"""Мост MyFLOW → DeepSeek Harness: установка, удаление и починка.

Зачем мост. Режим «DeepSeek Harness (без фокуса)» работает через плагин внутри
приложения: он вкладывает надиктованный текст в текущую сессию (никаких клавиш и
перехвата фокуса). Плагин живёт в КОНКРЕТНОЙ установке DSH — в её домашнем каталоге
(`$DSH_HOME/profiles/<profile>`), поэтому у нового пользователя его нет.

Ключевое решение: плагин ставится **копией** внутрь профиля, а не ссылкой на папку
MyFLOW. Со ссылкой (`link:`) удаление или перенос MyFLOW ломает профиль DSH — а DSH
на неразрешимом бандле падает при запуске. Копия делает установку независимой:
MyFLOW можно двигать и удалять, чужое приложение не пострадает.

Что умеет модуль:
  * найти установку DSH (запущенный процесс -> DSH_APP -> обычные места);
  * найти команду `dsh` (PATH -> рядом с приложением);
  * определить оконные профили (`desktop`, `web`, ...) — приложение может грузить
    не `web`, на этом мы уже спотыкались;
  * проверить, что мост стоит и ССЫЛКИ НЕ БИТЫЕ;
  * поставить / починить / удалить мост.

CLI:  python -m myflow.dsh_bridge --status | --install | --remove
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_NAME = "dsh-plugin-myflow"
DEFAULT_PROFILE = "web"
CREATE_NO_WINDOW = 0x08000000


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def plugin_dir() -> Path:
    """Каталог с нашим плагином (едет в поставке вместе с MyFLOW)."""
    return project_root() / "dsh-plugin-myflow"


def dsh_home() -> Path:
    """Домашний каталог DSH: $DSH_HOME, иначе ~/.dsh."""
    env = os.environ.get("DSH_HOME")
    return Path(env) if env else Path.home() / ".dsh"


def profile_dir(profile: str = DEFAULT_PROFILE) -> Path:
    return dsh_home() / "profiles" / profile


def _manifest(profile: str) -> dict | None:
    manifest = profile_dir(profile) / "package.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _write_manifest(profile: str, data: dict) -> None:
    manifest = profile_dir(profile) / "package.json"
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _bundles(data: dict) -> list:
    return ((data.get("dsh") or {}).get("profile") or {}).get("bundles") or []


# ── поиск установки ─────────────────────────────────────────────────────────

def _app_paths_from_processes() -> list[Path]:
    """Пути к запущенным приложениям DSH (WMI, как в opencode-инжекторе)."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name = 'DeepSeek Harness.exe'\" "
             "| Select-Object -ExpandProperty ExecutablePath"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            return [Path(line.strip()).parent for line in result.stdout.splitlines()
                    if line.strip()]
    except Exception:  # noqa: BLE001
        pass
    return []


def dsh_app_running() -> bool:
    """Запущено ли приложение DSH (важно для подсказки про перезапуск)."""
    return bool(_app_paths_from_processes())


def find_dsh_app() -> Path | None:
    """Где установлено приложение: процесс -> DSH_APP -> обычные места."""
    for path in _app_paths_from_processes():
        if path.exists():
            return path
    env = os.environ.get("DSH_APP")
    if env and Path(env).exists():
        return Path(env)
    local, progfiles = os.environ.get("LOCALAPPDATA"), os.environ.get("PROGRAMFILES")
    candidates = []
    if local:
        candidates += [Path(local) / "Programs" / "DeepSeek Harness",
                        Path(local) / "DeepSeek Harness"]
    if progfiles:
        candidates.append(Path(progfiles) / "DeepSeek Harness")
    for cand in candidates:
        if (cand / "DeepSeek Harness.exe").exists() or cand.is_dir():
            return cand
    return None


def find_dsh_cli() -> str | None:
    """Где `dsh`: PATH, затем рядом с приложением (PATH не нужен)."""
    for name in ("dsh", "dsh.cmd", "dsh.ps1", "dsh.exe"):
        found = shutil.which(name)
        if found:
            return found
    env_app = os.environ.get("DSH_APP")
    bases = [p for p in (find_dsh_app(), Path(env_app) if env_app else None) if p]
    for base in bases:
        for cand in (base / "resources" / "runtime" / "cli" / "bin" / "dsh.cmd",
                     base / "resources" / "runtime" / "cli" / "bin" / "dsh"):
            if cand.exists():
                return str(cand)
    return None


# ── профили ─────────────────────────────────────────────────────────────────

def gui_profiles() -> list[str]:
    """Профили с оконной оболочкой (web, desktop, ...) — по наличию dsh-web-app."""
    root = dsh_home() / "profiles"
    found: list[str] = []
    try:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            data = _manifest(entry.name)
            if not data:
                continue
            if any("dsh-web-app" in str(b) for b in _bundles(data)):
                found.append(entry.name)
    except Exception:  # noqa: BLE001
        pass
    return found


def target_profiles(profile: str | None = None) -> list[str]:
    if profile:
        return [profile]
    return gui_profiles() or [DEFAULT_PROFILE]


# ── состояние моста ─────────────────────────────────────────────────────────

PLUGIN_ID = "myflow-bridge"


def _plugin_disabled(profile: str) -> bool:
    """Выключен ли наш плагин тумблером в настройках DSH.

    Тумблер в списке плагинов записывается как `disabled: true` для строки с нашим
    id в слое патчей профиля (или в домашнем слое). Тогда мост формально стоит, но
    НЕ работает — MyFLOW должен честно сказать и откатиться на буфер обмена.
    """
    for patch in (profile_dir(profile) / "cordis.patch.yml",
                  dsh_home() / "cordis.patch.yml"):
        if not patch.exists():
            continue
        try:
            text = patch.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        for chunk in text.split("- id:"):
            if PLUGIN_ID in chunk and "disabled: true" in chunk.replace('"', ""):
                return True
    return False


def bridge_health(profile: str | None = None) -> dict:
    """Состояние моста: стоит ли и не битые ли ссылки.

    «Битым» считается профиль, где зависимость объявлена как `link:` на
    несуществующий путь (MyFLOW перенесли или удалили) или где нет копии плагина.
    """
    targets = target_profiles(profile)
    broken: list[str] = []
    installed = bool(targets)
    for name in targets:
        data = _manifest(name)
        if not data:
            installed = False
            broken.append(name)
            continue
        value = (data.get("dependencies") or {}).get(PACKAGE_NAME)
        if not value:
            installed = False
            continue
        if _plugin_disabled(name):
            broken.append(name)          # тумблер выключен — работать не будет
            continue
        text = str(value)
        if text.startswith("link:"):
            if not Path(text[5:]).exists():
                broken.append(name)
                continue
        else:
            rel = text.split(":", 1)[-1].lstrip("./\\")
            if not (profile_dir(name) / rel).exists() and not (
                    profile_dir(name) / "node_modules" / PACKAGE_NAME).exists():
                broken.append(name)
                continue
        copy = profile_dir(name) / "node_modules" / PACKAGE_NAME
        if not copy.exists():
            broken.append(name)
        elif copy.is_symlink() or os.path.isjunction(copy):
            # Ссылка на папку MyFLOW: перенос/удаление MyFLOW сломает профиль DSH,
            # поэтому считаем такую установку неполной и предлагаем починку.
            broken.append(name)
    return {"ok": installed and not broken, "installed": installed,
            "broken": broken,
            "disabled": [name for name in targets if _plugin_disabled(name)],
            "profiles": targets}


def plugin_fingerprint(source: Path | None = None) -> str:
    """Отпечаток плагина (содержимое файлов) — чтобы видеть расхождение копии."""
    import hashlib

    root = Path(source) if source else plugin_dir()
    digest = hashlib.sha256()
    if not root.exists():
        return ""
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def stale_profiles(profile: str | None = None) -> list[str]:
    """Где копия плагина отличается от поставки (например, после обновления MyFLOW)."""
    wanted = plugin_fingerprint()
    result: list[str] = []
    for name in target_profiles(profile):
        copy = profile_dir(name) / "node_modules" / PACKAGE_NAME
        if not copy.exists():
            continue
        if not wanted or plugin_fingerprint(copy) != wanted:
            result.append(name)
    return result


def sync_bridge(profile: str | None = None) -> tuple[bool, str]:
    """Обновить копии плагина, если они разошлись с поставкой.

    Нужно после обновления MyFLOW: у людей копия в профиле DSH остаётся старой, а
    плагин грузится именно из неё. Вызывается на старте MyFLOW (только когда выбран
    режим DeepSeek Harness) и, конечно, при обычной установке.
    """
    stale = stale_profiles(profile)
    if not stale:
        return True, "плагин в DSH совпадает с поставкой"
    ok, message = install_bridge(profile)
    if not ok:
        return False, message
    return True, (f"обновил плагин в {', '.join(stale)} — перезапусти "
                  f"DeepSeek Harness, чтобы применилось")


def bridge_installed(profile: str | None = None) -> bool:
    """Стоит ли мост и не сломаны ли ссылки (для панели и инжектора)."""
    return bool(bridge_health(profile)["ok"])


# ── установка / удаление ────────────────────────────────────────────────────

def _copy_into_profile(profile: str) -> Path:
    """Скопировать плагин ВНУТРЬ профиля (не ссылкой!) и вернуть путь копии."""
    dest = profile_dir(profile) / "node_modules" / PACKAGE_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    _remove_any(dest)
    shutil.copytree(plugin_dir(), dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest


def _remove_any(path: Path) -> None:
    """Убрать путь любого вида: обычный каталог, файл, симлинк или junction.

    pnpm ставит локальные зависимости ССЫЛКОЙ (симлинк/junction), а `rmtree` на
    такой штуке в Windows падает (FileExistsError при копировании следом) —
    поэтому ссылки сначала снимаем как ссылки.
    """
    try:
        if path.is_symlink() or os.path.isjunction(path):
            os.unlink(path)          # снять саму ссылку, цель не трогаем
            return
    except (AttributeError, OSError):
        pass
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def install_bridge(profile: str | None = None) -> tuple[bool, str]:
    """Поставить (или починить) мост во всех нужных профилях."""
    plugin = plugin_dir()
    if not (plugin / "index.js").exists():
        return False, f"нет плагина в поставке: {plugin}"

    targets = target_profiles(profile)
    installed_now, repaired, failed = [], [], []
    for name in targets:
        data = _manifest(name)
        if data is None:
            failed.append(name)
            continue
        already_ok = bridge_installed(name)
        try:
            _copy_into_profile(name)          # копия: MyFLOW можно переносить/удалять
            data.setdefault("dependencies", {})[PACKAGE_NAME] = (
                f"file:./node_modules/{PACKAGE_NAME}")
            bundles = _bundles(data)
            if PACKAGE_NAME not in bundles:
                data.setdefault("dsh", {}).setdefault("profile", {}).setdefault(
                    "bundles", []).append(PACKAGE_NAME)
            _write_manifest(name, data)
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"[myflow] мост в {name}: {exc!r}", file=sys.stderr)
            continue
        (installed_now if already_ok else repaired).append(name)

    if not installed_now and not repaired:
        return False, (f"не смог поставить мост ({', '.join(failed) or '—'}) — "
                       f"проверь права на {dsh_home() / 'profiles'}")

    what = []
    if repaired:
        what.append(f"поставлен в {', '.join(repaired)}")
    if installed_now:
        what.append(f"уже был в {', '.join(installed_now)}")
    tail = (" — перезапусти DeepSeek Harness" if dsh_app_running()
            else " — запусти DeepSeek Harness")
    return True, f"мост: {'; '.join(what)}{tail}"


def uninstall_bridge(profile: str | None = None) -> tuple[bool, str]:
    """Убрать мост: чистим зависимости, bundles и скопированный плагин."""
    targets = target_profiles(profile)
    removed: list[str] = []
    for name in targets:
        data = _manifest(name)
        if data is None:
            continue
        changed = False
        deps = data.get("dependencies") or {}
        if PACKAGE_NAME in deps:
            deps.pop(PACKAGE_NAME)
            changed = True
        bundles = _bundles(data)
        if PACKAGE_NAME in bundles:
            bundles.remove(PACKAGE_NAME)
            changed = True
        if changed:
            _write_manifest(name, data)
            removed.append(name)
        copy = profile_dir(name) / "node_modules" / PACKAGE_NAME
        if copy.exists():
            shutil.rmtree(copy, ignore_errors=True)
    if not removed:
        return True, "мост и не был установлен"
    return True, f"мост убран из {', '.join(removed)} — перезапусти DeepSeek Harness"


def repair_bridge(profile: str | None = None) -> tuple[bool, str]:
    """Починить мост (битые ссылки, переезд MyFLOW) — это та же установка."""
    return install_bridge(profile)


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Мост MyFLOW -> DeepSeek Harness")
    ap.add_argument("action", nargs="?", default="status",
                    choices=("status", "install", "remove", "repair"))
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    if args.action == "status":
        health = bridge_health(args.profile)
        print(f"  оконные профили: {gui_profiles() or '—'}")
        print(f"  приложение: {find_dsh_app() or 'не найдено'}"
              f"{' (запущено)' if dsh_app_running() else ''}")
        print(f"  команда dsh: {find_dsh_cli() or 'не найдена'}")
        print(f"  мост: {'готов' if health['ok'] else 'не готов'}")
        if health["broken"]:
            print(f"  битые/пустые: {', '.join(health['broken'])}")
        stale = stale_profiles(args.profile)
        if stale:
            print(f"  устаревшая копия: {', '.join(stale)}")
        return 0

    action = {"install": install_bridge, "repair": repair_bridge,
              "remove": uninstall_bridge}[args.action]
    ok, message = action(args.profile)
    print(f"  {'ok  ' if ok else 'НЕТ '} {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
