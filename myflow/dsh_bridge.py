"""Мост MyFLOW → DeepSeek Harness: проверка и установка плагина у нового пользователя.

Проблема, которую решает модуль. Плагин `dsh-plugin-myflow` ставится в КОНКРЕТНУЮ
установку DSH (в `$DSH_HOME/profiles/<profile>`). У нового пользователя MyFLOW его
нет, и без установки режим «вставка в чат DSH» молча не работал бы.

Что делаем:
  * плагин едет вместе с MyFLOW (каталог `dsh-plugin-myflow/` в репозитории);
  * MyFLOW умеет проверить, стоит ли мост в профиле пользователя;
  * и поставить его сам — через штатный CLI DSH (`dsh plugin --profile web add`).

Если мост не установлен, диктовка НЕ теряется: инжектор откатывается на обычную
вставку через буфер обмена (с фокусом) и сообщает об этом в журнал.
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


def find_dsh_app() -> Path | None:
    """Где установлено приложение DeepSeek Harness.

    Порядок поиска — от самого надёжного к догадкам:
      1. запущенный процесс `DeepSeek Harness.exe` (у работающего приложения путь
         всегда настоящий — это лучший источник);
      2. переменная DSH_APP (если пользователь её задал);
      3. обычные места установки на этой машине.
    """
    for path in _app_paths_from_processes():
        if path.exists():
            return path

    env = os.environ.get("DSH_APP")
    if env and Path(env).exists():
        return Path(env)

    local = os.environ.get("LOCALAPPDATA")
    progfiles = os.environ.get("PROGRAMFILES")
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


def _app_paths_from_processes() -> list[Path]:
    """Пути к запущенным приложениям DSH (через WMI, как в opencode-инжекторе)."""
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


def find_dsh_cli() -> str | None:
    """Где `dsh`: PATH, затем рядом с приложением, затем DSH_APP."""
    for name in ("dsh", "dsh.cmd", "dsh.ps1", "dsh.exe"):
        found = shutil.which(name)
        if found:
            return found

    bases: list[Path] = []
    app = find_dsh_app()
    if app:
        bases.append(app)
    env = os.environ.get("DSH_APP")
    if env:
        bases.append(Path(env))
    for base in bases:
        for cand in (base / "resources" / "runtime" / "cli" / "bin" / "dsh.cmd",
                     base / "resources" / "runtime" / "cli" / "bin" / "dsh"):
            if cand.exists():
                return str(cand)
    return None


def _manifest(profile: str) -> dict | None:
    manifest = profile_dir(profile) / "package.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def gui_profiles() -> list[str]:
    """Профили с оконной оболочкой (web, desktop, ...).

    Приложение может грузить НЕ `web`: у пользователя запускается профиль
    `desktop` (бандл desktop-оболочки), и мост, поставленный только в `web`,
    не загрузится — на этом мы уже спотыкались. Поэтому определяем цели по
    наличию бандла `dsh-web-app`.
    """
    root = dsh_home() / "profiles"
    found: list[str] = []
    try:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            data = _manifest(entry.name)
            if not data:
                continue
            bundles = ((data.get("dsh") or {}).get("profile") or {}).get("bundles") or []
            if any("dsh-web-app" in str(b) for b in bundles):
                found.append(entry.name)
    except Exception:  # noqa: BLE001
        pass
    return found


def target_profiles(profile: str | None = None) -> list[str]:
    """Куда ставить мост: явно указанный профиль или все оконные."""
    if profile:
        return [profile]
    return gui_profiles() or [DEFAULT_PROFILE]


def bridge_installed(profile: str | None = None) -> bool:
    """Стоит ли наш плагин во всех нужных профилях (по манифестам)."""
    targets = target_profiles(profile)
    if not targets:
        return False
    for name in targets:
        data = _manifest(name)
        if not data:
            return False
        if PACKAGE_NAME not in (data.get("dependencies") or {}):
            return False
    return True


def install_bridge(profile: str | None = None) -> tuple[bool, str]:
    """Поставить мост в нужные профили DSH. Возвращает (получилось, сообщение)."""
    plugin = plugin_dir()
    if not (plugin / "index.js").exists():
        return False, f"нет плагина в поставке: {plugin}"

    targets = target_profiles(profile)
    pending = [name for name in targets if not bridge_installed(name)]

    cli = find_dsh_cli()
    if cli is None:
        return False, ("не нашёл команду dsh — поставь мост вручную: "
                       f"dsh plugin --profile {targets[0]} add \"{plugin}\"")

    done: list[str] = []
    for name in pending:
        try:
            subprocess.run(
                [cli, "plugin", "--profile", name, "add", str(plugin)],
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=180, creationflags=CREATE_NO_WINDOW,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"не смог запустить dsh для профиля {name}: {exc!r}"
        if bridge_installed(name):
            done.append(name)

    if not pending:
        return True, f"мост уже установлен ({', '.join(targets)})"
    if not done:
        return False, ("dsh не поставил мост ни в один профиль "
                       f"({', '.join(pending)})")
    missing = [name for name in targets if name not in done]
    if missing:
        return True, (f"мост установлен в {', '.join(done)}, но не в "
                      f"{', '.join(missing)} — перезапусти DeepSeek Harness")
    if dsh_app_running():
        return True, ("мост установлен — перезапусти DeepSeek Harness, "
                      "чтобы плагин загрузился")
    return True, "мост установлен — запусти DeepSeek Harness"
