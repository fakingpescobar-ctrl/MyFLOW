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


def find_dsh_cli() -> str | None:
    """Где `dsh`: сначала PATH, потом рядом с установленным приложением."""
    for name in ("dsh", "dsh.cmd", "dsh.ps1", "dsh.exe"):
        found = shutil.which(name)
        if found:
            return found
    app = os.environ.get("DSH_APP")
    bases = [Path(app)] if app else []
    bases.append(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs")
    for base in bases:
        if not base:
            continue
        for cand in (base / "resources" / "runtime" / "cli" / "bin" / "dsh.cmd",
                     base / "resources" / "runtime" / "cli" / "bin" / "dsh"):
            if cand.exists():
                return str(cand)
    return None


def bridge_installed(profile: str = DEFAULT_PROFILE) -> bool:
    """Стоит ли наш плагин в профиле DSH (по манифесту профиля)."""
    manifest = profile_dir(profile) / "package.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    deps = data.get("dependencies") or {}
    return PACKAGE_NAME in deps


def install_bridge(profile: str = DEFAULT_PROFILE) -> tuple[bool, str]:
    """Поставить мост в профиль DSH. Возвращает (получилось, сообщение)."""
    plugin = plugin_dir()
    if not (plugin / "index.js").exists():
        return False, f"нет плагина в поставке: {plugin}"
    if bridge_installed(profile):
        return True, "мост уже установлен"

    cli = find_dsh_cli()
    if cli is None:
        return False, ("не нашёл команду dsh — поставь мост вручную: "
                       f"dsh plugin --profile {profile} add \"{plugin}\"")
    try:
        result = subprocess.run(
            [cli, "plugin", "--profile", profile, "add", str(plugin)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=180, creationflags=CREATE_NO_WINDOW,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"не смог запустить dsh: {exc!r}"

    if bridge_installed(profile):
        return True, ("мост установлен — перезапусти DeepSeek Harness, "
                      "чтобы плагин загрузился")
    tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:] or [""]
    return False, f"dsh не поставил мост: {tail[0][:120]}"
