"""Доктор MyFLOW: что готово, чего не хватает, что делать.

Зачем. Новому человеку (и нам при отладке) нужен один ответ на вопрос «почему не
работает». Доктор ничего не меняет: только смотрит и говорит.

    python tools/doctor.py
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "  ok  ", " warn ", " НЕТ  "
problems: list[str] = []


def line(status: str, text: str) -> None:
    print(f"  [{status}] {text}")


def check_python() -> None:
    version = sys.version_info
    text = f"Python {version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) >= (3, 11):
        line(OK, text)
    else:
        line(BAD, f"{text} — нужен 3.11 или новее")
        problems.append("обновить Python до 3.11+")


def check_modules() -> None:
    needed = {"numpy": "счёт и аудио", "sounddevice": "микрофон",
              "onnxruntime": "распознавание", "requests": "серверы моделей",
              "soundfile": "аудио фраз"}
    missing = [name for name in needed
               if importlib.util.find_spec(name) is None]
    if missing:
        line(BAD, f"нет модулей: {', '.join(missing)}")
        problems.append("установить зависимости: pip install -r requirements.txt")
    else:
        line(OK, f"зависимости на месте ({len(needed)})")


def check_models() -> None:
    models = ROOT / "models"
    if not models.exists():
        line(BAD, "нет папки models/")
        problems.append("скачать модели (см. README)")
        return

    files = [p for p in models.rglob("*") if p.is_file() and p.stat().st_size > 1024]
    gguf = [p for p in files if p.suffix == ".gguf"]
    whisper = [p for p in files if p.name.startswith("ggml-") and p.suffix == ".bin"]
    giga = [p for p in files if "giga" in p.name.lower() or "giga" in str(p.parent).lower()]

    # Рабочий постпроцессор — тот, что выбирает приложение (myflow-postproc-*).
    # Самый большой .gguf показывать нельзя: рядом лежат промежуточные сборки.
    shipped = [p for p in gguf if p.name.startswith("myflow-postproc")]
    for found, what in ((shipped or gguf, "постпроцессор (.gguf)"),
                        (whisper, "whisper (ggml-*.bin)"),
                        (giga, "GigaAM (onnx)")):
        if found:
            main = sorted(found, key=lambda p: -p.stat().st_size)[0]
            extra = f" (+ ещё {len(found) - 1})" if len(found) > 1 else ""
            line(OK, f"{what}: {main.name} ({main.stat().st_size / 2**20:.0f} МБ){extra}")
        else:
            line(WARN, f"{what}: не найдено")
            problems.append(f"скачать: {what}")


def check_venv() -> None:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv.exists():
        line(OK, "окружение .venv на месте")
    else:
        line(BAD, "нет .venv — запусти install-myflow.bat")
        problems.append("создать окружение: install-myflow.bat")


def check_history() -> None:
    db = ROOT / "data" / "myflow.sqlite"
    if not db.exists():
        line(WARN, "истории ещё нет (это нормально при первом запуске)")
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        count = con.execute("select count(*) from history").fetchone()[0]
        with_text = con.execute(
            "select count(*) from history where coalesce(asr_text,'') <> ''").fetchone()[0]
    finally:
        con.close()
    line(OK, f"история: {count} записей ({with_text} с текстом)")


def check_servers() -> None:
    """Порты моделей: занят — уже поднят, свободен — поднимет приложение."""
    import socket
    for port, what in ((8081, "whisper"), (8082, "Oracle 4B"), (8083, "постпроцессор")):
        with socket.socket() as sock:
            sock.settimeout(0.2)
            busy = sock.connect_ex(("127.0.0.1", port)) == 0
        line(WARN if busy else OK,
             f"порт {port} ({what}): {'занят — сервер работает' if busy else 'свободен'}")


def check_dsh_bridge() -> None:
    try:
        from myflow.dsh_bridge import bridge_health, dsh_app_running, find_dsh_app
        health = bridge_health()
        if health["ok"]:
            line(OK, "мост в DeepSeek Harness: готов")
        elif health.get("disabled"):
            line(WARN, "мост в DSH: плагин выключен в настройках")
        elif health.get("broken"):
            line(WARN, "мост в DSH: копия плагина повреждена (лечится в панели)")
        else:
            app = find_dsh_app()
            line(WARN, f"мост в DSH не установлен"
                       f"{' (DSH найден)' if app else ' (DSH не найден)'}"
                       f" — ставится из панели MyFLOW")
    except Exception as exc:  # noqa: BLE001
        line(WARN, f"мост в DSH: не удалось проверить ({exc!r})")


def main() -> int:
    print("  MyFLOW — проверка готовности\n")
    check_venv()
    check_python()
    check_modules()
    check_models()
    check_history()
    check_servers()
    check_dsh_bridge()

    print()
    if problems:
        print(f"  ЧТО СДЕЛАТЬ ({len(problems)}):")
        for item in dict.fromkeys(problems):
            print(f"    - {item}")
        return 1
    print("  ВСЁ ГОТОВО ✓ запускай start-myflow.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
