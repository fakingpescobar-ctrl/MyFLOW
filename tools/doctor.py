"""Доктор MyFLOW: что готово, чего не хватает, что делать.

Зачем. Новому человеку (и нам при отладке) нужен один ответ на вопрос «почему не
работает». Доктор ничего не меняет: только смотрит и говорит.

    python tools/doctor.py
"""

from __future__ import annotations

import argparse

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.bootstrap import (
    BUNDLES,
    MODEL_FILES,
    PACKAGES,
    confirm,
    missing_models,
    missing_packages,
    repair,
    train_env_ready,
)

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


def check_modules() -> tuple[list[str], list[str]]:
    """Пакеты. -> (что мешает запуску, что можно починить).

    Обязательные пакеты — стоп. Желательные (GigaAM, проверка голоса, OGG) —
    предупреждение: без них приложение работает, но часть функций молча
    отключается. Их тоже предлагаем поставить.
    """
    blocking: list[str] = []
    fixable: list[str] = []
    missing = missing_packages()
    required = [p for p in missing if p.required]
    optional = [p for p in missing if not p.required]
    if required:
        line(
            BAD, f"нет обязательных модулей: {', '.join(p.pip_name for p in required)}"
        )
        blocking.append("поставить пакеты: python -m myflow.bootstrap --fix")
    else:
        installed_required = sum(1 for p in PACKAGES if p.required and p.installed)
        line(OK, f"обязательные зависимости на месте ({installed_required})")
    for pkg in optional:
        line(WARN, f"нет модуля {pkg.pip_name} — {pkg.why} (желательно)")
        fixable.append(f"поставить пакет {pkg.pip_name} — {pkg.why}")
    return blocking, fixable


def check_training_assets() -> tuple[list[str], list[str]]:
    """Ассеты для обучения постпроцессора: база Qwen3-0.6B и корпус.

    Они нужны только тому, кто собирается обучать (у кого есть .venv-train).
    Без учебного окружения мы их не предлагаем качать — иначе доктор свалил
    бы каждому по 1.5 ГБ, о которых он и не думал.
    """
    blocking: list[str] = []
    fixable: list[str] = []
    train_ready = train_env_ready()
    for bundle in BUNDLES:
        if bundle.present():
            line(OK, f"{bundle.label}: на месте")
            continue
        if train_ready:
            line(WARN, f"{bundle.label}: нет — {bundle.why}")
            fixable.append(f"скачать: {bundle.label} ({bundle.why})")
        else:
            line(WARN, f"{bundle.label}: нет — нужно только для обучения "
                       "(install-train-env.bat), само не скачается")
    return blocking, fixable


def check_gigaam() -> None:
    """GigaAM — основной движок распознавания.

    Он НЕ в списке скачиваемого: onnx-asr тянет его сам при первом
    распознавании (нужен интернет один раз). Поэтому это предупреждение,
    а не проблема — доктор не должен блокировать запуск из-за него.
    """
    cache = ROOT / "models" / "hf-cache" / "hub"
    found = []
    if cache.exists():
        found = [p for p in cache.rglob("*.onnx") if p.stat().st_size > 50 * 2**20]
    if found:
        main = max(found, key=lambda p: p.stat().st_size)
        line(OK, f"GigaAM v3 e2e: {main.name} ({main.stat().st_size / 2**20:.0f} МБ)")
    else:
        line(WARN, "GigaAM v3 e2e: нет — скачается сам при первом распознавании")


def check_models() -> tuple[list[str], list[str]]:
    """Модели. -> (что мешает запуску, что можно скачать).

    Обязательная модель (VAD) попадает и туда, и туда: без неё приложение
    не стартит — но и скачать её можно.
    """
    blocking: list[str] = []
    fixable: list[str] = []
    for model in MODEL_FILES:
        if model.present():
            line(OK, f"{model.label}: {model.name_of()}")
            continue
        mark = "обязательна" if model.required else "желательна"
        line(
            BAD if model.required else WARN,
            f"{model.label}: не найдено ({mark}) — {model.why}",
        )
        item = f"скачать: {model.label} ({model.why})"
        if model.required:
            blocking.append(item)
        fixable.append(item)
    return blocking, fixable


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
            "select count(*) from history where coalesce(asr_text,'') <> ''"
        ).fetchone()[0]
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
        line(
            WARN if busy else OK,
            f"порт {port} ({what}): {'занят — сервер работает' if busy else 'свободен'}",
        )


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
            line(
                WARN,
                f"мост в DSH не установлен"
                f"{' (DSH найден)' if app else ' (DSH не найден)'}"
                f" — ставится из панели MyFLOW",
            )
    except Exception as exc:  # noqa: BLE001
        line(WARN, f"мост в DSH: не удалось проверить ({exc!r})")


def main() -> int:
    ap = argparse.ArgumentParser(description="MyFLOW: проверка готовности")
    ap.add_argument(
        "--fix", action="store_true", help="скачать недостающее (с вопросом)"
    )
    args = ap.parse_args()

    print("  MyFLOW — проверка готовности\n")
    check_venv()
    check_python()
    blocking, fixable = check_modules()
    blocking_models, fixable_models = check_models()
    check_gigaam()
    blocking_train, fixable_train = check_training_assets()
    problems.extend(blocking + blocking_models)
    fixable = fixable + fixable_models + fixable_train
    check_history()
    check_servers()
    check_dsh_bridge()

    print()
    if args.fix and (fixable or blocking):
        if confirm("  Поставить и скачать? [Y/n] "):
            repair()
            # Убираем ТОЛЬКО те проблемы, которые чинил --fix: чужие
            # (нет окружения, старый Python, не установленный мост в DSH)
            # остаются в списке — иначе доктор сказал бы «ВСЁ ГОТОВО» при
            # реальных проблемах.
            fixed = set(blocking + blocking_models)
            problems[:] = [p for p in problems if p not in fixed]
            print()
            print("  Перепроверяю:")
            blocking, fixable = check_modules()
            blocking_models, fixable_models = check_models()
            problems.extend(blocking + blocking_models)
            print()

    if problems:
        print(f"  ЧТО СДЕЛАТЬ ({len(problems)}):")
        for item in dict.fromkeys(problems):
            print(f"    - {item}")
        print("    (одной командой: python -m myflow.bootstrap --fix)")
        return 1
    print("  ВСЁ ГОТОВО ✓ запускай start-myflow.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
