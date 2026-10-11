"""Показать строки панели про модели: и на временных, и на ЖИВЫХ настройках.

Нужно, чтобы отвечать на вопрос «зачем две модели» не словами, а тем, что
пользователь реально видит в пульте.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from myflow.config import SettingsStore          # noqa: E402
from myflow.db import Database                   # noqa: E402
import myflow.main as M                          # noqa: E402

KEYS = ("asr", "router", "judge", "llm", "postproc")


def show(settings_path: Path | None, title: str) -> None:
    print(f"  --- {title} ---")
    tmp = tempfile.TemporaryDirectory()
    if settings_path is not None and settings_path.exists():
        store = SettingsStore(settings_path)
        db_path = settings_path.parent / "tmp_rows.sqlite"
    else:
        store = SettingsStore(Path(tmp.name) / "s.json")
        db_path = Path(tmp.name) / "d.sqlite"
    app = M.MyFlowApp(settings=store, db=Database(db_path))
    app._mic_cache = ([], time.monotonic())
    try:
        for row in app.control_snapshot()["modules"]:
            if row["key"] in KEYS:
                state = "вкл " if row.get("on") else "выкл"
                print(f"  [{state}] {row['label']:<16} | {row['value']}")
    finally:
        app.db.close()
        app.ax.stop()
        app._stopping.set()
        if db_path.exists() and settings_path is None:
            db_path.unlink(missing_ok=True)
        tmp.cleanup()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="использовать живые настройки")
    args = ap.parse_args()
    if args.live:
        show(ROOT / "data" / "settings.json", "на твоих живых настройках")
    else:
        show(None, "на настройках по умолчанию")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
