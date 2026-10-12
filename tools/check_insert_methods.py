"""Проверка: строка «Куда вставлять» и цикл способов, включая DSH."""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from myflow.config import SettingsStore          # noqa: E402
from myflow.db import Database                   # noqa: E402
import myflow.main as M                          # noqa: E402

tmp = tempfile.TemporaryDirectory()
db_path = Path(tmp.name) / "d.sqlite"
app = M.MyFlowApp(settings=SettingsStore(Path(tmp.name) / "s.json"),
                  db=Database(db_path))
app._mic_cache = ([], time.monotonic())


def row_value() -> str:
    return next(m for m in app.control_snapshot()["modules"]
                if m["key"] == "insert")["value"]


print(f"  сейчас: {row_value()}")
for _ in range(4):
    app.control_apply("adjust", "insert", +1)
    kind = type(app.paster).__name__
    print(f"  ←→ -> {row_value():<34} инжектор: {kind}")
    if kind == "DshInjector":
        print(f"        инбокс: {app.paster.inbox}")

app.db.close()
app.ax.stop()
app._stopping.set()
time.sleep(0.2)
tmp.cleanup()
