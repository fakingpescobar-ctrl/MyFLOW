"""Показать, где в панели TUI живёт строка «Мост в DSH»."""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from myflow.config import SettingsStore          # noqa: E402
from myflow.db import Database                   # noqa: E402
import myflow.main as M                          # noqa: E402

tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
store = SettingsStore(Path(tmp.name) / "s.json")
store.get().insert_method = "dsh"
app = M.MyFlowApp(settings=store, db=Database(Path(tmp.name) / "d.sqlite"))
app._mic_cache = ([], time.monotonic())

snap = app.control_snapshot()
print("  ЛЕВАЯ КОЛОНКА (модули):")
for m in snap["modules"]:
    print(f"    {m['label']:<18} {m.get('value', '')}")
print("  ПРАВАЯ КОЛОНКА (значения):")
for v in snap["values"]:
    print(f"    {v['label']:<18} {v.get('value', '')}")

rows = [m for m in snap["modules"] if m["key"] == "dshbridge"]
print()
if rows:
    print(f"  СТРОКА ВИДНА: {rows[0]['label']} | {rows[0]['value']} | {rows[0].get('note','')}")
else:
    print("  строки «Мост в DSH» НЕТ — значит мост уже стоит и работает")

app.db.close()
app.ax.stop()
app._stopping.set()
