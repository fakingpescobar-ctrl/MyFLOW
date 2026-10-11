"""Показать строку «Дообучение» так, как её увидит пользователь."""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from myflow.config import SettingsStore          # noqa: E402
from myflow.db import Database                   # noqa: E402
import myflow.main as M                          # noqa: E402

tmp = tempfile.TemporaryDirectory()
app = M.MyFlowApp(settings=SettingsStore(Path(tmp.name) / "s.json"),
                  db=Database(Path(tmp.name) / "d.sqlite"))
app._mic_cache = ([], time.monotonic())
state = app.retrain_state()
print(f"  новых диктовок: {state['new']} | порог: {state['threshold']} | "
      f"обучено на: {state['trained_on']} | окружение готово: {state['env_ready']}")
row = next((m for m in app.control_snapshot()["modules"] if m["key"] == "retrain"), None)
if row:
    print(f"  строка: {row['label']} | {row['value']} | {row['note']}")
    print(f"  подсказка: {row['hint']}")
else:
    print("  строки нет (не найдено учебное окружение)")
app.db.close()
app.ax.stop()
app._stopping.set()
tmp.cleanup()
