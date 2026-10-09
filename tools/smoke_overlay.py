"""Смоук Ф6: оверлей стартует в потоке, show/hide, клик-сквозь не падает."""
import sys
from pathlib import Path, time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
from myflow.overlay import StatusOverlay

ov = StatusOverlay(timeout_seconds=1.0)
ov.start()
time.sleep(0.8)  # дать Tk подняться

# до запуска show писал в очередь? guard не должен копить; поток жив:
assert ov._thread is not None and ov._thread.is_alive(), "thread не жив"

ov.show("● запись…", "#f5c542")
time.sleep(0.4)
# оверлей visible? проверим через root (доступен после старта)
root = ov._root
assert root is not None, "root не создан"
state = root.state()
print("state after show:", state)
assert state == "normal", f"окно не показано: {state}"

time.sleep(1.4)  # авто-скрытие через 1.0с
state2 = root.state()
print("state after timeout:", state2)
assert state2 == "withdrawn", f"авто-скрытие не сработало: {state2}"

ov.show("✓ вставлено", "#46a758")
time.sleep(0.3)
assert root.state() == "normal", "повторный show не сработал"
ov.stop()
time.sleep(0.5)
print("OK: оверлей поднялся, показал, скрылся по таймеру, повторный show ок, stop ок")