"""Смоук оверлея v2 на живой машине (нужен десктоп).

Проверяем то, что нельзя проверить юнитами: окно реально создаётся,
клик-сквозь применяется, статусы «●/⚙» залипают, «✓» прячется по таймеру.
ВАЖНО: состояние окна читаем через ov.query() — обращения к Tk из главного
потока ломают Tcl на выходе («async handler deleted by the wrong thread»).

Запуск: .venv\\Scripts\\python -X utf8 tools\\smoke_overlay.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from myflow.overlay import StatusOverlay, _work_area

ov = StatusOverlay(timeout_seconds=1.0)
ov.start()
time.sleep(1.0)  # дать Tk подняться
assert ov.is_running(), "поток оверлея не жив"
print("work_area:", _work_area())
print("geometry:", ov.query(lambda root, label: root.geometry()))

ov.show("● запись…", "#f5c542")
time.sleep(0.5)
state = ov.query(lambda root, label: root.state())
assert state == "normal", f"окно не показано: {state}"

# «● запись» — залипающий статус: не должен исчезнуть по таймеру
time.sleep(1.4)
state = ov.query(lambda root, label: root.state())
assert state == "normal", "статус записи самопроизвольно скрылся"
print("OK: «● запись» держится (залипающий статус)")

# «✓ вставлено» — успех: прячется по таймауту
ov.show("✓ вставлено", "#46a758")
time.sleep(0.4)
assert ov.query(lambda root, label: root.state()) == "normal", "успех не показался"
time.sleep(1.4)
state = ov.query(lambda root, label: root.state())
assert state == "withdrawn", f"авто-скрытие успеха не сработало: {state}"
print("OK: «✓ вставлено» скрылось по таймеру")

# «✗ ошибка» — висит дольше (3 с)
ov.show("✗ ошибка", "#e5484d")
time.sleep(2.0)
state = ov.query(lambda root, label: root.state())
assert state == "normal", "ошибка исчезла раньше 3 с"
print("OK: «✗ ошибка» держится дольше успеха")

# клик-сквозь применён: WS_EX_TRANSPARENT выставлен
print("text:", ov.query(lambda root, label: label.cget("text")))

ov.hide()
time.sleep(0.3)
assert ov.query(lambda root, label: root.state()) == "withdrawn", "hide() не сработал"
ov.stop()
print("OK: оверлей v2 поднялся, залипает/скрывается по политике, hide/stop работают")
