"""Тесты статус-оверлея v2.

GUI не поднимаем: проверяем чистую логику (таймауты статусов, позиция,
ограничение очереди, автоподъём потока) — сам Tk прогоняется в
tools/smoke_overlay.py на живой машине.
"""
import queue
import threading
import time
import unittest
from unittest import mock

from myflow.overlay import StatusOverlay, _pick_timeout, _place


class TimeoutPolicyTest(unittest.TestCase):
    def test_sticky_states(self):
        """● запись и ⚙ обработка не гаснут сами — их сменит следующий статус."""
        self.assertIsNone(_pick_timeout("● запись…"))
        self.assertIsNone(_pick_timeout("⚙ распознавание…"))
        self.assertIsNone(_pick_timeout("⚙ полировка…"))

    def test_error_lives_longer_than_success(self):
        self.assertEqual(_pick_timeout("✗ ошибка"), 3.0)
        self.assertEqual(_pick_timeout("✓ вставлено"), 1.4)
        self.assertGreater(_pick_timeout("✗ ошибка"), _pick_timeout("✓ вставлено"))

    def test_plain_status_uses_default(self):
        self.assertEqual(_pick_timeout("просто текст"), 1.8)
        self.assertEqual(_pick_timeout("просто текст", default=5.0), 5.0)

    def test_empty_text(self):
        self.assertEqual(_pick_timeout(""), 1.8)
        self.assertEqual(_pick_timeout(None), 1.8)  # type: ignore[arg-type]


class PlaceTest(unittest.TestCase):
    def test_bottom_right_of_work_area(self):
        x, y = _place((0, 0, 1920, 1080), 200, 40, margin=24)
        self.assertEqual((x, y), (1920 - 200 - 24, 1080 - 40 - 24))

    def test_taskbar_on_left_and_top(self):
        """Рабочая область может начинаться не с нуля (панель слева/сверху)."""
        x, y = _place((80, 40, 1920, 1080), 200, 40, margin=24)
        self.assertEqual((x, y), (1920 - 200 - 24, 1080 - 40 - 24))

    def test_second_monitor_with_negative_origin(self):
        x, y = _place((-1920, 0, 0, 1080), 200, 40, margin=24)
        self.assertEqual((x, y), (-224, 1016))

    def test_tiny_work_area_does_not_go_negative(self):
        x, y = _place((0, 0, 100, 50), 200, 40, margin=24)
        self.assertEqual((x, y), (0, 0))


class QueueDisciplineTest(unittest.TestCase):
    def test_overflow_drops_oldest(self):
        """Всплеск статусов не должен копиться: показываем свежее."""
        ov = StatusOverlay()
        ov._started = True                      # как после start(), но без Tk
        with mock.patch.object(ov, "start"):    # поток не поднимаем
            for i in range(12):
                ov.show(f"событие {i}", timeout=-1)
        messages = list(ov._q.queue)
        self.assertEqual(len(messages), 8, "очередь должна быть ограничена")
        self.assertEqual(messages[0][1], "событие 4", "старые статусы отброшены")
        self.assertEqual(messages[-1][1], "событие 11")

    def test_queue_does_not_grow_on_empty(self):
        ov = StatusOverlay()
        with mock.patch.object(ov, "_q", queue.Queue(maxsize=1)):
            ov._put(("show",))
            ov._put(("show",))
            self.assertEqual(ov._q.qsize(), 1)


class LifecycleTest(unittest.TestCase):
    def test_not_running_before_start(self):
        self.assertFalse(StatusOverlay().is_running())

    def test_show_does_not_start_when_disabled(self):
        """Фиче-флаг overlay=False не должен включаться сам от show()."""
        ov = StatusOverlay()
        with mock.patch.object(ov, "_run") as run:
            ov.show("✓ test", "#fff")
            time.sleep(0.1)
        run.assert_not_called()
        self.assertFalse(ov.is_running())

    def test_show_restarts_dead_thread(self):
        """Раньше умерший оверлей молча перестал бы показываться навсегда."""
        ov = StatusOverlay()
        started = threading.Event()

        def fake_run():
            started.set()
            time.sleep(0.3)

        with mock.patch.object(ov, "_run", side_effect=fake_run):
            ov.start()
            self.assertTrue(started.wait(1.0), "поток оверлея не запустился")
            started.clear()
            time.sleep(0.5)                     # поток «умер» (fake_run вернулся)
            self.assertFalse(ov.is_running())
            ov.show("✓ test", "#fff")           # должен поднять заново
            self.assertTrue(started.wait(1.0), "оверлей не перезапустился")

    def test_show_and_hide_do_not_raise_without_gui(self):
        ov = StatusOverlay()
        ov.hide()
        ov.stop()
        with mock.patch.object(ov, "_run"):     # не поднимаем реальный Tk
            ov.start()
            ov.show("тест", "#fff")
        self.assertFalse(ov.is_running())

    def test_query_without_thread_returns_none(self):
        self.assertIsNone(StatusOverlay().query(lambda root, label: 1))


if __name__ == "__main__":
    unittest.main()
