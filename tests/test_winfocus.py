"""Тесты «вставлять туда, где начал диктовать» (winfocus + оркестратор).

Симптом, который лечим: пользователь диктует в чат, случайно кликает мышью в
сторону — и Ctrl+V уходит в другое окно. Теперь цель (окно) фиксируется в момент
старта диктовки, а перед вставкой фокус возвращается в неё.
"""
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import myflow.main as M
from myflow.audio import Recording
from myflow.winfocus import FocusHistory, WindowRef


class FocusHistoryTest(unittest.TestCase):
    def test_finds_window_active_at_phrase_start(self):
        hist = FocusHistory()
        chat_a = WindowRef(hwnd=1, pid=10, title="Telegram — Чат A", ts=100.0)
        chat_b = WindowRef(hwnd=2, pid=20, title="Браузер", ts=105.0)
        hist.record(chat_a)
        hist.record(chat_b)
        # фраза началась в 100.2 (в чате A), закончилась в 106
        found = hist.at(100.2)
        self.assertIsNotNone(found)
        self.assertEqual(found.hwnd, 1, "должно найтись окно, активное на старте фразы")

    def test_returns_none_for_future_moment(self):
        hist = FocusHistory()
        hist.record(WindowRef(hwnd=1, pid=1, title="A", ts=100.0))
        self.assertIsNone(hist.at(50.0), "замеры из будущего не подходят")

    def test_empty_history(self):
        self.assertIsNone(FocusHistory().at(1.0))

    def test_prune_drops_old_samples(self):
        hist = FocusHistory(window_seconds=10)
        hist.record(WindowRef(hwnd=1, pid=1, title="A", ts=100.0))
        hist.record(WindowRef(hwnd=2, pid=2, title="B", ts=200.0))
        hist.prune(now=205.0)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist.last().hwnd, 2)


class TargetRefTest(unittest.TestCase):
    def test_label_is_readable(self):
        ref = WindowRef(hwnd=1, pid=1, title="Очень длинный заголовок окна чата, который не влезет",
                        process="telegram.exe")
        self.assertLessEqual(len(ref.label), 43)
        self.assertTrue(ref.label.startswith("Очень длинный"))

    def test_label_falls_back_to_process(self):
        ref = WindowRef(hwnd=1, pid=1, title="", process="telegram.exe")
        self.assertEqual(ref.label, "telegram.exe")

    def test_same_window(self):
        a = WindowRef(hwnd=7, pid=1, title="A")
        b = WindowRef(hwnd=7, pid=1, title="A (изменился заголовок)")
        c = WindowRef(hwnd=8, pid=1, title="A")
        self.assertTrue(a.same_window(b))
        self.assertFalse(a.same_window(c))


class _Paster:
    can_read = True

    def __init__(self):
        self.calls = []

    def paste(self, text):
        self.calls.append(text)
        return True

    def read(self):
        return ""


class _T:
    hwnd = 7

    def __init__(self, hwnd=7, pid=42, title="Telegram — чат"):
        self.hwnd, self.pid, self.title = hwnd, pid, title


class PasteToTargetTest(unittest.TestCase):
    """Оркестратор: вставка идёт в цель, а не в текущее активное окно."""

    def setUp(self):
        import tests.test_main_flow as tm
        self.tm = tm

    def _app(self, target, current, insert_target="start", refocus=True, focus_ok=True):
        app = self.tm.make_app()
        app.paster = _Paster()
        app.settings.get().insert_target = insert_target
        app.settings.get().refocus_on_insert = refocus
        app._target = target
        p1 = mock.patch.object(M, "current_window", return_value=current)
        p2 = mock.patch.object(M, "focus_window", return_value=focus_ok)
        self.m1, self.m2 = p1.start(), p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        return app

    def test_no_refocus_when_already_there(self):
        ref = WindowRef(hwnd=7, pid=42, title="Чат")
        app = self._app(ref, _T(7))
        self.assertTrue(app._paste_to_target("текст"))
        self.assertEqual(app.paster.calls, ["текст"])
        self.m2.assert_not_called()
        self.assertEqual(app._last_insert_pid, 42)

    def test_refocus_when_user_clicked_elsewhere(self):
        ref = WindowRef(hwnd=7, pid=42, title="Чат")
        app = self._app(ref, _T(99, title="Совсем другое окно"))
        self.assertTrue(app._paste_to_target("текст"))
        self.m2.assert_called_once_with(7)
        self.assertEqual(app.paster.calls, ["текст"], "текст всё равно вставлен")
        self.assertEqual(app._last_insert_pid, 42,
                         "наблюдатель правок должен знать процесс цели")

    def test_failed_refocus_still_pastes(self):
        ref = WindowRef(hwnd=7, pid=42, title="Чат")
        app = self._app(ref, _T(99), focus_ok=False)
        self.assertTrue(app._paste_to_target("текст"))
        self.assertEqual(app.paster.calls, ["текст"],
                         "если фокус вернуть не удалось — вставляем как есть")

    def test_insert_target_focus_ignores_target(self):
        ref = WindowRef(hwnd=7, pid=42, title="Чат")
        app = self._app(ref, _T(99), insert_target="focus")
        app._paste_to_target("текст")
        self.m2.assert_not_called()
        self.assertEqual(app.paster.calls, ["текст"])

    def test_refocus_disabled_does_not_move_focus(self):
        ref = WindowRef(hwnd=7, pid=42, title="Чат")
        app = self._app(ref, _T(99), refocus=False)
        app._paste_to_target("текст")
        self.m2.assert_not_called()

    def test_no_target_pastes_to_active(self):
        app = self._app(None, _T(5))
        app._paste_to_target("текст")
        self.m2.assert_not_called()
        self.assertEqual(app.paster.calls, ["текст"])


class CaptureTargetTest(unittest.TestCase):
    """Цель захватывается в момент старта диктовки."""

    def setUp(self):
        import tests.test_main_flow as tm
        self.tm = tm

    def test_ptt_captures_window_at_start(self):
        app = self.tm.make_app()
        app.settings.get().insert_target = "start"
        app._target_locked = None
        with mock.patch.object(M, "current_window",
                               return_value=WindowRef(hwnd=3, pid=9, title="Чат B")):
            app._begin_recording()
        self.assertIsNotNone(app._target)
        self.assertEqual(app._target.hwnd, 3)

    def test_ptt_respects_locked_target(self):
        app = self.tm.make_app()
        app.settings.get().insert_target = "start"
        app._target_locked = WindowRef(hwnd=1, pid=2, title="Закреплённый чат")
        with mock.patch.object(M, "current_window",
                               return_value=WindowRef(hwnd=3, pid=9, title="Другое")):
            app._begin_recording()
        self.assertEqual(app._target.hwnd, 1, "закреплённая цель важнее текущего окна")

    def test_auto_uses_history_at_phrase_start(self):
        app = self.tm.make_app()
        app.settings.get().insert_target = "start"
        app._target_locked = None
        app._focus_history = FocusHistory()
        import time
        start = time.monotonic() - 3.0
        app._focus_history.record(WindowRef(hwnd=11, pid=1, title="Чат на старте", ts=start))
        app._focus_history.record(WindowRef(hwnd=12, pid=1, title="Кликнули позже",
                                            ts=start + 2.5))
        rec = Recording(pcm_int16=np.zeros(16000, dtype=np.int16), duration_ms=3000,
                        max_amplitude=0.5)
        app._on_utterance(rec)
        self.assertIsNotNone(app._target)
        self.assertEqual(app._target.hwnd, 11,
                         "в AUTO берём окно, активное в начале фразы (3 с назад)")

    def test_insert_target_focus_leaves_target_empty(self):
        app = self.tm.make_app()
        app.settings.get().insert_target = "focus"
        app._target_locked = None
        app._on_utterance(Recording(pcm_int16=np.zeros(16000, dtype=np.int16),
                                    duration_ms=1000, max_amplitude=0.5))
        self.assertIsNone(app._target)


class TargetLockTest(unittest.TestCase):
    def setUp(self):
        import tests.test_main_flow as tm
        self.tm = tm

    def test_toggle_locks_and_unlocks(self):
        app = self.tm.make_app()
        app._target_locked = None
        with mock.patch.object(M, "current_window",
                               return_value=WindowRef(hwnd=5, pid=7, title="Мой чат")):
            app._toggle_target_lock()
            self.assertIsNotNone(app._target_locked)
            self.assertEqual(app._target_locked.hwnd, 5)
            app._toggle_target_lock()
        self.assertIsNone(app._target_locked, "повторное F6 снимает закрепление")


if __name__ == "__main__":
    unittest.main()
