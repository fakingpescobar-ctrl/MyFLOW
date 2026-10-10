"""Сторож микрофона: поток отвалился (Bluetooth/USB) — переоткрываем сами."""
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import myflow.main as M
from myflow.audio import DictationRecorder
from myflow.config import SettingsStore
from myflow.db import Database
import tempfile


class ReopenTest(unittest.TestCase):
    """Логика DictationRecorder.reopen(): поток умер -> поднимаем заново."""

    def _recorder(self):
        rec = DictationRecorder.__new__(DictationRecorder)   # без PortAudio в тесте
        rec._continuous = False
        rec._stream = None
        rec._device = 13
        rec._device_ok = staticmethod(lambda device: device == 13).__func__
        rec.stop_continuous = mock.Mock()
        rec.start_continuous = mock.Mock()
        return rec

    def test_noop_when_not_continuous(self):
        rec = self._recorder()
        self.assertIsNone(rec.reopen())
        rec.start_continuous.assert_not_called()

    def test_noop_when_stream_alive(self):
        rec = self._recorder()
        rec._continuous = True
        rec._stream = mock.Mock(active=True)
        self.assertIsNone(rec.reopen())
        rec.start_continuous.assert_not_called()

    def test_reopens_dead_stream(self):
        rec = self._recorder()
        rec._continuous = True
        rec._stream = mock.Mock(active=False)
        reason = rec.reopen()
        self.assertIsNotNone(reason)
        rec.stop_continuous.assert_called_once()
        rec.start_continuous.assert_called_once()

    def test_falls_back_to_default_when_device_gone(self):
        rec = self._recorder()
        rec._continuous = True
        rec._stream = mock.Mock(active=False)
        calls = []

        def start():
            calls.append(rec._device)
            if rec._device is not None:
                raise RuntimeError("устройство исчезло")

        rec.start_continuous = start
        reason = rec.reopen()
        self.assertEqual(calls, [13, None], "сначала пробуем 13, потом системный")
        self.assertEqual(rec._device, 13, "выбор пользователя не теряем")
        self.assertIn("системный", reason)

    def test_returns_none_when_nothing_works(self):
        rec = self._recorder()
        rec._continuous = True
        rec._stream = mock.Mock(active=False)
        rec.start_continuous = mock.Mock(side_effect=RuntimeError("нет микрофонов"))
        self.assertIsNone(rec.reopen())

    def test_stream_active_never_raises(self):
        rec = self._recorder()
        rec._stream = mock.Mock()
        type(rec._stream).active = mock.PropertyMock(side_effect=OSError("boom"))
        self.assertFalse(rec.stream_active())


class WatchdogLoopTest(unittest.TestCase):
    """Сторож в приложении: видит смерть потока и говорит об этом."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = SettingsStore(Path(tmp.name) / "s.json")
        db = Database(Path(tmp.name) / "w.sqlite")
        self.addCleanup(db.close)
        self.app = M.MyFlowApp(settings=store, db=db)
        self.app._stopping = threading.Event()
        self.app._pref_auto = True
        self.app.overlay = mock.Mock()
        self.app._mic_cache = ([], time.monotonic())

    def tearDown(self):
        self.app._stopping.set()
        try:
            self.app.ax.stop()
        except Exception:
            pass

    def test_watchdog_reopens_and_reports(self):
        self.app.recorder.reopen = mock.Mock(return_value="поток микрофона отвалился")
        self.app.recorder._last_status_text = "input overflow"
        t = threading.Thread(target=self.app._mic_watchdog_loop, args=(0.05,), daemon=True)
        t.start()
        time.sleep(0.3)
        self.app._stopping.set()
        t.join(timeout=2.0)
        self.assertTrue(self.app.recorder.reopen.called, "сторож должен проверять поток")
        self.assertTrue(self.app.overlay.show.called)

    def test_watchdog_silent_when_ptt(self):
        self.app._pref_auto = False
        self.app.recorder.reopen = mock.Mock(return_value="не важно")
        t = threading.Thread(target=self.app._mic_watchdog_loop, args=(0.05,), daemon=True)
        t.start()
        time.sleep(0.25)
        self.app._stopping.set()
        t.join(timeout=2.0)
        self.assertFalse(self.app.recorder.reopen.called,
                         "в PTT поток открывается по нажатию — переоткрывать нечего")

    def test_watchdog_survives_exceptions(self):
        self.app.recorder.reopen = mock.Mock(side_effect=RuntimeError("PortAudio умер"))
        t = threading.Thread(target=self.app._mic_watchdog_loop, args=(0.05,), daemon=True)
        t.start()
        time.sleep(0.2)
        self.app._stopping.set()
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(), "сторож не должен умирать от ошибки")


if __name__ == "__main__":
    unittest.main()
