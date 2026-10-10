"""Гигиена потоков: после stop() не остаётся зомби-потоков.

Реальная находка: `stop()` не гасил четыре фоновых цикла (`phrase-worker`,
`audio-worker`, `polish-watchdog`, `listener-watchdog`) — при перезапуске пульта
(клавиша `r` создаёт новое приложение) потоки копились, а старые воркеры могли
продолжать разбирать аудио и вставлять текст повторно.
"""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import myflow.main as M
from myflow.audio import DictationRecorder
from myflow.config import SettingsStore
from myflow.db import Database


def _app():
    tmp = tempfile.TemporaryDirectory()
    app = M.MyFlowApp(settings=SettingsStore(Path(tmp.name) / "s.json"),
                      db=Database(Path(tmp.name) / "h.sqlite"))
    app._mic_cache = ([], time.monotonic())
    app.recorder.start_continuous = lambda *a, **k: None
    app.recorder.stop_continuous = lambda *a, **k: None
    app.set_server_guard(lambda key: False)
    app.set_server_stopper(lambda key: None)
    return tmp, app


class ThreadHygieneTest(unittest.TestCase):
    def test_threads_do_not_accumulate_on_restarts(self):
        """Три старта/стопа подряд — потоки должны вернуться к исходному числу."""
        base = threading.active_count()
        for _ in range(3):
            tmp, app = _app()
            try:
                app.start()
                app.stop()
            finally:
                app._worker_stop.set()
                app._stopping.set()
                app.db.close()
                tmp.cleanup()
        deadline = time.monotonic() + 5.0
        while threading.active_count() > base and time.monotonic() < deadline:
            time.sleep(0.1)
        leftovers = [t.name for t in threading.enumerate()
                     if t.name not in ("MainThread",)]
        self.assertLessEqual(threading.active_count(), base,
                             f"потоки остались после stop(): {leftovers}")

    def test_stop_sets_worker_event(self):
        tmp, app = _app()
        try:
            app.start()
            app.stop()
            self.assertTrue(app._worker_stop.is_set(),
                            "воркер фраз должен получать сигнал остановки")
        finally:
            app.db.close()
            tmp.cleanup()

    def test_recorder_shutdown_stops_worker(self):
        rec = DictationRecorder()
        self.assertTrue(rec._worker.is_alive(), "воркер запускается с рекордером")
        rec.shutdown()
        self.assertFalse(rec._worker.is_alive(), "после shutdown() воркер завершён")

    def test_shutdown_is_idempotent(self):
        rec = DictationRecorder()
        rec.shutdown()
        rec.shutdown()          # повторный вызов не должен падать
        self.assertFalse(rec._worker.is_alive())


if __name__ == "__main__":
    unittest.main()
