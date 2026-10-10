"""Старт приложения не должен ждать серверы и AX.

Замер до правок: `start()` занимал ~554 мс, из них ~350 мс — блокирующая проверка
LLM (при лежащем сервере порт «висит» до connect-таймаута) и ~190 мс — ожидание
ответа PowerShell-хелпера AX. Пульт в это время не появлялся.
"""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import myflow.main as M
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
    # Оверлей (Tk) в тесте не нужен: его поток занимает GIL и искажает замер
    # времени старта. В TUI подставляется свой лёгкий оверлей.
    app.settings.get().flags["overlay"] = False
    app.overlay = mock.Mock()
    # Прогрев ASR грузит ONNX-модель (~0.6 с) и держит GIL: в тесте он искажает
    # замер (конкуренция, а не блокировка). Здесь он не нужен.
    app._warm_asr = lambda: None
    return tmp, app


class FastStartTest(unittest.TestCase):
    def test_start_does_not_wait_for_llm_or_ax(self):
        """Даже если проверки «медленные», start() возвращается быстро."""
        tmp, app = _app()
        try:
            def slow_health(*a, **k):
                time.sleep(0.5)
                return False

            with mock.patch.object(M, "_health_check_llm", slow_health), \
                 mock.patch.object(app, "_start_ax", lambda: time.sleep(0.5)):
                t0 = time.perf_counter()
                app.start()
                elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.25,
                            f"start() ждал блокирующие проверки: {elapsed*1000:.0f} мс")
        finally:
            app.stop()
            app._worker_stop.set()
            app._stopping.set()
            app.db.close()
            tmp.cleanup()

    def test_ax_starts_in_background_thread(self):
        tmp, app = _app()
        try:
            started = threading.Event()

            def fake_start_ax():
                started.set()

            with mock.patch.object(app, "_start_ax", fake_start_ax):
                app.start()
            self.assertTrue(started.wait(2.0), "AX должен подниматься в фоне")
        finally:
            app.stop()
            app._worker_stop.set()
            app._stopping.set()
            app.db.close()
            tmp.cleanup()

    def test_llm_prewarm_does_not_block_start(self):
        tmp, app = _app()
        try:
            called = threading.Event()

            def slow_health(*a, **k):
                called.set()
                time.sleep(0.3)
                return False

            with mock.patch.object(M, "_health_check_llm", slow_health):
                t0 = time.perf_counter()
                app.start()
                elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.25, "прогрев LLM не должен задерживать старт")
            self.assertTrue(called.wait(2.0), "прогрев всё равно выполняется (в фоне)")
        finally:
            app.stop()
            app._worker_stop.set()
            app._stopping.set()
            app.db.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
