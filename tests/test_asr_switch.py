"""Переключение ASR-движка не должно ломать диктовку.

Реальный случай: в панели нажали Enter на строке ASR -> движок сменился на
whisper и это записалось в настройки, а whisper-сервер (:8081) не поднялся.
Итог: все фразы падали с «whisper-server недоступен», ASR «сломался».
"""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import myflow.main as M
from myflow.asr import WhisperClient
from myflow.asr_gigaam import GigaAmClient
from myflow.config import SETTINGS_FILE, SettingsStore
from myflow.db import Database


def _app():
    tmp = tempfile.TemporaryDirectory()
    store = SettingsStore(Path(tmp.name) / "s.json")
    app = M.MyFlowApp(settings=store, db=Database(Path(tmp.name) / "h.sqlite"))
    app._tmp = tmp
    app._mic_cache = ([], time.monotonic())
    app.overlay = mock.Mock()
    app._stopping = threading.Event()
    return app


class _Rec:
    pcm_int16 = np.zeros(16000, dtype="int16")
    duration_ms = 1000
    max_amplitude = 0.5


class AsrSwitchSafetyTest(unittest.TestCase):
    def setUp(self):
        self.app = _app()

    def tearDown(self):
        self.app._stopping.set()
        self.app.db.close()
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._tmp.cleanup()

    def test_switch_to_whisper_refused_without_server(self):
        """Нет сервера — не переключаемся (иначе ASR «ломается»)."""
        before_backend = self.app.settings.get().asr.backend
        before_client = type(self.app.whisper).__name__
        ok, msg = self.app.control_apply("module", "asr")
        self.assertFalse(ok, msg)
        self.assertIn("whisper", msg)
        self.assertEqual(self.app.settings.get().asr.backend, before_backend,
                         "настройка не должна поменяться на нерабочий движок")
        self.assertEqual(type(self.app.whisper).__name__, before_client)

    def test_switch_to_whisper_allowed_with_server(self):
        self.app.set_server_guard(lambda key: True)
        ok, msg = self.app.control_apply("module", "asr")
        self.assertTrue(ok, msg)
        self.assertIsInstance(self.app.whisper, WhisperClient)
        self.assertEqual(self.app.settings.get().asr.backend, "whisper")

    def test_switch_back_unloads_whisper(self):
        self.app.set_server_guard(lambda key: True)
        stopped = []
        self.app.set_server_stopper(lambda key: stopped.append(key))
        self.app.control_apply("module", "asr")          # -> whisper
        ok, msg = self.app.control_apply("module", "asr")  # -> gigaam
        self.assertTrue(ok, msg)
        self.assertIsInstance(self.app.whisper, GigaAmClient)
        deadline = time.monotonic() + 2.0
        while not stopped and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(stopped, ["asr"], "whisper-сервер выгружается (VRAM)")

    def test_runtime_failure_falls_back_to_gigaam(self):
        """whisper отказал на живой фразе -> распознаём GigaAM, не теряем фразу."""
        s = self.app.settings.get()
        s.asr.backend = "whisper"
        self.app.whisper = WhisperClient(s.asr)
        self.app.whisper.transcribe = mock.Mock(
            side_effect=RuntimeError("whisper-server недоступен (8081)"))
        result = self.app._asr_fallback(_Rec(), RuntimeError("whisper-server недоступен"))
        self.assertIsNotNone(result, "резерв GigaAM должен сработать")
        self.assertIsInstance(self.app.whisper, GigaAmClient)
        self.assertEqual(s.asr.backend, "gigaam", "не рабочий движок не сохраняем")
        self.assertTrue(self.app.overlay.show.called)

    def test_fallback_returns_none_when_already_gigaam(self):
        self.assertIsNone(self.app._asr_fallback(_Rec(), RuntimeError("boom")))


class LiveSettingsIsolationTest(unittest.TestCase):
    """Тесты/пробы не должны менять ЖИВОЙ settings.json (так сломался ASR)."""

    def test_temp_store_does_not_touch_live_settings(self):
        before = SETTINGS_FILE.read_bytes() if SETTINGS_FILE.exists() else None
        app = _app()
        try:
            app.control_apply("style", "casual")
            app.control_apply("module", "llm")
            app.control_apply("module", "snippets")
            if app._server_guard is None:
                app.control_apply("module", "asr")     # отказ, настройку не пишем
        finally:
            app._stopping.set()
            app.db.close()
            try:
                app.ax.stop()
            except Exception:
                pass
            app._tmp.cleanup()
        after = SETTINGS_FILE.read_bytes() if SETTINGS_FILE.exists() else None
        self.assertEqual(before, after,
                         "живой data/settings.json изменён — так теряются настройки "
                         "пользователя (например, ASR переключается на whisper)")


if __name__ == "__main__":
    unittest.main()
