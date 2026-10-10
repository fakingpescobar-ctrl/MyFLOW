"""Выгрузка LLM под игры: F11, панель, авто-детект полного экрана."""
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
from myflow.tui import Supervisor, default_specs


def _app():
    tmp = tempfile.TemporaryDirectory()
    store = SettingsStore(Path(tmp.name) / "s.json")
    app = M.MyFlowApp(settings=store, db=Database(Path(tmp.name) / "h.sqlite"))
    app._tmp = tmp
    app._mic_cache = ([], time.monotonic())
    app.overlay = mock.Mock()
    app._stopping = threading.Event()
    return app


class UnloadToggleTest(unittest.TestCase):
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

    def test_unload_disables_polish_and_stops_server(self):
        stopped = []
        self.app.set_server_stopper(lambda key: stopped.append(key))
        self.app._toggle_llm_unload()
        self.assertFalse(self.app.settings.get().polish.enabled)
        self.assertTrue(self.app._polish_user_off, "автолечение не должно включать обратно")
        deadline = time.monotonic() + 2.0
        while not stopped and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(stopped, ["llm"], "сервер должен быть погашен (VRAM освобождена)")
        self.assertTrue(self.app.overlay.show.called)

    def test_toggle_back_enables_and_starts(self):
        started = []
        self.app.set_server_guard(lambda key: started.append(key) or True)
        self.app.set_server_stopper(lambda key: None)
        self.app._toggle_llm_unload()          # выгрузили
        self.app._toggle_llm_unload()          # вернули
        self.assertTrue(self.app.settings.get().polish.enabled)
        self.assertFalse(self.app._polish_user_off)
        deadline = time.monotonic() + 2.0
        while not started and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(started, ["llm"], "сервер должен подниматься в фоне")

    def test_default_hotkey_is_f11(self):
        self.assertEqual(self.app.settings.get().hotkey.unload_llm_key, "f11")

    def test_panel_toggle_frees_vram(self):
        """Enter на строке LLM в панели должен и погасить сервер, не только флаг."""
        stopped = []
        self.app.set_server_stopper(lambda key: stopped.append(key))
        self.app.control_apply("module", "llm")
        deadline = time.monotonic() + 2.0
        while not stopped and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(stopped, ["llm"])
        started = []
        self.app.set_server_guard(lambda key: started.append(key) or True)
        self.app.control_apply("module", "llm")
        deadline = time.monotonic() + 2.0
        while not started and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(started, ["llm"])
        self.assertTrue(self.app.settings.get().polish.enabled)

    def test_stopper_errors_do_not_break(self):
        self.app.set_server_stopper(mock.Mock(side_effect=RuntimeError("нет процесса")))
        self.app._toggle_llm_unload()
        time.sleep(0.2)
        self.assertFalse(self.app.settings.get().polish.enabled)


class GameWatchTest(unittest.TestCase):
    def setUp(self):
        self.app = _app()
        self.app.settings.get().polish.game_detect_seconds = 0.1

    def tearDown(self):
        self.app._stopping.set()
        self.app.db.close()
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._tmp.cleanup()

    def test_fullscreen_for_a_while_unloads(self):
        stopped = []
        self.app.set_server_stopper(lambda key: stopped.append(key))
        t = threading.Thread(target=self.app._game_watch_loop, args=(0.05,), daemon=True)
        with mock.patch("myflow.winfocus.is_fullscreen_foreground", return_value=True):
            t.start()
            deadline = time.monotonic() + 3.0
            while not stopped and time.monotonic() < deadline:
                time.sleep(0.05)
        self.app._stopping.set()
        t.join(timeout=2.0)
        self.assertTrue(stopped, "полный экран дольше порога -> LLM выгружена")
        self.assertFalse(self.app.settings.get().polish.enabled)

    def test_no_action_when_disabled(self):
        self.app.settings.get().polish.auto_unload_on_game = False
        stopped = []
        self.app.set_server_stopper(lambda key: stopped.append(key))
        t = threading.Thread(target=self.app._game_watch_loop, args=(0.05,), daemon=True)
        with mock.patch("myflow.winfocus.is_fullscreen_foreground", return_value=True):
            t.start()
            time.sleep(0.4)
        self.app._stopping.set()
        t.join(timeout=2.0)
        self.assertEqual(stopped, [], "флаг auto_unload_on_game выключает авто-выгрузку")

    def test_not_watching_when_windowed(self):
        stopped = []
        self.app.set_server_stopper(lambda key: stopped.append(key))
        t = threading.Thread(target=self.app._game_watch_loop, args=(0.05,), daemon=True)
        with mock.patch("myflow.winfocus.is_fullscreen_foreground", return_value=False):
            t.start()
            time.sleep(0.3)
        self.app._stopping.set()
        t.join(timeout=2.0)
        self.assertEqual(stopped, [])

    def test_watcher_starts_at_app_start(self):
        self.app._ensure_game_watch()
        self.assertIsNotNone(self.app._game_watch)
        self.assertTrue(self.app._game_watch.is_alive())


class SupervisorStopTest(unittest.TestCase):
    def test_stop_only_our_process(self):
        sup = Supervisor(default_specs(), enabled=True)
        spec = sup.specs[0]
        proc = mock.Mock()
        proc.poll.return_value = None
        sup._procs[spec.key] = proc
        sup._own.add(spec.key)
        self.assertTrue(sup.stop(spec))
        proc.terminate.assert_called_once()

    def test_stop_foreign_process_is_noop(self):
        sup = Supervisor(default_specs(), enabled=True)
        spec = sup.specs[0]
        self.assertFalse(sup.stop(spec), "чужой сервер не трогаем")


if __name__ == "__main__":
    unittest.main()
