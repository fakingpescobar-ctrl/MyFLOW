"""Тесты автолечения LLM: поднять сервер, дождаться, дополировать очередь."""
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
from myflow.db import Database, DictationRecord
from myflow.polish import PolishResult


def _app():
    tmp = tempfile.TemporaryDirectory()
    store = SettingsStore(Path(tmp.name) / "settings.json")
    app = M.MyFlowApp(settings=store, db=Database(Path(tmp.name) / "h.sqlite"))
    app._tmp = tmp
    app._mic_cache = ([], time.monotonic())
    app.paster = mock.Mock()
    return app


class DeferredPolishDbTest(unittest.TestCase):
    """Очередь недополированных фраз в базе."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "q.sqlite")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _add(self, text, status):
        return self.db.insert_history(DictationRecord(
            asr_text=text, polished_text=text, app_window="тест", duration_ms=1000,
            e2e_latency_ms=500, detected_language="ru", transcript_origin="gigaam",
            style="plain", status=status))

    def test_queue_counts_only_polish_errors(self):
        self._add("раз два три", "polish_error")
        self._add("всё хорошо", "ok")
        self._add("пустая", "asr_error")
        self._add("   ", "polish_error")            # пустой ASR в очередь не берём
        self.assertEqual(self.db.count_deferred_polish(), 1)
        rows = self.db.deferred_polish()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["asr_text"], "раз два три")

    def test_set_polished_updates_text_and_divergence(self):
        ident = self._add("раз два три", "polish_error")
        self.db.set_polished(ident, "Раз, два, три.", 0.4)
        self.db.mark_status(ident, "ok")
        row = self.db._conn.execute("SELECT * FROM history WHERE id=?", (ident,)).fetchone()
        self.assertEqual(row["polished_text"], "Раз, два, три.")
        self.assertAlmostEqual(row["divergence"], 0.4, places=3)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(self.db.count_deferred_polish(), 0)


class AutoHealTest(unittest.TestCase):
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

    # ---- 1. сервер поднимается сам ----

    def test_guard_is_called_when_llm_dead(self):
        calls = []
        self.app.set_server_guard(lambda key: calls.append(key) or True)
        self.app.polish.polish = mock.Mock(side_effect=RuntimeError("llama-server недоступен"))
        with mock.patch.object(M, "_health_check_llm", return_value=False), \
             mock.patch.object(self.app, "_wait_llm_ready", return_value=False):
            self.app._polish_safe("текст", "plain", [])
            deadline = time.monotonic() + 2.0
            while not calls and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertIn("llm", calls, "при мёртвой LLM должен звать подниматель сервера")

    def test_retry_after_wait_returns_polished(self):
        """Сервер поднялся во время ожидания — фраза полируется, а не уходит сырой."""
        self.app.polish.polish = mock.Mock(side_effect=[
            RuntimeError("llama-server оборвал соединение"),
            PolishResult(text="Раз, два, три.", latency_ms=90),
        ])
        self.app.set_server_guard(lambda key: True)
        # Живость: первая проверка (перед попыткой) — жива, значит идём в клиент,
        # он падает; дальше _revive_llm_and_retry видит живую LLM и повторяет.
        with mock.patch.object(M, "_health_check_llm", return_value=True):
            res = self.app._polish_safe("раз два три", "plain", [])
        self.assertEqual(res.text, "Раз, два, три.")
        self.assertFalse(res.skipped)
        self.assertEqual(self.app._polish_fails, 0, "успех после ожидания сбрасывает счётчик")
        self.assertEqual(self.app.polish.polish.call_count, 2, "должен быть повтор")

    def test_wait_expires_returns_raw_and_keeps_counter(self):
        self.app.polish.polish = mock.Mock(side_effect=RuntimeError("нет сервера"))
        self.app.set_server_guard(lambda key: False)
        with mock.patch.object(M, "_health_check_llm", return_value=False), \
             mock.patch.object(self.app, "_wait_llm_ready", return_value=False):
            res = self.app._polish_safe("раз два три", "plain", [])
        self.assertTrue(res.skipped, "не дождались — отдаём сырой ASR")
        self.assertEqual(res.text, "раз два три")
        self.assertGreater(self.app._polish_fails, 0)

    def test_guard_not_called_when_disabled(self):
        cfg = self.app.settings.get().polish
        cfg.auto_start_server = False
        calls = []
        self.app.set_server_guard(lambda key: calls.append(key) or True)
        with mock.patch.object(M, "_health_check_llm", return_value=False), \
             mock.patch.object(self.app, "_wait_llm_ready", return_value=False):
            self.app._polish_safe("текст", "plain", [])
        time.sleep(0.1)
        self.assertEqual(calls, [], "флаг auto_start_server выключает автолечение")

    # ---- 2. очередь дополировывается в фоне ----

    def test_repolish_loop_heals_queue(self):
        ident = self.app.db.insert_history(DictationRecord(
            asr_text="раз два три", polished_text="раз два три", app_window="тест",
            duration_ms=1000, e2e_latency_ms=500, detected_language="ru",
            transcript_origin="gigaam", style="plain", status="polish_error"))
        self.app.polish.polish = mock.Mock(
            return_value=PolishResult(text="Раз, два, три.", latency_ms=80))
        cfg = self.app.settings.get().polish
        cfg.auto_repolish = True
        cfg.auto_repolish_interval = 5.0
        with mock.patch.object(M, "_health_check_llm", return_value=True):
            self.app._ensure_repolish_worker()
            deadline = time.monotonic() + 8.0
            healed = False
            while time.monotonic() < deadline:
                if self.app.db.count_deferred_polish() == 0:
                    healed = True
                    break
                time.sleep(0.1)
        healed = False
        while time.monotonic() < deadline:
            if self.app.db.count_deferred_polish() == 0:
                healed = True
                break
            time.sleep(0.1)
        self.app._stopping.set()
        self.app._repolish_thread.join(timeout=2.0)
        self.assertTrue(healed, "фоновая дополировка должна вылечить очередь")
        row = self.app.db._conn.execute("SELECT * FROM history WHERE id=?", (ident,)).fetchone()
        self.assertEqual(row["polished_text"], "Раз, два, три.")
        self.assertEqual(row["status"], "ok")

    def test_repolish_skipped_while_llm_down(self):
        self.app.db.insert_history(DictationRecord(
            asr_text="раз два три", polished_text="раз два три", app_window="тест",
            duration_ms=1000, e2e_latency_ms=500, detected_language="ru",
            transcript_origin="gigaam", style="plain", status="polish_error"))
        self.app.polish.polish = mock.Mock()
        calls = []
        self.app.set_server_guard(lambda key: calls.append(key) or False)
        with mock.patch.object(M, "_health_check_llm", return_value=False), \
             mock.patch.object(M, "_health_check", return_value=False):
            cfg = self.app.settings.get().polish
            cfg.auto_repolish_interval = 1.0
            self.app.polish.cfg.enabled = True
            self.app._stopping.set()
            self.app._repolish_loop()          # один проход и выход (stop уже взведён)
        self.app.polish.polish.assert_not_called()
        self.assertGreaterEqual(self.app.db.count_deferred_polish(), 1)

    def test_snapshot_reports_deferred_count(self):
        self.app.db.insert_history(DictationRecord(
            asr_text="раз два три", polished_text="раз два три", app_window="тест",
            duration_ms=1000, e2e_latency_ms=500, detected_language="ru",
            transcript_origin="gigaam", style="plain", status="polish_error"))
        snap = self.app.control_snapshot()
        self.assertEqual(snap["counters"]["deferred_polish"], 1)

    def test_user_disabled_polish_is_not_re_enabled(self):
        """Выключил LLM в панели (освобождал VRAM под игру) — не поднимать обратно."""
        self.app.polish.polish = mock.Mock()
        ok, _ = self.app.control_apply("module", "llm")
        self.assertTrue(ok)
        self.assertFalse(self.app.settings.get().polish.enabled)
        self.assertTrue(self.app._polish_user_off, "флаг «выключил пользователь»")
        calls = []
        self.app.set_server_guard(lambda key: calls.append(key) or False)
        self.app.settings.get().polish.auto_repolish_interval = 1.0
        self.app._stopping.set()
        self.app._repolish_loop()          # один проход
        self.assertFalse(self.app.settings.get().polish.enabled,
                         "полировка не должна включаться сама")
        self.app.polish.polish.assert_not_called()

    def test_user_reenabled_polish_clears_flag(self):
        self.app.control_apply("module", "llm")      # выключили
        self.app.control_apply("module", "llm")      # включили
        self.assertTrue(self.app.settings.get().polish.enabled)
        self.assertFalse(self.app._polish_user_off)

    def test_autoheal_defaults_on(self):
        cfg = self.app.settings.get().polish
        self.assertTrue(cfg.auto_start_server)
        self.assertTrue(cfg.auto_repolish)
        self.assertGreater(cfg.wait_on_fail_seconds, 0)


# ---------------------------------------------------------------------------
# Тесты не должны трогать сеть и звуковое железо: фоновое сканирование
# микрофонов (PortAudio) доживало до выключения интерпретатора и роняло процесс
# повреждением кучи, а реальные HTTP-проверки тормозили прогон.
# ---------------------------------------------------------------------------
_MODULE_PATCHES: list = []


def setUpModule() -> None:
    from unittest import mock as _mock
    import myflow.main as _M
    import myflow.pick_mic as _PM
    _MODULE_PATCHES.extend([
        _mock.patch.object(_M, "_health_check_llm", return_value=False),
        _mock.patch.object(_M.MyFlowApp, "_refresh_mics_async", lambda self: None),
        _mock.patch.object(_PM, "fast_mic_list", return_value=[]),
        _mock.patch.object(_PM, "gather_mics", return_value=[]),
    ])
    for patcher in _MODULE_PATCHES:
        patcher.start()


def tearDownModule() -> None:
    for patcher in reversed(_MODULE_PATCHES):
        patcher.stop()
    _MODULE_PATCHES.clear()


if __name__ == "__main__":
    unittest.main()


class LlmStartupStateTest(unittest.TestCase):
    """Запуск LLM: «загружается» — не ошибка; полировка не гаснет на время загрузки."""

    def test_own_server_loading_is_starting_not_error(self):
        """Порт открыт, health ещё не отвечает, процесс НАШ -> «загружается»."""
        from myflow.tui import Supervisor, ServerSpec, default_specs
        spec = next(s for s in default_specs() if s.key == "llm")
        sup = Supervisor([spec], enabled=True)
        sup.health = lambda url, path, timeout=1.5: False      # модель ещё грузится
        sup.port_open = lambda port: True
        sup._own.add("llm")
        proc = mock.Mock()
        proc.poll.return_value = None                          # процесс жив
        sup._procs["llm"] = proc
        state, note = sup.status(spec)
        self.assertEqual(state, "starting", note)
        self.assertIn("загружает", note)

    def test_foreign_process_on_port_is_error(self):
        """Порт занят ЧУЖИМ процессом и health молчит -> честная ошибка."""
        from myflow.tui import Supervisor, default_specs
        spec = next(s for s in default_specs() if s.key == "llm")
        sup = Supervisor([spec], enabled=True)
        sup.health = lambda url, path, timeout=1.5: False
        sup.port_open = lambda port: True
        state, note = sup.status(spec)
        self.assertEqual(state, "error")
        self.assertIn("ЧУЖИМ", note)

    def test_polish_not_disabled_while_server_loads(self):
        """Прогрев: если LLM ожила во время ожидания — полировка остаётся включённой."""
        import types
        import myflow.main as M
        from test_main_flow import make_app
        app = make_app()
        # make_app() — заглушка: привязываем к ней НАСТОЯЩИЕ методы
        app._prewarm_llm_work = types.MethodType(M.MyFlowApp._prewarm_llm_work, app)
        app.polish.cfg.enabled = True
        app._ensure_llm_async = mock.Mock()
        app._ensure_polish_watchdog = mock.Mock()
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            return calls["n"] > 1          # первый раз «нет», второй — «да»

        with mock.patch.object(M, "_health_check_llm", flaky), \
             mock.patch.object(app, "_wait_llm_ready", lambda s: True):
            app._prewarm_llm_work()
        self.assertTrue(app.polish.cfg.enabled, "полировка не должна гаснуть")
        app._ensure_polish_watchdog.assert_not_called()
        app._ensure_llm_async.assert_called_once()

    def test_watchdog_interval_is_fast_while_polish_off(self):
        import types
        import myflow.main as M
        from test_main_flow import make_app
        app = make_app()
        app._watchdog_interval = types.MethodType(M.MyFlowApp._watchdog_interval, app)
        app.polish.cfg.enabled = False
        self.assertLessEqual(app._watchdog_interval(30.0), 5.0)
        app.polish.cfg.enabled = True
        self.assertEqual(app._watchdog_interval(30.0), 30.0)
