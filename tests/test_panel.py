"""Тесты единого пульта-центра: снимок состояния, переключения на лету, рендер."""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import myflow.main as M
import myflow.tui as T
from myflow.config import SettingsStore
from myflow.db import Database


class _Paster:
    can_read = True

    def paste(self, text):
        return True

    def read(self):
        return ""


def _app():
    """Приложение с ВРЕМЕННЫМИ настройками и базой (живой settings.json не трогаем)."""
    tmp = tempfile.TemporaryDirectory()
    store = SettingsStore(Path(tmp.name) / "settings.json")
    app = M.MyFlowApp(settings=store, db=Database(Path(tmp.name) / "panel.sqlite"))
    app.paster = _Paster()
    app._tmp = tmp
    # Микрофоны не сканируем (это долго): подкладываем кэш панели.
    # Кэш панели: (список, время) — порядок именно такой (см. _mic_choices).
    app._mic_cache = ([
        {"name": "Микрофон (Ardor Gaming Vox)", "idx": 13, "alive": True, "api": "DS"},
        {"name": "Набор микрофонов (Kinect USB Audio)", "idx": 15, "alive": True,
         "api": "DS"},
    ], time.monotonic())
    store.get().asr.input_device = 13
    return app


class ControlSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.app = _app()

    def tearDown(self):
        self.app.db.close()
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._stopping.set()
        self.app._tmp.cleanup()

    def test_snapshot_has_all_modules(self):
        snap = self.app.control_snapshot()
        keys = {m["key"] for m in snap["modules"]}
        for expected in ("asr", "vad", "llm", "voice", "ax", "snippets",
                         "fastpath", "learning", "overlay", "keep_audio"):
            self.assertIn(expected, keys, f"модуль {expected} должен быть на панели")

    def test_snapshot_has_control_rows(self):
        snap = self.app.control_snapshot()
        keys = {v["key"] for v in snap["values"]}
        self.assertEqual(keys, {"mic", "mode", "insert", "style", "target"})

    def test_every_row_has_label_value_and_hint(self):
        snap = self.app.control_snapshot()
        for row in snap["modules"] + snap["values"]:
            self.assertTrue(row.get("label"), row)
            self.assertIn("value", row, row)
            self.assertTrue(row.get("hint"), f"у строки {row['key']} нет подсказки")

    def test_asr_label_is_short(self):
        snap = self.app.control_snapshot()
        asr = next(m for m in snap["modules"] if m["key"] == "asr")
        self.assertLessEqual(len(asr["value"]), 30, asr["value"])
        self.assertIn("GigaAM", asr["value"], "панель показывает реальный движок")

    def test_snapshot_never_raises_without_servers(self):
        with mock.patch.object(M.MyFlowApp, "_mic_choices", return_value=[]):
            snap = self.app.control_snapshot()
        self.assertTrue(snap["modules"])


class ControlApplyTest(unittest.TestCase):
    def setUp(self):
        self.app = _app()

    def tearDown(self):
        self.app.db.close()
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._stopping.set()
        self.app._tmp.cleanup()

    def test_style_switch(self):
        ok, msg = self.app.control_apply("style", "casual")
        self.assertTrue(ok, msg)
        self.assertEqual(self.app.settings.get().default_style, "casual")
        self.app.control_apply("style", "plain")

    def test_module_toggle_snippets(self):
        before = self.app.settings.get().flags["snippets"]
        ok, _ = self.app.control_apply("module", "snippets")
        self.assertTrue(ok)
        self.assertEqual(self.app.settings.get().flags["snippets"], not before)
        self.app.control_apply("module", "snippets")
        self.assertEqual(self.app.settings.get().flags["snippets"], before)

    def test_module_toggle_llm(self):
        before = self.app.settings.get().polish.enabled
        ok, msg = self.app.control_apply("module", "llm")
        self.assertTrue(ok, msg)
        self.assertEqual(self.app.settings.get().polish.enabled, not before)
        self.app.control_apply("module", "llm")
        self.assertEqual(self.app.settings.get().polish.enabled, before)

    def test_vad_threshold_live(self):
        ok, msg = self.app.control_apply("vad_threshold", 0.42)
        self.assertTrue(ok, msg)
        self.assertAlmostEqual(self.app.recorder.vad_threshold, 0.42, places=3)
        self.app.control_apply("vad_threshold", 0.30)

    def test_voice_mode_cycles(self):
        s = self.app.settings.get().voiceprint
        seen = set()
        for _ in range(3):
            ok, _ = self.app.control_apply("voice_mode", 1)
            self.assertTrue(ok)
            seen.add(s.mode)
        self.assertEqual(seen, {"auto", "always", "off"}, "цикл из трёх режимов")

    def test_unknown_action_reports_error(self):
        ok, msg = self.app.control_apply("нет-такого", 1)
        self.assertFalse(ok)
        self.assertIn("неизвестное", msg)

    def test_module_switch_asr_backend(self):
        """Переключение на whisper только если сервер реально поднимается."""
        # без сервера — отказ, движок и настройка остаются прежними
        ok, msg = self.app.control_apply("module", "asr")
        self.assertFalse(ok, msg)
        self.assertIn("whisper", msg)
        # сервер поднимается — переключаемся
        self.app.set_server_guard(lambda key: True)
        ok, msg = self.app.control_apply("module", "asr")
        self.assertTrue(ok, msg)
        self.assertEqual(type(self.app.whisper).__name__, "WhisperClient")


class PanelRenderTest(unittest.TestCase):
    def _snap(self):
        return {"modules": [
            {"key": "asr", "label": "ASR", "value": "GigaAM v3 e2e (CPU)", "on": True,
             "ok": True, "note": "", "hint": "Enter — сменить движок"},
            {"key": "vad", "label": "VAD", "value": "Silero v6 · 0.30", "on": True,
             "ok": True, "note": "", "hint": "←→ порог"},
            {"key": "llm", "label": "LLM", "value": "qwen3-4b · :8082", "on": False,
             "ok": False, "note": "", "hint": "Enter вкл/выкл"},
        ], "values": [
            {"key": "mic", "label": "Микрофон", "value": "Ardor Gaming Vox [DS]",
             "hint": "←→ сменить микрофон"},
            {"key": "mode", "label": "Режим диктовки", "value": "авто (слушает)",
             "hint": "←→ авто ↔ PTT"},
        ], "counters": {"phrases": 3, "pending": 0, "queue": 0}}

    def test_panel_shows_modules_values_and_keys(self):
        frame = T.render_panel(self._snap(), 120, 34, "режим: авто",
                               ["строка журнала"], 0, 0, level=(0.05, ""),
                               status_line="✓ вставлено")
        plain = "\n".join(T.strip_ansi(x) for x in frame)
        for needle in ("Движки и модули", "Управление", "ASR", "GigaAM", "VAD",
                       "Silero", "LLM", "[вкл]", "[выкл]", "Микрофон",
                       "Ardor", "уровень", "Журнал", "строка журнала",
                       "F6", "F7", "F8", "F9", "F10", "PgUp/PgDn"):
            self.assertIn(needle, plain, f"на панели нет «{needle}»")

    def test_panel_selection_visible(self):
        frame = T.render_panel(self._snap(), 120, 34, "cfg", ["ж"], 0, 1,
                               level=(0.05, ""))
        self.assertIn("▸", "\n".join(T.strip_ansi(x) for x in frame),
                      "выделенная строка помечена")

    def test_panel_fits_any_width(self):
        for w, h in ((120, 34), (100, 26), (80, 24)):
            frame = T.render_panel(self._snap(), w, h, "cfg", ["ж"], 0, 0,
                                   level=(0.05, ""))
            self.assertEqual(len(frame), h)
            for ln in frame:
                self.assertEqual(T.vlen(ln), w, f"строка не по ширине {w}")

    def test_panel_scroll_indicator(self):
        log = [f"строка {i}" for i in range(50)]
        frame = T.render_panel(self._snap(), 120, 34, "cfg", log, 10, 0,
                               level=(0.05, ""))
        plain = "\n".join(T.strip_ansi(x) for x in frame)
        self.assertIn("прокрутка", plain)
        self.assertNotIn("строка 49", plain, "живой хвост ушёл из вида")

    def test_panel_legend_never_cut(self):
        frame = T.render_panel(self._snap(), 120, 34, "cfg", ["ж"], 0, 0,
                               level=(0.05, ""))
        plain = "\n".join(T.strip_ansi(x) for x in frame)
        self.assertIn("выход", plain, "легенда целиком видна")
        self.assertIn("пересканировать микрофоны", plain)


class PanelSpeedTest(unittest.TestCase):
    """Регресс: панель вставала на 11 секунд (скан микрофонов в цикле отрисовки)."""

    def setUp(self):
        self.app = _app()

    def tearDown(self):
        self.app.db.close()
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._stopping.set()
        self.app._tmp.cleanup()

    def test_snapshot_is_fast_without_cache(self):
        self.app._mic_cache = None
        t0 = time.perf_counter()
        for _ in range(10):
            self.app.control_snapshot()
            self.app._mic_cache = None          # худший случай: кэша нет
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 2.0,
                        f"снимок панели не должен сканировать микрофоны ({elapsed:.1f} с)")

    def test_mic_switch_is_instant(self):
        t0 = time.perf_counter()
        ok, msg = self.app.control_apply("mic", 15)
        elapsed = time.perf_counter() - t0
        self.assertTrue(ok, msg)
        self.assertLess(elapsed, 0.5, f"смена микрофона должна быть мгновенной ({elapsed:.2f} с)")

    def test_rescan_returns_immediately(self):
        t0 = time.perf_counter()
        ok, msg = self.app.control_apply("rescan_mics")
        elapsed = time.perf_counter() - t0
        self.assertTrue(ok, msg)
        self.assertLess(elapsed, 0.5, "F5 не должен блокировать панель")

    def test_scan_is_skipped_while_recording(self):
        from unittest import mock
        self.app.recorder._stream = object()          # «идёт запись»
        self.app._mic_cache = None
        with mock.patch("myflow.pick_mic.gather_mics") as scan:
            self.app._mic_choices()
            self.app._refresh_mics_async()
            if self.app._mic_thread is not None:
                self.app._mic_thread.join(timeout=3.0)
        scan.assert_not_called()
        self.app.recorder._stream = None


class PanelInteractTest(unittest.TestCase):
    """Регресс: стрелки на строке микрофона не работали («кнопки не нажимаются»)."""

    def setUp(self):
        self.app = _app()
        self.tui = T.TuiApp(no_servers=True)
        self.tui.app = self.app
        self.tui.store = self.app.settings        # как делает run(): один стор

    def tearDown(self):
        self.app.db.close()
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._stopping.set()
        self.app._tmp.cleanup()

    def _rows(self):
        snap = self.app.control_snapshot()
        return snap["modules"], snap["values"]

    def test_mic_row_is_treated_as_value_row(self):
        """Строка микрофона — не модуль: раньше нажатие уходило в «неизвестный модуль»."""
        modules, values = self._rows()
        mic_index = len(modules) + next(i for i, v in enumerate(values)
                                        if v["key"] == "mic")
        before = self.app.settings.get().asr.input_device
        self.tui._panel_apply(modules, values, mic_index, "right")
        after = self.app.settings.get().asr.input_device
        self.assertNotEqual(after, before, "←→ на строке микрофона должен её менять")
        self.assertNotIn("неизвестный модуль", "\n".join(self.tui.state.log))

    def test_mic_cycle_skips_dead_and_cycles_all(self):
        self.app._mic_cache = ([
            {"name": "Ardor Gaming Vox", "idx": 13, "alive": True, "api": "DS"},
            {"name": "Kinect USB Audio", "idx": 15, "alive": True, "api": "DS"},
            {"name": "Мёртвый", "idx": 2, "alive": False, "api": "MME"},
        ], time.monotonic())
        with mock.patch.object(M.MyFlowApp, "_refresh_mics_async", lambda self: None):
            seen = []
            for _ in range(3):
                self.tui._cycle_mic(1)
                seen.append(self.app.settings.get().asr.input_device)
        self.assertEqual(seen, [15, None, 13],
                         "цикл: 13 -> 15 -> системный -> 13, мёртвые пропускаем")

    def test_module_toggle_from_panel(self):
        modules, values = self._rows()
        idx = next(i for i, m in enumerate(modules) if m["key"] == "snippets")
        before = self.app.settings.get().flags["snippets"]
        self.tui._panel_apply(modules, values, idx, "enter")
        self.assertEqual(self.app.settings.get().flags["snippets"], not before)

    def test_style_row_changes_style(self):
        modules, values = self._rows()
        idx = len(modules) + next(i for i, v in enumerate(values) if v["key"] == "style")
        before = self.app.settings.get().default_style
        self.tui._panel_apply(modules, values, idx, "right")
        self.assertNotEqual(self.app.settings.get().default_style, before)

    def test_tui_uses_app_store(self):
        """TUI и приложение должны смотреть в ОДНИ настройки."""
        self.assertIs(self.tui._store(), self.app.settings)


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


class RouterToggleTest(unittest.TestCase):
    """Авто-выбор движка переключается из панели (строка «Авто-выбор»)."""

    def setUp(self):
        from unittest import mock
        self.app = _app()                       # реальное приложение, временные настройки
        self.app.overlay = mock.Mock()
        self.app._ensure_server = lambda key: True       # сервер «поднимается»
        self.app._stop_server = mock.Mock()
        self.app._ensure_second_opinion = mock.Mock()

    def tearDown(self):
        self.app.db.close()
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._stopping.set()
        self.app._tmp.cleanup()

    def _row(self):
        return next(m for m in self.app.control_snapshot()["modules"]
                    if m["key"] == "router")

    def test_row_present_and_on_by_default(self):
        row = self._row()
        self.assertTrue(row["on"])
        self.assertIn("whisper", row["value"])
        self.assertIn("второе мнение", row["note"])
        self.assertIn("Enter", row["hint"])

    def test_toggle_off_then_on(self):
        self.app.settings.get().asr.router = True
        ok, message = self.app.control_apply("module", "router")
        self.assertTrue(ok)
        self.assertFalse(self.app.settings.get().asr.router)
        self.assertIn("выключен", message)
        ok, message = self.app.control_apply("module", "router")
        self.assertTrue(ok)
        self.assertTrue(self.app.settings.get().asr.router)
        self.assertIn("включён", message)

    def test_row_reflects_off_state(self):
        self.app.settings.get().asr.router = False
        row = self._row()
        self.assertFalse(row["on"])
        self.assertIn("только GigaAM", row["value"])

    def test_enable_refuses_when_server_unavailable(self):
        self.app.settings.get().asr.router = False
        self.app._ensure_server = lambda key: False
        ok, message = self.app.control_apply("module", "router")
        self.assertFalse(ok)
        self.assertFalse(self.app.settings.get().asr.router,
                         "настройка не должна включаться без сервера")
        self.assertIn("НЕ", message)

    def test_disable_stops_whisper_server(self):
        """Выключили авто-выбор при GigaAM — whisper-сервер гасим (VRAM)."""
        self.app.settings.get().asr.router = True
        self.app.control_apply("module", "router")
        self.app._stop_server.assert_called_once_with("asr")

    def test_toggle_does_not_touch_live_settings(self):
        """Прогон панели не должен писать в живой data/settings.json."""
        live = Path(__file__).resolve().parent.parent / "data" / "settings.json"
        before = live.read_text(encoding="utf-8") if live.exists() else ""
        self.app.control_apply("module", "router")
        self.app.control_apply("module", "router")
        after = live.read_text(encoding="utf-8") if live.exists() else ""
        self.assertEqual(before, after, "живые настройки изменены тестом")


class JudgeToggleTest(unittest.TestCase):
    """Oracle LLM включается из панели (строка «Oracle LLM»)."""

    def setUp(self):
        from unittest import mock
        self.app = _app()
        self.app.overlay = mock.Mock()
        self.app._ensure_server = lambda key: True
        self.app._stop_server = mock.Mock()
        self.app._ensure_second_opinion = mock.Mock()
        self.app.polish = mock.Mock()
        self.app.polish.alive.return_value = True
        # Режимы несовместимы: для теста судьи авто-выбор должен быть выключен.
        self.app.settings.get().asr.router = False

    def tearDown(self):
        self.app.db.close()
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._stopping.set()
        self.app._tmp.cleanup()

    def _row(self):
        return next(m for m in self.app.control_snapshot()["modules"]
                    if m["key"] == "judge")

    def test_row_present_and_off_by_default(self):
        row = self._row()
        self.assertFalse(row["on"], "Oracle по умолчанию выключен (дорого по времени)")
        self.assertEqual(row["value"], "выключен")
        self.assertIn("Enter", row["hint"])

    def test_toggle_on_then_off(self):
        ok, message = self.app.control_apply("module", "judge")
        self.assertTrue(ok, message)
        self.assertTrue(self.app.settings.get().asr.judge)
        self.assertIn("включён", message)
        self.assertTrue(self._row()["on"])
        ok, message = self.app.control_apply("module", "judge")
        self.assertTrue(ok)
        self.assertFalse(self.app.settings.get().asr.judge)
        self.assertIn("выключен", message)

    def test_enable_requires_llm(self):
        """LLM лежит и поднять не удалось -> Oracle не включаем.

        Серверы грузятся лениво (полировка идёт либо на 4B, либо на
        постпроцессоре), поэтому при включении Oracle приложение СНАЧАЛА пытается
        поднять LLM и только потом отказывает.
        """
        self.app.polish.alive.return_value = False
        self.app._ensure_server = lambda key: False
        ok, message = self.app.control_apply("module", "judge")
        self.assertFalse(ok)
        self.assertIn("LLM", message)
        self.assertFalse(self.app.settings.get().asr.judge)

    def test_enable_starts_llm_lazily_when_needed(self):
        """LLM не запущена (её не грузили при старте), но режим включают — поднимаем."""
        self.app.polish.alive.return_value = False
        started = []
        self.app._ensure_server = lambda key: (started.append(key), True)[1]
        ok, message = self.app.control_apply("module", "judge")
        self.assertIn("llm", started, "Oracle должен поднять LLM лениво")
        self.assertTrue(ok, message)

    def test_enable_requires_whisper_server(self):
        self.app._ensure_server = lambda key: False
        ok, message = self.app.control_apply("module", "judge")
        self.assertFalse(ok)
        self.assertFalse(self.app.settings.get().asr.judge)

    def test_judge_needs_whisper_server_at_startup(self):
        from myflow.asr import whisper_needed
        cfg = self.app.settings.get().asr
        cfg.judge = True
        cfg.router = False
        self.assertTrue(whisper_needed(cfg), "судье нужен whisper-сервер")


class RouterJudgeCombinationTest(unittest.TestCase):
    """Режимы несовместимы: второй НЕ включается, пока первый активен."""

    def setUp(self):
        from unittest import mock
        self.app = _app()
        self.app.overlay = mock.Mock()
        self.app._ensure_server = lambda key: True
        self.app._stop_server = mock.Mock()
        self.app._ensure_second_opinion = mock.Mock()
        self.app.polish = mock.Mock()
        self.app.polish.alive.return_value = True

    def tearDown(self):
        self.app.db.close()
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._stopping.set()
        self.app._tmp.cleanup()

    def _cfg(self):
        return self.app.settings.get().asr

    def test_router_cannot_be_enabled_while_judge_on(self):
        cfg = self._cfg()
        cfg.judge = True
        cfg.router = False
        ok, message = self.app.control_apply("module", "router")
        self.assertFalse(ok, "должен быть отказ, а не автопереключение")
        self.assertFalse(cfg.router, "состояние не меняется")
        self.assertTrue(cfg.judge, "активный режим остаётся")
        self.assertIn("Oracle LLM", message)

    def test_judge_cannot_be_enabled_while_router_on(self):
        cfg = self._cfg()
        cfg.router = True
        cfg.judge = False
        ok, message = self.app.control_apply("module", "judge")
        self.assertFalse(ok)
        self.assertFalse(cfg.judge)
        self.assertTrue(cfg.router)
        self.assertIn("Авто-выбор", message)

    def test_after_disabling_first_mode_second_enables(self):
        cfg = self._cfg()
        cfg.router = True
        cfg.judge = False
        self.assertFalse(self.app.control_apply("module", "judge")[0])
        # выключаем активный режим — теперь судью включить можно
        self.assertTrue(self.app.control_apply("module", "router")[0])
        self.assertFalse(cfg.router)
        self.assertTrue(self.app.control_apply("module", "judge")[0])
        self.assertTrue(cfg.judge)

    def test_judge_alone_is_enough(self):
        from myflow.asr import whisper_needed
        cfg = self._cfg()
        cfg.router = False
        cfg.judge = False
        self.assertTrue(self.app.control_apply("module", "judge")[0])
        self.assertTrue(whisper_needed(cfg), "судье нужен whisper-сервер")

    def test_both_disabled_means_no_second_opinion_at_all(self):
        from myflow.asr import whisper_needed
        cfg = self._cfg()
        cfg.router = False
        cfg.judge = False
        self.assertFalse(whisper_needed(cfg))

    def test_rows_reflect_state(self):
        cfg = self._cfg()
        cfg.judge = True
        cfg.router = False
        rows = {m["key"]: m for m in self.app.control_snapshot()["modules"]}
        self.assertTrue(rows["judge"]["on"])
        self.assertFalse(rows["router"]["on"])
        self.assertIn("несовместим",
                      rows["judge"]["hint"] + rows["router"]["hint"],
                      "подсказки должны предупреждать о несовместимости")


class IncompatibleModesJournalTest(unittest.TestCase):
    """Отказ виден в журнале панели, а не только в настройке."""

    def test_journal_shows_refusal(self):
        from unittest import mock
        import myflow.tui as T

        app = _app()
        app.overlay = mock.Mock()
        app._ensure_server = lambda key: True
        app._stop_server = mock.Mock()
        app._ensure_second_opinion = mock.Mock()
        app.polish = mock.Mock()
        app.polish.alive.return_value = True
        try:
            cfg = app.settings.get().asr
            cfg.judge = True
            cfg.router = False
            tui = T.TuiApp(no_servers=True)
            tui.app = app
            tui._app_owned = False
            snap = app.control_snapshot()
            index = next(i for i, m in enumerate(snap["modules"]) if m["key"] == "router")
            tui._panel_apply(snap["modules"], snap["values"], index, "enter")
            journal = "\n".join(tui.state.log)
            self.assertIn("несовместимы", journal)
            # norm_text() заменяет ✗ на ASCII-маркер, поэтому проверяем любой из них
            self.assertTrue("!" in journal or "✗" in journal,
                            "отказ должен быть помечен как проблема")
            self.assertFalse(cfg.router)
        finally:
            app.db.close()
            try:
                app.ax.stop()
            except Exception:
                pass
            app._stopping.set()
            app._tmp.cleanup()
