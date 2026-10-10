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
        client = mock.Mock(backend_name="gigaam", warmup=lambda: True)
        with mock.patch.object(M, "make_asr_client", return_value=client) as make:
            ok, msg = self.app.control_apply("module", "asr")
        self.assertTrue(ok, msg)
        self.assertTrue(make.called, "движок ASR пересоздаётся на лету")
        self.assertIs(self.app.whisper, client)


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
