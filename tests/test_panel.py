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
    app._mic_cache = (time.monotonic(), [
        {"name": "Микрофон (Ardor Gaming Vox)", "idx": 13, "alive": True, "api": "DS"}])
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


if __name__ == "__main__":
    unittest.main()
