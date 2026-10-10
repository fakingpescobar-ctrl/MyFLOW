"""Подтверждение смены движка ASR: Enter на строке ASR не должен менять всё молча.

Реальный случай: случайное нажатие Enter на строке «ASR» переключало движок на
whisper и сохраняло настройку; сервер :8081 не поднимался, и распознавание
переставало работать. Теперь сначала спрашиваем «уверен?» (Да — Enter/y, Нет — n/Esc).
"""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import myflow.tui as T
from myflow.config import SettingsStore
from myflow.db import Database
import myflow.main as M


def _tui(script: list[str]):
    """TuiApp с приложением на временных настройках и скриптом клавиш."""
    tmp = tempfile.TemporaryDirectory()
    store = SettingsStore(Path(tmp.name) / "s.json")
    app = M.MyFlowApp(settings=store, db=Database(Path(tmp.name) / "h.sqlite"))
    app._mic_cache = ([], time.monotonic())
    app.overlay = mock.Mock()
    app._stopping = threading.Event()

    tui = T.TuiApp(no_servers=True)
    tui.app = app
    tui._app_owned = False       # приложение наше, TUI его не пересоздаёт и не гасит
    tui.store = store
    tui.state.add_log("тест")

    frames: list[list[str]] = []

    def fake_draw(self, lines):
        frames.append(list(lines))

    seq = iter(script)

    def fake_key(timeout=0.05):
        key = next(seq, "q")
        time.sleep(0.01)
        return key

    return tmp, store, app, tui, frames, fake_draw, fake_key


def _run(script: list[str], confirm_switch: bool | None = None):
    ctx = _tui(script)
    if confirm_switch is not None:
        ctx[1].get().asr.confirm_switch = confirm_switch
    tmp, store, app, tui, frames, fake_draw, fake_key = ctx
    with mock.patch.object(T.Console, "draw", fake_draw), \
         mock.patch.object(T.Console, "size", lambda self: (100, 26)), \
         mock.patch.object(T.Console, "clear", lambda self: None), \
         mock.patch.object(T.Console, "cursor", lambda self, v: None), \
         mock.patch.object(T.Console, "reset", lambda self: None), \
         mock.patch.object(T.TuiApp, "_read_key", staticmethod(fake_key)):
        tui._dashboard_phase()
    app._stopping.set()
    app.db.close()
    try:
        app.ax.stop()
    except Exception:
        pass
    return tmp, store, app, tui, frames


def _select_asr(tui, app) -> int:
    snap = app.control_snapshot()
    modules = snap.get("modules", [])
    return next(i for i, m in enumerate(modules) if m.get("key") == "asr")


class ConfirmModalTest(unittest.TestCase):
    def test_enter_on_asr_asks_instead_of_switching(self):
        tmp, store, app, tui, frames = _run(["enter", "n", "q"])
        try:
            self.assertIsNone(tui._confirm, "после «нет» окно закрыто")
            self.assertNotIn("whisper", str(store.get().asr.backend))
            text = "\n".join(T.strip_ansi(x) for f in frames for x in f)
            self.assertIn("Подтверждение", text, "окно подтверждения должно быть на экране")
            self.assertIn("Переключить распознавание", text)
            self.assertIn("Нет — n / Esc", text)
        finally:
            tmp.cleanup()

    def test_cancel_keeps_engine(self):
        tmp, store, app, tui, frames = _run(["enter", "n", "q"])
        try:
            self.assertEqual(store.get().asr.backend, "auto")
            self.assertIn("отменено", "\n".join(tui.state.log))
        finally:
            tmp.cleanup()

    def test_confirm_applies_switch(self):
        """«Да» приводит к попытке смены движка (её исход решает приложение)."""
        tmp, store, app, tui, frames = _run(["enter", "y", "q"])
        try:
            log = "\n".join(tui.state.log)
            self.assertIn("подтверждено", log)
            self.assertIsNone(tui._confirm)
        finally:
            tmp.cleanup()

    def test_other_keys_ignored_while_modal_open(self):
        """Пока окно открыто, стрелки/переключения не работают (кроме ответа)."""
        tmp, store, app, tui, frames = _run(["down", "enter", "down", "down", "n", "q"])
        try:
            # «n» закрыло окно; смена движка не произошла
            self.assertIsNone(tui._confirm)
            self.assertEqual(store.get().asr.backend, "auto")
        finally:
            tmp.cleanup()

    def test_confirm_switch_flag_disables_question(self):
        """С выключенным флагом вопрос не задаётся — действие применяется сразу."""
        tmp, store, app, tui, frames = _run(["enter", "q"], confirm_switch=False)
        try:
            self.assertFalse(store.get().asr.confirm_switch)
            text = "\n".join(T.strip_ansi(x) for f in frames for x in f)
            self.assertNotIn("Подтверждение", text, "окна быть не должно")
            self.assertIsNone(tui._confirm)
        finally:
            tmp.cleanup()

    def test_live_settings_untouched(self):
        """Скриптовые прогоны панели не должны писать в живой settings.json."""
        from myflow.config import SETTINGS_FILE
        before = SETTINGS_FILE.read_bytes() if SETTINGS_FILE.exists() else None
        tmp, store, app, tui, frames = _run(["down", "enter", "y", "n", "q"])
        try:
            after = SETTINGS_FILE.read_bytes() if SETTINGS_FILE.exists() else None
            self.assertEqual(before, after, "живой settings.json изменён прогоном панели")
        finally:
            tmp.cleanup()

    def test_question_depends_on_target_engine(self):
        tmp, store, app, tui, frames = _run(["q"])
        try:
            snap = app.control_snapshot()
            modules = snap.get("modules", [])
            values = snap.get("values", [])
            sel = _select_asr(tui, app)
            app.whisper = mock.Mock(backend_name="gigaam")
            tui._ask_asr_confirm(modules, values, sel, "enter")
            self.assertIn("whisper", tui._confirm["question"])
            app.whisper = mock.Mock(backend_name="whisper")
            tui._ask_asr_confirm(modules, values, sel, "enter")
            self.assertIn("GigaAM", tui._confirm["question"])
            self.assertIn("VRAM", tui._confirm["hint"])
        finally:
            tmp.cleanup()


class ModalRenderTest(unittest.TestCase):
    def _frame(self, confirm=None):
        snap = {"modules": [
            {"key": "asr", "label": "ASR", "value": "GigaAM v3 e2e (CPU)", "on": True,
             "ok": True, "note": "", "hint": "Enter — сменить движок"},
        ], "values": [], "counters": {}}
        return T.render_panel(snap, 100, 26, "режим: авто", ["строка журнала"], 0, 0,
                              level=(0.05, ""), confirm=confirm)

    def test_widths_stay_exact_with_modal(self):
        frame = self._frame(("Переключить распознавание на whisper?",
                             "Будет поднят whisper-сервер на :8081"))
        self.assertEqual(len(frame), 26)
        for ln in frame:
            self.assertEqual(T.vlen(ln), 100, "модальное окно не должно ломать ширину")

    def test_modal_text_visible(self):
        frame = self._frame(("Сменить движок?", "Подсказка про VRAM"))
        plain = "\n".join(T.strip_ansi(x) for x in frame)
        for needle in ("Подтверждение", "Сменить движок?", "Да — Enter / y",
                       "Нет — n / Esc", "Подсказка про VRAM"):
            self.assertIn(needle, plain)

    def test_long_question_is_truncated_not_wrapped(self):
        frame = self._frame(("Очень длинный вопрос " * 6, "и длинная подсказка " * 6))
        for ln in frame:
            self.assertEqual(T.vlen(ln), 100, "длинный вопрос не должен переносить строку")


if __name__ == "__main__":
    unittest.main()
