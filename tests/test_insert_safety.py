"""Бережная вставка: не уничтожаем картинки/файлы в буфере обмена."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import myflow.insertion as I


class RichClipboardTest(unittest.TestCase):
    def test_unicode_units(self):
        self.assertEqual(I._utf16_units("Я"), (1071,))
        self.assertEqual(I._utf16_units("z"), (122,))
        self.assertEqual(I._utf16_units("🙂"), (55357, 56898), "эмодзи — суррогатная пара")

    def test_send_unicode_text_sends_pairs(self):
        calls = []

        class FakeUser32:
            @staticmethod
            def SendInput(count, ptr, size):
                calls.append((count, size))
                return count

        with mock.patch.object(I.ctypes, "windll", mock.Mock(user32=FakeUser32())):
            sent = I.send_unicode_text("аб")
        self.assertEqual(sent, 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c[0] == 2 for c in calls), "на символ: нажатие+отпускание")

    def test_detects_rich_clipboard(self):
        class FakeUser32:
            @staticmethod
            def IsClipboardFormatAvailable(fmt):
                return fmt == 15          # CF_HDROP (файлы)

        with mock.patch.object(I.ctypes, "windll", mock.Mock(user32=FakeUser32())):
            self.assertTrue(I.clipboard_has_rich())

    def test_plain_clipboard_is_not_rich(self):
        class FakeUser32:
            @staticmethod
            def IsClipboardFormatAvailable(fmt):
                return False

        with mock.patch.object(I.ctypes, "windll", mock.Mock(user32=FakeUser32())):
            self.assertFalse(I.clipboard_has_rich())

    def test_detection_never_raises(self):
        with mock.patch.object(I.ctypes, "windll", mock.Mock(**{"user32.IsClipboardFormatAvailable.side_effect": OSError("boom")})):
            self.assertFalse(I.clipboard_has_rich())


class PasteStrategyTest(unittest.TestCase):
    """Главное: при картинке/файлах в буфере буфер НЕ трогаем вообще."""

    def test_rich_clipboard_uses_unicode_typing(self):
        paster = I.ClipboardPaste()
        with mock.patch.object(I, "clipboard_has_rich", return_value=True), \
             mock.patch.object(I, "send_unicode_text", return_value=30) as typed, \
             mock.patch.object(I.ClipboardPaste, "_copy") as copy, \
             mock.patch.object(I.ClipboardPaste, "_send_ctrl_v") as ctrl_v:
            paster.paste("Привет, мир!")
        typed.assert_called_once()
        copy.assert_not_called()
        ctrl_v.assert_not_called()
        self.assertIn("unicode", paster.last_method)

    def test_text_clipboard_keeps_clipboard_path(self):
        paster = I.ClipboardPaste()
        with mock.patch.object(I, "clipboard_has_rich", return_value=False), \
             mock.patch.object(I, "send_unicode_text") as typed, \
             mock.patch.object(I.ClipboardPaste, "_read_text", return_value="прежний текст"), \
             mock.patch.object(I.ClipboardPaste, "_copy") as copy, \
             mock.patch.object(I.ClipboardPaste, "_send_ctrl_v") as ctrl_v:
            paster.paste("Новый текст.")
        typed.assert_not_called()
        ctrl_v.assert_called_once()
        self.assertEqual(copy.call_count, 2, "наш текст + восстановление прежнего")
        self.assertEqual(paster.last_method, "clipboard")

    def test_empty_text_does_nothing(self):
        paster = I.ClipboardPaste()
        with mock.patch.object(I, "clipboard_has_rich") as rich:
            paster.paste("")
        rich.assert_not_called()


class HonestStatusTest(unittest.TestCase):
    """Если текста в поле не видно — говорим об этом, а не «✓ вставлено»."""

    def test_not_found_shows_warning(self):
        import tempfile
        import myflow.main as M
        from myflow.config import SettingsStore
        from myflow.db import Database
        from myflow.axtext import AxWatcher

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app = M.MyFlowApp(settings=SettingsStore(Path(tmp.name) / "s.json"),
                          db=Database(Path(tmp.name) / "h.sqlite"))
        self.addCleanup(app.db.close)
        self.addCleanup(lambda: app._stopping.set())
        self.addCleanup(lambda: app.ax.stop())
        app.overlay = mock.Mock()
        app._ax_ready = True
        app.ax = mock.Mock(available=True)
        app._last_insert_pid = 123
        app._watch_with_ax = mock.Mock(return_value=AxWatcher.NOT_FOUND)
        with mock.patch("myflow.main.Database", return_value=app.db):
            app._spawn_edit_watcher(1, "текст")
            for t in __import__("threading").enumerate():
                if t.name == "edit-watcher":
                    t.join(timeout=3.0)
        shown = [c.args[0] for c in app.overlay.show.call_args_list]
        self.assertTrue(any("проверь вставку" in s for s in shown),
                        f"ожидали предупреждение, получили: {shown}")


if __name__ == "__main__":
    unittest.main()
