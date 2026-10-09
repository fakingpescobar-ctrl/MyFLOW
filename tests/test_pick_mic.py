"""Тесты выбора микрофона (P0-6: «системный» != «отмена»; P2-10: без TTY)."""
import sys
import unittest
from unittest import mock

import myflow.pick_mic as pm


class PickMicSentinelTest(unittest.TestCase):
    def test_cancel_is_not_none(self):
        self.assertIsNot(pm._CANCEL, None,
                         "отмена не должна совпадать с «системным микрофоном» (None)")

    def test_no_devices_returns_cancel(self):
        store = mock.Mock()
        with mock.patch.object(pm, "_gather", return_value=[]):
            self.assertIs(pm._pick_mic_menu(store), pm._CANCEL)

    def test_system_default_selection_returns_none(self):
        store = mock.Mock()
        store.get.return_value.asr.input_device = None
        with mock.patch.object(pm, "_gather", return_value=[
                {"key": "mic", "name": "Микрофон (USB)", "idx": 1, "alive": True}]), \
             mock.patch.object(pm, "_probe", return_value=True), \
             mock.patch.object(pm, "_menu", return_value=pm._DEFAULT_SYS):
            self.assertIsNone(pm._pick_mic_menu(store),
                              "системный по умолчанию = None, и это НЕ отмена")

    def test_device_selection_returns_index(self):
        store = mock.Mock()
        store.get.return_value.asr.input_device = None
        with mock.patch.object(pm, "_gather", return_value=[
                {"key": "mic", "name": "Микрофон (USB)", "idx": 7, "alive": True}]), \
             mock.patch.object(pm, "_probe", return_value=True), \
             mock.patch.object(pm, "_menu", return_value=7):
            self.assertEqual(pm._pick_mic_menu(store), 7)

    def test_esc_returns_cancel(self):
        store = mock.Mock()
        store.get.return_value.asr.input_device = None
        with mock.patch.object(pm, "_gather", return_value=[
                {"key": "mic", "name": "Микрофон (USB)", "idx": 7, "alive": True}]), \
             mock.patch.object(pm, "_probe", return_value=True), \
             mock.patch.object(pm, "_menu", return_value=None):
            self.assertIs(pm._pick_mic_menu(store), pm._CANCEL)


class NoTtyTest(unittest.TestCase):
    def test_pick_mic_without_tty_keeps_settings(self):
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = False
        with mock.patch.object(sys, "stdin", fake_stdin), \
             mock.patch.object(pm, "_gather") as gather:
            self.assertEqual(pm.pick_mic(), 0)
        gather.assert_not_called()

    def test_probe_timeout_does_not_hang(self):
        """Зависший драйвер: _probe возвращает False, а не висит вечно."""
        import time

        def _hang():
            time.sleep(5)

        with mock.patch("sounddevice.rec", side_effect=lambda *a, **k: _hang()):
            t0 = time.monotonic()
            ok = pm._probe(None, timeout=0.2)
            elapsed = time.monotonic() - t0
        self.assertFalse(ok)
        self.assertLess(elapsed, 1.5, f"проба не уложилась в таймаут: {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
