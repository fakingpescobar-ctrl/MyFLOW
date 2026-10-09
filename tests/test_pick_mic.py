"""Тесты выбора микрофона (P0-6: «системный» != «отмена»; P2-10: без TTY).

Плюс регрессии пользователя: «виден только Kinect, а Ardor рабочий» и
«F5 не пересканирует микрофоны».
"""
import sys
import unittest
from unittest import mock

import numpy as np

import myflow.pick_mic as pm


class MicFilterTest(unittest.TestCase):
    """Раньше требовалось слово «микрофон»/«mic» в имени — реальные микрофоны
    с другим именем исчезали из списка целиком."""

    def test_inputs_without_mic_word_are_shown(self):
        for name in ("Ardor Gaming Vox", "Line In (USB Audio)", "Aux Input"):
            self.assertTrue(pm._is_mic(name), name)

    def test_outputs_and_virtual_cables_are_hidden(self):
        for name in ("Динамики (Realtek)", "CABLE Output (VB-Audio)",
                     "Стерео микшер (Realtek)", "Переназначение звуковых устр.",
                     "Первичный драйвер записи звука", "SPDIF Interface"):
            self.assertFalse(pm._is_mic(name), name)

    def test_mic_hint_is_only_for_priority(self):
        self.assertTrue(pm._has_mic_hint("Микрофон (Ardor Gaming Vox)"))
        self.assertTrue(pm._has_mic_hint("Набор микрофонов (Kinect USB Audio)"))
        self.assertFalse(pm._has_mic_hint("Line In (USB Audio)"))


class ProbeStatusTest(unittest.TestCase):
    """Проба: различаем «занят» / «тишина» / «не умеет 16 кГц»."""

    def _probe_with(self, rec_impl, exc=None):
        def fake_rec(n, samplerate, channels, dtype, device):
            if exc is not None:
                raise exc
            return rec_impl(n)
        with mock.patch.object(pm.sd, "rec", side_effect=fake_rec), \
             mock.patch.object(pm.sd, "wait", lambda: None), \
             mock.patch.object(pm.time, "sleep", lambda _s: None):
            return pm._probe_status(1)

    def test_loud_signal_is_alive(self):
        alive, reason = self._probe_with(
            lambda n: np.full((n, 1), 0.05, dtype=np.float32))
        self.assertTrue(alive)
        self.assertIn("жив", reason)

    def test_quiet_but_present_signal_is_alive_with_warning(self):
        # Ровно кейс Ardor: сигнал есть, но тихий (плавает 0.0003…0.0027)
        alive, reason = self._probe_with(
            lambda n: np.full((n, 1), 0.0006, dtype=np.float32))
        self.assertTrue(alive, "тихий вход — всё ещё рабочий микрофон")
        self.assertIn("тихий", reason)

    def test_digital_silence_is_dead(self):
        alive, reason = self._probe_with(
            lambda n: np.full((n, 1), 1.5e-5, dtype=np.float32))
        self.assertFalse(alive)
        self.assertIn("нет сигнала", reason)

    def test_unsupported_rate_reason(self):
        alive, reason = self._probe_with(None, exc=RuntimeError(
            "Error opening InputStream: Invalid sample rate [PaErrorCode -9997]"))
        self.assertFalse(alive)
        self.assertIn("16 кГц", reason)

    def test_busy_reason(self):
        alive, reason = self._probe_with(None, exc=RuntimeError(
            "Error opening InputStream: Device unavailable [PaErrorCode -9985]"))
        self.assertFalse(alive)
        self.assertIn("занят", reason)

    def test_reasons_are_short_enough_for_the_screen(self):
        """Причина показывается в строке списка — не должна быть простыней."""
        for exc in (RuntimeError("Error opening InputStream: Invalid device [PaErrorCode -9996]"),
                    RuntimeError("Error opening InputStream: Invalid sample rate [PaErrorCode -9997]"),
                    RuntimeError("Error opening InputStream: Device unavailable [PaErrorCode -9985]"),
                    RuntimeError("Что-то совсем неожиданное и очень длинное "*5)):
            _, reason = self._probe_with(None, exc=exc)
            self.assertLessEqual(len(reason), 40, reason)

    def test_retry_helps_transient_failure(self):
        """Первый заход после старта/BT-пробуждения часто падает — повторяем."""
        calls = {"n": 0}

        def fake_rec(n, samplerate, channels, dtype, device):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Device unavailable")
            return np.full((n, 1), 0.02, dtype=np.float32)

        with mock.patch.object(pm.sd, "rec", side_effect=fake_rec), \
             mock.patch.object(pm.sd, "wait", lambda: None), \
             mock.patch.object(pm.time, "sleep", lambda _s: None):
            alive, reason = pm._probe_status(1)
        self.assertTrue(alive, "после повтора устройство должно считаться живым")
        self.assertEqual(calls["n"], 2)


class GatherTest(unittest.TestCase):
    """Дубли одной физики в разных host API: берём живой вариант и его API."""

    def test_merge_prefers_alive_variant_with_api(self):
        devices = [
            {"name": "Микрофон (Ardor Gaming Vox)", "max_input_channels": 1,
             "hostapi": 0, "default_samplerate": 44100},
            {"name": "Микрофон (Ardor Gaming Vox)", "max_input_channels": 2,
             "hostapi": 1, "default_samplerate": 48000},
            {"name": "CABLE Output (VB-Audio)", "max_input_channels": 8,
             "hostapi": 0, "default_samplerate": 44100},
        ]
        hostapis = [{"name": "Windows MME"}, {"name": "Windows WASAPI"}]
        statuses = {0: (False, "не поддерживает 16 кГц"), 1: (True, "жив (rms=0.01)")}

        with mock.patch.object(pm.sd, "query_devices", side_effect=lambda i=None: (
                devices if i is None else devices[i])), \
             mock.patch.object(pm.sd, "query_hostapis",
                               side_effect=lambda i=None: hostapis if i is None else hostapis[i]), \
             mock.patch.object(pm, "_probe_status", side_effect=lambda i, timeout=2.5: statuses[i]):
            groups = pm._gather()
        self.assertEqual(len(groups), 1, "CABLE Output отфильтрован, Ardor склеен")
        self.assertTrue(groups[0]["alive"])
        self.assertEqual(groups[0]["idx"], 1, "берём ЖИВОЙ вариант, а не первый")
        self.assertIn("WASAPI", groups[0]["api"])


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
