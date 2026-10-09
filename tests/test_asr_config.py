"""Тесты ASR-клиента и конфига."""
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from myflow.asr import WhisperClient, pcm_int16_to_wav
from myflow.config import AsrConfig, SettingsStore


class WavHeaderTest(unittest.TestCase):
    def test_header_fields(self):
        pcm = np.arange(-100, 100, dtype=np.int16)
        wav = pcm_int16_to_wav(pcm, sample_rate=16000)
        riff, size, wave, fmt, fmt_len = struct.unpack("<4sI4s4sI", wav[:20])
        self.assertEqual(riff, b"RIFF")
        self.assertEqual(wave, b"WAVE")
        self.assertEqual(fmt, b"fmt ")
        self.assertEqual(fmt_len, 16)
        self.assertEqual(size, 36 + len(pcm.tobytes()))
        (audio_fmt, channels, rate, byte_rate, block_align, bits) = struct.unpack("<HHIIHH", wav[20:36])
        self.assertEqual(audio_fmt, 1)
        self.assertEqual(channels, 1)
        self.assertEqual(rate, 16000)
        self.assertEqual(byte_rate, 32000)
        self.assertEqual(block_align, 2)
        self.assertEqual(bits, 16)
        self.assertEqual(wav[36:40], b"data")
        self.assertEqual(struct.unpack("<I", wav[40:44])[0], len(pcm.tobytes()))
        self.assertEqual(len(wav), 44 + len(pcm.tobytes()))

    def test_bad_dtype(self):
        with self.assertRaises(TypeError):
            pcm_int16_to_wav(np.zeros(10, dtype=np.float32))


class TranscribeTest(unittest.TestCase):
    PCM = np.zeros(1600, dtype=np.int16)

    def setUp(self):
        self.client = WhisperClient(AsrConfig())

    def _fake_response(self, payload: bytes, status=200):
        class _R:
            status_code = status
            content = payload
            text = "err"

        return _R()

    def test_utf8_json_parsed(self):
        body = json.dumps({"text": "  Привет, брат.  ", "language": "ru"},
                          ensure_ascii=False).encode("utf-8")
        with mock.patch.object(self.client._session, "post",
                               return_value=self._fake_response(body)):
            res = self.client.transcribe(self.PCM)
        self.assertEqual(res.text, "Привет, брат.")
        self.assertEqual(res.language, "ru")

    def test_non_200_raises(self):
        with mock.patch.object(self.client._session, "post",
                               return_value=self._fake_response(b"oops", status=500)):
            with self.assertRaises(RuntimeError):
                self.client.transcribe(self.PCM)

    def test_broken_json_raises_readable(self):
        with mock.patch.object(self.client._session, "post",
                               return_value=self._fake_response(b"<html>not json</html>")):
            with self.assertRaises(RuntimeError) as ctx:
                self.client.transcribe(self.PCM)
        self.assertIn("нечитаемый", str(ctx.exception))

    def test_empty_and_bad_dtype(self):
        with self.assertRaises(ValueError):
            self.client.transcribe(np.zeros(0, dtype=np.int16))
        with self.assertRaises(TypeError):
            self.client.transcribe(np.zeros(10, dtype=np.float32))

    def test_retry_adapter_installed(self):
        """Ретраи настраиваются на сессии, а не на каждый запрос."""
        self.assertIn("http://", self.client._session.adapters)


class ConfigTest(unittest.TestCase):
    def _write(self, payload: dict) -> Path:
        import json as _json
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "settings.json"
        path.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_flags_merge_not_replace(self):
        """P2-7: отсутствующий флаг не должен давать KeyError."""
        path = self._write({"mode": "ptt", "flags": {"auto_insert": True}})
        store = SettingsStore(path)
        flags = store.get().flags
        self.assertTrue(flags["auto_insert"])
        self.assertIn("overlay", flags)
        self.assertIn("learning", flags)
        self.assertTrue(flags["overlay"], "дефолт должен сохраниться")

    def test_unknown_keys_ignored(self):
        path = self._write({"polish": {"url": "http://x/v1", "неизвестное": 1},
                            "новая_секция": True})
        store = SettingsStore(path)
        self.assertEqual(store.get().polish.url, "http://x/v1")

    def test_new_polish_defaults(self):
        path = self._write({"polish": {"url": "http://x/v1"}})
        cfg = SettingsStore(path).get().polish
        self.assertGreater(cfg.top_p, 0)
        self.assertGreater(cfg.repeat_penalty, 0)

    def test_broken_json_is_loud(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "settings.json"
        path.write_text("{это не json", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            SettingsStore(path)

    def test_roundtrip_save(self):
        path = self._write({"mode": "auto"})
        store = SettingsStore(path)
        store.get().mode = "ptt"
        store.save()
        self.assertEqual(SettingsStore(path).get().mode, "ptt")


if __name__ == "__main__":
    unittest.main()
