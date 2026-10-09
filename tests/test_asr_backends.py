"""Тесты ASR-бэкендов: выбор движка и GigaAM-клиент.

Пакет `onnx-asr` (и сама модель) в тестах не нужны: подменяем модуль
`onnx_asr` фейком, поэтому проверяем ровно нашу обвязку — выбор бэкенда,
отложенную загрузку, ошибки и формат результата.
"""
import sys
import types
import unittest
from unittest import mock

import numpy as np

from myflow.asr import AsrResult, WhisperClient, make_asr_client
from myflow.config import AsrConfig


class _FakeModel:
    def __init__(self, text="привет мир", fail=False):
        self.text = text
        self.fail = fail
        self.calls = []

    def recognize(self, audio, sample_rate=None):
        if self.fail:
            raise RuntimeError("инференс упал")
        self.calls.append((len(audio), sample_rate))
        return self.text


def _fake_onnx_asr(model: _FakeModel | None = None, load_error: Exception | None = None):
    module = types.ModuleType("onnx_asr")

    def load_model(name, quantization=None):
        if load_error is not None:
            raise load_error
        module.loaded_with = (name, quantization)
        return model if model is not None else _FakeModel()

    module.load_model = load_model
    return module


def _pcm(ms=200):
    n = int(16000 * ms / 1000)
    return (np.sin(np.arange(n) / 30) * 5000).astype(np.int16)


class BackendSelectionTest(unittest.TestCase):
    def test_whisper_backend(self):
        client = make_asr_client(AsrConfig(backend="whisper"))
        self.assertIsInstance(client, WhisperClient)

    def test_gigaam_backend_without_package_falls_back(self):
        with mock.patch("myflow.asr_gigaam.GigaAmClient.package_available",
                        staticmethod(lambda: False)):
            client = make_asr_client(AsrConfig(backend="gigaam"))
        self.assertIsInstance(client, WhisperClient, "без пакета работаем на whisper")

    def test_auto_prefers_gigaam_when_available(self):
        with mock.patch("myflow.asr_gigaam.GigaAmClient.package_available",
                        staticmethod(lambda: True)):
            client = make_asr_client(AsrConfig(backend="auto"))
        from myflow.asr_gigaam import GigaAmClient
        self.assertIsInstance(client, GigaAmClient)

    def test_auto_falls_back_to_whisper(self):
        with mock.patch("myflow.asr_gigaam.GigaAmClient.package_available",
                        staticmethod(lambda: False)):
            client = make_asr_client(AsrConfig(backend="auto"))
        self.assertIsInstance(client, WhisperClient)


class GigaAmClientTest(unittest.TestCase):
    def setUp(self):
        from myflow.asr_gigaam import GigaAmClient
        self.client = GigaAmClient(AsrConfig(backend="gigaam", gigaam_model="gigaam-v3-e2e-ctc",
                                             gigaam_quantization="int8"))

    def test_transcribe_returns_asr_result(self):
        model = _FakeModel("Привет, мир.")
        with mock.patch.dict(sys.modules, {"onnx_asr": _fake_onnx_asr(model)}):
            result = self.client.transcribe(_pcm(300))
        self.assertIsInstance(result, AsrResult)
        self.assertEqual(result.text, "Привет, мир.")
        self.assertEqual(result.language, "ru")
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(model.calls[0][1], 16000, "частота должна быть 16 кГц")

    def test_model_loaded_once(self):
        model = _FakeModel()
        module = _fake_onnx_asr(model)
        with mock.patch.dict(sys.modules, {"onnx_asr": module}):
            self.client.transcribe(_pcm())
            self.client.transcribe(_pcm())
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(module.loaded_with, ("gigaam-v3-e2e-ctc", "int8"))

    def test_warmup_reports_load_error(self):
        module = _fake_onnx_asr(load_error=OSError("нет файла модели"))
        with mock.patch.dict(sys.modules, {"onnx_asr": module}):
            self.assertFalse(self.client.warmup())
        self.assertFalse(self.client.available)
        self.assertIn("нет файла модели", self.client.load_error)
        with self.assertRaises(RuntimeError):
            self.client.transcribe(_pcm())

    def test_empty_and_wrong_dtype_rejected(self):
        with self.assertRaises(ValueError):
            self.client.transcribe(np.zeros(0, dtype=np.int16))
        with self.assertRaises(TypeError):
            self.client.transcribe(np.zeros(100, dtype=np.float32))

    def test_health(self):
        module = _fake_onnx_asr()
        with mock.patch.dict(sys.modules, {"onnx_asr": module}):
            self.assertTrue(self.client.health(), "пакет есть -> живой")
            self.client.warmup()
            self.assertTrue(self.client.health())

    def test_whitespace_stripped(self):
        model = _FakeModel("  текст с пробелами  ")
        with mock.patch.dict(sys.modules, {"onnx_asr": _fake_onnx_asr(model)}):
            self.assertEqual(self.client.transcribe(_pcm()).text, "текст с пробелами")


class LatinCommandTest(unittest.TestCase):
    """GigaAM пишет «Стоп» как «Stop.» — команда должна всё равно срабатывать."""

    def test_latin_stop_and_skip(self):
        from myflow.commands import classify

        for phrase, kind in (("Stop.", "stop"), ("stop", "stop"),
                             ("Skip", "skip"), ("стоп", "stop")):
            intent = classify(phrase)
            self.assertEqual(intent.kind, kind, phrase)

    def test_latin_does_not_break_normal_text(self):
        from myflow.commands import classify

        intent = classify("и тут я решил остановиться на этом варианте")
        self.assertTrue(intent is None or intent.kind not in ("stop", "skip"))


if __name__ == "__main__":
    unittest.main()
