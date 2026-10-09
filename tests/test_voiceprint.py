"""Тесты проверки голоса (speaker verification) и отказа от whisper-сервера.

Модель в тестах не нужна: подменяем эмбеддер фейком с управляемыми векторами,
поэтому проверяем ровно нашу логику — пороги, профиль, авто-пополнение,
деградацию без модели и поведение оркестратора.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from myflow.asr import whisper_needed
from myflow.config import AsrConfig
from myflow.voiceprint import SpeakerEmbedder, VoiceGate, VoiceProfile


class _FakeEmbedder:
    """Эмбеддер-заглушка: нужный вектор задаётся тестом."""

    def __init__(self, vector=None, available=True, model_path=None):
        self.vector = np.asarray(vector if vector is not None else [1.0, 0.0, 0.0],
                                 dtype=np.float32)
        self._available = available
        self.model_path = Path(model_path) if model_path else Path("fake.onnx")
        self.loaded = False
        self.load_error = "" if available else "нет модели"

    @property
    def available(self):
        return self._available

    def warmup(self):
        self.loaded = self._available
        return self._available

    def embed(self, pcm):
        return self.vector / np.linalg.norm(self.vector)


def _pcm(seconds=1.5):
    return np.zeros(int(16000 * seconds), dtype=np.int16)


class VoiceProfileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "vp.npz"

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_load_roundtrip(self):
        p = VoiceProfile(self.path)
        self.assertFalse(p.exists)
        p.add(np.array([1.0, 0.0], dtype=np.float32))
        p.add(np.array([1.0, 0.0], dtype=np.float32))
        self.assertTrue(p.save())
        self.assertEqual(p.count, 2)

        loaded = VoiceProfile(self.path)
        self.assertTrue(loaded.exists)
        self.assertEqual(loaded.count, 2)
        self.assertAlmostEqual(loaded.score(np.array([1.0, 0.0], dtype=np.float32)), 1.0,
                               places=5)

    def test_centroid_moves_towards_new_samples(self):
        p = VoiceProfile(self.path)
        p.add(np.array([1.0, 0.0], dtype=np.float32))
        p.add(np.array([0.0, 1.0], dtype=np.float32))
        # центроид не прыгает на новый вектор целиком (скользящее среднее)
        self.assertGreater(p.score(np.array([1.0, 0.0], dtype=np.float32)), 0.5)
        self.assertLess(p.score(np.array([1.0, 0.0], dtype=np.float32)), 1.0)

    def test_reset_removes_file(self):
        p = VoiceProfile(self.path)
        p.add(np.array([1.0, 0.0], dtype=np.float32))
        p.save()
        p.reset()
        self.assertFalse(p.exists)
        self.assertFalse(self.path.exists())

    def test_broken_file_is_not_fatal(self):
        self.path.write_bytes(b"not an npz")
        p = VoiceProfile(self.path)
        self.assertFalse(p.exists)


class VoiceGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.profile_path = Path(self._tmp.name) / "vp.npz"

    def tearDown(self):
        self._tmp.cleanup()

    def _gate(self, vector=(1.0, 0.0, 0.0), available=True, **kw):
        gate = VoiceGate(profile_path=self.profile_path, **kw)
        gate.embedder = _FakeEmbedder(vector, available=available)
        return gate

    def test_no_profile_passes_everything(self):
        gate = self._gate()
        ok, score, reason = gate.check(_pcm())
        self.assertTrue(ok)
        self.assertIsNone(score)
        self.assertIn("нет профиля", reason)
        self.assertFalse(gate.active)

    def test_owner_is_accepted(self):
        gate = self._gate()
        gate.profile.add(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        gate.profile.save()
        ok, score, reason = gate.check(_pcm())
        self.assertTrue(ok)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_other_speaker_is_rejected(self):
        gate = self._gate(vector=(0.0, 1.0, 0.0))
        gate.profile.add(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        gate.profile.save()
        ok, score, reason = gate.check(_pcm())
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.0, places=4)
        self.assertIn("чужой голос", reason)
        self.assertEqual(gate.stats["rejections"], 1)

    def test_short_phrase_is_not_checked(self):
        """Короткие фразы пропускаем: эмбеддинг по ним ненадёжен."""
        gate = self._gate(vector=(0.0, 1.0, 0.0))
        gate.profile.add(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        gate.profile.save()
        ok, score, reason = gate.check(_pcm(0.3))
        self.assertTrue(ok)
        self.assertIsNone(score)
        self.assertIn("короче", reason)

    def test_missing_model_passes_everything(self):
        gate = self._gate(available=False)
        gate.profile.add(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        gate.profile.save()
        ok, _, reason = gate.check(_pcm())
        self.assertTrue(ok, "без модели лучше пропустить фразу, чем потерять диктовку")
        self.assertIn("модель", reason)
        self.assertFalse(gate.active)

    def test_disabled_gate(self):
        gate = self._gate(enabled=False)
        gate.profile.add(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertTrue(gate.check(_pcm())[0])

    def test_enroll_from_empty_profile(self):
        gate = self._gate()
        ok, score = gate.enroll(_pcm())
        self.assertTrue(ok)
        self.assertIsNone(score)
        self.assertEqual(gate.profile.count, 1)
        self.assertTrue(gate.profile.exists)

    def test_auto_enroll_only_confident_own_phrases(self):
        gate = self._gate(vector=(1.0, 0.0, 0.0), auto_enroll=True,
                          enroll_min_score=0.45)
        gate.profile.add(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        gate.profile.save()
        # своя фраза (score 1.0) — добавляется
        self.assertTrue(gate.maybe_auto_enroll(_pcm(), accepted=True, score=1.0))
        self.assertEqual(gate.profile.count, 2)
        # слабое совпадение (0.3) — не добавляем, иначе профиль «размажется»
        self.assertFalse(gate.maybe_auto_enroll(_pcm(), accepted=True, score=0.3))
        # отклонённая фраза — тем более
        self.assertFalse(gate.maybe_auto_enroll(_pcm(), accepted=False, score=0.9))

    def test_auto_enroll_bootstraps_empty_profile(self):
        gate = self._gate(auto_enroll=True)
        self.assertTrue(gate.maybe_auto_enroll(_pcm(), accepted=True, score=None))
        self.assertEqual(gate.profile.count, 1)


class EmbedderAvailabilityTest(unittest.TestCase):
    def test_missing_model_reports_error(self):
        emb = SpeakerEmbedder(Path("нет-такого-файла.onnx"))
        self.assertFalse(emb.warmup())
        self.assertIn("нет модели", emb.load_error)
        self.assertFalse(emb.available)

    def test_embed_empty_returns_none(self):
        emb = SpeakerEmbedder(Path("нет-такого-файла.onnx"))
        self.assertIsNone(emb.embed(np.zeros(0, dtype=np.int16)))


class WhisperNeededTest(unittest.TestCase):
    def test_whisper_backend_always_needs_server(self):
        self.assertTrue(whisper_needed(AsrConfig(backend="whisper")))

    def test_gigaam_backend_does_not_need_server(self):
        with mock.patch("myflow.asr_gigaam.GigaAmClient.package_available",
                        staticmethod(lambda: True)):
            self.assertFalse(whisper_needed(AsrConfig(backend="gigaam")))
            self.assertFalse(whisper_needed(AsrConfig(backend="auto")))

    def test_fallback_when_package_missing(self):
        with mock.patch("myflow.asr_gigaam.GigaAmClient.package_available",
                        staticmethod(lambda: False)):
            self.assertTrue(whisper_needed(AsrConfig(backend="auto")),
                            "без GigaAM нужен whisper-сервер")
            self.assertTrue(whisper_needed(AsrConfig(backend="gigaam")))

    def test_env_override_keeps_whisper(self):
        with mock.patch.dict(os.environ, {"MYFLOW_KEEP_WHISPER": "1"}):
            with mock.patch("myflow.asr_gigaam.GigaAmClient.package_available",
                            staticmethod(lambda: True)):
                self.assertTrue(whisper_needed(AsrConfig(backend="gigaam")))


if __name__ == "__main__":
    unittest.main()
