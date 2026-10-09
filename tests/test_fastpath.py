"""Тесты быстрого пути (без LLM) и адаптивного порога по фоновому шуму."""
import unittest

from myflow.audio import DictationRecorder
from myflow.fastpath import fast_path_allowed, looks_wellformed


class WellformedTest(unittest.TestCase):
    def test_gigaam_output_is_fast_path(self):
        """Реальные ответы GigaAM e2e — уже оформлены, LLM не нужен."""
        for text in ("Привет! Как дела? Нужно купить хлеб и масло.",
                     "Сегодня отличная погода.",
                     "Надо не забыть позвонить маме.",
                     "Записываю очередную заметку."):
            ok, reason = looks_wellformed(text)
            self.assertTrue(ok, f"{text!r} -> {reason}")

    def test_whisper_like_text_is_not(self):
        """Вывод whisper: без конечной пунктуации/заглавной — полировка нужна."""
        ok, reason = looks_wellformed("так и чё а где живой индикатор")
        self.assertFalse(ok)
        self.assertIn("пунктуац", reason)
        ok, _ = looks_wellformed("нужно купить хлеб и масло.")
        self.assertFalse(ok, "нет заглавной в начале")

    def test_fillers_block_fast_path(self):
        ok, reason = looks_wellformed("Ну это типа проверка связи короче.")
        self.assertFalse(ok)
        self.assertIn("паразит", reason)

    def test_number_words_block_fast_path(self):
        """Числа словами полировка превращает в цифры — пропускать нельзя."""
        ok, reason = looks_wellformed("Открыть эндпоинт на порту восемь тысяч "
                                      "восемьдесят один и запустить тест.")
        self.assertFalse(ok)
        self.assertIn("числа словами", reason)

    def test_short_text_blocks_fast_path(self):
        ok, reason = looks_wellformed("Готово.")
        self.assertFalse(ok)
        self.assertIn("короче", reason)

    def test_digits_only_number_is_fine(self):
        """Уже цифры — правило чисел ничего не сделает."""
        ok, _ = looks_wellformed("Открыть эндпоинт на порту 8081 и запустить тест.")
        self.assertTrue(ok)

    def test_punctuation_defects_block_fast_path(self):
        """GigaAM e2e иногда выдаёт «телевизоре..» — такое пусть чинит LLM."""
        ok, reason = looks_wellformed("Диктую заметку при включённом телевизоре..")
        self.assertFalse(ok)
        self.assertIn("пунктуац", reason)
        for bad in ("Тут  двойной пробел.", "Слово , запятая.", "Странно?! Да."):
            self.assertFalse(looks_wellformed(bad)[0], bad)


class FastPathPolicyTest(unittest.TestCase):
    def test_only_gigaam_and_plain_style(self):
        text = "Привет! Как дела? Нужно купить хлеб и масло."
        self.assertTrue(fast_path_allowed("gigaam", "plain", text)[0])
        self.assertFalse(fast_path_allowed("whisper", "plain", text)[0],
                         "whisper не ставит пунктуацию — быстрый путь не для него")
        ok, reason = fast_path_allowed("gigaam", "terminal", text)
        self.assertFalse(ok)
        self.assertIn("terminal", reason)

    def test_whisper_backend_reason(self):
        ok, reason = fast_path_allowed("whisper", "plain", "Привет, мир, как дела.")
        self.assertFalse(ok)
        self.assertIn("GigaAM", reason)


class _FakeVad:
    """VAD-заглушка: речь на чанках, где пик выше порога громкости."""

    def __init__(self, speech_above=0.10):
        self.threshold = 0.30
        self.speech_above = speech_above

    def reset(self):
        pass

    def process(self, chunk):
        import numpy as np
        peak = float(np.abs(chunk).max()) if len(chunk) else 0.0
        return 0.9 if peak >= self.speech_above else 0.05


class AdaptiveNoiseGateTest(unittest.TestCase):
    def _recorder(self, **kw):
        return DictationRecorder(vad=_FakeVad(), on_utterance=lambda rec: None, **kw)

    def test_quiet_room_keeps_fixed_gate(self):
        rec = self._recorder()
        self.assertEqual(rec.noise_gate(), rec.silence_amplitude)

    def test_background_tv_raises_gate_but_capped(self):
        rec = self._recorder(noise_max_gate=0.04)
        # фон: 20 не-речевых чанков с пиком 0.026 (тихий телевизор)
        for _ in range(20):
            rec._track_noise_floor(0.026, is_speech=False)
        gate = rec.noise_gate()
        self.assertGreater(gate, 0.026, "фоновый ТВ должен отсекаться")
        self.assertLessEqual(gate, 0.04, "потолок защищает тихую речь")

    def test_loud_knock_does_not_break_gate(self):
        rec = self._recorder()
        rec._track_noise_floor(0.9, is_speech=False)      # стук по столу
        for _ in range(20):
            rec._track_noise_floor(0.02, is_speech=False)  # фон снова тихий
        self.assertLessEqual(rec.noise_gate(), 0.04)

    def test_disabled_gate(self):
        rec = self._recorder(adaptive_noise_gate=False)
        for _ in range(50):
            rec._track_noise_floor(0.03, is_speech=False)
        self.assertEqual(rec.noise_gate(), rec.silence_amplitude)

    def test_no_floor_without_enough_background(self):
        rec = self._recorder()
        for _ in range(5):                                 # меньше 15 чанков
            rec._track_noise_floor(0.03, is_speech=False)
        self.assertEqual(rec.noise_gate(), rec.silence_amplitude)

    def test_speech_chunks_do_not_lower_the_floor(self):
        rec = self._recorder()
        for _ in range(20):
            rec._track_noise_floor(0.026, is_speech=False)
        before = rec._noise_floor
        rec._track_noise_floor(0.5, is_speech=True)        # речь фон не поднимает
        self.assertEqual(rec._noise_floor, before)

    def test_real_utterance_passes_gate(self):
        """Реальная речь (пик 0.13+) проходит даже при фоне телевизора."""
        rec = self._recorder()
        for _ in range(20):
            rec._track_noise_floor(0.03, is_speech=False)
        gate = rec.noise_gate()
        self.assertGreater(0.13, gate, "самая тихая реальная запись — 0.13")


if __name__ == "__main__":
    unittest.main()
