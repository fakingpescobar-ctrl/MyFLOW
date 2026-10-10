"""Тесты LLM-судьи: склейка двух текстов + предохранитель.

Замер на 624 фразах: склейка даёт WER p50 0.040 против 0.071 у авто-выбора
и 0.083 у одного GigaAM. Но судья не слышит аудио, поэтому обязателен
предохранитель — один раз из 624 модель выдала эхо промпта.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from myflow.asr_judge import MergeResult, merge, rail_ok, word_error


class RailTest(unittest.TestCase):
    def test_accepts_close_to_input(self):
        ok, why = rail_ok("Проверка связи, один, два, три.",
                          "Проверка связи один два три", "проверка связи один два три")
        self.assertTrue(ok, why)

    def test_rejects_prompt_echo(self):
        """Реальный случай из замера: модель вернула фразу из промпта."""
        ok, why = rail_ok("То, что человек реально сказал.",
                          "Хотя я агента об этом не просил", "хотя я агента об этом не просил")
        self.assertFalse(ok)
        self.assertIn("эхо", why)

    def test_rejects_far_from_both(self):
        ok, why = rail_ok("Совершенно другой текст про погоду и рыбалку",
                          "Проверка связи один два три", "проверка связи один два три")
        self.assertFalse(ok)
        self.assertIn("не похоже", why)

    def test_rejects_empty(self):
        self.assertFalse(rail_ok("", "а", "б")[0])
        self.assertFalse(rail_ok("  ", "а", "б")[0])

    def test_similarity_is_symmetric_enough(self):
        self.assertLess(word_error("один два три", "один два три"), 1e-9)
        self.assertGreater(word_error("один два три", "четыре пять шесть"), 0.9)


class MergeTest(unittest.TestCase):
    def _client(self, answer, fail=False):
        client = mock.Mock()
        if fail:
            client.ask.side_effect = RuntimeError("LLM недоступна")
        else:
            client.ask.return_value = answer
        return client

    def test_merges_two_variants(self):
        client = self._client("Слушай, брат, я запустил модель Arnit 1.59b.")
        result = merge(client, "Слушай, брат, я запустил модель Aрнид 1.59B.",
                       "слушай брат я запустил модель arnit 1.59b")
        self.assertIsInstance(result, MergeResult)
        self.assertIn("Arnit", result.text)
        self.assertEqual(result.reason, "склейка")
        self.assertGreaterEqual(result.latency_ms, 0)

    def test_strips_letter_prefix(self):
        client = self._client("B: проверка связи один два три")
        result = merge(client, "Проверка связи один два три.", "проверка связи один два три")
        self.assertEqual(result.text, "проверка связи один два три")

    def test_rejects_echo_and_reports_reason(self):
        client = self._client("То, что человек реально сказал.")
        result = merge(client, "Проверка связи.", "проверка связи")
        self.assertEqual(result.text, "")
        self.assertIn("эхо", result.reason)

    def test_llm_failure_is_safe(self):
        client = self._client("", fail=True)
        result = merge(client, "Проверка связи.", "проверка связи")
        self.assertEqual(result.text, "")
        self.assertIn("недоступ", result.reason)

    def test_single_input_short_circuits(self):
        result = merge(self._client("не важно"), "", "только whisper")
        self.assertEqual(result.text, "только whisper")
        self.assertEqual(result.reason, "только whisper")

    def test_both_empty(self):
        result = merge(self._client("не важно"), "", "")
        self.assertEqual(result.text, "")
        self.assertIn("нет текстов", result.reason)


if __name__ == "__main__":
    unittest.main()
