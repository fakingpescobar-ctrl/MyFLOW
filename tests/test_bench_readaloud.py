"""Стенд «чтение вслух»: метрики WER по известному тексту.

Проверяем арифметику и разбор эталона — это то, на чём стенд может тихо соврать,
если ошибиться (и тогда решения о модели будут приниматься по мусорным числам).
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.bench_readaloud import error_rate, normalize, split_sentences  # noqa: E402


class NormalizeTest(unittest.TestCase):
    def test_case_yo_punct_and_spaces_do_not_count(self):
        self.assertEqual(normalize("Всё   ХОРОШО, брат!"), "все хорошо брат")

    def test_different_words_stay_different(self):
        self.assertNotEqual(normalize("алмазный клинок"), normalize("деревянный клинок"))


class ErrorRateTest(unittest.TestCase):
    def test_identical_text_has_zero_wer(self):
        rate, errors, words = error_rate("Проверка связи, брат.", "проверка связи брат")
        self.assertEqual(rate, 0.0)
        self.assertEqual(errors, 0)
        self.assertEqual(words, 3)

    def test_one_wrong_word_of_four(self):
        rate, errors, words = error_rate("алмазный деревянный клинок блестит",
                                         "алмазный деревянный клинок блестел")
        self.assertEqual(errors, 1)
        self.assertEqual(words, 4)
        self.assertAlmostEqual(rate, 0.25)

    def test_missing_word_counts_as_error(self):
        rate, errors, _ = error_rate("он взял клинок и ушёл", "он взял и ушёл")
        self.assertEqual(errors, 1)

    def test_extra_word_counts_as_error(self):
        _, errors, _ = error_rate("он взял клинок", "он взял этот клинок")
        self.assertEqual(errors, 1)

    def test_empty_reference_is_safe(self):
        rate, errors, words = error_rate("", "что-то")
        self.assertEqual((rate, errors, words), (0.0, 0, 0))


class SplitTest(unittest.TestCase):
    def test_splits_by_sentence_end(self):
        # Предложения короче трёх слов отсеиваются намеренно: читать «Да.» вслух
        # для замера бессмысленно, а сопоставление с эталоном оно ломает.
        text = ("Он вошёл в тёмный зал. Она молча ждала его там! "
                "Третье предложение тоже достаточно длинное?")
        self.assertEqual(len(split_sentences(text)), 3)

    def test_drops_too_short_fragments(self):
        self.assertEqual(split_sentences("Да.\n\nОн взял тяжёлый клинок и пошёл домой."),
                         ["Он взял тяжёлый клинок и пошёл домой."])

    def test_paragraphs_are_boundaries_too(self):
        text = "Он вошёл в зал\nи остановился у окна\n\nПотом обернулся к двери"
        self.assertEqual(len(split_sentences(text)), 3)


if __name__ == "__main__":
    unittest.main()
