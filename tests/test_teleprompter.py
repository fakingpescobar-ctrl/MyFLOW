"""Логика окна-подсказки: подсветка прочитанного и живые ошибки распознавания.

Главное, что здесь проверяется: окно должно двигаться ВПЕРЁД по тексту, отмечать
слова с ошибкой распознавания и не съезжать, если человек повторил фразу или
распознавание вернуло пустое.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.teleprompter import Teleprompter, align_from, split_words  # noqa: E402

TEXT = ("Он вошёл в тёмный зал и остановился у окна. "
        "Клинок лежал на столе и тускло блестел в свете свечи. "
        "Хранитель молча смотрел на гостя и ждал первого слова.")


class SplitTest(unittest.TestCase):
    def test_words_counted(self):
        self.assertEqual(len(split_words(TEXT)), 28)

    def test_empty_text_is_safe(self):
        prompter = Teleprompter("")
        self.assertEqual(prompter.progress(), 0.0)
        self.assertEqual(prompter.feed("что-то"), (0, set()))


class AlignTest(unittest.TestCase):
    def setUp(self):
        self.prompter = Teleprompter(TEXT)

    def test_exact_phrase_advances(self):
        # «Он вошёл в тёмный зал и остановился у окна» — девять слов.
        index, wrong = self.prompter.feed("Он вошёл в тёмный зал и остановился у окна")
        self.assertEqual(index, 9)
        self.assertEqual(wrong, set(), "идеально распознанная фраза — без ошибок")

    def test_wrong_word_is_marked_but_position_advances(self):
        index, wrong = self.prompter.feed("Он вошёл в тёмный зал и остановился у окно")
        self.assertGreaterEqual(index, 9, "фраза прочитана, позиция сдвинулась")
        self.assertIn(8, wrong, "последнее (девятое) слово распознано неверно")

    def test_empty_hearing_does_not_move(self):
        self.assertEqual(self.prompter.feed(""), (0, set()))
        self.assertEqual(self.prompter.feed("   "), (0, set()))

    def test_two_phrases_move_forward(self):
        self.prompter.feed("Он вошёл в тёмный зал и остановился у окна")
        index, _ = self.prompter.feed("Клинок лежал на столе и тускло блестел в свете свечи")
        self.assertEqual(index, 19, "9 слов первой фразы + 10 слов второй")

    def test_repeated_phrase_does_not_go_back(self):
        first, _ = self.prompter.feed("Он вошёл в тёмный зал и остановился у окна")
        second, _ = self.prompter.feed("Он вошёл в тёмный зал и остановился у окна")
        self.assertGreaterEqual(second, first, "повтор не должен откатывать позицию")

    def test_progress_grows(self):
        before = self.prompter.progress()
        self.prompter.feed("Клинок лежал на столе")
        self.assertGreater(self.prompter.progress(), before)

    def test_align_from_ignores_empty_words_list(self):
        self.assertEqual(align_from([], "он вошёл", 0), (0, set()))


if __name__ == "__main__":
    unittest.main()
