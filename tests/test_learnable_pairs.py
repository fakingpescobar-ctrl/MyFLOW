"""Защита словаря от ложных правил автообучения.

Правила вроде «она -> и» или «елай -> ну» появляются из-за неточной атрибуции правок
и молча портят текст: словарь уходит в подсказку полировщику. Здесь проверяется, что
такие пары не проходят, а осмысленные (номера, имена, специфичные слова) проходят.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.learning import learnable  # noqa: E402


class LearnableTest(unittest.TestCase):
    def test_rejects_content_word_into_function_word(self):
        for pair in (("она", "и"), ("идёт", "с"), ("там", "идет"),
                     ("надиктовывать", "и")):
            self.assertFalse(learnable(pair), f"ложная пара {pair} не должна учиться")

    def test_rejects_word_fragments(self):
        self.assertFalse(learnable(("елай", "ну")), "обрубок слова — не замена")

    def test_keeps_numbers(self):
        self.assertTrue(learnable(("8081", "8080")))

    def test_keeps_specific_words_and_names(self):
        for pair in (("подсветка", "подсветкой"), ("анфим", "Онфим"),
                     ("кобальдов", "кобольдов")):
            self.assertTrue(learnable(pair), f"осмысленная пара {pair} должна учиться")

    def test_keeps_insertions_and_deletions(self):
        self.assertTrue(learnable(("", "слово")))
        self.assertTrue(learnable(("слово", "")))

    def test_function_word_as_source_is_allowed(self):
        """Обратный случай: человек убрал лишний предлог — это настоящая правка."""
        self.assertTrue(learnable(("и", "")))
