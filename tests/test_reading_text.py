"""Форматирование текста для чтения и разбивка на абзацы в окне.

Два правила, которые здесь защищаются:
  1) текст главы — ЭТАЛОН для замера WER, форматирование не смеет менять ни одного
     слова (иначе мы будем мерить не распознавание, а свою же правку);
  2) окно чтения обязано сохранять строки-абзацы исходника, иначе получается одна
     простыня, и читать вслух невозможно.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from myflow.readwindow import paragraph_breaks          # noqa: E402
from myflow.teleprompter import split_words             # noqa: E402
from tools.format_reading_text import format_text       # noqa: E402


class ParagraphBreaksTest(unittest.TestCase):
    def test_last_word_of_each_line_ends_a_paragraph(self):
        text = "Первая строка тут.\nВторая строка тоже здесь.\nТретья."
        self.assertEqual(paragraph_breaks(text), {2, 6, 7})

    def test_matches_flat_word_list(self):
        text = "Одно два три\nчетыре пять\n\nшесть семь восемь"
        words = split_words(text)
        breaks = paragraph_breaks(text)
        self.assertTrue(all(0 <= index < len(words) for index in breaks))
        self.assertEqual(len(breaks), 3, "три непустых строки")

    def test_blank_lines_are_skipped(self):
        text = "Строка.\n\n\nЕщё строка с словами."
        self.assertEqual(paragraph_breaks(text), {0, 4})

    def test_empty_text_is_safe(self):
        self.assertEqual(paragraph_breaks(""), set())


class FormatTextTest(unittest.TestCase):
    def test_words_are_never_changed(self):
        text = "Привет,  мир!\n\n\nКак дела?   Хорошо."
        self.assertEqual(len(format_text(text).split()), len(text.split()))

    def test_collapses_spaces_and_trims(self):
        result = format_text("Строка с  двойным пробелом   \nвторая\tстрока  ")
        self.assertNotIn("  ", result)
        for line in result.splitlines():
            self.assertEqual(line, line.strip())

    def test_blank_lines_around_scene_separator(self):
        result = format_text("Текст до.\n***\nТекст после.")
        lines = result.splitlines()
        index = lines.index("***")
        self.assertEqual(lines[index - 1], "")
        self.assertEqual(lines[index + 1], "")

    def test_no_triple_blank_lines_or_trailing_blanks(self):
        result = format_text("А.\n\n\n\nБ.\n\n\n")
        self.assertNotIn("\n\n\n", result)
        self.assertTrue(result.endswith("Б.\n"))

    def test_idempotent(self):
        text = "Первый абзац.\nВторой абзац.\n***\nТретий абзац."
        once = format_text(text)
        self.assertEqual(format_text(once), once, "повторный прогон ничего не меняет")


if __name__ == "__main__":
    unittest.main()
