"""Логика окна чтения: сессия прохода, пропуск, сброс, отчёт.

Окно проверять тестами нельзя (нужен экран), а вот всё, что считает цифры, — нужно:
по этим числам мы потом решаем, как распознавание работает на живом чтении.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.readwindow import (ReadingSession, needs_scroll,   # noqa: E402
                              scroll_target)

TEXT = ("Он вошёл в тёмный зал и остановился у окна. "
        "Клинок лежал на столе и тускло блестел в свете свечи. "
        "Хранитель молча смотрел на гостя и ждал первого слова.")


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.session = ReadingSession(TEXT, model="GigaAM")

    def test_starts_at_zero(self):
        self.assertEqual(self.session.index, 0)
        self.assertEqual(self.session.errors, 0)
        self.assertEqual(self.session.accuracy(), 1.0)
        self.assertIn("GigaAM", self.session.status())

    def test_feed_advances_and_counts_phrases(self):
        moved = self.session.feed("Он вошёл в тёмный зал и остановился у окна")
        self.assertTrue(moved)
        self.assertEqual(self.session.index, 9)
        self.assertEqual(self.session.feed_count, 1)

    def test_empty_feed_is_ignored(self):
        self.assertFalse(self.session.feed(""))
        self.assertFalse(self.session.feed("   "))
        self.assertEqual(self.session.feed_count, 0)

    def test_errors_lower_accuracy(self):
        self.session.feed("Он вошёл в тёмный зал и остановился у окно")
        self.assertGreaterEqual(self.session.errors, 1)
        self.assertLess(self.session.accuracy(), 1.0)

    def test_skip_moves_one_word_forward(self):
        before = self.session.index
        self.session.skip()
        self.assertEqual(self.session.index, before + 1)
        self.assertEqual(self.session.skipped, 1)

    def test_reset_clears_everything(self):
        self.session.feed("Он вошёл в тёмный зал")
        self.session.skip()
        self.session.reset()
        self.assertEqual(self.session.index, 0)
        self.assertEqual(self.session.errors, 0)
        self.assertEqual(self.session.feed_count, 0)
        self.assertEqual(self.session.skipped, 0)

    def test_status_shows_progress_and_errors(self):
        self.session.feed("Он вошёл в тёмный зал и остановился у окна")
        status = self.session.status()
        self.assertIn("прочитано 9/28", status)
        self.assertIn("ошибок", status)

    def test_report_contains_numbers_and_problem_words(self):
        self.session.feed("Он вошёл в тёмный зал и остановился у окно")
        report = self.session.report()
        self.assertEqual(report["model"], "GigaAM")
        self.assertEqual(report["words_read"], self.session.index)
        self.assertGreaterEqual(report["errors"], 1)
        self.assertLessEqual(report["accuracy"], 1.0)
        self.assertTrue(report["problem_words"], "должны быть слова с ошибками")
        first = report["problem_words"][0]
        self.assertIn("expected", first)
        self.assertIn("context", first)

    def test_save_report_writes_json(self):
        self.session.feed("Он вошёл в тёмный зал и остановился у окна")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = self.session.save_report(Path(tmp))
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["words_total"], 28)
        self.assertIn("saved_at", data)


if __name__ == "__main__":
    unittest.main()


class ScrollTest(unittest.TestCase):
    """Прокрутка: чтение идёт сверху вниз, и текст не прыгает сам по себе.

    Ошибка, которую здесь ловим: на старте текущее слово первое, а see() в tkinter
    прокручивал так, что строка вставала ВНИЗУ экрана — окно «листало вниз» каждые
    полсекунды, и читать было невозможно.
    """

    def test_start_of_text_stays_at_top(self):
        self.assertTrue(needs_scroll(1, 1, 25), "первое слово вне удобной зоны -> к верху")
        self.assertEqual(scroll_target(1, 400), 0.0, "доля 0.0 — это самый верх")

    def test_middle_of_screen_is_left_alone(self):
        self.assertFalse(needs_scroll(12, 1, 25), "слово в удобной зоне — не трогаем")

    def test_near_bottom_scrolls_a_little(self):
        self.assertTrue(needs_scroll(23, 1, 25))
        self.assertLess(scroll_target(23, 400), 0.1, "прокрутка небольшая, на пятую строку")

    def test_near_top_scrolls_back(self):
        self.assertTrue(needs_scroll(11, 10, 34), "слово под самым верхом -> подтянуть")

    def test_short_text_never_scrolls(self):
        self.assertEqual(scroll_target(1, 1), 0.0)
        self.assertEqual(scroll_target(5, 1), 0.0)
