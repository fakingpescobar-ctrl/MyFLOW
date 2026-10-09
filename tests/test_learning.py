"""Тесты авто-обучения (P0-3: paste() без аргумента -> мёртвое обучение + busy-spin)."""
import tempfile
import time
import unittest
from pathlib import Path

from myflow.db import Database, DictationRecord
from myflow.learning import (
    extract_corrections,
    looks_like_edit_of,
    record_edit_and_learn,
    watch_pasted_edit,
)

BASE = "открыть эндпоинт на порту 8081 и запустить тест проверяем пайплайн"


class TestExtractCorrections(unittest.TestCase):
    def test_replace_pair(self):
        pairs = extract_corrections("открыть порт 8081", "открыть порт 8080")
        self.assertIn(("8081", "8080"), pairs)

    def test_long_text_no_autojunk_noise(self):
        """autojunk=False: на длинных текстах частые слова не считаются мусором."""
        words = " ".join(f"слово{i}" for i in range(250))
        pairs = extract_corrections(words, words)
        self.assertEqual(pairs, [], "идентичные длинные тексты не дают «правок»")


class TestLooksLikeEdit(unittest.TestCase):
    def test_edit_detected(self):
        self.assertTrue(looks_like_edit_of(BASE, BASE + " теперь"))

    def test_identical_is_not_edit(self):
        self.assertFalse(looks_like_edit_of(BASE, BASE))

    def test_foreign_text_is_not_edit(self):
        self.assertFalse(looks_like_edit_of(BASE, "привет мир как дела сегодня погода"))

    def test_empty(self):
        self.assertFalse(looks_like_edit_of(BASE, ""))


class TestWatchContract(unittest.TestCase):
    """P0-3: watch_pasted_edit принимает ЧИТАЛКУ буфера, а не paster."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "learn.sqlite")
        self.hid = self.db.insert_history(DictationRecord(
            asr_text=BASE, polished_text=BASE, app_window="test",
            duration_ms=1000, e2e_latency_ms=100))

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def test_accepts_reader_callable(self):
        edited = "открыть эндпоинт на порту 8080 и запустить тест проверяем пайплайн"
        res = watch_pasted_edit(self.db, self.hid, BASE, lambda: edited,
                                window_seconds=0.5, poll_interval=0.05)
        self.assertIsNotNone(res)
        self.assertGreater(res["edit_distance"], 0)

    def test_reader_exception_does_not_spin(self):
        """Ошибка чтения не должна крутить цикл на 100% CPU (был `continue`)."""
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise RuntimeError("клипборд занят")

        t0 = time.monotonic()
        res = watch_pasted_edit(self.db, self.hid, BASE, boom,
                                window_seconds=0.5, poll_interval=0.1)
        elapsed = time.monotonic() - t0
        self.assertIsNone(res)
        # ~5 итераций вместо тысяч: цикл спит
        self.assertLess(calls["n"], 15, f"слишком много итераций: {calls['n']} (busy-spin)")
        self.assertGreaterEqual(elapsed, 0.4)

    def test_no_edit_returns_none(self):
        res = watch_pasted_edit(self.db, self.hid, BASE, lambda: BASE,
                                window_seconds=0.3, poll_interval=0.05)
        self.assertIsNone(res)

    def test_learns_after_threshold_and_no_double_hit(self):
        """Порог 2 + отсутствие двойного hit_count (add_word уже инкрементит)."""
        db, hid = self.db, self.hid
        for _ in range(2):
            record_edit_and_learn(db, hid, BASE, BASE.replace("8081", "8080"))
        # delete_date: первый вызов накапливает кандидата, второй промотит
        rows = db._conn.execute("SELECT source_word, target_word, hit_count FROM dictionary").fetchall()
        promoted = [r for r in rows if r["source_word"] == "8081"]
        self.assertEqual(len(promoted), 1, f"пара 8081->8080 должна быть одна: {rows}")
        self.assertEqual(promoted[0]["target_word"], "8080")
        self.assertEqual(promoted[0]["hit_count"], 1,
                         "hit_count не должен удваиваться (bump_word убран)")


class TestCandidateKeyPerApp(unittest.TestCase):
    def test_different_windows_do_not_share_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "app.sqlite")
            h1 = db.insert_history(DictationRecord(
                asr_text=BASE, polished_text=BASE, app_window="opencode.exe",
                duration_ms=1000, e2e_latency_ms=100))
            h2 = db.insert_history(DictationRecord(
                asr_text=BASE, polished_text=BASE, app_window="telegram.exe",
                duration_ms=1000, e2e_latency_ms=100))
            edited = BASE.replace("8081", "8080")
            record_edit_and_learn(db, h1, BASE, edited)
            record_edit_and_learn(db, h2, BASE, edited)
            rows = db._conn.execute("SELECT COUNT(*) FROM dictionary").fetchone()[0]
            self.assertEqual(rows, 0, "правки в разных окнах не должны «складываться»")
            db.close()


if __name__ == "__main__":
    unittest.main()
