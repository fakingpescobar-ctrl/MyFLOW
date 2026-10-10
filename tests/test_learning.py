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


class TestConfidentLearning(unittest.TestCase):
    """Уверенные правки учим сразу: словарь иначе пуст (1 запись на 359 диктовок)."""

    def test_confident_rules(self):
        from myflow.learning import is_confident_correction
        yes = {
            ("туи", "TUI"): "латиница",
            ("туи", "tui"): "латиница",
            ("джигаам", "GigaAM"): "латиница",
            ("майфлоу", "MyFLOW"): "латиница",
            ("ообщеение", "общение"): "длинное похожее слово",
            ("воспризведение", "воспроизведение"): "длинное похожее слово",
        }
        for (src, dst), why in yes.items():
            with self.subTest(pair=(src, dst)):
                self.assertIsNotNone(is_confident_correction(src, dst), why)

    def test_grammar_is_not_confident(self):
        """«сказал -> сказала», «это -> этот», «делай -> сделай» — это НЕ ошибка ASR."""
        from myflow.learning import is_confident_correction
        for src, dst in (("сказал", "сказала"), ("это", "этот"), ("делай", "сделай"),
                         ("дом", "стол"), ("ну", "да"), ("идёт", "идут")):
            with self.subTest(pair=(src, dst)):
                self.assertIsNone(is_confident_correction(src, dst),
                                  f"«{src}» -> «{dst}» учить сразу нельзя")

    def _db_with(self, asr, polished, window="Тест — приложение"):
        import tempfile
        from myflow.db import Database, DictationRecord
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Database(Path(tmp.name) / "learn.sqlite")
        self.addCleanup(db.close)
        hid = db.insert_history(DictationRecord(
            asr_text=asr, polished_text=polished, app_window=window,
            duration_ms=2000, e2e_latency_ms=400, detected_language="ru",
            transcript_origin="gigaam", style="plain", status="ok"))
        return db, hid, window

    def test_term_learned_from_single_edit(self):
        from myflow.learning import record_edit_and_learn
        db, hid, window = self._db_with("Наш туи тормозит.", "Наш туи тормозит.")
        res = record_edit_and_learn(db, hid, "Наш туи тормозит.", "Наш TUI тормозит.")
        self.assertEqual(res["new_words"], 1, "термин должен выучиться с первой правки")
        words = db.get_dictionary(window)
        self.assertTrue(any(w["target_word"].lower() == "tui" for w in words), words)

    def test_grammar_edit_not_learned_immediately(self):
        from myflow.learning import record_edit_and_learn
        db, hid, window = self._db_with("Он сказал что придёт.", "Он сказал что придёт.")
        res = record_edit_and_learn(db, hid, "Он сказал что придёт.",
                                    "Он сказала что придёт.")
        self.assertEqual(res["new_words"], 0)
        self.assertEqual(db.get_dictionary(window), [])

    def test_validator_learns_current_pair_only(self):
        """Модель подтверждает пару — учим; про старые кандидаты не спрашиваем."""
        from myflow.learning import record_edit_and_learn
        db, hid, window = self._db_with("Он сказал что придёт.", "Он сказал что придёт.")
        record_edit_and_learn(db, hid, "Он сказал что придёт.", "Он сказала что придёт.")
        asked: list[tuple[str, str]] = []

        def validator(src, dst):
            asked.append((src, dst))
            return (src, dst) == ("каннект", "кинект")

        db2, hid2, window2 = self._db_with("Подключи каннект.", "Подключи каннект.",
                                          window=window)
        res = record_edit_and_learn(db2, hid2, "Подключи каннект.", "Подключи кинект.",
                                    validator=validator)
        self.assertEqual(res["new_words"], 1)
        self.assertIn(("каннект", "кинект"), asked)
        self.assertNotIn(("сказал", "сказала"), asked,
                         "старые кандидаты в этой правке не участвуют")


class TestNoiseFilter(unittest.TestCase):
    """Фильтр «слово -> число»: не учим совпадения с чужим скопированным текстом."""

    def test_word_to_number_is_not_learned(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "noise.sqlite")
            base = "кнопка overlay открывает меню"
            edited = "кнопка 243 открывает меню"
            hid = db.insert_history(DictationRecord(
                asr_text=base, polished_text=base, app_window="test",
                duration_ms=1000, e2e_latency_ms=100))
            for _ in range(3):
                record_edit_and_learn(db, hid, base, edited)
            rows = db._conn.execute("SELECT COUNT(*) FROM dictionary").fetchone()[0]
            self.assertEqual(rows, 0, "«overlay -> 243» не должно попасть в словарь")
            db.close()

    def test_number_to_number_is_still_learned(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "num.sqlite")
            hid = db.insert_history(DictationRecord(
                asr_text="порт 8081", polished_text="порт 8081", app_window="test",
                duration_ms=1000, e2e_latency_ms=100))
            for _ in range(2):
                record_edit_and_learn(db, hid, "порт 8081", "порт 8080")
            rows = db._conn.execute(
                "SELECT source_word, target_word FROM dictionary").fetchall()
            self.assertEqual([tuple(r) for r in rows], [("8081", "8080")])
            db.close()


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
