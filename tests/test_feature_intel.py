"""Тесты новых фич из Wispr-интела: сниппеты, divergence, аудио-хранилище, категории."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from myflow.audio_store import AudioStore
from myflow.config import Settings, detect_category
from myflow.db import Database, DictationRecord
from myflow.textdiff import divergence_score, similarity_score, tokens, word_distance


# ---------------------------------------------------------------- сниппеты
class SnippetsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "s.sqlite")

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def test_add_and_find_case_insensitive(self):
        self.db.add_snippet("Мой email", "user@example.com")
        found = self.db.find_snippet("мой   EMAIL")   # нормализация: регистр/пробелы
        self.assertIsNotNone(found)
        self.assertEqual(found["replacement"], "user@example.com")

    def test_update_same_phrase(self):
        self.db.add_snippet("промпт", "первый")
        self.db.add_snippet("промпт", "второй")
        rows = self.db.list_snippets()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["replacement"], "второй")

    def test_bump_and_order(self):
        self.db.add_snippet("а", "1")
        self.db.add_snippet("б", "2")
        for _ in range(3):
            self.db.bump_snippet("б")
        rows = self.db.list_snippets()
        self.assertEqual(rows[0]["phrase"], "б")
        self.assertEqual(rows[0]["hit_count"], 3)

    def test_delete_by_id_and_phrase(self):
        self.db.add_snippet("а", "1")
        self.db.add_snippet("б", "2")
        self.assertEqual(self.db.delete_snippet("а"), 1)
        rid = self.db.list_snippets()[0]["id"]
        self.assertEqual(self.db.delete_snippet(str(rid)), 1)
        self.assertEqual(self.db.list_snippets(), [])

    def test_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.db.add_snippet("  ", "текст")
        with self.assertRaises(ValueError):
            self.db.add_snippet("фраза", "   ")

    def test_no_match(self):
        self.db.add_snippet("мой email", "x@y.z")
        self.assertIsNone(self.db.find_snippet("совсем другое"))

    def test_history_stores_snippet_status(self):
        hid = self.db.insert_history(DictationRecord(
            asr_text="мой email", polished_text="user@example.com",
            app_window="chat.exe | Telegram", duration_ms=900,
            status="snippet", app_category="chat", divergence=0.0))
        row = self.db._conn.execute("SELECT status, app_category FROM history WHERE id=?",
                                    (hid,)).fetchone()
        self.assertEqual(row["status"], "snippet")
        self.assertEqual(row["app_category"], "chat")


# ---------------------------------------------------------------- divergence
class TextDiffTest(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(tokens("Привет, мир! 123"), ["привет", "мир", "123"])

    def test_distance(self):
        self.assertEqual(word_distance([], []), 0)
        self.assertEqual(word_distance(["a", "b"], ["a", "b"]), 0)
        self.assertEqual(word_distance(["a", "b"], ["a", "c"]), 1)
        self.assertEqual(word_distance(["a"], []), 1)

    def test_divergence_bounds(self):
        self.assertEqual(divergence_score("привет мир", "привет мир"), 0.0)
        self.assertEqual(divergence_score("a b c d", "1 2 3 4"), 1.0)
        self.assertGreater(divergence_score("открыть порт 8081", "открыть порт 8080"), 0.0)
        self.assertLess(divergence_score("открыть порт 8081", "Открыть порт 8080."), 0.5)
        self.assertEqual(similarity_score("а б", "а б"), 1.0)

    def test_punctuation_only_is_not_divergence(self):
        """Пунктуация и регистр — работа полировки, а не «переписывание»."""
        self.assertEqual(divergence_score("проверка связи", "Проверка связи."), 0.0)
        self.assertEqual(divergence_score("привет как дела", "Привет, как дела?"), 0.0)

    def test_truncation_has_high_divergence(self):
        raw = "сегодня обсудили план проекта сроки бюджеты риски и исполнителей"
        self.assertGreater(divergence_score(raw, "План проекта."), 0.5)


# ---------------------------------------------------------------- аудио
class AudioStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "audio"
        self.store = AudioStore(self.dir, keep=3)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _pcm(ms: int = 500) -> np.ndarray:
        n = int(16000 * ms / 1000)
        t = np.arange(n) / 16000
        return (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)

    def test_roundtrip_zlib(self):
        pcm = self._pcm()
        with mock.patch("myflow.audio_store._soundfile", return_value=None):
            path = self.store.save(1, pcm)
            self.assertIsNotNone(path)
            self.assertTrue(str(path).endswith(".pcm.z"))
            back = self.store.load(path)
        self.assertIsNotNone(back)
        self.assertEqual(len(back), len(pcm))
        self.assertTrue(np.array_equal(back, pcm))

    def test_compression_helps(self):
        pcm = self._pcm(1000)
        with mock.patch("myflow.audio_store._soundfile", return_value=None):
            path = self.store.save(2, pcm)
        raw = pcm.nbytes
        self.assertLess(path.stat().st_size, raw, "zlib должен сжимать PCM")
        ratio = path.stat().st_size / raw
        self.assertLess(ratio, 1.0)

    def test_prune_keeps_last(self):
        with mock.patch("myflow.audio_store._soundfile", return_value=None):
            for i in range(6):
                self.store.save(i, self._pcm(100))
        self.assertEqual(len(self.store.files()), 3, "ротация должна оставить 3 файла")

    def test_path_for_and_stats(self):
        with mock.patch("myflow.audio_store._soundfile", return_value=None):
            self.store.save(42, self._pcm(100))
        self.assertIsNotNone(self.store.path_for(42))
        st = self.store.stats()
        self.assertEqual(st["files"], 1)
        self.assertGreater(st["bytes"], 0)

    def test_load_missing_and_broken(self):
        self.assertIsNone(self.store.load(self.dir / "нет.pcm.z"))
        self.dir.mkdir(parents=True, exist_ok=True)
        bad = self.dir / "9.pcm.z"
        bad.write_bytes("мусор".encode("utf-8"))
        self.assertIsNone(self.store.load(bad))

    def test_save_empty_is_noop(self):
        self.assertIsNone(self.store.save(0, np.zeros(0, dtype=np.int16)))


# ---------------------------------------------------------------- категории
class CategoryTest(unittest.TestCase):
    def test_detect(self):
        cases = {
            "windowsterminal.exe | pwsh": "terminal",
            "powershell.exe | ...": "terminal",
            "code.exe | main.py - VS Code": "code",
            "telegram.exe | Чат": "chat",
            "outlook.exe | Входящие": "email",
            "notepad.exe | notes.txt": "notes",
            "winword.exe | Документ1": "docs",
            "chrome.exe | Google": "browser",
            "unknown": "other",
            "": "other",
        }
        for app, expected in cases.items():
            self.assertEqual(detect_category(app), expected, app)

    def test_no_false_positive_on_substring(self):
        # "encode.exe" не должен попасть в "code" по подстроке "code"
        self.assertEqual(detect_category("encode.exe | encoder"), "other")

    def test_style_by_category_defaults(self):
        s = Settings()
        self.assertEqual(s.style_by_category["terminal"], "terminal")
        self.assertEqual(s.style_by_category["chat"], "casual")
        self.assertEqual(s.style_by_category["email"], "formal")

    def test_flags_defaults(self):
        f = Settings().flags
        for flag in ("snippets", "keep_audio", "ax_observation", "learning", "live_preview"):
            self.assertIn(flag, f)


if __name__ == "__main__":
    unittest.main()
