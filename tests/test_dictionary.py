"""Тесты словаря (P0-5: NULL в UNIQUE давал дубли глобальных пар)."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from myflow.db import Database

OLD_SCHEMA_NO_MASK = """
CREATE TABLE dictionary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_word TEXT NOT NULL COLLATE NOCASE,
    target_word TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    hit_count INTEGER NOT NULL DEFAULT 1,
    last_used TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_word)
);
"""

OLD_SCHEMA_NULLABLE_MASK = """
CREATE TABLE dictionary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_word TEXT NOT NULL COLLATE NOCASE,
    target_word TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    app_mask TEXT,
    hit_count INTEGER NOT NULL DEFAULT 1,
    last_used TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_word, app_mask)
);
"""


class DictionaryTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "test.sqlite"

    def tearDown(self):
        self._tmp.cleanup()


class TestGlobalDedupe(DictionaryTestBase):
    def test_global_pair_is_updated_not_duplicated(self):
        db = Database(self.path)
        for _ in range(3):
            db.add_word("сухой лог", "Сухой Лог")
        rows = db._conn.execute("SELECT source_word, target_word, app_mask, hit_count FROM dictionary").fetchall()
        self.assertEqual(len(rows), 1, f"глобальная пара должна быть одна, а не {len(rows)}")
        self.assertEqual(rows[0]["app_mask"], "")
        self.assertEqual(rows[0]["hit_count"], 3)
        db.close()

    def test_per_app_pairs_are_separate(self):
        db = Database(self.path)
        db.add_word("тест", "тест-1", app_mask=None)
        db.add_word("тест", "тест-2", app_mask="code.exe")
        db.add_word("тест", "тест-2", app_mask="code.exe")
        rows = db._conn.execute("SELECT app_mask, target_word, hit_count FROM dictionary ORDER BY app_mask").fetchall()
        self.assertEqual(len(rows), 2)
        by_mask = {r["app_mask"]: r for r in rows}
        self.assertEqual(by_mask[""]["target_word"], "тест-1")
        self.assertEqual(by_mask["code.exe"]["target_word"], "тест-2")
        self.assertEqual(by_mask["code.exe"]["hit_count"], 2)
        db.close()

    def test_get_dictionary_global_and_per_app(self):
        db = Database(self.path)
        db.add_word("глобал", "Глобал")
        db.add_word("локально", "Локально", app_mask="opencode")
        self.assertEqual(len(db.get_dictionary(None)), 1)
        both = db.get_dictionary("OpenCode — chat")
        self.assertEqual(len(both), 2)
        db.close()


class TestMigrations(DictionaryTestBase):
    def test_migration_from_schema_without_mask(self):
        conn = sqlite3.connect(str(self.path))
        conn.executescript(OLD_SCHEMA_NO_MASK)
        conn.execute("INSERT INTO dictionary (source_word, target_word, source, hit_count) "
                     "VALUES ('а', 'Б', 'manual', 5)")
        conn.commit()
        conn.close()

        db = Database(self.path)
        rows = db._conn.execute("SELECT * FROM dictionary").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["app_mask"], "")
        self.assertEqual(rows[0]["hit_count"], 5)
        # и ON CONFLICT теперь работает
        db.add_word("а", "В")
        rows = db._conn.execute("SELECT target_word, hit_count FROM dictionary").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_word"], "В")
        self.assertEqual(rows[0]["hit_count"], 6)
        db.close()

    def test_migration_dedupes_existing_null_rows(self):
        conn = sqlite3.connect(str(self.path))
        conn.executescript(OLD_SCHEMA_NULLABLE_MASK)
        for _ in range(3):
            conn.execute("INSERT INTO dictionary (source_word, target_word, app_mask, hit_count) "
                         "VALUES ('дубль', 'Дубль', NULL, 2)")
        conn.execute("INSERT INTO dictionary (source_word, target_word, app_mask, hit_count) "
                     "VALUES ('дубль', 'Дубль-приложение', 'code.exe', 1)")
        conn.commit()
        conn.close()

        db = Database(self.path)
        rows = db._conn.execute("SELECT app_mask, hit_count FROM dictionary ORDER BY app_mask").fetchall()
        self.assertEqual(len(rows), 2, "три NULL-дубля должны схлопнуться в одну строку")
        self.assertEqual(rows[0]["app_mask"], "")
        self.assertEqual(rows[0]["hit_count"], 6)  # 2+2+2
        self.assertEqual(rows[1]["app_mask"], "code.exe")
        db.close()

    def test_migration_is_idempotent(self):
        db = Database(self.path)
        db.add_word("х", "Х")
        db.close()
        db2 = Database(self.path)  # повторное открытие не должно ломать схему
        self.assertEqual(len(db2.get_dictionary()), 1)
        db2.close()


class TestHistory(unittest.TestCase):
    def test_history_insert_and_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            from myflow.db import DictationRecord
            db = Database(Path(tmp) / "h.sqlite")
            hid = db.insert_history(DictationRecord(
                asr_text="привет", polished_text="Привет.", app_window="test",
                duration_ms=1000, e2e_latency_ms=100))
            self.assertEqual(db.get_polished_text(hid), "Привет.")
            db.attach_edit(hid, "Привет, мир.", 2, "[]")
            self.assertEqual(db.get_polished_text(hid), "Привет.")
            self.assertEqual(db.stats()["dictations"], 1)
            db.close()


if __name__ == "__main__":
    unittest.main()
