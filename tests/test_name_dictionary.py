"""Словарь имён: добавление, показ и удаление — обратимо и без мусора.

Замер на живой начитке показал: GigaAM чисто слышит обычные слова и путается в именах
собственных. Имена идут в словарь, который MyFLOW подмешивает полировщику. Здесь
проверяется, что записи добавляются идемпотентно, помечаются своим source и удаляются
целиком — то есть эксперимент всегда можно откатить.
"""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.name_dictionary import (MISHEARD, SOURCE, add_names,  # noqa: E402
                                   remove_names)


def make_db() -> tuple[sqlite3.Connection, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    con = sqlite3.connect(Path(tmp.name) / "test.sqlite")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE dictionary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_word TEXT NOT NULL COLLATE NOCASE,
        target_word TEXT NOT NULL,
        source TEXT,
        app_mask TEXT DEFAULT '',
        hit_count INTEGER DEFAULT 0,
        last_used TEXT,
        UNIQUE(source_word, app_mask))""")
    con.execute("insert into dictionary (source_word, target_word, source) "
                "values ('она', 'и', 'learned')")     # чужой мусор — не трогаем
    con.commit()
    return con, tmp


class NameDictionaryTest(unittest.TestCase):
    def test_adds_all_names_once(self):
        con, tmp = make_db()
        added, skipped = add_names(con)
        self.assertEqual(added, len(MISHEARD))
        self.assertEqual(skipped, 0)
        again, skipped_again = add_names(con)
        self.assertEqual(again, 0, "повторный запуск ничего не добавляет")
        self.assertEqual(skipped_again, len(MISHEARD))
        con.close()
        tmp.cleanup()

    def test_entries_are_marked_with_our_source(self):
        con, tmp = make_db()
        add_names(con)
        rows = list(con.execute("select source from dictionary where source=?",
                                (SOURCE,)))
        self.assertEqual(len(rows), len(MISHEARD))
        con.close()
        tmp.cleanup()

    def test_only_proper_names_no_common_words(self):
        """В словарь не должны попадать обычные слова — иначе испортим текст."""
        heard = {heard for heard, _correct in MISHEARD}
        for common in ("глотка", "племени", "и", "она", "там", "как"):
            self.assertNotIn(common, heard, f"обычное слово {common!r} в словаре имён")

    def test_remove_takes_only_ours(self):
        con, tmp = make_db()
        add_names(con)
        removed = remove_names(con)
        self.assertEqual(removed, len(MISHEARD))
        left = [row["source_word"] for row in con.execute("select source_word from dictionary")]
        self.assertEqual(left, ["она"], "чужие записи остались на месте")
        con.close()
        tmp.cleanup()

    def test_scope_is_respected(self):
        con, tmp = make_db()
        added, _ = add_names(con, scope="notepad.exe")
        self.assertEqual(added, len(MISHEARD))
        scoped = list(con.execute("select app_mask from dictionary where source=?",
                                  (SOURCE,)))
        self.assertTrue(all(row["app_mask"] == "notepad.exe" for row in scoped))
        con.close()
        tmp.cleanup()
