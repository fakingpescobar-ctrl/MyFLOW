"""Красивый интерфейс: состояние собирается только на чтение и не падает без базы.

Интерфейс показывает то же, что панель, но по-человечески. Проверяем главное: он
ничего не меняет в данных, справляется с отсутствующей базой и отдаёт нужные поля
(последняя фраза, история, счётчики, режим вставки, настройки).
"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.webui import PAGE, build_state  # noqa: E402


class StateTest(unittest.TestCase):
    def setUp(self):
        # Каталог живёт до конца теста: если вернуть только путь, временная папка
        # соберётся сборщиком мусора, база исчезнет и тест соврёт «ready: False».
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)

    def make_db(self, rows: list[tuple]) -> Path:
        path = Path(self.tmp.name) / "myflow.sqlite"
        con = sqlite3.connect(path)
        # Схема как в реальной базе: приложение читает e2e_latency_ms, и без колонки
        # карточка задержки падала на тестовой базе.
        con.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, created_at TEXT, "
                    "asr_text TEXT, polished_text TEXT, edited_text TEXT, "
                    "duration_ms INTEGER, e2e_latency_ms INTEGER, transcript_origin TEXT)")
        con.execute("CREATE TABLE dictionary (id INTEGER PRIMARY KEY, source_word TEXT)")
        con.execute("CREATE TABLE snippets (id INTEGER PRIMARY KEY, phrase TEXT)")
        con.executemany("INSERT INTO history VALUES (?,?,?,?,?,?,?,?)", rows)
        con.commit()
        con.close()
        return path

    def make_settings(self, payload: dict) -> Path:
        path = Path(self.tmp.name) / "settings.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_database_is_safe(self):
        state = build_state(Path(self.tmp.name) / "нет-такой-базы.sqlite")
        self.assertFalse(state["ready"])
        self.assertEqual(state["history"], [])
        self.assertIsNone(state["last"])

    def test_reads_history_and_counts(self):
        db = self.make_db([
            (1, "2026-10-12 10:00:00", "первая фраза тут", "", "", 1200, 200, "gigaam"),
            (2, "2026-10-12 10:01:00", "вторая фраза здесь", "", "", 900, 150, "fast_path"),
        ])
        state = build_state(db, Path(self.tmp.name) / "нет.json", limit=5)
        self.assertTrue(state["ready"])
        self.assertEqual(state["counts"]["dictations"], 2)
        self.assertEqual(state["counts"]["with_text"], 2)
        self.assertEqual(len(state["history"]), 2)
        self.assertEqual(state["last"]["id"], 2, "последняя фраза — самая свежая")

    def test_prefers_edited_text_then_polished(self):
        db = self.make_db([
            (1, "2026-10-12 10:00:00", "сырое", "полированное", "правка", 1000, 300, "gigaam"),
        ])
        state = build_state(db, Path(self.tmp.name) / "нет.json")
        self.assertEqual(state["history"][0]["text"], "правка")

    def test_settings_reflected(self):
        settings = self.make_settings({
            "insert_method": "dsh",
            "polish": {"enabled": True, "postproc": False},
        })
        state = build_state(Path(self.tmp.name) / "нет.sqlite", settings)
        self.assertEqual(state["insert_mode"], "dsh")
        self.assertFalse(state["postproc"])

    def test_page_has_no_external_resources(self):
        """Страница должна работать без интернета: ни одного внешнего файла."""
        for marker in ("http://", "https://", "cdn."):
            self.assertNotIn(marker, PAGE,
                             "интерфейс не должен тянуть внешние ресурсы")
        self.assertIn("MyFLOW", PAGE)


if __name__ == "__main__":
    unittest.main()


class LatencyCardTest(unittest.TestCase):
    """Карточка задержки: медиана и последняя берутся из истории."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "myflow.sqlite"
        con = sqlite3.connect(self.db)
        con.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, created_at TEXT, "
                    "asr_text TEXT, polished_text TEXT, edited_text TEXT, "
                    "duration_ms INTEGER, e2e_latency_ms INTEGER, transcript_origin TEXT)")
        con.executemany("INSERT INTO history VALUES (?,?,?,?,?,?,?,?)", [
            (1, "2026-10-12 10:00:00", "раз два три", "", "", 1000, 100, "gigaam"),
            (2, "2026-10-12 10:01:00", "четыре пять", "", "", 1100, 300, "gigaam"),
            (3, "2026-10-12 10:02:00", "шесть семь восемь", "", "", 1200, 200, "gigaam"),
        ])
        con.commit()
        con.close()

    def test_median_and_last(self):
        state = build_state(self.db, Path(self.tmp.name) / "нет.json")
        self.assertEqual(state["latency_median_ms"], 200)
        self.assertEqual(state["latency_ms"], 200, "последняя фраза")
        self.assertEqual(state["latency_worst_ms"], 300)
        self.assertEqual(state["words_last"], 3)
        self.assertEqual(state["duration_ms"], 1200)
