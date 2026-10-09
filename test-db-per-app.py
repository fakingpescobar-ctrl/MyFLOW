"""Временный тест per-app словаря (без реальных серверов).

Проверяем: миграцию старой схемы, add_word/get_dictionary с app-масками,
per-app обучение через record_edit_and_learn (2 правки в одном окне -> promo).
"""
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from myflow.db import Database
from myflow.learning import record_edit_and_learn

TMP = Path(tempfile.mkdtemp(prefix="myflow_db_test_"))


def test_migration():
    """Старая схема (без app_mask) мигрирует, данные сохраняются."""
    db_path = TMP / "migrate.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            asr_text TEXT NOT NULL, polished_text TEXT NOT NULL,
            edited_text TEXT, app_window TEXT, duration_ms INTEGER,
            e2e_latency_ms INTEGER, detected_language TEXT,
            transcript_origin TEXT, style TEXT, edit_distance INTEGER,
            corrections_json TEXT
        );
        CREATE TABLE dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_word TEXT NOT NULL UNIQUE COLLATE NOCASE,
            target_word TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual',
            hit_count INTEGER NOT NULL DEFAULT 1,
            last_used TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE settings_kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO dictionary (source_word, target_word, source) VALUES ('рeбот', 'ребят', 'manual');
        """
    )
    conn.close()
    db = Database(db_path)
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(dictionary)")}
    assert "app_mask" in cols, f"колонка app_mask не добавлена: {cols}"
    dic = db.get_dictionary()
    assert any(d["source_word"] == "рeбот" for d in dic), "данные потеряны при миграции"
    print("  1) миграция старой схемы + сохранение данных — OK")


def insert_history(db: Database, app: str) -> int:
    from myflow.db import DictationRecord
    return db.insert_history(DictationRecord(
        asr_text="тест", polished_text="тест", app_window=app, duration_ms=500))


def test_app_scoping():
    db = Database(TMP / "scope.sqlite")
    db.add_word("какчи", "кэш", source="learned")                    # глобальный
    db.add_word("комит", "commit", source="learned", app_mask="Visual Studio Code")
    db.add_word("комит", "commit", source="learned", app_mask="telegram")

    all_global = db.get_dictionary(None)
    assert [d["source_word"] for d in all_global] == ["какчи"], all_global

    vs = db.get_dictionary("Visual Studio Code — main.py")
    words = {d["source_word"] for d in vs}
    assert words == {"какчи", "комит"}, f"VS словарь неполный: {words}"

    tg = db.get_dictionary("TelegramDesktop (2)")
    words = {d["source_word"] for d in tg}
    assert words == {"какчи", "комит"}, tg

    other = db.get_dictionary("Notepad")
    words = {d["source_word"] for d in other}
    assert words == {"какчи"}, f"чужое окно подхватило per-app: {other}"
    print("  2) per-app выборка по активному окну — OK")


def test_learning_per_app():
    db = Database(TMP / "learn.sqlite")
    # Окно VS Code: дважды правим «спешалли» -> «специально» — учим ДЛЯ VS
    h1 = insert_history(db, "Visual Studio Code")
    r1 = record_edit_and_learn(db, h1, "сделать спешалли", "сделать специально")
    assert r1["new_words"] == 0, r1
    h2 = insert_history(db, "Visual Studio Code")
    r2 = record_edit_and_learn(db, h2, "проверить спешалли", "проверить специально")
    assert r2["new_words"] == 1, f"та же пара в том же окне не промо: {r2}"

    vs = db.get_dictionary("Visual Studio Code")
    assert any(d["source_word"] == "спешалли" and d["target_word"] == "специально" for d in vs), vs
    tg = db.get_dictionary("Telegram")
    assert all(d["source_word"] != "спешалли" for d in tg), f"per-app протёк в другое окно: {tg}"
    print("  3) обучение per-app (порог 2 в одном окне, изоляция окон) — OK")


def test_learning_unknown_global():
    db = Database(TMP / "learn_unknown.sqlite")
    h = insert_history(db, "unknown")
    record_edit_and_learn(db, h, "это рeбот", "это ребят")
    record_edit_and_learn(db, h, "тот рeбот", "тот ребят")
    dic = db.get_dictionary(None)  # глобально
    assert any(d["source_word"] == "рeбот" for d in dic), "unknown-окно не дало глобальный словарь"
    print("  4) правки в unknown-окне -> глобальный словарь — OK")


if __name__ == "__main__":
    test_migration()
    test_app_scoping()
    test_learning_per_app()
    test_learning_unknown_global()
    print("DB PER-APP TEST: ALL OK")