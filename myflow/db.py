"""MyFLOW — локальная БД (зеркало flow.sqlite у Wispr Flow).

Таблицы:
  - history:  каждая диктовка (asr -> polished -> edited для обучения)
  - dictionary: персональный словарь и автовыученные замены
  - settings_kv: служебные ключ-значения
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from myflow.config import DB_FILE

# Имя ASR-бэкенда для transcript_origin (как у Wispr: canto_1.0 / internal).
TRANSCRIPT_ORIGIN = "local_whisper_large_v3_turbo"


@dataclass
class DictationRecord:
    """Одна диктовка. edited_text заполняется юзером/системой позже."""
    asr_text: str
    polished_text: str
    app_window: str
    duration_ms: int
    e2e_latency_ms: int | None = None
    detected_language: str | None = None
    transcript_origin: str = TRANSCRIPT_ORIGIN
    style: str = "plain"
    edited_text: str | None = None
    edit_distance: int | None = None
    corrections_json: str | None = None


class Database:
    def __init__(self, path: Path = DB_FILE) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    asr_text TEXT NOT NULL,
                    polished_text TEXT NOT NULL,
                    edited_text TEXT,
                    app_window TEXT,
                    duration_ms INTEGER,
                    e2e_latency_ms INTEGER,
                    detected_language TEXT,
                    transcript_origin TEXT,
                    style TEXT,
                    edit_distance INTEGER,
                    corrections_json TEXT
                );
                CREATE TABLE IF NOT EXISTS dictionary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_word TEXT NOT NULL COLLATE NOCASE,
                    target_word TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',   -- manual | learned
                    app_mask TEXT,                            -- NULL = глобальный
                    hit_count INTEGER NOT NULL DEFAULT 1,
                    last_used TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(source_word, app_mask)
                );
                CREATE TABLE IF NOT EXISTS settings_kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            # Миграция старой схемы (без app_mask / с UNIQUE(source_word)):
            # пересоздаём таблицу с парным уникальным ключом, сохраняя данные.
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(dictionary)")}
            if "app_mask" not in cols:
                self._conn.executescript(
                    """
                    CREATE TABLE dictionary_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_word TEXT NOT NULL COLLATE NOCASE,
                        target_word TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'manual',
                        app_mask TEXT,
                        hit_count INTEGER NOT NULL DEFAULT 1,
                        last_used TEXT NOT NULL DEFAULT (datetime('now')),
                        UNIQUE(source_word, app_mask)
                    );
                    INSERT INTO dictionary_new
                        (source_word, target_word, source, hit_count, last_used)
                        SELECT source_word, target_word, source, hit_count, last_used
                        FROM dictionary;
                    DROP TABLE dictionary;
                    ALTER TABLE dictionary_new RENAME TO dictionary;
                    """
                )

    # ---- History ----

    def insert_history(self, rec: DictationRecord) -> int:
        d = asdict(rec)
        cols = ", ".join(d)
        qmarks = ", ".join("?" for _ in d)
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"INSERT INTO history ({cols}) VALUES ({qmarks})",
                list(d.values()),
            )
            return int(cur.lastrowid)

    def attach_edit(
        self, history_id: int, edited_text: str, edit_distance: int, corrections_json: str
    ) -> None:
        """Юзер вручную правил текст — фиксируем правку (сырьё для обучения)."""
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE history
                   SET edited_text=?, edit_distance=?, corrections_json=?
                   WHERE id=?""",
                (edited_text, edit_distance, corrections_json, history_id),
            )

    # ---- Dictionary ----

    def add_word(self, source_word: str, target_word: str,
                 source: str = "manual", app_mask: str | None = None) -> None:
        """Добавить пару слово -> замена. source: manual | learned.

        app_mask=None — глобальный (все окна); иначе пара действует только
        в окнах, чей заголовок содержит маску (lower).
        """
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO dictionary (source_word, target_word, source, app_mask)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(source_word, app_mask) DO UPDATE SET
                     target_word=excluded.target_word,
                     source=excluded.source,
                     hit_count=hit_count+1,
                     last_used=datetime('now')""",
                (source_word.strip().lower(), target_word.strip(), source,
                 (app_mask or "").strip().lower() or None),
            )

    def bump_word(self, source_word: str, app_mask: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE dictionary SET hit_count=hit_count+1, last_used=datetime('now')
                   WHERE source_word=? AND app_mask IS ?""",
                (source_word, None if not app_mask else app_mask.lower()),
            )

    def get_dictionary(self, app_title: str | None = None) -> list[dict[str, Any]]:
        """Словарь для диктовки: глобальный + per-app для активного окна.

        app_title — заголовок окна (lower-масок достаточно); None — только
        глобальные пары. Per-app пары идут после глобальных (LLM применит
        последнюю вхождение при дубликатах — специфичное перекрывает).
        """
        with self._lock, self._conn:
            if app_title:
                rows = self._conn.execute(
                    """SELECT source_word, target_word FROM dictionary
                       WHERE app_mask IS NULL
                          OR (? LIKE '%' || app_mask || '%')""",
                    (app_title.lower(),),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT source_word, target_word FROM dictionary WHERE app_mask IS NULL"
                ).fetchall()
            return [dict(r) for r in rows]

    # ---- Settings kv ----

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT value FROM settings_kv WHERE key=?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO settings_kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # ---- Statistics (для самодиагностики) ----

    def get_polished_text(self, history_id: int) -> str | None:
        """Текст диктовки, от которого юзер правил (для обучения)."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT polished_text FROM history WHERE id=?", (history_id,)
            ).fetchone()
            return row["polished_text"] if row else None

    def get_app_window(self, history_id: int) -> str | None:
        """Окно диктовки (для per-app обучения словаря)."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT app_window FROM history WHERE id=?", (history_id,)
            ).fetchone()
            return row["app_window"] if row else None

    def stats(self) -> dict[str, Any]:
        with self._lock, self._conn:
            d = self._conn.execute(
                """SELECT COUNT(*) n, ROUND(AVG(e2e_latency_ms),0) avg_lat,
                          ROUND(AVG(duration_ms),0) avg_dur
                   FROM history"""
            ).fetchone()
            words = self._conn.execute(
                "SELECT COUNT(*) FROM dictionary"
            ).fetchone()[0]
            return {"dictations": d["n"], "avg_latency_ms": d["avg_lat"], "avg_duration_ms": d["avg_dur"], "dictionary_words": words}