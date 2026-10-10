"""MyFLOW — локальная БД (зеркало flow.sqlite у Wispr Flow).

Таблицы:
  - history:  каждая диктовка (asr -> polished -> edited для обучения)
  - dictionary: персональный словарь и автовыученные замены
  - settings_kv: служебные ключ-значения
"""
from __future__ import annotations

import math
import re
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
    # Новое (аудит 10.10.2026): метрики надёжности и качества, как у Wispr.
    status: str = "ok"                  # ok | snippet | empty_asr | asr_error | polish_error | skipped
    app_category: str | None = None     # terminal | chat | email | code | notes | docs | browser
    divergence: float | None = None     # formattingDivergenceScore
    observation_end_reason: str | None = None   # edited | observation_window_elapsed | ...
    audio_path: str | None = None       # файл с аудио фразы (для retry)


class Database:
    def __init__(self, path: Path = DB_FILE) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        # WAL + busy_timeout: у watcher обучения своё соединение, без этого
        # ловим "database is locked" на каждой правке.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.DatabaseError:
            pass  # :memory: и экзотика — не критично
        self._init_schema()

    # Каноническая схема словаря. app_mask = '' значит «глобальная пара»:
    # NULL в UNIQUE-индексе SQLite считает РАЗНЫМИ значениями, из-за чего
    # ON CONFLICT не срабатывал и глобальные пары дублировались.
    _DICT_DDL = """
        CREATE TABLE IF NOT EXISTS dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_word TEXT NOT NULL COLLATE NOCASE,
            target_word TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual',   -- manual | learned
            app_mask TEXT NOT NULL DEFAULT '',       -- '' = глобальная пара
            hit_count INTEGER NOT NULL DEFAULT 1,
            last_used TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(source_word, app_mask)
        );
    """

    # Сниппеты (как Dictionary.isSnippet у Wispr): голосовая фраза -> готовый текст.
    # «мой email» -> адрес, «организуй мысли» -> длинный промпт и т.п.
    _SNIPPET_DDL = """
        CREATE TABLE IF NOT EXISTS snippets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phrase TEXT NOT NULL COLLATE NOCASE,      -- что произносим (нормализовано)
            replacement TEXT NOT NULL,                -- что вставляем
            source TEXT NOT NULL DEFAULT 'manual',    -- manual | default
            hit_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_used TEXT,
            UNIQUE(phrase)
        );
    """

    # Колонки истории, добавленные после первой версии (миграция ALTER-ом).
    _HISTORY_MIGRATIONS: tuple[tuple[str, str], ...] = (
        ("status", "TEXT NOT NULL DEFAULT 'ok'"),
        ("app_category", "TEXT"),
        ("divergence", "REAL"),
        ("observation_end_reason", "TEXT"),
        ("audio_path", "TEXT"),
    )

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
                CREATE TABLE IF NOT EXISTS settings_kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_history_created ON history(created_at);
                CREATE INDEX IF NOT EXISTS ix_history_app ON history(app_window);
                """
                + self._DICT_DDL
                + self._SNIPPET_DDL
            )
            # Миграции history: ALTER-ом добавляем колонки метрик (старые базы
            # обновляются на месте, данные не трогаем).
            have = {r[1] for r in self._conn.execute("PRAGMA table_info(history)")}
            for name, ddl in self._HISTORY_MIGRATIONS:
                if name not in have:
                    self._conn.execute(f"ALTER TABLE history ADD COLUMN {name} {ddl}")
            # Миграция словаря к канонической схеме:
            #  - старая таблица без app_mask;
            #  - app_mask nullable (NULL-дубли и неработающий ON CONFLICT);
            #  - уже накопленные NULL-строки.
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(dictionary)")}
            has_app_mask = "app_mask" in cols
            app_mask_notnull = any(
                r[1] == "app_mask" and r[3] for r in self._conn.execute("PRAGMA table_info(dictionary)")
            )
            has_nulls = False
            if has_app_mask:
                has_nulls = self._conn.execute(
                    "SELECT 1 FROM dictionary WHERE app_mask IS NULL LIMIT 1"
                ).fetchone() is not None
            if (not has_app_mask) or (not app_mask_notnull) or has_nulls:
                mask_expr = "COALESCE(app_mask, '')" if has_app_mask else "''"
                self._conn.executescript(
                    f"""
                    CREATE TABLE dictionary_mig (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_word TEXT NOT NULL COLLATE NOCASE,
                        target_word TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'manual',
                        app_mask TEXT NOT NULL DEFAULT '',
                        hit_count INTEGER NOT NULL DEFAULT 1,
                        last_used TEXT NOT NULL DEFAULT (datetime('now')),
                        UNIQUE(source_word, app_mask)
                    );
                    INSERT INTO dictionary_mig
                        (source_word, target_word, source, app_mask, hit_count, last_used)
                    SELECT source_word,
                           target_word,
                           CASE WHEN SUM(source = 'manual') > 0 THEN 'manual' ELSE 'learned' END,
                           {mask_expr},
                           SUM(hit_count),
                           MAX(last_used)
                      FROM dictionary
                     GROUP BY LOWER(source_word), {mask_expr};
                    DROP TABLE dictionary;
                    ALTER TABLE dictionary_mig RENAME TO dictionary;
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
        self, history_id: int, edited_text: str, edit_distance: int, corrections_json: str,
        observation_end_reason: str | None = None,
    ) -> None:
        """Юзер вручную правил текст — фиксируем правку (сырьё для обучения)."""
        with self._lock, self._conn:
            if observation_end_reason:
                self._conn.execute(
                    """UPDATE history
                       SET edited_text=?, edit_distance=?, corrections_json=?,
                           observation_end_reason=?
                       WHERE id=?""",
                    (edited_text, edit_distance, corrections_json,
                     observation_end_reason, history_id),
                )
            else:
                self._conn.execute(
                    """UPDATE history
                       SET edited_text=?, edit_distance=?, corrections_json=?
                       WHERE id=?""",
                    (edited_text, edit_distance, corrections_json, history_id),
                )

    def set_observation_reason(self, history_id: int, reason: str) -> None:
        """Причина окончания наблюдения за полем (аналог contentObservationEndReason)."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE history SET observation_end_reason=? WHERE id=?",
                (reason, history_id),
            )

    def mark_status(self, history_id: int, status: str) -> None:
        """Сменить статус записи (например, retried / audio_missing)."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE history SET status=? WHERE id=?", (status, history_id))

    def deferred_polish(self, limit: int = 20) -> list[dict[str, Any]]:
        """Фразы, у которых полировка не удалась (сервер LLM лежал).

        Это очередь автолечения: как только LLM оживёт, дополируем их из
        сохранённого текста ASR — аудио для этого не нужно.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, asr_text, polished_text, style, app_window, created_at
                   FROM history
                   WHERE status = 'polish_error'
                     AND asr_text IS NOT NULL AND TRIM(asr_text) <> ''
                   ORDER BY id DESC LIMIT ?""",
                (int(limit),)).fetchall()
            return [dict(r) for r in rows]

    def count_deferred_polish(self) -> int:
        """Сколько фраз ждут дополировки (для панели)."""
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) AS n FROM history
                   WHERE status = 'polish_error'
                     AND asr_text IS NOT NULL AND TRIM(asr_text) <> ''""").fetchone()
            return int(row["n"]) if row else 0

    def set_polished(self, history_id: int, text: str,
                     divergence: float | None = None) -> None:
        """Записать дополированный текст (после того как LLM ожила)."""
        with self._lock, self._conn:
            if divergence is None:
                self._conn.execute(
                    "UPDATE history SET polished_text=? WHERE id=?", (text, history_id))
            else:
                self._conn.execute(
                    "UPDATE history SET polished_text=?, divergence=? WHERE id=?",
                    (text, float(divergence), history_id))

    def last_retryable(self) -> dict | None:
        """Последняя неуспешная фраза, для которой сохранено аудио."""
        with self._lock, self._conn:
            row = self._conn.execute(
                """SELECT * FROM history
                   WHERE status IN ('asr_error', 'polish_error', 'empty_asr')
                     AND audio_path IS NOT NULL
                   ORDER BY id DESC LIMIT 1"""
            ).fetchone()
            return dict(row) if row else None

    # ---- Snippets (голос -> готовый текст, как isSnippet у Wispr) ----

    def add_snippet(self, phrase: str, replacement: str, source: str = "manual") -> None:
        """Добавить/обновить сниппет. phrase нормализуется (lower, один пробел)."""
        key = " ".join((phrase or "").strip().lower().split())
        if not key or not (replacement or "").strip():
            raise ValueError("сниппет: нужны непустые phrase и replacement")
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO snippets (phrase, replacement, source)
                   VALUES (?, ?, ?)
                   ON CONFLICT(phrase) DO UPDATE SET
                     replacement=excluded.replacement,
                     source=excluded.source""",
                (key, replacement, source),
            )

    def list_snippets(self) -> list[dict[str, Any]]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                """SELECT id, phrase, replacement, source, hit_count, created_at, last_used
                   FROM snippets ORDER BY hit_count DESC, phrase"""
            ).fetchall()
            return [dict(r) for r in rows]

    def find_snippet(self, phrase: str) -> dict[str, Any] | None:
        """Точное совпадение по нормализованной фразе."""
        key = " ".join((phrase or "").strip().lower().split())
        if not key:
            return None
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM snippets WHERE phrase=?", (key,)
            ).fetchone()
            return dict(row) if row else None

    def bump_snippet(self, phrase: str) -> None:
        key = " ".join((phrase or "").strip().lower().split())
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE snippets SET hit_count=hit_count+1, last_used=datetime('now')
                   WHERE phrase=?""",
                (key,),
            )

    def delete_snippet(self, ident: str) -> int:
        """Удалить по id (число) или по фразе. Возвращает число удалённых строк."""
        with self._lock, self._conn:
            if str(ident).strip().isdigit():
                cur = self._conn.execute("DELETE FROM snippets WHERE id=?", (int(ident),))
            else:
                key = " ".join(str(ident).strip().lower().split())
                cur = self._conn.execute("DELETE FROM snippets WHERE phrase=?", (key,))
            return int(cur.rowcount or 0)

    # ---- Dictionary ----

    def add_word(self, source_word: str, target_word: str,
                 source: str = "manual", app_mask: str | None = None) -> None:
        """Добавить пару слово -> замена. source: manual | learned.

        app_mask=None/'' — глобальная пара (все окна); иначе пара действует
        только в окнах, чей заголовок содержит маску (lower).
        """
        mask = (app_mask or "").strip().lower()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO dictionary (source_word, target_word, source, app_mask)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(source_word, app_mask) DO UPDATE SET
                     target_word=excluded.target_word,
                     source=excluded.source,
                     hit_count=hit_count+1,
                     last_used=datetime('now')""",
                (source_word.strip().lower(), target_word.strip(), source, mask),
            )

    def bump_word(self, source_word: str, app_mask: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE dictionary SET hit_count=hit_count+1, last_used=datetime('now')
                   WHERE source_word=? AND app_mask=?""",
                (source_word.strip().lower(), (app_mask or "").strip().lower()),
            )

    def get_dictionary(self, app_title: str | None = None) -> list[dict[str, Any]]:
        """Словарь для диктовки: глобальный + per-app для активного окна.

        Сортировка по частоте: в промпт попадают самые полезные пары
        (раньше [:50] резало произвольные строки).
        """
        with self._lock, self._conn:
            if app_title:
                rows = self._conn.execute(
                    """SELECT source_word, target_word FROM dictionary
                       WHERE app_mask = ''
                          OR (? LIKE '%' || app_mask || '%')
                       ORDER BY hit_count DESC, last_used DESC""",
                    (app_title.lower(),),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT source_word, target_word FROM dictionary
                       WHERE app_mask = ''
                       ORDER BY hit_count DESC, last_used DESC"""
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

    def close(self) -> None:
        """Закрыть соединение (watcher обучения держит своё).

        Перед закрытием ужимаем WAL: файл -wal иначе копится (в data/ видели
        378 КБ при базе 360 КБ) и остаётся мусором после выхода.
        """
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def prune_history(self, keep_days: int = 90) -> int:
        """Удалить историю старше keep_days (0/None = не чистить).

        Приватность (в истории — полный текст диктовок) и размер файла.
        """
        if not keep_days or keep_days <= 0:
            return 0
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM history WHERE created_at < datetime('now', ?)",
                (f"-{int(keep_days)} days",),
            )
            return int(cur.rowcount or 0)

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
        """Сводка: объёмы + метрики надёжности/качества (аналог их телеметрии)."""
        with self._lock, self._conn:
            d = self._conn.execute(
                """SELECT COUNT(*) n, ROUND(AVG(e2e_latency_ms),0) avg_lat,
                          ROUND(AVG(duration_ms),0) avg_dur,
                          ROUND(AVG(divergence),3) avg_div
                   FROM history"""
            ).fetchone()
            words = self._conn.execute("SELECT COUNT(*) FROM dictionary").fetchone()[0]
            snippets = self._conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
            by_status = {
                r["status"] or "ok": r["n"]
                for r in self._conn.execute(
                    "SELECT status, COUNT(*) n FROM history GROUP BY status ORDER BY n DESC")
            }
            total = d["n"] or 0
            errors = sum(v for k, v in by_status.items()
                         if k in ("asr_error", "polish_error", "empty_asr"))
            latencies = [r[0] for r in self._conn.execute(
                "SELECT e2e_latency_ms FROM history WHERE e2e_latency_ms IS NOT NULL "
                "ORDER BY e2e_latency_ms")]

        def pct(q: float) -> int | None:
            """Перцентиль методом nearest-rank (p50 из 4 значений = 2-е, не 3-е)."""
            if not latencies:
                return None
            rank = max(1, math.ceil(q * len(latencies)))
            return int(latencies[min(rank, len(latencies)) - 1])

        return {
            "dictations": total,
            "avg_latency_ms": d["avg_lat"],
            "avg_duration_ms": d["avg_dur"],
            "avg_divergence": d["avg_div"],
            "p50_latency_ms": pct(0.5),
            "p90_latency_ms": pct(0.9),
            "p99_latency_ms": pct(0.99),
            "errors": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "by_status": by_status,
            "dictionary_words": words,
            "snippets": snippets,
        }