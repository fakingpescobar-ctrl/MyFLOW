"""Разбор начитки: что услышал GigaAM против текста главы.

Пользователь прочитал главу вслух; каждая фраза легла в историю MyFLOW. Здесь мы
склеиваем начитку в один текст и сравниваем с эталоном по словам (WER): видно, где
распознавание ошибается и какого типа ошибки преобладают.

    python tools/analyze_readaloud.py                    # последние 3 часа
    python tools/analyze_readaloud.py --hours 6 --show 25
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_readaloud import normalize, split_sentences, word_distance  # noqa: E402

DB = ROOT / "data" / "myflow.sqlite"
DEFAULT_REFERENCE = ROOT / "data" / "bench" / "private" / "klinok-ch1.txt"


def load_session(hours: float) -> list[dict]:
    """Записи истории за последние N часов (created_at в базе — UTC)."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    rows = [dict(r) for r in con.execute(
        "select id, created_at, asr_text, polished_text, edited_text, status, "
        "duration_ms, e2e_latency_ms from history "
        "where created_at >= ? and coalesce(asr_text,'') <> '' order by id", (since,))]
    con.close()
    return rows


def looks_like_chapter(text: str, vocabulary: set[str], share: float = 0.7) -> bool:
    """Похоже ли, что фразу читали из главы, а не диктовали в чат.

    В истории лежат ВСЕ диктовки: и начитка главы, и сообщения в чат про код
    (оттуда в замер лезли слова вроде «myflow», «llm», «asr» и получался WER 4.0).
    Фраза из главы почти целиком состоит из её же слов — по этой доле и фильтруем.
    """
    words = normalize(text).split()
    if not words:
        return False
    inside = sum(1 for word in words if word in vocabulary)
    return inside / len(words) >= share


def best_hypothesis(row: dict) -> str:
    """Что считаем услышанным: правка человека, иначе полировка, иначе сырое ASR."""
    for key in ("edited_text", "polished_text", "asr_text"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--show", type=int, default=20, help="сколько ошибок показать")
    args = ap.parse_args()

    reference = args.reference.read_text(encoding="utf-8")
    sessions = load_session(args.hours)
    if not sessions:
        print(f"  за последние {args.hours} ч записей нет — начитки не видно")
        return 1

    ref_words = normalize(reference).split()
    vocabulary = set(ref_words)
    chapter_rows = [row for row in sessions
                    if looks_like_chapter(best_hypothesis(row), vocabulary)]
    chat_rows = len(sessions) - len(chapter_rows)

    heard = " ".join(best_hypothesis(row) for row in chapter_rows)
    hyp_words = normalize(heard).split()
    distance, operations = word_distance(ref_words, hyp_words)

    # Читали только часть главы: считаем метрику по ПРОЧИТАННОМУ куску, а не по всей
    # главе (иначе нечитанный хвост раздувает WER).
    read_words = sum(1 for kind, _e, _h in operations if kind in ("замена",))
    read_words += len(ref_words) - sum(1 for kind, _e, _h in operations
                                       if kind == "пропуск")
    trailing = 0
    for kind, _expected, _heard_word in reversed(operations):
        if kind == "пропуск":
            trailing += 1
        else:
            break
    errors = distance - trailing
    wer = errors / max(read_words, 1)

    kinds = {"замена": 0, "пропуск": 0, "вставка": 0}
    for kind, _expected, _heard in operations:
        kinds[kind] = kinds.get(kind, 0) + 1
    kinds["пропуск"] = max(0, kinds["пропуск"] - trailing)

    durations = [row["duration_ms"] for row in sessions if row.get("duration_ms")]
    latencies = [row["e2e_latency_ms"] for row in sessions if row.get("e2e_latency_ms")]

    print(f"  всего записей за {args.hours:g} ч: {len(sessions)} "
          f"(из них про главу {len(chapter_rows)}, про чат {chat_rows})")
    if not chapter_rows:
        print("  начитки главы не нашёл — только диктовки в чат")
        return 1
    first = str(chapter_rows[0]["created_at"])[:19]
    last = str(chapter_rows[-1]["created_at"])[:19]
    print(f"  начитка: {len(chapter_rows)} фраз (id {chapter_rows[0]['id']}"
          f"..{chapter_rows[-1]['id']})")
    print(f"  время (UTC): {first} .. {last}")
    print(f"  прочитано слов эталона: {read_words} из {len(ref_words)} "
          f"({read_words / max(len(ref_words), 1) * 100:.0f}% главы)")
    print(f"  услышано слов: {len(hyp_words)}")
    print(f"\n  WER на прочитанном куске: {wer:.4f}  ({errors} ошибок)")
    print(f"  разбор ошибок: замена {kinds['замена']}, пропуск {kinds['пропуск']}, "
          f"вставка {kinds['вставка']}")
    durations = [row["duration_ms"] for row in chapter_rows if row.get("duration_ms")]
    latencies = [row["e2e_latency_ms"] for row in chapter_rows if row.get("e2e_latency_ms")]
    if durations:
        print(f"  длительность фраз: средняя {sum(durations) / len(durations):.0f} мс")
    if latencies:
        lat = sorted(latencies)
        print(f"  задержка e2e: медиана {lat[len(lat) // 2]:.0f} мс, "
              f"худшая {lat[-1]:.0f} мс")

    print(f"\n  ЧАСТЫЕ ОШИБКИ (эталон -> услышано):")
    counts: dict[tuple[str, str], int] = {}
    for kind, expected, heard_word in operations:
        if kind == "замена":
            key = (expected, heard_word)
            counts[key] = counts.get(key, 0) + 1
    for (expected, heard_word), count in sorted(counts.items(),
                                                key=lambda kv: -kv[1])[:args.show]:
        print(f"    x{count:<3} {expected!r} -> {heard_word!r}")
    if not counts:
        print("    замен нет — распознавание слово в слово")

    print(f"\n  ЧТО ЧИТАТЬ ДАЛЬШЕ: загляни в конец — последняя фраза "
          f"{sessions[-1]['id']} ({last} UTC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
