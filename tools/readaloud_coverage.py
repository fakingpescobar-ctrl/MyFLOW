"""Честный разбор начитки с учётом перечитываний.

Проблема прежнего расчёта: текст начитки склеивался и сравнивался с главой целиком.
Если человек перечитывает предложения (а он перечитывал, пока не прочитает ровно),
повторы попадают в «вставки», а непрочитанные места — в «пропуски», и WER получается
бессмысленным (1.62 при 270 реальных заменах).

Как считаем теперь: для КАЖДОГО предложения главы ищем лучшее совпадение среди фраз
начитки (по расстоянию Левенштейна, кандидаты отбираются по пересечению слов, чтобы
не гонять 60 тысяч выравниваний). Тогда:
  * повторные прочтения не мешают: берётся лучшее;
  * непрочитанные предложения видны отдельно (покрытие главы);
  * WER считается только по прочитанным предложениям.
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
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    rows = [dict(r) for r in con.execute(
        "select id, created_at, asr_text, polished_text, edited_text from history "
        "where created_at >= ? and coalesce(asr_text,'') <> '' order by id", (since,))]
    con.close()
    return rows


def heard_text(row: dict) -> str:
    for key in ("edited_text", "polished_text", "asr_text"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    reference = args.reference.read_text(encoding="utf-8")
    sentences = [normalize(s).split() for s in split_sentences(reference)]
    sentences = [s for s in sentences if s]
    vocabulary = {word for sentence in sentences for word in sentence}

    rows = load_session(args.hours)
    records = []
    for row in rows:
        text = heard_text(row)
        words = normalize(text).split()
        if not words:
            continue
        share = sum(1 for word in words if word in vocabulary) / len(words)
        if share >= 0.7:
            records.append((row, set(words), words))

    if not records:
        print("  начитки главы не нашёл")
        return 1

    total_words = sum(len(s) for s in sentences)
    covered_words = 0
    errors = 0
    unread: list[int] = []
    substitutions: dict[tuple[str, str], int] = {}
    per_sentence: list[tuple[float, int, int]] = []

    for number, reference_words in enumerate(sentences):
        target = set(reference_words)
        candidates = []
        for row, word_set, words in records:
            overlap = len(target & word_set)
            if overlap:
                candidates.append((overlap, row, words))
        if not candidates:
            unread.append(number)
            continue
        candidates.sort(key=lambda item: -item[0])
        best = None
        for _overlap, row, words in candidates[:3]:
            distance, operations = word_distance(reference_words, words)
            if best is None or distance < best[0]:
                best = (distance, row, operations)
        if best is None:
            unread.append(number)
            continue

        distance, row, operations = best
        # предложение считаем прочитанным, если совпала хотя бы половина слов
        matched = sum(1 for kind, _e, _h in operations if kind == "замена")
        if matched + (len(reference_words) - sum(1 for k, _e, _h in operations
                                                 if k == "пропуск")) < len(reference_words) / 2:
            unread.append(number)
            continue

        covered_words += len(reference_words)
        errors += distance
        per_sentence.append((distance / max(len(reference_words), 1), number, row["id"]))
        for kind, expected, heard_word in operations:
            if kind == "замена":
                key = (expected, heard_word)
                substitutions[key] = substitutions.get(key, 0) + 1

    wer = errors / max(covered_words, 1)
    print(f"  фраз начитки (про главу): {len(records)}, предложений в главе: {len(sentences)}")
    print(f"  прочитано предложений: {len(sentences) - len(unread)} из {len(sentences)} "
          f"({(len(sentences) - len(unread)) / len(sentences) * 100:.0f}%)")
    print(f"  покрытие по словам: {covered_words} из {total_words} "
          f"({covered_words / total_words * 100:.0f}%)")
    print(f"\n  WER на прочитанных предложениях: {wer:.4f}  ({errors} ошибок)")
    if substitutions:
        print(f"\n  ЧАСТЫЕ ЗАМЕНЫ (эталон -> услышано):")
        for (expected, heard_word), count in sorted(substitutions.items(),
                                                    key=lambda kv: -kv[1])[:args.show]:
            print(f"    x{count:<3} {expected!r} -> {heard_word!r}")
    worst = sorted(per_sentence, reverse=True)[:5]
    if worst:
        print(f"\n  ХУДШИЕ ПРЕДЛОЖЕНИЯ (WER, номер, id записи):")
        for value, number, record_id in worst:
            print(f"    {value:.2f}  предложение #{number + 1}, запись {record_id}")
    if unread:
        print(f"\n  НЕ ПРОЧИТАНО предложений: {len(unread)} (первые номера: "
              f"{', '.join(str(n + 1) for n in unread[:8])}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
