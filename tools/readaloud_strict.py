"""Строгий подсчёт покрытия: считать прочитанным только уверенное совпадение.

Прежний расчёт давал 94% покрытия, хотя человек прочитал около 40%: короткие фразы
(«И он.», «Да.») пересекаются по словам с десятками предложений книги, и матчер
принимал их за прочитанные. Здесь требуется, чтобы совпало не меньше 65% слов
предложения — тогда случайные пересечения отсеиваются.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_readaloud import normalize, split_sentences, word_distance  # noqa: E402

DB = ROOT / "data" / "myflow.sqlite"
REFERENCE = ROOT / "data" / "bench" / "private" / "klinok-ch1.txt"
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
STRICT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65

sentences = [normalize(s).split() for s in split_sentences(REFERENCE.read_text(encoding="utf-8"))]
sentences = [s for s in sentences if s]
vocabulary = {word for sentence in sentences for word in sentence}

since = (datetime.now(timezone.utc) - timedelta(hours=HOURS)).strftime("%Y-%m-%d %H:%M:%S")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute(
    "select id, created_at, asr_text, polished_text, edited_text from history "
    "where created_at >= ? and coalesce(asr_text,'') <> '' order by id", (since,))]
con.close()

records = []
for row in rows:
    for key in ("edited_text", "polished_text", "asr_text"):
        text = (row.get(key) or "").strip()
        if text:
            break
    else:
        continue
    words = normalize(text).split()
    if not words:
        continue
    if sum(1 for w in words if w in vocabulary) / len(words) >= 0.7:
        records.append((row, words, sum(1 for w in words if w in vocabulary)))

covered_sentences = 0
covered_words = 0
errors = 0
total_words = sum(len(s) for s in sentences)
substitutions: dict[tuple[str, str], int] = {}

for reference_words in sentences:
    target = set(reference_words)
    candidates = sorted(records, key=lambda item: -len(target & set(item[1])))[:5]
    best = None
    for _row, words, _score in candidates:
        distance, operations = word_distance(reference_words, words)
        if best is None or distance < best[0]:
            best = (distance, operations)
    if best is None:
        continue
    distance, operations = best
    # доля подтверждения: сколько слов предложения реально прозвучало
    matched = sum(1 for kind, _e, _h in operations if kind == "замена")
    skipped = sum(1 for kind, _e, _h in operations if kind == "пропуск")
    confirmed = (len(reference_words) - skipped) / max(len(reference_words), 1)
    if confirmed < STRICT:
        continue
    covered_sentences += 1
    covered_words += len(reference_words)
    errors += distance
    for kind, expected, heard_word in operations:
        if kind == "замена":
            substitutions[(expected, heard_word)] = substitutions.get((expected, heard_word), 0) + 1

print(f"  фраз в истории (про главу): {len(records)}")
print(f"  предложений в главе: {len(sentences)}, уверенно прочитано: {covered_sentences} "
      f"({covered_sentences / len(sentences) * 100:.0f}%)")
print(f"  покрытие по словам: {covered_words} из {total_words} "
      f"({covered_words / total_words * 100:.0f}% главы)")
if covered_words:
    print(f"  WER на уверенно прочитанном: {errors / covered_words:.4f} ({errors} ошибок)")
print("  частые замены:")
for (expected, heard_word), count in sorted(substitutions.items(),
                                            key=lambda kv: -kv[1])[:10]:
    print(f"    x{count:<3} {expected!r} -> {heard_word!r}")
