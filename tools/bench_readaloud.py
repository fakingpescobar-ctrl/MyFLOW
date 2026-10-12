"""Стенд «чтение вслух»: WER по ЗВУКУ на известном тексте.

Чем это лучше прежних замеров. Раньше мы мерили по уже распознанным фразам (текстовый
уровень) и на «хорошо ли отполировано», а не «правильно ли услышано» ✗. Здесь человек
читает ИЗВЕСТНЫЙ текст вслух: эталон у нас на руках, и мы видим полную картину —
ошибки распознавания, ошибки полировки и задержку, с аудио каждой фразы на диске
(колонка `audio_path` в таблице history).

Как пользоваться:
  1. положить главу в data/bench/private/<имя>.txt  (в git НЕ попадает — авторское право);
  2. читать предложение за предложением в MyFLOW;
  3. запустить этот стенд — он возьмёт последние N записей истории, сопоставит их с
     предложениями эталона по порядку и посчитает метрики.

Запуск:
    python tools/bench_readaloud.py --reference data/bench/private/perumov-ch1.txt
    python tools/bench_readaloud.py --reference ... --records 40      # последние 40 фраз
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "myflow.sqlite"
REPORTS = ROOT / "data" / "bench" / "reports"

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Для WER: регистр, ё, пунктуация и лишние пробелы не считаются."""
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = PUNCT.sub(" ", text)
    return " ".join(text.split())


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    """Разбить главу на предложения, годные для чтения вслух."""
    text = text.replace("\r\n", "\n")
    # абзацы и переносы — тоже границы: длинные куски делим по концу предложения
    chunks: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        chunks.extend(part.strip() for part in SENTENCE_END.split(paragraph) if part.strip())
    sentences = []
    for chunk in chunks:
        if len(normalize(chunk).split()) >= min_words:
            sentences.append(chunk)
    return sentences


def word_distance(reference: list[str], hypothesis: list[str]) -> tuple[int, list[tuple]]:
    """Правки Левенштейна по словам + список операций (для отчёта)."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            table[i][j] = min(table[i - 1][j] + 1,       # пропуск
                              table[i][j - 1] + 1,       # вставка
                              table[i - 1][j - 1] + cost)  # замена
    # обратный проход — собираем операции
    operations: list[tuple] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + (
                0 if reference[i - 1] == hypothesis[j - 1] else 1):
            if reference[i - 1] != hypothesis[j - 1]:
                operations.append(("замена", reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            operations.append(("пропуск", reference[i - 1], ""))
            i -= 1
        else:
            operations.append(("вставка", "", hypothesis[j - 1]))
            j -= 1
    operations.reverse()
    return table[-1][-1], operations


def error_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0, 0, 0
    distance, _ = word_distance(ref, hyp)
    return distance / len(ref), distance, len(ref)


def load_records(limit: int, since_id: int | None = None) -> list[dict]:
    """Последние записи истории с текстами и путями к аудио."""
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    query = ("select id, created_at, asr_text, polished_text, edited_text, audio_path, "
             "duration_ms, e2e_latency_ms, transcript_origin, style, status "
             "from history where asr_text is not null and trim(asr_text) <> '' ")
    params: list = []
    if since_id is not None:
        query += "and id > ? "
        params.append(since_id)
    query += "order by id desc limit ?"
    params.append(limit)
    rows = [dict(row) for row in con.execute(query, params)]
    con.close()
    return list(reversed(rows))          # в порядке чтения


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="файл с эталонным текстом")
    ap.add_argument("--records", type=int, default=0, help="сколько записей взять (0 = по числу предложений)")
    ap.add_argument("--since-id", type=int, default=None, help="брать записи после этого id")
    ap.add_argument("--report", default=None, help="куда записать отчёт")
    args = ap.parse_args()

    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"  нет файла эталона: {reference_path}", file=sys.stderr)
        return 1
    sentences = split_sentences(reference_path.read_text(encoding="utf-8"))
    print(f"  эталон: {reference_path.name} — {len(sentences)} предложений")

    wanted = args.records or len(sentences)
    records = load_records(wanted, args.since_id)
    print(f"  записей истории: {len(records)}")
    if not records:
        print("  нет записей — сначала продиктуй главу", file=sys.stderr)
        return 1

    # Сопоставляем по порядку: последняя запись = последнее предложение.
    pairs = list(zip(sentences[-len(records):], records))
    if len(records) != len(sentences):
        print(f"  ВНИМАНИЕ: записей {len(records)}, предложений {len(sentences)} — "
              f"сопоставляю последние {len(pairs)}")

    rows = []
    for sentence, record in pairs:
        raw = record.get("asr_text") or ""
        polished = record.get("polished_text") or raw
        raw_rate, raw_err, words = error_rate(sentence, raw)
        pol_rate, pol_err, _ = error_rate(sentence, polished)
        rows.append({
            "id": record["id"],
            "reference": sentence,
            "asr": raw,
            "polished": polished,
            "asr_wer": raw_rate,
            "polished_wer": pol_rate,
            "asr_errors": raw_err,
            "polished_errors": pol_err,
            "words": words,
            "latency_ms": record.get("e2e_latency_ms"),
            "duration_ms": record.get("duration_ms"),
            "audio": record.get("audio_path"),
            "origin": record.get("transcript_origin"),
            "style": record.get("style"),
            "overedit": max(0, pol_err - raw_err),   # полировка испортила то, что было верно
        })

    def median(values):
        clean = [v for v in values if isinstance(v, (int, float))]
        return statistics.median(clean) if clean else 0

    total_words = sum(r["words"] for r in rows)
    total_asr = sum(r["asr_errors"] for r in rows)
    total_pol = sum(r["polished_errors"] for r in rows)
    overedit = sum(r["overedit"] for r in rows)
    latencies = [r["latency_ms"] for r in rows]

    print(f"\n  ФРАЗ: {len(rows)} | СЛОВ В ЭТАЛОНЕ: {total_words}")
    print(f"  WER сырого ASR:      {total_asr / max(total_words, 1):.4f}  ({total_asr} ошибок)")
    print(f"  WER после полировки: {total_pol / max(total_words, 1):.4f}  ({total_pol} ошибок)")
    print(f"  испорчено полировкой: {overedit} слов")
    print(f"  задержка e2e: медиана {median(latencies):.0f} мс | "
          f"худшая {max((v for v in latencies if v), default=0):.0f} мс")

    worst = sorted(rows, key=lambda r: r["asr_wer"], reverse=True)[:5]
    print("\n  ХУДШИЕ ПО РАСПОЗНАВАНИЮ:")
    for row in worst:
        print(f"    [{row['id']}] WER {row['asr_wer']:.2f} -> после полировки {row['polished_wer']:.2f}")
        print(f"        эталон: {row['reference'][:78]}")
        print(f"        услышал: {row['asr'][:78]}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = Path(args.report) if args.report else REPORTS / f"readaloud-{stamp}.json"
    out.write_text(json.dumps({
        "reference": reference_path.name,
        "sentences": len(sentences),
        "phrases": len(rows),
        "total_words": total_words,
        "wer_asr": total_asr / max(total_words, 1),
        "wer_polished": total_pol / max(total_words, 1),
        "overedited_words": overedit,
        "latency_median_ms": median(latencies),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  отчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
