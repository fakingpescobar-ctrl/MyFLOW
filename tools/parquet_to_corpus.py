"""Parquet с Hugging Face -> чистый корпус фраз для синтетики постпроцессора.

Часть датасетов не отдаёт строки через API (`/rows`) — у них нет вьюера, но есть
parquet. Тогда качаем parquet и читаем его pyarrow (он есть в обучающем venv).

    .venv-train\\Scripts\\python.exe tools\\parquet_to_corpus.py --parquet <файл> --rows 4000

Фильтры те же, что в fetch_corpus.py: одна фраза, похожая на диктовку, без ссылок,
тегов, цифрового мусора и дублей.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.fetch_corpus import clean, CYRILLIC  # noqa: E402


def russian_from(value) -> str:
    """Достать русский текст из значения любой формы (строка, dict, JSON-строка)."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                return russian_from(json.loads(text))
            except Exception:  # noqa: BLE001
                return text if CYRILLIC.search(text) else ""
        return text
    if isinstance(value, dict):
        for key in ("ru", "rus", "russian", "target", "translation"):
            if key in value:
                found = russian_from(value[key])
                if found and CYRILLIC.search(found):
                    return found
        for inner in value.values():
            found = russian_from(inner)
            if found and CYRILLIC.search(found):
                return found
    if isinstance(value, (list, tuple)):
        for inner in value:
            found = russian_from(inner)
            if found and CYRILLIC.search(found):
                return found
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=140)
    ap.add_argument("--out", default=str(ROOT / "data" / "train" / "corpus" /
                                         "conversational.txt"))
    args = ap.parse_args()

    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet)
    columns = table.column_names
    print(f"  колонок: {len(columns)} -> {columns[:6]}")

    kept: list[str] = []
    seen: set[str] = set()
    scanned = 0
    for batch in table.to_batches(max_chunksize=2000):
        data = batch.to_pydict()
        width = len(next(iter(data.values()))) if data else 0
        for index in range(width):
            scanned += 1
            text = ""
            for column in columns:
                candidate = russian_from(data[column][index])
                if candidate and CYRILLIC.search(candidate):
                    text = candidate
                    break
            line = clean(text, args.min_len, args.max_len) if text else None
            if not line:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(line)
            if len(kept) >= args.rows:
                break
        if len(kept) >= args.rows:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print(f"  просмотрено строк: {scanned}")
    print(f"  корпус: {len(kept)} фраз -> {out}")
    if kept:
        average = sum(len(x) for x in kept) / len(kept)
        print(f"  средняя длина: {average:.0f} символов")
        for sample in (kept[0], kept[len(kept) // 3], kept[-1]):
            print(f"    {sample[:78]!r}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
