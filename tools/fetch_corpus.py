"""Сбор чистого русского текста с Hugging Face для синтетики постпроцессора.

Зачем: ядро обучения — реальные диктовки пользователя (пары «что услышал ASR ->
что поправил человек»), но чистых текстов для порчи мало (свои же фразы). Чужой
корпус даёт разнообразие формулировок, оставаясь в НАШЕЙ модели ошибок: мы не берём
чужие «исправления», а только чистый текст, который сами портим по нашему профилю.

Берём РАЗГОВОРНЫЙ корпус: субтитры (Helsinki-NLP/opus-100, конфиг ru-en) — для
диктовки живая речь ближе, чем википедийная. Формальные корпуса вредны: тянут стиль
к «нормализации» и учат переписывать, а это наша главная болезнь.

Загрузка — через API датасетов (rows), потому что hf_hub в этом окружении подвисает.

    python tools/fetch_corpus.py --rows 3000
    python tools/fetch_corpus.py --dataset Helsinki-NLP/opus-100 --config ru-en
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "train" / "corpus"

API = "https://datasets-server.huggingface.co/rows"
CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
NOISE = re.compile(r"(https?://|www\.|@\w+|#\w+|\*\*|__|\{\{|\}\}|<[^>]{1,20}>)")
DEFAULT_DATASET = "Helsinki-NLP/opus-100"
DEFAULT_CONFIG = "ru-en"


def fetch_page(dataset: str, config: str, split: str, offset: int, length: int,
               timeout: float = 30.0) -> list[dict]:
    query = urllib.parse.urlencode({
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": length,
    })
    req = urllib.request.Request(f"{API}?{query}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("rows") or []


def pick_text(row: dict) -> str:
    """Достать русский текст из строки любого вида (структура у датасетов разная)."""
    payload = row.get("row") if isinstance(row, dict) else None
    if not isinstance(payload, dict):
        return ""
    for value in payload.values():
        if isinstance(value, str) and CYRILLIC.search(value):
            return value
        if isinstance(value, dict):
            for inner in value.values():
                if isinstance(inner, str) and CYRILLIC.search(inner):
                    return inner
    return ""


def clean(text: str, min_len: int, max_len: int) -> str | None:
    """Привести к одной фразе, годной для диктовки, или отбросить."""
    line = " ".join(text.split())
    if NOISE.search(line):
        return None
    # Обрывки субтитров: «Но сейчас ...я вверяю», склейки и висячая пунктуация.
    if "..." in line or "…" in line or " -" in line[:3]:
        return None
    if line[0] in ",;:.!?—-" or line[-1] in ",;:—-":
        return None
    if line.count(",") > 4 or "  " in line:
        return None
    if not CYRILLIC.search(line):
        return None
    if not (min_len <= len(line) <= max_len):
        return None
    letters = sum(ch.isalpha() for ch in line)
    if letters < 0.6 * len(line):                 # мусор из символов/цифр
        return None
    if len(set(line.lower())) < 8:                # «ааааа», «да да да да»
        return None
    return line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=int, default=3000, help="сколько фраз собрать")
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=140)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / "conversational.txt"

    seen: set[str] = set()
    kept: list[str] = []
    offset = 0
    page = 100
    started = time.time()

    while len(kept) < args.rows:
        try:
            rows = fetch_page(args.dataset, args.config, args.split, offset, page)
        except Exception as exc:  # noqa: BLE001
            print(f"  сбой на offset={offset}: {exc!r}", file=sys.stderr)
            break
        if not rows:
            break
        for row in rows:
            line = clean(pick_text(row), args.min_len, args.max_len)
            if not line:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(line)
        offset += page
        if offset % 1000 == 0:
            print(f"  просмотрено {offset}, взято {len(kept)}", flush=True)
        if offset > 200000:
            break

    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    seconds = time.time() - started
    print(f"\n  корпус: {len(kept)} фраз -> {out}")
    print(f"  просмотрено строк: {offset} за {seconds:.0f} с")
    if kept:
        print(f"  средняя длина: {sum(len(x) for x in kept) / len(kept):.0f} символов")
        print(f"  примеры: {kept[0][:70]!r}")
        print(f"           {kept[len(kept) // 2][:70]!r}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
