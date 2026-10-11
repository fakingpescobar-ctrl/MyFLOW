"""Сборка датасета для дообучения компактного постпроцессора MyFLOW.

Задача модели — узкая: «сырой вывод ASR → аккуратный текст». Именно такую роль
в superwhisper играет их s1-mini (462 МБ, Qwen3, дообучен под punctuation /
truecasing / text-normalization; см. их карточку на HuggingFace).

Данные собираются из двух источников:

1. **Реальные пары** из истории диктовок (`data/myflow.sqlite`):
   `asr_text` → итоговый текст (правка пользователя, если была, иначе полировка).
   Среди них есть и «текст не менялся» — это тоже ценно: учим модель НЕ трогать.
2. **Синтетика**: берём ЧИСТЫЕ тексты (итоги диктовок + эталонные фразы Wispr) и
   портим их нашим же «моделью ошибок» ASR:
     * нижний регистр, снятая пунктуация, ё→е;
     * цифры словами, лишние филлеры («ну,», «э-э,», «значит,», «типа,»);
     * потерянные запятые, обрывок последнего слова, путаница терминов из словаря.
   Одна чистая фраза → до 8 вариантов порчи (детерминированно по seed).

Выход: `data/train/postproc_{train,val}.jsonl` в формате чата (system/user/assistant),
готовом для Qwen3-шаблона. Данные остаются в `data/` (в git не попадают).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "train"
WISPR = ROOT / "data" / "eval_wispr" / "phrases.jsonl"

SYSTEM_PROMPT = (
    "Ты — постпроцессор русской диктовки. Тебе дают сырой текст распознавания "
    "речи. Верни тот же текст, но аккуратно оформленный: расставь знаки "
    "препинания и заглавные буквы, запиши числа цифрами, убери слова-филлеры "
    "(ну, э-э, эм, значит, типа, короче, как бы). НИЧЕГО не добавляй от себя, "
    "не пересказывай, не меняй смысл и порядок слов, не переводи. Если текст уже "
    "оформлен — верни его без изменений. Отвечай только текстом."
)

# Филлеры, которые наша детерминированная чистка и так умеет убирать.
FILLERS = ["ну,", "э-э,", "эм,", "значит,", "типа,", "короче,", "как бы,",
           "в общем,", "собственно,", "вот,"]
# Числа словами — самый частый вид «нормализации».
NUM_WORDS = {"ноль": "0", "один": "1", "одна": "1", "два": "2", "две": "2",
             "три": "3", "четыре": "4", "пять": "5", "шесть": "6", "семь": "7",
             "восемь": "8", "девять": "9", "десять": "10", "двадцать": "20",
             "тридцать": "30", "сорок": "40", "пятьдесят": "50", "сто": "100"}
_WORD = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")


def corrupt(text: str, rng: random.Random, dictionary: list[dict] | None = None) -> str:
    """Испортить чистый текст так, как это делает ASR (наш «шумовой профиль»)."""
    out = text.strip()
    if not out:
        return out
    # 1) цифры словами обратно (или наоборот) — ASR любит слова
    for word, digit in NUM_WORDS.items():
        out = re.sub(rf"\b{digit}\b", word, out, count=1)
    # 2) снять часть пунктуации (ASR-гипотеза часто без запятых)
    out = re.sub(r"(?<=[а-яё]),(?=\s)", "", out)
    # 3) нижний регистр (whisper особенно)
    if rng.random() < 0.7:
        out = out.lower()
    # 4) ё → е (частая мелочь распознавания)
    if rng.random() < 0.3:
        out = out.replace("ё", "е")
    # 5) вставить филлеры в 1-3 местах
    words = out.split()
    for _ in range(rng.randint(0, 3)):
        if len(words) < 3:
            break
        pos = rng.randrange(1, max(2, len(words) - 1))
        words.insert(pos, rng.choice(FILLERS))
    out = " ".join(words)
    # 6) термины из словаря — типичная путаница ASR
    if dictionary:
        pair = rng.choice(dictionary)
        wrong, right = str(pair.get("wrong", "")), str(pair.get("right", ""))
        if wrong and right and rng.random() < 0.5:
            out = out.replace(right, wrong, 1)
    # 7) редко — обрывок в конце (у GigaAM бывает)
    if rng.random() < 0.08:
        out = out[:-1].rstrip(" ,.") + "."
    # 8) точка в конце убирается — модель должна поставить
    if rng.random() < 0.5:
        out = out.rstrip(".!?…")
    return out.strip()


def finalize_clean(text: str) -> str:
    """Привести «эталон» к виду, который мы хотим от модели."""
    t = (text or "").strip()
    if not t:
        return t
    t = re.sub(r"\s+", " ", t)
    if t[0].islower():
        t = t[0].upper() + t[1:]
    if t[-1] not in ".!?…":
        t += "."
    return t


def real_pairs(limit: int | None = None) -> list[dict]:
    """Пары из истории диктовок: сырой ASR -> итоговый текст."""
    db = ROOT / "data" / "myflow.sqlite"
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = list(con.execute(
        "SELECT id, asr_text, polished_text, edited_text FROM history "
        "WHERE asr_text IS NOT NULL AND TRIM(asr_text) != '' ORDER BY id"))
    con.close()
    pairs = []
    for r in rows:
        final = (r["edited_text"] or "").strip() or (r["polished_text"] or "").strip()
        asr = (r["asr_text"] or "").strip()
        if not final or not asr:
            continue
        pairs.append({"src": asr, "dst": finalize_clean(final), "kind": "real",
                      "id": int(r["id"])})
    if limit:
        pairs = pairs[:limit]
    return pairs


def clean_texts() -> list[str]:
    """Чистые тексты для синтетики: итоги диктовок + эталоны Wispr."""
    texts = [p["dst"] for p in real_pairs()]
    if WISPR.exists():
        for line in WISPR.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ref = (row.get("their_edited") or row.get("their_final") or "").strip()
            if ref:
                texts.append(finalize_clean(ref))
    # уникализируем и отбрасываем слишком короткие/длинные
    seen, out = set(), []
    for t in texts:
        key = t.lower()
        if key in seen or len(_WORD.findall(t)) < 2:
            continue
        if len(t) > 400:
            continue
        seen.add(key)
        out.append(t)
    return out


def load_dictionary() -> list[dict]:
    """Словарь правок пользователя (для реалистичной путаницы терминов)."""
    db = ROOT / "data" / "myflow.sqlite"
    if not db.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = list(con.execute("SELECT src_text, dst_text FROM dictionary"))
        con.close()
        return [{"wrong": r["src_text"], "right": r["dst_text"]} for r in rows]
    except Exception:  # noqa: BLE001
        return []


def to_chat(src: str, dst: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": src},
        {"role": "assistant", "content": dst},
    ]}


def build(val_ratio: float = 0.15, variants: int = 8, seed: int = 20261011,
          limit: int | None = None) -> dict:
    rng = random.Random(seed)
    real = real_pairs(limit=limit)
    dictionary = load_dictionary()
    texts = clean_texts()

    # 1) синтетика из чистых текстов
    synthetic = []
    for text in texts:
        dst = text
        for _ in range(max(1, variants)):
            src = corrupt(text, rng, dictionary)
            if src.strip().lower() == dst.strip().lower():
                continue                     # порча ничего не изменила — пример бесполезен
            synthetic.append(to_chat(src, dst))
        # один «идентичный» пример: учим не трогать уже оформленный текст
        synthetic.append(to_chat(dst, dst))

    # 2) реальные пары
    real_examples = [to_chat(p["src"], p["dst"]) for p in real]

    # 3) разбиение: валидация — только РЕАЛЬНЫЕ пары (честная проверка на своих)
    rng.shuffle(real_examples)
    n_val = max(20, int(len(real_examples) * val_ratio)) if real_examples else 0
    val = real_examples[:n_val]
    train_real = real_examples[n_val:]

    rng.shuffle(synthetic)
    train = train_real + synthetic
    rng.shuffle(train)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "postproc_train.jsonl"
    val_path = OUT_DIR / "postproc_val.jsonl"
    train_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in train),
                          encoding="utf-8")
    val_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in val),
                        encoding="utf-8")
    return {
        "clean_texts": len(texts),
        "synthetic": len(synthetic),
        "real_total": len(real_examples),
        "real_train": len(train_real),
        "real_val": len(val),
        "train_total": len(train),
        "val_total": len(val),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Датасет для постпроцессора MyFLOW")
    ap.add_argument("--variants", type=int, default=8, help="порч на один чистый текст")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--limit", type=int, default=None, help="ограничить реальные пары")
    args = ap.parse_args()
    st = build(val_ratio=args.val_ratio, variants=args.variants,
               seed=args.seed, limit=args.limit)
    print("Датасет собран:")
    for key in ("clean_texts", "synthetic", "real_total", "real_train", "real_val",
                "train_total", "val_total"):
        print(f"  {key:<12} {st[key]}")
    print(f"  train: {st['train_path']}")
    print(f"  val:   {st['val_path']}")
    print("\nПример (первая синтетическая пара, усечённо):")
    first = json.loads((OUT_DIR / "postproc_train.jsonl").read_text(encoding="utf-8")
                       .splitlines()[0])
    for msg in first["messages"]:
        if msg["role"] == "system":
            continue
        print(f"  {msg['role']:<9}: {msg['content'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
