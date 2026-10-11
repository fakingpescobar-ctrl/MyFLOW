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


def _norm(text: str) -> str:
    """Нормализованный вид текста (для дедупликации и проверки утечки)."""
    return " ".join(w.replace("ё", "е") for w in _WORD.findall((text or "").lower()))


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


def clean_texts(exclude: set[str] | None = None) -> list[str]:
    """Чистые тексты для синтетики: итоги диктовок + эталоны Wispr.

    `exclude` — нормализованные тексты, которые НЕЛЬЗЯ использовать как источник
    (цели и входы валидационных пар): иначе модель увидит правильный ответ
    валидации в обучении и оценка превратится в проверку памяти.
    """
    exclude = exclude or set()
    texts = [p["dst"] for p in real_pairs()]
    if WISPR.exists():
        for line in WISPR.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ref = (row.get("their_edited") or row.get("their_final") or "").strip()
            if ref:
                texts.append(finalize_clean(ref))
    texts = [t for t in texts if _norm(t) not in exclude]
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


def parts_key(row: dict) -> tuple[str, str]:
    """Нормализованный ключ пары (вход, ответ) — для дедупликации."""
    src = next(m["content"] for m in row["messages"] if m["role"] == "user")
    dst = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    return _norm(src), _norm(dst)


def to_chat(src: str, dst: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": src},
        {"role": "assistant", "content": dst},
    ]}


def build(variants: int = 4, seed: int = 20261011, val_every: int = 7,
          identity_cap: int = 1200, limit: int | None = None,
          out_dir: Path | None = None) -> dict:
    """Собрать датасет.

    Разбиение — СТАБИЛЬНОЕ по id диктовки (`id % val_every == 0` идёт в валидацию):
    при добавлении новых диктовок старая разбивка не «переезжает», иначе одна и та
    же фраза могла оказаться и в обучении, и в проверке.
    """
    rng = random.Random(seed)
    all_pairs = real_pairs(limit=limit)
    dictionary = load_dictionary()

    val_ids = {p["id"] for p in all_pairs if p["id"] % val_every == 0}
    train_ids = {p["id"] for p in all_pairs} - val_ids
    val_rows = [p for p in all_pairs if p["id"] in val_ids]

    # Тексты валидации блокируем ПО СОДЕРЖИМОМУ, а не только по id: одну и ту же
    # фразу пользователь диктовал несколько раз, и её копия с другим id иначе
    # утечёт в обучение (первая версия так и делала: 4 утечки из 56).
    blocked = {_norm(p["src"]) for p in val_rows} | {_norm(p["dst"]) for p in val_rows}

    val = [to_chat(p["src"], p["dst"]) for p in val_rows]
    train_real = []
    for p in all_pairs:
        if p["id"] not in train_ids:
            continue
        if _norm(p["dst"]) in blocked or _norm(p["src"]) in blocked:
            continue                                  # дубль валидационной фразы
        train_real.append(to_chat(p["src"], p["dst"]))

    texts = clean_texts(exclude=blocked)

    # Реальные пары тоже дедуплицируем: одну фразу пользователь диктовал дважды,
    # и пара (вход, ответ) повторялась (поймано тестом: 182 дубля).
    seen_pairs: set[tuple[str, str]] = set()
    unique_real: list[dict] = []
    for row in train_real:
        key = parts_key(row)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_real.append(row)
    train_real = unique_real

    # 1) синтетика: не больше `variants` РАЗНЫХ порч на текст, без дублей пар
    synthetic: list[dict] = []
    identity: list[dict] = []
    for text in texts:
        dst = text
        made = 0
        for _ in range(max(1, variants) * 3):          # попыток больше, чем нужно
            if made >= variants:
                break
            src = corrupt(text, rng, dictionary)
            key = (_norm(src), _norm(dst))
            if not key[0] or key[0] == key[1] or key in seen_pairs:
                continue                                # пусто/не изменилось/дубль
            seen_pairs.add(key)
            synthetic.append(to_chat(src, dst))
            made += 1
        identity_key = (_norm(dst), _norm(dst))
        if identity_key not in seen_pairs:
            seen_pairs.add(identity_key)
            identity.append(to_chat(dst, dst))

    rng.shuffle(identity)
    synthetic += identity[:identity_cap]                # учим «не трогать» готовое

    rng.shuffle(synthetic)
    train = [row for row in train_real + synthetic
             if len(next(m["content"] for m in row["messages"]
                         if m["role"] == "assistant").strip()) >= 3]
    rng.shuffle(train)
    # реальные пары валидации тоже проверим на дубли с обучением
    train_keys = {parts_key(row) for row in train}
    val = [row for row in val if parts_key(row) not in train_keys]

    target_dir = Path(out_dir) if out_dir else OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    train_path = target_dir / "postproc_train.jsonl"
    val_path = target_dir / "postproc_val.jsonl"
    train_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in train),
                          encoding="utf-8")
    val_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in val),
                        encoding="utf-8")
    return {
        "clean_texts": len(texts),
        "synthetic": len(synthetic),
        "real_total": len(all_pairs),
        "real_train": len(train_real),
        "real_val": len(val),
        "val_ids_in_train": len(val_ids & train_ids),
        "train_total": len(train),
        "val_total": len(val),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Датасет для постпроцессора MyFLOW")
    ap.add_argument("--variants", type=int, default=4, help="разных порч на чистый текст")
    ap.add_argument("--val-every", type=int, default=7,
                    help="каждая N-я диктовка идёт в валидацию (стабильно по id)")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--limit", type=int, default=None, help="ограничить реальные пары")
    ap.add_argument("--out", type=Path, default=None, help="каталог для датасета")
    args = ap.parse_args()
    st = build(variants=args.variants, seed=args.seed,
               val_every=args.val_every, limit=args.limit, out_dir=args.out)
    print("Датасет собран:")
    for key in ("clean_texts", "synthetic", "real_total", "real_train", "real_val",
                "train_total", "val_total"):
        print(f"  {key:<12} {st[key]}")
    print(f"  train: {st['train_path']}")
    print(f"  val:   {st['val_path']}")
    print("\nПример (первая синтетическая пара, усечённо):")
    first = json.loads(Path(st["train_path"]).read_text(encoding="utf-8").splitlines()[0])
    for msg in first["messages"]:
        if msg["role"] == "system":
            continue
        print(f"  {msg['role']:<9}: {msg['content'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
