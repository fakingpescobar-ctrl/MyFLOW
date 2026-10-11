"""Сборка датасета для дообучения компактного постпроцессора MyFLOW.

Задача модели — узкая: «сырой вывод ASR → аккуратный текст» (как `s1-mini` у
superwhisper: 462 МБ, Qwen3, дообучен под punctuation / truecasing /
text-normalization).

Источники:

1. **Реальные пары** из истории диктовок: `asr_text` → итоговый текст. Среди них
   есть и «текст не менялся» — учим НЕ трогать.
2. **Синтетика** из чистых текстов (итоги диктовок + эталоны Wispr).

ВАЖНО про синтетику (урок первого обучения). Первая версия портила текст
тотально — весь в нижний регистр, снять всю пунктуацию, числа словами. Модель
этому и научилась: «чинить развалившееся». А реальные входы **уже оформлены**, и
правки в них тонкие: убрать вставленный филлер, поправить одно слово, вернуть
запятую. Обученная на тотальной порче 0.6B давала 0.262 при сыром ASR 0.267 —
то есть не делала ничего. Поэтому здесь порча **тонкая по умолчанию** (вставленные
филлеры, повторы, реальные путаницы слов, пунктуация, регистр), а тотальная
осталась малой долей (~8%), чтобы модель не разучилась и её.

Слова-ошибки берём из РЕАЛЬНЫХ правок пользователя (`word_error_pairs`), а не
выдумываем: иначе модель учится на неразрешимой задаче (случайное слово вместо
случайного слова восстановить нельзя).

Выход: `data/train/postproc_{train,val}.jsonl` (чат-формат Qwen3). Разбиение
стабильно по `id % 7`, валидация блокируется по содержимому — утечка должна быть 0.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "train"
WISPR = ROOT / "data" / "eval_wispr" / "phrases.jsonl"

# Промпт берём из боевого модуля: обучение и рантайм обязаны использовать ОДИН
# текст, иначе дообученная модель получит на входе не тот формализм.
sys.path.insert(0, str(ROOT))
from myflow.postproc_prompt import POSTPROC_SYSTEM as SYSTEM_PROMPT  # noqa: E402

# Филлеры, которые наша чистка реально убирает (и которые видно в правках истории).
FILLERS = ["ну", "э-э", "эм", "значит", "типа", "короче", "собственно",
           "вот", "так", "кстати"]
DISCOURSE = ["вроде бы", "как бы", "в общем", "наверное", "по-моему"]
NUM_WORDS = {"ноль": "0", "один": "1", "одна": "1", "два": "2", "две": "2",
             "три": "3", "четыре": "4", "пять": "5", "шесть": "6", "семь": "7",
             "восемь": "8", "девять": "9", "десять": "10", "двадцать": "20",
             "тридцать": "30", "сорок": "40", "пятьдесят": "50", "сто": "100"}

_WORD = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")


def _norm(text: str) -> str:
    """Нормализованный вид текста (для дедупликации и проверки утечки)."""
    return " ".join(w.replace("ё", "е") for w in _WORD.findall((text or "").lower()))


def _words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def corrupt(text: str, rng: random.Random, dictionary: list[dict] | None = None,
            error_pairs: list[tuple[str, str]] | None = None,
            stats: dict | None = None) -> str:
    """Испортить чистый текст так, как выглядит реальный вход постпроцессора.

    Профиль подобран под историю диктовок: в основном ТОНКИЕ правки.
    `stats` (если передан) считает, какие операции сработали — видно в отчёте.
    """
    out = (text or "").strip()
    if not out:
        return out
    applied: list[str] = []

    # 1) цифры словами — ASR любит слова (детерминированно, если цифры есть)
    for word, digit in NUM_WORDS.items():
        if re.search(rf"\b{digit}\b", out):
            out = re.sub(rf"\b{digit}\b", word, out, count=1)
            applied.append("число словами")

    # 2) ТОТАЛЬНАЯ порча — малая доля (8%): чтобы модель не разучилась чинить
    #    развалившееся (бывает с whisper и при плохом звуке)
    if rng.random() < 0.08:
        out = out.lower()
        out = re.sub(r"[.,!?;:…]", "", out)
        out = re.sub(r"\s+", " ", out).strip()
        applied.append("тотальная")
        if not out:
            return out
        if stats is not None:
            for op in applied:
                stats[op] = stats.get(op, 0) + 1
        return out

    words = out.split()

    # 3) вставить филлеры (75%) — самая частая реальная правка
    if rng.random() < 0.75 and len(words) >= 2:
        for _ in range(rng.randint(1, 2)):
            pos = rng.randrange(1, len(words))
            filler = rng.choice(FILLERS)
            if rng.random() < 0.4:
                words.insert(pos, rng.choice(DISCOURSE))
            else:
                words.insert(pos, filler + ("," if rng.random() < 0.6 else ""))
        applied.append("филлер")

    # 4) повтор слова / ложный старт (25%)
    if rng.random() < 0.25 and len(words) >= 3:
        pos = rng.randrange(1, len(words) - 1)
        words.insert(pos, words[pos])
        applied.append("повтор")

    # 5) реальная путаница слов из истории правок (45%)
    if error_pairs and rng.random() < 0.45:
        wrong, right = rng.choice(error_pairs)
        if right in out or right in " ".join(words):
            joined = " ".join(words)
            if right in joined:
                words = joined.replace(right, wrong, 1).split()
                applied.append("слово")
    elif dictionary and rng.random() < 0.2:
        pair = rng.choice(dictionary)
        wrong, right = str(pair.get("wrong", "")), str(pair.get("right", ""))
        if wrong and right and right in " ".join(words):
            words = " ".join(words).replace(right, wrong, 1).split()
            applied.append("слово")

    out = " ".join(words)

    # 6) пунктуация: убрать запятую (40%) или поставить лишнюю (20%)
    if rng.random() < 0.40:
        out2 = re.sub(r"(?<=[а-яёa-z]),(?=\s)", "", out, count=1)
        if out2 != out:
            out = out2
            applied.append("запятая минус")
    elif rng.random() < 0.25 and len(words) >= 3:
        pos = out.find(" ", len(out) // 3)
        if pos > 0:
            out = out[:pos] + "," + out[pos:]
            applied.append("запятая плюс")

    # 7) регистр: первое слово с маленькой (35%) или случайное (15%)
    if out and rng.random() < 0.35:
        out = out[0].lower() + out[1:]
        applied.append("регистр")
    elif rng.random() < 0.2:
        parts = out.split()
        if len(parts) >= 3:
            i = rng.randrange(1, len(parts))
            if parts[i][:1].isupper():
                parts[i] = parts[i][0].lower() + parts[i][1:]
                out = " ".join(parts)
                applied.append("регистр")

    # 8) ё → е (20%)
    if rng.random() < 0.2 and "ё" in out:
        out = out.replace("ё", "е")
        applied.append("ё")

    # 9) точка в конце пропала (20%)
    if rng.random() < 0.2:
        out = out.rstrip(".!?…")
        applied.append("точка")

    if stats is not None:
        for op in applied:
            stats[op] = stats.get(op, 0) + 1
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


def _word_distance(ref: str, hyp: str) -> float:
    """Словесное расстояние (1.0 = ничего общего) — для отсева мусорных пар."""
    r, h = _norm(ref).split(), _norm(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        cur = [i]
        for j, y in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / len(r)


def real_pairs(limit: int | None = None, max_edit: float = 0.5) -> list[dict]:
    """Пары из истории диктовок: сырой ASR -> итоговый текст.

    `max_edit` отсекает «склеенные» пары: в истории 25 % строк, где «итог»
    относится к ДРУГОЙ диктовке (наблюдатель AX подхватил чужой текст, либо в
    итоге лежит голосовая команда вроде «Повтори команду.»). Настоящая полировка
    так не переписывает — расхождение > половины слов означает мусор. Такие пары
    прямо вредят: учат модель переписывать фразу целиком и портят замер всем
    подходам сразу.
    """
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
        clean = finalize_clean(final)
        if max_edit is not None and _word_distance(asr, clean) > max_edit:
            continue                      # склейка разных диктовок / голосовая команда
        pairs.append({"src": asr, "dst": clean, "kind": "real", "id": int(r["id"])})
    if limit:
        pairs = pairs[:limit]
    return pairs


def word_error_pairs(min_len: int = 3) -> list[tuple[str, str]]:
    """Реальные путаницы слов из правок пользователя: (что было, что должно быть).

    Выдумывать их нельзя: «случайное слово вместо случайного» восстановить
    невозможно, и модель научится гадать. Берём только пары, где слова похожи
    (опечатка/окончание/омофон), — такие правки как раз решаемы.
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in real_pairs():
        a, b = _words(pair["src"]), _words(pair["dst"])
        if not a or not b:
            continue
        matcher = difflib.SequenceMatcher(a=[w.lower() for w in a],
                                          b=[w.lower() for w in b])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace" or (i2 - i1) != 1 or (j2 - j1) != 1:
                continue
            wrong, right = a[i1], b[j1]
            if len(wrong) < min_len or len(right) < min_len:
                continue
            ratio = difflib.SequenceMatcher(a=wrong.lower(), b=right.lower()).ratio()
            if ratio < 0.6:            # совсем разные слова — неразрешимо
                continue
            key = (wrong.lower(), right.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append((wrong, right))
    return out


def clean_texts(exclude: set[str] | None = None, include_wispr: bool = True) -> list[str]:
    """Чистые тексты для синтетики: итоги диктовок + эталоны Wispr.

    `exclude` — нормализованные тексты, которые НЕЛЬЗЯ использовать как источник
    (цели и входы валидации): иначе модель увидит правильный ответ валидации.
    """
    exclude = exclude or set()
    texts = [p["dst"] for p in real_pairs()]
    # `include_wispr=False` нужен для ЧЕСТНОГО замера на 624 фразах Wispr: их
    # эталоны иначе попадают в обучение как «чистые тексты», и кандидат получает
    # преимущество перед 4B, который этих текстов не видел.
    if include_wispr and WISPR.exists():
        for line in WISPR.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ref = (row.get("their_edited") or row.get("their_final") or "").strip()
            if ref:
                texts.append(finalize_clean(ref))
    texts = [t for t in texts if _norm(t) not in exclude]
    seen, out = set(), []
    for t in texts:
        key = t.lower()
        if key in seen or len(_words(t)) < 2 or len(t) > 400:
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


def build(variants: int = 3, seed: int = 20261011, val_every: int = 7,
          identity_cap: int = 1200, limit: int | None = None,
          out_dir: Path | None = None, include_wispr: bool = True) -> dict:
    """Собрать датасет.

    Разбиение — СТАБИЛЬНОЕ по id диктовки (`id % val_every == 0` идёт в валидацию):
    при добавлении новых диктовок старая разбивка не «переезжает».
    """
    rng = random.Random(seed)
    all_pairs = real_pairs(limit=limit)
    dictionary = load_dictionary()
    error_pairs = word_error_pairs()

    val_ids = {p["id"] for p in all_pairs if p["id"] % val_every == 0}
    train_ids = {p["id"] for p in all_pairs} - val_ids
    val_rows = [p for p in all_pairs if p["id"] in val_ids]

    # Тексты валидации блокируем ПО СОДЕРЖИМОМУ: одну и ту же фразу пользователь
    # диктовал несколько раз, и её копия с другим id иначе утечёт в обучение.
    blocked = {_norm(p["src"]) for p in val_rows} | {_norm(p["dst"]) for p in val_rows}

    val = [to_chat(p["src"], p["dst"]) for p in val_rows]
    train_real = []
    for p in all_pairs:
        if p["id"] not in train_ids:
            continue
        if _norm(p["dst"]) in blocked or _norm(p["src"]) in blocked:
            continue
        train_real.append(to_chat(p["src"], p["dst"]))

    texts = clean_texts(exclude=blocked, include_wispr=include_wispr)

    # Реальные пары дедуплицируем: одну фразу диктовали дважды.
    seen_pairs: set[tuple[str, str]] = set()
    unique_real: list[dict] = []
    for row in train_real:
        key = parts_key(row)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_real.append(row)
    train_real = unique_real

    # Синтетика: не больше `variants` РАЗНЫХ порч на текст, без дублей пар.
    stats: dict[str, int] = {}
    synthetic: list[dict] = []
    identity: list[dict] = []
    for text in texts:
        dst = text
        made = 0
        for _ in range(max(1, variants) * 4):
            if made >= variants:
                break
            src = corrupt(text, rng, dictionary, error_pairs, stats)
            key = (_norm(src), _norm(dst))
            if not key[0] or key[0] == key[1] or key in seen_pairs:
                continue
            seen_pairs.add(key)
            synthetic.append(to_chat(src, dst))
            made += 1
        identity_key = (_norm(dst), _norm(dst))
        if identity_key not in seen_pairs:
            seen_pairs.add(identity_key)
            identity.append(to_chat(dst, dst))

    rng.shuffle(identity)
    synthetic += identity[:identity_cap]

    rng.shuffle(synthetic)
    train = [row for row in train_real + synthetic
             if len(next(m["content"] for m in row["messages"]
                         if m["role"] == "assistant").strip()) >= 3]
    rng.shuffle(train)
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
        "train_total": len(train),
        "val_total": len(val),
        "error_pairs": len(error_pairs),
        "ops": stats,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Датасет для постпроцессора MyFLOW")
    ap.add_argument("--variants", type=int, default=3, help="разных порч на чистый текст")
    ap.add_argument("--val-every", type=int, default=7,
                    help="каждая N-я диктовка идёт в валидацию (стабильно по id)")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--limit", type=int, default=None, help="ограничить реальные пары")
    ap.add_argument("--out", type=Path, default=None, help="каталог для датасета")
    ap.add_argument("--no-wispr", dest="wispr", action="store_false",
                    help="не брать эталоны Wispr в синтетику (для честного замера на них)")
    args = ap.parse_args()
    db = ROOT / "data" / "myflow.sqlite"
    raw_count = 0
    if db.exists():
        import sqlite3 as _sql
        con = _sql.connect(f"file:{db}?mode=ro", uri=True)
        raw_count = con.execute(
            "SELECT COUNT(*) FROM history WHERE asr_text IS NOT NULL "
            "AND TRIM(asr_text) != '' AND (TRIM(COALESCE(edited_text,'')) != '' "
            "OR TRIM(COALESCE(polished_text,'')) != '')").fetchone()[0]
        con.close()
    st = build(variants=args.variants, seed=args.seed,
               val_every=args.val_every, limit=args.limit, out_dir=args.out,
               include_wispr=args.wispr)
    if raw_count:
        dropped = raw_count - st["real_total"]
        print(f"  отсеяно мусорных пар: {dropped} из {raw_count} "
              f"({dropped / raw_count * 100:.1f}%)")
    print("Датасет собран:")
    for key in ("clean_texts", "synthetic", "real_total", "real_train", "real_val",
                "train_total", "val_total", "error_pairs"):
        print(f"  {key:<12} {st[key]}")
    print("  операции порчи:", ", ".join(f"{k}={v}" for k, v in
                                        sorted(st["ops"].items(), key=lambda x: -x[1])))
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
