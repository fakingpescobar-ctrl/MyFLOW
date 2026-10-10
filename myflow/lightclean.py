"""Лёгкая чистка текста ASR без LLM.

Зачем: LLM-полировка — самая дорогая часть пайплайна (замер: p50 210 мс, p90 603,
max 6230; это ~60% задержки e2e). Но большая часть фраз требует лишь
МЕХАНИЧЕСКИХ правок, которые можно сделать детерминированно за микросекунды:

  * мусор в начале («...», «,», лишние точки) — GigaAM иногда начинает фразу
    обрывком, e2e-декодер ставит «...»;
  * слова-паразиты в начале: «ну», «так,», «короче,», «эээ», «эм»;
  * задвоенная пунктуация («телевизоре..»), пробелы перед знаками;
  * строчная буква после точки и в начале фразы.

После такой чистки текст часто становится «уже оформленным», и тогда полировку
можно не звать вовсе (быстрый путь) — это экономит ~280 мс на фразе.

ПРАВИЛА НАМЕРЕННО КОНСЕРВАТИВНЫЕ: ничего не переставляем и не переписываем,
не трогаем слова в середине фразы, не удаляем смысловые слова. «Так работает»
останется как есть (без запятой «так» — не филлер), а «Так, работает» -> «Работает».
"""
from __future__ import annotations

import re

# Мусор в начале: точки/многоточия, запятые, двоеточия, тире.
_LEADING_NOISE = re.compile(r"^[\s.,;:…]+")

# Звуки-заминки: их можно снимать в начале всегда (самостоятельным словом).
_HESITATIONS: tuple[str, ...] = ("эээ", "ээ", "эмм", "эм", "мм", "м-м", "э")

# «ну» в начале — практически всегда филлер (в отличие от «так»/«вот»).
_ALWAYS_LEAD: tuple[str, ...] = ("ну",)

# Эти снимаем только когда после них стоит запятая (иначе слово смысловое:
# «так работает», «вот дом» трогать нельзя).
_COMMA_FILLERS: tuple[str, ...] = ("так", "вот", "короче", "значит", "слушай",
                                   "смотри", "понимаешь", "кстати")

_TERMINAL = ".!?…"
_SENTENCE_END = re.compile(r"([.!?…])\s+([а-яёa-z])", re.IGNORECASE)
_DUP_PUNCT = (
    (re.compile(r"\.{2,}"), ".", "схлопнуты точки"),
    (re.compile(r",{2,}"), ",", "схлопнуты запятые"),
    (re.compile(r"([!?])\1+"), r"\1", "схлопнуты знаки"),
    (re.compile(r"\s+([,.;:!?])"), r"\1", "убран пробел перед знаком"),
    (re.compile(r"\s{2,}"), " ", "схлопнуты пробелы"),
)


def _strip_leading_filler(text: str, words: tuple[str, ...], *,
                          need_comma: bool, min_words_left: int = 2) -> tuple[str, str | None]:
    """Снять филлер в начале. Возвращает (текст, что сняли) или (текст, None)."""
    low = text.lower()
    for filler in sorted(words, key=len, reverse=True):
        if not low.startswith(filler):
            continue
        rest = text[len(filler):]
        # Филлер должен быть отдельным словом: за ним пробел/запятая/конец.
        if rest and rest[0] not in " ,.!?;:":
            continue
        if need_comma:
            if not rest.lstrip().startswith(","):
                continue
            rest = rest.lstrip()[1:]
        rest = rest.lstrip(" ,")
        if len(rest.split()) < min_words_left:
            continue
        return rest, filler
    return text, None


def light_clean(text: str) -> tuple[str, list[str]]:
    """Механическая чистка. Возвращает (текст, список применённых правил)."""
    rules: list[str] = []
    out = (text or "").strip()

    stripped = _LEADING_NOISE.sub("", out)
    if stripped != out:
        rules.append("убран мусор в начале")
        out = stripped

    # Заминки и «ну» — можно повторно (бывает «ну эээ …»). После снятия филлера
    # снова убираем мусор в начале: «ну ...проверка» -> «...проверка» -> «Проверка».
    for _ in range(4):
        changed = False
        stripped = _LEADING_NOISE.sub("", out)
        if stripped != out:
            if "убран мусор в начале" not in rules:
                rules.append("убран мусор в начале")
            out = stripped
            changed = True
        out2, removed = _strip_leading_filler(out, _HESITATIONS, need_comma=False)
        if removed:
            rules.append(f"снята заминка «{removed}»")
            out = out2
            changed = True
        else:
            out2, removed = _strip_leading_filler(out, _ALWAYS_LEAD, need_comma=False)
            if removed:
                rules.append(f"снят филлер «{removed}»")
                out = out2
                changed = True
        if not changed:
            break

    out2, removed = _strip_leading_filler(out, _COMMA_FILLERS, need_comma=True,
                                          min_words_left=3)
    if removed:
        rules.append(f"снят филлер «{removed},»")
        out = out2

    for pattern, repl, why in _DUP_PUNCT:
        new = pattern.sub(repl, out)
        if new != out:
            if why not in rules:
                rules.append(why)
            out = new

    # Заглавная в начале и после конца предложения.
    out = out.strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
        rules.append("заглавная в начале")
    fixed = _SENTENCE_END.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}", out)
    if fixed != out:
        rules.append("заглавная после точки")
        out = fixed

    # Точка в конце, если фраза «повисла» (есть слова, но нет знака).
    if out and out[-1] not in _TERMINAL and len(out.split()) >= 2:
        out = out.rstrip() + "."
        rules.append("добавлена точка")

    # Повторная уборка пробелов, если что-то появилось после правок.
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out, rules
