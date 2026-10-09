"""MyFLOW — метрика расхождения ASR ↔ результат LLM (аналог formattingDivergenceScore).

Зачем: эвристика «слов стало в 1.6 раза больше» ловит только грубые случаи.
Числовая метрика позволяет (а) логировать, насколько модель переписала фразу,
(б) принимать решение об откате на сырой ASR по порогу, (в) копить статистику
качества полировки по стилям и приложениям.

Считаем расстояние Левенштейна по СЛОВАМ (регистр/пунктуация не важны) и
нормируем на длину более длинной последовательности → [0, 1], где 0 — LLM ничего
не изменила, 1 — полностью другой текст.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+", re.UNICODE)

# Порог, выше которого считаем, что модель «переписала» фразу, а не поправила.
MAX_DIVERGENCE = 0.5


def tokens(text: str) -> list[str]:
    """Слова в нижнем регистре (пунктуация игнорируется)."""
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def word_distance(a: list[str], b: list[str]) -> int:
    """Расстояние Левенштейна на списках слов (сколько вставок/удалений/замен)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # удаление
                           cur[j - 1] + 1,       # вставка
                           prev[j - 1] + (wa != wb)))  # замена/совпадение
        prev = cur
    return prev[-1]


def divergence_score(asr_text: str, polished_text: str) -> float:
    """Насколько результат отличается от входа: 0.0 (идентично) .. 1.0 (другой текст)."""
    a, b = tokens(asr_text), tokens(polished_text)
    if not a and not b:
        return 0.0
    return word_distance(a, b) / max(len(a), len(b))


def similarity_score(asr_text: str, polished_text: str) -> float:
    return 1.0 - divergence_score(asr_text, polished_text)
