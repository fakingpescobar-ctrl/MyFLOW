"""LLM-судья: склеивает два текста распознавания в один, самый верный.

Замер на 624 реальных фразах (WER к «эталонному» тексту, ё=е):

| конфигурация | WER p50 | WER среднее | точных | ASR-время |
|---|---|---|---|---|
| только GigaAM | 0.083 | 0.138 | 42.1 % | 56 мс |
| только whisper | 0.111 | 0.147 | 36.7 % | 137 мс |
| авто-выбор (второе мнение по признакам) | 0.071 | 0.130 | 44.7 % | 67 мс |
| **LLM-судья: склейка двух текстов** | **0.040** | **0.107** | **49.0 %** | ~390 мс |

Ключевые находки эксперимента (`tools/experiment_llm_judge.py`):

* просить судью **выбрать** A или B — бесполезно: хуже обоих движков (среднее
  WER 0.143 против 0.135 у whisper) и совпадение с «оракулом» всего 69 %;
* просить **склеить** — работает: 64 % ответов совпадают с одним из входов
  (то есть судья неявно выбирает лучший), остальные аккуратно правят слова
  («комиты» → «коммит»);
* у судьи **нет аудио** — он оценивает правдоподобность. Поэтому один раз из 624
  он выдал эхо промпта («То, что человек реально сказал.»). Отсюда
  **предохранитель**: склейка обязана быть близка хотя бы к одному входу и не
  содержать служебных фраз из промпта — иначе берём обычный результат.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PROMPT = (
    "Два варианта распознавания одной и той же русской фразы. Собери из них один "
    "максимально верный текст: то, что человек реально сказал. Ничего не добавляй "
    "от себя, не пересказывай, сохрани порядок слов. Пунктуацию и регистр можно "
    "поправить.\nA: {a}\nB: {b}\nВерни только текст."
)

# Служебные фразы промпта: если они попали в ответ — это эхо, а не транскрипт.
_ECHO = re.compile(
    r"(человек реально сказал|ответь одной буквой|вариант[ыа]?\s+распознавания|"
    r"верни только текст|порядок слов|пунктуац\w*\s+и\s+регистр)",
    re.IGNORECASE)
# Модель иногда отвечает «A: текст» — префикс убираем.
_PREFIX = re.compile(r"^(?:[ABAB]\s*[:—–-]\s*)")
_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class MergeResult:
    """Итог судьи: текст (пустой — не смог), время и причина."""

    text: str
    latency_ms: int
    reason: str


def _words(text: str) -> list[str]:
    return [w.replace("ё", "е") for w in _WORD.findall((text or "").lower())]


def word_error(a: str, b: str) -> float:
    """Доля слов b, которых нет в a (нормализованный Левенштейн)."""
    ref, hyp = _words(a), _words(b)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, x in enumerate(ref, 1):
        cur = [i]
        for j, y in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / len(ref)


def rail_ok(merged: str, a: str, b: str, limit: float = 0.6) -> tuple[bool, str]:
    """Предохранитель: склейка должна быть похожа хотя бы на один вход.

    Ловит эхо промпта и выдумки модели. На выборке сработал 2 раза из 624 —
    и оба раза это были именно испорченные ответы.
    """
    text = (merged or "").strip()
    if len(text) < 2:
        return False, "пустой ответ"
    if _ECHO.search(text):
        return False, "эхо промпта"
    da, db = word_error(a, text), word_error(b, text)
    if da > limit and db > limit:
        return False, f"не похоже ни на один вариант ({da:.2f}/{db:.2f})"
    return True, "ок"


def merge(client, a: str, b: str, max_tokens: int = 220) -> MergeResult:
    """Спросить судью и применить предохранитель. Никогда не бросает исключений."""
    import time

    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return MergeResult("", 0, "нет текстов")
    if not a:
        return MergeResult(b, 0, "только whisper")
    if not b:
        return MergeResult(a, 0, "только GigaAM")

    started = time.perf_counter()
    try:
        answer = client.ask(PROMPT.format(a=a, b=b), max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 — судья не критичен
        return MergeResult("", int((time.perf_counter() - started) * 1000),
                           f"судья недоступен: {type(exc).__name__}")
    latency = int((time.perf_counter() - started) * 1000)

    text = _PREFIX.sub("", (answer or "").strip().strip('"').strip()).strip()
    ok, why = rail_ok(text, a, b)
    if not ok:
        return MergeResult("", latency, f"отклонено: {why}")
    return MergeResult(text, latency, "склейка")
