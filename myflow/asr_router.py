"""Авто-выбор движка распознавания: быстро для простых фраз, точно для сложных.

Замер на 624 реальных диктовках пользователя (`data/eval_wispr/ab_asr.json`):

| политика | WER p50 | WER среднее | задержка p50 | срабатываний |
|---|---|---|---|---|
| только GigaAM | 0.100 | 0.152 | 56 мс | 0 % |
| только whisper | 0.091 | 0.135 | 137 мс | 100 % |
| **обрывок ИЛИ темп < 1.8 сл/с** | **0.091** | 0.140 | **80 мс** | 27 % |
| + филлеры (режим точности) | **0.083** | **0.135** | 126 мс | 53 % |

То есть второе мнение по признакам **не хуже чистого whisper по точности**, но в
2-3 раза дешевле по задержке: GigaAM (56 мс) отвечает первым, а whisper
подключается только когда результат выглядит подозрительно.

Сигналы «подозрительно» (все проверяются по тексту GigaAM и длительности):
  * обрывок в конце («...ты по.») — короткий последний токен;
  * нет конечной пунктуации — e2e-вариант GigaAM обычно её ставит;
  * низкий темп речи (слов в секунду) — типичный признак пропуска/обрыва;
  * филлеры (опционально, режим точности);
  * очень длинная фраза (опционально).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from myflow.fastpath import _SHORT_WORDS_OK

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_TERMINAL = ".!?…"
_FILLER_RE = re.compile(
    r"\b(э+|эм+|мм+|значит|типа|короче|как бы|в общем|блин|это самое)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Decision:
    """Нужно ли второе мнение и почему."""

    use_second: bool
    reason: str


def review(text: str, *, duration_ms: int, cfg) -> Decision:
    """Решить, звать ли whisper вторым мнением. Все пороги — из `cfg.asr`."""
    t = (text or "").strip()
    if not t:
        return Decision(False, "пусто")
    words = _WORD_RE.findall(t)
    if not words:
        return Decision(False, "нет слов")

    # 1) обрывок в конце: GigaAM иногда режет слово и всё равно ставит точку
    last = words[-1].lower()
    if len(last) <= 2 and last not in _SHORT_WORDS_OK:
        return Decision(True, f"обрывок «{last}»")

    # 2) нет конечной пунктуации — e2e-вариант GigaAM её обычно ставит
    if t[-1] not in _TERMINAL:
        return Decision(True, "нет конечной пунктуации")

    seconds = max(0.0, duration_ms / 1000.0)
    min_seconds = float(getattr(cfg, "router_min_seconds", 1.5) or 1.5)
    if seconds >= min_seconds:
        # 3) низкий темп речи: пропуск слов или обрыв
        wps = len(words) / seconds
        min_wps = float(getattr(cfg, "router_min_wps", 1.8) or 1.8)
        if min_wps > 0 and wps < min_wps:
            return Decision(True, f"темп {wps:.1f} сл/с")

        # 4) длинные фразы (по умолчанию выключено: GigaAM на них не хуже прочих)
        long_seconds = float(getattr(cfg, "router_long_seconds", 0.0) or 0.0)
        if long_seconds > 0 and seconds >= long_seconds:
            return Decision(True, f"длинная фраза {seconds:.0f} с")

    # 5) филлеры (режим точности: whisper их обычно не слышит)
    if bool(getattr(cfg, "router_use_fillers", False)) and _FILLER_RE.search(t):
        return Decision(True, "филлеры в тексте")

    return Decision(False, "чисто")
