"""MyFLOW — самообучение из правок (аналог Wispr: user edits -> vocabulary).

Флоу как у Wispr: юзер правит распознанный текст -> мы диффим «было/стало»,
накапливаем повторяющиеся пары замен и по достижении порога (2+) фиксируем их
в dictionary с source='learned'. Словарик затем подставляется в промпт LLM
и в post-замену результата.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Callable

from myflow.db import Database

CANDIDATES_KEY = "learned_candidates"
LEARN_THRESHOLD = 2

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def extract_corrections(asr_text: str, edited_text: str) -> list[tuple[str, str]]:
    """Пары «было -> стало» по выравниванию слов (SequenceMatcher opcodes).

    Возвращает только осмысленные замены (by-position pairs), не тасование.
    """
    src = _tokens(asr_text)
    dst = _tokens(edited_text)
    pairs: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, src, dst, autojunk=False).get_opcodes():
        if tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                s = src[i1 + k] if i1 + k < i2 else ""
                d = dst[j1 + k] if j1 + k < j2 else ""
                if s and d and s != d:
                    pairs.append((s, d))
        elif tag == "delete":
            for k in range(i1, i2):
                pairs.append((src[k], ""))
        elif tag == "insert":
            for k in range(j1, j2):
                pairs.append(("", dst[k]))
    return pairs


def record_edit_and_learn(db: Database, history_id: int, polished_text: str,
                          edited_text: str, observation_end_reason: str | None = None) -> dict:
    """Зафиксировать ручную правку, посчитать edit_distance, обучить словарь.

    Обучение per-app: пара promo'тится в словарь с app-маской окна диктовки
    (history.app_window), глобально попадают только замены без привязки к окну
    (app не распознан/unknown). Повторяющиеся пары копятся в kv с контекстом.
    observation_end_reason — как мы узнали о правке (AX/буфер), аналог
    contentObservationEndReason у Wispr.
    """
    if not edited_text or edited_text == polished_text:
        return {"edit_distance": 0, "new_words": 0}

    # autojunk=False: иначе на текстах ≥200 токенов частые слова объявляются
    # «мусором» и рождают ложные пары замен (порча словаря на длинных диктовках).
    sm = SequenceMatcher(None, _tokens(polished_text), _tokens(edited_text), autojunk=False)
    edit_distance = sum(1 for tag, *_ in sm.get_opcodes() if tag != "equal")
    pairs = extract_corrections(polished_text, edited_text)
    app = db.get_app_window(history_id)
    # «unknown»/пустой заголовок — окно не определить: учим глобально.
    if app and app.strip().lower() not in ("unknown", ""):
        app_context = app[:64].lower()
    else:
        app_context = None

    db.attach_edit(
        history_id=history_id,
        edited_text=edited_text,
        edit_distance=edit_distance,
        corrections_json=json.dumps(pairs, ensure_ascii=False),
        observation_end_reason=observation_end_reason,
    )

    new_words = 0
    if pairs:
        candidates: dict[str, dict] = {}
        raw = db.kv_get(CANDIDATES_KEY)
        if raw:
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError:
                loaded = {}
            for key, value in loaded.items():
                if isinstance(value, dict):
                    candidates[key] = value
                else:  # старый формат: просто счётчик, глобальный
                    candidates[key] = {"n": int(value), "app": None}
        for src, dst in pairs:
            if not src or not dst:
                continue
            # Фильтр мусора: «слово -> число» почти всегда не правка пользователя
            # а совпадение с посторонним текстом в буфере (реальный случай:
            # выучилось «overlay -> 243» из скопированного числа). Число -> число
            # и число -> слово оставляем: это осмысленные правки.
            if dst.isdigit() and not src.isdigit() and len(src) > 3:
                continue
            # Ключ кандидата с контекстом окна: правки в разных окнах не
            # должны «складываться» в один глобальный счётчик.
            key = f"{src}\u0000{dst}\u0000{app_context or ''}"
            entry = candidates.get(key, {"n": 0, "app": app_context})
            entry["n"] += 1
            if app_context and not entry.get("app"):
                entry["app"] = app_context
            candidates[key] = entry
        promoted: list[str] = []
        for key, entry in candidates.items():
            if entry["n"] >= LEARN_THRESHOLD:
                parts = key.split("\u0000")
                src, dst = parts[0], parts[1]
                # add_word сам инкрементит hit_count через ON CONFLICT —
                # отдельный bump_word давал двойной счёт.
                db.add_word(src, dst, source="learned", app_mask=entry.get("app"))
                promoted.append(key)
        for key in promoted:
            candidates.pop(key, None)
        db.kv_set(CANDIDATES_KEY, json.dumps(candidates, ensure_ascii=False))
        new_words = len(promoted)

    return {"edit_distance": edit_distance, "new_words": new_words}


def looks_like_edit_of(base: str, candidate: str, min_overlap: float = 0.5) -> bool:
    """Является ли `candidate` RUM-правкой `base`, а не случайным скопированным куском?

    Пересечение словарных токенов >= min_overlap (доля от меньшего текста).
    Защита от ложных срабатываний, когда юзер скопировал посторонний текст
    в окне обучения (наш текст не совпал — это чужая копия, не правка).
    """
    a = _tokens(base)
    b = _tokens(candidate)
    if not a or not b:
        return False
    if b == a:
        return False  # идентично — не правка
    sm = SequenceMatcher(None, a, b, autojunk=False)
    matching = sum(n for _, _, n in sm.get_matching_blocks())
    overlap = matching / min(len(a), len(b))
    return overlap >= min_overlap


def watch_pasted_edit(
    db: Database,
    history_id: int,
    polished_text: str,
    read_clipboard: Callable[[], str],
    window_seconds: float = 6.0,
    poll_interval: float = 0.6,
) -> dict | None:
    """Дождаться правки вставленного текста и обучить словарь (Ф9).

    Запускается фоном после paste(). Периодически читает буфер обмена через
    `read_clipboard` (это ЧТЕНИЕ, `ClipboardPaste.read`, а не вставка!);
    если юзер скопировал (Ctrl+C) отредактированную версию нашего текста —
    diff -> record_edit_and_learn. Возвращает словарь с edit_distance/new_words,
    или None, если за окно не заметили осмысленной правки/скопирован посторонний текст.
    """
    import time

    deadline = time.monotonic() + window_seconds
    while time.monotonic() < deadline:
        try:
            current = read_clipboard() or ""
        except Exception:
            current = ""
        if current and current != polished_text and looks_like_edit_of(polished_text, current):
            try:
                result = record_edit_and_learn(db, history_id, polished_text, current)
            except Exception:
                result = None
            if result and result["edit_distance"]:
                return result
        # sleep ВСЕГДА: раньше `continue` при исключении его пропускал,
        # и поток 6 секунд крутил холостой цикл на 100% CPU.
        time.sleep(poll_interval)
    return None