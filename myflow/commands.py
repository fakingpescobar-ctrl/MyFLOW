"""MyFLOW — распознавание голосовых команд.

Три уровня, по возрастанию риска ложного срабатывания:

1) Точные фразы без аргумента (ровно совпадение):
     «стоп», «скип»           -> пропустить текущую фразу (не вставлять)
     «удали последнее»        -> Ctrl+Z (если последняя вставка свежая <60с)
     «новый абзац»            -> вставить пустую строку
     «отправь»                -> Enter (подтвердить последнюю вставку)
2) Префиксы с аргументом (фраза начинается со слова + пробел + текст):
     «перескажи …» / «перепиши …» -> style=summarize  (текст = остаток)
     «список …» / «списком …»     -> style=list       (текст = остаток)
   Без аргумента (ровно «перескажи»/«списком») -> переоформить ПОСЛЕДНЮЮ
   вставленную фразу (пересказ/список вставляется ниже оригинала).
3) LLM-интент: НЕ точные триггеры, но есть явные командные слова
   («удали последнюю фразу», «кратко резюмируй», «пиши списком» ...) —
   лёгкий LLM-классификатор решает kind/text.

СОЗНАТЕЛЬНО не перехватывается: «отправь X» с аргументом (обычная диктовка:
«отправь документы Тане») — только голый «отправь» = команда.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Точные фразы-команды (без аргумента), kind действия
EXACT_COMMANDS: dict[str, str] = {
    "стоп": "stop",
    "скип": "skip",
    "удали последнее": "delete_last",
    "удалить последнее": "delete_last",
    "новый абзац": "new_paragraph",
    "отправь": "send",           # без аргумента = Enter к последней вставке
    "перескажи": "summarize",    # без аргумента = пересказ последней вставки
    "перепиши": "summarize",
    "список": "list",            # без аргумента = оформить последнюю вставку
    "списком": "list",
}

# Префиксы с аргументом: слово -> kind
PREFIX_COMMANDS: dict[str, str] = {
    "перескажи": "summarize",
    "перепиши": "summarize",
    "сократи": "summarize",
    "список": "list",
    "списком": "list",
}

# Слова-маркеры для LLM-интента («свободная формулировка»)
# «отправь» сознательно НЕ здесь — «отправь X» это диктовка.
INTENT_MARKERS: tuple[str, ...] = (
    "перескажи", "перепиши", "сократи", "резюмируй", "кратко",
    "спис", "удали", "удалить", "очисти", "абзац", "стоп", "скип",
)

# Фразы, при которых префикс-команда отдаёт ЧИСТЫЙ текст (не перехватываем,
# хотя формально подходит под префикс). Защита от случайных срабатываний в
# реальных диктовках: «отправь это сообщение» — текст, не команда.
_PREFIX_SAFE_WORDS: dict[str, tuple[str, ...]] = {
    "список": ("продукты", "дел", "задач", "покупок", "подарков"),
}

_NORM_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS_RE = re.compile(r"\s+")


@dataclass
class VoiceCommand:
    kind: str                    # text | summarize | list | send | delete_last | new_paragraph | stop | skip
    text: str = ""               # аргумент (остаток фразы) или "" (без аргумента)
    rephrase_last: bool = False  # True: команда без аргумента -> к последней вставке
    source: str = "exact"        # exact | prefix | llm
    markers: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    """lower, без пунктуации, один пробел. Для сравнения триггеров."""
    return _WS_RE.sub(" ", _NORM_RE.sub(" ", (text or "").lower())).strip()


def classify(text: str) -> VoiceCommand | None:
    """Точные фразы и префиксы. None — обычный текст (или кандидат LLM-интента)."""
    norm = normalize(text)
    if not norm:
        return None

    # 1) Точные фразы без аргумента
    if norm in EXACT_COMMANDS:
        kind = EXACT_COMMANDS[norm]
        return VoiceCommand(kind=kind, rephrase_last=(kind in ("send", "summarize", "list")))

    # 2) Префикс + аргумент (или ровно слово-префикс без аргумента)
    first = norm.split(" ", 1)[0]
    if first in PREFIX_COMMANDS:
        # «список продуктов...» — вероятно настоящая диктовка, не команда
        if first in _PREFIX_SAFE_WORDS:
            rest = norm[len(first) + 1:].strip()
            first_rest = rest.split(" ", 1)[0]
            if first_rest in _PREFIX_SAFE_WORDS[first]:
                return None
        kind = PREFIX_COMMANDS[first]
        arg = normalize(text[len(first):]).strip(" :")
        if not arg:
            # Ровно «перескажи»/«списком» — применить к последней вставке
            return VoiceCommand(kind=kind, rephrase_last=True, source="prefix")
        return VoiceCommand(kind=kind, text=arg, source="prefix")

    return None


def intent_markers(text: str) -> list[str]:
    """Командные слова в тексте (для решения «LLM-классифицировать»)."""
    norm = normalize(text)
    hits: list[str] = []
    for word in INTENT_MARKERS:
        if word in norm:
            hits.append(word)
    return hits


def is_llm_intent_candidate(text: str) -> bool:
    """Есть явный командный маркер, но нет точного триггера -> спросить LLM."""
    return len(intent_markers(text)) > 0


# ---- LLM-классификатор (лёгкий промпт, если нет точного триггера) ----

_INTENT_SYSTEM = (
    "Ты — классификатор голосового диктанта. Определи: является ли фраза "
    "голосовой командой или обычным текстом для вставки.\n"
    "Ответ строго JSON (без markdown):\n"
    "{\"kind\":\"text\",\"text\":\"<полный текст>\"} — обычный текст\n"
    "{\"kind\":\"summarize\",\"text\":\"<текст для пересказа>\"}\n"
    "{\"kind\":\"list\",\"text\":\"<текст для списка>\"}\n"
    "{\"kind\":\"send\",\"text\":\"<текст>\"}\n"
    "{\"kind\":\"stop\",\"text\":\"\"} или \"skip\" — пропустить\n"
    "{\"kind\":\"delete_last\",\"text\":\"\"}\n"
    "{\"kind\":\"new_paragraph\",\"text\":\"\"}\n\n"
    "Если фраза — обычный текст (диктуешь предложения, мысли, текст), "
    "верни kind=text с полным текстом. Командные слова: "
    "перескажи, перепиши, сократи, списком, список, удали, очисти, абзац, стоп, скип."
)


def llm_classify(text: str, polish_url: str, model: str) -> VoiceCommand | None:
    """LLM-классификатор интента (быстрый, temperature=0, 150 токенов)."""
    import json
    import requests as _req
    try:
        resp = _req.post(
            f"{polish_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _INTENT_SYSTEM},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
                "max_tokens": 150,
            },
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        kind = str(data.get("kind", "text")).strip()
        arg = str(data.get("text", "")).strip()
        rephrase = kind in ("send", "summarize", "list") and not arg
        return VoiceCommand(kind=kind, text=arg, rephrase_last=rephrase, source="llm")
    except Exception:
        return None