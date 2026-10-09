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

import json
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

# Все kind, которые MyFLOW реально умеет исполнять (main._run_command).
# Всё, что не отсюда (в т.ч. «text» от LLM-классификатора), — не команда.
KNOWN_KINDS: frozenset[str] = frozenset(
    {"stop", "skip", "delete_last", "new_paragraph", "send", "summarize", "list"}
)

# Слова-маркеры для LLM-интента («свободная формулировка»).
# ВАЖНО: по границам слов — иначе «краткосрочный», «стопка», «списке»
# тянут лишний LLM-вызов на каждую обычную диктовку (латентность).
# «отправь» сознательно НЕ здесь — «отправь X» это диктовка.
_INTENT_RE = re.compile(
    r"\b(?:стоп|скип|абзац|кратко|резюмир\w*|перескаж\w*|перепиш\w*|"
    r"сократ\w*|удали\w*|очист\w*|список|списком)\b",
    re.IGNORECASE | re.UNICODE,
)
# Для обратной совместимости/чтения: те же маркеры списком.
INTENT_MARKERS: tuple[str, ...] = (
    "перескажи", "перепиши", "сократи", "резюмируй", "кратко",
    "спис", "удали", "очисти", "абзац", "стоп", "скип",
)

# Фразы, при которых префикс-команда отдаёт ЧИСТЫЙ текст (не перехватываем,
# хотя формально подходит под префикс). Защита от случайных срабатываний в
# реальных диктовках: «отправь это сообщение» — текст, не команда.
# Значения — СТЕМЫ: «продукт» покрывает «продукты/продуктов», «задач» — «задачи/задач».
_PREFIX_SAFE_WORDS: dict[str, tuple[str, ...]] = {
    "список": ("продукт", "дел", "задач", "покуп", "подарк"),
    "списком": ("продукт", "дел", "задач", "покуп", "подарк"),
}

_NORM_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS_RE = re.compile(r"\s+")
# Первое слово в ОРИГИНАЛЕ (чтобы срезать его, сохранив регистр остатка).
_FIRST_WORD_RE = re.compile(r"^\s*[^\W\d_]+", re.UNICODE)
_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+", re.UNICODE)


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
        # «отправь» = Enter к последней вставке (НЕ повторная вставка текста!),
        # «перескажи/список» = переоформить последнюю вставку.
        return VoiceCommand(kind=kind, rephrase_last=(kind in ("summarize", "list")))

    # 2) Префикс + аргумент (или ровно слово-префикс без аргумента)
    first = norm.split(" ", 1)[0]
    if first in PREFIX_COMMANDS:
        # «список продуктов...» — вероятно настоящая диктовка, не команда
        if first in _PREFIX_SAFE_WORDS:
            rest = norm[len(first) + 1:].strip()
            first_rest = rest.split(" ", 1)[0]
            if first_rest and any(
                first_rest.startswith(stem) for stem in _PREFIX_SAFE_WORDS[first]
            ):
                return None
        kind = PREFIX_COMMANDS[first]
        # Аргумент берём из ОРИГИНАЛА: normalize() ниже опускает регистр и
        # вырезает пунктуацию, а это текст будущей вставки («Про Москву»).
        arg = _FIRST_WORD_RE.sub("", text, count=1).strip(" :—-—\t«»\"'")
        if not arg:
            # Ровно «перескажи»/«списком» — применить к последней вставке
            return VoiceCommand(kind=kind, rephrase_last=True, source="prefix")
        return VoiceCommand(kind=kind, text=arg, source="prefix")

    return None


def intent_markers(text: str) -> list[str]:
    """Командные слова в тексте (для решения «LLM-классифицировать»).

    Маркеры ищутся по ГРАНИЦАМ СЛОВ: «краткосрочный план», «стопка книг»,
    «списке дел» больше не считаются командными — иначе каждая такая
    диктовка тянула лишний LLM-вызов (и риск потерять текст).
    """
    return _INTENT_RE.findall(normalize(text))


def is_llm_intent_candidate(text: str) -> bool:
    """Есть явный командный маркер, но нет точного триггера -> спросить LLM."""
    return len(intent_markers(text)) > 0


def arg_looks_like_source(arg: str, source: str, min_ratio: float = 0.6) -> bool:
    """Похож ли аргумент LLM-команды на исходную фразу (не переписал ли модель).

    Страховка: если классификатор вернул в «тексте для пересказа» совсем
    другой текст — это не команда, а обычная диктовка (её нельзя терять).
    """
    a = set(_WORD_RE.findall((arg or "").lower()))
    b = set(_WORD_RE.findall((source or "").lower()))
    if not a:
        return False
    return len(a & b) / len(a) >= min_ratio


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


def _parse_json_loose(content: str) -> dict:
    """JSON из ответа LLM, устойчиво к markdown-обёртке и мусору вокруг."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("не объект JSON")
    return data


def llm_classify(text: str, polish_url: str, model: str) -> VoiceCommand | None:
    """LLM-классификатор интента (быстрый, temperature=0, 150 токенов).

    ГЛАВНЫЙ ИНВАРИАНТ: этот путь НИКОГДА не возвращает kind=text/мусор —
    только известные команды. Иначе вызывающий код считает фразу
    «обработанной» и диктовка исчезает (был реальный баг).
    """
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
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"]["content"] or ""
        data = _parse_json_loose(content)
        kind = str(data.get("kind", "")).strip().lower()
        arg = str(data.get("text", "") or "").strip()
        if kind not in KNOWN_KINDS:
            return None                      # «text» и любой мусор => диктовка
        if kind in ("summarize", "list"):
            if not arg:
                return VoiceCommand(kind=kind, rephrase_last=True, source="llm")
            if not arg_looks_like_source(arg, text):
                return None                  # модель переписала текст — не команда
            return VoiceCommand(kind=kind, text=arg, source="llm")
        if kind == "send":
            # Голый «отправь» = Enter к последней вставке, БЕЗ повторной вставки.
            return VoiceCommand(kind="send", source="llm")
        return VoiceCommand(kind=kind, source="llm")
    except Exception:
        return None