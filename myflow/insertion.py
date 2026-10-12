"""MyFLOW — вставка текста в активное окно (аналог C# helper у Wispr Flow).

Wispr вставляет через нативный accessibility-хелпер. Мы проще и надёжнее:
Clipboard + Ctrl+V через SendInput (работает в терминалах, браузерах, IDE).
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import pyperclip

# Структуры WinAPI для SendInput (KEYEVENTF_KEYUP = 0x0002, VK_CONTROL = 0x11).
# ВАЖНО (x64): SendInput требует cbSize == sizeof(INPUT) == 40. Union должен
# содержать ВСЕ три типа (MOUSEINPUT 32B — самый большой); если оставить только
# KEYBDINPUT (24B) — размер станет 32 и SendInput молча вернёт 0 (ничего не шлёт).
_VK_CONTROL = 0x11
_KEYEVENTF_KEYUP = 0x0002
_CTRL_V = 0x56
_ULONG_PTR = ctypes.c_size_t  # 8 байт на x64, 4 на x86


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]


# Форматы буфера обмена, которые НЕЛЬЗЯ убивать копированием текста:
# CF_BITMAP, CF_METAFILEPICT, CF_DIB, CF_ENHMETAFILE, CF_HDROP (файлы), CF_DIBV5.
_RICH_CLIPBOARD_FORMATS = (2, 3, 5, 14, 15, 17)
_KEYEVENTF_UNICODE = 0x0004
_INPUT_KEYBOARD = 1


def clipboard_has_rich() -> bool:
    """Есть ли в буфере картинка/файлы (не только текст).

    Проверка дешёвая и НЕ читает данные: IsClipboardFormatAvailable только
    спрашивает, есть ли формат. Нужна, чтобы не уничтожить картинку/скопированные
    файлы: `pyperclip.copy()` делает EmptyClipboard и они исчезают навсегда.
    """
    try:
        k = ctypes.windll.user32
        return any(bool(k.IsClipboardFormatAvailable(fmt))
                   for fmt in _RICH_CLIPBOARD_FORMATS)
    except Exception:  # noqa: BLE001 — не Windows/нет доступа: считаем, что нет
        return False


def _utf16_units(ch: str) -> tuple[int, ...]:
    """Кодовые единицы UTF-16 для символа (2 для эмодзи и редких знаков)."""
    raw = ch.encode("utf-16-le")
    return tuple(int.from_bytes(raw[i:i + 2], "little")
                 for i in range(0, len(raw), 2))


def send_unicode_text(text: str, pause: float = 0.0) -> int:
    """Напечатать текст через SendInput(KEYEVENTF_UNICODE). Возвращает число знаков.

    Буфер обмена НЕ трогается — поэтому картинки и файлы в нём остаются целы.
    Так вставляем, когда в буфере что-то ценное (Ctrl+V его бы уничтожил).
    """
    if not text:
        return 0
    user32 = ctypes.windll.user32
    sent = 0
    for ch in text:
        for unit in _utf16_units(ch):
            down = _INPUT()
            down.type = _INPUT_KEYBOARD
            down.ki.wVk = 0
            down.ki.wScan = unit
            down.ki.dwFlags = _KEYEVENTF_UNICODE
            up = _INPUT()
            up.type = _INPUT_KEYBOARD
            up.ki.wVk = 0
            up.ki.wScan = unit
            up.ki.dwFlags = _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP
            arr = (_INPUT * 2)(down, up)
            if user32.SendInput(2, ctypes.byref(arr), ctypes.sizeof(_INPUT)) == 2:
                sent += 1
        if pause:
            time.sleep(pause)
    return sent


def _input_struct(vk: int, keyup: bool) -> _INPUT:
    inp = _INPUT()
    inp.type = 1  # INPUT_KEYBOARD
    inp.ki.wVk = vk
    if keyup:
        inp.ki.dwFlags = _KEYEVENTF_KEYUP
    return inp


class ClipboardPaste:
    """Вставить текст через буфер обмена. int16-безопасно: unicode — ок."""

    def __init__(self) -> None:
        self.last_method = ""      # как вставили: clipboard-unicode (для журнала)

    def paste(self, text: str) -> None:
        if not text:
            return
        # В буфере картинка или скопированные файлы? Ctrl+V их уничтожит
        # (pyperclip.copy делает EmptyClipboard, вернуть уже нечего) — поэтому
        # печатаем текст через SendInput(UNICODE) и буфер не трогаем вообще.
        if clipboard_has_rich():
            sent = send_unicode_text(text)
            self.last_method = "unicode без буфера" if sent else "unicode (не отправлено)"
            print(f"[myflow]   вставка: печатаю текст ({sent} знаков) — "
                  f"в буфере картинка/файлы, не трогаю")
            return
        # Сохранить текущий буфер, чтобы не убить чужой контент пользователя.
        saved = self._read_text()
        try:
            self._copy(text)
            time.sleep(0.05)  # дать буферу обмена устаканиться (clipboard race)
            self._send_ctrl_v()
            self.last_method = "clipboard"
        finally:
            # ВАЖНО: '' значит «в буфере НЕ текст» (картинка/файлы). Копирование
            # '' вызывает EmptyClipboard и БЕЗВОЗВРАТНО уничтожает CF_HDROP/
            # CF_BITMAP — поэтому пустую строку не восстанавливаем вообще.
            if saved:
                time.sleep(0.15)
                self._copy(saved, best_effort=True)

    @staticmethod
    def _read_text() -> str | None:
        try:
            return pyperclip.paste()
        except pyperclip.PyperclipException:
            return None

    @staticmethod
    def _copy(text: str, best_effort: bool = False) -> None:
        """Копирование с ретраями: буфер бывает занят чужим процессом."""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                pyperclip.copy(text)
                return
            except pyperclip.PyperclipException as exc:
                last_exc = exc
                time.sleep(0.05 * (attempt + 1))
        if last_exc is not None and not best_effort:
            raise last_exc

    def check_clipboard(self, expected: str) -> bool:
        """True, если в буфере сейчас всё ещё наш вставленный текст.

        (Резюм: для авто-обучения — берём последнюю вставленную строку.)
        """
        try:
            return pyperclip.paste() == expected
        except pyperclip.PyperclipException:
            return False

    can_read = True

    def read(self) -> str:
        """Текущий ТЕКСТ буфера обмена (для авто-обучения). Буфер не меняет.

        Возвращает '' для картинок/файлов и при недоступном буфере —
        вызывающий код не должен трактовать '' как «пустой текст вставили».
        """
        try:
            return pyperclip.paste() or ""
        except pyperclip.PyperclipException:
            return ""

    @staticmethod
    def _send_ctrl_v() -> None:
        events = (
            _input_struct(_VK_CONTROL, keyup=False),
            _input_struct(_CTRL_V, keyup=False),
            _input_struct(_CTRL_V, keyup=True),
            _input_struct(_VK_CONTROL, keyup=True),
        )
        arr = (_INPUT * len(events))(*events)
        sent = ctypes.windll.user32.SendInput(len(events), arr, ctypes.sizeof(_INPUT))
        if sent != len(events):
            raise RuntimeError(f"SendInput вставил {sent}/{len(events)} событий")

class HistoryOnlyPaste:
    """Ничего не вставляет: текст живёт только в истории MyFLOW.

    Для начитки вслух и замеров: читаешь в микрофон, ничего не засоряется, а каждая
    фраза ложится в историю — оттуда её и разбираем против эталонного текста.
    """

    can_read = False

    def read(self) -> str:
        return ""

    def paste(self, text: str) -> bool:
        # Успех возвращаем намеренно: конвейер (история, статистика) должен идти
        # штатно, просто без вставки в чужое окно.
        return bool(text and text.strip())
