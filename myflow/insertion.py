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


def _input_struct(vk: int, keyup: bool) -> _INPUT:
    inp = _INPUT()
    inp.type = 1  # INPUT_KEYBOARD
    inp.ki.wVk = vk
    if keyup:
        inp.ki.dwFlags = _KEYEVENTF_KEYUP
    return inp


class ClipboardPaste:
    """Вставить текст через буфер обмена. int16-безопасно: unicode — ок."""

    def paste(self, text: str) -> None:
        if not text:
            return
        # Сохранить текущий буфер, чтобы не убить чужой контент пользователя.
        try:
            saved = pyperclip.paste()
        except pyperclip.PyperclipException:
            saved = None
        try:
            pyperclip.copy(text)
            time.sleep(0.05)  # дать буферу обмена устаканиться (cliboard race)
            self._send_ctrl_v()
        finally:
            # Вернуть буфер спустя паузу, чтобы вставка успела пройти.
            if saved is not None:
                time.sleep(0.15)
                pyperclip.copy(saved)

    def check_clipboard(self, expected: str) -> bool:
        """True, если в буфере сейчас всё ещё наш вставленный текст.

        (Резюм: для авто-обучения — берём последнюю вставленную строку.)
        """
        try:
            return pyperclip.paste() == expected
        except pyperclip.PyperclipException:
            return False

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