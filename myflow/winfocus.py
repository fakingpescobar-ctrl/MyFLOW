"""MyFLOW — какое окно было активным и как в него вернуться.

Зачем: пользователь диктует в чат, случайно кликает мышью в сторону — и
Ctrl+V уходит уже в другое окно (текст «не туда»). Решение: запомнить окно,
активное в МОМЕНТ СТАРТА диктовки, и вставлять именно в него, вернув фокус.

Здесь три вещи:
  * `current_window()` — что сейчас в фокусе (hwnd + pid + заголовок);
  * `focus_window(hwnd)` — вернуть фокус (SetForegroundWindow с обходом
    ограничения Windows на «кражу» фокуса через AttachThreadInput);
  * `FocusHistory` — кольцевой буфер замеров фокуса, чтобы узнать, какое окно
    было активно N секунд назад (в AUTO-режиме фраза начинается сама, и момент
    старта мы узнаём только по её длительности).

Windows-only; на других платформах функции мягко возвращают None/False.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

_IS_WINDOWS = sys.platform.startswith("win")

SW_RESTORE = 9
ASFW_ANY = -1


@dataclass(frozen=True)
class WindowRef:
    """Окно, куда можно вставлять: hwnd + процесс + заголовок."""
    hwnd: int
    pid: int
    title: str
    process: str = ""
    ts: float = 0.0

    @property
    def label(self) -> str:
        """Короткое имя для логов/оверлея: «Telegram — чат» / «chrome.exe»."""
        text = (self.title or "").strip()
        if len(text) > 42:
            text = text[:41] + "…"
        return text or self.process or f"hwnd {self.hwnd}"

    def same_window(self, other: "WindowRef | None") -> bool:
        return bool(other) and self.hwnd == other.hwnd


def _process_name(pid: int) -> str:
    if not _IS_WINDOWS or not pid:
        return ""
    try:
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_ulong(len(buf))
            if ctypes.windll.kernel32.QueryFullProcessImageNameW(
                    h, 0, buf, ctypes.byref(size)):
                return Path(buf.value).name.lower()
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def window_info(hwnd: int) -> WindowRef | None:
    """Описание окна по hwnd (без проверки, что оно в фокусе)."""
    if not _IS_WINDOWS or not hwnd:
        return None
    try:
        if not ctypes.windll.user32.IsWindow(hwnd):
            return None
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1) if length else None
        title = ""
        if buf is not None:
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return WindowRef(hwnd=int(hwnd), pid=int(pid.value), title=title,
                         process=_process_name(int(pid.value)), ts=time.monotonic())
    except Exception:  # noqa: BLE001
        return None


def current_window() -> WindowRef | None:
    """Окно в фокусе прямо сейчас."""
    if not _IS_WINDOWS:
        return None
    try:
        hwnd = int(ctypes.windll.user32.GetForegroundWindow())
    except Exception:  # noqa: BLE001
        return None
    return window_info(hwnd)


def focus_window(hwnd: int) -> bool:
    """Вернуть фокус в окно. True, если окно стало активным.

    Windows не даёт чужому процессу просто «украсть» фокус
    (SetForegroundWindow молча отказывает). Рабочий приём: временно
    присоединить свою очередь ввода к потоку активного окна, переключить
    фокус и отсоединиться.
    """
    if not _IS_WINDOWS or not hwnd:
        return False
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    try:
        if not user32.IsWindow(hwnd):
            return False
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)

        fg = int(user32.GetForegroundWindow())
        if fg == hwnd:
            return True
        # Разрешаем любому процессу перевести нас на передний план.
        try:
            user32.AllowSetForegroundWindow(ASFW_ANY)
        except Exception:  # noqa: BLE001
            pass

        cur_thread = kernel32.GetCurrentThreadId()
        fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        target_thread = user32.GetWindowThreadProcessId(hwnd, None)
        attached = False
        if fg_thread and fg_thread != cur_thread:
            attached = bool(user32.AttachThreadInput(fg_thread, cur_thread, True))
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(fg_thread, cur_thread, False)

        # Даём оконному менеджеру мгновение переключиться.
        for _ in range(6):
            if int(user32.GetForegroundWindow()) == hwnd:
                return True
            time.sleep(0.02)
        return int(user32.GetForegroundWindow()) == hwnd
    except Exception:  # noqa: BLE001
        return False


class FocusHistory:
    """Кольцевой буфер замеров фокуса: «какое окно было активно тогда-то».

    Нужен для AUTO-режима: фраза начинается без нажатия клавиши, поэтому
    момент её старта известен только по длительности — и нужное окно
    находится в истории.
    """

    def __init__(self, window_seconds: float = 90.0, max_items: int = 300) -> None:
        self.window_seconds = window_seconds
        self._items: deque[WindowRef] = deque(maxlen=max_items)
        self._lock = threading.Lock()

    def record(self, ref: WindowRef | None = None) -> None:
        """Записать замер (по умолчанию — текущий фокус)."""
        if ref is None:
            ref = current_window()
        if ref is None:
            return
        with self._lock:
            self._items.append(ref)

    def at(self, when: float) -> WindowRef | None:
        """Окно, активное в момент `when` (monotonic-время)."""
        with self._lock:
            items = list(self._items)
        if not items:
            return None
        cutoff = when - 1.0     # замеры старше секунды от «тогда» не годятся
        best: WindowRef | None = None
        for ref in items:
            if ref.ts <= when + 0.6 and ref.ts >= cutoff:
                best = ref     # берём самый свежий, но не позже момента
        if best is not None:
            return best
        # Ничего рядом: вернём самый старый замер (фраза началась раньше истории).
        return items[0] if items[0].ts <= when else None

    def last(self) -> WindowRef | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def prune(self, now: float | None = None) -> None:
        """Выбросить замеры старше окна хранения."""
        now = time.monotonic() if now is None else now
        limit = now - self.window_seconds
        with self._lock:
            while self._items and self._items[0].ts < limit:
                self._items.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
