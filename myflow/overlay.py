"""MyFLOW — оверлей статуса (Ф6).

Прозрачное topmost-окно в правом нижнем углу: показывает текущее состояние
диктовки («🎤 запись», «⚙ обработка…», «✓ вставлено»), клики проходят сквозь.

Реализация без новых зависимостей: Tkinter (встроен в CPython).
Потокобезопасность: главный цикл Tk живёт в daemon-потоке, вызовы статусов
проходят через Queue — обновление в mainloop по poll-таймеру.
"""
from __future__ import annotations

import queue
import threading

# Прозрачный цвет фона (chroma-key на Windows) и цвет текста.
_TRANSPARENT = "#f0f0f0"    # совпадает с фоном окна -> «вырезается» Windows
_TEXT = "#e8e8e8"
_FONT = ("Segoe UI", 11, "bold")

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000


def _is_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


class StatusOverlay:
    """Живой оверлей; методы show/hide безопасны из любого потока."""

    def __init__(self, timeout_seconds: float = 1.8) -> None:
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._timeout = timeout_seconds
        self._thread: threading.Thread | None = None
        self._root = None
        self._label = None
        self._hide_job: str | None = None

    # ---- публичный API (можно из любого потока) ----

    def start(self) -> None:
        """Поднять поток Tk (daemon). Двойной вызов безопасен."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="status-overlay")
        self._thread.start()

    def show(self, text: str, color: str = _TEXT) -> None:
        """Показать статус; авто-скрытие по таймеру (сбрасывается при новых show)."""
        if self._thread is None or not self._thread.is_alive():
            return  # оверлей выключен флагом — не копим очередь
        self._q.put(("show", text, color))

    def hide(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._q.put(("hide",))

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._q.put(("stop",))

    # ---- поток Tk ----

    def _run(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)                 # без рамки/кнопок
        root.attributes("-topmost", True)
        # Важно: только transparentcolor, НЕ -alpha — на Windows они конфликтуют
        # (alpha делает весь прямоугольник полупрозрачным и гасит чб-вырезку).
        root.attributes("-transparentcolor", _TRANSPARENT)
        root.configure(bg=_TRANSPARENT)

        label = tk.Label(
            root, text="", font=_FONT, fg=_TEXT, bg=_TRANSPARENT,
            bd=0, padx=14, pady=8,
        )
        label.pack()
        root.update_idletasks()

        self._root, self._label = root, label
        self._apply_click_through()
        root.deiconify()
        self._position()

        def poll() -> None:
            try:
                while True:
                    msg = self._q.get_nowait()
                    if msg[0] == "show":
                        _, text, color = msg
                        self._show_now(text, color)
                    elif msg[0] == "hide":
                        self._hide_now()
                    elif msg[0] == "stop":
                        root.destroy()
                        return
            except queue.Empty:
                pass
            root.after(40, poll)

        root.after(40, poll)
        root.mainloop()

    def _apply_click_through(self) -> None:
        """Клики сквозь окно + не появляется в Alt+Tab (Win32 ex-style)."""
        import ctypes
        from ctypes import wintypes

        try:
            hwnd: wintypes.HWND = self._root.winfo_id()
            # Tk окно: winfo_id() даёт дочернее окно; ищем top-level владельца.
            toplevel = ctypes.windll.user32.GetAncestor(hwnd, 2)  # GA_ROOT
            style = ctypes.windll.user32.GetWindowLongW(toplevel, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                toplevel, GWL_EXSTYLE,
                style | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE,
            )
        except Exception:
            pass  # клик-сквозь не критичен для работоспособности

    def _position(self) -> None:
        """Правый нижний угол (над панелью задач)."""
        root = self._root
        root.update_idletasks()
        w = root.winfo_width() or 200
        h = root.winfo_height() or 40
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.geometry(f"+{screen_w - w - 24}+{screen_h - h - 90}")

    def _show_now(self, text: str, color: str) -> None:
        if self._hide_job:
            self._root.after_cancel(self._hide_job)
            self._hide_job = None
        self._label.configure(text=text, fg=color)
        self._root.deiconify()
        self._root.lift()
        self._root.attributes("-topmost", True)
        self._position()
        self._hide_job = self._root.after(int(self._timeout * 1000), self._hide_now)

    def _hide_now(self) -> None:
        if self._hide_job:
            try:
                self._root.after_cancel(self._hide_job)
            except Exception:
                pass
            self._hide_job = None
        self._root.withdraw()