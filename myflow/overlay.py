"""MyFLOW — статус-оверлей v2.

Небольшая тёмная плашка в правом нижнем углу рабочей области: «● запись»,
«⚙ распознавание», «✓ вставлено», «✗ ошибка». Оверлей информационный:
клики проходят сквозь, фокус не забирает, в Alt+Tab не виден.

Чем отличается от прежней версии (удалена как непригодная):
  * читаемость: тёмный фон + почти белый текст вместо chroma-key
    `#e8e8e8` на `#f0f0f0`, который сливался и с тёмными, и со светлыми
    обоями (старый вариант вообще нельзя было прочитать);
  * позиция: рабочая область (rcWork) монитора, на котором сейчас активное
    окно, а не «экран минус 90 пикселей» — корректно при taskbar сверху/слева
    и на мультимониторных конфигурациях;
  * клик-сквозь применяется с SetWindowPos(SWP_FRAMECHANGED) — без этого
    extended-style мог не примениться;
  * «залипающие» статусы: «● запись» и «⚙ обработка» не гаснут по таймеру
    (их сменит следующий статус), ошибка висит 3 с, успех — 1.4 с;
  * живучесть: если поток Tk умер, следующий show() поднимает его заново
    (раньше оверлей молча перестал бы показываться до перезапуска);
  * очередь ограничена: при всплеске статусов устаревшие отбрасываются,
    поэтому плашка не «догоняет» события с задержкой.

Потокобезопасность: show()/hide()/stop() можно звать из любого потока —
они только кладут сообщение в Queue; весь Tk живёт в одном выделенном
daemon-потоке (Tk не потокобезопасен, это единственная надёжная схема).
"""
from __future__ import annotations

import ctypes
import queue
import threading
from ctypes import wintypes  # обязательно явно: ctypes.wintypes не подгружается сам

# --- оформление -------------------------------------------------------------
_BG = "#14181d"          # почти чёрный фон плашки
_BORDER = "#2c343d"      # тонкая рамка
_TEXT = "#f2f5f7"        # основной текст
_FONT = ("Segoe UI", 11, "bold")
_MARGIN = 24             # отступ от края рабочей области
_ALPHA = 0.94
_MAX_QUEUE = 8           # при всплеске устаревшие статусы отбрасываем
_POLL_MS = 30

# Сколько держать статус по префиксу текста (сек). None = до следующего.
_TIMEOUTS: dict = {
    "●": None,
    "⚙": None,
    "✗": 3.0,
    "✓": 1.4,
    "¶": 1.2,
    "⎌": 1.4,
    "○": 1.8,
    "…": 1.2,
}
_DEFAULT_TIMEOUT = 1.8

# --- WinAPI -----------------------------------------------------------------
_GWL_EXSTYLE = -20
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000

_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SWP_FRAMECHANGED = 0x0020

_MONITOR_DEFAULTTONEAREST = 2


def _pick_timeout(text: str, default: float | None = _DEFAULT_TIMEOUT) -> float | None:
    """Сколько секунд держать статус (None — до следующего события).

    Правило по первому значимому символу: запись/обработка «залипают»,
    ошибка висит дольше успеха.
    """
    for ch in (text or "").strip():
        if ch in _TIMEOUTS:
            return _TIMEOUTS[ch]
        break
    return default


def _place(work: tuple, w: int, h: int, margin: int = _MARGIN) -> tuple:
    """Координаты правого нижнего угла внутри рабочей области (l, t, r, b)."""
    left, top, right, bottom = work
    x = max(left, right - w - margin)
    y = max(top, bottom - h - margin)
    return x, y


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", ctypes.c_ulong),
    ]


def _work_area():
    """Рабочая область монитора с активным окном (Windows).

    rcWork уже исключает панель задач, где бы она ни была.
    """
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        hmon = user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONEAREST)
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            r = info.rcWork
            if r.right > r.left and r.bottom > r.top:
                return (r.left, r.top, r.right, r.bottom)
    except Exception:  # noqa: BLE001 — не Windows/нет монитора: уйдём на фолбэк
        pass
    try:
        rect = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:  # noqa: BLE001
        pass
    return None


class StatusOverlay:
    """Плашка статуса. Все публичные методы безопасны из любого потока."""

    def __init__(self, timeout_seconds: float = _DEFAULT_TIMEOUT) -> None:
        self._default_timeout = timeout_seconds
        self._q: "queue.Queue" = queue.Queue(maxsize=_MAX_QUEUE)
        self._thread = None
        self._root = None
        self._label = None
        self._hide_job = None
        self._lock = threading.Lock()
        # Оверлей включается только явным start(): фичев-флаг overlay=False
        # не должен включаться сам от первого show().
        self._started = False

    # ---- публичный API ----

    def start(self) -> None:
        """Поднять поток Tk. Повторный вызов безопасен."""
        with self._lock:
            self._started = True
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="status-overlay")
            self._thread.start()

    def show(self, text: str, color: str = _TEXT, timeout: float | None = None) -> None:
        """Показать статус.

        timeout: None — по правилу `_pick_timeout` (●/⚙ залипают, ✗ висит 3 с).
        Отрицательный timeout — «висеть до следующего статуса».
        """
        if not self._started:
            return                 # оверлей выключен флагом — не копим и не стартуем
        if not self.is_running():
            self.start()           # поток Tk умер — поднимаем заново
        if timeout is None:
            timeout = _pick_timeout(text, self._default_timeout)
        self._put(("show", text, color, timeout))

    def hide(self) -> None:
        self._put(("hide",))

    def stop(self, timeout: float = 2.0) -> None:
        """Остановить оверлей и дождаться завершения потока Tk.

        Ждать обязательно: если интерпретатор завершится, пока Tk-объекты
        ещё живы, Tcl падает с «Tcl_AsyncDelete: async handler deleted by
        the wrong thread» (GC корневого окна из чужого потока) и процесс
        выходит с ненулевым кодом. Джойн переносит уничтожение Tk в его
        собственный поток.
        """
        self._put(("stop",))
        thread = self._thread
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def query(self, fn, timeout: float = 2.0):
        """Выполнить fn(root, label) В ПОТОКЕ Tk и вернуть результат.

        Нужно для диагностики/тестов: обращаться к Tk-объектам из чужого
        потока нельзя (Tcl не потокобезопасен, а на выходе это даёт
        «Tcl_AsyncDelete: async handler deleted by the wrong thread»).
        Вернёт None, если поток Tk не ответил за timeout.
        """
        if not self.is_running():
            return None
        box = {"done": threading.Event(), "value": None}
        def _call() -> None:
            try:
                box["value"] = fn(self._root, self._label)
            except Exception as exc:  # noqa: BLE001 — вернём исключение наружу
                box["value"] = exc
            finally:
                box["done"].set()

        self._put(("call", _call))
        if not box["done"].wait(timeout):
            return None
        return box["value"]

    # ---- внутреннее ----

    def _put(self, msg: tuple) -> None:
        """Положить сообщение, выбросив самое старое при переполнении."""
        try:
            self._q.put_nowait(msg)
            return
        except queue.Full:
            pass
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(msg)
        except queue.Full:
            pass

    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:  # noqa: BLE001 — нет Tk = нет оверлея
            print(f"[overlay] Tk недоступен: {exc!r}")
            return

        try:
            root = tk.Tk()
        except Exception as exc:  # noqa: BLE001
            print(f"[overlay] не удалось создать окно: {exc!r}")
            return

        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            # Полупрозрачность ВСЕГО окна: фон тёмный, отдельная chroma-key
            # прозрачность не нужна и не конфликтует с -alpha.
            root.attributes("-alpha", _ALPHA)
        except Exception:  # noqa: BLE001
            pass
        root.configure(bg=_BG)

        frame = tk.Frame(root, bg=_BG, highlightbackground=_BORDER,
                         highlightcolor=_BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True)
        label = tk.Label(frame, text="", font=_FONT, fg=_TEXT, bg=_BG,
                         bd=0, padx=16, pady=9, justify="left")
        label.pack()

        self._root, self._label = root, label
        root.update_idletasks()
        self._apply_click_through()
        self._position()
        root.deiconify()

        def poll() -> None:
            stop = False
            while True:
                try:
                    msg = self._q.get_nowait()
                except queue.Empty:
                    break
                kind = msg[0]
                if kind == "show":
                    _, text, color, timeout = msg
                    self._show_now(text, color, timeout)
                elif kind == "hide":
                    self._hide_now()
                elif kind == "call":
                    msg[1]()          # диагностический вызов в потоке Tk
                elif kind == "stop":
                    stop = True
                    break
            if stop:
                try:
                    root.quit()
                    root.destroy()
                except Exception:  # noqa: BLE001
                    pass
                # Сбрасываем ссылку tkinter на корень ИЗ ЭТОГО ЖЕ потока:
                # иначе GC корня произойдёт в главном потоке и Tcl упадёт
                # («async handler deleted by the wrong thread»).
                try:
                    tk._default_root = None
                except Exception:  # noqa: BLE001
                    pass
                return
            try:
                root.after(_POLL_MS, poll)
            except Exception:  # noqa: BLE001 — окно уже уничтожено
                pass

        try:
            root.after(_POLL_MS, poll)
            root.mainloop()
        except Exception as exc:  # noqa: BLE001
            print(f"[overlay] цикл Tk упал: {exc!r}")
        finally:
            # Локальные ссылки на Tk освобождаем в потоке Tk, а не в чужом.
            self._root = self._label = None
            del root, frame, label, poll
            # В Tcl остаются зарегистрированные Python-колбэки (after/команды),
            # они ссылаются на Tkapp -> цикл. Если его соберёт главный поток,
            # Tcl падает с «Tcl_AsyncDelete: async handler deleted by the wrong
            # thread». Поэтому собираем цикл здесь и сейчас.
            try:
                import gc
                gc.collect()
            except Exception:  # noqa: BLE001
                pass

    # ---- работа с окном (только поток Tk) ----

    def _show_now(self, text: str, color: str, timeout) -> None:
        root = self._root
        if root is None:
            return
        if self._hide_job is not None:
            try:
                root.after_cancel(self._hide_job)
            except Exception:  # noqa: BLE001
                pass
            self._hide_job = None
        self._label.configure(text=text, fg=color or _TEXT)
        self._position()          # активное окно могло переехать на другой монитор
        root.deiconify()
        root.lift()
        try:
            root.attributes("-topmost", True)
        except Exception:  # noqa: BLE001
            pass
        if timeout is not None and timeout >= 0:
            self._hide_job = root.after(int(timeout * 1000), self._hide_now)

    def _hide_now(self) -> None:
        root = self._root
        if root is None:
            return
        self._hide_job = None
        try:
            root.withdraw()
        except Exception:  # noqa: BLE001
            pass

    def _position(self) -> None:
        """Правый нижний угол рабочей области активного монитора."""
        root = self._root
        if root is None:
            return
        try:
            root.update_idletasks()
            w = root.winfo_reqwidth()
            h = root.winfo_reqheight()
            work = _work_area() or (0, 0, root.winfo_screenwidth(), root.winfo_screenheight())
            x, y = _place(work, w, h, _MARGIN)
            root.geometry(f"+{x}+{y}")
        except Exception:  # noqa: BLE001 — позиция не критична
            pass

    def _apply_click_through(self) -> None:
        """Клики сквозь, без активации и без записи в Alt+Tab."""
        root = self._root
        if root is None:
            return
        try:
            user32 = ctypes.windll.user32
            hwnd = root.winfo_id()
            toplevel = user32.GetAncestor(hwnd, 2)  # GA_ROOT
            style = user32.GetWindowLongW(toplevel, _GWL_EXSTYLE)
            user32.SetWindowLongW(
                toplevel, _GWL_EXSTYLE,
                style | _WS_EX_TOOLWINDOW | _WS_EX_TRANSPARENT
                | _WS_EX_LAYERED | _WS_EX_NOACTIVATE,
            )
            # Без SetWindowPos(SWP_FRAMECHANGED) новый ex-style может не примениться.
            user32.SetWindowPos(
                toplevel, 0, 0, 0, 0, 0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER
                | _SWP_NOACTIVATE | _SWP_FRAMECHANGED,
            )
        except Exception:  # noqa: BLE001 — клик-сквозь не критичен для работы
            pass
