"""MyFLOW — одноконсольный TUI: серверы, настройки и диктовка в ОДНОМ окне.

Раньше `start-myflow.bat` открывал три окна: пульт серверов (start-all.ps1),
окно main и своё собственное. Теперь всё живёт в одном окне:

    1. «Запуск»      — поднимает whisper-server и llama-server скрытыми
                       дочерними процессами (логи в data\\*.log), показывает
                       прогресс и статус каждого;
    2. «Настройки»   — один экран с тремя блоками (микрофон / режим / вставка),
                       ↑/↓ — выбор внутри блока, ←/→ — переход между блоками,
                       Enter — применить и стартовать, Esc — выход;
    3. «Пульт»       — живой журнал диктовки, состояние ASR/LLM, текущий статус
                       (запись/распознавание/полировка/вставка) и подсказки
                       по хоткеям; q — выход, r — вернуться в настройки,
                       s — перезапустить упавший сервер, c — очистить журнал.

Всё без внешних зависимостей: ANSI/VT через WinAPI, ввод через msvcrt.
Кириллица и рамки — UTF-8 (SetConsoleOutputCP(65001)).

Запуск:  .venv\\Scripts\\python -m myflow.tui
Флаги:   --no-servers (не поднимать серверы), --demo (витрина для скриншота),
         --screen panel|dashboard|settings, --hold N (секунд держать кадр).
"""
from __future__ import annotations

import argparse
import ctypes
import io
import msvcrt
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------- ANSI/цвета
CSI = "\x1b["
RESET = CSI + "0m"
BOLD = CSI + "1m"


def fg(n: int) -> str:
    return f"{CSI}38;5;{n}m"


C_TEXT = fg(252)
C_DIM = fg(244)
C_ACCENT = fg(45)
C_INV = CSI + "7m"       # инверсия — для главной «кнопки» и выделения
C_BORDER_DIM = fg(238)   # рамка кадра: тише, чем содержимое
C_KEY = fg(250)          # цвет скобок вокруг клавиш
C_OK = fg(114)
C_WARN = fg(214)
C_ERR = fg(203)
C_BORDER = fg(238)
C_TITLE = fg(81)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Многоколоночные эмодзи из логов ломают выравнивание рамки — заменяем на ASCII.
_GLYPHS = {
    "●": "*", "○": "-", "⚙": "~", "✓": "+", "✗": "!", "…": "...",
    "¶": "P", "⎌": "U", "↻": "R", "⌘": ">", "·": "-", "→": "->",
    "🎤": "mic", "«": '"', "»": '"',
}


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s or "")


def vlen(s: str) -> int:
    return len(strip_ansi(s or ""))


def norm_text(s: str) -> str:
    """Убрать ANSI и заменить широкие глифы: рамка не поедет."""
    out = strip_ansi(s or "")
    for src, dst in _GLYPHS.items():
        out = out.replace(src, dst)
    return out.replace("\t", "    ").replace("\r", "")


def pad_to(s: str, width: int) -> str:
    """Дополнить/обрезать строку до ровно width видимых символов."""
    n = vlen(s)
    if n > width:
        return strip_ansi(s)[:width]
    return s + " " * (width - n)


def outer_frame(title: str, body: Sequence[str], w: int, h: int) -> list[str]:
    """Двойная рамка на весь экран с заголовком (ровно h строк по w символов)."""
    w = max(40, w)
    h = max(8, h)
    label = f" {title} " if title else ""
    inner = w - 2
    # Рамка приглушённая, заголовок — акцентный: контент читается легче.
    frame_c = C_BORDER_DIM
    top = (f"{frame_c}╔═{RESET}{BOLD}{C_TITLE}{label}{RESET}"
           f"{frame_c}" + "═" * max(0, inner - 1 - len(label)) + f"╗{RESET}")
    bottom = f"{frame_c}╚" + "═" * inner + f"╝{RESET}"
    lines = [top]
    room = h - 2
    body = list(body)[:room]
    body += [""] * (room - len(body))
    for ln in body:
        lines.append(f"{frame_c}║{RESET}" + pad_to(ln, inner) + f"{frame_c}║{RESET}")
    lines.append(bottom)
    return lines


def block(title: str, rows: Sequence[str], w: int, focused: bool = False) -> list[str]:
    """Блок с одинарной рамкой."""
    w = max(16, w)
    inner = w - 2
    color = C_ACCENT if focused else C_BORDER
    label = f" {norm_text(title)} "
    if len(label) > inner - 2:
        label = label[: max(0, inner - 2)]
    dashes = "─" * max(0, inner - 1 - len(label))
    out = [f"{color}┌─{RESET}{BOLD if focused else ''}{label}{RESET}{color}{dashes}┐{RESET}"]
    for r in rows:
        out.append(f"{color}│{RESET}" + pad_to(" " + r, inner) + f"{color}│{RESET}")
    out.append(f"{color}└" + "─" * inner + f"┘{RESET}")
    return out


def status_color(state: str) -> str:
    return {"ok": C_OK, "external": C_OK, "starting": C_WARN,
            "down": C_DIM, "error": C_ERR}.get(state, C_DIM)


# main.py передаёт цвета в HEX (#f5c542 и т.п.) — переводим в ANSI 256.
_HEX_MAP = {
    "#f5c542": C_WARN, "#46a758": C_OK, "#e5484d": C_ERR,
    "#9aa0a6": C_DIM, "#f2f5f7": C_TEXT, "#e8e8e8": C_TEXT,
}


def hex_to_ansi(color: str) -> str:
    """HEX -> ближайший цвет xterm-256 (truecolor в старом conhost ненадёжен)."""
    c = (color or "").strip().lower()
    if c in _HEX_MAP:
        return _HEX_MAP[c]
    if len(c) == 7 and c.startswith("#"):
        try:
            r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        except ValueError:
            return C_TEXT
        q = lambda v: max(0, min(5, round(v / 51)))  # noqa: E731
        return f"{CSI}38;5;{16 + 36 * q(r) + 6 * q(g) + q(b)}m"
    return C_TEXT


def status_label(state: str) -> str:
    return {"ok": "работает", "external": "работает (внешний)", "starting": "запускается",
            "down": "не запущен", "error": "ОШИБКА"}.get(state, state)


# ---- разбор клавиш ----
# msvcrt отдаёт «расширенные» клавиши как префикс + скан-код. ВАЖНО: у стрелок
# префикс \xe0, у функциональных клавиш — \x00; раньше обе группы шли через одну
# таблицу {H,P,K,M}, поэтому F5 (\x00 + «?») не распознавался и «пересканировать
# микрофоны» не срабатывало вообще.
_EXT_KEYS: dict[int, dict[bytes, str]] = {
    0x00: {   # \x00 — функциональные клавиши
        b";": "f1", b"<": "f2", b"=": "f3", b">": "f4", b"?": "f5",
        b"@": "f6", b"A": "f7", b"B": "f8", b"C": "f9", b"D": "f10",
        b"W": "f11", b"X": "f12", b"S": "delete", b"G": "home", b"O": "end",
    },
    0xE0: {   # \xe0 — стрелки и навигация
        b"H": "up", b"P": "down", b"K": "left", b"M": "right",
        b"S": "delete", b"G": "home", b"O": "end",
    },
}

# Windows Terminal (VT-ввод) может прислать F5 как ESC-последовательность \x1b[15~
_VT_KEYS: dict[str, str] = {
    "A": "up", "B": "down", "C": "right", "D": "left",
    "H": "home", "F": "end", "2~": "insert", "3~": "delete",
    "5~": "pageup", "6~": "pagedown",
    "11~": "f1", "12~": "f2", "13~": "f3", "14~": "f4", "15~": "f5",
    "17~": "f6", "18~": "f7", "19~": "f8", "20~": "f9", "21~": "f10",
    "23~": "f11", "24~": "f12",
}


def map_extended_key(prefix: int, code: bytes) -> str | None:
    """Скан-код расширенной клавиши -> имя ('f5', 'up', …). Чистая функция."""
    return _EXT_KEYS.get(prefix, {}).get(code)


_MOUSE_SGR = re.compile(r"\[?<(\d+);(\d+);(\d+)([Mm])")


def map_mouse_sequence(seq: str) -> str | None:
    """Разобрать SGR-событие мыши: `[<64;120;34M`.

    Код 64/65 — колесо вверх/вниз, остальные события (клики, движение) сейчас
    не используются, но распознаются, чтобы не ломать разбор ESC-последовательности.
    Возвращает `wheelup:X,Y` / `wheeldown:X,Y` / `click:X,Y` / None.
    """
    m = _MOUSE_SGR.search(seq or "")
    if not m:
        return None
    code, x, y, kind = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    if code & 64:                                   # колесо
        direction = "down" if code & 1 else "up"
        return f"wheel{direction}:{x},{y}"
    if code & 32:                                   # движение с кнопкой
        return f"drag:{x},{y}"
    return f"click:{x},{y}" if kind == "M" else f"release:{x},{y}"


def map_vt_sequence(seq: str) -> str | None:
    """Хвост ESC-последовательности ('15~', '[15~', 'A') -> имя клавиши."""
    text = seq.strip().lstrip("[")
    return _VT_KEYS.get(text)


def fit_width(line: str, width: int) -> str:
    """Обрезать строку до видимой ширины, не ломая ANSI-последовательности.

    Нужно как страховка: если строка окажется шире окна, терминал перенесёт её
    на следующую строку и кадр «поедет» (именно так ломалась отрисовка, когда
    ширина считалась по буферу консоли, а не по видимому окну).
    """
    if width <= 0 or vlen(line) <= width:
        return line
    out: list[str] = []
    visible = 0
    i = 0
    while i < len(line) and visible < width:
        ch = line[i]
        if ch == "\x1b":
            m = re.match(r"\x1b\[[0-9;?]*[A-Za-z]", line[i:])
            if m:
                out.append(m.group(0))
                i += len(m.group(0))
                continue
        out.append(ch)
        visible += 1
        i += 1
    return "".join(out) + RESET


def button_bar(buttons: Sequence[tuple[str, str]], width: int,
               primary: str | None = None) -> list[str]:
    """Полоска «кнопок»: [F5] пересканировать  [Enter] запустить …

    buttons — пары (клавиша, подпись). primary — клавиша главного действия,
    её подсвечиваем инверсией. Подписи не выбрасываем: если не влезает в одну
    строку — переносим на следующую (раньше хвост молча обрезался вместе с
    подписями, из-за чего казалось, что «кнопок нет»).
    """
    def render(key: str, label: str) -> str:
        if primary and key == primary:
            return f"{C_INV}{BOLD} {key} {label} {RESET}"
        return (f"{C_KEY}[{RESET}{C_ACCENT}{key}{RESET}{C_KEY}]{RESET}"
                f"{C_DIM} {label} {RESET}")

    lines: list[str] = []
    current = ""
    for key, label in buttons:
        chunk = render(key, label)
        sep = "  " if current else ""
        if current and vlen(strip_ansi(current + sep + chunk)) > width:
            lines.append(current)
            current = chunk
        else:
            current += sep + chunk
    if current:
        lines.append(current)
    return lines or [""]


# ----------------------------------------------------------------- консоль
class Console:
    """Низкоуровневый вывод кадра: VT-режим, размер, курсор."""

    def __init__(self) -> None:
        self.out = sys.__stdout__ or sys.stdout
        self.vt = self._enable_vt()
        # Последний нарисованный кадр: пишем в консоль только изменившиеся строки
        # (полный кадр — ~8.5 КБ ANSI; при 20 кадрах/с это 170 КБ/с впустую).
        self._last_frame: list[str] = []
        self.mouse = False
        self._locked_raw: tuple[int, int] | None = None
        try:
            self.out.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _enable_vt() -> bool:
        try:
            k = ctypes.windll.kernel32
            handle = k.GetStdHandle(-11)                      # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            if not k.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            # ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            k.SetConsoleMode(handle, mode.value | 0x0001 | 0x0004)
            k.SetConsoleOutputCP(65001)
            k.SetConsoleCP(65001)
            return True
        except Exception:  # noqa: BLE001
            return False

    def raw_size(self) -> tuple[int, int]:
        """Истинный размер видимого окна (без вычета строки под легенду)."""
        try:
            class _Coord(ctypes.Structure):
                _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

            class _SmallRect(ctypes.Structure):
                _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short),
                            ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]

            class _Csbi(ctypes.Structure):
                _fields_ = [("dwSize", _Coord), ("dwCursorPosition", _Coord),
                            ("wAttributes", ctypes.c_ushort), ("srWindow", _SmallRect),
                            ("dwMaximumWindowSize", _Coord)]

            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            info = _Csbi()
            if ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
                width = info.srWindow.Right - info.srWindow.Left + 1
                height = info.srWindow.Bottom - info.srWindow.Top + 1
                if width > 0 and height > 0:
                    return int(width), int(height)
        except Exception:  # noqa: BLE001
            pass
        try:
            import shutil as _sh
            ts = _sh.get_terminal_size()
            return int(ts.columns), int(ts.lines)
        except Exception:  # noqa: BLE001
            return 120, 35

    def lock_size(self, cols: int = 0, rows: int = 0) -> tuple[int, int]:
        """Зафиксировать размер окна и убрать прокрутку буфера.

        Зачем: колесо мыши прокручивало буфер консоли, и весь пульт «уезжал
        вверх». Делаем окно ровно нужного размера и буфер — ровно под окно
        (без истории прокрутки), поэтому прокручивать нечего.

        `cols`/`rows` — желаемый ВИДИМЫЙ размер (0 = оставить как есть).
        Возвращает фактический размер окна (ширина, высота буфера-окна).
        """
        want_w = int(cols or 0)
        want_h = int(rows or 0)
        try:
            k = ctypes.windll.kernel32
            handle = k.GetStdHandle(-11)

            class _Coord(ctypes.Structure):
                _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

            class _SmallRect(ctypes.Structure):
                _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short),
                            ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]

            cur_w, cur_h = self.raw_size()
            want_w = max(60, want_w or cur_w)
            want_h = max(20, want_h or cur_h)
            if (want_w, want_h) == (cur_w, cur_h):
                self._locked_raw = (want_w, want_h)
                return want_w, want_h

            def set_buffer(w: int, h: int) -> bool:
                return bool(k.SetConsoleScreenBufferSize(handle, _Coord(w, h)))

            def set_window(w: int, h: int) -> bool:
                rect = _SmallRect(0, 0, w - 1, h - 1)
                return bool(k.SetConsoleWindowInfo(handle, True, ctypes.byref(rect)))

            if want_w > cur_w or want_h > cur_h:
                set_buffer(max(want_w, cur_w), max(want_h, cur_h))   # растим буфер
                set_window(want_w, want_h)
            else:
                set_window(want_w, want_h)                           # сперва окно
            set_buffer(want_w, want_h)         # буфер ровно под окно: прокрутки нет
        except Exception:  # noqa: BLE001 — не Windows/нет консоли
            pass
        raw = self.raw_size()
        self._locked_raw = raw
        return raw

    def enable_mouse(self) -> None:
        """Перехватить колесо мыши: события приходят нам, окно не прокручивается."""
        if not self.vt:
            return
        try:
            k = ctypes.windll.kernel32
            handle = k.GetStdHandle(-10)                      # STD_INPUT_HANDLE
            mode = ctypes.c_ulong()
            if k.GetConsoleMode(handle, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
                k.SetConsoleMode(handle, mode.value | 0x0200)
        except Exception:  # noqa: BLE001
            pass
        # 1000 — обычные события мыши, 1006 — расширенный (SGR) формат координат
        self.out.write(CSI + "?1000h" + CSI + "?1006h")
        self.out.flush()
        self.mouse = True

    def disable_mouse(self) -> None:
        """Вернуть мышь терминалу (иначе после выхода колесо «залипает»)."""
        if not getattr(self, "mouse", False):
            return
        try:
            self.out.write(CSI + "?1006l" + CSI + "?1000l")
            self.out.flush()
        except Exception:  # noqa: BLE001
            pass
        self.mouse = False

    def size(self) -> tuple[int, int]:
        """Размер ВИДИМОГО окна консоли (не буфера!).

        ВАЖНО: `shutil.get_terminal_size()`/`os.get_terminal_size()` на Windows
        отдают размер буфера (`dwSize`), а он часто ШИРЕ окна (у пользователя
        буфер был ~230 колонок при окне ~120). Кадр рисовался на 230 символов,
        терминал переносил строки — отсюда каша и «два пульта» на экране.
        Берём `srWindow` из GetConsoleScreenBufferInfo.
        """
        try:
            class _Coord(ctypes.Structure):
                _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

            class _SmallRect(ctypes.Structure):
                _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short),
                            ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]

            class _Csbi(ctypes.Structure):
                _fields_ = [("dwSize", _Coord), ("dwCursorPosition", _Coord),
                            ("wAttributes", ctypes.c_ushort), ("srWindow", _SmallRect),
                            ("dwMaximumWindowSize", _Coord)]

            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            info = _Csbi()
            if ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle,
                                                                 ctypes.byref(info)):
                width = info.srWindow.Right - info.srWindow.Left + 1
                height = info.srWindow.Bottom - info.srWindow.Top + 1
                if width > 0 and height > 0:
                    return max(60, int(width)), max(18, int(height) - 1)
        except Exception:  # noqa: BLE001 — не Windows/нет консоли: падаем на fallback
            pass
        try:
            sz = shutil.get_terminal_size((100, 30))
            return max(60, int(sz.columns)), max(18, int(sz.lines) - 1)
        except Exception:  # noqa: BLE001
            return 100, 29

    def cursor(self, visible: bool) -> None:
        if not self.vt:
            return
        self.out.write(CSI + ("?25h" if visible else "?25l"))
        self.out.flush()

    def draw(self, lines: Sequence[str]) -> bool:
        """Нарисовать кадр. True — в консоль что-то записали.

        Пишем ТОЛЬКО изменившиеся строки: журнал добавляет по строке раз в
        несколько секунд, а панель перерисовывалась целиком 20 раз в секунду.
        Если кадр идентичен предыдущему — не пишем вообще (0 байт).
        """
        if not self.vt:
            self.out.write("\n".join(strip_ansi(x) for x in lines) + "\n")
            self.out.flush()
            return True
        # Защита от «поехавшего» кадра: строка шире окна заставляет терминал
        # переносить её (и затирать следующие), а лишние строки прокручивают
        # экран. Поэтому жёстко обрезаем по видимому размеру окна.
        w, h = self.size()
        visible = [fit_width(ln, w) for ln in list(lines)[:h]]
        prev = getattr(self, "_last_frame", [])
        if visible == prev:
            return False
        buf: list[str] = []
        for i, ln in enumerate(visible):
            if i < len(prev) and prev[i] == ln:
                continue
            buf.append(f"{CSI}{i + 1};1H{ln}{CSI}K")
        for i in range(len(visible), len(prev)):       # кадр стал короче
            buf.append(f"{CSI}{i + 1};1H{CSI}K")
        if buf:
            self.out.write("".join(buf))
            self.out.flush()
        self._last_frame = visible
        return True

    def clear(self) -> None:
        if self.vt:
            self.out.write(CSI + "2J" + CSI + "H")
        self.out.flush()
        self._last_frame = []          # после очистки кадр рисуем целиком

    def reset(self) -> None:
        self.out.write(RESET + CSI + "?25h")
        self.out.flush()


# ----------------------------------------------------------------- журнал
class PaneSink(io.TextIOBase):
    """sys.stdout приложения -> журнал TUI (+ файл data/main.log).

    В консоль НЕ пишем: кадр рисует сам TUI, любой посторонний вывод его сломает.
    """

    def __init__(self, pane: "TuiState", log_file: Path | None, max_lines: int = 400) -> None:
        super().__init__()
        self.pane = pane
        self.max_lines = max_lines
        self._buf = ""
        self._lock = threading.Lock()
        self._fh = None
        if log_file is not None:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(log_file, "a", encoding="utf-8", errors="replace")
                self._fh.write(f"\n=== TUI-сессия {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                self._fh.flush()
            except Exception:  # noqa: BLE001
                self._fh = None

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.write(s)
                    self._fh.flush()
                except Exception:  # noqa: BLE001
                    pass
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self.pane.add_log(line)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            if self._buf.strip():
                self.pane.add_log(self._buf)
                self._buf = ""

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        """Закрыть файл лога (иначе Windows не даст удалить/переместить его)."""
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:  # noqa: BLE001
                    pass
                self._fh = None


# ----------------------------------------------------------------- состояние
@dataclass
class TuiState:
    """То, что читает рендер (обновляется из разных потоков — под локом)."""

    status_text: str = "ожидание"
    status_color: str = C_DIM
    log: list[str] = field(default_factory=list)
    servers: dict = field(default_factory=dict)   # key -> dict(state, note, ms)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float = field(default_factory=time.monotonic)
    phrases: int = 0
    last_e2e_ms: int = 0
    # Прокрутка журнала: 0 — «живой» режим (видно последние строки),
    # N — на сколько строк отступили вверх от конца. Новые строки не сбивают
    # позицию: пользователь может спокойно читать историю.
    scroll: int = 0

    def scroll_by(self, delta: int, page: int, total: int) -> None:
        """Прокрутить журнал: delta — строки (или страница), total — длина лога."""
        with self.lock:
            max_offset = max(0, total - max(1, page))
            self.scroll = max(0, min(max_offset, self.scroll + delta))

    def scroll_to(self, offset: int, page: int, total: int) -> None:
        with self.lock:
            max_offset = max(0, total - max(1, page))
            self.scroll = max(0, min(max_offset, offset))

    def add_log(self, line: str) -> None:
        text = norm_text(line).rstrip()
        if not text:
            return
        m = re.search(r"e2e=(\d+)", text)
        with self.lock:
            if m:
                self.last_e2e_ms = int(m.group(1))
            if "вставлено" in text:
                self.phrases += 1
            self.log.append(f"{time.strftime('%H:%M:%S')}  {text}")
            if len(self.log) > 400:
                del self.log[: len(self.log) - 400]

    def set_status(self, text: str, color: str) -> None:
        with self.lock:
            self.status_text, self.status_color = text, color

    def snapshot(self):
        with self.lock:
            return (self.status_text, self.status_color, list(self.log), dict(self.servers))


# ----------------------------------------------------------------- серверы
@dataclass
class ServerSpec:
    key: str
    title: str
    port: int
    url: str
    health_path: str
    exe: Path
    args: list[str]
    model: Path
    log: Path
    required: bool = True

    def missing(self) -> str | None:
        if not self.exe.exists():
            return f"нет файла {self.exe}"
        if not self.model.exists():
            return f"нет модели {self.model} (задай MYFLOW_MODELS_DIR)"
        return None


def models_dir() -> Path:
    """Каталог моделей: рядом с проектом (<repo>/models), см. myflow.config."""
    from myflow.config import models_dir as _models_dir
    return _models_dir()


def llama_dir() -> Path:
    """Каталог llama.cpp: рядом с проектом (<repo>/bin/llama-cpp)."""
    from myflow.config import llama_dir as _llama_dir
    return _llama_dir()


def default_specs(root: Path = ROOT) -> list[ServerSpec]:
    """Флаги серверов. Тюнинг 10.10.2026 (замеры: полировка p50 205 мс вместо 260-330,
    ASR точнее на beam=5, VRAM 7.2/10 ГБ):

    llama: -np 1 -kvu        один слот: контекст целиком нашему запросу, кэш
                             промпта всегда попадает (было 4 слота впустую);
           -ngl 99           все 36 слоёв на GPU — при -ngl 35 последний слой
                             с выходной проекцией (389 МБ) считался на CPU и
                             ронял генерацию до 80 tok/s (стало 105-115);
           -ctk/-ctv q8_0    KV-кэш вдвое компактнее — освобождает место под -ngl 99;
           -fa on            flash attention явно (было auto);
           -c 8192           ~2200 токенов системного промпта + длинная диктовка.
    whisper: -l ru           без авто-детекта (был случай распознавания русской
                             фразы как французской); -bs 5 beam search на запас
                             скорости (RTF 0.02); -sns против не-речевых токенов.
    """
    models = models_dir()
    from myflow.config import whisper_dir
    whisper = whisper_dir() / "Release" / "whisper-server.exe"
    from myflow.asr import whisper_needed      # локально: без циклических импортов
    # Модель ASR можно подменить через MYFLOW_WHISPER_MODEL (например, на
    # ggml-large-v3-q5_0.bin для «режима качества»: медленнее в ~4 раза, по
    # замерам на своих записях выигрыша по WER не даёт — см. README).
    from myflow.config import whisper_model_path
    whisper_model = whisper_model_path()
    specs: list[ServerSpec] = []
    # whisper-сервер поднимаем ТОЛЬКО если он реально нужен: при бэкенде GigaAM
    # (локальная ONNX-модель на CPU) он занимал бы ~1 ГБ VRAM впустую.
    if whisper_needed():
        specs.append(ServerSpec(
            key="asr", title="ASR whisper.cpp", port=8081,
            url="http://127.0.0.1:8081", health_path="/health",
            exe=whisper, model=whisper_model,
            args=["--host", "127.0.0.1", "--port", "8081", "-l", "ru",
                  "-bs", "5", "-sns", "-t", "4"],
            log=root / "data" / "asr.log", required=True,
        ))
    specs.append(ServerSpec(
        key="llm", title="LLM Qwen3-4B (полировка)", port=8082,
        url="http://127.0.0.1:8082", health_path="/v1/models",
        exe=llama_dir() / "llama-server.exe",
        model=models / "Qwen3-4B-Instruct-2507-Q8_0.gguf",
        args=["--host", "127.0.0.1", "--port", "8082", "-c", "8192",
              "-np", "1", "-kvu", "-ngl", "99", "-fa", "on",
              "-ctk", "q8_0", "-ctv", "q8_0", "--jinja"],
        log=root / "data" / "llm.log", required=False,
    ))
    return specs


class Supervisor:
    """Запуск/остановка локальных серверов скрытыми процессами."""

    CREATE_NO_WINDOW = 0x08000000

    def __init__(self, specs: Sequence[ServerSpec], *, enabled: bool = True) -> None:
        self.specs = list(specs)
        self.enabled = enabled
        self._procs: dict[str, subprocess.Popen] = {}
        self._own: set[str] = set()
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()
        # Job Object: серверы умрут вместе с процессом MyFLOW (даже при kill -9).
        from myflow.jobobj import ProcessJob
        self.job = ProcessJob()

    # ---- проверки ----
    @staticmethod

    def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
        try:
            with socket.socket() as s:
                s.settimeout(timeout)
                return s.connect_ex((host, port)) == 0
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def health(url: str, path: str, timeout: float = 1.5) -> bool:
        try:
            import requests
            return bool(requests.get(url + path, timeout=timeout).ok)
        except Exception:  # noqa: BLE001
            return False

    def status(self, spec: ServerSpec) -> tuple[str, str]:
        """(state, note) — для отрисовки."""
        if spec.key in self._errors:
            return "error", self._errors[spec.key]
        if self.enabled and spec.key in self._own:
            proc = self._procs.get(spec.key)
            if proc is not None and proc.poll() is not None:
                note = f"процесс завершился (код {proc.returncode}); лог: {spec.log.name}"
                return "error", note
        if self.health(spec.url, spec.health_path):
            return ("ok", "наш процесс") if spec.key in self._own else ("external", "был запущен раньше")
        if self.port_open(spec.port):
            return "error", f"порт {spec.port} занят, но health не отвечает"
        if self.enabled and spec.key in self._own:
            return "starting", "загружает модель..."
        if not self.enabled:
            return "down", "запуск серверов отключён (--no-servers)"
        return "down", "не запущен"

    def health_snapshot(self) -> dict:
        out = {}
        for spec in self.specs:
            state, note = self.status(spec)
            out[spec.key] = {"title": spec.title, "state": state, "note": note}
        return out

    # ---- запуск/останов ----
    def start(self, spec: ServerSpec) -> bool:
        """Поднять сервер, если порт свободен. True — процесс наш (или уже занят)."""
        if not self.enabled:
            return False
        missing = spec.missing()
        if missing:
            self._errors[spec.key] = missing
            return False
        if self.port_open(spec.port):
            return True                      # кто-то уже слушает — не трогаем
        try:
            spec.log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(spec.log, "ab")
            try:
                proc = subprocess.Popen(
                    [str(spec.exe), "-m", str(spec.model), *spec.args],
                    stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                    cwd=str(ROOT), creationflags=self.CREATE_NO_WINDOW,
                )
            finally:
                fh.close()
            # Привязываем сервер к Job Object: даже если MyFLOW убьют силой,
            # Windows сама снимет сервер (память и VRAM освободятся).
            self.job.assign(proc)
            with self._lock:
                self._procs[spec.key] = proc
                self._own.add(spec.key)
            self._errors.pop(spec.key, None)
            return True
        except Exception as exc:  # noqa: BLE001
            self._errors[spec.key] = f"не удалось запустить: {exc!r}"
            return False

    def wait_ready(self, spec: ServerSpec, timeout: float,
                   tick: Callable[[], None] | None = None) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.health(spec.url, spec.health_path):
                return True
            proc = self._procs.get(spec.key)
            if proc is not None and proc.poll() is not None:
                self._errors[spec.key] = (f"процесс завершился (код {proc.returncode}); "
                                          f"лог: {spec.log.name}")
                return False
            if tick:
                tick()
            time.sleep(0.4)
        self._errors[spec.key] = f"не ответил за {int(timeout)} с (лог: {spec.log.name})"
        return False

    def stop(self, spec: ServerSpec) -> bool:
        """Погасить ОДИН свой сервер (чужой не трогаем). True — процесс был наш."""
        with self._lock:
            proc = self._procs.pop(spec.key, None)
            own = spec.key in self._own
            self._own.discard(spec.key)
        if proc is None or not own:
            return False
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    proc.kill()
        except Exception:  # noqa: BLE001
            return False
        return True

    def stop_all(self) -> None:
        """Гасим только СВОИ процессы: чужие экземпляры на портах не трогаем."""
        with self._lock:
            procs = list(self._procs.items())
            self._procs.clear()
            self._own.clear()
        for key, proc in procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        for key, proc in procs:
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass


# ----------------------------------------------------------------- настройки
@dataclass
class Choice:
    key: str
    title: str
    options: list[tuple[str, object]]
    index: int = 0
    note: Callable[[int], str] | None = None


@dataclass
class SettingsState:
    choices: list[Choice]
    focus: int = 0

    def move_item(self, delta: int) -> None:
        ch = self.choices[self.focus]
        if not ch.options:
            return
        ch.index = (ch.index + delta) % len(ch.options)

    def move_block(self, delta: int) -> None:
        self.focus = (self.focus + delta) % len(self.choices)

    def values(self) -> dict:
        return {ch.key: ch.options[ch.index][1] for ch in self.choices if ch.options}


class LevelMeter:
    """Живой уровень микрофона: открывает поток и держит текущий пик.

    Зачем: «жив/мёртв» не отвечает на вопрос «хватит ли громкости». В рантайме
    фраза отсекается при пике < 0.01, поэтому в настройках показываем уровень
    и метку порога — видно, проходит ли голос и какой драйвер (API) его даёт.
    """

    def __init__(self, samplerate: int = 16000, stream_factory=None) -> None:
        self.samplerate = samplerate
        # Фабрика потока — точка внедрения для тестов (чтобы не подменять
        # sys.modules['sounddevice'], это ломает импорт numpy).
        self._factory = stream_factory
        self._stream = None
        self._device: object = "_unset"
        self._peak = 0.0
        self._lock = threading.Lock()
        self.error = ""

    @property
    def device(self):
        return self._device

    def switch(self, device: int | None) -> None:
        """Переключить индикатор на другое устройство (если изменилось)."""
        if device == self._device:
            return
        self.stop()
        self._device = device
        self.error = ""
        with self._lock:
            self._peak = 0.0
        try:
            # numpy/sounddevice импортируем локально: TUI стартует быстро, а
            # индикатор нужен только на экране настроек. ВАЖНО: numpy именно
            # здесь — раньше его не было в модуле, и колбэк падал с NameError,
            # показывая нулевой уровень (тест это теперь ловит).
            import numpy as _np

            def _cb(indata, frames, time_info, status) -> None:  # noqa: ANN001
                if not len(indata):
                    return
                peak = float(_np.abs(indata[:, 0]).max())
                with self._lock:
                    # Плавное затухание: столбик не дёргается на паузах.
                    self._peak = max(peak, self._peak * 0.99)

            kwargs = dict(samplerate=self.samplerate, channels=1, dtype="float32",
                          device=device, callback=_cb, blocksize=1024)
            if self._factory is not None:
                stream = self._factory(**kwargs)
            else:
                import sounddevice as sd
                stream = sd.InputStream(**kwargs)
            stream.start()
            self._stream = stream
        except Exception as exc:  # noqa: BLE001 — занятый/недоступный микрофон
            self.error = str(exc).split("[PaErrorCode")[0].strip()[:46]
            self._stream = None

    def peak(self) -> float:
        with self._lock:
            return self._peak

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._device = "_unset"

    @staticmethod
    def bar(peak: float, width: int = 20, gate: float = 0.01) -> str:
        """Полоска уровня + метка порога (порог = минимальный пик фразы)."""
        # Логарифмическая шкала: 0.001 -> 0, 1.0 -> ширина
        import math
        level = 0.0 if peak <= 0 else min(1.0, max(0.0, (math.log10(peak) + 3) / 3))
        filled = int(round(level * width))
        gate_pos = int(round(((math.log10(gate) + 3) / 3) * width)) if gate > 0 else 0
        cells = []
        for i in range(width):
            if i == gate_pos:
                cells.append("╎")       # метка порога видна всегда — по ней
            elif i < filled:            # видно, проходит ли голос
                cells.append("█")
            else:
                cells.append("░")
        return "".join(cells)


def build_choices(store, mics: list[dict], system_ok: bool) -> SettingsState:
    s = store.get()
    current = s.asr.input_device

    mic_opts: list[tuple[str, object]] = [("системный по умолчанию", None)]
    mic_notes: list[str] = ["жив" if system_ok else "нет сигнала"]
    for m in mics:
        mark = "жив" if m["alive"] else (m.get("status") or "нет сигнала")
        api = m.get("api") or ""
        if api:
            mark = f"{mark} [{api}]"
        mic_opts.append((m["name"], m["idx"]))
        mic_notes.append(mark)
    mic_idx = 0
    for i, (_, v) in enumerate(mic_opts):
        if v == current:
            mic_idx = i
            break

    def mic_note(i: int) -> str:
        return mic_notes[i] if i < len(mic_notes) else ""

    mode_opts = [("авто — слушает постоянно", "auto"), ("PTT — держишь F8", "ptt")]
    insert_opts = [("в активное окно (Ctrl+V)", "clipboard"),
                   ("в чат OpenCode (без фокуса)", "opencode")]

    def pick_index(options, value) -> int:
        for i, (_, v) in enumerate(options):
            if v == value:
                return i
        return 0

    state = SettingsState(choices=[
        Choice("mic", "Микрофон", mic_opts, mic_idx, mic_note),
        Choice("mode", "Режим диктовки", mode_opts, pick_index(mode_opts, s.mode)),
        Choice("insert", "Куда вставлять", insert_opts, pick_index(insert_opts, s.insert_method)),
    ])
    return state


def apply_choices(store, values: dict) -> None:
    s = store.get()
    s.asr.input_device = values.get("mic")
    s.mode = values.get("mode", s.mode)
    s.insert_method = values.get("insert", s.insert_method)
    store.save()


def render_settings(state: SettingsState, store, w: int, h: int,
                    level: tuple[float, str] | None = None) -> list[str]:
    """level = (пик, ошибка) для живого индикатора выделенного микрофона."""
    body: list[str] = []
    inner_w = w - 6
    for bi, ch in enumerate(state.choices):
        rows = []
        for i, (label, _value) in enumerate(ch.options):
            sel = i == ch.index
            plain = f"{'[x]' if sel else '[ ]'} {label}"
            note = ch.note(i) if ch.note else ""
            room = max(4, inner_w - 2 - len(note) - 1)
            plain = plain[:room]
            note_color = C_OK if note.startswith(("жив", "осн.")) else C_DIM
            row = (C_TEXT if sel else C_DIM) + plain + RESET
            row += " " * max(1, inner_w - 2 - len(plain) - len(note))
            row += note_color + norm_text(note) + RESET
            rows.append(row)
        body += block(ch.title, rows, inner_w + 2, focused=(bi == state.focus))
        body.append("")
        # Под блоком микрофона — живой уровень: видно, хватает ли громкости
        # (фраза в рантайме отсекается при пике ниже 0.01).
        if ch.key == "mic" and level is not None:
            peak, err = level
            if err:
                meter = f"{C_ERR}нет доступа: {err}{RESET}"
            else:
                bar = LevelMeter.bar(peak)
                ok = peak >= 0.01
                color = C_OK if ok else C_WARN
                verdict = "проходит порог 0.010" if ok else "тихо — фраза будет отсечена"
                meter = (f"{C_DIM}уровень {RESET}{color}{bar}{RESET}"
                         f"{C_DIM} {peak:.4f}{RESET}  "
                         f"{color}{verdict}{RESET}")
            body.append(f"   {meter}")
            body.append("")
    cur = store.get()
    info = (f"{C_DIM}сейчас: микрофон {cur.asr.input_device if cur.asr.input_device is not None else 'системный'}"
            f" · режим {cur.mode} · вставка {cur.insert_method}{RESET}")
    if h > len(body) + 8:
        body += ["", info, ""]
    body += ["   " + line for line in button_bar([
        ("↑↓", "выбор"),
        ("←→", "раздел"),
        ("F5", "пересканировать"),
        ("Enter", "запустить"),
        ("Esc", "выход"),
    ], inner_w - 4, primary="Enter")]
    return outer_frame("MyFLOW · настройки запуска", body, w, h)


def render_boot(rows: Sequence[str], w: int, h: int, spin: str, note: str = "") -> list[str]:
    body = ["", f"  {C_ACCENT}{spin}{RESET}  {C_TEXT}поднимаю локальные серверы...{RESET}", ""]
    for r in rows:
        body.append("   " + r)
    if note:
        body += ["", "   " + C_WARN + note + RESET]
    return outer_frame("MyFLOW · запуск", body, w, h)


def pill(text: str, color: str) -> str:
    """Компактная «пилюля» для статусов: ▐ авто ▌ с подсветкой."""
    return f"{color}{C_INV}▐ {text} ▌{RESET}"


def journal_style(line: str) -> str:
    """Раскрасить строку журнала по смыслу — так глаз цепляется за главное.

    Раньше весь журнал был серым: успех, предупреждение и ошибка выглядели
    одинаково. Теперь цвет несёт смысл: ✓/вставлено — зелёный, ⚠/WARNING —
    жёлтый, ✗/ошибка — красный, замеры (e2e) — приглушённые.
    """
    low = line.lower()
    if "✗" in line or "ошибка" in low or "error" in low or "traceback" in low:
        return f"{C_ERR}{line}{RESET}"
    if "⚠" in line or "warning" in low or "отложен" in low or "проверь" in low:
        return f"{C_WARN}{line}{RESET}"
    if "✓" in line or "+ " in line[:4] or "вставлено" in low:
        return f"{C_OK}{line}{RESET}"
    if "☺" in line or "голос" in low:
        return f"{C_ACCENT}{line}{RESET}"
    if "e2e=" in low or "asr=" in low or low.strip().startswith("..."):
        return f"{C_DIM}{line}{RESET}"
    if "⏏" in line or "▲" in line or "⟳" in line:
        return f"{C_WARN}{line}{RESET}"
    return f"{C_TEXT}{line}{RESET}"


def _module_row(item: dict, selected: bool, width: int, name_w: int = 13) -> str:
    """Строка модуля: значок состояния, имя, значение, [вкл/выкл].

    ВАЖНО: ширины считаем по ВИДИМОЙ длине (vlen/strip_ansi), иначе ANSI-коды
    съедают место и значения обрезаются («GigaAM v3 (C»).
    """
    if item.get("on") and item.get("ok"):
        mark, mcolor = "●", C_OK
    elif item.get("on"):
        mark, mcolor = "◐", C_WARN
    else:
        mark, mcolor = "○", C_DIM
    # Одинаковая ширина у обоих состояний: иначе колонка «ёлочек» прыгает.
    toggle_plain = "[вкл] " if item.get("on") else "[выкл]"
    toggle = (f"{C_OK}{toggle_plain}{RESET}" if item.get("on")
              else f"{C_DIM}{toggle_plain}{RESET}")
    prefix = "▸ " if selected else "  "
    name = f"{item['label']:<{name_w}}"
    room = max(6, width - 2 - 2 - name_w - len(toggle_plain) - 1)
    value_txt = item.get("value", "")[:room]
    pad = max(1, width - 2 - 2 - name_w - len(value_txt) - len(toggle_plain))
    mark_plain = f"{prefix}{mark} {name}{value_txt}"
    if selected:
        return f"{C_INV}{mark_plain}{' ' * pad}{toggle_plain}{RESET}"
    return (f"{C_DIM if not selected else C_TEXT}{prefix}{RESET}{mcolor}{mark}{RESET} "
            f"{C_TEXT if selected else C_DIM}{name}{RESET}"
            f"{C_DIM}{value_txt}{RESET}{' ' * pad}{toggle}")


def _value_row(item: dict, selected: bool, width: int, name_w: int = 17) -> str:
    """Строка управления: подпись и значение со стрелками ◀ ▶."""
    prefix = "▸ " if selected else "  "
    room = max(6, width - 2 - 2 - name_w - 6)
    value = item.get("value", "")[:room]
    label = f"{item['label']:<{name_w}}"
    pad = max(1, width - 2 - 2 - name_w - len(value) - 4)
    plain = f"{prefix}{label}◀ {value}{' ' * pad} ▶"
    if selected:
        return f"{C_INV}{plain}{RESET}"
    arrows = f"{C_KEY}◀{RESET}"
    arrows_r = f"{C_KEY}▶{RESET}"
    return (f"{C_DIM}{prefix}{RESET}{C_TEXT}{label}{RESET}"
            f" {arrows} {C_TEXT}{value}{RESET}{' ' * pad} {arrows_r}")


def key_legend(width: int, module_hint: str | None = None) -> list[str]:
    """Легенда клавиш с подписями (переносится по ширине, ничего не обрезается)."""
    chips = [("↑↓", "строка"), ("←→", "значение"), ("Enter", "вкл/выкл"),
             ("PgUp/PgDn", "журнал"), ("End", "живой лог"), ("s", "серверы"),
             ("r", "мастер"), ("q", "выход"),
             ("F6", "привязать окно"), ("F7", "стиль"), ("F8", "запись/стоп"),
             ("F9", "отмена"), ("F10", "вставить отложенное"),
             ("F5", "пересканировать микрофоны"),
             ("F11", "выгрузить/поднять LLM")]
    lines = button_bar(chips, width)
    if module_hint:
        lines.append(f"{C_WARN}подсказка: {module_hint}{RESET}")
    return lines


def stamp_confirm(lines: Sequence[str], question: str, hint: str,
                  w: int) -> list[str]:
    """Наложить модальное подтверждение по центру кадра (Да/Нет).

    Кадр уже нарисован (рамка на всю ширину), поэтому аккуратно подменяем
    несколько центральных строк: рамку и выравнивание сохраняем, ширина каждой
    строки остаётся w (draw() обрезает по ней).
    """
    out = list(lines)
    inner = max(10, w - 2)
    box_w = min(inner - 4, max(48, vlen(question) + 6, vlen(hint) + 6))
    box_w = max(30, box_w)
    pad = max(0, (inner - box_w) // 2)

    def centered(text: str) -> str:
        room = box_w - 2
        text = fit_width(text, room)
        visible = vlen(strip_ansi(text))
        left = max(0, (room - visible) // 2)
        return (" " * left + text + " " * max(0, room - visible - left))

    title = " Подтверждение "
    head = (f"{C_WARN}╔═{RESET}{BOLD}{C_WARN}{title}{RESET}"
            f"{C_WARN}" + "═" * max(0, box_w - 2 - len(title)) + f"╗{RESET}")
    bottom = f"{C_WARN}╚" + "═" * (box_w - 2) + f"╝{RESET}"
    body = [
        f"{C_WARN}║{RESET}{centered(C_TEXT + question + RESET)}{C_WARN}║{RESET}",
        f"{C_WARN}║{RESET}{centered('')}{C_WARN}║{RESET}",
        f"{C_WARN}║{RESET}{centered(C_OK + 'Да — Enter / y' + RESET + C_DIM + '     ' +
                                C_ERR + 'Нет — n / Esc' + RESET)}{C_WARN}║{RESET}",
        f"{C_WARN}║{RESET}{centered(C_DIM + hint + RESET)}{C_WARN}║{RESET}",
    ]
    box = [head] + body + [bottom]
    top = max(1, (len(out) - len(box)) // 2)
    for i, box_line in enumerate(box):
        idx = top + i
        if idx >= len(out) - 1:
            break
        visible = vlen(strip_ansi(box_line))
        right = max(0, inner - pad - visible)
        out[idx] = (f"{C_BORDER}║{RESET}" + " " * pad + box_line + " " * right
                    + f"{C_BORDER}║{RESET}")
    return out


# Разметка последнего кадра: нужна, чтобы понимать, куда попала мышь.
# {"journal": (первая_строка, последняя_строка)} — нумерация экранных строк с 1.
LAST_LAYOUT: dict = {}


def render_panel(snap: dict, w: int, h: int, cfg_line: str,
                 journal: Sequence[str], scroll: int, sel: int,
                 level: tuple[float, str] | None = None,
                 status_line: str = "", counters: dict | None = None,
                 confirm: tuple[str, str] | None = None) -> list[str]:
    """Единый пульт-центр: движки, управление, живой уровень, журнал, клавиши."""
    modules = snap.get("modules", [])
    values = snap.get("values", [])
    rows = modules + values
    counters = counters or {}

    body: list[str] = []
    # Живой уровень в шапке: видно, что микрофон работает, не уходя с экрана.
    if level is not None:
        peak, err = level
        if err:
            meter = f"{C_ERR}микрофон: нет доступа ({err}){RESET}"
        else:
            ok = peak >= 0.01
            meter = (f"{C_DIM}уровень {RESET}"
                     f"{C_OK if ok else C_WARN}{LevelMeter.bar(peak, width=14)}{RESET}"
                     f"{C_DIM} {peak:.4f}{RESET}")
    else:
        meter = ""
    body.append(f"  {cfg_line}")
    body.append(f"  {meter}   {status_line}")
    body.append("")

    sel_key = rows[sel].get("key") if 0 <= sel < len(rows) else None
    name_w_mod = max([len(m["label"]) for m in modules] or [12]) + 1
    name_w_val = max([len(v["label"]) for v in values] or [16]) + 1

    two_col = w >= 108
    if two_col:
        col = (w - 14) // 2
        left = [_module_row(m, i == sel, col - 4, name_w_mod)
                for i, m in enumerate(modules)]
        right = [_value_row(v, (len(modules) + i) == sel, col - 4, name_w_val)
                 for i, v in enumerate(values)]
        height = max(len(left), len(right))
        left += [""] * (height - len(left))
        right += [""] * (height - len(right))
        lblock = block("Движки и модули", left, col)
        rblock = block("Управление (меняется на лету)", right, col)
        for i in range(height + 2):
            body.append(" " + (lblock[i] if i < len(lblock) else " " * col) + " " +
                        (rblock[i] if i < len(rblock) else ""))
    else:
        body += [" " + x for x in block(
            "Движки и модули", [_module_row(m, i == sel, w - 10, name_w_mod)
                                for i, m in enumerate(modules)], w - 6)]
        body.append("")
        body += [" " + x for x in block(
            "Управление (меняется на лету)",
            [_value_row(v, (len(modules) + i) == sel, w - 10, name_w_val)
             for i, v in enumerate(values)], w - 6)]

    legend = key_legend(w - 6,
                        rows[sel].get("hint") if 0 <= sel < len(rows) else None)
    # Журнал забирает остаток высоты; окно строк считаем по отступу прокрутки
    # (в рендере, а не в вызывающем коде — только тут известна реальная высота).
    space = h - len(body) - len(legend) - 4 - 2      # 2 рамки кадра, 2 рамки блока
    if space >= 3 and journal:
        end = max(space, len(journal) - max(0, scroll))
        start = max(0, end - space)
        shown = list(journal[start:end])
        if scroll > 0:
            shown = [f"{C_WARN}↑ прокрутка: {scroll} строк назад "
                     f"(End — в конец){RESET}"] + shown[1:]
        body.append("")
        # Запоминаем, где на экране оказался журнал: колесо мыши должно листать
        # ТОЛЬКО его (см. TuiApp._on_wheel). Строки нумеруем с 1: строка 0 — рамка.
        _journal_rows = ["  " + x for x in block(
            "Журнал (PgUp/PgDn — листать)", [journal_style(r) for r in shown], w - 6)]
        LAST_LAYOUT["journal"] = (len(body) + 1, len(body) + len(_journal_rows))
        body += _journal_rows
    body.append("")
    body += ["  " + line for line in legend]
    frame = outer_frame("MyFLOW · пульт-центр", body, w, h)
    if confirm is not None:
        # Модальное подтверждение: перекрывает кадр, пока не ответишь.
        frame = stamp_confirm(frame, confirm[0], confirm[1], w)
    return frame


def render_dashboard(state: TuiState, w: int, h: int, cfg_line: str) -> list[str]:
    status_text, status_ansi, log, servers = state.snapshot()
    body: list[str] = ["", f"  {cfg_line}", ""]

    for key in ("asr", "llm"):
        info = servers.get(key)
        if not info:
            continue
        st = info.get("state", "down")
        title = info.get("title", key)
        note = info.get("note", "")
        pad = 26 - len(title)
        body.append(f"   {C_TEXT}{title}{RESET}{' ' * max(1, pad)}"
                    f"{status_color(st)}[{status_label(st)}]{RESET}   {C_DIM}{note}{RESET}")
    body.append("")

    inner_w = w - 8
    # Ниже журнала ещё: статус, пустая строка, полоска кнопок (2 строки) и рамка.
    rows_room = max(4, h - len(body) - 9)
    with state.lock:
        offset = state.scroll
    if log:
        end = max(rows_room, len(log) - offset)
        start = max(0, end - rows_room)
        tail = log[start:end]
        if offset > 0:
            tail = ([f"{C_WARN}↑ прокрутка: {offset} строк назад "
                     f"(End — в конец){RESET}"] + tail[1:])
    else:
        tail = [f"{C_DIM}пока пусто — нажми F8 и говори{RESET}"]
    body += ["   " + x for x in block("Журнал", [C_DIM + r + RESET for r in tail], inner_w + 2)]
    body.append("")
    uptime = int(time.monotonic() - state.started_at)
    body.append(f"   {status_ansi}{status_text}{RESET}"
                f"    {C_DIM}фраз: {state.phrases} · последняя e2e: {state.last_e2e_ms} мс"
                f" · сессия {uptime // 60} мин{RESET}")
    body.append("")
    body += ["   " + line for line in button_bar([
        ("PgUp/PgDn", "журнал"),
        ("End", "в конец"),
        ("r", "настройки"),
        ("s", "серверы"),
        ("c", "очистить"),
        ("q", "выход"),
    ], inner_w - 4, primary="q")]
    return outer_frame("MyFLOW · пульт", body, w, h)


def render_error(message: str, w: int, h: int) -> list[str]:
    body = ["", f"  {C_ERR}Не удалось запустить:{RESET}", ""]
    for part in str(message).split("\n"):
        body.append("   " + C_TEXT + norm_text(part) + RESET)
    body += ["", f"  {C_DIM}Enter — выход{RESET}"]
    return outer_frame("MyFLOW · ошибка", body, w, h)


# ----------------------------------------------------------------- оверлей
class TuiOverlay:
    """StatusOverlay-совместимый: статус идёт в TUI (+ графическая плашка, если включена)."""

    def __init__(self, state: TuiState, gfx=None) -> None:
        self.state = state
        self.gfx = gfx

    def start(self) -> None:
        if self.gfx is not None:
            self.gfx.start()

    def show(self, text: str, color: str = "#f2f5f7", timeout=None) -> None:
        self.state.set_status(norm_text(text), hex_to_ansi(color))
        if self.gfx is not None:
            self.gfx.show(text, color, timeout)

    def hide(self) -> None:
        self.state.set_status("ожидание", C_DIM)
        if self.gfx is not None:
            self.gfx.hide()

    def stop(self) -> None:
        if self.gfx is not None:
            self.gfx.stop()

    def is_running(self) -> bool:
        return True

    def query(self, fn, timeout: float = 2.0):
        return self.gfx.query(fn, timeout) if self.gfx is not None else None


# ----------------------------------------------------------------- приложение
class TuiApp:
    def __init__(self, *, no_servers: bool = False) -> None:
        from myflow.config import SettingsStore
        self.store = SettingsStore()
        self.console = Console()
        self.state = TuiState()
        self.supervisor = Supervisor(default_specs(), enabled=not no_servers)
        self.no_servers = no_servers
        self.app = None
        self._stop = threading.Event()
        # Зафиксированный размер окна (ширина, высота кадра) — см. run().
        self._locked_size: tuple[int, int] | None = None
        self._sink: PaneSink | None = None
        self._prev_stdout = None
        self._health_thread: threading.Thread | None = None
        self._mics: list[dict] = []
        self._system_ok = False
        self._gfx = None
        self._overlay = None
        self._demo = False
        self._force_settings = False   # --settings: показать мастер даже при готовом конфиге
        # True — приложение создали мы сами (тогда и гасим его при выходе).
        self._app_owned = False
        # Модальное подтверждение (смена движка ASR): (вопрос, подсказка, действие)
        self._confirm: dict | None = None

    # ---- инфраструктура ----
    def _install_sink(self) -> None:
        log_file = ROOT / "data" / "main.log"
        try:
            from myflow.main import _rotate_log
            _rotate_log(log_file)
        except Exception:  # noqa: BLE001
            pass
        self._prev_stdout = sys.stdout
        self._sink = PaneSink(self.state, log_file)
        sys.stdout = self._sink

    def _restore_stdout(self) -> None:
        if self._sink is not None:
            try:
                self._sink.close()
            except Exception:  # noqa: BLE001
                pass
        if self._prev_stdout is not None:
            sys.stdout = self._prev_stdout

    @staticmethod
    def _read_key(timeout: float = 0.1) -> str | None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"\x00", b"\xe0"):
                    nxt = msvcrt.getch()
                    return map_extended_key(ch[0], nxt)
                if ch == b"\x1b":
                    # Может быть ESC-последовательность: F5 (\x1b[15~) или событие
                    # мыши в SGR-формате (\x1b[<64;120;34M) — она длиннее, поэтому
                    # собираем до 24 символов.
                    seq = ""
                    deadline = time.monotonic() + 0.05
                    while time.monotonic() < deadline and len(seq) < 24:
                        if not msvcrt.kbhit():
                            break
                        nxt = msvcrt.getch()
                        seq += nxt.decode("latin-1", "ignore")
                        mouse = map_mouse_sequence(seq)
                        if mouse:
                            return mouse
                        name = map_vt_sequence(seq[1:] if seq.startswith("[") else seq)
                        if name:
                            return name
                    return "esc"
                if ch == b"\r":
                    return "enter"
                if ch == b" ":
                    return "space"
                if ch == b"\x03":
                    return "ctrl-c"
                try:
                    return ch.decode("utf-8", "ignore").lower()
                except Exception:  # noqa: BLE001
                    return None
            time.sleep(0.01)
        return None

    def _start_health_poller(self) -> None:
        if self._health_thread is not None and self._health_thread.is_alive():
            return

        def loop() -> None:
            while not self._stop.is_set():
                snap = self.supervisor.health_snapshot()
                with self.state.lock:
                    self.state.servers = snap
                time.sleep(2.0)

        self._health_thread = threading.Thread(target=loop, daemon=True, name="tui-health")
        self._health_thread.start()

    # ---- фаза 1: серверы ----
    def _boot_phase(self) -> None:
        specs = self.supervisor.specs
        spin = "|/-\\"
        i = 0
        for spec in specs:
            if self._stop.is_set():
                return
            if self.supervisor.health(spec.url, spec.health_path):
                continue
            missing = spec.missing()
            if missing:
                self._draw_boot(spin[i % 4], note=f"{spec.title}: {missing}")
                i += 1
                for _ in range(20):                   # ~2 c, чтобы успеть прочитать
                    self._draw_boot(spin[i % 4], note=f"{spec.title}: {missing}")
                    i += 1
                    time.sleep(0.1)
                continue
            if not self.supervisor.start(spec):
                continue
            timeout = 90.0 if spec.required else 60.0
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not self._stop.is_set():
                self._draw_boot(spin[i % 4], note=f"{spec.title}: загружает модель...")
                i += 1
                if self.supervisor.health(spec.url, spec.health_path):
                    break
                time.sleep(0.25)
            self.supervisor.status(spec)             # обновить внутренние ошибки

    def _draw_boot(self, spin: str, note: str = "") -> None:
        w, h = self.console.size()
        rows = []
        for key, info in self.supervisor.health_snapshot().items():
            st = info["state"]
            rows.append(f"{C_TEXT}{info['title']:<26}{RESET}{status_color(st)}"
                        f"[{status_label(st)}]{RESET} {C_DIM}{info['note']}{RESET}")
        self.console.draw(render_boot(rows, w, h, spin, note))

    # ---- фаза 2: настройки ----
    def _scan_mics(self) -> None:
        from myflow.pick_mic import gather_mics, probe_device
        self._draw_boot("*", note="сканирую микрофоны (проверяю сигнал)...")
        try:
            self._mics = gather_mics()
        except Exception:  # noqa: BLE001
            self._mics = []
        try:
            self._system_ok = bool(probe_device(None))
        except Exception:  # noqa: BLE001
            self._system_ok = False

    def _settings_phase(self) -> bool:
        self._scan_mics()
        st = self.store
        state = build_choices(st, self._mics, self._system_ok)
        meter = LevelMeter()
        try:
            while not self._stop.is_set():
                # Индикатор следит за ВЫДЕЛЕННЫМ микрофоном: стрелками можно
                # сравнить варианты (MME/DirectSound) по живому уровню.
                meter.switch(state.values().get("mic"))
                w, h = self.console.size()
                self.console.draw(render_settings(state, st, w, h,
                                                  level=(meter.peak(), meter.error)))
                key = self._read_key(0.15)
                if key == "up":
                    state.move_item(-1)
                elif key == "down":
                    state.move_item(1)
                elif key == "left":
                    state.move_block(-1)
                elif key in ("right", "tab"):
                    state.move_block(1)
                elif key in ("enter", "f8"):
                    apply_choices(st, state.values())
                    return True
                elif key in ("esc", "q", "ctrl-c"):
                    return False
                elif key == "f5":
                    meter.stop()            # не держим устройство во время скана
                    self._scan_mics()
                    state = build_choices(st, self._mics, self._system_ok)
        finally:
            # Обязательно: иначе поток удержит микрофон и запись не откроется.
            meter.stop()
        return False

    # ---- фаза 3: пульт ----
    def _store(self):
        """Активные настройки: у работающего приложения и у TUI должен быть ОДИН
        объект. Раньше TUI держал свою копию, поэтому панель меняла микрофон в
        приложении, а читала старое значение из своей копии — «кнопки не работают».
        """
        app = getattr(self, "app", None)
        if app is not None and getattr(app, "settings", None) is not None:
            return app.settings
        return self.store

    def _panel_header(self) -> str:
        """Короткая шапка панели: микрофон/вставка/цель есть отдельными строками."""
        s = self._store().get()
        mode = "авто (слушает)" if s.mode == "auto" else "PTT (F8)"
        target = "активное окно"
        app = getattr(self, "app", None)
        if app is not None:
            locked = getattr(app, "_target_locked", None)
            cur = getattr(app, "_target", None)
            if locked is not None:
                target = f"⚲ {locked.label[:24]}"
            elif cur is not None:
                target = cur.label[:24]
        return (f"{pill(mode, C_OK)}"
                f"  {C_DIM}стиль{RESET} {C_TEXT}{s.default_style}{RESET}"
                f"  {C_DIM}цель{RESET} {C_ACCENT}⚲ {target}{RESET}")

    def _config_line(self) -> str:
        s = self.store.get()
        mic = s.asr.input_device
        mode = "авто (слушает постоянно)" if s.mode == "auto" else "PTT (держишь F8)"
        # Куда уйдёт текст: цель закреплена (F6) / берётся из старта диктовки.
        target = "активное окно"
        app = getattr(self, "app", None)
        if app is not None:
            locked = getattr(app, "_target_locked", None)
            cur = getattr(app, "_target", None)
            if locked is not None:
                target = f"⚲ {locked.label[:26]}"
            elif cur is not None:
                target = cur.label[:26]
        return (f"{C_TEXT}{BOLD}режим:{RESET} {C_ACCENT}{mode}{RESET}"
                f"   {C_TEXT}{BOLD}стиль:{RESET} {C_ACCENT}{s.default_style}{RESET}"
                f"   {C_TEXT}{BOLD}вставляю в:{RESET} {C_ACCENT}{target}{RESET}"
                f"   {C_TEXT}{BOLD}микрофон:{RESET} {C_ACCENT}"
                f"{'#' + str(mic) if mic is not None else 'системный'}{RESET}"
                f"   {C_TEXT}{BOLD}вставка:{RESET} {C_ACCENT}{s.insert_method}{RESET}")

    def _dashboard_phase(self) -> str:
        """Возвращает 'quit' | 'settings'."""
        from myflow.main import MyFlowApp, _install_console_handler
        from myflow.overlay import StatusOverlay

        # Если пульт перезапускается (r — настройки), гасим прежнюю плашку.
        if self._gfx is not None:
            try:
                self._gfx.stop()
            except Exception:  # noqa: BLE001
                pass
            self._gfx = None

        flags = self.store.get().flags
        self._gfx = StatusOverlay() if flags.get("overlay") else None
        self._overlay = TuiOverlay(self.state, self._gfx)

        self.state.add_log("MyFLOW запущен. F8 — диктовка, F9 — отмена, F7 — стиль.")
        # ВАЖНО: если приложение уже подсунуто (тесты, встраивание) — НЕ создаём
        # своё. Раньше здесь всегда создавался новый MyFlowApp на ЖИВЫХ настройках,
        # поэтому скриптовые тесты панели незаметно писали в data/settings.json.
        if self.app is None:
            try:
                self.app = MyFlowApp(overlay=self._overlay)
            except Exception as exc:  # noqa: BLE001
                self._show_error(str(exc))
                return "quit"
            self._app_owned = True
            # Автолечение LLM: приложение просит поднять/погасить сервер — TUI умеет.
            self.app.set_server_guard(self._ensure_server_alive)
            self.app.set_server_stopper(self._stop_server_by_key)
            try:
                self.app.start()
            except Exception as exc:  # noqa: BLE001
                self._show_error(str(exc))
                return "quit"
        # Настройки — общие с приложением (одно хранилище, а не две копии).
        self.store = self.app.settings
        try:
            _install_console_handler(self._shutdown_now)
        except Exception:  # noqa: BLE001
            pass

        self._start_health_poller()
        # Панель-центр: выделенная строка + живой индикатор уровня текущего микрофона.
        sel = 0
        idle_frames = 0
        meter = LevelMeter()
        last_key = None
        snap: dict = {"modules": [], "values": [], "counters": {}}
        snap_at = 0.0
        try:
            while not self._stop.is_set():
                w, h = self.console.size()
                # Состояние модулей пересобираем не каждый кадр (0.4 с), а клавиши
                # читаем часто (0.05 с): управление отзывчивое, а панель не тратит
                # время на опрос серверов и сканирование микрофонов.
                now = time.monotonic()
                if now - snap_at > 0.4 or not snap["modules"]:
                    snap = self._control_snapshot()
                    snap_at = now
                rows = snap.get("modules", []) + snap.get("values", [])
                sel = max(0, min(sel, max(0, len(rows) - 1)))
                meter.switch(self._store().get().asr.input_device)
                with self.state.lock:
                    journal = list(self.state.log)
                    offset = self.state.scroll
                if not journal:
                    journal = [f"{C_DIM}пока пусто — нажми F8 и говори{RESET}"]
                status_line, status_ansi, _, _ = self.state.snapshot()
                if "запись" in status_line:
                    # Живая анимация: видно, что микрофон пишет прямо сейчас.
                    spinner = "·", "··", "···", "··"
                    status_line = f"{status_line} {spinner[int(time.monotonic() * 3) % 4]}"
                counters = snap.get("counters", {})
                notes = []
                if counters.get("mic_scanning"):
                    notes.append("сканирую микрофоны…")
                if counters.get("deferred_polish"):
                    notes.append(f"{counters['deferred_polish']} фраз ждут дополировки")
                scan_note = ("  ·  " + "  ·  ".join(notes)) if notes else ""
                try:
                    frame_lines = render_panel(
                    snap, w, h, self._panel_header(), journal, offset, sel,
                    level=(meter.peak(), meter.error),
                    confirm=((self._confirm["question"], self._confirm["hint"])
                             if self._confirm else None),
                    status_line=(f"{status_ansi}{norm_text(status_line)}{RESET}"
                                 f"  фраз: {counters.get('phrases', 0)}"
                                 f" · отложено: {counters.get('pending', 0)}"
                                 f" · в очереди: {counters.get('queue', 0)}"
                                 f"{scan_note}"),
                        counters=counters)
                except Exception as exc:  # noqa: BLE001 — панель важнее бага в рендере
                    # Ошибка отрисовки НЕ должна ронять TUI: показываем кадр-заглушку
                    # и пишем в журнал, чтобы баг было видно.
                    self.state.add_log(f"ошибка отрисовки: {exc!r}")
                    frame_lines = render_error(f"Ошибка отрисовки панели: {exc}", w, h)
                painted = self.console.draw(frame_lines)
                # Частота опроса клавиш адаптивная: при записи/анимации часто,
                # в покое редко — так TUI почти не занимает CPU, когда не нужен.
                busy = bool(counters.get("pending")) or bool(counters.get("queue"))
                animate = ("запись" in status_line) or bool(self._confirm)
                idle_frames = 0 if (painted or busy or animate) else idle_frames + 1
                timeout = 0.03 if (busy or animate or idle_frames < 20) else 0.15
                last_key = self._read_key(timeout)
                # Колесо мыши: листает журнал (вне журнала — молча игнорируем,
                # окно при этом не прокручивается: мышь перехвачена в enable_mouse).
                if last_key and last_key.startswith(("wheelup", "wheeldown")):
                    self._on_wheel(last_key, h)
                    last_key = None
                if last_key in ("q", "ctrl-c") and self._confirm is None:
                    return "quit"
                if last_key == "ctrl-c":
                    return "quit"
                # Модальное подтверждение перехватывает клавиши, пока не ответишь.
                if self._confirm is not None:
                    if last_key in ("enter", "space", "y", "д"):
                        action = self._confirm["action"]
                        self._confirm = None
                        self.state.add_log(f"подтверждено: {self._confirm_note(action)}")
                        self._panel_apply(snap.get("modules", []),
                                          snap.get("values", []), action["sel"],
                                          action["key"], force=True)
                    elif last_key in ("n", "esc", "backspace", "н", "q"):
                        # ВАЖНО: q тоже закрывает окно (иначе из него нет выхода,
                        # а выход из приложения заблокирован — окно модальное).
                        self.state.add_log("отменено: движок не меняю")
                        self._confirm = None
                    else:
                        # Любая другая клавиша при открытом окне — просто ждём
                        # ответа (никаких действий по панели).
                        pass
                    continue
                if last_key == "r":
                    return "settings"
                if last_key == "s":
                    self._restart_down_servers()
                elif last_key == "c":
                    with self.state.lock:
                        self.state.log = ["журнал очищен"]
                        self.state.scroll = 0
                elif last_key == "o":
                    self._toggle_gfx()
                elif last_key == "f5":
                    ok, msg = (self.app.control_apply("rescan_mics")
                               if self.app is not None else (False, "нет приложения"))
                    self.state.add_log(f"{'✓' if ok else '✗'} {msg}")
                elif last_key in ("pageup", "pagedown", "home", "end"):
                    self._scroll_journal(last_key, h)
                elif last_key in ("up", "down"):
                    sel = (sel + (1 if last_key == "down" else -1)) % max(1, len(rows))
                elif last_key in ("left", "right", "enter", "space"):
                    modules = snap.get("modules", [])
                    values = snap.get("values", [])
                    asr_sel = next((i for i, m in enumerate(modules)
                                    if m.get("key") == "asr"), None)
                    if (asr_sel == sel and self._confirm_needed()
                            and last_key in ("enter", "space", "left", "right")):
                        self._ask_asr_confirm(modules, values, sel, last_key)
                    else:
                        sel = self._panel_apply(modules, values, sel, last_key)
        finally:
            meter.stop()
            if self._app_owned:
                try:
                    self.app.stop()
                except Exception:  # noqa: BLE001
                    pass
                self.app = None
        return "quit"

    def _control_snapshot(self) -> dict:
        """Живое состояние от приложения; пустой снимок, если оно ещё не готово."""
        app = self.app
        if app is None or not hasattr(app, "control_snapshot"):
            return {"modules": [], "values": [], "counters": {}}
        try:
            return app.control_snapshot()
        except Exception as exc:  # noqa: BLE001 — панель не должна падать
            self.state.add_log(f"панель: {exc!r}")
            return {"modules": [], "values": [], "counters": {}}

    # ---- подтверждение опасных действий ----

    def _confirm_needed(self) -> bool:
        """Нужно ли подтверждать смену движка (настройка asr.confirm_switch)."""
        try:
            return bool(getattr(self._store().get().asr, "confirm_switch", True))
        except Exception:  # noqa: BLE001
            return True

    def _ask_asr_confirm(self, modules: list[dict], values: list[dict],
                         sel: int, key: str) -> None:
        """Спросить подтверждение перед сменой движка распознавания.

        Смена движка — не косметика: whisper требует поднять сервер на :8081, и
        если это не выйдет, распознавание перестанет работать. Поэтому Enter на
        строке ASR сначала спрашивает «уверен?».
        """
        engine = ""
        try:
            engine = getattr(getattr(self.app, "whisper", None), "backend_name", "")
        except Exception:  # noqa: BLE001
            engine = ""
        to_whisper = "gigaam" in (engine or "").lower()
        if to_whisper:
            question = "Переключить распознавание на whisper?"
            hint = "Будет поднят whisper-сервер на :8081; GigaAM останется резервом"
        else:
            question = "Вернуть распознавание на GigaAM?"
            hint = "Локальная модель на CPU; whisper-сервер выгрузится (~1 ГБ VRAM)"
        self._confirm = {"question": question, "hint": hint,
                         "action": {"sel": sel, "key": "enter", "engine": engine}}

    @staticmethod
    def _confirm_note(action: dict) -> str:
        engine = (action or {}).get("engine", "")
        return ("переключаю на GigaAM" if "gigaam" in (engine or "").lower()
                else "переключаю на whisper")

    def _panel_apply(self, modules: list[dict], values: list[dict],
                     sel: int, key: str, force: bool = False) -> int:
        """Применить нажатие к выделенной строке панели. Возвращает новый выбор.

        Тип строки определяем по позиции (индекс >= числа модулей). Раньше здесь
        заново собирался снимок и строки сравнивались через `is` — объекты каждый
        раз новые, поэтому строка микрофона считалась МОДУЛЕМ и нажатие уходило в
        «неизвестный модуль: mic». Это и было «кнопки не работают».
        """
        rows = modules + values
        if not rows:
            return sel
        sel = max(0, min(sel, len(rows) - 1))
        item = rows[sel]
        key_name = item.get("key", "")
        app = self.app
        if app is None:
            return sel
        is_value_row = sel >= len(modules)
        is_module = not is_value_row

        try:
            if is_module and key in ("enter", "space", "left", "right"):
                action, value = "module", key_name
                if key_name == "vad" and key == "right":
                    action, value = "vad_threshold", self._vad_step(+0.05)
                elif key_name == "vad" and key == "left":
                    action, value = "vad_threshold", self._vad_step(-0.05)
                elif key_name == "voice" and key in ("left", "right"):
                    action, value = "voice_mode", (1 if key == "right" else -1)
                ok, msg = app.control_apply(action, value)
                self.state.add_log(f"{'✓' if ok else '✗'} {msg}")

            elif is_value_row:
                if key_name == "mic":
                    self._cycle_mic(1 if key in ("right", "enter", "space") else -1)
                elif key_name == "mode":
                    ok, msg = app.control_apply("mode", self._store().get().mode != "auto")
                    self.state.add_log(f"{'✓' if ok else '✗'} {msg}")
                elif key_name == "insert":
                    ok, msg = app.control_apply(
                        "insert", self._store().get().insert_method != "opencode")
                    self.state.add_log(f"{'✓' if ok else '✗'} {msg}")
                elif key_name == "style":
                    self._cycle_style_live(1 if key == "right" else -1)
                elif key_name == "target":
                    ok, msg = app.control_apply("target")
                    self.state.add_log(f"{'✓' if ok else '✗'} {msg}")
        except Exception as exc:  # noqa: BLE001
            self.state.add_log(f"ошибка панели: {exc!r}")
        return sel

    def _vad_step(self, delta: float) -> float:
        cur = float(getattr(self.app.recorder, "vad_threshold", 0.30))
        return round(max(0.10, min(0.60, cur + delta)), 2)

    def _cycle_mic(self, direction: int) -> None:
        """Переключить микрофон на лету по списку (мгновенно, без сканирования)."""
        try:
            choices = self.app._mic_choices()
        except Exception:  # noqa: BLE001
            choices = []
        if not choices:
            from myflow.pick_mic import fast_mic_list
            choices = fast_mic_list()
        # WDM-KS-индексы нестабильны (после переподключений «устройство исчезло»),
        # поэтому в перебор берём только те, что не помечены мёртвыми и не WDM.
        good = [m for m in choices
                if m.get("alive") is not False and m.get("api") != "WDM"]
        choices = good or choices
        if not choices:
            self.state.add_log("микрофонов не найдено (F5 — пересканировать)")
            return
        cur = self._store().get().asr.input_device
        idxs: list[object] = [None]
        for m in choices:                      # уникальные, в порядке появления
            if m["idx"] not in idxs:
                idxs.append(m["idx"])
        pos = idxs.index(cur) if cur in idxs else 0
        new = idxs[(pos + direction) % len(idxs)]
        if new == cur and len(idxs) > 1:       # не «застреваем» на одном значении
            new = idxs[(pos + 2 * direction) % len(idxs)]
        ok, msg = self.app.control_apply("mic", new)
        self.state.add_log(f"{'✓' if ok else '✗'} {msg}")

    def _cycle_style_live(self, direction: int) -> None:
        styles = [s for s in ("plain", "casual", "formal", "terminal", "summarize")]
        cur = self._store().get().default_style
        pos = styles.index(cur) if cur in styles else 0
        new = styles[(pos + direction) % len(styles)]
        ok, msg = self.app.control_apply("style", new)
        self.state.add_log(f"{'✓' if ok else '✗'} {msg}")

    def _on_wheel(self, key: str, height: int) -> bool:
        """Колесо мыши: листаем журнал, и только если курсор над журналом.

        Мышиный режим (enable_mouse) перехватывает колесо у терминала, поэтому
        сам пульт больше не «уезжает вверх»; вне журнала событие игнорируется.
        """
        try:
            _, coords = key.split(":", 1)
            _x, y = (int(v) for v in coords.split(","))
        except Exception:  # noqa: BLE001
            return False
        top, bottom = LAST_LAYOUT.get("journal", (0, 0))
        if not (top and top <= y <= bottom):
            return False
        self._scroll_journal("up" if key.startswith("wheelup") else "down", height)
        return True

    def _scroll_journal(self, key: str, height: int) -> None:
        """Листание журнала в пульте (PgUp/PgDn/Home/End/↑/↓).

        Позиция хранится как отступ от конца, поэтому новые строки не «дёргают»
        то, что ты читаешь: чтобы вернуться к живому логу — End.
        """
        page = max(3, height - 12)
        with self.state.lock:
            total = len(self.state.log)
        if key == "pageup":
            self.state.scroll_by(page, page, total)
        elif key == "pagedown":
            self.state.scroll_by(-page, page, total)
        elif key == "up":
            self.state.scroll_by(1, page, total)
        elif key == "down":
            self.state.scroll_by(-1, page, total)
        elif key == "home":
            self.state.scroll_to(total, page, total)
        elif key == "end":
            self.state.scroll_to(0, page, total)

    def _toggle_gfx(self) -> None:
        from myflow.overlay import StatusOverlay
        if self._gfx is None:
            self._gfx = StatusOverlay()
            self._gfx.start()
            if self._overlay is not None:
                self._overlay.gfx = self._gfx
            self.state.add_log("графическая плашка включена")
        else:
            try:
                self._gfx.stop()
            except Exception:  # noqa: BLE001
                pass
            self._gfx = None
            if self._overlay is not None:
                self._overlay.gfx = None
            self.state.add_log("графическая плашка выключена")

    def _stop_server_by_key(self, key: str) -> bool:
        """Погасить сервер по ключу (для выгрузки LLM под игру). True — погасили."""
        spec = next((x for x in self.supervisor.specs if x.key == key), None)
        if spec is None:
            return False
        try:
            stopped = self.supervisor.stop(spec)
        except Exception:  # noqa: BLE001
            return False
        with self.state.lock:
            self.state.servers = self.supervisor.health_snapshot()
        if stopped:
            self.state.add_log(f"{spec.title}: выгружен (память освобождена)")
        return bool(stopped)

    def _ensure_server_alive(self, key: str) -> bool:
        """Поднять сервер по ключу (llm/asr) для автолечения. True — он доступен.

        Вызывается из фонового потока приложения, поэтому ничего не рисует:
        только запускает процесс и ждёт готовности.
        """
        spec = next((x for x in self.supervisor.specs if x.key == key), None)
        if spec is None:
            return False
        state, _ = self.supervisor.status(spec)
        if state in ("ok", "external"):
            return True
        if not self.supervisor.start(spec):
            return False
        ok = self.supervisor.wait_ready(spec, 60.0)
        with self.state.lock:
            self.state.servers = self.supervisor.health_snapshot()
        return ok

    def _restart_down_servers(self) -> None:
        for spec in self.supervisor.specs:
            state, _ = self.supervisor.status(spec)
            if state in ("down", "error"):
                self.state.add_log(f"перезапускаю {spec.title}...")
                if self.supervisor.start(spec):
                    ok = self.supervisor.wait_ready(spec, 90.0)
                    self.state.add_log(f"{spec.title}: {'готов' if ok else 'не поднялся'}")
        with self.state.lock:
            self.state.servers = self.supervisor.health_snapshot()

    def _show_error(self, message: str) -> None:
        w, h = self.console.size()
        for _ in range(150):                      # ~15 c или до Enter
            self.console.draw(render_error(message, w, h))
            if self._read_key(0.1) in ("enter", "esc", "q", "ctrl-c"):
                return

    def _shutdown_now(self) -> None:
        self._stop.set()

    # ---- жизненный цикл ----
    def run(self) -> int:
        # Размер окна фиксируем ДО первого кадра: 0 в настройках = взять текущий
        # и запомнить его, дальше окно всегда этого размера (см. lock_size).
        try:
            tui_cfg = self.store.get().tui
            if getattr(tui_cfg, "fixed_size", True):
                w0, h0 = self.console.lock_size(tui_cfg.cols, tui_cfg.rows)
                if not tui_cfg.cols or not tui_cfg.rows:
                    tui_cfg.cols, tui_cfg.rows = w0, max(20, h0)
                    self.store.save()
                self._locked_size = (tui_cfg.cols, tui_cfg.rows)
            if getattr(tui_cfg, "mouse_wheel", True):
                self.console.enable_mouse()
        except Exception:  # noqa: BLE001 — оформление не должно мешать запуску
            pass
        self.console.clear()
        self.console.cursor(False)
        self._install_sink()
        code = 0
        try:
            self._boot_phase()
            if self._stop.is_set():
                return 0
            # Мастер настройки нужен только на первом запуске: дальше всё
            # меняется прямо на панели (микрофон/режим/вставка/стиль).
            # Вернуться в мастер можно клавишей r.
            from myflow.config import SETTINGS_FILE
            first_run = not Path(SETTINGS_FILE).exists()
            if (first_run or self._force_settings) and not self._settings_phase():
                return 0
            while True:
                # Окно должно ВСЕГДА оставаться зафиксированного размера: если
                # пользователь (или терминал) его изменил — возвращаем как было.
                if getattr(self, "_locked_size", None):
                    try:
                        # Размер сравниваем «сырой» (как его вернул lock_size).
                        # Пере-фиксируем не чаще раза в 2 с и не больше 5 попыток:
                        # если терминал не даёт менять размер, не бьёмся в цикле.
                        if (self.console.raw_size() != self._locked_size
                                and time.monotonic() >= getattr(self, "_relock_at", 0.0)
                                and getattr(self, "_relock_tries", 0) < 5):
                            self._relock_at = time.monotonic() + 2.0
                            self._relock_tries = getattr(self, "_relock_tries", 0) + 1
                            self.console.lock_size(*self._locked_size)
                            self.console.clear()
                    except Exception:  # noqa: BLE001
                        pass
                self.console.clear()
                self.state.add_log(
                    f"конфигурация: режим={self.store.get().mode}, "
                    f"вставка={self.store.get().insert_method}, "
                    f"микрофон={self.store.get().asr.input_device}")
                action = self._dashboard_phase()
                if action != "settings" or self._stop.is_set():
                    break
                self.console.clear()
        except KeyboardInterrupt:
            code = 0
        finally:
            self._stop.set()
            # Мышь возвращаем терминалу: иначе после выхода колесо «залипает» в
            # режиме приложения и обычная прокрутка окна не работает.
            try:
                self.console.disable_mouse()
            except Exception:  # noqa: BLE001
                pass
            self._shutdown()
        return code

    def _shutdown(self) -> None:
        try:
            if self.app is not None:
                self.app.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._overlay is not None:
                self._overlay.stop()
        except Exception:  # noqa: BLE001
            pass
        self.supervisor.stop_all()
        self._restore_stdout()
        self.console.clear()
        self.console.reset()
        print("MyFLOW остановлен. Логи: data\\main.log, data\\asr.log, data\\llm.log")


# ----------------------------------------------------------------- демо
def _demo_snapshot() -> dict:
    """Пример снимка для витрины панели (реальные данные — из приложения)."""
    return {"modules": [
        {"key": "asr", "label": "ASR", "value": "GigaAM v3 e2e (CPU)", "on": True,
         "ok": True, "note": "", "hint": "Enter — сменить движок (GigaAM ↔ whisper)"},
        {"key": "router", "label": "Авто-выбор",
         "value": "GigaAM → whisper при сомнении", "on": True, "ok": True,
         "note": "второе мнение",
         "hint": "Enter — включить/выключить: подозрительные фразы уходят в whisper"},
        {"key": "judge", "label": "Судья LLM",
         "value": "выключен", "on": False, "ok": True, "note": "точность",
         "hint": "Enter — оба движка, LLM склеивает лучший текст"},
        {"key": "vad", "label": "VAD", "value": "Silero v6 · 0.30", "on": True, "ok": True,
         "note": "", "hint": "←→ порог, Enter — автостоп по паузе"},
        {"key": "llm", "label": "LLM", "value": "qwen3-4b · :8082", "on": True, "ok": True,
         "note": "", "hint": "Enter — включить/выключить полировку"},
        {"key": "voice", "label": "Голос", "value": "31 фраз · auto · 0.25", "on": True,
         "ok": True, "note": "", "hint": "Enter вкл/выкл, ←→ режим (auto/always/off)"},
        {"key": "ax", "label": "AX", "value": "UI Automation", "on": True, "ok": True,
         "note": "", "hint": "Enter — включить/выключить UI Automation"},
        {"key": "snippets", "label": "Сниппеты", "value": "фразы → готовый текст",
         "on": True, "ok": True, "note": "", "hint": "Enter — включить/выключить"},
        {"key": "fastpath", "label": "Быстрый путь", "value": "не звать LLM, если готово",
         "on": True, "ok": True, "note": "", "hint": "Enter — включить/выключить"},
        {"key": "learning", "label": "Обучение", "value": "словарь из правок", "on": True,
         "ok": True, "note": "", "hint": "Enter — включить/выключить"},
        {"key": "overlay", "label": "Оверлей", "value": "плашка статуса", "on": False,
         "ok": True, "note": "", "hint": "Enter — включить/выключить"},
        {"key": "keep_audio", "label": "Аудио фраз", "value": "последние 50", "on": True,
         "ok": True, "note": "", "hint": "Enter — включить/выключить"},
    ], "values": [
        {"key": "mic", "label": "Микрофон", "value": "Ardor Gaming Vox [DS]",
         "hint": "←→ сменить микрофон на лету (F5 — пересканировать)"},
        {"key": "mode", "label": "Режим диктовки", "value": "авто (слушает)",
         "hint": "←→ авто ↔ PTT"},
        {"key": "insert", "label": "Куда вставлять", "value": "активное окно (Ctrl+V)",
         "hint": "←→ сменить способ вставки"},
        {"key": "style", "label": "Стиль текста", "value": "plain",
         "hint": "←→ стиль по умолчанию (F7 — на лету)"},
        {"key": "target", "label": "Цель вставки", "value": "⚲ Анализ и улучшение проекта",
         "hint": "Enter — закрепить/снять окно (F6)"},
    ], "counters": {"phrases": 12, "pending": 0, "queue": 0}}


def demo_app(hold: float, screen: str = "panel") -> int:
    """Витрина: рисует пульт или экран настроек с примерными данными."""
    tui = TuiApp(no_servers=True)
    tui._demo = True
    st = tui.state
    st.set_status("запись…", hex_to_ansi("#f5c542"))
    st.phrases, st.last_e2e_ms = 12, 1180
    st.servers = {
        "asr": {"title": "ASR whisper.cpp", "state": "ok", "note": "large-v3-turbo, RTF 0.12"},
        "llm": {"title": "LLM Qwen3-4B (полировка)", "state": "ok", "note": "полировка ~230 мс"},
    }
    for line in [
        "● запись начата (окно: code.exe | main.py - Visual Studio Code)",
        "⚙ распознавание (1820 мс)...",
        "ASR : проверка связи один два три",
        "LLM : Проверка связи, один, два, три.  (style=plain, polish=236 мс)",
        "✓ Проверка связи, один, два, три.",
        "✓ вставлено",
        "e2e=1240 мс (asr=520, polish=236)",
    ]:
        st.add_log(line)
    mics = [{"name": "Микрофон (USB Audio Device)", "idx": 1, "alive": True,
             "status": "жив (rms=0.0121)", "api": "DS"},
            {"name": "Микрофон (Realtek(R) Audio)", "idx": 3, "alive": True,
             "status": "тихий вход (rms=0.00061) — подними усиление?", "api": "MME"},
            {"name": "Набор микрофонов (Kinect USB Audio)", "idx": 5, "alive": False,
             "status": "нет сигнала (rms=0.000015)", "api": "MME"}]
    settings = build_choices(tui.store, mics, True)
    settings.focus = 0
    tui.console.clear()
    tui.console.cursor(False)
    try:
        end = time.monotonic() + hold
        while time.monotonic() < end:
            w, h = tui.console.size()
            if screen == "settings":
                tui.console.draw(render_settings(settings, tui.store, w, h,
                                                 level=(0.045, "")))
            elif screen == "dashboard":
                tui.console.draw(render_dashboard(st, w, h, tui._config_line()))
            else:
                # Витрина единого пульта-центра: движки, управление, уровень, журнал.
                tui.console.draw(render_panel(
                    _demo_snapshot(), w, h, tui._panel_header(),
                    [strip_ansi(x) for x in st.log],
                    0, 3, level=(0.1027, ""),
                    status_line="✓ вставлено  фраз: 12 · отложено: 0 · в очереди: 0"))
            time.sleep(0.2)
    finally:
        tui.console.clear()
        tui.console.reset()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="myflow.tui", description="MyFLOW: один TUI вместо трёх окон")
    ap.add_argument("--no-servers", action="store_true",
                    help="не поднимать whisper/llama (уже запущены вручную)")
    ap.add_argument("--settings", action="store_true",
                    help="показать мастер настройки, даже если конфиг уже есть")
    ap.add_argument("--demo", action="store_true", help="показать экран с примерными данными")
    ap.add_argument("--screen", choices=("panel", "dashboard", "settings"),
                    default="panel",
                    help="какой экран показывать в демо")
    ap.add_argument("--hold", type=float, default=8.0, help="сколько секунд держать демо-кадр")
    args = ap.parse_args(argv)

    if args.demo:
        return demo_app(args.hold, args.screen)

    if sys.stdin is None or not sys.stdin.isatty():
        print("[MyFLOW] нет интерактивной консоли: TUI не запустить. "
              "Запусти через start-myflow.bat или python -m myflow.main.")
        return 3
    tui = TuiApp(no_servers=args.no_servers)
    tui._force_settings = bool(args.settings)
    return tui.run()


if __name__ == "__main__":
    sys.exit(main())
