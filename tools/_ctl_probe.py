"""Прочитать экран ЧУЖОГО окна консоли (по PID) — диагностика отрисовки TUI.

Запуск: python tools/_ctl_probe.py <pid> <файл-вывода>
Помогает увидеть ровно то, что видит пользователь в окне с TUI.
"""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path


class _Coord(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class _SmallRect(ctypes.Structure):
    _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short),
                ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]


class _Csbi(ctypes.Structure):
    _fields_ = [("dwSize", _Coord), ("dwCursorPosition", _Coord),
                ("wAttributes", ctypes.c_ushort), ("srWindow", _SmallRect),
                ("dwMaximumWindowSize", _Coord)]


K = ctypes.windll.kernel32


def dump(pid: int) -> list[str]:
    K.FreeConsole()
    if not K.AttachConsole(pid):
        return [f"не смог подключиться к консоли процесса {pid} (ошибка {K.GetLastError()})"]
    lines: list[str] = []
    try:
        # после AttachConsole нужен свежий дескриптор консоли: CONOUT$
        GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING = 0x80000000, 0x40000000, 3
        handle = K.CreateFileW("CONOUT$", GENERIC_READ | GENERIC_WRITE, 3, None,
                               OPEN_EXISTING, 0, None)
        if not handle or handle == -1:
            return [f"CONOUT$ не открылся (ошибка {K.GetLastError()})"]
        info = _Csbi()
        if not K.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return ["нет доступа к буферу консоли"]
        win_w = info.srWindow.Right - info.srWindow.Left + 1
        win_h = info.srWindow.Bottom - info.srWindow.Top + 1
        lines.append(f"окно={win_w}x{win_h} буфер={info.dwSize.X}x{info.dwSize.Y} "
                     f"вьюпорт_top={info.srWindow.Top}")
        buf = ctypes.create_unicode_buffer(win_w + 1)
        written = ctypes.c_ulong()
        for y in range(info.srWindow.Top, info.srWindow.Bottom + 1):
            K.ReadConsoleOutputCharacterW(handle, buf, win_w, _Coord(0, y),
                                          ctypes.byref(written))
            text = buf.value[: written.value]
            lines.append(f"{y:>3}|{text.rstrip()}")
        # ищем признак переноса: строки ровно по ширине окна
        full = [ln for ln in lines[1:] if len(ln) - 4 >= win_w - 1]
        lines.append(f"строк во всю ширину окна (признак переноса): {len(full)}")
    finally:
        K.FreeConsole()
    return lines


def main() -> int:
    pid = int(sys.argv[1])
    out = Path(sys.argv[2])
    time.sleep(0.5)
    out.write_text("\n".join(dump(pid)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
