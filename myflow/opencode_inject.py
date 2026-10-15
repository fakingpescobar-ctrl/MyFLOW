"""MyFLOW — инжектор текста в чат OpenCode (аналог whisper_listener в opencode-tts).

Единственный путь — WriteConsoleInputW напрямую в консольный буфер opencode:
не требует фокуса, работает даже когда активно другое окно или opencode свёрнут.

ПРЕДНАМЕРЕННО БЕЗ fallback с фокусом: подъём окна opencode (SetForegroundWindow /
ShowWindow(SW_RESTORE)) сворачивает полноэкранную игру (Dota 2 и т.п.) — это
неприемлемо. Если консоль opencode недоступна — возвращаем False и НЕ трогаем
ни фокус, ни чужие окна.

КРИТИЧНО: FreeConsole()/AttachConsole() выполняются ТОЛЬКО во временном
дочернем процессе (python -m myflow.opencode_inject). В главном процессе
MyFLOW эти вызовы никогда не делаются — иначе main теряет свою консоль
(окно MyFLOW немеет, а повторный AttachConsole даёт WinError 6).

Интерфейс совпадает с ClipboardPaste (метод paste(text)) — main подставляет
нужную реализацию по настройке insert_method.
"""
from __future__ import annotations

import base64
import ctypes
import json
import subprocess
import sys
from ctypes import wintypes

# --- WinAPI-прототипы (обязательны: без restype HANDLE усекается до 32 бит).
# Объявляются ПОСЛЕ структур (ниже) — ctypes требует готовые типы.
_kernel32 = ctypes.windll.kernel32

# --- WinAPI: WriteConsoleInputW (единственный путь, без фокуса) ---

KEY_EVENT = 0x0001
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3

VK_RETURN = 0x0D

CREATE_NO_WINDOW = 0x08000000


class KEY_EVENT_RECORD(ctypes.Structure):
    _fields_ = [
        ("bKeyDown", wintypes.BOOL),
        ("wRepeatCount", wintypes.WORD),
        ("wVirtualKeyCode", wintypes.WORD),
        ("wVirtualScanCode", wintypes.WORD),
        ("uChar", wintypes.WCHAR),
        ("dwControlKeyState", wintypes.DWORD),
    ]


class INPUT_RECORD(ctypes.Structure):
    _fields_ = [
        ("EventType", wintypes.WORD),
        ("Event", KEY_EVENT_RECORD),
    ]


if ctypes.sizeof(INPUT_RECORD) not in (20, 24):  # 20 на x86, 24 на x64 (pad)
    raise RuntimeError(f"неожиданный sizeof(INPUT_RECORD)={ctypes.sizeof(INPUT_RECORD)}")

# --- Прототипы (после структур) ---

_kernel32.FreeConsole.restype = wintypes.BOOL
_kernel32.AttachConsole.argtypes = [wintypes.DWORD]
_kernel32.AttachConsole.restype = wintypes.BOOL
_kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
]
_kernel32.CreateFileW.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.WriteConsoleInputW.argtypes = [
    wintypes.HANDLE, ctypes.POINTER(INPUT_RECORD), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
]
_kernel32.WriteConsoleInputW.restype = wintypes.BOOL


def _log_warn(msg: str) -> None:
    """Тихий warning в stderr — не роняем main, но не глотаем молча."""
    try:
        sys.stderr.write(f"[opencode_inject] {msg}\n")
    except Exception:
        pass


def _find_opencode_pids() -> list[int]:
    """PIDs ИНТЕРАКТИВНЫХ opencode.exe, серверные (serve/--port) - в конце списка.

    Почему это важно: `opencode serve --port N` - сервер, у него своя консоль, которую
    пользователь не видит. Когда такой процесс оказывался первым, инжект отрабатывал
    «успешно», а текст уходил в невидимую консоль - выглядело как «инжект не работает».
    """
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
             "Get-CimInstance Win32_Process -Filter \"Name = 'opencode.exe'\" "
             "| Select-Object ProcessId, CommandLine | ConvertTo-Json"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
            creationflags=CREATE_NO_WINDOW,
        )
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout)
            if isinstance(data, dict):
                data = [data]
            interactive: list[int] = []
            servers: list[int] = []
            for entry in data:
                try:
                    pid = int(entry["ProcessId"])
                except Exception:  # noqa: BLE001
                    continue
                cmdline = (entry.get("CommandLine") or "").lower()
                if " serve" in cmdline or "--port" in cmdline:
                    servers.append(pid)
                else:
                    interactive.append(pid)
            if servers:
                _log_warn(f"opencode: серверные процессы отложены ({servers}), "
                          f"вставляем в интерактивные ({interactive or 'нет'})")
            return interactive + servers
    except Exception as exc:
        _log_warn(f"WMI lookup opencode failed: {exc}")
    return []


def _char_record(ch: str, key_down: bool) -> INPUT_RECORD:
    r = INPUT_RECORD()
    r.EventType = KEY_EVENT
    r.Event.bKeyDown = key_down
    r.Event.wRepeatCount = 1
    r.Event.wVirtualKeyCode = 0
    r.Event.wVirtualScanCode = 0
    r.Event.uChar = ch
    r.Event.dwControlKeyState = 0
    return r


def _enter_record(key_down: bool) -> INPUT_RECORD:
    r = INPUT_RECORD()
    r.EventType = KEY_EVENT
    r.Event.bKeyDown = key_down
    r.Event.wRepeatCount = 1
    r.Event.wVirtualKeyCode = VK_RETURN
    r.Event.wVirtualScanCode = 0
    r.Event.uChar = "\r"
    r.Event.dwControlKeyState = 0
    return r


def _child_inject(text: str, auto_enter: bool) -> int:
    """Инжект в консоль opencode (В ДОЧЕРНЕМ ПРОЦЕССЕ).

    FreeConsole/AttachConsole здесь безопасны: это временный процесс,
    у него нет своей консоли (CREATE_NO_WINDOW), после выхода — всё
    подчищено. Главный процесс MyFLOW не затронут.
    Возвращает 0 при успехе, иначе 1 + предупреждение в stderr.
    """
    pids = _find_opencode_pids()
    if not pids:
        _log_warn("opencode не найден (нет живого процесса opencode.exe)")
        return 1
    pid = pids[0]
    _kernel32.FreeConsole()  # на всякий случай — процесса без консоли
    try:
        if not _kernel32.AttachConsole(pid):
            _log_warn(f"AttachConsole({pid}) не удался (err={ctypes.get_last_error()})")
            return 1
        h_con_in = _kernel32.CreateFileW(
            "CONIN$",
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None, OPEN_EXISTING, 0, None,
        )
        if not h_con_in or h_con_in == wintypes.HANDLE(-1).value:
            _log_warn(f"CONIN$ недоступен (err={ctypes.get_last_error()})")
            return 1
        try:
            records = [rec for ch in text if ch not in ("\r", "\n")
                       for rec in (_char_record(ch, True), _char_record(ch, False))]
            if auto_enter:
                records.extend((_enter_record(True), _enter_record(False)))
            n = len(records)
            if not n:
                return 1
            arr = (INPUT_RECORD * n)(*records)
            written = wintypes.DWORD(0)
            ok = _kernel32.WriteConsoleInputW(h_con_in, arr, n, ctypes.byref(written))
            if not ok or written.value != n:
                _log_warn(f"WriteConsoleInputW неполная: ok={ok}, written={written.value}/{n}")
                return 1
            return 0
        finally:
            _kernel32.CloseHandle(h_con_in)
    finally:
        _kernel32.FreeConsole()


class OpenCodeInjector:
    """Вставка текста в чат OpenCode. Интерфейс — как у ClipboardPaste.

    Консольные манипуляции (FreeConsole/AttachConsole) выполняются во
    временном дочернем процессе, чтобы главный MyFLOW сохранял свою
    консоль (окно статусов не немеет, повторные вставки работают).
    """

    def __init__(self, auto_enter: bool = True) -> None:
        self.auto_enter = auto_enter

    # Консольный инжект буфер обмена не использует: авто-обучение из правок
    # в этом режиме невозможно (и main его не запускает).
    can_read = False

    def read(self) -> str:
        return ""

    def paste(self, text: str) -> bool:
        """Отправить текст в чат OpenCode. True — ушло, False — не вышло."""
        if not text:
            return False
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        try:
            r = subprocess.run(
                [sys.executable, "-m", "myflow.opencode_inject",
                 "--payload", payload,
                 "--auto-enter", "1" if self.auto_enter else "0"],
                stdin=subprocess.DEVNULL,
                capture_output=True, text=True, errors="replace", timeout=15,
                creationflags=CREATE_NO_WINDOW,
            )
            if r.returncode == 0:
                return True
            warn = (r.stderr or "").strip()
            if warn:
                # Пробрасываем предупреждение дочернего в stderr main — видно в окне MyFLOW.
                try:
                    sys.stderr.write(f"{warn}\n")
                except Exception:
                    pass
            return False
        except Exception as exc:
            import traceback
            _log_warn(f"inject error: {exc!r}\n{traceback.format_exc(limit=8)}")
            return False


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="MyFLOW CLI-инжектор (только дочерний процесс)")
    ap.add_argument("--payload", required=True, help="base64(utf-8) текста")
    ap.add_argument("--auto-enter", default="1", choices=("0", "1"))
    args = ap.parse_args()
    try:
        text = base64.b64decode(args.payload).decode("utf-8")
    except Exception as exc:
        _log_warn(f"bad payload: {exc!r}")
        return 1
    return _child_inject(text, args.auto_enter == "1")


if __name__ == "__main__":
    sys.exit(_cli())