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
         --hold N (сколько секунд держать демо-кадр).
"""
from __future__ import annotations

import argparse
import ctypes
import io
import msvcrt
import os
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
    top = (f"{C_BORDER}╔═{RESET}{BOLD}{C_TITLE}{label}{RESET}"
           f"{C_BORDER}" + "═" * max(0, inner - 1 - len(label)) + f"╗{RESET}")
    bottom = f"{C_BORDER}╚" + "═" * inner + f"╝{RESET}"
    lines = [top]
    room = h - 2
    body = list(body)[:room]
    body += [""] * (room - len(body))
    for ln in body:
        lines.append(f"{C_BORDER}║{RESET}" + pad_to(ln, inner) + f"{C_BORDER}║{RESET}")
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


def map_vt_sequence(seq: str) -> str | None:
    """Хвост ESC-последовательности ('15~', '[15~', 'A') -> имя клавиши."""
    text = seq.strip().lstrip("[")
    return _VT_KEYS.get(text)


# ----------------------------------------------------------------- консоль
class Console:
    """Низкоуровневый вывод кадра: VT-режим, размер, курсор."""

    def __init__(self) -> None:
        self.out = sys.__stdout__ or sys.stdout
        self.vt = self._enable_vt()
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

    def size(self) -> tuple[int, int]:
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

    def draw(self, lines: Sequence[str]) -> None:
        if not self.vt:
            self.out.write("\n".join(strip_ansi(x) for x in lines) + "\n")
            self.out.flush()
            return
        buf = [CSI + "H"]
        for i, ln in enumerate(lines):
            buf.append(f"{CSI}{i + 1};1H{ln}{CSI}K")
        self.out.write("".join(buf))
        self.out.flush()

    def clear(self) -> None:
        if self.vt:
            self.out.write(CSI + "2J" + CSI + "H")
        self.out.flush()

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
    return Path(os.environ.get("MYFLOW_MODELS_DIR", r"X:\llm\models"))


def llama_dir() -> Path:
    return Path(os.environ.get("MYFLOW_LLAMA_DIR", r"X:\llm\llama-cpp"))


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
    whisper = root / "bin" / "whisper" / "Release" / "whisper-server.exe"
    from myflow.asr import whisper_needed      # локально: без циклических импортов
    # Модель ASR можно подменить через MYFLOW_WHISPER_MODEL (например, на
    # ggml-large-v3-q5_0.bin для «режима качества»: медленнее в ~4 раза, по
    # замерам на своих записях выигрыша по WER не даёт — см. README).
    whisper_model = Path(os.environ.get("MYFLOW_WHISPER_MODEL")
                         or (models / "ggml-large-v3-turbo-q5_0.bin"))
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


def render_settings(state: SettingsState, store, w: int, h: int) -> list[str]:
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
            note_color = C_OK if note in ("живой", "осн.") else C_DIM
            row = (C_TEXT if sel else C_DIM) + plain + RESET
            row += " " * max(1, inner_w - 2 - len(plain) - len(note))
            row += note_color + norm_text(note) + RESET
            rows.append(row)
        body += block(ch.title, rows, inner_w + 2, focused=(bi == state.focus))
        body.append("")
    cur = store.get()
    info = (f"{C_DIM}сейчас: микрофон {cur.asr.input_device if cur.asr.input_device is not None else 'системный'}"
            f" · режим {cur.mode} · вставка {cur.insert_method}{RESET}")
    if h > len(body) + 6:
        body += ["", info, ""]
    body.append(f"{C_ACCENT}↑/↓{RESET}{C_DIM} выбор в блоке   {RESET}"
                f"{C_ACCENT}←/→{RESET}{C_DIM} другой блок   {RESET}"
                f"{C_ACCENT}Enter{RESET}{C_DIM} старт   {RESET}"
                f"{C_ACCENT}Esc{RESET}{C_DIM} выход   {RESET}"
                f"{C_ACCENT}F5{RESET}{C_DIM} пересканировать микрофоны{RESET}")
    return outer_frame("MyFLOW · настройки запуска", body, w, h)


def render_boot(rows: Sequence[str], w: int, h: int, spin: str, note: str = "") -> list[str]:
    body = ["", f"  {C_ACCENT}{spin}{RESET}  {C_TEXT}поднимаю локальные серверы...{RESET}", ""]
    for r in rows:
        body.append("   " + r)
    if note:
        body += ["", "   " + C_WARN + note + RESET]
    return outer_frame("MyFLOW · запуск", body, w, h)


def render_dashboard(state: TuiState, w: int, h: int, cfg_line: str,
                     hint: str = "q выход · r настройки · s перезапуск серверов · c очистить журнал") -> list[str]:
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
    rows_room = max(4, h - len(body) - 8)
    tail = log[-rows_room:] if log else ["пока пусто — нажми F8 и говори"]
    body += ["   " + x for x in block("Журнал", [C_DIM + r + RESET for r in tail], inner_w + 2)]
    body.append("")
    uptime = int(time.monotonic() - state.started_at)
    body.append(f"   {status_ansi}{status_text}{RESET}"
                f"    {C_DIM}фраз: {state.phrases} · последняя e2e: {state.last_e2e_ms} мс"
                f" · сессия {uptime // 60} мин{RESET}")
    body.append("")
    body.append(f"   {C_DIM}{hint}{RESET}")
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
        self._sink: PaneSink | None = None
        self._prev_stdout = None
        self._health_thread: threading.Thread | None = None
        self._mics: list[dict] = []
        self._system_ok = False
        self._gfx = None
        self._overlay = None
        self._demo = False

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
                    # Может быть ESC-последовательность (F5 = \x1b[15~ в VT-режиме).
                    seq = ""
                    deadline = time.monotonic() + 0.05
                    while time.monotonic() < deadline and len(seq) < 6:
                        if not msvcrt.kbhit():
                            break
                        nxt = msvcrt.getch()
                        seq += nxt.decode("latin-1", "ignore")
                        name = map_vt_sequence(seq[1:] if seq.startswith("[") else seq)
                        if name:
                            return name
                    return "esc"
                if ch == b"\r":
                    return "enter"
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
        while not self._stop.is_set():
            w, h = self.console.size()
            self.console.draw(render_settings(state, st, w, h))
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
                self._scan_mics()
                state = build_choices(st, self._mics, self._system_ok)
        return False

    # ---- фаза 3: пульт ----
    def _config_line(self) -> str:
        s = self.store.get()
        mic = s.asr.input_device
        mode = "авто (слушает постоянно)" if s.mode == "auto" else "PTT (держишь F8)"
        return (f"{C_TEXT}{BOLD}режим:{RESET} {C_ACCENT}{mode}{RESET}"
                f"   {C_TEXT}{BOLD}стиль:{RESET} {C_ACCENT}{s.default_style}{RESET}"
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
        try:
            self.app = MyFlowApp(overlay=self._overlay)
        except Exception as exc:  # noqa: BLE001
            self._show_error(str(exc))
            return "quit"
        try:
            self.app.start()
        except Exception as exc:  # noqa: BLE001
            self._show_error(str(exc))
            return "quit"
        try:
            _install_console_handler(self._shutdown_now)
        except Exception:  # noqa: BLE001
            pass

        self._start_health_poller()
        last_key = None
        try:
            while not self._stop.is_set():
                w, h = self.console.size()
                self.console.draw(render_dashboard(self.state, w, h, self._config_line()))
                last_key = self._read_key(0.12)
                if last_key in ("q", "ctrl-c"):
                    return "quit"
                if last_key == "r":
                    return "settings"
                if last_key == "s":
                    self._restart_down_servers()
                elif last_key == "c":
                    with self.state.lock:
                        self.state.log = ["журнал очищен"]
                elif last_key == "o":
                    self._toggle_gfx()
        finally:
            try:
                self.app.stop()
            except Exception:  # noqa: BLE001
                pass
            self.app = None
        return "quit"

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
        self.console.clear()
        self.console.cursor(False)
        self._install_sink()
        code = 0
        try:
            self._boot_phase()
            if self._stop.is_set():
                return 0
            if not self._settings_phase():
                return 0
            while True:
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
def demo_app(hold: float, screen: str = "dashboard") -> int:
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
    mics = [{"name": "Микрофон (USB Audio Device)", "idx": 1, "alive": True},
            {"name": "Микрофон (Realtek(R) Audio)", "idx": 3, "alive": True},
            {"name": "Микрофон (ATK Mercury I)", "idx": 5, "alive": False}]
    settings = build_choices(tui.store, mics, True)
    settings.focus = 0
    tui.console.clear()
    tui.console.cursor(False)
    try:
        end = time.monotonic() + hold
        while time.monotonic() < end:
            w, h = tui.console.size()
            if screen == "settings":
                tui.console.draw(render_settings(settings, tui.store, w, h))
            else:
                tui.console.draw(render_dashboard(st, w, h, tui._config_line()))
            time.sleep(0.2)
    finally:
        tui.console.clear()
        tui.console.reset()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="myflow.tui", description="MyFLOW: один TUI вместо трёх окон")
    ap.add_argument("--no-servers", action="store_true",
                    help="не поднимать whisper/llama (уже запущены вручную)")
    ap.add_argument("--demo", action="store_true", help="показать экран с примерными данными")
    ap.add_argument("--screen", choices=("dashboard", "settings"), default="dashboard",
                    help="какой экран показывать в демо")
    ap.add_argument("--hold", type=float, default=8.0, help="сколько секунд держать демо-кадр")
    args = ap.parse_args(argv)

    if args.demo:
        return demo_app(args.hold, args.screen)

    if sys.stdin is None or not sys.stdin.isatty():
        print("[MyFLOW] нет интерактивной консоли: TUI не запустить. "
              "Запусти через start-myflow.bat или python -m myflow.main.")
        return 3
    return TuiApp(no_servers=args.no_servers).run()


if __name__ == "__main__":
    sys.exit(main())
