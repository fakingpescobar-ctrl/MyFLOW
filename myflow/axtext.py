"""MyFLOW — чтение текста поля в фокусе через UI Automation (AX).

Зачем: научиться ловить правки пользователя БЕЗ Ctrl+C. Раньше обучение
требовало, чтобы пользователь выделил исправленный текст и скопировал его в
буфер за 6 секунд — на практике не работало никогда. Wispr Flow держит для
этого нативный helper и наблюдает поле ввода (contentObservationEndReason).
Здесь то же самое, но без нативных зависимостей: PowerShell-процесс с .NET
UIAutomationClient, запущенный один раз и опрашиваемый по stdin/stdout.

Если helper недоступен (нет powershell, UIA молчит) — вызывающий код обязан
деградировать на старый путь через буфер обмена: `AxReader.read()` вернёт None.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "ax_read.ps1"
CREATE_NO_WINDOW = 0x08000000


@dataclass
class AxText:
    """Прочитанное поле: текст + контекст (какое окно/процесс)."""

    text: str
    source: str          # value | legacy | text | none | error
    pid: int = 0
    window: str = ""
    reliable: bool = True

    @property
    def empty(self) -> bool:
        return not self.text.strip()


class AxReader:
    """Постоянный процесс-хелпер UIA. Потокобезопасен (один внутренний lock)."""

    START_TIMEOUT = 8.0
    READ_TIMEOUT = 2.5

    def __init__(self, script: Path = SCRIPT, command: list[str] | None = None) -> None:
        self.script = script
        self._command = list(command) if command else None   # для тестов/диагностики
        self._proc: subprocess.Popen | None = None
        self._lines: "queue.Queue[str]" = queue.Queue()
        # RLock: start() при неудаче зовёт stop() изнутри захваченного лока
        # (обычный Lock здесь даёт вечный дедлок — проверено тестом).
        self._lock = threading.RLock()
        self._reader: threading.Thread | None = None
        self._failed = False          # helper не поднялся: больше не пробуем
        self._last_error = ""

    # ---- жизненный цикл ----

    @staticmethod
    def _powershell() -> str | None:
        """Windows PowerShell 5.1: в нём гарантированно есть UIAutomationClient."""
        import shutil
        for exe in ("powershell.exe", "powershell", "pwsh.exe"):
            path = shutil.which(exe)
            if path:
                return path
        return None

    def start(self) -> bool:
        with self._lock:
            if self._failed:
                return False
            if self._proc is not None and self._proc.poll() is None:
                return True
            if self._command:
                argv = list(self._command)
            else:
                exe = self._powershell()
                if exe is None or not self.script.exists():
                    self._failed = True
                    self._last_error = "нет powershell.exe или ax_read.ps1"
                    return False
                argv = [exe, "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-File", str(self.script)]
            try:
                self._proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                    errors="replace", bufsize=1, creationflags=CREATE_NO_WINDOW,
                )
            except Exception as exc:  # noqa: BLE001
                self._failed = True
                self._last_error = f"не удалось запустить helper: {exc!r}"
                return False
            self._lines = queue.Queue()
            self._reader = threading.Thread(target=self._pump, daemon=True,
                                            name="ax-reader")
            self._reader.start()
            # Проверяем живость и готовность первого ответа.
            probe = self._request(timeout=self.START_TIMEOUT)
            if probe is None:
                self._last_error = "helper не ответил на пробный запрос"
                self._failed = True
                self.stop()
                return False
            return True

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self._lines.put(line)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._lines.put("")          # сигнал «поток закончился»

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
        # Явно закрываем пайпы: иначе TextIOWrapper'ы живут до сборщика мусора
        # и на выходе дают ResourceWarning/OSError (Errno 22) на Windows.
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def available(self) -> bool:
        return not self._failed and self._proc is not None and self._proc.poll() is None

    @property
    def last_error(self) -> str:
        return self._last_error

    # ---- запросы ----

    def _request(self, timeout: float) -> AxText | None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return None
        try:
            proc.stdin.write("read\n")
            proc.stdin.flush()
        except Exception:  # noqa: BLE001
            return None
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if not line:
            return None
        return self._parse(line)

    @staticmethod
    def _parse(line: str) -> AxText | None:
        line = line.strip()
        if not line:
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None
        raw = data.get("text_b64") or ""
        try:
            text = base64.b64decode(raw).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            text = ""
        return AxText(
            text=text,
            source=str(data.get("source") or "none"),
            pid=int(data.get("pid") or 0),
            window=str(data.get("window") or ""),
            reliable=bool(data.get("ok")) and str(data.get("source")) != "text",
        )

    def read(self, timeout: float | None = None) -> AxText | None:
        """Прочитать поле в фокусе. None — AX недоступен (нужен fallback)."""
        with self._lock:
            if self._failed:
                return None
            if not self.available:
                if not self.start():
                    return None
            return self._request(timeout if timeout is not None else self.READ_TIMEOUT)


class AxWatcher:
    """Наблюдение за полем ввода после вставки (аналог их contentObservation).

    Возвращает причину окончания наблюдения — как contentObservationEndReason
    у Wispr: правка найдена, окно истекло, поле очищено, текст не найден и т.п.
    """

    # причины
    EDITED = "edited"                       # пользователь поправил текст -> учим
    TIMEOUT = "observation_window_elapsed"  # окно истекло без правки
    EMPTIED = "textbox_emptied"             # поле очистили (отмена/удаление)
    NOT_FOUND = "dictated_text_not_found"   # нашего текста в поле нет
    ANCHOR = "anchor_mismatch"              # другое окно/процесс
    UNAVAILABLE = "ax_unavailable"          # AX не поднялся

    def __init__(self, reader: AxReader, window_seconds: float = 6.0,
                 poll_interval: float = 0.5, max_divergence: float = 0.7) -> None:
        self.reader = reader
        self.window_seconds = window_seconds
        self.poll_interval = poll_interval
        # Для AX порог мягче, чем у буфера обмена: мы ЗНАЕМ, что читаем то же
        # поле, куда вставляли (pid/окно проверены), поэтому любое заметное
        # изменение текста — правка. Правка «один два три» -> «1 2 3» меняет
        # 3 слова из 5 и не проходит старую проверку перекрытия токенов (0.4).
        self.max_divergence = max_divergence

    @staticmethod
    def _is_edit(base: str, candidate: str, max_divergence: float) -> bool:
        from myflow.textdiff import divergence_score
        base, candidate = (base or "").strip(), (candidate or "").strip()
        if not candidate or candidate == base:
            return False
        if divergence_score(base, candidate) <= max_divergence:
            return True
        # Короткие фразы: одна замена слова даёт divergence 1.0 («Провер» ->
        # «Проверка»), поэтому разрешаем совпадение по началу/вхождению.
        if len(base.split()) <= 2:
            head = base[:4].lower()
            if head and (candidate.lower().startswith(head)
                         or base.lower().startswith(candidate[:4].lower())
                         or candidate.lower() in base.lower()
                         or base.lower() in candidate.lower()):
                return True
        return False

    def watch(self, inserted_text: str, expected_pid: int = 0):
        """Ждать правку. Возвращает (edited_text | None, reason).

        Пустая строка edited_text означает «правки не было».
        """
        if self.reader.read() is None:
            return None, self.UNAVAILABLE
        deadline = time.monotonic() + self.window_seconds
        while time.monotonic() < deadline:
            sample = self.reader.read()
            if sample is None:
                return None, self.UNAVAILABLE
            text = sample.text
            if expected_pid and sample.pid and sample.pid != expected_pid:
                return None, self.ANCHOR
            if not text.strip():
                return None, self.EMPTIED
            if text.strip() == inserted_text.strip():
                time.sleep(self.poll_interval)
                continue
            if self._is_edit(inserted_text, text, self.max_divergence):
                return text, self.EDITED
            # Поле не пустое, но наш текст в нём не угадывается: скорее всего
            # ушли в другое окно/поле (или наша вставка не прошла).
            return None, self.NOT_FOUND
        return None, self.TIMEOUT


def main() -> int:
    """CLI-проба: `python -m myflow.axtext` — прочитать поле в фокусе."""
    import sys
    reader = AxReader()
    if not reader.start():
        print(f"[ax] недоступно: {reader.last_error}")
        return 1
    try:
        for _ in range(3):
            sample = reader.read()
            if sample is None:
                print("[ax] нет ответа")
                break
            print(f"[ax] source={sample.source} pid={sample.pid} window={sample.window!r} "
                  f"reliable={sample.reliable}")
            print(f"[ax] text={sample.text[:200]!r}")
            time.sleep(0.5)
    finally:
        reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
