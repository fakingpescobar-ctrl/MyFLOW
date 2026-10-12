"""MyFLOW — инжектор текста в чат DeepSeek Harness (приложение, без фокуса).

Как это работает. У OpenCode мы писали клавиши прямо в консольный буфер
(`WriteConsoleInputW`) — там чат живёт в консоли. DSH — десктопное приложение
(Electron), поэтому консольный путь не подходит, а API приложения закрыт
браузерной сессией (проверено: 401 на всё).

Штатный путь — **плагин внутри DSH** (`dsh-plugin-myflow`): он живёт в приложении и
вызывает API агента (`agent.followup(...)`), то есть вкладывает реплику в текущую
сессию ровно так, как это делает человек, печатающий в строке ввода и нажимающий
Enter. Связь между MyFLOW и плагином — файл-инбокс в домашнем каталоге DSH:

    $DSH_HOME/myflow-inbox.jsonl   строки {"text": "...", "submit": true}

MyFLOW дописывает строку (атомарно, одной записью), плагин её читает и вкладывает.
Никаких портов, токенов и перехвата фокуса.

Интерфейс совпадает с ClipboardPaste (метод paste(text)) — main подставляет нужную
реализацию по настройке insert_method.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def dsh_home() -> Path:
    """Домашний каталог DSH: $DSH_HOME, иначе ~/.dsh (как в самом харнессе)."""
    env = os.environ.get("DSH_HOME")
    if env:
        return Path(env)
    return Path.home() / ".dsh"


def inbox_path() -> Path:
    return dsh_home() / "myflow-inbox.jsonl"


def _log_warn(message: str) -> None:
    try:
        sys.stderr.write(f"[dsh_inject] {message}\n")
    except Exception:
        pass


class DshInjector:
    """Вставка текста в текущую сессию DeepSeek Harness через плагин.

    Плагин сам решает, когда забрать строку: если активной сессии нет, она просто
    останется в инбоксе и уйдёт при следующем опросе (данные не теряются).
    """

    # Буфер обмена не используется: авто-обучение из правок в этом режиме
    # невозможно (и main его не запускает) — как и у opencode-инжектора.
    can_read = False

    def __init__(self, inbox: Path | None = None) -> None:
        self.inbox = Path(inbox) if inbox is not None else inbox_path()

    def read(self) -> str:
        return ""

    def paste(self, text: str) -> bool:
        """Дописать реплику в инбокс DSH. True — записано, False — не вышло."""
        if not text or not text.strip():
            return False
        try:
            self.inbox.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            # Одна запись целиком: плагин читает только целые строки, поэтому
            # недописанная строка просто подождёт следующего опроса.
            with open(self.inbox, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            return True
        except Exception as exc:  # noqa: BLE001
            _log_warn(f"не смог записать в инбокс DSH ({self.inbox}): {exc!r}")
            return False
