"""Окно чтения вслух: текст главы, живая подсветка и функции управления.

Зачем отдельное окно, а не просто текст в блокноте. Человек, читающий главу вслух,
должен видеть, где он остановился, и понимать, что распознаватель делает прямо
сейчас. Поэтому в окне есть:

  * сам текст главы крупно, с переносом по словам;
  * подсветка: прочитано (серым), ошибка распознавания (красным), текущее слово
    (жёлтым) — двигается по мере того, как MyFLOW распознаёт прочитанное;
  * функции кнопками и горячими клавишами:
      Заново      (или клавиша  R ) — начать чтение с начала
      Отчёт       (или клавиша  W ) — сохранить отчёт с цифрами по текущему проходу
      Пропустить  (или клавиша  S ) — пропустить фразу (распознали не то, читаю дальше)
      Стоп/выход  (или клавиша Esc)
  * строка состояния: прочитано слов и процент, сколько слов распознано с ошибкой,
    сколько времени идёт чтение, какой маршрут ASR сейчас работает.

Окно не забирает фокус: печатаешь в Блокноте или в приложении, а подсветка берётся
из истории MyFLOW (SQLite) — приложение менять не нужно.

    python -m myflow.readwindow --text data/bench/private/klinok-ch1.txt
    python -m myflow.readwindow --text ... --font 24
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.teleprompter import (Teleprompter, load_history,  # noqa: E402
                                  latest_record_id)

REPORTS = ROOT / "data" / "bench" / "reports"


def needs_scroll(line: int, first_visible: int, last_visible: int,
                 margin: int = 4) -> bool:
    """Нужно ли прокрутить текст, чтобы текущее слово было в удобной зоне."""
    comfort_top = first_visible + margin
    comfort_bottom = max(last_visible - margin, comfort_top)
    return not (comfort_top <= line <= comfort_bottom)


def scroll_target(line: int, total_lines: int, keep_from_top: int = 5) -> float:
    """Куда прокрутить: доля от начала текста (0.0 — самый верх)."""
    if total_lines <= 1:
        return 0.0
    target = max(0, min(line - keep_from_top, total_lines - 1))
    return target / total_lines


def paragraph_breaks(text: str) -> set[int]:
    """Индексы слов, после которых в тексте начинается новая строка-абзац.

    Файл главы уже разбит по строкам: каждая строка — абзац или реплика диалога.
    Окно раньше склеивало всё в одну простыню, и читать было невозможно. Теперь
    разбивка сохраняется: после последнего слова строки ставим перенос.
    """
    breaks: set[int] = set()
    index = 0
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        index += len(words)
        breaks.add(index - 1)
    return breaks


class ReadingSession:
    """Состояние прохода: текст, позиция, ошибки, время. Без окна — для тестов."""

    def __init__(self, text: str, model: str = "GigaAM") -> None:
        self.prompter = Teleprompter(text)
        self.breaks = paragraph_breaks(text)
        self.model = model
        self.started = time.monotonic()
        self.feed_count = 0
        self.skipped = 0

    @property
    def words(self) -> list[str]:
        return self.prompter.words

    @property
    def index(self) -> int:
        return self.prompter.index

    @property
    def errors(self) -> int:
        return len(self.prompter.mismatches)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def feed(self, heard: str) -> bool:
        """Скормить распознанную фразу. False — ничего нового не услышано."""
        if not heard or not heard.strip():
            return False
        before = self.prompter.index
        self.prompter.feed(heard)
        self.feed_count += 1
        return self.prompter.index > before

    def skip(self) -> None:
        """Пропустить одно слово (распознали не то, чтец идёт дальше)."""
        self.skipped += 1
        self.prompter.index = min(self.prompter.index + 1, len(self.prompter.words))

    def reset(self) -> None:
        self.prompter.index = 0
        self.prompter.mismatches = set()
        self.prompter.history = []
        self.started = time.monotonic()
        self.feed_count = 0
        self.skipped = 0

    def accuracy(self) -> float:
        """Доля прочитанных слов, распознанных верно."""
        if self.index <= 0:
            return 1.0
        return max(0.0, 1.0 - self.errors / self.index)

    def progress(self) -> float:
        return self.prompter.progress()

    def status(self) -> str:
        minutes, seconds = divmod(int(self.elapsed()), 60)
        return (f"прочитано {self.index}/{len(self.words)} слов "
                f"({self.progress() * 100:.0f}%) · ошибок {self.errors} "
                f"({(1 - self.accuracy()) * 100:.1f}%) · "
                f"фраз {self.feed_count} · {minutes:02d}:{seconds:02d} · ASR: {self.model}")

    def report(self) -> dict:
        words = self.words
        pairs = []
        for position in range(self.index):
            issue = position in self.prompter.mismatches
            context = " ".join(words[max(0, position - 4):position + 5])
            if issue:
                pairs.append({"word_index": position,
                              "expected": words[position],
                              "context": context})
        return {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "words_total": len(words),
            "words_read": self.index,
            "errors": self.errors,
            "accuracy": round(self.accuracy(), 4),
            "phrases_heard": self.feed_count,
            "skipped_manual": self.skipped,
            "seconds": round(self.elapsed(), 1),
            "problem_words": pairs[:200],
        }

    def save_report(self, directory: Path = REPORTS) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        out = directory / f"readwindow-{stamp}.json"
        out.write_text(json.dumps(self.report(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return out


class ReadingWindow:  # pragma: no cover - нужен экран
    """Окно поверх остальных: текст, подсветка, кнопки функций."""

    POLL_MS = 500

    def __init__(self, text: str, font_size: int, width: int, height: int,
                 model: str, highlight: bool = False) -> None:
        import tkinter as tk

        self.tk = tk
        # Подсветка по умолчанию ВЫКЛЮЧЕНА: при начитке она сбивает с толку.
        self.highlight = highlight
        self.session = ReadingSession(text, model=model)
        self.font_size = font_size
        self.word_positions: list[tuple[int, int]] = []

        self.root = tk.Tk()
        self.root.title("MyFLOW — чтение вслух")
        self.root.geometry(f"{width}x{height}+60+40")
        self.root.attributes("-topmost", True)

        self._build_widgets()
        self._render_text()
        self._bind_keys()

        # Следим только за НОВЫМИ фразами: иначе окно переигрывает всю историю
        # диктовок и уезжает в конец главы (симптом «всё время листает вниз»).
        self.last_id = latest_record_id()

        self.redraw()
        self.poll()

    def _build_widgets(self) -> None:
        tk = self.tk
        if not self.highlight:
            # Простой режим: только текст и подсказка про прокрутку.
            body = tk.Frame(self.root)
            body.pack(fill="both", expand=True, padx=16, pady=(14, 6))
            scroll = tk.Scrollbar(body)
            scroll.pack(side="right", fill="y")
            self.text = tk.Text(body, wrap="word",
                                font=("Georgia", self.font_size),
                                padx=28, pady=20, yscrollcommand=scroll.set,
                                highlightthickness=0, spacing1=6, spacing3=6,
                                relief="flat", borderwidth=0)
            self.text.pack(side="left", fill="both", expand=True)
            scroll.config(command=self.text.yview)
            hint = tk.Label(self.root, anchor="w", font=("Segoe UI", 10),
                            fg="#6b7280",
                            text="Прокрутка — колесо мыши или Page Down · "
                                 "закрыть — Esc")
            hint.pack(fill="x", padx=18, pady=(0, 10))
            return

        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=(8, 2))

        self.buttons = {}
        for key, label in (("reset", "Заново (R)"), ("report", "Отчёт (W)"),
                           ("skip", "Пропустить (S)"), ("quit", "Стоп (Esc)")):
            button = tk.Button(top, text=label, command=lambda k=key: self.command(k))
            button.pack(side="left", padx=3)
            self.buttons[key] = button

        self.model_label = tk.Label(top, text=f"маршрут ASR: {self.session.model}",
                                    font=("Segoe UI", 10))
        self.model_label.pack(side="right")

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10, pady=4)
        scroll = tk.Scrollbar(body)
        scroll.pack(side="right", fill="y")
        self.text = tk.Text(body, wrap="word", font=("Segoe UI", self.font_size),
                            padx=20, pady=14, yscrollcommand=scroll.set,
                            highlightthickness=0, spacing1=3, spacing3=3)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.text.yview)
        self.text.tag_configure("read", foreground="#6b7280")
        self.text.tag_configure("wrong", foreground="#dc2626", underline=True)
        self.text.tag_configure("current", background="#fde68a", foreground="#111827")

        self.hint = tk.Label(self.root, anchor="w", font=("Segoe UI", 9),
                             fg="#6b7280",
                             text="Читай по предложению с паузой. Подсветка идёт сама "
                                  "из истории MyFLOW.")
        self.hint.pack(fill="x", padx=10)
        self.status = tk.Label(self.root, anchor="w", font=("Consolas", 10))
        self.status.pack(fill="x", padx=10, pady=(0, 8))

    def _render_text(self) -> None:
        """Разложить слова по виджету, СОХРАНЯЯ строки-абзацы исходного текста."""
        self.text.delete("1.0", "end")
        self.word_positions = []
        words = self.session.words
        breaks = self.session.breaks
        for index, word in enumerate(words):
            start = self.text.index("end-1c")
            self.text.insert("end", word)
            end = self.text.index("end-1c")
            self.word_positions.append((start, end))
            if index == len(words) - 1:
                continue
            # конец строки-абзаца -> пустая строка для воздуха, иначе пробел
            self.text.insert("end", "\n\n" if index in breaks else " ")

    def _bind_keys(self) -> None:
        self.root.bind("<Escape>", lambda _e: self.command("quit"))
        if not self.highlight:
            return
        self.root.bind("r", lambda _e: self.command("reset"))
        self.root.bind("R", lambda _e: self.command("reset"))
        self.root.bind("w", lambda _e: self.command("report"))
        self.root.bind("W", lambda _e: self.command("report"))
        self.root.bind("s", lambda _e: self.command("skip"))
        self.root.bind("S", lambda _e: self.command("skip"))

    def command(self, key: str) -> None:
        if key == "quit":
            self.root.destroy()
            return
        if key == "reset":
            self.session.reset()
            self.redraw()
            return
        if key == "skip":
            self.session.skip()
            self.redraw()
            return
        if key == "report":
            path = self.session.save_report()
            self.hint.config(text=f"отчёт сохранён: {path}")

    def redraw(self) -> None:
        session = self.session
        self.text.tag_remove("read", "1.0", "end")
        self.text.tag_remove("wrong", "1.0", "end")
        self.text.tag_remove("current", "1.0", "end")

        follow = None
        for index, (start, end) in enumerate(self.word_positions):
            if index < session.index:
                tag = "wrong" if index in session.prompter.mismatches else "read"
                self.text.tag_add(tag, start, end)
            elif index == session.index:
                self.text.tag_add("current", start, end)
                follow = int(self.text.index(start).split(".")[0])

        if follow is not None:
            self._follow(follow)
        self.status.config(text=session.status())

    def _follow(self, line: int) -> None:
        """Держать текущее слово в удобной зоне, не дёргая текст каждый тик."""
        total = int(self.text.index("end-1c").split(".")[0])
        first_visible = int(self.text.index("@0,0").split(".")[0])
        last_visible = int(self.text.index(
            f"@0,{max(self.text.winfo_height(), 1)}").split(".")[0])
        if not needs_scroll(line, first_visible, last_visible):
            return
        self.text.yview_moveto(scroll_target(line, total))

    def poll(self) -> None:
        for record_id, heard in load_history(self.last_id):
            self.last_id = record_id
            if heard.strip():
                self.session.feed(heard)
        self.redraw()
        self.root.after(self.POLL_MS, self.poll)

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Окно чтения главы вслух")
    ap.add_argument("--text", required=True, help="файл с текстом главы")
    ap.add_argument("--font", type=int, default=22)
    ap.add_argument("--width", type=int, default=1150)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--model", default="GigaAM", help="какой маршрут ASR сейчас работает")
    ap.add_argument("--check", action="store_true", help="только проверить текст")
    ap.add_argument("--highlight", action="store_true",
                    help="режим с подсветкой прочитанного (по умолчанию выключен)")
    args = ap.parse_args()

    path = Path(args.text)
    if not path.exists():
        print(f"  нет файла: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")

    if args.check:
        session = ReadingSession(text, model=args.model)
        print(f"  слов в тексте: {len(session.words)}")
        print(f"  первые слова: {' '.join(session.words[:8])}")
        return 0

    return ReadingWindow(text, args.font, args.width, args.height, args.model,
                         highlight=args.highlight).run()


if __name__ == "__main__":
    raise SystemExit(main())
