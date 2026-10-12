"""Окно-подсказка для теста «чтение вслух»: текст и подсветка прочитанного.

Зачем. Без подсказки человек, читая главу вслух, теряет место: не понимает, где
остановился, начинает повторяться или пропускает куски — и замер WER съезжает по
вине чтеца, а не распознавателя ✗.

Как работает. Окно показывает текст крупно и красит слова по мере того, как MyFLOW
распознаёт прочитанное:
    * прочитано верно   — обычным «прочитанным» цветом;
    * распознано с ошибкой — красным (живой WER, видно сразу);
    * текущая позиция   — подчёркнуто и подсвечено.

Окно НЕ забирает фокус: ввод идёт в Notepad/поле, а подсветка обновляется сама из
истории MyFLOW (SQLite), и менять в приложении ничего не нужно.

Запуск (отдельным процессом, рядом с MyFLOW):
    python -m myflow.teleprompter --text data/bench/private/perumov-ch1.txt
    python -m myflow.teleprompter --text ... --font 26 --width 1100

Проверка логики без окна (для тестов): align_from().
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "myflow.sqlite"

PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WORD = re.compile(r"\S+")


def normalize_word(word: str) -> str:
    return PUNCT.sub("", unicodedata.normalize("NFKC", word).lower().replace("ё", "е"))


def split_words(text: str) -> list[str]:
    return WORD.findall(text)


def _cost(reference: list[str], heard: list[str]) -> int:
    """Расстояние Левенштейна по нормализованным словам."""
    previous = list(range(len(heard) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i] + [0] * len(heard)
        for j, hyp in enumerate(heard, start=1):
            same = normalize_word(ref) == normalize_word(hyp)
            current[j] = min(previous[j] + 1,          # пропуск в речи
                             current[j - 1] + 1,       # лишнее слово
                             previous[j - 1] + (0 if same else 1))
        previous = current
    return previous[-1]


def align_from(words: list[str], heard: str, index: int, lookahead: int = 40
               ) -> tuple[int, set[int]]:
    """Куда дошёл чтец: (новая позиция, индексы слов, распознанных с ошибкой).

    Ищем, где в тексте лучше всего стыкуется только что услышанная фраза, начиная
    с текущей позиции (немного вперёд — человек мог повторить или перескочить).
    """
    heard_words = split_words(heard)
    if not heard_words or index >= len(words):
        return index, set()

    best_start, best_cost = index, None
    span = min(len(heard_words) + 3, index + lookahead) - index
    for start in range(index, min(len(words), index + 25)):
        length = min(len(heard_words) + 3, len(words) - start)
        if length <= 0:
            break
        cost = _cost(words[start:start + length], heard_words)
        if best_cost is None or cost < best_cost:
            best_start, best_cost = start, cost
    _ = span

    length = min(len(heard_words) + 3, len(words) - best_start)
    slice_words = words[best_start:best_start + length]
    # выравниваем ещё раз, чтобы понять, какие слова разошлись
    mismatches: set[int] = set()
    i, j, consumed = 0, 0, 0
    previous = list(range(len(heard_words) + 1))
    rows: list[list[int]] = [previous]
    for ref in slice_words:
        current = [rows[-1][0] + 1] + [0] * len(heard_words)
        for k, hyp in enumerate(heard_words, start=1):
            same = normalize_word(ref) == normalize_word(hyp)
            current[k] = min(rows[-1][k] + 1, current[k - 1] + 1,
                             rows[-1][k - 1] + (0 if same else 1))
        rows.append(current)
    # Обратный проход: считаем, сколько слов ЭТАЛОНА реально сопоставлено речи.
    i, j = len(slice_words), len(heard_words)
    matched_ref = 0
    while i > 0 or j > 0:
        same = (i > 0 and j > 0 and normalize_word(slice_words[i - 1])
                == normalize_word(heard_words[j - 1]))
        if i > 0 and j > 0 and rows[i][j] == rows[i - 1][j - 1] + (0 if same else 1):
            if not same:
                mismatches.add(best_start + i - 1)
            matched_ref += 1                 # слово эталона сопоставлено прочитанному
            i, j = i - 1, j - 1
        elif i > 0 and rows[i][j] == rows[i - 1][j] + 1:
            # Слово эталона пропущено. Если речь ещё не кончилась — это ошибка
            # распознавания (не расслышал слово), и её надо показать красным.
            if j > 0:
                mismatches.add(best_start + i - 1)
            i -= 1
        else:
            j -= 1                           # лишнее слово речи

    new_index = best_start + matched_ref
    # Ошибки засчитываем только внутри сопоставленного куска (хвост «на вырост»
    # к делу не относится и давал сдвинутые индексы).
    mismatches = {index for index in mismatches if index < new_index}
    return min(new_index, len(words)), mismatches


class Teleprompter:
    """Вся логика подсветки без окна — так её можно проверять тестами."""

    def __init__(self, text: str) -> None:
        self.words = split_words(text)
        self.index = 0
        self.mismatches: set[int] = set()
        self.history: list[int] = []

    def feed(self, heard: str) -> tuple[int, set[int]]:
        """Скормить распознанную фразу; вернуть (позиция, ошибочные слова)."""
        new_index, wrong = align_from(self.words, heard, self.index)
        self.mismatches |= wrong
        if new_index > self.index:
            self.history.append(new_index)
            self.index = new_index
        return self.index, wrong

    def progress(self) -> float:
        return self.index / len(self.words) if self.words else 0.0


def load_history(start_id: int = 0, limit: int = 200) -> list[tuple[int, str]]:
    """Новые записи истории: [(id, распознанный текст)]."""
    if not DB.exists():
        return []
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = list(con.execute(
            "select id, coalesce(asr_text, '') from history "
            "where id > ? order by id limit ?", (start_id, limit)))
    finally:
        con.close()
    return rows


def run_window(text_path: Path, font: int, width: int) -> int:  # pragma: no cover
    """Окно-подсказка (нужен экран; логика выше проверяется тестами)."""
    import tkinter as tk

    prompter = Teleprompter(text_path.read_text(encoding="utf-8"))
    root = tk.Tk()
    root.title("MyFLOW — чтение вслух")
    root.geometry(f"{width}x520+80+60")
    root.attributes("-topmost", True)

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)
    scroll = tk.Scrollbar(frame)
    scroll.pack(side="right", fill="y")
    widget = tk.Text(frame, wrap="word", font=("Segoe UI", font), padx=24, pady=18,
                     yscrollcommand=scroll.set, highlightthickness=0)
    widget.pack(side="left", fill="both", expand=True)
    scroll.config(command=widget.yview)

    widget.tag_configure("read", foreground="#6b7280")
    widget.tag_configure("wrong", foreground="#dc2626", underline=True)
    widget.tag_configure("current", background="#fde68a", foreground="#111827")
    widget.tag_configure("todo", foreground="#111827")
    widget.insert("1.0", " ".join(prompter.words))

    status = tk.Label(root, anchor="w", font=("Segoe UI", 11))
    status.pack(fill="x")

    def redraw() -> None:
        widget.tag_remove("read", "1.0", "end")
        widget.tag_remove("wrong", "1.0", "end")
        widget.tag_remove("current", "1.0", "end")
        position = 0
        for index, word in enumerate(prompter.words):
            start = f"1.0+{position}c"
            end = f"1.0+{position + len(word)}c"
            if index < prompter.index:
                widget.tag_add("wrong" if index in prompter.mismatches else "read",
                               start, end)
            elif index == prompter.index:
                widget.tag_add("current", start, end)
                widget.see(start)
            position += len(word) + 1
        status.config(text=f"прочитано {prompter.index} из {len(prompter.words)} слов "
                           f"({prompter.progress() * 100:.0f}%) · ошибок "
                           f"{len(prompter.mismatches)} · Esc — выход")

    last_id = 0
    for record_id, _text in load_history(0, 1):
        last_id = max(0, record_id - 1)

    def poll() -> None:
        nonlocal last_id
        for record_id, heard in load_history(last_id):
            last_id = record_id
            if heard.strip():
                prompter.feed(heard)
        redraw()
        root.after(500, poll)

    root.bind("<Escape>", lambda _event: root.destroy())
    redraw()
    poll()
    root.mainloop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Окно-подсказка для чтения вслух")
    ap.add_argument("--text", required=True, help="эталонный текст (глава)")
    ap.add_argument("--font", type=int, default=22)
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--check", action="store_true", help="только проверка текста, без окна")
    args = ap.parse_args()

    path = Path(args.text)
    if not path.exists():
        print(f"  нет файла: {path}", file=sys.stderr)
        return 1
    if args.check:
        prompter = Teleprompter(path.read_text(encoding="utf-8"))
        print(f"  слов в тексте: {len(prompter.words)}")
        print(f"  первые слова: {' '.join(prompter.words[:8])}")
        return 0
    return run_window(path, args.font, args.width)


if __name__ == "__main__":
    raise SystemExit(main())
