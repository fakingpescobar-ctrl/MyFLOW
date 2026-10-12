"""Лёгкое форматирование текста для чтения вслух + абзацы в окне.

Что выяснилось про файл главы: он уже разбит правильно — 188 строк, каждая строка
это абзац или реплика диалога, лишних пробелов нет. Поэтому «форматировать» текст
почти не нужно: ломало чтение ОКНО, которое склеивало все строки в одну простыню.

Что делает этот скрипт:
  * сохраняет копию исходника (.orig.txt), чтобы эталон никогда не терялся;
  * вокруг разделителей сцен (***) ставит пустые строки — их видно глазом;
  * убирает случайные хвостовые пробелы и двойные пробелы;
  * НЕ трогает слова: текст — эталон для замера WER, менять в нём буквы нельзя.
"""
from pathlib import Path
import shutil
import re

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "bench" / "private" / "klinok-ch1.txt"


def format_text(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        lines.append(line)

    # вокруг разделителей сцен — пустая строка, чтобы их было видно
    out: list[str] = []
    for line in lines:
        is_separator = bool(line) and set(line) == {"*"}
        if is_separator:
            if out and out[-1] != "":
                out.append("")
            out.append(line)
            out.append("")
            continue
        out.append(line)

    # склеиваем: без трёх пустых строк подряд, без пустых в начале/конце
    cleaned: list[str] = []
    for line in out:
        if line == "" and (not cleaned or cleaned[-1] == ""):
            continue
        cleaned.append(line)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned) + "\n"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Форматирование текста для чтения вслух")
    ap.add_argument("--text", type=Path, default=DEFAULT_PATH)
    PATH = ap.parse_args().text
    if not PATH.exists():
        print(f"  нет файла: {PATH}")
        return 1
    original = PATH.read_text(encoding="utf-8")
    backup = PATH.with_suffix(".orig.txt")
    if not backup.exists():
        shutil.copyfile(PATH, backup)
        print(f"  копия исходника: {backup.name}")

    formatted = format_text(original)
    PATH.write_text(formatted, encoding="utf-8")

    words_before = len(original.split())
    words_after = len(formatted.split())
    print(f"  строк: {len(original.splitlines())} -> {len(formatted.splitlines())}")
    print(f"  слов: {words_before} -> {words_after}", end="")
    print(" (слово в слово, менять нельзя — это эталон)" if words_before == words_after
          else "  ВНИМАНИЕ: число слов изменилось!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
