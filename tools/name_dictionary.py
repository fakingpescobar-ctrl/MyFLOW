"""Словарь имён: то, что GigaAM слышит неверно, и как это должно быть.

Зачем. Замер на живой начитке показал: обычные слова GigaAM слышит чисто, а спотыкается
на именах собственных («Онфим» -> «Анфим», «Кицум» -> «Китсум»). Словарь MyFLOW уходит
в подсказку полировщику, поэтому имена в нём — самая дешёвая правка качества.

Важно, что здесь НЕ добавляется: замены обычных слов («глотка» -> «глоточка»,
«племени» -> «племен»). Это реальные слова, и глобальная замена испортила бы
нормальный текст — в словарь идут только однозначные имена.

Все записи помечены source='names', поэтому их легко найти, показать и удалить:
    python tools/name_dictionary.py --show
    python tools/name_dictionary.py --add
    python tools/name_dictionary.py --remove           # убрать только наши записи
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "myflow.sqlite"
SOURCE = "names"

# Имена из книги: (как слышит ASR, как должно быть)
MISHEARD = [
    ("анфим", "Онфим"), ("анфима", "Онфима"), ("онфим", "Онфим"),
    ("китсум", "Кицум"), ("китцум", "Кицум"), ("кицун", "Кицум"),
    ("кицума", "Кицума"),
    ("эвилин", "Эвелин"), ("эвелина", "Эвелина"),
    ("агата", "Агата"), ("троша", "Троша"), ("троши", "Троши"),
    ("нодлик", "Нодлик"), ("таньша", "Таньша"), ("еремей", "Еремей"),
    ("хвалин", "Хвалин"), ("хвалина", "Хвалина"),
    ("остраг", "Остраг"), ("друнг", "Друнг"),
    ("кобальдов", "кобольдов"), ("кобальды", "кобольды"),
]

# Просто правильные написания: чтобы полировщик держал их в голове
CORRECT = ["Агата", "Троша", "Нодлик", "Таньша", "Еремей", "Онфим", "Кицум",
           "Эвелин", "Хвалин", "Остраг", "Друнг", "Дадрроунтгот", "иммельсторн"]


def connect(readonly: bool = False, db: Path | None = None) -> sqlite3.Connection:
    path = Path(db) if db is not None else DB
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row      # иначе строки приходят кортежами
    return con


def add_names(con: sqlite3.Connection, scope: str = "") -> tuple[int, int]:
    """Добавить записи словаря. Возвращает (добавлено, уже было)."""
    added = skipped = 0
    for heard, correct in MISHEARD:
        exists = con.execute(
            "select 1 from dictionary where source_word=? and app_mask=?",
            (heard, scope)).fetchone()
        if exists:
            skipped += 1
            continue
        con.execute(
            "insert into dictionary (source_word, target_word, source, app_mask) "
            "values (?, ?, ?, ?)", (heard, correct, SOURCE, scope))
        added += 1
    con.commit()
    return added, skipped


def remove_names(con: sqlite3.Connection) -> int:
    cursor = con.execute("delete from dictionary where source=?", (SOURCE,))
    con.commit()
    return cursor.rowcount


def show(con: sqlite3.Connection) -> None:
    rows = list(con.execute(
        "select source_word, target_word, source from dictionary order by source, id"))
    ours = [r for r in rows if r["source"] == SOURCE]
    others = [r for r in rows if r["source"] != SOURCE]
    print(f"  наши имена ({len(ours)}):")
    for row in ours[:40]:
        print(f"    {row['source_word']} -> {row['target_word']}")
    print(f"\n  прочие записи ({len(others)}):")
    for row in others[:20]:
        print(f"    {row['source_word']!r} -> {row['target_word']!r} "
              f"source={row['source']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="store_true")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--scope", default="", help="ограничить словарь окном приложения")
    args = ap.parse_args()

    if not DB.exists():
        print(f"  нет базы: {DB}")
        return 1

    if args.add:
        con = connect()
        added, skipped = add_names(con, args.scope)
        con.close()
        print(f"  добавлено имён: {added} (уже было {skipped})")
    if args.remove:
        con = connect()
        count = remove_names(con)
        con.close()
        print(f"  убрано наших имён: {count}")
    if args.show or not (args.add or args.remove):
        show(connect(readonly=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
