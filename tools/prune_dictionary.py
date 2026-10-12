"""Чистка автообученного словаря от ложных правил.

Откуда берутся ложные правила. MyFLOW наблюдает поле, куда вставил текст, и учит
замены «что было -> что осталось». Атрибуция часто ошибается (правка относится к
другому месту, приложение перерисовывает строку, курсор уезжает) — и в словарь
попадает, например, «она -> и» или «елай -> ну». Такие правила молча портят текст.

Правило чистки: цель замены — СЛУЖЕБНОЕ слово (предлог, союз, частица, местоимение,
форма «быть»). Настоящая правка никогда не превращает значащее слово в предлог, а
вот сломанная атрибуция — постоянно. Записи с цифрами, именами и специфичными
словами остаются.

    python tools/prune_dictionary.py --show     # показать, что предлагается убрать
    python tools/prune_dictionary.py --apply    # убрать
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "myflow.sqlite"

# Служебные слова: если целью замены стало такое слово, правило почти наверняка ложное.
FUNCTION_WORDS = {
    "и", "а", "но", "да", "же", "ли", "бы", "не", "ни", "в", "во", "на", "с", "со",
    "к", "ко", "по", "о", "об", "от", "до", "из", "за", "над", "под", "при", "про",
    "для", "без", "у", "через", "между", "или", "либо", "то", "как", "что", "чем",
    "это", "этот", "эта", "эти", "тот", "та", "те", "он", "она", "оно", "они",
    "я", "ты", "мы", "вы", "ну", "вот", "там", "тут", "здесь", "уже", "ещё", "еще",
    "был", "была", "было", "были", "есть", "будет", "быть", "идёт", "идет",
}


def is_learned(entry: dict) -> bool:
    return (entry.get("source") or "") == "learned"


def looks_false(entry: dict) -> bool:
    """Ложное правило: целью стало служебное слово."""
    target = (entry.get("target_word") or "").strip().lower().strip(".,!?;:")
    source = (entry.get("source_word") or "").strip()
    if not target or not source:
        return False
    if any(ch.isdigit() for ch in target) or any(ch.isdigit() for ch in source):
        return False                      # номера портов и прочее — не трогаем
    if len(source) <= 2 and source.lower() not in FUNCTION_WORDS:
        return False                      # обрубок слова: чистим отдельно, ниже
    return target in FUNCTION_WORDS


def looks_broken(entry: dict) -> bool:
    """Обрубок: цель в одно-два слова, источник явно из середины слова."""
    source = (entry.get("source_word") or "").strip().lower()
    target = (entry.get("target_word") or "").strip().lower()
    if not source or not target:
        return False
    return (len(source) <= 5 and len(target) <= 3
            and target in FUNCTION_WORDS and source not in FUNCTION_WORDS)


def find_false(con: sqlite3.Connection) -> list[dict]:
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("select * from dictionary")]
    return [r for r in rows if (looks_false(r) or looks_broken(r))
            and (r.get("source") or "") == "learned"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True) if not args.apply \
        else sqlite3.connect(DB)
    if not args.apply:
        con.row_factory = sqlite3.Row

    bad = find_false(con)
    total = con.execute("select count(*) from dictionary").fetchone()[0]
    print(f"  всего записей: {total}, ложных по правилу: {len(bad)}")
    for row in bad:
        print(f"    убрать: {row['source_word']!r} -> {row['target_word']!r} "
              f"(source={row.get('source')})")

    if args.apply and bad:
        for row in bad:
            con.execute("delete from dictionary where id=?", (row["id"],))
        con.commit()
        print(f"  убрано: {len(bad)}")
    con.close()
    if not args.apply and bad:
        print("\n  чтобы убрать: python tools/prune_dictionary.py --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
