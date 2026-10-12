"""Скачать текст в общественном достоянии с Викитеки для теста «чтение вслух».

Почему не книга Перумова: она защищена авторским правом, а нам нужна целая глава —
это не короткая цитата. Поэтому берём авторов, чьи права давно истекли (Чехов,
Толстой, Гоголь и т.п.): для замера распознавания такой текст ничем не хуже, а
вопросов не вызывает.

    python tools/fetch_public_text.py --search "Хамелеон Чехов"
    python tools/fetch_public_text.py --title "Хамелеон (Чехов)" --out chehov.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "bench" / "private"
API = "https://ru.wikisource.org/w/api.php"


def call(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(f"{API}?{query}",
                                     headers={"User-Agent": "MyFLOW-bench/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def search(term: str, limit: int = 5) -> list[str]:
    data = call({"action": "query", "list": "search", "srsearch": term,
                 "srlimit": limit})
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def extract(title: str) -> str:
    data = call({"action": "query", "prop": "extracts", "explaintext": 1,
                 "redirects": 1, "titles": title})
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract") or ""
        if text:
            return text
    return ""


def cleanup(text: str) -> str:
    """Убираем служебные строки разметки, оставляем читаемую прозу."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # служебное: сноски, категории, ссылки на источники
        if re.match(r"^(Источник|Примечани|Категори|\[\[|\{\{)", stripped):
            continue
        if len(stripped) < 3:
            continue
        lines.append(re.sub(r"\s+", " ", stripped))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", help="что искать в Викитеке")
    ap.add_argument("--title", help="точное название страницы")
    ap.add_argument("--out", default=None, help="файл в data/bench/private/")
    args = ap.parse_args()

    title = args.title
    if not title:
        if not args.search:
            print("  нужно --search или --title", file=sys.stderr)
            return 1
        found = search(args.search)
        if not found:
            print("  ничего не найдено", file=sys.stderr)
            return 1
        print("  найдено:")
        for name in found:
            print(f"    {name}")
        title = found[0]
        print(f"  беру первый: {title}")

    text = cleanup(extract(title))
    if not text:
        print(f"  не удалось получить текст: {title}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / "public-text.txt"
    if not out.is_absolute():
        out = OUT_DIR / out
    out.write_text(text + "\n", encoding="utf-8")

    words = len(text.split())
    print(f"\n  страница: {title}")
    print(f"  слов: {words} | знаков: {len(text)}")
    print(f"  файл: {out}")
    print(f"  начало: {text[:100]!r}")
    if words < 200:
        print("  ВНИМАНИЕ: текст короткий для чтения вслух", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
