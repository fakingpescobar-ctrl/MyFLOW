"""Достать из app.asar строки, похожие на HTTP-маршруты и токены.

Нужно, чтобы понять, есть ли у DeepSeek Harness локальный API, через который можно
вложить надиктованный текст прямо в чат (как мы делаем для OpenCode через консоль).
Инструменты asar этот путь не видят, а сам архив — обычный файл: ищем в нём
печатаемые строки.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ASAR = Path(r"D:\DeepSeek Harness\resources\app.asar")
PATTERNS = {
    "маршруты": re.compile(rb"[\x20-\x7e]{4,120}(?:/api/|/v1/|/session|/message|/chat)[\x20-\x7e]{0,80}"),
    "методы": re.compile(rb"(?:app|router|server)\.(?:get|post|put|delete)\([\x20-\x7e]{0,80}"),
}


def main() -> int:
    if not ASAR.exists():
        print(f"нет файла: {ASAR}")
        return 1
    data = ASAR.read_bytes()
    print(f"  размер архива: {len(data) / 2**20:.0f} МБ")
    for name, pattern in PATTERNS.items():
        found: dict[str, int] = {}
        for m in pattern.finditer(data):
            text = m.group(0).decode("latin-1", "replace").strip()
            # отбрасываем мусор: слишком короткие и явно не пути
            if "/api/" in text or text.startswith(("app.", "router.", "server.")):
                found[text] = found.get(text, 0) + 1
        print(f"\n  --- {name}: {len(found)} уникальных ---")
        for text, count in sorted(found.items(), key=lambda kv: -kv[1])[:25]:
            print(f"    x{count:<4} {text[:118]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
