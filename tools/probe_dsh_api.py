"""Разведка локального API DSH: какие эндпоинты отвечают и что нужно для доступа.

У веб-профиля есть `/api` с «browser-trust fence»: скорее всего пускает запросы с
браузерным Origin. Если так — MyFLOW сможет вкладывать надиктованный текст прямо в
чат, без фокуса и без оконных фокусов (как мы делаем для OpenCode через консоль).
"""
from __future__ import annotations

import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:19387"
ASAR = Path(os.environ.get("DSH_ASAR", ""))   # см. peek_dsh_api.py
PATHS = ("/", "/api", "/api/health", "/api/status", "/api/sessions",
         "/api/session", "/api/state", "/api/events", "/api/workspace")


def probe(path: str, origin: str | None) -> str:
    req = urllib.request.Request(BASE + path)
    if origin:
        req.add_header("Origin", origin)
        req.add_header("Referer", origin + "/")
    req.add_header("Accept", "application/json, text/html")
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            body = resp.read(200)
            return f"HTTP {resp.status} | {body[:70]!r}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return f"ошибка: {type(exc).__name__}"


def main() -> int:
    print("  --- без Origin ---")
    for path in PATHS:
        print(f"    {path:<16} {probe(path, None)}")
    print("\n  --- с браузерным Origin ---")
    for path in PATHS:
        print(f"    {path:<16} {probe(path, BASE)}")

    if ASAR and ASAR.exists():
        print("\n  --- строки '/api/...' из asar ---")
        data = ASAR.read_bytes()
        found: dict[str, int] = {}
        for m in re.finditer(rb"[\x20-\x7e]{2,80}/api/[\x20-\x7e]{0,60}", data):
            text = m.group(0).decode("latin-1", "replace").strip().strip('"\'`,;')
            if len(text) < 60 and "api/" in text:
                found[text] = found.get(text, 0) + 1
        for text, count in sorted(found.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    x{count:<4} {text[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
