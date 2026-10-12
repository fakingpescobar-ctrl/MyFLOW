"""Проверка доступа к локальному API DSH с ключом из .credentials.yaml.

Цель: понять, можно ли вкладывать надиктованный текст прямо в чат DeepSeek Harness
без фокуса окна (как мы делаем для OpenCode через консоль).

СЕКРЕТЫ НЕ ПЕЧАТАЕМ: выводим только ИМЕНА ключей и коды ответов.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:19387"
CRED = Path.home() / ".dsh" / ".credentials.yaml"
PATHS = ("/", "/api", "/api/health", "/api/sessions", "/api/session", "/api/state",
         "/api/events", "/api/status")


def key_names(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*:", line)
        if m:
            names.append(m.group(1))
    return names


def local_key(text: str) -> str | None:
    """Ключ локального API: ищем по имени STRATA_LOCAL_API_KEY / local."""
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]*(?:LOCAL|STRATA)[A-Za-z0-9_.\-]*)\s*:\s*(\S+)", line)
        if m:
            return m.group(2).strip().strip('"\'')
    return None


def probe(path: str, headers: dict[str, str]) -> str:
    req = urllib.request.Request(BASE + path)
    for name, value in headers.items():
        req.add_header(name, value)
    req.add_header("Accept", "application/json, text/html")
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            return f"HTTP {resp.status} | {resp.read(120)[:90]!r}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        return f"ошибка: {type(exc).__name__}"


def main() -> int:
    if not CRED.exists():
        print(f"  нет файла: {CRED}")
        return 1
    text = CRED.read_text(encoding="utf-8", errors="replace")
    print(f"  в .credentials.yaml ключи (только имена): {', '.join(key_names(text))}")

    key = local_key(text)
    print(f"  найден ключ локального API: {'да' if key else 'НЕТ'}")

    variants = {
        "без ключа": {},
        "Bearer": {"Authorization": f"Bearer {key}"} if key else {},
        "x-api-key": {"x-api-key": key} if key else {},
        "Bearer + Origin": ({"Authorization": f"Bearer {key}", "Origin": BASE}
                            if key else {}),
    }
    for name, headers in variants.items():
        if not headers:
            continue
        print(f"\n  --- {name} ---")
        for path in PATHS:
            print(f"    {path:<16} {probe(path, headers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
