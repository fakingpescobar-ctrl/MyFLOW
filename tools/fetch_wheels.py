"""Скачать колёса пакетов curl-ом и поставить их офлайн.

Зачем: в этом окружении **pip зависает** на скачивании (`0 соединений`, файл не
растёт) — и на 2.5 ГБ torch, и на 12 МБ transformers. curl качает нормально и умеет
докачку, поэтому схема такая:

1. спрашиваем у PyPI JSON-API список файлов пакета (это крошечный запрос);
2. выбираем подходящее колесо (py3-none-any или cp313-win_amd64);
3. качаем его через curl с докачкой в `data/train/wheels/`;
4. ставим всё разом: `pip install --no-index --find-links <каталог>`.

Запуск:
    .venv\\Scripts\\python.exe -X utf8 tools\\fetch_wheels.py transformers peft ...
    .venv\\Scripts\\python.exe -X utf8 tools\\fetch_wheels.py --install --target <venv-python>
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELS = ROOT / "data" / "train" / "wheels"


def pick_wheel(data: dict) -> dict | None:
    """Выбрать колесо под нашу систему.

    Тонкости, на которых сборщик спотыкался:
      * `tokenizers` и `safetensors` выходят как **abi3** (cp39-abi3-win_amd64) —
        они подходят для любого Python 3, но не содержат "cp313";
      * для части пакетов есть **free-threaded** колесо (cp313t) — оно для Python
        без GIL и в обычном 3.13 не годится;
      * чистые python-колёса (py3-none-any) подходят всегда.
    """
    best = None
    for item in data.get("urls", []):
        name = item["filename"]
        if not name.endswith(".whl"):
            continue
        if "cp313t" in name or "-pp3" in name:
            continue                       # free-threaded / PyPy — не наш случай
        if "win32" in name and "win_amd64" not in name:
            continue                      # 32-битное колесо в 64-битный Python не встанет
        if name.endswith("py3-none-any.whl") or name.endswith("py2.py3-none-any.whl"):
            score = 3
        elif "cp313-cp313-win_amd64" in name:
            score = 2                     # точное совпадение версии
        elif "abi3-win_amd64" in name:
            score = 1                     # abi3: работает на любом Python 3 x64
        else:
            continue
        if best is None or score > best[0]:
            best = (score, item)
    return best[1] if best else None


def fetch(name: str) -> tuple[bool, str]:
    """`name`, `name==версия` или `name>=версия`.

    Точная версия важна (torch 2.6.0 требует sympy==1.13.1), а спецификаторы вида
    `>=` просто означают «любая подходящая» — берём последнюю.
    """
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=|<=|~=|!=|>|<)?\s*(.*)$", name.strip())
    if not m:
        return False, f"не понял имя пакета: {name}"
    target, op, version = m.group(1), m.group(2) or "", (m.group(3) or "").strip()
    url = (f"https://pypi.org/pypi/{target}/{version}/json" if op == "==" and version
           else f"https://pypi.org/pypi/{target}/json")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except Exception as exc:  # noqa: BLE001
        return False, f"нет данных о {name}: {exc}"
    chosen = pick_wheel(data)
    if not chosen:
        return False, f"не нашёл подходящего колеса для {name} (только sdist?)"
    out = WHEELS / chosen["filename"]
    if out.exists() and out.stat().st_size == chosen["size"]:
        return True, f"уже есть: {chosen['filename']}"
    cmd = ["curl.exe", "-L", "-C", "-", "--retry", "40", "--retry-delay", "5",
           "--retry-all-errors", "--connect-timeout", "30",
           "-o", str(out), chosen["url"]]
    for attempt in range(3):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if out.exists() and out.stat().st_size == chosen["size"]:
            return True, f"{chosen['filename']} ({chosen['size'] / 2**20:.1f} МБ)"
        if proc.returncode == 0 and out.exists():
            return True, f"{chosen['filename']} (~{out.stat().st_size / 2**20:.1f} МБ)"
    return False, f"curl не смог скачать {chosen['filename']}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Колёса для учебного окружения")
    ap.add_argument("packages", nargs="*", help="имена пакетов на PyPI")
    ap.add_argument("--install", action="store_true",
                    help="поставить всё скачанное в --target (по умолчанию .venv-train)")
    ap.add_argument("--target", default=str(ROOT / ".venv-train" / "Scripts" / "python.exe"))
    args = ap.parse_args()

    WHEELS.mkdir(parents=True, exist_ok=True)
    if args.packages:
        for name in args.packages:
            ok, note = fetch(name)
            print(f"  {'ok ' if ok else 'НЕТ'} {name:<18} {note}", flush=True)

    if args.install:
        wheels = sorted(str(p) for p in WHEELS.glob("*.whl"))
        if not wheels:
            print("нечего ставить: каталог колёс пуст")
            return 1
        print(f"\nставлю {len(wheels)} колёс офлайн в {args.target}")
        cmd = [args.target, "-m", "pip", "install", "--no-index",
               "--find-links", str(WHEELS), "--no-cache-dir", "--progress-bar", "off",
               *wheels]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        tail = (proc.stdout or "")[-1500:]
        print(tail)
        if proc.returncode != 0:
            print("STDERR:", (proc.stderr or "")[-1200:])
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
