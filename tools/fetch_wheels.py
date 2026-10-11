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
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELS = ROOT / "data" / "train" / "wheels"


def pick_wheel(data: dict) -> dict | None:
    """Выбрать колесо под нашу систему: чистое python3 или cp313 win_amd64."""
    best = None
    for item in data.get("urls", []):
        name = item["filename"]
        if not name.endswith(".whl"):
            continue
        ok = (name.endswith("py3-none-any.whl")
              or name.endswith("py2.py3-none-any.whl")
              or ("cp313" in name and "win_amd64" in name)
              or ("cp313" in name and "win32" not in name and platform.system() != "Windows"))
        if not ok:
            continue
        score = (name.endswith("py3-none-any.whl"), "cp313" in name)
        if best is None or score > best[0]:
            best = (score, item)
    return best[1] if best else None


def fetch(name: str) -> tuple[bool, str]:
    target = name.split("==")[0]
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{target}/json", timeout=30) as r:
            data = json.load(r)
    except Exception as exc:  # noqa: BLE001
        return False, f"нет данных о {target}: {exc}"
    chosen = pick_wheel(data)
    if not chosen:
        return False, f"не нашёл подходящего колеса для {target} (только sdist?)"
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
