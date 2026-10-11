"""Скачать модель с HuggingFace curl-ом (с докачкой и обрывом по скорости).

Зачем свой велосипед: `huggingface_hub` и pip в этом окружении **зависают** на
середине загрузки (процесс жив, соединений 0, файл не растёт, докачки нет).
curl с `-C -`, `--speed-limit/--speed-time` и `--retry` качает надёжно.

Запуск:
    .venv\\Scripts\\python.exe -X utf8 tools\\fetch_hf_model.py Qwen/Qwen3-1.7B
    .venv\\Scripts\\python.exe -X utf8 tools\\fetch_hf_model.py Qwen/Qwen3-1.7B --dir data\\train\\Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = ("config.json", "model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "generation_config.json")


def remote_size(repo: str, name: str) -> int:
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("content-length", 0))


def download(repo: str, name: str, target: Path) -> tuple[bool, str]:
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    try:
        want = remote_size(repo, name)
    except Exception as exc:  # noqa: BLE001
        return False, f"{name}: не смог узнать размер ({exc})"
    got = target.stat().st_size if target.exists() else 0
    if got == want and want > 0:
        return True, f"{name}: уже есть ({want / 2**20:.1f} МБ)"
    # --max-time обязателен: полностью вставшая передача НЕ ловится ни
    # --connect-timeout (соединение уже установлено), ни --speed-limit (скорость
    # считается по пришедшим данным). Без него curl висит бесконечно, а файл
    # остаётся нулевым — проверено на шардах 1.7 ГБ (12 минут, 0 байт).
    cmd = ["curl.exe", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5",
           "--retry-all-errors", "--connect-timeout", "30",
           "--speed-limit", "20480", "--speed-time", "20", "--max-time", "180",
           "-o", str(target), url]
    # при обрыве по скорости curl выходит с ненулевым кодом, но с уже скачанным
    # куском — повторяем, он продолжит с места
    for _ in range(60):
        subprocess.run(cmd, capture_output=True)
        now = target.stat().st_size if target.exists() else 0
        if now == want:
            return True, f"{name}: {want / 2**20:.1f} МБ"
        if now == got:
            break
        got = now
    now = target.stat().st_size if target.exists() else 0
    return False, f"{name}: скачано {now / 2**20:.1f} из {want / 2**20:.1f} МБ"


def main() -> int:
    ap = argparse.ArgumentParser(description="Скачать модель с HuggingFace (curl)")
    ap.add_argument("repo", help="например Qwen/Qwen3-1.7B")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    ap.add_argument("--copy-tokenizer-from", default=None,
                    help="каталог, откуда взять файлы токенизатора (у серии Qwen3 он общий)")
    args = ap.parse_args()

    name = args.repo.split("/")[-1]
    out = Path(args.dir) if args.dir else (ROOT / "data" / "train" / name)
    out.mkdir(parents=True, exist_ok=True)
    print(f"  качаю {args.repo} -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")

    tokenizer_files = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
    if args.copy_tokenizer_from:
        src = Path(args.copy_tokenizer_from)
        for f in tokenizer_files:
            if (src / f).exists():
                (out / f).write_bytes((src / f).read_bytes())
                print(f"    {f}: скопирован из {src.name}")
        args.files = [f for f in args.files if f not in tokenizer_files]

    ok_all = True
    for f in args.files:
        ok, note = download(args.repo, f, out / f)
        print(f"    {'ok ' if ok else 'НЕТ'} {note}", flush=True)
        ok_all = ok_all and ok

    # итог: читается ли модель
    if ok_all:
        try:
            cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
            print(f"  готова: {cfg.get('model_type')}, слоёв {cfg.get('num_hidden_layers')}, "
                  f"размер {cfg.get('hidden_size')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  конфиг не читается: {exc}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
