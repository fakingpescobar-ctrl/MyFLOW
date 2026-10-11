"""A/B: кандидатная модель постпроцессора против текущего Qwen3-4B.

Поднимает переданный GGUF на отдельном порту (8083), прогоняет его на тех же
парах, что и боевую модель, и печатает таблицу. Итог этапа 3 по цели.

Запуск:
    .venv\\Scripts\\python.exe -X utf8 tools\\ab_postproc.py ^
        --model models\\myflow-postproc-qwen3-06b-q8_0.gguf
Опции:
    --pairs   набор пар (по умолчанию валидация: data\\train\\postproc_val.jsonl)
    --also-all  дополнительно прогнать по всей истории (data\\train\\all_real_pairs.jsonl)
    --port    порт для кандидата (по умолчанию 8083)
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_postproc import distance, light_clean, load_pairs, run_llm  # noqa: E402
from myflow.config import SettingsStore                               # noqa: E402


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))]


def row(name: str, dists: list[float], lat: list[int] | None = None) -> str:
    exact = sum(1 for d in dists if d == 0) / max(1, len(dists)) * 100
    line = (f"  {name:<26} p50 {pct(dists, .5):.3f} | среднее "
            f"{statistics.fmean(dists):.3f} | точных {exact:>4.1f}%")
    if lat:
        line += f" | задержка p50 {pct([float(x) for x in lat], .5):>5.0f} мс"
    return line


def wait_health(url: str, seconds: float = 90.0) -> bool:
    import requests
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/models", timeout=1.5).ok:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B постпроцессора против 4B")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "train" / "postproc_val.jsonl")
    ap.add_argument("--also-all", action="store_true")
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()

    model = args.model if args.model.is_absolute() else ROOT / args.model
    if not model.exists():
        print(f"нет модели: {model}")
        return 1
    server = ROOT / "bin" / "llama-cpp" / "llama-server.exe"
    if not server.exists():
        print(f"нет сервера: {server}")
        return 1

    url = f"http://127.0.0.1:{args.port}/v1"
    log = ROOT / "data" / f"ab_postproc_{args.port}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"поднимаю кандидата на :{args.port} ({model.name})...")
    with open(log, "ab") as fh:
        proc = subprocess.Popen(
            [str(server), "-m", str(model), "--host", "127.0.0.1",
             "--port", str(args.port), "-c", str(args.ctx), "-np", "1",
             "-ngl", "99", "-fa", "on", "--jinja"],
            stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    try:
        if not wait_health(url):
            print(f"кандидат не поднялся, смотри {log}")
            return 1

        sets = [("валидация", args.pairs)]
        if args.also_all:
            sets.append(("вся история", ROOT / "data" / "train" / "all_real_pairs.jsonl"))
        for title, path in sets:
            pairs = load_pairs(path)
            if not pairs:
                continue
            print(f"\n=== {title} ({len(pairs)} пар) ===")
            print(row("сырой ASR", [distance(p["dst"], p["src"]) for p in pairs]))
            print(row("light_clean", [distance(p["dst"], light_clean(p["src"])[0])
                                      for p in pairs]))
            # Рантайм-тумблер polish.enabled здесь не важен: мерим САМУ модель.
            base_url = SettingsStore().get().polish.url
            if wait_health(base_url, 3.0):
                dists, lat = run_llm(pairs, base_url)
                print(row("Qwen3-4B (текущая)", dists, lat))
            else:
                print("  Qwen3-4B (текущая):        сервер :8082 не отвечает — пропуск")
            dists, lat = run_llm(pairs, url)
            print(row(f"КАНДИДАТ :{args.port}", dists, lat))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
        print("\nкандидат остановлен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
