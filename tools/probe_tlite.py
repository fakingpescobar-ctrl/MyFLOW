"""Проба T-lite как кандидата в учителя: качество полировки на наших парах.

T-lite — 8B (Q4 ~4.7 ГБ), поэтому 4B на время пробы не поднимаем: вместе они не
влезут в 10 ГБ. Сравниваем с уже измеренными цифрами 4B на тех же парах.

Запускает T-lite на порту 8087 (специально НЕ 8083, чтобы стенд не подменил стиль
на postproc — T-lite нашему промпту не обучен, ему нужен plain).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_postproc import distance, light_clean, load_pairs, run_llm  # noqa: E402
from myflow.config import SettingsStore                              # noqa: E402

MODEL = ROOT / "data" / "train" / "tlite" / "T-lite-instruct-0.1.Q4_K_M.gguf"
PORT = 8087


def wait_health(url: str, seconds: float = 180.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/models", timeout=2.0).ok:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return False


def main() -> int:
    if not MODEL.exists():
        print(f"нет модели: {MODEL}")
        return 1
    pairs_file = ROOT / "data" / "train" / "final" / "postproc_val.jsonl"
    if not pairs_file.exists():
        pairs_file = ROOT / "data" / "train" / "postproc_val.jsonl"
    pairs = load_pairs(pairs_file)
    print(f"  пар: {len(pairs)} | модель: {MODEL.name}")

    server = ROOT / "bin" / "llama-cpp" / "llama-server.exe"
    log = ROOT / "data" / "train" / "tlite_probe.log"
    print("  поднимаю T-lite на :8087 (это ~30-60 с для 8B Q4)...", flush=True)
    with open(log, "ab") as fh:
        proc = subprocess.Popen(
            [str(server), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
             "-c", "8192", "-np", "1", "-ngl", "99", "-fa", "on", "--jinja"],
            stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    try:
        url = f"http://127.0.0.1:{PORT}/v1"
        if not wait_health(url):
            print(f"  T-lite не поднялся, смотри {log.name}")
            return 1
        print("  T-lite готов", flush=True)
        print(f"  сырой ASR      среднее "
              f"{sum(distance(p['dst'], p['src']) for p in pairs) / len(pairs):.3f}")
        print(f"  light_clean    среднее "
              f"{sum(distance(p['dst'], light_clean(p['src'])[0]) for p in pairs) / len(pairs):.3f}")
        # ВАЖНО: сначала стиль plain — он написан под Qwen3 (с нашими правилами и
        # примерами). Чужая модель на нём «разговаривается»: первый прогон дал 2.783
        # при 0.098 у сырого текста, то есть это была проверка не модели, а промпта.
        # Поэтому меряем и с короткой универсальной инструкцией (postproc).
        for style in ("plain", "postproc"):
            dists, lat = run_llm(pairs, url, style=style)
            mean = sum(dists) / len(dists)
            exact = sum(1 for d in dists if d == 0) / len(dists) * 100
            lat_sorted = sorted(lat)
            print(f"  T-lite ({style:<8}) среднее {mean:.3f} | точных {exact:.1f}% | "
                  f"задержка p50 {lat_sorted[len(lat_sorted) // 2]} мс")
        print("\n  для сравнения (те же пары, ранее измерено):")
        print("    4B Qwen3-4B     0.073-0.085 | точных 75-79% | 141-145 мс")
        print("    наша 0.6B       0.079-0.098 | точных 60-69% |  47- 55 мс")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            proc.kill()
        print("\n  T-lite остановлен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
