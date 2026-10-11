"""Замер постпроцессора на парах «сырой ASR → эталон» (и сравнение подходов).

Считает одну и ту же метрику для нескольких вариантов обработки:
  * `raw`        — ничего не делаем (эталон для сравнения);
  * `light_clean`— наша детерминированная чистка (быстрый путь, 0 мс LLM);
  * `llm`        — текущая полировка локальной моделью через llama-server.

Инструмент не привязан к конкретной модели: URL/порт задаются ключами, поэтому
им же будем мерить дообученную 0.6B, подняв её на другом порту.

Запуск:
  python tools/eval_postproc.py --pairs data/train/postproc_val.jsonl
  python tools/eval_postproc.py --pairs ... --llm --url http://127.0.0.1:8083/v1
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from myflow.lightclean import light_clean                 # noqa: E402
from myflow.polish import PolishClient                    # noqa: E402
from myflow.config import SettingsStore                   # noqa: E402

_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def words(text: str) -> list[str]:
    return [w.replace("ё", "е") for w in _WORD.findall((text or "").lower())]


def distance(ref: str, hyp: str) -> float:
    """Словесное расстояние (1.0 = ничего не совпало)."""
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        cur = [i]
        for j, y in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / len(r)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))]


def load_pairs(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    pairs = []
    for row in rows:
        msgs = row.get("messages")
        if msgs:                                  # формат датасета
            src = next(m["content"] for m in msgs if m["role"] == "user")
            dst = next(m["content"] for m in msgs if m["role"] == "assistant")
        else:                                     # простой {"src","dst"}
            src, dst = row.get("src", ""), row.get("dst", "")
        if src and dst:
            pairs.append({"src": src, "dst": dst})
    return pairs[:limit] if limit else pairs


def run_llm(pairs: list[dict], url: str,
            style: str = "plain") -> tuple[list[float], list[int]]:
    # Дообученный постпроцессор (:8083) зовётся ТОЛЬКО своим стилем: с чужим
    # системным промптом он получает не тот вход, на котором обучался (замер:
    # 0.119 вместо 0.079 — та же ловушка, что была в боевом пути).
    if "8083" in url and style == "plain":
        from myflow.postproc_prompt import POSTPROC_STYLE
        style = POSTPROC_STYLE
    cfg = SettingsStore().get().polish
    cfg.url = url
    cfg.enabled = True          # мерим модель, а не рантайм-тумблер полировки
    client = PolishClient(cfg)
    dists, lat = [], []
    for row in pairs:
        started = time.perf_counter()
        try:
            result = client.polish(row["src"], style=style, dictionary=[])
            out = result.text or row["src"]
        except Exception:  # noqa: BLE001
            out = row["src"]
        lat.append(int((time.perf_counter() - started) * 1000))
        dists.append(distance(row["dst"], out))
    return dists, lat


def report(name: str, dists: list[float], lat: list[int] | None = None,
           srcs: list[str] | None = None) -> None:
    exact = sum(1 for d in dists if d == 0) / max(1, len(dists)) * 100
    line = (f"  {name:<16} расстояние p50 {pct(dists, .5):.3f} | среднее "
            f"{statistics.fmean(dists):.3f} | точных {exact:>4.1f}%")
    if lat:
        line += f" | задержка p50 {pct([float(x) for x in lat], .5):>5.0f} мс"
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Замер постпроцессора на реальных парах")
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "train" / "postproc_val.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", action="store_true", help="мерить и текущую LLM-полировку")
    ap.add_argument("--url", default=None, help="URL LLM (по умолчанию из настроек)")
    ap.add_argument("--style", default="plain")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs, args.limit)
    if not pairs:
        print(f"нет пар в {args.pairs}")
        return 1
    print(f"Пар: {len(pairs)} (источник: {args.pairs})")
    srcs = [p["src"] for p in pairs]

    report("raw ASR", [distance(p["dst"], p["src"]) for p in pairs])
    report("light_clean", [distance(p["dst"], light_clean(p["src"])[0]) for p in pairs])
    if args.llm:
        url = args.url or SettingsStore().get().polish.url
        dists, lat = run_llm(pairs, url, args.style)
        report(f"LLM {url.split('//')[-1]}", dists, lat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
