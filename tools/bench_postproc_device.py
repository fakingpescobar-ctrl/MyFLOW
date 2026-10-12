"""Замер: постпроцессор на видеоядре (Vulkan) против 3080 (CUDA).

Один и тот же файл модели, один и тот же системный промпт (общий с рантаймом),
одни и те же фразы. Меряем задержку на фразу и скорость генерации.

Запуск:
    python tools/bench_postproc_device.py            # 8084 против 8085
    python tools/bench_postproc_device.py --ports 8084
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

PHRASES = [
    "Так, брат, проверка связи.",
    "ну короче смотри я хотел сказать что этот файл надо переименовать",
    "Открой мне пожалуйста настройки и посмотри что там с портами",
    "я вчера начал делать таблицу и добавил там три столбца",
    "эээ давай перенесём эту встречу на завтра на десять утра",
    "нужно проверить логи и посмотреть почему упал сервер",
    "Позвони маме и скажи что я приеду позже",
    "Слушай а ты можешь сделать чтобы оно само это делало",
    "добавь в список задач пункт про тесты",
    "короче говоря всё работает но медленно",
]

ENDPOINTS = "http://127.0.0.1:{port}/v1/chat/completions"


def system_prompt() -> str:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from myflow.postproc_prompt import POSTPROC_SYSTEM
        return POSTPROC_SYSTEM
    except Exception:  # noqa: BLE001
        return "Приведи расшифровку диктовки в порядок, не меняя смысл."


def ask(port: int, text: str, system: str, timeout: float = 60.0) -> tuple[float, int]:
    """Один запрос: (секунды, сгенерированных токенов)."""
    payload = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": text}],
        "temperature": 0.0, "max_tokens": 256, "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(ENDPOINTS.format(port=port), data=data,
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    tokens = int((body.get("usage") or {}).get("completion_tokens") or 0)
    return elapsed, tokens


def measure(port: int, label: str, system: str) -> dict:
    print(f"\n  {label} (: {port})")
    try:
        ask(port, "разминка", system)                      # прогрев
    except Exception as exc:  # noqa: BLE001
        print(f"    недоступен: {type(exc).__name__}")
        return {"port": port, "label": label, "ok": False}

    times, rates = [], []
    for phrase in PHRASES:
        elapsed, tokens = ask(port, phrase, system)
        times.append(elapsed * 1000)
        if tokens and elapsed > 0:
            rates.append(tokens / elapsed)
    times.sort()
    result = {
        "port": port, "label": label, "ok": True,
        "p50": times[len(times) // 2],
        "mean": statistics.fmean(times),
        "min": times[0], "max": times[-1],
        "tok_s": statistics.fmean(rates) if rates else 0.0,
    }
    print(f"    p50 {result['p50']:.0f} мс | среднее {result['mean']:.0f} мс | "
          f"лучшее {result['min']:.0f} | худшее {result['max']:.0f} | "
          f"{result['tok_s']:.1f} ток/с")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="8084,8085")
    ap.add_argument("--labels", default="видеоядро (Vulkan),3080 (CUDA)")
    args = ap.parse_args()

    ports = [int(p) for p in args.ports.split(",")]
    labels = args.labels.split(",")
    system = system_prompt()
    results = [measure(port, labels[i] if i < len(labels) else str(port), system)
               for i, port in enumerate(ports)]

    good = [r for r in results if r["ok"]]
    if len(good) == 2:
        fast, slow = sorted(good, key=lambda r: r["p50"])
        ratio = slow["p50"] / fast["p50"] if fast["p50"] else 0
        print(f"\n  быстрее: {fast['label']} — в {ratio:.1f} раза по медиане")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
