"""MyFLOW — бенч локальной LLM-полировки (llama-server).

Что мерит:
  * скорость генерации (tok/s) на разной длине ответа — это главный показатель
    для полировки (у нас короткие ответы, важна именно скорость декодинга);
  * работает ли кэш системного промпта (prompt_n vs cache_n) — если нет,
    каждая диктовка платит повторный префилл ~2200 токенов (~250 мс);
  * реальную границу контекста (какой n_ctx у слота: при `-np 4` контекст
    делится, и запросы начинают падать с exceed_context_size);
  * латентность на живых фразах (p50/p90) и качество по набору ловушек
    (числа, омофоны, падежи) — печатает ответы, чтобы видеть регрессии.

Запуск:
    .venv\\Scripts\\python -X utf8 tools\\bench_llm.py
    .venv\\Scripts\\python -X utf8 tools\\bench_llm.py --url http://127.0.0.1:8083/v1 --label "9B Q4"
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from myflow.polish import DATA_GUARD, NUMBERS_RULE, OMOPHON_RULES, STYLE_PROMPTS

# Ловушки, на которых 4B спотыкалась (падежи/числа/омофоны/филлеры).
TRAPS = [
    "брат перезапусти батник и напиши что вышло",
    "открыть эндпоинт на порту восемь тысяч восемьдесят один и запустить тест",
    "ну это типа как бы проверка связи короче",
    "какая погода в сухом лагу сегодня",
    "скинь мне отчёт за первый квартал на почту",
    "проверка связи один два три четыре пять",
]


def system_prompt(style: str = "plain") -> str:
    return STYLE_PROMPTS[style] + NUMBERS_RULE + OMOPHON_RULES + DATA_GUARD


def load_model(url: str) -> str:
    return requests.get(f"{url}/models", timeout=5).json()["data"][0]["id"]


def ask(url: str, model: str, text: str, *, temperature: float = 0.2,
        max_tokens: int = 200, cache_prompt: bool | None = None) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt()},
                     {"role": "user", "content": f"<dictation>{text}</dictation>"}],
        "temperature": temperature, "top_p": 0.9, "repeat_penalty": 1.05,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if cache_prompt is not None:
        body["cache_prompt"] = cache_prompt
    t0 = time.perf_counter()
    resp = requests.post(f"{url}/chat/completions", json=body, timeout=180)
    wall = (time.perf_counter() - t0) * 1000
    if resp.status_code != 200:
        return {"error": resp.status_code, "text": resp.text[:200], "wall_ms": wall}
    payload = resp.json()
    tm = payload.get("timings") or {}
    return {
        "text": payload["choices"][0]["message"]["content"].strip(),
        "wall_ms": wall,
        "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
        "prompt_ms": tm.get("prompt_ms"),
        "predict_n": tm.get("predicted_n"), "predict_ms": tm.get("predicted_ms"),
        "tps": tm.get("predicted_per_second"),
    }


def bench_throughput(url: str, model: str) -> None:
    print("--- скорость генерации (длина ответа -> tok/s) ---")
    for mx in (20, 100, 300):
        r = ask(url, model, "перечисли числа от 1 до 60 словами", max_tokens=mx, temperature=0)
        if "error" in r:
            print(f"  max_tokens={mx}: ошибка {r['error']}: {r['text']}")
            continue
        tps = r["tps"] or 0
        print(f"  max_tokens={mx:<4} predicted={r['predict_n']:<4} "
              f"predict={r['predict_ms'] and round(r['predict_ms'])} мс  "
              f"wall={r['wall_ms']:6.0f} мс  tok/s={tps:.1f}")


def bench_cache(url: str, model: str) -> None:
    print("--- кэш системного промпта (cache_n должен быть ~2000+) ---")
    for i in range(3):
        r = ask(url, model, TRAPS[i % len(TRAPS)], max_tokens=40, temperature=0)
        if "error" in r:
            print(f"  запрос {i + 1}: ошибка {r['error']}")
            continue
        total = (r["prompt_n"] or 0) + (r["cache_n"] or 0)
        print(f"  запрос {i + 1}: prompt_n={r['prompt_n']} cache_n={r['cache_n']} "
              f"(всего ~{total}) префилл={r['prompt_ms'] and round(r['prompt_ms'])} мс "
              f"tps={r['tps'] and round(r['tps'], 1)}")


def bench_context(url: str, model: str) -> None:
    print("--- граница контекста (уникальный текст, без кэш-хитов) ---")
    alphabet = ["альфа", "бета", "гамма", "дельта", "эпсилон", "дзета", "эта", "тета"]
    for words in (1000, 2000, 2500, 3000):
        text = " ".join(alphabet[(i * 7 + i // 3) % len(alphabet)] + str(i) for i in range(words))
        body = {"model": model, "messages": [{"role": "user", "content": text}],
                "max_tokens": 4, "temperature": 0}
        t0 = time.perf_counter()
        resp = requests.post(f"{url}/chat/completions", json=body, timeout=180)
        dt = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            print(f"  ~{words} слов: {resp.status_code} -> {resp.text[:160]}")
            break
        tm = (resp.json().get("timings") or {})
        print(f"  ~{words} слов: ок, prompt_n={tm.get('prompt_n')} "
              f"префилл={tm.get('prompt_ms') and round(tm['prompt_ms'])} мс (wall {dt:.0f})")


def bench_latency_and_quality(url: str, model: str, temperature: float) -> list[float]:
    print(f"--- живые фразы (temperature={temperature}) ---")
    lat = []
    for text in TRAPS:
        r = ask(url, model, text, temperature=temperature, max_tokens=200)
        if "error" in r:
            print(f"  ошибка: {r['error']}")
            continue
        lat.append(r["wall_ms"])
        print(f"  RAW : {text}")
        print(f"  POLL: {r['text']}   ({r['wall_ms']:.0f} мс, {r['predict_n']} ток, "
              f"{r['tps'] and round(r['tps'], 1)} tok/s)")
    if lat:
        lat.sort()
        p50 = statistics.median(lat)
        p90 = lat[int(len(lat) * 0.9) - 1] if len(lat) > 1 else lat[0]
        print(f"  ИТОГ: p50={p50:.0f} мс, p90={p90:.0f} мс")
    return lat


def main() -> int:
    ap = argparse.ArgumentParser(description="Бенч LLM-полировки MyFLOW")
    ap.add_argument("--url", default="http://127.0.0.1:8082/v1")
    ap.add_argument("--label", default="")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--skip-context", action="store_true")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    try:
        model = load_model(url)
    except Exception as exc:  # noqa: BLE001
        print(f"сервер недоступен ({url}): {exc!r}")
        return 1
    label = args.label or Path(str(model)).name
    print(f"=== Бенч {label} ({url}) ===")
    print(f"system-промпт: {len(system_prompt())} символов")
    bench_cache(url, model)
    bench_throughput(url, model)
    if not args.skip_context:
        bench_context(url, model)
    bench_latency_and_quality(url, model, args.temperature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
