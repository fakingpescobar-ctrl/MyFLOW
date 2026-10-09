"""MyFLOW — A/B полировки на СВОЕЙ истории диктовок.

Берём реальные пары `asr_text` → `polished_text` из `data/myflow.sqlite`,
прогоняем `asr_text` через указанный llama-server и сравниваем результат с тем,
что ты уже видел и принял. Это объективный способ выбрать модель (4B/9B/12B),
не полагаясь на «на глаз».

Метрики:
  * латентность p50/p90 (и по длине ответа);
  * divergence(LLM, принятый текст) — насколько модель разошлась с тем, что
    тебя устраивало (меньше = ближе к эталону);
  * divergence(ASR, LLM) — сколько модель переписывает (больше = риск отсебятины);
  * доля откатов `_guard_polish` (модель исказила фразу и мы вернули сырой ASR);
  * худшие примеры — глазами.

Запуск:
    .venv\\Scripts\\python -X utf8 tools\\eval_polish.py --limit 60
    .venv\\Scripts\\python -X utf8 tools\\eval_polish.py --url http://127.0.0.1:8083/v1 --label "9B Q4"
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from myflow.config import DB_FILE
from myflow.db import Database
from myflow.main import MyFlowApp
from myflow.polish import DATA_GUARD, NUMBERS_RULE, OMOPHON_RULES, STYLE_PROMPTS
from myflow.textdiff import divergence_score


def system_prompt(style: str) -> str:
    return STYLE_PROMPTS.get(style, STYLE_PROMPTS["plain"]) + NUMBERS_RULE + OMOPHON_RULES + DATA_GUARD


def ask(url: str, model: str, text: str, style: str, max_tokens: int) -> tuple[str, float] | None:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt(style)},
                     {"role": "user", "content": f"<dictation>{text}</dictation>"}],
        "temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.05,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    import time
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{url}/chat/completions", json=body, timeout=120)
    except requests.RequestException:
        return None
    dt = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return None
    return r.json()["choices"][0]["message"]["content"].strip(), dt


def load_rows(limit: int, min_words: int = 2) -> list[dict]:
    db = Database(DB_FILE)
    try:
        rows = [dict(r) for r in db._conn.execute(  # noqa: SLF001 — читаем свою базу
            """SELECT id, asr_text, polished_text, style FROM history
               WHERE asr_text <> '' AND polished_text <> ''
               ORDER BY id DESC LIMIT ?""", (limit * 4,))]
    finally:
        db.close()
    out = [r for r in rows if len(r["asr_text"].split()) >= min_words]
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B полировки на своей истории")
    ap.add_argument("--url", default="http://127.0.0.1:8082/v1")
    ap.add_argument("--label", default="")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    url = args.url.rstrip("/")
    try:
        model = requests.get(f"{url}/models", timeout=5).json()["data"][0]["id"]
    except Exception as exc:  # noqa: BLE001
        print(f"сервер недоступен ({url}): {exc!r}")
        return 1

    rows = load_rows(args.limit)
    if not rows:
        print("в истории нет подходящих диктовок (нужны asr_text и polished_text)")
        return 1

    label = args.label or Path(str(model)).name
    print(f"=== A/B {label} · {len(rows)} фраз из истории ({url}) ===\n")

    lat, div_llm, div_asr, guarded = [], [], [], 0
    worst: list[tuple[float, str, str, str]] = []

    for i, row in enumerate(rows, 1):
        style = row["style"] or "plain"
        result = ask(url, model, row["asr_text"], style, args.max_tokens)
        if result is None:
            print(f"  #{row['id']}: запрос не удался")
            continue
        candidate, ms = result
        final = MyFlowApp._guard_polish(row["asr_text"], candidate, style)
        if final == row["asr_text"] and candidate.strip() != row["asr_text"].strip():
            guarded += 1
        d_ref = divergence_score(final, row["polished_text"])
        d_asr = divergence_score(row["asr_text"], final)
        lat.append(ms)
        div_llm.append(d_ref)
        div_asr.append(d_asr)
        worst.append((d_ref, row["asr_text"], final, row["polished_text"]))
        if i % 10 == 0:
            print(f"  ... {i}/{len(rows)}")

    if not lat:
        print("ни один запрос не прошёл")
        return 1

    lat.sort()
    p50 = statistics.median(lat)
    p90 = lat[min(len(lat) - 1, int(len(lat) * 0.9))]
    print(f"\n--- {label} ---")
    print(f"  фраз              : {len(lat)}")
    print(f"  латентность       : p50={p50:.0f} мс · p90={p90:.0f} мс · "
          f"мин={lat[0]:.0f} · макс={lat[-1]:.0f}")
    print(f"  divergence к эталону (наш принятый текст): "
          f"средне={statistics.mean(div_llm):.3f}")
    print(f"  divergence ASR→LLM (сколько модель переписывает): "
          f"средне={statistics.mean(div_asr):.3f}")
    print(f"  откатов guard     : {guarded} из {len(lat)} ({guarded / len(lat) * 100:.1f} %)")

    worst.sort(reverse=True)
    print("\n--- где модель сильнее всего разошлась с эталоном ---")
    for d, raw, got, ref in worst[:5]:
        print(f"  divergence={d:.2f}")
        print(f"    ASR : {raw[:110]}")
        print(f"    LLM : {got[:110]}")
        print(f"    эталон: {ref[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
