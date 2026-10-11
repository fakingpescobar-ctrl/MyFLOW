"""Дистилляция: 4B учит 0.6B своим поведением на нашем корпусе.

Почему так: на чистой валидации 4B точнее (0.073 против 0.081) — он делает более
меткие мелкие правки. Значит вместо «золотых» эталонов даём ученику ВЫХОД УЧИТЕЛЯ:
студент повторяет политику той модели, качество которой мы хотим догнать.

Входы берём из реального распределения:
  * `src` реальных пар из истории (без валидации — иначе утечка);
  * синтетика с нашим тонким профилем порчи;
  * «уже хорошие» тексты (учим не трогать).

Эталоны пишутся в `data/train/teacher_train.jsonl` (чат-формат со СВОИМ системным
промптом ученика — на нём же он будет вызван в бою).

Запуск:
    .venv\\Scripts\\python.exe -X utf8 tools\\gen_teacher_data.py --limit 3500
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_postproc_dataset import (clean_texts, corrupt, load_dictionary,  # noqa: E402
                                    real_pairs, word_error_pairs)
from eval_postproc import words                                          # noqa: E402
from myflow.config import SettingsStore                                  # noqa: E402
from myflow.polish import PolishClient                                   # noqa: E402
from myflow.postproc_prompt import POSTPROC_SYSTEM                       # noqa: E402
from myflow.tui import Supervisor, default_specs                         # noqa: E402

OUT = ROOT / "data" / "train" / "teacher_train.jsonl"
BAD_MARKERS = ("Задачи:", "Это пост-обработка", "То, что человек реально сказал",
               "Верни только текст", "Ты — постпроцессор")


def build_inputs(seed: int = 20261011) -> list[str]:
    """Входы: реальные + синтетика + «уже хорошие» (без валидации)."""
    rng = random.Random(seed)
    pairs = real_pairs()
    val_ids = {p["id"] for p in pairs if p["id"] % 7 == 0}
    blocked = set()
    for p in pairs:
        if p["id"] in val_ids:
            blocked.add(p["src"].lower())
            blocked.add(p["dst"].lower())

    inputs: list[str] = []
    # 1) реальные входы (тренировочная часть)
    for p in pairs:
        if p["id"] not in val_ids:
            inputs.append(p["src"])

    # 2) синтетика
    dictionary = load_dictionary()
    error_pairs = word_error_pairs()
    texts = clean_texts(exclude=set())
    rng.shuffle(texts)
    for text in texts:
        if text.lower() in blocked:
            continue
        for _ in range(3):
            src = corrupt(text, rng, dictionary, error_pairs)
            if src and src.strip().lower() != text.strip().lower():
                inputs.append(src)
        inputs.append(text)          # «уже хорошее» — учим не трогать

    # уникализируем, сохраняя порядок
    seen, out = set(), []
    for item in inputs:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def looks_bad(src: str, out: str) -> str:
    """Пустая/переписанная/эхо-выдача — в обучение не годится. '' = годится."""
    if not out or len(out.strip()) < 2:
        return "пусто"
    if any(marker in out for marker in BAD_MARKERS):
        return "эхо промпта"
    n_src, n_out = len(words(src)), len(words(out))
    if n_src and (n_out < 0.4 * n_src or n_out > 1.8 * n_src):
        return f"длина {n_out}/{n_src}"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Эталоны от 4B для дистилляции")
    ap.add_argument("--limit", type=int, default=3500)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--url", default=None)
    args = ap.parse_args()

    sup = Supervisor(default_specs(), enabled=True)
    llm = next(s for s in sup.specs if s.key == "llm")
    cfg = SettingsStore().get().polish
    if args.url:
        cfg.url = args.url
    cfg.enabled = True
    client = PolishClient(cfg)
    ours_started = False
    try:
        if not client.alive():
            print("  поднимаю учителя (4B) на :8082...", flush=True)
            sup.start(llm)
            ours_started = sup.wait_ready(llm, 180.0)
        print(f"  учитель готов: {client.alive()}", flush=True)

        inputs = build_inputs()[: args.limit]
        print(f"  входов: {len(inputs)}", flush=True)

        rows, dropped, lats = [], {}, []
        for i, src in enumerate(inputs, 1):
            started = time.perf_counter()
            try:
                result = client.polish(src, style="plain", dictionary=[])
                out = (result.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                out = ""
                dropped[f"ошибка {type(exc).__name__}"] = dropped.get(
                    f"ошибка {type(exc).__name__}", 0) + 1
            lats.append(int((time.perf_counter() - started) * 1000))
            reason = looks_bad(src, out)
            if reason and not out:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            if reason:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            rows.append({"messages": [
                {"role": "system", "content": POSTPROC_SYSTEM},
                {"role": "user", "content": src},
                {"role": "assistant", "content": out},
            ]})
            if i % 250 == 0:
                print(f"    {i}/{len(inputs)} готово, отброшено {sum(dropped.values())}, "
                      f"p50 {statistics.median(lats):.0f} мс", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                            encoding="utf-8")
        print(f"\n  записано: {len(rows)} пар -> {args.out.relative_to(ROOT)}")
        print(f"  отброшено: {sum(dropped.values())} ({dropped})")
        if lats:
            print(f"  скорость учителя: p50 {statistics.median(lats):.0f} мс")
        print("\n  примеры:")
        for row in rows[:4]:
            src = row["messages"][1]["content"]
            dst = row["messages"][2]["content"]
            print(f"    вход : {src[:76]}")
            print(f"    ученик получит: {dst[:76]}")
    finally:
        if ours_started:
            print("  гашу учителя, которого поднял я", flush=True)
            sup.stop(llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
