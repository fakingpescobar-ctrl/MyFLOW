"""Замер постпроцессора прямо в transformers: правильный формат, без обрезки.

Отвечает на вопрос, откуда провал (среднее 0.618 против 0.220 у 4B):
виноват формат входа (обёртка `<dictation>`, включённые «размышления», обрезка
на 256 токенах) или сама модель не тянет задачу.

Запуск:
    .venv-train\\Scripts\\python.exe tools\\eval_postproc_hf.py [--adapter ПУТЬ|--base]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from eval_postproc import distance, light_clean, load_pairs  # noqa: E402
from myflow.postproc_prompt import POSTPROC_SYSTEM          # noqa: E402

BASE = Path(__file__).resolve().parent.parent / "data" / "train" / "Qwen3-0.6B"


def pct(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "train" / "postproc_val.jsonl")
    ap.add_argument("--adapter", default=str(ROOT / "data" / "train" / "postproc_lora"))
    ap.add_argument("--base", action="store_true", help="без адаптера (базовая модель)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=0, help="0 = по длине входа")
    ap.add_argument("--thinking", action="store_true", help="разрешить блок размышлений")
    ap.add_argument("--wrap", action="store_true", help="обернуть вход как в бою (<dictation>)")
    args = ap.parse_args()

    print("  загружаю", BASE.name, "| адаптер:", "нет" if args.base else Path(args.adapter).name)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    model = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.float16)
    if not args.base:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to("cuda").eval()

    pairs = load_pairs(args.pairs, args.limit)
    print(f"  пар: {len(pairs)} | размышления: {args.thinking} | обёртка: {args.wrap}")

    dists, lat, samples = [], [], []
    for row in pairs:
        src, dst = row["src"], row["dst"]
        user = f"<dictation>{src}</dictation>" if args.wrap else src
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": POSTPROC_SYSTEM},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
            **({} if args.thinking else {"enable_thinking": False}))
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        budget = args.max_new or max(48, inputs["input_ids"].shape[1])
        started = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        lat.append(int((time.perf_counter() - started) * 1000))
        produced = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
        dists.append(distance(dst, produced))
        if len(samples) < 3:
            samples.append((src, dst, produced))

    exact = sum(1 for d in dists if d == 0) / len(dists) * 100
    print(f"\n  ИТОГ p50 {pct(dists, .5):.3f} | среднее {statistics.fmean(dists):.3f} | "
          f"точных {exact:.1f}% | задержка p50 {pct([float(x) for x in lat], .5):.0f} мс")
    print("\n  примеры:")
    for src, dst, produced in samples:
        print(f"    вход : {src[:70]}")
        print(f"    цель : {dst[:70]}")
        print(f"    выход: {produced[:70].replace(chr(10), ' | ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
