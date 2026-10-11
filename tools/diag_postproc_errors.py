"""Разбор ошибок постпроцессора: где именно он не дотягивает.

Показывает худшие случаи, средний промах по длине фразы и «переправляет ли модель
лишнее» (удалённые/добавленные слова). Нужен, чтобы понять следующий шаг: данные,
эпохи или размер модели.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from eval_postproc import distance, load_pairs, words  # noqa: E402
from myflow.postproc_prompt import POSTPROC_SYSTEM    # noqa: E402

BASE = ROOT / "data" / "train" / "Qwen3-0.6B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "data" / "train" / "postproc_lora_v2"))
    ap.add_argument("--pairs", type=Path, default=ROOT / "data" / "train" / "postproc_val.jsonl")
    ap.add_argument("--worst", type=int, default=6)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    model = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model = model.to("cuda").eval()

    rows = load_pairs(args.pairs)
    results = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": POSTPROC_SYSTEM},
             {"role": "user", "content": row["src"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=inputs["input_ids"].shape[1],
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        produced = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
        d = distance(row["dst"], produced)
        raw = distance(row["dst"], row["src"])
        results.append({"src": row["src"], "dst": row["dst"], "out": produced,
                        "d": d, "raw": raw, "len": len(words(row["dst"]))})

    mean = statistics.fmean(r["d"] for r in results)
    raw_mean = statistics.fmean(r["raw"] for r in results)
    helped = sum(1 for r in results if r["d"] < r["raw"] - 1e-9)
    hurt = sum(1 for r in results if r["d"] > r["raw"] + 1e-9)
    same = len(results) - helped - hurt
    print(f"  среднее {mean:.3f} (сырой ASR {raw_mean:.3f}) | "
          f"улучшил {helped}, испортил {hurt}, без изменений {same} из {len(results)}")

    print("\n  по длине фразы:")
    for lo, hi in ((1, 5), (6, 12), (13, 25), (26, 1000)):
        bucket = [r for r in results if lo <= r["len"] <= hi]
        if bucket:
            bmean = statistics.fmean(r["d"] for r in bucket)
            braw = statistics.fmean(r["raw"] for r in bucket)
            print(f"    {lo:>3}-{hi:<4} слов: n={len(bucket):>2} | "
                  f"модель {bmean:.3f} | сырой {braw:.3f}")

    print(f"\n  худшие {args.worst} случая:")
    for r in sorted(results, key=lambda x: -x["d"])[:args.worst]:
        print(f"    промах {r['d']:.2f} (сырой {r['raw']:.2f})")
        print(f"      вход : {r['src'][:78]}")
        print(f"      цель : {r['dst'][:78]}")
        print(f"      выход: {r['out'][:78].replace(chr(10), ' | ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
