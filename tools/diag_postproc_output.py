"""Диагностика вывода постпроцессора: что именно генерирует модель.

Сравниваем ДВА формата входа:
  1) как в обучении: apply_chat_template(system+user, thinking как в шаблоне);
  2) как в бою: с `<dictation>...</dictation>` и enable_thinking=False.

Если модель отвечает мусором/рассуждениями — проблема в несовпадении формата,
а не в размере модели.
"""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from myflow.postproc_prompt import POSTPROC_SYSTEM  # noqa: E402

BASE = Path(__file__).resolve().parent.parent / "data" / "train" / "Qwen3-0.6B"
ADAPTER = str(Path(__file__).resolve().parent.parent / "data" / "train" / "postproc_lora")
VAL = Path(__file__).resolve().parent.parent / "data" / "train" / "postproc_val.jsonl"


def generate(model, tokenizer, messages, **template_kwargs) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, **template_kwargs)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
    model = model.to("cuda").eval()

    rows = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  пар в валидации: {len(rows)}")
    for row in rows[:4]:
        msgs = row["messages"]
        src = next(m["content"] for m in msgs if m["role"] == "user")
        dst = next(m["content"] for m in msgs if m["role"] == "assistant")
        print("\n  ── вход:", src[:90])
        print("  цель    :", dst[:90])
        # 1) как обучали (шаблон по умолчанию)
        out1 = generate(model, tokenizer, msgs[:-1])
        print("  как учили :", out1[:120].replace("\n", " | "))
        # 2) боевой формат: dictation-обёртка + без размышлений
        out2 = generate(model, tokenizer,
                        [{"role": "system", "content": POSTPROC_SYSTEM},
                         {"role": "user", "content": f"<dictation>{src}</dictation>"}],
                        enable_thinking=False)
        print("  боевой    :", out2[:120].replace("\n", " | "))
        # 3) правильный формат: без обёртки, без размышлений
        out3 = generate(model, tokenizer,
                        [{"role": "system", "content": POSTPROC_SYSTEM},
                         {"role": "user", "content": src}],
                        enable_thinking=False)
        print("  чистый    :", out3[:120].replace("\n", " | "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
