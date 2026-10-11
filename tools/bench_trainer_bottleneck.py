"""Где тренер теряет время: сравнение raw-цикла и Trainer на одних данных.

Замер raw-цикла (см. bench_train_speed.py) даёт 117-192 мс на шаг, а `Trainer` на
тех же данных — 14.6 с. Здесь ищем причину: длину последовательностей, collator,
autocast, оптимизатор, число потоков CPU.
"""
import os
import time
from pathlib import Path
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import torch                                              # noqa: E402
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments  # noqa: E402
from peft import LoraConfig, get_peft_model                # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "train" / "Qwen3-0.6B"
VOCAB = 151643


class Tok:
    pad_token_id = 151643
    eos_token_id = 151645


def batch(batch_size: int, seq: int) -> dict:
    ids = torch.randint(0, 1000, (batch_size, seq))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": ids.clone()}


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    print("  CPU потоков у torch:", torch.get_num_threads(), "| ядер:", os.cpu_count())
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM: свободно {free / 2**30:.2f} ГБ")

    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.to("cuda")

    data = []
    for _ in range(40):
        ids1 = torch.randint(0, 1000, (256,))
        data.append({"input_ids": ids1, "attention_mask": torch.ones(256),
                     "labels": ids1.clone()})

    print("\n  --- raw цикл (batch 8, 256 токенов) ---")
    ids = torch.randint(0, 1000, (8, 256), device="cuda")
    mask = torch.ones_like(ids)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for _ in range(3):
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(10):
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - started) / 10
    print(f"      {per_step * 1000:.0f} мс/шаг | пик {torch.cuda.max_memory_allocated()/2**30:.2f} ГБ")

    print("\n  --- Trainer (batch 8, 256 токенов, 10 шагов) ---")
    args = TrainingArguments(output_dir=str(Path(__file__).resolve().parent.parent / "data" / "train" / "bench_out"),
                             max_steps=10, per_device_train_batch_size=8,
                             learning_rate=1e-4, bf16=True, logging_steps=5,
                             report_to=[], save_strategy="no", remove_unused_columns=False,
                             dataloader_num_workers=0, optim="adamw_torch")
    trainer = Trainer(model=model, args=args, train_dataset=data)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    took = time.perf_counter() - started
    print(f"      Trainer: {took / 10 * 1000:.0f} мс/шаг | "
          f"пик {torch.cuda.max_memory_allocated()/2**30:.2f} ГБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
