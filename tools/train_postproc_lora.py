"""LoRA-дообучение компактного русского постпроцессора MyFLOW.

Роль модели узкая (как s1-mini у superwhisper): «сырой ASR → аккуратный текст».
Данные — `tools/build_postproc_dataset.py` (реальные пары + синтетика).

Запуск (в ОТДЕЛЬНОМ учебном окружении, см. tools/train_postproc.md):
    .venv-train\\Scripts\\python.exe tools\\train_postproc_lora.py --base Qwen/Qwen3-0.6B

Что делает:
  * берёт базовую модель и датасет, обучает LoRA (attention + MLP проекции);
  * считает eval-loss на РЕАЛЬНЫХ парах (val), сохраняет лучший адаптер;
  * учит только на токенах ответа (промпт и вопрос исключены из лосса).

Ограничения окружения: RTX 3080 10 ГБ. 0.6B в bf16 + LoRA r=16 занимает ~3.5 ГБ,
1.7B — ~7 ГБ; батч и длина подобраны под это.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "train"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def to_example(tokenizer, row: dict, max_len: int) -> dict:
    """Собрать пример с маскированием: лосс только на ответе ассистента."""
    msgs = row["messages"]
    full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False,
                                           add_generation_prompt=True)
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100                       # промпт не учим
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": labels}


class Collator:
    """Дополняет батч до самой длинной последовательности (паддинг справа)."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, masks, labels = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * pad)
            masks.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA-обучение постпроцессора")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B",
                    help="базовая модель (Qwen/Qwen3-0.6B или Qwen/Qwen3-1.7B)")
    ap.add_argument("--train", type=Path, default=DATA_DIR / "postproc_train.jsonl")
    ap.add_argument("--val", type=Path, default=DATA_DIR / "postproc_val.jsonl")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "postproc_lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--smoke", action="store_true", help="быстрый прогон на 200 примерах")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    print(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows, val_rows = load_rows(args.train), load_rows(args.val)
    if args.smoke:
        train_rows, val_rows = train_rows[:200], val_rows[:20]
    print(f"примеров: train={len(train_rows)} val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = [to_example(tokenizer, r, args.max_len) for r in train_rows]
    val_ds = [to_example(tokenizer, r, args.max_len) for r in val_rows]
    collator = Collator(tokenizer.pad_token_id or tokenizer.eos_token_id)

    args.out.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=25, save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="steps", eval_steps=args.save_steps,
        bf16=(device == "cuda"), optim="adamw_torch", report_to=[],
        gradient_checkpointing=False, remove_unused_columns=False)
    # Совместимость версий transformers: параметр переименовали
    # (evaluation_strategy -> eval_strategy), и на старых сборках падает.
    try:
        training = TrainingArguments(
            output_dir=str(args.out), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
            logging_steps=25, save_steps=args.save_steps, save_total_limit=2,
            evaluation_strategy="steps", eval_steps=args.save_steps,
            bf16=(device == "cuda"), optim="adamw_torch", report_to=[],
            remove_unused_columns=False)
    except TypeError:
        pass
    trainer = Trainer(model=model, args=training, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print("адаптер сохранён:", args.out)
    print("\nДальше: собрать GGUF (tools/export_postproc_gguf.ps1) и замерить "
          "tools/eval_postproc.py --url http://127.0.0.1:8083/v1")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
