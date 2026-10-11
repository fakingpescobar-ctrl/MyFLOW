"""Слить LoRA-адаптер постпроцессора с базовой моделью в один HF-каталог.

Нужно перед конвертацией в GGUF: llama.cpp принимает уже слитую модель (или
отдельный LoRA в GGUF, но проще слить).

Запуск (в учебном окружении):
    .venv-train\\Scripts\\python.exe tools\\merge_postproc_lora.py ^
        --adapter data/train/postproc_lora --out data/train/postproc_merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Слить LoRA с базой")
    ap.add_argument("--adapter", type=Path,
                    default=ROOT / "data" / "train" / "postproc_lora")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "train" / "postproc_merged")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (args.adapter / "adapter_config.json").exists():
        raise SystemExit(f"нет адаптера в {args.adapter}")

    base_name = None
    import json
    cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base_name = cfg.get("base_model_name_or_path")
    print("база:", base_name)

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    tokenizer.save_pretrained(str(args.out))
    print("слитая модель:", args.out)
    print("дальше: tools/export_postproc_gguf.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
