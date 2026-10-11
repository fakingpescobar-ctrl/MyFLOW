"""Замер скорости обучения 0.6B: аккуратно, с малым батчем и контролем памяти.

Прошлый замер упал в OOM (23.8 ГБ аллокаций на 10 ГБ карте) — подозрение на
фрагментацию и удержание графов. Здесь: малый батч, освобождение между шагами,
`expandable_segments`, и печать пиковой памяти. Сравниваем eager и sdpa.
"""
import time
from pathlib import Path

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "train" / "Qwen3-0.6B"


def run(attn: str, seq: int, batch: int = 2) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.bfloat16, attn_implementation=attn).to("cuda").train()
    peak = 0.0
    try:
        ids = torch.randint(0, 1000, (batch, seq), device="cuda")
        mask = torch.ones_like(ids)
        # прогрев
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        del out
        started = time.perf_counter()
        for _ in range(4):
            out = model(input_ids=ids, attention_mask=mask, labels=ids)
            out.loss.backward()
            torch.cuda.synchronize()
            model.zero_grad(set_to_none=True)
            del out
        per_step = (time.perf_counter() - started) / 4
        peak = torch.cuda.max_memory_allocated() / 2**30
        tokens = batch * seq
        print(f"  {attn:<5} seq={seq:>3} batch={batch}: {per_step * 1000:6.0f} мс/шаг | "
              f"{tokens / per_step / 1000:7.1f} тыс. ток/с | пик VRAM {peak:.2f} ГБ")
    finally:
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("  torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
          "| TF32", torch.backends.cuda.matmul.allow_tf32)
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM: свободно {free / 2**30:.2f} ГБ из {total / 2**30:.2f} ГБ")
    for attn in ("sdpa", "eager"):
        for seq in (128, 256, 512):
            try:
                run(attn, seq)
            except Exception as exc:  # noqa: BLE001
                print(f"  {attn:<5} seq={seq:>3}: ошибка {type(exc).__name__}: "
                      f"{str(exc)[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
