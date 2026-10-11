"""Почему тренер медленный: проверка dtype и длины последовательностей.

Гипотеза: `dtype=bfloat16` в transformers 5.x не применяется, модель грузится в
fp32 (2.4 ГБ вместо 1.2), пик выходит за 10 ГБ карты, WDDM уводит в своп — отсюда
1.7 с на шаг. Проверяем dtype и заодно смотрим реальные длины наших текстов.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "train" / "Qwen3-0.6B"
TRAIN = Path(__file__).resolve().parent.parent / "data" / "train" / "postproc_train.jsonl"


def main() -> int:
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16)
    got = next(model.parameters()).dtype
    print(f"  после from_pretrained(dtype=bfloat16): веса в {got}",
          "-> ПАРАМЕТР НЕ ПРИМЕНИЛСЯ" if got != torch.bfloat16 else "-> ок")
    model = model.to(torch.bfloat16)
    print(f"  после model.to(bfloat16): веса в {next(model.parameters()).dtype}")

    tokenizer = AutoTokenizer.from_pretrained(PATH)
    rows = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    lengths = []
    for row in rows[:1500]:
        text = "".join(m["content"] for m in row["messages"])
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    lengths.sort()
    total = len(lengths)
    print(f"  длины примеров (символы+промпт), n={total}:")
    for q in (0.5, 0.9, 0.95, 0.99, 1.0):
        idx = min(total - 1, int(total * q) if q < 1 else total - 1)
        print(f"      p{int(q * 100):>3}: {lengths[idx]:>5} токенов")
    print(f"  доля примеров длиннее 256: "
          f"{sum(1 for x in lengths if x > 256) / total * 100:.1f}%")
    print(f"  доля примеров длиннее 384: "
          f"{sum(1 for x in lengths if x > 384) / total * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
