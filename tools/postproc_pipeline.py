"""Конвейер постпроцессора: слить адаптер -> GGUF -> квант -> проверить.

Собирает в одну команду то, что раньше делалось вручную (и на чём я спотыкался):

* слияние LoRA с базой (`merge_postproc_lora.py`);
* конвертация в GGUF. ВАЖНО: `convert_hf_to_gguf.py` из master-ветки llama.cpp
  стал пакетным (импортирует `conversion`) и по отдельности не работает; версия
  b9000 самодостаточна, её и берём (curl-ом — pip/wget тут зависают);
* квантование нашим `llama-quantize.exe`;
* опционально — сразу поднять сервер на :8083 и сравнить с боевой моделью
  (`ab_postproc.py`).

Запуск:
    .venv\\Scripts\\python.exe -X utf8 tools\\postproc_pipeline.py \\
        --adapter data\\train\\postproc_lora_v2 --name myflow-postproc-qwen3-06b
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / ".venv-train" / "Scripts" / "python.exe"
LLAMA = ROOT / "bin" / "llama-cpp"
WORK = ROOT / "data" / "train" / "gguf"
CONVERTER_TAG = "b9000"          # в этой версии конвертер ещё самодостаточен
CONVERTER = WORK / "convert_hf_to_gguf.py"
CONVERTER_URL = (f"https://raw.githubusercontent.com/ggml-org/llama.cpp/"
                 f"{CONVERTER_TAG}/convert_hf_to_gguf.py")


def run(cmd: list[str], title: str) -> bool:
    print(f"\n=== {title} ===")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"  ОШИБКА (код {proc.returncode})")
        return False
    return True


def ensure_converter() -> bool:
    if CONVERTER.exists() and CONVERTER.stat().st_size > 3000:
        return True
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"  качаю конвертер {CONVERTER_TAG} (curl — pip/wget зависают)...")
    cmd = ["curl.exe", "-sL", "--retry", "20", "--retry-delay", "3",
           "--retry-all-errors", "--speed-limit", "20480", "--speed-time", "20",
           "-o", str(CONVERTER), CONVERTER_URL]
    subprocess.run(cmd, cwd=str(ROOT))
    if not CONVERTER.exists() or CONVERTER.stat().st_size < 3000:
        print("  не удалось скачать конвертер")
        return False
    text = CONVERTER.read_text(encoding="utf-8", errors="ignore")[:4000]
    if "from conversion import" in text:
        print("  эта версия конвертера пакетная — нужна другая")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Адаптер -> GGUF -> квант")
    ap.add_argument("--adapter", default=str(ROOT / "data" / "train" / "postproc_lora_v2"))
    ap.add_argument("--merged", default=str(ROOT / "data" / "train" / "postproc_merged_v2"))
    ap.add_argument("--name", default="myflow-postproc-qwen3-06b")
    ap.add_argument("--quant", default="Q8_0", choices=("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"))
    ap.add_argument("--skip-merge", action="store_true")
    args = ap.parse_args()

    python = TRAIN_PY if TRAIN_PY.exists() else Path(sys.executable)
    merged = Path(args.merged)
    WORK.mkdir(parents=True, exist_ok=True)

    if not args.skip_merge:
        if not run([str(python), "-X", "utf8", "tools/merge_postproc_lora.py",
                    "--adapter", args.adapter, "--out", args.merged],
                   "сливаю адаптер с базой"):
            return 1
    if not (merged / "model.safetensors").exists():
        print(f"нет слитой модели: {merged}")
        return 1

    if not ensure_converter():
        return 1

    f16 = WORK / f"{args.name}-f16.gguf"
    if not run([str(python), "-X", "utf8", str(CONVERTER), str(merged),
                "--outfile", str(f16), "--outtype", "f16"],
               "конвертирую в GGUF (f16)"):
        return 1

    quantized = WORK / f"{args.name}-{args.quant.lower()}.gguf"
    if not run([str(LLAMA / "llama-quantize.exe"), str(f16), str(quantized), args.quant],
               f"квантую в {args.quant}"):
        return 1

    size = quantized.stat().st_size / 2**20
    print(f"\nготово: {quantized.relative_to(ROOT)} ({size:.1f} МБ)")
    print("Дальше:")
    print(f"  .venv\\Scripts\\python.exe -X utf8 tools\\eval_postproc_hf.py --adapter {args.adapter}")
    print(f"  .venv\\Scripts\\python.exe -X utf8 tools\\ab_postproc.py --model {quantized.relative_to(ROOT)} --also-all")
    print(f"  (если A/B выигран) Copy-Item {quantized.relative_to(ROOT)} models\\{args.name}-{args.quant.lower()}.gguf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
