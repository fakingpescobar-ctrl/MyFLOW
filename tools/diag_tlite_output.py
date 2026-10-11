"""Что реально отвечает T-lite: смотрим выводы, а не только цифры.

Расстояния 2.8-4.9 означают, что модель пишет НАМНОГО больше эталона (в 3-5 раз) —
то есть не оформляет текст, а что-то объясняет. Здесь видно, что именно, и есть ли
шанс поправить промптом.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from myflow.config import SettingsStore   # noqa: E402
from myflow.polish import PolishClient    # noqa: E402

MODEL = ROOT / "data" / "train" / "tlite" / "T-lite-instruct-0.1.Q4_K_M.gguf"
PORT = 8087
PAIRS = ROOT / "data" / "train" / "final" / "postproc_val.jsonl"


def main() -> int:
    pairs = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines()
             if l.strip()][:3]
    server = ROOT / "bin" / "llama-cpp" / "llama-server.exe"
    log = ROOT / "data" / "train" / "tlite_probe.log"
    with open(log, "ab") as fh:
        proc = subprocess.Popen(
            [str(server), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
             "-c", "8192", "-np", "1", "-ngl", "99", "-fa", "on", "--jinja"],
            stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    try:
        url = f"http://127.0.0.1:{PORT}/v1"
        for _ in range(180):
            try:
                if requests.get(f"{url}/models", timeout=2.0).ok:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
        print("  T-lite готов\n")
        cfg = SettingsStore().get().polish
        cfg.url = url
        cfg.enabled = True
        client = PolishClient(cfg)

        # есть ли у модели шаблон чата вообще
        info = requests.get(f"{url.rsplit('/v1', 1)[0]}/props", timeout=5).json()
        template = (info.get("default_generation_settings") or {})
        print(f"  шаблон чата в модели: {'есть' if info.get('chat_template') else 'НЕТ'}")
        print(f"  контекст: {info.get('n_ctx')} | модель: {info.get('model_path', '')[-40:]}\n")

        for style in ("plain", "postproc"):
            print(f"  === стиль {style} ===")
            for row in pairs:
                src = next(m["content"] for m in row["messages"] if m["role"] == "user")
                dst = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
                result = client.polish(src, style=style, dictionary=[])
                print(f"    вход : {src[:64]}")
                print(f"    цель : {dst[:64]}")
                print(f"    ответ: {(result.text or '')[:150].replace(chr(10), ' | ')}")
            print()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
