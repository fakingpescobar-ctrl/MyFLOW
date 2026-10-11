"""Поставить финальную модель в models/ и записать состояние дообучения."""
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "data" / "train" / "gguf" / "myflow-postproc-final-q5_k_m.gguf"
dst = ROOT / "models" / "myflow-postproc-qwen3-06b-q5_k_m.gguf"
if not src.exists():
    print(f"нет собранной модели: {src}")
    raise SystemExit(1)

if dst.exists():
    backup = dst.with_suffix(".previous.gguf")
    shutil.copyfile(dst, backup)
    print(f"  резервная копия: {backup.name}")

shutil.copyfile(src, dst)
print(f"  установлена: {dst.name} ({dst.stat().st_size / 2**20:.0f} МБ)")

con = sqlite3.connect(f"file:{ROOT / 'data' / 'myflow.sqlite'}?mode=ro", uri=True)
dictations = con.execute(
    "SELECT COUNT(*) FROM history WHERE asr_text IS NOT NULL "
    "AND TRIM(asr_text) != ''").fetchone()[0]
max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM history").fetchone()[0]
con.close()

state = ROOT / "data" / "train" / "state.json"
state.parent.mkdir(parents=True, exist_ok=True)
state.write_text(json.dumps({
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dictations": int(dictations),
    "max_id": int(max_id),
    "model": dst.name,
    "quant": "Q5_K_M",
    "identity_share": 0.30,
    "note": "обучено на всех диктовках, без эталонов Wispr (честность замера на 624)",
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  состояние: диктовок {dictations}, max id {max_id} -> {state.name}")
