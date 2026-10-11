"""Сколько данных накопилось для дообучения: состояние истории и модели."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

con = sqlite3.connect(f"file:{ROOT / 'data' / 'myflow.sqlite'}?mode=ro", uri=True)
total = con.execute("SELECT COUNT(*) FROM history").fetchone()[0]
with_text = con.execute(
    "SELECT COUNT(*) FROM history WHERE asr_text IS NOT NULL "
    "AND TRIM(asr_text) != ''").fetchone()[0]
with_edit = con.execute(
    "SELECT COUNT(*) FROM history WHERE TRIM(COALESCE(edited_text,'')) != ''").fetchone()[0]
max_id = con.execute("SELECT MAX(id) FROM history").fetchone()[0]
con.close()
print(f"  записей в истории: {total} | с текстом ASR: {with_text} | "
      f"с твоими правками: {with_edit} | max id: {max_id}")

state = ROOT / "data" / "train" / "state.json"
if state.exists():
    data = json.loads(state.read_text(encoding="utf-8"))
    print(f"  последнее обучение: {data.get('trained_at')} | "
          f"диктовок было {data.get('dictations')} | модель {data.get('model')}")
    print(f"  новых с тех пор: {max(0, (max_id or 0) - int(data.get('max_id') or 0))}")
else:
    print("  состояния обучения нет (модель ещё не переобучалась через пульт)")
