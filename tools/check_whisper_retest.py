"""Что есть для замера whisper на том же аудио (без нового чтения)."""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "myflow.sqlite"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute(
    "select id, audio_path, asr_text, polished_text, edited_text from history "
    "where id between 500 and 1000 and coalesce(audio_path,'') <> '' order by id")]
con.close()

existing = [r for r in rows if (ROOT / r["audio_path"]).exists()]
size = sum((ROOT / r["audio_path"]).stat().st_size for r in existing
           if (ROOT / r["audio_path"]).exists())
print(f"  записей этой начитки с аудио: {len(rows)}")
print(f"  файлов реально на диске: {len(existing)} "
      f"({size / 2**20:.1f} МБ)")
if existing:
    sample = existing[0]
    print(f"  пример: id={sample['id']}, {Path(sample['audio_path']).name}, "
          f"символов ASR={len(sample['asr_text'] or '')}")

print("\n  чем распознавать (whisper):")
bins = sorted((ROOT / "bin" / "whisper").rglob("*.exe"))
for path in bins:
    print(f"    {path.relative_to(ROOT)}")
models = sorted((ROOT / "models").glob("ggml-*.bin")) + \
    sorted((ROOT / "bin" / "whisper").rglob("ggml-*.bin"))
for path in models:
    print(f"    модель: {path.relative_to(ROOT)} "
          f"({path.stat().st_size / 2**20:.0f} МБ)")
if not models:
    print("    моделей whisper в models/ не нашёл")

print("\n  порты: 8081 (whisper), 8082 (Oracle), 8083 (постпроцессор)")
