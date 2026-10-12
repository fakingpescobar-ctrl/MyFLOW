"""Whisper против GigaAM на ОДНОМ И ТОМ ЖЕ звуке.

Читать заново не нужно: аудио начитки лежит в истории (колонка audio_path), у нас
сохранены последние 50 фраз. Прогоняем эти же файлы через whisper и сравниваем с
тем, что услышал GigaAM.

Честность сравнения:
  * звук один и тот же (один wav на оба движка);
  * эталонное предложение выбирается по лучшему совпадению с GigaAM, и ПОТОМ к этому
    же предложению меряется whisper — иначе каждый движок выбрал бы себе удобное;
  * аудио .ogg переводим в wav 16 кГц моно, как требует whisper-server.

    python tools/whisper_vs_gigaam.py --limit 50
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_readaloud import normalize, split_sentences, word_distance  # noqa: E402

DB = ROOT / "data" / "myflow.sqlite"
REFERENCE = ROOT / "data" / "bench" / "private" / "klinok-ch1.txt"


def load_records() -> list[dict]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "select id, audio_path, asr_text, polished_text, edited_text from history "
        "where coalesce(audio_path,'') <> '' order by id")]
    con.close()
    return [r for r in rows if (ROOT / r["audio_path"]).exists()]


def to_wav(source: Path, target: Path) -> bool:
    """ogg -> wav 16 кГц моно (whisper-server другого не принимает)."""
    try:
        import soundfile as sf
        data, rate = sf.read(str(source), always_2d=True)
        mono = data.mean(axis=1)
        if rate != 16000:
            import numpy as np
            count = int(len(mono) * 16000 / rate)
            mono = np.interp(np.linspace(0, len(mono), count, endpoint=False),
                             np.arange(len(mono)), mono)
            rate = 16000
        sf.write(str(target), mono, rate, subtype="PCM_16")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    не смог конвертировать {source.name}: {exc!r}")
        return False


def ask_whisper(server: str, wav: Path, timeout: float = 120.0) -> str:
    with open(wav, "rb") as handle:
        files = {"file": (wav.name, handle, "audio/wav")}
        data = {"language": "ru", "response_format": "json", "temperature": "0"}
        response = requests.post(f"{server}/inference", files=files, data=data,
                                 timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return (payload.get("text") or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    args = ap.parse_args()

    sentences = [normalize(s).split() for s in
                 split_sentences(args.reference.read_text(encoding="utf-8"))]
    sentences = [s for s in sentences if s]
    vocabulary = {word for sentence in sentences for word in sentence}

    records = [r for r in load_records()
               if (r["asr_text"] or "").strip()][-args.limit:]
    print(f"  фраз с аудио: {len(records)}")

    results = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for number, row in enumerate(records, start=1):
            source = ROOT / row["audio_path"]
            wav = Path(tmp) / f"{row['id']}.wav"
            if not to_wav(source, wav):
                continue
            started = time.perf_counter()
            try:
                whisper_text = ask_whisper(args.server, wav)
            except Exception as exc:  # noqa: BLE001
                print(f"  whisper-server не ответил на {row['id']}: {exc!r}")
                break
            elapsed = time.perf_counter() - started

            giga_words = normalize(row["asr_text"] or "").split()
            whisper_words = normalize(whisper_text).split()
            if not whisper_words:
                continue

            # предложение выбирает GigaAM, whisper меряется по нему же
            target = None
            best = None
            for sentence in sentences:
                if len(set(sentence) & set(giga_words)) == 0:
                    continue
                distance, _ops = word_distance(sentence, giga_words)
                if best is None or distance < best[0]:
                    best, target = (distance, sentence), sentence
            if target is None:
                continue
            giga_distance, _ = word_distance(target, giga_words)
            whisper_distance, _ = word_distance(target, whisper_words)
            results.append({
                "id": row["id"], "seconds": elapsed,
                "giga": giga_distance, "whisper": whisper_distance,
                "words": len(target),
            })
            if number % 10 == 0:
                print(f"    обработано {number}...", flush=True)

    if not results:
        print("  сравнить нечего")
        return 1

    total_words = sum(r["words"] for r in results)
    giga_errors = sum(r["giga"] for r in results)
    whisper_errors = sum(r["whisper"] for r in results)
    whisper_better = sum(1 for r in results if r["whisper"] < r["giga"])
    giga_better = sum(1 for r in results if r["giga"] < r["whisper"])
    same = len(results) - whisper_better - giga_better
    seconds = [r["seconds"] for r in results]

    print(f"\n  фраз сравнено: {len(results)}")
    print(f"  WER GigaAM:  {giga_errors / total_words:.4f}  ({giga_errors} ошибок)")
    print(f"  WER whisper: {whisper_errors / total_words:.4f}  ({whisper_errors} ошибок)")
    print(f"  по фразам: whisper лучше {whisper_better}, GigaAM лучше {giga_better}, "
          f"одинаково {same}")
    if seconds:
        ordered = sorted(seconds)
        print(f"  время whisper на фразу: медиана {ordered[len(ordered) // 2]:.1f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
