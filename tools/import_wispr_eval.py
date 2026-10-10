"""Импорт реальных диктовок из базы Wispr Flow для сравнения качества и скорости.

Идея: в базе Wispr (`%APPDATA%\\Wispr Flow\\flow.sqlite`) на каждую фразу лежат
аудио (WAV), их текст ASR, их итоговый текст и замеренная задержка. Это готовая
выборка НАСТОЯЩЕЙ речи пользователя — на ней можно честно измерить наш пайплайн
и сравнить «фраза к фразе» с тем, что делает облачный сервис.

Что делает скрипт (только чтение их базы):
  * достаёт фразы, где есть аудио + asrText + formattedText + задержка;
  * пишет WAV в `data/eval_wispr/audio/<id>.wav`;
  * пишет метаданные в `data/eval_wispr/phrases.jsonl`
    (их тексты, их задержки, длительность, приложение, метрики расхождения).

Данные остаются ЛОКАЛЬНО в проекте (`data/` не попадает в git): это записи
голоса пользователя.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WISPR_DB = Path.home() / "AppData" / "Roaming" / "Wispr Flow" / "flow.sqlite"
OUT_DIR = ROOT / "data" / "eval_wispr"


def export(limit: int | None = None, min_ms: int = 100, max_ms: int = 6000,
           db_path: Path = WISPR_DB, out_dir: Path = OUT_DIR) -> dict:
    """Выгрузить фразы. Возвращает статистику импорта."""
    if not db_path.exists():
        raise SystemExit(f"нет базы Wispr: {db_path}")

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "phrases.jsonl"

    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT transcriptEntityId k, audio, asrText, formattedText, editedText,
                  e2eLatency, clientNetworkLatency, duration, numWords, language, app,
                  micDevice, formattingDivergenceScore, editDistanceToDictated,
                  contentObservationEndReason, status, timestamp
           FROM History
           WHERE audio IS NOT NULL AND LENGTH(audio) > 5000
             AND asrText IS NOT NULL AND TRIM(asrText) != ''
             AND formattedText IS NOT NULL AND TRIM(formattedText) != ''
             AND e2eLatency BETWEEN ? AND ?
           ORDER BY timestamp""",
        (min_ms, max_ms)).fetchall()
    con.close()

    written = 0
    total_bytes = 0
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            if limit is not None and written >= limit:
                break
            key = str(r["k"])
            wav = r["audio"]
            if wav[:4] != b"RIFF":
                continue
            path = audio_dir / f"{key}.wav"
            if not path.exists():
                path.write_bytes(wav)
            total_bytes += len(wav)
            fh.write(json.dumps({
                "id": key,
                "audio": str(path.relative_to(ROOT)).replace("\\", "/"),
                "their_asr": r["asrText"],
                "their_final": r["formattedText"],
                "their_edited": r["editedText"],
                "their_e2e_ms": r["e2eLatency"],
                "their_network_ms": r["clientNetworkLatency"],
                "speech_seconds": r["duration"],
                "words": r["numWords"],
                "language": r["language"],
                "app": r["app"],
                "mic": r["micDevice"],
                "divergence": r["formattingDivergenceScore"],
                "edit_distance": r["editDistanceToDictated"],
                "observation_end": r["contentObservationEndReason"],
                "status": r["status"],
            }, ensure_ascii=False) + "\n")
            written += 1

    return {"phrases": written, "audio_mb": round(total_bytes / 2**20, 1),
            "index": str(out_jsonl), "audio_dir": str(audio_dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Импорт диктовок Wispr для eval")
    ap.add_argument("--limit", type=int, default=None, help="сколько фраз взять")
    ap.add_argument("--min-ms", type=int, default=100, help="минимальная их задержка")
    ap.add_argument("--max-ms", type=int, default=6000, help="максимальная их задержка")
    args = ap.parse_args()
    stats = export(limit=args.limit, min_ms=args.min_ms, max_ms=args.max_ms)
    print(f"Импортировано фраз: {stats['phrases']} ({stats['audio_mb']} МБ аудио)")
    print(f"  индекс: {stats['index']}")
    print(f"  аудио:  {stats['audio_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
