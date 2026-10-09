"""MyFLOW — A/B ASR-бэкендов на своих записях (whisper-server vs GigaAM).

Берём файлы из `data/validation/*.wav` (реальная русская диктовка, команды, ТВ-шум)
и прогоняем через оба бэкенда: печатаем латентность и оба текста рядом, чтобы
можно было судить качество глазами, плюс считаем:
  * WER относительно «эталона» (если он задан в файле `data/validation/gt.json`
    или в словаре ниже) — иначе только согласие между движками;
  * уверенность/галлюцинации: доля латиницы в выводе (whisper любит выдавать
    «Thank you.» / «Sous-titrage» на тишине и шуме).

Запуск:
    .venv\\Scripts\\python -X utf8 tools\\eval_asr.py
    .venv\\Scripts\\python -X utf8 tools\\eval_asr.py --only whisper
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.asr import WhisperClient, make_asr_client  # noqa: E402
from myflow.config import AsrConfig  # noqa: E402

VALIDATION = ROOT / "data" / "validation"

# Эталоны для самых длинных записей — из прошлых прогонов (можно уточнить руками
# в data/validation/gt.json: {"val_dictation.wav": "…текст…"}).
GT_FALLBACK: dict[str, str] = {
    "val_dictation.wav": ("Ну что, теперь ваша очередь. Так, и чё, а где живой индикатор? "
                          "Нихуя не понимаю. Что за ебота, блять? Ёбаный рот."),
}


def load_gt() -> dict[str, str]:
    import json
    path = VALIDATION / "gt.json"
    gt = dict(GT_FALLBACK)
    if path.exists():
        try:
            gt.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return gt


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise ValueError(f"{path.name}: ожидалось 16 кГц моно")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()


def normalize(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    r, h = normalize(reference), normalize(hypothesis)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isascii()) / len(letters)


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B ASR: whisper vs GigaAM")
    ap.add_argument("--only", choices=("whisper", "gigaam", "both"), default="both")
    ap.add_argument("--limit", type=int, default=0, help="сколько файлов брать (0 = все)")
    ap.add_argument("--whisper-url", default="http://127.0.0.1:8081",
                    help="URL whisper-сервера; можно несколько через запятую, "
                         "с меткой: \"http://127.0.0.1:8081|turbo,http://127.0.0.1:8084|large-v3\"")
    args = ap.parse_args()

    files = sorted(VALIDATION.glob("*.wav"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"нет wav в {VALIDATION}")
        return 1

    clients = {}
    if args.only in ("whisper", "both"):
        for spec in args.whisper_url.split(","):
            spec = spec.strip()
            if not spec:
                continue
            url, _, label = spec.partition("|")
            label = (label or url).strip()
            clients[f"whisper:{label}"] = WhisperClient(
                AsrConfig(backend="whisper", url=url.rstrip("/")))
    if args.only in ("gigaam", "both"):
        g = AsrConfig(backend="gigaam")
        client = make_asr_client(g)
        clients["gigaam"] = client
        warm = getattr(client, "warmup", None)
        if warm is not None:
            print("грею GigaAM (первая загрузка ONNX ~20-30 с)...")
            t0 = time.perf_counter()
            if not warm():
                print(f"GigaAM недоступна: {getattr(client, 'load_error', '?')}")
                clients.pop("gigaam", None)
            else:
                print(f"   готово за {time.perf_counter() - t0:.1f} с")

    if not clients:
        print("нет доступных бэкендов")
        return 1

    gt = load_gt()
    stats: dict[str, dict] = {name: {"lat": [], "rtf": [], "wer": [], "latin": [], "texts": []}
                              for name in clients}

    for path in files:
        pcm = read_wav(path)
        secs = len(pcm) / 16000
        print(f"\n=== {path.name} ({secs:.1f} с) ===")
        for name, client in clients.items():
            t0 = time.perf_counter()
            try:
                res = client.transcribe(pcm)
                ms = int((time.perf_counter() - t0) * 1000)
                text = res.text
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<8} ОШИБКА: {exc!r}")
                continue
            stats[name]["lat"].append(ms)
            stats[name]["rtf"].append(ms / 1000 / max(0.1, secs))
            stats[name]["latin"].append(latin_ratio(text))
            stats[name]["texts"].append(text)
            marker = ""
            if path.name in gt:
                w = wer(gt[path.name], text)
                stats[name]["wer"].append(w)
                marker = f" WER={w:.2f}"
            print(f"  {name:<8} {ms:5d} мс (RTF {ms / 1000 / secs:.3f}){marker}")
            print(f"           {text[:150]!r}")

    print("\n================ ИТОГ ================")
    for name, data in stats.items():
        if not data["lat"]:
            continue
        lat = sorted(data["lat"])
        line = (f"  {name:<8} файлов={len(lat):<3} "
                f"p50={statistics.median(lat):.0f} мс  "
                f"средний RTF={statistics.mean(data['rtf']):.3f}  "
                f"латиница={statistics.mean(data['latin']) * 100:.1f} %")
        if data["wer"]:
            line += f"  средний WER={statistics.mean(data['wer']):.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
