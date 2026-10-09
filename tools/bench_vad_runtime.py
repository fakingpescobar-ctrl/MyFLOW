"""MyFLOW — VAD-бенч на ПАРАМЕТРАХ РАНТАЙМА (а не на произвольных).

Прошлый бенч (`validate_silero.py`) гонял min_speech=15 кадров (480 мс) и
min_silence=20/40 кадров (640/1280 мс), тогда как в бою MyFLOW использует:
    порог Silero        = 0.30   (`vad_threshold`)
    минимум речи        = 180 мс (`audio._MIN_SPEECH_MS`)
    минимум фразы       = 300 мс (`_MIN_UTTERANCE_MS`)
    пауза-автостоп      = 900 мс (`silence_pause_seconds`, по умолчанию)
    пик тишины          = 0.01   (`silence_amplitude`)
Значит прошлый вывод «допустимый порог 0.30 при min_silence 40» ничего не говорил
о реальном режиме. Здесь всё считается в кадрах по 10 мс и ровно с этими гейтами.

GROUND TRUTH — проверено вручную (нарезка сегментов + распознавание GigaAM):
    val_dict_1.wav  = ДВЕ фразы («Привет. Как дела?» + «Нужно купить хлеб и масло.»)
    val_noise.wav   = речи нет вообще (шорох/стук) -> любое срабатывание = ложное
    val_tv_*.wav    = речь есть (пользователь + телевизор; VAD не различает, кто говорит)
Поэтому прошлый критерий «ровно 1 фраза на файл» был неверным: при паузе 1300 мс
(как в старом бенче) две разные фразы склеивались в одну, и бенч «проходил» на
артефакте. Здесь честные критерии:
    A (критично) val_noise -> 0 срабатываний (нет ложных на не-речи);
    B каждый файл с речью -> >= 1 фраза (ничего не потеряно);
    C цена по CPU (мс на секунду аудио).

Запуск:
    .venv\\Scripts\\python -X utf8 tools\\bench_vad_runtime.py
    .venv\\Scripts\\python -X utf8 tools\\bench_vad_runtime.py --models silero,ten
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import bench_vad as bv  # noqa: E402

VALIDATION = ROOT / "data" / "validation"
MODELS = ROOT / "data" / "vad_models"

FRAME_MS = 10                      # адаптеры bench_vad приводят всё к 10 мс кадрам
RUNTIME = dict(threshold=0.30, min_speech_ms=180, min_silence_ms=900,
               min_utterance_ms=300, silence_amplitude=0.01)

GROUPS = [
    ("dict", [f"val_dict_{i}.wav" for i in range(1, 6)], 1, "речь"),
    ("cmd", [f"val_cmd_{i}.wav" for i in range(1, 5)], 1, "речь"),
    ("noise", ["val_noise.wav"], 0, "не-речь (критично: 0 срабатываний)"),
    ("tv0", ["val_tv_0.wav"], 1, "речь ТВ"),
    ("tv", [f"val_tv_{i}.wav" for i in range(1, 5)], 1, "речь"),
]


def build_models(which: set[str]) -> dict[str, object]:
    models: dict[str, object] = {}
    if "silero" in which:
        models["silero"] = bv.SileroVad(MODELS / "silero_v6.onnx")
    if "ten" in which:
        models["ten"] = bv.TenVadAdapter(threshold=RUNTIME["threshold"])
    if "firered" in which:
        models["firered"] = bv.FireRedVad(MODELS / "fireredvad_vad.onnx",
                                          MODELS / "cmvn.ark")
    if "firered_stream" in which:
        models["firered_stream"] = bv.FireRedStreamVad(MODELS / "fireredvad_stream_vad.onnx",
                                                       MODELS / "cmvn.ark")
    return models


def run_file(model, wav: np.ndarray, thr: float, min_speech_frames: int,
             min_silence_frames: int) -> list[tuple[float, float]]:
    probs = model.compute(wav)
    return bv.segment_frames(probs, thr, min_speech_frames=min_speech_frames,
                             min_silence_frames=min_silence_frames, pad_frames=0)


def apply_runtime_gates(wav: np.ndarray, segs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Те же гейты, что в audio.py: длительность фразы и пик тишины."""
    kept = []
    for start, end in segs:
        duration_ms = (end - start) * 1000
        i0, i1 = int(start * 16000), int(end * 16000)
        peak = float(np.abs(wav[i0:i1]).max()) if i1 > i0 else 0.0
        if duration_ms < RUNTIME["min_utterance_ms"] or peak < RUNTIME["silence_amplitude"]:
            continue
        kept.append((start, end))
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description="VAD-бенч на параметрах рантайма")
    ap.add_argument("--models", default="silero,ten,firered,firered_stream")
    ap.add_argument("--thresholds", default="0.25,0.30,0.35,0.40")
    ap.add_argument("--silences", default="450,700,900", help="пауза в мс")
    args = ap.parse_args()

    which = {m.strip() for m in args.models.split(",") if m.strip()}
    models = build_models(which)
    if not models:
        print("нет доступных моделей")
        return 1

    # читаем все файлы один раз
    audio: dict[str, np.ndarray] = {}
    for _, files, _, _ in GROUPS:
        for name in files:
            wav, sr = sf.read(str(VALIDATION / name), dtype="float32")
            if sr != 16000:
                raise SystemExit(f"{name}: ожидалось 16 кГц, получено {sr}")
            audio[name] = wav

    min_speech_frames = RUNTIME["min_speech_ms"] // FRAME_MS     # 18
    print(f"Параметры рантайма: порог={RUNTIME['threshold']}, речь>={RUNTIME['min_speech_ms']} мс "
          f"({min_speech_frames} кадров), пауза>={RUNTIME['min_silence_ms']} мс, "
          f"фраза>={RUNTIME['min_utterance_ms']} мс, пик>={RUNTIME['silence_amplitude']}")
    print("Критерии: A=val_noise без ложных срабатываний (критично), "
          "B=файлы с речью дают >=1 фразу, C=цена CPU\n")

    # ---- 1) режим рантайма ----
    print("=== 1) Чистый режим рантайма (порог 0.30, пауза 900 мс) ===")
    header = f"{'модель':<16}" + "".join(f"{g[0]:>9}" for g in GROUPS) + \
             "   A    B    C: мс/с аудио"
    print(header)
    print("-" * len(header))
    for name, model in models.items():
        counts, total_audio, total_ms = {}, 0.0, 0.0
        for group, files, _, _ in GROUPS:
            cnt = []
            for fname in files:
                wav = audio[fname]
                t0 = time.perf_counter()
                segs = run_file(model, wav, RUNTIME["threshold"], min_speech_frames,
                                RUNTIME["min_silence_ms"] // FRAME_MS)
                total_ms += (time.perf_counter() - t0) * 1000
                total_audio += len(wav) / 16000
                cnt.append(len(apply_runtime_gates(wav, segs)))
            counts[group] = cnt
        noise_ok = max(counts["noise"]) == 0
        speech_ok = all(min(counts[g]) >= 1 for g, _, exp, _ in GROUPS if exp >= 1)
        row = f"{name:<16}"
        for group, _, _, _ in GROUPS:
            c = counts[group]
            row += f"{f'{min(c)}-{max(c)}':>9}"
        row += (f"   {'OK' if noise_ok else 'ЛОЖН':<4} "
                f"{'OK' if speech_ok else 'ПОТЕРЯ':<4} {total_ms / total_audio:6.1f}")
        print(row)

    # ---- 2) чувствительность ----
    print("\n=== 2) Чувствительность: порог × пауза (сколько файлов «звучит верно») ===")
    header2 = f"{'модель':<16}{'пауза':>8}" + "".join(f"{t:>8}" for t in args.thresholds.split(","))
    print(header2)
    print("-" * len(header2))
    for name, model in models.items():
        for silence_ms in [int(x) for x in args.silences.split(",")]:
            row = f"{name:<16}{silence_ms:>8}"
            for thr in [float(x) for x in args.thresholds.split(",")]:
                ok_files, total_files = 0, 0
                for group, files, exp, _ in GROUPS:
                    for fname in files:
                        total_files += 1
                        wav = audio[fname]
                        segs = apply_runtime_gates(wav, run_file(
                            model, wav, thr, min_speech_frames, silence_ms // FRAME_MS))
                        # не-речь: ни одного срабатывания; речь: хотя бы одна фраза
                        good = (len(segs) == 0) if exp == 0 else (len(segs) >= 1)
                        ok_files += int(good)
                row += f"{ok_files}/{total_files:>4}"
            print(row)

    # ---- 3) что слышно на шумных файлах при режиме рантайма ----
    print("\n=== 3) Детали по «сложным» файлам (режим рантайма) ===")
    for name, model in models.items():
        print(f"  [{name}]")
        for fname in ("val_noise.wav", "val_tv_0.wav", "val_tv_1.wav", "val_dict_1.wav"):
            wav = audio[fname]
            segs = apply_runtime_gates(wav, run_file(
                model, wav, RUNTIME["threshold"], min_speech_frames,
                RUNTIME["min_silence_ms"] // FRAME_MS))
            total = sum(e - s for s, e in segs)
            txt = ", ".join(f"{s:.2f}-{e:.2f}" for s, e in segs[:4]) or "—"
            print(f"    {fname:<16} фраз={len(segs):<2} речи={total:5.2f} с  {txt}")

    print("\nВывод: критерий A (нет ложных на не-речи) важнее всего — ложное срабатывание "
          "в AUTO-режиме даёт пустую запись и мусор в истории. Смотри также цену CPU: "
          "она умножается на каждую секунду диктовки.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
