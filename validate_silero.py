"""Валидация порога Silero: пофайлово, каждый файл = свой GT.

Группы:
  dict  (val_dict_1..5)   — 5 фраз, GT=1 сегмент на файл
  cmd   (val_cmd_1..4)    — 4 команды, GT=1 сегмент на файл
  noise (val_noise)       — шуршание/стук/тишина, GT=0
  tv0   (val_tv_0)        — телек без речи, GT=0
  tv    (val_tv_1..4)     — речь юзера при работающем телике, GT=1 на файл
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, r"C:\Projects\MyFLOW")
import bench_vad as b

VAL = Path(r"data\validation")
sil = b.SileroVad(b.MODELS_DIR / "silero_v6.onnx")

GROUPS = [
    ("dict", [f"val_dict_{i}.wav" for i in range(1, 6)], 1, "5 фраз (тишина)"),
    ("cmd", [f"val_cmd_{i}.wav" for i in range(1, 5)], 1, "4 команды"),
    ("noise", ["val_noise.wav"], 0, "шум без речи"),
    ("tv0", ["val_tv_0.wav"], 0, "телек без речи"),
    ("tv", [f"val_tv_{i}.wav" for i in range(1, 5)], 1, "речь + телек"),
]


def segs_of(path: Path, thr: float, msil: int) -> tuple[list, float]:
    wav, sr = sf.read(str(path), dtype="float32")
    probs = sil.compute(wav)
    segs = b.segment_frames(probs, thr, min_speech_frames=15, min_silence_frames=msil, pad_frames=3)
    rms = float(np.sqrt((wav ** 2).mean()))
    return segs, rms


print("Для каждой группы: кол-во сегментов (min-max по файлам) / файлов")
print(f"{'thr':<6}{'msil':<6}" + "".join(f"{g[0]:>10}" for g in GROUPS) + "   verdict")
print("-" * 74)
for thr in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
    for msil in (40, 20):
        counts = {}
        for key, files, expect, _ in GROUPS:
            cnts = [len(segs_of(VAL / f, thr, msil)[0]) for f in files]
            counts[key] = cnts
        ok = (max(counts["dict"]) == 1 and min(counts["dict"]) == 1 and
              max(counts["cmd"]) == 1 and min(counts["cmd"]) == 1 and
              max(counts["noise"]) == 0 and
              max(counts["tv0"]) == 0 and
              max(counts["tv"]) == 1 and min(counts["tv"]) == 1)
        row = f"{thr:<6}{msil:<6}"
        for key, _, _, _ in GROUPS:
            c = counts[key]
            row += f"{f'{min(c)}-{max(c)}':>10}"
        row += f"   {'OK' if ok else ''}"
        print(row)
        if ok and thr >= 0.30:
            print(f"\n  >>> ДОПУСТИМЫЙ порог: {thr}, min_silence={msil} фреймов")
print()

# чувствительность: для tv-группы покажем сегменты на пороге 0.30/msil40
for key, files, expect, label in GROUPS:
    print(f"\n[{key}] {label}")
    for f in files:
        segs, rms = segs_of(VAL / f, 0.30, 40)
        fmt = ", ".join(f"{a:.2f}-{b:.2f}" for a, b in segs[:6]) or "—"
        print(f"  {f:<14} rms={rms:.4f} {len(segs)}сег: {fmt}")