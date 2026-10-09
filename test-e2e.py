"""Интеграционный тест: запись с микрофона -> whisper -> (llm) -> вывод.

Запуск: .venv\\Scripts\\python test-e2e.py [seconds=4]
"""
import sys
import time

sys.path.insert(0, r"C:\Projects\MyFLOW")

import sounddevice as sd
import numpy as np
import requests

from myflow.asr import WhisperClient, pcm_int16_to_wav
from myflow.config import AsrConfig

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0

print(f"[e2e] записываю {SECONDS}s с микрофона — говори что-нибудь...")
audio = sd.rec(int(SECONDS * 16000), samplerate=16000, channels=1, dtype="float32")
sd.wait()
pcm = np.clip(audio[:, 0], -1, 1)
pcm = (np.where(pcm < 0, pcm * 32768, pcm * 32767)).astype(np.int16)
print(f"[e2e] записано {len(pcm)} сэмплов, max_amp={np.max(np.abs(pcm)):.3f}")

w = WhisperClient(AsrConfig())
t0 = time.perf_counter()
res = w.transcribe(pcm)
ms = int((time.perf_counter() - t0) * 1000)
print(f"[e2e] whisper ({ms}ms): lang={res.language}")
print(f"[e2e] TEXT: {res.text!r}")

print("[e2e] ok") if not res.text else print("[e2e] pipeline works")