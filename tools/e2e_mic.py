"""Интеграционная проба: запись с микрофона -> whisper -> печать.

Не автотест (нет assert'ов): инструмент ручной проверки пайплайна.
Запуск: .venv\\Scripts\\python -X utf8 tools\\e2e_mic.py [seconds=4]
Учитывает asr.input_device из data/settings.json (раньше брал системный).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd

from myflow.asr import WhisperClient
from myflow.config import AsrConfig, SettingsStore

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0

cfg = AsrConfig()
try:
    cfg = SettingsStore().get().asr          # url/язык/устройство из настроек
except Exception as exc:                     # битый settings.json — не повод падать
    print(f"[e2e] settings.json не прочитан ({exc}) — беру дефолты")

print(f"[e2e] пишу {SECONDS}s, устройство={cfg.input_device or 'системное'}, "
      f"язык={cfg.language} — говори что-нибудь...")
audio = sd.rec(int(SECONDS * 16000), samplerate=16000, channels=1,
               dtype="float32", device=cfg.input_device)
sd.wait()
pcm = np.clip(audio[:, 0], -1, 1)
pcm = (np.where(pcm < 0, pcm * 32768, pcm * 32767)).astype(np.int16)
peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
print(f"[e2e] записано {len(pcm)} сэмплов, peak={peak:.3f}")
if peak < 0.01:
    print("[e2e] ВНИМАНИЕ: сигнала почти нет — проверь микрофон/индекс устройства")

client = WhisperClient(cfg)
t0 = time.perf_counter()
res = client.transcribe(pcm)
ms = int((time.perf_counter() - t0) * 1000)
print(f"[e2e] whisper ({ms} мс, RTF={ms / max(1, len(pcm) / 16):.2f}): lang={res.language}")
print(f"[e2e] TEXT: {res.text!r}")
if not res.text:
    print("[e2e] FAIL: пустой текст (тишина или whisper-server лежит)")
    sys.exit(1)
print("[e2e] OK: пайплайн работает")
