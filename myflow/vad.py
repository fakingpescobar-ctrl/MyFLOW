"""MyFLOW — VAD (Silero v6, ONNX, стрим).

Класс `SileroStreamVad` инкрементально держит hidden-state и контекст (64 сэмпла,
как в faster-whisper), принимает входные чанки Float32 [-1,1] 16k и возвращает
вероятность речи на чанк. Это то, что используется в AUTO-режиме DictationRecorder
вместо старой энергетики (амплитудного порога).

Параметры (подтверждены в myflow/bench_vad.py):
  threshold      — порог вероятности «это речь» (валидация на реальных записях: 0.30)
  min_speech_ms  — мин. длительность речи подряд, чтобы считать её начивающейся фразой
  min_silence_ms — пауза тишины, завершающая фразу (по умолчанию 0.9с как silence_pause_seconds)
  pad_ms         — паддинг-бакет на краях фразы (pad_frames=3 из бенча)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Путь к модели VAD: переменная окружения MYFLOW_SILERO_PATH или
# data/vad_models/silero_v6.onnx рядом с репозиторием (раньше был жёстко
# считался от корня репозитория, поэтому проект можно переносить).
_REPO_ROOT = Path(__file__).resolve().parent.parent
SILERO_PATH = os.environ.get(
    "MYFLOW_SILERO_PATH", str(_REPO_ROOT / "data" / "vad_models" / "silero_v6.onnx")
)
SR = 16000


class SileroStreamVad:
    """Стрим-обёртка Silero: вход чанк Float32 [-1,1] 16k, выход вероятность [0,1]."""

    def __init__(self, model_path: str = SILERO_PATH, threshold: float = 0.30) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Сброс состояния (новый поток речи / тест)."""
        self.hidden = np.zeros((2, 1, 128), dtype=np.float32)
        self._tail = np.zeros(64, dtype=np.float32)   # контекст 64 сэмпла

    def process(self, indata: np.ndarray) -> float:
        """Один чанк Float32 [-1,1] (произвольной длины) → вероятность речи (float).

        Инкрементально апдейтит hidden и контекст. Вход режется на под-чанки
        ≤512 (модель v6: feed = контекст64 + под-чанк должен быть ≤640, а под-чанк
        ≥128). Возвращается вероятность последнего под-чанка — достаточно гладко
        для фреймового решения.
        """
        if len(indata) == 0:
            return 0.0
        piece = np.asarray(indata, dtype=np.float32).reshape(-1)
        prob = 0.0
        at = 0
        while at < len(piece):
            take = min(512, len(piece) - at)
            sub = piece[at:at + take]
            at += take
            feed_input = np.concatenate([self._tail, sub])   # [контекст 64 | под-чанк]
            self._tail = sub[-64:]
            out, self.hidden = self.session.run(
                ["output", "stateN"],
                {"input": feed_input[None].astype(np.float32),
                 "state": self.hidden,
                 "sr": np.array(SR, dtype=np.int64)})
            prob = float(out[0][0])
        return prob

def _silero_smoketest() -> None:
    import time
    v = SileroStreamVad()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        chunk = (rng.standard_normal(320) * 0.001).astype(np.float32)  # тишина
        p = v.process(chunk)
    dt = (time.perf_counter() - t0) / n
    print(f"smoketest: {n} чанков, среднее {dt*1e3:.2f} мс/чанк, последняя prob={p:.3f}")


if __name__ == "__main__":
    _silero_smoketest()