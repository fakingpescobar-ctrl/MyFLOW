"""MyFLOW — клиент локального whisper-server (аналог gRPC-стрима Wispr Flow).

whisper.cpp server принимает WAV (16k mono) через POST /inference и отдаёт
{"text", "language", "segments"}. Здесь же — конвертация Int16 PCM -> WAV.
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from myflow.config import AsrConfig, SAMPLE_RATE


@dataclass
class AsrResult:
    text: str
    language: str | None
    latency_ms: int


def pcm_int16_to_wav(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Int16 PCM -> WAV (44-байтный заголовок). Вход уже проверен каллингом."""
    if pcm.dtype != np.int16:
        raise TypeError(f"ожидался int16 PCM, получил {pcm.dtype}")
    n = len(pcm)
    data = pcm.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", len(data),
    )
    return header + data


class WhisperClient:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        # Keep-alive + ретраи: единичный сетевой сбой не должен терять фразу.
        # Ретраим только connect/502-504 (read-timeout не ретраим: повторный
        # инференс на GPU — дорого, лучше честно вернуть ошибку и взять сырой ASR).
        self._session = requests.Session()
        retry = Retry(
            total=2, connect=2, read=0, status=2, backoff_factor=0.3,
            status_forcelist=(502, 503, 504), allowed_methods=("POST",),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def transcribe(self, pcm_int16: np.ndarray) -> AsrResult:
        if pcm_int16.dtype != np.int16:
            raise TypeError(f"ожидался int16 PCM, получил {pcm_int16.dtype}")
        if pcm_int16.size == 0:
            raise ValueError("пустой аудио-буфер — транскрибировать нечего")

        wav = pcm_int16_to_wav(pcm_int16)
        started = time.perf_counter()
        params: dict = {"language": self.cfg.language}
        if self.cfg.initial_prompt:
            params["prompt"] = self.cfg.initial_prompt
        try:
            resp = self._session.post(
                f"{self.cfg.url}/inference",
                files={"file": ("audio.wav", wav, "audio/wav")},
                params=params,
                timeout=(self.cfg.timeout_connect, self.cfg.timeout_read),
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"whisper-server недоступен ({self.cfg.url}): {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(f"whisper-server ответил {resp.status_code}: {resp.text[:200]}")

        try:
            # Явный UTF-8: у whisper.cpp нет charset в заголовке, а resp.json()
            # в этом случае уходит в эвристику (мойибаке на русском).
            payload = json.loads(resp.content.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"whisper-server вернул нечитаемый ответ: {exc!r}") from exc

        text = (payload.get("text") or "").strip()
        language = payload.get("language")
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AsrResult(text=text, language=language, latency_ms=latency_ms)