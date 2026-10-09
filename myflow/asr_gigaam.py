"""MyFLOW — ASR-бэкенд GigaAM v3 (ru-only, ONNX, CPU).

Зачем: Whisper large-v3-turbo — дистиллят, и на русском он слабее ru-специфичных
моделей. GigaAM v3 от Sber (ai-sage/GigaAM-v3) обучен только на русском и даёт
меньше ошибок на именах/числах; вариант **e2e** сразу расставляет пунктуацию и
заглавные буквы (то есть полировке остаётся меньше работы).

Реализация: ONNX-экспорт от istupakov (`gigaam-v3-e2e-ctc`, int8, ~215 МБ) через
пакет `onnx-asr`. Работает на CPU: 30 с аудио ≈ 400 мс (RTF 0.014), то есть
GPU полностью свободен под LLM.

Пакет `onnx-asr` — ОПЦИОНАЛЬНАЯ зависимость: если его нет, бэкенд сообщает
`available = False`, и MyFLOW автоматически остаётся на whisper.

Интерфейс совпадает с WhisperClient: `transcribe(pcm_int16) -> AsrResult`.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

from myflow.config import AsrConfig, SAMPLE_RATE

DEFAULT_MODEL = "gigaam-v3-e2e-ctc"
DEFAULT_QUANTIZATION = "int8"


class GigaAmClient:
    """GigaAM через onnx-asr: модель грузится один раз, инференс на CPU."""

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self._loading = False

    # ---- доступность ----

    @property
    def model_name(self) -> str:
        return getattr(self.cfg, "gigaam_model", DEFAULT_MODEL) or DEFAULT_MODEL

    @property
    def quantization(self) -> str | None:
        return getattr(self.cfg, "gigaam_quantization", DEFAULT_QUANTIZATION) or None

    @staticmethod
    def package_available() -> bool:
        import importlib.util
        import sys
        # Модуль может быть уже подгружен (в тестах — подменён фейком):
        # find_spec на таком модуле падает с ValueError('__spec__ is None').
        if "onnx_asr" in sys.modules:
            return sys.modules["onnx_asr"] is not None
        try:
            return importlib.util.find_spec("onnx_asr") is not None
        except Exception:  # noqa: BLE001
            return False

    @property
    def available(self) -> bool:
        return self.package_available() and not self._load_error

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str:
        return self._load_error or ""

    # ---- загрузка ----

    def warmup(self, timeout: float | None = None) -> bool:
        """Загрузить модель. Возвращает True, если готова.

        Вызывается из фонового потока на старте приложения: первая загрузка
        ONNX-сессии занимает десятки секунд (инициализация + чтение 215 МБ),
        и мы не хотим платить это на первой же диктовке.
        """
        with self._lock:
            if self._model is not None:
                return True
            if self._load_error:
                return False
            if not self.package_available():
                self._load_error = "пакет onnx-asr не установлен (pip install onnx-asr[hub])"
                return False
            self._loading = True
            try:
                import onnx_asr
                # Кэш моделей HF держим рядом с остальными моделями, а не на C:.
                os.environ.setdefault("HF_HOME", str(_cache_dir()))
                started = time.perf_counter()
                self._model = onnx_asr.load_model(self.model_name,
                                                  quantization=self.quantization)
                print(f"[asr] GigaAM {self.model_name} ({self.quantization}) готова за "
                      f"{time.perf_counter() - started:.1f} c")
                return True
            except Exception as exc:  # noqa: BLE001 — не роняем приложение
                self._load_error = f"{exc!r}"
                print(f"[asr] GigaAM недоступна: {self._load_error}")
                return False
            finally:
                self._loading = False

    def health(self) -> bool:
        """Для gigaam «сервер» — это локальная модель: проверяем доступность."""
        if self._model is not None:
            return True
        if self._load_error:
            return False
        return self.package_available()

    # ---- инференс ----

    def transcribe(self, pcm_int16: np.ndarray):
        from myflow.asr import AsrResult
        if pcm_int16.dtype != np.int16:
            raise TypeError(f"ожидался int16 PCM, получил {pcm_int16.dtype}")
        if pcm_int16.size == 0:
            raise ValueError("пустой аудио-буфер — транскрибировать нечего")
        if self._model is None and not self.warmup():
            raise RuntimeError(f"GigaAM не загружена: {self._load_error}")

        started = time.perf_counter()
        audio = pcm_int16.astype(np.float32) / 32768.0
        text = self._model.recognize(audio, sample_rate=SAMPLE_RATE)
        latency_ms = int((time.perf_counter() - started) * 1000)
        from myflow.asr import AsrResult as _R
        return _R(text=(text or "").strip(), language="ru", latency_ms=latency_ms)


def _cache_dir() -> Path:
    env = os.environ.get("MYFLOW_MODELS_DIR")
    base = Path(env) if env else Path(r"X:\llm\models")
    return base / "hf-cache"
