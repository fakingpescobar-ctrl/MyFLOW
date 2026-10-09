"""MyFLOW — проверка голоса (speaker verification).

Зачем: VAD отличает речь от тишины, но не отличает ВАШУ речь от чужой. В комнате
с телевизором или коллегами он честно «слышит речь» и делает фразы из чужого
голоса — энергетикой это не лечится (замеры: тихая речь 0.019 против фона 0.026,
диапазоны пересекаются). Здесь мы это решаем нейросетью: считаем эмбеддинг
голоса и сравниваем с профилем владельца.

Модель: **WeSpeaker ResNet34** (VoxCeleb), ONNX-экспорт onnx-community
(`wespeaker-voxceleb-resnet34-LM`, 25 МБ). Вход — 80-мерный kaldi-fbank
(25/10 мс, hamming), выход — эмбеддинг 256. Инференс на CPU, ~10-20 мс на
фразу, GPU не нужен.

Ключевая деталь качества — **CMN** (вычитание среднего по кадрам): без него
разделение голосов плохое (свой 0.74-0.91 против чужого 0.55-0.70), с ним —
чистое (свой 0.56-0.83, чужой 0.01-0.09). Замерено на своих записях.

Профиль — центроид L2-нормированных эмбеддингов, лежит локально в
`data/voiceprint.npz` (это биометрический шаблон, наружу не уходит).
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from myflow.config import DATA_DIR, SAMPLE_RATE

DEFAULT_THRESHOLD = 0.25
# Замер на своих записях: свой голос 0.559-0.834, чужой 0.011-0.090.
# Порог 0.25 даёт запас 2.2x снизу (тихая своя фраза 0.364) и 2.8x сверху.
DEFAULT_MIN_SECONDS = 0.8
DEFAULT_ENROLL_MIN_SCORE = 0.45
FBANK_BINS = 80
FBANK_FRAME_MS = 25.0
FBANK_SHIFT_MS = 10.0
MIN_FRAMES = 9


def default_model_path() -> Path:
    import os
    env = os.environ.get("MYFLOW_VOICEPRINT_MODEL")
    if env:
        return Path(env)
    models = os.environ.get("MYFLOW_MODELS_DIR")
    base = Path(models) if models else Path(r"X:\llm\models")
    return base / "voiceprint" / "model.onnx"


def default_profile_path() -> Path:
    import os
    env = os.environ.get("MYFLOW_VOICEPRINT_PROFILE")
    return Path(env) if env else DATA_DIR / "voiceprint.npz"


class SpeakerEmbedder:
    """fbank (kaldi) -> эмбеддинг голоса, через onnxruntime на CPU."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else default_model_path()
        self._session = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # ---- доступность ----

    @staticmethod
    def deps_available() -> bool:
        import importlib.util
        for mod in ("onnxruntime", "kaldi_native_fbank"):
            try:
                if importlib.util.find_spec(mod) is None:
                    return False
            except Exception:  # noqa: BLE001
                return False
        return True

    @property
    def available(self) -> bool:
        return self.deps_available() and self.model_path.exists() and not self._load_error

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def load_error(self) -> str:
        return self._load_error or ""

    # ---- загрузка ----

    def warmup(self) -> bool:
        with self._lock:
            if self._session is not None:
                return True
            if self._load_error:
                return False
            if not self.deps_available():
                self._load_error = ("нет зависимостей: pip install onnxruntime "
                                    "kaldi_native_fbank")
                return False
            if not self.model_path.exists():
                self._load_error = f"нет модели: {self.model_path}"
                return False
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"])
                return True
            except Exception as exc:  # noqa: BLE001
                self._load_error = repr(exc)
                return False

    # ---- признаки и эмбеддинг ----

    def fbank(self, pcm_int16: np.ndarray) -> np.ndarray | None:
        """kaldi-совместимый fbank: то, на чём обучалась WeSpeaker."""
        import kaldi_native_fbank as knf
        pcm = np.clip(pcm_int16, -32768, 32767).astype(np.int16)
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = FBANK_FRAME_MS
        opts.frame_opts.frame_shift_ms = FBANK_SHIFT_MS
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.round_to_power_of_two = True
        opts.mel_opts.num_bins = FBANK_BINS
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(SAMPLE_RATE, pcm.tolist())
        n = fbank.num_frames_ready
        if n < MIN_FRAMES:
            return None
        return np.vstack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)

    def embed(self, pcm_int16: np.ndarray) -> np.ndarray | None:
        """L2-нормированный эмбеддинг голоса или None, если аудио мало."""
        if pcm_int16 is None or len(pcm_int16) == 0:
            return None
        if self._session is None and not self.warmup():
            return None
        feats = self.fbank(pcm_int16)
        if feats is None:
            return None
        feats = feats - feats.mean(axis=0, keepdims=True)      # CMN — см. докстринг
        out = self._session.run(None, {"input_features": feats[None]})[0][0]
        norm = float(np.linalg.norm(out))
        if norm <= 1e-9:
            return None
        return out / norm


class VoiceProfile:
    """Центроид голоса владельца; хранится локально в npz."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else default_profile_path()
        self.centroid: np.ndarray | None = None
        self.count: int = 0
        self.load()

    @property
    def exists(self) -> bool:
        return self.centroid is not None and self.count > 0

    def load(self) -> bool:
        try:
            if not self.path.exists():
                return False
            data = np.load(self.path)
            centroid = np.asarray(data["centroid"], dtype=np.float32)
            count = int(data["count"][0]) if data["count"].shape else int(data["count"])
            if centroid.ndim != 1 or centroid.size == 0:
                return False
            norm = float(np.linalg.norm(centroid))
            self.centroid = centroid / norm if norm > 1e-9 else None
            self.count = count
            return self.centroid is not None
        except Exception:  # noqa: BLE001 — битый профиль не должен ронять приложение
            self.centroid, self.count = None, 0
            return False

    def save(self) -> bool:
        if self.centroid is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, centroid=self.centroid,
                     count=np.asarray([self.count], dtype=np.int64))
            return True
        except Exception:  # noqa: BLE001
            return False

    def add(self, embedding: np.ndarray, max_samples: int = 40) -> None:
        """Онлайн-обновление центроида (скользящее среднее по эмбеддингам)."""
        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-9:
            return
        emb = emb / norm
        if self.centroid is None:
            self.centroid = emb
            self.count = 1
        else:
            # Ограничиваем «память» профиля: старые вклады постепенно размываются,
            # но и не перевешивают свежие записи (микрофон/комната меняются).
            weight = 1.0 / (min(self.count, max_samples) + 1)
            merged = (1.0 - weight) * self.centroid + weight * emb
            mnorm = float(np.linalg.norm(merged))
            self.centroid = merged / mnorm if mnorm > 1e-9 else self.centroid
            self.count += 1

    def score(self, embedding: np.ndarray) -> float:
        if self.centroid is None or embedding is None:
            return -1.0
        return float(np.dot(self.centroid, embedding))

    def reset(self) -> None:
        self.centroid, self.count = None, 0
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception:  # noqa: BLE001
            pass


class VoiceGate:
    """Проверка «это владелец?» + авто-пополнение профиля своими фразами."""

    def __init__(self, enabled: bool = True, threshold: float = DEFAULT_THRESHOLD,
                 min_seconds: float = DEFAULT_MIN_SECONDS,
                 auto_enroll: bool = True,
                 enroll_min_score: float = DEFAULT_ENROLL_MIN_SCORE,
                 model_path: Path | None = None,
                 profile_path: Path | None = None) -> None:
        self.enabled = enabled
        self.threshold = threshold
        self.min_seconds = min_seconds
        self.auto_enroll = auto_enroll
        self.enroll_min_score = enroll_min_score
        self.embedder = SpeakerEmbedder(model_path)
        self.profile = VoiceProfile(profile_path)
        self._rejections = 0
        self._enrollments = 0

    # ---- состояние ----

    @property
    def active(self) -> bool:
        """Проверять ли прямо сейчас: включено, есть модель и профиль."""
        return bool(self.enabled and self.profile.exists and self.embedder.available)

    @property
    def ready(self) -> bool:
        return bool(self.profile.exists and self.embedder.available)

    def warmup(self) -> bool:
        return self.embedder.warmup()

    @property
    def stats(self) -> dict:
        return {"rejections": self._rejections, "enrollments": self._enrollments,
                "profile_phrases": self.profile.count, "threshold": self.threshold,
                "loaded": self.embedder.loaded, "active": self.active}

    # ---- проверка ----

    def check(self, pcm_int16: np.ndarray) -> tuple[bool, float | None, str]:
        """(пропускать_ли, косинус, причина).

        Возвращает ok=True, когда проверка невозможна: короткая фраза, нет
        профиля, модель недоступна. Лучше пропустить фразу, чем потерять
        диктовку пользователя из-за проблем с моделью.
        """
        if not self.enabled:
            return True, None, "выключено"
        if not self.profile.exists:
            return True, None, "нет профиля голоса"
        if not self.embedder.available:
            return True, None, f"модель недоступна: {self.embedder.load_error}"
        if pcm_int16 is None or len(pcm_int16) < int(self.min_seconds * SAMPLE_RATE):
            return True, None, "фраза короче порога проверки"
        embedding = self.embedder.embed(pcm_int16)
        if embedding is None:
            return True, None, "не удалось посчитать эмбеддинг"
        score = self.profile.score(embedding)
        if score >= self.threshold:
            return True, score, "голос владельца"
        self._rejections += 1
        return False, score, f"чужой голос (cosine={score:.3f} < {self.threshold})"

    def enroll(self, pcm_int16: np.ndarray, force: bool = False) -> tuple[bool, float | None]:
        """Добавить фразу в профиль. Без force — только уверенно «свои» фразы."""
        if pcm_int16 is None or len(pcm_int16) < int(self.min_seconds * SAMPLE_RATE):
            return False, None
        embedding = self.embedder.embed(pcm_int16)
        if embedding is None:
            return False, None
        score = self.profile.score(embedding) if self.profile.exists else None
        if not force and self.profile.exists:
            # Пополняем профиль только тем, что уверенно похоже на владельца,
            # иначе чужой голос, случайно прошедший порог, «размажет» профиль.
            if score is None or score < self.enroll_min_score:
                return False, score
        self.profile.add(embedding)
        self.profile.save()
        self._enrollments += 1
        return True, score

    def maybe_auto_enroll(self, pcm_int16: np.ndarray, accepted: bool,
                          score: float | None) -> bool:
        """Авто-пополнение: только принятые фразы, уверенно похожие на владельца."""
        if not (self.enabled and self.auto_enroll and accepted):
            return False
        if score is not None and score < self.enroll_min_score:
            return False
        ok, _ = self.enroll(pcm_int16, force=not self.profile.exists)
        return ok
