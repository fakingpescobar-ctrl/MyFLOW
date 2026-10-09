# bench_vad.py — сравнение VAD-кандидатов для MyFLOW (осень 2026)
# Участники: MyFLOW-энергетика, Silero v6 (ONNX), WebRTC,
#            каскад WebRTC+Silero (RealtimeSTT-стиль), TEN-VAD, FireRedVAD (offline).
# Запуск:  python -X utf8 bench_vad.py

from __future__ import annotations

import itertools
import math
import time
from collections import deque  # noqa: F401  (зарезервировано — не используется)
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "vad_models"
SR = 16000


# ---------------------------------------------------------------- синтетика

def _ramped(times: np.ndarray, rise: float, fall: float) -> np.ndarray:
    """Cos-огибающая: подъём за rise, спад за fall."""
    env = np.ones_like(times)
    n_rise = int(rise * SR)
    if n_rise > 0:
        env[:n_rise] = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, n_rise))
    n_fall = int(fall * SR)
    if n_fall > 0:
        env[-n_fall:] = 0.5 + 0.5 * np.cos(np.linspace(0, math.pi, n_fall))
    return env


def make_speech(dur_s: float, amp: float,
                base_freqs: tuple[float, ...] = (180, 320, 520, 780),
                am_hz: float = 5.0, pad_edges: float = 0.02) -> np.ndarray:
    """Гармоническая «речь»: формантные синусы + амплитудная модуляция (слоги)."""
    t = np.arange(int(dur_s * SR)) / SR
    wav = np.zeros_like(t)
    weights = (0.55, 0.30, 0.10, 0.05)
    for i, f in enumerate(base_freqs):
        w = weights[i] if i < len(weights) else 0.25
        wav += w * np.sin(2 * math.pi * f * t)
    wav *= 1.0 + 0.25 * np.sin(2 * math.pi * am_hz * t)
    wav *= _ramped(t, pad_edges, pad_edges)
    return wav / np.max(np.abs(wav)) * amp


def make_pink_noise(dur_s: float, amp: float, cutoff_hz: float = 250.0) -> np.ndarray:
    """Гудящий шум (вентилятор): белый шум через экспоненциальный lowpass."""
    n = int(dur_s * SR)
    white = np.random.default_rng(42).standard_normal(n)
    alpha = math.exp(-2.0 * math.pi * cutoff_hz / SR)
    filt = np.zeros(n)
    y = 0.0
    for i in range(n):
        y = alpha * y + (1.0 - alpha) * white[i]
        filt[i] = y
    return filt / np.max(np.abs(filt)) * amp


def make_tv_noise(dur_s: float, amp: float) -> np.ndarray:
    """«Телевизор»: прерывистые аккорды с огибающими, амплитудой выше порога MyFLOW."""
    t = np.arange(int(dur_s * SR)) / SR
    wav = np.zeros_like(t)
    rng = np.random.default_rng(7)
    pos = 0.0
    while pos < dur_s:
        seg_len = min(rng.uniform(0.08, 0.35), dur_s - pos)
        i0, i1 = int(pos * SR), int((pos + seg_len) * SR)
        freqs = tuple(rng.choice((220, 277, 330, 370, 440, 554, 660)) for _ in range(3))
        tt = t[i0:i1] - pos
        seg = sum(np.sin(2 * math.pi * f * tt) for f in freqs) / 3.0
        env = _ramped(tt, 0.015, 0.02)[: len(seg)]
        if len(seg) > 0:
            wav[i0:i1] += seg * env
        pos = seg_end = pos + seg_len + rng.uniform(0.05, 0.5)
    return wav / np.max(np.abs(wav)) * amp


def make_clicks(dur_s: float, amp: float) -> np.ndarray:
    """Короткие клики (щелчки/клавиши): затухающие синусы 15мс."""
    wav = np.zeros(int(dur_s * SR))
    rng = np.random.default_rng(3)
    pos = 0.05
    while pos < dur_s - 0.05:
        i0, i1 = int(pos * SR), min(int((pos + 0.015) * SR), len(wav))
        tt = np.arange(i1 - i0) / SR
        wav[i0:i1] += amp * np.sin(2 * math.pi * 3000 * tt) * np.exp(-tt / 0.004)
        pos += rng.uniform(0.15, 0.5)
    return wav


def make_pops(dur_s: float, amp: float = 0.5) -> np.ndarray:
    """Реалистичные попы микрофона: НЧ-импульсы (50-90Гц) 30-80мс с затуханием —
    в отличие от кликов не широкополосные, а низкочастотные с треском оболочки."""
    rng = np.random.default_rng(42)
    out = np.zeros(int(dur_s * SR))
    pos = 0.0
    while pos < dur_s:
        p_len = rng.uniform(0.03, 0.08)
        n = int(p_len * SR)
        i0 = int(pos * SR)
        i1 = min(i0 + n, len(out))
        if i1 <= i0:
            break
        tt = np.arange(i1 - i0) / SR
        f = rng.uniform(50, 90)
        env = np.exp(-tt / p_len * 3.5)
        pop = np.sin(2 * math.pi * f * tt) * env * amp
        pop += 0.15 * rng.standard_normal(len(tt)) * env  # треск оболочки
        out[i0:i1] += pop
        pos += p_len + rng.uniform(0.25, 0.7)
    return out


def synth_cases() -> list[dict]:
    """Сценарии бенча с ground-truth интервалами речи.
    Речь — настоящий голос jfk.wav; шум/телек/клики — синтетика.
    Важно: нейро-VAD (Silero/FireRed) обучены на реальном голосе — синтезированный
    тон они «не узнают» (это фича: отсев не-речи), поэтому речь берём из jfk."""
    jfk = _load_jfk()
    # реальные речевые окна jfk (по RMS): фраза1 0.3..1.5с, фраза2 3.3..3.63с; 2.0..3.2с — пауза
    phrase_a = jfk[int(0.3 * SR):int(1.5 * SR)]
    phrase_b = jfk[int(3.3 * SR):int(3.63 * SR)]
    t2 = np.concatenate([phrase_a, np.zeros(int(0.9 * SR)), phrase_b])
    t6_chunk = jfk[int(0.3 * SR):int(1.55 * SR)].copy()
    ramp = np.linspace(0.004, 1.0, int(0.25 * SR))
    t6_chunk[:len(ramp)] *= ramp
    t6 = np.concatenate([t6_chunk, np.zeros(int(0.5 * SR))])
    t7 = jfk[int(0.3 * SR):int(1.7 * SR)] + make_pink_noise(1.4, 0.06)
    # T8 cold start: короткая фраза 0.3с (близко к min_speech 0.25с), тишина вокруг
    t8 = np.concatenate([np.zeros(int(0.5 * SR)),
                         jfk[int(3.3 * SR):int(3.6 * SR)],
                         np.zeros(int(0.5 * SR))])
    # T9 barge-in: речь поверх говорящего ТВ
    t9 = jfk[int(0.5 * SR):int(1.7 * SR)] + make_tv_noise(1.2, 0.06)
    # T10 попы: низкочастотные импульсы (реальные попы микрофона — не клики)
    t10 = make_pops(3.0)
    # T11 эхо/спикерфон: речь + задержанная 200мс копия ×0.4 (динамик), эхо-хвост — не речь
    speech = jfk[int(0.5 * SR):int(1.7 * SR)]
    echo = np.zeros(len(speech) + int(0.2 * SR))
    echo[:len(speech)] += speech
    echo[int(0.2 * SR):] += 0.4 * speech
    # T12 долгая речь: 2 длинные фразы (2.25с каждая) + пауза 1.0с — проверка удержания и склейки
    long_a = jfk[int(5.25 * SR):int(7.5 * SR)]
    long_b = jfk[int(8.0 * SR):int(10.25 * SR)]
    t12 = np.concatenate([long_a, np.zeros(int(1.0 * SR)), long_b])
    return [
        {"name": "T1 тишина",        "wav": np.zeros(3 * SR),            "gt": []},
        {"name": "T2 речь-пауза",    "wav": t2,                           "gt": [(0.0, 1.2), (2.1, 2.43)]},
        {"name": "T3 клики",         "wav": make_clicks(3.0, 0.5),        "gt": []},
        {"name": "T4 вентилятор",    "wav": make_pink_noise(3.0, 0.06),   "gt": []},
        {"name": "T5 телевизор",     "wav": make_tv_noise(3.0, 0.06),     "gt": []},
        {"name": "T6 слабый старт",  "wav": t6,                           "gt": [(0.0, 1.25)]},
        {"name": "T7 речь в шуме",   "wav": t7,                           "gt": [(0.0, 1.4)]},
        {"name": "T8 cold start",    "wav": t8,                           "gt": [(0.5, 0.8)]},
        {"name": "T9 barge-in",      "wav": t9,                           "gt": [(0.0, 1.2)]},
        {"name": "T10 попы",         "wav": t10,                          "gt": []},
        {"name": "T11 эхо",          "wav": echo,                         "gt": [(0.0, 1.2)]},
        {"name": "T12 долгая речь",  "wav": t12,                          "gt": [(0.0, 2.25), (3.25, 5.5)]},
    ]


def _load_jfk() -> np.ndarray:
    """Реальный голос (jfk.wav из whisper.cpp) как float [-1,1]."""
    path = MODELS_DIR / "jfk.wav"
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav", path)
    import wave
    with wave.open(str(path)) as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------- адаптеры

class BaseVad:
    name = "base"

    def compute(self, wav: np.ndarray) -> np.ndarray:
        """per-10ms вероятности речи (0..1). wav — float в PCM16-шкале."""
        raise NotImplementedError


class EnergyVad(BaseVad):
    """Текущий VAD MyFLOW: пиковый порог амплитуды, чанк 40мс."""
    name = "MyFLOW energy"

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def compute(self, wav: np.ndarray) -> np.ndarray:
        frame_samples = SR // 100
        n_frames = int(np.ceil(len(wav) / frame_samples))
        loudness = np.array([np.max(np.abs(wav[i * frame_samples:(i + 1) * frame_samples]))
                             for i in range(n_frames)])
        chunk = 4
        probs = np.zeros(n_frames)
        for start in range(0, n_frames - chunk + 1, chunk):
            is_loud = max(loudness[start:start + chunk]) >= self.threshold
            probs[start:start + chunk] = 1.0 if is_loud else 0.0
        return probs


class SileroVad(BaseVad):
    name = "Silero v6"

    def __init__(self, onnx_path: Path, context_samples: int = 64):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.hidden = np.zeros((2, 1, 128), dtype=np.float32)
        self.chunk = 512
        self.context = context_samples

    def compute(self, wav: np.ndarray) -> np.ndarray:
        self.hidden = np.zeros((2, 1, 128), dtype=np.float32)
        tail = np.zeros(self.context, dtype=np.float32)
        chunk_probs = []
        for start in range(0, len(wav), self.chunk):
            piece = wav[start:start + self.chunk]  # float [-1,1]
            if len(piece) < self.chunk:
                piece = np.pad(piece, (0, self.chunk - len(piece)))
            feed_input = np.concatenate([tail, piece])  # контекст 64 как в faster-whisper
            tail = piece[-self.context:]
            out, self.hidden = self.session.run(
                ["output", "stateN"],
                {"input": feed_input[None].astype(np.float32),
                 "state": self.hidden,
                 "sr": np.array(SR, dtype=np.int64)})
            chunk_probs.append(float(out[0][0]))
        n_frames = int(np.ceil(len(wav) / (SR // 100)))
        if len(chunk_probs) == 0:
            return np.zeros(n_frames)
        # растяжка equally-spaced: чанки 512 (32мс) -> сетка 10мс
        src = np.linspace(0, len(chunk_probs) - 1, n_frames)
        return np.interp(src, np.arange(len(chunk_probs)), chunk_probs)


class WebRtcVad(BaseVad):
    name = "WebRTC"

    def __init__(self, aggressiveness: int = 3):
        import webrtcvad
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(aggressiveness)

    def compute(self, wav: np.ndarray) -> np.ndarray:
        pcm16 = np.clip(np.round(wav * 32768.0), -32768, 32767).astype(np.int16)
        frame_len = SR // 50  # 20мс
        pcm16 = np.pad(pcm16, (0, frame_len - len(pcm16) % frame_len))
        verdicts = []
        for i in range(0, len(pcm16), frame_len):
            frame = pcm16[i:i + frame_len].tobytes()
            verdicts.append(1.0 if self.vad.is_speech(frame, SR) else 0.0)
        frames = np.repeat(np.array(verdicts), 2)
        n_frames = int(np.ceil(len(wav) / (SR // 100)))
        return frames[:n_frames]


class CascadeVad(BaseVad):
    """RealtimeSTT-стиль: WebRTC-претриггер И Silero-подтверждение на 10мс-фреймах."""
    name = "WebRTC+Silero cascade"

    def __init__(self, silero: SileroVad, threshold: float = 0.4):
        import webrtcvad
        self.webrtc = webrtcvad.Vad()
        self.webrtc.set_mode(3)
        self.silero = silero
        self.threshold = threshold

    def compute(self, wav: np.ndarray) -> np.ndarray:
        silero_probs = self.silero.compute(wav)
        n_frames = int(np.ceil(len(wav) / (SR // 100)))
        pcm16 = np.clip(np.round(wav * 32768.0), -32768, 32767).astype(np.int16)
        frame_len = SR // 100  # 10мс
        pcm16 = np.pad(pcm16, (0, n_frames * frame_len - len(pcm16)))
        agreement = np.zeros(n_frames)
        for i in range(n_frames):
            frame = pcm16[i * frame_len:(i + 1) * frame_len].tobytes()
            webrtc_says_speech = self.webrtc.is_speech(frame, SR)
            agreement[i] = 1.0 if (webrtc_says_speech and silero_probs[i] >= self.threshold) else 0.0
        return agreement


class TenVadAdapter(BaseVad):
    name = "TEN-VAD"

    def __init__(self, threshold: float = 0.5):
        import ten_vad
        self.vad = ten_vad.TenVad(hop_size=256, threshold=threshold)
        self.hop = 256

    def compute(self, wav: np.ndarray) -> np.ndarray:
        pcm16 = np.clip(np.round(wav * 32768.0), -32768, 32767).astype(np.int16)
        pcm16 = np.pad(pcm16, (0, self.hop - len(pcm16) % self.hop))
        verdicts = []
        for start in range(0, len(pcm16), self.hop):
            prob, _ = self.vad.process(pcm16[start:start + self.hop])
            verdicts.append(float(prob))
        frames = np.repeat(np.array(verdicts), 2)
        n_frames = int(np.ceil(len(wav) / (SR // 100)))
        return frames[:n_frames]


class FireRedVad(BaseVad):
    """FireRedVAD offline: kaldi-fbank(80, 25мс/10мс) + CMVN + onnx."""

    name = "FireRedVAD"

    def __init__(self, onnx_path: Path, cmvn_path: Path, threshold: float = 0.4):
        import kaldi_native_fbank as knf
        import kaldiio
        self.knf = knf
        self.threshold = threshold
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SR
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        self.opts = opts
        stats = kaldiio.load_mat(str(cmvn_path))
        count = stats[0, -1]
        dim = stats.shape[-1] - 1
        means = np.array([stats[0, d] / count for d in range(dim)])
        variance = np.array([max(stats[1, d] / count - (stats[0, d] / count) ** 2, 1e-20)
                             for d in range(dim)])
        self.means = means
        self.istd = 1.0 / np.sqrt(variance)

    def _feats(self, wav: np.ndarray) -> np.ndarray:
        pcm16 = np.clip(np.round(wav * 32768.0), -32768, 32767).astype(np.int16)
        fbank = self.knf.OnlineFbank(self.opts)
        fbank.accept_waveform(SR, pcm16.tolist())
        n = fbank.num_frames_ready
        if n == 0:
            return np.zeros((0, 80), dtype=np.float32)
        feat = np.vstack([fbank.get_frame(i) for i in range(n)])
        return (feat - self.means) * self.istd

    def compute(self, wav: np.ndarray) -> np.ndarray:
        feat = self._feats(wav)[None].astype(np.float32)
        if feat.shape[1] == 0:
            return np.zeros(0)
        probs = self.session.run(["probs"], {"feat": feat})[0][0, :, 0]
        return probs


class FireRedStreamVad(FireRedVad):
    """FireRedVAD со streaming CMVN (running EMA вместо глобального) — честная
    имитация стрима: статистики накапливаются с нуля, alpha=0.01."""

    name = "FireRedVAD stream"

    def __init__(self, onnx_path: Path, cmvn_path: Path, threshold: float = 0.4, alpha: float = 0.01):
        super().__init__(onnx_path, cmvn_path, threshold)
        self.alpha = alpha
        self.reset_stats()

    def reset_stats(self):
        dim = self.means.shape[0]
        self.r_mean = np.zeros(dim, dtype=np.float32)
        self.r_var = np.ones(dim, dtype=np.float32) * 1e-6

    def _feats(self, wav: np.ndarray) -> np.ndarray:
        pcm16 = np.clip(np.round(wav * 32768.0), -32768, 32767).astype(np.int16)
        fbank = self.knf.OnlineFbank(self.opts)
        fbank.accept_waveform(SR, pcm16.tolist())
        n = fbank.num_frames_ready
        if n == 0:
            return np.zeros((0, 80), dtype=np.float32)
        feat = np.vstack([fbank.get_frame(i) for i in range(n)])
        a = self.alpha
        out = np.empty_like(feat)
        for i in range(n):
            frame = feat[i]
            out[i] = (frame - self.r_mean) / np.sqrt(self.r_var + 1e-8)
            delta = frame - self.r_mean
            self.r_mean = (1 - a) * self.r_mean + a * frame
            self.r_var = (1 - a) * self.r_var + a * delta * delta
        return out

    def compute(self, wav: np.ndarray) -> np.ndarray:
        self.reset_stats()
        feat = self._feats(wav)[None].astype(np.float32)
        if feat.shape[1] == 0:
            return np.zeros(0)
        probs = self.session.run(["probs"], {"feat": feat})[0][0, :, 0]
        return probs


# ---------------------------------------------------------------- сегментация

def smooth_median(values: np.ndarray, window: int = 5) -> np.ndarray:
    out = values.copy()
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


def segment_frames(probs: np.ndarray, threshold: float,
                   min_speech_frames: int = 25,
                   min_silence_frames: int = 40,
                   pad_frames: int = 3) -> list[tuple[float, float]]:
    """per-10ms вероятности -> интервалы [(start_s, end_s)]."""
    if len(probs) == 0:
        return []
    is_speech = smooth_median(probs) >= threshold
    segments: list[tuple[int, int]] = []
    start: int | None = None
    silence_run = 0
    for i, speak in enumerate(is_speech):
        if speak:
            if start is None:
                start = i
            silence_run = 0
        elif start is not None:
            silence_run += 1
            if silence_run >= min_silence_frames:
                segments.append((start, i - silence_run + 1))
                start = None
                silence_run = 0
    if start is not None:
        segments.append((start, len(is_speech)))
    result = []
    for seg_start, seg_end in segments:
        if seg_end - seg_start < min_speech_frames:
            continue
        result.append((max(0, seg_start - pad_frames) / 100.0,
                       (seg_end + pad_frames) / 100.0))
    return result


# ---------------------------------------------------------------- метрики

def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _union_time(segments: list[tuple[float, float]], gt: list[tuple[float, float]]) -> float:
    points = sorted(set(itertools.chain.from_iterable([list(s) for s in segments]
                                                      + [list(g) for g in gt])))
    total = 0.0
    for a, b in zip(points, points[1:]):
        in_gt = any(a >= g[0] and b <= g[1] for g in gt)
        in_seg = any(a >= s[0] and b <= s[1] for s in segments)
        if in_gt or in_seg:
            total += b - a
    return total


def match_score(segments: list[tuple[float, float]], gt: list[tuple[float, float]]) -> dict:
    """Покрытие GT, ложное аудио и IoU по времени."""
    covered = sum(_overlap(s, g) for s in segments for g in gt)
    gt_len = sum(g[1] - g[0] for g in gt)
    seg_len = sum(s[1] - s[0] for s in segments)
    union = _union_time(segments, gt)
    return {
        "n_seg": len(segments),
        "coverage": covered / gt_len if gt_len else 0.0,
        "false_audio": max(seg_len - covered, 0.0),
        "iou": covered / union if union else 0.0,
    }


# ---------------------------------------------------------------- запуск

def main() -> None:
    cases = synth_cases()
    silero = SileroVad(MODELS_DIR / "silero_v6.onnx")
    fire = FireRedVad(MODELS_DIR / "fireredvad_vad.onnx", MODELS_DIR / "cmvn.ark")
    fire_s = FireRedStreamVad(MODELS_DIR / "fireredvad_vad.onnx", MODELS_DIR / "cmvn.ark")
    adapters: list[tuple[BaseVad, float, int]] = [
        (EnergyVad(), 0.01, 25),
        (silero, 0.30, 15),          # sweep: thr 0.30 + ms 15 ловит cold start 0.3с (T8 IoU 0.69)
        (WebRtcVad(3), 0.5, 25),
        (CascadeVad(silero), 0.5, 25),
        (TenVadAdapter(), 0.5, 25),
        (fire, 0.4, 25),
        (fire_s, 0.4, 25),
    ]
    seg_kw = dict(min_silence_frames=40, pad_frames=3)

    header = f"{'VAD':<24}" + "".join(f"{c['name'][:14]:>16}" for c in cases)
    print(header)
    print("-" * len(header))
    for adapter, threshold, ms in adapters:
        cells = []
        for case in cases:
            t0 = time.perf_counter()
            probs = adapter.compute(case["wav"])
            elapsed = time.perf_counter() - t0
            segs = segment_frames(probs, threshold, min_speech_frames=ms, **seg_kw)
            score = match_score(segs, case["gt"])
            if case["gt"]:
                cells.append(f"{score['n_seg']} IoU{score['iou']:.2f}")
            else:
                cells.append(f"{score['n_seg']} fa{score['false_audio']:.1f}s")
            adapter._last_elapsed = elapsed  # noqa: SLF001
        print(f"{adapter.name:<24}" + "".join(f"{v:>16}" for v in cells))
    print()

    # диагностика сегментов: pred vs GT (T2, T8, T12)
    for name in ("T2 речь-пауза", "T8 cold start", "T12 долгая речь"):
        case = next(c for c in cases if c["name"] == name)
        print(f"--- segs: {name}  GT={case['gt']} ---")
        for adapter, threshold, ms in adapters:
            probs = adapter.compute(case["wav"])
            segs = segment_frames(probs, threshold, min_speech_frames=ms, **seg_kw)
            fmt = ", ".join(f"{a:.2f}-{b:.2f}" for a, b in segs[:6]) or "—"
            print(f"  {adapter.name:<24} {fmt}")
        print()

    print("--- RTF (процессинг / длительность) на T2 (2.9s) ---")
    t2 = cases[1]["wav"]
    dur = len(t2) / SR
    for adapter, threshold, ms in adapters:
        t0 = time.perf_counter()
        adapter.compute(t2)
        elapsed = time.perf_counter() - t0
        print(f"{adapter.name:<24} RTF={elapsed / dur:.4f}  {elapsed * 1000:.1f} ms")


if __name__ == "__main__":
    main()