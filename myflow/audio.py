"""MyFLOW — запись с микрофона (аналог recorderWorklet.js у Wispr Flow).

Два режима:
  PTT  — start()/stop() по хоткею; опциональный VAD-автостоп по тишине
         (пауза длиннее silence_pause_seconds завершает запись).
  AUTO — start_continuous(): микрофон слушает постоянно, фраза начинается
         по голосовому триггеру (несколько «громких» чанков подряд) с
         pre-roll ~0.25с и заканчивается по тишине; результат отдаётся
         через on_utterance(Recording) из отдельного потока.

Конвертация Float32 -> Int16 с клампом (копия схемы Wispr: Gm()).
Никакие колбэки не блокируют audio-callback (threading daemon).
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd

from myflow.config import SAMPLE_RATE, CHUNK_SAMPLES, MAX_DICTATION_SECONDS
from myflow.vad import SileroStreamVad


@dataclass
class Recording:
    """Готовый буфер PCM Int16 16k mono. parse at boundary: только валидное."""
    pcm_int16: np.ndarray
    duration_ms: int
    max_amplitude: float


def _float_to_int16(samples: np.ndarray) -> np.ndarray:
    """Кламп [-1,1] и масштаб в Int16 (одно из мест «ошибок не бывает»)."""
    clamped = np.clip(samples, -1.0, 1.0)
    scaled = np.where(clamped < 0, clamped * 32768.0, clamped * 32767.0)
    return scaled.astype(np.int16)


class DictationRecorder:
    """Запись 16k mono: PTT + AUTO (постоянное слушание) с VAD.

    VAD-параметры:
      silence_pause_seconds — сколько секунд тишины подряд = конец фразы/записи
      silence_amplitude    — пиковая амплитуда чанка, ниже которой он «тихий»
      min_auto_stop_ms     — мин. длительность PTT-записи перед VAD-автостопом
      min_utterance_ms     — мин. длительность фразы в AUTO (отсев кликов/шума)
    on_auto_stop (PTT) / on_utterance (AUTO) вызываются из отдельных потоков.
    """

    # AUTO-режим: pre-roll и минимальная длительность речи.
    _PREROLL_SECONDS = 0.25
    _MIN_UTTERANCE_MS = 300
    _MIN_SPEECH_MS = 180              # мин. длительность РЕЧИ в фразе (клики отсечь)

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_samples: int = CHUNK_SAMPLES,
                 device: int | None = None,
                 silence_pause_seconds: float = 0.9,
                 silence_amplitude: float = 0.01,
                 min_auto_stop_ms: int = 500,
                 min_utterance_ms: int = _MIN_UTTERANCE_MS,
                 vad_threshold: float = 0.30,
                 vad: SileroStreamVad | None = None,
                 adaptive_noise_gate: bool = True,
                 noise_ratio: float = 2.5,
                 noise_max_gate: float = 0.04,
                 noise_min_floor: float = 0.012,
                 on_auto_stop: Callable[[], None] | None = None,
                 on_utterance: Callable[[Recording], None] | None = None,
                 on_interval: Callable[[np.ndarray], None] | None = None,
                 interval_seconds: float = 1.8,
                 interval_window_seconds: float = 4.0) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self._device = device          # None = устройство по умолчанию
        self._stream: sd.InputStream | None = None
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._total_samples = 0
        self._max_amp = 0.0
        self._continuous = False
        # VAD (голосовая активность) — Silero вместо старой энергетики.
        self.vad = vad or SileroStreamVad(threshold=vad_threshold)
        self.vad_threshold = self.vad.threshold
        self.silence_pause_seconds = silence_pause_seconds
        self.silence_amplitude = silence_amplitude
        # Адаптивный порог по фоновому шуму: если в комнате играет телевизор
        # (пик 0.026 при голосе 0.317), фиксированный порог 0.01 пропускает
        # «фразу» из фоновой речи. Оцениваем уровень не-речевых чанков и требуем,
        # чтобы пик фразы был заметно выше. Потолок (0.04) защищает тихую речь:
        # замеры на своих записях — реальная речь 0.13-1.06, фон 0.017-0.040.
        self.adaptive_noise_gate = adaptive_noise_gate
        self.noise_ratio = noise_ratio
        self.noise_max_gate = noise_max_gate
        self.noise_min_floor = noise_min_floor
        self._noise_floor = 0.0        # пик не-речевых чанков (медленно затухает)
        self._noise_chunks = 0         # сколько не-речевых чанков видели
        self._noise_rejects = 0        # сколько фраз отсек адаптивный порог
        self.on_auto_stop = on_auto_stop
        self.on_utterance = on_utterance
        # Live-превью: каждые interval_seconds отдаём хвост записи
        # (последние interval_window_seconds) — для инкрементального ASR.
        self.on_interval = on_interval
        self._interval_chunks = max(1, int(interval_seconds * sample_rate / chunk_samples))
        self._interval_window_samples = int(interval_window_seconds * sample_rate)
        self._prev_interval_samples = 0
        self._silence_chunks = max(1, int(silence_pause_seconds * sample_rate / chunk_samples))
        self._chunk_ms = chunk_samples * 1000 / sample_rate
        self._silence_ms = int(silence_pause_seconds * 1000)
        self._min_auto_stop_samples = int(min_auto_stop_ms * sample_rate / 1000)
        self._min_utterance_ms = min_utterance_ms
        self._min_speech_ms = self._MIN_SPEECH_MS
        self._silent_run = 0
        self._auto_stop_fired = False
        # AUTO-состояние
        self._state = "idle"           # idle | speech
        self._speech_run = 0
        self._preroll_samples = int(self._PREROLL_SECONDS * sample_rate)
        self._ring: list[np.ndarray] = []
        self._phrase: list[np.ndarray] = []
        self._phrase_peak = 0.0
        self._phrase_speech_ok_chunks = 0
        # Счётчик превью внутри фразы: инициализируем здесь, а не только в
        # start_continuous (иначе атрибут отсутствует до первого запуска AUTO).
        self._prev_interval_samples_in_phrase = 0
        self._max_samples = MAX_DICTATION_SECONDS * sample_rate
        self._hard_limit_fired = False
        self._status_count = 0
        self._dropped_chunks = 0
        self._utter_ready = False
        # Тяжёлая работа (ONNX-инференс VAD + конверсия в Int16) вынесена из
        # PortAudio-callback в отдельный поток: в колбэке только копия буфера.
        # Раньше инференс шёл в RT-потоке и рядом с whisper/llama давал underflow.
        self._work_q: queue.Queue = queue.Queue(maxsize=512)
        self._worker = threading.Thread(target=self._work_loop, daemon=True,
                                        name="audio-worker")
        self._worker.start()

    # ---------- общий открытие/закрытие стрима ----------

    def _open_stream(self) -> None:
        device = self._device
        if device is not None and not self._device_ok(device):
            # Устройство из настроек отвалилось (наушники выключили?) —
            # не молчим, а падаем на системный дефолт, иначе цикл «мёртвый».
            print(f"[audio] микрофон #{device} недоступен — беру системный по умолчанию")
            device = None
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_samples,   # 640 -> callback тянет 40 мс
                callback=self._on_audio,
                device=device,
            )
            self._stream.start()
        except Exception as exc:
            raise RuntimeError(f"не удалось открыть микрофон: {exc}") from exc

    @staticmethod
    def _device_ok(device: int | None) -> bool:
        """Открывается ли поток на устройстве (дёшево, без записи)."""
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                device=device,
            ):
                pass
            return True
        except Exception:  # noqa: BLE001 — любой сбой = устройство мертво
            return False

    # ---------- PTT-режим ----------

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("запись уже идёт")  # Fail loud
        self._reset_ptt()
        self._continuous = False
        self._open_stream()

    def _reset_ptt(self) -> None:
        self._buffer = []
        self._total_samples = 0
        self._max_amp = 0.0
        self._silent_run = 0
        self._auto_stop_fired = False
        self._hard_limit_fired = False
        self._prev_interval_samples = 0
        # Фон пересчитываем заново: устройство/устройство-микшер могло смениться.
        self._noise_floor = 0.0
        self._noise_chunks = 0
        self.vad.reset()

    def _on_audio_ptt(self, indata: np.ndarray, frames: int) -> None:
        mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        is_speech = self.vad.process(mono) >= self.vad_threshold
        trigger = False
        limit_hit = False
        interval = False
        with self._lock:
            self._total_samples += frames
            if self._hard_limit_fired:
                # Лимит диктовки уже достигнут: время считаем, аудио не копим
                # (раньше при громком фоне буфер рос без границы — 32 КБ/с).
                pass
            else:
                self._buffer.append(_float_to_int16(mono))
                self._max_amp = max(self._max_amp, peak)
                if self._total_samples >= self._max_samples:
                    self._hard_limit_fired = True
                    limit_hit = True
            # Live-превью: каждый интервал отдаём хвост записи для драфт-ASR.
            # Не на самом старте (мы — миним. период как у авто-стопа) и только
            # если уже была речь — иначе превью будет крутиться на тишине.
            if (not self._hard_limit_fired
                    and self.on_interval is not None
                    and self._max_amp >= self.silence_amplitude
                    and self._total_samples >= self._min_auto_stop_samples
                    and self._total_samples - self._prev_interval_samples
                    >= self._interval_chunks * self.chunk_samples):
                self._prev_interval_samples = self._total_samples
                interval = True
            # VAD (Silero): считаем подряд идущие не-речевые чанки; пауза завершает
            # запись. Авто-стоп срабатывает только если в записи БЫЛА речь (иначе
            # F8-пресс без слов рубил бы запись до начала разговора).
            if is_speech:
                self._silent_run = 0
            else:
                self._silent_run += 1
            if (not self._hard_limit_fired
                    and not self._auto_stop_fired
                    and self._max_amp >= self.silence_amplitude
                    and self._silent_run >= self._silence_chunks
                    and self._total_samples >= self._min_auto_stop_samples):
                self._auto_stop_fired = True
                trigger = True
        if interval:
            self._emit_interval()
        if limit_hit:
            print(f"[audio] лимит диктовки {MAX_DICTATION_SECONDS}с — завершаю запись")
            if self.on_auto_stop:
                threading.Thread(target=self.on_auto_stop, name="hard-limit",
                                 daemon=True).start()
        if trigger and self.on_auto_stop:
            threading.Thread(target=self.on_auto_stop, name="auto-stop",
                             daemon=True).start()

    @property
    def recording_duration_ms(self) -> int:
        with self._lock:
            return int(self._total_samples * 1000 / self.sample_rate)

    def stop(self) -> Recording:
        stream, self._stream = self._stream, None
        if stream is None:
            raise RuntimeError("нет активной записи")
        stream.stop()
        stream.close()
        self._drain_work_queue()
        with self._lock:
            pcm = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.int16)
            duration_ms = int(len(pcm) * 1000 / self.sample_rate)
            max_amp = self._max_amp
        return Recording(pcm_int16=pcm, duration_ms=duration_ms, max_amplitude=max_amp)

    def _drain_work_queue(self, timeout: float = 0.5) -> None:
        """Дождаться, пока воркер разберёт уже принятые чанки.

        Иначе хвост фразы (последние 40-80 мс) остаётся в очереди и не
        попадает в запись: колбэк теперь только кладёт чанк.
        join() ждём в отдельном потоке — у Queue нет таймаута на join.
        """
        done = threading.Event()

        def _joiner() -> None:
            try:
                self._work_q.join()
            except Exception:  # noqa: BLE001
                pass
            finally:
                done.set()

        threading.Thread(target=_joiner, daemon=True, name="audio-drain").start()
        done.wait(timeout)

    def too_long(self) -> bool:
        """Лимит диктовки (Wispr: DICTATION_DURATION_TIMEOUT=19мин, мы меньше)."""
        return self.recording_duration_ms > MAX_DICTATION_SECONDS * 1000

    # ---------- AUTO-режим (постоянное слушание) ----------

    def start_continuous(self, on_utterance: Callable[[Recording], None] | None = None) -> None:
        if self._stream is not None:
            raise RuntimeError("прослушивание уже идёт")
        if on_utterance is not None:
            self.on_utterance = on_utterance
        self._continuous = True
        self._state = "idle"
        self._speech_run = 0
        self._silent_run = 0
        self._ring = []
        self._phrase = []
        self._phrase_peak = 0.0
        self._phrase_speech_ok_chunks = 0
        self._prev_interval_samples_in_phrase = 0
        self._utter_ready = False
        self.vad.reset()
        self._open_stream()

    def stop_continuous(self) -> None:
        stream, self._stream = self._stream, None
        self._continuous = False
        if stream is not None:
            stream.stop()
            stream.close()
        self._drain_work_queue()

    def force_finish(self) -> bool:
        """AUTO: ручной финиш текущей фразы (кнопка). True — фраза была."""
        trigger = False
        with self._lock:
            if self._state == "speech":
                self._state = "idle"
                self._speech_run = 0
                self._utter_ready = True
                trigger = True
        if trigger:
            self._emit_utterance()
        return trigger

    def discard(self) -> None:
        """AUTO: скип текущей фразы без распознавания (отмена)."""
        with self._lock:
            self._state = "idle"
            self._speech_run = 0
            self._silent_run = 0
            self._phrase = []
            self._phrase_peak = 0.0
            self._phrase_speech_ok_chunks = 0
            self._prev_interval_samples_in_phrase = 0
            self._utter_ready = False

    def _track_noise_floor(self, peak: float, is_speech: bool) -> None:
        """Оценить уровень фона по не-речевым чанкам.

        Берём ПИК чанка (а не RMS): фоновая речь из телевизора в тихих местах
        даёт RMS заметно ниже пика, и порог по RMS не отсекал бы её.
        Затухание медленное (0.5 % на чанк ≈ 4 с), чтобы одиночный стук не
        держал порог высоко; сверху всё ограничено noise_max_gate.
        """
        if not self.adaptive_noise_gate or is_speech:
            return
        self._noise_chunks += 1
        self._noise_floor = max(peak, self._noise_floor * 0.995)

    def noise_gate(self) -> float:
        """Минимальный пик фразы: фиксированный порог или адаптивный по фону."""
        required = self.silence_amplitude
        if not self.adaptive_noise_gate:
            return required
        # Пока не набрали статистику по фону (~0.6 с не-речи) — не рискуем.
        if self._noise_chunks < 15 or self._noise_floor < self.noise_min_floor:
            return required
        return min(max(required, self._noise_floor * self.noise_ratio),
                   self.noise_max_gate)

    def _on_audio_continuous(self, indata: np.ndarray) -> None:
        mono = _float_to_int16(indata[:, 0])          # храним Int16 (как раньше)
        peak = float(np.max(np.abs(indata[:, 0]))) if len(indata) else 0.0
        prob = self.vad.process(np.ascontiguousarray(indata[:, 0], dtype=np.float32))
        is_speech = prob >= self.vad_threshold
        self._track_noise_floor(peak, is_speech)
        emit = False
        interval = False
        with self._lock:
            # Кольцевой pre-roll: держим последние ~0.25с для захвата начала слова.
            self._ring.append(mono)
            ring_len = sum(len(x) for x in self._ring)
            while ring_len > self._preroll_samples:
                ring_len -= len(self._ring[0])
                self._ring.pop(0)

            if self._state == "idle":
                if is_speech:
                    self._speech_run += 1
                    if self._speech_run * self._chunk_ms >= self._min_speech_ms:
                        # Речь началась: фраза = pre-roll + дальше всё.
                        self._state = "speech"
                        self._phrase = list(self._ring)
                        self._phrase_peak = peak
                        self._silent_run = 0
                        # Считаем речь и в pre-roll (начало слова могло попасть туда)
                        self._phrase_speech_ok_chunks = self._speech_run
                else:
                    self._speech_run = 0
            else:  # speech
                self._phrase.append(mono)
                self._phrase_peak = max(self._phrase_peak, peak)
                if is_speech:
                    self._phrase_speech_ok_chunks += 1
                    self._silent_run = 0
                else:
                    self._silent_run += 1
                # Live-превью длинной фразы (раз в интервал)
                if (self.on_interval is not None
                        and len(self._phrase) * self.chunk_samples
                        - self._prev_interval_samples_in_phrase
                        >= self._interval_chunks * self.chunk_samples):
                    self._prev_interval_samples_in_phrase = \
                        len(self._phrase) * self.chunk_samples
                    interval = True
                if self._silent_run * self._chunk_ms >= self._silence_ms:
                    self._state = "idle"
                    self._speech_run = 0
                    self._utter_ready = True
                    emit = True
                # Жёсткий лимит длины фразы — принудительная отдача.
                if (len(self._phrase) * self.chunk_samples
                        >= self._max_samples + self._preroll_samples):
                    self._state = "idle"
                    self._speech_run = 0
                    self._utter_ready = True
                    emit = True
        if interval:
            self._emit_interval(source="phrase")
        if emit:
            self._emit_utterance()

    @staticmethod
    def _tail_samples(chunks: list[np.ndarray], n_samples: int) -> np.ndarray:
        """Хвост записи: последние n_samples сэмплов из списка чанков."""
        out: list[np.ndarray] = []
        got = 0
        for c in reversed(chunks):
            out.append(c)
            got += len(c)
            if got >= n_samples:
                break
        if not out:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(out[::-1]) if len(out) > 1 else out[0]

    def _emit_interval(self, source: str = "buffer") -> None:
        """Live-превью: хвост записи/фразы для драфт-распознавания."""
        with self._lock:
            src = self._buffer if source == "buffer" else self._phrase
            pcm = self._tail_samples(src, self._interval_window_samples)
        if len(pcm) == 0 or self.on_interval is None:
            return
        threading.Thread(target=self.on_interval, args=(pcm,), name="live-preview",
                         daemon=True).start()

    def _emit_utterance(self) -> None:
        """Собирает готовую фразу и отдаёт on_utterance из отдельного потока."""
        with self._lock:
            if not self._utter_ready:
                return
            self._utter_ready = False
            pcm = np.concatenate(self._phrase) if self._phrase else np.zeros(0, dtype=np.int16)
            duration_ms = int(len(pcm) * 1000 / self.sample_rate)
            peak = self._phrase_peak
            speech_ms = int(self._phrase_speech_ok_chunks * self._chunk_ms)
            self._phrase = []
            self._phrase_peak = 0.0
            self._phrase_speech_ok_chunks = 0
            # Сброс счётчика превью на границе фразы: без него live-превью
            # следующей фразы включалось только после длины предыдущей.
            self._prev_interval_samples_in_phrase = 0
        # Отсев мусора: Silero-фраза без достаточной речи/длины/громкости — не фраза.
        # Громкость сравниваем с адаптивным порогом по фону: в комнате с
        # телевизором фиксированные 0.01 пропускали фоновую речь.
        required = self.noise_gate()
        if speech_ms < self._min_speech_ms or duration_ms < self._min_utterance_ms:
            return
        if peak < required:
            if required > self.silence_amplitude:
                self._noise_rejects += 1
                print(f"[recorder] ○ тише фонового шума: пик {peak:.3f} < порога "
                      f"{required:.3f} (фон {self._noise_floor:.3f}) — фраза отброшена")
            return
        if self.on_utterance:
            rec = Recording(pcm_int16=pcm, duration_ms=duration_ms, max_amplitude=peak)
            threading.Thread(target=self.on_utterance, args=(rec,),
                             name="utterance", daemon=True).start()

    # ---------- диспетчер callback ----------

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio-callback: только копия буфера, никакой тяжёлой работы."""
        if status:
            # НЕ выбрасываем блок: overflow/underflow = дыра в речи, раньше
            # она исчезала бесследно. Считаем и изредка сообщаем.
            self._status_count += 1
            if self._status_count <= 3 or self._status_count % 500 == 0:
                print(f"[audio] статус потока: {status} (всего {self._status_count})")
        try:
            self._work_q.put_nowait((indata.copy(), frames, status))
        except queue.Full:
            self._dropped_chunks += 1
            if self._dropped_chunks <= 3 or self._dropped_chunks % 500 == 0:
                print(f"[audio] очередь обработки переполнена "
                      f"(потеряно чанков: {self._dropped_chunks})")

    def _work_loop(self) -> None:
        """Разбор чанков вне RT-потока: VAD, Int16, состояние фразы."""
        while True:
            indata, frames, status = self._work_q.get()
            try:
                if self._continuous:
                    self._on_audio_continuous(indata)
                else:
                    self._on_audio_ptt(indata, frames)
            except Exception as exc:  # noqa: BLE001 — воркер не должен умирать
                print(f"[audio] ошибка обработки чанка: {exc!r}")
            finally:
                self._work_q.task_done()