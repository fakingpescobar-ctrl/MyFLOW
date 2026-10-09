"""FSM рекордера: AUTO-сегментация, PTT-автостоп, live-превью, лимит.

Гоняем state machine синтетическими чанками с ДЕТЕРМИНИРОВАННЫМ VAD:
реальный Silero здесь не нужен (его качество проверяют tools/bench_vad.py и
tools/validate_silero.py), а зависимость от ONNX делала тесты хрупкими.
"""
import queue
import time
import unittest

import numpy as np

import myflow.audio as audio_mod

SILENCE = np.zeros((640, 1), dtype=np.float32)
SPEECH = (0.3 * np.sin(np.arange(640) * 0.1)).astype(np.float32)[:, None]
QUIET = (0.004 * np.sin(np.arange(640) * 0.1)).astype(np.float32)[:, None]


class FakeVad:
    """Речь = RMS > 0.05. Полностью детерминирован (в отличие от Silero)."""

    threshold = 0.5

    def reset(self):
        pass

    def process(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        return 0.9 if rms > 0.05 else 0.0


class RecorderFsmTest(unittest.TestCase):
    """Стрим не открываем — гоняем state machine напрямую."""

    def _rec(self, **kwargs):
        kwargs.setdefault("vad", FakeVad())
        return audio_mod.DictationRecorder(**kwargs)

    def _auto(self, **kwargs):
        rec = self._rec(**kwargs)
        out = []
        rec.on_utterance = out.append
        return rec, out

    # ---------- AUTO ----------

    def test_phrase_after_pause(self):
        rec, out = self._auto()
        for _ in range(10):
            rec._on_audio_continuous(SILENCE)
        for _ in range(20):
            rec._on_audio_continuous(SPEECH)
        for _ in range(30):
            rec._on_audio_continuous(SILENCE)
        self.assertEqual(len(out), 1, f"ожидалась 1 фраза, got {len(out)}")
        self.assertGreaterEqual(out[0].duration_ms, 700)

    def test_second_phrase(self):
        rec, out = self._auto()
        for _ in range(20):
            rec._on_audio_continuous(SPEECH)
        for _ in range(30):
            rec._on_audio_continuous(SILENCE)
        for _ in range(5):
            rec._on_audio_continuous(SPEECH)
        for _ in range(30):
            rec._on_audio_continuous(SILENCE)
        self.assertEqual(len(out), 2, "вторая фраза не пришла")

    def test_force_finish(self):
        rec, out = self._auto()
        for _ in range(10):   # 400 мс речи (>= min_utterance_ms 300)
            rec._on_audio_continuous(SPEECH)
        self.assertTrue(rec.force_finish())
        self.assertEqual(len(out), 1)

    def test_force_finish_without_speech_is_noop(self):
        rec, out = self._auto()
        self.assertFalse(rec.force_finish())
        self.assertEqual(len(out), 0)

    def test_discard_skips_phrase(self):
        rec, out = self._auto()
        for _ in range(5):
            rec._on_audio_continuous(SPEECH)
        rec.discard()
        for _ in range(30):
            rec._on_audio_continuous(SILENCE)
        self.assertEqual(len(out), 0, "discard не скипнул фразу")

    def test_short_click_rejected(self):
        rec, out = self._auto()
        for _ in range(4):  # 160 мс < 180 мс речи
            rec._on_audio_continuous(SPEECH)
        for _ in range(30):
            rec._on_audio_continuous(SILENCE)
        self.assertEqual(len(out), 0, "клик прошёл как фраза")

    def test_quiet_background_ignored(self):
        rec, out = self._auto()
        for _ in range(30):
            rec._on_audio_continuous(QUIET)
        self.assertEqual(len(out), 0, "тихий фон разбудил фразу")

    def test_long_phrase_hard_limit(self):
        """AUTO: фраза не растёт бесконечно (MAX_DICTATION_SECONDS)."""
        old = audio_mod.MAX_DICTATION_SECONDS
        audio_mod.MAX_DICTATION_SECONDS = 1
        try:
            rec, out = self._auto()
            for _ in range(40):   # 1.6 с речи без пауз
                rec._on_audio_continuous(SPEECH)
            self.assertEqual(len(out), 1, "жёсткий лимит длины фразы не сработал")
        finally:
            audio_mod.MAX_DICTATION_SECONDS = old

    # ---------- PTT ----------

    def test_ptt_autostop_once(self):
        hits = []
        rec = self._rec(silence_pause_seconds=0.9, min_auto_stop_ms=500,
                        on_auto_stop=lambda: hits.append(1))
        rec._reset_ptt()
        for _ in range(20):  # 800 мс речи
            rec._on_audio_ptt(SPEECH, 640)
        self.assertEqual(rec.recording_duration_ms, 800)
        for _ in range(23):  # 920 мс паузы > 900
            rec._on_audio_ptt(SILENCE, 640)
        self.assertGreaterEqual(len(hits), 1, "авто-стоп не сработал")
        count = len(hits)
        for _ in range(23):
            rec._on_audio_ptt(SILENCE, 640)
        self.assertEqual(len(hits), count, "авто-стоп сработал повторно")

    def test_ptt_startup_silence_no_autostop(self):
        hits = []
        rec = self._rec(silence_pause_seconds=0.9, min_auto_stop_ms=500,
                        on_auto_stop=lambda: hits.append(1))
        rec._reset_ptt()
        for _ in range(30):  # 1.2 с тишины с самого начала
            rec._on_audio_ptt(SILENCE, 640)
        self.assertEqual(len(hits), 0, "стартовая тишина дала авто-стоп")

    def test_ptt_live_preview_window(self):
        intervals = []
        rec = self._rec(silence_pause_seconds=0.9, min_auto_stop_ms=500,
                        on_interval=intervals.append,
                        interval_seconds=0.6, interval_window_seconds=2.0)
        rec._reset_ptt()
        for _ in range(5):  # 200 мс тишины — превью молчит
            rec._on_audio_ptt(SILENCE, 640)
        time.sleep(0.05)
        self.assertEqual(len(intervals), 0, "превью на тишине")
        for _ in range(65):
            rec._on_audio_ptt(SPEECH, 640)
        time.sleep(0.1)
        self.assertGreaterEqual(len(intervals), 1, "нет превью")
        self.assertEqual(len(intervals[-1]), 2 * 16000)
        self.assertTrue(all(len(x) <= 2 * 16000 for x in intervals))

    def test_auto_preview_works_for_second_phrase(self):
        """P1-1: счётчик превью не сбрасывался -> вторая фраза без live-превью."""
        previews = []
        rec = self._rec(silence_pause_seconds=0.9, min_utterance_ms=300,
                        on_interval=previews.append,
                        interval_seconds=0.6, interval_window_seconds=2.0)
        rec._continuous = True
        for _ in range(45):   # фраза 1: 1.8 с речи
            rec._on_audio_continuous(SPEECH)
        for _ in range(30):   # пауза 1.2 с — фраза закрыта
            rec._on_audio_continuous(SILENCE)
        time.sleep(0.1)
        after_first = len(previews)
        self.assertGreater(after_first, 0, "у первой фразы не было превью")
        for _ in range(25):   # фраза 2: всего 1.0 с (меньше длины первой)
            rec._on_audio_continuous(SPEECH)
        time.sleep(0.1)
        self.assertGreater(len(previews), after_first,
                           "у второй фразы нет live-превью (счётчик не сброшен)")

    def test_ptt_hard_limit_stops_recording(self):
        """P1-10: лимит диктовки работает и в PTT (раньше буфер рос без границы)."""
        hits = []
        old = audio_mod.MAX_DICTATION_SECONDS
        audio_mod.MAX_DICTATION_SECONDS = 2
        try:
            rec = self._rec(silence_pause_seconds=99, min_auto_stop_ms=500,
                            on_auto_stop=lambda: hits.append(1))
            rec._reset_ptt()
            for _ in range(60):  # 2.4 с речи без пауз
                rec._on_audio_ptt(SPEECH, 640)
            time.sleep(0.05)
            self.assertEqual(len(hits), 1, "жёсткий лимит не сработал")
            buffered = sum(len(x) for x in rec._buffer)
            self.assertLessEqual(buffered, 2 * 16000 + 640, "буфер растёт после лимита")
        finally:
            audio_mod.MAX_DICTATION_SECONDS = old

    def test_status_blocks_are_not_dropped(self):
        """P2-3: блок со status больше не выбрасывается молча."""
        rec, out = self._auto()
        rec._on_audio(SPEECH, 640, None, "input overflow")
        self.assertEqual(rec._status_count, 1)

    def test_callback_path_reaches_worker(self):
        """P2-2: колбэк только кладёт чанк, состояние обновляет воркер."""
        rec, out = self._auto()
        rec._continuous = True
        for _ in range(20):
            rec._on_audio(SPEECH, 640, None, None)
        rec._drain_work_queue()
        self.assertTrue(rec._ring, "воркер не наполнил кольцо pre-roll")
        self.assertEqual(rec._state, "speech", "воркер не перевёл FSM в речь")

    def test_worker_drop_counter_on_overflow(self):
        """При переполнении очереди считаем потерянные чанки, не теряя поток.

        Подменяем очередь на maxsize=1: воркер остался ждать на СТАРОЙ
        очереди, поэтому новая гарантированно переполняется.
        """
        rec, out = self._auto()
        rec._work_q = queue.Queue(maxsize=1)
        for _ in range(3):
            rec._on_audio(SPEECH, 640, None, None)
        self.assertGreaterEqual(rec._dropped_chunks, 1)


if __name__ == "__main__":
    unittest.main()
