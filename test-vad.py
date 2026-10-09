"""Временный тест VAD-автостопа и AUTO-режима (без реального микрофона).

Гоняем state-machine DictationRecorder._on_audio_continuous синтетическими
чанками: тишина -> речь -> пауза -> снова речь -> force_finish -> discard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from myflow.audio import DictationRecorder

silence = np.zeros((640, 1), dtype=np.float32)
speech = (0.3 * np.sin(np.arange(640) * 0.1)).astype(np.float32)[:, None]


def main() -> int:
    rec = DictationRecorder()  # стрим НЕ открываем — гоняем machine напрямую
    out = []
    rec.on_utterance = lambda r: out.append(r)

    # 1) тишина 10 чанков -> речь 20 чанков (800мс) -> пауза 30 чанков (1200мс>900)
    for _ in range(10):
        rec._on_audio_continuous(silence)
    for _ in range(20):
        rec._on_audio_continuous(speech)
    for _ in range(30):
        rec._on_audio_continuous(silence)
    assert len(out) == 1, f"ожидалась 1 фраза, got {len(out)}"
    r1 = out[0]
    assert r1.duration_ms >= 700, f"короткая фраза: {r1.duration_ms}"
    print(f"  1) фраза после паузы: {r1.duration_ms} мс, peak={r1.max_amplitude:.3f}")

    # 2) снова речь + пауза -> вторая фраза
    for _ in range(5):
        rec._on_audio_continuous(speech)
    for _ in range(30):
        rec._on_audio_continuous(silence)
    assert len(out) == 2, f"вторая фраза не пришла: {len(out)}"
    print(f"  2) вторая фраза: {out[1].duration_ms} мс")

    # 3) речь -> force_finish (ручной стоп кнопкой)
    for _ in range(5):
        rec._on_audio_continuous(speech)
    assert rec.force_finish() is True
    assert len(out) == 3, f"force_finish не отдал фразу: {len(out)}"
    print(f"  3) force_finish: фраза {out[2].duration_ms} мс")

    # 4) речь -> discard (скип) — фраза НЕ уходит
    for _ in range(5):
        rec._on_audio_continuous(speech)
    rec.discard()
    for _ in range(30):
        rec._on_audio_continuous(silence)
    assert len(out) == 3, f"discard не скипнул: {len(out)}"
    print("  4) discard: скип без эмита — OK")

    # 5) клик-шум (короткий, <300мс) — отсев
    for _ in range(4):  # 160мс
        rec._on_audio_continuous(speech)
    for _ in range(30):
        rec._on_audio_continuous(silence)
    assert len(out) == 3, f"клик прошёл: {len(out)}"
    print("  5) отсев короткого клика — OK")

    # 6) тихая фраза (амплитуда ниже порога) — не триггерится вообще
    quiet = (0.004 * np.sin(np.arange(640) * 0.1)).astype(np.float32)[:, None]
    for _ in range(30):
        rec._on_audio_continuous(quiet)
    assert len(out) == 3, f"тихий фон разбудил: {len(out)}"
    print("  6) тихий фон не триггерит — OK")

    # 7) PTT-режим: авто-стоп по паузе (on_auto_stop один раз)
    hits = []
    ptt = DictationRecorder(silence_pause_seconds=0.9,
                            min_auto_stop_ms=500,
                            on_auto_stop=lambda: hits.append(1))
    ptt._stream = object()  # имитация открытого стрима (стрим не трогаем)
    ptt._reset_ptt()
    for _ in range(20):  # 800мс речи
        ptt._on_audio_ptt(speech, 640)
    assert ptt.recording_duration_ms == 800
    for _ in range(23):  # 920мс паузы > 900мс
        ptt._on_audio_ptt(silence, 640)
    assert len(hits) == 1, f"авто-стоп: {len(hits)}"
    for _ in range(23):  # повторная тишина — не дёргает снова
        ptt._on_audio_ptt(silence, 640)
    assert len(hits) == 1, f"повторный авто-стоп: {len(hits)}"
    print("  7) PTT авто-стоп по паузе (1 раз) — OK")

    # 8) PTT: чистая тишина с самого начала (< 500мс записи) — триггера нет
    hits2 = []
    ptt2 = DictationRecorder(silence_pause_seconds=0.9,
                             min_auto_stop_ms=500,
                             on_auto_stop=lambda: hits2.append(1))
    ptt2._stream = object()
    ptt2._reset_ptt()
    for _ in range(30):  # 1200мс тишины, но запись изначально тихая
        ptt2._on_audio_ptt(silence, 640)
    assert len(hits2) == 0, f"тишина-с-начала дала авто-стоп: {len(hits2)}"
    print("  8) PTT: стартовая тишина авто-стоп не даёт — OK")

    # 9) live-превью: интервалы отдают хвост (окно 2с), на тишине — молчат
    import time
    intervals = []
    ptt3 = DictationRecorder(silence_pause_seconds=0.9, min_auto_stop_ms=500,
                             on_interval=lambda p: intervals.append(len(p)),
                             interval_seconds=0.6, interval_window_seconds=2.0)
    ptt3._stream = object()
    ptt3._reset_ptt()
    for _ in range(5):  # 200мс тишины
        ptt3._on_audio_ptt(silence, 640)
    time.sleep(0.05)
    assert len(intervals) == 0, f"интервал на тишине: {len(intervals)}"
    for _ in range(65):  # 2.6с речи — запись перерастёт окно 2с
        ptt3._on_audio_ptt(speech, 640)
    time.sleep(0.1)
    assert len(intervals) >= 1, f"нет интервалов: {len(intervals)}"
    assert intervals[-1] == 2 * 16000, f"последнее окно не 2с: {intervals[-1]}"
    assert all(x <= 2 * 16000 for x in intervals), f"окно больше 2с: {intervals}"
    print(f"  9) PTT live-превью (окно 2с, интервалов={len(intervals)}) — OK")

    # 10) AUTO (continuous): превью длинной фразы, на старте-тишине — нет
    intervals_c = []
    cont = DictationRecorder(silence_pause_seconds=0.9, min_utterance_ms=300,
                             on_interval=lambda p: intervals_c.append(len(p)),
                             interval_seconds=0.6, interval_window_seconds=2.0)
    cont._stream = object()
    cont._continuous = True
    cont._state = "idle"
    cont._speech_run = 0
    cont._silent_run = 0
    cont._ring = []
    cont._phrase = []
    cont._phrase_peak = 0.0
    cont._phrase_speech_samples = 0
    cont._prev_interval_samples_in_phrase = 0
    for _ in range(10):  # 400мс тишины — idle, интервалов нет
        cont._on_audio_continuous(silence)
    time.sleep(0.05)
    assert len(intervals_c) == 0, f"превью до речи: {len(intervals_c)}"
    for _ in range(70):  # 2.8с речи — фраза активна, интервалы идут
        cont._on_audio_continuous(speech)
    time.sleep(0.1)
    assert len(intervals_c) >= 1, f"нет превью фразы: {len(intervals_c)}"
    assert intervals_c[-1] == 2 * 16000, f"окно не 2с: {intervals_c[-1]}"
    print(f"  10) AUTO live-превью длинной фразы (интервалов={len(intervals_c)}) — OK")
    # фраза всё ещё идёт (тишина 0.9с ещё не набрана) — продолжаем, потом стоп
    cont._stream = None
    cont._continuous = False

    print("VAD TEST: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())