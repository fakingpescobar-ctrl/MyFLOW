"""Регресс-тесты оркестратора: P0-1, P0-2, P0-4, P0-7, P0-8, P1-2.

Тесты работают без микрофона, серверов и GUI: все внешние зависимости —
заглушки, у реального MyFlowApp берутся только методы.
"""
import queue
import types
import unittest
from unittest import mock

import numpy as np
from pynput import keyboard

from myflow.asr import AsrResult
from myflow.audio import Recording
from myflow.commands import VoiceCommand
from myflow.main import _Job, _LivePreview, MyFlowApp
from myflow.polish import PolishResult


# ---------- заглушки ----------

class _HK:
    ptt_key = "f8"
    cancel_key = "f9"
    style_cycle_key = "f7"
    suppress = False


class _Hotkeys:
    hotkey = _HK()


class _Flags(dict):
    pass


class _Settings:
    hotkey = _HK()

    def __init__(self):
        self.style_by_app = {}
        self.default_style = "plain"
        self.flags = _Flags(auto_insert=True, learning=False, overlay=False,
                            live_preview=False, beep_on_start=False)

    def get(self):
        return self


class _Overlay:
    def __init__(self):
        self.shown = []

    def show(self, *args, **kwargs):
        self.shown.append(args)


class _Recorder:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self._current = None

    def start(self):
        if self._current is not None:
            raise RuntimeError("запись уже идёт")
        self.started += 1
        self._current = Recording(
            pcm_int16=np.zeros(16000, dtype=np.int16),
            duration_ms=1000, max_amplitude=0.5)

    def stop(self):
        if self._current is None:
            raise RuntimeError("нет активной записи")
        self.stopped += 1
        rec, self._current = self._current, None
        return rec

    def force_finish(self):
        return False

    def discard(self):
        self._current = None


class _Paster:
    can_read = True

    def __init__(self):
        self.calls = []

    def paste(self, text):
        self.calls.append(text)

    def read(self):
        return ""


class _Whisper:
    def __init__(self, text="привет мир"):
        self.text = text
        self.calls = 0

    def transcribe(self, pcm):
        self.calls += 1
        return AsrResult(text=self.text, language="ru", latency_ms=10)


class _DB:
    def __init__(self, gen_bump_on_insert=0):
        self.records = []
        self._bump = gen_bump_on_insert
        self.app = None

    def get_dictionary(self, app):
        return []

    def insert_history(self, rec):
        self.records.append(rec)
        if self.app is not None:
            self.app._gen += self._bump   # отмена «во время обработки»
        return 7

    def get_polished_text(self, hid):
        return "привет мир"


def make_app(auto=False, busy=False, gen=0, polish_enabled=False, paster=None):
    app = types.SimpleNamespace()
    app._pref_auto = auto
    app._key_down = False
    app._ptt_held = False
    app._busy = busy
    app._gen = gen
    app._auto_stopped = False
    app._polish_fails = 0
    app._polish_watchdog = None
    app._style_override = None
    app._style_cycle = ["plain"]
    app._last_style_switch_t = 0.0
    app._last_history_id = None
    app._last_insert_ts = 0.0
    app._queue = queue.Queue()
    app.settings = _Settings()
    app.recorder = _Recorder()
    app.overlay = _Overlay()
    app.paster = paster or _Paster()
    app.whisper = _Whisper()
    app.polish = types.SimpleNamespace(
        cfg=types.SimpleNamespace(enabled=polish_enabled, url="http://127.0.0.1:8082/v1"),
        polish=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("LLM лежит")),
        _effective_model=lambda: "qwen",
    )
    app.db = _DB()
    app.db.app = app
    for name in ("_handle_press", "_handle_release", "_begin_recording", "_finish_recording",
                 "_enqueue", "_cancel_all", "_polish_safe", "_ensure_polish_watchdog",
                 "_polish_watchdog_loop", "_worker_loop", "_process_recording",
                 "_dispatch_voice_command", "_run_command", "_resolve_style",
                 "_spawn_edit_watcher", "_process_preview", "_cycle_style",
                 "_require_last_insert", "_press_key", "_listener_watchdog_loop"):
        setattr(app, name, types.MethodType(getattr(MyFlowApp, name), app))
    app._guard_polish = MyFlowApp._guard_polish
    return app


# ---------- P0-1: интент не съедает диктовку ----------

class IntentNoDropTest(unittest.TestCase):
    def test_text_kind_is_not_a_command(self):
        """Реальный баг: LLM ответил kind=text -> _process_recording делал return."""
        app = make_app(polish_enabled=True)
        with mock.patch("myflow.main.llm_classify",
                        return_value=VoiceCommand(kind="text", text="что угодно")):
            cmd = app._dispatch_voice_command("брат кратко удали этот файл потом")
        self.assertIsNone(cmd, "kind=text должен обрабатываться как обычная диктовка")

    def test_unknown_kind_is_not_a_command(self):
        app = make_app(polish_enabled=True)
        with mock.patch("myflow.main.llm_classify",
                        return_value=VoiceCommand(kind="translate", text="x")):
            self.assertIsNone(app._dispatch_voice_command("кратко переведи текст"))

    def test_real_command_still_works(self):
        app = make_app(polish_enabled=True)
        self.assertEqual(app._dispatch_voice_command("стоп").kind, "stop")
        self.assertEqual(app._dispatch_voice_command("новый абзац").kind, "new_paragraph")

    def test_plain_dictation_is_not_a_command(self):
        app = make_app(polish_enabled=True)
        self.assertIsNone(app._dispatch_voice_command("сегодня хорошая погода"))


# ---------- P0-2: полировка не теряет фразу ----------

class PolishFallbackTest(unittest.TestCase):
    def setUp(self):
        # Не пишем тестовые трейсбеки в боевой data/main.log
        self.p1 = mock.patch("myflow.main._tb_to_file")
        self.p1.start()
        self.p2 = mock.patch("myflow.main._health_check_llm", return_value=False)
        self.p2.start()
        self.addCleanup(self.p1.stop)
        self.addCleanup(self.p2.stop)

    def test_llm_error_falls_back_to_raw(self):
        app = make_app(polish_enabled=True)
        result = app._polish_safe("привет мир", "plain", [])
        self.assertTrue(result.skipped)
        self.assertEqual(result.text, "привет мир", "фраза не должна теряться")

    def test_three_failures_disable_polish(self):
        app = make_app(polish_enabled=True)
        for _ in range(3):
            app._polish_safe("привет", "plain", [])
        self.assertFalse(app.polish.cfg.enabled)
        self.assertIsNotNone(app._polish_watchdog)

    def test_success_resets_failure_counter(self):
        app = make_app(polish_enabled=True)
        app._polish_safe("раз", "plain", [])
        app.polish.polish = lambda *a, **k: PolishResult(text="Раз.", latency_ms=5)
        app._polish_safe("раз", "plain", [])
        self.assertEqual(app._polish_fails, 0)

    def test_disabled_polish_returns_raw(self):
        app = make_app(polish_enabled=False)
        result = app._polish_safe("привет", "plain", [])
        self.assertTrue(result.skipped)
        self.assertEqual(result.text, "привет")

    def test_truncated_response_is_not_inserted(self):
        """P1-4: обрезанный ответ LLM -> сырой ASR, а не «полуслово»."""
        app = make_app(polish_enabled=True)
        app.polish.polish = lambda *a, **k: PolishResult(
            text="Привет, бр", latency_ms=5, truncated=True)
        paster = app.paster
        rec = Recording(pcm_int16=np.zeros(16000, dtype=np.int16),
                        duration_ms=1000, max_amplitude=0.5)
        app.whisper = _Whisper("привет брат")
        app._process_recording(0, rec)
        self.assertEqual(paster.calls, ["привет брат"])


# ---------- P0-8 / P1-4: guard_polish ----------

class GuardPolishTest(unittest.TestCase):
    def test_logged_truncation_case(self):
        """data/main.log: ASR «Проверка связи.» -> LLM «Провер» -> вставка «Провер»."""
        self.assertEqual(
            MyFlowApp._guard_polish("Проверка связи.", "Провер", "plain"),
            "Проверка связи.")

    def test_short_phrase_word_drop(self):
        self.assertEqual(MyFlowApp._guard_polish("да брат", "брат", "plain"), "да брат")

    def test_one_word_not_truncated(self):
        self.assertEqual(MyFlowApp._guard_polish("перезапусти", "перезапусти", "plain"),
                         "перезапусти")
        self.assertEqual(MyFlowApp._guard_polish("перезапусти", "Пере", "plain"),
                         "перезапусти")

    def test_hallucination_rejected(self):
        raw = "слушай скажи какая погода"
        hallucination = raw + " " + "Ошибка распознавания: " * 20
        self.assertEqual(MyFlowApp._guard_polish(raw, hallucination, "plain"), raw)

    def test_empty_answer_falls_back_to_raw(self):
        self.assertEqual(MyFlowApp._guard_polish("привет", "   ", "plain"), "привет")

    def test_normal_polish_passes(self):
        self.assertEqual(
            MyFlowApp._guard_polish("привет как дела", "Привет, как дела?", "plain"),
            "Привет, как дела?")

    def test_summarize_can_shorten(self):
        raw = "сегодня мы обсудили план проекта и сроки и бюджеты и риски"
        summary = "Обсудили план проекта и сроки."
        self.assertEqual(MyFlowApp._guard_polish(raw, summary, "summarize"), summary)

    def test_summarize_losing_numbers_rejected(self):
        raw = "сервер слушает порт 8081 и 8080"
        summary = "Сервер слушает порт."
        self.assertEqual(MyFlowApp._guard_polish(raw, summary, "summarize"), raw)


# ---------- P0-7 / P0-4: PTT и хоткеи ----------

class PttHotkeyTest(unittest.TestCase):
    def setUp(self):
        self.p1 = mock.patch("myflow.main._tb_to_file")
        self.p1.start()
        self.addCleanup(self.p1.stop)

    def test_press_while_busy_still_starts_recording(self):
        """Раньше F8 при _busy молча игнорировался — начало речи терялось."""
        app = make_app(busy=True)
        app._handle_press(keyboard.Key.f8)
        self.assertEqual(app.recorder.started, 1)
        self.assertTrue(app._ptt_held)

    def test_key_autorepeat_does_not_restart(self):
        app = make_app()
        app._handle_press(keyboard.Key.f8)
        app._handle_press(keyboard.Key.f8)   # автоповтор
        app._handle_press(keyboard.Key.f8)
        self.assertEqual(app.recorder.started, 1)

    def test_autostop_then_autorepeat_does_not_record_again(self):
        """P0-7: после VAD-автостопа удержание F8 не должно начинать новую запись."""
        app = make_app()
        app._handle_press(keyboard.Key.f8)
        app._ptt_held = False          # как делает _on_auto_stop
        app._handle_press(keyboard.Key.f8)   # клавиша всё ещё удерживается
        self.assertEqual(app.recorder.started, 1, "фраза не должна резаться на куски")

    def test_release_enqueues_job_with_generation(self):
        app = make_app()
        app._handle_press(keyboard.Key.f8)
        app._handle_release(keyboard.Key.f8)
        self.assertEqual(app.recorder.stopped, 1)
        job = app._queue.get_nowait()
        self.assertIsInstance(job, _Job)
        self.assertEqual(job.gen, 0)
        self.assertIsInstance(job.item, Recording)

    def test_release_without_press_is_safe(self):
        app = make_app()
        app._handle_release(keyboard.Key.f8)
        self.assertEqual(app.recorder.stopped, 0)

    def test_release_when_already_stopped_is_safe(self):
        """Гонка автостопа и отпускания не должна валить хоткей (P0-4)."""
        app = make_app()
        app._handle_press(keyboard.Key.f8)
        app.recorder._current = None      # «автостоп уже остановил запись»
        app._handle_release(keyboard.Key.f8)   # не должно бросить исключение

    def test_recorder_error_in_callback_does_not_propagate(self):
        """P0-4: исключение в колбэке убивало pynput-listener навсегда."""
        app = make_app()

        def boom():
            raise RuntimeError("не удалось открыть микрофон")

        app.recorder.start = boom
        app._on_press = types.MethodType(MyFlowApp._on_press, app)
        app._on_press(keyboard.Key.f8)   # не должно бросить наружу
        self.assertTrue(app.overlay.shown, "ошибка должна быть видна в оверлее")

    def test_auto_mode_press_does_not_start_recording(self):
        app = make_app(auto=True)
        app._handle_press(keyboard.Key.f8)
        self.assertEqual(app.recorder.started, 0)

    def test_style_key_cycles(self):
        app = make_app()
        app._cycle_style(debounce=0.0)
        self.assertEqual(app._style_override, "plain")


# ---------- P1-2: отмена по поколению ----------

class CancelTest(unittest.TestCase):
    def test_cancel_bumps_generation_and_stops_recording(self):
        app = make_app()
        app._handle_press(keyboard.Key.f8)
        app._cancel_all()
        self.assertEqual(app._gen, 1)
        self.assertEqual(app.recorder.stopped, 1)
        self.assertFalse(app._ptt_held)

    def test_worker_skips_stale_preview(self):
        """P1-2: превью старого поколения не должно доходить до whisper."""
        import threading
        import time
        app = make_app(gen=5)
        app._queue.put(_Job(gen=1, item=_LivePreview(pcm_int16=np.zeros(320, dtype=np.int16))))
        threading.Thread(target=app._worker_loop, daemon=True).start()
        time.sleep(0.25)
        self.assertEqual(app.whisper.calls, 0)

    def test_worker_processes_fresh_preview(self):
        import threading
        import time
        app = make_app(gen=5)
        app._queue.put(_Job(gen=5, item=_LivePreview(pcm_int16=np.zeros(320, dtype=np.int16))))
        threading.Thread(target=app._worker_loop, daemon=True).start()
        time.sleep(0.25)
        self.assertEqual(app.whisper.calls, 1)

    def test_cancelled_before_processing_is_dropped(self):
        app = make_app(gen=1)
        rec = Recording(pcm_int16=np.zeros(16000, dtype=np.int16),
                        duration_ms=1000, max_amplitude=0.5)
        app._process_recording(0, rec)      # gen устарел
        self.assertEqual(app.whisper.calls, 0, "ASR не должен запускаться")
        self.assertEqual(app.paster.calls, [])

    def test_cancel_during_processing_skips_insert_but_keeps_history(self):
        """F9 «не отменяет»: отмена во время ASR/полировки не вставляла текст."""
        app = make_app()
        app.db._bump = 1                     # отмена приходит во время обработки
        rec = Recording(pcm_int16=np.zeros(16000, dtype=np.int16),
                        duration_ms=1000, max_amplitude=0.5)
        app._process_recording(0, rec)
        self.assertEqual(len(app.db.records), 1, "текст должен остаться в истории")
        self.assertEqual(app.paster.calls, [], "вставлять отменённое нельзя")

    def test_skip_command_bumps_generation(self):
        app = make_app()
        app._run_command(VoiceCommand(kind="skip"))
        self.assertEqual(app._gen, 1)


# ---------- P0-3 (интеграция): watcher не запускается для opencode ----------

class WatsonTest(unittest.TestCase):
    def test_watcher_skipped_without_read_api(self):
        paster = _Paster()
        paster.can_read = False
        app = make_app(paster=paster)
        with mock.patch("myflow.main.threading.Thread") as thread:
            app._spawn_edit_watcher(1, "текст")
        thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
