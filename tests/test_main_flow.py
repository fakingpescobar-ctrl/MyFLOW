"""Регресс-тесты оркестратора: P0-1, P0-2, P0-4, P0-7, P0-8, P1-2.

Тесты работают без микрофона, серверов и GUI: все внешние зависимости —
заглушки, у реального MyFlowApp берутся только методы.
"""
import queue
import threading
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


class _VoiceCfg:
    mode = "always"          # в тестах проверяем сам факт вызова гейта


class _PolishCfg:
    def __init__(self, skip_wellformed=False):
        self.skip_wellformed = skip_wellformed


class _AsrCfg:
    initial_prompt = ""
    prompt_from_dictionary = True
    language = "ru"


class _Settings:
    hotkey = _HK()
    asr = _AsrCfg()
    polish = _PolishCfg()
    voiceprint = _VoiceCfg()
    use_ax_observation = False          # в тестах AX не поднимаем
    observation_seconds = 0.2
    app_categories = None                # -> DEFAULT_APP_CATEGORIES
    style_by_category = {"terminal": "terminal", "chat": "casual", "email": "formal"}
    audio_keep = 10
    insert_target = "focus"          # в тестах окон нет -> вставляем «в активное»
    insert_idle_seconds = 0.8
    refocus_on_insert = False

    def __init__(self):
        self.style_by_app = {}
        self.default_style = "plain"
        self.asr = _AsrCfg()
        self.polish = _PolishCfg()
        # Своя копия на каждый тест: иначе правка mode протекает между тестами.
        self.voiceprint = _VoiceCfg()
        self.flags = _Flags(auto_insert=True, learning=False, overlay=False,
                            live_preview=False, beep_on_start=False,
                            snippets=True, keep_audio=False, ax_observation=False)

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
        self.snippets: dict = {}

    def get_dictionary(self, app):
        return []

    def find_snippet(self, phrase):
        key = " ".join((phrase or "").strip().lower().split())
        return self.snippets.get(key)

    def bump_snippet(self, phrase):
        key = " ".join((phrase or "").strip().lower().split())
        if key in self.snippets:
            self.snippets[key]["hit_count"] += 1

    def insert_history(self, rec):
        self.records.append(rec)
        if self.app is not None:
            self.app._gen += self._bump   # отмена «во время обработки»
        return 7

    def get_polished_text(self, hid):
        return "привет мир"

    def mark_status(self, hid, status):
        pass


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
    app._preview_lock = threading.Lock()
    app._preview_slot = None
    app.settings = _Settings()
    app.recorder = _Recorder()
    app.overlay = _Overlay()
    app.paster = paster or _Paster()
    app.whisper = _Whisper()
    app.audio = mock.Mock()
    app.audio.save.return_value = None
    app.ax = mock.Mock()
    app.ax.available = False
    app._ax_ready = False
    app._last_insert_pid = 0
    # Проверка голоса: в тестах профиля нет -> gate пропускает всё.
    app.voice = mock.Mock()
    app.voice.check.return_value = (True, None, "нет профиля голоса")
    app.voice.maybe_auto_enroll.return_value = False
    app.voice.profile = types.SimpleNamespace(count=0)
    app.voice.threshold = 0.25
    app.voice.ready = False
    app._asr_prompt_base = "тестовая подсказка"
    app._asr_prompt_ts = 0.0
    # Цель вставки: в тестах окна нет -> вставляем «в активное» (как раньше).
    app._target = None
    app._target_locked = None
    app._stopping = threading.Event()
    app._pending = __import__('collections').deque()
    app._last_insert_ts = 0.0
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
                 "_require_last_insert", "_press_key", "_listener_watchdog_loop",
                 "_take_preview", "_on_live_preview", "_resolve_category",
                 "_save_audio", "_record_failure", "_try_snippet",
                 "_remember_insert_target", "_watch_with_ax", "_watch_with_clipboard",
                 "retry_pending", "_log_stats", "_refresh_asr_prompt",
                 "_paste_to_target", "_toggle_target_lock", "_on_utterance",
                 "_deliver", "_pending_worker", "_flush_pending",
                 "_deliver_pending_once"):
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


class PreviewPriorityTest(unittest.TestCase):
    """P2-1: превью живёт в одном слоте и не стоит впереди реальных фраз."""

    def test_latest_preview_wins(self):
        app = make_app()
        app._on_live_preview(np.zeros(640, dtype=np.int16))
        app._on_live_preview(np.zeros(320, dtype=np.int16))
        job = app._take_preview()
        self.assertIsNotNone(job)
        self.assertEqual(len(job.item.pcm_int16), 320, "должен остаться свежий драфт")
        self.assertIsNone(app._take_preview(), "слот должен опустошаться")

    def test_cancel_clears_preview(self):
        app = make_app()
        app._on_live_preview(np.zeros(640, dtype=np.int16))
        app._cancel_all()
        self.assertIsNone(app._take_preview(), "после отмены драфт не показываем")

    def test_real_phrase_has_priority_in_queue(self):
        """Фраза идёт через очередь, превью — только когда очередь пуста."""
        import time
        app = make_app()
        rec = Recording(pcm_int16=np.zeros(16000, dtype=np.int16),
                        duration_ms=1000, max_amplitude=0.5)
        app.polish.cfg.enabled = False
        app._enqueue(rec)                      # реальная фраза
        app._on_live_preview(np.zeros(640, dtype=np.int16))   # драфт в слот
        threading.Thread(target=app._worker_loop, daemon=True).start()
        time.sleep(0.4)
        self.assertEqual(len(app.paster.calls), 1, "фраза должна быть обработана")
        self.assertEqual(len(app.db.records), 1)
        self.assertEqual(app.whisper.calls, 2, "после фразы воркер берёт драфт")
        self.assertIsNone(app._take_preview())


# ---------- P0-3 (интеграция): watcher не запускается для opencode ----------

class SnippetFlowTest(unittest.TestCase):
    """Сниппеты: голосовая фраза -> готовый текст, ДО команд и полировки."""

    def setUp(self):
        self.p1 = mock.patch("myflow.main._tb_to_file")
        self.p1.start()
        self.addCleanup(self.p1.stop)

    @staticmethod
    def _rec():
        return Recording(pcm_int16=np.zeros(16000, dtype=np.int16),
                         duration_ms=1000, max_amplitude=0.5)

    def test_snippet_inserts_replacement(self):
        app = make_app()
        app.whisper = _Whisper("мой email")
        app.db.snippets["мой email"] = {"phrase": "мой email",
                                        "replacement": "user@example.com",
                                        "hit_count": 0}
        app._process_recording(0, self._rec())
        self.assertEqual(app.paster.calls, ["user@example.com"])
        self.assertEqual(len(app.db.records), 1)
        self.assertEqual(app.db.records[0].status, "snippet")
        self.assertEqual(app.db.snippets["мой email"]["hit_count"], 1,
                         "использование сниппета должно учитываться")

    def test_snippet_wins_over_polish(self):
        app = make_app(polish_enabled=True)
        app.polish.polish = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("полировка не должна вызываться для сниппета"))
        app.whisper = _Whisper("организуй мысли")
        app.db.snippets["организуй мысли"] = {"phrase": "организуй мысли",
                                              "replacement": "Organize these...",
                                              "hit_count": 0}
        app._process_recording(0, self._rec())
        self.assertEqual(app.paster.calls, ["Organize these..."])

    def test_no_snippet_goes_normal_path(self):
        app = make_app()
        app.whisper = _Whisper("обычная фраза")
        app._process_recording(0, self._rec())
        self.assertEqual(app.paster.calls, ["обычная фраза"])
        self.assertEqual(app.db.records[0].status, "ok")


class FailureRecordingTest(unittest.TestCase):
    """Неуспехи должны попадать в историю (для статистики и retry)."""

    def setUp(self):
        self.p1 = mock.patch("myflow.main._tb_to_file")
        self.p1.start()
        self.addCleanup(self.p1.stop)

    @staticmethod
    def _rec():
        return Recording(pcm_int16=np.zeros(16000, dtype=np.int16),
                         duration_ms=1000, max_amplitude=0.5)

    def test_asr_error_is_recorded(self):
        app = make_app()
        app.whisper = _Whisper()
        app.whisper.transcribe = lambda pcm: (_ for _ in ()).throw(RuntimeError("сервер упал"))
        app._process_recording(0, self._rec())
        self.assertEqual(len(app.db.records), 1)
        self.assertEqual(app.db.records[0].status, "asr_error")
        self.assertEqual(app.paster.calls, [])

    def test_empty_asr_is_recorded(self):
        app = make_app()
        app.whisper = _Whisper("")
        app._process_recording(0, self._rec())
        self.assertEqual(app.db.records[0].status, "empty_asr")

    def test_divergence_and_category_in_record(self):
        app = make_app()
        app.whisper = _Whisper("проверка связи")
        app._process_recording(0, self._rec())
        rec = app.db.records[0]
        self.assertIsNotNone(rec.divergence)
        self.assertEqual(rec.status, "ok")
        self.assertTrue(rec.app_category)


class RetryFlowTest(unittest.TestCase):
    """retry_pending: сохранённое аудио -> повторное распознавание."""

    def test_retry_processes_stored_audio(self):
        import tempfile
        from pathlib import Path
        from myflow.audio_store import AudioStore
        from myflow.db import Database, DictationRecord

        with tempfile.TemporaryDirectory() as tmp:
            store = AudioStore(Path(tmp) / "audio", keep=5)
            pcm = (np.sin(np.arange(16000) / 50) * 6000).astype(np.int16)
            path = store.save(1, pcm)
            db = Database(Path(tmp) / "h.sqlite")
            db.insert_history(DictationRecord(
                asr_text="", polished_text="", app_window="code.exe | x",
                duration_ms=1000, status="asr_error", audio_path=str(path)))

            app = make_app()
            app.db = db
            app.audio = store
            app.whisper = _Whisper("восстановленная фраза")
            done = app.retry_pending(limit=3)
            self.assertEqual(done, 1, "фраза должна быть перераспознана")
            self.assertEqual(app.paster.calls, ["восстановленная фраза"])
            statuses = [r[0] for r in db._conn.execute("SELECT status FROM history ORDER BY id")]
            self.assertIn("retried", statuses)
            db.close()

    def test_retry_marks_missing_audio(self):
        import tempfile
        from pathlib import Path
        from myflow.db import Database, DictationRecord

        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "h.sqlite")
            db.insert_history(DictationRecord(
                asr_text="", polished_text="", app_window="x",
                duration_ms=100, status="asr_error", audio_path="нет-такого-файла"))
            app = make_app()
            app.db = db
            app.audio = mock.Mock()
            app.audio.load.return_value = None
            self.assertEqual(app.retry_pending(limit=3), 0)
            status = db._conn.execute("SELECT status FROM history").fetchone()[0]
            self.assertEqual(status, "audio_missing")
            db.close()


class FastPathFlowTest(unittest.TestCase):
    """Быстрый путь в оркестраторе: оформленный текст GigaAM — без LLM."""

    def setUp(self):
        self.p1 = mock.patch("myflow.main._tb_to_file")
        self.p1.start()
        self.addCleanup(self.p1.stop)

    @staticmethod
    def _rec():
        return Recording(pcm_int16=np.zeros(16000, dtype=np.int16),
                         duration_ms=1000, max_amplitude=0.5)

    def _app(self, text, skip_wellformed=True, style="plain"):
        from myflow.polish import PolishResult
        app = make_app(polish_enabled=True)      # быстрый путь — альтернатива полировке
        app.whisper = _Whisper(text)
        app.whisper.backend_name = "gigaam"
        app.settings.get().polish.skip_wellformed = skip_wellformed
        # Стиль фиксируем: в тестах активное окно — терминал, иначе стиль
        # определился бы как terminal (быстрый путь для него не работает).
        app._resolve_style = lambda app_descriptor=None, _s=style: _s
        # LLM подменена: тесты проверяют, ЗВАЛИ её или нет, а не её ответ.
        app.polish.polish = mock.Mock(return_value=PolishResult(
            text=text + " [LLM]", latency_ms=5))
        return app

    def test_wellformed_text_skips_llm(self):
        app = self._app("Сегодня отличная погода.")
        app._process_recording(0, self._rec())
        self.assertFalse(app.polish.polish.called, "LLM звать не нужно")
        self.assertEqual(app.paster.calls, ["Сегодня отличная погода."])
        self.assertEqual(app.db.records[-1].status, "fast_path")
        self.assertEqual(app.db.records[-1].divergence, 0.0)

    def test_text_needing_polish_uses_llm(self):
        app = self._app("сегодня отличная погода")      # без пунктуации
        app._process_recording(0, self._rec())
        self.assertTrue(app.polish.polish.called, "текст без пунктуации идёт через LLM")
        self.assertEqual(app.db.records[-1].status, "ok")

    def test_whisper_backend_not_fast_pathed(self):
        app = self._app("Сегодня отличная погода.")
        app.whisper.backend_name = "whisper"
        app._process_recording(0, self._rec())
        self.assertTrue(app.polish.polish.called,
                        "whisper не ставит пунктуацию — идём через LLM")

    def test_flag_disables_fast_path(self):
        app = self._app("Сегодня отличная погода.", skip_wellformed=False)
        app._process_recording(0, self._rec())
        self.assertTrue(app.polish.polish.called)

    def test_terminal_style_never_fast_pathed(self):
        """Для стиля terminal полировка реально переформатирует текст."""
        app = self._app("Сегодня отличная погода.", style="terminal")
        app._process_recording(0, self._rec())
        self.assertTrue(app.polish.polish.called)


class VoiceGateFlowTest(unittest.TestCase):
    """Проверка голоса в оркестраторе: чужую речь не распознаём и не вставляем."""

    def setUp(self):
        self.p1 = mock.patch("myflow.main._tb_to_file")
        self.p1.start()
        self.addCleanup(self.p1.stop)

    @staticmethod
    def _rec():
        return Recording(pcm_int16=np.zeros(16000, dtype=np.int16),
                         duration_ms=1000, max_amplitude=0.5)

    def test_other_speaker_is_dropped_before_asr(self):
        app = make_app()
        app.voice.check.return_value = (False, 0.05, "чужой голос (cosine=0.050 < 0.25)")
        app.whisper = mock.Mock()
        app.polish.polish = mock.Mock()
        app._process_recording(0, self._rec())
        self.assertFalse(app.whisper.transcribe.called, "чужую речь не распознаём")
        self.assertEqual(app.paster.calls, [])
        self.assertEqual(app.db.records[-1].status, "other_speaker")

    def test_owner_voice_goes_through(self):
        app = make_app()
        app.voice.check.return_value = (True, 0.72, "голос владельца")
        app.whisper = _Whisper("Проверка голоса.")
        app._process_recording(0, self._rec())
        self.assertEqual(app.paster.calls, ["Проверка голоса."])
        self.assertEqual(app.db.records[-1].status, "ok")

    def test_auto_enroll_called_for_accepted_phrase(self):
        app = make_app()
        app.voice.check.return_value = (True, 0.72, "голос владельца")
        app.voice.maybe_auto_enroll.return_value = True
        app.whisper = _Whisper("Проверка голоса.")
        app._process_recording(0, self._rec())
        self.assertTrue(app.voice.maybe_auto_enroll.called)
        args, kwargs = app.voice.maybe_auto_enroll.call_args
        self.assertTrue(kwargs.get("accepted"))
        self.assertAlmostEqual(kwargs.get("score"), 0.72)

    def test_ptt_mode_skips_voice_check_by_default(self):
        """В PTT (удержание F8) — явное намерение, лишний отказ вреднее."""
        app = make_app(auto=False)                      # PTT
        app.settings.get().voiceprint.mode = "auto"
        app.voice.check.return_value = (False, 0.05, "чужой голос")
        app.whisper = _Whisper("Проверка PTT.")
        app._process_recording(0, self._rec())
        self.assertFalse(app.voice.check.called, "в PTT проверку не делаем")
        self.assertEqual(app.paster.calls, ["Проверка PTT."])

    def test_auto_mode_checks_voice(self):
        app = make_app(auto=True)                       # AUTO: слушает постоянно
        app.settings.get().voiceprint.mode = "auto"
        app.voice.check.return_value = (False, 0.05, "чужой голос")
        app._process_recording(0, self._rec())
        self.assertTrue(app.voice.check.called)
        self.assertEqual(app.db.records[-1].status, "other_speaker")

    def test_mode_off_skips_check_everywhere(self):
        app = make_app(auto=True)
        app.settings.get().voiceprint.mode = "off"
        app.voice.check.return_value = (False, 0.05, "чужой голос")
        app.whisper = _Whisper("Проверка off.")
        app._process_recording(0, self._rec())
        self.assertFalse(app.voice.check.called)


class AsrPromptTest(unittest.TestCase):
    """Подсказка whisper собирается из словаря (имена/термины пишутся верно)."""

    def test_dictionary_words_are_added(self):
        app = make_app()
        app._asr_prompt_base = "Пользователь диктует по-русски."
        app.db.get_dictionary = lambda app_title=None: [
            {"source_word": "сухой лаг", "target_word": "Сухой Лог"},
            {"source_word": "квен", "target_word": "Qwen"},
            {"source_word": "дубль", "target_word": "Сухой Лог"},   # дубль не нужен
        ]
        app._refresh_asr_prompt(force=True)
        prompt = app.settings.get().asr.initial_prompt
        self.assertIn("Сухой Лог", prompt)
        self.assertIn("Qwen", prompt)
        self.assertEqual(prompt.count("Сухой Лог"), 1, "дубликаты не добавляем")
        self.assertLessEqual(len(prompt), 420, "промпт ограничен по длине")

    def test_empty_dictionary_keeps_base(self):
        app = make_app()
        app._asr_prompt_base = "базовая подсказка"
        app.db.get_dictionary = lambda app_title=None: []
        app._refresh_asr_prompt(force=True)
        self.assertEqual(app.settings.get().asr.initial_prompt, "базовая подсказка")

    def test_flag_disables_dictionary_prompt(self):
        app = make_app()
        app._asr_prompt_base = "база"
        app.settings.get().asr.prompt_from_dictionary = False
        app.db.get_dictionary = lambda app_title=None: [{"target_word": "Qwen"}]
        app._refresh_asr_prompt(force=True)
        self.assertEqual(app.settings.get().asr.initial_prompt, "база")

    def test_throttled_without_force(self):
        app = make_app()
        app._asr_prompt_base = "база"
        app.db.get_dictionary = lambda app_title=None: [{"target_word": "Qwen"}]
        app._refresh_asr_prompt(force=True)
        first = app.settings.get().asr.initial_prompt
        app.db.get_dictionary = lambda app_title=None: [{"target_word": "Другое"}]
        app._refresh_asr_prompt()          # без force — в пределах минуты не трогаем
        self.assertEqual(app.settings.get().asr.initial_prompt, first)


class WatcherTest(unittest.TestCase):
    def test_watcher_skipped_without_read_api(self):
        paster = _Paster()
        paster.can_read = False
        app = make_app(paster=paster)
        with mock.patch("myflow.main.threading.Thread") as thread:
            app._spawn_edit_watcher(1, "текст")
        thread.assert_not_called()


class HelperTest(unittest.TestCase):
    """P1-7 / P2-8 / P2-9: вспомогательные функции оркестратора."""

    def test_app_descriptor_has_process_and_title(self):
        import myflow.main as M
        with mock.patch.object(M, "active_window_title", return_value="main.py — Code"), \
             mock.patch.object(M, "active_process_name", return_value="code.exe"):
            self.assertEqual(M.app_descriptor(), "code.exe | main.py — Code")
        with mock.patch.object(M, "active_window_title", return_value="unknown"), \
             mock.patch.object(M, "active_process_name", return_value=""):
            self.assertEqual(M.app_descriptor(), "unknown")

    def test_rotate_log(self):
        import tempfile
        from pathlib import Path as _P
        import myflow.main as M
        with tempfile.TemporaryDirectory() as tmp:
            path = _P(tmp) / "main.log"
            path.write_text("x" * 100, encoding="utf-8")
            M._rotate_log(path, max_bytes=50)
            self.assertFalse(path.exists())
            self.assertTrue(_P(tmp, "main.log.1").exists())
            # маленький файл не ротируется
            path.write_text("y", encoding="utf-8")
            M._rotate_log(path, max_bytes=50)
            self.assertTrue(path.exists())

    def test_console_handler_installs_without_crash(self):
        import myflow.main as M
        calls = []
        M._install_console_handler(lambda: calls.append(1))
        self.assertFalse(calls, "обработчик не должен звать exit при установке")

    def test_ansi_stripped_for_log(self):
        import myflow.main as M
        self.assertEqual(M._ANSI_RE.sub("", "\x1b[31mкрасный\x1b[0m"), "красный")


if __name__ == "__main__":
    unittest.main()
