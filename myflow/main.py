"""MyFLOW — оркестратор (аналог main/index.js у Wispr Flow).

Пайплайн одной диктовки:
  PTT (F8) или голосовой триггер (AUTO) -> запись фразы
  -> whisper (asr_text) -> LLM polish (+словарь, стиль) -> вставка
  -> история в SQLite (+ обучение из последующих правок).

Режимы (settings.mode, выбирается при запуске в pick_mic):
  ptt  — запись по нажатию кнопки (держишь F8); пауза >0.9с тоже завершает
  auto — микрофон слушает постоянно: фраза режется по тишине, вставляется,
         дальше слушает снова (F8 = срочный стоп фразы, F9 = скип)

Запуск:  python -m myflow.main
"""
from __future__ import annotations

import ctypes
import queue
import re
import sys
import threading
from collections import deque
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pynput import keyboard

from myflow.audio import DictationRecorder, Recording
from myflow.asr import WhisperClient, AsrResult, make_asr_client
from myflow.audio_store import AudioStore
from myflow.axtext import AxReader, AxWatcher
from myflow.db import Database, DictationRecord
from myflow.config import SettingsStore, DATA_DIR, DEFAULT_APP_CATEGORIES, detect_category
from myflow.fastpath import fast_path_allowed, looks_wellformed
from myflow.lightclean import light_clean, strip_inner_fillers
from myflow.pick_mic import fast_mic_list
from myflow.voiceprint import VoiceGate
from myflow.winfocus import (FocusHistory, WindowRef, current_window,
                             focus_window, idle_seconds, window_exists)
from myflow.insertion import ClipboardPaste
from myflow.opencode_inject import OpenCodeInjector
from myflow.learning import record_edit_and_learn, watch_pasted_edit
from myflow.overlay import StatusOverlay
from myflow.pick_mic import pick_mic
from myflow.polish import PolishClient, PolishResult, STYLE_PROMPTS
from myflow.textdiff import divergence_score, MAX_DIVERGENCE
from myflow.commands import (
    classify as cmd_classify,
    is_llm_intent_candidate,
    llm_classify,
    VoiceCommand,
    KNOWN_KINDS,
)

# Подсказка whisper: типичные разговорные фразы, чтобы «ты можешь» не
# сливалось в «вывести» и т.п. Задаётся как initial_prompt, если не
# переопределено в settings.json (asr.initial_prompt).
ASR_INITIAL_PROMPT = (
    "Пользователь диктует по-русски короткие сообщения для чата. "
    "Часто начинает с обращения «брат» или: «А ты можешь», «Можешь», "
    "«Ты можешь», «Покажи», «Вывести», «Сделай», «Напиши», "
    "«Перезапусти», «Какая погода». "
    "Примеры: «брат, какая погода в Сухом Логу?», «брат, ты можешь картинкой показать?», "
    "«брат, перезапусти батник». "
    "Распознай речь точно, не добавляя и не пропуская слова из подсказки."
)


def _tb_to_file(marker: str) -> None:
    """Записать текущий traceback в data/main.log в обход Tee/консоли.

    После FreeConsole() stdout может указывать на мёртвый дескриптор —
    вывод в файл должен идти ПЕРВЫМ, до любых консольных операций.
    """
    import traceback

    try:
        log_path = Path(__file__).resolve().parent.parent / "data" / "main.log"
        with open(log_path, "a", encoding="utf-8", errors="replace") as _lf:
            _lf.write(f"\n=== {marker} ===\n")
            traceback.print_exc(file=_lf)
            _lf.write("\n")
    except Exception:
        pass


@dataclass
class _LivePreview:
    """Драфт-кусок для инкрементального ASR (не вставляется, не в БД)."""
    pcm_int16: np.ndarray


@dataclass
class _Job:
    """Элемент очереди: нагрузка + поколение (для честной отмены).

    Отмена (F9 или голосовые «стоп/скип») инкрементит self._gen: всё, что
    лежит в очереди со старым поколением, не вставляется. Раньше F9 в AUTO
    отменял только текущую фразу, а уже поставленные в очередь всё равно
    вставлялись.
    """
    gen: int
    item: Recording | _LivePreview


_NUM_RE = re.compile(r"\d+")


def _numbers(text: str) -> set[str]:
    """Числа в тексте (для проверки, что пересказ их не потерял)."""
    return set(_NUM_RE.findall(text or ""))


def active_window_title() -> str:
    """Имя активного окна (Wispr логирует session_dictation_apps). Windows-only."""
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    if not length:
        return "unknown"
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value[:64] or "unknown"


def active_process_name() -> str:
    """Имя процесса активного окна ("code.exe") — надёжная маска per-app.

    Заголовок окна меняется при смене файла/чата/вкладки, поэтому маски
    словаря по заголовку почти никогда не совпадали. Имя процесса стабильно.
    """
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return ""
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_ulong(len(buf))
            if ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return Path(buf.value).name.lower()
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:  # noqa: BLE001 — Windows-only, не критично
        return ""
    return ""


def app_descriptor() -> str:
    """Строка для history.app_window: «процесс | заголовок».

    Маски per-app (словарь и стили) ищутся как подстрока, поэтому работают
    и по имени процесса, и по фрагменту заголовка (обратная совместимость).
    """
    title = active_window_title()
    proc = active_process_name()
    return f"{proc} | {title}" if proc else title


def _set_thread_priority_below_normal() -> None:
    """Понизить приоритет ТЕКУЩЕГО потока (Windows): фоновые загрузки не мешают UI."""
    if sys.platform != "win32":
        return
    try:
        THREAD_PRIORITY_BELOW_NORMAL = -1
        ctypes.windll.kernel32.SetThreadPriority(
            ctypes.windll.kernel32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)
    except Exception:  # noqa: BLE001 — не критично
        pass


def _health_check(whisper) -> bool:
    """Fail fast на старте: если ASR не готов, сказать громко и упасть.

    Работает и для whisper-server (HTTP /health), и для локального GigaAM
    (у него свой health(): доступен ли пакет/модель).
    """
    if hasattr(whisper, "health"):
        try:
            return bool(whisper.health())
        except Exception:  # noqa: BLE001
            return False
    import requests
    try:
        r = requests.get(f"{whisper.cfg.url}/health", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


# Пока LLM лежит, не долбим её на каждой фразе: на этой машине отказ соединения
# стоит до ~2 секунд (порт не отвергает, а «висит»), и каждое обращение
# превращалось в заметную задержку диктовки. Держим короткий кэш «лежит».
_LLM_DOWN_UNTIL = 0.0
_LLM_DOWN_TTL = 3.0


def _health_check_llm(polish: PolishClient, *, cached: bool = True) -> bool:
    """Жива ли LLM. Быстро (~0.3 с) и с кэшем неудачи на несколько секунд."""
    global _LLM_DOWN_UNTIL
    now = time.monotonic()
    if cached and now < _LLM_DOWN_UNTIL:
        return False
    alive = False
    checker = getattr(polish, "alive", None)
    try:
        if checker is not None:
            alive = bool(checker())
        else:
            # Клиент без alive() (тестовые заглушки, старые сборки) — проверяем
            # обычным запросом, но с коротким connect-таймаутом.
            import requests
            url = getattr(getattr(polish, "cfg", None), "url", "")
            if url:
                alive = bool(requests.get(f"{url}/models", timeout=(0.35, 2.0)).ok)
    except Exception:  # noqa: BLE001
        alive = False
    if alive:
        _LLM_DOWN_UNTIL = 0.0
    else:
        _LLM_DOWN_UNTIL = now + _LLM_DOWN_TTL
    return alive


class MyFlowApp:
    def __init__(self, overlay: StatusOverlay | None = None,
                 settings: SettingsStore | None = None,
                 db: Database | None = None) -> None:
        """overlay — свой статус-слой (TUI подставляет свой); None = обычная плашка.

        settings/db можно подсунуть свои: это нужно тестам, чтобы переключения
        на панели не писались в ЖИВОЙ settings.json и в рабочую базу.
        """
        self.settings = settings or SettingsStore()
        cfg = self.settings.get()
        self.db = db or Database()
        self._pref_auto = cfg.mode == "auto"
        # VAD: общий для обоих режимов параметр паузы; в ptt работает как
        # автостоп при удержании F8, в auto — это конец фразы.
        self.recorder = DictationRecorder(
            device=cfg.asr.input_device,
            silence_pause_seconds=cfg.silence_pause_seconds,
            silence_amplitude=cfg.silence_amplitude,
            adaptive_noise_gate=cfg.adaptive_noise_gate,
            noise_ratio=cfg.noise_ratio,
            noise_max_gate=cfg.noise_max_gate,
            on_auto_stop=self._on_auto_stop if (not self._pref_auto and cfg.auto_stop_by_silence) else None,
            on_utterance=self._on_utterance if self._pref_auto else None,
            on_interval=self._on_live_preview if cfg.flags.get("live_preview", True) else None,
        )
        self.whisper = make_asr_client(cfg.asr)
        # Фразы-подсказки для ASR: дефолт, если не заданы в settings.json.
        if not cfg.asr.initial_prompt:
            cfg.asr.initial_prompt = ASR_INITIAL_PROMPT
        self._asr_prompt_base = cfg.asr.initial_prompt
        self._asr_prompt_ts = 0.0
        self.polish = PolishClient(cfg.polish)
        # Проверка голоса (speaker verification): отсекает чужую речь, которую
        # VAD честно считает речью. Модель грузится в фоне (см. _warm_asr).
        self.voice = VoiceGate(
            enabled=cfg.voiceprint.enabled,
            threshold=cfg.voiceprint.threshold,
            min_seconds=cfg.voiceprint.min_seconds,
            auto_enroll=cfg.voiceprint.auto_enroll,
            enroll_min_score=cfg.voiceprint.enroll_min_score,
            model_path=Path(cfg.voiceprint.model_path) if cfg.voiceprint.model_path else None,
            profile_path=(Path(cfg.voiceprint.profile_path)
                          if cfg.voiceprint.profile_path else None),
        )
        # Куда вставлять: обычно — буфер+Ctrl+V в активное окно; в opencode-режиме —
        # WriteConsoleInputW напрямую в консоль (не требует фокуса, Enter по флагу).
        if cfg.insert_method == "opencode":
            self.paster = OpenCodeInjector(auto_enter=cfg.opencode_auto_enter)
        elif cfg.insert_method == "clipboard":
            self.paster = ClipboardPaste()
        else:
            raise ValueError(
                f"неизвестный insert_method: {cfg.insert_method!r} "
                "(допустимо: 'clipboard' | 'opencode')"
            )
        self.overlay = overlay if overlay is not None else StatusOverlay()
        # Аудио последних фраз: позволяет перераспознать упавшую фразу (retry).
        self.audio = AudioStore(DATA_DIR / "audio", keep=cfg.audio_keep)
        # Наблюдение за полем ввода через UI Automation (обучение без Ctrl+C).
        self.ax = AxReader()
        self._ax_ready = False
        self._last_insert_pid = 0
        self._last_insert_ts: float = 0.0    # момент последней вставки (delete_last: защита <60с)
        # Цель вставки: окно, активное в момент старта диктовки. Нужна, чтобы
        # случайный клик мышью во время фразы не увёл текст в другое окно.
        self._focus_history = FocusHistory()
        self._target: WindowRef | None = None
        self._target_locked: WindowRef | None = None   # явно закреплённая цель (F6)
        self._focus_thread: threading.Thread | None = None
        # Микрофоны для панели: (список, время) + фоновое сканирование.
        self._mic_cache: tuple[list[dict], float] | None = None
        self._mic_thread: threading.Thread | None = None
        self._mic_scanning = False
        # Отложенные вставки (пользователь был занят) — FIFO, ничего не теряем.
        self._pending: deque[tuple[str, WindowRef | None, int | None]] = deque()
        self._stopping = threading.Event()             # сигнал остановки фоновых петель
        # Очередь фраз: worker обрабатывает последовательно, а recorder в
        # AUTO-режиме тем временем продолжает слушать (не теряем фразы).
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._gen = 0                    # поколение отмены (см. _Job)
        # Live-превью держим в ОДНОМ слоте (последнее побеждает) и разбираем
        # только когда очередь фраз пуста: драфты не должны стоять впереди
        # реальной диктовки и тормозить вставку.
        self._preview_lock = threading.Lock()
        self._preview_slot: _Job | None = None
        self._worker: threading.Thread | None = None
        self._busy = False
        self._ptt_held = False           # логически «идёт PTT-запись»
        self._key_down = False           # F8 физически удерживается (автоповтор)
        self._auto_stopped = False
        self._listener: keyboard.Listener | None = None
        self._listener_watchdog: threading.Thread | None = None
        self._last_history_id: int | None = None
        self._polish_fails = 0
        self._polish_watchdog: threading.Thread | None = None
        # Автолечение LLM: «подниматель» серверов + очередь недополированных.
        self._server_guard = None
        self._llm_revive: threading.Thread | None = None
        self._repolish_thread: threading.Thread | None = None
        self._deferred_polish_count = 0
        self._polish_user_off = False   # полировку выключил сам пользователь
        self._mic_watchdog: threading.Thread | None = None
        # Живость движков для панели: считается в фоне, панель читает кэш.
        self._health: dict = {}
        self._health_thread: threading.Thread | None = None
        # Остановка воркера фраз (отдельно от _stopping).
        self._worker_stop = threading.Event()
        # Выгрузка LLM под игры: стоп сервера + сторож полного экрана.
        self._server_stopper = None
        self._game_watch: threading.Thread | None = None
        self._fullscreen_since: float | None = None
        # Ф8: ручной стиль (F7) перекрывает per-app; None = авто по окну.
        self._style_override: str | None = None
        self._last_style_switch_t = 0.0
        # Порядок цикла F7 — как в STYLE_PROMPTS (стабильный, не сеты).
        self._style_cycle = [s for s in ("plain", "casual", "formal", "terminal", "summarize")
                             if s in STYLE_PROMPTS]

    # ---- запуск / останов ----

    def start(self) -> None:
        # Проверка ASR — в ФОНЕ: для whisper это HTTP (до ~0.7 с ожидания при
        # закрытом порте) и запуск приложения ждал её. Если сервер мёртв, мы это
        # скажем, а фраза не потеряется: отказ ASR пишется в историю с аудио,
        # плюс работает резерв на GigaAM (_asr_fallback).
        threading.Thread(target=self._check_asr_background, daemon=True,
                         name="asr-check").start()
        # Проверку LLM здесь НЕ делаем: при лежащем сервере она стоит ~0.35 с и
        # задерживала запуск. Этим занимается фоновый прогрев (_prewarm_llm_work)
        # и сторож полировки; _polish_safe умеет быстро отказаться без сервера.
        pass
        if self.settings.get().flags["overlay"]:
            self.overlay.start()
        # AX поднимаем в ФОНЕ: его старт ждёт ответа PowerShell-хелпера (~0.19 с)
        # и держал запуск приложения. Пока не готов — правки ловим через буфер.
        threading.Thread(target=self._start_ax, daemon=True,
                         name="ax-start").start()
        self._warm_asr()
        # Приватность/размер: чистим историю старше retention (по умолчанию 90 дней).
        try:
            removed = self.db.prune_history(self.settings.get().history_retention_days)
            if removed:
                print(f"[myflow] история: удалено старых диктовок — {removed}")
        except Exception as exc:  # noqa: BLE001 — не мешает запуску
            print(f"[myflow] не удалось почистить историю: {exc!r}")
        self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                        name="phrase-worker")
        self._worker.start()
        # Слушатель хоткеев — в ФОНЕ: при повторной установке Win32-хука pynput
        # ждёт ~0.65 с (замер), и старт/перезапуск пульта висел. Сторож
        # (_listener_watchdog_loop) поднимет его заново, если что-то не так.
        self._start_listener_async()
        self._listener_watchdog = threading.Thread(
            target=self._listener_watchdog_loop, daemon=True, name="listener-watchdog")
        self._listener_watchdog.start()
        self._start_focus_tracker()
        self._pending_thread = threading.Thread(
            target=self._pending_worker, daemon=True, name="pending-insert")
        self._pending_thread.start()
        if (self.settings.get().insert_target or "start") == "start":
            hk = self.settings.get().hotkey
            print("[myflow] цель вставки: окно, из которого начата диктовка "
                  "(клик мышью в сторону текст не уведёт; "
                  f"{hk.lock_target_key}=закрепить окно, "
                  f"{hk.flush_pending_key}=вставить отложенное сейчас)")
            print("[myflow] если в момент вставки ты сам работаешь мышью — "
                  "вставка подождёт паузы и не отберёт управление")
        if self._pref_auto:
            self.recorder.start_continuous()
            print("[myflow] готов (AUTO: слушаю постоянно, фраза по паузе). "
                  f"F8=стоп фразы, F9=скип, {self.settings.get().hotkey.style_cycle_key}=стиль")
        else:
            polish_state = "LLM" if self.polish.cfg.enabled else "без полировки"
            print(f"[myflow] готов ({polish_state}, PTT). "
                  f"PTT={self.settings.get().hotkey.ptt_key} (держи — диктуешь), "
                  f"cancel={self.settings.get().hotkey.cancel_key}, "
                  f"style={self.settings.get().hotkey.style_cycle_key} (цикл стилей)")
        self._log_stats()

        # Автолечение LLM: следим за очередью недополированных фраз.
        self._ensure_repolish_worker()
        # Сторож микрофона: Bluetooth/USB отвалился -> переоткрываем поток.
        self._ensure_mic_watchdog()
        # Сторож игр: полный экран -> выгружаем LLM, VRAM уходит игре.
        self._ensure_game_watch()
        # Живость движков считаем в фоне: панель не должна ждать сеть.
        self._ensure_health_worker()
        # Гигиена: не даём логам расти без предела.
        rotated = rotate_all_logs()
        if rotated:
            print(f"[myflow] логи перевёрнуты: {', '.join(rotated)}")
        # И прогреваем LLM заранее: холодный старт llama-server — это ~8 секунд, и
        # если пользователь начнёт диктовать раньше, фраза уйдёт сырым ASR.
        self._prewarm_llm()

    def _mic_watchdog_loop(self, interval: float = 2.0) -> None:
        """Переоткрывать поток, если микрофон отвалился (Bluetooth/USB/дрова).

        Без этого приложение «слушало» тишину: PortAudio закрывал поток, а в
        журнале не было ни слова — фразы просто переставали появляться.
        """
        while not self._stopping.wait(interval):
            try:
                if not self._pref_auto:
                    continue          # в PTT поток открывается по нажатию
                reason = self.recorder.reopen()
                if reason:
                    status = getattr(self.recorder, "_last_status_text", "")
                    note = f" ({status})" if status else ""
                    print(f"[myflow] ⟳ микрофон переподключён: {reason}{note}")
                    self.overlay.show("⟳ микрофон переподключён", "#f5c542")
            except Exception as exc:  # noqa: BLE001 — сторож не должен падать
                print(f"[myflow] сторож микрофона: {exc!r}")

    def _ensure_mic_watchdog(self) -> None:
        if self._mic_watchdog is not None and self._mic_watchdog.is_alive():
            return
        self._mic_watchdog = threading.Thread(target=self._mic_watchdog_loop,
                                              daemon=True, name="mic-watchdog")
        self._mic_watchdog.start()

    def _health_loop(self, interval: float = 2.0) -> None:
        """Фон: обновлять живость движков, чтобы панель не ждала сеть.

        Замер: `control_snapshot` сам опрашивал LLM/ASR и панель замирала на
        0.36 с каждые 3 секунды (клавиши в это время не читались). Теперь все
        проверки — здесь, а снимок панели читает готовый результат (0.1 мс).
        """
        while not self._stopping.is_set():
            try:
                try:
                    asr_ok = bool(self.whisper.health())
                except Exception:  # noqa: BLE001
                    asr_ok = False
                llm_ok = (bool(self.polish.cfg.enabled)
                          and _health_check_llm(self.polish))
                try:
                    # Список микрофонов тоже обновляем в фоне (sd.query_devices
                    # ~10-20 мс — в цикле отрисовки это лишняя задержка).
                    from myflow.pick_mic import fast_mic_list
                    self._mic_cache = (fast_mic_list(), time.monotonic())
                except Exception:  # noqa: BLE001
                    pass
                self._health = {"asr": asr_ok, "llm": llm_ok,
                                "ts": time.monotonic()}
            except Exception:  # noqa: BLE001 — сторож не должен падать
                pass
            if self._stopping.wait(interval):
                break

    def _ensure_health_worker(self) -> None:
        if self._health_thread is not None and self._health_thread.is_alive():
            return
        self._health_thread = threading.Thread(target=self._health_loop,
                                              daemon=True, name="health")
        self._health_thread.start()

    def _prewarm_llm(self) -> None:
        """Поднять LLM сразу при запуске — НЕ блокируя старт приложения.

        Проверка живости при лежащем сервере стоит ~0.36 с (порт не отвергает,
        а «висит» до connect-таймаута), и старт приложения ждал её. Теперь
        проверка и запуск сервера идут отдельным потоком: пульт появляется
        мгновенно, а LLM догоняет в фоне.
        """
        if not self.polish.cfg.enabled:
            return
        if not getattr(self.polish.cfg, "auto_start_server", True):
            return
        threading.Thread(target=self._prewarm_llm_work, daemon=True,
                         name="llm-prewarm").start()

    def _prewarm_llm_work(self) -> None:
        try:
            if _health_check_llm(self.polish):
                print("[myflow] LLM уже отвечает — прогрев не нужен")
                return
        except Exception:  # noqa: BLE001
            pass
        # Сервера нет: честно говорим и выключаем полировку до его поднятия —
        # раньше это делал start(), но проверка блокировала запуск на ~0.35 с.
        if self.polish.cfg.enabled:
            print(f"[myflow] WARNING: llama-server не отвечает на "
                  f"{self.polish.cfg.url} — полировка отключена до поднятия "
                  f"сервера (scripts/start-llm.ps1)")
            self.polish.cfg.enabled = False
            self._ensure_polish_watchdog()
        print("[myflow] LLM не отвечает — поднимаю заранее (в фоне)")
        self._ensure_llm_async()

    def _log_stats(self) -> None:
        """Сводка надёжности по истории (аналог их телеметрии: ошибки/перцентили)."""
        try:
            st = self.db.stats()
        except Exception as exc:  # noqa: BLE001
            print(f"[myflow] статистика недоступна: {exc!r}")
            return
        if not st["dictations"]:
            return
        print(f"[myflow] история: {st['dictations']} фраз, ошибок {st['errors']} "
              f"({st['error_rate'] * 100:.2f}%), p50={st['p50_latency_ms']} мс, "
              f"p90={st['p90_latency_ms']} мс, p99={st['p99_latency_ms']} мс, "
              f"avg divergence={st['avg_divergence']}")
        if st["by_status"]:
            print("[myflow]   статусы: " + ", ".join(
                f"{k}={v}" for k, v in st["by_status"].items()))

    def _check_asr_background(self) -> None:
        """Предупредить, если ASR недоступен (не блокируя старт приложения)."""
        try:
            if not _health_check(self.whisper):
                url = getattr(getattr(self.whisper, "cfg", None), "url", "")
                print(f"[myflow] WARNING: ASR не отвечает ({url or 'локальная модель'}) — "
                      f"фразы пойдут через резерв (GigaAM) или в retry")
        except Exception as exc:  # noqa: BLE001
            print(f"[myflow] проверка ASR не удалась: {exc!r}")

    def _warm_asr(self) -> None:
        """Прогреть локальную ASR-модель в фоне (GigaAM грузится ~20-30 с).

        Whisper-серверу прогрев не нужен (он отдельный процесс), поэтому
        работаем только с клиентами, у которых есть warmup().
        """
        warmup = getattr(self.whisper, "warmup", None)
        if warmup is None:
            return

        def _run() -> None:
            # Понижаем приоритет потока: загрузка ONNX-модели держит GIL и
            # подтормаживает отрисовку панели (~0.6 с). Так интерфейс плавнее.
            _set_thread_priority_below_normal()
            try:
                if not warmup():
                    print("[myflow] локальная ASR-модель не загрузилась — "
                          "первая диктовка может не распознаться")
            except Exception as exc:  # noqa: BLE001
                print(f"[myflow] прогрев ASR: {exc!r}")
            # Заодно греем модель голоса (ONNX, ~25 МБ) — тоже в фоне.
            try:
                if self.voice.ready and self.voice.warmup():
                    print(f"[myflow] ☺ профиль голоса найден "
                          f"({self.voice.profile.count} фраз, "
                          f"порог {self.voice.threshold})")
            except Exception as exc:  # noqa: BLE001
                print(f"[myflow] модель голоса: {exc!r}")

        threading.Thread(target=_run, daemon=True, name="asr-warmup").start()

    def _start_ax(self) -> None:
        """Поднять AX-хелпер (UI Automation) для наблюдения за полем ввода."""
        if not self.settings.get().use_ax_observation:
            return
        if not self.settings.get().flags.get("ax_observation", True):
            return
        try:
            self._ax_ready = self.ax.start()
            if self._ax_ready:
                print("[myflow] наблюдение за полем ввода: UI Automation активен "
                      "(правки ловим без Ctrl+C)")
            else:
                print(f"[myflow] UI Automation недоступен ({self.ax.last_error}) — "
                      "правки ловим через буфер обмена")
        except Exception as exc:  # noqa: BLE001
            self._ax_ready = False
            print(f"[myflow] AX-хелпер не поднялся: {exc!r}")

    # ---- панель управления: живой статус и переключение на лету ----

    def _asr_label(self) -> str:
        """Короткое имя текущего ASR-движка для панели."""
        if type(self.whisper).__name__ == "GigaAmClient":
            model = str(getattr(self.whisper, "model_name", "v3"))
            short = model.replace("gigaam-", "").replace("-e2e", " e2e").replace("-ctc", "")
            return f"GigaAM {short} (CPU)"
        cfg = getattr(self.whisper, "cfg", None)
        url = getattr(cfg, "url", "")
        # Защита от нестандартных клиентов (тесты, будущие бэкенды): без проверки
        # типа url.split() падал на чужом объекте и панель теряла строку ASR.
        port = url.split("//")[-1] if isinstance(url, str) and "//" in url else ""
        return f"whisper {port}".strip() or "whisper"

    @staticmethod
    def _short_mic_name(name: str) -> str:
        """«Микрофон (Ardor Gaming Vox)» -> «Ardor Gaming Vox» (для панели)."""
        text = (name or "").strip()
        for prefix in ("Набор микрофонов (", "Микрофон (", "Микрофон "):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        return text.rstrip(") ").strip() or name

    def control_snapshot(self) -> dict:
        """Живое состояние всех узлов для панели управления.

        Возвращает {"modules": [...], "values": {...}, "counters": {...}}:
        modules — список строк-модулей (key, label, value, on, ok, note).
        """
        s = self.settings.get()
        flags = s.flags

        # Живость берём из фонового кэша (_health_loop) — НИКАКИХ сетевых
        # вызовов в цикле отрисовки: иначе панель замирала на ~0.36 с.
        health = getattr(self, "_health", None) or {}
        asr_ok = bool(health.get("asr", True))
        llm_ok = bool(health.get("llm", False))
        health_age = time.monotonic() - float(health.get("ts", 0.0))
        health_stale = health_age > 6.0

        voice = getattr(self, "voice", None)
        voice_on = bool(voice and voice.enabled)
        voice_profile = voice.profile.count if voice else 0
        vad_threshold = float(getattr(self.recorder, "vad_threshold", 0.3))

        modules = [
            {"key": "asr", "label": "ASR", "value": self._asr_label(), "on": True,
             "ok": asr_ok, "note": "проверка…" if health_stale else "распознавание речи",
             "hint": "Enter — сменить движок (спросит подтверждение): whisper "
                     "поднимает сервер :8081, GigaAM работает всегда"},
            {"key": "vad", "label": "VAD", "value": f"Silero v6 · {vad_threshold:.2f}",
             "on": bool(s.auto_stop_by_silence), "ok": True, "note": "детектор речи",
             "hint": "←→ порог, Enter — автостоп по паузе"},
            {"key": "llm", "label": "LLM",
             "value": f"{s.polish.model.split('-instruct')[0]} · :8082",
             "on": bool(s.polish.enabled), "ok": llm_ok,
             "note": "проверка…" if health_stale else "полировка текста",
             "hint": "Enter — включить/выключить полировку"},
            {"key": "voice", "label": "Голос",
             "value": f"{voice_profile} фраз · {s.voiceprint.mode} · {s.voiceprint.threshold:.2f}",
             "on": voice_on, "ok": voice_profile > 0, "note": "проверка владельца",
             "hint": "Enter вкл/выкл, ←→ режим (auto/always/off)"},
            {"key": "ax", "label": "AX",
             "value": "UI Automation" if self._ax_ready else "буфер обмена",
             "on": bool(self._ax_ready), "ok": bool(self._ax_ready),
             "note": "наблюдение за полем ввода",
             "hint": "Enter — включить/выключить UI Automation"},
            {"key": "snippets", "label": "Сниппеты", "value": "фразы → готовый текст",
             "on": bool(flags.get("snippets", True)), "ok": True, "note": "замены фраз",
             "hint": "Enter — включить/выключить"},
            {"key": "fastpath", "label": "Быстрый путь",
             "value": "не звать LLM, если готово",
             "on": bool(getattr(s.polish, "skip_wellformed", True)), "ok": True,
             "note": "экономия 100-180 мс", "hint": "Enter — включить/выключить"},
            {"key": "learning", "label": "Обучение", "value": "словарь из правок",
             "on": bool(flags.get("learning", True)), "ok": True, "note": "самообучение",
             "hint": "Enter — включить/выключить"},
            {"key": "overlay", "label": "Оверлей", "value": "плашка статуса",
             "on": bool(flags.get("overlay", False)), "ok": True, "note": "поверх окон",
             "hint": "Enter — включить/выключить"},
            {"key": "keep_audio", "label": "Аудио фраз", "value": f"последние {s.audio_keep}",
             "on": bool(flags.get("keep_audio", True)), "ok": True,
             "note": "для повторного распознавания", "hint": "Enter — включить/выключить"},
        ]

        # Ввод: микрофон/режим/вставка/стиль/цель — меняются на лету.
        mic = s.asr.input_device
        mic_label = "системный по умолчанию"
        for m in self._mic_choices():
            if m["idx"] == mic:
                api = f" [{m.get('api')}]" if m.get("api") else ""
                mic_label = f"{self._short_mic_name(m['name'])}{api}"
                break
        if mic is not None and mic_label.startswith("системный"):
            # Сканирование микрофонов могло не успеть — покажем хотя бы имя
            # устройства по индексу, чтобы панель не врала «системный».
            try:
                import sounddevice as sd
                mic_label = (f"{self._short_mic_name(sd.query_devices(mic)['name'])}"
                             f" [#{mic}]")
            except Exception:  # noqa: BLE001
                mic_label = f"#{mic}"
        mode_label = "авто (слушает)" if self._pref_auto else "PTT (F8)"
        target = self._target_locked or self._target
        target_label = ("⚲ " + target.label) if self._target_locked is not None else (
            target.label if target is not None else "активное окно")
        values = [
            {"key": "mic", "label": "Микрофон", "value": mic_label,
             "hint": "←→ сменить микрофон на лету (F5 — пересканировать)"},
            {"key": "mode", "label": "Режим диктовки", "value": mode_label,
             "hint": "←→ авто ↔ PTT"},
            {"key": "insert", "label": "Куда вставлять",
             "value": "активное окно (Ctrl+V)" if s.insert_method == "clipboard"
                      else "чат OpenCode (без фокуса)",
             "hint": "←→ сменить способ вставки"},
            {"key": "style", "label": "Стиль текста", "value": s.default_style,
             "hint": "←→ стиль по умолчанию (F7 — на лету)"},
            {"key": "target", "label": "Цель вставки", "value": target_label,
             "hint": "Enter — закрепить/снять окно (F6)"},
        ]

        try:
            self._deferred_polish_count = self.db.count_deferred_polish()
        except Exception:  # noqa: BLE001
            pass
        counters = {
            "mic_scanning": bool(getattr(self, "_mic_scanning", False)),
            "deferred_polish": int(getattr(self, "_deferred_polish_count", 0)),
            "phrases": int(getattr(self, "_phrases", 0)),
            "pending": len(getattr(self, "_pending", [])),
            "queue": self._queue.qsize() if hasattr(self, "_queue") else 0,
        }
        return {"modules": modules, "values": values, "counters": counters}

    def _mic_choices(self) -> list[dict]:
        """Список живых микрофонов для панели. НИКОГДА не блокирует отрисовку.

        Полное сканирование с пробой звука занимает ~12 секунд (17 устройств ×
        два замера). Раньше оно выполнялось прямо в `control_snapshot()`, то есть
        в цикле панели: кадр вставал на 11+ секунд, клавиши не читались
        («кнопки не работают»), а кэш был бесполезен — время бралось до скана,
        и запись сразу считалась устаревшей.

        Теперь: отдаём кэш мгновенно, а обновление запускаем в ФОНЕ.
        """
        now = time.monotonic()
        cached = getattr(self, "_mic_cache", None)
        if cached:
            mics, started = cached
            if mics or now - started < 30.0:     # пустой кэш обновляем по таймауту
                return mics
        # Кэша нет/устарел — отдаём мгновенный список без пробы и греем полный
        # в фоне. Метку времени пишем сразу, иначе один скан запускался бы на
        # каждый кадр (именно это и тормозило панель).
        fast = fast_mic_list()
        self._mic_cache = (fast, now)
        self._mic_scanning = True
        self._refresh_mics_async()
        return fast

    def _refresh_mics_async(self) -> None:
        """Полное сканирование (проба звука) в фоне; результат кладём в кэш."""
        if getattr(self, "_mic_thread", None) is not None and self._mic_thread.is_alive():
            return

        def worker() -> None:
            try:
                # Пока идёт запись, пробу не делаем: скан открывает потоки на тех
                # же устройствах и дерётся с микрофоном («занят»).
                if getattr(self.recorder, "_stream", None) is not None:
                    self._mic_cache = (fast_mic_list(), time.monotonic())
                    return
                from myflow.pick_mic import gather_mics
                current = self.settings.get().asr.input_device
                alive = [m for m in gather_mics() if m.get("alive")]
                if current is not None and not any(m["idx"] == current for m in alive):
                    # Выбранное устройство могло не пройти пробу — оставляем его
                    # в списке, чтобы можно было вернуться к нему стрелками.
                    for m in fast_mic_list():
                        if m["idx"] == current:
                            alive.append(dict(m, alive=True, status="выбран"))
                            break
                self._mic_cache = (alive or fast_mic_list(), time.monotonic())
            except Exception:  # noqa: BLE001 — панель не должна падать
                self._mic_cache = (fast_mic_list(), time.monotonic())
            finally:
                self._mic_scanning = False

        self._mic_thread = threading.Thread(target=worker, daemon=True,
                                            name="mic-scan")
        self._mic_thread.start()

    def control_apply(self, action: str, value=None) -> tuple[bool, str]:
        """Применить изменение с панели управления на лету. (ok, сообщение)."""
        s = self.settings.get()
        store = self.settings
        try:
            if action == "mic":
                idx = None if value is None else int(value)
                self.recorder.set_device(idx)
                s.asr.input_device = idx
                store.save()
                # Кэш НЕ сбрасываем: полный скан — 12 секунд, а он нам тут не нужен.
                # Выбранное устройство гарантированно окажется в списке после
                # очередного фонового обновления; переключение работает мгновенно.
                return True, f"микрофон переключён на {idx if idx is not None else 'системный'}"

            if action == "mode":
                new_mode = "auto" if value else "ptt"
                if new_mode == s.mode:
                    return True, f"режим уже {new_mode}"
                s.mode = new_mode
                store.save()
                self._pref_auto = new_mode == "auto"
                if self._pref_auto:
                    self.recorder.on_utterance = self._on_utterance
                    if self.recorder._stream is None:
                        self.recorder.start_continuous()
                    print("[myflow] режим: авто (слушаю постоянно)")
                else:
                    self.recorder.stop_continuous()
                    print("[myflow] режим: PTT (F8)")
                return True, f"режим: {'авто' if self._pref_auto else 'PTT'}"

            if action == "insert":
                method = "opencode" if value else "clipboard"
                s.insert_method = method
                store.save()
                self.paster = (OpenCodeInjector(s.opencode_auto_enter)
                               if method == "opencode" else ClipboardPaste())
                return True, f"вставка: {method}"

            if action == "style":
                s.default_style = str(value)
                self._style_override = None
                store.save()
                return True, f"стиль: {value}"

            if action == "target":
                self._toggle_target_lock()
                return True, "цель вставки переключена"

            if action == "voice_mode":
                modes = ["auto", "always", "off"]
                cur = s.voiceprint.mode if s.voiceprint.mode in modes else "auto"
                nxt = modes[(modes.index(cur) + (1 if float(value) > 0 else -1)) % len(modes)]
                s.voiceprint.mode = nxt
                store.save()
                print(f"[myflow] проверка голоса: режим {nxt}")
                return True, f"проверка голоса: {nxt}"

            if action == "rescan_mics":
                self._mic_cache = None
                self._refresh_mics_async()
                return True, "сканирую микрофоны (живость проверяется в фоне)"

            if action == "vad_threshold":
                applied = self.recorder.set_vad_threshold(float(value))
                return True, f"порог VAD: {applied:.2f}"

            if action == "module":
                key = str(value)
                return self._toggle_module(key)
        except Exception as exc:  # noqa: BLE001 — панель не должна падать
            return False, f"не удалось: {exc!r}"
        return False, f"неизвестное действие: {action}"

    def _toggle_module(self, key: str) -> tuple[bool, str]:
        """Включить/выключить модуль на лету."""
        s = self.settings.get()
        flags = s.flags

        if key == "asr":
            # Меняем движок распознавания: gigaam <-> whisper.
            # ВАЖНО: переключение должно быть РАБОТОСПОСОБНЫМ. Реальный случай:
            # пользователь нажал Enter на строке ASR, движок сменился на whisper,
            # а whisper-сервер (порт 8081) не был поднят — и все фразы падали
            # с «whisper-server недоступен». Поэтому сначала проверяем цель:
            # для whisper поднимаем сервер и ждём, для GigaAM — проверяем пакет.
            cur = getattr(self.whisper, "backend_name", "whisper")
            target = "whisper" if cur == "gigaam" else "gigaam"
            previous = s.asr.backend

            if target == "gigaam":
                from myflow.asr_gigaam import GigaAmClient
                if not GigaAmClient.package_available():
                    return False, "GigaAM недоступен (нет onnx-asr) — оставил whisper"
                client = GigaAmClient(s.asr)
                self.whisper = client
                s.asr.backend = "gigaam"
                self.settings.save()
                threading.Thread(target=client.warmup, daemon=True,
                                 name="asr-switch-warmup").start()
                # whisper-сервер больше не нужен: гасим, освобождаем ~1 ГБ VRAM.
                self._stop_server("asr")
                print(f"[myflow] ASR: {self._asr_label()}")
                return True, "ASR: GigaAM (whisper-сервер выгружен)"

            # target == "whisper": без живого сервера он бесполезен.
            if not self._ensure_server("asr"):
                s.asr.backend = previous          # откатываем настройку
                self.settings.save()
                return False, ("whisper-сервер не поднялся (проверь bin\\whisper и "
                               "модель) — оставил " + (previous or "gigaam"))
            from myflow.asr import WhisperClient
            self.whisper = WhisperClient(s.asr)
            s.asr.backend = "whisper"
            self.settings.save()
            print("[myflow] ASR: whisper (сервер поднят и проверен)")
            return True, "ASR: whisper (сервер поднят)"

        if key == "vad":
            s.auto_stop_by_silence = not s.auto_stop_by_silence
            self.settings.save()
            self.recorder.on_auto_stop = (
                self._on_auto_stop if (not self._pref_auto and s.auto_stop_by_silence) else None)
            print(f"[myflow] VAD-автостоп: {'вкл' if s.auto_stop_by_silence else 'выкл'}")
            return True, f"VAD-автостоп: {'вкл' if s.auto_stop_by_silence else 'выкл'}"

        if key == "llm":
            s.polish.enabled = not s.polish.enabled
            self._polish_user_off = not s.polish.enabled
            self.settings.save()
            # Выключил — сразу гасим сервер (VRAM освобождается), включил — поднимаем.
            if s.polish.enabled:
                self._ensure_llm_async()
            else:
                self._stop_llm_server()
            print(f"[myflow] полировка: {'вкл' if s.polish.enabled else 'выкл'}")
            return True, f"полировка: {'вкл' if s.polish.enabled else 'выкл'}"

        if key == "voice":
            s.voiceprint.enabled = not s.voiceprint.enabled
            self.settings.save()
            if self.voice is not None:
                self.voice.enabled = s.voiceprint.enabled
            print(f"[myflow] проверка голоса: {'вкл' if s.voiceprint.enabled else 'выкл'}")
            return True, f"проверка голоса: {'вкл' if s.voiceprint.enabled else 'выкл'}"

        if key == "ax":
            if self._ax_ready:
                self._ax_ready = False
                try:
                    self.ax.stop()
                except Exception:  # noqa: BLE001
                    pass
                return True, "AX выключен (правки — через буфер)"
            self._start_ax()
            return True, ("AX включён" if self._ax_ready else "AX недоступен")

        if key in ("snippets", "learning", "overlay", "keep_audio"):
            flags[key] = not flags.get(key, True)
            self.settings.save()
            if key == "overlay":
                try:
                    if flags[key]:
                        self.overlay.start()
                    else:
                        self.overlay.hide()
                except Exception:  # noqa: BLE001
                    pass
            label = {"snippets": "сниппеты", "learning": "обучение",
                     "overlay": "оверлей", "keep_audio": "запись аудио фраз"}[key]
            print(f"[myflow] {label}: {'вкл' if flags[key] else 'выкл'}")
            return True, f"{label}: {'вкл' if flags[key] else 'выкл'}"

        if key == "fastpath":
            s.polish.skip_wellformed = not getattr(s.polish, "skip_wellformed", True)
            self.settings.save()
            state = "вкл" if s.polish.skip_wellformed else "выкл"
            print(f"[myflow] быстрый путь: {state}")
            return True, f"быстрый путь: {state}"

        return False, f"неизвестный модуль: {key}"

    def stop(self) -> None:
        self._stopping.set()
        # Гасим воркер рекордера: иначе он живёт после stop()
        self._worker_stop.set()
        try:
            self.recorder.shutdown()
        except Exception:  # noqa: BLE001
            pass      # гасим фоновые петли (в т.ч. сэмплер фокуса)
        if self._listener:
            self._listener.stop()
        if self._pref_auto:
            self.recorder.stop_continuous()
        else:
            try:
                self.recorder.stop()
            except RuntimeError:
                pass
        try:
            self.ax.stop()
        except Exception:  # noqa: BLE001
            pass
        self.overlay.stop()

    # ---- хоткеи ----
    #
    # ВАЖНО: тело колбэков обёрнуто в try/except. Pynput при исключении в
    # колбэке делает self.stop() (pynput/_util/__init__.py) — слушатель
    # умирал НАВСЕГДА, и F7/F8/F9 молча переставали работать до перезапуска.

    def _on_press(self, key) -> None:
        try:
            self._handle_press(key)
        except Exception as exc:  # noqa: BLE001 — хоткей не должен умирать
            _tb_to_file("ОШИБКА ХОТКЕЯ (press)")
            print(f"[myflow] ✗ хоткей press: {exc!r}")
            self.overlay.show("✗ хоткей", "#e5484d")

    def _on_release(self, key) -> None:
        try:
            self._handle_release(key)
        except Exception as exc:  # noqa: BLE001
            _tb_to_file("ОШИБКА ХОТКЕЯ (release)")
            print(f"[myflow] ✗ хоткей release: {exc!r}")

    def _handle_press(self, key) -> None:
        hk = self.settings.get().hotkey
        if _matches(key, hk.style_cycle_key):
            self._cycle_style()
            return
        if _matches(key, getattr(hk, "lock_target_key", "f6")):
            self._toggle_target_lock()
            return
        if _matches(key, getattr(hk, "flush_pending_key", "f10")):
            self._flush_pending()
            return
        if _matches(key, getattr(hk, "unload_llm_key", "f11")):
            self._toggle_llm_unload()
            return
        if _matches(key, hk.cancel_key):
            self._cancel_all()
            return
        if not _matches(key, hk.ptt_key):
            return
        if self._pref_auto:
            # Срочный стоп текущей фразы (без ожидания паузы).
            if self.recorder.force_finish():
                print("[myflow] ○ ручной стоп фразы")
            return
        if self._key_down:
            return  # автоповтор WM_KEYDOWN — уже обработали это нажатие
        self._key_down = True
        if self._ptt_held:
            return
        self._ptt_held = True
        # Запись можно начинать ВСЕГДА: очередь разберётся. Раньше при
        # self._busy нажатие F8 молча игнорировалось, и начало речи терялось.
        if self._busy:
            print("[myflow]   (предыдущая фраза ещё обрабатывается — пишу в очередь)")
        self._begin_recording()

    def _handle_release(self, key) -> None:
        if self._pref_auto:
            return  # в AUTO режим кнопка не управляет записью
        if not _matches(key, self.settings.get().hotkey.ptt_key):
            return
        self._key_down = False
        if not self._ptt_held:
            return
        self._ptt_held = False
        self._finish_recording()

    def _start_listener(self) -> None:
        hk = self.settings.get().hotkey
        # suppress=True (по желанию): наши F-клавиши не улетают в активное окно
        # (в IDE F8 = шаг отладки). По умолчанию False — как было.
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release,
            suppress=bool(getattr(hk, "suppress", False)),
        )
        self._listener.start()

    def _start_listener_async(self) -> None:
        """Поднять слушатель хоткеев, не блокируя старт приложения."""
        threading.Thread(target=self._start_listener_safe, daemon=True,
                         name="listener-start").start()

    def _start_listener_safe(self) -> None:
        try:
            self._start_listener()
        except Exception as exc:  # noqa: BLE001
            print(f"[myflow] слушатель хоткеев не поднялся: {exc!r} "
                  f"(сторож попробует снова)")

    def _listener_watchdog_loop(self, interval: float = 2.0) -> None:
        """Страховка: если слушатель хоткеев умер — поднять заново.

        Цикл завершается по stop(): иначе при перезапуске пульта (клавиша r)
        поток оставался жить и копился (замер: 4 потока на каждый перезапуск).
        """
        while not self._stopping.wait(interval):
            try:
                if self._stopping.is_set():
                    break
                if self._listener is None or not self._listener.is_alive():
                    print("[myflow] ! слушатель хоткеев умер — перезапускаю")
                    self._start_listener()
                    self.overlay.show("↻ хоткеи восстановлены", "#46a758")
            except Exception:  # noqa: BLE001
                pass

    # ---- отмена (F9 / голосовые «стоп»/«скип») ----

    def _cancel_all(self) -> None:
        """Отменить текущую запись И всё, что ещё не вставлено.

        Поколение отмены инкрементится — воркер увидит старый gen и не
        вставит ни уже обрабатываемую фразу, ни элементы очереди.
        """
        self._gen += 1
        self._key_down = False
        self._ptt_held = False
        with self._preview_lock:
            self._preview_slot = None   # отменённый драфт не показываем
        if self._pref_auto:
            try:
                self.recorder.discard()
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                self.recorder.stop()
            except RuntimeError:
                pass
        print("[myflow] ○ отменено")
        self.overlay.show("○ отменено", "#9aa0a6")

    # ---- наблюдение за LLM (полировка) ----

    def _polish_watchdog_loop(self, interval: float = 30.0) -> None:
        """Если полировка выключена (сервер лежал) — вернуть её, когда поднимется."""
        while not self._stopping.wait(interval):
            try:
                if self.polish.cfg.enabled:
                    continue
                if _health_check_llm(self.polish):
                    self.polish.cfg.enabled = True
                    self._polish_fails = 0
                    print("[myflow] LLM снова доступна — полировка включена")
                    self.overlay.show("✓ полировка включена", "#46a758")
            except Exception:  # noqa: BLE001
                pass

    def _ensure_polish_watchdog(self) -> None:
        if self._polish_watchdog is not None and self._polish_watchdog.is_alive():
            return
        self._polish_watchdog = threading.Thread(
            target=self._polish_watchdog_loop, daemon=True, name="polish-watchdog")
        self._polish_watchdog.start()

    def _polish_safe(self, text: str, style: str, dic: list[dict]) -> PolishResult:
        """Полировка, которая НИКОГДА не теряет фразу.

        Любая ошибка LLM (таймаут, 400, битый JSON, мёртвый сервер) -> сырой
        ASR. Раньше исключение улетало в общий handler воркера: ни вставки,
        ни записи в историю (реальный лог: 4 потерянные фразы).
        """
        if not self.polish.cfg.enabled:
            return PolishResult(text=text, latency_ms=0, skipped=True)
        # Быстрый отказ: если LLM заведомо лежит (или её только что не стало),
        # не тратим на неё секунды — сразу сырой ASR, автолечение в фоне и
        # очередь на дополировку.
        if not _health_check_llm(self.polish):
            self._polish_fails += 1
            print("[myflow] ! LLM не отвечает — вставлю сырой ASR "
                  "(поднимаю сервер в фоне, фраза в очереди на дополировку)")
            # Ждать здесь НЕЛЬЗЯ: холодный старт llama-server ~8 с, диктовка
            # встала бы. Поднимаем в фоне, а фразу дополирует очередь.
            self._ensure_llm_async()
            if self._polish_fails >= 3:
                self.polish.cfg.enabled = False
                print("[myflow] ! LLM недоступна 3 раза подряд — полировка выключена "
                      "до восстановления сервера (проверяю раз в 30с)")
                self._ensure_polish_watchdog()
            return PolishResult(text=text, latency_ms=0, skipped=True)
        try:
            result = self.polish.polish(text, style=style, dictionary=dic)
            self._polish_fails = 0
            return result
        except Exception as exc:  # noqa: BLE001 — это защитный слой
            self._polish_fails += 1
            _tb_to_file("ОШИБКА ПОЛИРОВКИ")
            print(f"[myflow] ! полировка не удалась ({exc})")
            # АВТОЛЕЧЕНИЕ: поднимаем сервер и ждём его до wait_on_fail_seconds,
            # затем пробуем ещё раз — иначе фраза уйдёт сырым ASR из-за того,
            # что llama-server просто не успел стартовать.
            if self._revive_llm_and_retry(text, style, dic):
                # ВАЖНО: счётчик ошибок сбрасываем только при успехе повтора —
                # иначе фраза попадёт в историю как polish_error, хотя полировка
                # на самом деле удалась (это поймал тест).
                try:
                    result = self.polish.polish(text, style=style, dictionary=dic)
                    self._polish_fails = 0
                    return result
                except Exception as exc2:  # noqa: BLE001
                    print(f"[myflow] ! повторная полировка не удалась ({exc2})")
            if self._polish_fails >= 3:
                self.polish.cfg.enabled = False
                print("[myflow] ! LLM недоступна 3 раза подряд — полировка выключена "
                      "до восстановления сервера (проверяю раз в 30с)")
                self._ensure_polish_watchdog()
            print("[myflow]   вставлю сырой ASR; фраза в очереди на дополировку")
            return PolishResult(text=text, latency_ms=0, skipped=True)

    # ---- автолечение LLM ----

    def set_server_stopper(self, stopper) -> None:
        """Подключить «останови сервер» (тоже даёт TUI/Supervisor).

        Нужно для выгрузки LLM: гасим llama-server и освобождаем VRAM, а не
        просто помечаем полировку выключенной.
        """
        self._server_stopper = stopper

    def _ensure_server(self, key: str, timeout: float = 90.0) -> bool:
        """Синхронно поднять сервер (для осознанного действия пользователя).

        Обычная диктовка серверы не ждёт, но переключение движка — действие
        явное: лучше подождать пару секунд, чем получить нерабочий ASR.
        """
        guard = self._server_guard
        if guard is None:
            return False
        try:
            return bool(guard(key))
        except Exception as exc:  # noqa: BLE001
            print(f"[myflow] не удалось поднять {key}: {exc!r}")
            return False

    def _stop_server(self, key: str) -> None:
        """Погасить сервер в фоне (освободить VRAM/память)."""
        stopper = self._server_stopper
        if stopper is None:
            return
        threading.Thread(target=lambda: self._safe_stop(stopper, key),
                         daemon=True, name=f"stop-{key}").start()

    def _stop_llm_server(self) -> None:
        """Погасить LLM в фоне (не блокируя диктовку и UI)."""
        stopper = self._server_stopper
        if stopper is None:
            return
        threading.Thread(target=lambda: self._safe_stop(stopper),
                         daemon=True, name="llm-unload").start()

    @staticmethod
    def _safe_stop(stopper, key: str = "llm") -> None:
        try:
            stopper(key)
        except Exception:  # noqa: BLE001 — выгрузка не должна ломать приложение
            pass

    def set_server_guard(self, guard) -> None:
        """Подключить «подниматель» серверов (его даёт TUI/Supervisor).

        guard(key) -> bool: True, если сервер уже поднят или поднялся.
        Приложение не знает, как запускать llama-server, поэтому спрашивает.
        """
        self._server_guard = guard

    def _ensure_llm_async(self) -> None:
        """Поднять LLM в фоне (не блокируя вставку фразы)."""
        guard = self._server_guard
        cfg = self.polish.cfg
        # getattr — чтобы старые конфиги (и тестовые заглушки) не падали.
        if not getattr(cfg, "auto_start_server", True) or guard is None:
            return
        if self._llm_revive is not None and self._llm_revive.is_alive():
            return

        def worker() -> None:
            try:
                started = bool(guard("llm"))
                if started:
                    print("[myflow] ▲ LLM поднята автоматически")
            except Exception:  # noqa: BLE001 — автолечение не должно ломать диктовку
                pass

        self._llm_revive = threading.Thread(target=worker, daemon=True,
                                            name="llm-revive")
        self._llm_revive.start()

    def _wait_llm_ready(self, seconds: float) -> bool:
        """Ждать живости LLM до seconds (шаг 0.25 с). True — дождались."""
        if seconds <= 0:
            return False
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stopping.is_set():
                return False
            if _health_check_llm(self.polish):
                return True
            time.sleep(0.25)
        return False

    def _revive_llm_and_retry(self, text: str, style: str, dic: list[dict]) -> bool:
        """Поднять LLM и дождаться её. True — можно пробовать полировку снова."""
        cfg = self.polish.cfg
        if not getattr(cfg, "auto_start_server", True):
            return False
        # Сервер мог просто перезапускаться — короткое ожидание без запуска.
        if _health_check_llm(self.polish):
            return True
        self._ensure_llm_async()
        waited = self._wait_llm_ready(float(getattr(cfg, "wait_on_fail_seconds", 2.5)))
        if waited:
            print(f"[myflow]   LLM поднялась — повторяю полировку "
                  f"(ждали до {getattr(cfg, 'wait_on_fail_seconds', 2.5):.1f} с)")
        return waited

    def _repolish_loop(self) -> None:
        """Фон: когда LLM ожила — дополировать фразы из очереди (status=polish_error)."""
        cfg0 = self.polish.cfg
        interval = max(5.0, float(getattr(cfg0, "auto_repolish_interval", 15.0)))
        while not self._stopping.wait(interval):
            try:
                if not getattr(self.polish.cfg, "auto_repolish", True):
                    continue
                pending = self.db.count_deferred_polish()
                self._deferred_polish_count = pending
                if not pending:
                    continue
                if not self.polish.cfg.enabled:
                    # Сами включаем полировку только если её выключил сторож
                    # (сервер лежал). Если пользователь выключил её руками —
                    # не поднимаем сервер: он мог освобождать VRAM под игру.
                    if getattr(self, "_polish_user_off", False):
                        continue
                    self.polish.cfg.enabled = True     # дадим шанс после автолечения
                if not _health_check_llm(self.polish):
                    self._ensure_llm_async()
                    continue
                done = 0
                for row in self.db.deferred_polish(limit=5):
                    try:
                        res = self.polish.polish(row["asr_text"],
                                                 style=row.get("style") or "plain",
                                                 dictionary=self.db.get_dictionary(
                                                     row.get("app_window")))
                    except Exception as exc:  # noqa: BLE001
                        print(f"[myflow] ! дополировка #{row['id']} не удалась: {exc!r}")
                        break
                    score = divergence_score(row["asr_text"], res.text)
                    self.db.set_polished(row["id"], res.text, score)
                    self.db.mark_status(row["id"], "ok")
                    done += 1
                    print(f"[myflow] ✓ дополировано #{row['id']}: {res.text[:60]}")
                left = self.db.count_deferred_polish()
                self._deferred_polish_count = left
                if done:
                    print(f"[myflow] дополировано фраз: {done}, осталось {left}")
                    self.overlay.show(f"✓ дополировано {done}", "#46a758")
            except Exception:  # noqa: BLE001 — фон не должен падать
                pass

    def _ensure_repolish_worker(self) -> None:
        if self._repolish_thread is not None and self._repolish_thread.is_alive():
            return
        try:
            self._deferred_polish_count = self.db.count_deferred_polish()
        except Exception:  # noqa: BLE001
            self._deferred_polish_count = 0
        if self._deferred_polish_count:
            print(f"[myflow] ! фраз в очереди на дополировку: "
                  f"{self._deferred_polish_count} (LLM оживёт — дополирую)")
        self._repolish_thread = threading.Thread(target=self._repolish_loop, daemon=True,
                                                name="repolish")
        self._repolish_thread.start()


    # ---- жизненный цикл записи (PTT) ----

    def _begin_recording(self) -> None:
        # Цель вставки — окно, из которого пользователь начал диктовать.
        # Если он потом случайно кликнет в сторону, текст всё равно уйдёт сюда.
        self._target = self._target_locked or current_window()
        self.recorder.start()
        self._auto_stopped = False
        print(f"[myflow] ● запись начата (окно: {app_descriptor()})")
        if self._target is not None:
            print(f"[myflow]   цель вставки: {self._target.label}")
        self.overlay.show("● запись…", "#f5c542")

    def _cancel_recording(self) -> None:
        try:
            self.recorder.stop()
        except RuntimeError:
            pass
        print("[myflow] ○ запись отменена")
        self.overlay.show("○ отменено", "#9aa0a6")

    def _on_auto_stop(self) -> None:
        """PTT+VAD: тишина дольше паузы завершает запись как отпускание F8."""
        if self._auto_stopped:
            return
        self._auto_stopped = True
        self._ptt_held = False  # отпускание не вызовет повторный finish
        print("[myflow] ○ пауза — автостоп")
        try:
            self._finish_recording()
        except Exception as exc:  # авто-стоп не должен валить main
            print(f"[myflow] авто-стоп: {exc}")

    def _finish_recording(self) -> None:
        try:
            recording = self.recorder.stop()
        except RuntimeError:
            # Уже остановлена (гонка автостопа и отпускания F8) — не ошибка.
            return
        if recording.duration_ms < 200:
            print("[myflow] ○ слишком короткая запись, пропуск")
            self.overlay.show("○ слишком коротко", "#9aa0a6")
            return
        if recording.max_amplitude < 0.01:
            print("[myflow] ○ тишина (нет сигнала с микрофона), пропуск")
            self.overlay.show("○ тишина", "#9aa0a6")
            return
        self._enqueue(recording)

    # ---- AUTO: фраза от рекордера / live-превью ----

    def _enqueue(self, item: Recording | _LivePreview) -> None:
        """Поставить элемент в очередь с текущим поколением отмены."""
        self._queue.put(_Job(gen=self._gen, item=item))

    def _on_utterance(self, rec: Recording) -> None:
        """Вызывается из отдельного потока (recorder). Очередь не блокирует."""
        # В AUTO фраза начинается сама (VAD), поэтому окно-цель ищем в истории
        # фокуса: оно было активным в момент старта этой фразы.
        if self._target_locked is not None:
            self._target = self._target_locked
        elif (self.settings.get().insert_target or "start") == "start":
            started_at = time.monotonic() - max(0.0, rec.duration_ms / 1000.0)
            self._target = self._focus_history.at(started_at) or current_window()
        else:
            self._target = None       # insert_target="focus": решаем в момент вставки
        self._enqueue(rec)

    def _start_focus_tracker(self) -> None:
        """Сэмплер фокуса для AUTO-режима (нужен только при insert_target='start').

        Замеры раз в 0.4 с: по ним находим окно, активное в момент начала фразы.
        Отдельный поток, ничего не блокирует и почти не стоит по CPU.
        """
        if (self.settings.get().insert_target or "start") != "start":
            return

        def loop() -> None:
            while not self._stopping.is_set():
                self._focus_history.record()
                self._focus_history.prune()
                time.sleep(0.4)

        self._focus_thread = threading.Thread(target=loop, daemon=True,
                                              name="focus-tracker")
        self._focus_thread.start()

    def _on_live_preview(self, pcm_int16) -> None:
        """Инкрементальный ASR: кладём драфт в единственный слот.

        Свежий драфт вытесняет предыдущий (устаревшие кадры бесполезны), а
        реальные фразы идут через основную очередь и всегда обрабатываются
        первыми. Раньше превью лежали в общей FIFO и задерживали вставку.
        """
        with self._preview_lock:
            self._preview_slot = _Job(gen=self._gen, item=_LivePreview(pcm_int16=pcm_int16))

    def _take_preview(self) -> _Job | None:
        with self._preview_lock:
            slot, self._preview_slot = self._preview_slot, None
        return slot

    # ---- worker: последовательный пайплайн ----

    def _worker_loop(self) -> None:
        # Отдельное событие: _stopping взводится ещё и тестами/встраиванием, а
        # воркер должен жить до настоящего stop() приложения.
        while not self._worker_stop.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                # Очередь фраз пуста — можно показать свежий драфт.
                preview = self._take_preview()
                if preview is not None and preview.gen == self._gen:
                    try:
                        self._process_preview(preview.item)
                    except Exception as exc:  # noqa: BLE001 — драфт не критичен
                        print(f"[myflow] preview: {exc!r}")
                continue
            self._busy = True
            try:
                if isinstance(job.item, _LivePreview):
                    if job.gen == self._gen:
                        self._process_preview(job.item)
                else:
                    self._process_recording(job.gen, job.item)
            except Exception as exc:  # фраза не должна валить worker
                _tb_to_file("ОШИБКА фразы")
                print(f"[myflow] ✗ ОШИБКА фразы: {exc}")
                self.overlay.show("✗ ошибка", "#e5484d")
            finally:
                self._busy = False
                self._queue.task_done()

    def _process_preview(self, preview: _LivePreview) -> None:
        """Драфт-распознавание живого хвоста: только печать, без вставки/БД."""
        try:
            asr = self.whisper.transcribe(preview.pcm_int16)
        except RuntimeError:
            return  # сервер занят/упал — драфт не критичен
        except Exception as exc_pv:
            _tb_to_file("preview fail")
            print(f"[dbg] preview fail: {exc_pv!r}")
        if asr.text:
            print(f"[myflow]   … {asr.text}")
            self.overlay.show(f"… {asr.text[:40]}", "#f5c542")

    # ---- голосовые команды ----

    def _dispatch_voice_command(self, text: str) -> VoiceCommand | None:
        """Определить и выполнить голосовую команду. None — обычный текст.

        Порядок: точные триггеры > префиксы > LLM-интент (если есть маркер
        и LLM доступна). Ложные срабатывания минимизированы сознательно.

        ИНВАРИАНТ: возвращаем только известные команды. Всё остальное
        (в т.ч. kind="text" от LLM) — обычная диктовка, текст не теряется.
        """
        cmd = cmd_classify(text)
        if cmd is None and is_llm_intent_candidate(text) and self.polish.cfg.enabled:
            try:
                cmd = llm_classify(text, self.polish.cfg.url, self.polish._effective_model())
            except Exception:
                cmd = None
        if cmd is None or cmd.kind not in KNOWN_KINDS:
            return None
        try:
            self._run_command(cmd)
        except Exception as exc:
            print(f"[myflow] ✗ команда «{cmd.kind}»: {exc}")
            self.overlay.show("✗ команда", "#e5484d")
            return cmd
        return cmd

    def _run_command(self, cmd: VoiceCommand) -> None:
        if cmd.kind == "stop" or cmd.kind == "skip":
            # Ничего из очереди не вставляем (поколение отмены) + стоп записи.
            self._gen += 1
            if not self._pref_auto:
                self._ptt_held = False
                try:
                    self.recorder.stop()
                except RuntimeError:
                    pass
            self.overlay.show("○ пропуск", "#9aa0a6")
            return
        if cmd.kind == "new_paragraph":
            self._paste_to_target("\n")
            self.overlay.show("¶ абзац", "#46a758")
            return
        if cmd.kind == "delete_last":
            if time.monotonic() - self._last_insert_ts > 60:
                print("[myflow]   delete_last: последняя вставка >60с назад — игнор")
                self.overlay.show("нечего удалять", "#9aa0a6")
                return
            self._press_key("BACK")  # Ctrl+Z
            self.overlay.show("⎌ отменено", "#46a758")
            return
        if cmd.kind == "send":
            # Голый «отправь» = Enter к последней вставке. Раньше здесь была
            # повторная вставка текста (rephrase_last=True) — уходил дубль.
            self._press_key("ENTER")
            self.overlay.show("✓ отправлено", "#46a758")
            return
        if cmd.kind in ("summarize", "list"):
            if cmd.rephrase_last:
                text = self._require_last_insert(cmd)
            else:
                text = cmd.text
            if not text.strip():
                print(f"[myflow]   {cmd.kind}: пустой аргумент — нечего переоформлять")
                self.overlay.show("нечего переоформлять", "#9aa0a6")
                return
            polished = self._polish_safe(text, cmd.kind, [])
            final = polished.text if not polished.skipped else text
            self._paste_to_target(final)
            self.overlay.show(f"✓ {cmd.kind}", "#46a758")
            return

    def _require_last_insert(self, cmd: VoiceCommand) -> str:
        """Текст последней вставленной фразы (для переоформления без аргумента)."""
        if self._last_history_id is None:
            raise RuntimeError("нет последней диктовки")
        if time.monotonic() - self._last_insert_ts > 300:
            print("[myflow]   последняя вставка >5 мин назад — команда без аргумента")
        text = self.db.get_polished_text(self._last_history_id)
        if not text:
            raise RuntimeError(f"последняя диктовка {self._last_history_id} не найдена")
        return text

    def _press_key(self, what: str) -> None:
        """Нажать клавишу. Для Ctrl+Z используем VK-код (не раскладку!).

        pynput.press("z") на русской раскладке даёт VkKeyScan -> -1 и уходит
        в UNICODE-инъекцию: Ctrl+Z не срабатывает, а оверлей рапортует успех.
        """
        from pynput.keyboard import Key, KeyCode, Controller
        kb = Controller()
        if what == "ENTER":
            kb.press(Key.enter)
            kb.release(Key.enter)
        elif what == "BACK":
            vk_z = KeyCode.from_vk(0x5A)  # VK_Z — раскладко-независимо
            with kb.pressed(Key.ctrl):
                kb.press(vk_z)
                kb.release(vk_z)
        elif what == "HOME":
            kb.press(Key.home)
            kb.release(Key.home)

    def _process_recording(self, gen: int, recording: Recording) -> None:
        if gen != self._gen:
            print("[myflow] ○ фраза отменена до обработки")
            return
        started = time.perf_counter()
        if recording.duration_ms < 200:
            print("[myflow] ○ слишком короткая фраза, пропуск")
            self.overlay.show("○ слишком коротко", "#9aa0a6")
            return
        if recording.max_amplitude < 0.01:
            print("[myflow] ○ тишина (нет сигнала), пропуск")
            self.overlay.show("○ тишина", "#9aa0a6")
            return

        flags = self.settings.get().flags
        app = app_descriptor()
        app_category = self._resolve_category(app)

        # Аудио фразы сохраняем ДО распознавания: если ASR/LLM упадут,
        # фразу можно будет перераспознать (myflow.cli retry).
        audio_path = None
        if flags.get("keep_audio", True):
            audio_path = self._save_audio(recording)

        # Проверка голоса ДО распознавания: если фразу сказал не владелец
        # (телевизор, коллега, видео) — не тратим ASR/LLM и ничего не вставляем.
        # В PTT-режиме по умолчанию не проверяем: удержание F8 — явное намерение.
        voice_cfg = self.settings.get().voiceprint
        voice_active = (voice_cfg.mode == "always"
                        or (voice_cfg.mode == "auto" and self._pref_auto))
        voice_ok, voice_score, voice_reason = (True, None, "проверка не в этом режиме")
        if voice_active:
            voice_ok, voice_score, voice_reason = self.voice.check(recording.pcm_int16)
        if not voice_ok:
            e2e_ms = int((time.perf_counter() - started) * 1000)
            print(f"[myflow] ✗ {voice_reason} — фраза пропущена "
                  f"({recording.duration_ms} мс, {e2e_ms} мс)")
            self.overlay.show("✗ не ваш голос", "#e5484d")
            self.db.insert_history(DictationRecord(
                asr_text="", polished_text="", app_window=app,
                duration_ms=recording.duration_ms, e2e_latency_ms=e2e_ms,
                style=self._resolve_style(app), status="other_speaker",
                app_category=app_category,
                audio_path=str(audio_path) if audio_path else None,
            ))
            return
        if voice_score is not None:
            print(f"[myflow] ☺ голос владельца (cosine={voice_score:.3f})")

        print(f"[myflow] ⚙ распознавание ({recording.duration_ms} мс, {app_category})...")
        self.overlay.show("⚙ распознавание…", "#f5c542")
        self._refresh_asr_prompt()   # словарь -> подсказка whisper (раз в минуту)

        try:
            asr: AsrResult = self.whisper.transcribe(recording.pcm_int16)
        except Exception as exc_asr:
            # Отказ ASR не должен терять фразу: если это whisper — падаем на
            # локальную GigaAM и повторяем распознавание ТОГО ЖЕ аудио.
            asr = self._asr_fallback(recording, exc_asr)
            if asr is None:
                # Раньше ошибка ASR просто теряла фразу: в истории её не было,
                # а в статистике — ни следа. Теперь это строка со статусом.
                _tb_to_file("ОШИБКА ASR")
                print(f"[myflow] ✗ ASR не справился: {exc_asr}")
                self.overlay.show("✗ ASR ошибка", "#e5484d")
                self._record_failure("asr_error", app, app_category, recording,
                                     audio_path)
                return
        if not asr.text:
            print("[myflow] ✗ пустой результат распознавания")
            self.overlay.show("✗ не распознано", "#e5484d")
            self._record_failure("empty_asr", app, app_category, recording, audio_path,
                                 asr_latency_ms=asr.latency_ms)
            return

        # СНИППЕТЫ — ДО команд и полировки: явная пользовательская фраза
        # («мой email», «организуй мысли») должна подставить готовый текст.
        if flags.get("snippets", True):
            snippet = self._try_snippet(asr.text, app, app_category, recording,
                                        audio_path, asr, gen)
            if snippet:
                return

        # Голосовые команды (после распознавания, до полировки): если это
        # команда — обрабатываем и выходим (никакой вставки/БД).
        cmd = self._dispatch_voice_command(asr.text)
        if cmd:
            print(f"[myflow]   ⌘ команда: {cmd.kind}"
                  + (f" <{cmd.text[:40]}>" if cmd.text else "")
                  + f" ({cmd.source})")
            return

        dic = self.db.get_dictionary(app) if flags["learning"] else []
        style = self._resolve_style(app)

        # Быстрый путь: GigaAM e2e уже отдаёт текст с пунктуацией и заглавными,
        # и если он оформлен (нет филлеров и чисел словами) — LLM не нужен.
        # Экономит 110-180 мс на фразе (полировка дороже распознавания).
        fast_ok, fast_reason = False, ""
        cfg_polish = self.settings.get().polish
        backend_name = getattr(self.whisper, "backend_name", "")
        if self.polish.cfg.enabled and getattr(cfg_polish, "skip_wellformed", True):
            fast_ok, fast_reason = fast_path_allowed(backend_name, style, asr.text)
        # Лёгкая чистка БЕЗ LLM: филлеры/«...»/двойная пунктуация/заглавная.
        # Если после неё текст стал оформленным — полировку не зовём вовсе
        # (замер: такие фразы стоят ~280 мс LLM, а чистка — микросекунды).
        local_rules: list[str] = []
        fast_text = asr.text
        if (not fast_ok and self.polish.cfg.enabled
                and getattr(cfg_polish, "light_clean", True)
                and "gigaam" in (backend_name or "").lower()
                and style in ("plain", "casual")):
            cleaned, local_rules = light_clean(asr.text)
            # ВАЖНО: без конечной пунктуации фраза часто ОБОРВАНА («Наш туи
            # тормозит. То») — там нужен LLM (он выбрасывает хвост), а не
            # локальная точка. Поэтому локальный путь только для текста, у
            # которого пунктуация в конце уже была.
            has_terminal = asr.text.rstrip()[-1:] in ".!?…"
            if local_rules and has_terminal and looks_wellformed(cleaned)[0]:
                fast_text = cleaned
                fast_ok = True
                fast_reason = "лёгкая чистка: " + ", ".join(local_rules)
            else:
                local_rules = []
        if fast_ok:
            final_text = fast_text
            divergence = divergence_score(asr.text, final_text) if local_rules else 0.0
            polish_failed = False
            record_status = "local_polish" if local_rules else "fast_path"
        else:
            if self.polish.cfg.enabled:
                self.overlay.show("⚙ полировка…", "#f5c542")
            # _polish_safe: любая ошибка LLM -> сырой ASR, фраза не теряется.
            fails_before = self._polish_fails
            polished: PolishResult = self._polish_safe(asr.text, style, dic)
            polish_failed = self._polish_fails > fails_before
            if polished.skipped or polished.truncated:
                final_text = asr.text
                if polished.truncated:
                    print("[myflow]   WARNING: ответ LLM обрезан (max_tokens) — вставлен сырой ASR")
            else:
                final_text = self._guard_polish(asr.text, polished.text, style)

            # Детерминированная зачистка филлеров ВНУТРИ фразы: локальная 4B-модель
            # их часто оставляет («Второе, значит, э-э, насчёт»), хотя облако
            # убирает — проверено стендом на реальных диктовках Wispr.
            if style in ("plain", "casual", "formal") and final_text:
                cleaned_final, inner_rules = strip_inner_fillers(final_text)
                if inner_rules and cleaned_final:
                    print(f"[myflow]   чистка: {', '.join(inner_rules)}")
                    final_text = cleaned_final

            # Метрика расхождения ASR <-> LLM (аналог formattingDivergenceScore).
            divergence = divergence_score(asr.text, final_text)
            record_status = "polish_error" if polish_failed else "ok"

        e2e_ms = int((time.perf_counter() - started) * 1000)

        # Видимость полировки: показываем исходный ASR и результат LLM,
        # чтобы было понятно, что именно изменила модель.
        if fast_ok and local_rules:
            print(f"[myflow]   LLM: лёгкая чистка без модели ({', '.join(local_rules)})")
            print(f"[myflow]   ASR : {asr.text}")
            print(f"[myflow]   было→стало: {final_text}")
        elif fast_ok:
            print(f"[myflow]   LLM: быстрый путь ({fast_reason}) — полировка не нужна")
        elif polished.skipped:
            print("[myflow]   LLM: выключена — вставлен сырой ASR")
        elif final_text == asr.text:
            print(f"[myflow]   LLM: текст не изменён (divergence={divergence:.2f})")
        else:
            print(f"[myflow]   ASR : {asr.text}")
            print(f"[myflow]   LLM : {final_text}  (style={style}, "
                  f"divergence={divergence:.2f}, polish={polished.latency_ms} мс)")

        history_id = self.db.insert_history(
            DictationRecord(
                asr_text=asr.text,
                polished_text=final_text,
                app_window=app,
                duration_ms=recording.duration_ms,
                e2e_latency_ms=e2e_ms,
                detected_language=asr.language,
                style=style,
                status=record_status,
                app_category=app_category,
                divergence=divergence,
                audio_path=str(audio_path) if audio_path else None,
            )
        )
        self._last_history_id = history_id

        print(f"[myflow] ✓ {final_text}")
        # Профиль голоса пополняем своими же удачными фразами: чем больше
        # записей, тем устойчивее центроид (микрофон/комната меняются).
        if self.voice.maybe_auto_enroll(recording.pcm_int16, accepted=True,
                                        score=voice_score):
            print(f"[myflow] ☺ профиль голоса обновлён "
                  f"({self.voice.profile.count} фраз)")
        # Отмена могла прийти, пока шли ASR и полировка: текст уже в истории,
        # но вставлять его не нужно (иначе F9 «не отменяет»).
        if gen != self._gen:
            print("[myflow] ○ отменено во время обработки — вставлять не буду (текст в истории)")
            self.overlay.show("○ отменено", "#9aa0a6")
            return
        if flags["auto_insert"]:
            try:
                ok = self._paste_to_target(final_text, history_id=history_id)
                # Запоминаем процесс/окно, в которое вставили: наблюдатель
                # правок должен убедиться, что читает то же самое окно.
                # (При отложенной вставке это сделает сама доставка.)
                if ok:
                    self._remember_insert_target()
                if isinstance(self.paster, OpenCodeInjector) and not ok:
                    print("[myflow] ✗ инжект не прошёл: opencode не найден или консоль недоступна")
                    self.overlay.show("✗ инжект не прошёл", "#e5484d")
                elif not ok:
                    # Вставка отложена (пользователь занят) — оверлей уже сказал
                    # «вставлю на паузе», не перекрываем его ложным «вставлено».
                    print("[myflow] ⏳ фраза отложена до паузы (F10 — вставить сразу)")
                else:
                    self.overlay.show("✓ вставлено", "#46a758")
                # Обучение запускает сама вставка (в т.ч. отложенная).
            except Exception as exc_in:
                # Любое падение на вставке — отдельным маркером, НЕ валит worker
                _tb_to_file("ОШИБКА ВСТАВКИ")
                print(f"[myflow] ✗ ОШИБКА ВСТАВКИ: {exc_in!r}")
        else:
            self.overlay.show("✓ готово", "#46a758")
        polish_ms = 0 if fast_ok else polished.latency_ms
        print(f"[myflow] e2e={e2e_ms} мс (asr={asr.latency_ms}, polish={polish_ms})")

    def _asr_fallback(self, recording: Recording,
                      exc: Exception) -> AsrResult | None:
        """Резервное распознавание через GigaAM, если текущий движок отказал.

        Реальный случай: backend=whisper, а whisper-сервер не поднялся — каждая
        фраза падала с «whisper-server недоступен» и диктовка не работала. Теперь
        мы один раз пробуем локальную GigaAM (CPU, сервер не нужен), и если она
        справилась — переключаемся на неё и запоминаем это в настройках.
        """
        if getattr(self.whisper, "backend_name", "") == "gigaam":
            return None                     # уже GigaAM — резерва нет
        try:
            from myflow.asr_gigaam import GigaAmClient
            if not GigaAmClient.package_available():
                return None
        except Exception:  # noqa: BLE001
            return None
        try:
            client = GigaAmClient(self.settings.get().asr)
            if not client.warmup():
                print(f"[myflow] ✗ ASR-резерв: GigaAM не загрузилась "
                      f"({client._load_error})")
                return None
            result = client.transcribe(recording.pcm_int16)
            self.whisper = client
            self.settings.get().asr.backend = "gigaam"
            self.settings.save()
            print(f"[myflow] ⚠ {type(exc).__name__}: основной ASR отказал "
                  f"({str(exc)[:80]}) — распознал через GigaAM, переключился на неё")
            self.overlay.show("⚠ ASR переключён на GigaAM", "#f5c542")
            return result
        except Exception as exc2:  # noqa: BLE001
            print(f"[myflow] ✗ ASR-резерв не сработал: {exc2!r}")
            return None

    def _refresh_asr_prompt(self, force: bool = False) -> None:
        """Подмешать словарь в подсказку whisper (имена, термины, топонимы).

        Whisper учитывает initial_prompt как «контекст предыдущего текста» —
        если в подсказке есть «Сухой Лог», «Qwen», «opencode», он пишет их
        правильно. Промпт ограничен (~224 токена), поэтому база обрезается,
        а словарь идёт коротким списком.
        """
        cfg = self.settings.get().asr
        if not getattr(cfg, "prompt_from_dictionary", True):
            cfg.initial_prompt = self._asr_prompt_base   # словарь не подмешиваем
            return
        now = time.monotonic()
        if not force and now - self._asr_prompt_ts < 60.0:
            return
        self._asr_prompt_ts = now
        try:
            words = [d.get("target_word") for d in self.db.get_dictionary(None)]
        except Exception:  # noqa: BLE001 — подсказка не критична
            words = []
        seen: set[str] = set()
        vocab: list[str] = []
        for word in words:
            text = str(word or "").strip()
            key = text.lower()
            if not text or key in seen or len(text) > 24 or len(text.split()) > 3:
                continue
            seen.add(key)
            vocab.append(text)
            if len(vocab) >= 12:
                break
        base = self._asr_prompt_base
        if not vocab:
            cfg.initial_prompt = base
            return
        vocab_text = ", ".join(vocab)
        budget = 380 - len(vocab_text) - 12      # символов (~190 токенов на всё)
        head = base[:max(60, budget)].rstrip(" .,;:")
        cfg.initial_prompt = f"{head}. Термины: {vocab_text}."

    # ---- сниппеты / аудио / статистика неудач ----

    def _toggle_target_lock(self) -> None:
        """F6: закрепить текущее окно как цель вставки (или снять закрепление).

        Нужно, когда диктуешь в один чат, а «кликать в него» перед каждой фразой
        не хочется: закреплённая цель не пересчитывается по моменту старта.
        """
        cur = current_window()
        if self._target_locked is not None:
            print(f"[myflow] ○ закрепление снято (было: «{self._target_locked.label}»)")
            self.overlay.show("○ цель не закреплена", "#9aa0a6")
            self._target_locked = None
            return
        if cur is None:
            print("[myflow] ! не вижу активного окна — закрепить нечего")
            return
        self._target_locked = cur
        print(f"[myflow] ⚲ цель закреплена: «{cur.label}» (F6 — снять)")
        self.overlay.show(f"⚲ вставляю в: {cur.label[:28]}", "#46a758")

    def _paste_to_target(self, text: str, history_id: int | None = None) -> bool:
        """Вставить текст в ЦЕЛЕВОЕ окно (где началась диктовка).

        Раньше вставка шла в то окно, которое активно в момент Ctrl+V: случайный
        клик мышью во время фразы — и текст уходил не туда («не в тот чат»).
        Теперь цель зафиксирована с начала диктовки, а перед вставкой фокус
        возвращается в неё.

        ВАЖНО: фокус не отбирается, пока пользователь сам работает мышью или
        клавиатурой. Если он в этот момент открывает документы/переключает окна,
        вставка откладывается (status=pending) и выполняется на естественной
        паузе или когда он сам вернётся в целевое окно — управление не теряется.
        """
        target = self._target if (self.settings.get().insert_target or "start") == "start" else None
        cur = current_window()
        needs_move = bool(target) and (cur is None or cur.hwnd != target.hwnd)
        if needs_move:
            idle = idle_seconds()
            if idle < float(self.settings.get().insert_idle_seconds):
                # Пользователь занят: не мешаем, положим в отложенные.
                self._pending.append((text, target, history_id))
                print(f"[myflow] ⏳ пользователь занят ({idle:.1f} с назад была активность) — "
                      f"вставлю в «{target.label}» на паузе (F10 — вставить сразу)")
                self.overlay.show("⏳ вставлю на паузе", "#f5c542")
                return False
        return self._deliver(text, target, history_id)

    def _deliver(self, text: str, target: WindowRef | None, history_id: int | None) -> bool:
        """Непосредственная вставка (без проверки занятости)."""
        restored = False
        if target is not None:
            cur = current_window()
            if cur is None or cur.hwnd != target.hwnd:
                if self.settings.get().refocus_on_insert:
                    restored = focus_window(target.hwnd)
                    if restored:
                        time.sleep(0.06)      # окну нужно мгновение стать активным
                    else:
                        print(f"[myflow] ! не удалось вернуть фокус в «{target.label}» — "
                              "вставляю в активное окно")
                else:
                    print(f"[myflow]   цель «{target.label}» не в фокусе, "
                          "вставляю в активное (refocus_on_insert=false)")
        ok = self.paster.paste(text)
        self._last_insert_ts = time.monotonic()
        # Наблюдателю правок важно знать, в какое окно вставили.
        if target is not None and target.pid:
            self._last_insert_pid = target.pid
        else:
            self._remember_insert_target()
        if target is not None and restored:
            print(f"[myflow]   вставлено в «{target.label}» (фокус возвращён)")
        if history_id is not None and self.settings.get().flags.get("learning", False):
            self._spawn_edit_watcher(history_id, text)
        return ok

    def _deliver_pending_once(self) -> bool:
        """Одна попытка доставить отложенную вставку. True — что-то вставили.

        Отдельный метод (а не тело цикла) — чтобы поведение проверялось тестами
        без запуска фонового потока.
        """
        if not self._pending:
            return False
        text, target, history_id = self._pending[0]
        cur = current_window()
        if target is not None and not window_exists(target.hwnd):
            print(f"[myflow] ! целевое окно «{target.label}» закрылось — "
                  "вставляю в активное")
            self._pending.popleft()
            self._deliver(text, None, history_id)
            return True
        if target is None or (cur is not None and cur.hwnd == target.hwnd):
            self._pending.popleft()
            self._deliver(text, target, history_id)
            return True
        idle = idle_seconds()
        if idle >= float(self.settings.get().insert_idle_seconds):
            self._pending.popleft()
            print(f"[myflow] ⏳ пауза {idle:.1f} с — вставляю отложенное в "
                  f"«{target.label}»")
            self._deliver(text, target, history_id)
            return True
        return False

    def _pending_worker(self) -> None:
        """Отложенная вставка: ждём паузу пользователя или возврат в цель.

        Отложенное никогда не теряется: если целевое окно закрылось — вставляем
        в активное (и пишем об этом); если пользователь вернулся в цель — вставляем
        без перехвата фокуса; если он сам ничего не делает — вставляем на паузе.
        """
        while not self._stopping.is_set():
            self._stopping.wait(0.2)
            try:
                self._deliver_pending_once()
            except Exception as exc:  # noqa: BLE001 — воркер не должен падать
                print(f"[myflow] отложенная вставка: {exc!r}")

    def _flush_pending(self) -> int:
        """F10: вставить отложенное немедленно (явная команда пользователя)."""
        count = 0
        while self._pending:
            text, target, history_id = self._pending.popleft()
            self._deliver(text, target, history_id)
            count += 1
        if count:
            print(f"[myflow] ⏎ вставил отложенное вручную: {count} фраз(ы)")
            self.overlay.show("✓ отложенное вставлено", "#46a758")
        else:
            print("[myflow] отложенных вставок нет")
        return count

    def _remember_insert_target(self) -> None:
        """Запомнить pid окна, куда вставили (для AX-наблюдения за правкой)."""
        self._last_insert_pid = 0
        if not (self._ax_ready and self.ax.available):
            return
        try:
            sample = self.ax.read(timeout=1.5)
            if sample is not None:
                self._last_insert_pid = sample.pid
        except Exception:  # noqa: BLE001
            pass

    def _resolve_category(self, app_descriptor_text: str) -> str:
        """Категория приложения: пользовательские маски дополняют дефолтные."""
        user = self.settings.get().app_categories or {}
        if user:
            merged: dict[str, list[str]] = {
                k: list(v) for k, v in DEFAULT_APP_CATEGORIES.items()}
            for category, masks in user.items():
                merged.setdefault(str(category), [])
                merged[str(category)] = list(merged[str(category)]) + [str(m) for m in masks]
            return detect_category(app_descriptor_text, merged)
        return detect_category(app_descriptor_text)

    def _save_audio(self, recording: Recording):
        """Сохранить аудио фразы (своя метка времени вместо id: id ещё нет)."""
        try:
            return self.audio.save(int(time.time() * 1000), recording.pcm_int16)
        except Exception as exc:  # noqa: BLE001 — хранение не критично
            print(f"[myflow]   аудио не сохранено: {exc!r}")
            return None

    def _record_failure(self, status: str, app: str, category: str,
                        recording: Recording, audio_path, asr_latency_ms: int = 0) -> None:
        """Зафиксировать неуспех в истории (для статистики и retry)."""
        try:
            self.db.insert_history(DictationRecord(
                asr_text="", polished_text="", app_window=app,
                duration_ms=recording.duration_ms,
                e2e_latency_ms=asr_latency_ms or None,
                style=self._resolve_style(app), status=status,
                app_category=category,
                audio_path=str(audio_path) if audio_path else None,
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"[myflow]   не удалось записать неуспех: {exc!r}")

    def _try_snippet(self, asr_text: str, app: str, category: str, recording: Recording,
                     audio_path, asr: AsrResult, gen: int) -> bool:
        """Подставить сниппет, если распознанное совпало с его фразой."""
        try:
            snippet = self.db.find_snippet(asr_text)
        except Exception as exc:  # noqa: BLE001
            print(f"[myflow]   сниппеты недоступны: {exc!r}")
            return False
        if not snippet:
            return False
        replacement = snippet["replacement"]
        self.db.bump_snippet(snippet["phrase"])
        e2e_ms = 0
        history_id = self.db.insert_history(DictationRecord(
            asr_text=asr_text, polished_text=replacement, app_window=app,
            duration_ms=recording.duration_ms, e2e_latency_ms=e2e_ms,
            detected_language=asr.language, style="snippet", status="snippet",
            app_category=category, divergence=0.0,
            audio_path=str(audio_path) if audio_path else None,
        ))
        self._last_history_id = history_id
        print(f"[myflow] ✓ сниппет «{snippet['phrase']}» -> {replacement[:60]}")
        if self.settings.get().flags["auto_insert"] and gen == self._gen:
            try:
                self._paste_to_target(replacement)
                self._last_insert_ts = time.monotonic()
                self.overlay.show("✓ сниппет", "#46a758")
            except Exception as exc:  # noqa: BLE001
                _tb_to_file("ОШИБКА ВСТАВКИ (сниппет)")
                print(f"[myflow] ✗ сниппет не вставился: {exc!r}")
        return True

    def retry_pending(self, limit: int = 5) -> int:
        """Перераспознать сохранённые неуспешные фразы (аудио -> ASR -> LLM -> вставка).

        Возвращает число повторно обработанных фраз. Каждая попытка — новая
        строка истории, старая помечается status='retried'.
        """
        done = 0
        for _ in range(max(1, limit)):
            row = self.db.last_retryable()
            if not row:
                break
            pcm = self.audio.load(row["audio_path"]) if row.get("audio_path") else None
            if pcm is None:
                self.db.mark_status(int(row["id"]), "audio_missing")
                print(f"[myflow] retry #{row['id']}: аудио не найдено")
                continue
            peak = float(abs(pcm).max()) / 32768.0 if len(pcm) else 0.0
            rec = Recording(pcm_int16=pcm,
                            duration_ms=int(len(pcm) * 1000 / 16000),
                            max_amplitude=peak)
            print(f"[myflow] retry #{row['id']} ({rec.duration_ms} мс, {row['status']})")
            self.db.mark_status(int(row["id"]), "retried")
            before = self._last_history_id
            self._process_recording(self._gen, rec)
            if self._last_history_id != before:
                done += 1
        return done

    @staticmethod
    def _guard_polish(raw: str, polished_text: str, style: str,
                      divergence: float | None = None) -> str:
        """Страховка от LLM-саботажа: обрезка ИЛИ переписывание -> сырой ASR.

        Урок из лога: «Проверка связи.» -> «Провер». Старая версия требовала
        n_raw >= 3, поэтому КОРОТКИЕ фразы (а их большинство) не защищались.

        Теперь решение принимается ещё и по числовой метрике расхождения
        (аналог formattingDivergenceScore у Wispr): если модель переписала
        больше половины слов — это не полировка, а генерация.
        """
        if not polished_text.strip():
            return raw                      # пустой ответ ≠ пустая вставка
        raw = raw.strip()
        polished_text = polished_text.strip()
        n_raw = len(raw.split())
        n_pol = len(polished_text.split())
        c_raw, c_pol = len(raw), len(polished_text)
        div = divergence if divergence is not None else divergence_score(raw, polished_text)
        tag = f"divergence={div:.2f}"

        if style == "summarize":
            # Пересказ короче исходника — норма. Но не в разы длиннее и не
            # должен терять числа (они важны: «восемь тысяч» -> 8080 и т.п.).
            if n_raw and n_pol > max(n_raw * 2, n_raw + 20):
                print(f"[myflow]   WARNING: пересказ раздулся ({n_pol}/{n_raw} слов, {tag}) — сырой ASR")
                return raw
            if _numbers(raw) - _numbers(polished_text):
                print("[myflow]   WARNING: пересказ потерял числа — сырой ASR")
                return raw
            return polished_text

        if c_raw and c_pol < c_raw * 0.6:
            print(f"[myflow]   WARNING: LLM обрезал фразу ({c_pol}/{c_raw} симв., {tag}) — сырой ASR")
            return raw
        if div > MAX_DIVERGENCE:
            print(f"[myflow]   WARNING: LLM переписала фразу ({tag}, {n_pol}/{n_raw} слов) — сырой ASR")
            return raw
        if n_raw and n_pol <= n_raw * 0.5:
            print(f"[myflow]   WARNING: LLM выбросил половину слов ({n_pol}/{n_raw}) — сырой ASR")
            return raw
        if n_raw and n_pol > max(n_raw * 1.6, n_raw + 12):
            print(f"[myflow]   WARNING: LLM исказил фразу ({n_pol}/{n_raw} слов) — сырой ASR")
            return raw
        return polished_text

    # ---- стили (Ф8) ----

    def _cycle_style(self, debounce: float = 0.3) -> None:
        """F7: следующий стиль из цикла. Debounce от автоповтора клавиши."""
        now = time.monotonic()
        if now - self._last_style_switch_t < debounce:
            return
        self._last_style_switch_t = now
        if self._style_override is None:
            self._style_override = self._style_cycle[0]
            print(f"[myflow] стиль: {self._style_override} (ручной)")
        else:
            i = self._style_cycle.index(self._style_override) + 1
            if i >= len(self._style_cycle):
                self._style_override = None
                print(f"[myflow] стиль: авто (по окну), базовый={self.settings.get().default_style}")
            else:
                self._style_override = self._style_cycle[i]
                print(f"[myflow] стиль: {self._style_override} (ручной)")

    def _resolve_style(self, app_title: str) -> str:
        """Стиль для текущей диктовки.

        Приоритет (как у Wispr): ручной override (F7) > КАТЕГОРИЯ приложения >
        legacy-маска заголовка > стиль по умолчанию. Категория стабильнее
        заголовка (заголовок меняется при смене файла/вкладки).
        """
        if self._style_override is not None:
            if self._style_override not in STYLE_PROMPTS:
                raise KeyError(f"неизвестный стиль: {self._style_override}")
            return self._style_override
        s = self.settings.get()
        category = self._resolve_category(app_title)
        by_category = s.style_by_category or {}
        style = by_category.get(category)
        if style in STYLE_PROMPTS:
            return style
        title = app_title.lower()
        for mask, style in s.style_by_app.items():
            if mask.lower() in title and style in STYLE_PROMPTS:
                return style
        return s.default_style

    def learn_from_edit(self, edited_text: str) -> None:
        """Вызов после ручной правки текста (например, отдельной командой)."""
        if self._last_history_id is None:
            raise RuntimeError("нет последней диктовки для обучения")
        polished = self.db.get_polished_text(self._last_history_id)
        if polished is None:
            raise RuntimeError(f"диктовка {self._last_history_id} не найдена")
        result = record_edit_and_learn(
            self.db, self._last_history_id, polished, edited_text
        )
        print(f"[myflow] обучение: distance={result['edit_distance']}, новые слова={result['new_words']}")

    def _spawn_edit_watcher(self, history_id: int, polished_text: str) -> None:
        """Ф9: фоном ловим правку вставленного текста и учим словарь.

        Два пути (как у Wispr, который наблюдает поле ввода через AX):
          1. UI Automation — читаем текст поля в фокусе сами, Ctrl+C не нужен;
          2. буфер обмена — старый путь, если AX недоступен (нет PowerShell/UIA).
        Причина окончания наблюдения пишется в историю
        (observation_end_reason), как contentObservationEndReason у Wispr.
        """
        flags = self.settings.get().flags
        use_ax = flags.get("ax_observation", True) and self.settings.get().use_ax_observation
        reader = getattr(self.paster, "can_read", False)
        if not use_ax and not reader:
            return  # ни AX, ни клипборда: наблюдать нечем

        def _run() -> None:
            # sqlite3 check_same_thread=True: отдельное соединение в этом потоке.
            db = Database()
            try:
                if use_ax and self._ax_ready and self.ax.available:
                    reason = self._watch_with_ax(db, history_id, polished_text)
                else:
                    reason = self._watch_with_clipboard(db, history_id, polished_text)
                if reason:
                    db.set_observation_reason(history_id, reason)
                    print(f"[myflow] наблюдение: {reason}")
                    # Честный статус: если нашего текста в поле не нашли, «вставлено»
                    # могло быть ложным (поле без UI Automation, другое окно,
                    # вставка не прошла) — говорим об этом, а не молчим.
                    if reason == AxWatcher.NOT_FOUND:
                        print("[myflow] ⚠ текста в поле не видно — проверь, что вставка "
                              "прошла (поле может не поддерживать UI Automation)")
                        self.overlay.show("⚠ проверь вставку", "#f5c542")
            except Exception as exc:  # watcher не должен убивать основной поток
                print(f"[myflow] watcher: {exc}")
            finally:
                try:
                    db.close()
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True, name="edit-watcher").start()

    def _toggle_llm_unload(self) -> None:
        """F11: выгрузить LLM (освободить VRAM) или поднять обратно.

        Для игр: одно нажатие — и 6 ГБ VRAM свободны. Полировка при этом
        выключается и НЕ включается автолечением, пока не вернёшь сам.
        """
        cfg = self.settings.get().polish
        if cfg.enabled:
            cfg.enabled = False
            self._polish_user_off = True
            self.settings.save()
            self._stop_llm_server()
            print("[myflow] ⏏ LLM выгружена — VRAM освобождается "
                  "(F11 — вернуть, полировка включится сама)")
            self.overlay.show("⏏ LLM выгружена — VRAM свободна", "#f5c542")
        else:
            cfg.enabled = True
            self._polish_user_off = False
            self.settings.save()
            print("[myflow] ▲ LLM включается — поднимаю сервер в фоне")
            self.overlay.show("▲ LLM включается…", "#46a758")
            self._ensure_llm_async()

    def _game_watch_loop(self, interval: float = 5.0) -> None:
        """Полноэкранная игра -> выгрузить LLM (не отнимать VRAM)."""
        from myflow.winfocus import is_fullscreen_foreground
        while not self._stopping.wait(interval):
            try:
                cfg = self.settings.get().polish
                if not getattr(cfg, "auto_unload_on_game", True):
                    continue
                fullscreen = is_fullscreen_foreground()
                now = time.monotonic()
                if not fullscreen:
                    if self._fullscreen_since is not None:
                        self._fullscreen_since = None
                        if self._polish_user_off and cfg.enabled is False:
                            print("[myflow] игра/полный экран закрыт — LLM поднимется "
                                  "при следующей фразе (или F11)")
                    continue
                if self._fullscreen_since is None:
                    self._fullscreen_since = now
                    continue
                hold = float(getattr(cfg, "game_detect_seconds", 20.0))
                if (now - self._fullscreen_since >= hold and cfg.enabled
                        and not self._polish_user_off):
                    print(f"[myflow] ⏏ полный экран уже {hold:.0f} с — выгружаю LLM, "
                          f"чтобы не занимать VRAM")
                    self._toggle_llm_unload()
            except Exception as exc:  # noqa: BLE001 — сторож не должен падать
                print(f"[myflow] сторож игр: {exc!r}")

    def _ensure_game_watch(self) -> None:
        if self._game_watch is not None and self._game_watch.is_alive():
            return
        self._game_watch = threading.Thread(target=self._game_watch_loop,
                                           daemon=True, name="game-watch")
        self._game_watch.start()

    def _llm_confirms_correction(self, src: str, dst: str) -> bool:
        """Спросить у LLM: это правка ОШИБКИ распознавания, а не грамматика?

        Нужна для спорных пар вроде «каннект -> кинект»: учить их сразу полезно,
        но «сказал -> сказала» учить нельзя (сломает будущий текст). Спрашиваем
        только когда модель доступна; иначе пара ждёт второго повтора.
        """
        if not self.polish.cfg.enabled or not _health_check_llm(self.polish):
            return False
        try:
            prompt = (f"Распознавание речи выдало слово «{src}», пользователь "
                      f"исправил на «{dst}». Это ошибка распознавания (да) или "
                      f"грамматическая правка/перефраз (нет)? Ответь одним словом: да или нет.")
            answer = (self.polish.ask(prompt) or "").strip().lower()
            return answer.startswith("да") or answer.startswith("yes")
        except Exception:  # noqa: BLE001 — проверка не должна мешать обучению
            return False

    def _watch_with_ax(self, db: Database, history_id: int, polished_text: str) -> str:
        """Наблюдение через UI Automation. Возвращает причину окончания."""
        watcher = AxWatcher(self.ax,
                            window_seconds=self.settings.get().observation_seconds)
        edited, reason = watcher.watch(polished_text, expected_pid=self._last_insert_pid)
        if edited and reason == AxWatcher.EDITED:
            result = record_edit_and_learn(db, history_id, polished_text, edited,
                                           observation_end_reason=reason,
                                           validator=self._llm_confirms_correction)
            print(f"[myflow] авто-обучение (AX): distance={result['edit_distance']}, "
                  f"новые слова={result['new_words']}")
        return reason

    def _watch_with_clipboard(self, db: Database, history_id: int, polished_text: str) -> str:
        """Старый путь: пользователь должен скопировать правку в буфер."""
        if not getattr(self.paster, "can_read", False):
            return "ax_unavailable"
        result = watch_pasted_edit(db, history_id, polished_text, self.paster.read)
        if result:
            print(f"[myflow] авто-обучение (буфер): distance={result['edit_distance']}, "
                  f"новые слова={result['new_words']}")
            return "edited"
        return "observation_window_elapsed"


_KEY_NAMES = {
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "space", "enter", "tab", "esc", "backspace",
}


def _matches(key, hotkey_name: str) -> bool:
    """Совпадает ли событие клавиатуры с именем хоткея (f8, ctrl, char...)."""
    try:
        if hotkey_name in _KEY_NAMES:
            return key == getattr(keyboard.Key, hotkey_name)
        return getattr(key, "char", None) == hotkey_name
    except AttributeError:
        return False


_CTRL_HANDLERS: list = []   # держим ссылки: без них ctypes-колбэк соберёт GC


def _install_console_handler(on_exit) -> None:
    """Закрытие окна консоли / logoff / shutdown: успеть остановить запись.

    Без этого процесс убивался системой, минуя app.stop(): микрофон и
    потоки не освобождались, а очередь фраз терялась.
    """
    if sys.platform != "win32":
        return
    try:
        handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)

        def _handler(event: int) -> bool:
            # 2 = CTRL_CLOSE_EVENT, 5 = LOGOFF, 6 = SHUTDOWN
            if event in (2, 5, 6):
                try:
                    on_exit()
                except Exception:  # noqa: BLE001
                    pass
            return False

        cb = handler_type(_handler)
        _CTRL_HANDLERS.append(cb)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(cb, True)
    except Exception:  # noqa: BLE001 — не критично для работы
        pass


def _acquire_single_instance() -> bool:
    """True, если этот процесс — первый. Синглтон через TCP-порт 127.0.0.1:8765.

    Почему не мьютекс/pid-файл: tkinter/Tk на этой машине умеет порождать
    клон-процесс (та же командная строка), который НАСЛЕДУЕТ хэндл мьютекса
    и обходит мутекс-guard. Сокет не наследуется и bind атомарен — второй
    процесс гарантированно получает EADDRINUSE и выходит.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 8765))
    except OSError:
        return False
    s.listen(1)
    _acquire_single_instance._lock_sock = s
    return True


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _rotate_log(path: Path, max_bytes: int = 2_000_000) -> None:
    """Простая ротация: main.log -> main.log.1 (не даём файлу расти вечно)."""
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            backup = path.with_name(path.name + ".1")
            if backup.exists():
                backup.unlink()
            path.rename(backup)
    except OSError:
        pass


def rotate_all_logs(max_bytes: int = 2_000_000) -> list[str]:
    """Ротация всех логов проекта: main.log, llm.log, asr.log и *.log.err.

    Серверные логи (llama/whisper) росли без ограничений: в data/ накопилось
    больше десятка *.log.err от экспериментов с моделями. Теперь при старте
    каждый лог больше лимита уезжает в <имя>.1 (один бэкап, старый удаляется).
    """
    rotated: list[str] = []
    try:
        data = Path(__file__).resolve().parent.parent / "data"
        if not data.is_dir():
            return rotated
        for path in data.glob("*.log*"):
            if path.name.endswith(".1") or not path.is_file():
                continue
            if path.stat().st_size > max_bytes:
                _rotate_log(path, max_bytes)
                rotated.append(path.name)
    except OSError:
        pass
    return rotated


def _tee_log(max_bytes: int = 2_000_000) -> None:
    """Дублируем stdout в data/main.log — чтобы читать окно main из тестов.

    Пишем без ANSI-последовательностей (TUI pick_mic их сыпет), под локом
    (пишут несколько потоков) и с ротацией по размеру.
    """
    import io
    import sys
    log_path = Path(__file__).resolve().parent.parent / "data" / "main.log"
    _rotate_log(log_path, max_bytes)

    class _Tee(io.TextIOBase):
        def __init__(self, real) -> None:
            self.real = real
            self.fh = open(log_path, "a", encoding="utf-8", errors="replace")
            self._lock = threading.Lock()

        def write(self, s: str) -> int:
            with self._lock:
                try:
                    clean = _ANSI_RE.sub("", s)
                    if clean:
                        self.fh.write(clean)
                        self.fh.flush()
                except Exception:  # noqa: BLE001 — логирование не роняет main
                    pass
                try:
                    return self.real.write(s)
                except (OSError, ValueError, AttributeError):
                    return len(s)  # консоль умерла (FreeConsole/закрыт поток)

        def flush(self) -> None:
            with self._lock:
                for stream in (self.fh, self.real):
                    try:
                        stream.flush()
                    except Exception:  # noqa: BLE001
                        pass

        def isatty(self) -> bool:
            return False

    sys.stdout = _Tee(sys.stdout)
    print(f"\n=== сессия {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"(python {sys.version.split()[0]}) ===")


def main() -> None:
    import sys
    if sys.stdout is None:      # pythonw.exe: stdout отсутствует
        sys.stdout = open(Path(__file__).resolve().parent.parent / "data" / "main.log",
                          "a", encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(line_buffering=True)  # вывод виден сразу в консоли
    _tee_log()
    if not _acquire_single_instance():
        print("[myflow] другой экземпляр уже работает — выходим (single-instance).")
        return
    # Единственный guard от двойного запуска — атомарный bind порта 8765
    # (см. _acquire_single_instance). WMI/eldest-wins вариант удалён: он
    # каждые 5с поднимал powershell и путал venv-шим с настоящим python.
    # Микрофон при старте: программа сама проверяет доступность (по сигналу)
    # и выдаёт список живых устройств. Отмена = выход без запуска.
    if pick_mic() == 3:
        print("[myflow] запуск отменён (микрофон не выбран).")
        return
    try:
        app = MyFlowApp()
    except (ValueError, RuntimeError) as exc:
        print(f"[myflow] FATAL: {exc}")
        return
    try:
        app.start()
    except RuntimeError as exc:
        print(f"[myflow] FATAL: {exc}")
        return
    _install_console_handler(app.stop)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[myflow] остановлен")
        if app.db.stats()["dictations"]:
            print("[myflow] статистика:", app.db.stats())
    finally:
        # Гарантированный stop: освобождаем микрофон, слушатель, оверлей.
        try:
            app.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[myflow] ошибка остановки: {exc!r}")


if __name__ == "__main__":
    main()