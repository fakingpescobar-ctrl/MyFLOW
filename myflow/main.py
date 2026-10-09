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


def _health_check_llm(polish: PolishClient) -> bool:
    import requests
    try:
        r = requests.get(f"{polish.cfg.url}/models", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


class MyFlowApp:
    def __init__(self, overlay: StatusOverlay | None = None) -> None:
        """overlay — свой статус-слой (TUI подставляет свой); None = обычная плашка."""
        self.settings = SettingsStore()
        cfg = self.settings.get()
        self.db = Database()
        self._pref_auto = cfg.mode == "auto"
        # VAD: общий для обоих режимов параметр паузы; в ptt работает как
        # автостоп при удержании F8, в auto — это конец фразы.
        self.recorder = DictationRecorder(
            device=cfg.asr.input_device,
            silence_pause_seconds=cfg.silence_pause_seconds,
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
        # Ф8: ручной стиль (F7) перекрывает per-app; None = авто по окну.
        self._style_override: str | None = None
        self._last_style_switch_t = 0.0
        # Порядок цикла F7 — как в STYLE_PROMPTS (стабильный, не сеты).
        self._style_cycle = [s for s in ("plain", "casual", "formal", "terminal", "summarize")
                             if s in STYLE_PROMPTS]

    # ---- запуск / останов ----

    def start(self) -> None:
        if not _health_check(self.whisper):
            raise RuntimeError(
                f"whisper-server не отвечает на {self.whisper.cfg.url}/health — "
                "запусти scripts/start-asr.ps1"
            )
        if self.polish.cfg.enabled and not _health_check_llm(self.polish):
            print(f"[myflow] WARNING: llama-server не отвечает на {self.polish.cfg.url} — "
                  f"полировка отключена до поднятия сервера (scripts/start-llm.ps1)")
            self.polish.cfg.enabled = False
            self._ensure_polish_watchdog()
        if self.settings.get().flags["overlay"]:
            self.overlay.start()
        self._start_ax()
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
        self._start_listener()
        self._listener_watchdog = threading.Thread(
            target=self._listener_watchdog_loop, daemon=True, name="listener-watchdog")
        self._listener_watchdog.start()
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

    def _warm_asr(self) -> None:
        """Прогреть локальную ASR-модель в фоне (GigaAM грузится ~20-30 с).

        Whisper-серверу прогрев не нужен (он отдельный процесс), поэтому
        работаем только с клиентами, у которых есть warmup().
        """
        warmup = getattr(self.whisper, "warmup", None)
        if warmup is None:
            return

        def _run() -> None:
            try:
                if not warmup():
                    print("[myflow] локальная ASR-модель не загрузилась — "
                          "первая диктовка может не распознаться")
            except Exception as exc:  # noqa: BLE001
                print(f"[myflow] прогрев ASR: {exc!r}")

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

    def stop(self) -> None:
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

    def _listener_watchdog_loop(self, interval: float = 2.0) -> None:
        """Страховка: если слушатель хоткеев умер — поднять заново."""
        while True:
            time.sleep(interval)
            try:
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
        while True:
            time.sleep(interval)
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
        try:
            result = self.polish.polish(text, style=style, dictionary=dic)
            self._polish_fails = 0
            return result
        except Exception as exc:  # noqa: BLE001 — это защитный слой
            self._polish_fails += 1
            _tb_to_file("ОШИБКА ПОЛИРОВКИ")
            print(f"[myflow] ! полировка не удалась ({exc}) — вставлю сырой ASR")
            if self._polish_fails >= 3:
                self.polish.cfg.enabled = False
                print("[myflow] ! LLM недоступна 3 раза подряд — полировка выключена "
                      "до восстановления сервера (проверяю раз в 30с)")
                self._ensure_polish_watchdog()
            return PolishResult(text=text, latency_ms=0, skipped=True)

    # ---- жизненный цикл записи (PTT) ----

    def _begin_recording(self) -> None:
        self.recorder.start()
        self._auto_stopped = False
        print(f"[myflow] ● запись начата (окно: {app_descriptor()})")
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
        self._enqueue(rec)

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
        while True:
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
            self.paster.paste("\n")
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
            self.paster.paste(final)
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

        print(f"[myflow] ⚙ распознавание ({recording.duration_ms} мс, {app_category})...")
        self.overlay.show("⚙ распознавание…", "#f5c542")
        self._refresh_asr_prompt()   # словарь -> подсказка whisper (раз в минуту)

        try:
            asr: AsrResult = self.whisper.transcribe(recording.pcm_int16)
        except Exception as exc_asr:
            # Раньше ошибка ASR просто теряла фразу: в истории её не было,
            # а в статистике — ни следа. Теперь это строка со статусом.
            _tb_to_file("ОШИБКА ASR")
            print(f"[myflow] ✗ ASR не справился: {exc_asr}")
            self.overlay.show("✗ ASR ошибка", "#e5484d")
            self._record_failure("asr_error", app, app_category, recording, audio_path)
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

        # Метрика расхождения ASR <-> LLM (аналог formattingDivergenceScore).
        divergence = divergence_score(asr.text, final_text)
        e2e_ms = int((time.perf_counter() - started) * 1000)

        # Видимость полировки: показываем исходный ASR и результат LLM,
        # чтобы было понятно, что именно изменила модель.
        if polished.skipped:
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
                status="polish_error" if polish_failed else "ok",
                app_category=app_category,
                divergence=divergence,
                audio_path=str(audio_path) if audio_path else None,
            )
        )
        self._last_history_id = history_id

        print(f"[myflow] ✓ {final_text}")
        # Отмена могла прийти, пока шли ASR и полировка: текст уже в истории,
        # но вставлять его не нужно (иначе F9 «не отменяет»).
        if gen != self._gen:
            print("[myflow] ○ отменено во время обработки — вставлять не буду (текст в истории)")
            self.overlay.show("○ отменено", "#9aa0a6")
            return
        if flags["auto_insert"]:
            try:
                ok = self.paster.paste(final_text)
                self._last_insert_ts = time.monotonic()
                # Запоминаем процесс/окно, в которое вставили: наблюдатель
                # правок должен убедиться, что читает то же самое окно.
                self._remember_insert_target()
                if isinstance(self.paster, OpenCodeInjector) and not ok:
                    print("[myflow] ✗ инжект не прошёл: opencode не найден или консоль недоступна")
                    self.overlay.show("✗ инжект не прошёл", "#e5484d")
                else:
                    self.overlay.show("✓ вставлено", "#46a758")
                if flags["learning"]:
                    self._spawn_edit_watcher(history_id, final_text)
            except Exception as exc_in:
                # Любое падение на вставке — отдельным маркером, НЕ валит worker
                _tb_to_file("ОШИБКА ВСТАВКИ")
                print(f"[myflow] ✗ ОШИБКА ВСТАВКИ: {exc_in!r}")
        else:
            self.overlay.show("✓ готово", "#46a758")
        print(f"[myflow] e2e={e2e_ms} мс (asr={asr.latency_ms}, polish={polished.latency_ms})")

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
                self.paster.paste(replacement)
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
            except Exception as exc:  # watcher не должен убивать основной поток
                print(f"[myflow] watcher: {exc}")
            finally:
                try:
                    db.close()
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True, name="edit-watcher").start()

    def _watch_with_ax(self, db: Database, history_id: int, polished_text: str) -> str:
        """Наблюдение через UI Automation. Возвращает причину окончания."""
        watcher = AxWatcher(self.ax,
                            window_seconds=self.settings.get().observation_seconds)
        edited, reason = watcher.watch(polished_text, expected_pid=self._last_insert_pid)
        if edited and reason == AxWatcher.EDITED:
            result = record_edit_and_learn(db, history_id, polished_text, edited,
                                           observation_end_reason=reason)
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