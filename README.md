# MyFLOW — локальный голосовой диктант (копия архитектуры Wispr Flow)

Полностью локальный диктант-ассистент по образцу Wispr Flow, но без облака:
**Whisper large-v3-turbo (ASR) + Qwen3-4B-Instruct-2507 Q8 (полировка) на RTX 3080**, всё в SQLite.

> Аудит проекта и план работ: [AUDIT.md](AUDIT.md). Что уже исправлено — в разделе
> «Что починено (аудит 09.10.2026)» ниже.

## Установка (с нуля)

Нужны: **Windows 10/11**, **Python 3.13**, NVIDIA GPU (CUDA 12) для серверов.

```powershell
# 1) окружение
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt

# 2) бинарники whisper.cpp (CUDA-сборка) — положить в bin\whisper\Release\
#    нужен whisper-server.exe (+ dll-ки CUDA 12)
#    релизы: https://github.com/ggml-org/whisper.cpp/releases

# 3) модели (≈4.6 ГБ) — по умолчанию ждём на диске E:
#    X:\llm\models\ggml-large-v3-turbo-q5_0.bin          (Whisper, ~547 МБ)
#    X:\llm\models\Qwen3-4B-Instruct-2507-Q8_0.gguf      (LLM, ~4 ГБ)
#    X:\llm\llama-cpp\llama-server.exe                   (llama.cpp)
#    Другой диск/путь — задай переменные окружения (см. ниже).

# 4) проверка
.venv\Scripts\python -m unittest discover -s tests -t .
```

### Переменные окружения (вместо хардкода путей)

| Переменная | По умолчанию | Что задаёт |
|---|---|---|
| `MYFLOW_MODELS_DIR` | `X:\llm\models` | каталог с `.bin`/`.gguf` моделями |
| `MYFLOW_LLAMA_DIR` | `X:\llm\llama-cpp` | каталог с `llama-server.exe` |
| `MYFLOW_SILERO_PATH` | `data\vad_models\silero_v6.onnx` | ONNX-модель Silero VAD |

Модель VAD (`data/vad_models/silero_v6.onnx`) лежит в репозитории данных и нужна для работы
(исключена из git — при переносе скопируй `data/vad_models/` вручную).

## Запуск

**В один клик:** `start-myflow.bat` — поднимает оба сервера (фоновые, логи в `data\`),
ждёт готовности и запускает main в отдельном окне.

**Вручную** (три окна):

```
powershell -File scripts\start-asr.ps1      # whisper-server :8081
powershell -File scripts\start-llm.ps1      # llama-server   :8082
.venv\Scripts\python -m myflow.main         # сам MyFLOW (PTT=F8)
```

При старте — TUI-выбор микрофона, режима (`ptt`/`auto`) и цели вставки
(активное окно / чат OpenCode).

**Тесты:** `.venv\Scripts\python -m unittest discover -s tests -t .`
(122 теста, без серверов/микрофона/GUI; pytest тоже подойдёт — конфиг в `pyproject.toml`).

**Проба пайплайна с микрофоном:** `.venv\Scripts\python -X utf8 tools\e2e_mic.py 4`
(пишет 4 с → распознаёт, учитывает `asr.input_device` из настроек).

### Настройки (`data/settings.json`)

Ключевые ручки (остальное — в `myflow/config.py`):

| Ключ | Смысл |
|---|---|
| `hotkey.ptt_key` / `cancel_key` / `style_cycle_key` | клавиши PTT/отмены/стиля (`f8`/`f9`/`f7`) |
| `hotkey.suppress` | `true` — не отдавать эти клавиши активному окну (в IDE F8 = шаг отладки) |
| `mode` | `ptt` (по кнопке) или `auto` (слушает постоянно) |
| `insert_method` | `clipboard` (Ctrl+V) или `opencode` (инжект в консоль) |
| `silence_pause_seconds` | пауза, завершающая фразу (0.9 с) |
| `history_retention_days` | сколько дней хранить историю (0 = вечно) |
| `flags.live_preview` / `learning` / `overlay` / `auto_insert` | фиче-флаги |
| `polish.top_p` / `repeat_penalty` / `max_tokens` | параметры LLM-полировки |
| `style_by_app` | маска окна/процесса → стиль (например `"code.exe": "terminal"`) |

## Структура

```
myflow/           # ядро (пакет)
  main.py         # оркестратор: хоткеи, очередь, пайплайн, отмена
  audio.py        # запись 16k mono, PTT/AUTO, автостоп по VAD, превью
  vad.py          # Silero v6 (ONNX, CPU) — стрим с hidden-state
  asr.py          # клиент whisper-server (Session+Retry, WAV в памяти)
  polish.py       # клиент llama-server: стили, правила, защита от инъекций
  commands.py     # голосовые команды: точные/префиксы/LLM-интент
  learning.py     # авто-обучение словаря из правок (diff -> dictionary)
  db.py           # SQLite: history / dictionary / settings_kv (WAL)
  insertion.py    # вставка через буфер + SendInput Ctrl+V
  opencode_inject.py  # инжект в консоль OpenCode (WriteConsoleInputW)
  overlay.py      # статус-оверлей (Tk, topmost, click-through)
  pick_mic.py     # TUI выбора микрофона/режима/цели
  config.py       # настройки (data/settings.json) и фиче-флаги
scripts/          # start-asr / start-llm / start-all (пути от расположения скрипта)
tools/            # ручные инструменты: e2e_mic, smoke_overlay, bench_vad,
                  # validate_silero, record_validation
tests/            # unittest-набор (регрессы на все P0-баги из аудита)
data/             # рантайм: settings.json, myflow.sqlite, логи, модели VAD
```

## Архитектура (зеркало Wispr Flow)

| Слой | Wispr Flow | MyFLOW | Статус |
|---|---|---|---|
| Запись | AudioWorklet 16k mono, чанк 640 (40мс) | `audio.py`: sounddevice 16k mono, чанк 640, Float32→Int16 (кламп как у них) | ✅ |
| Горячие клавиши | PTT Hold, Esc cancel, шорткаты | `main.py`: глобальный PTT (F8), cancel (F9), стиль (F7) через pynput | ✅ |
| Режимы записи | — | `pick_mic.py` при старте (стрелки ↑/↓, Enter, Esc): микрофон → режим (`ptt` — по кнопке, `auto` — слушает постоянно) | ✅ |
| VAD авто-стоп | — | пауза >0.9с завершает фразу; авто-стоп PTT только при наличии речи; жёсткий лимит 60с и в PTT | ✅ |
| Инкрементальный ASR | streaming → live text | `on_interval`: слайдинг-окно (4с) каждые ~1.8с → драфт-превью «… текст» (не вставляется) | ✅ |
| ASR | gRPC streaming → Baseten «Canto 1.0» | `asr.py` → whisper-server (large-v3-turbo, CUDA) `:8081` | ✅ |
| Полировка | gpt-5-nano / gpt-oss-120b | `polish.py` → llama-server (Qwen3-4B Q8) `:8082`; стили plain/casual/formal/terminal/list/summarize + правило чисел + омофоны | ✅ |
| Instruct (пересказ, команды) | gpt-oss-120b | тот же Qwen3-4B, стиль `summarize` (0.3-0.5с, пересказ 2-3 пр.) | ✅ |
| Голосовые команды | — | `commands.py`: точные фразы/префиксы/LLM-интент (стоп, скип, перескажи, списком, отправь, удали последнее, новый абзац) | ✅ |
| Вставка | C# helper (accessibility) | `insertion.py`: Clipboard + SendInput Ctrl+V (буфер не трогает картинки/файлы) | ✅ |
| БД | flow.sqlite (History/Dictionary/Settings) | `db.py`: SQLite history/dictionary/settings_kv, WAL, миграции | ✅ |
| Словарь | personal dictionary, авто-обучение из правок | `learning.py`: diff правок → learned-слова (порог 2); per-app маска окна; глобальные пары без дублей | ✅ |
| Телеметрия | Sentry/PostHog (облако) | поведение НЕ телеметрируется (локально) | ✅ |

## Что починено (аудит 09.10.2026)

Подробности и доказательства — в [AUDIT.md](AUDIT.md).

- **Не терять речь:** ошибка полировки → вставляется сырой ASR (+ watchdog возврата LLM);
  LLM-интент с `kind=text` больше не «съедает» диктовку; отмена (F9/«стоп») действует
  и на уже обработанные/очередные фразы.
- **Авто-обучение ожило:** watcher читает буфер через `read()` (раньше вызывался метод
  вставки без аргумента → TypeError → 6 с холостого цикла на 100 % CPU).
- **Хоткеи не умирают:** исключение в колбэке больше не убивает pynput-слушатель,
  добавлен супервизор, F8-автоповтор не рвёт фразу после автостопа.
- **Словарь:** `app_mask=''` вместо NULL (NULL в UNIQUE давал дубли), убран двойной `hit_count`.
- **Защита от LLM-саботажа:** пустой/обрезанный (`finish_reason=length`) ответ → сырой ASR;
  короткие фразы тоже защищены (кейс «Проверка связи.» → «Провер»).
- **Прочее:** live-превью снова работает на второй и последующих фразах, Ctrl+Z по VK-коду
  (зависит от раскладки раньше), ASR с keep-alive/ретраями и UTF-8, лимит диктовки в PTT,
  буфер обмена не уничтожает картинки/файлы, требования и пути больше не хардкодятся.

## Замеры (RTX 3080 10GB)

| Шаг | Латентность |
|---|---|
| Whisper large-v3-turbo (4с аудио) | ~500 ms |
| Полировка Qwen3-4B Q8 (короткий текст) | 598 ms cold / 232 ms warm |
| e2e (диктовка → вставка) | ~1.2–1.5 s |

## Ограничения / честно

- **Пак моделей:** единственный годный — **Qwen3-4B-Instruct-2507 Q8** (полировка 0.3-0.9с,
  пересказ 0.3-1.3с, числа верны с NUMBERS_RULE). Забракованы: Qwen3.5-9B UD/Coder и
  Nemotron-3-Nano-4B (thinking-режим: пустой content при пересказе), gemma-3n-E4B
  (копирует/самоповторяется). Nemotron неплох для полировки (0.5с) — резерв.
- `-c 8192` у llama-server: системный промпт (~1.5-2к токенов) + диктовка + ответ должны
  влезать в контекст (на `-c 2048` в логе были реальные 400 `exceed_context_size`).
- 10 ГБ VRAM = одновременно 1 LLM + whisper (две LLM не влезают, оффлоад убивает скорость).
- **Авто-обучение** ловит правки только если они скопированы в буфер (выделение + Ctrl+C)
  в окне 6 с после вставки; правки мышью без копирования не видны. Правки в разных окнах
  не складываются (порог 2 — в рамках одного окна).
- **Голосовые команды:** «отправь X» с аргументом намеренно НЕ перехватывается; команды с
  аргументом могут сработать, если ты реально диктуешь о пересказе списком. «удали последнее»
  работает через Ctrl+Z не дольше 60 с после вставки. Переоформление последней фразы без
  аргумента вставляется НИЖЕ оригинала (замена выделения небезопасна).
- **Live-превью** не ускоряет финальную вставку (полная транскрипция — при отпускании
  кнопки/паузе), это живой отклик во время длинных диктовок.
- **Вставка через буфер** может конфликтовать с чужой вставкой; restore-логика есть и
  не трогает нетекстовые форматы (картинки/файлы).
- **pynput-хоткеи** требуют прав; при фокусе на elevated-окне Windows (UIPI) хук/вставка
  могут не сработать — статус в оверлее в этом случае не гарантирован.
- Логи (`data/main.log`) содержат распознанный текст — не выкладывай их; каталог `data/`
  исключён из git.
- Ограничения инструментов: `tools/bench_vad.py` тянет `jfk.wav` из интернета при отсутствии
  файла и гоняет другие параметры VAD, чем рантайм (сравнительный бенч, не эталон).
