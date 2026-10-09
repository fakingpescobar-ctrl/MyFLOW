"""MyFLOW — конфигурация и фиче-флаги (аналог Wispr Flow feature flags).

Все настройки читаются из data/settings.json (перезаписывается через команды),
дефолты хардкодятся здесь, чтобы коду не приходилось валидировать мусор.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"
DB_FILE = DATA_DIR / "myflow.sqlite"

# Границы записи (как у Wispr: CHUNK_DURATION_MS=40, лимиты диктовки).
SAMPLE_RATE = 16_000          # RECORDING_SAMPLE_RATE (Wispr: S=16e3)
CHUNK_SAMPLES = 640           # 40 мс на чанк (Wispr: CHUNK_DURATION_MS=40)
MAX_DICTATION_SECONDS = 60    # жёсткий лимит одной диктовки
MAX_DICTATION_WARN_SECONDS = 50

# Категории приложений (аналог personalizationAppType у Wispr): маска по
# имени процесса/заголовку -> категория. Первое совпадение побеждает.
# Матч по подстроке в нижнем регистре, поэтому "code" поймает "code.exe" и
# "visual studio code", но не "encode".
DEFAULT_APP_CATEGORIES: dict[str, tuple[str, ...]] = {
    "terminal": ("windowsterminal", "powershell", "pwsh", "cmd.exe", "conhost",
                 "wt.exe", "alacritty", "wezterm", "mintty", "git-bash"),
    "code": ("code.exe", "devenv", "pycharm", "idea64", "webstorm", "goland",
             "sublime_text", "notepad++", "cursor", "rider64", "clion"),
    "chat": ("telegram", "whatsapp", "discord", "slack", "teams", "skype",
             "signal", "viber", "element", "happ"),
    "email": ("outlook", "thunderbird", "mail.exe", "thebat", "em client"),
    "notes": ("notepad", "obsidian", "logseq", "joplin", "onenote", "evernote"),
    "docs": ("winword", "excel", "powerpnt", "acrobat", "libreoffice", "wps"),
    "browser": ("chrome", "browser", "msedge", "firefox", "opera", "brave", "yandex"),
}

# Категория -> стиль полировки по умолчанию (у Wispr 4 стиля: default/formal/
# casual/very casual; у нас 6 промптов, поэтому маппинг шире).
DEFAULT_CATEGORY_STYLES: dict[str, str] = {
    "terminal": "terminal",
    "code": "plain",
    "chat": "casual",
    "email": "formal",
    "notes": "plain",
    "docs": "formal",
    "browser": "plain",
    "other": "plain",
}


def detect_category(app_descriptor: str,
                    categories: dict[str, list[str]] | None = None) -> str:
    """Определить категорию приложения по «процесс | заголовок».

    Аналог их personalizationAppType: стиль подбирается по КАТЕГОРИИ, а не по
    полному заголовку окна (который меняется при смене файла/вкладки).

    Матч по границам слов, а не «просто подстрока»: иначе «encode.exe»
    попадал бы в категорию code по подстроке «code.exe».
    """
    text = (app_descriptor or "").lower()
    rules = categories or DEFAULT_APP_CATEGORIES
    for category, masks in rules.items():
        for mask in masks:
            if not mask:
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(str(mask).lower()) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                return category
    return "other"


@dataclass
class AsrConfig:
    """Параметры локального whisper-server (аналог gRPC-стоика Wispr)."""
    url: str = "http://127.0.0.1:8081"
    language: str = "auto"          # auto | ru | en
    translate_to_english: bool = False
    temperature: float = 0.0
    timeout_seconds: float = 60.0   # общий (legacy)
    timeout_connect: float = 3.0    # отдельный connect-таймаут
    timeout_read: float = 45.0      # отдельный read-таймаут (инференс)
    input_device: int | None = None    # индекс sounddevice для записи; None = системный
    initial_prompt: str = ""           # подсказка для whisper (типичные фразы диктовки)


@dataclass
class PolishConfig:
    """Параметры локальной LLM-полировки (аналог Instruct/Polish у Wispr)."""
    url: str = "http://127.0.0.1:8082/v1"   # qwen3-4b форматтер (быстрый)
    # Fallback-имя модели, если /v1/models не ответил. Реальное имя клиент
    # спрашивает у сервера (polish.PolishClient._effective_model).
    model: str = "qwen3-4b-instruct-2507"
    enabled: bool = True            # фиче-флаг polish-provider
    max_tokens: int = 512           # верхняя граница ответа (реально — по бюджету)
    temperature: float = 0.2
    top_p: float = 0.9
    repeat_penalty: float = 1.05     # против дегенеративных повторов
    timeout_seconds: float = 60.0


@dataclass
class HotkeyConfig:
    """Глобальные хоткеи (аналог шорткатов Wispr)."""
    ptt_key: str = "f8"             # Push-to-Talk: держишь — диктуешь
    # https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key
    cancel_key: str = "f9"          # отмена текущей диктовки
    style_cycle_key: str = "f7"     # переключение стиля (цикл plain→casual→formal→terminal→summarize)
    suppress: bool = False          # True: не отдавать эти клавиши активному окну


@dataclass
class Settings:
    asr: AsrConfig = field(default_factory=AsrConfig)
    polish: PolishConfig = field(default_factory=PolishConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    # Режим активации записи: "ptt" — по нажатию кнопки (F8),
    # "auto" — постоянно слушает микрофон, фраза режется по паузе.
    mode: str = "ptt"
    # Куда вставлять распознанный текст:
    #   "clipboard" — обычная вставка в активное окно (буфер + Ctrl+V);
    #   "opencode"  — инжект прямо в чат OpenCode (WriteConsoleInputW, без фокуса).
    insert_method: str = "clipboard"
    # После вставки в чат OpenCode — нажать Enter (отправить сообщение).
    opencode_auto_enter: bool = True
    # VAD-автостоп: пауза между словами длиннее silence_pause_seconds
    # завершает запись. Работает и в ptt (при удержании F8 тоже режет),
    # и в auto (это и есть конец фразы).
    auto_stop_by_silence: bool = True
    silence_pause_seconds: float = 0.9
    # Стиль по умолчанию: минимум форматирования, как в терминале.
    default_style: str = "plain"
    # Сколько дней хранить историю диктовок (в ней полный текст; 0 = вечно).
    history_retention_days: int = 90
    # Стили per-app (аналог Wispr PersonalizationStyleSettings):
    # маска заголовка окна -> стиль. Первое совпадение побеждает (case-insensitive).
    style_by_app: dict[str, str] = field(default_factory=lambda: {
        "terminal": "terminal",
        "powershell": "terminal",
        "cmd": "terminal",
        "command prompt": "terminal",
    })
    # Категория приложения -> стиль (приоритетнее style_by_app): как у Wispr,
    # где стиль задаётся типом приложения, а не заголовком окна.
    style_by_category: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_STYLES))
    # Свои маски категорий: {"terminal": ["windowsterminal", "my-shell"]}.
    # Дополняют дефолтные (не заменяют), матч по границам слов.
    app_categories: dict[str, list[str]] = field(default_factory=dict)
    # Наблюдение за полем ввода через UI Automation (обучение из правок без Ctrl+C).
    use_ax_observation: bool = True
    observation_seconds: float = 6.0
    # Сколько последних фраз держать в аудио-хранилище (перераспознавание упавших).
    audio_keep: int = 50
    # Фиче-флаги (у Wispr: cloud-sync, edge-proxy, dark-mode ...).
    flags: dict[str, bool] = field(default_factory=lambda: {
        "auto_insert": True,      # вставлять текст сразу после распознавания
        "learning": True,         # авто-обучение словаря из правок
        "beep_on_start": False,   # звуковой сигнал начала записи
        "overlay": True,          # Ф6: статус-оверлей в углу экрана
        "live_preview": True,     # инкрементальный ASR: превью во время записи
        "snippets": True,         # голосовые сниппеты (фраза -> готовый текст)
        "keep_audio": True,       # хранить аудио последних фраз (для retry)
        "ax_observation": True,   # наблюдать поле через UIA (иначе — буфер обмена)
    })


class SettingsStore:
    """Потокобезопасное хранилище настроек (load в main, save редкий)."""

    def __init__(self, path: Path = SETTINGS_FILE) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> Settings:
        if not self._path.exists():
            return Settings()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("не объект")
            data = Settings()
            # Вложенные dataclass материализуем явно (Settings(**raw) с dict не работает).
            # Неизвестные ключи внутри секции игнорируем (совместимость вперёд/назад),
            # отсутствующие — берём из дефолтов dataclass.
            for name, cls in (("asr", AsrConfig), ("polish", PolishConfig), ("hotkey", HotkeyConfig)):
                section = raw.get(name)
                if isinstance(section, dict):
                    known = {f.name for f in fields(cls)}
                    setattr(data, name, cls(**{k: v for k, v in section.items() if k in known}))
            for name, value in raw.items():
                if name in ("asr", "polish", "hotkey"):
                    continue
                if name == "flags":
                    # MERGE с дефолтами, а не замена: иначе отсутствие нового
                    # флага в старом settings.json = KeyError на каждой фразе.
                    if isinstance(value, dict):
                        data.flags.update(
                            {k: bool(v) for k, v in value.items() if k in data.flags}
                        )
                    continue
                setattr(data, name, value)
            return data
        except (json.JSONDecodeError, TypeError, ValueError):
            # Fail loud: не молча подставляем дефолты, а сообщаем о битой конфе.
            raise RuntimeError(f"settings.json повреждён: {self._path}")

    def get(self) -> Settings:
        with self._lock:
            return self._data

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(asdict(self._data), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def set_flag(self, name: str, value: bool) -> None:
        with self._lock:
            if name not in self._data.flags:
                raise KeyError(f"неизвестный флаг: {name}")
            self._data.flags[name] = value