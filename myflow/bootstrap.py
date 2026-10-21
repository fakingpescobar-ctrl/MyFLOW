"""MyFLOW — самопочинка: чего не хватает и предложение это скачать.

Зачем. Новичку не нужно знать, какие пакеты ставить и откуда брать модели.
Приложение само находит недостающее, честно говорит об этом и — только после
согласия пользователя — докачивает.

    python -m myflow.bootstrap          # что не хватает (ничего не качает)
    python -m myflow.bootstrap --fix    # спросить и починить

Модуль намеренно ничего тяжёлого не импортирует: его можно звать ДО поднятия
ядра, чтобы поймать отсутствующие пакеты раньше, чем приложение упадёт на
`import numpy`.

Принцип: сначала спрашиваем, потом качаем. Ничего не ставится и не скачивается
без явного «да» пользователя.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from myflow.config import ROOT, models_dir, voiceprint_model_path, whisper_model_path

HF = "https://huggingface.co/{repo}/resolve/main/{name}"
LLM_SIZE = 4_280_405_600  # размер Qwen3-4B-Instruct-2507-Q8_0.gguf
SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)


# --------------------------------------------------------------------- модели


@dataclass(frozen=True)
class Package:
    """Python-пакет, который нужен приложению."""

    pip_name: str
    module: str
    why: str
    required: bool = True

    @property
    def installed(self) -> bool:
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            # У пакета нет родительского пакета — тоже «не установлен».
            return False


@dataclass(frozen=True)
class ModelFile:
    """Файл-модель, который нужен приложению.

    `present` — «на месте ли» (сверяется РАЗМЕР, а не только факт файла:
    оборванная загрузка не должна считаться за модель).
    `found_name` — что именно лежит на месте (у LLM это может быть своя
    дообученная LoRA вместо базовой Qwen3-4B).
    """

    label: str
    url: str
    size_bytes: int
    why: str
    destination: Callable[[], Path]
    present: Callable[[], bool]
    found_name: Callable[[], str] | None = None
    required: bool = False

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 2**20

    def name_of(self) -> str:
        """Имя файла, который реально на месте (для отчётов)."""
        if self.found_name is not None:
            return self.found_name()
        return self.destination().name


def _silero_path() -> Path:
    env = os.environ.get("MYFLOW_SILERO_PATH")
    return Path(env) if env else ROOT / "data" / "vad_models" / "silero_v6.onnx"


def _base_llm_path() -> Path:
    return models_dir() / "Qwen3-4B-Instruct-2507-Q8_0.gguf"


def _oracle_enabled() -> bool:
    """Включён ли Oracle LLM: ему нужна именно базовая Qwen3-4B.

    Если Oracle выключен и есть своя LoRA — база не нужна, и качать 4 ГБ
    незачем.
    """
    try:
        from myflow.config import SettingsStore

        return bool(SettingsStore().get().asr.judge)
    except Exception:  # noqa: BLE001 — настройки не должны ломать проверку
        return False


def _llm_satisfied() -> bool:
    """Есть ли чем полировать.

    Своя LoRA подходит для полировки. Базовая Qwen3-4B подходит и для
    полировки, и для Oracle — а тот без неё не стартует.
    """
    if _file_present(_base_llm_path(), LLM_SIZE):
        return True
    return _llm_found() is not None and not _oracle_enabled()


def _llm_found() -> Path | None:
    """Какая модель полировки реально на месте.

    Своя дообученная LoRA (myflow-postproc-*.gguf) важнее базовой Qwen3-4B —
    именно её приложение выбирает в первую очередь.
    """
    posts = sorted(models_dir().glob("myflow-postproc-*.gguf"),
                   key=lambda p: -p.stat().st_size)
    for path in posts:
        if path.stat().st_size > 1024 * 1024:
            return path
    base = _base_llm_path()
    return base if _file_present(base, LLM_SIZE) else None


def _file_present(path: Path, expected_bytes: int) -> bool:
    """Файл на месте И того размера, который ожидается.

    Проверка размера важна: оборванная загрузка (обрыв сети, убитый curl)
    оставляет файл — и без этой проверки он считался бы готовой моделью.
    Допуск 1 %: размер на диске может чуть отличаться (так у Qwen3-4B).
    """
    if not path.exists():
        return False
    return path.stat().st_size >= expected_bytes * 0.99


def _model(label: str, url: str, size_bytes: int, why: str,
           destination: Callable[[], Path], *, required: bool = False,
           present: Callable[[], bool] | None = None,
           found_name: Callable[[], str] | None = None) -> ModelFile:
    """Собрать запись о модели: по умолчанию «на месте» = файл нужного размера."""
    if present is None:
        present = lambda: _file_present(destination(), size_bytes)  # noqa: E731
    if found_name is None:
        found_name = lambda: destination().name  # noqa: E731
    return ModelFile(label, url, size_bytes, why, destination, present,
                     found_name, required)


MODEL_FILES: tuple[ModelFile, ...] = (
    _model(
        "VAD (Silero)",
        SILERO_URL,
        2_327_524,
        "детектор речи — без него приложение не стартует",
        _silero_path,
        required=True,
    ),
    _model(
        "ASR whisper",
        HF.format(
            repo="ggerganov/whisper.cpp", name="ggml-large-v3-turbo-q5_0.bin"
        ),
        574_041_195,
        "второй движок распознавания (когда GigaAM сомневается)",
        whisper_model_path,
    ),
    _model(
        "LLM Qwen3-4B",
        HF.format(
            repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
            name="Qwen3-4B-Instruct-2507-Q8_0.gguf",
        ),
        LLM_SIZE,
        "полировка текста: пунктуация, стиль, числа",
        _base_llm_path,
        present=_llm_satisfied,
        found_name=lambda: (_llm_found() or _base_llm_path()).name,
    ),
    _model(
        "Проверка голоса",
        url=HF.format(
            repo="onnx-community/wespeaker-voxceleb-resnet34-LM", name="onnx/model.onnx"
        ),
        size_bytes=26_535_549,
        why="отсекает чужую речь (телевизор, коллеги)",
        destination=voiceprint_model_path,
    ),
)

PACKAGES: tuple[Package, ...] = (
    Package("numpy", "numpy", "счёт и аудио"),
    Package("sounddevice", "sounddevice", "микрофон"),
    Package("onnxruntime", "onnxruntime", "распознавание"),
    Package("requests", "requests", "серверы моделей"),
    Package("pyperclip", "pyperclip", "вставка в чужое окно"),
    Package("pynput", "pynput", "горячие клавиши"),
    Package("soundfile", "soundfile", "аудио фраз: OGG вместо zlib", required=False),
    Package(
        "onnx-asr[hub]",
        "onnx_asr",
        "GigaAM — основной движок распознавания (56 мс)",
        required=False,
    ),
    Package(
        "kaldi_native_fbank", "kaldi_native_fbank", "проверка голоса", required=False
    ),
)


# --------------------------------------------------------------------- проверка


def missing_packages() -> list[Package]:
    """Пакеты, которых нет (обязательные и желательные)."""
    return [p for p in PACKAGES if not p.installed]


def missing_models() -> list[ModelFile]:
    """Файлы-модели, которых нет."""
    return [m for m in MODEL_FILES if not m.present()]


def blocking_gaps(
    packages: list[Package] | None = None, models: list[ModelFile] | None = None
) -> list[str]:
    """Что именно мешает запуску. Пустой список — запускать можно."""
    packages = missing_packages() if packages is None else packages
    models = missing_models() if models is None else models
    gaps = [f"пакет {p.pip_name} ({p.why})" for p in packages if p.required]
    gaps += [f"файл {m.destination()} ({m.why})" for m in models if m.required]
    return gaps


def total_download_mb(models: list[ModelFile] | None = None) -> float:
    models = missing_models() if models is None else models
    return sum(m.size_mb for m in models)


# --------------------------------------------------------------------- действия


def install_packages(
    packages: list[Package], log: Callable[[str], None] = print
) -> bool:
    """Поставить пакеты в то окружение, из которого запущено приложение."""
    if not packages:
        return True
    names = [p.pip_name for p in packages]
    log(f"  ставлю пакеты: {', '.join(names)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *names],
        cwd=str(ROOT),
    )
    return result.returncode == 0


def _draw_progress(label: str, done: int, total: int) -> None:
    pct = 0 if total <= 0 else min(100, int(done * 100 / total))
    filled = pct * 25 // 100
    bar = "#" * filled + "." * (25 - filled)
    sys.stdout.write(
        f"\r    [{bar}] {pct:3d}%  {done / 2**20:,.0f} / {total / 2**20:,.0f} МБ"
    )
    sys.stdout.flush()


def download_model(model: ModelFile, log: Callable[[str], None] = print) -> bool:
    """Скачать файл с докачкой (curl). Прерванную загрузку можно продолжить."""
    dest = model.destination()
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  ↓ {model.label} ({model.size_mb:,.0f} МБ) → {dest}")

    cmd = [
        "curl.exe",
        "-L",
        "-C",
        "-",
        # --fail обязателен: без него curl на 404/403 завершится с кодом 0
        # и сохранит HTML-страницу ошибки — она «весит» больше килобайта,
        # прошла бы проверку размера и стала бы «моделью» из мусора.
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--speed-limit",
        "50000",
        "--speed-time",
        "60",
        "--max-time",
        "7200",
        "-s",
        "-S",
        "-o",
        str(dest),
        model.url,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except OSError:
        log("  ! curl не найден — файл придётся скачать вручную")
        return False

    while proc.poll() is None:
        _draw_progress(
            model.label, dest.stat().st_size if dest.exists() else 0, model.size_bytes
        )
        time.sleep(0.5)
    sys.stdout.write("\r" + " " * 78 + "\r")
    sys.stdout.flush()

    if proc.returncode != 0:
        log(
            f"  ! {model.label}: загрузка прервалась — запусти ещё раз, докачка продолжится"
        )
        return False
    if not dest.exists():
        log(f"  ! {model.label}: файл не скачался")
        return False
    actual = dest.stat().st_size
    if actual < model.size_bytes * 0.99:
        log(
            f"  ! {model.label}: скачалось {actual / 2**20:,.1f} МБ, "
            f"ожидалось {model.size_mb:,.1f} МБ — файл битый, запусти ещё раз"
        )
        return False
    log(f"  ✓ {model.label} готов")
    return True


def confirm(question: str) -> bool:
    """Согласие пользователя. Без интерактивной консоли — отказ."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        answer = input(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("", "y", "yes", "д", "да", "о")


# --------------------------------------------------------------------- отчёт


def format_gaps(
    packages: list[Package] | None = None, models: list[ModelFile] | None = None
) -> list[str]:
    """Строки отчёта о том, чего не хватает."""
    packages = missing_packages() if packages is None else packages
    models = missing_models() if models is None else models
    lines: list[str] = []
    for pkg in packages:
        mark = "обязателен" if pkg.required else "желателен"
        lines.append(f"  [ пакет ] {pkg.pip_name} — {pkg.why} ({mark})")
    for model in models:
        mark = "обязателен" if model.required else "желателен"
        lines.append(
            f"  [ файл  ] {model.destination()} — {model.why}, "
            f"{model.size_mb:,.0f} МБ ({mark})"
        )
    return lines


def manual_instructions(
    packages: list[Package] | None = None, models: list[ModelFile] | None = None
) -> list[str]:
    """Что делать руками, если пользователь отказался от автоматической загрузки."""
    packages = missing_packages() if packages is None else packages
    models = missing_models() if models is None else models
    lines: list[str] = []
    if packages:
        names = " ".join(p.pip_name for p in packages)
        lines.append(f"  поставь пакеты:  .venv\\Scripts\\pip install {names}")
    for model in models:
        lines.append(f"  скачай вручную:  {model.url}")
        lines.append(f"                   → {model.destination()}")
    return lines


def repair(
    log: Callable[[str], None] = print,
    packages: list[Package] | None = None,
    models: list[ModelFile] | None = None,
) -> bool:
    """Поставить пакеты и скачать файлы (согласие уже получено)."""
    packages = missing_packages() if packages is None else packages
    models = missing_models() if models is None else models
    ok = True
    if packages:
        ok = install_packages(packages, log) and ok
    for model in models:
        ok = download_model(model, log) and ok
    return ok


def preflight(
    log: Callable[[str], None] = print,
    packages: list[Package] | None = None,
    models: list[ModelFile] | None = None,
) -> bool:
    """Проверка перед стартом. False — запускаться нельзя.

    Спрашивает разрешение и только после согласия качает. Если чего-то
    обязательного не хватает и пользователь отказался — возвращает False.
    """
    packages = missing_packages() if packages is None else packages
    models = missing_models() if models is None else models
    if not packages and not models:
        return True

    log("")
    log("  MyFLOW — кое-чего не хватает:")
    for line in format_gaps(packages, models):
        log(line)
    log("")

    if not confirm("  Поставить и скачать? [Y/n] "):
        log("")
        log("  Скачай руками:")
        for line in manual_instructions(packages, models):
            log(line)
        log("")
        gaps = blocking_gaps(packages, models)
        if gaps:
            log("  Без этого не запустится:")
            for gap in gaps:
                log(f"    - {gap}")
        return not gaps

    log("")
    repair(log, packages, models)
    log("")
    gaps = blocking_gaps()
    if gaps:
        log("  Всё ещё не хватает:")
        for gap in gaps:
            log(f"    - {gap}")
        return False
    log("  Готово, продолжаю.")
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fix = "--fix" in argv

    packages = missing_packages()
    models = missing_models()
    if not packages and not models:
        print("  Всё на месте — ничего скачивать не нужно.")
        return 0

    for line in format_gaps(packages, models):
        print(line)
    print("")
    if not fix:
        print("  Чтобы починить автоматически: python -m myflow.bootstrap --fix")
        return 1

    if not confirm("  Поставить и скачать? [Y/n] "):
        print("")
        for line in manual_instructions(packages, models):
            print(line)
        return 1
    repair()
    return 0 if not blocking_gaps() else 1


if __name__ == "__main__":
    raise SystemExit(main())
