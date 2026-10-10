"""Страж самодостаточности проекта.

Требование: ВСЁ, что относится к MyFLOW, лежит внутри папки проекта — модели,
бинарники, VAD, профиль голоса. Никаких «половинок» на других дисках: раньше
модели жили в E:\\llm\\..., и проект без них не работал.

Этот тест падает, если:
  * какой-то ассет разрешается ВНЕ корня репозитория;
  * в коде/скриптах появляется абсолютный путь с буквой диска;
  * нужный файл пропал из проекта;
  * models/ или bin/ перестали игнорироваться git (гигабайты в коммит).
"""
import re
import subprocess
import unittest
from pathlib import Path

from myflow.config import (
    DATA_DIR,
    ROOT,
    hf_cache_dir,
    llama_dir,
    models_dir,
    voiceprint_model_path,
    whisper_dir,
    whisper_model_path,
)

# Ассеты, без которых проект не работает (или работает частично).
REQUIRED_FILES = {
    "llama-server.exe": lambda: llama_dir() / "llama-server.exe",
    "LLM-модель Qwen3-4B Q8": lambda: models_dir() / "Qwen3-4B-Instruct-2507-Q8_0.gguf",
    "GigaAM (HF-кэш)": lambda: hf_cache_dir(),
    "модель голоса WeSpeaker": voiceprint_model_path,
    "VAD Silero": lambda: DATA_DIR / "vad_models" / "silero_v6.onnx",
    "whisper-server.exe": lambda: whisper_dir() / "Release" / "whisper-server.exe",
    "модель whisper (fallback)": whisper_model_path,
}

DIRS_WITH_ASSETS = {
    "каталог моделей": models_dir,
    "каталог llama.cpp": llama_dir,
    "каталог whisper.cpp": whisper_dir,
    "кэш GigaAM": hf_cache_dir,
    "модель голоса": voiceprint_model_path,
    "VAD Silero": lambda: DATA_DIR / "vad_models" / "silero_v6.onnx",
}


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


class SelfContainedProjectTest(unittest.TestCase):
    def test_assets_are_inside_repo(self):
        for name, getter in DIRS_WITH_ASSETS.items():
            path = getter()
            self.assertTrue(_inside_repo(path),
                            f"{name} лежит вне проекта: {path}")

    def test_required_assets_exist(self):
        missing = [name for name, getter in REQUIRED_FILES.items() if not getter().exists()]
        self.assertEqual(missing, [], f"нет файлов внутри проекта: {missing}")

    def test_models_and_bin_are_gitignored(self):
        """Модели весят гигабайты — в git они попадать не должны."""
        out = subprocess.run(["git", "check-ignore", "models", "bin"],
                             cwd=str(ROOT), capture_output=True, text=True)
        self.assertIn("models", out.stdout)
        self.assertIn("bin", out.stdout)

    def test_no_absolute_drive_paths_in_code(self):
        """Пути — только через myflow.config; хардкод диска ломает переносимость."""
        pattern = re.compile(r"[A-Za-z]:\\")
        roots = [ROOT / "myflow", ROOT / "scripts", ROOT / "tools"]
        files: list[Path] = []
        for base in roots:
            if base.exists():
                files += [p for p in base.rglob("*")
                          if p.is_file() and p.suffix in (".py", ".ps1", ".bat", ".cmd")]
        files += [p for p in ROOT.glob("*.bat")] + [p for p in ROOT.glob("*.ps1")]
        offenders = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for num, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{num}: {line.strip()[:80]}")
        self.assertEqual(offenders, [],
                         "в коде появились абсолютные пути (используй myflow.config):\n"
                         + "\n".join(offenders))

    def test_env_override_is_opt_in(self):
        """Переменные окружения переопределяют пути, но по умолчанию — проект."""
        import os
        for var in ("MYFLOW_MODELS_DIR", "MYFLOW_LLAMA_DIR",
                    "MYFLOW_WHISPER_DIR", "MYFLOW_WHISPER_MODEL"):
            self.assertNotIn(var, os.environ,
                             f"{var} задан в окружении: проект перестанет быть "
                             f"самодостаточным (это допустимо только осознанно)")


if __name__ == "__main__":
    unittest.main()


class WindowSizeLaunchTest(unittest.TestCase):
    """Размер окна применяется ДО старта TUI (консоль не помнит размер сама)."""

    ROOT = Path(__file__).resolve().parent.parent

    def test_bat_applies_saved_window_size(self):
        bat = (self.ROOT / "start-myflow.bat").read_text(encoding="utf-8")
        self.assertIn("apply-window-size.ps1", bat,
                      "start-myflow.bat должен применять размер из настроек")
        # вызов обязан идти ДО запуска TUI
        self.assertLess(bat.index("apply-window-size.ps1"), bat.index("myflow.tui"),
                        "размер ставим до старта TUI")

    def test_size_script_exists_and_reads_settings(self):
        script = self.ROOT / "scripts" / "apply-window-size.ps1"
        self.assertTrue(script.exists())
        text = script.read_text(encoding="utf-8")
        self.assertIn("settings.json", text)
        self.assertIn("tui", text)
        self.assertIn("SetWindowSize", text)

    def test_size_script_has_bom(self):
        """Windows PowerShell 5.1 читает UTF-8 только с BOM (иначе кириллица ломается)."""
        raw = (self.ROOT / "scripts" / "apply-window-size.ps1").read_bytes()
        self.assertEqual(raw[:3], b"\xef\xbb\xbf", "нужен UTF-8 BOM")
