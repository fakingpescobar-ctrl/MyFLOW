"""Установщик для нового пользователя: батник должен работать без правки.

Он первый, что запустит чужой человек, поэтому проверяем то, на чём батники уже
ломались: скобки и && внутри блоков if (cmd их не переваривает), чужие абсолютные
пути, запуск из своей папки и понятную подсказку, если Python не найден.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAT = ROOT / "install-myflow.bat"


class InstallBatTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(BAT.exists(), "install-myflow.bat должен лежать в корне")
        self.text = BAT.read_text(encoding="utf-8", errors="replace")

    def test_creates_venv_and_installs_requirements(self):
        self.assertIn("py -3.13 -m venv", self.text)
        self.assertIn("pip install -r", self.text)
        self.assertIn("requirements.txt", self.text)

    def test_runs_doctor_at_the_end(self):
        self.assertIn("doctor.py", self.text)

    def test_runs_from_own_folder(self):
        self.assertIn('cd /d "%~dp0"', self.text)

    def test_no_absolute_drive_paths(self):
        found = re.findall(r"[A-Za-z]:\\", self.text)
        self.assertEqual(found, [], "в батнике не должно быть чужих абсолютных путей")

    def test_explains_missing_python(self):
        self.assertIn(":no_python", self.text)
        self.assertIn("python.org", self.text.lower())

    def test_no_parentheses_inside_if_blocks(self):
        blocks = re.findall(r"if\s+[^\n]*\(\n(.*?)\n\)", self.text, re.S)
        for block in blocks:
            for line in block.splitlines():
                stripped = line.strip().lower()
                if not stripped or stripped.startswith(("echo", "rem")):
                    continue
                self.assertNotIn("(", line, f"скобка внутри if-блока: {line!r}")

    def test_mentions_models_are_separate(self):
        """Модели весят гигабайты — установщик должен честно сказать про README."""
        self.assertIn("README", self.text)


if __name__ == "__main__":
    unittest.main()


class FetchModelsBatTest(unittest.TestCase):
    """Батник скачивания моделей: первый шаг чужого человека после установки."""

    def setUp(self):
        self.bat = ROOT / "fetch-models.bat"
        self.assertTrue(self.bat.exists(), "fetch-models.bat должен лежать в корне")
        self.text = self.bat.read_text(encoding="utf-8", errors="replace")

    def test_runs_fetch_models(self):
        self.assertIn("fetch_models.py", self.text)

    def test_requires_venv_first(self):
        self.assertIn("install-myflow.bat", self.text)

    def test_no_absolute_drive_paths(self):
        self.assertEqual(re.findall(r"[A-Za-z]:\\\\", self.text), [])

    def test_no_parentheses_inside_if_blocks(self):
        for block in re.findall(r"if\s+[^\n]*\(\n(.*?)\n\)", self.text, re.S):
            for line in block.splitlines():
                stripped = line.strip().lower()
                if not stripped or stripped.startswith(("echo", "rem")):
                    continue
                self.assertNotIn("(", line, f"скобка внутри if-блока: {line!r}")


class UiBatTest(unittest.TestCase):
    """Интерфейс должен открываться ОТДЕЛЬНЫМ окном, а не вкладкой браузера."""

    def setUp(self):
        self.bat = ROOT / "ui.bat"
        self.assertTrue(self.bat.exists(), "ui.bat должен лежать в корне")
        self.text = self.bat.read_text(encoding="utf-8", errors="replace")

    def test_uses_app_window_mode(self):
        self.assertIn("--app=", self.text,
                      "режим приложения даёт окно без вкладок и адресной строки")

    def test_tries_edge_then_chrome(self):
        self.assertIn("msedge", self.text)
        self.assertIn("chrome", self.text)

    def test_does_not_start_second_server(self):
        """Если сервер уже поднят, окно открывается без второго запуска."""
        self.assertIn("LISTENING", self.text)

    def test_no_absolute_drive_paths(self):
        self.assertEqual(re.findall(r"[A-Za-z]:\\\\", self.text), [])

    def test_no_parentheses_inside_if_blocks(self):
        for block in re.findall(r"if\s+[^\n]*\(\n(.*?)\n\)", self.text, re.S):
            for line in block.splitlines():
                stripped = line.strip().lower()
                if not stripped or stripped.startswith(("echo", "rem")):
                    continue
                self.assertNotIn("(", line, f"скобка внутри if-блока: {line!r}")
