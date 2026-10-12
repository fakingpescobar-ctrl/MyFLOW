"""Запускалка окна чтения: батник должен работать без правки и без абсолютных путей.

Пользователь запускает чтение главы одним кликом, поэтому батник — часть продукта:
если он упадёт из-за скобок в блоке `if` (cmd такое не переваривает) или из-за
жёсткого пути к чьей-то папке, чтение не начнётся, а понять причину будет тяжело.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAT = ROOT / "read-chapter.bat"


class ReadChapterBatTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(BAT.exists(), "батник read-chapter.bat должен лежать в корне")
        self.text = BAT.read_text(encoding="utf-8", errors="replace")

    def test_launches_teleprompter(self):
        self.assertIn("myflow.readwindow", self.text, "окно с подсветкой остаётся")

    def test_opens_text_in_notepad_by_default(self):
        """Простая начитка: текст открывается в Блокноте, ничего вставлять не надо."""
        self.assertIn("notepad.exe", self.text)
        self.assertIn("--window", self.text, "подсветка доступна по флагу")

    def test_uses_project_venv(self):
        self.assertIn(".venv\\Scripts\\python.exe", self.text)

    def test_no_absolute_drive_paths(self):
        bad = re.findall(r"[A-Za-z]:\\", self.text)
        self.assertEqual(bad, [], "в батнике не должно быть чужих абсолютных путей")

    def test_runs_from_own_folder(self):
        self.assertIn('cd /d "%~dp0"', self.text)

    def test_no_parentheses_inside_if_blocks(self):
        """Скобки внутри блока `if (...)` ломают разбор cmd — это уже случалось."""
        blocks = re.findall(r"if\s+not\s+exist[^\n]*\(\n(.*?)\n\)", self.text, re.S)
        for block in blocks:
            for line in block.splitlines():
                if line.strip().lower().startswith(("echo", "rem")):
                    continue
                self.assertNotIn("(", line, f"скобка внутри if-блока: {line!r}")

    def test_default_text_is_in_private_folder(self):
        self.assertIn("data\\bench\\private", self.text)

    def test_mentions_active_window_mode(self):
        self.assertIn("aktivnoe okno", self.text)

    def test_mentions_controls(self):
        """В окне есть функции, а не только текст."""
        for marker in ("Zanova", "Otchet", "Propustit", "Stop"):
            self.assertIn(marker, self.text)


if __name__ == "__main__":
    unittest.main()
