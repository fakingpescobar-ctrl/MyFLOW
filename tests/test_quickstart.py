"""Быстрый старт должен совпадать с тем, что реально лежит в проекте.

Инструкция - первое, что читает чужой человек. Если в ней назван несуществующий
батник или пропущен шаг, он застрянет на ровном месте, поэтому сверяем текст с
файлами: все упомянутые батники существуют, порядок шагов на месте, доктор и
скачивание моделей описаны.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "QUICKSTART.md"


class QuickstartTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(DOC.exists(), "QUICKSTART.md должен лежать в корне")
        self.text = DOC.read_text(encoding="utf-8")

    def test_all_mentioned_bats_exist(self):
        mentioned = set(re.findall(r"[a-z-]+\.bat", self.text))
        self.assertTrue(mentioned, "в инструкции должны быть батники")
        for name in mentioned:
            self.assertTrue((ROOT / name).exists(), f"в инструкции есть {name}, а файла нет")

    def test_chain_order(self):
        order = ["install-myflow.bat", "fetch-models.bat", "doctor.py", "start-myflow.bat"]
        positions = [self.text.index(item) for item in order]
        self.assertEqual(positions, sorted(positions),
                         "шаги должны идти в правильном порядке: "
                         "установка -> модели -> доктор -> запуск")

    def test_readme_links_to_quickstart(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("QUICKSTART.md", readme, "из README должна быть ссылка")

    def test_dsh_bridge_explained(self):
        self.assertIn("Мост в DSH", self.text)

    def test_insert_modes_listed(self):
        for mode in ("активное окно", "OpenCode", "DeepSeek Harness", "никуда"):
            self.assertIn(mode, self.text, f"режим {mode} должен быть описан")

    def test_where_things_live(self):
        for item in ("myflow.sqlite", "settings.json", "models"):
            self.assertIn(item, self.text)
