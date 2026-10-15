"""Режимы запуска не должны включаться вместе.

Пользователь: «UI и TUI должны быть разные варианты запуска, а не сразу всё вместе».
TUI-батник не поднимает сервер интерфейса вообще; UI-батник поднимает приложение
(свёрнутым) вместе с сервером интерфейса и открывает окно.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class LaunchModesTest(unittest.TestCase):
    def test_tui_mode_does_not_serve_ui(self):
        text = (ROOT / "start-myflow.bat").read_text(encoding="utf-8", errors="replace")
        self.assertIn('set "MYFLOW_WEBUI=0"', text,
                      "режим TUI не должен поднимать интерфейс")
        self.assertNotIn('set "MYFLOW_WEBUI=1"', text)

    def test_ui_mode_serves_ui(self):
        text = (ROOT / "ui.bat").read_text(encoding="utf-8", errors="replace")
        self.assertIn('set "MYFLOW_WEBUI=1"', text,
                      "режим UI обязан поднять сервер интерфейса")

    def test_modes_do_not_start_each_other(self):
        tui = (ROOT / "start-myflow.bat").read_text(encoding="utf-8", errors="replace")
        self.assertNotIn("ui.bat", tui.replace("sm. ui.bat", ""),
                         "TUI-батник не должен запускать окно")
        ui = (ROOT / "ui.bat").read_text(encoding="utf-8", errors="replace")
        self.assertNotIn("start-myflow", ui,
                         "UI-батник не должен запускать TUI-батник")

    def test_ui_mode_starts_app_minimized(self):
        ui = (ROOT / "ui.bat").read_text(encoding="utf-8", errors="replace")
        self.assertIn("/min", ui, "приложение в режиме UI стартует свёрнутым")
        self.assertIn("myflow.tui", ui)
