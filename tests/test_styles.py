"""Стили: разрешение по окну, override F7, цикл (порт test-f8.py).

Не создаёт MyFlowApp: работаем на заглушке (иначе тест читал бы живой
settings.json и открывал продовую БД).
"""
import types
import unittest

from myflow.config import Settings
from myflow.main import MyFlowApp
from myflow.polish import STYLE_PROMPTS


class _SettingsStore:
    def __init__(self, settings):
        self._s = settings

    def get(self):
        return self._s


def make_style_app():
    app = types.SimpleNamespace()
    settings = Settings()
    settings.style_by_app = {"terminal": "terminal", "powershell": "terminal"}
    settings.default_style = "plain"
    app.settings = _SettingsStore(settings)
    app._style_override = None
    app._last_style_switch_t = 0.0
    app._style_cycle = [s for s in ("plain", "casual", "formal", "terminal", "summarize")
                        if s in STYLE_PROMPTS]
    app._resolve_style = types.MethodType(MyFlowApp._resolve_style, app)
    app._cycle_style = types.MethodType(MyFlowApp._cycle_style, app)
    return app


class StyleTest(unittest.TestCase):
    def test_per_app_mask(self):
        app = make_style_app()
        self.assertEqual(app._resolve_style("Terminal — Ubuntu"), "terminal")
        self.assertEqual(app._resolve_style("Windows PowerShell"), "terminal")
        self.assertEqual(app._resolve_style("Visual Studio Code"), "plain")

    def test_override_beats_mask(self):
        app = make_style_app()
        app._style_override = "formal"
        self.assertEqual(app._resolve_style("Terminal"), "formal")

    def test_unknown_override_raises(self):
        app = make_style_app()
        app._style_override = "несуществующий"
        with self.assertRaises(KeyError):
            app._resolve_style("Terminal")

    def test_cycle_order_and_reset(self):
        app = make_style_app()
        seen = []
        for _ in range(len(app._style_cycle) + 1):
            app._cycle_style(debounce=0.0)
            seen.append(app._style_override)
        self.assertEqual(seen[0], "plain")
        self.assertEqual(seen[1], "casual")
        self.assertEqual(seen[-1], None, "после последнего стиля — снова авто")
        self.assertEqual(len(seen), len(app._style_cycle) + 1)

    def test_debounce_ignores_fast_repeat(self):
        app = make_style_app()
        app._cycle_style(debounce=0.3)
        first = app._style_override
        app._cycle_style(debounce=0.3)   # автоповтор клавиши
        self.assertEqual(app._style_override, first, "debounce не сработал")

    def test_all_styles_have_prompts(self):
        for style in ("plain", "casual", "formal", "terminal", "summarize", "list"):
            self.assertIn(style, STYLE_PROMPTS)


if __name__ == "__main__":
    unittest.main()
