"""Кнопки окна должны доходить до пульта, каким бы ни был точный контракт.

Сбой, который здесь ловится: окно слало action="adjust" с кортежем, а приложение такого
обработчика не имело - кнопки молча не работали, и пользователь видел «толку 0».
"""
import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.webui import serve  # noqa: E402


class PickyPanel:
    """Пульт, который понимает ТОЛЬКО форму TUI: control_apply(ключ, значение)."""

    def __init__(self):
        self.calls = []

    def snapshot(self) -> dict:
        return {"modules": [{"key": "llm", "label": "Post-Processor", "on": True,
                             "ok": True, "value": "включён", "hint": ""}],
                "values": [{"key": "insert", "label": "Куда вставлять",
                            "value": "opencode", "hint": ""}]}

    def apply(self, action, value):
        self.calls.append((action, value))
        if action == "llm":
            return True, "модуль переключён"
        if action == "insert":
            return True, "вставка: чат OpenCode"
        return False, "не понимаю"


class ActionFormsTest(unittest.TestCase):
    def setUp(self):
        self.panel = PickyPanel()
        self.server, self.url = serve(0, control=self.panel.snapshot,
                                      apply=self.panel.apply, quiet=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url.rstrip("/") + "/api/action",
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))

    def test_module_toggle_reaches_pult(self):
        result = self.post({"action": "module", "value": "llm"})
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.panel.calls[0], ("llm", None),
                         "первая попытка - форма TUI")

    def test_adjust_reaches_pult(self):
        result = self.post({"action": "adjust", "value": "insert:+1"})
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.panel.calls[0], ("insert", 1),
                         "первая попытка - форма TUI: control_apply(ключ, шаг)")

    def test_unknown_action_reports_honestly(self):
        result = self.post({"action": "module", "value": "нет-такого"})
        self.assertFalse(result["ok"])
        self.assertTrue(result["message"])


if __name__ == "__main__":
    unittest.main()
