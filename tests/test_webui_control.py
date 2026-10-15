"""Мост управления: окно вызывает тот же пульт, что и TUI.

Проверяем без запуска приложения: поднимаем сервер с ФЕЙКОВЫМ пультом и дёргаем
эндпоинты так, как это делает страница. Если мост сломан, кнопки в окне перестанут
работать, а панель будет работать - это и есть расхождение, которого мы избегаем.
"""
import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.webui import serve, snapshot_for  # noqa: E402


class FakePanel:
    """Путь, который ведёт себя как пульт MyFLOW: снимок + применение."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.on = True

    def snapshot(self) -> dict:
        return {
            "modules": [{"key": "llm", "label": "Post-Processor", "on": self.on,
                         "ok": True, "value": "включён", "hint": "своя 0.6B"},
                        {"key": "oracle", "label": "Oracle LLM", "on": False,
                         "ok": True, "value": "выключен", "hint": ""}],
            "values": [{"key": "insert", "label": "Куда вставлять",
                        "value": "никуда (только история)", "hint": ""}],
        }

    def apply(self, action, value):
        self.calls.append((action, value))
        return True, f"применено: {action} {value}"


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.panel = FakePanel()
        self.server, self.url = serve(0, control=self.panel.snapshot,
                                      apply=self.panel.apply, quiet=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        # Порядок важен: сначала останавливаем цикл, потом закрываем сокет.
        # Наоборот - WinError 10038 («операция не на сокете»).
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url.rstrip("/") + path, timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))

    def post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url.rstrip("/") + "/api/action",
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))

    def test_snapshot_reports_control_attached(self):
        data = self.get("/api/snapshot")
        self.assertTrue(data["control"], "окно должно знать, что управление доступно")
        labels = [m["label"] for m in data["modules"]]
        self.assertIn("Post-Processor", labels)
        self.assertIn("Oracle LLM", labels)
        self.assertEqual(data["values"][0]["key"], "insert")

    def test_toggle_calls_the_same_pult(self):
        """Первой пробуется форма TUI: control_apply(ключ, ...)."""
        result = self.post({"action": "module", "value": "llm"})
        self.assertTrue(result["ok"])
        self.assertEqual(self.panel.calls, [("llm", None)])

    def test_adjust_calls_with_delta(self):
        """И здесь форма TUI: control_apply(ключ, шаг)."""
        self.post({"action": "adjust", "value": "insert:+1"})
        self.assertEqual(self.panel.calls, [("insert", 1)])

    def test_without_control_it_says_so(self):
        state = snapshot_for(None)
        self.assertFalse(state["control"])
        self.assertIn("history", state)


if __name__ == "__main__":
    unittest.main()
