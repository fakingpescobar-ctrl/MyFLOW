"""Вставка в чат DeepSeek Harness: режим «без фокуса» через плагин.

Как это устроено: MyFLOW дописывает строку в файл-инбокс в домашнем каталоге DSH
(`$DSH_HOME/myflow-inbox.jsonl`), а плагин `dsh-plugin-myflow`, живущий внутри
приложения, читает её и вкладывает текст в текущую сессию через API агента
(`agent.followup`). Никаких портов, токенов и перехвата фокуса.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from myflow import dsh_inject                        # noqa: E402
from myflow.dsh_inject import DshInjector            # noqa: E402


class DshInjectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.inbox = Path(self.tmp.name) / "myflow-inbox.jsonl"
        self.injector = DshInjector(self.inbox)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_json_line(self):
        self.assertTrue(self.injector.paste("Проверка связи раз, два, три."))
        lines = [ln for ln in self.inbox.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]),
                         {"text": "Проверка связи раз, два, три."})

    def test_appends_not_overwrites(self):
        self.injector.paste("первая")
        self.injector.paste("вторая")
        lines = [json.loads(ln) for ln in
                 self.inbox.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual([item["text"] for item in lines], ["первая", "вторая"])

    def test_creates_missing_directory(self):
        deep = Path(self.tmp.name) / "нет" / "такой" / "папки" / "inbox.jsonl"
        self.assertTrue(DshInjector(deep).paste("текст"))
        self.assertTrue(deep.exists())

    def test_empty_text_is_refused(self):
        self.assertFalse(self.injector.paste(""))
        self.assertFalse(self.injector.paste("   "))
        self.assertFalse(self.inbox.exists(), "пустой текст не должен создавать файл")

    def test_no_clipboard_access(self):
        """Авто-обучение из правок в этом режиме невозможно — как и у OpenCode."""
        self.assertFalse(self.injector.can_read)
        self.assertEqual(self.injector.read(), "")

    def test_inbox_path_follows_dsh_home(self):
        with mock.patch.dict(os.environ, {"DSH_HOME": r"C:\tmp\dsh-home"}):
            self.assertEqual(dsh_inject.inbox_path(),
                             Path(r"C:\tmp\dsh-home") / "myflow-inbox.jsonl")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSH_HOME", None)
            self.assertTrue(str(dsh_inject.inbox_path()).endswith(".dsh\\myflow-inbox.jsonl")
                            or str(dsh_inject.inbox_path()).endswith(".dsh/myflow-inbox.jsonl"))


class InsertMethodTest(unittest.TestCase):
    """Режим выбирается строкой «Куда вставлять»: окно → OpenCode → DSH."""

    def _app(self, method: str):
        import myflow.main as M
        from myflow.config import SettingsStore
        from myflow.db import Database
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        store = SettingsStore(Path(tmp.name) / "s.json")
        store.get().insert_method = method
        app = M.MyFlowApp(settings=store, db=Database(Path(tmp.name) / "d.sqlite"))
        app._mic_cache = ([], __import__("time").monotonic())
        app._tmp = tmp
        return app

    def test_dsh_method_builds_dsh_injector(self):
        app = self._app("dsh")
        self.assertEqual(type(app.paster).__name__, "DshInjector")
        app.db.close()
        app.ax.stop()
        app._stopping.set()

    def test_unknown_method_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._app("telepathy")
        self.assertIn("insert_method", str(ctx.exception))

    def test_panel_row_shows_dsh_and_cycles(self):
        """←→ проходит окно → OpenCode → DSH, и подпись строки это отражает."""
        app = self._app("clipboard")
        # Строка «Куда вставлять» лежит в колонке ЗНАЧЕНИЙ, не модулей.
        value = lambda: next(v for v in app.control_snapshot()["values"]
                             if v["key"] == "insert")["value"]
        seen = []
        for _ in range(3):
            app.control_apply("insert", +1)
            seen.append((type(app.paster).__name__, value()))
        kinds = [kind for kind, _ in seen]
        self.assertIn("DshInjector", kinds, "цикл должен доходить до DSH")
        self.assertIn("OpenCodeInjector", kinds, "и до OpenCode")
        dsh_label = [label for kind, label in seen if kind == "DshInjector"][0]
        self.assertIn("DeepSeek Harness", dsh_label)
        app.db.close()
        app.ax.stop()
        app._stopping.set()


if __name__ == "__main__":
    unittest.main()
