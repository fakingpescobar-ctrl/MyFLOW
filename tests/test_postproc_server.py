"""Два сервера LLM: 4B для Oracle и дообученный 0.6B для полировки.

Схема: маленькая модель оформляет текст (~50 мс), а 4B остаётся для Oracle LLM,
где нужна смекалка. Полировка выбирает клиента по флагу `polish.postproc`, а
сервер постпроцессора появляется в списке ТОЛЬКО когда модель собрана — иначе
пульт пытался бы поднять несуществующий файл.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from myflow.tui import default_specs                    # noqa: E402


class PostprocServerSpecTest(unittest.TestCase):
    def test_no_spec_when_model_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("myflow.tui.models_dir", lambda: Path(tmp)):
                keys = [s.key for s in default_specs()]
        self.assertIn("llm", keys)
        self.assertNotIn("postproc", keys, "без собранной модели сервер не нужен")

    def test_spec_appears_when_model_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "myflow-postproc-qwen3-06b-q8_0.gguf").write_bytes(b"GGUF")
            with mock.patch("myflow.tui.models_dir", lambda: Path(tmp)):
                specs = {s.key: s for s in default_specs()}
        self.assertIn("postproc", specs)
        spec = specs["postproc"]
        self.assertEqual(spec.port, 8083, "постпроцессор должен жить на своём порту")
        self.assertIn("8083", spec.url)
        self.assertIn("myflow-postproc", str(spec.model))
        self.assertTrue(str(spec.model).endswith(".gguf"))

    def test_both_servers_coexist(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "myflow-postproc-x.gguf").write_bytes(b"GGUF")
            with mock.patch("myflow.tui.models_dir", lambda: Path(tmp)):
                specs = {s.key: s for s in default_specs()}
        self.assertEqual(specs["llm"].port, 8082)
        self.assertEqual(specs["postproc"].port, 8083)
        self.assertNotEqual(specs["llm"].model, specs["postproc"].model)


class PolishClientSelectionTest(unittest.TestCase):
    def _app(self):
        import types
        from test_main_flow import make_app
        import myflow.main as M
        app = make_app()
        # make_app() — заглушка: привязываем к ней НАСТОЯЩИЕ методы
        app._polish_client = types.MethodType(M.MyFlowApp._polish_client, app)
        return app

    def test_default_uses_main_client(self):
        app = self._app()
        app.polish.cfg.postproc = False
        self.assertIs(app._polish_client(), app.polish)

    def test_postproc_flag_uses_separate_client(self):
        app = self._app()
        app.polish.cfg.enabled = True
        app.polish.cfg.postproc = True
        app.polish.cfg.url_postproc = "http://127.0.0.1:8083/v1"
        client = app._polish_client()
        self.assertIsNot(client, app.polish, "для 0.6B нужен свой клиент")
        self.assertEqual(client.cfg.url, "http://127.0.0.1:8083/v1")
        self.assertTrue(client.cfg.enabled, "главный тумблер полировки включён")
        # клиент кэшируется: второй вызов отдаёт тот же объект
        self.assertIs(app._polish_client(), client)
        # и главный клиент не переключился на чужой URL
        self.assertNotEqual(app.polish.cfg.url, "http://127.0.0.1:8083/v1")

    def test_polish_safe_uses_postproc_client_when_enabled(self):
        import types
        import myflow.main as M

        app = self._app()
        app.settings.get().polish.postproc = True
        app.polish.cfg.enabled = True
        app.polish.cfg.postproc = True
        app.polish.cfg.url_postproc = "http://127.0.0.1:8083/v1"
        called = {}

        class FakeClient:
            cfg = types.SimpleNamespace(enabled=True, postproc=True)

            def polish(self, text, style="plain", dictionary=None):
                called["style"] = style
                return types.SimpleNamespace(text=text, latency_ms=1, skipped=False)

        with mock.patch.object(app, "_polish_client", lambda: FakeClient()), \
             mock.patch.object(M, "_health_check_llm", lambda *a, **k: True):
            app._polish_safe("проверка связи ну", "plain", [])
        self.assertIn("style", called)


if __name__ == "__main__":
    unittest.main()


class MasterSwitchTest(unittest.TestCase):
    """Выключенная полировка глушит и постпроцессор: тумблер главный."""

    def test_disabled_master_switch_skips_postproc(self):
        import types
        import myflow.main as M
        from test_main_flow import make_app

        app = make_app()
        app._polish_client = types.MethodType(M.MyFlowApp._polish_client, app)
        app._polish_safe = types.MethodType(M.MyFlowApp._polish_safe, app)
        app.polish.cfg.enabled = False
        app.polish.cfg.postproc = True
        app.polish.cfg.url_postproc = "http://127.0.0.1:8083/v1"
        result = app._polish_safe("проверка связи", "plain", [])
        self.assertTrue(getattr(result, "skipped", False),
                        "при выключенной полировке постпроцессор не должен зваться")


class PostprocPanelRowTest(unittest.TestCase):
    """Строка «Постпроцессор» в панели: появляется с моделью, переключает режим."""

    def setUp(self):
        # Настоящее приложение с временными настройками (как в тестах панели):
        # нужны control_snapshot/control_apply, а не заглушка.
        from test_panel import _app
        self.app = _app()
        self.app.overlay = mock.Mock()
        self.app._stop_server = mock.Mock()
        self.app._ensure_server = mock.Mock(return_value=True)
        self.app.settings.get().polish.enabled = True
        self.app.settings.get().polish.postproc = False

    def tearDown(self):
        try:
            self.app.ax.stop()
        except Exception:
            pass
        self.app._stopping.set()
        try:
            self.app.db.close()
        except Exception:
            pass
        self.app._tmp.cleanup()

    def _rows(self):
        return {m["key"]: m for m in self.app.control_snapshot()["modules"]}

    def test_no_row_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("myflow.main.models_dir", lambda: Path(tmp)):
                rows = self._rows()
        self.assertNotIn("postproc", rows, "без модели строка не должна показываться")

    def test_row_appears_with_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "myflow-postproc-qwen3-06b-q8_0.gguf").write_bytes(b"GGUF")
            with mock.patch("myflow.main.models_dir", lambda: Path(tmp)):
                rows = self._rows()
        self.assertIn("postproc", rows)
        self.assertFalse(rows["postproc"]["on"])
        self.assertIn("Qwen3-4B", rows["postproc"]["value"])

    def test_toggle_on_ensures_server_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "myflow-postproc-x.gguf").write_bytes(b"GGUF")
            with mock.patch("myflow.main.models_dir", lambda: Path(tmp)):
                ok, message = self.app.control_apply("module", "postproc")
        self.assertTrue(ok, message)
        self.assertTrue(self.app.settings.get().polish.postproc)
        self.app._ensure_server.assert_called_once_with("postproc")
        self.assertIn("включён", message)

    def test_toggle_on_without_model_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("myflow.main.models_dir", lambda: Path(tmp)):
                ok, message = self.app.control_apply("module", "postproc")
        self.assertFalse(ok)
        self.assertIn("не собрана", message)
        self.assertFalse(self.app.settings.get().polish.postproc)

    def test_toggle_on_refused_when_polish_disabled(self):
        self.app.settings.get().polish.enabled = False
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "myflow-postproc-x.gguf").write_bytes(b"GGUF")
            with mock.patch("myflow.main.models_dir", lambda: Path(tmp)):
                ok, message = self.app.control_apply("module", "postproc")
        self.assertFalse(ok)
        self.assertIn("полировка выключена", message)

    def test_toggle_off_stops_server_and_drops_client(self):
        self.app.settings.get().polish.postproc = True
        self.app._postproc_client = object()
        ok, message = self.app.control_apply("module", "postproc")
        self.assertTrue(ok, message)
        self.assertFalse(self.app.settings.get().polish.postproc)
        self.assertIsNone(self.app._postproc_client)
        self.app._stop_server.assert_called_once_with("postproc")
        self.assertIn("Qwen3-4B", message)
