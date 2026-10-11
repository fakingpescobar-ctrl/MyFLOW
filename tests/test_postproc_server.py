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

    def test_prefers_best_quality_quantization(self):
        """Из нескольких квантизаций берём самую тяжёлую (она точнее).

        Алфавитный порядок выбирал Q6_K (472 МБ) вместо Q8_0 (610 МБ), а замер
        показал 0.113 против 0.098 — терять качество на ровном месте нельзя.
        """
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "myflow-postproc-qwen3-06b-q6_k.gguf").write_bytes(b"x" * 100)
            (Path(tmp) / "myflow-postproc-qwen3-06b-q8_0.gguf").write_bytes(b"x" * 300)
            with mock.patch("myflow.tui.models_dir", lambda: Path(tmp)):
                specs = {s.key: s for s in default_specs()}
        self.assertIn("q8_0", str(specs["postproc"].model))

    def test_both_servers_coexist(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "myflow-postproc-x.gguf").write_bytes(b"GGUF")
            with mock.patch("myflow.tui.models_dir", lambda: Path(tmp)):
                specs = {s.key: s for s in default_specs()}
        self.assertEqual(specs["llm"].port, 8082)
        self.assertEqual(specs["postproc"].port, 8083)
        self.assertNotEqual(specs["llm"].model, specs["postproc"].model)


class PolishClientSelectionTest(unittest.TestCase):
    def _app(self, model_available: bool = True):
        import types
        import myflow.main as M
        from test_main_flow import make_app
        app = make_app()
        # make_app() — заглушка: привязываем к ней НАСТОЯЩИЕ методы
        app._polish_client = types.MethodType(M.MyFlowApp._polish_client, app)
        app.postproc_model_available = lambda: model_available
        return app

    def test_falls_back_to_4b_when_model_missing(self):
        """Своей модели ещё нет — полируем 4B, чтобы функция работала из коробки."""
        app = self._app(model_available=False)
        app.polish.cfg.postproc = True
        self.assertIs(app._polish_client(), app.polish,
                      "без своей модели клиент должен остаться на 4B")

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
        app.postproc_model_available = lambda: True
        app.polish.cfg.enabled = False
        app.polish.cfg.postproc = True
        app.polish.cfg.url_postproc = "http://127.0.0.1:8083/v1"
        result = app._polish_safe("проверка связи", "plain", [])
        self.assertTrue(getattr(result, "skipped", False),
                        "при выключенной полировке постпроцессор не должен зваться")


class PolishRowShowsModelTest(unittest.TestCase):
    """Строка «Полировка» показывает, чем полируем; выбора модели больше нет.

    Решение пользователя: 4B держим ТОЛЬКО для Oracle LLM, полирует всегда своя
    0.6B (а если её ещё нет — откат на 4B).
    """

    def setUp(self):
        from test_panel import _app
        self.app = _app()
        self.app.overlay = mock.Mock()
        self.app.settings.get().polish.enabled = True

    def tearDown(self):
        try:
            self.app.ax.stop()
        except Exception:  # noqa: BLE001
            pass
        self.app._stopping.set()
        try:
            self.app.db.close()
        except Exception:  # noqa: BLE001
            pass
        self.app._tmp.cleanup()

    def _rows(self, available: bool):
        with mock.patch("myflow.main.models_dir", lambda: Path("/nonexistent")), \
             mock.patch.object(self.app, "postproc_model_available",
                               lambda: available):
            return {m["key"]: m for m in self.app.control_snapshot()["modules"]}

    def test_no_model_choice_row(self):
        rows = self._rows(available=True)
        self.assertNotIn("postproc", rows,
                         "выбора модели нет: полирует своя, 4B — только Oracle")

    def test_polish_row_names_own_model(self):
        rows = self._rows(available=True)
        self.assertIn("своя 0.6B", rows["llm"]["note"])

    def test_polish_row_admits_fallback_to_4b(self):
        rows = self._rows(available=False)
        self.assertIn("Qwen3-4B", rows["llm"]["note"])
        self.assertIn("своей ещё нет", rows["llm"]["note"])
