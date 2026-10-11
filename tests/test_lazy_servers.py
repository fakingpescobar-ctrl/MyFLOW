"""Ленивая загрузка серверов: держим в памяти только нужную модель.

Пользователь справедливо спросил: зачем две LLM? Ответ — они АЛЬТЕРНАТИВЫ для
полировки (0.6B ИЛИ 4B), а 4B дополнительно нужен только Oracle LLM. Значит, если
полировка на 0.6B и Oracle выключен, 4 ГБ модели грузить незачем. Раньше пульт
всё равно поднимал все серверы при старте.
"""
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from myflow.tui import needed_servers                # noqa: E402


def settings(polish_enabled: bool, postproc: bool, judge: bool):
    return types.SimpleNamespace(
        polish=types.SimpleNamespace(enabled=polish_enabled, postproc=postproc),
        asr=types.SimpleNamespace(judge=judge))


class NeededServersTest(unittest.TestCase):
    def test_nothing_enabled_keeps_only_asr(self):
        self.assertEqual(needed_servers(settings(False, False, False)), {"asr"})

    def test_postproc_polish_does_not_load_4b(self):
        need = needed_servers(settings(True, True, False))
        self.assertEqual(need, {"asr", "postproc"})
        self.assertNotIn("llm", need, "полирует 0.6B — 4B незачем грузить")

    def test_4b_polish_loads_llm(self):
        self.assertEqual(needed_servers(settings(True, False, False)), {"asr", "llm"})

    def test_oracle_loads_llm(self):
        self.assertEqual(needed_servers(settings(False, False, True)), {"asr", "llm"})

    def test_oracle_plus_postproc_need_both(self):
        self.assertEqual(needed_servers(settings(True, True, True)),
                         {"asr", "llm", "postproc"})

    def test_survives_settings_without_attributes(self):
        """Старые конфиги (без postproc) не должны ломать расчёт."""
        old = types.SimpleNamespace(polish=types.SimpleNamespace(enabled=True),
                                    asr=types.SimpleNamespace())
        self.assertEqual(needed_servers(old), {"asr", "llm"})


class SkippedServersNoteTest(unittest.TestCase):
    def test_skipped_server_is_marked_as_not_needed(self):
        from myflow.tui import ServerSpec, Supervisor
        spec = ServerSpec(key="llm", title="LLM", port=8082,
                          url="http://127.0.0.1:8082", health_path="/v1/models",
                          exe=Path("x"), model=Path("y"), args=[], log=Path("z"))
        sup = Supervisor([spec], enabled=True)
        sup._skipped_servers = {"llm"}          # так помечает фаза загрузки
        snap = sup.health_snapshot()
        self.assertIn("не нужен", snap["llm"]["note"])


class PrewarmTargetTest(unittest.TestCase):
    """Прогрев поднимает ту модель, которая будет полировать."""

    def _app(self, postproc: bool):
        import myflow.main as M
        from test_main_flow import make_app
        app = make_app()
        app._prewarm_llm = types.MethodType(M.MyFlowApp._prewarm_llm, app)
        app.polish.cfg.enabled = True
        app.polish.cfg.postproc = postproc
        app.polish.cfg.auto_start_server = True
        started = []
        app._ensure_server = lambda key: (started.append(key), True)[1]
        return app, started

    def _run_sync(self, app):
        """Выполнить поток прогрева синхронно (иначе тест не дождётся)."""
        class _Sync:
            """Заглушка Thread: выполняет цель сразу, с её аргументами."""

            def __init__(self, target=None, daemon=None, name=None,
                         args=(), kwargs=None, **extra):
                self._target = target
                self._args = tuple(args or ())
                self._kwargs = dict(kwargs or {})

            def start(self):
                if self._target:
                    self._target(*self._args, **self._kwargs)

        with mock.patch("myflow.main.threading.Thread", _Sync):
            app._prewarm_llm()

    def test_prewarm_targets_postproc_when_enabled(self):
        app, started = self._app(postproc=True)
        self._run_sync(app)
        self.assertEqual(started, ["postproc"], "должны грузить 0.6B, а не 4 ГБ")

    def test_prewarm_targets_llm_otherwise(self):
        app, started = self._app(postproc=False)
        self._run_sync(app)
        self.assertEqual(started, ["llm"])


if __name__ == "__main__":
    unittest.main()
