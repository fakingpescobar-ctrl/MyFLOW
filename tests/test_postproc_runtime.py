"""Рантайм постпроцессора: единый промпт, стиль postproc, подмена в пайплайне.

Страж от «дрейфа промпта»: дообученная модель запоминает формализм инструкции,
поэтому обучение (сборщик датасета) и боевой запуск обязаны использовать ОДИН
текст. Если кто-то поменяет промпт в одном месте — тест упадёт.
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from myflow.config import SettingsStore                     # noqa: E402
from myflow.postproc_prompt import POSTPROC_STYLE, POSTPROC_SYSTEM  # noqa: E402
from myflow.polish import PolishClient                      # noqa: E402


def _load_dataset_builder():
    spec = importlib.util.spec_from_file_location(
        "build_postproc_dataset", ROOT / "tools" / "build_postproc_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptUnityTest(unittest.TestCase):
    def test_dataset_and_runtime_use_same_prompt(self):
        builder = _load_dataset_builder()
        self.assertEqual(
            builder.SYSTEM_PROMPT, POSTPROC_SYSTEM,
            "промпт обучения и рантайма разошлись — модель будет получать не тот вход")

    def test_dataset_rows_carry_that_prompt(self):
        builder = _load_dataset_builder()
        row = builder.to_chat("проверка связи", "Проверка связи.")
        system = next(m["content"] for m in row["messages"] if m["role"] == "system")
        self.assertEqual(system, POSTPROC_SYSTEM)


class PostprocStyleTest(unittest.TestCase):
    def _client(self):
        cfg = SettingsStore().get().polish
        cfg.enabled = True
        cfg.postproc = True
        client = PolishClient(cfg)
        captured = {}

        class _Resp:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "Проверка связи."},
                                     "finish_reason": "stop"}]}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            return _Resp()

        client.session = mock.Mock()
        client.session.post.side_effect = fake_post
        # кэш имени модели «свежий», иначе клиент полезет в /models через мок
        import time as _time
        client._model_cache = "test-model"
        client._model_cache_t = _time.monotonic()
        return client, captured

    def test_postproc_style_sends_trained_prompt(self):
        client, captured = self._client()
        client.polish("проверка связи ну", style=POSTPROC_STYLE, dictionary=[])
        messages = captured["payload"]["messages"]
        system = next(m["content"] for m in messages if m["role"] == "system")
        self.assertEqual(system, POSTPROC_SYSTEM,
                         "для стиля postproc должен уходить промпт обучения")
        self.assertNotIn("ЗАПРЕЩЕНО удалять осмысленные слова", system,
                         "правила обычной полировки не должны подмешиваться")

    def test_plain_style_still_uses_regular_prompt(self):
        client, captured = self._client()
        client.polish("проверка связи ну", style="plain", dictionary=[])
        system = next(m["content"] for m in captured["payload"]["messages"]
                      if m["role"] == "system")
        self.assertNotEqual(system, POSTPROC_SYSTEM)
        self.assertIn("пост-обработка распознанной речи", system)


class PipelinePostprocTest(unittest.TestCase):
    def test_polish_safe_switches_style_when_enabled(self):
        import myflow.main as M
        from test_main_flow import make_app

        app = make_app()
        app.settings.get().polish.postproc = True
        app.polish.cfg.enabled = True
        app.polish.cfg.postproc = True
        seen = {}

        def fake_polish(text, style="plain", dictionary=None):
            seen["style"] = style
            return types.SimpleNamespace(text=text, latency_ms=1, skipped=False)

        # клиент подменяем целиком: у него свой cfg (enabled) и свой polish
        fake_client = types.SimpleNamespace(
            cfg=types.SimpleNamespace(enabled=True, postproc=True), polish=fake_polish)
        app._polish_client = lambda: fake_client
        with mock.patch.object(M, "_health_check_llm", lambda *a, **k: True):
            app._polish_safe("проверка связи ну", "plain", [])
        self.assertEqual(seen.get("style"), POSTPROC_STYLE,
                         "при включённом postproc пайплайн должен звать его стиль")

    def test_polish_safe_keeps_style_when_disabled(self):
        import myflow.main as M
        from test_main_flow import make_app

        app = make_app()
        app.settings.get().polish.postproc = False
        app.polish.cfg.enabled = True
        app.polish.cfg.postproc = False
        seen = {}

        def fake_polish(text, style="plain", dictionary=None):
            seen["style"] = style
            return types.SimpleNamespace(text=text, latency_ms=1, skipped=False)

        fake_client = types.SimpleNamespace(
            cfg=types.SimpleNamespace(enabled=True, postproc=False), polish=fake_polish)
        app._polish_client = lambda: fake_client
        with mock.patch.object(M, "_health_check_llm", lambda *a, **k: True):
            app._polish_safe("проверка связи ну", "plain", [])
        self.assertEqual(seen.get("style"), "plain")

    def test_config_default_is_off(self):
        """По умолчанию режим выключен: дообученной модели может ещё не быть.

        Настройки берём из ВРЕМЕННОГО файла: живые пользователь мог переключить
        (и переключил — включил постпроцессор), тест не должен от этого зависеть.
        """
        import tempfile
        from pathlib import Path as _Path
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(_Path(tmp) / "settings.json")
        self.assertFalse(store.get().polish.postproc)


if __name__ == "__main__":
    unittest.main()


class InputFormatTest(unittest.TestCase):
    """Формат входа постпроцессора: найденный баг ломал качество втрое.

    Боевой путь оборачивал текст в <dictation>, а обучение шло на голом тексте.
    Замер: 0.618 против 0.262 после снятия обёртки.
    """

    def _client(self, captured):
        from myflow.config import SettingsStore
        from myflow.polish import PolishClient

        cfg = SettingsStore().get().polish
        cfg.enabled = True
        client = PolishClient(cfg)
        import time as _time
        client._model_cache = "m"
        client._model_cache_t = _time.monotonic()

        class _Resp:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "Привет."}}]}

        def fake_post(url, json=None, timeout=None):
            captured.update(json)
            return _Resp()

        client.session = mock.Mock()
        client.session.post.side_effect = fake_post
        return client

    def test_postproc_style_sends_bare_text(self):
        captured = {}
        client = self._client(captured)
        client.polish("привет ну как дела", style=POSTPROC_STYLE, dictionary=[])
        user = next(m["content"] for m in captured["messages"] if m["role"] == "user")
        self.assertEqual(user, "привет ну как дела",
                         "постпроцессор обучался на голом тексте — обёртки быть не должно")
        self.assertNotIn("<dictation>", user)

    def test_plain_style_keeps_wrapper(self):
        captured = {}
        client = self._client(captured)
        client.polish("привет ну как дела", style="plain", dictionary=[])
        user = next(m["content"] for m in captured["messages"] if m["role"] == "user")
        self.assertIn("<dictation>", user,
                      "обычная полировка ждёт обёртку — её поведение менять нельзя")
