"""Авто-выбор движка распознавания: юнит-проверки решателя и интеграция.

Замеры на 624 реальных фразах (`data/eval_wispr/`):
  только GigaAM      — WER p50 0.083, задержка p50 57 мс
  только whisper     — WER p50 0.091, задержка p50 137 мс
  авто-выбор (этот)  — WER p50 0.071, задержка p50 67 мс (второе мнение в 20 % фраз)
"""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from myflow.asr_router import review
from myflow.config import AsrConfig


def cfg(**kwargs) -> AsrConfig:
    base = AsrConfig()
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


class ReviewTest(unittest.TestCase):
    def test_clean_text_stays_on_fast_engine(self):
        d = review("Проверка связи, один, два, три.", duration_ms=2500, cfg=cfg())
        self.assertFalse(d.use_second)

    def test_short_suspicious_tail_triggers_second(self):
        d = review("А что быы ты по.", duration_ms=1500, cfg=cfg())
        self.assertTrue(d.use_second)
        self.assertIn("обрывок", d.reason)

    def test_missing_terminal_punctuation_triggers_second(self):
        d = review("Второе, насчёт", duration_ms=2000, cfg=cfg())
        self.assertTrue(d.use_second)
        self.assertIn("пунктуац", d.reason)

    def test_slow_speech_rate_triggers_second(self):
        # 3 слова за 5 секунд = 0.6 сл/с — типичный признак пропуска слов
        d = review("Так вот оно что.", duration_ms=5000, cfg=cfg())
        self.assertTrue(d.use_second)
        self.assertIn("темп", d.reason)

    def test_rate_not_applied_to_very_short_phrases(self):
        """На короткой фразе «медленный темп» ничего не значит."""
        d = review("Так.", duration_ms=900, cfg=cfg(router_min_seconds=1.5))
        self.assertFalse(d.use_second)

    def test_fillers_only_when_enabled(self):
        text = "Типа, например, чтобы агент мог пройти авторизацию."
        self.assertFalse(review(text, duration_ms=3000, cfg=cfg()).use_second)
        self.assertTrue(review(text, duration_ms=3000,
                               cfg=cfg(router_use_fillers=True)).use_second)

    def test_long_phrase_rule_when_enabled(self):
        # 45 слов за 20 с = 2.25 сл/с: темп нормальный, сработать должно только
        # правило длины (иначе тест ловит срабатывание по темпу).
        text = ("Проверка длинной фразы идёт сейчас ровно чтобы посчитать темп речи "
                "и убедиться что правило длины срабатывает отдельно от темпа "
                "который здесь в норме и не должен мешать совсем никак вот так вот "
                "да именно так и происходит в жизни когда диктуешь долго и нудно "
                "продолжая говорить дальше и дальше без остановки и пауз вообще.")
        self.assertFalse(review(text, duration_ms=20000, cfg=cfg()).use_second)
        d = review(text, duration_ms=20000, cfg=cfg(router_long_seconds=15.0))
        self.assertTrue(d.use_second)
        self.assertIn("длинная", d.reason)

    def test_empty_text_does_not_trigger(self):
        self.assertFalse(review("", duration_ms=1000, cfg=cfg()).use_second)
        self.assertFalse(review("   ", duration_ms=1000, cfg=cfg()).use_second)

    def test_threshold_is_configurable(self):
        text = "Проверка связи идёт сейчас."          # 4 слова
        # 4 слова / 4 с = 1.0 сл/с
        self.assertTrue(review(text, duration_ms=4000, cfg=cfg(router_min_wps=1.5)).use_second)
        self.assertFalse(review(text, duration_ms=4000, cfg=cfg(router_min_wps=0.8)).use_second)


class TranscribeRoutingTest(unittest.TestCase):
    """`MyFlowApp._transcribe`: второе мнение зовётся только по подозрению."""

    def _app(self, primary_text: str, second_text: str = "текст от whisper."):
        from unittest import mock
        from test_main_flow import make_app

        app = make_app()
        # make_app() собирает приложение-заглушку и подменяет _transcribe простым
        # вызовом. Здесь проверяем НАСТОЯЩИЙ метод — привязываем его к заглушке.
        from myflow.main import MyFlowApp
        app._transcribe = types.MethodType(MyFlowApp._transcribe, app)
        app._ensure_second_opinion = types.MethodType(
            MyFlowApp._ensure_second_opinion, app)
        app.settings.get().asr.router = True
        primary = mock.Mock(backend_name="gigaam")
        primary.transcribe.return_value = types.SimpleNamespace(
            text=primary_text, language="ru", latency_ms=50)
        second = mock.Mock(backend_name="whisper")
        second.transcribe.return_value = types.SimpleNamespace(
            text=second_text, language="ru", latency_ms=120)
        second.alive.return_value = True
        app.whisper = primary
        app._second = second
        return app, primary, second

    def test_clean_phrase_uses_primary_only(self):
        app, primary, second = self._app("Проверка связи, один, два, три.")
        res, origin = app._transcribe(None, duration_ms=2500)
        self.assertEqual(origin, "gigaam")
        self.assertEqual(res.text, "Проверка связи, один, два, три.")
        second.transcribe.assert_not_called()

    def test_doubtful_phrase_asks_second_opinion(self):
        app, primary, second = self._app("А что быы ты по.")
        res, origin = app._transcribe(None, duration_ms=1500)
        second.transcribe.assert_called_once()
        self.assertEqual(res.text, "текст от whisper.")
        self.assertTrue(origin.startswith("whisper:"), origin)

    def test_router_disabled_never_asks_second(self):
        app, primary, second = self._app("А что быы ты по.")
        app.settings.get().asr.router = False
        res, origin = app._transcribe(None, duration_ms=1500)
        second.transcribe.assert_not_called()
        self.assertEqual(origin, "gigaam")
        self.assertEqual(res.text, "А что быы ты по.")

    def test_second_opinion_failure_falls_back_to_primary(self):
        app, primary, second = self._app("А что быы ты по.")
        second.transcribe.side_effect = RuntimeError("whisper-server недоступен")
        res, origin = app._transcribe(None, duration_ms=1500)
        self.assertEqual(origin, "gigaam")
        self.assertEqual(res.text, "А что быы ты по.")

    def test_empty_second_opinion_keeps_primary(self):
        app, primary, second = self._app("А что быы ты по.", second_text="   ")
        res, origin = app._transcribe(None, duration_ms=1500)
        self.assertEqual(origin, "gigaam")
        self.assertEqual(res.text, "А что быы ты по.")


if __name__ == "__main__":
    unittest.main()
