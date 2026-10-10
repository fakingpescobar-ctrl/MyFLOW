"""Лёгкая чистка без LLM: филлеры, мусор в начале, пунктуация, заглавные."""
import sys
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from myflow.fastpath import looks_wellformed
from myflow.lightclean import light_clean


class LightCleanTest(unittest.TestCase):
    def test_removes_leading_fillers(self):
        cases = {
            "Ну что там, брат? Как дела?": "Что там, брат? Как дела?",
            "Короче, я двигаю твоё окно.": "Я двигаю твоё окно.",
            "Так, смотри, оно работает.": "Смотри, оно работает.",
            "эээ ну это самое, проверка связи.": "Это самое, проверка связи.",
            "Ну да, тогда делай всё по очереди.": "Да, тогда делай всё по очереди.",
        }
        for src, want in cases.items():
            with self.subTest(src=src):
                got, rules = light_clean(src)
                self.assertEqual(got, want)
                self.assertTrue(rules, "должны быть применённые правила")

    def test_keeps_meaningful_words(self):
        """«так»/«вот» без запятой — смысловые, их трогать нельзя."""
        for src in ("Так работает.", "Вот дом стоит.", "Так и должно быть."):
            with self.subTest(src=src):
                got, rules = light_clean(src)
                self.assertEqual(got, src)
                self.assertEqual(rules, [])

    def test_cleans_leading_noise_and_capitalizes(self):
        got, rules = light_clean("...проверка связи идёт")
        self.assertEqual(got, "Проверка связи идёт.")
        self.assertIn("убран мусор в начале", rules)

    def test_collapses_duplicate_punctuation(self):
        got, _ = light_clean("Это телевизоре.. Работает .")
        self.assertEqual(got, "Это телевизоре. Работает.")

    def test_capitalizes_after_sentence_end(self):
        got, rules = light_clean("Проверка связи. потому что важно.")
        self.assertEqual(got, "Проверка связи. Потому что важно.")
        self.assertIn("заглавная после точки", rules)

    def test_adds_trailing_period(self):
        got, rules = light_clean("проверка связи идёт")
        self.assertEqual(got, "Проверка связи идёт.")
        self.assertIn("добавлена точка", rules)

    def test_already_clean_text_untouched(self):
        for src in ("Проверка связи, один, два, три.",
                    "Сегодня хорошая погода, пойдём гулять.",
                    "Открой пожалуйста окно."):
            with self.subTest(src=src):
                got, rules = light_clean(src)
                self.assertEqual(got, src)

    def test_idempotent(self):
        once, _ = light_clean("ну ...проверка связи.. как дела")
        twice, rules = light_clean(once)
        self.assertEqual(once, twice)
        self.assertEqual(rules, [], "повторная чистка ничего не меняет")

    def test_empty_and_short_safe(self):
        for src in ("", "   ", "ну", "да."):
            with self.subTest(src=src):
                got, rules = light_clean(src)
                self.assertIsInstance(got, str)
                self.assertIsInstance(rules, list)

    def test_does_not_break_word_with_filler_prefix(self):
        """«нужно»/«такой» не должны попадать под правило филлеров."""
        for src in ("Нужно проверить связь.", "Такой вариант подходит."):
            with self.subTest(src=src):
                got, rules = light_clean(src)
                self.assertEqual(got, src)
                self.assertEqual(rules, [])

    def test_result_passes_fast_path(self):
        """Смысл всей затеи: после чистки текст считается оформленным."""
        for src in ("Ну что там, брат? Как дела?",
                    "Короче, я двигаю твоё окно и смотрю.",
                    "...проверка связи. потому что бывает",
                    "Ну да, тогда делай всё по очереди."):
            with self.subTest(src=src):
                got, rules = light_clean(src)
                self.assertTrue(rules)
                ok, why = looks_wellformed(got)
                self.assertTrue(ok, f"после чистки всё ещё не оформлено: {why}")


class LightCleanInPipelineTest(unittest.TestCase):
    """Фраза с филлером не должна звать LLM вообще."""

    def setUp(self):
        from unittest import mock
        import myflow.main as M
        from test_main_flow import make_app
        self.mock = mock
        self.app = make_app(polish_enabled=True)
        calls: list[str] = []

        def _no_llm(*a, **k):
            import traceback
            calls.append("".join(traceback.format_stack(limit=8)))
            raise AssertionError("LLM не должна вызываться при лёгкой чистке")

        self._llm_calls = calls
        self.app.polish.polish = mock.Mock(side_effect=_no_llm)
        r = mock.Mock()
        r.text = "Ну что там, брат? Как дела?"
        r.language = "ru"
        r.latency_ms = 120
        self.app.whisper = mock.Mock(backend_name="gigaam", health=lambda: True,
                                     transcribe=lambda *a, **k: r)
        # Стиль берём фиксированный: пайплайн спрашивает _resolve_style, а тот
        # смотрит на РЕАЛЬНОЕ активное окно (в терминале стиль terminal, там
        # полировка переформатирует текст и локальная чистка не применяется) —
        # из-за этого тест вёл себя по-разному в зависимости от фокуса.
        self.app._pick_style = lambda: "plain"
        self.app._resolve_style = lambda app: "plain"
        self.window_patch = self.mock.patch("myflow.main.app_descriptor",
                                            return_value="тест")
        self.window_patch.start()
        self.addCleanup(self.window_patch.stop)
        self.app.voice.check = lambda pcm: (True, None, "ок")

    def _run(self, text):
        from test_main_flow import Recording
        import numpy as np
        r = self.mock.Mock()
        r.text = text
        r.language = "ru"
        r.latency_ms = 120
        self.app.whisper.transcribe = lambda *a, **k: r
        rec = Recording(pcm_int16=np.zeros(32000, dtype=np.int16),
                        duration_ms=2000, max_amplitude=0.5)
        self.app._process_recording(0, rec)

    def test_filler_phrase_bypasses_llm(self):
        self._run("Ну что там, брат? Как дела?")
        self.assertFalse(self.app.polish.polish.called,
                         "LLM вызвана при лёгкой чистке, стек:\n"
                         + "\n".join(self._llm_calls))
        inserted = " ".join(self.app.paster.calls)
        self.assertIn("Что там", inserted, f"вставлено: {inserted!r}")
        statuses = [r.status for r in self.app.db.records]
        self.assertIn("local_polish", statuses, f"статусы: {statuses}")

    def test_phrase_needing_real_polish_still_uses_llm(self):
        """Опечатки/числа словами — по-прежнему в LLM."""
        self.app.polish.polish = self.mock.Mock(
            side_effect=AssertionError("LLM не должна вызываться"))
        # «один, два, три» — числа словами, чистка их не трогает -> должен пойти LLM
        try:
            self._run("Проверка связи, один, два, три.")
        except AssertionError:
            self.fail("фраза с числами словами должна идти в LLM, а не в чистку")


    def test_truncated_phrase_without_terminal_goes_to_llm(self):
        """Обрывок («... тормозит. То») должен дочищаться LLM, а не локально."""
        called = {"n": 0}

        def fake_polish(*a, **k):
            called["n"] += 1
            from myflow.polish import PolishResult
            return PolishResult(text="Наш TUI тормозит.", latency_ms=90)

        self.app.polish.polish = fake_polish
        self._run("Наш туи тормозит. То")
        self.assertEqual(called["n"], 1, "фраза без конечной пунктуации — в LLM")

    def test_filler_phrase_with_terminal_skips_llm(self):
        """А вот «Ну да, тогда делай всё по очереди.» — чисто механическая."""
        self._run("Ну да, тогда делай всё по очереди.")
        self.assertFalse(self.app.polish.polish.called,
                         "LLM вызвана при лёгкой чистке, стек:\n"
                         + "\n".join(self._llm_calls))

# ---------------------------------------------------------------------------
# Тесты маршрутизации (идёт фраза в LLM или нет) не должны зависеть от того,
# запущен ли сейчас llama-server: _polish_safe сначала спрашивает живость LLM и
# при мёртвом сервере честно уходит в быстрый отказ, не вызывая клиента.
# ---------------------------------------------------------------------------
_MODULE_PATCHES: list = []


def setUpModule() -> None:
    import myflow.main as _M
    p = mock.patch.object(_M, "_health_check_llm", return_value=True)
    p.start()
    _MODULE_PATCHES.append(p)


def tearDownModule() -> None:
    for patcher in reversed(_MODULE_PATCHES):
        patcher.stop()
    _MODULE_PATCHES.clear()


if __name__ == "__main__":
    unittest.main()


class StripInnerFillersTest(unittest.TestCase):
    """Зачистка филлеров внутри фразы (найдено стендом на диктовках Wispr)."""

    def test_removes_comma_set_fillers(self):
        from myflow.lightclean import strip_inner_fillers
        out, rules = strip_inner_fillers("Второе, значит, э-э, насчёт")
        self.assertEqual(out, "Второе, насчёт")
        self.assertTrue(rules)

    def test_keeps_meaningful_znachit_before_chto(self):
        """«это значит, что…» — не филлер: смысл ломать нельзя."""
        from myflow.lightclean import strip_inner_fillers
        src = "Так, это значит, что нужно проверить."
        out, rules = strip_inner_fillers(src)
        self.assertEqual(out, src)
        self.assertEqual(rules, [])

    def test_cleans_leading_pile(self):
        from myflow.lightclean import strip_inner_fillers
        out, _ = strip_inner_fillers("Ну, вот, короче, всё работает.")
        self.assertEqual(out, "Всё работает.")

    def test_idempotent(self):
        from myflow.lightclean import strip_inner_fillers
        once, _ = strip_inner_fillers("Второе, значит, э-э, насчёт")
        twice, _ = strip_inner_fillers(once)
        self.assertEqual(once, twice)

    def test_leaves_clean_text_alone(self):
        from myflow.lightclean import strip_inner_fillers
        src = "Проверка связи, один, два, три."
        self.assertEqual(strip_inner_fillers(src), (src, []))

    def test_does_not_break_numbers_or_names(self):
        from myflow.lightclean import strip_inner_fillers
        src = "Скиллы, MCP, GigaAM — всё на месте."
        self.assertEqual(strip_inner_fillers(src)[0], src)
