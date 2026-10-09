"""Тесты классификатора команд (P0-1: LLM-интент не должен съедать диктовку)."""
import unittest

from myflow.commands import (
    VoiceCommand,
    KNOWN_KINDS,
    arg_looks_like_source,
    classify,
    intent_markers,
    is_llm_intent_candidate,
    llm_classify,
)


class TestExactAndPrefix(unittest.TestCase):
    def test_exact_commands(self):
        self.assertEqual(classify("стоп").kind, "stop")
        self.assertEqual(classify("скип.").kind, "skip")
        self.assertEqual(classify("удали последнее").kind, "delete_last")
        self.assertEqual(classify("новый абзац").kind, "new_paragraph")

    def test_send_does_not_rephrase(self):
        """P1-5: голый «отправь» = Enter, БЕЗ повторной вставки текста."""
        cmd = classify("отправь")
        self.assertEqual(cmd.kind, "send")
        self.assertFalse(cmd.rephrase_last, "«отправь» не должен перевставлять текст")

    def test_summarize_last_insert(self):
        cmd = classify("перескажи")
        self.assertEqual(cmd.kind, "summarize")
        self.assertTrue(cmd.rephrase_last)

    def test_prefix_keeps_original_case(self):
        """P1-6: аргумент не должен терять регистр/пунктуацию."""
        cmd = classify("перескажи Про Москву и Питер.")
        self.assertEqual(cmd.kind, "summarize")
        self.assertEqual(cmd.text, "Про Москву и Питер.")

    def test_prefix_safelist(self):
        self.assertIsNone(classify("список задач на завтра"))
        self.assertIsNone(classify("списком продуктов купить молоко"))
        self.assertEqual(classify("списком поставщиков цены").kind, "list")

    def test_send_with_argument_is_dictation(self):
        self.assertIsNone(classify("отправь документы Тане"))

    def test_list_inside_sentence_is_dictation(self):
        self.assertIsNone(classify("дай мне список продуктов"))


class TestIntentMarkers(unittest.TestCase):
    def test_word_boundaries(self):
        """P1-1: маркеры по границам слов, а не подстрокой."""
        self.assertEqual(intent_markers("краткосрочный план на неделю"), [])
        self.assertEqual(intent_markers("у меня стопка книг"), [])
        self.assertEqual(intent_markers("в списке дел на сегодня"), [])
        self.assertFalse(is_llm_intent_candidate("краткосрочный план"))
        self.assertFalse(is_llm_intent_candidate("стопка книг"))

    def test_real_markers(self):
        self.assertTrue(is_llm_intent_candidate("удали последнюю фразу"))
        self.assertTrue(is_llm_intent_candidate("кратко резюмируй встречу"))
        self.assertTrue(is_llm_intent_candidate("очисти историю"))
        self.assertFalse(is_llm_intent_candidate("завтра еду в аэропорт"))


class TestArgOverlap(unittest.TestCase):
    def test_overlap(self):
        self.assertTrue(arg_looks_like_source("встречу", "кратко резюмируй встречу"))
        self.assertFalse(arg_looks_like_source(
            "совершенно другой текст про погоду",
            "кратко резюмируй встречу"))

    def test_empty_arg(self):
        self.assertFalse(arg_looks_like_source("", "кратко резюмируй встречу"))


class TestLLMClassifyContract(unittest.TestCase):
    """P0-1: llm_classify НИКОГДА не возвращает kind вне KNOWN_KINDS.

    Проверяем без сети: подменяем requests.post внутри commands.
    """
    def _fake_post(self, payload_content):
        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": payload_content},
                                     "finish_reason": "stop"}]}

        def _post(url, json=None, timeout=None):  # noqa: A002 - как в requests
            return _Resp()

        return _post

    def _run(self, content, text="обычная диктовка про погоду"):
        from unittest import mock

        import requests
        # commands.llm_classify делает `import requests as _req` в момент
        # вызова, поэтому патч requests.post подхватывается.
        with mock.patch.object(requests, "post", self._fake_post(content)):
            return llm_classify(text, "http://127.0.0.1:8082/v1", "qwen")

    def test_text_kind_returns_none(self):
        cmd = self._run('{"kind":"text","text":"обычная диктовка про погоду"}')
        self.assertIsNone(cmd, "kind=text должен означать «обычная диктовка»")

    def test_unknown_kind_returns_none(self):
        self.assertIsNone(self._run('{"kind":"translate","text":"x"}'))

    def test_garbage_returns_none(self):
        self.assertIsNone(self._run("не json вообще"))

    def test_summarize_with_foreign_arg_returns_none(self):
        cmd = self._run('{"kind":"summarize","text":"совсем другой длинный текст"}',
                        text="брат, перескажи про встречу")
        self.assertIsNone(cmd, "модель переписала текст — это не команда")

    def test_summarize_with_matching_arg(self):
        cmd = self._run('{"kind":"summarize","text":"про встречу"}',
                        text="брат, перескажи про встречу")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "summarize")
        self.assertIn(cmd.kind, KNOWN_KINDS)

    def test_send_never_rephrases(self):
        cmd = self._run('{"kind":"send","text":""}')
        self.assertIsNotNone(cmd)
        self.assertFalse(cmd.rephrase_last)

    def test_markdown_fence_tolerated(self):
        cmd = self._run('```json\n{"kind":"stop","text":""}\n```')
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "stop")


if __name__ == "__main__":
    unittest.main()
