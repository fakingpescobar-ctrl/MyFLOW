"""Тесты подготовки данных для LoRA: маскирование промпта и паддинг батча.

Логика тонкая и молчаливо ломается: если забыть замаскировать промпт, модель
учится «пересказывать инструкцию» вместо оформления текста; если неверно
паддить батч, метки съезжают относительно входов. Поэтому проверяем на заглушке
токенизатора — без скачивания моделей и без GPU.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_postproc_lora", ROOT / "tools" / "train_postproc_lora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    """Токенизатор-заглушка: один символ = один токен, шаблон простой."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            **kwargs):
        # kwargs: enable_thinking=False передаётся тренером (обучение как в бою)
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) % 100 + 2 for ch in text]}


class MaskingTest(unittest.TestCase):
    def setUp(self):
        self.trainer = _load_trainer()
        self.tok = FakeTokenizer()
        self.row = {"messages": [
            {"role": "system", "content": "СИС"},
            {"role": "user", "content": "ПРИВЕТ"},
            {"role": "assistant", "content": "Привет."},
        ]}

    def test_prompt_is_masked(self):
        ex = self.trainer.to_example(self.tok, self.row, max_len=512)
        labels = ex["labels"]
        self.assertIn(-100, labels, "промпт должен быть замаскирован")
        # ответ обязан быть НЕ замаскирован
        self.assertTrue(any(x != -100 for x in labels), "ответ должен учиться")

    def test_answer_tail_is_unmasked(self):
        ex = self.trainer.to_example(self.tok, self.row, max_len=512)
        unmasked = [x for x in ex["labels"] if x != -100]
        expected = len(self.tok.apply_chat_template(
            [{"role": "assistant", "content": "Привет."}], tokenize=False)) - len("<assistant>")
        self.assertGreaterEqual(len(unmasked), expected - 2)
        self.assertLessEqual(len(unmasked), len("<assistant>Привет.</assistant>") + 2)

    def test_lengths_match(self):
        ex = self.trainer.to_example(self.tok, self.row, max_len=512)
        self.assertEqual(len(ex["input_ids"]), len(ex["labels"]))
        self.assertEqual(len(ex["input_ids"]), len(ex["attention_mask"]))

    def test_max_len_is_respected(self):
        ex = self.trainer.to_example(self.tok, self.row, max_len=10)
        self.assertLessEqual(len(ex["input_ids"]), 10)

    def test_masking_covers_whole_prompt(self):
        ex = self.trainer.to_example(self.tok, self.row, max_len=512)
        labels = ex["labels"]
        first_unmasked = next(i for i, x in enumerate(labels) if x != -100)
        self.assertGreater(first_unmasked, 0, "первые токены — это промпт, они маскированы")
        self.assertTrue(all(x == -100 for x in labels[:first_unmasked]))


class CollatorTest(unittest.TestCase):
    def setUp(self):
        self.trainer = _load_trainer()
        self.collator = self.trainer.Collator(pad_id=0)

    def test_pads_to_longest(self):
        batch = [{"input_ids": [5, 6], "attention_mask": [1, 1], "labels": [-100, 7]},
                 {"input_ids": [9], "attention_mask": [1], "labels": [8]}]
        out = self.collator(batch)
        self.assertEqual(out["input_ids"].shape, (2, 2))
        self.assertEqual(out["attention_mask"][1].tolist(), [1, 0], "паддинг не должен быть виден")
        self.assertEqual(out["labels"][1].tolist(), [8, -100], "паддинг не должен учиться")
        self.assertEqual(out["input_ids"][0].tolist(), [5, 6])

    def test_uses_pad_id(self):
        collator = self.trainer.Collator(pad_id=99)
        out = collator([{"input_ids": [5, 6], "attention_mask": [1, 1], "labels": [1, 2]},
                        {"input_ids": [7], "attention_mask": [1], "labels": [3]}])
        self.assertEqual(out["input_ids"][1].tolist(), [7, 99])


class DatasetFormatTest(unittest.TestCase):
    def test_real_dataset_rows_are_chat_format(self):
        import json
        path = ROOT / "data" / "train" / "postproc_train.jsonl"
        if not path.exists():
            self.skipTest("датасет не собран (нет data/train/postproc_train.jsonl)")
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()][:50]
        for row in rows:
            self.assertEqual([m["role"] for m in row["messages"]],
                             ["system", "user", "assistant"])

    def test_trainer_reads_the_dataset(self):
        trainer = _load_trainer()
        path = ROOT / "data" / "train" / "postproc_train.jsonl"
        if not path.exists():
            self.skipTest("датасет не собран")
        rows = trainer.load_rows(path)
        self.assertGreater(len(rows), 100)


if __name__ == "__main__":
    unittest.main()
