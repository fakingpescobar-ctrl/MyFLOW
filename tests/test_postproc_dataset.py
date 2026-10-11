"""Датасет для дообучения постпроцессора: утечка, детерминизм, качество пар.

Утечка в этом деле — главный риск: если цель валидационной фразы попадёт в
обучение (а одну и ту же фразу пользователь диктовал не раз), оценка превратится
в проверку памяти. Первая версия сборщика именно так и делала (60 из 64 целей).
"""
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "build_postproc_dataset", ROOT / "tools" / "build_postproc_dataset.py")
bpd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bpd)

WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def norm(text: str) -> str:
    return " ".join(w.replace("ё", "е") for w in WORD.findall((text or "").lower()))


def parts(row: dict) -> tuple[str, str]:
    src = next(m["content"] for m in row["messages"] if m["role"] == "user")
    dst = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    return norm(src), norm(dst)


class DatasetBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.stats = bpd.build(out_dir=cls.out, identity_share=0.45)
        cls.train = [json.loads(l) for l in
                     (cls.out / "postproc_train.jsonl").read_text(encoding="utf-8").splitlines()
                     if l.strip()]
        cls.val = [json.loads(l) for l in
                   (cls.out / "postproc_val.jsonl").read_text(encoding="utf-8").splitlines()
                   if l.strip()]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dataset_is_not_empty(self):
        self.assertGreater(len(self.train), 500)
        self.assertGreaterEqual(len(self.val), 20)

    def test_no_validation_target_leaks_into_training(self):
        train_answers = {dst for _, dst in map(parts, self.train)}
        leaks = [dst for _, dst in map(parts, self.val) if dst in train_answers]
        self.assertEqual(leaks, [], f"утечка целей: {len(leaks)}")

    def test_no_validation_input_leaks_into_training(self):
        train_inputs = {src for src, _ in map(parts, self.train)}
        leaks = [src for src, _ in map(parts, self.val) if src in train_inputs]
        self.assertEqual(leaks, [], f"утечка входов: {len(leaks)}")

    def test_no_duplicate_change_pairs_in_training(self):
        """Дубли допустимы ТОЛЬКО среди «ничего не менять»: это взвешивание класса.

        Перекос был настоящей проблемой: при 86 % примеров «поменяй» модель правила
        даже готовый текст (0.104 против 0.029 у «не трогать»), поэтому идентичности
        повторяются намеренно.
        """
        pairs = list(map(parts, self.train))
        change = [p for p in pairs if p[0] != p[1]]
        identity = [p for p in pairs if p[0] == p[1]]
        self.assertEqual(len(change), len(set(change)), "дубли среди пар с правкой")
        self.assertGreaterEqual(len(identity), len(change) * 0.2,
                                "идентичностей слишком мало для баланса")

    def test_every_example_has_system_prompt(self):
        for row in self.train[:200]:
            roles = [m["role"] for m in row["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant"])

    def test_identity_examples_present(self):
        """Учим не трогать уже оформленный текст — иначе модель будет всё переписывать."""
        identity = [1 for src, dst in map(parts, self.train) if src == dst]
        self.assertGreater(len(identity), 50, "нет примеров «ничего не менять»")

    def test_build_is_deterministic(self):
        other = Path(self.tmp.name) / "again"
        bpd.build(out_dir=other, identity_share=0.45)     # те же параметры, что в setUpClass
        a = (self.out / "postproc_train.jsonl").read_text(encoding="utf-8")
        b = (other / "postproc_train.jsonl").read_text(encoding="utf-8")
        self.assertEqual(a, b, "при том же seed датасет должен совпадать")

    def test_validation_is_stable_by_id(self):
        """Одни и те же диктовки остаются в валидации при пересборке."""
        v1 = {parts(r)[1] for r in self.val}
        other = Path(self.tmp.name) / "stable"
        bpd.build(out_dir=other, identity_share=0.45)
        val2 = [json.loads(l) for l in
                (other / "postproc_val.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
        self.assertEqual(v1, {parts(r)[1] for r in val2})


class CorruptionTest(unittest.TestCase):
    def test_digits_become_words(self):
        import random
        out = bpd.corrupt("Проверил 3 раза.", random.Random(1))
        self.assertIn("три", out.lower())

    def test_fillers_can_appear(self):
        import random
        outs = [bpd.corrupt("Проверка связи идёт сейчас нормально.", random.Random(i))
                for i in range(30)]
        self.assertTrue(any(any(f in o.lower() for f in ("ну,", "э-э,", "значит,", "типа,"))
                            for o in outs), "филлеры должны иногда появляться")

    def test_finalize_adds_period_and_capital(self):
        self.assertEqual(bpd.finalize_clean("проверка связи"), "Проверка связи.")
        self.assertEqual(bpd.finalize_clean("Уже готово!"), "Уже готово!")

    def test_finalize_collapses_spaces(self):
        self.assertEqual(bpd.finalize_clean("  два   слова  "), "Два слова.")
