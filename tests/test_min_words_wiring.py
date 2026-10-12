"""Фильтр коротких диктовок обязан действовать на ВСЕ пути сборки датасета.

История ошибки (12.10.2026): три прогона подряд собрали одинаковый датасет
(366 реальных пар), потому что `--min-words` доходил только до одного вызова
`real_pairs()` — и то в функции, где не было `args`. Эксперимент проверял не то,
что задумано, а заметить это можно было только по цифрам через 20 минут.

Тесты герметичные: база подменяется временной, живая история не используется.
"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools import build_postproc_dataset as B  # noqa: E402


class MinWordsWiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        (root / "data").mkdir(parents=True)
        db = root / "data" / "myflow.sqlite"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, asr_text TEXT, "
                    "polished_text TEXT, edited_text TEXT)")
        rows = [
            # Короткие пары без правок: иначе max_edit отсеет их как «склейки»,
            # и тест будет проверять не фильтр коротких, а фильтр мусора.
            (1, "Да", "Да", "Да"),                                 # 1 слово
            (2, "Открой окно", "Открой окно", "Открой окно"),      # 2 слова
            (3, "сегодня мы читаем главу и смотрим где ошибки",
             "Сегодня мы читаем главу и смотрим, где ошибки.",
             "Сегодня мы читаем главу и смотрим, где ошибки."),  # длинная
            (4, "проверка связи идёт хорошо микрофон слышит каждое слово",
             "Проверка связи идёт хорошо, микрофон слышит каждое слово.",
             "Проверка связи идёт хорошо, микрофон слышит каждое слово."),
        ]
        con.executemany("INSERT INTO history VALUES (?,?,?,?)", rows)
        con.commit()
        con.close()
        self.root = root
        self._saved_root = B.ROOT
        B.ROOT = root

    def tearDown(self):
        B.ROOT = self._saved_root
        self.tmp.cleanup()

    def test_module_setting_filters_every_path(self):
        """MIN_WORDS действует, даже если real_pairs() вызван без аргументов."""
        all_pairs = B.real_pairs()
        self.assertEqual(len(all_pairs), 4)

        saved, B.MIN_WORDS = B.MIN_WORDS, 4
        try:
            filtered = B.real_pairs()
        finally:
            B.MIN_WORDS = saved
        self.assertEqual(len(filtered), 2, "короткие диктовки отсеяны")
        self.assertTrue(all(len(pair["dst"].split()) >= 4 for pair in filtered))

    def test_explicit_zero_overrides_module_setting(self):
        saved, B.MIN_WORDS = B.MIN_WORDS, 10
        try:
            pairs = B.real_pairs(min_words=0)
        finally:
            B.MIN_WORDS = saved
        self.assertEqual(len(pairs), 4, "явный min_words=0 сильнее настройки модуля")

    def test_clean_texts_respects_setting(self):
        """Синтетика строится из чистых текстов — там фильтр тоже нужен."""
        saved, B.MIN_WORDS = B.MIN_WORDS, 5
        try:
            texts = B.clean_texts(include_wispr=False)
        finally:
            B.MIN_WORDS = saved
        self.assertTrue(texts)
        self.assertTrue(all(len(text.split()) >= 5 for text in texts),
                        "в синтетику не должны попадать короткие тексты")


if __name__ == "__main__":
    unittest.main()
