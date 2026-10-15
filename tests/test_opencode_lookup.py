"""Поиск процессов opencode не должен падать на не-UTF-8 выводе.

Случилось именно это: PowerShell отдавал кириллицу в своей кодировке, text=True
декодировал строго в UTF-8, и UnicodeDecodeError убивал поток чтения - TUI падал с
трейсбеком. Теперь вывод читается с errors='replace'.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import myflow.opencode_inject as OI  # noqa: E402


class LookupTest(unittest.TestCase):
    def test_lookup_does_not_raise(self):
        pids = OI._find_opencode_pids()
        self.assertIsInstance(pids, list)
        self.assertTrue(all(isinstance(p, int) for p in pids))

    def test_uses_replace_errors(self):
        """Защита от повторения: чтение без падения на кривых байтах."""
        source = (ROOT / "myflow" / "opencode_inject.py").read_text(encoding="utf-8")
        self.assertIn('errors="replace"', source,
                      "вывод подпроцессов читаем с errors='replace'")

    def test_bad_bytes_do_not_break_decoding(self):
        """Тот же приём, что в модуле: байт 0x98 ломает строгий UTF-8."""
        raw = b"opencode \x98 \xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82"
        with self.assertRaises(UnicodeDecodeError):
            raw.decode("utf-8")
        self.assertIn("opencode", raw.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    unittest.main()
