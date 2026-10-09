"""Тесты AX-наблюдения (UI Automation) и статистики/CLI.

Хелпер UIA подменяется крошечным скриптом по протоколу stdin/stdout, поэтому
тесты не зависят от Windows/десктопа и проверяют именно нашу логику.
"""
import base64
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from myflow.axtext import AxReader, AxText, AxWatcher
from myflow.db import Database, DictationRecord
from myflow.learning import record_edit_and_learn

FAKE_HELPER = textwrap.dedent('''
    import base64, json, sys, time
    TEXT = sys.argv[1] if len(sys.argv) > 1 else ""
    PID = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    MODE = sys.argv[3] if len(sys.argv) > 3 else "value"
    if MODE == "crash":
        sys.exit(1)
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "quit":
            break
        if line.strip() != "read":
            continue
        if MODE == "silent":
            continue                      # молчим: проверяем таймаут
        payload = {"ok": True, "source": "value", "pid": PID, "window": "Fake",
                   "text_b64": base64.b64encode(TEXT.encode("utf-8")).decode("ascii")}
        sys.stdout.write(json.dumps(payload) + "\\n")
        sys.stdout.flush()
    ''')
PROTOCOL_DOC = 1


def _helper(text: str, pid: int = 1234, mode: str = "value") -> list[str]:
    return [sys.executable, "-c", FAKE_HELPER, text, str(pid), mode]


class AxReaderTest(unittest.TestCase):
    def test_read_text(self):
        reader = AxReader(command=_helper("привет из поля"))
        self.assertTrue(reader.start(), reader.last_error)
        try:
            sample = reader.read()
            self.assertIsNotNone(sample)
            self.assertEqual(sample.text, "привет из поля")
            self.assertEqual(sample.pid, 1234)
            self.assertTrue(sample.reliable)
        finally:
            reader.stop()

    def test_unavailable_when_helper_dies(self):
        reader = AxReader(command=_helper("x", mode="crash"))
        self.assertFalse(reader.start())
        self.assertFalse(reader.available)
        self.assertTrue(reader.last_error)
        self.assertIsNone(reader.read())

    def test_timeout_when_silent(self):
        reader = AxReader(command=_helper("x", mode="silent"))
        self.assertFalse(reader.start(), "молчащий хелпер не должен считаться готовым")
        self.assertIsNone(reader.read(timeout=0.3))

    def test_parse_malformed_line(self):
        self.assertIsNone(AxReader._parse("не json"))
        self.assertIsNone(AxReader._parse(""))
        parsed = AxReader._parse(json.dumps({
            "ok": False, "source": "text", "pid": 7, "window": "W",
            "text_b64": base64.b64encode("Привет".encode()).decode()}))
        self.assertEqual(parsed.text, "Привет")
        self.assertFalse(parsed.reliable, "источник text помечается как ненадёжный")


class AxWatcherTest(unittest.TestCase):
    def _reader(self, samples):
        """Поддельный reader: отдаёт samples по очереди (последний — повтор)."""
        class Fake:
            def __init__(self):
                self.i = 0

            def read(self, timeout=None):
                s = samples[min(self.i, len(samples) - 1)]
                self.i += 1
                return s

        return Fake()

    def _watch(self, inserted, samples, **kw):
        w = AxWatcher(self._reader(samples), window_seconds=kw.pop("window", 1.0),
                      poll_interval=kw.pop("poll", 0.01))
        return w.watch(inserted, expected_pid=kw.pop("pid", 0))

    def test_edited_text_is_detected(self):
        base = "открыть эндпоинт на порту 8081 и запустить тест"
        samples = [AxText(text=base, source="value", pid=1),
                   AxText(text=base + " теперь", source="value", pid=1)]
        edited, reason = self._watch(base, samples)
        self.assertEqual(reason, AxWatcher.EDITED)
        self.assertIn("теперь", edited)

    def test_digits_rewrite_is_detected(self):
        """Живой кейс: «один два три» -> «1 2 3» меняет 3 слова из 5.

        Старая проверка перекрытия токенов (0.5) такое не признавала правкой.
        """
        base = "проверка связи один два три"
        edited, reason = self._watch(base, [AxText(text="проверка связи 1 2 3",
                                                   source="value", pid=1)])
        self.assertEqual(reason, AxWatcher.EDITED)
        self.assertEqual(edited, "проверка связи 1 2 3")

    def test_short_word_fix_is_detected(self):
        """«Провер» -> «Проверка»: divergence 1.0, но это правка того же поля."""
        edited, reason = self._watch("Провер", [AxText(text="Проверка", source="value", pid=1)])
        self.assertEqual(reason, AxWatcher.EDITED)

    def test_unchanged_then_timeout(self):
        base = "привет мир"
        edited, reason = self._watch(base, [AxText(text=base, source="value", pid=1)])
        self.assertIsNone(edited)
        self.assertEqual(reason, AxWatcher.TIMEOUT)

    def test_textbox_emptied(self):
        edited, reason = self._watch("привет мир", [AxText(text="", source="value", pid=1)])
        self.assertIsNone(edited)
        self.assertEqual(reason, AxWatcher.EMPTIED)

    def test_foreign_text_not_found(self):
        edited, reason = self._watch("привет мир",
                                     [AxText(text="совершенно другой текст", source="value", pid=1)])
        self.assertIsNone(edited)
        self.assertEqual(reason, AxWatcher.NOT_FOUND)

    def test_anchor_mismatch_on_other_process(self):
        edited, reason = self._watch("привет мир",
                                     [AxText(text="привет мир", source="value", pid=99)],
                                     pid=1)
        self.assertIsNone(edited)
        self.assertEqual(reason, AxWatcher.ANCHOR)

    def test_unavailable_when_reader_returns_none(self):
        class Dead:
            def read(self, timeout=None):
                return None

        w = AxWatcher(Dead(), window_seconds=0.2)
        edited, reason = w.watch("привет")
        self.assertIsNone(edited)
        self.assertEqual(reason, AxWatcher.UNAVAILABLE)


class ObservationReasonInHistoryTest(unittest.TestCase):
    def test_reason_is_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "s.sqlite")
            hid = db.insert_history(DictationRecord(
                asr_text="привет мир", polished_text="Привет, мир.",
                app_window="chat.exe | Telegram", duration_ms=1000))
            record_edit_and_learn(db, hid, "Привет, мир.", "Привет, мир!",
                                  observation_end_reason="edited")
            row = db._conn.execute(
                "SELECT observation_end_reason, edited_text FROM history WHERE id=?",
                (hid,)).fetchone()
            self.assertEqual(row["observation_end_reason"], "edited")
            self.assertIsNotNone(row["edited_text"])
            db.close()


class StatsAndRetryDbTest(unittest.TestCase):
    def test_stats_counts_errors_and_percentiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "s.db")
            for i, lat in enumerate([100, 200, 300, 400]):
                db.insert_history(DictationRecord(
                    asr_text=f"фраза {i}", polished_text=f"Фраза {i}.",
                    app_window="code.exe | x", duration_ms=1000,
                    e2e_latency_ms=lat, status="ok", divergence=0.1))
            db.insert_history(DictationRecord(
                asr_text="", polished_text="", app_window="code.exe | x",
                duration_ms=500, status="asr_error", audio_path="data/audio/1.pcm.z"))
            st = db.stats()
            self.assertEqual(st["dictations"], 5)
            self.assertEqual(st["errors"], 1)
            self.assertAlmostEqual(st["error_rate"], 0.2, places=3)
            self.assertEqual(st["p50_latency_ms"], 200)
            self.assertIn("asr_error", st["by_status"])
            row = db.last_retryable()
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "asr_error")
            db.mark_status(int(row["id"]), "retried")
            self.assertIsNone(db.last_retryable())
            db.close()

    def test_divergence_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "d.db")
            for d in (0.0, 1.0):
                db.insert_history(DictationRecord(
                    asr_text="a", polished_text="b", app_window="x",
                    duration_ms=100, divergence=d))
            self.assertAlmostEqual(db.stats()["avg_divergence"], 0.5, places=3)
            db.close()


if __name__ == "__main__":
    unittest.main()
