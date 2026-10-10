"""Тесты освобождения памяти: серверы не переживают закрытие MyFLOW.

Ключевой сценарий пользователя: закрыл окно MyFLOW (или процесс убили силой) —
в памяти и VRAM не должно остаться ни llama-server, ни whisper-server.
"""
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from myflow.config import ROOT
from myflow.jobobj import ProcessJob
from myflow.tui import Supervisor, default_specs


class ProcessJobTest(unittest.TestCase):
    def test_job_is_created_on_windows(self):
        job = ProcessJob()
        self.assertTrue(job.active, f"job не создался: {job.error}")
        self.assertIsNone(job.error)

    def test_assign_real_child(self):
        job = ProcessJob()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(job.assign(child), f"привязка не удалась: {job.error}")
            self.assertEqual(job.assigned, 1)
        finally:
            child.terminate()
            child.wait(timeout=10)

    def test_assign_without_handle(self):
        job = ProcessJob()
        self.assertFalse(job.assign(None))
        fake = mock.Mock()
        fake._handle = 0
        self.assertFalse(job.assign(fake))

    def test_degradation_without_api(self):
        """Если Job Object недоступен — работаем как раньше, без падения."""
        with mock.patch("ctypes.windll", create=True) as win:
            win.kernel32.CreateJobObjectW.return_value = 0
            job = ProcessJob()
        self.assertFalse(job.active)
        self.assertIsNotNone(job.error)
        self.assertFalse(job.assign(mock.Mock()))


class SupervisorJobTest(unittest.TestCase):
    def test_supervisor_has_job(self):
        sup = Supervisor(default_specs(), enabled=True)
        self.assertTrue(sup.job.active, "у Supervisor должен быть Job Object")

    def test_started_process_is_assigned_to_job(self):
        """Каждый запущенный нами сервер привязывается к job'у."""
        specs = default_specs()
        sup = Supervisor(specs, enabled=True)
        spec = specs[0]
        sup.job = mock.Mock(active=True, assigned=0)
        proc = mock.Mock()
        proc.poll.return_value = None
        with mock.patch("subprocess.Popen", return_value=proc), \
             mock.patch.object(type(sup), "port_open", staticmethod(lambda *a, **k: False)):
            self.assertTrue(sup.start(spec))
        sup.job.assign.assert_called_once_with(proc)


class ShutdownAssetsTest(unittest.TestCase):
    """Скрипты и .bat должны гасить серверы проекта и не трогать чужие."""

    def _text(self, rel: str) -> str:
        return (ROOT / rel).read_text(encoding="utf-8-sig")

    def test_stop_script_exists_and_filters_by_project_path(self):
        text = self._text("scripts/stop-servers.ps1")
        self.assertIn("llama-server.exe", text)
        self.assertIn("whisper-server.exe", text)
        # Фильтр по пути проекта: чужие сборки llama.cpp не задеваем.
        self.assertIn("ExecutablePath", text)
        self.assertIn("StartsWith", text)

    def test_bat_stops_servers_before_and_after(self):
        text = self._text("start-myflow.bat")
        self.assertGreaterEqual(text.count("stop-servers.ps1"), 2,
                                "нужно гасить хвосты и до, и после TUI")

    def test_scripts_have_utf8_bom(self):
        """Windows PowerShell 5.1 читает файл без BOM как ANSI — русский в мусор."""
        for f in (ROOT / "scripts").glob("*.ps1"):
            with self.subTest(script=f.name):
                self.assertTrue(f.read_bytes().startswith(b"\xef\xbb\xbf"),
                                f"{f.name}: нет UTF-8 BOM")

    def test_no_other_drive_paths_in_scripts(self):
        for f in (ROOT / "scripts").glob("*.ps1"):
            text = f.read_text(encoding="utf-8-sig")
            self.assertNotIn("E:\\llm", text, f"{f.name}: внешний путь")


if __name__ == "__main__":
    unittest.main()
