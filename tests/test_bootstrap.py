"""Самопочинка MyFLOW: приложение само находит, чего не хватает, и предлагает скачать.

Требования, которые проверяет этот файл:
  * недостающее находится точно — и пакеты, и файлы-модели;
  * БЕЗ согласия пользователя ничего не скачивается и не ставится (главное
    обещание модуля: сначала спрашиваем, потом качаем);
  * при согласии пакеты ставятся в то окружение, из которого запущено;
  * модуль лёгкий: его можно импортировать ДО ядра — иначе без numpy
    приложение упадёт на import раньше, чем сможет что-то предложить.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from myflow import bootstrap
from myflow.bootstrap import ModelFile, Package, blocking_gaps, confirm


def _package(name: str = "нет_такого", required: bool = True) -> Package:
    return Package(name, name, "для теста", required=required)


def _model(tmp: Path, name: str = "m.bin", required: bool = True) -> ModelFile:
    path = tmp / name
    return ModelFile(
        label="тест",
        url="https://example.invalid/model.bin",
        size_bytes=2048,
        why="для теста",
        destination=lambda: path,
        present=lambda: path.exists(),
        required=required,
    )


class DetectionTest(unittest.TestCase):
    def test_installed_module_is_found(self):
        self.assertTrue(Package("numpy", "numpy", "тест").installed)

    def test_absent_module_is_found(self):
        self.assertFalse(Package("нет", "модуля_нет_точно", "тест").installed)

    def test_model_present_follows_its_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _model(Path(tmp))
            self.assertFalse(model.present())
            (Path(tmp) / "m.bin").write_bytes(b"0" * 2048)
            self.assertTrue(model.present())

    def test_blocking_gaps_lists_only_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            gaps = blocking_gaps(
                packages=[
                    _package("обязательный"),
                    _package("желательный", required=False),
                ],
                models=[
                    _model(Path(tmp), "a.bin"),
                    _model(Path(tmp), "b.bin", required=False),
                ],
            )
        self.assertEqual(len(gaps), 2, gaps)
        self.assertIn("обязательный", gaps[0])
        self.assertNotIn("желательный", gaps[0])

    def test_report_says_what_and_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = bootstrap.format_gaps(
                packages=[_package()], models=[_model(Path(tmp))]
            )
        text = "\n".join(lines)
        self.assertIn("пакет", text)
        self.assertIn("для теста", text)
        self.assertIn("обязателен", text)


class ConsentTest(unittest.TestCase):
    """Главное обещание: без «да» ничего не происходит."""

    def setUp(self):
        self.downloaded: list[ModelFile] = []
        self.installed: list[Package] = []
        self._orig_download = bootstrap.download_model
        self._orig_install = bootstrap.install_packages
        self._orig_confirm = bootstrap.confirm
        bootstrap.download_model = lambda model, log=print: (
            self.downloaded.append(model) or True
        )
        bootstrap.install_packages = lambda packages, log=print: (
            self.installed.extend(packages) or True
        )

    def tearDown(self):
        bootstrap.download_model = self._orig_download
        bootstrap.install_packages = self._orig_install
        bootstrap.confirm = self._orig_confirm

    def test_nothing_downloaded_without_consent(self):
        bootstrap.confirm = lambda question: False
        with tempfile.TemporaryDirectory() as tmp:
            ok = bootstrap.preflight(
                log=lambda s: None, packages=[_package()], models=[_model(Path(tmp))]
            )
        self.assertEqual(self.downloaded, [])
        self.assertEqual(self.installed, [])
        self.assertFalse(ok, "без обязательного компонента запускать нельзя")

    def test_everything_downloaded_after_consent(self):
        bootstrap.confirm = lambda question: True
        with tempfile.TemporaryDirectory() as tmp:
            bootstrap.preflight(
                log=lambda s: None, packages=[_package()], models=[_model(Path(tmp))]
            )
        self.assertEqual(len(self.downloaded), 1)
        self.assertEqual(len(self.installed), 1)

    def test_optional_only_does_not_block_start(self):
        """Недостающее желательное — не повод не запускаться."""
        bootstrap.confirm = lambda question: False
        with tempfile.TemporaryDirectory() as tmp:
            ok = bootstrap.preflight(
                log=lambda s: None,
                packages=[_package("желательный", required=False)],
                models=[_model(Path(tmp), "b.bin", required=False)],
            )
        self.assertTrue(ok)

    def test_nothing_missing_means_no_question(self):
        asked: list[str] = []
        bootstrap.confirm = lambda question: asked.append(question) or True
        ok = bootstrap.preflight(log=lambda s: None, packages=[], models=[])
        self.assertTrue(ok)
        self.assertEqual(asked, [], "если всё на месте, спрашивать не о чем")


class InstallTest(unittest.TestCase):
    def test_pip_uses_current_interpreter(self):
        """Пакеты должны встать в ТО ЖЕ окружение, из которого запущено."""
        seen: list[list[str]] = []
        original = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            seen.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0)

        subprocess.run = fake_run
        try:
            bootstrap.install_packages([_package("numpy")], log=lambda s: None)
        finally:
            subprocess.run = original
        self.assertTrue(seen, "pip не вызван")
        self.assertEqual(seen[0][:4], [sys.executable, "-m", "pip", "install"])

    def test_install_of_nothing_is_success(self):
        self.assertTrue(bootstrap.install_packages([], log=lambda s: None))


class LightModuleTest(unittest.TestCase):
    def test_import_does_not_pull_heavy_dependencies(self):
        """Модуль зовут ДО ядра: тяжёлые импорты сломали бы сам отчёт."""
        code = (
            "import sys; import myflow.bootstrap; "
            "heavy = [m for m in ('numpy', 'sounddevice', 'onnxruntime') "
            "if m in sys.modules]; print(heavy)"
        )
        out = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True
        )
        self.assertEqual(
            out.stdout.strip(), "[]", f"bootstrap тянет тяжёлое: {out.stdout.strip()}"
        )


class IntegrationTest(unittest.TestCase):
    """Точки входа обязаны звать preflight — иначе «самочинилка» не сработает."""

    ROOT = Path(__file__).resolve().parent.parent

    def _text(self, rel: str) -> str:
        return (self.ROOT / rel).read_text(encoding="utf-8")

    def test_tui_calls_preflight(self):
        self.assertIn("preflight", self._text("myflow/tui.py"))

    def test_main_calls_preflight(self):
        self.assertIn("preflight", self._text("myflow/main.py"))

    def test_doctor_uses_bootstrap(self):
        doctor = self._text("tools/doctor.py")
        self.assertIn("myflow.bootstrap", doctor)
        self.assertIn("--fix", doctor)

    def test_bootstrap_knows_the_assets_the_app_uses(self):
        """Модели из bootstrap — те же, что реально использует приложение."""
        vad = self._text("myflow/vad.py")
        config = self._text("myflow/config.py")
        self.assertIn("silero_v6.onnx", vad)
        self.assertIn("ggml-large-v3-turbo", config)
        self.assertIn("qwen3-4b", config.lower())


if __name__ == "__main__":
    unittest.main()
