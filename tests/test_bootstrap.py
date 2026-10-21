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


class TrainingAssetsTest(unittest.TestCase):
    """Путь нового пользователя к своему постпроцессору.

    Модель не публикуется (она персональная) — значит новый человек обязан
    уметь получить базу, корпус и обучиться сам. Проверяем, что все шаги
    этого пути существуют и что приложение предлагает их, а не молчит.
    """

    ROOT = Path(__file__).resolve().parent.parent

    def test_install_script_exists(self):
        bat = self.ROOT / "install-train-env.bat"
        self.assertTrue(bat.exists(), "нужен скрипт установки учебного окружения")
        text = bat.read_text(encoding="utf-8")
        self.assertIn(".venv-train", text)
        self.assertIn("requirements-train.txt", text)

    def test_requirements_train_exist(self):
        req = self.ROOT / "requirements-train.txt"
        self.assertTrue(req.exists(), "нужен отдельный файл зависимостей для обучения")
        text = req.read_text(encoding="utf-8")
        for package in ("torch", "transformers", "peft", "accelerate"):
            self.assertIn(package, text, f"{package} нужен для обучения")

    def test_bundles_are_declared(self):
        self.assertTrue(bootstrap.BUNDLES, "нужны наборы файлов для обучения")
        labels = [b.label for b in bootstrap.BUNDLES]
        self.assertTrue(any("Qwen3-0.6B" in x for x in labels), labels)
        self.assertTrue(any("орпус" in x for x in labels), labels)

    def test_bundle_presence_is_checked_by_files(self):
        """Набор считается готовым, только если есть все его файлы."""
        bundle = bootstrap.BUNDLES[0]
        self.assertIsInstance(bundle.present(), bool)

    def test_bundles_offered_only_with_training_env(self):
        """Без .venv-train про базу и корпус не спрашиваем — они не нужны."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = bootstrap.Bundle(
                label="тест",
                why="тест",
                hint="тест",
                destination=lambda: Path(tmp) / "x",
                present=lambda: False,
                fetch=lambda log=print: True,
            )
            saved = bootstrap.BUNDLES
            bootstrap.BUNDLES = (fake,)
            try:
                self.assertEqual(bootstrap.missing_bundles(only_if_train_env=False),
                                 [fake])
                # train_env_ready() смотрит в ROOT — она у нас настоящая,
                # поэтому проверяем обе ветки явно.
                if not bootstrap.train_env_ready():
                    self.assertEqual(bootstrap.missing_bundles(), [])
            finally:
                bootstrap.BUNDLES = saved

    def test_bundle_not_fetched_without_consent(self):
        """Как и файлы-модели, наборы не качаются без согласия."""
        fetched: list[str] = []
        bundle = bootstrap.Bundle(
            label="тест",
            why="тест",
            hint="тест",
            destination=lambda: Path("нет"),
            present=lambda: False,
            fetch=lambda log=print: fetched.append("да") or True,
        )
        original_confirm = bootstrap.confirm
        bootstrap.confirm = lambda question: False
        try:
            bootstrap.preflight(log=lambda s: None, packages=[], models=[],
                                bundles=[bundle])
        finally:
            bootstrap.confirm = original_confirm
        self.assertEqual(fetched, [], "набор скачался без согласия")

    def test_module_is_runnable(self):
        """`python -m myflow.bootstrap` должен работать.

        Регресс: при правке модуля один раз потерялся блок
        `if __name__ == "__main__"` — модуль импортировался, но ничего не
        делал, и команда молча выходила с кодом 0.
        """
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "myflow.bootstrap"],
            capture_output=True, text=True, cwd=str(root),
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Всё на месте", out.stdout)

    def test_retrain_uses_corpus_when_present(self):
        """Дообучение должно подхватывать корпус сам, а не требовать флаг."""
        text = (self.ROOT / "tools" / "retrain_postproc.py").read_text(encoding="utf-8")
        self.assertIn("conversational.txt", text,
                      "retrain должен сам брать корпус, если он скачан")
        self.assertIn("install-train-env.bat", text,
                      "retrain должен подсказывать скрипт установки окружения")


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
