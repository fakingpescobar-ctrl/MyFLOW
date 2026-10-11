"""Дообучение из пульта: строка «Дообучение», индикация и запуск.

Каждая диктовка — размеченный пример («что услышали» -> «что осталось»), поэтому
пульт должен показывать, сколько свежих накопилось, и запускать дообучение в фоне,
не мешая диктовать. Тесты герметичны: работают на временном каталоге вместо
живых data/ и myflow.sqlite.
"""
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import myflow.main as M  # noqa: E402


def make_world(dictations: int = 150, trained_max_id: int | None = None) -> Path:
    """Временный «мир»: база с диктовками, состояние обучения, учебное окружение."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data").mkdir()
    db = tmp / "data" / "myflow.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, asr_text TEXT, "
                "polished_text TEXT, edited_text TEXT)")
    for i in range(1, dictations + 1):
        con.execute("INSERT INTO history VALUES (?,?,?,?)",
                    (i, f"фраза номер {i}", f"Фраза номер {i}.", ""))
    con.commit()
    con.close()
    if trained_max_id is not None:
        (tmp / "data" / "train").mkdir(parents=True)
        (tmp / "data" / "train" / "state.json").write_text(json.dumps({
            "trained_at": "2026-10-11 18:00:00", "dictations": trained_max_id,
            "max_id": trained_max_id, "model": "myflow-postproc.gguf",
        }, ensure_ascii=False), encoding="utf-8")
    (tmp / "tools").mkdir()
    (tmp / "tools" / "retrain_postproc.py").write_text("# заглушка", encoding="utf-8")
    (tmp / ".venv-train" / "Scripts").mkdir(parents=True)
    (tmp / ".venv-train" / "Scripts" / "python.exe").write_bytes(b"")
    return tmp


class RetrainStateTest(unittest.TestCase):
    def _app(self, world: Path):
        from test_main_flow import make_app
        app = make_app()
        app._train_python = types.MethodType(M.MyFlowApp._train_python, app)
        app.retrain_state = types.MethodType(M.MyFlowApp.retrain_state, app)
        app.start_retrain = types.MethodType(M.MyFlowApp.start_retrain, app)
        return app, mock.patch("myflow.main.ROOT", world)

    def test_counts_new_dictations_since_last_training(self):
        world = make_world(dictations=150, trained_max_id=100)
        app, patcher = self._app(world)
        with patcher:
            state = app.retrain_state()
        self.assertEqual(state["dictations"], 150)
        self.assertEqual(state["new"], 50)
        self.assertEqual(state["trained_on"], 100)
        self.assertTrue(state["env_ready"])

    def test_without_state_all_dictations_are_new(self):
        world = make_world(dictations=42)
        app, patcher = self._app(world)
        with patcher:
            state = app.retrain_state()
        self.assertEqual(state["new"], 42)
        self.assertEqual(state["trained_on"], 0)

    def test_env_missing_is_reported(self):
        world = make_world(dictations=10)
        (world / ".venv-train" / "Scripts" / "python.exe").unlink()
        app, patcher = self._app(world)
        with patcher:
            self.assertFalse(app.retrain_state()["env_ready"])


class RetrainRowTest(unittest.TestCase):
    def _rows(self, world: Path):
        from test_panel import _app
        app = _app()
        with mock.patch("myflow.main.ROOT", world):
            rows = {m["key"]: m for m in app.control_snapshot()["modules"]}
        app._stopping.set()
        try:
            app.db.close()
        except Exception:  # noqa: BLE001
            pass
        app._tmp.cleanup()
        return rows

    def test_row_shows_progress_toward_threshold(self):
        rows = self._rows(make_world(dictations=30, trained_max_id=10))
        row = rows["retrain"]
        self.assertIn("+20", row["value"])         # новых диктовок с прошлого раза
        self.assertIn("100", row["value"])         # и порог, после которого пора
        self.assertFalse(row["on"])

    def test_row_asks_to_retrain_when_threshold_reached(self):
        rows = self._rows(make_world(dictations=200, trained_max_id=50))
        row = rows["retrain"]
        self.assertIn("пора", row["value"])
        self.assertIn("150", row["value"])

    def test_row_hidden_without_training_env(self):
        world = make_world(dictations=30)
        (world / ".venv-train" / "Scripts" / "python.exe").unlink()
        rows = self._rows(world)
        self.assertNotIn("retrain", rows, "без учебного окружения строка бесполезна")


class RetrainStartTest(unittest.TestCase):
    def _app(self, world: Path):
        from test_main_flow import make_app
        app = make_app()
        for name in ("_train_python", "retrain_state", "start_retrain"):
            setattr(app, name, types.MethodType(getattr(M.MyFlowApp, name), app))
        return app

    def test_start_refuses_without_env(self):
        world = make_world(dictations=10)
        (world / ".venv-train" / "Scripts" / "python.exe").unlink()
        app = self._app(world)
        with mock.patch("myflow.main.ROOT", world):
            ok, message = app.start_retrain()
        self.assertFalse(ok)
        self.assertIn("учебного окружения", message)

    def test_start_spawns_background_process(self):
        world = make_world(dictations=10)
        app = self._app(world)
        fake = mock.Mock()
        fake.poll.return_value = None
        with mock.patch("myflow.main.ROOT", world), \
             mock.patch("subprocess.Popen", return_value=fake) as popen:
            ok, message = app.start_retrain()
        self.assertTrue(ok, message)
        self.assertIn("запущено", message)
        popen.assert_called_once()
        started = popen.call_args[0][0]
        self.assertIn("retrain_postproc.py", " ".join(str(x) for x in started))

    def test_second_start_is_refused_while_running(self):
        world = make_world(dictations=10)
        app = self._app(world)
        fake = mock.Mock()
        fake.poll.return_value = None
        app._retrain_proc = fake
        with mock.patch("myflow.main.ROOT", world):
            ok, message = app.start_retrain()
        self.assertFalse(ok)
        self.assertIn("уже идёт", message)

    def test_row_marks_running_process(self):
        from test_panel import _app as panel_app
        world = make_world(dictations=10)
        app = panel_app()
        app._retrain_proc = mock.Mock()
        app._retrain_proc.poll.return_value = None
        with mock.patch("myflow.main.ROOT", world):
            rows = {m["key"]: m for m in app.control_snapshot()["modules"]}
        self.assertTrue(rows["retrain"]["on"])
        self.assertIn("идёт", rows["retrain"]["value"])
        app._stopping.set()
        try:
            app.db.close()
        except Exception:  # noqa: BLE001
            pass
        app._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()


class InstallGuardTest(unittest.TestCase):
    """Защита от автоматической установки ХУДШЕЙ модели.

    Реальный случай: обучение на всех диктовках (включая последние 80 — короткие и
    уже чистые) дало 0.170 против 0.140 у текущей. Без защиты такая модель уехала бы
    в пульт молча.
    """

    def _should_install(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from compare_postproc_models import should_install
        return should_install

    def test_better_model_is_installed(self):
        self.assertTrue(self._should_install()(0.13, 0.14))

    def test_equal_model_is_installed(self):
        self.assertTrue(self._should_install()(0.14, 0.14))

    def test_slightly_worse_within_tolerance(self):
        self.assertTrue(self._should_install()(0.145, 0.14, tolerance=0.05))

    def test_clearly_worse_is_refused(self):
        self.assertFalse(self._should_install()(0.17, 0.14),
                         "модель заметно хуже — ставить нельзя")

    def test_force_overrides_refusal(self):
        self.assertTrue(self._should_install()(0.17, 0.14, force=True))
