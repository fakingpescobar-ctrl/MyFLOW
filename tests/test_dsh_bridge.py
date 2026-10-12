"""Мост MyFLOW → DeepSeek Harness: установка у нового пользователя.

Плагин ставится в КОНКРЕТНУЮ установку DSH ($DSH_HOME/profiles/web), поэтому у
нового пользователя его нет. MyFLOW проверяет это сам, умеет поставить плагин через
штатный CLI DSH и — если моста нет — вставляет текст обычным способом (буфер
обмена), чтобы надиктованное не терялось.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from myflow import dsh_bridge                          # noqa: E402
from myflow.dsh_inject import DshInjector              # noqa: E402


def make_home(with_bridge: bool) -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    profile = Path(tmp.name) / "profiles" / "web"
    profile.mkdir(parents=True)
    # новая схема установки — КОПИЯ внутри профиля (file:), не ссылка
    deps = ({"dsh-plugin-myflow": "file:./node_modules/dsh-plugin-myflow"}
            if with_bridge else {})
    (profile / "package.json").write_text(
        json.dumps({"name": "dsh-profile-web", "dependencies": deps}), encoding="utf-8")
    return tmp


class BridgeDetectionTest(unittest.TestCase):
    def test_not_installed_for_new_user(self):
        tmp = make_home(with_bridge=False)
        with mock.patch.dict(os.environ, {"DSH_HOME": tmp.name}):
            self.assertFalse(dsh_bridge.bridge_installed())
        tmp.cleanup()

    def test_detected_when_profile_has_plugin(self):
        """Мост считается готовым, когда есть и зависимость, и КОПИЯ плагина."""
        tmp = make_home(with_bridge=True)
        copied = Path(tmp.name) / "profiles" / "web" / "node_modules" / \
            dsh_bridge.PACKAGE_NAME
        copied.mkdir(parents=True)
        (copied / "index.js").write_text("", encoding="utf-8")
        with mock.patch.dict(os.environ, {"DSH_HOME": tmp.name}):
            self.assertTrue(dsh_bridge.bridge_installed())
        # зависимость без копии — не готов (битая/неполная установка)
        import shutil as _sh
        _sh.rmtree(copied)
        with mock.patch.dict(os.environ, {"DSH_HOME": tmp.name}):
            self.assertFalse(dsh_bridge.bridge_installed())
        tmp.cleanup()

    def test_missing_profile_is_not_an_error(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                self.assertFalse(dsh_bridge.bridge_installed())

    def test_broken_manifest_is_tolerated(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            profile = Path(tmp) / "profiles" / "web"
            profile.mkdir(parents=True)
            (profile / "package.json").write_text("{ это не json", encoding="utf-8")
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                self.assertFalse(dsh_bridge.bridge_installed())

    def test_plugin_ships_with_myflow(self):
        self.assertTrue((dsh_bridge.plugin_dir() / "index.js").exists(),
                        "плагин должен ехать вместе с MyFLOW")
        self.assertTrue((dsh_bridge.plugin_dir() / "cordis.patch.yml").exists())


class GuiProfilesTest(unittest.TestCase):
    """Цели определяются по оболочке: web, desktop — а не жёстко «web».

    Приложение пользователя грузит профиль `desktop`; мост, поставленный только
    в `web`, не загружался — на этом мы споткнулись.
    """

    def _home(self, profiles: dict) -> tempfile.TemporaryDirectory:
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        for name, bundles in profiles.items():
            d = Path(tmp.name) / "profiles" / name
            d.mkdir(parents=True)
            (d / "package.json").write_text(json.dumps(
                {"name": f"dsh-profile-{name}",
                 "dsh": {"profile": {"bundles": bundles}},
                 "dependencies": {}}), encoding="utf-8")
        return tmp

    def test_detects_both_windowed_profiles(self):
        tmp = self._home({
            "desktop": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"],
            "web": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"],
            "headless": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-headless"],
        })
        with mock.patch.dict(os.environ, {"DSH_HOME": tmp.name}):
            self.assertEqual(dsh_bridge.gui_profiles(), ["desktop", "web"])
        tmp.cleanup()

    def test_bridge_requires_every_windowed_profile(self):
        tmp = self._home({
            "desktop": ["@deepseek-ai/dsh-web-app"],
            "web": ["@deepseek-ai/dsh-web-app"],
        })
        with mock.patch.dict(os.environ, {"DSH_HOME": tmp.name}):
            self.assertFalse(dsh_bridge.bridge_installed(), "нигде нет — не установлен")
            # поставили только в web -> всё ещё не готово (desktop без моста)
            manifest = Path(tmp.name) / "profiles" / "web" / "package.json"
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["dependencies"] = {"dsh-plugin-myflow": "link:..."}
            manifest.write_text(json.dumps(data), encoding="utf-8")
            self.assertFalse(dsh_bridge.bridge_installed())
        tmp.cleanup()


class InstallTest(unittest.TestCase):
    """Мост ставится КОПИЕЙ внутрь профиля — MyFLOW можно переносить и удалять.

    Со ссылкой (`link:`) удаление MyFLOW ломало бы профиль DSH, а DSH на
    неразрешимом бандле падает при запуске.
    """

    def _profile(self, tmp: str, name: str = "web") -> Path:
        d = Path(tmp) / "profiles" / name
        d.mkdir(parents=True)
        (d / "package.json").write_text(json.dumps({
            "name": f"dsh-profile-{name}",
            "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-web-app"]}},
            "dependencies": {},
        }, ensure_ascii=False), encoding="utf-8")
        return d

    def test_installs_copy_and_wires_profile(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._profile(tmp)
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}), \
                 mock.patch.object(dsh_bridge, "dsh_app_running", lambda: True):
                ok, message = dsh_bridge.install_bridge()
                self.assertTrue(ok, message)
                self.assertIn("перезапусти", message)
                # копия плагина внутри профиля (не ссылка на MyFLOW)
                copied = Path(tmp) / "profiles" / "web" / "node_modules" / \
                    dsh_bridge.PACKAGE_NAME
                self.assertTrue((copied / "index.js").exists())
                data = json.loads((Path(tmp) / "profiles" / "web" /
                                   "package.json").read_text(encoding="utf-8"))
                self.assertTrue(str(data["dependencies"][dsh_bridge.PACKAGE_NAME])
                                .startswith("file:"))
                self.assertIn(dsh_bridge.PACKAGE_NAME,
                              data["dsh"]["profile"]["bundles"])
                self.assertTrue(dsh_bridge.bridge_installed())

    def test_reports_ready_after_install(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._profile(tmp)
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                dsh_bridge.install_bridge()
                ok, message = dsh_bridge.install_bridge()
        self.assertTrue(ok)
        self.assertIn("уже был", message)

    def test_broken_link_is_detected_and_repaired(self):
        """MyFLOW перенесли/удалили -> ссылка битая -> мост «не готов» -> починка."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            profile = self._profile(tmp)
            (profile / "package.json").write_text(json.dumps({
                "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-web-app",
                                                dsh_bridge.PACKAGE_NAME]}},
                "dependencies": {dsh_bridge.PACKAGE_NAME:
                                 "link:C:/нет/такой/папки/dsh-plugin-myflow"},
            }, ensure_ascii=False), encoding="utf-8")
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                health = dsh_bridge.bridge_health()
                self.assertFalse(health["ok"])
                self.assertIn("web", health["broken"])
                self.assertFalse(dsh_bridge.bridge_installed())
                ok, message = dsh_bridge.repair_bridge()
                self.assertTrue(ok, message)
                self.assertTrue(dsh_bridge.bridge_installed(),
                                "после починки ссылка заменяется копией")

    def test_uninstall_removes_everything(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._profile(tmp)
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                dsh_bridge.install_bridge()
                self.assertTrue(dsh_bridge.bridge_installed())
                ok, message = dsh_bridge.uninstall_bridge()
                self.assertTrue(ok, message)
                self.assertFalse(dsh_bridge.bridge_installed())
                data = json.loads((Path(tmp) / "profiles" / "web" /
                                   "package.json").read_text(encoding="utf-8"))
                self.assertNotIn(dsh_bridge.PACKAGE_NAME, data["dependencies"])
                self.assertNotIn(dsh_bridge.PACKAGE_NAME,
                                 data["dsh"]["profile"]["bundles"])

    def test_uninstall_when_not_installed(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._profile(tmp)
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                ok, message = dsh_bridge.uninstall_bridge()
        self.assertTrue(ok)
        self.assertIn("и не был", message)


class FallbackTest(unittest.TestCase):
    def test_falls_back_to_clipboard_without_bridge(self):
        class FakeFallback:
            def __init__(self):
                self.calls = []

            def paste(self, text):
                self.calls.append(text)
                return True

        fallback = FakeFallback()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = Path(tmp) / "inbox.jsonl"
            injector = DshInjector(inbox, fallback=fallback)
            injector._bridge_ok = False              # моста нет
            self.assertTrue(injector.paste("Проверка связи"))
            self.assertEqual(fallback.calls, ["Проверка связи"])
            self.assertFalse(inbox.exists(), "без моста в инбокс писать нечего")

    def test_no_bridge_and_no_fallback_reports_failure(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            injector = DshInjector(Path(tmp) / "inbox.jsonl")
            injector._bridge_ok = False
            self.assertFalse(injector.paste("текст"))

    def test_writes_to_inbox_when_bridge_ready(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            inbox = Path(tmp) / "inbox.jsonl"
            injector = DshInjector(inbox, fallback=None)
            injector._bridge_ok = True
            self.assertTrue(injector.paste("Проверка связи"))
            self.assertIn("Проверка связи", inbox.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()


class AppDiscoveryTest(unittest.TestCase):
    """Поиск установки DSH: без него новый пользователь не поставит мост.

    `dsh` в PATH есть не у всех, а переменную DSH_APP никто не задаёт. Поэтому
    основной источник — путь запущенного приложения (`DeepSeek Harness.exe`),
    плюс обычные места установки как запасной вариант.
    """

    def test_finds_app_from_running_process(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with mock.patch.object(dsh_bridge, "_app_paths_from_processes",
                                   lambda: [Path(tmp)]):
                self.assertEqual(dsh_bridge.find_dsh_app(), Path(tmp))
                self.assertTrue(dsh_bridge.dsh_app_running())

    def test_falls_back_to_localappdata(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            app = Path(tmp) / "Programs" / "DeepSeek Harness"
            app.mkdir(parents=True)
            (app / "DeepSeek Harness.exe").write_bytes(b"")
            with mock.patch.object(dsh_bridge, "_app_paths_from_processes", lambda: []), \
                 mock.patch.dict(os.environ, {"LOCALAPPDATA": tmp}, clear=False):
                os.environ.pop("DSH_APP", None)
                self.assertEqual(dsh_bridge.find_dsh_app(), app)

    def test_cli_found_next_to_app_without_path(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            app = Path(tmp) / "DeepSeek Harness"
            cli = app / "resources" / "runtime" / "cli" / "bin" / "dsh.cmd"
            cli.parent.mkdir(parents=True)
            cli.write_bytes(b"")
            with mock.patch.object(dsh_bridge, "_app_paths_from_processes",
                                   lambda: [app]), \
                 mock.patch("shutil.which", lambda name: None), \
                 mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DSH_APP", None)
                self.assertEqual(Path(dsh_bridge.find_dsh_cli()), cli)

    def test_nothing_found_returns_none(self):
        with mock.patch.object(dsh_bridge, "_app_paths_from_processes", lambda: []), \
             mock.patch("shutil.which", lambda name: None), \
             mock.patch.dict(os.environ, {}, clear=False):
            for var in ("DSH_APP", "LOCALAPPDATA", "PROGRAMFILES"):
                os.environ.pop(var, None)
            self.assertIsNone(dsh_bridge.find_dsh_app())
            self.assertIsNone(dsh_bridge.find_dsh_cli())


class DisabledToggleTest(unittest.TestCase):
    """Тумблер плагина в настройках DSH выключает мост — MyFLOW должен это видеть.

    Без этого MyFLOW считал бы мост рабочим, писал в инбокс, и диктовка молча
    пропадала бы.
    """

    def _installed_profile(self, tmp: str) -> Path:
        d = Path(tmp) / "profiles" / "web"
        (d / "node_modules" / dsh_bridge.PACKAGE_NAME).mkdir(parents=True)
        (d / "node_modules" / dsh_bridge.PACKAGE_NAME / "index.js").write_text("")
        (d / "package.json").write_text(json.dumps({
            "name": "dsh-profile-web",
            "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-web-app",
                                            dsh_bridge.PACKAGE_NAME]}},
            "dependencies": {dsh_bridge.PACKAGE_NAME:
                             f"file:./node_modules/{dsh_bridge.PACKAGE_NAME}"},
        }, ensure_ascii=False), encoding="utf-8")
        return d

    def test_toggle_off_is_detected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            d = self._installed_profile(tmp)
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                self.assertTrue(dsh_bridge.bridge_installed(), "пока тумблер включён")
                # пользователь выключил плагин в настройках DSH
                (d / "cordis.patch.yml").write_text(
                    "- id: myflow-bridge\n  disabled: true\n", encoding="utf-8")
                health = dsh_bridge.bridge_health()
                self.assertFalse(health["ok"])
                self.assertIn("web", health["disabled"])
                self.assertFalse(dsh_bridge.bridge_installed())

    def test_toggle_off_in_home_layer_is_detected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._installed_profile(tmp)
            (Path(tmp) / "cordis.patch.yml").write_text(
                '- id: myflow-bridge\n  disabled: true   # выключено в UI\n',
                encoding="utf-8")
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                health = dsh_bridge.bridge_health()
                self.assertIn("web", health["disabled"])

    def test_other_plugin_disabled_does_not_affect_us(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            d = self._installed_profile(tmp)
            (d / "cordis.patch.yml").write_text(
                "- id: orange-assistant\n  disabled: true\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                self.assertTrue(dsh_bridge.bridge_installed(),
                                "чужой выключенный плагин нас не касается")


class SyncTest(unittest.TestCase):
    """После обновления MyFLOW копию плагина в DSH надо обновлять самим.

    Плагин грузится из копии в профиле DSH, поэтому свежие исправления MyFLOW
    иначе до пользователя не доходили бы.
    """

    def _installed(self, tmp: str) -> Path:
        d = Path(tmp) / "profiles" / "web"
        (d / "node_modules" / dsh_bridge.PACKAGE_NAME).mkdir(parents=True)
        (d / "package.json").write_text(json.dumps({
            "name": "dsh-profile-web",
            "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-web-app",
                                            dsh_bridge.PACKAGE_NAME]}},
            "dependencies": {dsh_bridge.PACKAGE_NAME:
                             f"file:./node_modules/{dsh_bridge.PACKAGE_NAME}"},
        }, ensure_ascii=False), encoding="utf-8")
        with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
            dsh_bridge.install_bridge()
        return d

    def test_fresh_copy_is_not_stale(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._installed(tmp)
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                self.assertEqual(dsh_bridge.stale_profiles(), [])
                ok, message = dsh_bridge.sync_bridge()
        self.assertTrue(ok)
        self.assertIn("совпадает", message)

    def test_modified_copy_is_detected_and_updated(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            d = self._installed(tmp)
            copy = d / "node_modules" / dsh_bridge.PACKAGE_NAME / "index.js"
            copy.write_text("// старая версия плагина\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                self.assertEqual(dsh_bridge.stale_profiles(), ["web"])
                ok, message = dsh_bridge.sync_bridge()
                self.assertTrue(ok, message)
                self.assertIn("обновил", message)
                self.assertEqual(dsh_bridge.stale_profiles(), [], "после синхронизации")
                # содержимое копии снова равно поставке
                self.assertEqual(copy.read_text(encoding="utf-8"),
                                 (dsh_bridge.plugin_dir() / "index.js")
                                 .read_text(encoding="utf-8"))

    def test_fingerprint_changes_with_content(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp) / "plugin"
            root.mkdir()
            (root / "index.js").write_text("версия 1", encoding="utf-8")
            first = dsh_bridge.plugin_fingerprint(root)
            (root / "index.js").write_text("версия 2", encoding="utf-8")
            self.assertNotEqual(first, dsh_bridge.plugin_fingerprint(root))

    def test_no_copy_means_nothing_to_sync(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            profile = Path(tmp) / "profiles" / "web"
            profile.mkdir(parents=True)
            (profile / "package.json").write_text(json.dumps(
                {"dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-web-app"]}}}),
                encoding="utf-8")
            with mock.patch.dict(os.environ, {"DSH_HOME": tmp}):
                self.assertEqual(dsh_bridge.stale_profiles(), [])
