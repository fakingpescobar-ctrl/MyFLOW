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
    deps = {"dsh-plugin-myflow": "link:.../dsh-plugin-myflow"} if with_bridge else {}
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
        tmp = make_home(with_bridge=True)
        with mock.patch.dict(os.environ, {"DSH_HOME": tmp.name}):
            self.assertTrue(dsh_bridge.bridge_installed())
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
    def test_reports_already_installed(self):
        tmp = make_home(with_bridge=True)
        with mock.patch.dict(os.environ, {"DSH_HOME": tmp.name}):
            ok, message = dsh_bridge.install_bridge()
        self.assertTrue(ok)
        self.assertIn("уже установлен", message)
        tmp.cleanup()

    def test_refuses_without_dsh_command(self):
        tmp = make_home(with_bridge=False)
        with mock.patch.dict(os.environ, {"DSH_HOME": tmp.name}), \
             mock.patch.object(dsh_bridge, "find_dsh_cli", lambda: None):
            ok, message = dsh_bridge.install_bridge()
        self.assertFalse(ok)
        self.assertIn("вручную", message)
        tmp.cleanup()

    def test_runs_dsh_plugin_add(self):
        tmp = make_home(with_bridge=False)
        calls = {}

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            # как настоящий dsh: прописывает зависимость в профиль
            profile = Path(tmp.name) / "profiles" / "web" / "package.json"
            profile.write_text(json.dumps({
                "name": "dsh-profile-web",
                "dependencies": {"dsh-plugin-myflow": "link:.../dsh-plugin-myflow"},
            }), encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.dict(os.environ, {"DSH_HOME": tmp.name}), \
             mock.patch.object(dsh_bridge, "find_dsh_cli", lambda: "dsh.cmd"), \
             mock.patch("subprocess.run", fake_run):
            ok, message = dsh_bridge.install_bridge()
        self.assertTrue(ok, message)
        self.assertIn("перезапусти", message)
        self.assertEqual(calls["cmd"][:4], ["dsh.cmd", "plugin", "--profile", "web"])
        tmp.cleanup()


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
