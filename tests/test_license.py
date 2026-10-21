"""Лицензия проекта должна быть одна и везде.

MIT выбрана осознанно: весь стек зависимостей permissive (MIT/Apache/BSD),
GPL-обязательств нет, а для личного инструмента короткая и понятная лицензия
важнее формальностей.

Проверяем, что заявленная лицензия не разъехалась:
  * файл LICENSE есть и это именно MIT;
  * в метаданных пакета указано то же самое;
  * README и список сторонних компонентов не врут про лицензию.
"""

import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class LicenseTest(unittest.TestCase):
    def test_license_file_is_mit(self):
        license_file = ROOT / "LICENSE"
        self.assertTrue(license_file.exists(), "нужен файл LICENSE")
        text = license_file.read_text(encoding="utf-8")
        self.assertIn("MIT License", text)
        self.assertIn("Permission is hereby granted, free of charge", text)
        self.assertRegex(text, r"Copyright \(c\) \d{4}")

    def test_pyproject_declares_same_license(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = data["project"]
        self.assertEqual(
            project.get("license"),
            "MIT",
            "в метаданных пакета лицензия должна совпадать с LICENSE",
        )
        self.assertIn("LICENSE", project.get("license-files", []))

    def test_readme_links_license(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("(LICENSE)", readme, "README должен ссылаться на LICENSE")
        self.assertIn("MIT", readme)

    def test_third_party_mentions_project_license(self):
        text = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
        self.assertIn(
            "MIT",
            text,
            "файл сторонних лицензий должен отличать MIT проекта "
            "от лицензий зависимостей",
        )

    def test_no_license_classifier_conflict(self):
        """Классификатор лицензии и SPDX-выражение вместе нельзя (setuptools>=77)."""
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        classifiers = data["project"].get("classifiers", [])
        self.assertFalse(
            [c for c in classifiers if c.startswith("License ::")],
            'при license = "MIT" классификатор лицензии не нужен (и запрещён)',
        )

    def test_dependencies_allow_mit(self):
        """В MIT-проекте нельзя затягивать copyleft-зависимости."""
        text = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
        for row in text.splitlines():
            if not row.startswith("| ") or "GPL" not in row:
                continue
            # LGPL у pynput — динамическая линковка, обязательств по раскрытию нет.
            self.assertIn("LGPL", row)
            self.assertNotRegex(
                row,
                r"(?<!L)GPL",
                "copyleft-зависимость (GPL) в MIT-проекте требует пересмотра лицензии",
            )


if __name__ == "__main__":
    unittest.main()
