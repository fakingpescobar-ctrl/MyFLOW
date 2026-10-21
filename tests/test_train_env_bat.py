"""Батник install-train-env.bat: парсинг cmd и переходы по меткам.

Полный прогон качает 2.5 ГБ (torch с CUDA) — для теста не годится.
Поэтому тест собирает КОПИЮ скрипта, где тяжёлые команды заменены на echo,
а все пути указывают на временный каталог. Ничего настоящего он не трогает
и не требует: тест должен проходить и на чистой машине без .venv-train.

Что ловится:
  * непарные скобки или && в echo внутри блока if (...) — cmd закрывает блок
    по ПЕРВОЙ ')' , даже из обычного текста, и весь скрипт после ломается;
  * потерянная метка или goto не туда — сценарий не доходит до конца;
  * команда, разрезанная переносом, после чего остаётся мусор.
"""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAT = ROOT / "install-train-env.bat"

# Команды, которые нельзя выполнять в тесте: заменяем на echo.
HEAVY_PREFIXES = (
    '"%TRAIN_PY%" -m pip',
    '"%TRAIN_PY%" -c',
    '"%RUNTIME_PY%" -X utf8',
    "powershell ",
    "curl.exe ",
    "  curl.exe ",
    "py -3.13 -m venv ",
)


def logical_commands(text: str) -> list[list[str]]:
    """Разбить .bat на команды: перенос строки через ^ продолжает команду."""
    commands: list[list[str]] = []
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not current and not line.strip():
            continue
        if line.rstrip().endswith("^"):
            current.append(line.rstrip()[:-1].strip())
            continue
        current.append(line.strip())
        commands.append(current)
        current = []
    if current:
        commands.append(current)
    return commands


def make_stub(source: str, workdir: Path) -> str:
    """Собрать заглушку: тяжёлые команды -> echo, пути -> временный каталог."""
    out_lines: list[str] = []
    for command in logical_commands(source):
        joined = " ".join(command)
        if joined.strip().lower() == "pause":
            out_lines.append("echo   [STUB] pause")
            continue
        if any(joined.lstrip().startswith(prefix.strip()) for prefix in HEAVY_PREFIXES):
            out_lines.append("echo   [STUB] " + joined.strip()[:60])
            continue
        out_lines.extend(command)
    text = "\n".join(out_lines) + "\n"
    # Пути: и %~dp0, и жёстко прописанные TRAIN_PY/RUNTIME_PY/WHEELS.
    text = text.replace("%~dp0", str(workdir) + "\\\\")
    # Замены через lambda: re.sub иначе трактует backslash в замене как escape.
    text = re.sub(
        r'set "TRAIN_PY=.*"',
        lambda _: f'set "TRAIN_PY={workdir}\\\\.venv-train\\\\Scripts\\\\python.exe"',
        text,
    )
    text = re.sub(
        r'set "RUNTIME_PY=.*"',
        lambda _: f'set "RUNTIME_PY={workdir}\\\\.venv\\\\Scripts\\\\python.exe"',
        text,
    )
    text = re.sub(
        r'set "WHEELS=.*"',
        lambda _: f'set "WHEELS={workdir}\\\\wheels"',
        text,
    )
    return text


class TrainEnvScriptTest(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(BAT.exists(), "нужен install-train-env.bat")
        self.assertTrue(
            (ROOT / "requirements-train.txt").exists(), "нужен requirements-train.txt"
        )

    def test_no_special_chars_in_echo_inside_block(self):
        """Внутри блока if (...) скобка или && в тексте echo ломает cmd.

        cmd закрывает блок по ПЕРВОЙ ')' — даже если это просто текст.
        Незакрытая '(' и '&&' ломают разбор так же.
        """
        offenders = []
        inside = False
        for num, line in enumerate(BAT.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if re.search(r"\bif\b.*\($", stripped, re.IGNORECASE):
                inside = True
            if inside and stripped.lower().startswith("echo "):
                if ")" in stripped or "(" in stripped or "&&" in stripped:
                    offenders.append(f"{num}: {stripped}")
            if inside and stripped == ")":
                inside = False
        self.assertEqual(
            offenders,
            [],
            "в echo внутри блока if нельзя скобки и &&:\n" + "\n".join(offenders),
        )

    def test_torch_comes_with_cuda(self):
        """Колесо с PyPI для Windows — CPU-only (194 МБ). Нужен индекс PyTorch."""
        text = BAT.read_text(encoding="utf-8")
        self.assertIn(
            "download.pytorch.org/whl/cu124",
            text,
            "torch должен браться с индекса PyTorch (там CUDA-сборка)",
        )
        req = (ROOT / "requirements-train.txt").read_text(encoding="utf-8")
        self.assertIn(
            "torch==2.6.0+cu124",
            req,
            "в requirements-train нужен именно CUDA-пин (+cu124)",
        )

    def test_offline_path_pins_sympy(self):
        """torch 2.6.0 требует sympy ровно 1.13.1 — иначе не импортируется."""
        text = BAT.read_text(encoding="utf-8")
        self.assertIn(
            "sympy==1.13.1",
            text,
            "офлайн-ветка должна тянуть sympy==1.13.1, а не последний",
        )

    def test_script_reaches_the_end(self):
        """Прогон заглушки: скрипт должен дойти до GOTOVO без ошибок.

        Тест герметичен: работает на временном каталоге и не требует
        установленного .venv-train.
        """
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            # Заглушка «окружения»: файл есть — значит скрипт идёт по
            # ветке «уже есть» и не пытается ничего создавать.
            fake = work / ".venv-train" / "Scripts" / "python.exe"
            fake.parent.mkdir(parents=True, exist_ok=True)
            fake.write_bytes(b"")
            (work / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)

            stub = make_stub(BAT.read_text(encoding="utf-8"), work)
            copy = work / "smoke.bat"
            copy.write_text(stub, encoding="utf-8", newline="\r\n")
            res = subprocess.run(
                ["cmd", "/c", str(copy)], capture_output=True, text=True, timeout=180
            )
        out = res.stdout or ""
        self.assertIn("GOTOVO", out, f"скрипт не дошёл до конца:\n{out[-1500:]}")
        self.assertNotIn("[oshibka]", out, f"скрипт ругнулся:\n{out[-1500:]}")
        self.assertNotIn(
            "9009", out, f"команда не найдена — огрызок разбора:\n{out[-1500:]}"
        )


if __name__ == "__main__":
    unittest.main()
