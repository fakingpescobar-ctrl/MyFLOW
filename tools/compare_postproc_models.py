"""Сравнить две модели постпроцессора на отложенных диктовках.

Зачем: дообучение — автоматический процесс, и оно МОЖЕТ сделать модель хуже.
Проверено на себе: обучение на всех диктовках (включая последние 80, короткие и
уже чистые) дало 0.170 против 0.140 — то есть установка «свежей» модели ухудшила
полировку. Поэтому перед заменой модели в `models/` надо ОТДЕЛЬНО измерить новую
на тех диктовках, которых она не видела, и ставить только если не хуже.

Здесь только решающая часть: запустить обе модели на своих портах и посчитать
словесное расстояние до эталона боевым путём (ворота быстрого пути -> правила ->
модель), чтобы сравнение было честным.
"""
from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_postproc import distance                       # noqa: E402
from myflow.config import SettingsStore                  # noqa: E402
from myflow.fastpath import fast_path_allowed, looks_wellformed  # noqa: E402
from myflow.lightclean import light_clean, strip_inner_fillers   # noqa: E402
from myflow.polish import PolishClient                   # noqa: E402
from myflow.postproc_prompt import POSTPROC_STYLE        # noqa: E402


def should_install(new_mean: float, current_mean: float,
                   tolerance: float = 0.05, force: bool = False) -> bool:
    """Ставить ли новую модель: не хуже текущей (с допуском) или принудительно."""
    if force:
        return True
    return new_mean <= current_mean * (1.0 + tolerance)


def free_port() -> int:
    """Свободный локальный порт для замеров.

    Жёсткие номера (были 8085/8086) занимают чужие программы: 12.10.2026 порт 8085
    держала сторонняя ScpTrayApp, сравнение моделей упало, и хранитель поставил
    непроверенную модель. Пусть порт выбирает система.
    """
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_health(url: str, seconds: float = 120.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/models", timeout=1.5).ok:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return False


def gated_mean(model_path: Path, port: int, pairs: list[dict],
               log: Path) -> float:
    """Среднее расстояние до эталона боевым путём для одной модели."""
    server = ROOT / "bin" / "llama-cpp" / "llama-server.exe"
    url = f"http://127.0.0.1:{port}/v1"
    with open(log, "ab") as fh:
        proc = subprocess.Popen(
            [str(server), "-m", str(model_path), "--host", "127.0.0.1",
             "--port", str(port), "-c", "4096", "-np", "1", "-ngl", "99",
             "-fa", "on", "--jinja"],
            stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    try:
        if not wait_health(url):
            raise RuntimeError(f"модель не поднялась на :{port}")
        cfg = SettingsStore().get().polish
        cfg.url = url
        cfg.enabled = True
        client = PolishClient(cfg)
        dists = []
        for row in pairs:
            src = row["src"]
            fast_ok, _ = fast_path_allowed("gigaam", "plain", src)
            if fast_ok:
                out = src
            else:
                cleaned, rules = light_clean(src)
                terminal = src.rstrip()[-1:] in ".!?…"
                if rules and terminal and looks_wellformed(cleaned)[0]:
                    out = cleaned
                else:
                    try:
                        out = client.polish(src, style=POSTPROC_STYLE,
                                            dictionary=[]).text or src
                    except Exception:  # noqa: BLE001
                        out = src
            cleaned_final, inner = strip_inner_fillers(out)
            if inner and cleaned_final:
                out = cleaned_final
            dists.append(distance(row["dst"], out))
        return statistics.fmean(dists) if dists else 0.0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()


def compare(current: Path, new: Path, pairs: list[dict],
            log: Path | None = None) -> dict:
    """Сравнить две модели на одних парах. Возвращает средние расстояния."""
    log = log or (ROOT / "data" / "train" / "compare_models.log")
    new_mean = gated_mean(new, free_port(), pairs, log)
    current_mean = gated_mean(current, free_port(), pairs, log)
    return {"new": new_mean, "current": current_mean}
