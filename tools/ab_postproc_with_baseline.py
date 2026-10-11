"""A/B постпроцессора против 4B с гарантией, что 4B поднят и потом погашен.

Отдельный сценарий нужен потому, что сервер 4B поднимает приложение, а оно может
быть закрыто (тогда сравнивать не с чем). Здесь: поднять наш 4B на :8082, дождаться
готовности, прогнать `ab_postproc.py`, погасить поднятое.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.polish import PolishClient        # noqa: E402
from myflow.config import SettingsStore       # noqa: E402
from myflow.tui import Supervisor, default_specs  # noqa: E402


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "data/train/gguf/myflow-postproc-v3-q8_0.gguf"
    cfg = SettingsStore().get().polish
    client = PolishClient(cfg)
    started = False
    sup = Supervisor(default_specs(), enabled=True)
    llm = next(s for s in sup.specs if s.key == "llm")
    try:
        if not client.alive():
            print("  поднимаю 4B на :8082...", flush=True)
            sup.start(llm)
            started = sup.wait_ready(llm, 120.0)
            print("  готов:", started, flush=True)
        else:
            print("  4B уже работает", flush=True)
        cmd = [sys.executable, "-X", "utf8", str(ROOT / "tools" / "ab_postproc.py"),
               "--model", model, "--also-all", "--port", "8083"]
        print("  --- запускаю A/B ---", flush=True)
        subprocess.run(cmd, cwd=str(ROOT))
    finally:
        if started:
            print("  гашу 4B, который поднял я", flush=True)
            sup.stop(llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
