"""Сравнение на 624 фразах Wispr: 4B против нашего постпроцессора.

Главный критерий цели: «не уступать Qwen3-4B на 624 реальных фразах». Здесь оба
варианта прогоняются ОДНИМ И ТЕМ ЖЕ стендом (`eval_vs_wispr.py`) на одних и тех же
аудиофайлах и с одинаковым ASR, отличается только модель полировки:

* :8082 — боевая Qwen3-4B (стиль plain);
* :8083 — наша модель (стиль postproc подставляется автоматически).

Важно: кандидат обучен на датасете, из которого ИСКЛЮЧЕНЫ эталоны Wispr
(`build_postproc_dataset.py --no-wispr`), иначе он бы «узнавал» их и сравнение
было бы нечестным.

Запуск:
    .venv\\Scripts\\python.exe -X utf8 tools\\eval_wispr_compare.py \\
        --candidate data\\train\\gguf\\myflow-postproc-v4-q8_0.gguf --limit 624
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.config import SettingsStore                       # noqa: E402
from myflow.tui import Supervisor, default_specs              # noqa: E402

PORT_CANDIDATE = 8083


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


def run_eval(url: str, limit: int, tag: str) -> dict:
    report = ROOT / "data" / "eval_wispr" / f"report_{tag}.md"
    results = ROOT / "data" / "eval_wispr" / f"results_{tag}.jsonl"
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "tools" / "eval_vs_wispr.py"),
           "--limit", str(limit), "--polish", "on", "--url", url,
           "--report", str(report), "--save", str(results)]
    print(f"\n  --- {tag}: {url} ---", flush=True)
    subprocess.run(cmd, cwd=str(ROOT))
    summary = {}
    if report.exists():
        text = report.read_text(encoding="utf-8")
        for key in ("p50", "p90"):
            m = re.search(rf"{key}\s+([\d.]+)", text)
            if m:
                summary[key] = float(m.group(1))
        m = re.search(r"WER\([^)]*\)\s+p50\s+([\d.]+)\s+\(среднее\s+([\d.]+)\)", text)
        if m:
            summary["wer_p50"], summary["wer_mean"] = float(m.group(1)), float(m.group(2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="624 фразы: 4B против кандидата")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--limit", type=int, default=624)
    args = ap.parse_args()

    cand = Path(args.candidate)
    if not cand.is_absolute():
        cand = ROOT / cand
    if not cand.exists():
        print(f"нет модели кандидата: {cand}")
        return 1
    server = ROOT / "bin" / "llama-cpp" / "llama-server.exe"

    sup = Supervisor(default_specs(), enabled=True)
    llm = next(s for s in sup.specs if s.key == "llm")
    base_url = SettingsStore().get().polish.url
    started_llm = started_cand = False
    log = ROOT / "data" / "eval_wispr" / "candidate_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        if not wait_health(base_url, 2.0):
            print("  поднимаю 4B (:8082)...", flush=True)
            sup.start(llm)
            started_llm = sup.wait_ready(llm, 180.0)
        print(f"  4B готова: {wait_health(base_url, 5.0)}", flush=True)

        print(f"  поднимаю кандидата на :{PORT_CANDIDATE}...", flush=True)
        with open(log, "ab") as fh:
            proc = subprocess.Popen(
                [str(server), "-m", str(cand), "--host", "127.0.0.1",
                 "--port", str(PORT_CANDIDATE), "-c", "4096", "-np", "1",
                 "-ngl", "99", "-fa", "on", "--jinja"],
                stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        started_cand = wait_health(f"http://127.0.0.1:{PORT_CANDIDATE}/v1")
        print(f"  кандидат готов: {started_cand}", flush=True)

        summary = {}
        summary["4B"] = run_eval(base_url, args.limit, "cmp_4b")
        summary["candidate"] = run_eval(f"http://127.0.0.1:{PORT_CANDIDATE}/v1",
                                        args.limit, "cmp_cand")

        print("\n  === ИТОГ СРАВНЕНИЯ ===")
        for tag, data in summary.items():
            wer = data.get("wer_p50")
            lat = data.get("p50")
            print(f"    {tag:<10} WER p50 {wer if wer is not None else '—'} | "
                  f"задержка p50 {lat if lat is not None else '—'} мс")
        (ROOT / "data" / "eval_wispr" / "compare_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()
            print("  кандидат остановлен", flush=True)
        if started_llm:
            print("  гашу 4B, который поднял я", flush=True)
            sup.stop(llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
