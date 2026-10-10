"""Полный прогон сравнения с Wispr ВМЕСТЕ с локальной LLM.

Почему отдельный скрипт: серверы поднимает Supervisor с Job Object
(KILL_ON_JOB_CLOSE) — если запустить LLM в одном процессе, а прогон в другом,
сервер умрёт вместе с первым (это фича: закрыл MyFLOW — серверы погасли). Значит,
поднимаем сервер и гоняем сравнение в ОДНОМ процессе.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import eval_vs_wispr  # noqa: E402
from myflow.config import SettingsStore  # noqa: E402
from myflow.polish import PolishClient  # noqa: E402
from myflow.tui import Supervisor, default_specs  # noqa: E402


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    only_edited = "--only-edited" in sys.argv
    sup = Supervisor(default_specs(), enabled=True)
    spec = next(s for s in sup.specs if s.key == "llm")
    try:
        print(f"поднимаю LLM: {sup.start(spec)} | готов: {sup.wait_ready(spec, 120.0)}")
        print("жива:", PolishClient(SettingsStore().get().polish).alive(), flush=True)
        rows = [json.loads(line) for line in
                (eval_vs_wispr.WISPR_DIR / "phrases.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        if only_edited:
            rows = [r for r in rows if r.get("their_edited")]
        print(f"фраз к прогону: {len(rows)}", flush=True)
        report = eval_vs_wispr.evaluate(rows, use_polish="on", limit=limit)
        suffix = "_edited" if only_edited else ""
        out_md = eval_vs_wispr.WISPR_DIR / f"report_llm{suffix}.md"
        out_json = eval_vs_wispr.WISPR_DIR / f"results_llm{suffix}.jsonl"
        eval_vs_wispr.write_report(report, out_md)
        out_json.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                                      for r in report["results"]), encoding="utf-8")
        print(f"\nЗадержка  MyFLOW p50 {report['our_e2e']['p50']} / p90 {report['our_e2e']['p90']}"
              f" / p99 {report['our_e2e']['p99']} мс")
        print(f"          Wispr  p50 {report['their_e2e']['p50']} / p90 {report['their_e2e']['p90']}"
              f" / p99 {report['their_e2e']['p99']} мс")
        print(f"Качество  WER(наш итог, эталон) p50 {report['wer_vs_reference']['p50']}"
              f" | p90 {report['wer_vs_reference']['p90']}"
              f" | среднее {report['wer_vs_reference']['mean']}")
        print(f"Маршруты  {report['routes']}")
        print(f"Отчёты: {out_md} | {out_json}")
        return 0
    finally:
        sup.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
