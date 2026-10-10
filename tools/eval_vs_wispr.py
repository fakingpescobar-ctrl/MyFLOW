"""Сравнение MyFLOW и Wispr Flow «фраза к фразе» на ОДНОМ И ТОМ ЖЕ аудио.

Берём выборку, импортированную `tools/import_wispr_eval.py` (реальные диктовки
пользователя: WAV + их asrText/formattedText + их замеренная задержка), прогоняем
наш пайплайн и считаем:

  * качество: WER нашего итога против их итога (а где есть — против правки
    пользователя `editedText`, это самый честный эталон);
  * скорость: наш e2e (ASR + полировка, измеряем сами) против их e2e (их запись);
  * сколько фраз ушло без LLM (быстрый путь/лёгкая чистка).

Данные и отчёт остаются локально в `data/eval_wispr/`.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from myflow.config import SettingsStore                      # noqa: E402
from myflow.fastpath import fast_path_allowed, looks_wellformed  # noqa: E402
from myflow.lightclean import light_clean, strip_inner_fillers                    # noqa: E402

WISPR_DIR = ROOT / "data" / "eval_wispr"
_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate (Левенштейн по словам / длина эталона)."""
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"неожиданный формат {path.name}: "
                             f"{w.getframerate()} Гц, {w.getnchannels()} кан., {w.getsampwidth()*8} бит")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))]


def evaluate(rows: list[dict], use_polish: str = "auto", limit: int | None = None) -> dict:
    from myflow.asr import make_asr_client, whisper_needed  # noqa: PLC0415
    from myflow.polish import PolishClient                 # noqa: PLC0415

    cfg = SettingsStore().get()
    asr = make_asr_client(cfg.asr)
    if hasattr(asr, "warmup"):
        asr.warmup()
    polish = PolishClient(cfg.polish)
    llm_alive = polish.alive()
    do_polish = (use_polish == "on") or (use_polish == "auto" and llm_alive)
    print(f"ASR: {type(asr).__name__} | LLM: {'есть' if llm_alive else 'нет'} "
          f"| полировка: {'вкл' if do_polish else 'выкл'}")

    results: list[dict] = []
    for row in rows[:limit] if limit else rows:
        try:
            pcm = read_wav(ROOT / row["audio"])
        except Exception as exc:  # noqa: BLE001
            print(f"  {row['id'][:8]}… пропуск: {exc}")
            continue

        t0 = time.perf_counter()
        res = asr.transcribe(pcm)
        asr_ms = int((time.perf_counter() - t0) * 1000)
        our_asr = res.text

        style = "plain"
        fast_ok, reason = fast_path_allowed(getattr(asr, "backend_name", ""), style, our_asr)
        rules: list[str] = []
        our_final = our_asr
        polish_ms = 0
        route = "fast_path" if fast_ok else ""
        if not fast_ok and do_polish:
            cleaned, rules = light_clean(our_asr)
            has_terminal = our_asr.rstrip()[-1:] in ".!?…"
            if rules and has_terminal and looks_wellformed(cleaned)[0]:
                our_final, route = cleaned, "local_polish"
            else:
                t1 = time.perf_counter()
                try:
                    polished = polish.polish(our_asr, style=style, dictionary=[])
                    our_final = polished.text or our_asr
                    route = "llm"
                except Exception as exc:  # noqa: BLE001
                    route = f"llm_error:{type(exc).__name__}"
                polish_ms = int((time.perf_counter() - t1) * 1000)
        elif not fast_ok:
            route = "llm_off"

        # та же детерминированная зачистка, что и в боевом пайплайне
        cleaned_final, inner_rules = strip_inner_fillers(our_final)
        if inner_rules and cleaned_final:
            our_final = cleaned_final
        our_e2e = asr_ms + polish_ms
        reference = row.get("their_edited") or row["their_final"]
        results.append({
            "id": row["id"],
            "route": route,
            "our_asr": our_asr,
            "our_final": our_final,
            "their_asr": row["their_asr"],
            "their_final": row["their_final"],
            "their_edited": row.get("their_edited"),
            "reference": reference,
            "our_asr_ms": asr_ms,
            "our_polish_ms": polish_ms,
            "our_e2e_ms": our_e2e,
            "their_e2e_ms": row["their_e2e_ms"],
            "their_network_ms": row.get("their_network_ms"),
            "speech_seconds": row.get("speech_seconds"),
            "wer_vs_reference": round(wer(reference, our_final), 4),
            "wer_their_vs_reference": round(wer(reference, row["their_final"]), 4),
            "wer_asr_vs_their_asr": round(wer(row["their_asr"], our_asr), 4),
            "has_user_edit": bool(row.get("their_edited")),
        })

    ours = [r["our_e2e_ms"] for r in results]
    theirs = [r["their_e2e_ms"] for r in results]
    ref_wer = [r["wer_vs_reference"] for r in results]
    asr_wer = [r["wer_asr_vs_their_asr"] for r in results]
    return {
        "n": len(results),
        "routes": {k: sum(1 for r in results if r["route"] == k)
                   for k in {r["route"] for r in results}},
        "our_e2e": {"p50": pct(ours, .5), "p90": pct(ours, .9), "p99": pct(ours, .99)},
        "their_e2e": {"p50": pct(theirs, .5), "p90": pct(theirs, .9), "p99": pct(theirs, .99)},
        "wer_vs_reference": {"p50": round(pct(ref_wer, .5), 3), "p90": round(pct(ref_wer, .9), 3),
                             "mean": round(statistics.fmean(ref_wer), 3) if ref_wer else 0},
        "wer_asr_vs_their_asr": {"p50": round(pct(asr_wer, .5), 3),
                                 "mean": round(statistics.fmean(asr_wer), 3) if asr_wer else 0},
        "results": results,
    }


def write_report(report: dict, path: Path, worst: int = 10) -> None:
    r = report
    lines = [
        "# MyFLOW против Wispr Flow — сравнение на одной и той же речи",
        "",
        f"Фраз: **{r['n']}** (импорт из базы Wispr, аудио — реальные диктовки).",
        f"Маршруты: {r['routes']}",
        "",
        "## Задержка (мс)",
        "",
        "| | p50 | p90 | p99 |",
        "|---|---|---|---|",
        f"| MyFLOW (наш замер) | {r['our_e2e']['p50']} | {r['our_e2e']['p90']} | {r['our_e2e']['p99']} |",
        f"| Wispr (их запись) | {r['their_e2e']['p50']} | {r['their_e2e']['p90']} | {r['their_e2e']['p99']} |",
        "",
        "## Качество (WER к эталону)",
        "",
        "Эталон — их итоговый текст, а где есть — правка пользователя (`editedText`).",
        "",
        f"* наш итог против эталона: p50 **{r['wer_vs_reference']['p50']}**, "
        f"p90 {r['wer_vs_reference']['p90']}, среднее {r['wer_vs_reference']['mean']}",
        f"* наш ASR против их ASR: p50 {r['wer_asr_vs_their_asr']['p50']}, "
        f"среднее {r['wer_asr_vs_their_asr']['mean']} (разные движки — это расхождение, не ошибка)",
        "",
        f"## Худшие {worst} случаев (локально, для разбора)",
        "",
    ]
    for item in sorted(r["results"], key=lambda x: -x["wer_vs_reference"])[:worst]:
        lines.append(f"* `{item['id'][:8]}` WER {item['wer_vs_reference']:.2f} "
                     f"({item['our_e2e_ms']} мс, {item['route']}): "
                     f"наш «{item['our_final'][:70]}» ↔ эталон «{item['reference'][:70]}»")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Сравнение MyFLOW и Wispr на одном аудио")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--polish", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--only-edited", action="store_true",
                    help="только фразы, где пользователь сам правил текст")
    ap.add_argument("--report", type=Path, default=WISPR_DIR / "report.md")
    ap.add_argument("--save", type=Path, default=WISPR_DIR / "results.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            (WISPR_DIR / "phrases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only_edited:
        rows = [r for r in rows if r.get("their_edited")]
    print(f"Фраз к прогону: {len(rows)}")
    report = evaluate(rows, use_polish=args.polish, limit=args.limit)
    write_report(report, args.report)
    args.save.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in report["results"]),
                         encoding="utf-8")
    print(f"\nЗадержка  MyFLOW p50 {report['our_e2e']['p50']} / p90 {report['our_e2e']['p90']} мс"
          f"   Wispr p50 {report['their_e2e']['p50']} / p90 {report['their_e2e']['p90']} мс")
    print(f"Качество  WER(наш итог, эталон) p50 {report['wer_vs_reference']['p50']} "
          f"(среднее {report['wer_vs_reference']['mean']})")
    print(f"Маршруты  {report['routes']}")
    print(f"Отчёт: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
