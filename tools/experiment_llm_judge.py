"""Эксперимент: LLM-«судья» выбирает лучший из двух текстов распознавания.

У судьи НЕТ аудио — только два текста (GigaAM и whisper). Значит он оценивает
правдоподобность: связность, термины,完整ность. Замер на 624 реальных фразах:

  * `choice`  — просим выбрать A или B (дешёво, ~4 токена);
  * `merge`   — просим выдать один лучший/склеенный текст (дороже);
  * сравнение с «оракулом» (выбор по фактическому WER) и текущим авто-выбором.

Запуск: python tools/experiment_llm_judge.py [--limit N] [--mode choice|merge|both]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_vs_wispr import wer, pct                      # noqa: E402
from myflow.config import SettingsStore                 # noqa: E402
from myflow.polish import PolishClient                  # noqa: E402
from myflow.tui import Supervisor, default_specs        # noqa: E402

WISPR_DIR = ROOT / "data" / "eval_wispr"

CHOICE_PROMPT = (
    "Два варианта распознавания одной и той же русской фразы. Пунктуация и регистр "
    "не важны — важно, что человек реально сказал: слова, термины, окончания.\n"
    "A: {a}\nB: {b}\n"
    "Ответь ОДНОЙ буквой: A или B. Если оба одинаковы — A."
)
MERGE_PROMPT = (
    "Два варианта распознавания одной и той же русской фразы. Собери из них один "
    "максимально верный текст: то, что человек реально сказал. Ничего не добавляй "
    "от себя, не пересказывай, сохрани порядок слов. Пунктуацию и регистр можно "
    "поправить.\nA: {a}\nB: {b}\nВерни только текст."
)


def judge_choice(polish: PolishClient, a: str, b: str) -> tuple[str, int]:
    started = time.perf_counter()
    try:
        answer = polish.ask(CHOICE_PROMPT.format(a=a.strip(), b=b.strip()), max_tokens=4)
    except Exception as exc:  # noqa: BLE001
        return "A", int((time.perf_counter() - started) * 1000)
    ms = int((time.perf_counter() - started) * 1000)
    letters = re.findall(r"\b([AB])\b", (answer or "").upper())
    return (letters[0] if letters else "A"), ms


def judge_merge(polish: PolishClient, a: str, b: str) -> tuple[str, int]:
    started = time.perf_counter()
    try:
        merged = polish.ask(MERGE_PROMPT.format(a=a.strip(), b=b.strip()), max_tokens=200)
    except Exception as exc:  # noqa: BLE001
        return "", int((time.perf_counter() - started) * 1000)
    ms = int((time.perf_counter() - started) * 1000)
    merged = (merged or "").strip().strip('"').strip()
    # Модель иногда отвечает «A: ...» — вычищаем префикс
    merged = re.sub(r"^(?:[ABAB]\s*[:—-]\s*)", "", merged)
    return merged, ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", choices=("choice", "merge", "both"), default="both")
    args = ap.parse_args()

    idx = {r["id"]: r for r in (json.loads(line) for line in
           (WISPR_DIR / "phrases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    ab = json.loads((WISPR_DIR / "ab_asr.json").read_text(encoding="utf-8"))
    pairs = list(zip(ab["gigaam"], ab["whisper"]))
    if args.limit:
        pairs = pairs[: args.limit]

    sup = Supervisor(default_specs(), enabled=True)
    spec = next(s for s in sup.specs if s.key == "llm")
    try:
        print(f"LLM: {sup.start(spec)} | готов: {sup.wait_ready(spec, 120.0)}", flush=True)
        polish = PolishClient(SettingsStore().get().polish)
        print("жива:", polish.alive(), flush=True)

        stats = {"choice": [], "merge": [], "oracle": [], "gigaam": [], "whisper": []}
        judge_ms, merge_ms, agree = [], [], 0
        judge_merged: list[str] = []
        judge_merge_ms: list[int] = []
        for i, (g, w) in enumerate(pairs):
            ref = idx[g["id"]]["their_final"]
            a, b = g["text"], w["text"]
            wa, wb = wer(ref, a), wer(ref, b)
            stats["gigaam"].append(wa)
            stats["whisper"].append(wb)
            stats["oracle"].append(min(wa, wb))
            if args.mode in ("choice", "both"):
                pick, ms = judge_choice(polish, a, b)
                judge_ms.append(ms)
                chosen = a if pick == "A" else b
                stats["choice"].append(wer(ref, chosen))
                best = "A" if wa <= wb else "B"
                agree += int(pick == best)
            if args.mode in ("merge", "both"):
                merged, ms = judge_merge(polish, a, b)
                merge_ms.append(ms)
                judge_merged.append(merged)
                judge_merge_ms.append(ms)
                stats["merge"].append(wer(ref, merged) if merged else min(wa, wb))
            else:
                judge_merged.append("")
                judge_merge_ms.append(0)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(pairs)}", flush=True)

        # Симуляция гибрида: GigaAM на уверенных фразах, судья-склейка на спорных.
        try:
            router_rows = {r["id"]: r for r in (json.loads(line) for line in
                           (WISPR_DIR / "results_llm_router.jsonl").read_text(encoding="utf-8").splitlines()
                           if line.strip())}
        except Exception:  # noqa: BLE001
            router_rows = {}
        hybrid, hybrid_ms, merged_n = [], [], 0
        for i, (g, w) in enumerate(pairs):
            ref = idx[g["id"]]["their_final"]
            doubtful = router_rows.get(g["id"], {}).get("asr_route", "primary") != "primary"
            if doubtful and i < len(judge_merged):
                text = judge_merged[i] or w["text"]
                hybrid.append(wer(ref, text)); hybrid_ms.append(g["ms"] + w["ms"] + judge_merge_ms[i])
                merged_n += 1
            else:
                hybrid.append(wer(ref, g["text"])); hybrid_ms.append(g["ms"])
        stats["hybrid"] = hybrid

        print("\n=== WER к их тексту (ё=е) ===")
        rows = [("только GigaAM", stats["gigaam"]), ("только whisper", stats["whisper"]),
                ("авто-выбор (сейчас)", None), ("судья: выбор A/B", stats["choice"]),
                ("судья: склейка (все фразы)", stats["merge"]),
                ("ГИБРИД: склейка на спорных", stats.get("hybrid")),
                ("ОРАКУЛ (потолок)", stats["oracle"])]
        for name, values in rows:
            if values is None:
                print(f"  {name:<22} см. результаты results_llm_router.jsonl")
                continue
            print(f"  {name:<22} p50 {pct(values,.5):.3f} | среднее {statistics.fmean(values):.3f} | "
                  f"точных {sum(1 for x in values if x == 0)/len(values)*100:>4.1f}%")
        if judge_ms:
            print(f"\n  задержка судьи (выбор): p50 {pct(judge_ms,.5)} | p90 {pct(judge_ms,.9)} мс | "
                  f"совпал с оракулом {agree/len(judge_ms)*100:.1f}%")
        if merge_ms:
            print(f"  задержка судьи (склейка): p50 {pct(merge_ms,.5)} | p90 {pct(merge_ms,.9)} мс")
        (WISPR_DIR / "judge_merge.json").write_text(json.dumps(
            {"merged": judge_merged, "ms": judge_merge_ms,
             "choice": [round(x, 4) for x in stats["choice"]]}, ensure_ascii=False),
            encoding="utf-8")
        if "hybrid" in stats:
            print(f"\n  гибрид: судья на {merged_n} спорных фразах, задержка p50 "
                  f"{pct(hybrid_ms,.5):.0f} | p90 {pct(hybrid_ms,.9):.0f} мс")
        return 0
    finally:
        sup.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
