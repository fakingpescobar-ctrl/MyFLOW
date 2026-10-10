"""Замер полного пайплайна поверх склеек LLM-судьи (с предохранителем).

Берём уже посчитанные склейки (`data/eval_wispr/judge_merge.json`), применяем
предохранитель (склейка обязана быть близка хотя бы к одному входу — иначе это
галлюцинация вида «То, что человек реально сказал.») и прогоняем обычный конвейер:
быстрый путь → лёгкая чистка → полировка LLM.

Сравнение идёт с теми же 624 фразами, что и остальные конфигурации:
  results_llm.jsonl            — только GigaAM
  results_llm_whisperonly.jsonl — только whisper
  results_llm_router.jsonl     — авто-выбор (текущий)
  results_llm_merge.jsonl      — эта: склейка судьи
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import eval_vs_wispr                                      # noqa: E402
import myflow.asr as asr_mod                              # noqa: E402
from myflow.asr import AsrResult                          # noqa: E402
from myflow.config import SettingsStore, whisper_dir, whisper_model_path  # noqa: E402
from myflow.polish import PolishClient                    # noqa: E402
from myflow.tui import ServerSpec, Supervisor, default_specs  # noqa: E402

BAD_MARKERS = re.compile(
    r"(человек реально сказал|ответь одной буквой|вариант[ыа]? распознавания|верни только текст)",
    re.IGNORECASE)
WISPR_DIR = eval_vs_wispr.WISPR_DIR


def _words(text: str) -> list[str]:
    return [w.replace("ё", "е") for w in re.findall(r"[а-яёa-z0-9]+", (text or "").lower())]


def wer(a: str, b: str) -> float:
    ref, hyp = _words(a), _words(b)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, x in enumerate(ref, 1):
        cur = [i]
        for j, y in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / len(ref)


def rail_ok(merged: str, a: str, b: str, limit: float = 0.6) -> bool:
    """Предохранитель: склейка должна быть близка хотя бы к одному входу."""
    if not merged or len(merged.strip()) < 2:
        return False
    if BAD_MARKERS.search(merged):
        return False
    return wer(a, merged) <= limit or wer(b, merged) <= limit


class StoredAsr:
    """Отдаёт заранее подготовленный текст (склейку) вместо вызова движка.

    backend_name="gigaam" — чтобы быстрый путь вёл себя как в бою: он смотрит на
    движок и на вид текста, а текст здесь уже итоговый от двух движков.
    """

    backend_name = "gigaam"

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def transcribe(self, pcm):
        key = hashlib.sha1(pcm.tobytes()).hexdigest()
        return AsrResult(text=self.mapping.get(key, ""), language="ru", latency_ms=390)

    def health(self) -> bool:
        return True


def main() -> int:
    idx = {r["id"]: r for r in (json.loads(line) for line in
           (WISPR_DIR / "phrases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    ab = json.loads((WISPR_DIR / "ab_asr.json").read_text(encoding="utf-8"))
    merged = json.loads((WISPR_DIR / "judge_merge.json").read_text(encoding="utf-8"))["merged"]

    mapping: dict[str, str] = {}
    rejected = 0
    for i, g in enumerate(ab["gigaam"]):
        row = idx[g["id"]]
        a, b, m = g["text"], ab["whisper"][i]["text"], merged[i]
        if rail_ok(m, a, b):
            text = m
        else:
            rejected += 1
            text = b                    # отказ -> whisper (сырой он точнее GigaAM)
        with wave.open(str(ROOT / row["audio"]), "rb") as fh:
            pcm_bytes = fh.readframes(fh.getnframes())
        mapping[hashlib.sha1(pcm_bytes).hexdigest()] = text
    print(f"предохранитель отклонил: {rejected} из {len(ab['gigaam'])}")

    whisper_spec = ServerSpec(
        key="asr", title="ASR whisper.cpp", port=8081,
        url="http://127.0.0.1:8081", health_path="/health",
        log=ROOT / "data" / "logs" / "whisper-merge.log",
        exe=whisper_dir() / "Release" / "whisper-server.exe",
        model=whisper_model_path(),
        args=["--host", "127.0.0.1", "--port", "8081", "-l", "ru", "-bs", "5", "-sns", "-t", "4"])
    llm_spec = next(s for s in default_specs() if s.key == "llm")
    sup = Supervisor([whisper_spec, llm_spec], enabled=True)
    try:
        print("whisper:", sup.start(whisper_spec),
              "| LLM:", sup.start(llm_spec), flush=True)
        print("whisper готов:", sup.wait_ready(whisper_spec, 180.0),
              "| LLM готов:", sup.wait_ready(llm_spec, 180.0), flush=True)
        print("LLM жива:", PolishClient(SettingsStore().get().polish).alive(), flush=True)
        asr_mod.make_asr_client = lambda cfg: StoredAsr(mapping)
        rows = [json.loads(line) for line in
                (WISPR_DIR / "phrases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        report = eval_vs_wispr.evaluate(rows, use_polish="on", router_client=None)
        eval_vs_wispr.write_report(report, WISPR_DIR / "report_llm_merge.md")
        (WISPR_DIR / "results_llm_merge.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in report["results"]),
            encoding="utf-8")
        print(f"\nWER после пайплайна: p50 {report['wer_vs_reference']['p50']} "
              f"| среднее {report['wer_vs_reference']['mean']}")
        print(f"Задержка: p50 {report['our_e2e']['p50']} | p90 {report['our_e2e']['p90']} мс")
        print(f"Маршруты: {report['routes']}")
        return 0
    finally:
        sup.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
