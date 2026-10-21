"""Дообучение постпроцессора на свежих диктовках — одна команда.

Запускается вручную или из пульта (строка «Дообучение»). Что делает:

1. собирает датасет из ТЕКУЩЕЙ истории (`data/myflow.sqlite`): реальные пары
   «что услышали -> что осталось» + синтетика с нашим профилем ошибок, с балансом
   «ничего не менять» (иначе модель начинает править готовый текст);
2. обучает LoRA в учебном окружении `.venv-train` (CUDA-torch);
3. сливает адаптер, конвертирует в GGUF и квантует (Q5_K_M — целевой размер);
4. **делает резервную копию** текущей модели и ставит новую в `models/`;
5. пишет состояние в `data/train/state.json` (сколько диктовок, когда, какая модель).

Прогресс пишется в `data/retrain.log` (пульт показывает последнюю строку).

Запуск:
    .venv\\Scripts\\python.exe -X utf8 tools\\retrain_postproc.py            # полный цикл
    .venv\\Scripts\\python.exe -X utf8 tools\\retrain_postproc.py --dry-run  # только план
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

LOG = ROOT / "data" / "retrain.log"
STATE = ROOT / "data" / "train" / "state.json"
WORK = ROOT / "data" / "train" / "live"
MODELS = ROOT / "models"
SHIPPED = MODELS / "myflow-postproc-qwen3-06b-q5_k_m.gguf"
TRAIN_PY = ROOT / ".venv-train" / "Scripts" / "python.exe"
RUNTIME_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def count_dictations() -> tuple[int, int]:
    import sqlite3
    db = ROOT / "data" / "myflow.sqlite"
    if not db.exists():
        return 0, 0
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    with_text = con.execute(
        "SELECT COUNT(*) FROM history WHERE asr_text IS NOT NULL "
        "AND TRIM(asr_text) != ''").fetchone()[0]
    max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM history").fetchone()[0]
    con.close()
    return int(with_text), int(max_id)


def run(cmd: list[str], title: str) -> bool:
    log(title)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        log(f"ОШИБКА на шаге «{title}» (код {proc.returncode})")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Дообучение постпроцессора на свежих диктовках")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--variants", type=int, default=4, help="порч на чистый текст")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="чистый текст для синтетики (напр. corpus/conversational.txt)")
    ap.add_argument("--corpus-share", type=float, default=1.0)
    ap.add_argument("--min-words", type=int, default=0,
                    help="не учить на диктовках короче N слов (короткие учат бездействию)")
    ap.add_argument("--identity-share", type=float, default=0.45,
                    help="доля примеров «ничего не менять»")
    ap.add_argument("--quant", default="Q5_K_M")
    ap.add_argument("--backup", action="store_true", default=True)
    ap.add_argument("--holdout", type=int, default=80,
                    help="последние N диктовок не учим, а проверяем на них новую модель")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="допустимое ухудшение относительно текущей модели (5%%)")
    ap.add_argument("--force", action="store_true",
                    help="установить новую модель даже если она хуже")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TRAIN_PY.exists():
        log("нет учебного окружения .venv-train — запусти install-train-env.bat")
        return 1

    base = ROOT / "data" / "train" / "Qwen3-0.6B"
    if not (base / "model.safetensors").exists():
        log("нет базы Qwen3-0.6B для обучения — скачай её:")
        log("  python tools/fetch_hf_model.py Qwen/Qwen3-0.6B --dir data\\train\\Qwen3-0.6B")
        return 1

    # Корпус для синтетики берём сами, если он уже скачан: без него датасет
    # состоит только из твоих диктовок, а для первого обучения их мало.
    corpus = args.corpus
    if corpus is None:
        default_corpus = ROOT / "data" / "train" / "corpus" / "conversational.txt"
        corpus = default_corpus if default_corpus.exists() else None
    if corpus is None:
        log("корпуса для синтетики нет (покажем на одних твоих диктовках)")
    else:
        log(f"корпус для синтетики: {corpus.name}")

    dictations, max_id = count_dictations()
    log(f"старт дообучения: диктовок с текстом {dictations}, max id {max_id}")
    WORK.mkdir(parents=True, exist_ok=True)

    # 1) датасет
    if not run([str(RUNTIME_PY), "-X", "utf8", "tools/build_postproc_dataset.py",
                "--out", str(WORK), "--variants", str(args.variants),
                "--identity-share", str(args.identity_share),
                "--min-words", str(args.min_words),
                "--corpus", str(corpus) if corpus else "",
                "--corpus-share", str(args.corpus_share),
                "--holdout-last", str(args.holdout)],
               "собираю датасет из текущей истории"):
        return 1
    train_file = WORK / "postproc_train.jsonl"
    val_file = WORK / "postproc_val.jsonl"
    rows = sum(1 for _ in train_file.read_text(encoding="utf-8").splitlines() if _.strip())
    log(f"датасет: {rows} примеров, проверка {sum(1 for _ in val_file.read_text(encoding='utf-8').splitlines() if _.strip())}")

    if args.dry_run:
        log("dry-run: обучение не запускаю")
        return 0

    # 2) обучение
    adapter = WORK / "lora"
    if not run([str(TRAIN_PY), "-X", "utf8", "tools/train_postproc_lora.py",
                "--base", "data/train/Qwen3-0.6B", "--train", str(train_file),
                "--val", str(val_file), "--max-len", "384", "--batch", "2",
                "--accum", "8", "--epochs", str(args.epochs), "--out", str(adapter)],
               "обучаю LoRA на Qwen3-0.6B"):
        return 1

    # 3) GGUF
    if not run([str(RUNTIME_PY), "-X", "utf8", "tools/postproc_pipeline.py",
                "--adapter", str(adapter), "--merged", str(WORK / "merged"),
                "--name", "myflow-postproc-live", "--quant", args.quant],
               "собираю GGUF"):
        return 1
    built = ROOT / "data" / "train" / "gguf" / f"myflow-postproc-live-{args.quant.lower()}.gguf"
    if not built.exists():
        log(f"не нашёл собранную модель: {built}")
        return 1

    # 3.5) ЗАЩИТА: измеряем новую модель на диктовках, которых она не видела, и
    # ставим только если она не хуже текущей. Проверено на себе: обучение на всех
    # диктовках (включая короткие «уже чистые») дало 0.170 против 0.140 — то есть
    # автоматическая установка «свежей» модели ухудшила полировку.
    if SHIPPED.exists() and args.holdout > 0:
        from compare_postproc_models import compare, should_install
        from build_postproc_dataset import real_pairs
        ordered = sorted(real_pairs(), key=lambda x: x["id"])
        holdout_pairs = ordered[-args.holdout:]
        log(f"проверяю новую модель на {len(holdout_pairs)} отложенных диктовках")
        try:
            means = compare(SHIPPED, built, holdout_pairs)
            log(f"среднее расстояние: новая {means['new']:.3f} "
                f"против текущей {means['current']:.3f}")
            if not should_install(means["new"], means["current"],
                                  tolerance=args.tolerance, force=args.force):
                log("новая модель ХУЖЕ текущей — оставляю прежнюю "
                    "(переопределить: --force)")
                return 0
        except Exception as exc:  # noqa: BLE001
            # Раньше здесь была автоматическая установка «на свой страх» — и именно
            # так 12.10.2026 непроверенная модель попала в пульт: сравнение не смогло
            # поднять сервер (порт для замера заняла сторонняя программа), а цикл всё
            # равно поставил новую модель. Хранитель обязан отказывать.
            if not args.force:
                log(f"сравнение не удалось ({type(exc).__name__}: {exc})")
                log("НЕ устанавливаю: проверка не прошла. Текущая модель в пульте "
                    "не тронута, новая лежит в data/train/gguf")
                log("подсказка: порт для замера может занимать другая программа — "
                    "освободи его и запусти проверку заново, либо --force осознанно")
                return 1
            log(f"сравнение не удалось ({type(exc).__name__}: {exc}) — "
                "--force указан, ставлю без проверки")

    # 4) ставим в пульт с резервной копией
    if args.backup and SHIPPED.exists():
        backup = SHIPPED.with_suffix(".previous.gguf")
        shutil.copyfile(SHIPPED, backup)
        log(f"резервная копия прежней модели: {backup.name}")
    MODELS.mkdir(parents=True, exist_ok=True)
    if SHIPPED.exists():
        SHIPPED.unlink()
    shutil.copyfile(built, SHIPPED)
    log(f"новая модель на месте: {SHIPPED.name} ({SHIPPED.stat().st_size / 2**20:.0f} МБ)")

    # 5) состояние
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dictations": dictations,
        "max_id": max_id,
        "examples": rows,
        "model": SHIPPED.name,
        "quant": args.quant,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("готово: модель обновлена, состояние записано")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
