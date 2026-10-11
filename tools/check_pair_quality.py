"""Доля «склеенных» пар в истории: вход и цель — про разные диктовки.

`edited_text` в истории иногда относится к ДРУГОЙ фразе (наблюдатель AX подхватил
чужой текст поля ввода). Такие пары не шумят, а прямо вредят: обучают модель
переписывать фразы целиком и портят замер всем подходам сразу.

Признак склейки — большое расхождение между сырым ASR и «итогом»: настоящая
полировка так не переписывает.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from build_postproc_dataset import real_pairs          # noqa: E402
from eval_postproc import distance, words              # noqa: E402

THRESHOLDS = (0.4, 0.5, 0.6, 0.8, 1.0)


def main() -> int:
    pairs = real_pairs()
    dists = [distance(p["dst"], p["src"]) for p in pairs]
    total = len(dists)
    print(f"  реальных пар с текстом: {total}")
    for thr in THRESHOLDS:
        bad = sum(1 for d in dists if d > thr)
        print(f"    расхождение > {thr:.1f}: {bad:>4} ({bad / total * 100:4.1f}%) "
              f"-> останется {total - bad}")

    print("\n  примеры подозрительных пар (расхождение >= 0.8):")
    shown = 0
    for p in pairs:
        d = distance(p["dst"], p["src"])
        if d >= 0.8 and shown < 6:
            shown += 1
            print(f"    {d:.2f} | вход: {p['src'][:66]}")
            print(f"         | цель: {p['dst'][:66]}")
    print("\n  примеры нормальных пар (расхождение <= 0.3):")
    shown = 0
    for p in pairs:
        d = distance(p["dst"], p["src"])
        if d <= 0.3 and shown < 4:
            shown += 1
            print(f"    {d:.2f} | вход: {p['src'][:66]}")
            print(f"         | цель: {p['dst'][:66]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
