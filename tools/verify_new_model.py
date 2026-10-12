"""Проверить непроверенную модель из дообучения и решить, ставить ли её.

Инцидент 12.10.2026: цикл не смог сравнить модели (порт 8085 держала сторонняя
ScpTrayApp) и — из-за тогдашней ошибки в хранителе — поставил новую модель без
проверки. Модель откатили, хранитель починили, а саму новую модель надо проверить:
если она не хуже, поставить; если хуже — оставить как есть и выбросить.

    python tools/verify_new_model.py                      # новая из data/train/gguf
    python tools/verify_new_model.py --new <путь.gguf>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from compare_postproc_models import compare, should_install  # noqa: E402
from build_postproc_dataset import real_pairs                # noqa: E402

SHIPPED = ROOT / "models" / "myflow-postproc-qwen3-06b-q5_k_m.gguf"
DEFAULT_NEW = ROOT / "data" / "train" / "gguf" / "myflow-postproc-live-q5_k_m.gguf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, default=DEFAULT_NEW)
    ap.add_argument("--current", type=Path, default=SHIPPED)
    ap.add_argument("--holdout", type=int, default=80)
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--install", action="store_true", help="поставить, если не хуже")
    args = ap.parse_args()

    for path, label in ((args.current, "текущая"), (args.new, "новая")):
        if not path.exists():
            print(f"  нет файла ({label}): {path}", file=sys.stderr)
            return 1

    ordered = sorted(real_pairs(), key=lambda item: item["id"])
    pairs = ordered[-args.holdout:]
    print(f"  проверяю на {len(pairs)} отложенных диктовках (id "
          f"{pairs[0]['id']}..{pairs[-1]['id']})")
    print(f"  текущая: {args.current.name}")
    print(f"  новая:   {args.new.name}")

    means = compare(args.current, args.new, pairs)
    print(f"\n  среднее расстояние до эталона:")
    print(f"    текущая {means['current']:.4f}")
    print(f"    новая   {means['new']:.4f}")

    verdict = should_install(means["new"], means["current"], tolerance=args.tolerance)
    delta = means["new"] - means["current"]
    print(f"  разница: {delta:+.4f} ({'хуже' if delta > 0 else 'не хуже'} текущей)")

    if verdict:
        print("\n  ВЕРДИКТ: новая не хуже — можно ставить")
        if args.install:
            import shutil
            backup = args.current.with_suffix(".previous.gguf")
            shutil.copyfile(args.current, backup)
            args.current.unlink()
            shutil.copyfile(args.new, args.current)
            print(f"  поставлена (прежняя в {backup.name})")
        else:
            print("  (чтобы поставить: --install)")
    else:
        print("\n  ВЕРДИКТ: новая ХУЖЕ — оставляем текущую, новую не ставим")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
