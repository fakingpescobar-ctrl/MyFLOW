"""Скачать модели одной командой (для нового пользователя).

Что нужно MyFLOW:
  * whisper (ggml-large-v3-turbo, 547 МБ) — обычный файл, качаем здесь;
  * GigaAM v3 e2e (ONNX int8, 214 МБ) — его тянет сам `onnx-asr` при первом запуске,
    поэтому мы только проверяем и подсказываем;
  * постпроцессор (наша LoRA, 424 МБ) — едет в поставке вместе с MyFLOW, из сети его
    брать негде.

Почему свой велосипед: `huggingface_hub` и pip в этом окружении зависают на середине
загрузки, а curl с `-C -` (докачка), `--speed-limit/--speed-time` и `--retry` качает
надёжно — на этом уже писали tools/fetch_hf_model.py.

    python tools/fetch_models.py            # скачать недостающее
    python tools/fetch_models.py --check     # только показать, чего нет
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from myflow.bootstrap import MODEL_FILES, download_model

MODELS = ROOT / "models"
HF = "https://huggingface.co/{repo}/resolve/main/{name}"

# Whisper живёт в bootstrap (там же — URL, размер и докачка).
# Здесь только порог «файл считается скачанным».
WHISPER_MODEL = next(m for m in MODEL_FILES if m.label == "ASR whisper")
WHISPER = {
    "what": "whisper large-v3-turbo",
    "path": WHISPER_MODEL.destination(),
    "min_mb": 400,
}


def size_mb(path: Path) -> float:
    return path.stat().st_size / 2**20 if path.exists() else 0.0


def check_whisper() -> bool:
    """Готово ли. Порог — тот же, что в bootstrap (размер с допуском 1 %):
    оборванный файл не должен считаться скачанным."""
    return WHISPER_MODEL.present()


def check_gigaam() -> tuple[bool, Path | None]:
    """GigaAM лежит в кэше HF; его скачивает onnx-asr при первом запуске."""
    cache = MODELS / "hf-cache" / "hub"
    if not cache.exists():
        return False, None
    found = [p for p in cache.rglob("*.onnx") if p.stat().st_size > 50 * 2**20]
    return bool(found), (found[0] if found else None)


def check_postproc() -> tuple[bool, float]:
    found = [p for p in MODELS.glob("myflow-postproc-*.gguf")]
    if not found:
        return False, 0.0
    best = max(found, key=lambda p: p.stat().st_size)
    return True, size_mb(best)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="ничего не качать")
    args = ap.parse_args()

    print("\n  МОДЕЛИ MyFLOW")
    missing: list[str] = []

    whisper_ok = check_whisper()
    print(
        f"  [{'ok' if whisper_ok else 'НЕТ'}] whisper: "
        f"{size_mb(WHISPER['path']):.0f} МБ"
    )
    if not whisper_ok and not args.check:
        if download_model(WHISPER_MODEL):
            whisper_ok = True
            print(f"    готово: {size_mb(WHISPER['path']):.0f} МБ")
        else:
            print("    не получилось — повтори запуск, докачка продолжится")
    if not whisper_ok:
        missing.append("whisper (этот скрипт)")

    giga_ok, giga_path = check_gigaam()
    print(
        f"  [{'ok' if giga_ok else 'НЕТ'}] GigaAM v3 e2e: "
        f"{size_mb(giga_path) if giga_path else 0:.0f} МБ"
    )
    if not giga_ok:
        missing.append(
            "GigaAM — скачается сам при первом запуске "
            "распознавания (нужен интернет один раз)"
        )

    post_ok, post_mb = check_postproc()
    print(
        f"  [{'ok' if post_ok else 'НЕТ'}] постпроцессор (наша LoRA): {post_mb:.0f} МБ"
    )
    if not post_ok:
        missing.append(
            "постпроцессор — он едет в поставке вместе с MyFLOW "
            "(myflow-postproc-*.gguf в папке models/)"
        )

    print()
    if missing:
        print("  ЧТО ЕЩЁ НУЖНО:")
        for item in missing:
            print(f"    - {item}")
        return 1
    print("  ВСЕ МОДЕЛИ НА МЕСТЕ ✓ дальше: python tools/doctor.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
