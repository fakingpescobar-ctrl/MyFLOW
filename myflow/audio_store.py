"""MyFLOW — хранение аудио последних фраз (аналог audio/opusChunks у Wispr).

Зачем: если распознавание или полировка упали, фраза сейчас теряется — в логе
остаётся только ошибка. С сохранённым аудио её можно перераспознать позже
(`myflow.cli retry`) или дослать в LLM, когда сервер поднимется.

Хранение: Int16 PCM 16 кГц моно, сжатый zlib (stdlib, без зависимостей).
Речь сжимается примерно в 1.5-2 раза; 50 фраз по ~5 с — это ~3-5 МБ на диске.
Если установлен soundfile (libsndfile), пишем OGG/Vorbis — в 5-8 раз компактнее.

Файлы: data/audio/<history_id>.pcm.z (или .ogg) + ротация по количеству.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_KEEP = 50
_MAGIC_Z = b"MFA1"          # MyFLOW audio, zlib-сжатый Int16 PCM


def _soundfile():
    """soundfile опционален: без него остаётся zlib-путь."""
    try:
        import soundfile as sf
        return sf
    except Exception:  # noqa: BLE001
        return None


@dataclass
class AudioStore:
    """Каталог с аудио последних фраз и ротацией."""

    directory: Path
    keep: int = DEFAULT_KEEP

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    # ---- запись/чтение ----

    def save(self, history_id: int, pcm_int16: np.ndarray) -> Path | None:
        """Сохранить фразу. None — не получилось (не роняем пайплайн)."""
        if pcm_int16 is None or len(pcm_int16) == 0:
            return None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            sf = _soundfile()
            if sf is not None:
                path = self.directory / f"{history_id}.ogg"
                try:
                    sf.write(str(path), np.asarray(pcm_int16, dtype=np.int16),
                             16000, format="OGG", subtype="VORBIS")
                    self.prune()
                    return path
                except Exception:  # noqa: BLE001 — нет кодека: падаем на zlib
                    pass
            path = self.directory / f"{history_id}.pcm.z"
            raw = np.ascontiguousarray(pcm_int16, dtype=np.int16).tobytes()
            path.write_bytes(_MAGIC_Z + zlib.compress(raw, 6))
            self.prune()
            return path
        except Exception:  # noqa: BLE001
            return None

    def load(self, path: str | Path) -> np.ndarray | None:
        """Прочитать аудио обратно (Int16 PCM)."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            if path.suffix == ".ogg":
                sf = _soundfile()
                if sf is None:
                    return None
                data, _rate = sf.read(str(path), dtype="int16")
                return np.asarray(data, dtype=np.int16).reshape(-1)
            blob = path.read_bytes()
            if blob[:4] != _MAGIC_Z:
                return None
            raw = zlib.decompress(blob[4:])
            return np.frombuffer(raw, dtype=np.int16).copy()
        except Exception:  # noqa: BLE001
            return None

    def path_for(self, history_id: int) -> Path | None:
        for ext in (".ogg", ".pcm.z"):
            p = self.directory / f"{history_id}{ext}"
            if p.exists():
                return p
        return None

    # ---- ротация ----

    def files(self) -> list[Path]:
        if not self.directory.exists():
            return []
        out = [p for p in self.directory.iterdir()
               if p.suffix in (".z", ".ogg") and p.is_file()]
        return sorted(out, key=lambda p: p.stat().st_mtime)

    def prune(self, keep: int | None = None) -> int:
        """Оставить последние keep файлов. Возвращает число удалённых."""
        limit = self.keep if keep is None else keep
        files = self.files()
        removed = 0
        for path in files[:-limit] if limit > 0 else files:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.files())

    def stats(self) -> dict:
        files = self.files()
        return {"files": len(files), "bytes": self.total_bytes(),
                "keep": self.keep}
