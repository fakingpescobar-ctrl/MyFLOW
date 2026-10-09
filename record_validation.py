"""Запись реальных данных для валидации порога VAD (v3 — автоматические тайминги).

Юзер САМ запускает:  & ".venv\Scripts\python.exe" -X utf8 record_validation.py
Скрипт сам ведёт по таймингам: отсчёт -> ГОВОРИ -> пауза -> следующая.
  - 5 фраз диктовки (по 7с записи каждая)
  - 4 короткие команды (по 4с каждая)
  - шумовой файл: 5с тишина + 10с шуршание/стук (НЕ ГОВОРИТЬ) + 5с тишина
Микрофон: Kinect USB Audio.
"""
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000
OUT_DIR = Path(__file__).resolve().parent / "data" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHRASES = [("Привет.", 7),
           ("Сегодня отличная погода.", 7),
           ("Проверим, как работает распознавание речи.", 8),
           ("Надо купить молоко и хлеб.", 7),
           ("Записываю очередную заметку.", 7)]
COMMANDS = [("Ок", 4), ("Стоп", 4), ("Дальше", 4), ("Назад", 4)]
TV_PHRASES = [("Сейчас работает телевизор.", 8),
              ("Диктую заметку при включённом телевизоре.", 9),
              ("Надо не забыть позвонить маме.", 7),
              ("Это последняя фраза с шумом на фоне.", 8)]


def find_kinect() -> int:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "kinect" in d["name"].lower():
            return i
    return int(sd.default.device[0])


def countdown(n: int) -> None:
    for i in range(n, 0, -1):
        print(f"    {i}...", flush=True)
        time.sleep(1)


def record_auto(device: int, out_name: str, prompt: str, dur: float, prep: int = 3) -> Path:
    path = OUT_DIR / out_name
    print(f"\n  >>> {prompt}")
    countdown(prep)
    print(f"  ГОВОРИ! (запись {dur:.0f}с)", flush=True)
    frames: list[np.ndarray] = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=device) as stream:
        got = 0.0
        while got < dur:
            chunk, _ = stream.read(int(0.25 * SR))
            frames.append(chunk[:, 0])
            got += 0.25
    data = np.concatenate(frames)
    sf.write(str(path), data, SR)
    rms = float(np.sqrt((data ** 2).mean()))
    print(f"  готово -> {path.name} (rms={rms:.4f})")
    time.sleep(1.5)
    return path


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    dev = find_kinect()
    print(f"Микрофон: [{dev}] {sd.query_devices(dev)['name']}")
    print("Сейчас пойдут фразы. Читай спокойно, без спешки. Между фразами скрипт сам подождёт.")

    if mode in ("all", "dictation"):
        print("\n=== ДИКТОВКА (5 фраз) ===")
        for i, (ph, dur) in enumerate(PHRASES):
            record_auto(dev, f"val_dict_{i+1}.wav", f"Фраза {i+1}/5: «{ph}»", dur)
    if mode in ("all", "commands"):
        print("\n=== КОМАНДЫ (4 штуки) ===")
        for i, (cmd, dur) in enumerate(COMMANDS):
            record_auto(dev, f"val_cmd_{i+1}.wav", f"Команда {i+1}/4: скажи «{cmd}»", dur)
    if mode in ("all", "noise"):
        print("\n=== ШУМ (НЕ ГОВОРИТЬ НИ СЛОВА!) ===")
        record_auto(dev, "val_noise.wav",
                    "ШУМ: 5с тишина + 10с шурши/стучи/кликай + 5с тишина. БЕЗ РЕЧИ!", 20, prep=5)
    if mode in ("all", "tv"):
        print("\n=== ТВОЯ РЕЧЬ + ТЕЛЕК НА ФОНЕ ===")
        print("СЕЙЧАС ВКЛЮЧИ ТЕЛЕВИЗОР/ВИДЕО на фоне (пусть говорит диктор/герои).")
        record_auto(dev, "val_tv_0.wav",
                    "Пауза на включение телека (10с): пусть играет, мы молчим", 10, prep=3)
        for i, (ph, dur) in enumerate(TV_PHRASES):
            record_auto(dev, f"val_tv_{i+1}.wav",
                        f"ТЕЛЕК ИГРАЕТ. Твоя фраза {i+1}/4: «{ph}»", dur)

    print("\nГотово. Файлы в:", OUT_DIR)


if __name__ == "__main__":
    main()