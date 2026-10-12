"""Статистика текста для чтения: структура абзацев без вывода содержимого.

Нужна, чтобы понять, как форматировать: одни ли строки = абзацы, есть ли переносы
внутри абзацев, сколько разделителей сцен.
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            r"data\bench\private\klinok-ch1.txt")
if not path.exists():
    print(f"  нет файла: {path}")
    raise SystemExit(1)

text = path.read_text(encoding="utf-8")
lines = text.splitlines()

non_empty = [line for line in lines if line.strip()]
empty = [line for line in lines if not line.strip()]
lengths = [len(line.strip()) for line in non_empty]

print(f"  файл: {path.name} — {len(text)} знаков, {len(text.split())} слов")
print(f"  строк: {len(lines)} (непустых {len(non_empty)}, пустых {len(empty)})")
print(f"  длина строки: средняя {sum(lengths) / max(len(lengths), 1):.0f}, "
      f"максимум {max(lengths) if lengths else 0}")
print(f"  разделителей сцен (***): {sum(1 for line in non_empty if set(line.strip()) == {'*'})}")
print(f"  строк с диалогом (начинаются с тире): "
      f"{sum(1 for line in non_empty if line.strip().startswith(('–', '—', '-')))}")
print(f"  строк с двойными пробелами: {sum(1 for line in lines if '  ' in line)}")
print(f"  строк с хвостовыми пробелами: "
      f"{sum(1 for line in lines if line != line.rstrip())}")
# похоже ли на жёсткие переносы внутри абзаца: строка не заканчивается знаком
mid_sentence = sum(1 for line in non_empty
                   if line.strip()[-1:] not in '.!?…»"*' and not line.strip().endswith('*'))
print(f"  строк, кончающихся без точки (намек на перенос внутри абзаца): {mid_sentence}")
