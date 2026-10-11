# Дообучение постпроцессора MyFLOW (Qwen3-0.6B/1.7B + LoRA)

Зачем: у лидера ниши (superwhisper) для этой роли работает **маленькая
специализированная** модель — их `s1-mini` весит 462 МБ, основан на Qwen3 и
дообучен под punctuation / truecasing / text-normalization (теги их карточки на
HuggingFace). Топовость здесь даёт **специализация**, а не размер. У нас есть
свои данные (реальные диктовки + правки), значит мы можем собрать такое же
решение, но **под русский язык и под наш словарь** — чего у них нет.

## Что уже готово (этап 0)

| Инструмент | Что делает |
|---|---|
| `tools/build_postproc_dataset.py` | датасет: реальные пары из истории + синтетика |
| `tools/eval_postproc.py` | замер на парах «сырой ASR → эталон» (raw / light_clean / LLM) |
| `tools/train_postproc_lora.py` | LoRA-обучение с маскированием промпта |

Датасет (11.10.2026): **8482 примера** на обучение (8114 синтетики + 368 реальных),
**64 реальные пары** на проверку (они НЕ попадают в обучение).

Цель для сравнения (те же 64 пары, модель Qwen3-4B Q8):

| Подход | расстояние p50 | среднее | точных | задержка p50 |
|---|---|---|---|---|
| сырой ASR | 0.100 | 0.357 | 43.8 % | — |
| `light_clean` (правила) | 0.100 | 0.357 | 42.2 % | 0 мс |
| LLM Qwen3-4B (сейчас) | **0.000** | **0.289** | **59.4 %** | 178 мс |

Задача дообученной 0.6B — **не уступить последней строке**, но за ~50 мс и при
400-600 МБ вместо 4 ГБ.

## Этап 1 — учебное окружение (отдельно от боевого!)

Боевой `.venv` трогать нельзя: в нём стоит torch **CPU-сборки**, а обучение нужно
на GPU. Поэтому учебное окружение — своё:

```powershell
py -3.13 -m venv .venv-train
.venv-train\Scripts\python.exe -m pip install --upgrade pip
# CUDA-сборка torch (≈2.5 ГБ). CUDA 12.4 подходит для драйвера 616.x
.venv-train\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv-train\Scripts\python.exe -m pip install transformers peft accelerate datasets
```

Проверка, что GPU виден:

```powershell
.venv-train\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Этап 2 — обучение

```powershell
# пересобрать датасет, если появились новые диктовки
.venv\Scripts\python.exe -X utf8 tools\build_postproc_dataset.py

# быстрый прогон (200 примеров) — проверить, что всё работает
.venv-train\Scripts\python.exe tools\train_postproc_lora.py --smoke

# полное обучение 0.6B
.venv-train\Scripts\python.exe tools\train_postproc_lora.py --base Qwen/Qwen3-0.6B --epochs 3
# если качество не дотянет — то же для 1.7B (влезает в 10 ГБ):
.venv-train\Scripts\python.exe tools\train_postproc_lora.py --base Qwen/Qwen3-1.7B --batch 4
```

Адаптер сохраняется в `data/train/postproc_lora`.

## Этап 3 — замер против 4B

```powershell
# 1) слить адаптер с базой, собрать GGUF и поднять на отдельном порту
.venv-train\Scripts\python.exe tools\merge_postproc_lora.py --adapter data/train/postproc_lora --out data/train/postproc_merged
powershell -File tools\export_postproc_gguf.ps1 -Merged data\train\postproc_merged
bin\llama-cpp\llama-server.exe -m models\myflow-postproc-qwen3-06b-q8_0.gguf --host 127.0.0.1 --port 8083 -c 8192 -np 1 -ngl 99 -fa on --jinja

# 2) сравнить с текущей моделью на одних и тех же парах
.venv\Scripts\python.exe -X utf8 tools\eval_postproc.py --llm --url http://127.0.0.1:8083/v1
```

И на 624 реальных фразах (наш основной стенд):

```powershell
.venv\Scripts\python.exe -X utf8 tools\eval_vs_wispr.py --polish auto --report data\eval_wispr\report_postproc.md
```

## Этап 4 — встройка в пульт

Если 0.6B не уступает 4B: собрать GGUF (`tools/export_postproc_gguf.ps1`), положить
в `models/`, оставить 4B только для «Oracle LLM» (там нужна смекалка, а не
форматирование), а полировку перевести на маленькую модель. Ожидаемый эффект:
полировка ~50 мс вместо 178, освобождение ~3.5 ГБ VRAM под whisper/Oracle.

## Ограничения и честные риски

* 432 реальные диктовки — мало для обучения «с нуля»; поэтому 95 % датасета —
  синтетика с нашим же профилем ошибок ASR (регистр, пунктуация, числа словами,
  филлеры, путаница терминов). Это стандартный приём, но он учит модель чинить
  именно эти ошибки, а не любые;
* синтетика может «переучить» на механическое форматирование: следим по
  eval-loss на реальных парах и по 624 фразам Wispr;
* маленькая модель рискует галлюцинировать на длинных фразах — предохранитель
  `_guard_polish` в пайплайне остаётся и откатывает подозрительные правки.
