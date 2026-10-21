# Сторонние компоненты и лицензии

> **Сам проект MyFLOW** распространяется под **MIT** — см. [LICENSE](LICENSE).
> Этот файл перечисляет только **сторонние** компоненты: они идут на своих
> условиях, и те условия нужно соблюдать при распространении сборки.

MyFLOW — локальное приложение: модели и бинарники лежат внутри репозитория
(каталоги `models/` и `bin/`, в git не попадают). Ниже — что именно используется,
версии и лицензии. Данные о Python-пакетах взяты из метаданных установленных
дистрибутивов (`importlib.metadata`), о моделях — из карточек HuggingFace
(поле `license`), о бинарниках — с их сайтов.

> Полные тексты лицензий — по ссылкам в столбце «Источник». Этот файл нужен,
> чтобы было видно, что и на каких условиях мы используем.

## Модели

| Компонент | Где лежит | Лицензия | Источник |
|---|---|---|---|
| GigaAM v3 e2e CTC (ONNX, int8) — распознавание речи | `models/hf-cache/` | **MIT** | [istupakov/gigaam-v3-onnx](https://huggingface.co/istupakov/gigaam-v3-onnx) (база: ai-sage/GigaAM-v3) |
| Qwen3-4B-Instruct-2507 (Q8_0) — полировка текста | `models/Qwen3-4B-Instruct-2507-Q8_0.gguf` | **Apache-2.0** | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| WeSpeaker ResNet34-LM (ONNX) — проверка голоса | `models/voiceprint/model.onnx` | **CC-BY-4.0** | [onnx-community/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM) |
| Silero VAD v6 (ONNX) — детектор речи | `data/vad_models/silero_v6.onnx` | **MIT** | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| Whisper large-v3-turbo (ggml, q5_0) — опциональный ASR-fallback | `models/ggml-large-v3-turbo-q5_0.bin` | **MIT** | [openai/whisper](https://github.com/openai/whisper) |

**CC-BY-4.0 (модель голоса) требует указания авторства.** Оно выполнено этой
таблицей: модель создана сообществом WeSpeaker / onnx-community, конвертация и
публикация — [onnx-community/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM).

## Бинарники

| Компонент | Где лежит | Лицензия | Источник |
|---|---|---|---|
| llama.cpp (`llama-server.exe` + CUDA-сборка) | `bin/llama-cpp/` | **MIT** | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| whisper.cpp (`whisper-server.exe`) | `bin/whisper/Release/` | **MIT** | [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) |
| CUDA runtime (в составе сборок) | там же | NVIDIA CUDA Toolkit EULA | [NVIDIA](https://docs.nvidia.com/cuda/eula/index.html) |

## Python-пакеты (обязательные)

| Пакет | Версия | Лицензия |
|---|---|---|
| numpy | 2.5.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| sounddevice | 0.5.6 | MIT |
| pyperclip | 1.11.0 | BSD |
| requests | 2.34.2 | Apache-2.0 |
| onnxruntime | 1.20.1 | MIT |
| pynput | 1.8.2 | LGPLv3 |
| cffi | 2.1.1 | MIT-0 |
| six | 1.17.0 | MIT |
| pycparser | 3.0 | BSD-3-Clause |
| certifi | 2026.7.22 | MPL-2.0 |
| charset-normalizer | 3.5.1 | MIT |
| idna | 3.19 | BSD-3-Clause |
| urllib3 | 2.7.0 | MIT |

`pynput` распространяется под **LGPLv3**: мы используем его как библиотеку без
изменений и не статически линкуем — обязательств по раскрытию исходников MyFLOW
это не создаёт.

## Python-пакеты (опциональные — ASR GigaAM и проверка голоса)

| Пакет | Версия | Лицензия |
|---|---|---|
| onnx-asr | 0.12.0 | MIT |
| huggingface_hub | 2.2.0 | Apache-2.0 |
| hf-xet | 1.7.0 | Apache-2.0 |
| kaldi-native-fbank | 1.22.3 | Apache-2.0 |
| kaldiio | 2.18.1 | не указана в метаданных пакета (используется только в tools/) |
| protobuf | 7.36.1 | BSD-3-Clause |
| flatbuffers | 25.12.19 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| filelock | 3.32.5 | MIT |
| fsspec | 2026.7.0 | BSD-3-Clause |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| PyYAML | 6.0.3 | MIT |
| coloredlogs | 15.0.1 | MIT |
| humanfriendly | 10.0 | MIT |
| Jinja2 | 3.1.6 | BSD-3-Clause |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| mpmath / networkx / sympy | 1.3.0 / 3.6.1 / 1.14.0 | BSD / BSD-3-Clause / BSD |
| anyio, h11, httpcore2, httpx2, click, colorama | 4.15.1 / 0.16.0 / 2.13.1 / 2.13.1 / 8.5.0 / 0.4.6 | MIT / MIT / BSD-3-Clause / BSD-3-Clause / BSD-3-Clause / BSD |
| pyreadline3 | 3.5.6 | BSD |

## Только для инструментов в `tools/` (не нужны для работы приложения)

| Пакет | Версия | Лицензия |
|---|---|---|
| soundfile | 0.14.0 | BSD-3-Clause |
| silero-vad | 6.2.1 | MIT |
| webrtcvad-wheels | 2.0.14 | MIT |
| ten-vad | 1.0.6.8 | не указана в метаданных пакета |
| fireredvad | 0.0.2 | Apache-2.0 |

## Благодарности

Спасибо авторам всех перечисленных проектов: llama.cpp и whisper.cpp (Georgi
Gerganov и сообщество), GigaAM (SberDevices / ai-sage) и конвертации istupakov,
Qwen (Alibaba), WeSpeaker (wenet-community), Silero (snakers4), onnx-asr
(istupakov), onnxruntime (Microsoft).
