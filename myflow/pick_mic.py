"""MyFLOW — интерактивный выбор микрофона и режима при запуске.

TUI с клавиатурным управлением (без ввода цифр):
  стрелки ↑/↓ — перемещение, Enter — выбор, Esc — отмена запуска.
  Первое меню: микрофон (живые первыми). Второе: режим работы
  (auto — постоянно слушает / ptt — по нажатию кнопки F8).
  Третье: куда вставлять (обычно в активное окно / в чат OpenCode).

Алгоритм:
  1) собрать входные устройства sounddevice,
  2) отфильтровать технический мусор (мапперы, виртуальные CABLE, микшеры),
  3) ПРОВЕРИТЬ доступность каждого реального микрофона (по сигналу: 0.25с, RMS),
  4) показать список стрелками (живые — первыми с ✓, мёртвые — ✗),
  5) показать меню режима (auto/ptt),
  6) показать меню цели вставки (clipboard/opencode),
  7) сохранить выбор в data/settings.json (asr.input_device + mode + insert_method).

Exit codes:
    0 — выбор сделан / оставлен текущий
    3 — пользователь отменил запуск (Esc) / нет микрофонов
"""
from __future__ import annotations

import msvcrt
import re
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from myflow.config import SettingsStore, SAMPLE_RATE

# Технический мусор, который юзеру показывать незачем.
_BLACKLIST = (
    "переназначение", "первичный драйвер", "первичный звуковой",
    "cable output", "cable input", "vb-audio", "stereo mixer", "стерио микшер",
    "стерео микшер", "микшер", "spdif", "волновой", "wave out", "mapper",
    "loopback", "что слышно", "наушники", "headphones", "динамики", "speakers",
    "output", "монитор",
)

# Признаки микрофона в имени — только ДЛЯ ПОДСКАЗКИ в сортировке. Раньше по ним
# устройство отбрасывалось целиком, и микрофон с нестандартным именем
# (например, USB-интерфейс «Ardor Gaming») не попадал в список вообще.
_MIC_HINT = ("микрофон", "mic", "набор микрофонов")

# Порог «живости»: полностью мёртвое устройство даёт цифровую тишину ~1.5e-5,
# поэтому 3e-5 отделяет «нет сигнала» от «что-то идёт». Всё, что между 3e-5 и
# 1e-3, помечаем живым, но с предупреждением: это либо очень тихая комната,
# либо убавленное усиление входа — и тогда фразы будут отсекаться как «тишина»
# (в рантайме порог пика 0.01).
_DEAD_RMS = 3e-5
_QUIET_RMS = 1e-3
# Проба: короткий 0.25 с недостаточно — USB-интерфейсы и BT-микрофоны
# «раскачиваются» не мгновенно. Пишем 0.45 с и меряем RMS по хвосту.
_PROBE_SECONDS = 0.45
_PROBE_SKIP_SECONDS = 0.15


def _has_mic_hint(name: str) -> bool:
    """Похоже ли имя на микрофон (для приоритета в списке, не для фильтра)."""
    low = name.lower()
    return any(h in low for h in _MIC_HINT)


def _is_mic(name: str) -> bool:
    """Показывать ли устройство в списке входов.

    Критерий — «это вход, а не выход/виртуальный мусор». Раньше требовалось
    слово «микрофон» в имени: микрофоны с другим именем исчезали из списка
    (реальный кейс: «Ardor Gaming Vox» показывался только как «Kinect»).
    """
    low = name.lower()
    if any(b in low for b in _BLACKLIST):
        return False
    return True


def _probe_status(device: int | None, timeout: float = 2.5) -> tuple[bool, str]:
    """(жив_ли, причина). Причина важна: «занят» и «тишина» — разные проблемы.

    Открытие потока НЕ достаточно: MME/DirectSound «открываются» даже на
    выключенном блютуз-устройстве и молча отдают нули — поэтому смотрим RMS.
    Но и «не открылось» нельзя путать с «молчит»: первое обычно значит, что
    устройство занято другим приложением.
    """
    result: dict[str, object] = {"alive": False, "reason": "нет сигнала"}

    def _attempt() -> None:
        try:
            n = int(SAMPLE_RATE * _PROBE_SECONDS)
            rec = sd.rec(n, samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         device=device)
            sd.wait()
            x = np.nan_to_num(rec[:, 0]) if len(rec) else np.zeros(0)
            if not len(x):
                result["reason"] = "нет данных от устройства"
                return
            skip = int(SAMPLE_RATE * _PROBE_SKIP_SECONDS)
            tail = x[skip:] if len(x) > skip else x
            rms = float(np.sqrt(np.mean(tail ** 2)))
            if rms > _QUIET_RMS:
                result["alive"] = True
                result["reason"] = f"жив (rms={rms:.4f})"
            elif rms > _DEAD_RMS:
                # Сигнал есть, но тихий: микрофон рабочий, однако проверь
                # громкость входа — иначе фразы будут отсекаться как «тишина».
                result["alive"] = True
                result["reason"] = f"тихий вход (rms={rms:.5f}) — подними усиление?"
            else:
                result["reason"] = f"нет сигнала (rms={rms:.6f})"
        except Exception as exc:  # noqa: BLE001 — мёртвое/занятое так и помечаем
            msg = str(exc)
            short = msg.split("[PaErrorCode")[0].strip()
            short = short.replace("Error opening InputStream:", "").strip() or "ошибка"
            if "Invalid sample rate" in msg:
                result["reason"] = "не умеет 16 кГц — смени API"
            elif "busy" in msg.lower() or "unavailable" in msg.lower():
                result["reason"] = "занят другим приложением"
            elif "Invalid device" in msg:
                result["reason"] = "устройство исчезло (пересканируй)"
            else:
                reason = f"не открылся: {short}"
                result["reason"] = reason[:40]

    def _run() -> None:
        _attempt()
        if not result["alive"]:
            # Один повтор: сразу после старта/BT-пробуждения первый заход часто
            # падает, хотя устройство рабочее.
            time.sleep(0.15)
            _attempt()

    t = threading.Thread(target=_run, daemon=True, name="mic-probe")
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, "таймаут (драйвер завис)"
    return bool(result["alive"]), str(result["reason"])


def _probe(device: int | None, timeout: float = 2.5) -> bool:
    """Совместимость: просто «жив ли». Подробности — _probe_status()."""
    return _probe_status(device, timeout)[0]


def _hostapi_short(device: int) -> str:
    """Короткое имя host API устройства (MME/WASAPI/…) — для диагностики."""
    try:
        api = sd.query_devices(device)["hostapi"]
        name = sd.query_hostapis(api)["name"]
        return (name.replace("Windows ", "")
                    .replace("DirectSound", "DS")
                    .replace("WDM-KS", "WDM")
                    .strip())
    except Exception:  # noqa: BLE001
        return ""


def _cluster_key(name: str) -> str:
    """Ядро имени без пробелов/скобок — для склейки дублей одной физики."""
    return re.sub(r"[\s_()\[\]-]+", "", name.lower())


def _gather() -> list[dict]:
    """Входные устройства: одна запись на физику, с живостью и причиной.

    Возвращает [{"name", "idx", "alive", "status", "api"}], живые — первыми.
    Обрезанные имена сопоставляем префиксом: «...DualSense W» ⊆
    «...DualSense Wireless Controller».
    """
    cands: list[tuple[int, str]] = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        if _is_mic(d["name"].strip()):
            cands.append((i, d["name"].strip()))

    if not cands:
        return []

    probed = {i: _probe_status(i) for i, _ in cands}

    groups: list[dict] = []
    # Сначала имена с «микрофон/mic», потом остальные входы (Line In и т.п.)
    ordered = sorted(cands, key=lambda c: (not _has_mic_hint(c[1]), -len(c[1])))
    for idx, name in ordered:
        key = _cluster_key(name)
        match = next((g for g in groups
                      if g["key"].startswith(key) or key.startswith(g["key"])), None)
        alive, reason = probed[idx]
        if match is None:
            groups.append({"key": key, "name": name, "idx": idx, "alive": alive,
                           "status": reason, "api": _hostapi_short(idx)})
        else:
            # Дубли одной физики в разных API: берём первый ЖИВОЙ вариант,
            # его причину и его API (Ardor: MME живой, WASAPI не умеет 16 кГц).
            if alive and not match["alive"]:
                match["idx"], match["alive"] = idx, True
                match["status"], match["api"] = reason, _hostapi_short(idx)
            if len(match["name"]) < len(name):
                match["name"], match["key"] = name, key
    groups.sort(key=lambda g: (not g["alive"], g["idx"]))
    return groups


# ---------- TUI: клавиатурные меню (стрелки + Enter, без ввода цифр) ----------

def _render(title: str, items: list[tuple[str, object]], idx: int, hint: str) -> None:
    """Рисует блок меню и поднимает курсор на его начало (ANSI)."""
    lines = [f"\x1b[K{title}"]
    for n, (label, _) in enumerate(items):
        mark = "→ " if n == idx else "   "
        lines.append(f"\x1b[K{mark}{label}")
    lines.append(f"\x1b[K{hint}")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"\x1b[{len(lines)}A")
    sys.stdout.flush()


def _clear_block(n_lines: int) -> None:
    """Спуститься вниз на n_lines и очистить хвост экрана."""
    sys.stdout.write(f"\x1b[{n_lines}B\x1b[0J")
    sys.stdout.flush()


def _menu(title: str, items: list[tuple[str, object]], current_idx: int = 0,
          hint: str = "↑/↓ — выбор, Enter — принять, Esc — отмена") -> object | None:
    """Клавиатурное меню. Возвращает value выбранного пункта или None (отмена)."""
    if not items:
        return None
    idx = max(0, min(current_idx, len(items) - 1))
    n_lines = len(items) + 2
    try:
        while True:
            _render(title, items, idx, hint)
            ch = msvcrt.getch()
            if ch in (b"\x00", b"\xe0"):        # extended-сканкод (стрелки)
                ch = msvcrt.getch()
                if ch == b"H":                  # ↑
                    idx = (idx - 1) % len(items)
                elif ch == b"P":                # ↓
                    idx = (idx + 1) % len(items)
            elif ch in (b"\r", b"\n"):          # Enter
                _clear_block(n_lines)
                return items[idx][1]
            elif ch in (b"\x1b", b"q", b"Q"):   # Esc / q
                _clear_block(n_lines)
                return None
    except KeyboardInterrupt:
        _clear_block(n_lines)
        return None


_DEFAULT_SYS = "__default__"  # sentinel пункта «системный по умолчанию»
_CANCEL = object()           # отмена запуска; НЕ путать с None (системный!)


# ---------- выбор микрофона ----------

def _pick_mic_menu(store: SettingsStore) -> int | None | object:
    """Меню микрофона.

    Возвращает: int (индекс устройства) | None (системный по умолчанию)
    | _CANCEL (отмена пользователем / живых устройств нет).
    Раньше отмена и «системный по умолчанию» обе возвращали None — выбор
    системного микрофона приводил к «запуск отменён».
    """
    print("== MyFLOW: проверка доступности микрофонов ==")
    mics = _gather()
    if not mics:
        print("[MyFLOW] реальных микрофонов не найдено! Проверь подключение (блютуз/USB).")
        return _CANCEL

    current = store.get().asr.input_device
    sys_default_ok = _probe(None)
    live = [m for m in mics if m["alive"]]
    cur = next((m for m in mics if m["idx"] == current), None)

    print("  (живые микрофоны помечены ✓ — проверено по сигналу)")
    if cur is None:
        print(f"  Сейчас: {'#' + str(current) + ' (не в списке — отключён?)' if current is not None else 'системный по умолчанию'}")
    else:
        state = "доступен" if cur["alive"] else "НЕДОСТУПЕН — выбери живой!"
        print(f"  Сейчас: {cur['name']} — {state}")

    items: list[tuple[str, object]] = []
    for m in live:
        mark = "  [сейчас]" if m["idx"] == current else ""
        api = f" [{m['api']}]" if m.get("api") else ""
        items.append((f"\x1b[32m✓\x1b[0m {m['name']}{api}{mark}", m["idx"]))
    for m in mics:
        if not m["alive"]:
            why = f" — {m['status']}" if m.get("status") else ""
            items.append((f"\x1b[31m✗\x1b[0m {m['name']}\x1b[90m{why}\x1b[0m", m["idx"]))
    if sys_default_ok:
        items.append(("системный по умолчанию", _DEFAULT_SYS))
    else:
        items.append(("\x1b[31m✗\x1b[0m системный по умолчанию (недоступен)", _DEFAULT_SYS))

    if not live and not sys_default_ok:
        print("  Все микрофоны недоступны (выключены/отключены?). Проверь подключение.")
        return _CANCEL

    # Текущий выбираем по умолчанию, иначе первый живой.
    cur_idx = 0
    if current is not None:
        for n, (_, v) in enumerate(items):
            if v == current:
                cur_idx = n
                break
    elif not live:  # системный
        cur_idx = len(items) - 1

    print()
    picked = _menu(
        "== MyFLOW: выбери микрофон ==",
        items,
        current_idx=cur_idx,
        hint="↑/↓ — выбор, Enter — принять, Esc — отмена запуска",
    )
    if picked is None:
        return _CANCEL
    if picked == _DEFAULT_SYS:
        return None  # системный по умолчанию: input_device = None
    return picked


# ---------- выбор режима ----------

def _pick_mode_menu(store: SettingsStore) -> str | None:
    """Меню режима: auto (постоянно слушает) / ptt (по нажатию кнопки)."""
    mode_items = [
        ("авто — постоянно слушает микрофон, без кнопки", "auto"),
        ("по нажатию кнопки — PTT: держишь F8 и говоришь", "ptt"),
    ]
    current = store.get().mode
    cur_idx = 0
    for n, (_, v) in enumerate(mode_items):
        if v == current:
            cur_idx = n
            break
    return _menu(
        "== MyFLOW: режим работы ==",
        mode_items,
        current_idx=cur_idx,
        hint="↑/↓ — выбор, Enter — принять, Esc — отмена запуска",
    )


def _pick_insert_menu(store: SettingsStore) -> str | None:
    """Меню цели вставки: обычная (в активное окно) / в чат OpenCode."""
    insert_items = [
        ("обычно — в активное окно (буфер обмена + Ctrl+V)", "clipboard"),
        ("в чат OpenCode — инжект без фокуса (WriteConIn)", "opencode"),
    ]
    current = store.get().insert_method
    cur_idx = 0
    for n, (_, v) in enumerate(insert_items):
        if v == current:
            cur_idx = n
            break
    return _menu(
        "== MyFLOW: куда вставлять текст ==",
        insert_items,
        current_idx=cur_idx,
        hint="↑/↓ — выбор, Enter — принять, Esc — отмена запуска",
    )


def pick_mic() -> int:
    store = SettingsStore()
    # Без интерактивной консоли msvcrt.getch() бросает исключение (например,
    # запуск из планировщика/пайпа) — тогда просто оставляем текущие настройки.
    if sys.stdin is None or not sys.stdin.isatty():
        print("[MyFLOW] нет интерактивной консоли — оставляю текущие настройки "
              "(микрофон/режим/вставку можно поменять в data/settings.json).")
        return 0
    print()
    device = _pick_mic_menu(store)
    if device is _CANCEL:
        print("[MyFLOW] запуск отменён.")
        return 3
    print()
    mode = _pick_mode_menu(store)
    if mode is None:
        print("[MyFLOW] запуск отменён.")
        return 3
    print()
    insert = _pick_insert_menu(store)
    if insert is None:
        print("[MyFLOW] запуск отменён.")
        return 3

    store.get().asr.input_device = device
    store.get().mode = mode
    store.get().insert_method = insert
    store.save()
    mode_label = "авто (постоянно слушает)" if mode == "auto" else "по нажатию кнопки (F8)"
    insert_label = "в чат OpenCode" if insert == "opencode" else "в активное окно"
    dev_label = "системный" if device is None else f"#{device}"
    print(f"[MyFLOW] готово: микрофон {dev_label}, режим — {mode_label}, вставка — {insert_label}.")
    print("[MyFLOW] запускаю MyFLOW...")
    return 0


def main() -> None:
    # Имена устройств бывают с символами вне кодировки консоли — не паникуем.
    sys.stdout.reconfigure(errors="replace")
    sys.stdin.reconfigure(errors="replace")
    sys.exit(pick_mic())


# Публичные алиасы для TUI (внутренние имена оставлены, чтобы ничего не ломать).
gather_mics = _gather
probe_device = _probe
is_mic_device = _is_mic


if __name__ == "__main__":
    main()