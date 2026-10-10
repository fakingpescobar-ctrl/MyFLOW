"""Тесты одноконсольного TUI: рендер кадров, состояние выбора, супервизор серверов.

Реальную консоль не трогаем (msvcrt/WinAPI проверяются в tools/smoke_tui.py):
здесь — чистая логика, которая ломалась бы незаметно.
"""
import io
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from pathlib import Path
from unittest import mock

import myflow.tui as T
from myflow.config import Settings


class FakeStore:
    def __init__(self):
        self.s = Settings()
        self.saved = 0

    def get(self):
        return self.s

    def save(self):
        self.saved += 1


# ---------------------------------------------------------------- утилиты
class KeyMappingTest(unittest.TestCase):
    """F5 (пересканировать микрофоны) не работал: скан-код \\x00 + '?' не
    распознавался, потому что стрелки (\\xe0) и функциональные клавиши (\\x00)
    обрабатывались одной таблицей."""

    def test_function_keys(self):
        expected = {b"?": "f5", b"B": "f8", b"A": "f7", b"C": "f9", b"D": "f10",
                    b"W": "f11", b"X": "f12"}
        for code, name in expected.items():
            self.assertEqual(T.map_extended_key(0x00, code), name, code)

    def test_arrows(self):
        self.assertEqual(T.map_extended_key(0xE0, b"H"), "up")
        self.assertEqual(T.map_extended_key(0xE0, b"P"), "down")
        self.assertEqual(T.map_extended_key(0xE0, b"K"), "left")
        self.assertEqual(T.map_extended_key(0xE0, b"M"), "right")

    def test_prefixes_do_not_collide(self):
        """'A'/'B' — это F7/F8 под префиксом \\x00, но не стрелки."""
        self.assertEqual(T.map_extended_key(0x00, b"A"), "f7")
        self.assertIsNone(T.map_extended_key(0xE0, b"A"))

    def test_unknown_code(self):
        self.assertIsNone(T.map_extended_key(0x00, b"\xff"))

    def test_vt_sequences(self):
        """Windows Terminal может прислать F5 как ESC[15~."""
        self.assertEqual(T.map_vt_sequence("15~"), "f5")
        self.assertEqual(T.map_vt_sequence("[15~"), "f5")
        self.assertEqual(T.map_vt_sequence("A"), "up")
        self.assertEqual(T.map_vt_sequence("3~"), "delete")
        self.assertIsNone(T.map_vt_sequence("999"))


class MicNoteTest(unittest.TestCase):
    """Строка микрофона в настройках показывает причину и host API."""

    def test_note_contains_status_and_api(self):
        store = FakeStore()
        mics = [{"name": "Микрофон (Ardor Gaming Vox)", "idx": 1, "alive": True,
                 "status": "жив (rms=0.0021)", "api": "MME"},
                {"name": "Микрофон (Vault )", "idx": 2, "alive": False,
                 "status": "нет сигнала (rms=0.000015)", "api": "MME"}]
        state = T.build_choices(store, mics, system_ok=True)
        mic = next(c for c in state.choices if c.key == "mic")
        labels = [c[0] for c in mic.options]
        self.assertIn("Ardor", " ".join(labels))
        self.assertEqual(mic.note(0), "жив")
        self.assertIn("MME", mic.note(1))
        self.assertIn("нет сигнала", mic.note(2))


class LevelMeterTest(unittest.TestCase):
    """Живой индикатор уровня: ловит регресс «колбэк падает, уровень всегда 0»."""

    class _FakeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            pass

        def close(self):
            pass

    def _meter(self, recorder=None, error=None):
        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
            stream = self._FakeStream(**kwargs)
            if recorder is not None:
                recorder["cb"] = kwargs["callback"]
                recorder["stream"] = stream
            return stream

        return T.LevelMeter(stream_factory=factory), calls

    def test_peak_updates_from_callback(self):
        rec = {}
        meter, calls = self._meter(rec)
        meter.switch(13)
        self.assertEqual(calls[0]["device"], 13)
        self.assertEqual(calls[0]["samplerate"], 16000)
        self.assertTrue(rec["stream"].started)
        self.assertEqual(meter.peak(), 0.0)
        rec["cb"](np.full((1024, 1), 0.05, dtype=np.float32), 1024, None, None)
        self.assertAlmostEqual(meter.peak(), 0.05, places=4)
        meter.stop()

    def test_switch_does_not_reopen_same_device(self):
        meter, calls = self._meter()
        meter.switch(5)
        meter.switch(5)
        self.assertEqual(len(calls), 1, "то же устройство — поток не переоткрываем")
        meter.switch(6)
        self.assertEqual(len(calls), 2)
        meter.stop()

    def test_device_error_is_reported(self):
        meter, _ = self._meter(error=RuntimeError(
            "Error opening InputStream: Device unavailable [PaErrorCode -9985]"))
        meter.switch(2)
        self.assertIn("Device unavailable", meter.error)
        self.assertEqual(meter.peak(), 0.0)

    def test_bar_marks_gate_always(self):
        bar = T.LevelMeter.bar(0.05, width=20, gate=0.01)
        self.assertEqual(len(bar), 20)
        self.assertIn("╎", bar, "метка порога должна быть видна и под столбиком")
        self.assertEqual(T.LevelMeter.bar(0.0, width=10).count("█"), 0)
        self.assertGreater(T.LevelMeter.bar(0.5, width=10).count("█"), 5)

    def test_bar_shows_unreachable_gate(self):
        """Уровень ниже порога: столбик не дотягивается до метки."""
        bar = T.LevelMeter.bar(0.002, width=20, gate=0.01)
        gate_pos = bar.index("╎")
        self.assertLess(bar.count("█"), gate_pos, "столбик должен быть левее метки")


class TextHelpersTest(unittest.TestCase):
    def test_strip_ansi_and_vlen(self):
        s = f"{T.C_OK}готово{T.RESET}"
        self.assertEqual(T.strip_ansi(s), "готово")
        self.assertEqual(T.vlen(s), 6)

    def test_norm_text_replaces_wide_glyphs(self):
        """Эмодзи из логов ломают выравнивание рамки — их заменяем на ASCII."""
        self.assertEqual(T.norm_text("● запись…"), "* запись...")
        self.assertEqual(T.norm_text("⚙ ✓ ✗"), "~ + !")
        self.assertNotIn("\t", T.norm_text("a\tb"))

    def test_pad_to_exact_width(self):
        self.assertEqual(T.vlen(T.pad_to("abc", 10)), 10)
        self.assertEqual(T.vlen(T.pad_to("abcdefghijkl", 5)), 5)
        self.assertEqual(T.vlen(T.pad_to(f"{T.C_OK}abc{T.RESET}", 8)), 8)

    def test_hex_to_ansi(self):
        self.assertEqual(T.hex_to_ansi("#f5c542"), T.C_WARN)
        self.assertEqual(T.hex_to_ansi("#46a758"), T.C_OK)
        self.assertEqual(T.hex_to_ansi("#e5484d"), T.C_ERR)
        self.assertTrue(T.hex_to_ansi("#123456").startswith("\x1b[38;5;"))
        self.assertEqual(T.hex_to_ansi("мусор"), T.C_TEXT)

    def test_status_labels(self):
        for st in ("ok", "external", "starting", "down", "error"):
            self.assertTrue(T.status_label(st))
            self.assertTrue(T.status_color(st).startswith("\x1b["))


# ---------------------------------------------------------------- рамки
class FrameTest(unittest.TestCase):
    def _assert_rect(self, lines, w, h):
        self.assertEqual(len(lines), h, "неверное число строк кадра")
        for ln in lines:
            self.assertEqual(T.vlen(ln), w, f"строка не по ширине: {T.strip_ansi(ln)!r}")

    def test_outer_frame_exact_size(self):
        for w, h in ((80, 24), (96, 26), (120, 40), (40, 8)):
            self._assert_rect(T.outer_frame("тест", ["строка"], w, h), w, h)

    def test_outer_frame_truncates_body(self):
        lines = T.outer_frame("t", [f"строка {i}" for i in range(100)], 80, 10)
        self._assert_rect(lines, 80, 10)

    def test_block_exact_width(self):
        for w in (20, 40, 96):
            for ln in T.block("заголовок", ["[x] пункт", "длинная строка " * 5], w):
                self.assertEqual(T.vlen(ln), w)

    def test_block_very_long_title(self):
        for ln in T.block("очень длинный заголовок блока", ["x"], 24):
            self.assertEqual(T.vlen(ln), 24)


# ---------------------------------------------------------------- настройки
class SettingsStateTest(unittest.TestCase):
    def _state(self):
        store = FakeStore()
        store.s.asr.input_device = 3
        mics = [{"name": "Микрофон (USB)", "idx": 1, "alive": True},
                {"name": "Микрофон (Realtek)", "idx": 3, "alive": False}]
        return store, T.build_choices(store, mics, True)

    def test_mic_options_and_preselect(self):
        store, st = self._state()
        self.assertEqual(st.choices[0].title, "Микрофон")
        self.assertEqual(st.choices[0].options[0][1], None)          # системный
        self.assertEqual(st.choices[0].options[2][1], 3)             # текущий
        self.assertEqual(st.choices[0].index, 2, "текущий микрофон должен быть выбран")
        self.assertEqual(st.choices[0].note(2), "нет сигнала")

    def test_navigation_wraps(self):
        store, st = self._state()
        st.focus = 0
        st.choices[0].index = 0
        st.move_item(-1)
        self.assertEqual(st.choices[0].index, len(st.choices[0].options) - 1,
                         "с первого пункта вверх — на последний")
        st.move_item(1)
        self.assertEqual(st.choices[0].index, 0, "и обратно по кругу")
        st.move_block(-1)
        self.assertEqual(st.focus, 2, "блоки должны листаться по кругу")
        st.move_block(1)
        self.assertEqual(st.focus, 0)

    def test_values_and_apply(self):
        store, st = self._state()
        st.choices[0].index = 1      # USB-микрофон
        st.choices[1].index = 1      # ptt
        st.choices[2].index = 1      # opencode
        values = st.values()
        self.assertEqual(values, {"mic": 1, "mode": "ptt", "insert": "opencode"})
        T.apply_choices(store, values)
        self.assertEqual(store.get().asr.input_device, 1)
        self.assertEqual(store.get().mode, "ptt")
        self.assertEqual(store.get().insert_method, "opencode")
        self.assertEqual(store.saved, 1)

    def test_no_mics_only_system(self):
        store = FakeStore()
        st = T.build_choices(store, [], False)
        self.assertEqual(len(st.choices[0].options), 1)
        self.assertEqual(st.choices[0].note(0), "нет сигнала")


# ---------------------------------------------------------------- супервизор
def _spec(tmp: Path, key: str = "asr") -> T.ServerSpec:
    exe = tmp / "server.exe"
    model = tmp / "model.bin"
    exe.write_bytes(b"x")
    model.write_bytes(b"x")
    return T.ServerSpec(key=key, title="Тест", port=18081, url="http://127.0.0.1:18081",
                        health_path="/health", exe=exe, args=[], model=model,
                        log=tmp / "s.log")


class SupervisorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_missing_files_reported(self):
        spec = _spec(self.dir)
        spec.exe = self.dir / "нет.exe"
        self.assertIn("нет файла", spec.missing() or "")

    def test_disabled_supervisor_reports_down(self):
        sup = T.Supervisor([_spec(self.dir)], enabled=False)
        state, note = sup.status(sup.specs[0])
        self.assertEqual(state, "down")
        self.assertIn("--no-servers", note)

    def test_external_instance_is_not_ours(self):
        spec = _spec(self.dir)
        sup = T.Supervisor([spec])
        with mock.patch.object(T.Supervisor, "health", return_value=True):
            state, note = sup.status(spec)
        self.assertEqual(state, "external")
        self.assertIn("раньше", note)

    def test_own_process_ok_and_error(self):
        spec = _spec(self.dir)
        sup = T.Supervisor([spec])
        proc = mock.Mock()
        proc.poll.return_value = None
        sup._procs[spec.key] = proc
        sup._own.add(spec.key)
        with mock.patch.object(T.Supervisor, "health", return_value=True):
            self.assertEqual(sup.status(spec)[0], "ok")
        with mock.patch.object(T.Supervisor, "health", return_value=False), \
             mock.patch.object(T.Supervisor, "port_open", return_value=True):
            proc.poll.return_value = 1          # процесс умер
            self.assertEqual(sup.status(spec)[0], "error")

    def test_start_skips_when_port_busy(self):
        spec = _spec(self.dir)
        sup = T.Supervisor([spec])
        with mock.patch.object(T.Supervisor, "port_open", return_value=True), \
             mock.patch("subprocess.Popen") as popen:
            self.assertTrue(sup.start(spec))
        popen.assert_not_called()
        self.assertNotIn(spec.key, sup._own)

    def test_start_reports_spawn_error(self):
        spec = _spec(self.dir)
        sup = T.Supervisor([spec])
        with mock.patch.object(T.Supervisor, "port_open", return_value=False), \
             mock.patch("subprocess.Popen", side_effect=OSError("boom")):
            self.assertFalse(sup.start(spec))
        self.assertEqual(sup.status(spec)[0], "error")
        self.assertIn("boom", sup.status(spec)[1])

    def test_start_spawns_hidden_and_owns_process(self):
        spec = _spec(self.dir)
        sup = T.Supervisor([spec])
        proc = mock.Mock()
        with mock.patch.object(T.Supervisor, "port_open", return_value=False), \
             mock.patch("subprocess.Popen", return_value=proc) as popen:
            self.assertTrue(sup.start(spec))
        kwargs = popen.call_args.kwargs
        self.assertEqual(kwargs["creationflags"], T.Supervisor.CREATE_NO_WINDOW,
                         "сервер не должен открывать своё окно")
        self.assertIn(spec.key, sup._own)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], str(spec.exe))
        self.assertIn(str(spec.model), argv)

    def test_stop_all_terminates_only_own(self):
        spec = _spec(self.dir)
        sup = T.Supervisor([spec])
        mine, theirs = mock.Mock(), mock.Mock()
        mine.poll.return_value = None
        sup._procs[spec.key] = mine
        sup._own.add(spec.key)
        sup.stop_all()
        mine.terminate.assert_called_once()
        theirs.terminate.assert_not_called()
        self.assertFalse(sup._own)

    def test_health_uses_requests(self):
        with mock.patch("requests.get") as get:
            get.return_value = mock.Mock(ok=True)
            self.assertTrue(T.Supervisor.health("http://x", "/health"))
        get.assert_called_once()


# ---------------------------------------------------------------- журнал и оверлей
class PaneSinkTest(unittest.TestCase):
    def test_lines_go_to_pane_not_console(self):
        st = T.TuiState()
        sink = T.PaneSink(st, log_file=None)
        sink.write("первая строка\nвторая ")
        sink.write("строка\n")
        sink.flush()
        self.assertEqual(len(st.log), 2)
        self.assertIn("первая строка", st.log[0])
        self.assertIn("вторая строка", st.log[1])

    def test_counters_from_log(self):
        st = T.TuiState()
        sink = T.PaneSink(st, log_file=None)
        sink.write("✓ вставлено\ne2e=1240 мс (asr=520, polish=236)\n")
        self.assertEqual(st.phrases, 1)
        self.assertEqual(st.last_e2e_ms, 1240)

    def test_ansi_and_glyphs_normalized(self):
        st = T.TuiState()
        sink = T.PaneSink(st, log_file=None)
        sink.write(f"{T.C_OK}\x1b[1m● запись\x1b[0m\n")
        self.assertEqual(st.log[0].split("  ", 1)[1], "* запись")

    def test_log_is_bounded(self):
        st = T.TuiState()
        sink = T.PaneSink(st, log_file=None)
        sink.write("\n".join(f"строка {i}" for i in range(600)) + "\n")
        self.assertLessEqual(len(st.log), 400)

    def test_writes_file(self):
        st = T.TuiState()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "main.log"
            sink = T.PaneSink(st, log_file=path)
            sink.write("в файл\n")
            sink.close()
            self.assertIn("в файл", path.read_text(encoding="utf-8"))


class TuiOverlayTest(unittest.TestCase):
    def test_show_sets_status_with_ansi_color(self):
        st = T.TuiState()
        ov = T.TuiOverlay(st)
        self.assertTrue(ov.is_running())
        ov.show("запись…", "#f5c542")
        text, color, _log, _srv = st.snapshot()
        self.assertEqual(text, "запись...", "широкие глифы заменяются на ASCII")
        self.assertEqual(color, T.C_WARN)
        ov.hide()
        self.assertEqual(st.snapshot()[0], "ожидание")

    def test_show_does_not_start_gfx_when_disabled(self):
        st = T.TuiState()
        gfx = mock.Mock()
        ov = T.TuiOverlay(st, gfx=None)
        ov.start()
        self.assertIsNone(ov.gfx)
        ov2 = T.TuiOverlay(st, gfx=gfx)
        ov2.start()
        gfx.start.assert_called_once()
        ov2.stop()
        gfx.stop.assert_called_once()

    def test_show_forwards_to_gfx(self):
        st = T.TuiState()
        gfx = mock.Mock()
        T.TuiOverlay(st, gfx=gfx).show("✓ вставлено", "#46a758")
        gfx.show.assert_called_once()


# ---------------------------------------------------------------- экраны
class ScreensTest(unittest.TestCase):
    def _settings_state(self):
        store = FakeStore()
        return store, T.build_choices(store, [{"name": "Микрофон (USB)", "idx": 1, "alive": True}], True)

    def test_settings_screen_content(self):
        store, st = self._settings_state()
        frame = T.render_settings(st, store, 96, 26)
        plain = "\n".join(T.strip_ansi(x) for x in frame)
        self.assertIn("Микрофон", plain)
        self.assertIn("Режим диктовки", plain)
        self.assertIn("Куда вставлять", plain)
        self.assertIn("[x]", plain)
        self.assertIn("жив", plain)
        for ln in frame:
            self.assertEqual(T.vlen(ln), 96)

    def test_dashboard_screen_content(self):
        st = T.TuiState()
        st.servers = {"asr": {"title": "ASR whisper.cpp", "state": "ok", "note": "RTF 0.12"},
                      "llm": {"title": "LLM", "state": "error", "note": "упал"}}
        st.add_log("✓ вставлено")
        st.set_status("запись…", T.C_WARN)
        frame = T.render_dashboard(st, 100, 30, "режим: авто")
        plain = "\n".join(T.strip_ansi(x) for x in frame)
        self.assertIn("ASR whisper.cpp", plain)
        self.assertIn("[работает]", plain)
        self.assertIn("ОШИБКА", plain)
        self.assertIn("запись…", plain)
        self.assertIn("Журнал", plain)
        self.assertIn("настройки", plain)
        self.assertIn("выход", plain)
        for ln in frame:
            self.assertEqual(T.vlen(ln), 100)

    def test_dashboard_small_console(self):
        st = T.TuiState()
        frame = T.render_dashboard(st, 60, 18, "cfg")
        self.assertEqual(len(frame), 18)
        for ln in frame:
            self.assertEqual(T.vlen(ln), 60)

    def test_error_screen(self):
        frame = T.render_error("whisper-server не отвечает", 80, 12)
        plain = "\n".join(T.strip_ansi(x) for x in frame)
        self.assertIn("whisper-server не отвечает", plain)
        self.assertIn("Enter", plain)


class FitWidthTest(unittest.TestCase):
    """Строка шире окна заставляет терминал переносить её — кадр «едет»."""

    def test_truncates_visible_length(self):
        line = "абвгдеёжзийклмнопрст"
        cut = T.fit_width(line, 5)
        self.assertLessEqual(T.vlen(cut), 5 + len(T.RESET))
        self.assertEqual(T.vlen(T.strip_ansi(cut)), 5)

    def test_keeps_ansi_and_resets(self):
        line = f"{T.C_OK}зелёный{T.RESET} текст"
        cut = T.fit_width(line, 7)
        self.assertTrue(cut.startswith(T.C_OK), "ANSI-код сохраняется")
        self.assertTrue(cut.endswith(T.RESET), "в конце сброс цвета")
        self.assertEqual(T.vlen(T.strip_ansi(cut)), 7)

    def test_short_line_untouched(self):
        line = f"{T.C_DIM}ок{T.RESET}"
        self.assertEqual(T.fit_width(line, 20), line)

    def test_zero_width(self):
        self.assertEqual(T.fit_width("текст", 0), "текст")


class ButtonBarTest(unittest.TestCase):
    def test_contains_keys_and_labels(self):
        bar = "\n".join(T.button_bar([("F5", "пересканировать"),
                                        ("Enter", "запустить")], 80))
        plain = T.strip_ansi(bar)
        self.assertIn("F5", plain)
        self.assertIn("пересканировать", plain)
        self.assertIn("Enter", plain)
        self.assertIn("запустить", plain)

    def test_primary_is_highlighted(self):
        bar = "\n".join(T.button_bar([("Enter", "запустить")], 80, primary="Enter"))
        self.assertIn(T.C_INV, bar, "главная кнопка выделяется инверсией")

    def test_fits_width(self):
        lines = T.button_bar([("PgUp/PgDn", "журнал"), ("End", "в конец"),
                              ("r", "настройки"), ("q", "выход")], 40)
        for line in lines:
            self.assertLessEqual(T.vlen(T.strip_ansi(line)), 40)
        joined = T.strip_ansi("\n".join(lines))
        for label in ("журнал", "в конец", "настройки", "выход"):
            self.assertIn(label, joined, "подписи не выбрасываем даже на узком окне")


class JournalScrollTest(unittest.TestCase):
    """Прокрутка журнала: можно вернуться и прочитать прошлое."""

    def test_scroll_by_and_clamp(self):
        st = T.TuiState()
        st.scroll_by(5, page=10, total=100)
        self.assertEqual(st.scroll, 5)
        st.scroll_by(-50, page=10, total=100)
        self.assertEqual(st.scroll, 0, "ниже нуля не уходим")
        st.scroll_to(999, page=10, total=100)
        self.assertEqual(st.scroll, 90, "выше конца не уходим")

    def test_render_shows_older_lines_and_indicator(self):
        st = T.TuiState()
        for i in range(40):
            st.add_log(f"строка {i}")
        st.scroll_by(10, page=8, total=40)
        frame = T.render_dashboard(st, 100, 30, "cfg")
        plain = "\n".join(T.strip_ansi(x) for x in frame)
        self.assertIn("прокрутка", plain, "видно, что журнал прокручен")
        self.assertNotIn("строка 39", plain, "самая свежая строка ушла из вида")
        for ln in frame:
            self.assertEqual(T.vlen(ln), 100)

    def test_live_mode_shows_tail(self):
        st = T.TuiState()
        for i in range(40):
            st.add_log(f"строка {i}")
        frame = T.render_dashboard(st, 100, 30, "cfg")
        plain = "\n".join(T.strip_ansi(x) for x in frame)
        self.assertIn("строка 39", plain)
        self.assertNotIn("прокрутка", plain)


class ConsoleDrawTest(unittest.TestCase):
    """draw() обязан обрезать кадр по ВИДИМОМУ окну (а не по буферу)."""

    class _Out:
        def __init__(self):
            self.data = ""

        def write(self, s):
            self.data += s

        def flush(self):
            pass

    def _console(self, size=(20, 3)):
        console = T.Console.__new__(T.Console)      # без реальной консоли
        console.out = self._Out()
        console.vt = True
        console._last_frame = []
        console.size = lambda: size
        return console

    def test_draw_clips_width_and_height(self):
        console = self._console(size=(20, 3))
        console.draw(["x" * 50, "y" * 50, "z" * 50, "лишняя", "ещё"])
        data = console.out.data
        self.assertNotIn("x" * 50, data, "длинная строка обрезана")
        self.assertNotIn(f"{T.CSI}4;1H", data, "строки сверх высоты не рисуем")

    def test_draw_writes_nothing_when_frame_unchanged(self):
        """Панель перерисовывалась 20 раз в секунду целиком — это лишние 170 КБ/с."""
        console = self._console(size=(40, 4))
        frame = ["строка один", "строка два", "строка три"]
        self.assertTrue(console.draw(frame), "первый кадр рисуем")
        first = len(console.out.data)
        self.assertFalse(console.draw(list(frame)), "идентичный кадр — не пишем")
        self.assertEqual(len(console.out.data), first, "в консоль не ушло ни байта")

    def test_draw_repaints_only_changed_lines(self):
        console = self._console(size=(40, 4))
        console.draw(["aaa", "bbb", "ccc"])
        console.out.data = ""
        console.draw(["aaa", "ZZZ", "ccc"])
        data = console.out.data
        self.assertIn("ZZZ", data)
        self.assertNotIn("aaa", data, "неизменившуюся строку не перерисовываем")
        self.assertNotIn("ccc", data)

    def test_draw_clears_tail_when_frame_shrinks(self):
        console = self._console(size=(40, 6))
        console.draw(["a", "b", "c", "d"])
        console.out.data = ""
        console.draw(["a", "b"])
        self.assertIn(f"{T.CSI}3;1H", console.out.data, "остаток старого кадра стираем")

    def test_draw_plain_when_no_vt(self):
        console = T.Console.__new__(T.Console)
        console.out = self._Out()
        console.vt = False
        console.draw([f"{T.C_OK}привет{T.RESET}"])
        self.assertIn("привет", console.out.data)
        self.assertNotIn(T.C_OK, console.out.data, "без VT цвета не пишем")


class PhaseTest(unittest.TestCase):
    """Фазы TUI без реальной консоли: рисуем в никуда, проверяем решения."""

    def _app(self, **kw):
        with mock.patch.object(T.Console, "draw"), mock.patch.object(T.Console, "clear"), \
             mock.patch.object(T.Console, "cursor"), mock.patch.object(T.Console, "reset"):
            app = T.TuiApp(**kw)
        return app

    def test_boot_phase_with_servers_disabled_starts_nothing(self):
        app = self._app(no_servers=True)
        app._draw_boot = lambda *a, **k: None
        app._boot_phase()                       # не должно падать и что-то запускать
        self.assertEqual(app.supervisor._own, set())

    def test_boot_phase_skips_healthy_server(self):
        app = self._app(no_servers=False)
        app._draw_boot = lambda *a, **k: None
        with mock.patch.object(T.Supervisor, "health", return_value=True), \
             mock.patch.object(T.Supervisor, "start") as start:
            app._boot_phase()
        start.assert_not_called()

    def test_restart_down_servers_starts_only_downed(self):
        app = self._app(no_servers=False)
        with mock.patch.object(T.Supervisor, "status", return_value=("down", "нет")), \
             mock.patch.object(T.Supervisor, "start", return_value=True) as start, \
             mock.patch.object(T.Supervisor, "wait_ready", return_value=True), \
             mock.patch.object(T.Supervisor, "health_snapshot", return_value={}):
            app._restart_down_servers()
        self.assertEqual(start.call_count, len(app.supervisor.specs))

    def test_shutdown_stops_servers_and_restores_stdout(self):
        app = self._app(no_servers=False)
        prev = sys.stdout
        app._install_sink()
        self.assertIsNot(sys.stdout, prev)
        with mock.patch.object(T.Supervisor, "stop_all") as stop_all:
            app._shutdown()
        stop_all.assert_called_once()
        self.assertIs(sys.stdout, prev, "stdout должен вернуться как был")

    def test_config_line_contains_settings(self):
        app = self._app(no_servers=True)
        line = T.strip_ansi(app._config_line())
        self.assertIn("режим:", line)
        self.assertIn("микрофон:", line)
        self.assertIn("вставка:", line)

    def test_toggle_gfx_off_when_absent(self):
        app = self._app(no_servers=True)
        app._overlay = T.TuiOverlay(app.state, None)
        app._toggle_gfx()                       # включаем плашку
        self.assertIsNotNone(app._gfx)
        app._toggle_gfx()                       # выключаем
        self.assertIsNone(app._gfx)


if __name__ == "__main__":
    unittest.main()


class MouseWheelTest(unittest.TestCase):
    """Колесо мыши: листает только журнал, окно не прокручивается."""

    class _Out:
        def __init__(self):
            self.data = ""

        def write(self, s):
            self.data += s

        def flush(self):
            pass

    @staticmethod
    def _snap():
        return {"modules": [
            {"key": "asr", "label": "ASR", "value": "GigaAM", "on": True,
             "ok": True, "note": "", "hint": "h"},
        ], "values": [], "counters": {"phrases": 1, "pending": 0, "queue": 0}}

    def test_parses_sgr_wheel_events(self):
        self.assertEqual(T.map_mouse_sequence("[<64;120;34M"), "wheelup:120,34")
        self.assertEqual(T.map_mouse_sequence("[<65;10;5M"), "wheeldown:10,5")

    def test_parses_clicks_and_ignores_other_sequences(self):
        self.assertEqual(T.map_mouse_sequence("[<0;15;20M"), "click:15,20")
        self.assertIsNone(T.map_mouse_sequence("[15~"))
        self.assertIsNone(T.map_mouse_sequence(""))

    def test_layout_records_journal_rows(self):
        frame = T.render_panel(self._snap(), 100, 30, "cfg", ["строка 1", "строка 2"],
                               0, 0, level=(0.1, ""))
        top, bottom = T.LAST_LAYOUT["journal"]
        self.assertGreaterEqual(top, 1)
        self.assertGreater(bottom, top, "журнал должен занимать несколько строк")
        self.assertLessEqual(bottom, len(frame), "границы внутри кадра")

    def test_wheel_inside_journal_scrolls(self):
        from unittest import mock
        tui = T.TuiApp(no_servers=True)
        frame = T.render_panel(self._snap(), 100, 30, "cfg", ["a", "b", "c"], 0, 0)
        top, _bottom = T.LAST_LAYOUT["journal"]
        tui.state.log = [f"строка {i}" for i in range(50)]
        with mock.patch.object(tui, "_scroll_journal") as scroll:
            handled = tui._on_wheel(f"wheelup:10,{top + 1}", 30)
        self.assertTrue(handled)
        scroll.assert_called_once()

    def test_wheel_outside_journal_is_ignored(self):
        """Вне журнала колесо не должно ничего прокручивать."""
        from unittest import mock
        tui = T.TuiApp(no_servers=True)
        T.render_panel(self._snap(), 100, 30, "cfg", ["a"], 0, 0)
        with mock.patch.object(tui, "_scroll_journal") as scroll:
            handled = tui._on_wheel("wheelup:10,1", 30)
        self.assertFalse(handled)
        scroll.assert_not_called()

    def test_mouse_mode_toggles_escape_sequences(self):
        console = T.Console.__new__(T.Console)
        console.out = self._Out()
        console.vt = True
        console.mouse = False
        console.enable_mouse()
        self.assertIn(T.CSI + "?1000h", console.out.data)
        self.assertIn(T.CSI + "?1006h", console.out.data)
        self.assertTrue(console.mouse)
        console.out.data = ""
        console.disable_mouse()
        self.assertIn(T.CSI + "?1006l", console.out.data)
        self.assertIn(T.CSI + "?1000l", console.out.data)
        self.assertFalse(console.mouse)

    def test_lock_size_keeps_size_when_disabled(self):
        """lock_size(0,0) ничего не ломает и возвращает актуальный размер."""
        console = T.Console.__new__(T.Console)
        console.vt = False
        console.raw_size = lambda: (120, 35)
        w, h = console.lock_size(0, 0)
        self.assertEqual((w, h), (120, 35))


class WindowSizeTest(unittest.TestCase):
    """Размер окна: применяем сохранённый, а ручное растягивание ЗАПОМИНАЕМ."""

    def _tui(self, cols=120, rows=30):
        import tempfile
        from pathlib import Path as _P
        from myflow.config import SettingsStore
        tmp = tempfile.TemporaryDirectory()
        store = SettingsStore(_P(tmp.name) / "s.json")
        store.get().tui.cols, store.get().tui.rows = cols, rows
        store.save()
        tui = T.TuiApp(no_servers=True)
        tui.store = store
        tui._tmp = tmp
        return tui, store

    def test_adopt_manual_resize_is_remembered(self):
        tui, store = self._tui()
        tui._locked_size = (120, 30)
        tui._pending_size = (140, 42)
        tui._adopt_size((140, 42))
        self.assertEqual(tui._locked_size, (140, 42))
        cfg = store.get().tui
        self.assertEqual((cfg.cols, cfg.rows), (140, 42),
                         "растянутое окно должно сохраниться в настройках")
        self.assertIsNone(tui._pending_size)
        tui._tmp.cleanup()

    def test_adopt_ignores_same_size(self):
        tui, store = self._tui()
        tui._locked_size = (120, 30)
        tui._adopt_size((120, 30))
        self.assertEqual((store.get().tui.cols, store.get().tui.rows), (120, 30))
        tui._tmp.cleanup()

    def test_apply_window_size_updates_lock(self):
        tui, store = self._tui()
        tui.console = self._FakeConsole()
        tui.apply_window_size(132, 40)
        self.assertEqual(tui._locked_size, (132, 40))
        self.assertEqual(tui.console.locked, (132, 40))
        tui._tmp.cleanup()

    class _FakeConsole:
        def __init__(self):
            self.locked = None
            self.cleared = 0

        def lock_size(self, cols=0, rows=0):
            self.locked = (cols, rows)
            return (cols, rows)

        def clear(self):
            self.cleared += 1

    def test_settings_row_presets_cycle(self):
        """Строка «Размер окна» перебирает пресеты и зовёт хук применения."""
        import tempfile
        from pathlib import Path as _P
        from myflow.config import SettingsStore
        from myflow.db import Database
        import myflow.main as M
        from unittest import mock

        tmp = tempfile.TemporaryDirectory()
        store = SettingsStore(_P(tmp.name) / "s.json")
        store.get().tui.cols, store.get().tui.rows = 100, 28
        app = M.MyFlowApp(settings=store, db=Database(_P(tmp.name) / "d.sqlite"))
        app._mic_cache = ([], 0.0)
        app.overlay = mock.Mock()
        calls = []
        app.set_console_resize(lambda c, r: calls.append((c, r)))
        try:
            ok, message = app.control_apply("module", "size")
            self.assertTrue(ok)
            self.assertEqual(calls[-1], (120, 34), "следующий пресет после 100x28")
            self.assertIn("120x34", message)
            row = next(m for m in app.control_snapshot()["modules"] if m["key"] == "size")
            self.assertEqual(row["value"], "120x34")
        finally:
            app.db.close()
            app._stopping.set()
            tmp.cleanup()
