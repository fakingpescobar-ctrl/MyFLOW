"""MyFLOW — CLI: сниппеты, словарь, статистика надёжности, retry, диагностика.

Запуск:
    .venv\\Scripts\\python -m myflow.cli snippets list
    .venv\\Scripts\\python -m myflow.cli snippets add "мой email" "user@example.com"
    .venv\\Scripts\\python -m myflow.cli snippets rm 3
    .venv\\Scripts\\python -m myflow.cli dict list
    .venv\\Scripts\\python -m myflow.cli dict add "сухой лаг" "Сухой Лог" [--app code.exe]
    .venv\\Scripts\\python -m myflow.cli stats
    .venv\\Scripts\\python -m myflow.cli retry [--limit 5]
    .venv\\Scripts\\python -m myflow.cli ax       # что видит UI Automation в поле фокуса
    .venv\\Scripts\\python -m myflow.cli doctor   # проверка окружения и серверов
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _out(text: str = "") -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(text)


# ---------------------------------------------------------------- сниппеты
def cmd_snippets(args) -> int:
    from myflow.db import Database
    db = Database()
    try:
        if args.action == "list":
            rows = db.list_snippets()
            if not rows:
                _out("Сниппетов нет. Добавить: myflow.cli snippets add \"фраза\" \"текст\"")
                return 0
            _out(f"Сниппетов: {len(rows)}")
            for r in rows:
                repl = r["replacement"].replace("\n", " ")
                if len(repl) > 60:
                    repl = repl[:57] + "..."
                _out(f"  #{r['id']:<3} [{r['source']:<7}] {r['phrase']!r} -> {repl!r} "
                     f"(использован {r['hit_count']} раз)")
            return 0
        if args.action == "add":
            db.add_snippet(args.phrase, args.replacement)
            _out(f"Сниппет добавлен: {args.phrase!r} -> ({len(args.replacement)} симв.)")
            return 0
        if args.action == "rm":
            removed = db.delete_snippet(args.ident)
            _out(f"Удалено сниппетов: {removed}" if removed else "Не найдено")
            return 0 if removed else 1
    finally:
        db.close()
    return 2


# ---------------------------------------------------------------- словарь
def cmd_dict(args) -> int:
    from myflow.db import Database
    db = Database()
    try:
        if args.action == "list":
            rows = db.get_dictionary(args.app)
            _out(f"Замен в словаре: {len(rows)}"
                 + (f" (окно: {args.app})" if args.app else " (глобальные)"))
            for r in rows:
                _out(f"  {r['source_word']} -> {r['target_word']}")
            return 0
        if args.action == "add":
            db.add_word(args.source, args.target, source="manual", app_mask=args.app)
            _out(f"Добавлено: {args.source} -> {args.target}"
                 + (f" (только для {args.app})" if args.app else " (глобально)"))
            return 0
        if args.action == "rm":
            with db._lock, db._conn:  # noqa: SLF001 — простая административная операция
                cur = db._conn.execute(
                    "DELETE FROM dictionary WHERE source_word=? AND app_mask=?",
                    (args.source.strip().lower(), (args.app or "").strip().lower()))
            _out(f"Удалено: {cur.rowcount}")
            return 0 if cur.rowcount else 1
    finally:
        db.close()
    return 2


# ---------------------------------------------------------------- статистика
def cmd_stats(args) -> int:
    from myflow.db import Database
    from myflow.audio_store import AudioStore
    from myflow.config import DATA_DIR
    db = Database()
    try:
        st = db.stats()
    finally:
        db.close()
    _out("== Надёжность и качество MyFLOW ==")
    _out(f"  диктовок          : {st['dictations']}")
    _out(f"  ошибок            : {st['errors']} ({st['error_rate'] * 100:.2f}%)")
    _out(f"  задержка e2e      : p50={st['p50_latency_ms']} мс · "
         f"p90={st['p90_latency_ms']} мс · p99={st['p99_latency_ms']} мс")
    _out(f"  среднее divergence: {st['avg_divergence']} (0 = LLM не меняла текст)")
    _out(f"  словарь/сниппеты  : {st['dictionary_words']} / {st['snippets']}")
    if st["by_status"]:
        _out("  статусы           : " + ", ".join(f"{k}={v}" for k, v in st["by_status"].items()))
    audio = AudioStore(DATA_DIR / "audio").stats()
    _out(f"  аудио для retry   : {audio['files']} файлов, "
         f"{audio['bytes'] / 1024 / 1024:.1f} МБ (держим {audio['keep']})")
    return 0


# ---------------------------------------------------------------- retry
def cmd_retry(args) -> int:
    from myflow.main import MyFlowApp, _health_check, _health_check_llm
    app = MyFlowApp()
    try:
        if not _health_check(app.whisper):
            _out(f"whisper-server недоступен ({app.whisper.cfg.url}) — сначала подними серверы")
            return 1
        if app.polish.cfg.enabled and not _health_check_llm(app.polish):
            app.polish.cfg.enabled = False
            _out("llama-server недоступен — перераспознаю без полировки")
        done = app.retry_pending(limit=args.limit)
        _out(f"Перераспознано фраз: {done}")
        return 0 if done else 1
    finally:
        try:
            app.ax.stop()
        except Exception:  # noqa: BLE001
            pass
        app.db.close()


# ---------------------------------------------------------------- AX-диагностика
def cmd_ax(args) -> int:
    from myflow.axtext import AxReader
    reader = AxReader()
    if not reader.start():
        _out(f"UI Automation недоступен: {reader.last_error}")
        return 1
    try:
        sample = reader.read()
        if sample is None:
            _out("Нет ответа от AX-хелпера")
            return 1
        _out(f"источник={sample.source} pid={sample.pid} окно={sample.window!r} "
             f"надёжный={sample.reliable}")
        _out(f"текст поля: {sample.text[:400]!r}")
        return 0
    finally:
        reader.stop()


# ---------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    """Проверка окружения: модели, серверы, БД, AX, аудио-хранилище."""
    import importlib
    from myflow.config import SettingsStore, DATA_DIR
    from myflow.tui import default_specs, Supervisor

    ok = True
    _out("== MyFLOW doctor ==")
    _out(f"  python            : {sys.version.split()[0]} ({sys.executable})")

    for mod in ("numpy", "sounddevice", "pyperclip", "requests", "onnxruntime", "pynput"):
        try:
            importlib.import_module(mod)
            _out(f"  пакет {mod:<13}: ok")
        except Exception as exc:  # noqa: BLE001
            ok = False
            _out(f"  пакет {mod:<13}: НЕТ ({exc})")

    try:
        from myflow.vad import SILERO_PATH
        exists = Path(SILERO_PATH).exists()
        ok &= exists
        _out(f"  Silero VAD        : {'ok' if exists else 'НЕТ'} ({SILERO_PATH})")
    except Exception as exc:  # noqa: BLE001
        ok = False
        _out(f"  Silero VAD        : ошибка ({exc})")

    try:
        settings = SettingsStore().get()
        _out(f"  settings.json     : режим={settings.mode}, вставка={settings.insert_method}, "
             f"микрофон={settings.asr.input_device}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        _out(f"  settings.json     : ОШИБКА ({exc})")

    from myflow.asr import make_asr_client, whisper_needed
    try:
        asr_cfg = SettingsStore().get().asr
        client = make_asr_client(asr_cfg)
        name = type(client).__name__
        healthy = client.health()
        _out(f"  ASR               : {name} "
             f"(backend={asr_cfg.backend}) — {'готов' if healthy else 'НЕ ГОТОВ'}")
        if not healthy:
            ok = False
        extra = getattr(client, "load_error", "")
        if extra:
            _out(f"    {extra}")
        if not whisper_needed(asr_cfg):
            _out("    whisper-сервер не нужен: ASR работает на локальной ONNX-модели (CPU)")
    except Exception as exc:  # noqa: BLE001
        ok = False
        _out(f"  ASR               : ОШИБКА ({exc})")

    sup = Supervisor(default_specs())
    for key, info in sup.health_snapshot().items():
        mark = "ok" if info["state"] in ("ok", "external") else info["state"].upper()
        if info["state"] not in ("ok", "external"):
            ok = False
        _out(f"  сервер {key:<10}: [{mark}] {info['note']}")

    from myflow.voiceprint import VoiceGate
    try:
        vcfg = SettingsStore().get().voiceprint
        gate = VoiceGate(enabled=vcfg.enabled, threshold=vcfg.threshold,
                         model_path=Path(vcfg.model_path) if vcfg.model_path else None,
                         profile_path=Path(vcfg.profile_path) if vcfg.profile_path else None)
        if not vcfg.enabled:
            _out("  голос             : проверка выключена")
        elif not gate.embedder.model_path.exists():
            _out(f"  голос             : нет модели ({gate.embedder.model_path})")
        elif not gate.profile.exists:
            _out("  голос             : профиль пуст -> myflow.cli voice enroll …")
        else:
            _out(f"  голос             : профиль {gate.profile.count} фраз, "
                 f"порог {vcfg.threshold}, модель {gate.embedder.model_path.name}")
    except Exception as exc:  # noqa: BLE001
        _out(f"  голос             : ОШИБКА ({exc})")

    from myflow.db import Database
    from myflow.audio_store import AudioStore
    db = Database()
    try:
        st = db.stats()
        _out(f"  база              : {st['dictations']} фраз, ошибок {st['errors']}, "
             f"сниппетов {st['snippets']}, словарь {st['dictionary_words']}")
    finally:
        db.close()
    audio = AudioStore(DATA_DIR / "audio").stats()
    _out(f"  аудио-хранилище   : {audio['files']} файлов, {audio['bytes'] / 1024 / 1024:.1f} МБ")

    from myflow.axtext import AxReader
    reader = AxReader()
    ax_ok = reader.start()
    _out(f"  UI Automation     : {'ok' if ax_ok else 'недоступен (' + reader.last_error + ')'}")
    reader.stop()
    if not ax_ok:
        _out("    (правки будут ловиться через буфер обмена: выдели правку и Ctrl+C)")

    _out("")
    _out("ИТОГ: всё в порядке" if ok else "ИТОГ: есть проблемы (см. выше)")
    return 0 if ok else 1


# ---------------------------------------------------------------- голос
def _load_wav_mono16k(path: str):
    """Читаем wav в int16 mono 16 кГц (что нужно и ASR, и модели голоса)."""
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1:
            raise SystemExit(f"{path}: нужен wav 16 кГц моно "
                             f"(получено {w.getframerate()} Гц, {w.getnchannels()} канал.)")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()


def cmd_voice(args) -> int:
    from myflow.config import SettingsStore
    from myflow.voiceprint import VoiceGate

    cfg = SettingsStore().get()
    vp = cfg.voiceprint
    gate = VoiceGate(
        enabled=vp.enabled, threshold=vp.threshold, min_seconds=vp.min_seconds,
        auto_enroll=vp.auto_enroll, enroll_min_score=vp.enroll_min_score,
        model_path=Path(vp.model_path) if vp.model_path else None,
        profile_path=Path(vp.profile_path) if vp.profile_path else None,
    )

    if args.action == "status":
        _out("== Профиль голоса ==")
        _out(f"  проверка           : {'включена' if vp.enabled else 'выключена'}")
        _out(f"  порог (cosine)     : {vp.threshold}")
        _out(f"  минимальная фраза  : {vp.min_seconds} с")
        _out(f"  авто-пополнение    : {'да' if vp.auto_enroll else 'нет'}"
             f" (берём фразы со score >= {vp.enroll_min_score})")
        _out(f"  модель             : {gate.embedder.model_path} "
             f"({'есть' if gate.embedder.model_path.exists() else 'НЕТ'})")
        _out(f"  профиль            : {gate.profile.path} "
             f"({'есть' if gate.profile.exists else 'НЕТ'})")
        if gate.profile.exists:
            _out(f"  фраз в профиле     : {gate.profile.count}")
        if not gate.embedder.deps_available():
            _out("  зависимости        : НЕТ (pip install onnxruntime kaldi_native_fbank)")
        elif not gate.profile.exists:
            _out("\n  Профиль пуст — проверка голоса не работает.")
            _out("  Записать: myflow.cli voice enroll data\\validation\\val_dict_1.wav …")
        return 0

    if args.action == "reset":
        gate.profile.reset()
        _out(f"Профиль голоса удалён ({gate.profile.path})")
        return 0

    if args.action == "enroll":
        if not gate.embedder.deps_available():
            _out("Нет зависимостей: pip install onnxruntime kaldi_native_fbank")
            return 1
        if not gate.embedder.model_path.exists():
            _out(f"Нет модели: {gate.embedder.model_path}")
            _out("Скачать: см. README, раздел «Проверка голоса»")
            return 1
        files = list(args.files)
        if args.from_history:
            from myflow.db import Database
            db = Database()
            try:
                rows = db._conn.execute(  # noqa: SLF001
                    """SELECT audio_path FROM history
                       WHERE audio_path IS NOT NULL AND audio_path <> ''
                         AND status IN ('ok','fast_path','polished')
                       ORDER BY id DESC LIMIT ?""", (args.from_history,)).fetchall()
            finally:
                db.close()
            found = [r[0] for r in rows if Path(r[0]).exists()]
            _out(f"Из истории взято файлов: {len(found)} (запрошено {args.from_history})")
            files.extend(found)
        if not files:
            _out("Нечего enroll-ить: укажи wav-файлы или --from-history N")
            return 1
        added = skipped = 0
        for path in files:
            pcm = _load_wav_mono16k(str(path))
            ok, score = gate.enroll(pcm)
            tag = "добавлен" if ok else "пропущен"
            score_text = "нет профиля" if score is None else f"cosine={score:+.3f}"
            _out(f"  {Path(path).name:<28} {tag:<9} {score_text}")
            added += int(ok)
            skipped += int(not ok)
        _out(f"\nДобавлено фраз: {added}, пропущено: {skipped}. "
             f"Всего в профиле: {gate.profile.count}")
        if added == 0 and gate.profile.exists:
            _out("Ни одна фраза не прошла порог похожести — это не ваш голос?")
        return 0

    if args.action == "test":
        # Проверить конкретный файл: наш голос или чужой?
        if not gate.profile.exists:
            _out("Профиль пуст — сначала voice enroll")
            return 1
        for path in args.files:
            pcm = _load_wav_mono16k(str(path))
            ok, score, reason = gate.check(pcm)
            verdict = "СВОЙ" if ok else "ЧУЖОЙ"
            score_text = "—" if score is None else f"{score:+.3f}"
            _out(f"  {Path(path).name:<28} {verdict:<6} cosine={score_text}  ({reason})")
        return 0
    return 1


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="myflow.cli", description="MyFLOW: служебные команды")
    sub = ap.add_subparsers(dest="group", required=True)

    sn = sub.add_parser("snippets", help="голосовые сниппеты (фраза -> готовый текст)")
    sn_sub = sn.add_subparsers(dest="action", required=True)
    sn_sub.add_parser("list", help="показать сниппеты")
    sn_add = sn_sub.add_parser("add", help="добавить сниппет")
    sn_add.add_argument("phrase", help="что произносим, например \"мой email\"")
    sn_add.add_argument("replacement", help="что вставляем")
    sn_rm = sn_sub.add_parser("rm", help="удалить сниппет")
    sn_rm.add_argument("ident", help="id или фраза")
    sn.set_defaults(func=cmd_snippets)

    d = sub.add_parser("dict", help="словарь замен")
    d_sub = d.add_subparsers(dest="action", required=True)
    d_list = d_sub.add_parser("list")
    d_list.add_argument("--app", default=None, help="маска окна/процесса")
    d_add = d_sub.add_parser("add")
    d_add.add_argument("source")
    d_add.add_argument("target")
    d_add.add_argument("--app", default=None, help="только для этого окна/процесса")
    d_rm = d_sub.add_parser("rm")
    d_rm.add_argument("source")
    d_rm.add_argument("--app", default=None)
    d.set_defaults(func=cmd_dict)

    st = sub.add_parser("stats", help="метрики надёжности и качества")
    st.set_defaults(func=cmd_stats)

    rt = sub.add_parser("retry", help="перераспознать упавшие фразы из сохранённого аудио")
    rt.add_argument("--limit", type=int, default=5)
    rt.set_defaults(func=cmd_retry)

    ax = sub.add_parser("ax", help="что видит UI Automation в поле фокуса")
    ax.set_defaults(func=cmd_ax)

    doc = sub.add_parser("doctor", help="проверка окружения")
    doc.set_defaults(func=cmd_doctor)

    vp = sub.add_parser("voice", help="профиль голоса (speaker verification)")
    vp_sub = vp.add_subparsers(dest="action", required=True)
    vp_sub.add_parser("status", help="состояние профиля и модели")
    vp_sub.add_parser("reset", help="удалить профиль голоса")
    vp_en = vp_sub.add_parser("enroll", help="добавить фразы в профиль")
    vp_en.add_argument("files", nargs="*", help="wav 16 кГц моно (твой голос)")
    vp_en.add_argument("--from-history", type=int, default=0, metavar="N",
                       help="взять до N своих записей из истории (сохранённое аудио)")
    vp_te = vp_sub.add_parser("test", help="проверить файлы: свой голос или чужой")
    vp_te.add_argument("files", nargs="+")
    vp.set_defaults(func=cmd_voice)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
