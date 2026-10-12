"""Красивый интерфейс MyFLOW: отдельное окно с ПОЛНЫМ управлением.

Главное решение. TUI управляет приложением через пульт: `control_snapshot()` отдаёт
строки (что можно включать, какие значения), `control_apply(action, value)` применяет
изменение на лету. Второй такой же пульт писать нельзя: он разойдётся с первым и
получится две правды.

Поэтому окно подключается к ТОМУ ЖЕ пульту:
  * приложение запускает сервер интерфейса и передаёт ему свои функции управления
    (см. main.py: _start_webui);
  * `GET /api/snapshot` отдаёт то же, что видит TUI;
  * `POST /api/action` вызывает `control_apply` — то есть кнопки в окне делают ровно
    то же, что клавиши в панели;
  * если сервер запущен отдельно (приложение не подключено), окно работает в режиме
    витрины: показывает историю и счётчики из базы, но кнопки честно говорят, что
    управление недоступно.

Запуск отдельно (витрина):
    python -m myflow.webui --port 8790
Запуск из приложения (управление):
    в main.py сервер поднимается с control_snapshot/control_apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "myflow.sqlite"
SETTINGS = ROOT / "data" / "settings.json"
RETRAIN_THRESHOLD = 100


def read_settings(path: Path | None = None) -> dict:
    file = Path(path) if path is not None else SETTINGS
    if not file.exists():
        return {}
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _model_name(pattern: str) -> str:
    """Имя рабочей модели из папки models (для честной подписи в интерфейсе)."""
    folder = ROOT / "models"
    if not folder.exists():
        return ""
    found = sorted(folder.glob(pattern))
    return found[0].name if found else ""


def build_state(db: Path | None = None, settings: Path | None = None,
                limit: int = 12) -> dict:
    """Состояние из базы (режим витрины): только чтение, ничего не меняет."""
    path = Path(db) if db is not None else DB
    cfg = read_settings(settings)
    state = {
        "ready": path.exists(),
        "asr": "GigaAM v3 e2e (CPU)",
        "postproc_model": _model_name("myflow-postproc-*.gguf"),
        "oracle_model": (cfg.get("polish") or {}).get("model", "qwen3-4b-instruct-2507"),
        "polish": bool((cfg.get("polish") or {}).get("enabled", True)),
        "postproc": bool((cfg.get("polish") or {}).get("postproc", True)),
        "insert_mode": (cfg.get("insert_method") or "clipboard"),
        "history": [], "last": None,
        "latency_ms": None, "latency_median_ms": None, "latency_worst_ms": None,
        "duration_ms": None, "words_last": 0,
        # Живое состояние приложения: подставляет приложение, когда подключено.
        "live": {},
        "counts": {"dictations": 0, "with_text": 0, "dictionary": 0,
                   "snippets": 0, "since_training": 0, "threshold": RETRAIN_THRESHOLD},
    }
    if not path.exists():
        return state
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        state["counts"]["dictations"] = con.execute("select count(*) from history").fetchone()[0]
        state["counts"]["with_text"] = con.execute(
            "select count(*) from history where coalesce(asr_text,'') <> ''").fetchone()[0]
        for table, key in (("dictionary", "dictionary"), ("snippets", "snippets")):
            try:
                state["counts"][key] = con.execute(
                    f"select count(*) from {table}").fetchone()[0]
            except Exception:  # noqa: BLE001
                pass
        rows = [dict(r) for r in con.execute(
            "select id, created_at, coalesce(edited_text, polished_text, asr_text) as text, "
            "duration_ms, e2e_latency_ms, transcript_origin from history "
            "where coalesce(asr_text,'') <> '' order by id desc limit ?", (limit,))]
        state["history"] = rows
        state["last"] = rows[0] if rows else None
        # Живые цифры, которые можно взять из истории: задержка и объём фразы.
        try:
            recent = [row[0] for row in con.execute(
                "select e2e_latency_ms from history where e2e_latency_ms is not null "
                "order by id desc limit 20")]
        except Exception:  # noqa: BLE001
            # Старая или чужая база без этой колонки: карточка задержки просто пустая,
            # а окно продолжает работать.
            recent = []
        if state["last"]:
            state["latency_ms"] = state["last"].get("e2e_latency_ms")
            state["duration_ms"] = state["last"].get("duration_ms")
            state["words_last"] = len((state["last"].get("text") or "").split())
        if recent:
            ordered = sorted(recent)
            state["latency_median_ms"] = ordered[len(ordered) // 2]
            state["latency_worst_ms"] = ordered[-1]
        state["counts"]["since_training"] = _since_training(con)
    finally:
        con.close()
    return state


def _since_training(con: sqlite3.Connection) -> int:
    state_file = ROOT / "data" / "train" / "state.json"
    if not state_file.exists():
        return 0
    try:
        trained_id = int(json.loads(state_file.read_text(encoding="utf-8")).get("max_id", 0))
    except Exception:  # noqa: BLE001
        return 0
    return con.execute("select count(*) from history where id > ?", (trained_id,)).fetchone()[0]


def snapshot_for(control) -> dict:
    """Данные для окна: пульт приложения, если он подключён, иначе витрина из базы."""
    if control is None:
        state = build_state()
        state["control"] = False
        return state
    try:
        panel = control()                      # это control_snapshot() приложения
    except Exception as exc:  # noqa: BLE001
        state = build_state()
        state["control"] = False
        state["error"] = repr(exc)
        return state
    state = build_state(limit=12)
    state["control"] = True
    # Приложение может отдавать живое состояние (уровень микрофона, «слушаю», стадия).
    # Если не отдаёт - окно честно покажет «—», а не нули.
    if isinstance(panel.get("live"), dict):
        state["live"] = panel["live"]
    state["modules"] = [
        {"key": row.get("key"), "label": row.get("label"),
         "on": bool(row.get("on")), "ok": row.get("ok", True),
         "value": row.get("value", ""), "hint": row.get("hint", "")}
        for row in panel.get("modules", [])]
    state["values"] = [
        {"key": row.get("key"), "label": row.get("label"),
         "value": row.get("value", ""), "hint": row.get("hint", "")}
        for row in panel.get("values", [])]
    return state


# Разметка страницы вынесена отдельно: её переписывают целиком, и резать
# её по живому рядом с логикой сервера - путь к опечаткам.
from myflow.webui_page import PAGE  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    db: Path = DB
    settings_path: Path = SETTINGS
    control = None            # control_snapshot() приложения, если подключено
    apply = None              # control_apply(action, value) приложения

    def _send(self, payload: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/snapshot") or self.path.startswith("/api/state"):
            self._send(json.dumps(snapshot_for(self.control), ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return
        self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/api/action"):
            self._send(b'{"ok":false}', "application/json")
            return
        if self.apply is None:
            self._send(json.dumps({"ok": False, "message":
                                   "окно открыто без приложения: управление недоступно"},
                                  ensure_ascii=False).encode(),
                       "application/json; charset=utf-8", 409)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            action, value = body.get("action"), body.get("value")
            if action == "module":
                ok, message = self.apply("toggle", value)
            elif action == "adjust":
                key, _, delta = str(value).partition(":")
                ok, message = self.apply("adjust", (key, int(delta or 1)))
            else:
                ok, message = self.apply(action, value)
            self._send(json.dumps({"ok": bool(ok), "message": message},
                                  ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        except Exception as exc:  # noqa: BLE001
            self._send(json.dumps({"ok": False, "message": repr(exc)},
                                  ensure_ascii=False).encode(),
                       "application/json; charset=utf-8", 500)

    def log_message(self, *_args) -> None:
        return


def serve(port: int = 0, host: str = "127.0.0.1", control=None, apply=None,
          open_browser: bool = False, quiet: bool = False):
    """Поднять сервер интерфейса. Возвращает (server, url)."""
    handler = type("BoundHandler", (Handler,), {"control": control, "apply": apply})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_address[1]}/"
    if not quiet:
        print(f"  MyFLOW: интерфейс на {url}"
              f"{' (с управлением)' if control else ' (витрина)'}")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    return server, url


def main() -> int:
    ap = argparse.ArgumentParser(description="Красивый интерфейс MyFLOW")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    server, url = serve(args.port, args.host, open_browser=not args.no_open)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  остановлено")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
