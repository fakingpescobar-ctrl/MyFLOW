"""Красивый интерфейс MyFLOW: страница в браузере поверх того же ядра.

Зачем так, а не Electron. Всё «мозги» MyFLOW уже на Python: распознавание,
постпроцессор, история, серверы моделей. Electron требует либо второй процесс с
мостом, либо переезд на Node — то есть переписывание всего, что мы измеряли и
отлаживали. Здесь наоборот: Python отдаёт HTML (только стандартная библиотека, без
новых зависимостей), а интерфейс получается такой же, как веб. Эту же страницу потом
можно завернуть в pywebview или Electron — интерфейс переносится без переписывания.

Что показывает: состояние (какой ASR, постпроцессор, Oracle), последнюю фразу,
историю, счётчики (диктовок, словарь, накоплено на дообучение), режим вставки.

    python -m myflow.webui              # откроет страницу в браузере
    python -m myflow.webui --port 8790 --no-open
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
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


def build_state(db: Path | None = None, settings: Path | None = None,
                limit: int = 12) -> dict:
    """Состояние для интерфейса: только чтение, ничего не меняет."""
    path = Path(db) if db is not None else DB
    cfg = read_settings(settings)

    state = {
        "ready": path.exists(),
        "asr": "GigaAM v3 e2e (CPU)",
        "polish": bool((cfg.get("polish") or {}).get("enabled", True)),
        "postproc": bool((cfg.get("polish") or {}).get("postproc", True)),
        "insert_mode": ((cfg.get("insert_method") or "clipboard")),
        "history": [],
        "counts": {"dictations": 0, "with_text": 0, "dictionary": 0,
                   "snippets": 0, "since_training": 0, "threshold": RETRAIN_THRESHOLD},
        "last": None,
    }
    if not path.exists():
        return state

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        state["counts"]["dictations"] = con.execute(
            "select count(*) from history").fetchone()[0]
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
            "duration_ms, transcript_origin from history "
            "where coalesce(asr_text,'') <> '' order by id desc limit ?", (limit,))]
        state["history"] = rows
        if rows:
            state["last"] = rows[0]
        state["counts"]["since_training"] = _since_training(con)
    finally:
        con.close()
    return state


def _since_training(con: sqlite3.Connection) -> int:
    """Сколько диктовок накопилось с прошлого дообучения (для строки «Дообучение»)."""
    state_file = ROOT / "data" / "train" / "state.json"
    if not state_file.exists():
        return 0
    try:
        trained_id = int(json.loads(state_file.read_text(encoding="utf-8")).get("max_id", 0))
    except Exception:  # noqa: BLE001
        return 0
    return con.execute("select count(*) from history where id > ?", (trained_id,)).fetchone()[0]


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>MyFLOW</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0f1115; --card:#171a21; --line:#242833; --text:#e6e8ee;
          --dim:#8b93a7; --accent:#5eead4; --warn:#fbbf24; --bad:#f87171; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:15px/1.5 "Segoe UI",system-ui,sans-serif; padding:28px; }
  h1 { font-size:19px; margin:0 0 4px; letter-spacing:.2px; }
  .sub { color:var(--dim); font-size:13px; margin-bottom:22px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
          gap:14px; margin-bottom:22px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:16px 18px; }
  .card .label { color:var(--dim); font-size:12px; text-transform:uppercase;
                 letter-spacing:.08em; margin-bottom:6px; }
  .card .value { font-size:17px; font-weight:600; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%;
         margin-right:8px; background:var(--accent); vertical-align:middle; }
  .dot.off { background:var(--dim); }
  .dot.warn { background:var(--warn); }
  .last { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:18px; margin-bottom:22px; }
  .last .text { font-size:16px; white-space:pre-wrap; }
  table { width:100%; border-collapse:collapse; background:var(--card);
          border:1px solid var(--line); border-radius:14px; overflow:hidden; }
  th,td { text-align:left; padding:10px 14px; border-bottom:1px solid var(--line);
          font-size:14px; vertical-align:top; }
  th { color:var(--dim); font-weight:500; font-size:12px; text-transform:uppercase;
       letter-spacing:.06em; }
  tr:last-child td { border-bottom:none; }
  .time { color:var(--dim); white-space:nowrap; font-variant-numeric:tabular-nums; }
  .pill { font-size:12px; color:var(--dim); border:1px solid var(--line);
          border-radius:999px; padding:2px 10px; }
  footer { color:var(--dim); font-size:12px; margin-top:20px; }
</style></head>
<body>
  <h1><span class="dot" id="dot"></span>MyFLOW</h1>
  <div class="sub">локальная диктовка · голос никуда не отправляется</div>

  <div class="grid">
    <div class="card"><div class="label">Распознавание</div>
      <div class="value" id="asr">—</div></div>
    <div class="card"><div class="label">Постпроцессор</div>
      <div class="value" id="postproc">—</div></div>
    <div class="card"><div class="label">Куда вставлять</div>
      <div class="value" id="insert">—</div></div>
    <div class="card"><div class="label">Дообучение</div>
      <div class="value" id="retrain">—</div></div>
  </div>

  <div class="last">
    <div class="label" style="color:var(--dim);font-size:12px;letter-spacing:.08em;
         text-transform:uppercase;margin-bottom:8px">Последняя фраза</div>
    <div class="text" id="last">пока ничего не надиктовано</div>
  </div>

  <table>
    <thead><tr><th style="width:110px">Время</th><th>Текст</th>
      <th style="width:120px">Источник</th></tr></thead>
    <tbody id="rows"><tr><td colspan="3" class="time">загрузка…</td></tr></tbody>
  </table>
  <footer id="counts"></footer>

<script>
const MODES = { clipboard:"активное окно", opencode:"OpenCode (без фокуса)",
                dsh:"DeepSeek Harness (без фокуса)", none:"никуда (только история)" };
function esc(s){ return (s||"").replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c])); }
function shortTime(t){ return (t||"").slice(11,19) || "—"; }
async function tick(){
  try {
    const r = await fetch("/api/state", {cache:"no-store"});
    const s = await r.json();
    document.getElementById("asr").textContent = s.asr;
    document.getElementById("postproc").textContent = s.polish
      ? (s.postproc ? "своя 0.6B (включён)" : "—") : "выключен";
    document.getElementById("insert").textContent = MODES[s.insert_mode] || s.insert_mode;
    const need = s.counts.threshold - s.counts.since_training;
    document.getElementById("retrain").textContent = need > 0
      ? `через ${need} диктовок` : "пора дообучить";
    const dot = document.getElementById("dot");
    dot.className = "dot" + (s.ready ? "" : " off");
    if (s.last) {
      document.getElementById("last").textContent = s.last.text || "—";
    }
    document.getElementById("rows").innerHTML = (s.history||[]).map(h =>
      `<tr><td class="time">${shortTime(h.created_at)}</td><td>${esc(h.text)}</td>` +
      `<td><span class="pill">${esc(h.transcript_origin||"—")}</span></td></tr>`).join("")
      || `<tr><td colspan="3" class="time">история пуста</td></tr>`;
    document.getElementById("counts").textContent =
      `диктовок ${s.counts.dictations} · с текстом ${s.counts.with_text} · ` +
      `словарь ${s.counts.dictionary} · сниппетов ${s.counts.snippets}`;
  } catch (e) {
    document.getElementById("dot").className = "dot warn";
  }
}
tick(); setInterval(tick, 1000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    db: Path = DB
    settings_path: Path = SETTINGS

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/state"):
            payload = json.dumps(build_state(self.db, self.settings_path),
                                 ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:  # тишина в консоли
        return


def main() -> int:
    ap = argparse.ArgumentParser(description="Красивый интерфейс MyFLOW")
    ap.add_argument("--port", type=int, default=0, help="0 — взять свободный")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    port = server.server_address[1]
    url = f"http://{args.host}:{port}/"
    print(f"  MyFLOW: интерфейс на {url}  (Ctrl+C — остановить)")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  остановлено")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
