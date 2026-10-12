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
            "duration_ms, transcript_origin from history "
            "where coalesce(asr_text,'') <> '' order by id desc limit ?", (limit,))]
        state["history"] = rows
        state["last"] = rows[0] if rows else None
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


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>MyFLOW</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --bg:#06070b; --bg2:#0c0e14; --card:#101219; --line:#1d2230;
    --text:#f2f4f8; --dim:#8b93a7;
    --orange:#ff8a1f; --orange2:#ffb347; --neon:#22d3ee; --magenta:#e879f9;
    --ok:#3ddc97; --warn:#fbbf24; --bad:#fb7185;
    --glow:0 0 22px rgba(255,138,31,.28);
  }
  *{box-sizing:border-box}
  body{margin:0;color:var(--text);font:15px/1.55 "Segoe UI",system-ui,sans-serif;
       padding:26px 30px 44px;background:
         radial-gradient(900px 480px at 12% -12%, rgba(255,138,31,.14) 0%, transparent 60%),
         radial-gradient(800px 420px at 92% 0%, rgba(34,211,238,.10) 0%, transparent 55%),
         linear-gradient(180deg,#0a0c12 0%,var(--bg) 60%);
       background-attachment:fixed}
  body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
       background-image:linear-gradient(rgba(255,255,255,.028) 1px,transparent 1px),
                        linear-gradient(90deg,rgba(255,255,255,.028) 1px,transparent 1px);
       background-size:44px 44px;mask-image:linear-gradient(180deg,#000 0%,transparent 78%)}
  body>*{position:relative;z-index:1}
  header{display:flex;align-items:center;gap:14px;margin-bottom:22px}
  .logo{width:40px;height:40px;border-radius:12px;letter-spacing:.5px;
        background:linear-gradient(135deg,var(--orange),var(--orange2));
        display:flex;align-items:center;justify-content:center;font-weight:800;font-size:16px;
        color:#120a02;box-shadow:var(--glow)}
  h1{font-size:20px;margin:0;letter-spacing:.2px}
  .sub{color:var(--dim);font-size:13px}
  .badge{margin-left:auto;font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;
         padding:6px 14px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
  .badge.live{border-color:rgba(255,138,31,.55);color:var(--orange2);
              box-shadow:0 0 16px rgba(255,138,31,.22)}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:13px;
        margin-bottom:24px}
  .card{background:linear-gradient(180deg,var(--card),var(--bg2));border:1px solid var(--line);
        border-radius:15px;padding:15px 17px;transition:border-color .15s,box-shadow .15s;
        position:relative;overflow:hidden}
  .card::after{content:"";position:absolute;left:0;top:0;height:2px;width:100%;opacity:.55;
        background:linear-gradient(90deg,transparent,var(--orange),transparent)}
  .card:hover{border-color:rgba(255,138,31,.35);box-shadow:0 0 20px rgba(255,138,31,.10)}
  .card .label{color:var(--dim);font-size:11px;text-transform:uppercase;
               letter-spacing:.14em;margin-bottom:7px}
  .card .mono{font-family:Consolas,"Cascadia Mono",monospace;font-size:13px;
              color:var(--orange2);word-break:break-all;line-height:1.35}
  .card .value{font-size:16px;font-weight:600}
  .card .hint{color:var(--dim);font-size:12px;margin-top:5px}
  .on .value{color:var(--ok);text-shadow:0 0 12px rgba(61,220,151,.35)}
  .off .value{color:var(--dim)}
  h2{font-size:12px;text-transform:uppercase;letter-spacing:.18em;color:var(--orange);
     margin:28px 0 12px;display:flex;align-items:center;gap:10px}
  h2::after{content:"";flex:1;height:1px;
     background:linear-gradient(90deg,rgba(255,138,31,.45),transparent)}
  .row{display:flex;align-items:center;gap:14px;background:var(--card);
       border:1px solid var(--line);border-radius:13px;padding:12px 16px;margin-bottom:9px}
  .row .name{flex:1;min-width:0}
  .row .name b{font-weight:600}
  .row .name .v{color:var(--dim);font-size:13px;margin-top:2px}
  .sw{width:52px;height:28px;border-radius:999px;border:1px solid var(--line);
      background:#0e1119;position:relative;cursor:pointer;flex:none;transition:.15s}
  .sw i{position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;
        background:#4b5468;transition:.15s}
  .sw.on{background:linear-gradient(135deg,var(--orange),var(--orange2));
         border-color:transparent;box-shadow:0 0 16px rgba(255,138,31,.45)}
  .sw.on i{left:27px;background:#fff}
  .sw.na{opacity:.35;cursor:not-allowed}
  .stepper{display:flex;align-items:center;gap:8px;flex:none}
  button.ctl{background:#151824;color:var(--orange2);border:1px solid var(--line);
             border-radius:10px;padding:7px 13px;font-size:14px;cursor:pointer;
             transition:.12s}
  button.ctl:hover{border-color:var(--orange);color:#fff;box-shadow:0 0 14px rgba(255,138,31,.35)}
  .last{background:linear-gradient(180deg,var(--card),var(--bg2));border:1px solid var(--line);
        border-radius:15px;padding:18px;margin-bottom:22px}
  .last .t{font-size:16px;white-space:pre-wrap}
  table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
        border-radius:15px;overflow:hidden}
  th,td{text-align:left;padding:10px 15px;border-bottom:1px solid var(--line);font-size:14px;
        vertical-align:top}
  th{color:var(--dim);font-weight:500;font-size:11.5px;text-transform:uppercase;letter-spacing:.08em}
  tr:last-child td{border-bottom:none}
  .time{color:var(--dim);white-space:nowrap;font-variant-numeric:tabular-nums}
  .pill{font-size:11px;letter-spacing:.04em;color:var(--neon);
        border:1px solid rgba(34,211,238,.3);border-radius:999px;padding:2px 10px;
        font-family:Consolas,monospace}
  footer{color:var(--dim);font-size:12px;margin-top:18px}
  .toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);
         background:#12141d;border:1px solid rgba(255,138,31,.4);
         box-shadow:0 0 24px rgba(255,138,31,.22);color:var(--text);
         padding:10px 18px;border-radius:12px;font-size:14px;opacity:0;transition:.2s;
         pointer-events:none}
  .toast.show{opacity:1}
</style></head>
<body>
<header>
  <div class="logo">MF</div>
  <div><h1>MyFLOW</h1>
    <div class="sub">локальная диктовка · голос никуда не отправляется</div></div>
  <div class="badge" id="mode">определяю…</div>
</header>

<div class="grid">
  <div class="card"><div class="label">Распознавание</div>
    <div class="value" id="asr">—</div></div>
  <div class="card" id="c-postproc"><div class="label">Постпроцессор</div>
    <div class="value" id="postproc">—</div></div>
  <div class="card"><div class="label">Куда вставлять</div>
    <div class="value" id="insert">—</div></div>
  <div class="card"><div class="label">Дообучение</div>
    <div class="value" id="retrain">—</div></div>
</div>

<h2>Управление</h2>
<div id="controls"><div class="row"><div class="name">Загрузка…</div></div></div>

<h2>Последняя фраза</h2>
<div class="last"><div class="t" id="last">пока ничего не надиктовано</div></div>

<h2>История</h2>
<table><thead><tr><th style="width:100px">Время</th><th>Текст</th>
  <th style="width:110px">Источник</th></tr></thead>
<tbody id="rows"><tr><td colspan="3" class="time">загрузка…</td></tr></tbody></table>
<footer id="counts"></footer>
<div class="toast" id="toast"></div>

<script>
const MODES = { clipboard:"активное окно", opencode:"OpenCode (без фокуса)",
                dsh:"DeepSeek Harness (без фокуса)", none:"никуда (только история)" };
const esc = s => (s||"").replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
const shortTime = t => (t||"").slice(11,19) || "—";
let attached = false;

function toast(msg){
  const el = document.getElementById("toast");
  el.textContent = msg; el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 1800);
}

async function act(action, value){
  try{
    const r = await fetch("/api/action", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({action, value})});
    const j = await r.json();
    toast(j.message || (j.ok ? "готово" : "не вышло"));
    tick();
  }catch(e){ toast("окно не может управлять: приложение не подключено"); }
}

function renderControls(s){
  const box = document.getElementById("controls");
  if(!s.control){
    box.innerHTML = `<div class="row"><div class="name"><b>Управление недоступно</b>
      <div class="v">Окно открыто отдельно от приложения: видно историю и счётчики,
      но кнопки не работают. Запусти интерфейс из MyFLOW (ui.bat при работающем
      приложении) — и здесь появится полный пульт.</div></div></div>`;
    return;
  }
  const mods = (s.modules||[]).map(m => `
    <div class="row">
      <div class="name"><b>${esc(m.label)}</b>
        <div class="v">${esc(m.value||"")}${m.hint ? " · " + esc(m.hint) : ""}</div></div>
      <div class="sw ${m.on ? "on" : ""} ${m.ok === false ? "na" : ""}"
           onclick="act('module','${esc(m.key)}')"><i></i></div>
    </div>`).join("");
  const vals = (s.values||[]).map(v => `
    <div class="row">
      <div class="name"><b>${esc(v.label)}</b>
        <div class="v">${esc(v.value||"")}${v.hint ? " · " + esc(v.hint) : ""}</div></div>
      <div class="stepper">
        <button class="ctl" onclick="act('adjust','${esc(v.key)}:-1')">◀</button>
        <button class="ctl" onclick="act('adjust','${esc(v.key)}:+1')">▶</button>
      </div>
    </div>`).join("");
  box.innerHTML = mods + vals;
}

async function tick(){
  try{
    const r = await fetch("/api/snapshot", {cache:"no-store"});
    const s = await r.json();
    attached = !!s.control;
    const badge = document.getElementById("mode");
    badge.textContent = attached ? "управление активно" : "витрина (без управления)";
    badge.className = "badge" + (attached ? " live" : "");
    document.getElementById("asr").textContent = s.asr || "—";
    document.getElementById("postproc").textContent =
      s.postproc_model || "модель не найдена";
    document.getElementById("postproc-state").textContent = s.polish
      ? (s.postproc ? "включён · :8083" : "выключен в настройках")
      : "полировка выключена";
    document.getElementById("c-postproc").className =
      "card " + (s.polish && s.postproc ? "on" : "off");
    document.getElementById("insert").textContent = MODES[s.insert_mode] || s.insert_mode || "—";
    const need = (s.counts.threshold||100) - (s.counts.since_training||0);
    document.getElementById("retrain").textContent =
      need > 0 ? `через ${need} диктовок` : "пора дообучить";
    if(s.last) document.getElementById("last").textContent = s.last.text || "—";
    document.getElementById("rows").innerHTML = (s.history||[]).map(h =>
      `<tr><td class="time">${shortTime(h.created_at)}</td><td>${esc(h.text)}</td>` +
      `<td><span class="pill">${esc(h.transcript_origin||"—")}</span></td></tr>`).join("")
      || `<tr><td colspan="3" class="time">история пуста</td></tr>`;
    document.getElementById("counts").textContent =
      `диктовок ${s.counts.dictations} · с текстом ${s.counts.with_text} · ` +
      `словарь ${s.counts.dictionary} · сниппетов ${s.counts.snippets}`;
    if(s.control) renderControls(s);
    else if(!window._painted){ window._painted = 1; renderControls(s); }
  }catch(e){ document.getElementById("mode").className = "badge"; }
}
tick(); setInterval(tick, 1200);
</script></body></html>
"""


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
