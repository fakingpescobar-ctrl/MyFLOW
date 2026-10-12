"""Разметка страницы MyFLOW: компактно, по вкладкам, без прокрутки ради кнопок.

Почему отдельный файл: страница переписывается целиком, и резать её по живому в
webui.py (где рядом сервер и логика) — путь к опечаткам. Здесь только разметка.

Раскладка:
  * шапка: логотип, название, бейдж управления, счётчики в одну строку;
  * вкладки: Обзор / Управление / История — переключаются без перезагрузки;
  * Обзор: карточки в плотной сетке (распознавание, модель, куда вставлять,
    дообучение, микрофон, задержка, объём фразы) и последняя фраза;
  * Управление: две колонки — модули с тумблерами и значения со стрелками; влезает
    в экран, чтобы не приходилось крутить мышью;
  * История: список фраз, прокручивается внутри вкладки.
"""
PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>MyFLOW</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{
    --bg:#06070b; --card:#101219; --card2:#0c0e14; --line:#1d2230;
    --text:#f2f4f8; --dim:#8b93a7;
    --orange:#ff8a1f; --orange2:#ffb347; --neon:#22d3ee; --magenta:#e879f9;
    --ok:#3ddc97; --warn:#fbbf24; --bad:#fb7185;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;color:var(--text);font:14px/1.45 "Segoe UI",system-ui,sans-serif;
       background:
         radial-gradient(760px 380px at 10% -10%, rgba(255,138,31,.13) 0%, transparent 60%),
         radial-gradient(720px 360px at 95% 0%, rgba(34,211,238,.09) 0%, transparent 55%),
         linear-gradient(180deg,#0a0c12 0%,var(--bg) 60%);
       background-attachment:fixed;display:flex;flex-direction:column;
       height:100vh;overflow:hidden}
  body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.45;
       background-image:linear-gradient(rgba(255,255,255,.026) 1px,transparent 1px),
                        linear-gradient(90deg,rgba(255,255,255,.026) 1px,transparent 1px);
       background-size:40px 40px;mask-image:linear-gradient(180deg,#000,transparent 70%)}
  body>*{position:relative;z-index:1}

  header{display:flex;align-items:center;gap:12px;padding:14px 18px 10px}
  .logo{width:34px;height:34px;border-radius:10px;flex:none;
        background:linear-gradient(135deg,var(--orange),var(--orange2));
        display:flex;align-items:center;justify-content:center;font-weight:800;font-size:14px;
        color:#120a02;box-shadow:0 0 20px rgba(255,138,31,.35)}
  .title{font-size:16px;font-weight:600;letter-spacing:.2px}
  .sub{color:var(--dim);font-size:11.5px}
  .badge{margin-left:auto;font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
         padding:5px 12px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
  .badge.live{border-color:rgba(255,138,31,.55);color:var(--orange2);
              box-shadow:0 0 14px rgba(255,138,31,.22)}
  .counts{color:var(--dim);font-size:11.5px;font-family:Consolas,monospace}

  nav{display:flex;gap:6px;padding:0 18px 10px}
  .tab{padding:7px 15px;border-radius:10px;border:1px solid var(--line);color:var(--dim);
       cursor:pointer;font-size:13px;background:#0d1017;transition:.12s;user-select:none}
  .tab:hover{border-color:rgba(255,138,31,.35);color:var(--text)}
  .tab.active{color:#120a02;border-color:transparent;font-weight:600;
       background:linear-gradient(135deg,var(--orange),var(--orange2));
       box-shadow:0 0 16px rgba(255,138,31,.3)}

  main{flex:1;min-height:0;overflow:auto;padding:4px 18px 18px}
  .panel{display:none} .panel.active{display:block}

  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
  .card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
        border-radius:12px;padding:11px 13px;position:relative;overflow:hidden}
  .card::after{content:"";position:absolute;left:0;top:0;height:2px;width:100%;opacity:.5;
        background:linear-gradient(90deg,transparent,var(--orange),transparent)}
  .card .label{color:var(--dim);font-size:10px;text-transform:uppercase;
               letter-spacing:.13em;margin-bottom:5px}
  .card .value{font-size:14px;font-weight:600}
  .card .mono{font-family:Consolas,"Cascadia Mono",monospace;font-size:11.5px;
              color:var(--orange2);word-break:break-all;line-height:1.3}
  .card .hint{color:var(--dim);font-size:11px;margin-top:3px}
  .on .value{color:var(--ok)} .off .value{color:var(--dim)}
  .level{margin-top:8px;height:5px;border-radius:999px;background:#12151f;
         border:1px solid var(--line);overflow:hidden}
  .level i{display:block;height:100%;width:0%;border-radius:999px;
           background:linear-gradient(90deg,var(--ok),var(--orange),var(--bad));
           transition:width .12s linear}

  h2{font-size:11px;text-transform:uppercase;letter-spacing:.17em;color:var(--orange);
     margin:16px 0 8px;display:flex;align-items:center;gap:9px}
  h2::after{content:"";flex:1;height:1px;
     background:linear-gradient(90deg,rgba(255,138,31,.45),transparent)}
  h2:first-child{margin-top:4px}

  .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .row{display:flex;align-items:center;gap:10px;background:var(--card);
       border:1px solid var(--line);border-radius:10px;padding:7px 11px;margin-bottom:6px}
  .row .name{flex:1;min-width:0}
  .row b{font-weight:600;font-size:13px}
  .row .v{color:var(--dim);font-size:11.5px;margin-top:1px;overflow:hidden;
          text-overflow:ellipsis;white-space:nowrap}
  .sw{width:44px;height:24px;border-radius:999px;border:1px solid var(--line);
      background:#0e1119;position:relative;cursor:pointer;flex:none;transition:.15s}
  .sw i{position:absolute;top:3px;left:3px;width:16px;height:16px;border-radius:50%;
        background:#4b5468;transition:.15s}
  .sw.on{background:linear-gradient(135deg,var(--orange),var(--orange2));
         border-color:transparent;box-shadow:0 0 12px rgba(255,138,31,.4)}
  .sw.on i{left:23px;background:#fff}
  .sw.na{opacity:.35;cursor:not-allowed}
  .stepper{display:flex;gap:6px;flex:none}
  button.ctl{background:#151824;color:var(--orange2);border:1px solid var(--line);
             border-radius:8px;padding:4px 10px;font-size:13px;cursor:pointer;transition:.12s}
  button.ctl:hover{border-color:var(--orange);color:#fff;box-shadow:0 0 12px rgba(255,138,31,.3)}

  .last{background:linear-gradient(180deg,var(--card),var(--card2));
        border:1px solid var(--line);border-radius:12px;padding:13px 15px}
  .last .t{font-size:14px;white-space:pre-wrap}

  table{width:100%;border-collapse:collapse;background:var(--card);
        border:1px solid var(--line);border-radius:12px;overflow:hidden}
  th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--line);font-size:13px}
  th{color:var(--dim);font-weight:500;font-size:10.5px;text-transform:uppercase;letter-spacing:.08em}
  tr:last-child td{border-bottom:none}
  .time{color:var(--dim);white-space:nowrap;font-variant-numeric:tabular-nums;
        font-family:Consolas,monospace;font-size:12px}
  .pill{font-size:10.5px;letter-spacing:.03em;color:var(--neon);
        border:1px solid rgba(34,211,238,.3);border-radius:999px;padding:1px 9px;
        font-family:Consolas,monospace}
  .toast{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);
         background:#12141d;border:1px solid rgba(255,138,31,.4);
         box-shadow:0 0 20px rgba(255,138,31,.22);color:var(--text);
         padding:9px 16px;border-radius:11px;font-size:13px;opacity:0;transition:.2s;
         pointer-events:none}
  .toast.show{opacity:1}
</style></head>
<body>
<header>
  <div class="logo">MF</div>
  <div><div class="title">MyFLOW</div>
    <div class="sub" id="counts">локальная диктовка</div></div>
  <div class="badge" id="mode">определяю…</div>
</header>

<nav>
  <div class="tab active" data-tab="overview">Обзор</div>
  <div class="tab" data-tab="control">Управление</div>
  <div class="tab" data-tab="history">История</div>
</nav>

<main>
  <section class="panel active" id="p-overview">
    <div class="grid">
      <div class="card"><div class="label">Распознавание</div>
        <div class="value" id="asr">—</div></div>
      <div class="card" id="c-postproc"><div class="label">Постпроцессор</div>
        <div class="mono" id="postproc">—</div>
        <div class="hint" id="postproc-state"></div></div>
      <div class="card"><div class="label">Куда вставлять</div>
        <div class="value" id="insert">—</div></div>
      <div class="card"><div class="label">Дообучение</div>
        <div class="value" id="retrain">—</div></div>
      <div class="card" id="c-mic"><div class="label">Микрофон</div>
        <div class="value" id="mic-state">—</div>
        <div class="level"><i id="mic-level"></i></div></div>
      <div class="card"><div class="label">Задержка</div>
        <div class="value" id="latency">—</div>
        <div class="hint" id="latency-last"></div></div>
      <div class="card"><div class="label">Последняя фраза</div>
        <div class="value" id="words">—</div>
        <div class="hint" id="duration"></div></div>
    </div>
    <h2>Последняя фраза</h2>
    <div class="last"><div class="t" id="last">пока ничего не надиктовано</div></div>
  </section>

  <section class="panel" id="p-control">
    <div class="cols">
      <div>
        <h2>Модули</h2>
        <div id="modules"><div class="row"><div class="name">Загрузка…</div></div></div>
      </div>
      <div>
        <h2>Значения</h2>
        <div id="values"><div class="row"><div class="name">Загрузка…</div></div></div>
      </div>
    </div>
    <div id="control-note"></div>
  </section>

  <section class="panel" id="p-history">
    <h2>История</h2>
    <table><thead><tr><th style="width:92px">Время</th><th>Текст</th>
      <th style="width:104px">Источник</th></tr></thead>
    <tbody id="rows"><tr><td colspan="3" class="time">загрузка…</td></tr></tbody></table>
  </section>
</main>

<div class="toast" id="toast"></div>

<script>
const MODES = { clipboard:"активное окно", opencode:"OpenCode (без фокуса)",
                dsh:"DeepSeek Harness (без фокуса)", none:"никуда (только история)" };
const esc = s => (s||"").replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
// Доступ к элементу, который не бросает исключение: окно не должно ломаться целиком
// из-за одного отсутствующего id.
const el = id => document.getElementById(id) || {style:{}, classList:{add(){},remove(){}}};
const setText = (id, value) => { el(id).textContent = value; };
const shortTime = t => (t||"").slice(11,19) || "—";
let attached = false;

document.querySelectorAll(".tab").forEach(tab => tab.onclick = () => {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  tab.classList.add("active");
  const panel = el("p-" + tab.dataset.tab); panel.classList && panel.classList.add("active");
});

function toast(msg){
  const box = el("toast");
  box.textContent = msg; box.classList.add("show");
  setTimeout(() => box.classList.remove("show"), 1800);
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
  if(!s.control){
    el("modules").innerHTML = "";
    el("values").innerHTML = "";
    setText("control-note",
      "Управление недоступно: окно открыто отдельно от приложения. Запусти ui.bat " +
      "(или start-myflow.bat) - и здесь появится полный пульт.");
    return;
  }
  setText("control-note", "");
  el("modules").innerHTML = (s.modules||[]).map(m => `
    <div class="row">
      <div class="name"><b>${esc(m.label)}</b>
        <div class="v">${esc(m.value||"")}${m.hint ? " · " + esc(m.hint) : ""}</div></div>
      <div class="sw ${m.on ? "on" : ""} ${m.ok === false ? "na" : ""}"
           onclick="act('module','${esc(m.key)}')"><i></i></div>
    </div>`).join("") || `<div class="row"><div class="name">пусто</div></div>`;
  el("values").innerHTML = (s.values||[]).map(v => `
    <div class="row">
      <div class="name"><b>${esc(v.label)}</b>
        <div class="v">${esc(v.value||"")}${v.hint ? " · " + esc(v.hint) : ""}</div></div>
      <div class="stepper">
        <button class="ctl" onclick="act('adjust','${esc(v.key)}:-1')">◀</button>
        <button class="ctl" onclick="act('adjust','${esc(v.key)}:+1')">▶</button>
      </div>
    </div>`).join("") || `<div class="row"><div class="name">пусто</div></div>`;
}

async function tick(){
  try{
    const r = await fetch("/api/snapshot", {cache:"no-store"});
    const s = await r.json();
    attached = !!s.control;
    const badge = el("mode");
    badge.textContent = attached ? "управление активно" : "витрина (без управления)";
    badge.className = "badge" + (attached ? " live" : "");

    if(s.control || !window._painted){ window._painted = 1; renderControls(s); }

    setText("asr", s.asr || "—");
    setText("postproc", s.postproc_model || "модель не найдена");
    setText("postproc-state", s.polish
      ? (s.postproc ? "включён · :8083" : "выключен в настройках")
      : "полировка выключена");
    el("c-postproc").className = "card " + (s.polish && s.postproc ? "on" : "off");
    setText("insert", MODES[s.insert_mode] || s.insert_mode || "—");

    const live = s.live || {};
    if (live.listening !== undefined || live.level !== undefined) {
      setText("mic-state", live.listening ? (live.state || "слушаю")
                                          : (live.state || "пауза"));
      el("c-mic").className = "card " + (live.listening ? "on" : "off");
      el("mic-level").style.width = Math.round((live.level || 0) * 100) + "%";
    } else {
      setText("mic-state", s.control ? "нет данных" : "нужен запуск");
      el("c-mic").className = "card off";
      el("mic-level").style.width = "0%";
    }

    const lat = s.latency_median_ms;
    setText("latency", lat ? `${Math.round(lat)} мс`
      : (s.latency_ms ? `${Math.round(s.latency_ms)} мс` : "—"));
    setText("latency-last", s.latency_ms
      ? `последняя ${Math.round(s.latency_ms)} · худшая ${Math.round(s.latency_worst_ms||0)} мс`
      : "надиктуй фразу");
    setText("words", s.words_last ? `${s.words_last} слов` : "—");
    setText("duration", s.duration_ms ? `звучала ${(s.duration_ms/1000).toFixed(1)} с` : "");

    const need = (s.counts.threshold||100) - (s.counts.since_training||0);
    setText("retrain", need > 0 ? `через ${need} диктовок` : "пора дообучить");
    if(s.last) setText("last", s.last.text || "—");

    el("rows").innerHTML = (s.history||[]).map(h =>
      `<tr><td class="time">${shortTime(h.created_at)}</td><td>${esc(h.text)}</td>` +
      `<td><span class="pill">${esc(h.transcript_origin||"—")}</span></td></tr>`).join("")
      || `<tr><td colspan="3" class="time">история пуста</td></tr>`;

    setText("counts",
      `диктовок ${s.counts.dictations} · слов-правок ${s.counts.with_text} · ` +
      `словарь ${s.counts.dictionary} · сниппетов ${s.counts.snippets}`);
  }catch(e){
    el("mode").className = "badge";
    console.error(e);
  }
}
tick(); setInterval(tick, 1200);
</script></body></html>
"""
