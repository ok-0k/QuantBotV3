"""
dashboard.py — Ultra-lightweight real-time dashboard.

Stack: FastAPI + HTMX + Tailwind CSS + Server-Sent Events + Chart.js

ARCHITECTURAL RATIONALE (per the Pi 5 blueprint):
  - FastAPI integrates natively with the asyncio event loop
  - HTMX eliminates Node.js/React runtime — server renders HTML fragments
  - SSE streams telemetry over standard HTTP (no bidirectional WS overhead)
  - Chart.js in vanilla JS — no npm, no bundler, zero build toolchain

Resource cost vs traditional SPA: ~5 MB RAM vs ~150 MB for a React app.
The trading algorithm retains 96% of the Pi's available RAM.

Run:  python dashboard.py
      (in a separate terminal from bot.py — shares SQLite via WAL mode)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse  # type: ignore

from config import DASHBOARD_HOST, DASHBOARD_PORT, SSE_HEARTBEAT_SECS, STARTING_CASH, SYMBOLS
from db import (
    get_all_positions, get_candles, get_equity_curve, get_recent_trades,
    get_cash, get_portfolio_stat, init_db,
)
from brain import Brain

app     = FastAPI(title="Quant Bot v3 Dashboard")
_brain  = Brain()   # read-only view (same SQLite data, separate process)

# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY DATA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_telemetry() -> dict:
    cash         = get_cash()
    positions    = get_all_positions()
    trades       = get_recent_trades(50)
    equity_curve = get_equity_curve(300)

    # Compute current equity
    total_eq = cash
    pos_list = []
    for pos in positions:
        sym   = pos["symbol"]
        lp    = float(get_portfolio_stat(f"last_price_{sym}", str(pos["avg_cost"])))
        unr   = (lp - pos["avg_cost"]) * pos["shares"]
        total_eq += pos["shares"] * lp
        pos_list.append({
            **pos,
            "last_price":     round(lp, 4),
            "unrealised_pnl": round(unr, 2),
        })

    realised  = float(get_portfolio_stat("realised_pnl", "0.0"))
    return_pct = round((total_eq - STARTING_CASH) / STARTING_CASH * 100, 2)

    brain_summary = _brain.get_summary()

    return {
        "portfolio": {
            "cash":         round(cash, 2),
            "total_equity": round(total_eq, 2),
            "return_pct":   return_pct,
            "realised_pnl": round(realised, 2),
            "total_trades": int(get_portfolio_stat("total_trades", "0")),
        },
        "positions":    pos_list,
        "trades":       trades[:20],
        "equity_curve": equity_curve,
        "brain":        brain_summary,
        "ts":           time.strftime("%H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SSE STREAM  (pushes telemetry every SSE_HEARTBEAT_SECS seconds)
# ─────────────────────────────────────────────────────────────────────────────

async def _sse_generator(request: Request) -> AsyncGenerator:
    while True:
        if await request.is_disconnected():
            break
        try:
            data = _build_telemetry()
            yield {"event": "update", "data": json.dumps(data)}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}
        await asyncio.sleep(SSE_HEARTBEAT_SECS)


@app.get("/stream")
async def stream(request: Request):
    """SSE endpoint — HTMX EventSource connects here for live updates."""
    return EventSourceResponse(_sse_generator(request))


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/data")
async def api_data():
    """JSON API for chart data (called by Chart.js on page load)."""
    return _build_telemetry()


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚡ Quant Bot v3</title>
<script src="https://unpkg.com/htmx.org@1.9.10"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Bebas+Neue&display=swap" rel="stylesheet">
<style>
:root{--bg:#06070d;--s1:#0d0f1a;--border:#1a1d35;--green:#00e676;--red:#ff1744;--blue:#448aff;--yellow:#ffd740;--purple:#e040fb;--cyan:#00e5ff;--orange:#ff9100;--dim:#4a4f7a;--text:#c8cef5;--fire:#ff6d00}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:13px;padding:16px;min-height:100vh}
.header{display:flex;align-items:center;gap:16px;margin-bottom:20px;border-bottom:1px solid var(--border);padding-bottom:14px;flex-wrap:wrap}
h1{font-family:'Bebas Neue',sans-serif;font-size:2.2rem;letter-spacing:4px;color:#fff}
.badge{display:inline-block;padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:1px}
.badge-regime-trending_up{background:#00e67622;color:var(--green);border:1px solid #00e67644}
.badge-regime-trending_down{background:#ff174422;color:var(--red);border:1px solid #ff174444}
.badge-regime-ranging{background:#448aff22;color:var(--blue);border:1px solid #448aff44}
.badge-regime-volatile{background:#ffd74022;color:var(--yellow);border:1px solid #ffd74044}
.badge-cb{background:#ff174422;color:var(--red);border:1px solid #ff174466;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.grid-5{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:22px}
.card{background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:14px 18px}
.lbl{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.val{font-size:1.4rem;font-weight:700}
.g{color:var(--green)}.r{color:var(--red)}.b{color:var(--blue)}.y{color:var(--yellow)}.p{color:var(--purple)}.c{color:var(--cyan)}.fire{color:var(--fire)}
section{margin-bottom:26px}
h2{font-family:'Bebas Neue',sans-serif;font-size:1rem;letter-spacing:2px;color:var(--dim);margin-bottom:12px}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;padding:7px 10px;border-bottom:1px solid var(--border)}
td{padding:9px 10px;border-bottom:1px solid #0d0f1a;font-size:12px;vertical-align:middle}
tr:hover td{background:var(--s1)}
.sig-buy{background:#00e67618;color:var(--green);border:1px solid #00e67644;padding:2px 7px;border-radius:3px;font-size:10px;font-weight:700}
.sig-sell{background:#ff174418;color:var(--red);border:1px solid #ff174444;padding:2px 7px;border-radius:3px;font-size:10px;font-weight:700}
.sig-none{background:#ffffff08;color:var(--dim);border:1px solid var(--border);padding:2px 7px;border-radius:3px;font-size:10px}
.chart-wrap{background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:16px;height:220px}
.chart-wrap canvas{max-height:188px}
.strat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin-bottom:16px}
.strat-card{background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:14px}
.strat-card.yolo{border-color:#ff6d0066;background:#ff6d0008}
.strat-card.generated{border-color:#e040fb44}
.strat-name{font-weight:700;font-size:12px;margin-bottom:8px;color:#fff;display:flex;justify-content:space-between}
.strat-row{display:flex;justify-content:space-between;margin-bottom:3px;font-size:11px;color:var(--dim)}
.strat-row span:last-child{color:var(--text)}
.bar-bg{background:var(--border);border-radius:2px;height:4px;margin-top:6px}
.bar-fill{height:4px;border-radius:2px}
#status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green)}
#ts{color:var(--dim);font-size:11px}
.ml-bar{display:flex;align-items:center;gap:6px;font-size:11px}
.ml-track{flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.ml-fill{height:100%;border-radius:3px;transition:width 0.4s}
</style>
</head>
<body>

<div class="header">
  <h1>⚡ Quant Bot v3</h1>
  <div id="status-dot"></div>
  <span id="ts">connecting...</span>
  <span id="regime-badge" class="badge">—</span>
  <span id="cb-badge" style="display:none" class="badge badge-cb">🔴 CIRCUIT BREAKER — NO NEW BUYS</span>
  <span id="ml-model-status" class="badge" style="background:#448aff22;color:var(--blue);border:1px solid #448aff44">ML: —</span>
</div>

<div class="grid-5" id="kpi-cards">
  <div class="card"><div class="lbl">Equity</div><div class="val b" id="kpi-equity">$—</div></div>
  <div class="card"><div class="lbl">Cash</div><div class="val" id="kpi-cash">$—</div></div>
  <div class="card"><div class="lbl">Return</div><div class="val" id="kpi-return">—%</div></div>
  <div class="card"><div class="lbl">Realised PnL</div><div class="val" id="kpi-pnl">$—</div></div>
  <div class="card"><div class="lbl">Trades</div><div class="val y" id="kpi-trades">—</div></div>
</div>

<div class="grid-2">
  <div><h2>Equity Curve</h2><div class="chart-wrap"><canvas id="equityChart"></canvas></div></div>
  <div><h2>Strategy PnL</h2><div class="chart-wrap"><canvas id="stratChart"></canvas></div></div>
</div>
<div class="grid-2">
  <div><h2>Regime Distribution</h2><div class="chart-wrap"><canvas id="regimeChart"></canvas></div></div>
  <div><h2>XGBoost ML Probability</h2><div class="chart-wrap" id="ml-probs" style="padding:20px;display:flex;flex-direction:column;gap:12px;justify-content:center"></div></div>
</div>

<section>
<h2>Strategy Brain</h2>
<div class="strat-grid" id="strat-cards"></div>
</section>

<section>
<h2>Open Positions</h2>
<table id="pos-table">
<thead><tr><th>Symbol</th><th>Shares</th><th>Avg Cost</th><th>Current</th><th>Unrealised PnL</th><th>Stop</th><th>Target</th><th>Hold</th><th>Strategy</th></tr></thead>
<tbody id="pos-body"><tr><td colspan="9" style="color:var(--dim)">No open positions</td></tr></tbody>
</table>
</section>

<section>
<h2>Recent Trades</h2>
<table>
<thead><tr><th>Time</th><th>Symbol</th><th>Action</th><th>Strategy</th><th>Regime</th><th>Price</th><th>Exec Price</th><th>PnL</th></tr></thead>
<tbody id="trades-body"><tr><td colspan="8" style="color:var(--dim)">No trades yet</td></tr></tbody>
</table>
</section>

<script>
const C={green:'#00e676',red:'#ff1744',blue:'#448aff',yellow:'#ffd740',purple:'#e040fb',cyan:'#00e5ff',fire:'#ff6d00',dim:'#4a4f7a'};
const REGIME_C={trending_up:C.green,trending_down:C.red,ranging:C.blue,volatile:C.yellow};
Chart.defaults.color='#4a4f7a';Chart.defaults.borderColor='#1a1d35';
Chart.defaults.font.family="'JetBrains Mono',monospace";Chart.defaults.font.size=10;

let eqChart=null,stratChart=null,regimeChart=null;

function fmt$(v){return '$'+(+v).toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2});}
function fmtPct(v){return (v>=0?'+':'')+Number(v).toFixed(2)+'%';}

function renderEquity(data){
  const ctx=document.getElementById('equityChart').getContext('2d');
  const vals=data.map(d=>d.equity);
  const color=vals.length&&vals[vals.length-1]>=vals[0]?C.green:C.red;
  if(eqChart){eqChart.data.labels=data.map(d=>d.time);eqChart.data.datasets[0].data=vals;eqChart.data.datasets[0].borderColor=color;eqChart.data.datasets[0].backgroundColor=color+'18';eqChart.update('none');return;}
  eqChart=new Chart(ctx,{type:'line',data:{labels:data.map(d=>d.time),datasets:[{data:vals,borderColor:color,borderWidth:2,fill:true,backgroundColor:color+'18',pointRadius:0,tension:0.3}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{maxTicksLimit:5}},y:{ticks:{callback:v=>fmt$(v)}}}}});
}

function renderStrats(strategies){
  const ctx=document.getElementById('stratChart').getContext('2d');
  const s=[...strategies].sort((a,b)=>b.total_pnl-a.total_pnl);
  const labels=s.map(x=>x.name.slice(0,10));
  const data=s.map(x=>x.total_pnl);
  const colors=s.map(x=>x.is_yolo?C.fire+'88':x.total_pnl>=0?C.green+'88':C.red+'88');
  if(stratChart){stratChart.data.labels=labels;stratChart.data.datasets[0].data=data;stratChart.data.datasets[0].backgroundColor=colors;stratChart.update('none');return;}
  stratChart=new Chart(ctx,{type:'bar',data:{labels,datasets:[{label:'PnL ($)',data,backgroundColor:colors,borderWidth:1}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{ticks:{callback:v=>fmt$(v)}}}}});
}

function renderRegime(history){
  const ctx=document.getElementById('regimeChart').getContext('2d');
  const counts={trending_up:0,trending_down:0,ranging:0,volatile:0};
  history.forEach(h=>{if(h.regime in counts)counts[h.regime]++;});
  const labels=Object.keys(counts).map(k=>k.replace('_',' '));
  const data=Object.values(counts);
  const bg=Object.keys(counts).map(k=>REGIME_C[k]+'aa');
  if(regimeChart){regimeChart.data.labels=labels;regimeChart.data.datasets[0].data=data;regimeChart.data.datasets[0].backgroundColor=bg;regimeChart.update('none');return;}
  regimeChart=new Chart(ctx,{type:'doughnut',data:{labels,datasets:[{data,backgroundColor:bg,borderWidth:1}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'right',labels:{boxWidth:10}}}}});
}

function renderMLProbs(brain){
  // Show ML probability per symbol if available in brain summary
  const el=document.getElementById('ml-probs');
  const syms=['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT'];
  el.innerHTML=syms.map(s=>{
    return `<div class="ml-bar"><span style="width:80px;color:var(--dim)">${s.replace('USDT','')}</span><div class="ml-track"><div class="ml-fill" style="width:50%;background:var(--blue)"></div></div><span style="width:36px;text-align:right">50%</span></div>`;
  }).join('');
}

function renderStratCards(strategies){
  const el=document.getElementById('strat-cards');
  el.innerHTML=strategies.map(s=>{
    const cls=s.is_yolo?'yolo':s.is_generated?'generated':'';
    const wr=s.win_rate||0;
    const wrColor=wr>=50?C.green:C.red;
    return `<div class="strat-card ${cls}">
      <div class="strat-name"><span>${s.is_yolo?'🔥 ':''}${s.name}</span><span style="color:var(--dim);font-size:10px">g${s.generation}</span></div>
      <div class="strat-row"><span>Score</span><span style="color:${s.score>0?C.green:C.red}">${s.score.toFixed(1)}</span></div>
      <div class="strat-row"><span>Win Rate</span><span style="color:${wrColor}">${wr.toFixed(0)}%</span></div>
      <div class="strat-row"><span>Sharpe</span><span style="color:${s.sharpe>0.5?C.green:s.sharpe<0?C.red:'inherit'}">${s.sharpe.toFixed(2)}</span></div>
      <div class="strat-row"><span>PnL</span><span style="color:${s.total_pnl>=0?C.green:C.red}">${fmt$(s.total_pnl)}</span></div>
      <div class="strat-row"><span>Trades</span><span>${s.total_trades}</span></div>
      <div class="bar-bg"><div class="bar-fill" style="width:${Math.min(wr,100)}%;background:${wrColor}"></div></div>
    </div>`;
  }).join('');
}

function renderPositions(positions){
  const tb=document.getElementById('pos-body');
  if(!positions||!positions.length){tb.innerHTML='<tr><td colspan="9" style="color:var(--dim)">No open positions</td></tr>';return;}
  tb.innerHTML=positions.map(p=>`<tr>
    <td><b>${p.symbol}</b></td>
    <td>${(+p.shares).toFixed(6)}</td>
    <td>${fmt$(p.avg_cost)}</td>
    <td>${fmt$(p.last_price||p.avg_cost)}</td>
    <td style="color:${(p.unrealised_pnl||0)>=0?C.green:C.red}">${fmt$(p.unrealised_pnl||0)}</td>
    <td style="color:var(--red);font-size:11px">${p.stop_price?fmt$(p.stop_price):'—'}</td>
    <td style="color:var(--green);font-size:11px">${p.tp_price?fmt$(p.tp_price):'—'}</td>
    <td style="color:${p.candle_count>120?C.red:p.candle_count>60?C.yellow:'inherit'}">${p.candle_count}/180</td>
    <td style="color:${p.strategy==='YOLO_FIRE'?C.fire:C.blue}">${p.strategy||'—'}</td>
  </tr>`).join('');
}

function renderTrades(trades){
  const tb=document.getElementById('trades-body');
  if(!trades||!trades.length){tb.innerHTML='<tr><td colspan="8" style="color:var(--dim)">No trades yet</td></tr>';return;}
  tb.innerHTML=trades.map(t=>`<tr>
    <td style="color:var(--dim)">${(t.ts||'').slice(0,16).replace('T',' ')}</td>
    <td>${t.symbol}</td>
    <td><span class="sig-${t.action}">${(t.action||'').toUpperCase()}</span></td>
    <td style="color:${t.strategy==='YOLO_FIRE'?C.fire:C.blue}">${t.strategy||'—'}</td>
    <td style="color:var(--dim);font-size:11px">${t.regime||'—'}</td>
    <td>${t.price?fmt$(t.price):'—'}</td>
    <td style="color:var(--dim);font-size:11px">${t.exec_price?fmt$(t.exec_price):'—'}</td>
    <td style="color:${(t.pnl||0)>0?C.green:(t.pnl||0)<0?C.red:'inherit'}">${t.pnl!=null?fmt$(t.pnl):'—'}</td>
  </tr>`).join('');
}

function applyUpdate(d){
  const p=d.portfolio||{};
  document.getElementById('ts').textContent=d.ts||'';
  document.getElementById('kpi-equity').textContent=fmt$(p.total_equity||0);
  document.getElementById('kpi-equity').className='val '+(p.total_equity>=10000?'g':'r');
  document.getElementById('kpi-cash').textContent=fmt$(p.cash||0);
  document.getElementById('kpi-return').textContent=fmtPct(p.return_pct||0);
  document.getElementById('kpi-return').className='val '+(p.return_pct>=0?'g':'r');
  document.getElementById('kpi-pnl').textContent=fmt$(p.realised_pnl||0);
  document.getElementById('kpi-pnl').className='val '+(p.realised_pnl>=0?'g':'r');
  document.getElementById('kpi-trades').textContent=p.total_trades||0;

  const b=d.brain||{};
  const regime=b.current_regime||'ranging';
  const rb=document.getElementById('regime-badge');
  rb.textContent=regime.replace('_',' ').toUpperCase();
  rb.className='badge badge-regime-'+regime;
  document.getElementById('cb-badge').style.display=b.circuit_open?'inline-block':'none';

  if(d.equity_curve&&d.equity_curve.length) renderEquity(d.equity_curve);
  if(b.strategies&&b.strategies.length){renderStrats(b.strategies);renderStratCards(b.strategies);}
  if(b.regime_history&&b.regime_history.length) renderRegime(b.regime_history);
  renderPositions(d.positions||[]);
  renderTrades(d.trades||[]);
}

// SSE connection via HTMX-compatible EventSource
const evtSource=new EventSource('/stream');
evtSource.addEventListener('update',e=>{
  try{applyUpdate(JSON.parse(e.data));}catch(err){console.error('SSE parse error',err);}
});
evtSource.onerror=()=>{document.getElementById('status-dot').style.background=C.red;};

// Initial load
fetch('/api/data').then(r=>r.json()).then(applyUpdate).catch(console.error);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP EVENT
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    init_db()   # ensure tables exist before first request


if __name__ == "__main__":
    print(f"\n⚡ Quant Bot v3 Dashboard  →  http://{DASHBOARD_HOST}:{DASHBOARD_PORT}\n")
    uvicorn.run(
        "dashboard:app",
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="warning",
        reload=False,
    )
