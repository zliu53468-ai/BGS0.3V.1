"""Local button-driven decision lab for BGS0.3V.1.

Run:
    python -m uvicorn local_decision_lab:app --host 127.0.0.1 --port 8787 --reload

Then open http://127.0.0.1:8787

This module is intentionally separate from app.py / OCR / LINE flows.  It calls
production predictor.predict() for each live preview, so the displayed direction
uses the same formal decision chain as production while giving extra diagnostics
for follow-last / chase behaviour.
"""
from __future__ import annotations

from typing import Any, Iterable
import html
import math
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from predictor import predict

app = FastAPI(title="BGS Local Decision Lab", version="1.0.0")


class PredictRequest(BaseModel):
    history: str = ""
    bankroll: float = 10000.0
    session_id: str = ""


class BacktestRequest(BaseModel):
    history: str = Field(default="", description="B/P/T sequence")
    bankroll: float = 10000.0
    session_id: str = ""


def _normalize_raw(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = "".join(ch for ch in history.upper() if ch not in " ,|\t\r\n")
        if any(ch not in {"B", "P", "T"} for ch in compact):
            raise ValueError("History must contain only B / P / T")
        return list(compact)
    out: list[str] = []
    for item in history:
        value = str(item or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out


def _bp(values: Iterable[str]) -> list[str]:
    return [v for v in values if v in {"B", "P"}]


def _last_bp(values: Iterable[str]) -> str:
    bp = _bp(values)
    return bp[-1] if bp else ""


def _run_length(values: Iterable[str]) -> tuple[str, int]:
    bp = _bp(values)
    if not bp:
        return "", 0
    side = bp[-1]
    length = 0
    for value in reversed(bp):
        if value != side:
            break
        length += 1
    return side, length


def _side_text(side: str) -> str:
    return "莊" if side == "B" else "閒" if side == "P" else "和"


def _component_direction(component: dict[str, Any]) -> str:
    try:
        p_b = float(component.get("p_b", 0.5) or 0.5)
    except (TypeError, ValueError):
        p_b = 0.5
    return "B" if p_b >= 0.5 else "P"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _predict_payload(req: PredictRequest) -> dict[str, Any]:
    raw = _normalize_raw(req.history)
    sid = (req.session_id or "").strip() or uuid.uuid4().hex[:12]
    result = predict(
        history=raw,
        venue="LOCAL",
        room="DECISION_LAB",
        shoe_id=sid,
        user_id=f"LOCAL-{sid}",
        shoe_context={"bankroll": max(0.0, float(req.bankroll or 0.0))},
    )

    direction = str(result.get("direction") or result.get("recommend") or "B").upper()
    probabilities = dict(result.get("probabilities") or {})
    raw_probabilities = dict(result.get("raw_direction_probabilities") or probabilities)
    road_pattern = dict(result.get("road_pattern_model") or {})
    components = dict(road_pattern.get("components") or {})
    weights = dict(road_pattern.get("component_weights") or {})
    last_side = _last_bp(raw)
    run_side, run_length = _run_length(raw)

    component_rows: list[dict[str, Any]] = []
    same_votes = 0
    active_votes = 0
    for name in ("multi_window", "pattern_replay", "ngram", "pattern_survival"):
        component = dict(components.get(name) or {})
        weight = dict(weights.get(name) or {})
        comp_direction = _component_direction(component)
        relation = "SAME" if last_side and comp_direction == last_side else "SWITCH" if last_side else "N/A"
        effective_weight = _safe_float(weight.get("effective_weight", 0.0))
        if effective_weight > 0.0:
            active_votes += 1
            if relation == "SAME":
                same_votes += 1
        component_rows.append(
            {
                "name": name,
                "direction": comp_direction,
                "direction_text": _side_text(comp_direction),
                "relation_to_last": relation,
                "p_b": _safe_float(component.get("p_b", 0.5), 0.5),
                "p_p": _safe_float(component.get("p_p", 0.5), 0.5),
                "reliability": _safe_float(component.get("reliability", 0.0)),
                "effective_weight": effective_weight,
                "support": component.get("support", 0),
                "pattern": component.get("pattern", ""),
                "desired_relation": component.get("desired_relation", ""),
            }
        )

    relation = "SAME" if last_side and direction == last_side else "SWITCH" if last_side else "N/A"
    chase_ratio = (same_votes / active_votes) if active_votes else 0.0
    chase_risk = (
        "HIGH" if active_votes >= 3 and chase_ratio >= 0.75
        else "MEDIUM" if active_votes >= 2 and chase_ratio >= 0.50
        else "LOW"
    )

    derived = dict(road_pattern.get("derived_ask_road") or {})
    regime = dict(road_pattern.get("regime_gate") or {})
    return {
        "ok": True,
        "session_id": sid,
        "history": "".join(raw),
        "rounds": len(raw),
        "bp_rounds": len(_bp(raw)),
        "direction": direction,
        "direction_text": _side_text(direction),
        "last_side": last_side,
        "last_side_text": _side_text(last_side) if last_side else "—",
        "relation_to_last": relation,
        "run_side": run_side,
        "run_length": run_length,
        "banker_probability": _safe_float(probabilities.get("B", 0.5), 0.5),
        "player_probability": _safe_float(probabilities.get("P", 0.5), 0.5),
        "raw_banker_probability": _safe_float(raw_probabilities.get("B", 0.5), 0.5),
        "raw_player_probability": _safe_float(raw_probabilities.get("P", 0.5), 0.5),
        "confidence": _safe_float(result.get("confidence", 0.5), 0.5),
        "bet_percentage": _safe_float(result.get("bet_percentage", 0.0)),
        "bet_amount": _safe_float(result.get("bet_amount", 0.0)),
        "formal_direction_source": result.get("formal_direction_source", ""),
        "model_version": result.get("model_version", ""),
        "pattern": road_pattern.get("pattern", ""),
        "raw_edge": _safe_float(road_pattern.get("raw_edge", 0.0)),
        "final_edge": _safe_float(road_pattern.get("final_edge", 0.0)),
        "maturity": _safe_float(road_pattern.get("maturity", 0.0)),
        "component_rows": component_rows,
        "active_component_votes": active_votes,
        "same_component_votes": same_votes,
        "same_vote_ratio": chase_ratio,
        "chase_risk": chase_risk,
        "derived_road": {
            "p_b": _safe_float((dict(derived.get("likelihood") or {})).get("B", 0.5), 0.5),
            "reliability": _safe_float(derived.get("reliability", 0.0)),
            "active_roads": list(derived.get("active_roads") or []),
        },
        "regime_gate": {
            "available": bool(regime.get("available")),
            "p_b": _safe_float((dict(regime.get("likelihood") or {})).get("B", 0.5), 0.5),
            "reliability": _safe_float(regime.get("reliability", 0.0)),
            "state": regime.get("state", regime.get("dominant_state", "")),
        },
        "diagnosis": {
            "is_follow_last_call": relation == "SAME",
            "note": (
                "HIGH means at least 3 active V1 components are simultaneously calling SAME. "
                "This usually indicates correlated continuation evidence rather than an independent consensus."
            ),
        },
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "BGS Local Decision Lab"}


@app.post("/api/predict")
def api_predict(req: PredictRequest) -> dict[str, Any]:
    try:
        return _predict_payload(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"predict failed: {exc}") from exc


@app.post("/api/backtest")
def api_backtest(req: BacktestRequest) -> dict[str, Any]:
    try:
        raw = _normalize_raw(req.history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sid = (req.session_id or "").strip() or uuid.uuid4().hex[:12]
    rows: list[dict[str, Any]] = []
    hits = misses = ties = 0
    follow_last_calls = 0
    resolved_calls = 0
    run3_follow_calls = 0
    run3_calls = 0

    for target_index in range(1, len(raw)):
        prefix = raw[:target_index]
        actual = raw[target_index]
        payload = _predict_payload(
            PredictRequest(
                history="".join(prefix),
                bankroll=req.bankroll,
                session_id=f"{sid}-bt-{target_index}",
            )
        )
        predicted = payload["direction"]
        last_side = payload["last_side"]
        relation = payload["relation_to_last"]
        if relation == "SAME":
            follow_last_calls += 1
        if last_side:
            resolved_calls += 1
        if payload["run_length"] >= 3:
            run3_calls += 1
            if relation == "SAME":
                run3_follow_calls += 1

        if actual == "T":
            verdict = "TIE"
            ties += 1
        elif predicted == actual:
            verdict = "HIT"
            hits += 1
        else:
            verdict = "MISS"
            misses += 1

        rows.append(
            {
                "round": target_index + 1,
                "prefix": "".join(prefix),
                "actual": actual,
                "actual_text": _side_text(actual),
                "predicted": predicted,
                "predicted_text": _side_text(predicted),
                "last_side": last_side,
                "relation_to_last": relation,
                "run_length": payload["run_length"],
                "confidence": payload["confidence"],
                "chase_risk": payload["chase_risk"],
                "same_vote_ratio": payload["same_vote_ratio"],
                "verdict": verdict,
            }
        )

    scored = hits + misses
    return {
        "ok": True,
        "session_id": sid,
        "history": "".join(raw),
        "rounds": len(raw),
        "samples": len(rows),
        "scored_samples": scored,
        "hits": hits,
        "misses": misses,
        "ties_skipped": ties,
        "accuracy": (hits / scored) if scored else None,
        "follow_last_call_rate": (follow_last_calls / resolved_calls) if resolved_calls else None,
        "switch_call_rate": (1.0 - follow_last_calls / resolved_calls) if resolved_calls else None,
        "run_ge3_follow_last_rate": (run3_follow_calls / run3_calls) if run3_calls else None,
        "rows": rows,
        "interpretation": (
            "If follow_last_call_rate is persistently high across mixed shoes, inspect correlated SAME/SWITCH "
            "evidence in multi_window + pattern_replay + ngram + pattern_survival before changing LinUCB."
        ),
    }


HTML = r'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>BGS 本地決策測試台</title>
<style>
:root{font-family:Inter,"Noto Sans TC","Microsoft JhengHei",sans-serif;color:#eef2ff;background:#080b14}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 20% 0,#18233d 0,#080b14 42%);min-height:100vh}
.wrap{max-width:1180px;margin:0 auto;padding:24px}.title{font-size:28px;font-weight:900;margin:0}.sub{color:#9aa6c4;margin:6px 0 18px}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:16px}.card{background:rgba(15,20,35,.92);border:1px solid #27324d;border-radius:18px;padding:18px;box-shadow:0 18px 55px rgba(0,0,0,.22)}
.controls{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.btn{border:0;border-radius:14px;padding:15px 22px;font-weight:900;font-size:18px;cursor:pointer;color:white}.b{background:#d84d4d}.p{background:#356ee8}.t{background:#34a853}.ghost{background:#252d42;color:#dbe5ff}.warn{background:#7b4fd6}
textarea,input{width:100%;background:#0a0f1d;color:#eaf0ff;border:1px solid #34405d;border-radius:12px;padding:12px;font-size:16px}textarea{min-height:90px;resize:vertical}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}.kpi{background:#0b1120;border:1px solid #25304a;border-radius:14px;padding:12px}.kpi .v{font-size:22px;font-weight:900}.kpi .l{color:#8f9ab7;font-size:12px;margin-top:4px}
.pred{display:flex;align-items:center;justify-content:space-between;gap:14px;background:#0a1020;border-radius:16px;padding:18px;border:1px solid #303b59}.dir{font-size:42px;font-weight:1000}.same{color:#ffcf5a}.switch{color:#7de1b1}.muted{color:#8f9ab7}
.bar{height:12px;background:#202840;border-radius:999px;overflow:hidden}.bar>i{display:block;height:100%;background:linear-gradient(90deg,#356ee8,#d84d4d)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px 8px;border-bottom:1px solid #25304a;text-align:left}th{color:#9aa6c4;position:sticky;top:0;background:#0f1423}.scroll{max-height:390px;overflow:auto;border:1px solid #25304a;border-radius:12px}
.badge{display:inline-block;border-radius:999px;padding:4px 8px;font-size:12px;font-weight:800;background:#222b40}.high{background:#6b2834}.medium{background:#665522}.low{background:#24513c}
.small{font-size:12px;color:#8f9ab7}.history-chips{display:flex;gap:5px;flex-wrap:wrap;margin:10px 0}.chip{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-size:12px;font-weight:900}.chip.B{background:#d84d4d}.chip.P{background:#356ee8}.chip.T{background:#34a853}
@media(max-width:850px){.grid{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body><div class="wrap">
<h1 class="title">BGS 本地決策測試台</h1>
<div class="sub">正式呼叫 predictor.predict()。按莊／閒／和輸入已開結果，畫面會即時算「下一手」並標記是否追上一手。</div>
<div class="grid">
<section class="card">
<label>牌路歷史（B=莊、P=閒、T=和）</label>
<textarea id="history" placeholder="例如：BPPBBP"></textarea>
<div id="chips" class="history-chips"></div>
<div class="controls">
<button class="btn b" onclick="appendOutcome('B')">莊 B</button>
<button class="btn p" onclick="appendOutcome('P')">閒 P</button>
<button class="btn t" onclick="appendOutcome('T')">和 T</button>
<button class="btn ghost" onclick="undoOne()">復原</button>
<button class="btn ghost" onclick="resetAll()">清空</button>
<button class="btn warn" onclick="runBacktest()">整段回測</button>
</div>
<label>測試本金</label><input id="bankroll" type="number" value="10000" min="0" step="100" />
<div style="height:12px"></div>
<div class="pred">
<div><div class="small">下一手方向</div><div id="direction" class="dir">—</div><div id="relation" class="muted">尚未輸入</div></div>
<div style="min-width:220px"><div class="small">P(B) / P(P)</div><div id="probText" style="font-weight:900;margin:5px 0">—</div><div class="bar"><i id="pbar" style="width:50%"></i></div></div>
</div>
<div class="kpis">
<div class="kpi"><div id="confidence" class="v">—</div><div class="l">Confidence</div></div>
<div class="kpi"><div id="run" class="v">—</div><div class="l">目前連段</div></div>
<div class="kpi"><div id="sameVotes" class="v">—</div><div class="l">有效元件 SAME 票</div></div>
<div class="kpi"><div id="risk" class="v">—</div><div class="l">追單風險</div></div>
</div>
<div class="small" id="modelInfo"></div>
</section>
<section class="card">
<h3 style="margin-top:0">Road 元件診斷</h3>
<div class="scroll"><table><thead><tr><th>元件</th><th>方向</th><th>對上一手</th><th>P(B)</th><th>可靠度</th><th>有效權重</th></tr></thead><tbody id="components"></tbody></table></div>
<div style="height:14px"></div>
<div class="small">判讀重點：如果 multi_window、pattern_replay、ngram、pattern_survival 同時大量顯示 SAME，代表它們是高度相關的延續訊號，不等於四個獨立模型都支持同一邊。</div>
</section>
</div>
<section class="card" style="margin-top:16px">
<h3 style="margin-top:0">整段回測</h3>
<div class="kpis">
<div class="kpi"><div id="btAcc" class="v">—</div><div class="l">命中率（和局略過）</div></div>
<div class="kpi"><div id="btFollow" class="v">—</div><div class="l">預測跟上一手同邊</div></div>
<div class="kpi"><div id="btRun3" class="v">—</div><div class="l">連 3+ 時繼續追同邊</div></div>
<div class="kpi"><div id="btSamples" class="v">—</div><div class="l">有效樣本</div></div>
</div>
<div class="scroll"><table><thead><tr><th>局</th><th>上一手</th><th>預測</th><th>關係</th><th>實際</th><th>連段</th><th>信心</th><th>結果</th></tr></thead><tbody id="btRows"></tbody></table></div>
</section>
</div>
<script>
const sessionId=(crypto.randomUUID?crypto.randomUUID():String(Date.now())).replaceAll('-','').slice(0,12);
const $=id=>document.getElementById(id); const pct=v=>v==null?'—':(v*100).toFixed(1)+'%';
function clean(){const v=$('history').value.toUpperCase().replace(/[^BPT]/g,'');$('history').value=v;return v}
function renderChips(){const h=clean();$('chips').innerHTML=[...h].map(x=>`<span class="chip ${x}">${x==='B'?'莊':x==='P'?'閒':'和'}</span>`).join('')}
async function predict(){renderChips();const history=clean();const bankroll=Number($('bankroll').value||0);try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({history,bankroll,session_id:sessionId})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'predict failed');
$('direction').textContent=d.direction_text+' '+d.direction;$('direction').style.color=d.direction==='B'?'#ff7777':'#74a0ff';$('relation').textContent=d.relation_to_last==='SAME'?'⚠ 跟上一手同邊（SAME）':d.relation_to_last==='SWITCH'?'↔ 跟上一手反邊（SWITCH）':'冷啟動';$('relation').className=d.relation_to_last==='SAME'?'same':'switch';$('probText').textContent=`莊 ${(d.banker_probability*100).toFixed(2)}% / 閒 ${(d.player_probability*100).toFixed(2)}%`;$('pbar').style.width=(d.banker_probability*100)+'%';$('confidence').textContent=pct(d.confidence);$('run').textContent=d.run_side?`${d.run_side==='B'?'莊':'閒'} × ${d.run_length}`:'—';$('sameVotes').textContent=`${d.same_component_votes}/${d.active_component_votes}`;$('risk').textContent=d.chase_risk;$('risk').className='v '+d.chase_risk.toLowerCase();$('modelInfo').textContent=`正式來源：${d.formal_direction_source} ｜ pattern=${d.pattern||'—'} ｜ rawEdge=${d.raw_edge.toFixed(4)} ｜ finalEdge=${d.final_edge.toFixed(4)} ｜ maturity=${d.maturity.toFixed(2)}`;
$('components').innerHTML=d.component_rows.map(x=>`<tr><td>${x.name}${x.pattern?'<br><span class="small">'+x.pattern+'</span>':''}</td><td>${x.direction_text}</td><td><span class="badge ${x.relation_to_last==='SAME'?'medium':'low'}">${x.relation_to_last}</span></td><td>${(x.p_b*100).toFixed(1)}%</td><td>${(x.reliability*100).toFixed(1)}%</td><td>${x.effective_weight.toFixed(4)}</td></tr>`).join('');}catch(e){alert(e.message)}}
function appendOutcome(x){$('history').value=clean()+x;predict()}function undoOne(){$('history').value=clean().slice(0,-1);predict()}function resetAll(){$('history').value='';$('btRows').innerHTML='';['btAcc','btFollow','btRun3','btSamples'].forEach(x=>$(x).textContent='—');predict()}
async function runBacktest(){renderChips();const history=clean();if(history.length<2){alert('至少輸入 2 手');return}const bankroll=Number($('bankroll').value||0);try{const r=await fetch('/api/backtest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({history,bankroll,session_id:sessionId})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'backtest failed');$('btAcc').textContent=pct(d.accuracy);$('btFollow').textContent=pct(d.follow_last_call_rate);$('btRun3').textContent=pct(d.run_ge3_follow_last_rate);$('btSamples').textContent=d.scored_samples;$('btRows').innerHTML=d.rows.map(x=>`<tr><td>${x.round}</td><td>${x.last_side||'—'}</td><td>${x.predicted_text}</td><td><span class="badge ${x.relation_to_last==='SAME'?'medium':'low'}">${x.relation_to_last}</span></td><td>${x.actual_text}</td><td>${x.run_length}</td><td>${pct(x.confidence)}</td><td>${x.verdict}</td></tr>`).join('');}catch(e){alert(e.message)}}
$('history').addEventListener('input',()=>{renderChips()});$('bankroll').addEventListener('change',predict);predict();
</script></body></html>'''
