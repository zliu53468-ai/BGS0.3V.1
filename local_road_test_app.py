"""Local browser tester for BGS road recognition + LinUCB prediction.

Run:
    uvicorn local_road_test_app:app --host 127.0.0.1 --port 8765

This process keeps a separate local LinUCB state file under the OS temp folder,
so browser experiments do not reuse or overwrite the production state file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import os
import tempfile
import uuid

# Isolate the local browser tester before importing predictor/contextual_bandit.
_LOCAL_DIR = Path(tempfile.gettempdir()) / "bgs_local_road_tester"
_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LINUCB_STATE_FILE", str(_LOCAL_DIR / "contextual_linucb_state.json"))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from road_detector import detect_road_sequence_detailed
from screenshot_predictor import predict_from_screenshot

app = FastAPI(title="BGS Local Road Tester", version="1.0.0")


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _prediction_summary(prediction: Dict[str, Any]) -> Dict[str, Any]:
    money = dict(prediction.get("money_management") or {})
    return {
        "direction": prediction.get("direction") or prediction.get("action") or prediction.get("recommend"),
        "direction_text": prediction.get("direction_text") or prediction.get("action_text") or prediction.get("recommend_text"),
        "confidence": prediction.get("confidence"),
        "exploit_confidence": prediction.get("exploit_confidence", prediction.get("confidence")),
        "exploit_probabilities": prediction.get("exploit_probabilities", prediction.get("probabilities")),
        "direction_probabilities_ucb": prediction.get("direction_probabilities_ucb", prediction.get("raw_direction_probabilities")),
        "mean_scores": prediction.get("mean_scores") or dict(prediction.get("dynamic_prediction_policy") or {}).get("mean_scores"),
        "score_gap": prediction.get("score_gap") or dict(prediction.get("dynamic_prediction_policy") or {}).get("score_gap"),
        "pure_ev": prediction.get("pure_ev", money.get("pure_ev", money.get("virtual_ev"))),
        "bet_percentage": prediction.get("bet_percentage"),
        "bet_amount": prediction.get("bet_amount"),
        "bet_allowed": prediction.get("bet_allowed"),
        "formal_direction_source": prediction.get("formal_direction_source"),
        "probability_semantics": prediction.get("probability_semantics"),
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BGS 本地牌路測試器</title>
<style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif;color:#eceff4;background:#10131a}
body{margin:0;padding:24px;background:linear-gradient(180deg,#10131a,#171c26);min-height:100vh}
.wrap{max-width:1100px;margin:auto}.card{background:#1d2430;border:1px solid #313a49;border-radius:18px;padding:18px;margin-bottom:16px;box-shadow:0 12px 36px #0005}
h1{margin:0 0 8px;font-size:28px}.sub{color:#9eabc0;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
label{display:block;color:#b8c1d1;font-size:13px;margin-bottom:6px}input,select,button{width:100%;box-sizing:border-box;border-radius:10px;border:1px solid #3a4558;background:#111722;color:#fff;padding:11px}
button{background:#2f6feb;border:0;font-weight:700;cursor:pointer}.secondary{background:#394356}.drop{border:2px dashed #4a5870;border-radius:14px;padding:18px;text-align:center;margin:14px 0}img{max-width:100%;max-height:420px;border-radius:12px;display:none;margin:auto}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.kpi{background:#121925;border-radius:12px;padding:12px}.kpi b{display:block;font-size:20px;margin-top:6px}.good{color:#66d9a3}.warn{color:#ffd166}.bad{color:#ff7b7b}
pre{white-space:pre-wrap;word-break:break-word;background:#0c1119;border-radius:12px;padding:14px;max-height:520px;overflow:auto;font-size:12px}.seq{font-family:ui-monospace,Menlo,monospace;word-break:break-all;font-size:18px;line-height:1.7}
.row{display:flex;gap:10px}.row>*{flex:1}@media(max-width:700px){body{padding:12px}.row{display:block}.row>*{margin-bottom:8px}}
</style>
</head>
<body><div class="wrap">
<div class="card"><h1>BGS 本地牌路圖預測測試器</h1><div class="sub">上傳完整桌面截圖或只截大路。先跑大路辨識，再把可信 B/P/T 序列送進目前 Single-Brain LinUCB 模型。這個網站使用獨立的本地狀態檔，不會覆蓋正式 production brain。</div>
<form id="form"><div class="grid">
<div><label>館別</label><select name="venue"><option>DG</option><option>MT</option><option>DB</option><option>SA</option><option>OB</option><option>T9</option><option value="">自動</option></select></div>
<div><label>輸入類型</label><select name="input_type"><option value="auto">自動判斷</option><option value="full_screen">完整畫面</option><option value="road_crop">只截大路</option><option value="wide_multi_road">多路橫向裁圖</option></select></div>
<div><label>本金</label><input name="bankroll" type="number" value="10000" min="0"></div>
<div><label>剩餘牌數（不知道可留 416）</label><input name="remaining_cards" type="number" value="416" min="1"></div>
<div><label>桌號</label><input name="room" value="local"></div>
</div>
<div class="drop"><input id="file" name="file" type="file" accept="image/*" required><p>可直接選擇 PNG / JPG 截圖</p><img id="preview"></div>
<div class="row"><button type="submit">開始辨識＋預測</button><button type="button" class="secondary" id="reset">清除本地測試腦</button></div></form></div>
<div class="card" id="summary" style="display:none"><div class="kpis" id="kpis"></div><h3>辨識序列</h3><div class="seq" id="seq"></div></div>
<div class="card" id="details" style="display:none"><h3>完整診斷</h3><pre id="json"></pre></div>
</div>
<script>
const form=document.getElementById('form'), file=document.getElementById('file'), preview=document.getElementById('preview');
file.onchange=()=>{const f=file.files[0]; if(!f)return; preview.src=URL.createObjectURL(f); preview.style.display='block'};
const fmt=v=>v===null||v===undefined?'—':(typeof v==='number'?Math.round(v*100000)/100000:v);
form.onsubmit=async(e)=>{e.preventDefault(); const btn=form.querySelector('button[type=submit]');btn.disabled=true;btn.textContent='分析中…';
 try{const r=await fetch('/api/test-image',{method:'POST',body:new FormData(form)}); const data=await r.json();
 document.getElementById('summary').style.display='block'; document.getElementById('details').style.display='block'; document.getElementById('json').textContent=JSON.stringify(data,null,2);
 const road=data.road||{}, p=data.prediction||{}; const ok=!!road.quality_ok; document.getElementById('seq').textContent=(road.raw_outcomes||road.sequence||[]).join(' ');
 document.getElementById('kpis').innerHTML=`<div class="kpi">大路品質<b class="${ok?'good':'bad'}">${ok?'通過':'失敗'}</b></div><div class="kpi">辨識顆數<b>${fmt(road.recognized_count)}</b></div><div class="kpi">區域<b>${road.selected_region||'—'}</b></div><div class="kpi">AutoLocate<b class="${road.autolocate_fallback_used?'warn':''}">${road.autolocate_fallback_used?'有啟用':'未啟用'}</b></div><div class="kpi">方向<b>${p.direction_text||p.direction||'—'}</b></div><div class="kpi">純期望信心<b>${p.exploit_confidence===undefined?'—':(Number(p.exploit_confidence)*100).toFixed(2)+'%'}</b></div><div class="kpi">Pure EV<b>${p.pure_ev===undefined?'—':(Number(p.pure_ev)*100).toFixed(3)+'%'}</b></div><div class="kpi">下注比例<b>${p.bet_percentage===undefined?'—':Number(p.bet_percentage).toFixed(2)+'%'}</b></div>`;
 }catch(err){alert(String(err))}finally{btn.disabled=false;btn.textContent='開始辨識＋預測'}};
document.getElementById('reset').onclick=async()=>{const r=await fetch('/api/reset-local-state',{method:'POST'});const d=await r.json();alert(d.ok?'本地測試狀態已清除':'清除失敗')};
</script></body></html>"""


@app.post("/api/test-image")
async def test_image(
    file: UploadFile = File(...),
    venue: str = Form("DG"),
    room: str = Form("local"),
    input_type: str = Form("auto"),
    bankroll: float = Form(10000.0),
    remaining_cards: int = Form(416),
) -> JSONResponse:
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        suffix = ".png"
    temp_path = _LOCAL_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    payload = await file.read()
    if not payload:
        return JSONResponse({"ok": False, "error": "empty_upload"}, status_code=400)
    if len(payload) > 20_000_000:
        return JSONResponse({"ok": False, "error": "image_too_large"}, status_code=413)
    temp_path.write_bytes(payload)

    try:
        road = dict(
            detect_road_sequence_detailed(
                str(temp_path),
                venue=str(venue or ""),
                input_type=str(input_type or "auto"),
            )
            or {}
        )
        road_ok = bool(road.get("ok")) and bool(road.get("quality_ok", True)) and bool(road.get("sequence"))
        prediction_summary: Dict[str, Any] = {}
        prediction_raw: Dict[str, Any] = {}
        if road_ok:
            raw_outcomes = list(road.get("raw_outcomes") or road.get("sequence") or [])
            prediction_raw = dict(
                predict_from_screenshot(
                    list(road.get("sequence") or []),
                    remaining_cards=max(1, int(remaining_cards or 416)),
                    raw_outcomes=raw_outcomes,
                    tie_markers=dict(road.get("tie_markers") or {}),
                    shoe_context={"bankroll": max(0.0, _safe_float(bankroll, 10000.0))},
                    venue=str(venue or ""),
                    room=str(room or "local"),
                    shoe_id="local-browser-test",
                    user_id="local-browser-test",
                    road_context=road,
                    screen_metadata={"input_type": str(input_type or "auto"), "local_tester": True},
                    initial_grid_cells=list(road.get("grid_cells") or []),
                    initial_image_history=raw_outcomes,
                    manual_outcome_history=[],
                    record_for_learning=False,
                )
                or {}
            )
            prediction_summary = _prediction_summary(prediction_raw)

        road_view = {
            "ok": road.get("ok"),
            "quality_ok": road.get("quality_ok"),
            "sequence": road.get("sequence"),
            "raw_outcomes": road.get("raw_outcomes"),
            "tie_markers": road.get("tie_markers"),
            "recognized_count": road.get("recognized_count"),
            "uncertain_count": road.get("uncertain_count", road.get("unknown_candidates")),
            "reconstructed_all": road.get("reconstructed_all"),
            "fallback_reason": road.get("fallback_reason"),
            "selected_region": road.get("selected_region"),
            "layout_profile": road.get("layout_profile"),
            "input_type": road.get("input_type"),
            "autolocate_fallback_used": road.get("autolocate_fallback_used", False),
            "autolocate_primary_fallback_reason": road.get("autolocate_primary_fallback_reason"),
            "autolocate_candidates": road.get("autolocate_candidates", []),
            "candidate_regions": road.get("candidate_regions", []),
            "effective_grid": road.get("effective_grid", {}),
            "median_cell_confidence": road.get("median_cell_confidence"),
            "errors": road.get("errors", []),
        }
        return JSONResponse(
            {
                "ok": road_ok,
                "road": road_view,
                "prediction": prediction_summary,
                "prediction_raw": prediction_raw if os.getenv("LOCAL_TESTER_FULL_PREDICTION", "0") == "1" else {},
                "local_state_file": str(_LOCAL_DIR / "contextual_linucb_state.json"),
                "production_state_isolated": True,
            }
        )
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/api/reset-local-state")
def reset_local_state() -> Dict[str, Any]:
    state_file = _LOCAL_DIR / "contextual_linucb_state.json"
    try:
        state_file.unlink(missing_ok=True)
        return {"ok": True, "state_file": str(state_file)}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "state_file": str(state_file)}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "bgs-local-road-tester",
        "local_state_file": str(_LOCAL_DIR / "contextual_linucb_state.json"),
        "production_state_isolated": True,
    }
