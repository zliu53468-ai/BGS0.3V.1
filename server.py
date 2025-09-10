#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGS Server (Dashboard + Dual-key predict)
- Flask API
- Optional LINE webhook (enabled only if LINE env vars are set)
- Seed ingest + synthetic training + model-first predict
- 30-min trial timer per user (memory-based; single worker recommended)
- Graphical dashboard (/) and /health

Env:
  LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN  # optional
  TRIAL_MINUTES=30
  ADMIN_UIDS=Uxxxxxxxx,Uyyyyyyyy  # optional for TRAIN/STATUS guard
  LOGIN_CODE=123456                # optional demo login code
  DATA_ROOT=/opt/data              # optional; default "."
  PORT=8000                        # local run
"""
from __future__ import annotations

import os
import json
import time
import random
from typing import List, Dict, Any

from pathlib import Path as _Path
from flask import Flask, request, jsonify, abort

app = Flask(__name__)
try:
    from flask_cors import CORS
    CORS(app)
except Exception:
    pass

# ---------- Trial settings ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
_trial_sessions: Dict[str, Dict[str, Any]] = {}  # userId -> {"start": epoch, "warned": False, "unlocked": False}

def trial_touch(user_id: str) -> Dict[str, Any]:
    s = _trial_sessions.get(user_id)
    if not s:
        s = {"start": time.time(), "warned": False, "unlocked": False}
        _trial_sessions[user_id] = s
    return s

def trial_check_and_maybe_warn(user_id: str, reply_fn):
    s = trial_touch(user_id)
    if s.get("unlocked"):
        return
    elapsed_min = (time.time() - s["start"]) / 60.0
    if elapsed_min >= TRIAL_MINUTES and not s.get("warned"):
        s["warned"] = True
        reply_fn(f"⏰ 試用已超過 {TRIAL_MINUTES} 分鐘，請完成登入或續期。若已完成登入但仍看到此訊息，輸入：RESET TRIAL")

# ---------- Storage paths ----------
ROOT = _Path(os.getenv("DATA_ROOT", "."))
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)

SEED_CSV   = DATA / "seed.csv"
SIM_ROWS   = DATA / "sim_rows.csv"
MODEL_PATH = MODELS / "baseline.joblib"
PRIORS_JSON= REPORTS / "priors.json"

# ---------- Heuristic baseline ----------
OUTMAP = {"B":0, "P":1, "T":2}
INV_OUTMAP = {0:"B", 1:"P", 2:"T"}

def parse_text_seq(s: str) -> List[str]:
    toks = [t.strip().upper() for t in s.replace(",", " ").split() if t.strip().upper() in ("B","P","T")]
    return toks

def estimate_probs(seq: List[str]):
    k = min(12, len(seq)) or 1
    win = seq[-k:]
    cB = win.count("B") + 0.5
    cP = win.count("P") + 0.5
    cT = win.count("T") + 0.5
    s = cB + cP + cT
    probs = (cB/s, cP/s, cT/s)
    detail = {"window": k, "counts": {"B": cB, "P": cP, "T": cT}}
    return probs, detail

# ---------- Hot-train (synth + train) ----------
import threading
_hot_lock = threading.Lock()
_hot_training = False
_hot_last_metrics: Dict[str, Any] | None = None

try:
    import joblib
except Exception:
    joblib = None

def _append_seed_history(history: str) -> int:
    import pandas as pd
    toks = parse_text_seq(history)
    if len(toks) < 6:
        raise ValueError("history 長度至少 6")
    df_new = pd.DataFrame({"history":[" ".join(toks)]})
    if SEED_CSV.exists():
        df = pd.read_csv(SEED_CSV)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(SEED_CSV, index=False)
    return len(df)

def _read_seed_histories() -> List[List[str]]:
    import pandas as pd
    if not SEED_CSV.exists(): return []
    df = pd.read_csv(SEED_CSV)
    out = []
    for s in df["history"].astype(str).tolist():
        toks = parse_text_seq(s)
        if len(toks) >= 6:
            out.append(toks)
    return out

def _estimate_ngram(seqs: List[List[str]], order:int=2, laplace:float=0.5):
    from collections import defaultdict, Counter
    counts = defaultdict(Counter)
    for seq in seqs:
        if len(seq) <= order: continue
        for i in range(order, len(seq)):
            ctx = tuple(seq[i-order:i]); nxt = seq[i]
            counts[ctx][nxt] += 1
    vocab = ["B","P","T"]
    trans = {}
    for ctx, ctr in counts.items():
        total = sum(ctr.values()) + laplace*len(vocab)
        trans[ctx] = {v:(ctr[v]+laplace)/total for v in vocab}
    return trans

def _style_adjust(probs: Dict[str,float], last: str|None, style:str='hybrid', long_strength:float=0.5, jumpy_strength:float=0.5, tie_rate:float=0.06):
    pB, pP, pT = probs.get('B',1/3), probs.get('P',1/3), probs.get('T',1/3)
    pT = 0.85*pT + 0.15*tie_rate
    remain = max(1e-9, 1 - pT)
    s = max(pB+pP, 1e-12)
    pB, pP = remain*(pB/s), remain*(pP/s)
    if last in ("B","P"):
        if style in ("long","hybrid"):
            if last=="B": pB += 0.2*long_strength
            else: pP += 0.2*long_strength
        if style in ("jumpy","hybrid"):
            if last=="B": pP += 0.2*jumpy_strength
            else: pB += 0.2*jumpy_strength
    tot = pB+pP+pT
    if tot <= 0: return {"B":1/3,"P":1/3,"T":1/3}
    return {"B":pB/tot,"P":pP/tot,"T":pT/tot}

def _sample_next(probs: Dict[str,float]) -> str:
    r = random.random(); acc = 0.0
    for k in ("B","P","T"):
        acc += probs[k]
        if r <= acc: return k
    return "T"

def _gen_sequences(trans:dict, order:int, n_seq:int, min_len:int, max_len:int, style:str, long_strength:float, jumpy_strength:float, tie_rate:float):
    contexts = list(trans.keys())
    if not contexts: raise ValueError("轉移為空，seed 不足或 order 過大")
    seqs = []
    for _ in range(n_seq):
        cur = list(random.choice(contexts))
        L = random.randint(min_len, max_len)
        last = cur[-1] if cur else None
        while len(cur) < L:
            probs = trans.get(tuple(cur[-order:]), {"B":1/3,"P":1/3,"T":1/3})
            probs = _style_adjust(probs, last, style, 0.5, 0.5, tie_rate)
            nxt = _sample_next(probs)
            cur.append(nxt); last = nxt
        seqs.append(cur)
    return seqs

def _expand_rows(seqs: List[List[str]], max_history:int=12):
    import pandas as _pd
    rows = []
    for sid, seq in enumerate(seqs):
        streak = 0
        for i in range(len(seq)-1):
            cur = seq[:i+1]; nxt = seq[i+1]
            streak = 1 if i==0 else (streak+1 if cur[-1]==cur[-2] else 1)
            k = min(max_history, len(cur)); win = cur[-k:]
            wB=win.count('B')/k; wP=win.count('P')/k; wT=win.count('T')/k
            switches = sum(1 for j in range(1,len(win)) if win[j]!=win[j-1])
            osc = switches/(len(win)-1) if len(win)>1 else 0.0
            ctx1=cur[-1]; ctx2=''.join(cur[-2:]) if i>=1 else '_'+ctx1; ctx3=''.join(cur[-3:]) if i>=2 else '_'+ctx2
            rows.append({'seq_id':sid,'step':i,'streak':streak,'wB':wB,'wP':wP,'wT':wT,'osc':osc,'last':ctx1,'ctx1':ctx1,'ctx2':ctx2,'ctx3':ctx3,'y':OUTMAP[nxt]})
    df = _pd.DataFrame(rows)
    for col in ['last','ctx1','ctx2','ctx3']:
        d = _pd.get_dummies(df[col], prefix=col)
        df = _pd.concat([df.drop(columns=[col]), d], axis=1)
    return df

def _train_baseline(df, valid_ratio:float=0.1):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['y']).values; y = df['y'].values
    Xtr, Xva, ytr, yva = train_test_split(X,y,test_size=valid_ratio,random_state=42,stratify=y)

    model_name = "LogisticRegression"
    try:
        import lightgbm as lgb  # optional
        clf = lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=63,subsample=0.9,colsample_bytree=0.9,random_state=42)
        model_name = "LightGBM"
    except Exception:
        try:
            import xgboost as xgb
            clf = xgb.XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,objective='multi:softprob',num_class=3,random_state=42,tree_method='hist')
            model_name = "XGBoost"
        except Exception:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=200, multi_class='multinomial')
            model_name = "LogisticRegression"
    clf.fit(Xtr,ytr)
    from sklearn.metrics import log_loss
    import numpy as np
    pva = clf.predict_proba(Xva)
    acc = float((pva.argmax(1)==yva).mean())
    ll  = float(log_loss(yva,pva))
    if joblib is not None:
        try: joblib.dump(clf, MODEL_PATH)
        except Exception: pass
    return clf, {"valid_acc":acc,"logloss":ll,"model":model_name,"rows":int(len(df))}

def _synth_and_train(target_rows:int=300_000, order:int=2, style:str='hybrid', tie_rate:float=0.06, random_seed:int=2025):
    import numpy as np
    np.random.seed(random_seed); random.seed(random_seed)
    seqs = _read_seed_histories()
    if not seqs: raise ValueError("沒有 seed，請先 /ingest-seed 或在 LINE 打：SEED: <B/P/T 串>")
    trans = _estimate_ngram(seqs, order=order, laplace=0.5)
    sim: List[List[str]] = []; rows_est = 0
    while rows_est < target_rows:
        batch = _gen_sequences(trans, order, n_seq=200, min_len=60, max_len=120, style=style, long_strength=0.5, jumpy_strength=0.5, tie_rate=tie_rate)
        sim.extend(batch)
        rows_est = sum(max(0,len(s)-1) for s in sim)
    df = _expand_rows(sim, max_history=12)
    if len(df) > target_rows:
        df = df.sample(n=target_rows, random_state=random_seed).sort_index()
    df.to_csv(SIM_ROWS, index=False)
    _, metrics = _train_baseline(df, valid_ratio=0.1)
    with open(PRIORS_JSON,'w',encoding='utf-8') as f:
        json.dump({"order":order,"style":style,"tie_rate":tie_rate,"target_rows":target_rows, **metrics}, f, ensure_ascii=False, indent=2)
    return metrics

def _predict_next_with_model(history: str):
    if not (MODEL_PATH.exists() and joblib is not None):
        return None
    try:
        clf = joblib.load(MODEL_PATH)
    except Exception:
        return None
    import pandas as _pd
    toks = parse_text_seq(history)
    if len(toks) < 3: return None
    cur = toks; i = len(cur)-1
    k = min(12, len(cur)); win = cur[-k:]
    wB=win.count('B')/k; wP=win.count('P')/k; wT=win.count('T')/k
    switches = sum(1 for j in range(1,len(win)) if win[j]!=win[j-1])
    osc = switches/(len(win)-1) if len(win)>1 else 0.0
    ctx1=cur[-1]; ctx2=''.join(cur[-2:]) if i>=1 else '_'+ctx1; ctx3=''.join(cur[-3:]) if i>=2 else '_'+ctx2
    row={'seq_id':0,'step':i,'streak':1,'wB':wB,'wP':wP,'wT':wT,'osc':osc,'last':ctx1,'ctx1':ctx1,'ctx2':ctx2,'ctx3':ctx3}
    df=_pd.DataFrame([row])
    for col in ['last','ctx1','ctx2','ctx3']:
        d=_pd.get_dummies(df[col], prefix=col); df=_pd.concat([df.drop(columns=[col]), d], axis=1)
    if SIM_ROWS.exists():
        ref_cols=_pd.read_csv(SIM_ROWS, nrows=1).drop(columns=['y']).columns.tolist()
        for c in ref_cols:
            if c not in df.columns: df[c]=0
        df=df[ref_cols]
    proba = clf.predict_proba(df.values)[0]
    return {'B':float(proba[0]),'P':float(proba[1]),'T':float(proba[2])}

# ---------- REST ----------
@app.post("/ingest-seed")
def ingest_seed():
    data = request.get_json(silent=True) or {}
    history = str(data.get("history","")).strip()
    if not history:
        return jsonify({"ok":False,"msg":"history 必填"}), 400
    try:
        n = _append_seed_history(history)
        return jsonify({"ok":True,"seed_records": n}), 200
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 400

def _bg_train_hot(target_rows:int, style:str, tie_rate:float):
    global _hot_training, _hot_last_metrics
    try:
        with _hot_lock:
            _hot_training = True
        m = _synth_and_train(target_rows=target_rows, style=style, tie_rate=tie_rate)
        _hot_last_metrics = m
    except Exception as e:
        _hot_last_metrics = {"error": str(e)}
    finally:
        with _hot_lock:
            _hot_training = False

@app.post("/synth-train")
def synth_train():
    data = request.get_json(silent=True) or {}
    target_rows = int(data.get("target_rows", 300000))
    style = str(data.get("style","hybrid"))
    tie_rate = float(data.get("tie_rate", 0.06))
    with _hot_lock:
        if _hot_training:
            return jsonify({"ok":False,"msg":"training in progress"}), 409
    t = threading.Thread(target=_bg_train_hot, args=(target_rows, style, tie_rate), daemon=True)
    t.start()
    return jsonify({"ok":True,"msg":"started"}), 200

@app.post("/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    seq_text = str(data.get("history",""))
    model_proba = _predict_next_with_model(seq_text)
    if model_proba is not None:
        probs = model_proba
        top_label = max(probs, key=probs.get)
        resp = {
            "ok": True,
            "source": "model",
            "probabilities": probs,  # 舊前端鍵名
            "probs": probs,          # 新前端兼容
            "top": top_label,
            "label": top_label
        }
        return jsonify(resp), 200
    seq = parse_text_seq(seq_text)
    p, detail = estimate_probs(seq)
    probs = {"B":p[0],"P":p[1],"T":p[2]}
    top_label = max(probs, key=probs.get)
    resp = {
        "ok": True,
        "source": "heuristic",
        "probabilities": probs,
        "probs": probs,
        "top": top_label,
        "label": top_label,
        "detail": detail
    }
    return jsonify(resp), 200

# ---------- Dashboard & Health ----------
@app.get("/health")
def health():
    return "OK", 200

@app.get("/")
def home():
    seed_n = 0
    if SEED_CSV.exists():
        try:
            import pandas as _pd
            seed_n = len(_pd.read_csv(SEED_CSV))
        except Exception:
            seed_n = 0
    status = {
        "seed_records": seed_n,
        "model_exists": MODEL_PATH.exists(),
        "is_training": _hot_training,
        "last_metrics": _hot_last_metrics,
        "webhook": "/line-webhook",
        "health": "/health",
    }
    accept = request.headers.get("Accept","")
    if "application/json" in accept:
        return jsonify(status)
    # Graphical dashboard (pure HTML+JS)
    return f"""<!doctype html>
<html lang="zh-Hant"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BGS Dashboard</title>
<style>
  body{{font-family:ui-sans-serif,system-ui;background:#0d1117;color:#c9d1d9;padding:24px}}
  .card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;margin:12px 0}}
  .row{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
  .grid{{display:grid;gap:12px}}
  .grid-2{{grid-template-columns:1fr 1fr}}
  input,textarea,select,button{{background:#0b1220;color:#e5e7eb;border:1px solid #374151;border-radius:8px;padding:8px}}
  button{{cursor:pointer}}
  pre{{white-space:pre-wrap;word-break:break-word;background:#0b1220;padding:12px;border-radius:8px}}
  .muted{{color:#94a3b8}}
</style></head>
<body>
  <h2>BGS Dashboard</h2>
  <div class="card">
    <div class="row">
      <button onclick="refreshStatus()">重新整理狀態</button>
      <span class="muted">Webhook：<code>/line-webhook</code></span>
      <span class="muted">健康檢查：<code>/health</code></span>
    </div>
    <pre id="statusBox">載入中…</pre>
  </div>

  <div class="grid grid-2">
    <div class="card">
      <h3>即時預測 /predict</h3>
      <div class="row">
        <input id="predInput" placeholder="例如：B P P T B" style="flex:1"/>
        <button onclick="doPredict()">送出</button>
      </div>
      <pre id="predBox"></pre>
    </div>
    <div class="card">
      <h3>追加種子 /ingest-seed</h3>
      <textarea id="seedInput" rows="4" placeholder="貼上真實歷史：B P B P T B …"></textarea>
      <div class="row"><button onclick="doSeed()">追加</button></div>
      <pre id="seedBox"></pre>
    </div>
  </div>

  <div class="card">
    <h3>啟動訓練 /synth-train</h3>
    <div class="row">
      <label>rows</label><input id="rows" type="number" value="300000" style="width:160px"/>
      <label>style</label><select id="style"><option>hybrid</option><option>jumpy</option><option>long</option></select>
      <label>tie_rate</label><input id="tie" type="number" step="0.01" value="0.06" style="width:120px"/>
      <button onclick="doTrain()">開始訓練</button>
    </div>
    <pre class="muted">提示：免費方案先以 120000～200000 測試。</pre>
    <pre id="trainBox"></pre>
  </div>

<script>
async function refreshStatus(){try{const r=await fetch(location.origin + "/",{headers:{"Accept":"application/json"}});const j=await r.json();document.getElementById("statusBox").textContent=JSON.stringify(j,null,2);}catch(e){document.getElementById("statusBox").textContent="讀取失敗："+e;}}
async function doPredict(){const history=document.getElementById("predInput").value.trim();const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({history})});const j=await r.json();document.getElementById("predBox").textContent=JSON.stringify(j,null,2);}
async function doSeed(){const history=document.getElementById("seedInput").value.trim();const r=await fetch("/ingest-seed",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({history})});const j=await r.json();document.getElementById("seedBox").textContent=JSON.stringify(j,null,2);refreshStatus();}
async function doTrain(){const target_rows=parseInt(document.getElementById("rows").value||"300000",10);const style=document.getElementById("style").value;const tie_rate=parseFloat(document.getElementById("tie").value||"0.06");const r=await fetch("/synth-train",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({target_rows,style,tie_rate})});const j=await r.json();document.getElementById("trainBox").textContent=JSON.stringify(j,null,2);refreshStatus();}
refreshStatus();
</script>
</body></html>"""

# ---------- Optional LINE webhook ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
ADMIN_UIDS = [x.strip() for x in os.getenv("ADMIN_UIDS","").split(",") if x.strip()]

_line_enabled = bool(LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN)
if _line_enabled:
    from linebot import LineBotApi, WebhookParser
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
    from linebot.exceptions import InvalidSignatureError

    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    parser = WebhookParser(LINE_CHANNEL_SECRET)

    def reply_or_push(event, message: TextSendMessage | str):
        if isinstance(message, str):
            message = TextSendMessage(text=message)
        try:
            if getattr(event, "reply_token", None):
                line_bot_api.reply_message(event.reply_token, message)
            else:
                uid = getattr(event.source, "user_id", None)
                if uid:
                    line_bot_api.push_message(uid, message)
        except Exception as e:
            print("LINE send error:", e, flush=True)

    @app.post("/line-webhook")
    def line_webhook():
        sig = request.headers.get("X-Line-Signature", "")
        body = request.get_data(as_text=True)
        try:
            events = parser.parse(body, sig)
        except InvalidSignatureError:
            abort(400)
        for event in events:
            if isinstance(event, MessageEvent) and isinstance(event.message, TextMessage):
                on_text(event)
        return "OK"

    def on_text(event):
        text = (event.message.text or "").strip()
        user_id = getattr(event.source, "user_id", "unknown")

        def _reply(msg: str):
            reply_or_push(event, TextSendMessage(text=msg))

        # Trial check (per message)
        trial_check_and_maybe_warn(user_id, _reply)

        up = text.upper()

        # Maintenance commands
        if up == "RESET TRIAL":
            _trial_sessions.pop(user_id, None)
            _reply("✅ 已重置你的試用倒數與解鎖狀態。")
            return
        if up.startswith("LOGIN "):
            code = text.split(" ",1)[1].strip()
            ok_code = os.getenv("LOGIN_CODE", "123456")
            if code == ok_code:
                s = trial_touch(user_id); s["unlocked"] = True
                _reply("✅ 登入成功，已解除試用限制。")
            else:
                _reply("❌ 登入碼錯誤")
            return

        def is_admin(uid: str) -> bool:
            return (uid in ADMIN_UIDS) or (not ADMIN_UIDS)  # 沒設定 ADMIN_UIDS 則不限制

        # Hot-train commands
        if up.startswith("SEED:"):
            history = text.split(":",1)[1]
            try:
                n = _append_seed_history(history)
                _reply(f"✅ 已追加 seed（共 {n} 筆）。可下：TRAIN 300000 hybrid 0.06")
            except Exception as e:
                _reply(f"❌ 追加失敗：{e}")
            return

        if up.startswith("TRAIN"):
            if not is_admin(user_id):
                _reply("⛔ 僅管理員可啟動訓練")
                return
            parts = text.split()
            target = int(parts[1]) if len(parts)>=2 else 300000
            style  = parts[2] if len(parts)>=3 else "hybrid"
            try:
                tie = float(parts[3]) if len(parts)>=4 else 0.06
            except Exception:
                tie = 0.06
            with _hot_lock:
                if _hot_training:
                    _reply("⚠️ 目前已有訓練在進行中")
                    return
            threading.Thread(target=_bg_train_hot, args=(target, style, tie), daemon=True).start()
            _reply(f"🚀 開始訓練：rows={target} style={style} tie={tie}")
            return

        if up.startswith("STATUS"):
            if not is_admin(user_id):
                _reply("⛔ 僅管理員可查詢訓練狀態")
                return
            if _hot_training:
                _reply("🔄 訓練中…")
            else:
                if _hot_last_metrics and "error" not in _hot_last_metrics:
                    _reply(f"✅ 最近模型：{_hot_last_metrics.get('model')} acc={_hot_last_metrics.get('valid_acc'):.3f} logloss={_hot_last_metrics.get('logloss'):.3f}")
                elif _hot_last_metrics and "error" in _hot_last_metrics:
                    _reply(f"❌ 上次訓練錯誤：{_hot_last_metrics['error']}")
                else:
                    _reply("ℹ️ 尚無訓練紀錄")
            return

        if up.startswith("PRED "):
            hist = text.split(" ",1)[1]
            model_proba = _predict_next_with_model(hist)
            if model_proba is not None:
                _reply(f"模型機率 → 莊 {model_proba['B']:.2f}｜閒 {model_proba['P']:.2f}｜和 {model_proba['T']:.2f}")
            else:
                seq_tmp = parse_text_seq(hist)
                p, _ = estimate_probs(seq_tmp)
                _reply(f"啟發式機率 → 莊 {p[0]:.2f}｜閒 {p[1]:.2f}｜和 {p[2]:.2f}")
            return

        _reply("指令：SEED: <路單> ｜ TRAIN <rows> [style] [tie] ｜ STATUS ｜ PRED <路單> ｜ LOGIN <code> ｜ RESET TRIAL")

else:
    def on_text(event):
        pass

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
