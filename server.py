
# -*- coding: utf-8 -*-
# BGS server (ensemble + MC prediction)
import os, json, re, random, time, threading
from pathlib import Path as _Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

from flask import Flask, request, jsonify, Response

# Optional CORS for dashboard
try:
    from flask_cors import CORS
except Exception:
    CORS = None

import numpy as np
import pandas as pd

# ML libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Optional boosters
_HAVE_LGBM = False
_HAVE_XGB  = False
try:
    from lightgbm import LGBMClassifier
    _HAVE_LGBM = True
except Exception:
    pass
try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:
    pass

# -------- storage roots with fallback --------
ROOT = _Path(os.getenv("DATA_ROOT", "."))
try:
    (ROOT/".probe").parent.mkdir(parents=True, exist_ok=True)
    (ROOT/".probe").write_text("ok", encoding="utf-8")
    (ROOT/".probe").unlink(missing_ok=True)
except Exception:
    ROOT = _Path("./storage")

DATA    = ROOT / "data";    DATA.mkdir(parents=True, exist_ok=True)
MODELS  = ROOT / "models";  MODELS.mkdir(parents=True, exist_ok=True)
REPORTS = ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)

SEED_FILE   = DATA / "seed.txt"
META_FILE   = MODELS / "meta.json"
LGBM_FILE   = MODELS / "lgbm.joblib"
LR_FILE     = MODELS / "lr.joblib"
XGB_FILE    = MODELS / "xgb.joblib"

app = Flask(__name__)
if CORS:
    CORS(app)

# ---- simple in-memory user state (bankroll/session) ----
_user_state = {}  # user_id -> {"bankroll": int, "in_session": bool, "hands": int}

def _get_state(uid: str):
    st = _user_state.get(uid) or {"bankroll": 5000, "in_session": False, "hands": 0}
    _user_state[uid] = st
    return st

def _fmt_money(n: float) -> str:
    try:
        return f"{int(round(n)):,.0f}"
    except Exception:
        return str(int(n))

# ------------- utils -------------
LABELS = ["B","P","T"]

def parse_text_seq(txt: str) -> List[str]:
    """Accept 'B P T', 'B,P,T', or 'BPT...' and normalize to list."""
    txt = (txt or "").strip().upper()
    if not txt:
        return []
    # If contains space/comma, split; else take only B/P/T chars
    if (" " in txt) or ("," in txt):
        tokens = re.split(r"[,\s]+", txt)
        seq = [t for t in tokens if t in LABELS]
    else:
        seq = [ch for ch in txt if ch in LABELS]
    return seq

def sliding_features(seq: List[str], k_context:int=3, win:int=12) -> Tuple[pd.DataFrame, List[str]]:
    """From a sequence of labels, create supervised rows to predict next label.
       Returns X (features) and y (next label).
    """
    rows = []
    y = []
    for i in range(max(k_context,1), len(seq)):
        prev = seq[max(0, i-k_context):i]
        # basic counts in a window of last win
        w = seq[max(0, i-win):i]
        cnt = Counter(w)
        wB = cnt.get("B",0)/max(1,len(w))
        wP = cnt.get("P",0)/max(1,len(w))
        wT = cnt.get("T",0)/max(1,len(w))
        # streak length of last symbol
        streak = 0
        for j in range(i-1, -1, -1):
            if seq[j]==seq[i-1]:
                streak += 1
            else:
                break
        # oscillation rate (changes / length)
        changes = sum(1 for a,b in zip(w, w[1:]) if a!=b)
        osc = changes / max(1,len(w)-1)
        # last 3 one-hots
        ctx = prev[-k_context:]
        row = {"wB":wB,"wP":wP,"wT":wT,"streak":streak,"osc":osc}
        for j,lab in enumerate(["ctx1","ctx2","ctx3"]):
            val = ctx[-(j+1)] if len(ctx)>j else "_"
            for L in LABELS+["_"]:
                row[f"{lab}_{L}"] = 1.0 if val==L else 0.0
        rows.append(row)
        y.append(seq[i])
    X = pd.DataFrame(rows)
    return X, y

def ngram_model(seq: List[str], n:int=2) -> Dict[Tuple[str,...], Dict[str,float]]:
    trans = defaultdict(Counter)
    for i in range(len(seq)-n):
        key = tuple(seq[i:i+n])
        nxt = seq[i+n]
        trans[key][nxt]+=1
    probs = {}
    for k,c in trans.items():
        s = sum(c.values())
        probs[k] = {lab:c.get(lab,0)/s for lab in LABELS}
    return probs

def sample_next_from_ngram(ctx: List[str], ng: Dict[Tuple[str,...], Dict[str,float]], n:int=2) -> str:
    key = tuple(ctx[-n:]) if len(ctx)>=n else None
    p = ng.get(key)
    if not p:
        # uniform fallback
        return random.choice(LABELS)
    labs = LABELS
    ps = np.array([p.get(L,0.0) for L in labs], dtype=float)
    if ps.sum()<=0: ps = np.ones_like(ps)/len(ps)
    ps = ps/ps.sum()
    return random.choices(labs, weights=ps, k=1)[0]

def jitter_numeric(X: pd.DataFrame, scale: float=0.02) -> pd.DataFrame:
    X = X.copy()
    num_cols = [c for c in X.columns if X[c].dtype!=object]
    for c in num_cols:
        X[c] = X[c].astype(float) + np.random.normal(0.0, scale, size=len(X))
    return X


# --------- global training state ---------
# Auto-train controls (env)
AUTO_TRAIN       = int(os.getenv("AUTO_TRAIN", "0"))          # 1=enable auto-train
TRAIN_ROWS_DEF   = int(os.getenv("TRAIN_ROWS", "100000"))
TRAIN_MODE_DEF   = os.getenv("TRAIN_MODE", "feature")
TRAIN_STYLE_DEF  = os.getenv("TRAIN_STYLE", "hybrid")
TRAIN_TIE_DEF    = float(os.getenv("TRAIN_TIE", "0.06"))
TRAIN_JITTER_DEF = float(os.getenv("TRAIN_JITTER", "0.02"))
TRAIN_WITH_XGB   = int(os.getenv("TRAIN_WITH_XGB", "0"))
MIN_SEED_FOR_TRAIN = int(os.getenv("TRAIN_MIN_SEED", "6"))
PRED_ENSEMBLE_DEF = os.getenv("PRED_ENSEMBLE", "light").lower()  # none|light|full
PRED_MC_DEF       = int(os.getenv("PRED_MC", "0"))

_last_seed_mtime = 0.0

_train_lock = threading.Lock()
_is_training = False
_last_metrics = None

def _save_meta(meta: dict):
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_meta() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    return {}

# --------- synth + train (ensemble) ---------
def _synth_from_seed(seed_seq: List[str], rows:int=100000, mode:str="feature", style:str="hybrid", tie_rate:float=0.06, jitter:float=0.02) -> Tuple[pd.DataFrame,List[str]]:
    """Return (X,y) synthesized dataset based on seed"""
    seed_seq = [s for s in seed_seq if s in LABELS]
    if len(seed_seq) < 6:
        raise ValueError("Seed sequence too short")
    # Expand features from seed
    X_seed, y_seed = sliding_features(seed_seq)
    if mode=="feature":
        # resample rows with replacement + jitter
        idx = np.random.choice(len(X_seed), size=min(rows, len(X_seed)), replace=True)
        X = X_seed.iloc[idx].reset_index(drop=True)
        y = [y_seed[i] for i in idx]
        X = jitter_numeric(X, scale=jitter)
        return X, y
    elif mode=="ngram":
        # use n-gram generator to simulate a long sequence
        n = 2
        ng = ngram_model(seed_seq, n=n)
        cur = seed_seq[:n]
        full = list(seed_seq[:])  # keep original too
        while len(full) < len(seed_seq) + rows//2:
            nxt = sample_next_from_ngram(full, ng, n=n)
            full.append(nxt)
        X, y = sliding_features(full)
        return X.iloc[-rows:].reset_index(drop=True), y[-rows:]
    else:
        # uniform baseline
        sim = [random.choice(LABELS) for _ in range(rows + 8)]
        X, y = sliding_features(sim)
        return X, y

def _train_ensemble(seed_seq: List[str], rows:int, mode:str, style:str, tie_rate:float, jitter:float, with_xgb:int) -> dict:
    X, y = _synth_from_seed(seed_seq, rows=rows, mode=mode, style=style, tie_rate=tie_rate, jitter=jitter)
    # simple split
    Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    metrics = {}
    # Standardize numeric for LR
    num_cols = [c for c in X.columns if X[c].dtype!=object]
    scaler = StandardScaler()
    Xtrain_num = Xtrain[num_cols].values
    Xval_num   = Xval[num_cols].values
    scaler.fit(Xtrain_num)

    # LGBM
    if _HAVE_LGBM:
        lgbm = LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42)
        lgbm.fit(Xtrain, ytrain)
        joblib.dump(lgbm, LGBM_FILE)
        p = lgbm.predict_proba(Xval)
        metrics["lgbm"] = {"acc": float(accuracy_score(yval, np.argmax(p,axis=1).tolist())),
                           "logloss": float(log_loss(yval, p, labels=LABELS))}
    # LR
    lr = LogisticRegression(max_iter=200, multi_class="multinomial")
    lr.fit(scaler.transform(Xtrain_num), ytrain)
    joblib.dump({"model":lr, "scaler":scaler, "cols":num_cols}, LR_FILE)
    p = lr.predict_proba(scaler.transform(Xval_num))
    metrics["lr"] = {"acc": float(accuracy_score(yval, np.argmax(p,axis=1).tolist())),
                     "logloss": float(log_loss(yval, p, labels=LABELS))}
    # XGB optional
    if with_xgb and _HAVE_XGB:
        xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, objective="multi:softprob", num_class=3, tree_method="hist", random_state=42)
        xgb.fit(Xtrain, np.array([LABELS.index(v) for v in ytrain]))
        joblib.dump(xgb, XGB_FILE)
        p = xgb.predict_proba(Xval)
        metrics["xgb"] = {"acc": float(accuracy_score([LABELS.index(v) for v in yval], np.argmax(p,axis=1))),
                          "logloss": float(log_loss([LABELS.index(v) for v in yval], p))}
    meta = {"labels":LABELS, "time":int(time.time()), "metrics":metrics, "feature_cols":list(X.columns)}
    _save_meta(meta)
    return meta


def _start_training_async(seed_seq: List[str],
                          rows:int=None, mode:str=None, style:str=None,
                          tie:float=None, jitter:float=None, with_xgb:int=None):
    rows   = rows   if rows   is not None else TRAIN_ROWS_DEF
    mode   = mode   if mode   is not None else TRAIN_MODE_DEF
    style  = style  if style  is not None else TRAIN_STYLE_DEF
    tie    = tie    if tie    is not None else TRAIN_TIE_DEF
    jitter = jitter if jitter is not None else TRAIN_JITTER_DEF
    with_xgb = with_xgb if with_xgb is not None else TRAIN_WITH_XGB
    def _job():
        global _is_training, _last_metrics
        with _train_lock:
            _is_training=True
            try:
                meta = _train_ensemble(seed_seq, rows, mode, style, tie, jitter, with_xgb)
                _last_metrics = meta.get("metrics")
            finally:
                _is_training=False
    threading.Thread(target=_job, daemon=True).start()

def _need_train() -> bool:
    """If no model or seed is newer than models -> need train"""
    if not SEED_FILE.exists():
        return False
    seed_mtime = SEED_FILE.stat().st_mtime
    model_times = []
    for f in (LGBM_FILE, LR_FILE, XGB_FILE):
        if f.exists(): model_times.append(f.stat().st_mtime)
    if not model_times:
        return True
    return seed_mtime > max(model_times)

def _load_models(need_xgb:bool=False):
    models = []
    if LGBM_FILE.exists():
        try:
            models.append(("lgbm", joblib.load(LGBM_FILE)))
        except Exception: pass
    if LR_FILE.exists():
        try:
            obj = joblib.load(LR_FILE)
            models.append(("lr", obj))
        except Exception: pass
    if need_xgb and XGB_FILE.exists():
        try:
            models.append(("xgb", joblib.load(XGB_FILE)))
        except Exception: pass
    return models



def _bet_size_pct(prob_top: float) -> float:
    # Simple monotonic sizing: 0% at 0.5, up to 30% at 0.85+
    # clip between 0 and 0.3
    pct = max(0.0, min(0.3, (prob_top - 0.5) * 0.75))
    return pct

def _format_line_reply(total_hands:int, probs, top_label:str, bankroll:int) -> str:
    mB = float(probs.get("B",0)); mP=float(probs.get("P",0)); mT=float(probs.get("T",0))
    zh = {"B":"莊","P":"閒","T":"和"}
    pct = _bet_size_pct(max(mB,mP,mT))
    bet_amt = bankroll * pct
    tip = (f"🧱 10%={_fmt_money(bankroll*0.10)} ｜ 20%={_fmt_money(bankroll*0.20)} ｜ 30%={_fmt_money(bankroll*0.30)}")
    msg = (f"🇮🇹 已解析 {total_hands} 手 (0 ms)\n"
           f"機率：莊 {mB:.3f} ｜ 閒 {mP:.3f} ｜ 和 {mT:.3f}\n"
           f"👉 下一手建議：{zh.get(top_label, top_label)} 🎯\n"
           f"💰 本金：{_fmt_money(bankroll)}\n"
           f"✅ 建議下注：{_fmt_money(bet_amt)} = { _fmt_money(bankroll) } × {pct*100:.1f}%\n"
           f"{tip}\n"
           f"📨 直接輸入下一手結果（莊／閒／和 或 B/P/T），我會再幫你算下一局。")
    return msg

def _predict_proba_from_models(models, Xrow:pd.DataFrame) -> np.ndarray:
    probs = []
    for name, model in models:
        if name=="lr":
            scaler = model["scaler"]; cols = model["cols"]; lr = model["model"]
            p = lr.predict_proba(scaler.transform(Xrow[cols].values))
        elif name=="lgbm":
            p = model.predict_proba(Xrow)
        elif name=="xgb":
            p = model.predict_proba(Xrow)
        else:
            continue
        probs.append(np.array(p))
    if not probs:
        # uniform fallback
        return np.ones((1,3))/3.0
    P = np.mean(probs, axis=0)
    return P

def _feature_from_prefix(prefix: List[str]) -> pd.DataFrame:
    if len(prefix)<1: prefix = ["B"]
    X,_ = sliding_features(prefix + ["B"])  # add a dummy to get one row
    return X.tail(1).reset_index(drop=True)

def _mc_augment(Xrow: pd.DataFrame, mc:int, seed_seq: Optional[List[str]]=None, jitter_scale:float=0.02) -> pd.DataFrame:
    if mc<=1:
        return Xrow
    Xrep = pd.concat([Xrow]*mc, ignore_index=True)
    Xrep = jitter_numeric(Xrep, scale=jitter_scale)
    return Xrep

# ------------- API endpoints -------------

@app.route("/health")
def health():
    return "OK"

@app.route("/", methods=["GET"])
def home():
    if request.headers.get("Accept")=="application/json":
        meta = _load_meta()
        return jsonify({
            "health": "/health",
            "is_training": _is_training,
            "last_metrics": meta.get("metrics"),
            "model_exists": LGBM_FILE.exists() or LR_FILE.exists() or XGB_FILE.exists(),
            "seed_records": len(parse_text_seq(SEED_FILE.read_text(encoding='utf-8'))) if SEED_FILE.exists() else 0,
            "webhook": "/line-webhook"
        })
    # Simple dashboard HTML (no f-string to avoid braces parsing)
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>BGS Dashboard</title></head>
<body style="font-family:ui-monospace,Menlo,Consolas,monospace;max-width:880px;margin:20px auto;line-height:1.4">
<h2>BGS Dashboard</h2>
<button onclick="refresh()">重新整理狀態</button>
<span> Webhook：/line-webhook 健康檢查：/health</span>
<pre id="statusBox">{}</pre>

<h3>即時預測 /predict</h3>
<input id="seq" placeholder="例如：B P P T B"><input id="ensemble" placeholder="ensemble=none|light|full" style="width:200px">
<input id="mc" placeholder="mc=0/5000" style="width:120px"><button onclick="doPredict()">送出</button>
<pre id="predBox"></pre>

<h3>追加種子 /ingest-seed</h3>
<textarea id="seed" rows="6" cols="60" placeholder="貼上真實歷史：B P P T B ..."></textarea><br>
<button onclick="doSeed()">追加</button>
<pre id="seedBox"></pre>

<h3>啟動訓練 /synth-train</h3>
rows <input id="rows" value="100000" style="width:110px">
style <input id="style" value="hybrid" style="width:120px">
tie_rate <input id="tie" value="0.06" style="width:80px">
mode <select id="mode"><option>feature</option><option>ngram</option><option>uniform</option></select>
jitter <input id="jitter" value="0.02" style="width:80px">
with_xgb <input id="withxgb" value="0" style="width:60px">
<button onclick="doTrain()">開始訓練</button>
<pre id="trainBox"></pre>

<script>
async function refresh(){
  try{
    const r=await fetch(location.origin+"/",{headers:{"Accept":"application/json"}});
    const j=await r.json(); document.getElementById("statusBox").textContent=JSON.stringify(j,null,2);
  }catch(e){document.getElementById("statusBox").textContent="讀取失敗: "+e;}
}
async function doPredict(){
  const s=document.getElementById("seq").value;
  const ens=document.getElementById("ensemble").value||"light";
  const mc=document.getElementById("mc").value||"0";
  const qp=new URLSearchParams({seq:s, ensemble:ens, mc:mc});
  const r=await fetch("/predict?"+qp.toString());
  document.getElementById("predBox").textContent=await r.text();
}
async function doSeed(){
  const t=document.getElementById("seed").value;
  const r=await fetch("/ingest-seed",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:t})});
  document.getElementById("seedBox").textContent=await r.text();
}
async function doTrain(){
  const rows=document.getElementById("rows").value;
  const style=document.getElementById("style").value;
  const tie=document.getElementById("tie").value;
  const mode=document.getElementById("mode").value;
  const jitter=document.getElementById("jitter").value;
  const withxgb=document.getElementById("withxgb").value;
  const r=await fetch("/synth-train",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({rows:parseInt(rows),style:style,tie_rate:parseFloat(tie),mode:mode,jitter:parseFloat(jitter),with_xgb:parseInt(withxgb)})});
  document.getElementById("trainBox").textContent=await r.text();
}
refresh();
</script>
</body></html>"""

@app.route("/ingest-seed", methods=["POST"])
def ingest_seed():
    data = request.get_json(silent=True) or {}
    text = data.get("text","")
    seq = parse_text_seq(text)
    if not seq:
        return jsonify({"ok": False, "msg":"no valid tokens"}), 400
    old = []
    if SEED_FILE.exists():
        old = parse_text_seq(SEED_FILE.read_text(encoding="utf-8"))
    merged = old + seq
    SEED_FILE.write_text(" ".join(merged), encoding="utf-8")
    # auto-train on seed append
    if AUTO_TRAIN and len(merged)>=MIN_SEED_FOR_TRAIN and not _is_training:
        _start_training_async(merged)
    return jsonify({"ok": True, "added": len(seq), "total": len(merged)})

@app.route("/synth-train", methods=["POST"])
def synth_train():
    data = request.get_json(silent=True) or {}
    rows = int(data.get("rows", 100000))
    style = str(data.get("style","hybrid"))
    tie_rate = float(data.get("tie_rate", 0.06))
    mode = str(data.get("mode","feature"))
    jitter = float(data.get("jitter", 0.02))
    with_xgb = int(data.get("with_xgb", 0))
    if not SEED_FILE.exists():
        return jsonify({"ok":False,"msg":"no seed yet"}), 400
    seed_seq = parse_text_seq(SEED_FILE.read_text(encoding="utf-8"))
    def _job():
        global _is_training, _last_metrics
        with _train_lock:
            _is_training=True
            try:
                meta = _train_ensemble(seed_seq, rows, mode, style, tie_rate, jitter, with_xgb)
                _last_metrics = meta.get("metrics")
            finally:
                _is_training=False
    threading.Thread(target=_job, daemon=True).start()
    return jsonify({"ok":True,"msg":"training started","rows":rows,"mode":mode,"with_xgb":with_xgb})

@app.route("/predict", methods=["GET"])
def predict():
    seq_param = request.args.get("seq","")
    seq = parse_text_seq(seq_param)
    if not seq:
        return jsonify({"ok":False,"msg":"seq required like 'B P T' or 'BPT'"}), 400
    ensemble = (request.args.get("ensemble", PRED_ENSEMBLE_DEF) or PRED_ENSEMBLE_DEF).lower()
    mc = int(request.args.get("mc", str(PRED_MC_DEF)) or str(PRED_MC_DEF))
    need_xgb = (ensemble=="full")
    models = _load_models(need_xgb=need_xgb)
    if not models:
        # no model yet -> heuristic on counts
        cnt=Counter(seq)
        probs = np.array([[cnt.get("B",1), cnt.get("P",1), cnt.get("T",1)]], dtype=float)
        probs = probs / probs.sum()
        labs = LABELS
        resp = {"source":"heuristic","probabilities":{labs[i]:float(probs[0,i]) for i in range(3)},
                "probs":{labs[i]:float(probs[0,i]) for i in range(3)},
                "top": labs[int(np.argmax(probs))], "label": labs[int(np.argmax(probs))]}
        return jsonify(resp)
    Xrow = _feature_from_prefix(seq)
    if mc>1:
        Xmc  = _mc_augment(Xrow, mc=mc, seed_seq=None, jitter_scale=0.02)
        P = _predict_proba_from_models(models, Xmc)
        P = np.mean(P, axis=0, keepdims=True)
    else:
        P = _predict_proba_from_models(models, Xrow)
    labs = LABELS
    out = {"source":"ensemble" if len(models)>1 else models[0][0],
           "probabilities":{labs[i]:float(P[0,i]) for i in range(3)},
           "probs":{labs[i]:float(P[0,i]) for i in range(3)},
           "top": labs[int(np.argmax(P))], "label": labs[int(np.argmax(P))]}
    return jsonify(out)

# ---------------- LINE webhook (minimal) ----------------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET","").strip()
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN","").strip()

_use_line = bool(LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN)
if _use_line:
    # Using v3 sdk to avoid deprecation warnings
    try:
        from linebot.v3.webhook import WebhookParser
        from linebot.v3.messaging import MessagingApi, Configuration, ReplyMessageRequest, TextMessage
        from linebot.v3.exceptions import InvalidSignatureError
        _LINE_V3 = True
        _line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
        _line_api = MessagingApi(_line_config)
        _line_parser = WebhookParser(LINE_CHANNEL_SECRET)
    except Exception:
        _LINE_V3 = False
        _use_line = False  # fallback disable if import fails
else:
    _LINE_V3 = False

HELP_TEXT = ("指令： SEED: <路單> ｜\n"
             "TRAIN <rows> [style] [tie] [mode] [jitter] [with_xgb] ｜ STATUS ｜\n"
             "PRED <路單> ｜ PRED[ENS] <路單> ｜ PRED[FULL] <路單>\n"
             "例：SEED: B P P T B\n"
             "例：TRAIN 100000 hybrid 0.06 feature 0.02 0\n"
             "例：PRED[ENS] B P T B P\n")

def _handle_command(txt: str) -> str:
    # bankroll setter, examples: "本金 50000" or "50000"
    mb = re.match(r"(?i)^(?:本金\s*[:：]?\s*)?([1-9]\d{2,})$", t.replace(',', ''))  # >= 100
    if mb:
        val = int(mb.group(1))
        # state will be set at webhook using user id; here we just echo & instruct
        return f"👍 已設定本金：{_fmt_money(val)} 元。接著貼上歷史（B/P/T 或 莊/閒/和），然後輸入『開始分析』即可！🚀"
    if re.match(r"^(開始分析)$", t):
        return "🔄 模式已開啟。請直接輸入下一手結果（莊/閒/和 或 B/P/T），我會持續幫你算下一局。"
    if re.match(r"^(結束分析)$", t):
        return "⛔ 已結束，本金設定保留。要重新開始請先貼歷史，然後輸入『開始分析』。"

    t = txt.strip()
    m = re.match(r"(?i)^SEED\s*:\s*(.+)$", t)
    if m:
        seq = parse_text_seq(m.group(1))
        if not seq: return "格式錯誤，請用：SEED: B P P ..."
        old = []
        if SEED_FILE.exists():
            old = parse_text_seq(SEED_FILE.read_text(encoding="utf-8"))
        merged = old + seq
        SEED_FILE.write_text(" ".join(merged), encoding="utf-8")
        if AUTO_TRAIN and len(merged)>=MIN_SEED_FOR_TRAIN and not _is_training:
            _start_training_async(merged)
        return f"📝 已接收歷史共 {len(seq)} 手，目前累計 {_fmt_money(len(merged))} 手。\n輸入『開始分析』即可啟動。"
    m = re.match(r"(?i)^TRAIN\s+(\d+)(?:\s+(\S+))?(?:\s+([\d\.]+))?(?:\s+(\S+))?(?:\s+([\d\.]+))?(?:\s+(\d+))?", t)
    if m:
        rows = int(m.group(1)); style = m.group(2) or "hybrid"; tie = float(m.group(3) or "0.06")
        mode = m.group(4) or "feature"; jitter=float(m.group(5) or "0.02"); with_xgb=int(m.group(6) or "0")
        if not SEED_FILE.exists(): return "尚未有種子資料，請先 SEED:"
        seed_seq = parse_text_seq(SEED_FILE.read_text(encoding="utf-8"))
        def _job():
            global _is_training, _last_metrics
            with _train_lock:
                _is_training=True
                try:
                    meta = _train_ensemble(seed_seq, rows, mode, style, tie, jitter, with_xgb)
                    _last_metrics = meta.get("metrics")
                finally:
                    _is_training=False
        threading.Thread(target=_job, daemon=True).start()
        return f"訓練啟動 rows={rows} mode={mode} with_xgb={with_xgb}"
    if re.match(r"(?i)^STATUS$", t):
        meta = _load_meta()
        return json.dumps({"is_training":_is_training,"metrics":meta.get("metrics")}, ensure_ascii=False)
    m = re.match(r"(?i)^PRED(\[(ENS|FULL)\])?\s+(.+)$", t)
    if m:
        ens_key = (m.group(2) or "").lower()
        if ens_key=="full": ens="full"
        elif ens_key=="ens": ens="light"
        else: ens="light"
        seq = parse_text_seq(m.group(3))
        if not seq: return "格式錯誤，請用：PRED B P T ..."
        models = _load_models(need_xgb=(ens=="full"))
        if not models: return "尚未訓練，請先 TRAIN"
        Xrow = _feature_from_prefix(seq)
        P = _predict_proba_from_models(models, Xrow)
        labs = LABELS
        return f"預測：{labs[int(np.argmax(P))]} 機率 { {labs[i]:round(float(P[0,i]),3) for i in range(3)} }"
    return HELP_TEXT

@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not _use_line:
        return "LINE not configured", 200
    signature = request.headers.get("X-Line-Signature","")
    body = request.get_data(as_text=True)
    try:
        events = _line_parser.parse(body, signature)
    except Exception:
        return "invalid signature", 400
    
    for ev in events:
        if ev.type!="message" or ev.message.type!="text": 
            continue
        user_text = ev.message.text or ""
        uid = getattr(getattr(ev, "source", None), "user_id", None) or getattr(getattr(ev, "source", None), "userId", None) or "anon"
        st = _get_state(uid)
        # bankroll quick set
        mb = re.match(r"(?i)^(?:本金\s*[:：]?\s*)?([1-9]\d{2,})$", user_text.replace(',', ''))
        if mb:
            st["bankroll"] = int(mb.group(1))
            reply = f"👍 已設定本金：{_fmt_money(st['bankroll'])} 元。接著貼上歷史（B/P/T 或 莊/閒/和），然後輸入『開始分析』即可！🚀"
        elif user_text.strip()=="開始分析":
            st["in_session"]=True
            reply = "✅ 已開始分析。直接輸入下一手結果（莊／閒／和 或 B/P/T），我會再幫你算下一局。"
        elif user_text.strip()=="結束分析":
            st["in_session"]=False
            reply = "⛔ 已結束，本金設定保留。要重新開始請先貼歷史，然後輸入『開始分析』。"
        else:
            # map Chinese outcome to B/P/T if in session
            zhmap = {"莊":"B","閒":"P","和":"T"}
            simple = user_text.strip().upper()
            if st.get("in_session") and simple in zhmap.keys()|set(["B","P","T"]):
                lab = zhmap.get(simple, simple)
                # append to seed
                old = parse_text_seq(SEED_FILE.read_text(encoding="utf-8")) if SEED_FILE.exists() else []
                merged = old + [lab]
                SEED_FILE.write_text(" ".join(merged), encoding="utf-8")
                st["hands"] = len(merged)
                # do prediction for next
                need_xgb = (PRED_ENSEMBLE_DEF=="full")
                models = _load_models(need_xgb=need_xgb)
                if not models:
                    reply = "尚未訓練，請先貼歷史並等待訓練完成或輸入 TRAIN。"
                else:
                    Xrow = _feature_from_prefix(merged)
                    P = _predict_proba_from_models(models, Xrow)
                    probs = {"B": float(P[0,0]), "P": float(P[0,1]), "T": float(P[0,2])}
                    labs = LABELS
                    top = labs[int(np.argmax(P))]
                    reply = _format_line_reply(st["hands"], probs, top, st["bankroll"])
            else:
                reply = _handle_command(user_text)
        try:
            _line_api.reply_message(ReplyMessageRequest(
                replyToken=ev.reply_token,
                messages=[TextMessage(text=reply[:4950])]
            ))
        except Exception:
            pass
    return "OK", 200
