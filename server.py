#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGS LINE Bot backend — v12.1
Ensemble(Full-history) + Regime Gating + Page-Hinkley Drift
+ Big-Road-like mapping (6x20) for features
+ CSV I/O (/export) + hot reload (/reload)
+ LINE buttons: 莊(紅)/閒(藍)/和(綠)/開始分析/結束分析/返回
"""

import os, csv, time, logging
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-backend")

# ========= 路徑 / 環境變數 =========
DATA_CSV_PATH = os.getenv("DATA_LOG_PATH", "/data/logs/rounds.csv")
os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)

RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "")
RNN_PATH = os.getenv("RNN_PATH", "/data/models/rnn.pt")
XGB_PATH = os.getenv("XGB_PATH", "/data/models/xgb.json")
LGBM_PATH = os.getenv("LGBM_PATH", "/data/models/lgbm.txt")

# ========= 基礎常數 =========
CLASS_ORDER = ("B", "P", "T")
LAB_ZH = {"B": "莊", "P": "閒", "T": "和"}
THEORETICAL_PROBS: Dict[str, float] = {"B": 0.458, "P": 0.446, "T": 0.096}

def parse_history(payload) -> List[str]:
    if payload is None: return []
    seq: List[str] = []
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s, str) and s.strip().upper() in CLASS_ORDER:
                seq.append(s.strip().upper())
    elif isinstance(payload, str):
        for ch in payload:
            up = ch.upper()
            if up in CLASS_ORDER: seq.append(up)
    return seq

# ========= 可選模型（lazy import） =========
try:
    import torch  # type: ignore
    import torch.nn as tnn  # type: ignore
except Exception:
    torch = None; tnn = None

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None

if tnn is not None:
    class TinyRNN(tnn.Module):  # type: ignore
        def __init__(self, in_dim=3, hidden=16, out_dim=3):
            super().__init__()
            self.rnn = tnn.GRU(in_dim, hidden, batch_first=True)
            self.fc  = tnn.Linear(hidden, out_dim)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])
else:
    TinyRNN = None  # type: ignore

# ========= 模型載入/重載 =========
RNN_MODEL: Optional[Any] = None
XGB_MODEL: Optional[Any] = None
LGBM_MODEL: Optional[Any] = None

def load_models() -> None:
    global RNN_MODEL, XGB_MODEL, LGBM_MODEL
    # RNN
    if TinyRNN is not None and torch is not None and RNN_PATH and os.path.exists(RNN_PATH):
        try:
            _m = TinyRNN()
            _m.load_state_dict(torch.load(RNN_PATH, map_location="cpu"))
            _m.eval()
            RNN_MODEL = _m
            logger.info("Loaded RNN from %s", RNN_PATH)
        except Exception as e:
            logger.warning("Load RNN failed: %s", e); RNN_MODEL = None
    else:
        RNN_MODEL = None
    # XGB
    if xgb is not None and XGB_PATH and os.path.exists(XGB_PATH):
        try:
            booster = xgb.Booster()
            booster.load_model(XGB_PATH)
            XGB_MODEL = booster
            logger.info("Loaded XGB from %s", XGB_PATH)
        except Exception as e:
            logger.warning("Load XGB failed: %s", e); XGB_MODEL = None
    else:
        XGB_MODEL = None
    # LGBM
    if lgb is not None and LGBM_PATH and os.path.exists(LGBM_PATH):
        try:
            booster = lgb.Booster(model_file=LGBM_PATH)
            LGBM_MODEL = booster
            logger.info("Loaded LGBM from %s", LGBM_PATH)
        except Exception as e:
            logger.warning("Load LGBM failed: %s", e); LGBM_MODEL = None
    else:
        LGBM_MODEL = None

load_models()

# ========= Tie（和）處理 + 二分類模型補 T =========
def exp_decay_freq(seq: List[str], gamma: float = None) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    if gamma is None: gamma = float(os.getenv("EW_GAMMA","0.96"))
    wB = wP = wT = 0.0; w = 1.0
    for r in reversed(seq):
        if r=="B": wB += w
        elif r=="P": wP += w
        else: wT += w
        w *= gamma
    alpha = float(os.getenv("LAPLACE", "0.5"))
    wB += alpha; wP += alpha; wT += alpha
    S = wB+wP+wT
    return [wB/S, wP/S, wT/S]

def _estimate_tie_prob(seq: List[str]) -> float:
    prior_T = THEORETICAL_PROBS["T"]
    long_T  = exp_decay_freq(seq)[2]
    w       = float(os.getenv("T_BLEND", "0.5"))
    floor_T = float(os.getenv("T_MIN", "0.03"))
    cap_T   = float(os.getenv("T_MAX", "0.18"))
    pT = (1-w)*prior_T + w*long_T
    return max(floor_T, min(cap_T, pT))

def _merge_bp_with_t(bp: List[float], pT: float) -> List[float]:
    b, p = float(bp[0]), float(bp[1])
    s = max(1e-12, b+p); b/=s; p/=s
    scale = 1.0 - pT
    return [b*scale, p*scale, pT]

# ========= 單模型推論 =========
def rnn_predict(seq: List[str]) -> Optional[List[float]]:
    if RNN_MODEL is None or torch is None or not seq: return None
    try:
        def onehot(label: str): return [1 if label == lab else 0 for lab in CLASS_ORDER]
        inp = torch.tensor([[onehot(ch) for ch in seq]], dtype=torch.float32)
        with torch.no_grad():
            logits = RNN_MODEL(inp)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        return [float(p) for p in probs]
    except Exception as e:
        logger.warning("RNN inference failed: %s", e); return None

def xgb_predict(seq: List[str]) -> Optional[List[float]]:
    if XGB_MODEL is None or not seq: return None
    try:
        import numpy as np
        K = int(os.getenv("FEAT_WIN", "20"))
        vec: List[float] = []
        for label in seq[-K:]:
            vec.extend([1.0 if label==lab else 0.0 for lab in CLASS_ORDER])
        pad = K*3 - len(vec)
        if pad>0: vec = [0.0]*pad + vec
        dmatrix = xgb.DMatrix(np.array([vec], dtype=float))
        prob = XGB_MODEL.predict(dmatrix)[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            pT = _estimate_tie_prob(seq)
            return _merge_bp_with_t([float(prob[0]), float(prob[1])], pT)
        return None
    except Exception as e:
        logger.warning("XGB inference failed: %s", e); return None

def lgbm_predict(seq: List[str]) -> Optional[List[float]]:
    if LGBM_MODEL is None or not seq: return None
    try:
        K = int(os.getenv("FEAT_WIN", "20"))
        vec: List[float] = []
        for label in seq[-K:]:
            vec.extend([1.0 if label==lab else 0.0 for lab in CLASS_ORDER])
        pad = K*3 - len(vec)
        if pad>0: vec = [0.0]*pad + vec
        prob = LGBM_MODEL.predict([vec])[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            pT = _estimate_tie_prob(seq)
            return _merge_bp_with_t([float(prob[0]), float(prob[1])], pT)
        return None
    except Exception as e:
        logger.warning("LGBM inference failed: %s", e); return None

# ========= 短期 / Markov / 工具 =========
def recent_freq(seq: List[str], win: int) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    cut = seq[-win:] if win>0 else seq
    alpha = float(os.getenv("LAPLACE","0.5"))
    nB = cut.count("B")+alpha; nP = cut.count("P")+alpha; nT = cut.count("T")+alpha
    tot = max(1,len(cut))+3*alpha
    return [nB/tot, nP/tot, nT/tot]

def markov_next_prob(seq: List[str], decay: float = None) -> List[float]:
    if not seq or len(seq)<2: return [1/3,1/3,1/3]
    if decay is None: decay = float(os.getenv("MKV_DECAY","0.98"))
    idx = {"B":0,"P":1,"T":2}
    C = [[0.0]*3 for _ in range(3)]
    w = 1.0
    for a,b in zip(seq[:-1], seq[1:]):
        C[idx[a]][idx[b]] += w; w *= decay
    flow_to = [C[0][0]+C[1][0]+C[2][0],
               C[0][1]+C[1][1]+C[2][1],
               C[0][2]+C[1][2]+C[2][2]]
    alpha = float(os.getenv("MKV_LAPLACE","0.5"))
    flow_to = [x+alpha for x in flow_to]
    S = sum(flow_to)
    return [x/S for x in flow_to]

def norm(v: List[float]) -> List[float]:
    s = sum(v); s = s if s>1e-12 else 1.0
    return [max(0.0,x)/s for x in v]

def blend(a: List[float], b: List[float], w: float) -> List[float]:
    return [(1-w)*a[i] + w*b[i] for i in range(3)]

def temperature_scale(p: List[float], tau: float) -> List[float]:
    if tau<=1e-6: return p
    ex = [pow(max(pi,1e-9), 1.0/tau) for pi in p]
    s = sum(ex); return [e/s for e in ex]

# ========= 路型偵測（Regime gating） =========
def last_run(seq: List[str]) -> Tuple[str,int]:
    if not seq: return ("",0)
    ch = seq[-1]; i=len(seq)-2; n=1
    while i>=0 and seq[i]==ch: n+=1; i-=1
    return (ch,n)

def run_lengths(seq: List[str], win: int = 14) -> List[int]:
    if not seq: return []
    s = seq[-win:] if win>0 else seq[:]
    lens=[]; cur=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]: cur+=1
        else: lens.append(cur); cur=1
    lens.append(cur); return lens

def is_zigzag(seq: List[str], k:int=6)->bool:
    s = seq[-k:] if len(seq)>=k else seq
    if len(s)<4: return False
    alt = all(s[i]!=s[i-1] for i in range(1,len(s)))
    if alt: return True
    if len(s)%2==0:
        pairs=[s[i:i+2] for i in range(0,len(s),2)]
        if all(len(p)==2 and p[0]==p[1] for p in pairs):
            if all(pairs[i][0]!=pairs[i-1][0] for i in range(1,len(pairs))):
                return True
    return False

def is_qijiao(seq: List[str], win:int=20, tol:float=0.1)->bool:
    s = seq[-win:] if len(seq)>=win else seq
    if not s: return False
    b = s.count("B"); p = s.count("P"); t = s.count("T")
    tot_bp = max(1,b+p); ratio=b/tot_bp
    return (abs(ratio-0.5)<=tol) and (t <= max(1,int(0.15*len(s))))

def is_oscillating(seq: List[str], win:int=12)->bool:
    lens = run_lengths(seq, win)
    if not lens: return False
    avg = sum(lens)/len(lens)
    return 1.0 <= avg <= 2.1

def shape_1room2hall(seq: List[str], win:int=18)->bool:
    lens=run_lengths(seq, win)
    if len(lens)<6: return False
    if not all(1<=x<=2 for x in lens[-6:]): return False
    alt = all((lens[i]%2)!=(lens[i-1]%2) for i in range(1,min(len(lens),10)))
    return alt

def shape_2room1hall(seq: List[str], win:int=18)->bool:
    lens=run_lengths(seq, win)
    if len(lens)<5: return False
    last = lens[-6:] if len(lens)>=6 else lens
    cnt_1 = sum(1 for x in last if x==1)
    cnt_2 = sum(1 for x in last if x==2)
    return (cnt_1+cnt_2) >= max(4,int(0.7*len(last))) and cnt_2 >= cnt_1

def regime_boosts(seq: List[str]) -> List[float]:
    if not seq: return [1.0,1.0,1.0]
    b=[1.0,1.0,1.0]  # [B,P,T]
    last,rlen = last_run(seq)
    DRAGON_TH    = int(os.getenv("BOOST_DRAGON_LEN", "4"))
    BOOST_DRAGON = float(os.getenv("BOOST_DRAGON", "1.12"))
    BOOST_ALT    = float(os.getenv("BOOST_ALT", "1.08"))
    BOOST_QJ     = float(os.getenv("BOOST_QIJIAO", "1.05"))
    BOOST_ROOM   = float(os.getenv("BOOST_ROOM", "1.06"))
    BOOST_T      = float(os.getenv("BOOST_T", "1.03"))
    if rlen>=DRAGON_TH:
        if last=="B": b[0]*=BOOST_DRAGON
        elif last=="P": b[1]*=BOOST_DRAGON
        else: b[2]*=BOOST_DRAGON
    if is_zigzag(seq,6) or is_oscillating(seq,12) or shape_1room2hall(seq):
        if seq[-1]=="B": b[1]*=BOOST_ALT
        elif seq[-1]=="P": b[0]*=BOOST_ALT
    if shape_2room1hall(seq):
        s = seq[-10:] if len(seq)>=10 else seq
        if s.count("B")>s.count("P"): b[0]*=BOOST_ROOM
        elif s.count("P")>s.count("B"): b[1]*=BOOST_ROOM
    if is_qijiao(seq): b[0]*=BOOST_QJ; b[1]*=BOOST_QJ
    ew = exp_decay_freq(seq)
    if ew[2] > THEORETICAL_PROBS["T"]*1.15: b[2]*=BOOST_T
    return b

def _apply_boosts_and_norm(probs: List[float], boosts: List[float]) -> List[float]:
    p=[max(1e-12, probs[i]*boosts[i]) for i in range(3)]
    s=sum(p); return [x/s for x in p]

# ========= Page-Hinkley 漂移偵測 =========
def js_divergence(p: List[float], q: List[float]) -> float:
    import math
    eps=1e-12; m=[(p[i]+q[i])/2.0 for i in range(3)]
    def _kl(a,b): return sum((ai+eps)*math.log((ai+eps)/(bi+eps)) for ai,bi in zip(a,b))
    return 0.5*_kl(p,m)+0.5*_kl(q,m)

USER_DRIFT: Dict[str, Dict[str, float]] = {}  # cum / min / cooldown
def _get_drift_state(uid: str) -> Dict[str, float]:
    st = USER_DRIFT.get(uid)
    if st is None:
        st={'cum':0.0,'min':0.0,'cooldown':0.0}; USER_DRIFT[uid]=st
    return st

def update_ph_state(uid: str, seq: List[str]) -> bool:
    if not seq: return False
    st = _get_drift_state(uid)
    REC_WIN = int(os.getenv("REC_WIN_FOR_PH","12"))
    p_short = recent_freq(seq, REC_WIN)
    p_long  = exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.96")))
    D_t     = js_divergence(p_short, p_long)
    PH_DELTA   = float(os.getenv("PH_DELTA","0.005"))
    PH_LAMBDA  = float(os.getenv("PH_LAMBDA","0.08"))
    DRIFT_STEPS= float(os.getenv("DRIFT_STEPS","5"))
    st['cum'] += (D_t - PH_DELTA)
    st['min']  = min(st['min'], st['cum'])
    if (st['cum'] - st['min']) > PH_LAMBDA:
        st['cum']=0.0; st['min']=0.0; st['cooldown']=DRIFT_STEPS
        logger.info(f"[PH] drift triggered for {uid}: D_t={D_t:.4f}")
        return True
    return False

def consume_cooldown(uid: str) -> bool:
    st=_get_drift_state(uid)
    if st['cooldown']>0: st['cooldown']=max(0.0, st['cooldown']-1.0); return True
    return False

def in_drift(uid: str) -> bool:
    return _get_drift_state(uid)['cooldown']>0.0

# ========= Big-Road-like mapping（6x20，僅作為特徵參考，不輸出） =========
def map_to_grid(seq: List[str], rows:int=6, cols:int=20)->List[List[str]]:
    grid=[["" for _ in range(cols)] for _ in range(rows)]
    if not seq: return grid
    r=c=0; last=None
    for ch in seq:
        if ch==last:
            if r+1<rows and grid[r+1][c]=="":
                r+=1
            else:
                c=min(cols-1,c+1)
        else:
            last=ch; r=0; c=min(cols-1,c+1) if grid[r][c]!="" else c
            while c<cols and grid[0][c]!="": c+=1
            if c>=cols: c=cols-1
        if grid[r][c]=="": grid[r][c]=ch
    return grid

# ========= Ensemble（含修正一行 MKV 調用） =========
def ensemble_with_anti_stuck(seq: List[str], weight_overrides: Optional[Dict[str,float]]=None) -> List[float]:
    # 模型
    rule  = [THEORETICAL_PROBS["B"], THEORETICAL_PROBS["P"], THEORETICAL_PROBS["T"]]
    pr_rnn = rnn_predict(seq)
    pr_xgb = xgb_predict(seq)
    pr_lgb = lgbm_predict(seq)

    # 權重
    w_rule = float(os.getenv("RULE_W","0.40"))
    w_rnn  = float(os.getenv("RNN_W","0.25"))
    w_xgb  = float(os.getenv("XGB_W","0.20"))
    w_lgb  = float(os.getenv("LGBM_W","0.15"))

    total = w_rule + (w_rnn if pr_rnn else 0) + (w_xgb if pr_xgb else 0) + (w_lgb if pr_lgb else 0)
    base = [w_rule*rule[i] for i in range(3)]
    if pr_rnn: base=[base[i]+w_rnn*pr_rnn[i] for i in range(3)]
    if pr_xgb: base=[base[i]+w_xgb*pr_xgb[i] for i in range(3)]
    if pr_lgb: base=[base[i]+w_lgb*pr_lgb[i] for i in range(3)]
    probs=[b/max(total,1e-9) for b in base]

    # 全盤訊號
    REC_WIN = int(os.getenv("REC_WIN","16"))
    REC_W   = float(os.getenv("REC_W","0.25"))
    LONG_W  = float(os.getenv("LONG_W","0.35"))
    MKV_W   = float(os.getenv("MKV_W","0.25"))
    PRIOR_W = float(os.getenv("PRIOR_W","0.15"))
    if weight_overrides:
        REC_W  = weight_overrides.get("REC_W",REC_W)
        LONG_W = weight_overrides.get("LONG_W",LONG_W)
        MKV_W  = weight_overrides.get("MKV_W",MKV_W)
        PRIOR_W= weight_overrides.get("PRIOR_W",PRIOR_W)

    p_rec  = recent_freq(seq, REC_WIN)
    p_long = exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.96")))
    # ★ 修正一行：正確傳入 (seq, decay)
    p_mkv  = markov_next_prob(seq, float(os.getenv("MKV_DECAY","0.98")))

    probs = blend(probs, p_rec,  REC_W)
    probs = blend(probs, p_long, LONG_W)
    probs = blend(probs, p_mkv,  MKV_W)
    probs = blend(probs, rule,   PRIOR_W)

    # 安全處理
    EPS = float(os.getenv("EPSILON_FLOOR","0.06"))
    CAP = float(os.getenv("MAX_CAP","0.88"))
    TAU = float(os.getenv("TEMP","1.10"))
    probs=[min(CAP,max(EPS,p)) for p in probs]
    probs=norm(probs); probs=temperature_scale(probs, TAU)

    # Regime gating
    boosts = regime_boosts(seq)
    probs  = _apply_boosts_and_norm(probs, boosts)
    return norm(probs)

def recommend_from_probs(probs: List[float]) -> str:
    return CLASS_ORDER[probs.index(max(probs))]

# ========= Health / Predict / Export / Reload =========
@app.route("/", methods=["GET"])
def index(): return "ok"

@app.route("/health", methods=["GET"])
def health(): return jsonify(status="healthy", version="v12.1")

@app.route("/healthz", methods=["GET"])
def healthz(): return jsonify(status="healthy")

@app.route("/predict", methods=["POST"])
def predict():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    seq = parse_history(data.get("history"))
    probs = ensemble_with_anti_stuck(seq)
    rec   = recommend_from_probs(probs)
    labels = list(CLASS_ORDER)
    return jsonify({
        "history_len": len(seq),
        "probabilities": {labels[i]: probs[i] for i in range(3)},
        "recommendation": rec
    })

def append_round_csv(user_id: str, history_before: str, label: str) -> None:
    try:
        os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)
        with open(DATA_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([user_id, int(time.time()), history_before, label])
    except Exception as e:
        logger.warning("append_round_csv failed: %s", e)

@app.route("/export", methods=["GET"])
def export_csv():
    n = int(request.args.get("n", "1000"))
    rows: List[List[str]] = []
    try:
        if os.path.exists(DATA_CSV_PATH):
            with open(DATA_CSV_PATH, "r", encoding="utf-8") as f:
                data = list(csv.reader(f))
                rows = data[-n:] if n > 0 else data
    except Exception as e:
        logger.warning("export read failed: %s", e); rows=[]
    output = "user_id,ts,history_before,label\n" + "\n".join([",".join(r) for r in rows])
    return Response(output, mimetype="text/csv",
        headers={"Content-Disposition":"attachment; filename=rounds.csv"})

@app.route("/reload", methods=["POST"])
def reload_models():
    token = request.headers.get("X-Reload-Token","") or request.args.get("token","")
    if not RELOAD_TOKEN or token != RELOAD_TOKEN:
        return jsonify(ok=False, error="unauthorized"), 401
    load_models()
    return jsonify(ok=True, rnn=bool(RNN_MODEL), xgb=bool(XGB_MODEL), lgbm=bool(LGBM_MODEL))

# ========= LINE Webhook =========
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET", "")

USE_LINE=False
try:
    from linebot import LineBotApi, WebhookHandler  # type: ignore
    from linebot.models import (  # type: ignore
        MessageEvent, TextMessage, TextSendMessage,
        PostbackEvent, PostbackAction,
        FlexSendMessage,
        QuickReply, QuickReplyButton
    )
    USE_LINE = bool(LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET)
except Exception as e:
    logger.warning("LINE SDK not available or env not set: %s", e)
    USE_LINE=False

if USE_LINE:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
else:
    line_bot_api=None; handler=None

USER_HISTORY: Dict[str, List[str]] = {}
USER_READY:   Dict[str, bool]      = {}
USER_DRIFT:   Dict[str, Dict[str, float]] = USER_DRIFT  # alias

def flex_buttons_card() -> 'FlexSendMessage':
    # 三個主按鈕用 primary + color；返回/開始/結束用 secondary
    contents = {
        "type": "bubble",
        "body": {
            "type": "box", "layout": "vertical", "spacing": "md",
            "contents": [
                {"type": "text", "text": "🤖 請開始輸入歷史數據", "weight": "bold", "size": "lg"},
                {"type": "text", "text": "先輸入莊/閒/和；按「開始分析」後才會給出下注建議。", "wrap": True, "size": "sm", "color": "#555555"},
                {"type":"box","layout":"horizontal","spacing":"sm","contents":[
                    {"type":"button","style":"primary","color":"#E74C3C","action":{"type":"postback","label":"莊","data":"B"}},
                    {"type":"button","style":"primary","color":"#2980B9","action":{"type":"postback","label":"閒","data":"P"}},
                    {"type":"button","style":"primary","color":"#27AE60","action":{"type":"postback","label":"和","data":"T"}}
                ]},
                {"type":"box","layout":"horizontal","spacing":"sm","contents":[
                    {"type":"button","style":"secondary","action":{"type":"postback","label":"開始分析","data":"START"}},
                    {"type":"button","style":"secondary","action":{"type":"postback","label":"結束分析","data":"END"}},
                    {"type":"button","style":"secondary","action":{"type":"postback","label":"返回","data":"UNDO"}}
                ]}
            ]
        }
    }
    from linebot.models import FlexSendMessage  # type: ignore
    return FlexSendMessage(alt_text="請開始輸入歷史數據", contents=contents)

def quick_reply_bar():
    from linebot.models import QuickReply, QuickReplyButton, PostbackAction  # type: ignore
    return QuickReply(items=[
        QuickReplyButton(action=PostbackAction(label="莊", data="B")),
        QuickReplyButton(action=PostbackAction(label="閒", data="P")),
        QuickReplyButton(action=PostbackAction(label="和", data="T")),
        QuickReplyButton(action=PostbackAction(label="開始分析", data="START")),
        QuickReplyButton(action=PostbackAction(label="結束分析", data="END")),
        QuickReplyButton(action=PostbackAction(label="返回", data="UNDO")),
    ])

@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not USE_LINE or handler is None:
        logger.warning("LINE webhook hit but LINE SDK/env not configured.")
        return "ok", 200
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        logger.error("LINE handle error: %s", e)
        return "ok", 200
    return "ok", 200

if USE_LINE and handler is not None:
    from linebot.models import MessageEvent, TextMessage, TextSendMessage, PostbackEvent  # type: ignore

    @handler.add(MessageEvent, message=TextMessage)
    def handle_text(event):
        uid = event.source.user_id
        USER_HISTORY.setdefault(uid, [])
        USER_READY.setdefault(uid, False)
        _get_drift_state(uid)
        sB = USER_HISTORY[uid].count("B")
        sP = USER_HISTORY[uid].count("P")
        sT = USER_HISTORY[uid].count("T")
        msg = f"請使用下方按鈕輸入：莊/閒/和。\n目前已輸入：{len(USER_HISTORY[uid])} 手（莊{sB} / 閒{sP} / 和{sT}）。\n按「開始分析」後才會給出下注建議。"
        line_bot_api.reply_message(
            event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()), flex_buttons_card()]
        )

    @handler.add(PostbackEvent)
    def handle_postback(event):
        uid  = event.source.user_id
        data = (event.postback.data or "").upper()
        seq  = USER_HISTORY.get(uid, [])
        ready= USER_READY.get(uid, False)

        # 控制流程
        if data == "START":
            USER_READY[uid] = True
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="🔎 已開始分析。請繼續輸入莊/閒/和，我會根據資料給出建議。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            ); return

        if data == "END":
            USER_HISTORY[uid] = []
            USER_READY[uid]   = False
            USER_DRIFT[uid]   = {'cum':0.0,'min':0.0,'cooldown':0.0}
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="✅ 已結束分析，紀錄已清空。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            ); return

        if data == "UNDO":
            if seq:
                removed = seq.pop()
                USER_HISTORY[uid] = seq
                msg = f"↩ 已返回一步（移除：{LAB_ZH.get(removed, removed)}）。\n目前 {len(seq)} 手：莊{seq.count('B')}｜閒{seq.count('P')}｜和{seq.count('T')}。"
            else:
                msg = "沒有可返回的紀錄。"
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            ); return

        # 僅接受 B/P/T
        if data not in CLASS_ORDER:
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="請用按鈕輸入（莊/閒/和），或選開始/結束分析/返回。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            ); return

        # 追加 & 落地
        history_before = "".join(seq)
        seq.append(data); USER_HISTORY[uid] = seq
        append_round_csv(uid, history_before, data)

        # 未開始：只顯示累積與 B/P/T 計數（不給建議）
        if not ready:
            sB = seq.count("B"); sP = seq.count("P"); sT = seq.count("T")
            s_tail = "".join(seq[-20:])
            msg = f"已記錄 {len(seq)} 手：{s_tail}\n目前統計：莊{sB}｜閒{sP}｜和{sT}\n按「開始分析」後才會給出下注建議。"
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            ); return

        # 已開始：PH 偵測 + 集成
        drift_now = update_ph_state(uid, seq)
        active = in_drift(uid)
        if active: consume_cooldown(uid)

        overrides = None
        if active:
            REC_W   = float(os.getenv("REC_W","0.25"))
            LONG_W  = float(os.getenv("LONG_W","0.35"))
            MKV_W   = float(os.getenv("MKV_W","0.25"))
            PRIOR_W = float(os.getenv("PRIOR_W","0.15"))
            SHORT_BOOST = float(os.getenv("PH_SHORT_BOOST","0.30"))
            LONG_CUT    = float(os.getenv("PH_LONG_CUT","0.40"))
            MKV_CUT     = float(os.getenv("PH_MKV_CUT","0.40"))
            PRIOR_KEEP  = float(os.getenv("PH_PRIOR_KEEP","1.00"))
            overrides = {
                "REC_W":   REC_W*(1.0+SHORT_BOOST),
                "LONG_W":  max(0.0, LONG_W*(1.0-LONG_CUT)),
                "MKV_W":   max(0.0, MKV_W*(1.0-MKV_CUT)),
                "PRIOR_W": PRIOR_W*PRIOR_KEEP
            }

        probs = ensemble_with_anti_stuck(seq, overrides)
        rec   = recommend_from_probs(probs)
        suffix = "（⚡偵測到路型變化，短期權重暫時提高）" if active else ""
        msg = (
            f"已解析 {len(seq)} 手\n"
            f"機率：莊 {probs[0]:.3f}｜閒 {probs[1]:.3f}｜和 {probs[2]:.3f}\n"
            f"建議：{LAB_ZH[rec]} {suffix}"
        )
        line_bot_api.reply_message(
            event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
             flex_buttons_card()]
        )

# ========= Entrypoint =========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
