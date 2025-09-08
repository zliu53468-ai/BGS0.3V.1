#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGS LINE Bot backend — v16
- Big-Road(6x20) mapping & features (col depth / wall / early-dragon)
- Conditional Markov (conditioned on LAST result)
- Break arbitration = run-hazard + wall + swing + mean-revert
- Momentum (short + conditional Markov) vs Reversion dynamic mixing
- Page-Hinkley drift gating (short-term overweight during drift)
- Tie(T) calibration across all paths
- MAX_HISTORY cap & FEAT_WIN alignment
- CSV I/O (/export) + hot reload (/reload)
- LINE buttons: 莊(紅)/閒(藍)/和(綠)/開始分析/結束分析/返回
- Before START: 只回手數與統計；START 後才回建議
"""

import os, csv, time, logging
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-backend")

# ========= Paths / ENV =========
DATA_CSV_PATH = os.getenv("DATA_LOG_PATH", "/data/logs/rounds.csv")
os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)

RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "")

RNN_PATH = os.getenv("RNN_PATH",  "/opt/models/rnn.pt")
XGB_PATH = os.getenv("XGB_PATH",  "/opt/models/xgb.json")
LGBM_PATH= os.getenv("LGBM_PATH", "/opt/models/lgbm.txt")

FEAT_WIN    = int(os.getenv("FEAT_WIN", "120"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "400"))

# ========= Constants =========
CLASS_ORDER = ("B","P","T")
LAB_ZH = {"B":"莊","P":"閒","T":"和"}
THEORETICAL_PROBS: Dict[str,float] = {"B":0.458,"P":0.446,"T":0.096}

def _cap_history(seq: List[str]) -> List[str]:
    if MAX_HISTORY>0 and len(seq)>MAX_HISTORY:
        return seq[-MAX_HISTORY:]
    return seq

def parse_history(payload) -> List[str]:
    if payload is None: return []
    out: List[str] = []
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s, str) and s.strip().upper() in CLASS_ORDER:
                out.append(s.strip().upper())
    elif isinstance(payload, str):
        for ch in payload:
            up = ch.upper()
            if up in CLASS_ORDER: out.append(up)
    return _cap_history(out)

# ========= Optional Models =========
try:
    import torch
    import torch.nn as tnn
except Exception:
    torch=None; tnn=None
try:
    import xgboost as xgb
except Exception:
    xgb=None
try:
    import lightgbm as lgb
except Exception:
    lgb=None

if tnn is not None:
    class TinyRNN(tnn.Module):
        def __init__(self, in_dim=3, hidden=16, out_dim=3):
            super().__init__()
            self.rnn = tnn.GRU(in_dim, hidden, batch_first=True)
            self.fc  = tnn.Linear(hidden, out_dim)
        def forward(self, x):
            o,_ = self.rnn(x)
            return self.fc(o[:,-1,:])
else:
    TinyRNN=None

# ========= Load / Reload =========
RNN_MODEL: Optional[Any] = None
XGB_MODEL: Optional[Any] = None
LGBM_MODEL: Optional[Any] = None

def load_models() -> None:
    global RNN_MODEL,XGB_MODEL,LGBM_MODEL
    # RNN
    if TinyRNN is not None and torch is not None and os.path.exists(RNN_PATH):
        try:
            m = TinyRNN()
            m.load_state_dict(torch.load(RNN_PATH, map_location="cpu"))
            m.eval(); RNN_MODEL = m
            logger.info("Loaded RNN: %s", RNN_PATH)
        except Exception as e:
            logger.warning("Load RNN failed: %s", e); RNN_MODEL=None
    else:
        RNN_MODEL=None
    # XGB
    if xgb is not None and XGB_PATH and os.path.exists(XGB_PATH):
        try:
            booster = xgb.Booster(); booster.load_model(XGB_PATH)
            XGB_MODEL = booster
            logger.info("Loaded XGB: %s", XGB_PATH)
        except Exception as e:
            logger.warning("Load XGB failed: %s", e); XGB_MODEL=None
    else:
        XGB_MODEL=None
    # LGBM
    if lgb is not None and LGBM_PATH and os.path.exists(LGBM_PATH):
        try:
            booster = lgb.Booster(model_file=LGBM_PATH)
            LGBM_MODEL = booster
            logger.info("Loaded LGBM: %s", LGBM_PATH)
        except Exception as e:
            logger.warning("Load LGBM failed: %s", e); LGBM_MODEL=None
    else:
        LGBM_MODEL=None

load_models()

# ========= Tie calibration =========
def exp_decay_freq(seq: List[str], gamma: float=None) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    if gamma is None: gamma=float(os.getenv("EW_GAMMA","0.985"))
    wB=wP=wT=0.0; w=1.0
    for r in reversed(seq):
        if r=="B": wB+=w
        elif r=="P": wP+=w
        else: wT+=w
        w*=gamma
    a=float(os.getenv("LAPLACE","0.5"))
    wB+=a; wP+=a; wT+=a
    S=wB+wP+wT
    return [wB/S,wP/S,wT/S]

def _estimate_tie_prob(seq: List[str]) -> float:
    prior = THEORETICAL_PROBS["T"]
    longT = exp_decay_freq(seq)[2]
    w     = float(os.getenv("T_BLEND","0.5"))
    lo    = float(os.getenv("T_MIN","0.03"))
    hi    = float(os.getenv("T_MAX","0.18"))
    p = (1-w)*prior + w*longT
    return max(lo, min(hi, p))

def _merge_bp_with_t(bp: List[float], pT: float) -> List[float]:
    b,p = float(bp[0]), float(bp[1])
    s = max(1e-12, b+p)
    b/=s; p/=s
    scale = 1.0 - pT
    return [b*scale, p*scale, pT]

# ========= Single-model inference =========
def rnn_predict(seq: List[str]) -> Optional[List[float]]:
    if RNN_MODEL is None or torch is None or not seq: return None
    try:
        def oh(x:str): return [1 if x==c else 0 for c in CLASS_ORDER]
        x = torch.tensor([[oh(ch) for ch in seq[-FEAT_WIN:]]], dtype=torch.float32)
        with torch.no_grad():
            logits = RNN_MODEL(x)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        return [float(v) for v in probs]
    except Exception as e:
        logger.warning("RNN infer fail: %s", e); return None

def _vec_from_seq(seq: List[str]) -> List[float]:
    vec: List[float] = []
    for lab in seq[-FEAT_WIN:]:
        vec.extend([1.0 if lab==c else 0.0 for c in CLASS_ORDER])
    need = FEAT_WIN*3 - len(vec)
    if need>0: vec = [0.0]*need + vec
    return vec

def xgb_predict(seq: List[str]) -> Optional[List[float]]:
    if XGB_MODEL is None or not seq: return None
    try:
        import numpy as np
        d = xgb.DMatrix(np.array([_vec_from_seq(seq)], dtype=float))
        prob = XGB_MODEL.predict(d)[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            return _merge_bp_with_t([float(prob[0]), float(prob[1])], _estimate_tie_prob(seq))
        return None
    except Exception as e:
        logger.warning("XGB infer fail: %s", e); return None

def lgbm_predict(seq: List[str]) -> Optional[List[float]]:
    if LGBM_MODEL is None or not seq: return None
    try:
        prob = LGBM_MODEL.predict([_vec_from_seq(seq)])[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            return _merge_bp_with_t([float(prob[0]), float(prob[1])], _estimate_tie_prob(seq))
        return None
    except Exception as e:
        logger.warning("LGBM infer fail: %s", e); return None

# ========= Utils =========
def norm(v: List[float]) -> List[float]:
    s=sum(v); s=s if s>1e-12 else 1.0
    return [max(0.0,x)/s for x in v]

def blend(a: List[float], b: List[float], w: float) -> List[float]:
    return [(1-w)*a[i] + w*b[i] for i in range(3)]

def temperature_scale(p: List[float], tau: float) -> List[float]:
    if tau<=1e-6: return p
    ex=[pow(max(pi,1e-9), 1.0/tau) for pi in p]; s=sum(ex)
    return [e/s for e in ex]

# ========= Big-Road 6x20 =========
def features_like_early_dragon(seq: List[str]) -> bool:
    k=min(6, len(seq))
    if k<4: return False
    tail=seq[-k:]
    most=max(tail.count("B"), tail.count("P"))
    return (most>=k-1)

def map_to_big_road(seq: List[str], rows:int=6, cols:int=20) -> Tuple[List[List[str]], Dict[str,Any]]:
    grid=[["" for _ in range(cols)] for _ in range(rows)]
    if not seq:
        return grid, {"cur_run":0,"col_depth":0,"blocked":False,"r":0,"c":0,"early_dragon_hint":False}

    r=c=0; last=None
    for ch in seq:
        if last is None:
            grid[r][c]=ch; last=ch; continue
        if ch==last:
            if r+1<rows and grid[r+1][c]=="":
                r+=1
            else:
                c=min(cols-1, c+1)
                while c<cols and grid[r][c]!="":
                    c=min(cols-1, c+1)
                if c>=cols: c=cols-1
        else:
            last=ch
            c=min(cols-1, c+1)
            r=0
            while c<cols and grid[r][c]!="":
                c=min(cols-1, c+1)
            if c>=cols: c=cols-1
        if grid[r][c]=="": grid[r][c]=ch

    cur_depth=0
    for rr in range(rows):
        if grid[rr][c]!="": cur_depth=rr+1
    blocked = (cur_depth>=rows) or (r==rows-1) or (r+1<rows and grid[r+1][c]!="" and last==grid[r][c])

    def last_run_len(s: List[str])->int:
        if not s: return 0
        ch=s[-1]; i=len(s)-2; n=1
        while i>=0 and s[i]==ch: n+=1; i-=1
        return n

    feats = {
        "cur_run": last_run_len(seq),
        "col_depth": cur_depth,
        "blocked": blocked,
        "r": r, "c": c,
        "early_dragon_hint": (cur_depth>=3 and features_like_early_dragon(seq))
    }
    return grid, feats

# ========= Short/Mid =========
def recent_freq(seq: List[str], win: int) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    s = seq[-win:] if win>0 else seq
    a = float(os.getenv("LAPLACE","0.5"))
    nB=s.count("B")+a; nP=s.count("P")+a; nT=s.count("T")+a
    tot=max(1,len(s))+3*a
    return [nB/tot,nP/tot,nT/tot]

# ========= Conditional Markov (conditioned on LAST) =========
def markov_conditional(seq: List[str], decay: float=None) -> List[float]:
    if len(seq)<2: return [1/3,1/3,1/3]
    if decay is None: decay=float(os.getenv("MKV_DECAY","0.98"))
    idx={"B":0,"P":1,"T":2}
    last=seq[-1]
    row=[0.0,0.0,0.0]; w=1.0
    # iterate from the end for stronger recency effect
    for a,b in zip(reversed(seq[:-1]), reversed(seq[1:])):
        if a==last:
            row[idx[b]] += w
        w *= decay
    a=float(os.getenv("MKV_LAPLACE","0.5"))
    row=[x+a for x in row]; S=sum(row)
    return [x/S for x in row]

# ========= Regime boosts (mild) =========
def is_zigzag(seq: List[str], k:int=6)->bool:
    s=seq[-k:] if len(seq)>=k else seq
    if len(s)<4: return False
    alt = all(s[i]!=s[i-1] for i in range(1,len(s)))
    if alt: return True
    if len(s)%2==0:
        pairs=[s[i:i+2] for i in range(0,len(s),2)]
        if all(len(p)==2 and p[0]==p[1] for p in pairs):
            if all(pairs[i][0]!=pairs[i-1][0] for i in range(1,len(pairs))):
                return True
    return False

def regime_boosts(seq: List[str], feat: Dict[str,Any]) -> List[float]:
    if not seq: return [1.0,1.0,1.0]
    b=[1.0,1.0,1.0]
    last=seq[-1]
    # last-run length
    n=1; i=len(seq)-2
    while i>=0 and seq[i]==last: n+=1; i-=1
    DRAGON_TH    = int(os.getenv("BOOST_DRAGON_LEN","4"))
    BOOST_DRAGON = float(os.getenv("BOOST_DRAGON","1.08"))
    BOOST_EARLYD = float(os.getenv("BOOST_EARLY_DRAGON","1.04"))
    BOOST_ALT    = float(os.getenv("BOOST_ALT","1.05"))
    BOOST_T      = float(os.getenv("BOOST_T","1.02"))

    if feat.get("early_dragon_hint",False) and not feat.get("blocked",False):
        if last=="B": b[0]*=BOOST_EARLYD
        elif last=="P": b[1]*=BOOST_EARLYD
    if n>=DRAGON_TH and not feat.get("blocked",False):
        if last=="B": b[0]*=BOOST_DRAGON
        elif last=="P": b[1]*=BOOST_DRAGON
    if is_zigzag(seq,6):
        if last=="B": b[1]*=BOOST_ALT
        elif last=="P": b[0]*=BOOST_ALT
    if exp_decay_freq(seq)[2] > THEORETICAL_PROBS["T"]*1.15:
        b[2]*=BOOST_T
    return b

def _apply_boosts_and_norm(probs: List[float], boosts: List[float]) -> List[float]:
    p=[max(1e-12, probs[i]*boosts[i]) for i in range(3)]
    s=sum(p)
    return [x/s for x in p]

# ========= Page-Hinkley =========
def js_divergence(p: List[float], q: List[float]) -> float:
    import math
    eps=1e-12
    m=[(p[i]+q[i])/2.0 for i in range(3)]
    def _kl(a,b): return sum((ai+eps)*math.log((ai+eps)/(bi+eps)) for ai,bi in zip(a,b))
    return 0.5*_kl(p,m)+0.5*_kl(q,m)

USER_DRIFT: Dict[str, Dict[str,float]] = {}
def _get_drift_state(uid: str) -> Dict[str,float]:
    st = USER_DRIFT.get(uid)
    if st is None:
        st={'cum':0.0,'min':0.0,'cooldown':0.0}; USER_DRIFT[uid]=st
    return st

def update_ph_state(uid: str, seq: List[str]) -> bool:
    if not seq: return False
    st = _get_drift_state(uid)
    REC_WIN   = int(os.getenv("REC_WIN_FOR_PH","12"))
    p_short   = recent_freq(seq, REC_WIN)
    p_long    = exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.985")))
    D_t       = js_divergence(p_short, p_long)
    PH_DELTA  = float(os.getenv("PH_DELTA","0.005"))
    PH_LAMBDA = float(os.getenv("PH_LAMBDA","0.08"))
    DRIFT_STEPS=float(os.getenv("DRIFT_STEPS","5"))
    st['cum'] += (D_t - PH_DELTA)
    st['min']  = min(st['min'], st['cum'])
    if (st['cum'] - st['min']) > PH_LAMBDA:
        st['cum']=0.0; st['min']=0.0; st['cooldown']=float(DRIFT_STEPS)
        logger.info("[PH] drift triggered for %s: D=%.4f", uid, D_t)
        return True
    return False

def in_drift(uid: str)->bool:
    return _get_drift_state(uid)['cooldown']>0.0

def consume_cooldown(uid: str)->None:
    st=_get_drift_state(uid)
    if st['cooldown']>0: st['cooldown']=max(0.0, st['cooldown']-1.0)

# ========= Ensemble with Arbitration =========
def ensemble_with_anti_stuck(seq: List[str], weight_overrides: Optional[Dict[str,float]]=None) -> List[float]:
    # Base models (rule + optional ML)
    rule  = [THEORETICAL_PROBS["B"], THEORETICAL_PROBS["P"], THEORETICAL_PROBS["T"]]
    pr_rnn = rnn_predict(seq)
    pr_xgb = xgb_predict(seq)
    pr_lgb = lgbm_predict(seq)

    w_rule = float(os.getenv("RULE_W","0.30"))
    w_rnn  = float(os.getenv("RNN_W","0.25"))
    w_xgb  = float(os.getenv("XGB_W","0.20"))
    w_lgb  = float(os.getenv("LGBM_W","0.25"))

    total = w_rule + (w_rnn if pr_rnn else 0) + (w_xgb if pr_xgb else 0) + (w_lgb if pr_lgb else 0)
    base  = [w_rule*rule[i] for i in range(3)]
    if pr_rnn: base=[base[i]+w_rnn*pr_rnn[i] for i in range(3)]
    if pr_xgb: base=[base[i]+w_xgb*pr_xgb[i] for i in range(3)]
    if pr_lgb: base=[base[i]+w_lgb*pr_lgb[i] for i in range(3)]
    probs=[b/max(total,1e-9) for b in base]

    # ==== Reversion path (hazard + wall + swing + mean-revert) ====
    grid, feat = map_to_big_road(seq)
    seq_bp = [x for x in seq if x in ("B","P")]

    def _run_hist(bp: List[str]) -> Dict[int,int]:
        h: Dict[int,int] = {}
        if not bp: return h
        cur = 1
        for i in range(1, len(bp)):
            if bp[i]==bp[i-1]: cur+=1
            else:
                h[cur]=h.get(cur,0)+1
                cur=1
        h[cur]=h.get(cur,0)+1
        return h

    hist = _run_hist(seq_bp)

    def _last_run_len(bp: List[str]) -> int:
        if not bp: return 0
        n=1
        for i in range(len(bp)-2,-1,-1):
            if bp[i]==bp[-1]: n+=1
            else: break
        return n

    L = _last_run_len(seq_bp)
    last = seq_bp[-1] if seq_bp else ""
    opp  = "P" if last=="B" else ("B" if last=="P" else "")

    wall = 1.0 if feat.get("blocked", False) else 0.0

    def _hazard(L:int, hist:Dict[int,int]) -> float:
        if L<=0: return 0.0
        a = float(os.getenv("HZD_ALPHA","0.6"))
        ge = sum(v for k,v in hist.items() if k>=L)
        end= hist.get(L,0)
        return (end + a) / (ge + a*max(1, len(hist)))

    hz = _hazard(L, hist)

    SW_S  = int(os.getenv("SWING_SHORT","6"))
    SW_TH = float(os.getenv("SWING_EDGE","0.62"))
    short = seq_bp[-SW_S:] if len(seq_bp)>=SW_S else seq_bp
    swing = 0.0
    if len(short)>=4:
        cnt_opp = sum(1 for x in short if x==opp)
        swing = 1.0 if (cnt_opp/max(1,len(short)))>=SW_TH else 0.0

    b = seq_bp.count("B"); p = seq_bp.count("P")
    tot=max(1,b+p)
    diff=abs(b-p)/tot
    mr_side = "P" if b>p else ("B" if p>b else "")
    mr = diff if mr_side==opp else 0.0

    W_HZ   = float(os.getenv("W_HAZARD","0.55"))
    W_WALL = float(os.getenv("W_WALL","0.25"))
    W_SW   = float(os.getenv("W_SWING","0.12"))
    W_MR   = float(os.getenv("W_MEANREV","0.08"))
    p_break = max(0.0, min(1.0, W_HZ*hz + W_WALL*wall + W_SW*swing + W_MR*mr))

    EPS_BRK = float(os.getenv("EPS_BREAK_EPS","0.02"))
    if last=="B":
        p_rev_bp = [1.0 - p_break*(1.0-EPS_BRK), p_break*(1.0-EPS_BRK)]
    elif last=="P":
        p_rev_bp = [p_break*(1.0-EPS_BRK), 1.0 - p_break*(1.0-EPS_BRK)]
    else:
        p_rev_bp = [0.5,0.5]
    pT_est = _estimate_tie_prob(seq)
    p_rev  = _merge_bp_with_t([p_rev_bp[0], p_rev_bp[1]], pT_est)

    # ==== Momentum path (short + conditional Markov) ====
    W_S = int(os.getenv("WIN_SHORT","6"))
    W_M = int(os.getenv("WIN_MID","12"))
    p_short = blend(recent_freq(seq, W_S), recent_freq(seq, W_M), 0.5)
    p_mkv   = markov_conditional(seq, float(os.getenv("MKV_DECAY","0.98")))
    p_momBP = [(p_short[0]+p_mkv[0])*0.5, (p_short[1]+p_mkv[1])*0.5]
    p_mom   = _merge_bp_with_t(p_momBP, pT_est)

    # 仲裁：以 p_break 混合 momentum / reversion
    p_mix = blend(p_mom, p_rev, p_break)

    # Long-term & Prior blending
    p_long = exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.985")))
    PRIOR_W= float(os.getenv("PRIOR_W","0.15"))
    LONG_W = float(os.getenv("LONG_W","0.25"))
    REC_W  = float(os.getenv("REC_W","0.25"))
    MKV_W  = float(os.getenv("MKV_W","0.25"))  # 用於 drift 覆蓋時

    if weight_overrides:
        REC_W  = weight_overrides.get("REC_W",  REC_W)
        LONG_W = weight_overrides.get("LONG_W", LONG_W)
        PRIOR_W= weight_overrides.get("PRIOR_W",PRIOR_W)
        MKV_W  = weight_overrides.get("MKV_W",  MKV_W)  # ← 尊重覆蓋

    probs = blend(probs, p_mix,  REC_W)
    probs = blend(probs, p_long, LONG_W)
    probs = blend(probs, [THEORETICAL_PROBS["B"],THEORETICAL_PROBS["P"],THEORETICAL_PROBS["T"]], PRIOR_W)

    # Safety caps + temperature
    EPS = float(os.getenv("EPSILON_FLOOR","0.06"))
    CAP = float(os.getenv("MAX_CAP","0.86"))
    TAU = float(os.getenv("TEMP","1.06"))
    probs=[min(CAP, max(EPS, p)) for p in probs]
    probs=norm(probs); probs=temperature_scale(probs, TAU)

    # Mild regime boosts
    boosts = regime_boosts(seq, feat)
    probs  = _apply_boosts_and_norm(probs, boosts)
    return norm(probs)

def recommend_from_probs(p: List[float]) -> str:
    return CLASS_ORDER[p.index(max(p))]

# ========= Health / Predict / Export / Reload =========
@app.route("/", methods=["GET"])
def index(): return "ok"

@app.route("/health", methods=["GET"])
def health(): return jsonify(status="healthy", version="v16")

@app.route("/healthz", methods=["GET"])
def healthz(): return jsonify(status="healthy")

@app.route("/predict", methods=["POST"])
def predict():
    data: Dict[str,Any] = request.get_json(silent=True) or {}
    seq = parse_history(data.get("history"))
    probs = ensemble_with_anti_stuck(seq)
    rec   = recommend_from_probs(probs)
    return jsonify({
        "history_len": len(seq),
        "probabilities": {"B":probs[0],"P":probs[1],"T":probs[2]},
        "recommendation": rec
    })

def append_round_csv(user_id: str, history_before: str, label: str) -> None:
    try:
        os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)
        with open(DATA_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([user_id, int(time.time()), history_before, label])
    except Exception as e:
        logger.warning("append csv fail: %s", e)

@app.route("/export", methods=["GET"])
def export_csv():
    n=int(request.args.get("n","1000"))
    rows: List[List[str]]=[]
    try:
        if os.path.exists(DATA_CSV_PATH):
            with open(DATA_CSV_PATH,"r",encoding="utf-8") as f:
                data=list(csv.reader(f))
                rows=data[-n:] if n>0 else data
    except Exception as e:
        logger.warning("export read fail: %s", e); rows=[]
    out="user_id,ts,history_before,label\n"+"\n".join([",".join(r) for r in rows])
    return Response(out, mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=rounds.csv"})

@app.route("/reload", methods=["POST"])
def reload_models():
    token = request.headers.get("X-Reload-Token","") or request.args.get("token","")
    if not RELOAD_TOKEN or token != RELOAD_TOKEN:
        return jsonify(ok=False,error="unauthorized"),401
    load_models()
    return jsonify(ok=True, rnn=bool(RNN_MODEL), xgb=bool(XGB_MODEL), lgbm=bool(LGBM_MODEL))

# ========= LINE Webhook =========
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET","")

USE_LINE=False
try:
    from linebot import LineBotApi, WebhookHandler  # type: ignore
    from linebot.models import (  # type: ignore
        MessageEvent, TextMessage, TextSendMessage,
        PostbackEvent, PostbackAction, FlexSendMessage,
        QuickReply, QuickReplyButton
    )
    USE_LINE = bool(LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET)
except Exception as e:
    logger.warning("LINE SDK/env not ready: %s", e)
    USE_LINE=False

if USE_LINE:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler      = WebhookHandler(LINE_CHANNEL_SECRET)
else:
    line_bot_api=None; handler=None

USER_HISTORY: Dict[str,List[str]] = {}
USER_READY:   Dict[str,bool]      = {}

def flex_buttons_card() -> 'FlexSendMessage':
    contents={
        "type":"bubble",
        "body":{
            "type":"box","layout":"vertical","spacing":"md",
            "contents":[
                {"type":"text","text":"🤖 請輸入莊/閒/和；按「開始分析」後才會給出建議。","wrap":True,"size":"sm","color":"#555555"},
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
    return FlexSendMessage(alt_text="輸入/開始/返回", contents=contents)

def quick_reply_bar():
    return QuickReply(items=[
        QuickReplyButton(action=PostbackAction(label="莊",data="B")),
        QuickReplyButton(action=PostbackAction(label="閒",data="P")),
        QuickReplyButton(action=PostbackAction(label="和",data="T")),
        QuickReplyButton(action=PostbackAction(label="開始分析",data="START")),
        QuickReplyButton(action=PostbackAction(label="結束分析",data="END")),
        QuickReplyButton(action=PostbackAction(label="返回",data="UNDO")),
    ])

@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not USE_LINE or handler is None:
        logger.warning("LINE webhook hit but LINE SDK/env not configured.")
        return "ok",200
    signature = request.headers.get("X-Line-Signature","")
    body      = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        logger.error("LINE handle error: %s", e)
        return "ok",200
    return "ok",200

if USE_LINE and handler is not None:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_text(event):
        uid=event.source.user_id
        USER_HISTORY.setdefault(uid, [])
        USER_READY.setdefault(uid, False)
        sB=USER_HISTORY[uid].count("B"); sP=USER_HISTORY[uid].count("P"); sT=USER_HISTORY[uid].count("T")
        msg=(f"請使用按鈕輸入：莊/閒/和。\n"
             f"目前已輸入 {len(USER_HISTORY[uid])} 手（莊{sB} / 閒{sP} / 和{sT}）。\n"
             "按「開始分析」後才會給出下注建議；如需核對可用「返回」。")
        line_bot_api.reply_message(event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
             flex_buttons_card()])

    @handler.add(PostbackEvent)
    def handle_postback(event):
        uid  = event.source.user_id
        data = (event.postback.data or "").upper()
        seq  = USER_HISTORY.get(uid, [])
        ready= USER_READY.get(uid, False)

        if data=="START":
            USER_READY[uid]=True
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text="🔎 已開始分析。之後每輸入一手我會回覆機率與建議。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        if data=="END":
            USER_HISTORY[uid]=[]
            USER_READY[uid]=False
            USER_DRIFT[uid]={'cum':0.0,'min':0.0,'cooldown':0.0}
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text="✅ 已結束分析並清空紀錄。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        if data=="UNDO":
            if seq:
                removed = seq.pop()
                USER_HISTORY[uid]=seq
                msg=f"↩ 已返回一步（移除：{LAB_ZH.get(removed, removed)}）。\n目前 {len(seq)} 手：莊{seq.count('B')}｜閒{seq.count('P')}｜和{seq.count('T')}。"
            else:
                msg="沒有可返回的紀錄。"
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        if data not in CLASS_ORDER:
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text="請用按鈕輸入（莊/閒/和），或選開始/結束分析/返回。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        # append + log
        hist_before = "".join(seq)
        seq.append(data); USER_HISTORY[uid]=_cap_history(seq)
        append_round_csv(uid, hist_before, data)

        if not ready:
            sB=seq.count("B"); sP=seq.count("P"); sT=seq.count("T")
            s_tail="".join(seq[-20:])
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text=f"已記錄 {len(seq)} 手：{s_tail}\n統計：莊{sB}｜閒{sP}｜和{sT}\n補齊後請按「開始分析」。",
                                 quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        # drift gating
        drift_now = update_ph_state(uid, seq)
        active = in_drift(uid)
        if active: consume_cooldown(uid)

        overrides=None
        if active:
            overrides={"REC_W":0.32, "LONG_W":0.20, "PRIOR_W":0.15, "MKV_W":0.33}

        probs = ensemble_with_anti_stuck(seq, overrides)
        rec   = recommend_from_probs(probs)
        suffix = "（⚡偵測到路型變化，已暫時提高短期/馬可夫權重）" if active else ""
        line_bot_api.reply_message(event.reply_token,
            [TextSendMessage(text=f"已解析 {len(seq)} 手\n機率：莊 {probs[0]:.3f}｜閒 {probs[1]:.3f}｜和 {probs[2]:.3f}\n建議：{LAB_ZH[rec]} {suffix}",
                             quick_reply=quick_reply_bar()),
             flex_buttons_card()])

# ========= Entrypoint =========
if __name__=="__main__":
    port=int(os.environ.get("PORT","8080"))
    app.run(host="0.0.0.0", port=port)
