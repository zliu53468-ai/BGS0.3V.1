#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGS backend — v16.1
- Big-Road 6x20 繪製 + 牆阻(wall)/柱深(col_depth)/早龍偵測
- 條件式馬可夫 (以最後一手為條件)
- 模式偵測：齊腳/單雙跳/2連反覆(2-2-2-2)、龍斷 Hazard、疲勞
- Page-Hinkley 漂移：可覆寫 REC_W/LONG_W/MKV_W/PRIOR_W
- 三模型(RNN/XGB/LGBM)可選；缺時自動降階
- /predict 入口；可直接部署
"""

import os, csv, time, logging
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-backend")

# =================== 路徑/環境 ===================
DATA_CSV_PATH = os.getenv("DATA_LOG_PATH", "/data/logs/rounds.csv")
os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)

RNN_PATH = os.getenv("RNN_PATH", "/data/models/rnn.pt")
XGB_PATH = os.getenv("XGB_PATH", "/data/models/xgb.json")
LGBM_PATH = os.getenv("LGBM_PATH", "/data/models/lgbm.txt")

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "240"))

# =================== 常數 ===================
CLASS_ORDER = ("B","P","T")
LAB_ZH = {"B":"莊","P":"閒","T":"和"}
THEORETICAL_PROBS = {"B":0.458,"P":0.446,"T":0.096}

# =================== 解析 ===================
def parse_history(payload)->List[str]:
    seq: List[str] = []
    if payload is None: return seq
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s, str) and s.strip().upper() in CLASS_ORDER:
                seq.append(s.strip().upper())
    elif isinstance(payload, str):
        for ch in payload:
            up = ch.upper()
            if up in CLASS_ORDER: seq.append(up)
    # 限長，避免超長歷史拖慢
    return seq[-MAX_HISTORY:] if len(seq) > MAX_HISTORY else seq

# =================== 可選模型載入 ===================
try:
    import torch; import torch.nn as tnn
except Exception:
    torch = None; tnn = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

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
    TinyRNN = None

RNN_MODEL: Optional[Any] = None
XGB_MODEL: Optional[Any] = None
LGBM_MODEL: Optional[Any] = None

def load_models() -> None:
    global RNN_MODEL, XGB_MODEL, LGBM_MODEL
    # RNN
    if TinyRNN and torch and os.path.exists(RNN_PATH):
        try:
            m = TinyRNN()
            m.load_state_dict(torch.load(RNN_PATH, map_location="cpu"))
            m.eval(); RNN_MODEL = m
            logger.info("Loaded RNN %s", RNN_PATH)
        except Exception as e:
            logger.warning("Load RNN fail: %s", e); RNN_MODEL=None
    # XGB
    if xgb and os.path.exists(XGB_PATH):
        try:
            booster = xgb.Booster(); booster.load_model(XGB_PATH)
            XGB_MODEL = booster; logger.info("Loaded XGB %s", XGB_PATH)
        except Exception as e:
            logger.warning("Load XGB fail: %s", e); XGB_MODEL=None
    # LGBM
    if lgb and os.path.exists(LGBM_PATH):
        try:
            booster = lgb.Booster(model_file=LGBM_PATH)
            LGBM_MODEL = booster; logger.info("Loaded LGBM %s", LGBM_PATH)
        except Exception as e:
            logger.warning("Load LGBM fail: %s", e); LGBM_MODEL=None

load_models()

# =================== 公用工具 ===================
def norm(v: List[float]) -> List[float]:
    s = sum(v); s = s if s > 1e-12 else 1.0
    return [max(0.0, x)/s for x in v]

def blend(a: List[float], b: List[float], w: float) -> List[float]:
    return [(1-w)*a[i] + w*b[i] for i in range(3)]

def temperature_scale(p: List[float], tau: float) -> List[float]:
    if tau <= 1e-6: return p
    ex = [pow(max(pi,1e-9), 1.0/tau) for pi in p]
    s  = sum(ex)
    return [e/s for e in ex]

# =================== Tie 校正 ===================
def exp_decay_freq(seq: List[str], gamma: float=None) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    if gamma is None: gamma = float(os.getenv("EW_GAMMA","0.96"))
    wB=wP=wT=0.0; w=1.0
    for r in reversed(seq):
        if r=="B": wB+=w
        elif r=="P": wP+=w
        else: wT+=w
        w*=gamma
    a=float(os.getenv("LAPLACE","0.5"))
    wB+=a; wP+=a; wT+=a; S=wB+wP+wT
    return [wB/S,wP/S,wT/S]

def _estimate_tie_prob(seq: List[str]) -> float:
    prior_T = THEORETICAL_PROBS["T"]
    long_T  = exp_decay_freq(seq)[2]
    w       = float(os.getenv("T_BLEND","0.5"))
    floor_T = float(os.getenv("T_MIN","0.03"))
    cap_T   = float(os.getenv("T_MAX","0.18"))
    pT = (1-w)*prior_T + w*long_T
    return max(floor_T, min(cap_T, pT))

def _merge_bp_with_t(bp: List[float], pT: float) -> List[float]:
    b,p = float(bp[0]), float(bp[1])
    s = max(1e-12, b+p)
    b/=s; p/=s
    sc = 1.0 - pT
    return [b*sc, p*sc, pT]

# =================== Big-Road 6x20 ===================
def map_to_big_road(seq: List[str], rows:int=6, cols:int=20) -> Tuple[List[List[str]], Dict[str,Any]]:
    """同色向下；到底或被占→往右；換色→右移，從最上放"""
    grid=[["" for _ in range(cols)] for _ in range(rows)]
    if not seq: return grid, {"cur_run":0,"col_depth":0,"blocked":False,"r":0,"c":0}
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
        else:
            last=ch
            c=min(cols-1, c+1); r=0
            while c<cols and grid[r][c]!="":
                c=min(cols-1, c+1)
        if grid[r][c]=="": grid[r][c]=ch
    # 深度/牆阻
    depth=0
    for rr in range(rows):
        if grid[rr][c]!="": depth=rr+1
    blocked = (depth>=rows) or (r==rows-1) or (r+1<rows and grid[r+1][c]!="" and last==grid[r][c])
    # 當前 run 長
    def last_run_len(s: List[str])->int:
        if not s: return 0
        ch=s[-1]; i=len(s)-2; n=1
        while i>=0 and s[i]==ch: n+=1; i-=1
        return n
    feats={"cur_run": last_run_len(seq), "col_depth": depth, "blocked": blocked, "r": r, "c": c}
    return grid, feats

# =================== 模式偵測 ===================
def bp_only(seq: List[str]) -> List[str]:
    return [x for x in seq if x in ("B","P")]

def runs_bp(seq_bp: List[str]) -> Tuple[List[int], List[str]]:
    if not seq_bp: return [],[]
    lens=[1]; cols=[seq_bp[0]]
    for i in range(1,len(seq_bp)):
        if seq_bp[i]==seq_bp[i-1]: lens[-1]+=1
        else: lens.append(1); cols.append(seq_bp[i])
    return lens, cols

def detect_qijiao(seq_bp: List[str]) -> bool:
    lens,_ = runs_bp(seq_bp)
    if len(lens) < 4: return False
    return len(set(lens[-4:])) == 1

def detect_single_double(seq_bp: List[str]) -> bool:
    lens,_ = runs_bp(seq_bp)
    if not lens or len(lens) < 6: return False
    return all(L in (1,2) for L in lens[-6:])

def detect_pair2(seq_bp: List[str]) -> bool:
    lens,_ = runs_bp(seq_bp)
    return lens[-4:] == [2,2,2,2] if len(lens)>=4 else False

def hazard_from_hist(L:int, hist:Dict[int,int]) -> float:
    a = float(os.getenv("HZD_ALPHA","0.5"))
    ge = sum(v for k,v in hist.items() if k>=L)
    end= hist.get(L,0)
    return (end + a) / (ge + a*max(1,len(hist)))

def run_hist(seq_bp: List[str]) -> Dict[int,int]:
    h: Dict[int,int] = {}
    if not seq_bp: return h
    cur=1
    for i in range(1,len(seq_bp)):
        if seq_bp[i]==seq_bp[i-1]: cur+=1
        else: h[cur]=h.get(cur,0)+1; cur=1
    h[cur]=h.get(cur,0)+1
    return h

def mean_revert_score(seq: List[str]) -> Tuple[float,str]:
    b = seq.count("B"); p = seq.count("P"); tot=max(1,b+p)
    diff=(b-p)/tot
    side = "P" if diff>0 else ("B" if diff<0 else "")
    return abs(diff), side

# =================== 條件式馬可夫 ===================
def markov_next_prob(seq: List[str], decay: float=None) -> List[float]:
    if not seq or len(seq)<2: return [1/3,1/3,1/3]
    if decay is None: decay=float(os.getenv("MKV_DECAY","0.98"))
    idx={"B":0,"P":1,"T":2}; last=seq[-1]
    out=[0.0,0.0,0.0]; w=1.0
    for i in range(len(seq)-1):
        if seq[i]==last:
            out[idx[seq[i+1]]] += w
        w*=decay
    a=float(os.getenv("MKV_LAPLACE","0.5"))
    out=[x+a for x in out]; S=sum(out)
    return [x/S for x in out]

# =================== Page-Hinkley ===================
def js_divergence(p: List[float], q: List[float]) -> float:
    import math
    eps=1e-12; m=[(p[i]+q[i])/2.0 for i in range(3)]
    def _kl(a,b): return sum((ai+eps)*math.log((ai+eps)/(bi+eps)) for ai,bi in zip(a,b))
    return 0.5*_kl(p,m)+0.5*_kl(q,m)

USER_DRIFT: Dict[str, Dict[str,float]] = {}

def _get_drift_state(uid:str)->Dict[str,float]:
    st = USER_DRIFT.get(uid)
    if st is None:
        st={'cum':0.0,'min':0.0,'cooldown':0.0}; USER_DRIFT[uid]=st
    return st

def update_ph_state(uid:str, seq: List[str]) -> bool:
    if not seq: return False
    st = _get_drift_state(uid)
    REC_WIN = int(os.getenv("REC_WIN_FOR_PH","12"))
    p_short = exp_decay_freq(seq[-REC_WIN:]) if len(seq)>REC_WIN else exp_decay_freq(seq)
    p_long  = exp_decay_freq(seq)
    D_t = js_divergence(p_short, p_long)
    PH_DELTA   = float(os.getenv("PH_DELTA","0.005"))
    PH_LAMBDA  = float(os.getenv("PH_LAMBDA","0.08"))
    DRIFT_STEPS= float(os.getenv("DRIFT_STEPS","5"))
    st['cum'] += (D_t - PH_DELTA)
    st['min']  = min(st['min'], st['cum'])
    if (st['cum'] - st['min']) > PH_LAMBDA:
        st['cum']=0.0; st['min']=0.0; st['cooldown']=DRIFT_STEPS
        logger.info("[PH] drift triggered (uid=%s) D=%.4f", uid, D_t)
        return True
    return False

def in_drift(uid:str)->bool:
    return _get_drift_state(uid)['cooldown']>0.0

def consume_cooldown(uid:str)->None:
    st=_get_drift_state(uid)
    if st['cooldown']>0: st['cooldown']=max(0.0, st['cooldown']-1.0)

# =================== 單模型推論 ===================
def rnn_predict(seq: List[str]) -> Optional[List[float]]:
    if RNN_MODEL is None or torch is None or not seq: return None
    try:
        def onehot(y): return [1 if y==c else 0 for c in CLASS_ORDER]
        x = torch.tensor([[onehot(ch) for ch in seq]], dtype=torch.float32)
        with torch.no_grad():
            logits = RNN_MODEL(x)
            p = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        return [float(v) for v in p]
    except Exception as e:
        logger.warning("RNN infer fail: %s", e); return None

def xgb_predict(seq: List[str]) -> Optional[List[float]]:
    if XGB_MODEL is None or not seq: return None
    try:
        import numpy as np
        K = int(os.getenv("FEAT_WIN","20")); vec=[]
        for lab in seq[-K:]:
            vec.extend([1.0 if lab==c else 0.0 for c in CLASS_ORDER])
        need = K*3 - len(vec)
        if need>0: vec=[0.0]*need + vec
        d = xgb.DMatrix(np.array([vec], dtype=float))
        prob = XGB_MODEL.predict(d)[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]),float(prob[1]),float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            pT = _estimate_tie_prob(seq)
            return _merge_bp_with_t([float(prob[0]),float(prob[1])], pT)
        return None
    except Exception as e:
        logger.warning("XGB infer fail: %s", e); return None

def lgbm_predict(seq: List[str]) -> Optional[List[float]]:
    if LGBM_MODEL is None or not seq: return None
    try:
        K = int(os.getenv("FEAT_WIN","20")); vec=[]
        for lab in seq[-K:]:
            vec.extend([1.0 if lab==c else 0.0 for c in CLASS_ORDER])
        need = K*3 - len(vec)
        if need>0: vec=[0.0]*need + vec
        prob = LGBM_MODEL.predict([vec])[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]),float(prob[1]),float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            pT = _estimate_tie_prob(seq)
            return _merge_bp_with_t([float(prob[0]),float(prob[1])], pT)
        return None
    except Exception as e:
        logger.warning("LGBM infer fail: %s", e); return None

# =================== 集成 (含模式→加權) ===================
def ensemble_with_anti_stuck(seq: List[str], uid: str="") -> List[float]:
    # --- 基礎三路：規則 + 已訓練模型 ---
    rule  = [THEORETICAL_PROBS["B"], THEORETICAL_PROBS["P"], THEORETICAL_PROBS["T"]]
    pr_rnn = rnn_predict(seq)
    pr_xgb = xgb_predict(seq)
    pr_lgb = lgbm_predict(seq)

    w_rule = float(os.getenv("RULE_W","0.28"))
    w_rnn  = float(os.getenv("RNN_W","0.24"))
    w_xgb  = float(os.getenv("XGB_W","0.24"))
    w_lgb  = float(os.getenv("LGBM_W","0.24"))

    total = w_rule + (w_rnn if pr_rnn else 0) + (w_xgb if pr_xgb else 0) + (w_lgb if pr_lgb else 0)
    base = [w_rule*rule[i] for i in range(3)]
    if pr_rnn: base=[base[i]+w_rnn*pr_rnn[i] for i in range(3)]
    if pr_xgb: base=[base[i]+w_xgb*pr_xgb[i] for i in range(3)]
    if pr_lgb: base=[base[i]+w_lgb*pr_lgb[i] for i in range(3)]
    probs = [b/max(total,1e-9) for b in base]

    # --- Big-Road 與偵測 ---
    grid, feat = map_to_big_road(seq)
    seq_bp = bp_only(seq)
    lens, cols = runs_bp(seq_bp)
    cur_run = lens[-1] if lens else 1
    last    = seq[-1] if seq else ""

    # 龍斷 Hazard / 疲勞
    hist = run_hist(seq_bp)
    hz = hazard_from_hist(cur_run, hist) if cur_run>0 else 0.0
    fatigue = min(1.0, max(0.0, (cur_run-4)/8.0))  # 4+ 之後逐步疲勞

    # 模式旗標
    flag_qj  = detect_qijiao(seq_bp)
    flag_sd  = detect_single_double(seq_bp)
    flag_p22 = detect_pair2(seq_bp)

    # --- Momentum 路徑 (短窗 + 條件馬可夫) ---
    W_S = int(os.getenv("WIN_SHORT","6"))
    W_M = int(os.getenv("WIN_MID","12"))
    p_short = exp_decay_freq(seq[-W_S:] if len(seq)>=W_S else seq)
    p_mid   = exp_decay_freq(seq[-W_M:] if len(seq)>=W_M else seq)
    p_sm    = blend(p_short, p_mid, 0.5)
    p_mkv   = markov_next_prob(seq, float(os.getenv("MKV_DECAY","0.98")))
    MOM_W_MKV = float(os.getenv("MOM_W_MKV","0.50"))
    p_momentum = blend(p_sm, p_mkv, MOM_W_MKV)

    # --- Reversion 路徑 (牆阻/龍斷/齊腳/均值回歸) ---
    opp = "P" if last=="B" else ("B" if last=="P" else "")
    eps = 0.02
    if opp=="B":  p_rev_bp = [1.0-eps, eps]   # 反打 B
    elif opp=="P":p_rev_bp = [eps, 1.0-eps]   # 反打 P
    else:         p_rev_bp = [0.5, 0.5]
    # reversion 強度：牆阻 + 龍斷 + 疲勞 + 齊腳/單雙跳/2-2-2-2 + 均值回歸
    wall   = 1.0 if feat.get("blocked", False) else 0.0
    mr_sc, mr_side = mean_revert_score(seq)
    alpha_hz   = float(os.getenv("W_HAZARD","0.45"))
    alpha_wall = float(os.getenv("W_WALL","0.30"))
    alpha_fat  = float(os.getenv("W_FATIGUE","0.20"))
    alpha_qj   = float(os.getenv("W_QIJIAO","0.15"))
    alpha_sd   = float(os.getenv("W_SINGLEDOUBLE","0.15"))
    alpha_p22  = float(os.getenv("W_PAIR22","0.15"))
    alpha_mr   = float(os.getenv("W_MEANREV","0.20"))

    rev_strength = 0.0
    rev_strength += alpha_hz*hz + alpha_wall*wall + alpha_fat*fatigue
    if flag_qj:  rev_strength += alpha_qj
    if flag_sd:  rev_strength += alpha_sd
    if flag_p22: rev_strength += alpha_p22
    rev_strength += alpha_mr*mr_sc
    rev_strength = max(0.0, min(1.0, rev_strength))

    pT = _estimate_tie_prob(seq)
    p_mom = _merge_bp_with_t([p_momentum[0], p_momentum[1]], pT)
    p_rev = _merge_bp_with_t([p_rev_bp[0],   p_rev_bp[1]],   pT)

    # --- 漂移期權重覆寫（含 MKV_W） ---
    overrides = None
    if uid:
        drift_now = update_ph_state(uid, seq)
        active = in_drift(uid)
        if active:
            consume_cooldown(uid)
            overrides = {
                "REC_W":  float(os.getenv("PH_REC_W","0.34")),
                "LONG_W": float(os.getenv("PH_LONG_W","0.18")),
                "MKV_W":  float(os.getenv("PH_MKV_W","0.38")),
                "PRIOR_W":float(os.getenv("PH_PRIOR_W","0.10")),
            }

    # --- Arbitration: Momentum vs Reversion ---
    # 先用 rev_strength 在 MOM/REV 間插值
    p_mix = blend(p_mom, p_rev, rev_strength)

    # --- 長期EW + 先驗 + 基底模型 ---
    p_long  = exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.96")))
    REC_W   = float(os.getenv("REC_W","0.28"))   # p_mix 權重
    LONG_W  = float(os.getenv("LONG_W","0.24"))
    MKV_W   = float(os.getenv("MKV_W","0.25"))   # 影響到上面的 MOM_W_MKV 已內含；此處保留接口
    PRIOR_W = float(os.getenv("PRIOR_W","0.12"))
    if overrides:
        REC_W   = overrides.get("REC_W",  REC_W)
        LONG_W  = overrides.get("LONG_W", LONG_W)
        PRIOR_W = overrides.get("PRIOR_W",PRIOR_W)
        # 重要：尊重 MKV_W 覆蓋，轉為調 p_momentum 內的 mkv 影響
        adj = overrides.get("MKV_W", None)
        if adj is not None:
            mom_re = blend(p_sm, p_mkv, adj)         # 重新組 MOM
            p_mom  = _merge_bp_with_t([mom_re[0], mom_re[1]], pT)
            p_mix  = blend(p_mom, p_rev, rev_strength)

    probs = blend(probs, p_mix,  REC_W)
    probs = blend(probs, p_long, LONG_W)
    probs = blend(probs, [THEORETICAL_PROBS["B"],THEORETICAL_PROBS["P"],THEORETICAL_PROBS["T"]], PRIOR_W)

    # --- 安全處理 + 輕度 regime boost ---
    EPS = float(os.getenv("EPSILON_FLOOR","0.06"))
    CAP = float(os.getenv("MAX_CAP","0.86"))
    TAU = float(os.getenv("TEMP","1.06"))
    probs = [min(CAP, max(EPS, p)) for p in probs]
    probs = norm(probs); probs = temperature_scale(probs, TAU)

    # 早龍但非牆阻 → 輕推同邊；強烈交錯 → 輕推反邊
    boosts=[1.0,1.0,1.0]
    early_dragon = (feat.get("col_depth",0)>=3 and not feat.get("blocked",False))
    if early_dragon and last in ("B","P"):
        k = 0 if last=="B" else 1
        boosts[k] *= float(os.getenv("BOOST_EARLY_DRAGON","1.05"))
    # 交錯（近6手全交錯或 1-1/2-2 反覆）
    if detect_single_double(seq_bp):
        if last=="B": boosts[1]*=float(os.getenv("BOOST_ZIGZAG","1.04"))
        elif last=="P": boosts[0]*=float(os.getenv("BOOST_ZIGZAG","1.04"))
    # 和局升溫
    if exp_decay_freq(seq)[2] > THEORETICAL_PROBS["T"]*1.18:
        boosts[2]*=float(os.getenv("BOOST_T","1.02"))

    probs = norm([probs[i]*boosts[i] for i in range(3)])
    return probs

def recommend_from_probs(p: List[float]) -> str:
    return CLASS_ORDER[p.index(max(p))]

# =================== API ===================
@app.route("/", methods=["GET"])
def index(): return "ok"

@app.route("/predict", methods=["POST"])
def predict():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    seq = parse_history(data.get("history"))
    uid = str(data.get("user_id",""))  # 可選：帶 user_id 進 PH
    probs = ensemble_with_anti_stuck(seq, uid=uid)
    rec   = recommend_from_probs(probs)
    return jsonify(
        history_len=len(seq),
        probabilities={"B":probs[0], "P":probs[1], "T":probs[2]},
        recommendation=rec
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT","8080"))
    app.run(host="0.0.0.0", port=port)
