#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGS LINE Bot backend — v8 (Oscillation-Enhanced)
Full-history Ensemble + Regime Gating + PH Drift
+ Oscillation Experts（Alt / PairAlt / RunLen）+ Dynamic Gating
+ 2nd/3rd-Order Markov + Momentum
+ Anti one-sided（抑制同向、B/P 回正、可選觀望）
+ CSV落地 + /export + /reload + LINE 按鈕
"""
import os, csv, time, logging, math
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-backend")

# ========= 路徑（含 /tmp fallback） =========
def _ensure_parent(p: str) -> str:
    d = os.path.dirname(p) or "."
    try:
        os.makedirs(d, exist_ok=True)
        tf = os.path.join(d, ".wtest"); open(tf, "w").write("ok"); os.remove(tf)
        return p
    except Exception:
        alt = os.path.join("/tmp", os.path.relpath(p, "/"))
        os.makedirs(os.path.dirname(alt), exist_ok=True)
        return alt

DATA_CSV_PATH = _ensure_parent(os.getenv("DATA_LOG_PATH", "/tmp/logs/rounds.csv"))
RELOAD_TOKEN  = os.getenv("RELOAD_TOKEN", "")
RNN_PATH = os.getenv("RNN_PATH", "/opt/models/rnn.pt")
XGB_PATH = os.getenv("XGB_PATH", "/opt/models/xgb.json")
LGBM_PATH = os.getenv("LGBM_PATH", "/opt/models/lgbm.txt")

# ========= 防單邊/觀望 =========
SIDE_REPEAT_TH   = int(os.getenv("SIDE_REPEAT_TH", "3"))
SIDE_REPEAT_PEN  = float(os.getenv("SIDE_REPEAT_PEN", "0.15"))
SIDE_REPEAT_MAX  = int(os.getenv("SIDE_REPEAT_MAX", "3"))
BP_BAL_WIN       = int(os.getenv("BP_BAL_WIN", "30"))
BP_BAL_STRENGTH  = float(os.getenv("BP_BAL_STRENGTH", "0.20"))
ALLOW_NO_BET     = os.getenv("ALLOW_NO_BET", "false").lower() == "true"
MIN_GAP          = float(os.getenv("MIN_GAP", "0.06"))

# ========= 常數 =========
CLASS_ORDER = ("B", "P", "T")
LAB_ZH = {"B": "莊", "P": "閒", "T": "和"}
THEORETICAL_PROBS: Dict[str, float] = {"B": 0.458, "P": 0.446, "T": 0.096}

# ========= 解析 =========
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

# ========= 可選模型 =========
try:
    import torch
    import torch.nn as tnn
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
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])
else:
    TinyRNN = None

RNN_MODEL: Optional[Any] = None
XGB_MODEL: Optional[Any] = None
LGBM_MODEL: Optional[Any] = None

def load_models() -> None:
    global RNN_MODEL, XGB_MODEL, LGBM_MODEL
    if TinyRNN is not None and torch is not None and os.path.exists(RNN_PATH):
        try:
            m = TinyRNN(); m.load_state_dict(torch.load(RNN_PATH, map_location="cpu")); m.eval()
            RNN_MODEL = m; logger.info("Loaded RNN from %s", RNN_PATH)
        except Exception as e:
            logger.warning("Load RNN failed: %s", e); RNN_MODEL=None
    if xgb is not None and os.path.exists(XGB_PATH):
        try:
            b = xgb.Booster(); b.load_model(XGB_PATH); XGB_MODEL = b
            logger.info("Loaded XGB from %s", XGB_PATH)
        except Exception as e:
            logger.warning("Load XGB failed: %s", e); XGB_MODEL=None
    if lgb is not None and os.path.exists(LGBM_PATH):
        try:
            b = lgb.Booster(model_file=LGBM_PATH); LGBM_MODEL = b
            logger.info("Loaded LGBM from %s", LGBM_PATH)
        except Exception as e:
            logger.warning("Load LGBM failed: %s", e); LGBM_MODEL=None
load_models()

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

# ---- Tie 補正 + 二/三類重建 ----
def exp_decay_freq(seq: List[str], gamma: float = None) -> List[float]:
    if not seq: return [1/3, 1/3, 1/3]
    if gamma is None: gamma = float(os.getenv("EW_GAMMA", "0.96"))
    wB = wP = wT = 0.0; w = 1.0
    for r in reversed(seq):
        if r == "B": wB += w
        elif r == "P": wP += w
        else: wT += w
        w *= gamma
    alpha = float(os.getenv("LAPLACE", "0.5"))
    wB += alpha; wP += alpha; wT += alpha
    S = wB + wP + wT
    return [wB/S, wP/S, wT/S]

def _estimate_tie_prob(seq: List[str]) -> float:
    prior_T = THEORETICAL_PROBS["T"]
    long_T  = exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.96")))[2]
    w       = float(os.getenv("T_BLEND", "0.5"))
    floor_T = float(os.getenv("T_MIN", "0.03"))
    cap_T   = float(os.getenv("T_MAX", "0.18"))
    pT = (1 - w) * prior_T + w * long_T
    return max(floor_T, min(cap_T, pT))

def _merge_bp_with_t(bp: List[float], pT: float) -> List[float]:
    b, p = float(bp[0]), float(bp[1]); s = max(1e-12, b + p)
    b, p = b / s, p / s; scale = 1.0 - pT
    return [b * scale, p * scale, pT]

def _vec_from_seq(seq: List[str], K:int) -> List[float]:
    vec: List[float] = []
    for label in seq[-K:]:
        vec.extend([1.0 if label == lab else 0.0 for lab in CLASS_ORDER])
    pad = K*3 - len(vec)
    if pad > 0: vec = [0.0]*pad + vec
    return vec

def xgb_predict(seq: List[str]) -> Optional[List[float]]:
    if XGB_MODEL is None or not seq: return None
    try:
        import numpy as np
        K = int(os.getenv("FEAT_WIN", "20"))
        dmatrix = xgb.DMatrix(np.array([_vec_from_seq(seq, K)], dtype=float))
        prob = XGB_MODEL.predict(dmatrix)[0]
        if isinstance(prob, (list, tuple)) and len(prob) == 3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        if isinstance(prob, (list, tuple)) and len(prob) == 2:
            pT = _estimate_tie_prob(seq)
            return _merge_bp_with_t([float(prob[0]), float(prob[1])], pT)
        return None
    except Exception as e:
        logger.warning("XGB inference failed: %s", e); return None

def lgbm_predict(seq: List[str]) -> Optional[List[float]]:
    if LGBM_MODEL is None or not seq: return None
    try:
        K = int(os.getenv("FEAT_WIN", "20"))
        prob = LGBM_MODEL.predict([_vec_from_seq(seq, K)])[0]
        if isinstance(prob, (list, tuple)) and len(prob) == 3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        if isinstance(prob, (list, tuple)) and len(prob) == 2:
            pT = _estimate_tie_prob(seq)
            return _merge_bp_with_t([float(prob[0]), float(prob[1])], pT)
        return None
    except Exception as e:
        logger.warning("LGBM inference failed: %s", e); return None

# ========= 路型/統計 =========
def last_run(seq: List[str]) -> Tuple[str, int]:
    if not seq: return ("", 0)
    ch = seq[-1]; i = len(seq) - 2; n = 1
    while i >= 0 and seq[i] == ch: n += 1; i -= 1
    return (ch, n)

def run_lengths(seq: List[str], win: int = 14) -> List[int]:
    if not seq: return []
    s = seq[-win:] if win>0 else seq[:]
    lens = []; cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]: cur += 1
        else: lens.append(cur); cur = 1
    lens.append(cur); return lens

def alt_ratio(seq: List[str], win:int=12) -> float:
    s = seq[-win:] if len(seq) >= win else seq
    if len(s) < 2: return 0.0
    diff = sum(1 for i in range(1,len(s)) if s[i]!=s[i-1])
    return diff/(len(s)-1)

def period2_score(seq: List[str], win:int=12) -> float:
    """量化 2 期交錯結構（簡化的自相關/二元交替）"""
    s = seq[-win:] if len(seq)>=win else seq
    if len(s) < 4: return 0.0
    # 若連續的 pair 多為 '11' '22' 交替，視為高分
    pairs = [s[i:i+2] for i in range(0, len(s)-1, 1)]
    ok = 0; total = 0
    for i in range(1, len(pairs)):
        a, b = pairs[i-1], pairs[i]
        if len(a)==2 and len(b)==2 and a[0]==a[1] and b[0]==b[1] and a[0]!=b[0]:
            ok += 1
        total += 1
    return ok/max(1,total)

def is_qijiao(seq: List[str], win: int = 20, tol: float = 0.1) -> bool:
    s = seq[-win:] if len(seq) >= win else seq
    if not s: return False
    b = s.count("B"); p = s.count("P"); t = s.count("T")
    tot_bp = max(1, b+p)
    ratio = b / tot_bp
    return (abs(ratio - 0.5) <= tol) and (t <= max(1, int(0.15*len(s))))

def is_oscillating(seq: List[str], win: int = 12) -> bool:
    lens = run_lengths(seq, win)
    if not lens: return False
    avg = sum(lens)/len(lens)
    return 1.0 <= avg <= 2.1

def shape_1room2hall(seq: List[str], win: int = 18) -> bool:
    lens = run_lengths(seq, win)
    if len(lens) < 6: return False
    if not all(1 <= x <= 2 for x in lens[-6:]): return False
    alt = all((lens[i]%2) != (lens[i-1]%2) for i in range(1, min(len(lens), 10)))
    return alt

def shape_2room1hall(seq: List[str], win: int = 18) -> bool:
    lens = run_lengths(seq, win)
    if len(lens) < 5: return False
    last = lens[-6:] if len(lens)>=6 else lens
    cnt_1 = sum(1 for x in last if x==1)
    cnt_2 = sum(1 for x in last if x==2)
    return (cnt_1 + cnt_2) >= max(4, int(0.7*len(last))) and cnt_2 >= cnt_1

# ========= Regime gating（含 Momentum）=========
def regime_boosts(seq: List[str]) -> List[float]:
    if not seq: return [1.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0]
    last, rlen = last_run(seq)
    DRAGON_TH    = int(os.getenv("BOOST_DRAGON_LEN", "4"))
    BOOST_DRAGON = float(os.getenv("BOOST_DRAGON", "1.12"))
    BOOST_ALT    = float(os.getenv("BOOST_ALT", "1.08"))
    BOOST_QJ     = float(os.getenv("BOOST_QIJIAO", "1.05"))
    BOOST_ROOM   = float(os.getenv("BOOST_ROOM", "1.06"))
    BOOST_T      = float(os.getenv("BOOST_T", "1.03"))
    if rlen >= DRAGON_TH:
        if last == "B": b[0] *= BOOST_DRAGON
        elif last == "P": b[1] *= BOOST_DRAGON
        else: b[2] *= BOOST_DRAGON
    if is_oscillating(seq, win=12) or shape_1room2hall(seq):
        if seq[-1] == "B": b[1] *= BOOST_ALT
        elif seq[-1] == "P": b[0] *= BOOST_ALT
    if shape_2room1hall(seq):
        s = seq[-10:] if len(seq)>=10 else seq
        if s.count("B") > s.count("P"): b[0] *= BOOST_ROOM
        elif s.count("P") > s.count("B"): b[1] *= BOOST_ROOM
    if is_qijiao(seq): b[0] *= BOOST_QJ; b[1] *= BOOST_QJ
    ew = exp_decay_freq(seq)
    if ew[2] > THEORETICAL_PROBS["T"] * 1.15: b[2] *= BOOST_T
    return b

# Momentum 追龍
MOM_WIN          = int(os.getenv("MOM_WIN", "8"))
MOM_MAX_BOOST    = float(os.getenv("MOM_MAX_BOOST", "1.15"))
MOM_BASE_BOOST   = float(os.getenv("MOM_BASE_BOOST", "1.03"))
MOM_RLEN_THRESH  = int(os.getenv("MOM_RLEN_THRESH", "3"))
MOM_ALIGN_THRESH = float(os.getenv("MOM_ALIGN_THRESH", "0.58"))
def momentum_boost(seq: List[str]) -> List[float]:
    if not seq: return [1.0, 1.0, 1.0]
    last, rlen = last_run(seq)
    s = seq[-MOM_WIN:] if len(seq) >= MOM_WIN else seq
    b = s.count("B"); p = s.count("P"); t = s.count("T")
    tot = max(1, b+p+t)
    rb, rp = b/tot, p/tot
    boosts = [1.0,1.0,1.0]
    if last == "B" and rlen >= MOM_RLEN_THRESH and rb >= MOM_ALIGN_THRESH:
        k = min(1.0, (rlen - MOM_RLEN_THRESH + 1)/5.0) * min(1.0, (rb - MOM_ALIGN_THRESH)/(1.0 - MOM_ALIGN_THRESH + 1e-9))
        boosts[0] *= MOM_BASE_BOOST + (MOM_MAX_BOOST - MOM_BASE_BOOST)*k
    if last == "P" and rlen >= MOM_RLEN_THRESH and rp >= MOM_ALIGN_THRESH:
        k = min(1.0, (rlen - MOM_RLEN_THRESH + 1)/5.0) * min(1.0, (rp - MOM_ALIGN_THRESH)/(1.0 - MOM_ALIGN_THRESH + 1e-9))
        boosts[1] *= MOM_BASE_BOOST + (MOM_MAX_BOOST - MOM_BASE_BOOST)*k
    return boosts

def _apply_boosts_and_norm(probs: List[float], boosts: List[float]) -> List[float]:
    p = [max(1e-12, probs[i] * boosts[i]) for i in range(3)]
    s = sum(p);  return [x / s for x in p]

# ========= PH 漂移（Page-Hinkley）=========
def js_divergence(p: List[float], q: List[float]) -> float:
    eps = 1e-12
    m = [(p[i]+q[i])/2.0 for i in range(3)]
    def _kl(a, b): return sum((ai+eps)*math.log((ai+eps)/(bi+eps)) for ai,bi in zip(a,b))
    return 0.5*_kl(p, m) + 0.5*_kl(q, m)

USER_DRIFT: Dict[str, Dict[str, float]] = {}
def _get_drift_state(uid: str) -> Dict[str, float]:
    st = USER_DRIFT.get(uid)
    if st is None: st = {'cum': 0.0, 'min': 0.0, 'cooldown': 0.0}; USER_DRIFT[uid] = st
    return st
def recent_freq(seq: List[str], win: int) -> List[float]:
    if not seq: return [1/3, 1/3, 1/3]
    cut = seq[-win:] if win>0 else seq
    alpha = float(os.getenv("LAPLACE", "0.5"))
    nB = cut.count("B") + alpha; nP = cut.count("P") + alpha; nT = cut.count("T") + alpha
    tot = max(1, len(cut)) + 3*alpha
    return [nB/tot, nP/tot, nT/tot]
def update_ph_state(uid: str, seq: List[str]) -> bool:
    if not seq: return False
    st = _get_drift_state(uid)
    REC_WIN = int(os.getenv("REC_WIN_FOR_PH", "12"))
    p_short = recent_freq(seq, REC_WIN)
    p_long  = exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.96")))
    D_t     = js_divergence(p_short, p_long)
    PH_DELTA   = float(os.getenv("PH_DELTA", "0.005"))
    PH_LAMBDA  = float(os.getenv("PH_LAMBDA", "0.08"))
    DRIFT_STEPS= float(os.getenv("DRIFT_STEPS", "5"))
    st['cum'] += (D_t - PH_DELTA); st['min'] = min(st['min'], st['cum'])
    if (st['cum'] - st['min']) > PH_LAMBDA:
        st['cum']=0.0; st['min']=0.0; st['cooldown']=DRIFT_STEPS
        logger.info(f"[PH] drift triggered for {uid}"); return True
    return False
def consume_cooldown(uid: str) -> bool:
    st=_get_drift_state(uid)
    if st['cooldown']>0: st['cooldown']=max(0.0, st['cooldown']-1.0); return True
    return False
def in_drift(uid: str) -> bool:
    return _get_drift_state(uid)['cooldown'] > 0.0

# ========= Markov（1/2/3 階）=========
def markov_next_prob(seq: List[str], decay: float = None) -> List[float]:
    if len(seq) < 2: return [1/3,1/3,1/3]
    if decay is None: decay = float(os.getenv("MKV_DECAY", "0.98"))
    idx = {"B":0, "P":1, "T":2}
    C = [[0.0]*3 for _ in range(3)]; w=1.0
    for a,b in zip(seq[:-1], seq[1:]):
        C[idx[a]][idx[b]] += w; w*=decay
    flow = [C[0][0]+C[1][0]+C[2][0], C[0][1]+C[1][1]+C[2][1], C[0][2]+C[1][2]+C[2][2]]
    alpha = float(os.getenv("MKV_LAPLACE","0.5"))
    flow = [x+alpha for x in flow]; S=sum(flow); return [x/S for x in flow]

def markov2_next_prob(seq: List[str], decay: float = None) -> List[float]:
    if len(seq) < 3: return [1/3,1/3,1/3]
    if decay is None: decay = float(os.getenv("MKV2_DECAY", "0.985"))
    idx = {"B":0, "P":1, "T":2}
    C = [[[0.0]*3 for _ in range(3)] for __ in range(3)]; w=1.0
    for a,b,c in zip(seq[:-2], seq[1:-1], seq[2:]):
        C[idx[a]][idx[b]][idx[c]] += w; w*=decay
    a=idx[seq[-2]]; b=idx[seq[-1]]
    flow=[C[a][b][0], C[a][b][1], C[a][b][2]]
    alpha=float(os.getenv("MKV2_LAPLACE","0.5"))
    flow=[x+alpha for x in flow]; S=sum(flow); return [x/S for x in flow]

def markov3_next_prob(seq: List[str], decay: float = None) -> List[float]:
    if len(seq) < 4: return [1/3,1/3,1/3]
    if decay is None: decay = float(os.getenv("MKV3_DECAY", "0.99"))
    idx = {"B":0, "P":1, "T":2}
    from collections import defaultdict
    C = defaultdict(lambda:[0.0,0.0,0.0]); w=1.0
    for a,b,c,d in zip(seq[:-3], seq[1:-2], seq[2:-1], seq[3:]):
        C[(idx[a],idx[b],idx[c])][idx[d]] += w; w*=decay
    key=(idx[seq[-3]], idx[seq[-2]], idx[seq[-1]])
    flow=C[key]; alpha=float(os.getenv("MKV3_LAPLACE","0.5"))
    flow=[x+alpha for x in flow]; S=sum(flow); return [x/S for x in flow]

# ========= Oscillation Experts =========
def alt_expert(seq: List[str]) -> List[float]:
    """單手交錯：預期翻邊（忽略 T）"""
    if not seq: return [1/3,1/3,1/3]
    last = seq[-1]
    p = [1e-6,1e-6,_estimate_tie_prob(seq)]
    if last == "B": p[1] = 1.0
    elif last == "P": p[0] = 1.0
    else:
        # 上一手 T → 取最近非 T
        for x in reversed(seq[:-1]):
            if x != "T": last=x; break
        if last == "B": p[1]=1.0
        elif last == "P": p[0]=1.0
        else: p[0]=p[1]=0.5
    S=sum(p); return [x/S for x in p]

def pair_alt_expert(seq: List[str]) -> List[float]:
    """兩手一換：若末兩手相同，預期翻邊；若末兩手不同，預期本手再湊對"""
    if len(seq) < 2: return [1/3,1/3,1/3]
    last2 = [x for x in seq if x in ("B","P")]
    if len(last2) < 2: return [1/3,1/3,1/3]
    a, b = last2[-2], last2[-1]
    p = [1e-6,1e-6,_estimate_tie_prob(seq)]
    if a == b:   # e.g., BB，預期轉向到 P
        if b == "B": p[1]=1.0
        else: p[0]=1.0
    else:        # e.g., BP，預期本手補成 PP
        if b == "B": p[0]=1.0
        else: p[1]=1.0
    S=sum(p); return [x/S for x in p]

def runlen_expert(seq: List[str], win:int=12) -> List[float]:
    """依 run-length 均值/方差：均值小→震盪偏翻，均值大→偏續龍"""
    s = seq[-win:] if len(seq)>=win else seq
    lens = run_lengths(s, win=len(s))
    if not lens: return [1/3,1/3,1/3]
    mean = sum(lens)/len(lens)
    var  = sum((x-mean)**2 for x in lens)/len(lens)
    last, rlen = last_run(seq)
    base = [1e-6,1e-6,_estimate_tie_prob(seq)]
    if mean <= 1.6 and var <= 0.6:  # 高震盪
        # 偏翻邊
        if last == "B": base[1]=1.0
        elif last == "P": base[0]=1.0
        else: base[0]=base[1]=0.5
    else:
        # 偏續龍（若 rlen >=2 更強）
        if last == "B": base[0]=1.0 if rlen>=2 else 0.7; base[1]=1.0-base[0]-base[2]
        elif last == "P": base[1]=1.0 if rlen>=2 else 0.7; base[0]=1.0-base[1]-base[2]
        else: base[0]=base[1]=0.5
    S=sum(base); return [x/S for x in base]

# ========= 防單邊 =========
USER_RECS: Dict[str, List[str]] = {}
def _apply_bp_balance_regularizer(seq: List[str], probs: List[float]) -> List[float]:
    if not seq: return probs
    s = seq[-BP_BAL_WIN:] if len(seq) >= BP_BAL_WIN else seq
    bp = [ch for ch in s if ch in ("B","P")]
    if not bp or len(bp) < max(6, BP_BAL_WIN//3): return probs
    b = bp.count("B"); p = bp.count("P"); tot = b+p
    rb = b/tot; rp = p/tot; thr = 0.62; strength = BP_BAL_STRENGTH
    p2 = probs[:]
    if rb > thr and probs[0] >= probs[1]:
        scale = min(1.0, (rb - 0.5)/0.5); p2[0] *= (1.0 - strength*scale)
    if rp > thr and probs[1] >= probs[0]:
        scale = min(1.0, (rp - 0.5)/0.5); p2[1] *= (1.0 - strength*scale)
    S=sum(p2); return [x/S for x in p2]
def _apply_side_repeat_penalty(uid: str, probs: List[float]) -> List[float]:
    recs = USER_RECS.get(uid, [])
    if not recs: return probs
    last = recs[-1]; k=1; i=len(recs)-2
    while i>=0 and recs[i]==last: k+=1; i-=1
    if k >= SIDE_REPEAT_TH:
        over = min(SIDE_REPEAT_MAX, k - SIDE_REPEAT_TH + 1)
        factor = (1.0 - SIDE_REPEAT_PEN) ** over
        p = probs[:]
        if last == "B": p[0]*=factor
        elif last == "P": p[1]*=factor
        else: p[2]*=factor
        S=sum(p); return [x/S for x in p]
    return probs
def _maybe_no_bet(probs: List[float]) -> Optional[str]:
    if not ALLOW_NO_BET: return None
    a = sorted(probs, reverse=True)
    return 'N' if a[0] - a[1] < MIN_GAP else None

# ========= 集成主體（震盪強化版）=========
def norm(v: List[float]) -> List[float]:
    s = sum(v);  s = s if s > 1e-12 else 1.0
    return [max(0.0, x)/s for x in v]
def blend(a: List[float], b: List[float], w: float) -> List[float]:
    return [ (1-w)*a[i] + w*b[i] for i in range(3) ]
def temperature_scale(p: List[float], tau: float) -> List[float]:
    if tau <= 1e-6: return p
    ex = [pow(max(pi,1e-9), 1.0/tau) for pi in p]; s=sum(ex); return [e/s for e in ex]

def ensemble_with_anti_stuck(seq: List[str], weight_overrides: Optional[Dict[str, float]] = None) -> List[float]:
    # 基底：理論 + RNN/XGB/LGBM
    rule  = [THEORETICAL_PROBS["B"], THEORETICAL_PROBS["P"], THEORETICAL_PROBS["T"]]
    pr_rnn = rnn_predict(seq); pr_xgb = xgb_predict(seq); pr_lgb = lgbm_predict(seq)
    w_rule = float(os.getenv("RULE_W", "0.30"))
    w_rnn  = float(os.getenv("RNN_W",  "0.22"))
    w_xgb  = float(os.getenv("XGB_W",  "0.22"))
    w_lgb  = float(os.getenv("LGBM_W", "0.26"))
    total = w_rule + (w_rnn if pr_rnn else 0) + (w_xgb if pr_xgb else 0) + (w_lgb if pr_lgb else 0)
    base = [w_rule*rule[i] for i in range(3)]
    if pr_rnn: base = [base[i] + w_rnn*pr_rnn[i] for i in range(3)]
    if pr_xgb: base = [base[i] + w_xgb*pr_xgb[i] for i in range(3)]
    if pr_lgb: base = [base[i] + w_lgb*pr_lgb[i] for i in range(3)]
    probs = [b / max(total, 1e-9) for b in base]

    # 多來源訊號（傳統）
    REC_WIN = int(os.getenv("REC_WIN", "16"))
    p_rec  = recent_freq(seq, REC_WIN)
    p_long = exp_decay_freq(seq, float(os.getenv("EW_GAMMA","0.96")))
    p_mkv1 = markov_next_prob(seq, float(os.getenv("MKV_DECAY","0.98")))
    p_mkv2 = markov2_next_prob(seq, float(os.getenv("MKV2_DECAY","0.985")))
    p_mkv3 = markov3_next_prob(seq, float(os.getenv("MKV3_DECAY","0.99")))

    # 震盪專家
    p_alt   = alt_expert(seq)
    p_pair  = pair_alt_expert(seq)
    p_rl    = runlen_expert(seq, win=max(10, REC_WIN))

    # 震盪強度量測
    altR = alt_ratio(seq, win=max(8, REC_WIN))
    per2 = period2_score(seq, win=max(8, REC_WIN))
    lens = run_lengths(seq, win=max(8, REC_WIN))
    rvar = 0.0 if not lens else sum((x-(sum(lens)/len(lens)))**2 for x in lens)/len(lens)

    # 自適應權重（越震盪→越信專家與 Markov；越平順→長期/Momentum）
    REC_W   = float(os.getenv("REC_W",  "0.20"))
    LONG_W  = float(os.getenv("LONG_W", "0.22"))
    MKV1_W  = float(os.getenv("MKV_W",  "0.16"))
    MKV2_W  = float(os.getenv("MKV2_W", "0.14"))
    MKV3_W  = float(os.getenv("MKV3_W", "0.08"))
    ALT_W   = float(os.getenv("ALT_W",  "0.12"))
    PAIR_W  = float(os.getenv("PAIR_W", "0.12"))
    RL_W    = float(os.getenv("RL_W",   "0.12"))
    PRIOR_W = float(os.getenv("PRIOR_W","0.08"))

    # 放大因子：交錯強、period2 強 → 專家加權提升
    osc = min(1.0, 0.6*altR + 0.4*per2)  # 0~1
    mkv_amp = 1.0 + 0.6*osc
    exp_amp = 1.0 + 0.8*osc
    long_cut = max(0.5, 1.0 - 0.7*osc)   # 震盪越強，長期越縮
    REC_W   *= (1.0 + 0.3*osc)
    LONG_W  *= long_cut
    MKV1_W  *= mkv_amp; MKV2_W *= mkv_amp; MKV3_W *= (mkv_amp*0.9)
    ALT_W   *= exp_amp; PAIR_W *= exp_amp; RL_W *= (exp_amp*0.9)

    if weight_overrides:
        REC_W   = weight_overrides.get("REC_W",  REC_W)
        LONG_W  = weight_overrides.get("LONG_W", LONG_W)
        MKV1_W  = weight_overrides.get("MKV_W",  MKV1_W)
        PRIOR_W = weight_overrides.get("PRIOR_W",PRIOR_W)

    # 融合
    probs = blend(probs, p_rec,  REC_W)
    probs = blend(probs, p_long, LONG_W)
    probs = blend(probs, p_mkv1, MKV1_W)
    probs = blend(probs, p_mkv2, MKV2_W)
    probs = blend(probs, p_mkv3, MKV3_W)
    probs = blend(probs, p_alt,  ALT_W)
    probs = blend(probs, p_pair, PAIR_W)
    probs = blend(probs, p_rl,   RL_W)
    probs = blend(probs, rule,   PRIOR_W)

    # 安全處理
    EPS = float(os.getenv("EPSILON_FLOOR", "0.06"))
    CAP = float(os.getenv("MAX_CAP", "0.88"))
    TAU = float(os.getenv("TEMP", "1.06"))
    probs = [min(CAP, max(EPS, p)) for p in probs]
    probs = norm(probs); probs = temperature_scale(probs, TAU)

    # Regime + Momentum
    probs = _apply_boosts_and_norm(probs, regime_boosts(seq))
    probs = _apply_boosts_and_norm(probs, momentum_boost(seq))
    return norm(probs)

def recommend_from_probs(probs: List[float]) -> str:
    return CLASS_ORDER[probs.index(max(probs))]

# ========= Health & Predict =========
@app.route("/", methods=["GET"])
def index(): return "ok"
@app.route("/health", methods=["GET"])
def health(): return jsonify(status="healthy", version="v8-oscillation-experts")
@app.route("/healthz", methods=["GET"])
def healthz(): return jsonify(status="healthy")
@app.route("/predict", methods=["POST"])
def predict():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    seq = parse_history(data.get("history"))
    probs = ensemble_with_anti_stuck(seq)
    probs = _apply_bp_balance_regularizer(seq, probs)
    nb = _maybe_no_bet(probs)
    rec = 'N' if nb == 'N' else recommend_from_probs(probs)
    labels = list(CLASS_ORDER)
    return jsonify({
        "history_len": len(seq),
        "probabilities": {labels[i]: probs[i] for i in range(3)},
        "recommendation": rec
    })

# ========= Data I/O =========
def append_round_csv(user_id: str, history_before: str, label: str) -> None:
    try:
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
                data = list(csv.reader(f)); rows = data[-n:] if n>0 else data
    except Exception as e:
        logger.warning("export read failed: %s", e); rows=[]
    output = "user_id,ts,history_before,label\n" + "\n".join([",".join(r) for r in rows])
    return Response(output, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=rounds.csv"})

@app.route("/reload", methods=["POST"])
def reload_models():
    token = request.headers.get("X-Reload-Token", "") or request.args.get("token", "")
    if not RELOAD_TOKEN or token != RELOAD_TOKEN:
        return jsonify(ok=False, error="unauthorized"), 401
    load_models()
    return jsonify(ok=True, rnn=bool(RNN_MODEL), xgb=bool(XGB_MODEL), lgbm=bool(LGBM_MODEL))

# ========= LINE Webhook =========
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET", "")
USE_LINE = False
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import (
        MessageEvent, TextMessage, TextSendMessage,
        PostbackEvent, PostbackAction,
        FlexSendMessage,
        QuickReply, QuickReplyButton
    )
    USE_LINE = bool(LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET)
except Exception as e:
    logger.warning("LINE SDK not available or env not set: %s", e); USE_LINE=False
if USE_LINE:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
else:
    line_bot_api = None; handler=None

USER_HISTORY: Dict[str, List[str]] = {}
USER_READY:   Dict[str, bool]      = {}

def flex_buttons_card() -> 'FlexSendMessage':
    contents = {
        "type": "bubble",
        "body": {
            "type": "box", "layout": "vertical", "spacing": "md",
            "contents": [
                {"type": "text", "text": "🤖 請開始輸入歷史數據", "weight": "bold", "size": "lg"},
                {"type": "text", "text": "先輸入莊/閒/和；按「開始分析」後才會給出下注建議。", "wrap": True, "size": "sm", "color": "#555"},
                {"type": "box", "layout": "horizontal", "spacing": "sm",
                 "contents": [
                    {"type":"button","style":"primary","color":"#E74C3C","action":{"type":"postback","label":"莊","data":"B"}},
                    {"type":"button","style":"primary","color":"#2980B9","action":{"type":"postback","label":"閒","data":"P"}},
                    {"type":"button","style":"primary","color":"#27AE60","action":{"type":"postback","label":"和","data":"T"}}
                 ]},
                {"type": "box", "layout": "horizontal", "spacing": "sm",
                 "contents": [
                    {"type":"button","style":"secondary","action":{"type":"postback","label":"開始分析","data":"START"}},
                    {"type":"button","style":"secondary","action":{"type":"postback","label":"結束分析","data":"END"}}
                 ]}
            ]
        }
    }
    return FlexSendMessage(alt_text="請開始輸入歷史數據", contents=contents)

def quick_reply_bar():
    return QuickReply(items=[
        QuickReplyButton(action=PostbackAction(label="莊", data="B")),
        QuickReplyButton(action=PostbackAction(label="閒", data="P")),
        QuickReplyButton(action=PostbackAction(label="和", data="T")),
        QuickReplyButton(action=PostbackAction(label="開始分析", data="START")),
        QuickReplyButton(action=PostbackAction(label="結束分析", data="END")),
    ])

@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not USE_LINE or handler is None:
        logger.warning("LINE webhook hit but LINE SDK/env not configured."); return "ok", 200
    signature = request.headers.get("X-Line-Signature", ""); body = request.get_data(as_text=True)
    try: handler.handle(body, signature)
    except Exception as e:
        logger.error("LINE handle error: %s", e); return "ok", 200
    return "ok", 200

if USE_LINE and handler is not None:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_text(event):
        uid = event.source.user_id
        USER_HISTORY.setdefault(uid, []); USER_READY.setdefault(uid, False)
        USER_RECS.setdefault(uid, []); _get_drift_state(uid)
        msg = "請使用下方按鈕輸入：莊/閒/和；按「開始分析」後才會給出下注建議。"
        line_bot_api.reply_message(event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()), flex_buttons_card()])

    @handler.add(PostbackEvent)
    def handle_postback(event):
        uid  = event.source.user_id
        data = (event.postback.data or "").upper()
        seq  = USER_HISTORY.get(uid, [])
        ready= USER_READY.get(uid, False)
        USER_RECS.setdefault(uid, [])

        if data == "START":
            USER_READY[uid] = True
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text="🔎 已開始分析。請繼續輸入莊/閒/和，我會根據資料給出建議。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return
        if data == "END":
            USER_HISTORY[uid]=[]; USER_READY[uid]=False; USER_RECS[uid]=[]; USER_DRIFT[uid]={'cum':0.0,'min':0.0,'cooldown':0.0}
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text="✅ 已結束分析，紀錄已清空。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return
        if data not in CLASS_ORDER:
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text="請用按鈕輸入（莊/閒/和），或選開始/結束分析。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        # 追加資料 & 落地
        history_before = "".join(seq)
        seq.append(data); USER_HISTORY[uid] = seq
        append_round_csv(uid, history_before, data)

        if not ready:
            s = "".join(seq[-20:])
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text=f"已記錄 {len(seq)} 手：{s}\n按「開始分析」後才會給出下注建議。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        drift_now = update_ph_state(uid, seq); active = in_drift(uid)
        if active: consume_cooldown(uid)
        overrides = None
        if active:
            REC_W=float(os.getenv("REC_W","0.20")); LONG_W=float(os.getenv("LONG_W","0.22"))
            MKV_W=float(os.getenv("MKV_W","0.16")); PRIOR_W=float(os.getenv("PRIOR_W","0.08"))
            SHORT_BOOST=float(os.getenv("PH_SHORT_BOOST","0.30"))
            LONG_CUT=float(os.getenv("PH_LONG_CUT","0.40")); MKV_CUT=float(os.getenv("PH_MKV_CUT","0.40"))
            PRIOR_KEEP=float(os.getenv("PH_PRIOR_KEEP","1.00"))
            overrides={"REC_W":REC_W*(1.0+SHORT_BOOST),"LONG_W":max(0.0,LONG_W*(1.0-LONG_CUT)),
                       "MKV_W":max(0.0,MKV_W*(1.0-MKV_CUT)),"PRIOR_W":PRIOR_W*PRIOR_KEEP}

        probs = ensemble_with_anti_stuck(seq, overrides)
        probs = _apply_bp_balance_regularizer(seq, probs)
        probs = _apply_side_repeat_penalty(uid, probs)
        nb = _maybe_no_bet(probs)
        if nb == 'N':
            msg = (f"已解析 {len(seq)} 手\n"
                   f"機率：莊 {probs[0]:.3f}｜閒 {probs[1]:.3f}｜和 {probs[2]:.3f}\n"
                   f"建議：觀望")
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text=msg, quick_reply=quick_reply_bar()), flex_buttons_card()])
            return

        rec = CLASS_ORDER[probs.index(max(probs))]
        USER_RECS[uid].append(rec)
        if len(USER_RECS[uid])>200: USER_RECS[uid]=USER_RECS[uid][-100:]
        suffix = "（⚡偵測到路型變化，短期權重已暫時提高）" if active else ""
        msg = (f"已解析 {len(seq)} 手\n"
               f"機率：莊 {probs[0]:.3f}｜閒 {probs[1]:.3f}｜和 {probs[2]:.3f}\n"
               f"建議：{LAB_ZH[rec]} {suffix}")
        line_bot_api.reply_message(event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()), flex_buttons_card()])

# ========= Entry =========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
