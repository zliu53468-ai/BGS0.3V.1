#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGS LINE Bot — v12 Late/Early Dragon Slope + Quiet Prestart (B/P/T 摘要)
- 未開始分析：只記錄；每 N 手回覆一次 B/P/T 摘要（PRESTART_EVERY_N）
- 強化：早龍/晚龍/形成中龍加速、短龍斷點、n-gram、Markov(1/2/3)、震盪翻邊偏置、PH 漂移、動態權重
- Flex 按鈕：莊=紅、閒=藍、和=綠
Routes: /, /health,/healthz,/predict,/export,/reload,/line-webhook
"""
import os, csv, time, logging, math
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-backend")

# ---------- 路徑（Render 免費盤請寫 /tmp） ----------
def _ensure_parent(p: str) -> str:
    d = os.path.dirname(p) or "."
    try:
        os.makedirs(d, exist_ok=True)
        testf = os.path.join(d, ".wtest"); open(testf, "w").write("ok"); os.remove(testf)
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

# ---------- 參數 ----------
CLASS_ORDER = ("B","P","T")
LAB_ZH = {"B":"莊","P":"閒","T":"和"}
THEORETICAL_PROBS = {"B":0.458,"P":0.446,"T":0.096}

# 未開始分析時，每 N 手回覆一次摘要（0=完全不回覆）
PRESTART_EVERY_N = int(os.getenv("PRESTART_EVERY_N", "3"))

# 觀望/防單邊
ALLOW_NO_BET   = os.getenv("ALLOW_NO_BET","false").lower()=="true"
MIN_GAP        = float(os.getenv("MIN_GAP","0.06"))
SIDE_REPEAT_TH = int(os.getenv("SIDE_REPEAT_TH","3"))
SIDE_REPEAT_PEN= float(os.getenv("SIDE_REPEAT_PEN","0.15"))
SIDE_REPEAT_MAX= int(os.getenv("SIDE_REPEAT_MAX","3"))
BP_BAL_WIN     = int(os.getenv("BP_BAL_WIN","30"))
BP_BAL_STRENGTH= float(os.getenv("BP_BAL_STRENGTH","0.20"))

# 早/晚 龍控制
HZ_BASE   = float(os.getenv("HZ_BASE","0.68"))   # 調高一點，加速早龍
HZ_DECAY  = float(os.getenv("HZ_DECAY","0.90"))
EARLY_ALT_MAX = float(os.getenv("EARLY_ALT_MAX","0.48"))
EARLY_W_MULT  = float(os.getenv("EARLY_W_MULT","1.35"))

LATE_RUN_TH   = int(os.getenv("LATE_RUN_TH","3"))
LATE_ALT_MAX  = float(os.getenv("LATE_ALT_MAX","0.45"))
LATE_WIN      = int(os.getenv("LATE_WIN","10"))
LATE_W_MULT   = float(os.getenv("LATE_W_MULT","1.45"))

# ---------- 工具 ----------
def parse_history(payload) -> List[str]:
    if payload is None: return []
    seq: List[str] = []
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s,str) and s.strip().upper() in CLASS_ORDER:
                seq.append(s.strip().upper())
    elif isinstance(payload,str):
        for ch in payload:
            up=ch.upper()
            if up in CLASS_ORDER: seq.append(up)
    return seq

def clean_bp(seq: List[str]) -> List[str]:
    return [x for x in seq if x in ("B","P")]

def bpt_counts(seq: List[str]) -> Tuple[int,int,int]:
    return (seq.count("B"), seq.count("P"), seq.count("T"))

def norm(v: List[float]) -> List[float]:
    s=sum(v); s=s if s>1e-12 else 1.0
    return [max(0.0,x)/s for x in v]

def blend(a: List[float], b: List[float], w: float) -> List[float]:
    return [(1-w)*a[i]+w*b[i] for i in range(3)]

def temperature_scale(p: List[float], tau: float) -> List[float]:
    if tau<=1e-6: return p
    ex=[pow(max(pi,1e-9),1.0/tau) for pi in p]; s=sum(ex); return [e/s for e in ex]

def last_run(seq: List[str]) -> Tuple[str,int]:
    if not seq: return ("",0)
    ch=seq[-1]; i=len(seq)-2; n=1
    while i>=0 and seq[i]==ch: n+=1; i-=1
    return (ch,n)

def run_lengths(seq: List[str], win:int=14) -> List[int]:
    if not seq: return []
    s=seq[-win:] if win>0 else seq[:]
    lens=[]; cur=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]: cur+=1
        else: lens.append(cur); cur=1
    lens.append(cur); return lens

# ---------- 頻率 ----------
def exp_decay_freq(seq: List[str], gamma: float=None) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    if gamma is None: gamma=float(os.getenv("EW_GAMMA","0.96"))
    wB=wP=wT=0.0; w=1.0
    for r in reversed(seq):
        if r=="B": wB+=w
        elif r=="P": wP+=w
        else: wT+=w
        w*=gamma
    alpha=float(os.getenv("LAPLACE","0.5"))
    wB+=alpha; wP+=alpha; wT+=alpha
    S=wB+wP+wT; return [wB/S,wP/S,wT/S]

def _estimate_tie_prob(seq: List[str]) -> float:
    prior_T=THEORETICAL_PROBS["T"]
    long_T=exp_decay_freq(seq,float(os.getenv("EW_GAMMA","0.96")))[2]
    w=float(os.getenv("T_BLEND","0.5"))
    floor=float(os.getenv("T_MIN","0.03")); cap=float(os.getenv("T_MAX","0.18"))
    pT=(1-w)*prior_T + w*long_T
    return max(floor, min(cap, pT))

def recent_freq(seq: List[str], win:int) -> List[float]:
    if not seq: return [1/3,1/3,1/3]
    cut=seq[-win:] if win>0 else seq
    alpha=float(os.getenv("LAPLACE","0.5"))
    nB=cut.count("B")+alpha; nP=cut.count("P")+alpha; nT=cut.count("T")+alpha
    tot=max(1,len(cut))+3*alpha
    return [nB/tot,nP/tot,nT/tot]

def alt_ratio(seq: List[str], win:int=12) -> float:
    s=clean_bp(seq[-win:] if len(seq)>=win else seq)
    if len(s)<2: return 0.0
    diff=sum(1 for i in range(1,len(s)) if s[i]!=s[i-1])
    return diff/(len(s)-1)

def period2_score(seq: List[str], win:int=12) -> float:
    s=clean_bp(seq[-win:] if len(seq)>=win else seq)
    if len(s)<4: return 0.0
    ok=0; tot=0
    for i in range(3,len(s)):
        a1,a2,b1,b2=s[i-3],s[i-2],s[i-1],s[i]
        if a1==a2 and b1==b2 and a2!=b1: ok+=1
        tot+=1
    return ok/max(1,tot)

# ---------- Markov 1/2/3 ----------
def markov_next_prob(seq: List[str], decay: float=None) -> List[float]:
    if len(seq)<2: return [1/3,1/3,1/3]
    if decay is None: decay=float(os.getenv("MKV_DECAY","0.98"))
    idx={"B":0,"P":1,"T":2}; C=[[0.0]*3 for _ in range(3)]; w=1.0
    for a,b in zip(seq[:-1],seq[1:]):
        C[idx[a]][idx[b]]+=w; w*=decay
    flow=[C[0][0]+C[1][0]+C[2][0], C[0][1]+C[1][1]+C[2][1], C[0][2]+C[1][2]+C[2][2]]
    lap=float(os.getenv("MKV_LAPLACE","0.5")); flow=[x+lap for x in flow]
    S=sum(flow); return [x/S for x in flow]

def markov2_next_prob(seq: List[str], decay: float=None) -> List[float]:
    if len(seq)<3: return [1/3,1/3,1/3]
    if decay is None: decay=float(os.getenv("MKV2_DECAY","0.985"))
    idx={"B":0,"P":1,"T":2}; C=[[[0.0]*3 for _ in range(3)] for __ in range(3)]; w=1.0
    for a,b,c in zip(seq[:-2],seq[1:-1],seq[2:]):
        C[idx[a]][idx[b]][idx[c]]+=w; w*=decay
    a=idx[seq[-2]]; b=idx[seq[-1]]; flow=[C[a][b][0],C[a][b][1],C[a][b][2]]
    lap=float(os.getenv("MKV2_LAPLACE","0.5")); flow=[x+lap for x in flow]
    S=sum(flow); return [x/S for x in flow]

def markov3_next_prob(seq: List[str], decay: float=None) -> List[float]:
    if len(seq)<4: return [1/3,1/3,1/3]
    if decay is None: decay=float(os.getenv("MKV3_DECAY","0.99"))
    idx={"B":0,"P":1,"T":2}; from collections import defaultdict
    C=defaultdict(lambda:[0.0,0.0,0.0]); w=1.0
    for a,b,c,d in zip(seq[:-3],seq[1:-2],seq[2:-1],seq[3:]):
        C[(idx[a],idx[b],idx[c])][idx[d]]+=w; w*=decay
    key=(idx[seq[-3]],idx[seq[-2]],idx[seq[-1]])
    flow=C[key]; lap=float(os.getenv("MKV3_LAPLACE","0.5")); flow=[x+lap for x in flow]
    S=sum(flow); return [x/S for x in flow]

# ---------- n-gram（忽略T） ----------
def ngram_expert(seq: List[str]) -> List[float]:
    bp=clean_bp(seq)
    if len(bp)<3: return [1/3,1/3,1/3]
    from collections import defaultdict
    decay3=float(os.getenv("NGRAM3_DECAY","0.985"))
    decay4=float(os.getenv("NGRAM4_DECAY","0.99"))
    lap=float(os.getenv("NGRAM_LAPLACE","0.5"))
    C3=defaultdict(lambda:[0.0,0.0]); w=1.0
    for a,b,c in zip(bp[:-2],bp[1:-1],bp[2:]): C3[(a,b)][0 if c=="B" else 1]+=w; w*=decay3
    p3=[0.5,0.5]; k3=(bp[-2],bp[-1])
    if k3 in C3:
        f=[C3[k3][0]+lap,C3[k3][1]+lap]; S=sum(f); p3=[f[0]/S,f[1]/S]
    if len(bp)<4:
        pB,pP=p3[0],p3[1]
    else:
        C4=defaultdict(lambda:[0.0,0.0]); w=1.0
        for a,b,c,d in zip(bp[:-3],bp[1:-2],bp[2:-1],bp[3:]): C4[(a,b,c)][0 if d=="B" else 1]+=w; w*=decay4
        p4=[0.5,0.5]; k4=(bp[-3],bp[-2],bp[-1])
        if k4 in C4:
            f=[C4[k4][0]+lap,C4[k4][1]+lap]; S=sum(f); p4=[f[0]/S,f[1]/S]
        alpha=float(os.getenv("NGRAM_BLEND","0.6"))
        pB=alpha*p4[0]+(1-alpha)*p3[0]; pP=alpha*p4[1]+(1-alpha)*p3[1]
    pT=_estimate_tie_prob(seq); sc=1.0-pT
    return [pB*sc,pP*sc,pT]

# ---------- 短龍斷點 ----------
def short_dragon_break_expert(seq: List[str]) -> List[float]:
    bp=clean_bp(seq)
    if not bp: return [1/3,1/3,1/3]
    ch=bp[-1]; i=len(bp)-2; rlen=1
    while i>=0 and bp[i]==ch: rlen+=1; i-=1
    altR=alt_ratio(seq, max(6,int(os.getenv("SDB_WIN","12"))))
    if 2<=rlen<=5 and altR>=float(os.getenv("SDB_ALT_TH","0.5")):
        pT=_estimate_tie_prob(seq)
        if ch=="B": base=[0.0,1.0,pT]
        else:       base=[1.0,0.0,pT]
        S=sum(base); return [x/S for x in base]
    return [1/3,1/3,1/3]

# ---------- 早龍 / 晚龍 / 形成中龍加速 ----------
def hazard_continue_prob(rlen:int) -> float:
    return max(0.0, min(0.99, HZ_BASE * (HZ_DECAY ** max(0, rlen-1))))

def early_dragon_expert(seq: List[str]) -> List[float]:
    bp=clean_bp(seq)
    if len(bp)<2: return [1/3,1/3,1/3]
    last=bp[-1]; i=len(bp)-2; rlen=1
    while i>=0 and bp[i]==last: rlen+=1; i-=1
    if 2<=rlen<=3 and alt_ratio(seq,8)<=EARLY_ALT_MAX:
        pT=_estimate_tie_prob(seq); cont=hazard_continue_prob(rlen)
        stay=cont*(1-pT); flip=(1-cont)*(1-pT)
        return [stay,flip,pT] if last=="B" else [flip,stay,pT]
    return [1/3,1/3,1/3]

def late_dragon_accel_expert(seq: List[str]) -> List[float]:
    bp=clean_bp(seq)
    if len(bp)<LATE_RUN_TH: return [1/3,1/3,1/3]
    last=bp[-1]; i=len(bp)-2; rlen=1
    while i>=0 and bp[i]==last: rlen+=1; i-=1
    win=LATE_WIN; s=bp[-win:] if len(bp)>=win else bp
    altR=alt_ratio(seq, max(8,win))
    if rlen>=LATE_RUN_TH and altR<=LATE_ALT_MAX:
        pT=_estimate_tie_prob(seq)
        dom=max(s.count("B"),s.count("P"))/max(1,len(s))
        cont=min(0.93, 0.62 + 0.10*(rlen-LATE_RUN_TH) + 0.25*dom)*(1.0-0.25*altR)
        stay=cont*(1-pT); flip=(1-cont)*(1-pT)
        return [stay,flip,pT] if last=="B" else [flip,stay,pT]
    return [1/3,1/3,1/3]

def forming_streak_slope_expert(seq: List[str]) -> List[float]:
    """偵測『剛開始牽』：最後兩手相同、再往前不同，且近窗交錯低"""
    bp=clean_bp(seq)
    if len(bp)<3: return [1/3,1/3,1/3]
    a,b,c = bp[-3], bp[-2], bp[-1]
    if b==c and a!=b and alt_ratio(seq, win=8)<=0.4:
        pT=_estimate_tie_prob(seq)
        cont=0.60  # 勇敢一點
        stay=cont*(1-pT); flip=(1-cont)*(1-pT)
        return [stay,flip,pT] if c=="B" else [flip,stay,pT]
    return [1/3,1/3,1/3]

# ---------- Regime / Momentum ----------
def is_oscillating(seq: List[str], win:int=12)->bool:
    lens=run_lengths(seq,win)
    if not lens: return False
    avg=sum(lens)/len(lens)
    return 1.0<=avg<=2.1

def is_qijiao(seq: List[str], win:int=20, tol:float=0.1)->bool:
    s=seq[-win:] if len(seq)>=win else seq
    if not s: return False
    b=s.count("B"); p=s.count("P"); t=s.count("T")
    tot=max(1,b+p)
    ratio=b/tot
    return (abs(ratio-0.5)<=tol) and (t<=max(1,int(0.15*len(s))))

def shape_1room2hall(seq: List[str], win:int=18)->bool:
    lens=run_lengths(seq,win)
    if len(lens)<6: return False
    if not all(1<=x<=2 for x in lens[-6:]): return False
    alt=all((lens[i]%2)!=(lens[i-1]%2) for i in range(1,min(len(lens),10)))
    return alt

def shape_2room1hall(seq: List[str], win:int=18)->bool:
    lens=run_lengths(seq,win)
    if len(lens)<5: return False
    last=lens[-6:] if len(lens)>=6 else lens
    c1=sum(1 for x in last if x==1); c2=sum(1 for x in last if x==2)
    return (c1+c2)>=max(4,int(0.7*len(last))) and c2>=c1

def regime_boosts(seq: List[str]) -> List[float]:
    if not seq: return [1.0,1.0,1.0]
    b=[1.0,1.0,1.0]
    last,rlen=last_run(seq)
    DRAGON_TH=int(os.getenv("BOOST_DRAGON_LEN","4"))
    BOOST_DRAGON=float(os.getenv("BOOST_DRAGON","1.12"))
    BOOST_ALT=float(os.getenv("BOOST_ALT","1.08"))
    BOOST_QJ=float(os.getenv("BOOST_QIJIAO","1.05"))
    BOOST_ROOM=float(os.getenv("BOOST_ROOM","1.06"))
    BOOST_T=float(os.getenv("BOOST_T","1.03"))
    if rlen>=DRAGON_TH:
        if last=="B": b[0]*=BOOST_DRAGON
        elif last=="P": b[1]*=BOOST_DRAGON
        else: b[2]*=BOOST_DRAGON
    if is_oscillating(seq,12) or shape_1room2hall(seq):
        if seq[-1]=="B": b[1]*=BOOST_ALT
        elif seq[-1]=="P": b[0]*=BOOST_ALT
    if shape_2room1hall(seq):
        s=seq[-10:] if len(seq)>=10 else seq
        if s.count("B")>s.count("P"): b[0]*=BOOST_ROOM
        elif s.count("P")>s.count("B"): b[1]*=BOOST_ROOM
    if is_qijiao(seq): b[0]*=BOOST_QJ; b[1]*=BOOST_QJ
    ew=exp_decay_freq(seq)
    if ew[2]>THEORETICAL_PROBS["T"]*1.15: b[2]*=BOOST_T
    return b

MOM_WIN=int(os.getenv("MOM_WIN","8"))
MOM_MAX_BOOST=float(os.getenv("MOM_MAX_BOOST","1.15"))
MOM_BASE_BOOST=float(os.getenv("MOM_BASE_BOOST","1.03"))
MOM_RLEN_THRESH=int(os.getenv("MOM_RLEN_THRESH","3"))
MOM_ALIGN_THRESH=float(os.getenv("MOM_ALIGN_THRESH","0.58"))
def momentum_boost(seq: List[str]) -> List[float]:
    if not seq: return [1.0,1.0,1.0]
    last,rlen=last_run(seq)
    s=seq[-MOM_WIN:] if len(seq)>=MOM_WIN else seq
    b=s.count("B"); p=s.count("P"); t=s.count("T"); tot=max(1,b+p+t)
    rb, rp=b/tot, p/tot
    boosts=[1.0,1.0,1.0]
    if last=="B" and rlen>=MOM_RLEN_THRESH and rb>=MOM_ALIGN_THRESH:
        k=min(1.0,(rlen-MOM_RLEN_THRESH+1)/5.0)*min(1.0,(rb-MOM_ALIGN_THRESH)/(1.0-MOM_ALIGN_THRESH+1e-9))
        boosts[0]*=MOM_BASE_BOOST+(MOM_MAX_BOOST-MOM_BASE_BOOST)*k
    if last=="P" and rlen>=MOM_RLEN_THRESH and rp>=MOM_ALIGN_THRESH:
        k=min(1.0,(rlen-MOM_RLEN_THRESH+1)/5.0)*min(1.0,(rp-MOM_ALIGN_THRESH)/(1.0-MOM_ALIGN_THRESH+1e-9))
        boosts[1]*=MOM_BASE_BOOST+(MOM_MAX_BOOST-MOM_BASE_BOOST)*k
    return boosts

def _apply_boosts_and_norm(probs: List[float], boosts: List[float]) -> List[float]:
    p=[max(1e-12, probs[i]*boosts[i]) for i in range(3)]
    s=sum(p); return [x/s for x in p]

# ---------- PH 漂移 ----------
def js_divergence(p: List[float], q: List[float]) -> float:
    eps=1e-12; m=[(p[i]+q[i])/2.0 for i in range(3)]
    def _kl(a,b): return sum((ai+eps)*math.log((ai+eps)/(bi+eps)) for ai,bi in zip(a,b))
    return 0.5*_kl(p,m)+0.5*_kl(q,m)

USER_DRIFT: Dict[str, Dict[str, float]] = {}
def _get_drift_state(uid: str) -> Dict[str, float]:
    st=USER_DRIFT.get(uid)
    if st is None:
        st={'cum':0.0,'min':0.0,'cooldown':0.0}; USER_DRIFT[uid]=st
    return st

def update_ph_state(uid: str, seq: List[str]) -> Tuple[bool,bool]:
    if not seq: return (False,False)
    st=_get_drift_state(uid)
    REC_WIN=int(os.getenv("REC_WIN_FOR_PH","12"))
    p_short=recent_freq(seq, REC_WIN)
    p_long =exp_decay_freq(seq,float(os.getenv("EW_GAMMA","0.96")))
    D=js_divergence(p_short,p_long)
    PH_DELTA=float(os.getenv("PH_DELTA","0.005"))
    PH_LAMBDA=float(os.getenv("PH_LAMBDA","0.08"))
    st['cum']+=(D-PH_DELTA); st['min']=min(st['min'], st['cum'])
    drift=False
    if (st['cum']-st['min'])>PH_LAMBDA:
        st['cum']=0.0; st['min']=0.0; st['cooldown']=float(os.getenv("DRIFT_STEPS","5")); drift=True
    active=st['cooldown']>0.0
    if active: st['cooldown']=max(0.0, st['cooldown']-1.0)
    return (drift,active)

# ---------- 防單邊/觀望 ----------
USER_RECS: Dict[str, List[str]] = {}
def _apply_bp_balance_regularizer(seq: List[str], probs: List[float]) -> List[float]:
    if not seq: return probs
    s=seq[-BP_BAL_WIN:] if len(seq)>=BP_BAL_WIN else seq
    bp=[ch for ch in s if ch in ("B","P")]
    if not bp or len(bp)<max(6,BP_BAL_WIN//3): return probs
    b=bp.count("B"); p=bp.count("P"); tot=b+p
    rb=b/tot; rp=p/tot; thr=0.62; strength=BP_BAL_STRENGTH
    p2=probs[:]
    if rb>thr and probs[0]>=probs[1]:
        scale=min(1.0,(rb-0.5)/0.5); p2[0]*=(1.0-strength*scale)
    if rp>thr and probs[1]>=probs[0]:
        scale=min(1.0,(rp-0.5)/0.5); p2[1]*=(1.0-strength*scale)
    S=sum(p2); return [x/S for x in p2]

def _apply_side_repeat_penalty(uid: str, probs: List[float]) -> List[float]:
    recs=USER_RECS.get(uid,[])
    if not recs: return probs
    last=recs[-1]; k=1; i=len(recs)-2
    while i>=0 and recs[i]==last: k+=1; i-=1
    if k>=SIDE_REPEAT_TH:
        over=min(SIDE_REPEAT_MAX, k-SIDE_REPEAT_TH+1)
        factor=(1.0-SIDE_REPEAT_PEN)**over
        p=probs[:]
        if last=="B": p[0]*=factor
        elif last=="P": p[1]*=factor
        else: p[2]*=factor
        S=sum(p); return [x/S for x in p]
    return probs

def _maybe_no_bet(probs: List[float]) -> Optional[str]:
    if not ALLOW_NO_BET: return None
    a=sorted(probs, reverse=True)
    return 'N' if a[0]-a[1]<MIN_GAP else None

# ---------- 可選模型 ----------
try:
    import torch; import torch.nn as tnn
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
            self.rnn=tnn.GRU(in_dim, hidden, batch_first=True)
            self.fc=tnn.Linear(hidden, out_dim)
        def forward(self, x):
            out,_=self.rnn(x); return self.fc(out[:,-1,:])
else:
    TinyRNN=None

RNN_MODEL: Optional[Any]=None
XGB_MODEL: Optional[Any]=None
LGBM_MODEL: Optional[Any]=None

def load_models()->None:
    global RNN_MODEL,XGB_MODEL,LGBM_MODEL
    if TinyRNN is not None and torch is not None and os.path.exists(RNN_PATH):
        try:
            m=TinyRNN(); m.load_state_dict(torch.load(RNN_PATH, map_location="cpu")); m.eval()
            RNN_MODEL=m; logger.info("Loaded RNN from %s", RNN_PATH)
        except Exception as e: logger.warning("Load RNN failed: %s", e); RNN_MODEL=None
    if xgb is not None and os.path.exists(XGB_PATH):
        try:
            b=xgb.Booster(); b.load_model(XGB_PATH); XGB_MODEL=b
            logger.info("Loaded XGB from %s", XGB_PATH)
        except Exception as e: logger.warning("Load XGB failed: %s", e); XGB_MODEL=None
    if lgb is not None and os.path.exists(LGBM_PATH):
        try:
            b=lgb.Booster(model_file=LGBM_PATH); LGBM_MODEL=b
            logger.info("Loaded LGBM from %s", LGBM_PATH)
        except Exception as e: logger.warning("Load LGBM failed: %s", e); LGBM_MODEL=None
load_models()

def rnn_predict(seq: List[str]) -> Optional[List[float]]:
    if RNN_MODEL is None or torch is None or not seq: return None
    try:
        def onehot(label:str): return [1 if label==lab else 0 for lab in CLASS_ORDER]
        x=torch.tensor([[onehot(ch) for ch in seq]], dtype=torch.float32)
        with torch.no_grad(): p=torch.softmax(RNN_MODEL(x), dim=-1).cpu().numpy()[0].tolist()
        return [float(v) for v in p]
    except Exception as e: logger.warning("RNN inference failed: %s", e); return None

def _vec_from_seq(seq: List[str], K:int)->List[float]:
    vec=[]
    for label in seq[-K:]:
        vec.extend([1.0 if label==lab else 0.0 for lab in CLASS_ORDER])
    pad=K*3-len(vec)
    if pad>0: vec=[0.0]*pad+vec
    return vec

def xgb_predict(seq: List[str]) -> Optional[List[float]]:
    if XGB_MODEL is None or not seq: return None
    try:
        import numpy as np
        K=int(os.getenv("FEAT_WIN","20"))
        prob=XGB_MODEL.predict(xgb.DMatrix(np.array([_vec_from_seq(seq,K)],dtype=float)))[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]),float(prob[1]),float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            pT=_estimate_tie_prob(seq); b,p=float(prob[0]),float(prob[1])
            s=max(1e-12,b+p); b/=s; p/=s; sc=1.0-pT; return [b*sc,p*sc,pT]
        return None
    except Exception as e: logger.warning("XGB inference failed: %s", e); return None

def lgbm_predict(seq: List[str]) -> Optional[List[float]]:
    if LGBM_MODEL is None or not seq: return None
    try:
        K=int(os.getenv("FEAT_WIN","20"))
        prob=LGBM_MODEL.predict([_vec_from_seq(seq,K)])[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]),float(prob[1]),float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            pT=_estimate_tie_prob(seq); b,p=float(prob[0]),float(prob[1])
            s=max(1e-12,b+p); b/=s; p/=s; sc=1.0-pT; return [b*sc,p*sc,pT]
        return None
    except Exception as e: logger.warning("LGBM inference failed: %s", e); return None

# ---------- 集成 ----------
def ensemble_with_anti_stuck(seq: List[str], weight_overrides: Optional[Dict[str,float]]=None) -> List[float]:
    rule=[THEORETICAL_PROBS["B"],THEORETICAL_PROBS["P"],THEORETICAL_PROBS["T"]]
    pr_rnn=rnn_predict(seq); pr_xgb=xgb_predict(seq); pr_lgb=lgbm_predict(seq)

    w_rule=float(os.getenv("RULE_W","0.22"))
    w_rnn =float(os.getenv("RNN_W" ,"0.22"))
    w_xgb =float(os.getenv("XGB_W" ,"0.22"))
    w_lgb =float(os.getenv("LGBM_W","0.34"))

    total=w_rule + (w_rnn if pr_rnn else 0) + (w_xgb if pr_xgb else 0) + (w_lgb if pr_lgb else 0)
    base=[w_rule*rule[i] for i in range(3)]
    if pr_rnn: base=[base[i]+w_rnn*pr_rnn[i] for i in range(3)]
    if pr_xgb: base=[base[i]+w_xgb*pr_xgb[i] for i in range(3)]
    if pr_lgb: base=[base[i]+w_lgb*pr_lgb[i] for i in range(3)]
    probs=[b/max(total,1e-9) for b in base]

    REC_WIN=int(os.getenv("REC_WIN","16"))
    p_rec =recent_freq(seq, REC_WIN)
    p_long=exp_decay_freq(seq,float(os.getenv("EW_GAMMA","0.96")))
    p_mkv1=markov_next_prob(seq,float(os.getenv("MKV_DECAY","0.98")))
    p_mkv2=markov2_next_prob(seq,float(os.getenv("MKV2_DECAY","0.985")))
    p_mkv3=markov3_next_prob(seq,float(os.getenv("MKV3_DECAY","0.99")))
    p_ng  =ngram_expert(seq)
    p_sdb =short_dragon_break_expert(seq)
    p_early=early_dragon_expert(seq)
    p_late =late_dragon_accel_expert(seq)
    p_slope=forming_streak_slope_expert(seq)

    altR=alt_ratio(seq, max(8,REC_WIN))
    per2=period2_score(seq, max(8,REC_WIN))

    # 權重
    REC_W  =float(os.getenv("REC_W","0.16"))
    LONG_W =float(os.getenv("LONG_W","0.18"))
    MKV1_W =float(os.getenv("MKV_W" ,"0.14"))
    MKV2_W =float(os.getenv("MKV2_W","0.12"))
    MKV3_W =float(os.getenv("MKV3_W","0.08"))
    NGRAM_W=float(os.getenv("NGRAM_W","0.16"))
    SDB_W  =float(os.getenv("SDB_W"  ,"0.14"))
    EARLY_W=float(os.getenv("EARLY_W","0.14"))
    LATE_W =float(os.getenv("LATE_W" ,"0.16"))
    SLOPE_W=float(os.getenv("SLOPE_W","0.12"))
    PRIOR_W=float(os.getenv("PRIOR_W","0.08"))

    # 震盪加權
    osc=min(1.0, 0.6*altR + 0.4*per2)
    scale=1.0 + 0.7*osc
    REC_W  *= (1.0 + 0.2*osc)
    LONG_W *= max(0.5, 1.0 - 0.6*osc)
    MKV1_W *= scale; MKV2_W *= scale; MKV3_W *= scale*0.9
    NGRAM_W*= scale*1.1; SDB_W *= scale*1.1

    # 早/晚/形成中龍
    bp=clean_bp(seq); last_bp_run=0
    if bp:
        ch=bp[-1]; i=len(bp)-2; last_bp_run=1
        while i>=0 and bp[i]==ch: last_bp_run+=1; i-=1
    if 2<=last_bp_run<=3 and altR<=EARLY_ALT_MAX:
        EARLY_W *= EARLY_W_MULT
        MKV1_W  *= 1.10; REC_W *= 1.10
    if last_bp_run>=LATE_RUN_TH and altR<=LATE_ALT_MAX:
        LATE_W  *= LATE_W_MULT
        LONG_W  *= 0.8

    if weight_overrides:
        REC_W  =weight_overrides.get("REC_W",REC_W)
        LONG_W =weight_overrides.get("LONG_W",LONG_W)
        MKV1_W =weight_overrides.get("MKV_W",MKV1_W)
        PRIOR_W=weight_overrides.get("PRIOR_W",PRIOR_W)

    def B(p,q,w): return [(1-w)*p[i]+w*q[i] for i in range(3)]
    probs=B(probs,p_rec ,REC_W)
    probs=B(probs,p_long,LONG_W)
    probs=B(probs,p_mkv1,MKV1_W); probs=B(probs,p_mkv2,MKV2_W); probs=B(probs,p_mkv3,MKV3_W)
    probs=B(probs,p_ng  ,NGRAM_W)
    probs=B(probs,p_sdb ,SDB_W)
    probs=B(probs,p_early,EARLY_W)
    probs=B(probs,p_late ,LATE_W)
    probs=B(probs,p_slope,SLOPE_W)
    probs=B(probs,[THEORETICAL_PROBS["B"],THEORETICAL_PROBS["P"],THEORETICAL_PROBS["T"]],PRIOR_W)

    # 交錯期翻邊偏置
    if len(seq)>=2 and (altR>=float(os.getenv("FLIP_ALT_TH","0.55")) or per2>=float(os.getenv("FLIP_PER2_TH","0.35"))):
        last=None
        for x in reversed(seq):
            if x in ("B","P"): last=x; break
        if last:
            flip=float(os.getenv("FLIP_BIAS","0.10"))*(0.6*altR+0.4*per2)
            p=probs[:]
            if last=="B": p[1]+=flip; p[0]=max(0.0,p[0]-flip*0.8)
            else:         p[0]+=flip; p[1]=max(0.0,p[1]-flip*0.8)
            s=sum(p); probs=[x/s for x in p]

    # 安全處理 + Regime + Momentum
    EPS=float(os.getenv("EPSILON_FLOOR","0.06"))
    CAP=float(os.getenv("MAX_CAP","0.88"))
    TAU=float(os.getenv("TEMP","1.06"))
    probs=[min(CAP,max(EPS,p)) for p in probs]
    probs=norm(probs); probs=temperature_scale(probs,TAU)
    probs=_apply_boosts_and_norm(probs, regime_boosts(seq))
    probs=_apply_boosts_and_norm(probs, momentum_boost(seq))
    return norm(probs)

def recommend_from_probs(probs: List[float]) -> str:
    return CLASS_ORDER[probs.index(max(probs))]

# ---------- 健康/Predict ----------
@app.route("/", methods=["GET"])
def index(): return "ok"
@app.route("/health", methods=["GET"])
def health(): return jsonify(status="healthy", version="v12-slope-quiet")
@app.route("/healthz", methods=["GET"])
def healthz(): return jsonify(status="healthy")
@app.route("/predict", methods=["POST"])
def predict():
    data=request.get_json(silent=True) or {}
    seq=parse_history(data.get("history"))
    probs=ensemble_with_anti_stuck(seq)
    probs=_apply_bp_balance_regularizer(seq, probs)
    nb=_maybe_no_bet(probs)
    rec='N' if nb=='N' else recommend_from_probs(probs)
    labels=list(CLASS_ORDER)
    return jsonify({"history_len":len(seq),
                    "probabilities":{labels[i]:probs[i] for i in range(3)},
                    "recommendation":rec})

# ---------- CSV I/O / Reload ----------
def append_round_csv(uid:str, history_before:str, label:str)->None:
    try:
        with open(DATA_CSV_PATH,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([uid, int(time.time()), history_before, label])
    except Exception as e: logger.warning("append_round_csv failed: %s", e)

@app.route("/export", methods=["GET"])
def export_csv():
    n=int(request.args.get("n","1000")); rows=[]
    try:
        if os.path.exists(DATA_CSV_PATH):
            with open(DATA_CSV_PATH,"r",encoding="utf-8") as f:
                data=list(csv.reader(f)); rows=data[-n:] if n>0 else data
    except Exception as e: logger.warning("export read failed: %s", e); rows=[]
    out="user_id,ts,history_before,label\n" + "\n".join([",".join(r) for r in rows])
    return Response(out, mimetype="text/csv",
        headers={"Content-Disposition":"attachment; filename=rounds.csv"})

@app.route("/reload", methods=["POST"])
def reload_models():
    token=request.headers.get("X-Reload-Token","") or request.args.get("token","")
    if not RELOAD_TOKEN or token!=RELOAD_TOKEN:
        return jsonify(ok=False,error="unauthorized"), 401
    load_models()
    return jsonify(ok=True, rnn=bool(RNN_MODEL), xgb=bool(XGB_MODEL), lgbm=bool(LGBM_MODEL))

# ---------- LINE（彩色按鈕 + 未開始顯示 B/P/T 摘要(節流)） ----------
LINE_CHANNEL_ACCESS_TOKEN=os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")
LINE_CHANNEL_SECRET      =os.getenv("LINE_CHANNEL_SECRET","")
USE_LINE=False
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import (
        MessageEvent, TextMessage, TextSendMessage,
        PostbackEvent, PostbackAction, FlexSendMessage,
        QuickReply, QuickReplyButton
    )
    USE_LINE=bool(LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET)
except Exception as e:
    logger.warning("LINE SDK not available or env not set: %s", e); USE_LINE=False

if USE_LINE:
    line_bot_api=LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler=WebhookHandler(LINE_CHANNEL_SECRET)
else:
    line_bot_api=None; handler=None

USER_HISTORY: Dict[str, List[str]] = {}
USER_READY:   Dict[str, bool]      = {}
USER_RECS:    Dict[str, List[str]] = USER_RECS

def flex_buttons_card() -> 'FlexSendMessage':
    contents={
        "type":"bubble",
        "body":{
            "type":"box","layout":"vertical","spacing":"md",
            "contents":[
                {"type":"text","text":"🤖 請先把當前『歷史牌局』輸入完畢，再按【開始分析】！", "wrap":True, "size":"sm"},
                {"type":"box","layout":"horizontal","spacing":"sm","contents":[
                    {"type":"button","style":"primary","color":"#E74C3C","action":{"type":"postback","label":"莊","data":"B"}},
                    {"type":"button","style":"primary","color":"#2980B9","action":{"type":"postback","label":"閒","data":"P"}},
                    {"type":"button","style":"primary","color":"#27AE60","action":{"type":"postback","label":"和","data":"T"}}
                ]},
                {"type":"box","layout":"horizontal","spacing":"sm","contents":[
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
        logger.warning("LINE webhook hit but LINE SDK/env not configured."); return "ok",200
    signature=request.headers.get("X-Line-Signature",""); body=request.get_data(as_text=True)
    try: handler.handle(body, signature)
    except Exception as e:
        logger.error("LINE handle error: %s", e); return "ok",200
    return "ok",200

if USE_LINE and handler is not None:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_text(event):
        uid=event.source.user_id
        USER_HISTORY.setdefault(uid,[]); USER_READY.setdefault(uid,False)
        b,p,t=bpt_counts(USER_HISTORY[uid])
        msg=f"請先把『歷史牌局』輸入完畢再按【開始分析】。\n目前已輸入：B={b}｜P={p}｜T={t}"
        line_bot_api.reply_message(event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()), flex_buttons_card()])

    @handler.add(PostbackEvent)
    def handle_postback(event):
        uid=event.source.user_id
        data=(event.postback.data or "").upper()
        seq=USER_HISTORY.get(uid,[])
        ready=USER_READY.get(uid,False)

        if data=="START":
            USER_READY[uid]=True
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text="🔎 已開始分析。請繼續輸入莊/閒/和，我會根據資料給出建議。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        if data=="END":
            USER_HISTORY[uid]=[]; USER_READY[uid]=False; USER_RECS[uid]=[]
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text="✅ 已結束分析，紀錄已清空。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        if data not in CLASS_ORDER:
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text="請用按鈕輸入（莊/閒/和），或選開始/結束分析。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()])
            return

        # 記錄 & 落地
        history_before="".join(seq)
        seq.append(data); USER_HISTORY[uid]=seq
        try: append_round_csv(uid, history_before, data)
        except Exception as e: logger.warning("csv log failed: %s", e)

        # 未開始：只記錄；每 N 手回一則 B/P/T 摘要
        if not ready:
            if PRESTART_EVERY_N>0 and (len(seq)%PRESTART_EVERY_N==0):
                b,p,t=bpt_counts(seq); s="".join(seq[-20:])
                msg=(f"已記錄 {len(seq)} 手（近20：{s}）\n"
                     f"目前統計：B={b}｜P={p}｜T={t}\n👉 請在全部歷史輸入完畢後再按【開始分析】")
                line_bot_api.reply_message(event.reply_token,
                    [TextSendMessage(text=msg, quick_reply=quick_reply_bar()), flex_buttons_card()])
            return

        # 漂移
        _, active = update_ph_state(uid, seq)
        overrides=None
        if active:
            REC_W=float(os.getenv("REC_W","0.16")); LONG_W=float(os.getenv("LONG_W","0.18"))
            MKV_W=float(os.getenv("MKV_W","0.14")); PRIOR_W=float(os.getenv("PRIOR_W","0.08"))
            SHORT_BOOST=float(os.getenv("PH_SHORT_BOOST","0.30"))
            LONG_CUT=float(os.getenv("PH_LONG_CUT","0.40"))
            MKV_CUT=float(os.getenv("PH_MKV_CUT","0.40"))
            PRIOR_KEEP=float(os.getenv("PH_PRIOR_KEEP","1.00"))
            overrides={"REC_W":REC_W*(1.0+SHORT_BOOST),
                       "LONG_W":max(0.0,LONG_W*(1.0-LONG_CUT)),
                       "MKV_W":max(0.0,MKV_W*(1.0-MKV_CUT)),
                       "PRIOR_W":PRIOR_W*PRIOR_KEEP}

        # 推論 + 防單邊
        probs=ensemble_with_anti_stuck(seq, overrides)
        probs=_apply_bp_balance_regularizer(seq, probs)
        probs=_apply_side_repeat_penalty(uid, probs)

        nb=_maybe_no_bet(probs)
        if nb=='N':
            msg=(f"已解析 {len(seq)} 手\n"
                 f"機率：莊 {probs[0]:.3f}｜閒 {probs[1]:.3f}｜和 {probs[2]:.3f}\n"
                 f"建議：觀望")
            line_bot_api.reply_message(event.reply_token,
                [TextSendMessage(text=msg, quick_reply=quick_reply_bar()), flex_buttons_card()])
            return

        rec=recommend_from_probs(probs)
        USER_RECS.setdefault(uid,[]).append(rec)
        if len(USER_RECS[uid])>200: USER_RECS[uid]=USER_RECS[uid][-100:]

        suffix="（⚡偵測到路型變化，短期權重暫時提高）" if active else ""
        msg=(f"已解析 {len(seq)} 手\n"
             f"機率：莊 {probs[0]:.3f}｜閒 {probs[1]:.3f}｜和 {probs[2]:.3f}\n"
             f"建議：{LAB_ZH[rec]} {suffix}")
        line_bot_api.reply_message(event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()), flex_buttons_card()])

# ---------- Entrypoint ----------
if __name__=="__main__":
    port=int(os.environ.get("PORT","8080"))
    app.run(host="0.0.0.0", port=port)
