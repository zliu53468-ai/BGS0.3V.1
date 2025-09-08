#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGS LINE Bot backend v12.3
- Buttons: B(紅)/P(藍)/T(綠)/開始分析/結束分析/返回
- 未開始：每次輸入都顯示已輸入手數 + B/P/T 統計（便於核對）
- 已開始：每次輸入才給建議；【返回】可用（僅回退與統計，不立即推建議）
- Ensemble(全歷史) + Regime gating + PH 漂移 + 齊腳提早偵測 + 弱轉強回補補丁
- NEW: 大路 6×20 繪製（相同往下、不同右移起頂；遇牆/被佔用則右移）→ 提供 wall/深度/早期龍訊號
- 大路特徵會影響權重（靠牆時降低追勢、提高反轉；早期龍但未撞牆時才微增動量）
- Tie(和) 機率三模型一致化
- CSV 追加、/export 匯出、/reload 熱重載
"""
import os, csv, time, logging
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-backend")

# ========= 路徑 & 環境 =========
DATA_CSV_PATH = os.getenv("DATA_LOG_PATH", "/data/logs/rounds.csv")
os.makedirs(os.path.dirname(DATA_CSV_PATH), exist_ok=True)

RELOAD_TOKEN = os.getenv("RELOAD_TOKEN", "")
RNN_PATH = os.getenv("RNN_PATH", "/data/models/rnn.pt")
XGB_PATH = os.getenv("XGB_PATH", "")
LGBM_PATH = os.getenv("LGBM_PATH", "")

# ========= 常數 =========
CLASS_ORDER = ("B","P","T")
LAB_ZH = {"B":"莊","P":"閒","T":"和"}
THEORETICAL_PROBS = {"B":0.458,"P":0.446,"T":0.096}

def parse_history(payload)->List[str]:
    if payload is None: return []
    out=[]
    if isinstance(payload,list):
        for s in payload:
            if isinstance(s,str) and s.strip().upper() in CLASS_ORDER:
                out.append(s.strip().upper())
    elif isinstance(payload,str):
        for ch in payload:
            up=ch.upper()
            if up in CLASS_ORDER: out.append(up)
    return out

# ========= 可選模型安全載入 =========
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
        def __init__(self,in_dim=3,hidden=16,out_dim=3):
            super().__init__()
            self.rnn=tnn.GRU(in_dim,hidden,batch_first=True)
            self.fc=tnn.Linear(hidden,out_dim)
        def forward(self,x):
            o,_=self.rnn(x)
            return self.fc(o[:,-1,:])
else:
    TinyRNN=None

# ========= 模型載入/重載 =========
RNN_MODEL=None
XGB_MODEL=None
LGBM_MODEL=None

def load_models():
    global RNN_MODEL,XGB_MODEL,LGBM_MODEL
    # RNN
    if TinyRNN is not None and torch is not None and os.path.exists(RNN_PATH):
        try:
            m=TinyRNN()
            m.load_state_dict(torch.load(RNN_PATH,map_location="cpu"))
            m.eval(); RNN_MODEL=m
            logger.info("Loaded RNN: %s",RNN_PATH)
        except Exception as e:
            logger.warning("Load RNN failed: %s",e); RNN_MODEL=None
    else: RNN_MODEL=None
    # XGB
    if xgb is not None and XGB_PATH and os.path.exists(XGB_PATH):
        try:
            booster=xgb.Booster(); booster.load_model(XGB_PATH)
            XGB_MODEL=booster; logger.info("Loaded XGB: %s",XGB_PATH)
        except Exception as e:
            logger.warning("Load XGB failed: %s",e); XGB_MODEL=None
    else: XGB_MODEL=None
    # LGBM
    if lgb is not None and LGBM_PATH and os.path.exists(LGBM_PATH):
        try:
            booster=lgb.Booster(model_file=LGBM_PATH)
            LGBM_MODEL=booster; logger.info("Loaded LGBM: %s",LGBM_PATH)
        except Exception as e:
            logger.warning("Load LGBM failed: %s",e); LGBM_MODEL=None
    else: LGBM_MODEL=None

load_models()

# ========= Tie (和) 一致化處理 =========
def exp_decay_freq(seq:List[str], gamma:float=None)->List[float]:
    if not seq: return [1/3,1/3,1/3]
    if gamma is None: gamma=float(os.getenv("EW_GAMMA","0.96"))
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

def _estimate_tie_prob(seq:List[str])->float:
    prior=THEORETICAL_PROBS["T"]
    longT=exp_decay_freq(seq)[2]
    w=float(os.getenv("T_BLEND","0.5"))
    floor=float(os.getenv("T_MIN","0.03"))
    cap=float(os.getenv("T_MAX","0.18"))
    p=(1-w)*prior+w*longT
    return max(floor,min(cap,p))

def _merge_bp_with_t(bp:List[float], pT:float)->List[float]:
    b,p=float(bp[0]),float(bp[1]); s=max(1e-12,b+p)
    b/=s; p/=s; scale=1.0-pT
    return [b*scale,p*scale,pT]

# ========= 單模型推論 =========
def rnn_predict(seq:List[str])->Optional[List[float]]:
    if RNN_MODEL is None or torch is None or not seq: return None
    try:
        def onehot(y): return [1 if y==c else 0 for c in CLASS_ORDER]
        x=torch.tensor([[onehot(ch) for ch in seq]],dtype=torch.float32)
        with torch.no_grad():
            logits=RNN_MODEL(x)
            p=torch.softmax(logits,dim=-1).cpu().numpy()[0].tolist()
        return [float(v) for v in p]
    except Exception as e:
        logger.warning("RNN infer fail: %s",e); return None

def xgb_predict(seq:List[str])->Optional[List[float]]:
    if XGB_MODEL is None or not seq: return None
    try:
        import numpy as np
        K=int(os.getenv("FEAT_WIN","20")); vec=[]
        for lab in seq[-K:]:
            vec.extend([1.0 if lab==c else 0.0 for c in CLASS_ORDER])
        need=K*3-len(vec)
        if need>0: vec=[0.0]*need+vec
        d=xgb.DMatrix(np.array([vec],dtype=float))
        prob=XGB_MODEL.predict(d)[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]),float(prob[1]),float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            return _merge_bp_with_t([float(prob[0]),float(prob[1])], _estimate_tie_prob(seq))
        return None
    except Exception as e:
        logger.warning("XGB infer fail: %s",e); return None

def lgbm_predict(seq:List[str])->Optional[List[float]]:
    if LGBM_MODEL is None or not seq: return None
    try:
        K=int(os.getenv("FEAT_WIN","20")); vec=[]
        for lab in seq[-K:]:
            vec.extend([1.0 if lab==c else 0.0 for c in CLASS_ORDER])
        need=K*3-len(vec)
        if need>0: vec=[0.0]*need+vec
        prob=LGBM_MODEL.predict([vec])[0]
        if isinstance(prob,(list,tuple)) and len(prob)==3:
            return [float(prob[0]),float(prob[1]),float(prob[2])]
        if isinstance(prob,(list,tuple)) and len(prob)==2:
            return _merge_bp_with_t([float(prob[0]),float(prob[1])], _estimate_tie_prob(seq))
        return None
    except Exception as e:
        logger.warning("LGBM infer fail: %s",e); return None

# ========= 工具 =========
def norm(v:List[float])->List[float]:
    s=sum(v); s=s if s>1e-12 else 1.0
    return [max(0.0,x)/s for x in v]

def blend(a:List[float], b:List[float], w:float)->List[float]:
    return [(1-w)*a[i]+w*b[i] for i in range(3)]

def temperature_scale(p:List[float], tau:float)->List[float]:
    if tau<=1e-6: return p
    ex=[pow(max(pi,1e-9),1.0/tau) for pi in p]; s=sum(ex)
    return [e/s for e in ex]

def recent_freq(seq:List[str], win:int)->List[float]:
    if not seq: return [1/3,1/3,1/3]
    s=seq[-win:] if win>0 else seq
    a=float(os.getenv("LAPLACE","0.5"))
    nB=s.count("B")+a; nP=s.count("P")+a; nT=s.count("T")+a
    tot=max(1,len(s))+3*a
    return [nB/tot,nP/tot,nT/tot]

def run_lengths(seq:List[str], win:int=40)->List[int]:
    if not seq: return []
    s=seq[-win:] if win>0 else seq[:]
    out=[]; cur=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]: cur+=1
        else: out.append(cur); cur=1
    out.append(cur)
    return out

def last_run(seq:List[str])->Tuple[str,int]:
    if not seq: return ("",0)
    ch=seq[-1]; i=len(seq)-2; n=1
    while i>=0 and seq[i]==ch: n+=1; i-=1
    return ch,n

def is_qijiao_early(seq:List[str])->bool:
    if len(seq)<8: return False
    N=int(os.getenv("QJ_WIN","18"))
    tol=float(os.getenv("QJ_TOL","0.12"))
    s=seq[-N:]
    b=s.count("B"); p=s.count("P"); bp=max(1,b+p)
    if abs(b/bp-0.5)>tol: return False
    lens=run_lengths(s,N)
    if not lens: return False
    from statistics import mode
    try:
        m=mode([min(3,x) for x in lens])
    except Exception:
        m=1
    if m not in (1,2): return False
    tail=lens[-4:] if len(lens)>=4 else lens
    return all(x in (1,2) for x in tail)

# ========= 大路 6x20 =========
def map_to_big_road(seq: List[str], rows:int=6, cols:int=20) -> Tuple[List[List[str]], Dict[str,Any]]:
    """大路繪製（簡化版）：
       - 與上一手同色：嘗試往下；若到底或下一格被占，則同列右移
       - 與上一手異色：右移至新列，從頂端 (row=0) 開始；若被占，用右移尋找空位
    """
    grid=[["" for _ in range(cols)] for _ in range(rows)]
    if not seq:
        return grid, {"col_depth":0,"blocked":False,"r":0,"c":0,"early_dragon":False}

    r=c=0; last=None
    for ch in seq:
        if last is None:
            grid[r][c]=ch; last=ch; continue
        if ch==last:
            # try down
            if r+1<rows and grid[r+1][c]=="":
                r+=1
            else:
                # wall or occupied -> move right on same row
                c=min(cols-1, c+1)
                while c<cols and grid[r][c]!="":
                    c=min(cols-1, c+1)
                if c>=cols: c=cols-1
        else:
            # switch color -> new column from top
            last=ch
            c=min(cols-1, c+1)
            r=0
            while c<cols and grid[r][c]!="":
                c=min(cols-1, c+1)
            if c>=cols: c=cols-1
        if grid[r][c]=="":
            grid[r][c]=ch

    # 現在列深度 & 是否靠牆/被擋
    cur_depth=0
    for rr in range(rows):
        if grid[rr][c]!="": cur_depth=rr+1
    blocked = (cur_depth>=rows) or (r==rows-1) or (r+1<rows and grid[r+1][c]!="" and last==grid[r][c])
    # 早期龍：近 5~6 手幾乎同色，且目前列深>=3
    tail=seq[-6:] if len(seq)>=6 else seq
    early_dragon = (max(tail.count("B"), tail.count("P"))>=len(tail)-1) and (cur_depth>=3)
    return grid, {"col_depth":cur_depth, "blocked":blocked, "r":r, "c":c, "early_dragon":early_dragon}

# ========= Regime boosts（含齊腳提早 + 大路微調）=========
def regime_boosts(seq:List[str], big_feat:Dict[str,Any])->List[float]:
    b=[1.0,1.0,1.0]
    if not seq: return b
    last,ln=last_run(seq)
    DRAGON_TH=int(os.getenv("BOOST_DRAGON_LEN","4"))
    BOOST_DRAGON=float(os.getenv("BOOST_DRAGON","1.08"))
    BOOST_ALT=float(os.getenv("BOOST_ALT","1.05"))
    BOOST_QJ=float(os.getenv("BOOST_QIJIAO","1.06"))
    BOOST_T=float(os.getenv("BOOST_T","1.02"))
    BOOST_EARLY=float(os.getenv("BOOST_EARLY_DRAGON","1.03"))
    NERF_WALL=float(os.getenv("NERF_WALL","0.92"))  # 靠牆時降低當前色追勢

    # 大路：靠牆 → 降低當前色
    if big_feat.get("blocked", False):
        if last=="B": b[0]*=NERF_WALL
        elif last=="P": b[1]*=NERF_WALL

    # 大路：早期龍且未靠牆 → 略增動量
    if big_feat.get("early_dragon", False) and not big_feat.get("blocked", False):
        if last=="B": b[0]*=BOOST_EARLY
        elif last=="P": b[1]*=BOOST_EARLY

    # 長龍續押（仍保留，但已被靠牆因子抑制）
    if ln>=DRAGON_TH and not big_feat.get("blocked", False):
        if last=="B": b[0]*=BOOST_DRAGON
        elif last=="P": b[1]*=BOOST_DRAGON

    # 蛇/單雙跳：反向微調
    tail=seq[-6:] if len(seq)>=6 else seq
    if len(tail)>=4 and all(tail[i]!=tail[i-1] for i in range(1,len(tail))):
        if tail[-1]=="B": b[1]*=BOOST_ALT
        else: b[0]*=BOOST_ALT

    # 齊腳提早
    if is_qijiao_early(seq):
        b[0]*=BOOST_QJ; b[1]*=BOOST_QJ

    # 和局升溫
    if exp_decay_freq(seq)[2] > THEORETICAL_PROBS["T"]*1.2:
        b[2]*=BOOST_T
    return b

# ========= Page-Hinkley 漂移 =========
def js_divergence(p:List[float], q:List[float])->float:
    import math
    eps=1e-12
    m=[(p[i]+q[i])/2 for i in range(3)]
    def _kl(a,b): return sum((ai+eps)*math.log((ai+eps)/(bi+eps)) for ai,bi in zip(a,b))
    return 0.5*_kl(p,m)+0.5*_kl(q,m)

USER_DRIFT: Dict[str, Dict[str,float]] = {}

def _get_drift_state(uid:str)->Dict[str,float]:
    st=USER_DRIFT.get(uid)
    if st is None:
        st={'cum':0.0,'min':0.0,'cooldown':0.0}
        USER_DRIFT[uid]=st
    return st

def update_ph_state(uid:str, seq:List[str])->bool:
    if not seq: return False
    st=_get_drift_state(uid)
    REC_WIN=int(os.getenv("REC_WIN_FOR_PH","12"))
    p_short=recent_freq(seq,REC_WIN)
    p_long =exp_decay_freq(seq)
    D=js_divergence(p_short,p_long)
    PH_DELTA=float(os.getenv("PH_DELTA","0.005"))
    PH_LAMBDA=float(os.getenv("PH_LAMBDA","0.08"))
    DRIFT_STEPS=float(os.getenv("DRIFT_STEPS","5"))
    st['cum']+=(D-PH_DELTA); st['min']=min(st['min'],st['cum'])
    if (st['cum']-st['min'])>PH_LAMBDA:
        st['cum']=0.0; st['min']=0.0; st['cooldown']=DRIFT_STEPS
        logger.info("[PH] drift triggered uid=%s D=%.4f",uid,D)
        return True
    return False

def in_drift(uid:str)->bool:
    return _get_drift_state(uid)['cooldown']>0.0

def consume_cooldown(uid:str)->None:
    st=_get_drift_state(uid)
    if st['cooldown']>0: st['cooldown']=max(0.0,st['cooldown']-1.0)

# ========= Markov =========
def markov_next_prob(seq:List[str], decay:float=0.98)->List[float]:
    if not seq or len(seq)<2: return [1/3,1/3,1/3]
    idx={"B":0,"P":1,"T":2}
    C=[[0.0]*3 for _ in range(3)]
    w=1.0
    for a,b in zip(seq[:-1],seq[1:]):
        C[idx[a]][idx[b]]+=w; w*=decay
    flow=[C[0][0]+C[1][0]+C[2][0],
          C[0][1]+C[1][1]+C[2][1],
          C[0][2]+C[1][2]+C[2][2]]
    a=float(os.getenv("MKV_LAPLACE","0.5"))
    flow=[x+a for x in flow]; S=sum(flow)
    return [x/S for x in flow]

# ========= 集成 =========
def ensemble_with_anti_stuck(seq:List[str], overrides:Optional[Dict[str,float]]=None)->List[float]:
    rule=[THEORETICAL_PROBS["B"],THEORETICAL_PROBS["P"],THEORETICAL_PROBS["T"]]
    pr_rnn=rnn_predict(seq); pr_xgb=xgb_predict(seq); pr_lgb=lgbm_predict(seq)
    w_rule=float(os.getenv("RULE_W","0.36"))
    w_rnn =float(os.getenv("RNN_W","0.25"))
    w_xgb =float(os.getenv("XGB_W","0.22"))
    w_lgb =float(os.getenv("LGBM_W","0.17"))
    total=w_rule+(w_rnn if pr_rnn else 0)+(w_xgb if pr_xgb else 0)+(w_lgb if pr_lgb else 0)
    base=[w_rule*rule[i] for i in range(3)]
    if pr_rnn: base=[base[i]+w_rnn*pr_rnn[i] for i in range(3)]
    if pr_xgb: base=[base[i]+w_xgb*pr_xgb[i] for i in range(3)]
    if pr_lgb: base=[base[i]+w_lgb*pr_lgb[i] for i in range(3)]
    probs=[b/max(total,1e-9) for b in base]

    # 短/長/馬可夫
    REC_W=float(os.getenv("REC_W","0.25"))
    LONG_W=float(os.getenv("LONG_W","0.28"))
    MKV_W=float(os.getenv("MKV_W","0.25"))
    PRIOR_W=float(os.getenv("PRIOR_W","0.14"))
    if overrides:
        REC_W =overrides.get("REC_W",REC_W)
        LONG_W=overrides.get("LONG_W",LONG_W)
        MKV_W =overrides.get("MKV_W",MKV_W)
        PRIOR_W=overrides.get("PRIOR_W",PRIOR_W)

    p_rec =recent_freq(seq, int(os.getenv("REC_WIN","16")))
    p_long=exp_decay_freq(seq)
    # 修正：正確傳入 seq 與 decay（避免 float has no len()）
    p_mkv =markov_next_prob(seq, float(os.getenv("MKV_DECAY","0.98")))

    # 大路特徵：靠牆→降低追勢；未靠牆且早期龍→略增追勢
    _, feat = map_to_big_road(seq)
    wall_nerf = float(os.getenv("WALL_NERF","0.25"))  # 靠牆時減少短期/Markov 權重
    early_gain= float(os.getenv("EARLY_GAIN","0.12")) # 早期龍時增加短期/Markov 權重（未靠牆）
    rec_w = REC_W
    mkv_w = MKV_W
    if feat.get("blocked", False):
        rec_w = max(0.0, rec_w*(1.0-wall_nerf))
        mkv_w = max(0.0, mkv_w*(1.0-wall_nerf))
    elif feat.get("early_dragon", False):
        rec_w = rec_w*(1.0+early_gain)
        mkv_w = mkv_w*(1.0+early_gain)

    probs=blend(probs,p_rec, rec_w)
    probs=blend(probs,p_long,LONG_W)
    probs=blend(probs,p_mkv, mkv_w)
    probs=blend(probs,rule,  PRIOR_W)

    # 安全 + 溫度
    EPS=float(os.getenv("EPSILON_FLOOR","0.06"))
    CAP=float(os.getenv("MAX_CAP","0.88"))
    TAU=float(os.getenv("TEMP","1.07"))
    probs=[min(CAP,max(EPS,p)) for p in probs]
    probs=norm(probs); probs=temperature_scale(probs,TAU)

    # Regime boosts（含齊腳提早 + 大路靠牆/早期龍）
    boosts=regime_boosts(seq, feat)
    probs=[probs[i]*boosts[i] for i in range(3)]
    probs=norm(probs)

    return probs

def recommend_from_probs(p:List[float])->str:
    return CLASS_ORDER[p.index(max(p))]

# ========= API =========
@app.route("/", methods=["GET"])
def index(): return "ok"

@app.route("/health", methods=["GET"])
def health(): return jsonify(status="healthy", version="v12.3")

@app.route("/healthz", methods=["GET"])
def healthz(): return jsonify(status="healthy")

@app.route("/predict", methods=["POST"])
def predict():
    data=request.get_json(silent=True) or {}
    seq=parse_history(data.get("history"))
    probs=ensemble_with_anti_stuck(seq)
    rec=recommend_from_probs(probs)
    return jsonify(history_len=len(seq),
                   probabilities={"B":probs[0],"P":probs[1],"T":probs[2]},
                   recommendation=rec)

# -------- CSV I/O ----------
def append_round_csv(uid:str, hist_before:str, label:str)->None:
    try:
        os.makedirs(os.path.dirname(DATA_CSV_PATH),exist_ok=True)
        with open(DATA_CSV_PATH,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([uid,int(time.time()),hist_before,label])
    except Exception as e:
        logger.warning("csv append fail: %s",e)

@app.route("/export", methods=["GET"])
def export_csv():
    n=int(request.args.get("n","1000"))
    rows=[]
    try:
        if os.path.exists(DATA_CSV_PATH):
            with open(DATA_CSV_PATH,"r",encoding="utf-8") as f:
                data=list(csv.reader(f)); rows=data[-n:] if n>0 else data
    except Exception as e:
        logger.warning("export fail: %s",e); rows=[]
    out="user_id,ts,history_before,label\n"+"\n".join([",".join(r) for r in rows])
    return Response(out, mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=rounds.csv"})

@app.route("/reload", methods=["POST"])
def reload_models_api():
    token=request.headers.get("X-Reload-Token","") or request.args.get("token","")
    if not RELOAD_TOKEN or token!=RELOAD_TOKEN:
        return jsonify(ok=False,error="unauthorized"),401
    load_models()
    return jsonify(ok=True, rnn=bool(RNN_MODEL), xgb=bool(XGB_MODEL), lgbm=bool(LGBM_MODEL))

# ========= LINE =========
LINE_CHANNEL_ACCESS_TOKEN=os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")
LINE_CHANNEL_SECRET=os.getenv("LINE_CHANNEL_SECRET","")

USE_LINE=False
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import (MessageEvent, TextMessage, TextSendMessage,
                                PostbackEvent, PostbackAction,
                                FlexSendMessage, QuickReply, QuickReplyButton)
    USE_LINE=bool(LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET)
except Exception as e:
    logger.warning("LINE SDK/env not ready: %s",e); USE_LINE=False

if USE_LINE:
    line_bot_api=LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler=WebhookHandler(LINE_CHANNEL_SECRET)
else:
    line_bot_api=None; handler=None

USER_HISTORY: Dict[str,List[str]]={}
USER_READY:   Dict[str,bool]={}
USER_DRIFT:   Dict[str, Dict[str,float]] = {}

def _counts(seq:List[str])->Tuple[int,int,int]:
    return seq.count("B"), seq.count("P"), seq.count("T")

def flex_buttons_card()->'FlexSendMessage':
    contents={
      "type":"bubble",
      "body":{
        "type":"box","layout":"vertical","spacing":"md",
        "contents":[
          {"type":"text","text":"🤖 請先補齊當前靴的歷史，再按「開始分析」","wrap":True,"size":"sm","color":"#555555"},
          {"type":"box","layout":"horizontal","spacing":"sm","contents":[
              {"type":"button","style":"primary","color":"#E74C3C",
               "action":{"type":"postback","label":"莊","data":"B"}},
              {"type":"button","style":"primary","color":"#2980B9",
               "action":{"type":"postback","label":"閒","data":"P"}},
              {"type":"button","style":"primary","color":"#27AE60",
               "action":{"type":"postback","label":"和","data":"T"}}
          ]},
          {"type":"box","layout":"horizontal","spacing":"sm","contents":[
              {"type":"button","style":"secondary",
               "action":{"type":"postback","label":"開始分析","data":"START"}},
              {"type":"button","style":"secondary",
               "action":{"type":"postback","label":"結束分析","data":"END"}},
              {"type":"button","style":"secondary",
               "action":{"type":"postback","label":"返回","data":"RETURN"}}
          ]}
        ]
      }
    }
    return FlexSendMessage(alt_text="輸入/開始/返回", contents=contents)

def quick_reply_bar()->QuickReply:
    return QuickReply(items=[
        QuickReplyButton(action=PostbackAction(label="莊",data="B")),
        QuickReplyButton(action=PostbackAction(label="閒",data="P")),
        QuickReplyButton(action=PostbackAction(label="和",data="T")),
        QuickReplyButton(action=PostbackAction(label="開始分析",data="START")),
        QuickReplyButton(action=PostbackAction(label="結束分析",data="END")),
        QuickReplyButton(action=PostbackAction(label="返回",data="RETURN")),
    ])

@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not USE_LINE or handler is None:
        logger.warning("LINE webhook hit but SDK/env not configured.")
        return "ok",200
    signature=request.headers.get("X-Line-Signature","")
    body=request.get_data(as_text=True)
    try:
        handler.handle(body,signature)
    except Exception as e:
        logger.error("LINE handle error: %s",e)
        return "ok",200
    return "ok",200

if USE_LINE and handler is not None:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_text(event):
        uid=event.source.user_id
        USER_HISTORY.setdefault(uid,[])
        USER_READY.setdefault(uid,False)
        USER_DRIFT.setdefault(uid, {'cum':0.0,'min':0.0,'cooldown':0.0})
        b,p,t=_counts(USER_HISTORY[uid])
        msg=("請用按鈕輸入：莊/閒/和。補齊當前靴後再按「開始分析」。\n"
             f"目前已輸入：{len(USER_HISTORY[uid])} 手（莊{b} / 閒{p} / 和{t}）。")
        line_bot_api.reply_message(
            event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
             flex_buttons_card()]
        )

    @handler.add(PostbackEvent)
    def handle_postback(event):
        uid=event.source.user_id
        data=(event.postback.data or "").upper()
        seq=USER_HISTORY.get(uid,[])
        ready=USER_READY.get(uid,False)

        # 控制鍵
        if data=="START":
            USER_READY[uid]=True
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="🔎 已開始分析。之後每輸入一手我會回覆機率與建議。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            ); return

        if data=="END":
            USER_HISTORY[uid]=[]
            USER_READY[uid]=False
            USER_DRIFT[uid]={'cum':0.0,'min':0.0,'cooldown':0.0}
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="✅ 已結束分析並清空紀錄。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            ); return

        if data=="RETURN":
            if seq:
                seq.pop(); USER_HISTORY[uid]=seq
                b,p,t=_counts(seq); s="".join(seq[-20:])
                tip="（已開始分析：僅回退，不立即提供建議）" if ready else ""
                line_bot_api.reply_message(
                    event.reply_token,
                    [TextSendMessage(
                        text=f"↩️ 已回退 1 手。已輸入 {len(seq)} 手：{s}\n統計：莊{b}｜閒{p}｜和{t} {tip}",
                        quick_reply=quick_reply_bar()),
                     flex_buttons_card()]
                )
            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    [TextSendMessage(text="沒有可回退的手數。", quick_reply=quick_reply_bar()),
                     flex_buttons_card()]
                )
            return

        # 非法
        if data not in CLASS_ORDER:
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="請使用按鈕（莊/閒/和/開始/結束/返回）。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            ); return

        # 記錄 + CSV
        hist_before="".join(seq)
        seq.append(data); USER_HISTORY[uid]=seq
        append_round_csv(uid, hist_before, data)

        # 未開始：只顯示統計
        if not ready:
            b,p,t=_counts(seq); s="".join(seq[-20:])
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(
                    text=f"已輸入 {len(seq)} 手：{s}\n統計：莊{b}｜閒{p}｜和{t}\n補齊後請按「開始分析」。",
                    quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            ); return

        # 已開始：PH 狀態與動態調權
        drift_now=update_ph_state(uid, seq)
        active=in_drift(uid)
        if active: consume_cooldown(uid)
        overrides=None
        if active:
            REC_W=float(os.getenv("REC_W","0.25"))
            LONG_W=float(os.getenv("LONG_W","0.28"))
            MKV_W=float(os.getenv("MKV_W","0.25"))
            PRIOR_W=float(os.getenv("PRIOR_W","0.14"))
            SHORT_BOOST=float(os.getenv("PH_SHORT_BOOST","0.32"))
            CUT=float(os.getenv("PH_CUT","0.40"))
            overrides={"REC_W":REC_W*(1+SHORT_BOOST),
                       "LONG_W":max(0.0,LONG_W*(1-CUT)),
                       "MKV_W": max(0.0,MKV_W *(1-CUT)),
                       "PRIOR_W":PRIOR_W}

        probs=ensemble_with_anti_stuck(seq, overrides)
        rec=recommend_from_probs(probs)
        suffix="（⚡偵測到路型變化，短期權重已暫時提高）" if active else ""
        line_bot_api.reply_message(
            event.reply_token,
            [TextSendMessage(
                text=f"已解析 {len(seq)} 手\n機率：莊 {probs[0]:.3f}｜閒 {probs[1]:.3f}｜和 {probs[2]:.3f}\n建議：{LAB_ZH[rec]} {suffix}",
                quick_reply=quick_reply_bar()),
             flex_buttons_card()]
        )

# ========= Entrypoint =========
if __name__=="__main__":
    port=int(os.environ.get("PORT","8080"))
    app.run(host="0.0.0.0", port=port)
