# server.py — LiveBoot Baccarat AI (Simple Reply + Soft Regime + Trial Auto-Notify + Feedback Log)

import os, logging, time, csv
from typing import List, Tuple, Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort

log = logging.getLogger("liveboot-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
app = Flask(__name__)

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    if v == "1/0": return 1
    try: return 1 if int(float(v)) != 0 else 0
    except: return 1 if default else 0

# ===== 基本 / 特徵 =====
FEAT_WIN   = int(os.getenv("FEAT_WIN", "40"))
GRID_ROWS  = int(os.getenv("GRID_ROWS", "6"))
GRID_COLS  = int(os.getenv("GRID_COLS", "20"))
MIN_EDGE   = float(os.getenv("MIN_EDGE", "0.07"))
CLIP_T_MIN = float(os.getenv("CLIP_T_MIN", "0.02"))
CLIP_T_MAX = float(os.getenv("CLIP_T_MAX", "0.12"))
SEED       = int(os.getenv("SEED", "42"))
np.random.seed(SEED)

USE_FULL_SHOE = env_flag("USE_FULL_SHOE", 1)
LOCAL_WEIGHT  = float(os.getenv("LOCAL_WEIGHT", "0.65"))
GLOBAL_WEIGHT = float(os.getenv("GLOBAL_WEIGHT", "0.35"))
MAX_RNN_LEN   = int(os.getenv("MAX_RNN_LEN", "256"))

# ===== 試用 / 開通 =====
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@jins888")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")
SHOW_REMAINING_TIME = env_flag("SHOW_REMAINING_TIME", 1)

API_TRIAL_ENFORCE = env_flag("API_TRIAL_ENFORCE", 0)
API_TRIAL_MINUTES = int(os.getenv("API_TRIAL_MINUTES", str(TRIAL_MINUTES)))
API_MINIMAL_JSON  = env_flag("API_MINIMAL_JSON", 0)
CRON_TOKEN = os.getenv("CRON_TOKEN", "")

# ===== 模型與權重 =====
DEEP_ONLY   = int(os.getenv("DEEP_ONLY", "0"))
DISABLE_RNN = int(os.getenv("DISABLE_RNN", "0"))
RNN_HIDDEN  = int(os.getenv("RNN_HIDDEN", "32"))
ENSEMBLE_WEIGHTS = os.getenv("ENSEMBLE_WEIGHTS", "xgb:0.2,lgb:0.2,rnn:0.6")
TEMP_XGB = float(os.getenv("TEMP_XGB", "0.95"))
TEMP_LGB = float(os.getenv("TEMP_LGB", "0.95"))
TEMP_RNN = float(os.getenv("TEMP_RNN", "0.85"))

ABSTAIN_EDGE  = float(os.getenv("ABSTAIN_EDGE", "0.08"))
ABSTAIN_VOTES = int(os.getenv("ABSTAIN_VOTES", "2"))
EDGE_ENTER    = float(os.getenv("EDGE_ENTER", "0.08"))

VOL_GUARD      = int(os.getenv("VOL_GUARD", "1"))
ALT_WIN        = int(os.getenv("ALT_WIN", "24"))
VOL_ALT_BAND   = float(os.getenv("VOL_ALT_BAND", "0.08"))
VOL_ALT_BOOST  = float(os.getenv("VOL_ALT_BOOST", "0.02"))
VOL_FLIP_TH    = float(os.getenv("VOL_FLIP_TH", "0.65"))
VOL_FLIP_BOOST = float(os.getenv("VOL_FLIP_BOOST", "0.02"))

ONLINE_ADAPT       = int(os.getenv("ONLINE_ADAPT", "1"))
ONLINE_MIN_SAMPLES = int(os.getenv("ONLINE_MIN_SAMPLES", "10"))
ONLINE_ACC_LOW     = float(os.getenv("ONLINE_ACC_LOW", "0.45"))
ONLINE_ACC_HIGH    = float(os.getenv("ONLINE_ACC_HIGH", "0.60"))
EDGE_STEP_UP       = float(os.getenv("EDGE_STEP_UP", "0.02"))
EDGE_STEP_DOWN     = float(os.getenv("EDGE_STEP_DOWN", "0.005"))
EDGE_ADAPT_CAP     = float(os.getenv("EDGE_ADAPT_CAP", "0.04"))

REGIME_CTRL   = int(os.getenv("REGIME_CTRL", "1"))
REG_WIN       = int(os.getenv("REG_WIN", "32"))
REG_STREAK_TH = float(os.getenv("REG_STREAK_TH", "0.62"))
REG_CHOP_TH   = float(os.getenv("REG_CHOP_TH", "0.62"))
REG_SIDE_BIAS = float(os.getenv("REG_SIDE_BIAS", "0.58"))
REG_WEIGHTS   = os.getenv("REG_WEIGHTS",
    "0.20/0.20/0.60,0.10/0.10/0.80,0.30/0.30/0.40,0.15/0.15/0.70,0.15/0.15/0.70")
REG_ALIGN_EDGE_BONUS      = float(os.getenv("REG_ALIGN_EDGE_BONUS", "0.01"))
REG_ALIGN_REQUIRE         = int(os.getenv("REG_ALIGN_REQUIRE", "1"))
REG_MISMATCH_EDGE_PENALTY = float(os.getenv("REG_MISMATCH_EDGE_PENALTY", "0.02"))
REGIME_PRIMARY = int(os.getenv("REGIME_PRIMARY", "1"))

EMA_ENABLE    = int(os.getenv("EMA_ENABLE", "1"))
EMA_PROB_A    = float(os.getenv("EMA_PROB_A", "0.30"))
EMA_BET_A     = float(os.getenv("EMA_BET_A", "0.20"))

SHOW_BIAS_ON_ABSTAIN = int(os.getenv("SHOW_BIAS_ON_ABSTAIN", "1"))
FORCE_DIRECTION_WHEN_UNDEREDGE = int(os.getenv("FORCE_DIRECTION_WHEN_UNDEREDGE", "0"))

SAME_SIDE_SOFT_CAP = int(os.getenv("SAME_SIDE_SOFT_CAP", "3"))
SAME_SIDE_PENALTY  = float(os.getenv("SAME_SIDE_PENALTY", "0.02"))

FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", "data/feedback.csv")

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, FollowEvent, TextSendMessage, QuickReply, QuickReplyButton, MessageAction
    line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
    line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None
except Exception as e:
    line_api = None; line_handler = None
    log.warning("LINE SDK not fully available: %s", e)

SESS: Dict[str, Dict[str, object]] = {}
SESS_API: Dict[str, Dict[str, object]] = {}
XGB_MODEL = None; LGB_MODEL = None; RNN_MODEL = None

def _load_xgb():
    global XGB_MODEL
    if DEEP_ONLY == 1: return
    try:
        import xgboost as xgb, os
        path = os.getenv("XGB_OUT_PATH", "data/models/xgb.json")
        if os.path.exists(path):
            booster = xgb.Booster(); booster.load_model(path)
            XGB_MODEL = booster; log.info("[MODEL] XGB loaded: %s", path)
    except Exception as e:
        log.warning("[MODEL] XGB load failed: %s", e)

def _load_lgb():
    global LGB_MODEL
    if DEEP_ONLY == 1: return
    try:
        import lightgbm as lgb, os
        path = os.getenv("LGBM_OUT_PATH", "data/models/lgbm.txt")
        if os.path.exists(path):
            LGB_MODEL = lgb.Booster(model_file=path)
            log.info("[MODEL] LGBM loaded: %s", path)
    except Exception as e:
        log.warning("[MODEL] LGBM load failed: %s", e)

def _load_rnn():
    global RNN_MODEL
    if DISABLE_RNN == 1:
        log.info("[MODEL] RNN disabled by env")
        return
    try:
        import torch, torch.nn as nn
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS","1")))
        class TinyRNN(nn.Module):
            def __init__(self, in_dim=3, hid=int(os.getenv("RNN_HIDDEN","32")), out_dim=3):
                super().__init__()
                self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
                self.fc  = nn.Linear(hid, out_dim)
            def forward(self, x):
                o,_ = self.gru(x); return self.fc(o[:,-1,:])
        path = os.getenv("RNN_OUT_PATH", "data/models/rnn.pt")
        if os.path.exists(path):
            RNN_MODEL = TinyRNN()
            state = __import__("torch").load(path, map_location="cpu")
            RNN_MODEL.load_state_dict(state); RNN_MODEL.eval()
            log.info("[MODEL] RNN loaded: %s", path)
        else:
            log.warning("[MODEL] RNN file not found at %s", path)
    except Exception as e:
        log.warning("[MODEL] RNN load failed: %s", e)

_load_xgb(); _load_lgb(); _load_rnn()

MAP = {"B":0, "P":1, "T":2, "莊":0, "閒":1, "和":2}
INV = {0:"B", 1:"P", 2:"T"}

def parse_history(s: str) -> List[int]:
    s = (s or "").strip().upper()
    if not s: return []
    toks = s.split()
    seq = list(s) if len(toks)==1 else toks
    out=[]
    for ch in seq:
        ch = ch.strip().upper()
        if ch in MAP: out.append(MAP[ch])
    return out

def encode_history(seq: List[int]) -> str:
    return " ".join(INV.get(v,"?") for v in seq)

def big_road_grid(seq: List[int], rows:int=6, cols:int=20):
    gs = np.zeros((rows, cols), dtype=np.int8)
    gt = np.zeros((rows, cols), dtype=np.int16)
    r=c=0; last_bp=None
    for v in seq:
        if v==2:
            if 0<=r<rows and 0<=c<cols: gt[r,c]+=1
            continue
        cur = +1 if v==0 else -1
        if last_bp is None:
            r=c=0; gs[r,c]=cur; last_bp=cur; continue
        if cur==last_bp:
            nr=r+1; nc=c
            if nr>=rows or gs[nr,nc]!=0: nr=r; nc=c+1
            r,c=nr,nc; 
            if 0<=r<rows and 0<=c<cols: gs[r,c]=cur
        else:
            c=c+1; r=0; last_bp=cur
            if c<cols: gs[r,c]=cur
    return gs, gt, (r,c)

def _global_aggregates(seq: List[int]) -> np.ndarray:
    n=len(seq)
    if n==0:
        return np.array([0.49,0.49,0.02, 0.5,0.5, 0,0,0,0, 0.5,0.5,0.5,0.5, 0.0], dtype=np.float32)
    arr=np.array(seq, dtype=np.int16)
    cnt=np.bincount(arr, minlength=3).astype(np.float32); freq=cnt/n
    bp=arr[arr!=2]
    altern=0.5 if len(bp)<2 else float(np.mean(bp[1:]!=bp[:-1]))
    def run_stats(side):
        x=(bp==side).astype(np.int8)
        if x.size==0: return 0.0,0.0
        runs=[]; cur=0
        for v in x:
            if v==1: cur+=1
            elif cur>0: runs.append(cur); cur=0
        if cur>0: runs.append(cur)
        if not runs: return 0.0,0.0
        r=np.array(runs, dtype=np.float32)
        return float(r.mean()), float(r.var()) if r.size>1 else 0.0
    b_mean,b_var = run_stats(0); p_mean,p_var = run_stats(1)
    b2b=p2p=b2p=p2b=0; cb=cp=0
    for i in range(len(bp)-1):
        a,b=bp[i], bp[i+1]
        if a==0: cb+=1; b2b+=(b==0); b2p+=(b==1)
        else:    cp+=1; p2p+=(b==1); p2b+=(b==0)
    B2B=(b2b/cb) if cb>0 else 0.5
    P2P=(p2p/cp) if cp>0 else 0.5
    B2P=(b2p/cb) if cb>0 else 0.5
    P2B=(p2b/cp) if cp>0 else 0.5
    tie_rate=float((arr==2).mean())
    return np.array([freq[0],freq[1],freq[2], altern,1.0-altern,
                     b_mean,b_var,p_mean,p_var, B2B,P2P,B2P,P2B, tie_rate], dtype=np.float32)

def _local_bigroad_feat(seq: List[int], rows:int, cols:int, win:int) -> np.ndarray:
    sub = seq[-win:] if len(seq)>win else seq[:]
    gs, gt, (r,c) = big_road_grid(sub, rows, cols)
    grid_sign_flat = gs.flatten().astype(np.float32)
    grid_tie_flat  = np.clip(gt.flatten(),0,3).astype(np.float32)/3.0
    bp_only=[x for x in sub if x in (0,1)]
    streak_len=0; streak_side=0.0
    if bp_only:
        last=bp_only[-1]
        for v in reversed(bp_only):
            if v==last: streak_len+=1
            else: break
        streak_side=+1.0 if last==0 else -1.0
    col_heights=[]
    for cc in range(cols-1,-1,-1):
        h=int((gs[:,cc]!=0).sum())
        if h>0: col_heights.append(h)
        if len(col_heights)>=6: break
    while len(col_heights)<6: col_heights.append(0)
    col_heights=np.array(col_heights, dtype=np.float32)/rows
    cur_col_height=float((gs[:,c]!=0).sum())/rows if 0<=c<cols else 0.0
    cur_col_side=float(gs[0,c]) if 0<=c<cols else 0.0
    cnt=np.bincount(sub, minlength=3).astype(np.float32); freq=cnt/max(1,len(sub))
    return np.concatenate([grid_sign_flat, grid_tie_flat,
                           np.array([streak_len/rows, streak_side], dtype=np.float32),
                           col_heights,
                           np.array([cur_col_height, cur_col_side], dtype=np.float32),
                           freq], axis=0)

def big_road_features(seq: List[int], rows:int=6, cols:int=20, win:int=40) -> np.ndarray:
    local=_local_bigroad_feat(seq, rows, cols, win).astype(np.float32)
    if USE_FULL_SHOE:
        glob=_global_aggregates(seq).astype(np.float32)
        lw=max(0.0, LOCAL_WEIGHT); gw=max(0.0, GLOBAL_WEIGHT); s=lw+gw
        if s==0: lw,gw=1.0,0.0
        else: lw,gw=lw/s,gw/s
        return np.concatenate([local*lw, glob*gw], axis=0).astype(np.float32)
    else:
        return local

def one_hot_seq(seq: List[int], win:int) -> np.ndarray:
    if USE_FULL_SHOE:
        sub = seq[-MAX_RNN_LEN:] if len(seq)>MAX_RNN_LEN else seq[:]
        L=len(sub); oh=np.zeros((1,L,3), dtype=np.float32)
        for i,v in enumerate(sub):
            if v in (0,1,2): oh[0,i,v]=1.0
        return oh
    else:
        sub = seq[-win:] if len(seq)>win else seq[:]
        pad = [-1]*max(0, win-len(sub))
        final=(pad+sub)[-win:]
        oh=[]
        for v in final:
            a=[0,0,0]
            if v in (0,1,2): a[v]=1
            oh.append(a)
        return np.array(oh, dtype=np.float32)[np.newaxis,:,:]

def softmax_log(p: np.ndarray, temp: float=1.0) -> np.ndarray:
    x = np.log(np.clip(p,1e-9,None)) / max(1e-9, temp)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def _regime_detect(seq: List[int]) -> Tuple[str, Optional[str]]:
    if not REGIME_CTRL or len(seq) < 8: return "neutral", None
    bp=[v for v in seq[-REG_WIN:] if v in (0,1)]
    if len(bp)<6: return "neutral", None
    arr=np.array(bp, dtype=np.int8)
    dif = arr[1:] != arr[:-1]
    chop_ratio=float(dif.mean())
    same_ratio=1.0 - chop_ratio
    b_rate=float((arr==0).mean()); p_rate=1.0-b_rate
    if same_ratio >= REG_STREAK_TH:
        last=arr[-1]; return "streak", ("莊" if last==0 else "閒")
    if chop_ratio >= REG_CHOP_TH:
        last=arr[-1]; return "chop", ("閒" if last==0 else "莊")
    if b_rate >= REG_SIDE_BIAS: return "banker", "莊"
    if p_rate >= REG_SIDE_BIAS: return "player", "閒"
    return "neutral", None

def _parse_triplets(spec: str) -> Dict[str, Tuple[float,float,float]]:
    parts=[s.strip() for s in (spec or "").split(",")]
    pads=["0.20/0.20/0.60","0.10/0.10/0.80","0.30/0.30/0.40","0.15/0.15/0.70","0.15/0.15/0.70"]
    while len(parts)<5: parts.append(pads[len(parts)])
    def one(tri:str):
        try:
            x,y,z=[max(0.0, float(v)) for v in tri.split("/")]
            s=x+y+z; return (x/s,y/s,z/s) if s>0 else (1/3,1/3,1/3)
        except: return (1/3,1/3,1/3)
    t=list(map(one, parts[:5]))
    return {"neutral":t[0], "streak":t[1], "chop":t[2], "banker":t[3], "player":t[4]}

def xgb_probs(seq: List[int]) -> Optional[np.ndarray]:
    if XGB_MODEL is None: return None
    import xgboost as xgb
    feat=big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN).reshape(1,-1)
    p=XGB_MODEL.predict(xgb.DMatrix(feat))[0]
    return np.array(p, dtype=np.float32)

def lgb_probs(seq: List[int]) -> Optional[np.ndarray]:
    if LGB_MODEL is None: return None
    feat=big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN).reshape(1,-1)
    p=LGB_MODEL.predict(feat)[0]
    return np.array(p, dtype=np.float32)

def rnn_probs(seq: List[int]) -> Optional[np.ndarray]:
    if RNN_MODEL is None: return None
    try:
        import torch
    except Exception:
        return None
    x=one_hot_seq(seq, FEAT_WIN)
    with __import__("torch").no_grad():
        logits=RNN_MODEL(__import__("torch").from_numpy(x))
        logits = logits / max(1e-6, TEMP_RNN)
        p = __import__("torch").softmax(logits, dim=-1).cpu().numpy()[0]
    return p.astype(np.float32)

def _parse_weights(spec: str) -> Dict[str, float]:
    out={"XGB":0.33,"LGBM":0.33,"RNN":0.34}
    try:
        tmp={}
        for part in (spec or "").split(","):
            if ":" in part:
                k,v=part.split(":",1)
                k=k.strip().lower(); v=float(v)
                if k=="xgb": tmp["XGB"]=v
                if k=="lgb": tmp["LGBM"]=v
                if k=="rnn": tmp["RNN"]=v
        if tmp:
            s=sum(max(0.0,x) for x in tmp.values()) or 1.0
            for k in tmp: tmp[k]=max(0.0,tmp[k])/s
            out.update(tmp)
    except: pass
    return out

def heuristic_probs(seq: List[int]) -> np.ndarray:
    if not seq: return np.array([0.49,0.49,0.02], dtype=np.float32)
    sub=seq[-FEAT_WIN:] if len(seq)>FEAT_WIN else seq
    cnt=np.bincount(sub, minlength=3).astype(np.float32)
    freq=cnt/max(1,len(sub))
    p0=0.90*freq + 0.10*np.array([0.49,0.49,0.02], dtype=np.float32)
    p0[2]=np.clip(p0[2], CLIP_T_MIN, CLIP_T_MAX)
    p0=np.clip(p0,1e-6,None); p0=p0/p0.sum()
    return p0

def vote_and_average(seq: List[int]) -> Tuple[np.ndarray, Dict[str,str], Dict[str,int], Tuple[str,Optional[str]]]:
    weights_global=_parse_weights(ENSEMBLE_WEIGHTS)
    preds=[]; names=[]; vote_labels={}; vote_counts={'莊':0,'閒':0,'和':0}
    label_map=["莊","閒","和"]

    px = None if DEEP_ONLY==1 else xgb_probs(seq)
    if px is not None:
        p = softmax_log(px, TEMP_XGB); preds.append(p); names.append("XGB")
        vote_labels['XGB']=label_map[int(px.argmax())]; vote_counts[vote_labels['XGB']]+=1

    pl = None if DEEP_ONLY==1 else lgb_probs(seq)
    if pl is not None:
        p = softmax_log(pl, TEMP_LGB); preds.append(p); names.append("LGBM")
        vote_labels['LGBM']=label_map[int(pl.argmax())]; vote_counts[vote_labels['LGBM']]+=1

    pr = rnn_probs(seq)
    if pr is not None:
        p = softmax_log(pr, 1.0); preds.append(p); names.append("RNN")
        vote_labels['RNN']=label_map[int(pr.argmax())]; vote_counts[vote_labels['RNN']]+=1

    regime, prefer = _regime_detect(seq)

    if not preds:
        ph=heuristic_probs(seq)
        return ph, {}, {'莊':0,'閒':0,'和':0}, (regime, prefer)

    rx, rl, rr = _parse_triplets(REG_WEIGHTS).get(regime, (0.33,0.33,0.34))
    regime_w={"XGB":rx,"LGBM":rl,"RNN":rr}
    raw=[]
    for n in names:
        raw.append(max(0.0, weights_global.get(n,0.0)) * max(0.0, regime_w.get(n,0.0)))
    W=np.array(raw, dtype=np.float32)
    if W.sum()<=0: W=np.ones_like(W)/len(W)
    W=W/W.sum()

    P=np.stack(preds, axis=0).astype(np.float32)
    p_avg=(P*W[:,None]).sum(axis=0)
    p_avg[2]=np.clip(p_avg[2], CLIP_T_MIN, CLIP_T_MAX)
    p_avg=np.clip(p_avg,1e-6,None); p_avg=p_avg/p_avg.sum()

    return p_avg, vote_labels, vote_counts, (regime, prefer)

def _alt_flip_metrics(seq: List[int], win: int = 24) -> Tuple[float, float]:
    if not seq: return 0.5, 0.5
    sub=[x for x in seq[-win:] if x in (0,1)]
    if len(sub)<=1: return 0.5,0.5
    dif=np.array(sub[1:])!=np.array(sub[:-1])
    flip_ratio=float(dif.mean())
    return flip_ratio, flip_ratio

def edge_to_base_pct(edge: float) -> float:
    if edge >= max(0.10, MIN_EDGE+0.02): return 0.30
    if edge >= max(0.08, MIN_EDGE):      return 0.20
    if edge >= max(0.05, MIN_EDGE-0.01): return 0.10
    return 0.0

def _online_edge_boost(sess: Optional[Dict[str,object]]) -> float:
    if not ONLINE_ADAPT or not sess: return 0.0
    stat=sess.get("perf", {"ok":0,"ng":0,"boost":0.0})
    ok,ng=int(stat.get("ok",0)), int(stat.get("ng",0))
    n=ok+ng
    if n < ONLINE_MIN_SAMPLES: return float(stat.get("boost",0.0))
    acc = ok / max(1,n)
    boost=float(stat.get("boost",0.0))
    if acc < ONLINE_ACC_LOW: boost=min(EDGE_ADAPT_CAP, boost+EDGE_STEP_UP)
    elif acc >= ONLINE_ACC_HIGH and boost>0: boost=max(0.0, boost-EDGE_STEP_DOWN)
    stat["boost"]=boost; sess["perf"]=stat
    return boost

def decide_bet_from_votes(p: np.ndarray, votes: Dict[str,int], models_used:int,
                          seq: Optional[List[int]] = None, sess: Optional[Dict[str,object]] = None,
                          regime_info: Tuple[str,Optional[str]]=("neutral", None)) -> Tuple[str,float,float]:
    regime, prefer = regime_info
    best = int(np.argmax(p)); lab1 = "莊" if best==0 else ("閒" if best==1 else "和")
    p1 = float(p[best]); p2 = float(sorted([p[0],p[1],p[2]], reverse=True)[1])
    edge = p1 - p2

    enter_th = max(MIN_EDGE, ABSTAIN_EDGE, EDGE_ENTER)
    if REGIME_CTRL and prefer in ("莊","閒"):
        if lab1 == prefer:
            enter_th = max(0.0, enter_th - REG_ALIGN_EDGE_BONUS)
        else:
            if REG_ALIGN_REQUIRE == 1:
                enter_th += REG_MISMATCH_EDGE_PENALTY

    if sess:
        same_run = int(sess.get("same_side_run", 0))
        last_sug = sess.get("last_suggestion")
        if last_sug and lab1 in ("莊","閒"):
            if last_sug == lab1: same_run += 1
            else: same_run = 1
        else:
            same_run = 1
        sess["same_side_run"] = same_run
        if same_run > max(1, SAME_SIDE_SOFT_CAP):
            enter_th += SAME_SIDE_PENALTY

    if VOL_GUARD and seq is not None:
        alt, flip = _alt_flip_metrics(seq, ALT_WIN)
        if abs(alt-0.5) < VOL_ALT_BAND: enter_th += VOL_ALT_BOOST
        if flip >= VOL_FLIP_TH:          enter_th += VOL_FLIP_BOOST

    enter_th += _online_edge_boost(sess)

    if edge < enter_th or lab1=="和":
        return lab1, edge, 0.0
    bet_pct = edge_to_base_pct(edge)
    return lab1, edge, bet_pct

def bet_amount(bankroll:int, pct:float) -> int:
    if not bankroll or bankroll<=0 or pct<=0: return 0
    return int(round(bankroll*pct))

def simple_reply(n_hand:int, lab:str, edge:float, p:np.ndarray, bankroll:int, bet_pct:float) -> str:
    conf = int(round(100*max(p[0], p[1], p[2])))
    amt = bet_amount(bankroll, bet_pct)
    if bet_pct > 0 and amt > 0:
        return f"📝 已記 {n_hand} 手\n👉 下一局：{lab}（{conf}%）🎯\n💰 建議：{amt:,}"
    else:
        return f"📝 已記 {n_hand} 手\n👉 下一局：{lab}（{conf}%）🟡"

def trial_over_text() -> str:
    return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"

def quick_reply_buttons():
    try:
        return QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="莊", text="莊")),
            QuickReplyButton(action=MessageAction(label="閒", text="閒")),
            QuickReplyButton(action=MessageAction(label="和", text="和")),
            QuickReplyButton(action=MessageAction(label="開始分析", text="開始分析")),
            QuickReplyButton(action=MessageAction(label="返回 ⬅️", text="返回")),
            QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
        ])
    except Exception:
        return None

@app.get("/")
def root(): return "LiveBoot ok", 200

@app.get("/health")
def health(): return jsonify(status="ok"), 200

@app.get("/healthz")
def healthz(): return jsonify(status="ok"), 200

@app.get("/cron")
def cron():
    token = request.args.get("token","")
    if not CRON_TOKEN or token != CRON_TOKEN:
        abort(403)
    now = int(time.time()); cnt = 0
    for uid, sess in list(SESS.items()):
        if sess.get("premium"): continue
        start = int(sess.get("trial_start", now))
        if (now - start) // 60 >= TRIAL_MINUTES and not sess.get("trial_notified"):
            try:
                line_api.push_message(uid, TextSendMessage(text=trial_over_text(), quick_reply=quick_reply_buttons()))
                sess["trial_notified"] = True; cnt += 1
            except Exception as e:
                log.warning("cron push failed: %s", e)
    return jsonify(pushed=cnt), 200

@app.post("/predict")
def predict_api():
    data = request.get_json(silent=True) or {}
    session_key = data.get("session_key")
    bankroll_in = data.get("bankroll")
    history = data.get("history", "")
    activation_code = str(data.get("activation_code","")).strip()
    action = str(data.get("action","")).strip().lower()

    if API_TRIAL_ENFORCE and not session_key:
        return jsonify(error="session_key_required",
                       message="API trial is ON. Provide session_key or activation_code."), 400

    if session_key:
        sess = SESS_API.setdefault(session_key, {"bankroll": 0, "seq": [], "trial_start": int(time.time()), "premium": False})
        if activation_code and ADMIN_ACTIVATION_SECRET and (activation_code == ADMIN_ACTIVATION_SECRET):
            sess["premium"] = True
        if API_TRIAL_ENFORCE and not sess.get("premium", False):
            now = int(time.time()); start = int(sess.get("trial_start", now))
            elapsed_min = (now - start) // 60
            if elapsed_min >= API_TRIAL_MINUTES:
                return jsonify(error="trial_expired",
                               message="⛔ API 試用已結束。請提供 activation_code 開通。",
                               contact=ADMIN_CONTACT,
                               minutes=API_TRIAL_MINUTES), 403
        if bankroll_in is not None:
            try: sess["bankroll"] = int(bankroll_in)
            except: pass
        if history:
            sess["seq"] = parse_history(history)
        seq = list(sess.get("seq", []))
        bankroll = int(sess.get("bankroll", 0) or 0)
    else:
        seq = parse_history(history)
        bankroll = int(bankroll_in or 0)
        sess = None

    if action == "undo":
        if seq: seq.pop(-1)
        if session_key: SESS_API[session_key]["seq"] = seq
    elif action == "reset":
        seq = []
        if session_key: SESS_API[session_key]["seq"] = []

    p_avg, vote_labels, vote_counts, regime_info = vote_and_average(seq)
    lab, edge, bet_pct = decide_bet_from_votes(p_avg, vote_counts, len(vote_labels), seq, None, regime_info)
    if API_MINIMAL_JSON:
        return jsonify(
            hands=len(seq),
            suggestion=lab,
            confidence=round(float(max(p_avg)),3),
            edge=round(float(edge),3),
            bet_pct=float(bet_pct),
            bet_amount=bet_amount(bankroll, bet_pct)
        ), 200
    text = simple_reply(len(seq), lab, edge, p_avg, bankroll, bet_pct)
    return jsonify(message=text, hands=len(seq), suggestion=lab,
                   bet_pct=float(bet_pct), bet_amount=bet_amount(bankroll, bet_pct)), 200

def _init_user(uid:str):
    now = int(time.time())
    SESS[uid] = {
        "bankroll": 0, "seq": [], "trial_start": now, "premium": False,
        "perf": {"ok":0,"ng":0,"boost":0.0},
        "ema_p": None, "ema_b": None,
        "last_suggestion": None,
        "same_side_run": 0,
        "trial_notified": False,
    }

@line_handler.add(FollowEvent)
def on_follow(event):
    uid = event.source.user_id
    _init_user(uid)
    mins = TRIAL_MINUTES
    msg = (f"🤖 歡迎！已啟用 {mins} 分鐘試用\n"
           "先輸入本金（例：5000）→ 貼歷史（B/P/T 或 莊/閒/和）→ 輸入『開始分析』📊\n"
           f"到期請輸入：開通 你的密碼（向管理員索取）{ADMIN_CONTACT}")
    line_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=quick_reply_buttons()))

def trial_guard(uid:str, text_in:str, reply_token:str) -> bool:
    sess = SESS.get(uid) or {}
    if sess.get("premium", False): return False
    start = int(sess.get("trial_start", int(time.time())))
    now   = int(time.time())
    elapsed_min = (now - start) // 60
    if elapsed_min >= TRIAL_MINUTES:
        if not sess.get("trial_notified"):
            safe_reply(reply_token, trial_over_text(), uid)
            sess["trial_notified"] = True
        else:
            safe_reply(reply_token, trial_over_text(), uid)
        return True
    return False

def feedback_log(row: Dict[str, object]):
    try:
        path = FEEDBACK_LOG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts","uid","hands","history","sug","p_b","p_p","p_t","edge","bankroll","bet_pct","bet_amt","result"
            ])
            if new_file: w.writeheader()
            w.writerow(row)
    except Exception as e:
        log.warning("feedback log failed: %s", e)

def safe_reply(reply_token: str, text: str, uid: Optional[str] = None):
    try:
        line_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=quick_reply_buttons()))
    except Exception as e:
        log.warning("[LINE] reply failed, try push: %s", e)
        if uid:
            try:
                line_api.push_message(uid, TextSendMessage(text=text, quick_reply=quick_reply_buttons()))
            except Exception as e2:
                log.error("[LINE] push failed: %s", e2)

def validate_activation_code(code: str) -> bool:
    if not ADMIN_ACTIVATION_SECRET: return False
    return bool(code) and (code == ADMIN_ACTIVATION_SECRET)

@line_handler.add(MessageEvent, message=TextMessage)
def on_text(event):
    uid = event.source.user_id
    text = (event.message.text or "").strip()
    if uid not in SESS: _init_user(uid)
    sess = SESS[uid]

    if trial_guard(uid, text, event.reply_token): return

    if text in ["返回", "undo", "回上一步"]:
        seq: List[int] = sess.get("seq", [])
        if seq:
            last = seq.pop(-1); sess["seq"] = seq
            safe_reply(event.reply_token, f"↩️ 撤回 {INV.get(last,'?')}，共 {len(seq)} 手。", uid)
        else:
            safe_reply(event.reply_token, "ℹ️ 沒有可撤回的紀錄。", uid)
        return

    if text in ["結束分析", "清空", "reset"]:
        sess["seq"] = []; sess["bankroll"] = 0
        sess["ema_p"] = None; sess["ema_b"] = None
        sess["last_suggestion"] = None; sess["same_side_run"]=0
        safe_reply(event.reply_token, "🧹 已清空，請輸入本金（例：5000）後貼歷史或輸入「開始分析」📊", uid)
        return

    if text.startswith("開通") or text.lower().startswith("activate"):
        code = text.split(" ",1)[1].strip() if " " in text else ""
        if validate_activation_code(code):
            sess["premium"] = True
            safe_reply(event.reply_token, "✅ 已開通成功！🎉", uid)
        else:
            safe_reply(event.reply_token, "❌ 密碼錯誤，請向管理員索取。", uid)
        return

    if text.isdigit():
        sess["bankroll"] = int(text)
        safe_reply(event.reply_token, f"👍 已設定本金：{int(text):,}", uid)
        return

    if text.startswith("結果") or text.lower().startswith("result"):
        parts = text.split()
        if len(parts) >= 2:
            token = parts[1].strip().upper()
            mapping = {"莊":"B","閒":"P","和":"T","B":"B","P":"P","T":"T"}
            outcome = mapping.get(token)
            last_sug = sess.get("last_suggestion")
            perf = sess.setdefault("perf", {"ok":0,"ng":0,"boost":0.0})
            if outcome and last_sug:
                ok = 1 if ((last_sug=="莊" and outcome=="B") or
                           (last_sug=="閒" and outcome=="P") or
                           (last_sug=="和" and outcome=="T")) else 0
                perf["ok"] = int(perf.get("ok",0)) + (1 if ok else 0)
                perf["ng"] = int(perf.get("ng",0)) + (0 if ok else 1)
            hist = encode_history(sess.get("seq", []))
            p = sess.get("last_probs") or [0.0,0.0,0.0]
            row = {
                "ts": int(time.time()), "uid": uid, "hands": len(sess.get("seq",[])),
                "history": hist, "sug": last_sug or "",
                "p_b": round(float(p[0]),4), "p_p": round(float(p[1]),4), "p_t": round(float(p[2]),4),
                "edge": round(float(sess.get("last_edge",0.0)),4),
                "bankroll": int(sess.get("bankroll",0) or 0),
                "bet_pct": float(sess.get("last_bet_pct",0.0)),
                "bet_amt": bet_amount(int(sess.get("bankroll",0) or 0), float(sess.get("last_bet_pct",0.0))),
                "result": token if outcome else "",
            }
            feedback_log(row)
            safe_reply(event.reply_token, "📥 已記錄結果", uid)
        else:
            safe_reply(event.reply_token, "ℹ️ 用法：結果 莊/閒/和", uid)
        return

    zh2eng = {"莊":"B","閒":"P","和":"T"}
    norm = "".join(zh2eng.get(ch, ch) for ch in text.upper())
    seq_in = parse_history(norm)
    if seq_in and ("開始分析" not in text):
        if len(seq_in) == 1:
            sess.setdefault("seq", []); sess["seq"].append(seq_in[0])
            safe_reply(event.reply_token, f"✅ 已記 1 手：{norm}（共 {len(sess['seq'])}）", uid)
        else:
            sess["seq"] = seq_in
            safe_reply(event.reply_token, f"✅ 已覆蓋歷史：{len(seq_in)} 手", uid)
        return

    if ("開始分析" in text) or (text in ["分析", "開始", "GO", "go"]):
        sseq: List[int] = sess.get("seq", [])
        bankroll: int = int(sess.get("bankroll", 0) or 0)
        p_avg, vote_labels, vote_counts, regime_info = vote_and_average(sseq)
        if EMA_ENABLE:
            prev = sess.get("ema_p")
            p_use = (1-EMA_PROB_A)*(prev if prev is not None else p_avg) + EMA_PROB_A*p_avg
            sess["ema_p"] = p_use
        else:
            p_use = p_avg
        lab, edge, bet_pct = decide_bet_from_votes(p_use, vote_counts, len(vote_labels), sseq, sess, regime_info)
        sess["last_suggestion"] = lab if lab in ("莊","閒","和") else None
        sess["last_edge"] = float(edge); sess["last_bet_pct"] = float(bet_pct)
        sess["last_probs"] = [float(p_use[0]), float(p_use[1]), float(p_use[2])]
        reply = simple_reply(len(sseq), lab, edge, p_use, bankroll, bet_pct)
        safe_reply(event.reply_token, reply, uid); return

    safe_reply(event.reply_token, "🧭 指令：設定本金→貼歷史→『開始分析』｜『返回』撤回｜『結束分析』清空｜『結果 ⋯』回報上一手", uid)

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    app.run(host="0.0.0.0", port=port)
