# server.py — LiveBoot Baccarat AI (Ensemble++ & RNN-First Ready)
# 功能總覽：
# • 三模型：XGB / LGBM / RNN（可加權、可深度主導 DEEP_ONLY）
# • 特徵：大路(6×20) Local + 整盤 Global 融合；RNN 可吃整盤（MAX_RNN_LEN）
# • 決策：溫度縮放、Tie 夾限、票數/邊際雙門檻、震盪防護、線上表現回饋(動態門檻)
# • 風控：邊際驅動的 10/20/30% 配注 × 投票共識縮放
# • LINE：30 分鐘試用鎖、開通密碼、返回/結束分析、結果回報（強化模型表現紀錄）
# • API：/predict 支援 session_key / action=undo|reset / activation_code / API_MINIMAL_JSON
# • 健康檢查：/health, /healthz

import os, logging, time
from typing import List, Tuple, Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort

# ===== App / Log =====
log = logging.getLogger("liveboot-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
app = Flask(__name__)

# ===== Env helpers =====
def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    if v == "1/0": return 1
    try:
        return 1 if int(float(v)) != 0 else 0
    except:
        return 1 if default else 0

# ===== Base Config =====
FEAT_WIN   = int(os.getenv("FEAT_WIN", "40"))
GRID_ROWS  = int(os.getenv("GRID_ROWS", "6"))
GRID_COLS  = int(os.getenv("GRID_COLS", "20"))

MIN_EDGE   = float(os.getenv("MIN_EDGE", "0.07"))
TEMP       = float(os.getenv("TEMP", "0.95"))
CLIP_T_MIN = float(os.getenv("CLIP_T_MIN", "0.02"))
CLIP_T_MAX = float(os.getenv("CLIP_T_MAX", "0.12"))
SEED       = int(os.getenv("SEED", "42"))
np.random.seed(SEED)

# Local + Global 融合
USE_FULL_SHOE = env_flag("USE_FULL_SHOE", 1)
LOCAL_WEIGHT  = float(os.getenv("LOCAL_WEIGHT", "0.65"))
GLOBAL_WEIGHT = float(os.getenv("GLOBAL_WEIGHT", "0.35"))
MAX_RNN_LEN   = int(os.getenv("MAX_RNN_LEN", "256"))

# 試用/開通
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@jins888")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")
SHOW_REMAINING_TIME = env_flag("SHOW_REMAINING_TIME", 1)

# API 試用鎖
API_TRIAL_ENFORCE  = env_flag("API_TRIAL_ENFORCE", 0)
API_TRIAL_MINUTES  = int(os.getenv("API_TRIAL_MINUTES", str(TRIAL_MINUTES)))
API_MINIMAL_JSON   = env_flag("API_MINIMAL_JSON", 0)

# ===== Ensemble / 深度主導 & 溫度 =====
DEEP_ONLY   = int(os.getenv("DEEP_ONLY", "0"))
DISABLE_RNN = int(os.getenv("DISABLE_RNN", "0"))
RNN_HIDDEN  = int(os.getenv("RNN_HIDDEN", "32"))

ENSEMBLE_WEIGHTS = os.getenv("ENSEMBLE_WEIGHTS", "xgb:0.2,lgb:0.2,rnn:0.6")
TEMP_XGB = float(os.getenv("TEMP_XGB", "0.95"))
TEMP_LGB = float(os.getenv("TEMP_LGB", "0.95"))
TEMP_RNN = float(os.getenv("TEMP_RNN", "0.85"))

# ===== 投入門檻（票數/邊際）=====
ABSTAIN_EDGE  = float(os.getenv("ABSTAIN_EDGE", "0.08"))
ABSTAIN_VOTES = int(os.getenv("ABSTAIN_VOTES", "2"))

# ===== Volatility Guard（震盪防護）=====
VOL_GUARD     = int(os.getenv("VOL_GUARD", "1"))
ALT_WIN       = int(os.getenv("ALT_WIN", "24"))
VOL_ALT_BAND  = float(os.getenv("VOL_ALT_BAND", "0.08"))
VOL_ALT_BOOST = float(os.getenv("VOL_ALT_BOOST", "0.02"))
VOL_FLIP_TH   = float(os.getenv("VOL_FLIP_TH", "0.65"))
VOL_FLIP_BOOST= float(os.getenv("VOL_FLIP_BOOST", "0.02"))
EDGE_ENTER    = float(os.getenv("EDGE_ENTER", "0.08"))

# ===== Online feedback（線上表現回饋→動態門檻）=====
ONLINE_ADAPT        = int(os.getenv("ONLINE_ADAPT", "1"))
ONLINE_MIN_SAMPLES  = int(os.getenv("ONLINE_MIN_SAMPLES", "10"))
ONLINE_ACC_LOW      = float(os.getenv("ONLINE_ACC_LOW", "0.45"))
ONLINE_ACC_HIGH     = float(os.getenv("ONLINE_ACC_HIGH", "0.60"))
EDGE_STEP_UP        = float(os.getenv("EDGE_STEP_UP", "0.02"))
EDGE_STEP_DOWN      = float(os.getenv("EDGE_STEP_DOWN", "0.005"))
EDGE_ADAPT_CAP      = float(os.getenv("EDGE_ADAPT_CAP", "0.04"))

# ===== LINE SDK =====
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import (
        MessageEvent, TextMessage, FollowEvent, TextSendMessage,
        QuickReply, QuickReplyButton, MessageAction
    )
    line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
    line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None
except Exception as e:
    line_api = None
    line_handler = None
    log.warning("LINE SDK not fully available: %s", e)

# ===== Sessions =====
SESS: Dict[str, Dict[str, object]] = {}      # for LINE
SESS_API: Dict[str, Dict[str, object]] = {}  # for /predict

# ===== Models (lazy) =====
XGB_MODEL = None
LGB_MODEL = None
RNN_MODEL = None

def _load_xgb():
    global XGB_MODEL
    if DEEP_ONLY == 1:
        log.info("[MODEL] DEEP_ONLY=1 → skip XGB")
        return
    try:
        import xgboost as xgb, os
        path = os.getenv("XGB_OUT_PATH", "/data/models/xgb.json")
        if os.path.exists(path):
            booster = xgb.Booster(); booster.load_model(path)
            XGB_MODEL = booster
            log.info("[MODEL] XGB loaded: %s", path)
    except Exception as e:
        log.warning("[MODEL] XGB load failed: %s", e)

def _load_lgb():
    global LGB_MODEL
    if DEEP_ONLY == 1:
        log.info("[MODEL] DEEP_ONLY=1 → skip LGBM")
        return
    try:
        import lightgbm as lgb, os
        path = os.getenv("LGBM_OUT_PATH", "/data/models/lgbm.txt")
        if os.path.exists(path):
            LGB_MODEL = lgb.Booster(model_file=path)
            log.info("[MODEL] LGBM loaded: %s", path)
    except Exception as e:
        log.warning("[MODEL] LGBM load failed: %s", e)

def _load_rnn():
    global RNN_MODEL
    if DISABLE_RNN == 1:
        log.info("[MODEL] DISABLE_RNN=1 → skip RNN")
        return
    try:
        import torch
        import torch.nn as nn
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS","1")))
        class TinyRNN(nn.Module):
            def __init__(self, in_dim=3, hid=RNN_HIDDEN, out_dim=3):
                super().__init__()
                self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
                self.fc  = nn.Linear(hid, out_dim)
            def forward(self, x):
                o,_ = self.gru(x); return self.fc(o[:, -1, :])
        path = os.getenv("RNN_OUT_PATH", "/data/models/rnn.pt")
        if os.path.exists(path):
            RNN_MODEL = TinyRNN()
            state = torch.load(path, map_location="cpu")
            RNN_MODEL.load_state_dict(state); RNN_MODEL.eval()
            log.info("[MODEL] RNN loaded: %s (hidden=%s)", path, RNN_HIDDEN)
        else:
            log.warning("[MODEL] RNN file not found at %s", path)
    except Exception as e:
        log.warning("[MODEL] RNN load failed: %s", e)

_load_xgb(); _load_lgb(); _load_rnn()

# ===== Big Road & Features (self-contained) =====
MAP = {"B":0, "P":1, "T":2}
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
    grid_sign = np.zeros((rows, cols), dtype=np.int8)
    grid_ties = np.zeros((rows, cols), dtype=np.int16)
    r = 0; c = 0; last_bp = None
    for v in seq:
        if v == 2:
            if 0 <= r < rows and 0 <= c < cols: grid_ties[r, c] += 1
            continue
        cur_bp = +1 if v==0 else -1
        if last_bp is None:
            r,c=0,0; grid_sign[r,c]=cur_bp; last_bp=cur_bp; continue
        if cur_bp == last_bp:
            nr=r+1; nc=c
            if nr>=rows or grid_sign[nr,nc]!=0:
                nr=r; nc=c+1
            r,c=nr,nc
            if 0<=r<rows and 0<=c<cols: grid_sign[r,c]=cur_bp
        else:
            c=c+1; r=0; last_bp=cur_bp
            if c<cols: grid_sign[r,c]=cur_bp
    return grid_sign, grid_ties, (r,c)

def _global_aggregates(seq: List[int]) -> np.ndarray:
    n = len(seq)
    if n == 0:
        return np.array([0.49,0.49,0.02, 0.5,0.5, 0.0,0.0,0.0,0.0, 0.5,0.5,0.5,0.5, 0.0], dtype=np.float32)
    arr = np.array(seq, dtype=np.int16)
    cnt = np.bincount(arr, minlength=3).astype(np.float32); freq = cnt / n
    bp = arr[arr != 2]
    altern = 0.5 if len(bp)<2 else float(np.mean(bp[1:] != bp[:-1]))
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
        a,b = bp[i], bp[i+1]
        if a==0:
            cb+=1; b2b+=(b==0); b2p+=(b==1)
        else:
            cp+=1; p2p+=(b==1); p2b+=(b==0)
    B2B=(b2b/cb) if cb>0 else 0.5
    P2P=(p2p/cp) if cp>0 else 0.5
    B2P=(b2p/cb) if cb>0 else 0.5
    P2B=(p2b/cp) if cp>0 else 0.5
    tie_rate=float((arr==2).mean())
    return np.array([
        float(freq[0]), float(freq[1]), float(freq[2]),
        float(altern), float(1.0-altern),
        float(b_mean), float(b_var), float(p_mean), float(p_var),
        float(B2B), float(P2P), float(B2P), float(P2B),
        float(tie_rate)
    ], dtype=np.float32)

def _local_bigroad_feat(seq: List[int], rows:int, cols:int, win:int) -> np.ndarray:
    sub = seq[-win:] if len(seq)>win else seq[:]
    gs, gt, (r,c) = big_road_grid(sub, rows, cols)
    grid_sign_flat = gs.flatten().astype(np.float32)
    grid_tie_flat  = np.clip(gt.flatten(), 0, 3).astype(np.float32) / 3.0
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
    return np.concatenate([
        grid_sign_flat, grid_tie_flat,
        np.array([streak_len/rows, streak_side], dtype=np.float32),
        col_heights,
        np.array([cur_col_height, cur_col_side], dtype=np.float32),
        freq
    ], axis=0)

def big_road_features(seq: List[int], rows:int=6, cols:int=20, win:int=40) -> np.ndarray:
    local=_local_bigroad_feat(seq, rows, cols, win).astype(np.float32)
    if USE_FULL_SHOE:
        glob=_global_aggregates(seq).astype(np.float32)
        lw=max(0.0, LOCAL_WEIGHT); gw=max(0.0, GLOBAL_WEIGHT); s=lw+gw
        if s==0: lw,gw=1.0,0.0
        else:    lw,gw=lw/s,gw/s
        return np.concatenate([local*lw, glob*gw], axis=0).astype(np.float32)
    else:
        return local

def one_hot_seq(seq: List[int], win:int) -> np.ndarray:
    if USE_FULL_SHOE:
        sub = seq[-MAX_RNN_LEN:] if len(seq) > MAX_RNN_LEN else seq[:]
        L = len(sub); oh = np.zeros((1, L, 3), dtype=np.float32)
        for i, v in enumerate(sub):
            if v in (0,1,2): oh[0, i, v] = 1.0
        return oh
    else:
        sub = seq[-win:] if len(seq)>win else seq[:]
        pad = [-1]*max(0, win-len(sub))
        final = (pad+sub)[-win:]
        oh=[]
        for v in final:
            a=[0,0,0]
            if v in (0,1,2): a[v]=1
            oh.append(a)
        return np.array(oh, dtype=np.float32)[np.newaxis, :, :]

def softmax_log(p: np.ndarray, temp: float=1.0) -> np.ndarray:
    x = np.log(np.clip(p,1e-9,None)) / max(1e-9, temp)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

# ===== Model prob functions =====
def xgb_probs(seq: List[int]) -> Optional[np.ndarray]:
    if XGB_MODEL is None: return None
    import xgboost as xgb
    feat = big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN).astype(np.float32)
    d = xgb.DMatrix(feat.reshape(1,-1))
    p = XGB_MODEL.predict(d)[0]
    return np.array(p, dtype=np.float32)

def lgb_probs(seq: List[int]) -> Optional[np.ndarray]:
    if LGB_MODEL is None: return None
    feat = big_road_features(seq, GRID_ROWS, GRID_COLS, FEAT_WIN).astype(np.float32).reshape(1,-1)
    p = LGB_MODEL.predict(feat)[0]
    return np.array(p, dtype=np.float32)

def rnn_probs(seq: List[int]) -> Optional[np.ndarray]:
    if RNN_MODEL is None: return None
    import torch
    x = one_hot_seq(seq, FEAT_WIN)
    with torch.no_grad():
        logits = RNN_MODEL(torch.from_numpy(x))
        logits = logits / max(1e-6, TEMP_RNN)  # RNN 專屬溫度
        p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return p.astype(np.float32)

def _parse_weights(spec: str) -> Dict[str, float]:
    out={"XGB":0.33,"LGBM":0.33,"RNN":0.34}
    try:
        tmp={}
        for part in (spec or "").split(","):
            if ":" in part:
                k,v=part.split(":",1)
                k=k.strip().lower(); v=float(v)
                if k=="xgb":  tmp["XGB"]=v
                if k=="lgb":  tmp["LGBM"]=v
                if k=="rnn":  tmp["RNN"]=v
        if tmp:
            s=sum(max(0.0,x) for x in tmp.values()) or 1.0
            for k in tmp: tmp[k]=max(0.0,tmp[k])/s
            out.update(tmp)
    except: pass
    return out

def heuristic_probs(seq: List[int]) -> Tuple[np.ndarray, str]:
    if not seq:
        return np.array([0.49,0.49,0.02], dtype=np.float32), "prior"
    sub = seq[-FEAT_WIN:] if len(seq)>FEAT_WIN else seq
    cnt = np.bincount(sub, minlength=3).astype(np.float32)
    freq = cnt / max(1,len(sub))
    p0 = 0.90*freq + 0.10*np.array([0.49,0.49,0.02], dtype=np.float32)
    p0[2] = np.clip(p0[2], CLIP_T_MIN, CLIP_T_MAX)
    p0 = np.clip(p0,1e-6,None); p0 = p0/p0.sum()
    return p0, "heuristic"

def vote_and_average(seq: List[int]) -> Tuple[np.ndarray, Dict[str,str], Dict[str,int]]:
    weights=_parse_weights(ENSEMBLE_WEIGHTS)
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
        p = softmax_log(pr, 1.0)  # TEMP_RNN 已用於 logits
        preds.append(p); names.append("RNN")
        vote_labels['RNN']=label_map[int(pr.argmax())]; vote_counts[vote_labels['RNN']]+=1

    if not preds:
        ph, _ = heuristic_probs(seq)
        return ph, {}, {'莊':0,'閒':0,'和':0}

    W=[]
    for n in names:
        W.append(weights.get(n, 0.0))
    W=np.array(W, dtype=np.float32)
    if W.sum()<=0: W=np.ones_like(W)/len(W)
    W=W/W.sum()

    P=np.stack(preds, axis=0).astype(np.float32)
    p_avg=(P*W[:,None]).sum(axis=0)
    p_avg[2]=np.clip(p_avg[2], CLIP_T_MIN, CLIP_T_MAX)
    p_avg=np.clip(p_avg, 1e-6, None); p_avg=p_avg/p_avg.sum()
    return p_avg, vote_labels, vote_counts

# ===== Volatility / Decision =====
def _alt_flip_metrics(seq: List[int], win: int = 24) -> Tuple[float, float]:
    if not seq: return 0.5, 0.5
    sub = [x for x in seq[-win:] if x in (0,1)]
    if len(sub) <= 1: return 0.5, 0.5
    diffs = np.array(sub[1:]) != np.array(sub[:-1])
    flip_ratio = float(diffs.mean())
    alternation = flip_ratio
    return alternation, flip_ratio

def edge_to_base_pct(edge: float) -> float:
    if edge >= max(0.10, MIN_EDGE+0.02): return 0.30
    if edge >= max(0.08, MIN_EDGE):      return 0.20
    if edge >= max(0.05, MIN_EDGE-0.01): return 0.10
    return 0.0

def _online_edge_boost(sess: Optional[Dict[str,object]]) -> float:
    if not ONLINE_ADAPT or not sess: return 0.0
    stat = sess.get("perf", {"ok":0,"ng":0,"boost":0.0})
    ok, ng = int(stat.get("ok",0)), int(stat.get("ng",0))
    n = ok + ng
    if n < ONLINE_MIN_SAMPLES: return float(stat.get("boost",0.0))
    acc = ok / max(1,n)
    boost = float(stat.get("boost",0.0))
    if acc < ONLINE_ACC_LOW:
        boost = min(EDGE_ADAPT_CAP, boost + EDGE_STEP_UP)
    elif acc >= ONLINE_ACC_HIGH and boost > 0.0:
        boost = max(0.0, boost - EDGE_STEP_DOWN)
    stat["boost"]=boost
    sess["perf"]=stat
    return boost

def decide_bet_from_votes(p: np.ndarray, votes: Dict[str,int], models_used:int,
                          seq: Optional[List[int]] = None, sess: Optional[Dict[str,object]] = None) -> Tuple[str,float,float,float,str]:
    arr = [(float(p[0]),"莊"), (float(p[1]),"閒"), (float(p[2]),"和")]
    arr.sort(reverse=True, key=lambda x: x[0])
    (p1, lab1), (p2, _) = arr[0], arr[1]
    edge = p1 - p2

    max_votes = max(votes.get("莊",0), votes.get("閒",0), votes.get("和",0)) if models_used>0 else 0
    vote_conf = (max_votes / models_used) if models_used>0 else 0.0

    # 票數門檻
    if models_used>0 and max_votes < ABSTAIN_VOTES:
        return "觀望", edge, 0.0, vote_conf, f"🟡 票數不足：{max_votes}/{models_used}"

    # Tie 避免
    if lab1 == "和" and p[2] < max(0.05, CLIP_T_MIN + 0.01):
        return "觀望", edge, 0.0, vote_conf, ""

    # 進場門檻：基礎 + 震盪 + 線上回饋
    vol_note=""; enter_th = max(MIN_EDGE, ABSTAIN_EDGE, EDGE_ENTER)
    if VOL_GUARD and seq is not None:
        alt, flip = _alt_flip_metrics(seq, ALT_WIN)
        if abs(alt-0.5) < VOL_ALT_BAND:
            enter_th += VOL_ALT_BOOST; vol_note += f"交替≈{alt:.2f}+{VOL_ALT_BOOST:.2f}；"
        if flip >= VOL_FLIP_TH:
            enter_th += VOL_FLIP_BOOST; vol_note += f"翻轉{flip:.2f}+{VOL_FLIP_BOOST:.2f}；"
    boost = _online_edge_boost(sess)
    enter_th += boost
    if vol_note or boost>0:
        vol_note = f"⚙️ 門檻 {enter_th:.3f}（{vol_note}{'回饋+'+str(round(boost,3)) if boost>0 else ''}）".strip("（ ）")

    if edge < enter_th:
        return "觀望", edge, 0.0, vote_conf, vol_note

    base_pct = edge_to_base_pct(edge)
    if base_pct == 0.0:
        return "觀望", edge, 0.0, vote_conf, vol_note

    scale = 0.5 + 0.5*vote_conf
    bet_pct = base_pct * scale
    bet_pct = float(np.clip(bet_pct, 0.05 if base_pct>0 else 0.0, 0.30))
    return lab1, edge, bet_pct, vote_conf, vol_note

def vote_summary_text(vote_counts: Dict[str,int], models_used:int) -> str:
    return f"莊 {vote_counts.get('莊',0)}/{models_used}, 閒 {vote_counts.get('閒',0)}/{models_used}, 和 {vote_counts.get('和',0)}/{models_used}"

# ===== 文案 =====
def fmt_line_reply(n_hand:int, p:np.ndarray, sug:str, edge:float,
                   bankroll:int, bet_pct:float, vote_labels:Dict[str,str],
                   vote_counts:Dict[str,int], models_used:int, remain_min:Optional[int],
                   vol_note:Optional[str]=None) -> str:
    b, pl, t = p[0], p[1], p[2]
    lines = []
    lines.append(f"📊 已解析 {n_hand} 手")
    lines.append(f"📈 平均機率：莊 {b:.3f}｜閒 {pl:.3f}｜和 {t:.3f}")
    if models_used>0:
        vline = f"🗳️ 投票（{models_used} 模型）：{vote_summary_text(vote_counts, models_used)}"
        who = []
        for k in ["XGB","LGBM","RNN"]:
            if k in vote_labels: who.append(f"{k}→{vote_labels[k]}")
        if who: vline += "｜" + "，".join(who)
        lines.append(vline)
    lines.append(f"🧩 特徵權重：Local {int(LOCAL_WEIGHT*100)}% / Global {int(GLOBAL_WEIGHT*100)}%")
    badge = "🎯" if sug != "觀望" else "🟡"
    lines.append(f"👉 下一手建議：{sug} {badge}（邊際 {edge:.3f}）")
    if vol_note: lines.append(vol_note)
    if bankroll and bet_pct>0:
        bet_amt = int(round(bankroll * bet_pct))
        lines.append(f"💵 本金：{bankroll:,}")
        lines.append(f"✅ 建議下注：{bet_amt:,} ＝ {bankroll:,} × {bet_pct*100:.1f}%")
        lines.append(f"🧮 10%={int(round(bankroll*0.10)):,}｜20%={int(round(bankroll*0.20)):,}｜30%={int(round(bankroll*0.30)):,}")
    if remain_min is not None and SHOW_REMAINING_TIME:
        lines.append(f"⏳ 試用剩餘：約 {max(0, remain_min)} 分鐘")
    lines.append("📝 操作：輸入莊/閒/和（或 B/P/T）；『開始分析』；『返回』撤回；『結束分析』清空；『結果 莊/閒/和』回報上一手實盤。")
    return "\n".join(lines)

def fmt_trial_over() -> str:
    return (
        "⛔ 免費試用已結束。\n"
        f"📬 請聯繫管理員官方 LINE：{ADMIN_CONTACT} 取得開通密碼。\n"
        "🔐 直接輸入：【開通 你的密碼】（例如：開通 abc123）"
    )

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

# ===== API =====
@app.route("/", methods=["GET"])
def root():
    return "LiveBoot ok", 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify(status="ok"), 200

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    action = str(data.get("action","")).strip().lower()
    session_key = data.get("session_key")
    bankroll_in = data.get("bankroll")
    history = data.get("history", "")
    activation_code = str(data.get("activation_code","")).strip()

    if API_TRIAL_ENFORCE and not session_key:
        return jsonify(error="session_key_required",
                       message="API trial is ON. Provide session_key or activation_code."), 400

    if session_key:
        sess = SESS_API.setdefault(session_key, {"bankroll": 0, "seq": [], "trial_start": int(time.time()), "premium": False})
        if activation_code and ADMIN_ACTIVATION_SECRET and (activation_code == ADMIN_ACTIVATION_SECRET):
            sess["premium"] = True
        if API_TRIAL_ENFORCE and not sess.get("premium", False):
            now = int(time.time())
            start = int(sess.get("trial_start", now))
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

    p_avg, vote_labels, vote_counts = vote_and_average(seq)
    models_used = len(vote_labels)
    sug, edge, bet_pct, vote_conf, vol_note = decide_bet_from_votes(p_avg, vote_counts, models_used, seq, None)

    if API_MINIMAL_JSON:
        return jsonify(
            hands=len(seq),
            probs={"banker": round(float(p_avg[0]),3), "player": round(float(p_avg[1]),3), "tie": round(float(p_avg[2]),3)},
            suggestion=sug,
            edge=round(float(edge),3),
            bet_pct=float(bet_pct),
            bet_amount=int(round(bankroll*bet_pct)) if bankroll and bet_pct>0 else 0
        )

    history_str = encode_history(seq)
    text = fmt_line_reply(len(seq), p_avg, sug, edge, bankroll, bet_pct, vote_labels, vote_counts, models_used, None, vol_note)

    return jsonify({
        "history_str": history_str,
        "hands": len(seq),
        "probs": {"banker": round(float(p_avg[0]),3), "player": round(float(p_avg[1]),3), "tie": round(float(p_avg[2]),3)},
        "suggestion": sug,
        "edge": round(float(edge),3),
        "bet_pct": float(bet_pct),
        "bet_amount": int(round(bankroll*bet_pct)) if bankroll and bet_pct>0 else 0,
        "votes": {"models_used": models_used, "莊": vote_counts.get("莊",0), "閒": vote_counts.get("閒",0), "和": vote_counts.get("和",0)},
        "vote_summary": vote_summary_text(vote_counts, models_used),
        "vol_note": vol_note,
        "message": text
    })

# ===== LINE webhook =====
@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not line_handler or not line_api:
        abort(503, "LINE not configured")
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400, "Invalid signature")
    return "OK", 200

@line_handler.add(FollowEvent)
def on_follow(event):
    uid = event.source.user_id
    now = int(time.time())
    SESS[uid] = {"bankroll": 0, "seq": [], "trial_start": now, "premium": False, "perf": {"ok":0,"ng":0,"boost":0.0}}
    mins = TRIAL_MINUTES
    msg = (
        "🤖 歡迎加入！\n"
        f"🎁 已啟用 {mins} 分鐘免費試用，現在就開始吧！\n"
        "先輸入本金（例如：5000），再貼歷史（B/P/T 或 莊/閒/和），輸入『開始分析』即可！📊\n"
        "🔐 到期後輸入：開通 你的密碼（向管理員索取）\n"
        f"👤 管理員：{ADMIN_CONTACT}"
    )
    line_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=quick_reply_buttons()))

@line_handler.add(MessageEvent, message=TextMessage)
def on_text(event):
    uid = event.source.user_id
    text = (event.message.text or "").strip()
    sess = SESS.setdefault(uid, {"bankroll": 0, "seq": [], "trial_start": int(time.time()), "premium": False, "perf": {"ok":0,"ng":0,"boost":0.0}})

    # 試用檢查
    if not sess.get("premium", False):
        start = int(sess.get("trial_start", int(time.time())))
        now   = int(time.time())
        elapsed_min = (now - start) // 60
        remain_min = max(0, TRIAL_MINUTES - elapsed_min)
        if elapsed_min >= TRIAL_MINUTES:
            if text.startswith("開通") or text.lower().startswith("activate"):
                code = text.split(" ",1)[1].strip() if " " in text else ""
                if validate_activation_code(code):
                    sess["premium"] = True
                    safe_reply(event.reply_token, "✅ 已開通成功！現在可以繼續使用所有功能。🎉", uid)
                else:
                    safe_reply(event.reply_token, "❌ 開通密碼不正確，請向管理員索取正確密碼。", uid)
            else:
                safe_reply(event.reply_token, fmt_trial_over(), uid)
            return
    else:
        remain_min = None

    # 系統指令：返回
    if text in ["返回", "undo", "回上一步"]:
        seq: List[int] = sess.get("seq", [])
        if seq:
            last = seq.pop(-1)
            sess["seq"] = seq
            msg = f"↩️ 已撤回上一手（{INV.get(last,'?')}）。目前共 {len(seq)} 手。輸入『開始分析』或再輸入下一手。"
        else:
            msg = "ℹ️ 目前沒有可撤回的紀錄。請先輸入歷史或單手結果。"
        safe_reply(event.reply_token, msg, uid); return

    # 系統指令：結束分析
    if text in ["結束分析", "清空", "reset"]:
        sess["seq"] = []
        msg = "🧹 已清空歷史。保留本金設定不變。\n貼上新歷史（B/P/T 或 莊/閒/和），或直接輸入單手結果開始紀錄。"
        safe_reply(event.reply_token, msg, uid); return

    # 開通
    if text.startswith("開通") or text.lower().startswith("activate"):
        code = text.split(" ",1)[1].strip() if " " in text else ""
        if validate_activation_code(code):
            sess["premium"] = True
            safe_reply(event.reply_token, "✅ 已開通成功！現在可以繼續使用所有功能。🎉", uid)
        else:
            safe_reply(event.reply_token, "❌ 開通密碼不正確，請向管理員索取正確密碼。", uid)
        return

    # 本金設定（純數字）
    if text.isdigit():
        sess["bankroll"] = int(text)
        msg = f"👍 已設定本金：{int(text):,} 元。\n貼上歷史（B/P/T 或 莊/閒/和）後輸入『開始分析』即可！🚀"
        safe_reply(event.reply_token, msg, uid); return

    # 結果回報（線上表現回饋）
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
                sess["perf"] = perf
                acc = perf["ok"]/max(1,(perf["ok"]+perf["ng"]))
                msg = f"📥 已記錄結果：{token}（{'✅ 命中' if ok else '❌ 未中'}）\n"
                msg+= f"近況：{perf['ok']} 中 / {perf['ok']+perf['ng']} 次（命中率 {acc:.2f}）"
            else:
                msg = "ℹ️ 尚無上一手建議或結果格式錯誤。用法：結果 莊/閒/和（或 B/P/T）。"
        else:
            msg = "ℹ️ 用法：結果 莊/閒/和（或 B/P/T）。"
        safe_reply(event.reply_token, msg, uid); return

    # 歷史/單手輸入
    zh2eng = {"莊":"B","閒":"P","和":"T"}
    norm = "".join(zh2eng.get(ch, ch) for ch in text.upper())
    seq_in = parse_history(norm)
    if seq_in and ("開始分析" not in text):
        if len(seq_in) == 1:
            sess.setdefault("seq", [])
            sess["seq"].append(seq_in[0])
            n = len(sess["seq"])
            msg = f"✅ 已記錄 1 手：{norm}。目前累計 {n} 手。\n輸入『開始分析』或繼續輸入下一手（或用『返回』撤回）。"
        else:
            sess["seq"] = seq_in
            n = len(seq_in)
            msg = f"✅ 已覆蓋歷史共 {n} 手。\n輸入『開始分析』即可啟動。🧪"
        safe_reply(event.reply_token, msg, uid); return

    # 分析
    if ("開始分析" in text) or (text in ["分析", "開始", "GO", "go"]):
        sseq: List[int] = sess.get("seq", [])
        bankroll: int = int(sess.get("bankroll", 0) or 0)
        p_avg, vote_labels, vote_counts = vote_and_average(sseq)
        models_used = len(vote_labels)
        sug, edge, bet_pct, vote_conf, vol_note = decide_bet_from_votes(p_avg, vote_counts, models_used, sseq, sess)
        sess["last_suggestion"] = sug if sug in ("莊","閒","和") else None
        reply = fmt_line_reply(len(sseq), p_avg, sug, edge, bankroll, bet_pct, vote_labels, vote_counts, models_used, remain_min, vol_note)
        safe_reply(event.reply_token, reply, uid); return

    # 說明
    msg = (
        "🧭 指令說明：\n"
        "• 數字：設定本金（例：5000）\n"
        "• 貼歷史：B/P/T 或 莊/閒/和（可含空白）\n"
        "• 『開始分析』：三模型加權＋RNN 溫度強化\n"
        "• 『返回』撤回上一手；『結束分析』清空歷史\n"
        "• 『結果 莊/閒/和』回報上一手實盤（用於線上回饋）\n"
        "• 試用到期後：『開通 你的密碼』\n"
        f"• 管理員：{ADMIN_CONTACT}"
    )
    safe_reply(event.reply_token, msg, uid)

# ===== Utils =====
def validate_activation_code(code: str) -> bool:
    if not ADMIN_ACTIVATION_SECRET:
        return False
    return bool(code) and (code == ADMIN_ACTIVATION_SECRET)

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

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    app.run(host="0.0.0.0", port=port)
