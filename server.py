# server.py - 改進版百家樂預測系統（整合場態控制與平衡決策）

import os, logging, time, csv
from typing import List, Tuple, Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort

log = logging.getLogger("liveboot-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
app = Flask(__name__)

# ---------- 配置參數（經修正與平衡）----------
def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    if v == "1/0": return 1
    try: return 1 if int(float(v)) != 0 else 0
    except: return 1 if default else 0

# 基礎參數 - 提高判斷穩定性
FEAT_WIN   = int(os.getenv("FEAT_WIN", "40"))  # 稍稍加長特徵窗口
GRID_ROWS  = int(os.getenv("GRID_ROWS", "6"))
GRID_COLS  = int(os.getenv("GRID_COLS", "20"))
MIN_EDGE   = float(os.getenv("MIN_EDGE", "0.035")) # 略微提高最小邊際，避免過度頻繁進場
CLIP_T_MIN = float(os.getenv("CLIP_T_MIN", "0.06"))
CLIP_T_MAX = float(os.getenv("CLIP_T_MAX", "0.20"))
SEED       = int(os.getenv("SEED", "42"))
np.random.seed(SEED)

# 權重平衡 - 解決趨勢跟隨偏誤的關鍵
USE_FULL_SHOE = env_flag("USE_FULL_SHOE", 1)
LOCAL_WEIGHT  = float(os.getenv("LOCAL_WEIGHT", "0.5"))  # [修改] 平衡本地與全域權重
GLOBAL_WEIGHT = float(os.getenv("GLOBAL_WEIGHT", "0.5"))  # [修改] 平衡本地與全域權重
MAX_RNN_LEN   = int(os.getenv("MAX_RNN_LEN", "128"))

# 試用設定
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@jins888")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")
API_TRIAL_ENFORCE = env_flag("API_TRIAL_ENFORCE", 0)
API_TRIAL_MINUTES = int(os.getenv("API_TRIAL_MINUTES", str(TRIAL_MINUTES)))
CRON_TOKEN = os.getenv("CRON_TOKEN", "")

# 模型設定
DEEP_ONLY   = int(os.getenv("DEEP_ONLY", "0"))
DISABLE_RNN = int(os.getenv("DISABLE_RNN", "0"))
RNN_HIDDEN  = int(os.getenv("RNN_HIDDEN", "32"))
ENSEMBLE_WEIGHTS = os.getenv("ENSEMBLE_WEIGHTS", "xgb:0.3,lgb:0.3,rnn:0.4")

# 溫度參數
TEMP_XGB = float(os.getenv("TEMP_XGB", "0.9"))
TEMP_LGB = float(os.getenv("TEMP_LGB", "0.9"))
TEMP_RNN = float(os.getenv("TEMP_RNN", "0.8"))

# 進場門檻
EDGE_ENTER    = float(os.getenv("EDGE_ENTER", "0.03"))

# 場態控制 - 解決問題的核心
REGIME_CTRL   = int(os.getenv("REGIME_CTRL", "1"))  # [修改] 預設啟用場態控制
REG_WIN       = int(os.getenv("REG_WIN", "25"))     # [修改] 加長場態分析窗口
REG_STREAK_TH = float(os.getenv("REG_STREAK_TH", "0.65")) # 長龍趨勢門檻
REG_CHOP_TH   = float(os.getenv("REG_CHOP_TH", "0.60")) # 單跳趨勢門檻
REG_SIDE_BIAS = float(os.getenv("REG_SIDE_BIAS", "0.70")) # 強勢邊門檻

# 場態調整參數
REG_ALIGN_EDGE_BONUS      = float(os.getenv("REG_ALIGN_EDGE_BONUS", "0.02")) # 順勢加成
REG_MISMATCH_EDGE_PENALTY = float(os.getenv("REG_MISMATCH_EDGE_PENALTY", "0.04")) # 逆勢懲罰

# 同邊投注限制
SAME_SIDE_SOFT_CAP = int(os.getenv("SAME_SIDE_SOFT_CAP", "4")) # 連續同邊4次後開始謹慎
SAME_SIDE_PENALTY  = float(os.getenv("SAME_SIDE_PENALTY", "0.015")) # 同邊懲罰

FORCE_DIRECTION_WHEN_UNDEREDGE = int(os.getenv("FORCE_DIRECTION_WHEN_UNDEREDGE", "1"))

# ---------- LINE設定 ----------
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
    line_api = None; line_handler = None
    log.warning("LINE SDK not fully available: %s", e)

# ---------- 全域變數 ----------
SESS: Dict[str, Dict[str, object]] = {}
SESS_API: Dict[str, Dict[str, object]] = {}
XGB_MODEL = None; LGB_MODEL = None; RNN_MODEL = None

# ---------- 模型載入 (與原版相同) ----------
def _load_xgb():
    global XGB_MODEL
    if DEEP_ONLY == 1: return
    try:
        import xgboost as xgb
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
        import lightgbm as lgb
        path = os.getenv("LGBM_OUT_PATH", "data/models/lgbm.txt")
        if os.path.exists(path):
            LGB_MODEL = lgb.Booster(model_file=path)
            log.info("[MODEL] LGBM loaded: %s", path)
    except Exception as e:
        log.warning("[MODEL] LGBM load failed: %s", e)

def _load_rnn():
    global RNN_MODEL
    if DISABLE_RNN == 1:
        log.info("[MODEL] RNN disabled by env"); return
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

# ---------- 基礎功能 (與原版大致相同) ----------
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
    if n==0: return np.array([0.45,0.45,0.1, 0.5,0.5, 0,0,0,0, 0.5,0.5,0.5,0.5, 0.1], dtype=np.float32)
    arr=np.array(seq, dtype=np.int16)
    cnt=np.bincount(arr, minlength=3).astype(np.float32); freq=cnt/n
    bp=arr[arr!=2]
    altern=0.5 if len(bp)<2 else float(np.mean(bp[1:]!=bp[:-1]))
    def run_stats(side):
        x=(bp==side).astype(np.int8);
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
    B2B=(b2b/cb) if cb>0 else 0.5; P2P=(p2p/cp) if cp>0 else 0.5
    B2P=(b2p/cb) if cb>0 else 0.5; P2B=(p2b/cp) if cp>0 else 0.5
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
                           col_heights, np.array([cur_col_height, cur_col_side], dtype=np.float32),
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
        pad = [-1]*max(0, win-len(sub)); final=(pad+sub)[-win:]
        oh=[]
        for v in final: a=[0,0,0]; (v in (0,1,2)) and (a[v]:=1); oh.append(a)
        return np.array(oh, dtype=np.float32)[np.newaxis,:,:]

def softmax_log(p: np.ndarray, temp: float=1.0) -> np.ndarray:
    x = np.log(np.clip(p,1e-9,None)) / max(1e-9, temp)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

# ---------- [新增] 場態分析 ----------
def detect_regime(seq: List[int]) -> str:
    """分析最近的歷史紀錄以判斷當前的場態"""
    bp = [x for x in seq if x != 2]
    if len(bp) < REG_WIN: return "NORMAL"
    
    sub = bp[-REG_WIN:]
    
    # 單跳檢測
    chops = sum(1 for i in range(1, len(sub)) if sub[i] != sub[i-1])
    chop_rate = chops / (len(sub) - 1)
    if chop_rate >= REG_CHOP_TH: return "CHOP"
    
    # 長龍檢測
    streaks = sum(1 for i in range(1, len(sub)) if sub[i] == sub[i-1])
    streak_rate = streaks / (len(sub) - 1)
    if streak_rate >= REG_STREAK_TH: return "STREAK"

    # 強勢邊檢測
    b_count = sub.count(0)
    p_count = sub.count(1)
    if b_count / len(sub) >= REG_SIDE_BIAS or p_count / len(sub) >= REG_SIDE_BIAS:
        return "SIDE_BIAS"

    return "NORMAL"

# ---------- 模型預測 (與原版大致相同) ----------
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
    import torch
    x=one_hot_seq(seq, FEAT_WIN)
    with torch.no_grad():
        logits=RNN_MODEL(torch.from_numpy(x))
        logits = logits / max(1e-6, TEMP_RNN)
        p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return p.astype(np.float32)

def _parse_weights(spec: str) -> Dict[str, float]:
    out={"XGB":0.33,"LGBM":0.33,"RNN":0.34};
    try:
        tmp={};
        for part in (spec or "").split(","):
            if ":" in part:
                k,v=part.split(":",1); k=k.strip().lower(); v=float(v)
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
    if not seq: return np.array([0.45,0.45,0.1], dtype=np.float32)
    sub=seq[-FEAT_WIN:] if len(seq)>FEAT_WIN else seq
    cnt=np.bincount(sub, minlength=3).astype(np.float32); freq=cnt/max(1,len(sub))
    p0=0.8*freq + 0.2*np.array([0.42,0.42,0.16], dtype=np.float32)
    p0[2]=np.clip(p0[2], CLIP_T_MIN, CLIP_T_MAX)
    p0=np.clip(p0,1e-6,None); p0=p0/p0.sum()
    return p0

def enhanced_ensemble(seq: List[int]) -> Tuple[np.ndarray, Dict[str,str], Dict[str,int], str]:
    weights=_parse_weights(ENSEMBLE_WEIGHTS); preds=[]; names=[];
    vote_labels={}; vote_counts={'莊':0,'閒':0,'和':0}; label_map=["莊","閒","和"]
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
        p = softmax_log(pr, TEMP_RNN); preds.append(p); names.append("RNN")
        vote_labels['RNN']=label_map[int(pr.argmax())]; vote_counts[vote_labels['RNN']]+=1
    
    regime = detect_regime(seq) if REGIME_CTRL else "DISABLED"

    if not preds: return heuristic_probs(seq), {}, {'莊':0,'閒':0,'和':0}, regime

    W=np.array([weights.get(n,0.0) for n in names], dtype=np.float32)
    if W.sum()<=0: W=np.ones_like(W)/len(W)
    W=W/W.sum()
    P=np.stack(preds, axis=0).astype(np.float32)
    p_avg=(P*W[:,None]).sum(axis=0)
    p_avg[2]=np.clip(p_avg[2], CLIP_T_MIN, CLIP_T_MAX)
    p_avg=np.clip(p_avg,1e-6,None); p_avg=p_avg/p_avg.sum()

    return p_avg, vote_labels, vote_counts, regime

# ---------- [修改] 決策邏輯 ----------
def flexible_decision(p: np.ndarray, seq: List[int], regime: str) -> Tuple[str, float, float]:
    best_idx = int(np.argmax(p))
    best_label = "莊" if best_idx==0 else ("閒" if best_idx==1 else "和")
    
    sorted_probs = sorted(p, reverse=True)
    edge = float(sorted_probs[0] - sorted_probs[1])
    
    # 修正信心指數邏輯
    confidence_multiplier = 1.0
    if seq and len(seq) > 10:
        recent = [x for x in seq[-10:] if x != 2]
        if len(recent) > 1:
            changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
            change_rate = changes / (len(recent) - 1)
            # 變化率越高(單跳)，信心越低
            confidence_multiplier = 1.2 - 0.6 * change_rate
    
    adjusted_edge = edge * confidence_multiplier
    
    # 獲取當前連勝狀態
    bp_only = [x for x in seq if x != 2]
    streak_len = 0
    if bp_only:
        last = bp_only[-1]
        for v in reversed(bp_only):
            if v == last: streak_len += 1
            else: break

    # 場態調整
    if REGIME_CTRL:
        prediction_is_streak = (best_idx == (bp_only[-1] if bp_only else -1))
        prediction_is_chop = (best_idx != (bp_only[-1] if bp_only else -1))

        if regime == "STREAK" and prediction_is_streak:
            adjusted_edge += REG_ALIGN_EDGE_BONUS
        elif regime == "STREAK" and prediction_is_chop:
            adjusted_edge -= REG_MISMATCH_EDGE_PENALTY
        elif regime == "CHOP" and prediction_is_chop:
            adjusted_edge += REG_ALIGN_EDGE_BONUS
        elif regime == "CHOP" and prediction_is_streak and streak_len >= 2:
            adjusted_edge -= REG_MISMATCH_EDGE_PENALTY

    # 連續同邊懲罰
    if streak_len >= SAME_SIDE_SOFT_CAP and best_idx == (bp_only[-1] if bp_only else -1):
        adjusted_edge -= (streak_len - SAME_SIDE_SOFT_CAP + 1) * SAME_SIDE_PENALTY

    # 最終決策
    if adjusted_edge < EDGE_ENTER:
        if FORCE_DIRECTION_WHEN_UNDEREDGE:
            return f"{best_label}（觀）", edge, 0.0
        else:
            return "觀望", edge, 0.0
    
    # 動態下注比例
    if adjusted_edge >= 0.10: bet_pct = 0.25
    elif adjusted_edge >= 0.07: bet_pct = 0.15
    elif adjusted_edge >= 0.04: bet_pct = 0.10
    else: bet_pct = 0.05
    
    return best_label, edge, bet_pct

def bet_amount(bankroll:int, pct:float) -> int:
    if not bankroll or bankroll<=0 or pct<=0: return 0
    return int(round(bankroll*pct))

def bet_ladder_text(bankroll: int) -> str:
    if not bankroll or bankroll <= 0: return "💴 配注：5%｜10%｜15%｜25%（先輸入本金）"
    a=int(round(bankroll*0.05)); b=int(round(bankroll*0.10))
    c=int(round(bankroll*0.15)); d=int(round(bankroll*0.25))
    return f"💴 配注 5% {a:,}｜10% {b:,}｜15% {c:,}｜25% {d:,}"

def simple_reply(n_hand:int, lab:str, edge:float, p:np.ndarray, bankroll:int, bet_pct:float, regime:str) -> str:
    conf = int(round(100*max(p)))
    amt = bet_amount(bankroll, bet_pct)
    b_pct, p_pct, t_pct = int(round(100*p[0])), int(round(100*p[1])), int(round(100*p[2]))
    prob_display = f"莊{b_pct}%｜閒{p_pct}%｜和{t_pct}%"
    
    regime_map = {"NORMAL": "標準", "STREAK": "長龍", "CHOP": "單跳", "SIDE_BIAS": "偏格", "DISABLED": "關閉"}
    regime_text = f"場態：{regime_map.get(regime, '未知')}"

    if bet_pct > 0 and amt > 0:
        return f"🎯 下一局：{lab}（{conf}%）💰 {amt:,}\n📊 {prob_display}\n🧠 {regime_text}"
    else:
        return f"👁️ 下一局：{lab}（{conf}%）⚪\n📊 {prob_display}\n🧠 {regime_text}"

def trial_over_text() -> str:
    return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"

# ---------- LINE功能 & HTTP API (與原版大致相同, 僅修改 predict 調用) ----------
def quick_reply_buttons():
    try:
        return QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="莊", text="莊")), QuickReplyButton(action=MessageAction(label="閒", text="閒")),
            QuickReplyButton(action=MessageAction(label="和", text="和")), QuickReplyButton(action=MessageAction(label="開始分析", text="開始分析")),
            QuickReplyButton(action=MessageAction(label="返回 ⬅️", text="返回")), QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
        ])
    except Exception: return None

def _init_user(uid:str):
    now = int(time.time())
    SESS[uid] = {"bankroll": 0, "seq": [], "trial_start": now, "premium": False, "last_suggestion": None}

def safe_reply(reply_token: str, text: str, uid: Optional[str] = None):
    try: line_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=quick_reply_buttons()))
    except Exception as e:
        log.warning("[LINE] reply failed, try push: %s", e)
        if uid:
            try: line_api.push_message(uid, TextSendMessage(text=text, quick_reply=quick_reply_buttons()))
            except Exception as e2: log.error("[LINE] push failed: %s", e2)

def validate_activation_code(code: str) -> bool:
    return bool(ADMIN_ACTIVATION_SECRET) and bool(code) and (code == ADMIN_ACTIVATION_SECRET)

def trial_guard(uid:str, reply_token:str) -> bool:
    sess = SESS.get(uid) or {}
    if sess.get("premium", False): return False
    now = int(time.time()); start = int(sess.get("trial_start", now))
    if (now - start) // 60 >= TRIAL_MINUTES:
        safe_reply(reply_token, trial_over_text(), uid); return True
    return False

@app.get("/")
def root(): return "LiveBoot Enhanced (Fixed) ok", 200
@app.get("/health")
def health(): return jsonify(status="ok"), 200
@app.get("/healthz")
def healthz(): return jsonify(status="ok"), 200

@app.get("/cron")
def cron():
    token = request.args.get("token","")
    if not CRON_TOKEN or token != CRON_TOKEN: abort(403)
    now = int(time.time()); cnt = 0
    for uid, sess in list(SESS.items()):
        if sess.get("premium"): continue
        start = int(sess.get("trial_start", now))
        if (now - start) // 60 >= TRIAL_MINUTES and not sess.get("trial_notified"):
            try:
                line_api.push_message(uid, TextSendMessage(text=trial_over_text(), quick_reply=quick_reply_buttons()))
                sess["trial_notified"] = True; cnt += 1
            except Exception as e: log.warning("cron push failed: %s", e)
    return jsonify(pushed=cnt), 200

@app.post("/predict")
def predict_api():
    data = request.get_json(silent=True) or {}
    session_key = data.get("session_key")
    bankroll_in = data.get("bankroll")
    history = data.get("history", "")
    activation_code = str(data.get("activation_code","")).strip()
    action = str(data.get("action","")).strip().lower()

    if API_TRIAL_ENFORCE and not session_key: return jsonify(error="session_key_required"), 400

    if session_key:
        sess = SESS_API.setdefault(session_key, {"bankroll": 0, "seq": [], "trial_start": int(time.time()), "premium": False})
        if activation_code and validate_activation_code(activation_code): sess["premium"] = True
        if API_TRIAL_ENFORCE and not sess.get("premium", False):
            now = int(time.time()); start = int(sess.get("trial_start", now))
            if (now - start) // 60 >= API_TRIAL_MINUTES: return jsonify(error="trial_expired"), 403
        if bankroll_in is not None:
            try: sess["bankroll"] = int(bankroll_in)
            except: pass
        if history: sess["seq"] = parse_history(history)
        seq = list(sess.get("seq", []))
        bankroll = int(sess.get("bankroll", 0) or 0)
    else:
        seq = parse_history(history)
        bankroll = int(bankroll_in or 0)

    if action == "undo":
        if seq: seq.pop(-1)
        if session_key: SESS_API[session_key]["seq"] = seq
    elif action == "reset":
        seq = []
        if session_key: SESS_API[session_key]["seq"] = []

    p_avg, _, _, regime = enhanced_ensemble(seq)
    lab, edge, bet_pct = flexible_decision(p_avg, seq, regime)

    text = simple_reply(len(seq), lab, edge, p_avg, bankroll, bet_pct, regime)
    return jsonify(message=text, hands=len(seq), suggestion=lab, regime=regime,
                   bet_pct=float(bet_pct), bet_amount=bet_amount(bankroll, bet_pct),
                   probabilities={"banker": float(p_avg[0]), "player": float(p_avg[1]), "tie": float(p_avg[2])}), 200

if line_handler and line_api:
    @line_handler.add(FollowEvent)
    def on_follow(event):
        uid = event.source.user_id; _init_user(uid)
        msg = (f"🤖 歡迎！已啟用 {TRIAL_MINUTES} 分鐘試用\n"
               "請先輸入本金（例：5000）💵\n"
               "再貼歷史（B/P/T 或 莊/閒/和）→『開始分析』📊\n"
               "配注：5%｜10%｜15%｜25%\n"
               f"到期請輸入：開通 你的密碼（向管理員索取）{ADMIN_CONTACT}")
        line_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=quick_reply_buttons()))

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event):
        uid = event.source.user_id; text = (event.message.text or "").strip()
        if uid not in SESS: _init_user(uid)
        sess = SESS[uid]
        if trial_guard(uid, event.reply_token): return

        if text in ["返回", "undo", "回上一步"]:
            if sess.get("seq", []):
                last = sess["seq"].pop(-1)
                safe_reply(event.reply_token, f"↩️ 撤回 {INV.get(last,'?')}，共 {len(sess['seq'])} 手。", uid)
            else:
                safe_reply(event.reply_token, "ℹ️ 沒有可撤回的紀錄。", uid)
            return

        if text in ["結束分析", "清空", "reset"]:
            sess["seq"] = []; sess["bankroll"] = 0; sess["last_suggestion"] = None
            safe_reply(event.reply_token, "🧹 已清空。\n請輸入本金（例：5000）💵\n貼歷史後輸入「開始分析」📊", uid)
            return

        if text.startswith("開通") or text.lower().startswith("activate"):
            code = text.split(" ",1)[1].strip() if " " in text else ""
            if validate_activation_code(code):
                sess["premium"] = True; safe_reply(event.reply_token, "✅ 已開通成功！🎉", uid)
            else:
                safe_reply(event.reply_token, "❌ 密碼錯誤，請向管理員索取。", uid)
            return

        if text.isdigit():
            sess["bankroll"] = int(text)
            safe_reply(event.reply_token, f"👍 已設定本金：{int(text):,}\n{bet_ladder_text(sess['bankroll'])}", uid)
            return

        zh2eng = {"莊":"B","閒":"P","和":"T"}
        norm = "".join(zh2eng.get(ch, ch) for ch in text.upper())
        seq_in = parse_history(norm)
        if seq_in and ("開始分析" not in text):
            if len(seq_in) == 1:
                sess.setdefault("seq", []).append(seq_in[0])
                safe_reply(event.reply_token, f"✅ 已記 1 手：{norm}（共 {len(sess['seq'])}）", uid)
            else:
                sess["seq"] = seq_in
                safe_reply(event.reply_token, f"✅ 已覆蓋歷史：{len(seq_in)} 手", uid)
            return

        if ("開始分析" in text) or (text in ["分析", "開始", "GO", "go"]):
            sseq = sess.get("seq", [])
            bankroll = int(sess.get("bankroll", 0) or 0)
            
            p_avg, _, _, regime = enhanced_ensemble(sseq)
            lab, edge, bet_pct = flexible_decision(p_avg, sseq, regime)
            
            sess["last_suggestion"] = lab if lab in ("莊","閒","和") else None
            reply = simple_reply(len(sseq), lab, edge, p_avg, bankroll, bet_pct, regime)
            
            if not bankroll or bankroll <= 0:
                reply += f"\n{bet_ladder_text(0)}"
            safe_reply(event.reply_token, reply, uid)
            return

        safe_reply(event.reply_token, "🧭 指令：設定本金→貼歷史→『開始分析』｜『返回』撤回｜『結束分析』清空", uid)

@app.post("/line-webhook")
def line_webhook():
    if not line_handler or not line_api: abort(503, "LINE not configured")
    signature = request.headers.get("X-Line-Signature", ""); body = request.get_data(as_text=True)
    try: line_handler.handle(body, signature)
    except InvalidSignatureError: abort(400, "Invalid signature")
    return "OK", 200

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
