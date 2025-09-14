# server.py - 改進版百家樂預測系統（更靈活的預測邏輯）

import os, logging, time, csv
from typing import List, Tuple, Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort

log = logging.getLogger("liveboot-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
app = Flask(__name__)

# ---------- 配置參數（降低閾值，提高靈活性） ----------
def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    if v == "1/0": return 1
    try: return 1 if int(float(v)) != 0 else 0
    except: return 1 if default else 0

# 基礎參數 - 降低門檻提高靈活性
FEAT_WIN   = int(os.getenv("FEAT_WIN", "30"))  # 減少特徵窗口
GRID_ROWS  = int(os.getenv("GRID_ROWS", "6"))
GRID_COLS  = int(os.getenv("GRID_COLS", "20"))
MIN_EDGE   = float(os.getenv("MIN_EDGE", "0.03"))  # 大幅降低最小邊際
CLIP_T_MIN = float(os.getenv("CLIP_T_MIN", "0.05"))  # 提高和局最小概率
CLIP_T_MAX = float(os.getenv("CLIP_T_MAX", "0.25"))  # 提高和局最大概率
SEED       = int(os.getenv("SEED", "42"))
np.random.seed(SEED)

USE_FULL_SHOE = env_flag("USE_FULL_SHOE", 1)
LOCAL_WEIGHT  = float(os.getenv("LOCAL_WEIGHT", "0.7"))
GLOBAL_WEIGHT = float(os.getenv("GLOBAL_WEIGHT", "0.3"))
MAX_RNN_LEN   = int(os.getenv("MAX_RNN_LEN", "128"))  # 減少RNN長度

# 試用設定
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@jins888")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")
SHOW_REMAINING_TIME = env_flag("SHOW_REMAINING_TIME", 1)
API_TRIAL_ENFORCE = env_flag("API_TRIAL_ENFORCE", 0)
API_TRIAL_MINUTES = int(os.getenv("API_TRIAL_MINUTES", str(TRIAL_MINUTES)))
API_MINIMAL_JSON  = env_flag("API_MINIMAL_JSON", 0)
CRON_TOKEN = os.getenv("CRON_TOKEN", "")

# 模型設定 - 增加靈活性
DEEP_ONLY   = int(os.getenv("DEEP_ONLY", "0"))
DISABLE_RNN = int(os.getenv("DISABLE_RNN", "0"))
RNN_HIDDEN  = int(os.getenv("RNN_HIDDEN", "32"))
ENSEMBLE_WEIGHTS = os.getenv("ENSEMBLE_WEIGHTS", "xgb:0.25,lgb:0.25,rnn:0.5")

# 降低溫度參數，提高預測敏感度
TEMP_XGB = float(os.getenv("TEMP_XGB", "0.8"))  # 降低溫度
TEMP_LGB = float(os.getenv("TEMP_LGB", "0.8"))  # 降低溫度
TEMP_RNN = float(os.getenv("TEMP_RNN", "0.7"))  # 降低溫度

# 大幅降低進場門檻
ABSTAIN_EDGE  = float(os.getenv("ABSTAIN_EDGE", "0.02"))  # 降低棄權門檻
ABSTAIN_VOTES = int(os.getenv("ABSTAIN_VOTES", "1"))      # 降低棄權投票需求
EDGE_ENTER    = float(os.getenv("EDGE_ENTER", "0.02"))    # 降低進場門檻

# 簡化場態控制
REGIME_CTRL   = int(os.getenv("REGIME_CTRL", "0"))  # 關閉場態控制，提高靈活性
REG_WIN       = int(os.getenv("REG_WIN", "20"))     # 減少場態窗口
REG_STREAK_TH = float(os.getenv("REG_STREAK_TH", "0.7"))
REG_CHOP_TH   = float(os.getenv("REG_CHOP_TH", "0.7"))
REG_SIDE_BIAS = float(os.getenv("REG_SIDE_BIAS", "0.65"))

# 降低各種懲罰和加成
REG_ALIGN_EDGE_BONUS      = float(os.getenv("REG_ALIGN_EDGE_BONUS", "0.005"))
REG_ALIGN_REQUIRE         = int(os.getenv("REG_ALIGN_REQUIRE", "0"))  # 關閉對齊要求
REG_MISMATCH_EDGE_PENALTY = float(os.getenv("REG_MISMATCH_EDGE_PENALTY", "0.005"))
REGIME_PRIMARY = int(os.getenv("REGIME_PRIMARY", "0"))  # 關閉場態優先

# 關閉EMA平滑
EMA_ENABLE    = int(os.getenv("EMA_ENABLE", "0"))  # 關閉EMA，提高響應速度
EMA_PROB_A    = float(os.getenv("EMA_PROB_A", "0.5"))
EMA_BET_A     = float(os.getenv("EMA_BET_A", "0.3"))

# 其他靈活性設定
VOL_GUARD      = int(os.getenv("VOL_GUARD", "0"))  # 關閉波動保護
SAME_SIDE_SOFT_CAP = int(os.getenv("SAME_SIDE_SOFT_CAP", "5"))  # 放寬同邊限制
SAME_SIDE_PENALTY  = float(os.getenv("SAME_SIDE_PENALTY", "0.01"))

SHOW_BIAS_ON_ABSTAIN = int(os.getenv("SHOW_BIAS_ON_ABSTAIN", "1"))
FORCE_DIRECTION_WHEN_UNDEREDGE = int(os.getenv("FORCE_DIRECTION_WHEN_UNDEREDGE", "1"))  # 強制給出方向

FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", "data/feedback.csv")

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

# ---------- 模型載入 ----------
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

# ---------- 基礎功能 ----------
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
        return np.array([0.45,0.45,0.1, 0.5,0.5, 0,0,0,0, 0.5,0.5,0.5,0.5, 0.1], dtype=np.float32)
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

# ---------- 模型預測（提高靈敏度） ----------
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
    if not seq: return np.array([0.45,0.45,0.1], dtype=np.float32)
    sub=seq[-FEAT_WIN:] if len(seq)>FEAT_WIN else seq
    cnt=np.bincount(sub, minlength=3).astype(np.float32)
    freq=cnt/max(1,len(sub))
    # 提高和局基礎概率，增加預測靈活性
    p0=0.8*freq + 0.2*np.array([0.42,0.42,0.16], dtype=np.float32)
    p0[2]=np.clip(p0[2], CLIP_T_MIN, CLIP_T_MAX)
    p0=np.clip(p0,1e-6,None); p0=p0/p0.sum()
    return p0

def enhanced_ensemble(seq: List[int]) -> Tuple[np.ndarray, Dict[str,str], Dict[str,int]]:
    """改進的集成預測 - 提高靈活性和平衡性"""
    weights=_parse_weights(ENSEMBLE_WEIGHTS)
    preds=[]; names=[]; vote_labels={}; vote_counts={'莊':0,'閒':0,'和':0}
    label_map=["莊","閒","和"]

    # XGBoost預測
    px = None if DEEP_ONLY==1 else xgb_probs(seq)
    if px is not None:
        p = softmax_log(px, TEMP_XGB)
        preds.append(p); names.append("XGB")
        vote_labels['XGB']=label_map[int(px.argmax())]
        vote_counts[vote_labels['XGB']]+=1

    # LightGBM預測
    pl = None if DEEP_ONLY==1 else lgb_probs(seq)
    if pl is not None:
        p = softmax_log(pl, TEMP_LGB)
        preds.append(p); names.append("LGBM")
        vote_labels['LGBM']=label_map[int(pl.argmax())]
        vote_counts[vote_labels['LGBM']]+=1

    # RNN預測
    pr = rnn_probs(seq)
    if pr is not None:
        p = softmax_log(pr, TEMP_RNN)
        preds.append(p); names.append("RNN")
        vote_labels['RNN']=label_map[int(pr.argmax())]
        vote_counts[vote_labels['RNN']]+=1

    if not preds:
        ph=heuristic_probs(seq)
        return ph, {}, {'莊':0,'閒':0,'和':0}

    # 簡化的權重平均（不受場態影響）
    W=np.array([weights.get(n,0.0) for n in names], dtype=np.float32)
    if W.sum()<=0: W=np.ones_like(W)/len(W)
    W=W/W.sum()

    P=np.stack(preds, axis=0).astype(np.float32)
    p_avg=(P*W[:,None]).sum(axis=0)
    
    # 確保和局有合理的預測範圍
    p_avg[2]=np.clip(p_avg[2], CLIP_T_MIN, CLIP_T_MAX)
    p_avg=np.clip(p_avg,1e-6,None); p_avg=p_avg/p_avg.sum()

    return p_avg, vote_labels, vote_counts

def flexible_decision(p: np.ndarray, seq: Optional[List[int]] = None) -> Tuple[str, float, float]:
    """靈活的決策邏輯 - 大幅簡化判斷條件"""
    
    # 找出最高概率的預測
    best_idx = int(np.argmax(p))
    best_label = "莊" if best_idx==0 else ("閒" if best_idx==1 else "和")
    
    # 計算邊際優勢
    sorted_probs = sorted([float(p[0]), float(p[1]), float(p[2])], reverse=True)
    edge = sorted_probs[0] - sorted_probs[1]
    
    # 動態調整信心度 - 根據歷史長度和波動性
    confidence_multiplier = 1.0
    if seq and len(seq) > 10:
        # 計算最近的變化率，提高對變化的敏感度
        recent = seq[-10:]
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        change_rate = changes / max(1, len(recent)-1)
        # 變化率高時提高信心，低時保守
        confidence_multiplier = 0.8 + 0.6 * change_rate
    
    adjusted_edge = edge * confidence_multiplier
    
    # 非常低的進場門檻 - 更積極的預測
    min_threshold = MIN_EDGE * 0.5  # 進一步降低門檻
    
    if adjusted_edge < min_threshold:
        # 即使邊際不足，也給出傾向性建議
        if FORCE_DIRECTION_WHEN_UNDEREDGE:
            return f"{best_label}（觀望）", edge, 0.0
        else:
            return "觀望", edge, 0.0
    
    # 動態下注比例 - 更積極
    if adjusted_edge >= 0.08:
        bet_pct = 0.25  # 高信心時25%
    elif adjusted_edge >= 0.05:
        bet_pct = 0.15  # 中等信心15%
    elif adjusted_edge >= 0.03:
        bet_pct = 0.08  # 低信心8%
    else:
        bet_pct = 0.05  # 最低5%
    
    return best_label, edge, bet_pct

def bet_amount(bankroll:int, pct:float) -> int:
    if not bankroll or bankroll<=0 or pct<=0: return 0
    return int(round(bankroll*pct))

def bet_ladder_text(bankroll: int) -> str:
    if not bankroll or bankroll <= 0:
        return "💴 配注：5%｜10%｜15%｜25%（先輸入本金以顯示金額）"
    a = int(round(bankroll * 0.05))
    b = int(round(bankroll * 0.10))
    c = int(round(bankroll * 0.15))
    d = int(round(bankroll * 0.25))
    return f"💴 配注 5% {a:,}｜10% {b:,}｜15% {c:,}｜25% {d:,}"

def simple_reply(n_hand:int, lab:str, edge:float, p:np.ndarray, bankroll:int, bet_pct:float) -> str:
    conf = int(round(100*max(p[0], p[1], p[2])))
    amt = bet_amount(bankroll, bet_pct)
    
    # 顯示三個概率，增加透明度
    b_pct = int(round(100*p[0]))
    p_pct = int(round(100*p[1]))
    t_pct = int(round(100*p[2]))
    
    prob_display = f"莊{b_pct}%｜閒{p_pct}%｜和{t_pct}%"
    
    if bet_pct > 0 and amt > 0:
        return f"🎯 下一局：{lab}（{conf}%）💰 {amt:,}\n📊 {prob_display}"
    else:
        return f"👁️ 下一局：{lab}（{conf}%）⚪\n📊 {prob_display}"

def trial_over_text() -> str:
    return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"

# ---------- LINE功能 ----------
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

def _init_user(uid:str):
    now = int(time.time())
    SESS[uid] = {
        "bankroll": 0, "seq": [], "trial_start": now, "premium": False,
        "perf": {"ok":0,"ng":0,"boost":0.0},
        "last_suggestion": None,
        "turn": 0, "last_turn": 0,
    }

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

def trial_guard(uid:str, reply_token:str) -> bool:
    sess = SESS.get(uid) or {}
    if sess.get("premium", False): return False
    start = int(sess.get("trial_start", int(time.time())))
    now   = int(time.time())
    elapsed_min = (now - start) // 60
    if elapsed_min >= TRIAL_MINUTES:
        safe_reply(reply_token, trial_over_text(), uid)
        return True
    return False

# ---------- HTTP API ----------
@app.get("/")
def root(): return "LiveBoot Enhanced ok", 200

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

    # 使用改進的預測系統
    p_avg, vote_labels, vote_counts = enhanced_ensemble(seq)
    lab, edge, bet_pct = flexible_decision(p_avg, seq)

    if API_MINIMAL_JSON:
        return jsonify(
            hands=len(seq),
            suggestion=lab,
            confidence=round(float(max(p_avg)),3),
            edge=round(float(edge),3),
            bet_pct=float(bet_pct),
            bet_amount=bet_amount(bankroll, bet_pct),
            probabilities={
                "banker": round(float(p_avg[0]), 3),
                "player": round(float(p_avg[1]), 3),
                "tie": round(float(p_avg[2]), 3)
            }
        ), 200
    
    text = simple_reply(len(seq), lab, edge, p_avg, bankroll, bet_pct)
    return jsonify(message=text, hands=len(seq), suggestion=lab,
                   bet_pct=float(bet_pct), bet_amount=bet_amount(bankroll, bet_pct),
                   probabilities={
                       "banker": float(p_avg[0]),
                       "player": float(p_avg[1]), 
                       "tie": float(p_avg[2])
                   }), 200

# ---------- LINE Webhook處理 ----------
if line_handler and line_api:
    @line_handler.add(FollowEvent)
    def on_follow(event):
        uid = event.source.user_id
        _init_user(uid)
        mins = TRIAL_MINUTES
        msg = (
            f"🤖 歡迎！已啟用 {mins} 分鐘試用\n"
            "請先輸入本金（例：5000）💵\n"
            "再貼歷史（B/P/T 或 莊/閒/和）→『開始分析』📊\n"
            "配注：5%｜10%｜15%｜25%（輸入本金後顯示金額）\n"
            f"到期請輸入：開通 你的密碼（向管理員索取）{ADMIN_CONTACT}"
        )
        line_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=quick_reply_buttons()))

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event):
        uid = event.source.user_id
        text = (event.message.text or "").strip()
        if uid not in SESS: _init_user(uid)
        sess = SESS[uid]

        # 試用守門
        if trial_guard(uid, event.reply_token): return

        # 系統指令
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
            sess["last_suggestion"] = None
            safe_reply(
                event.reply_token,
                "🧹 已清空。\n請輸入本金（例：5000）💵\n💴 配注：5%｜10%｜15%｜25%｜輸入本金後顯示金額\n貼歷史後輸入「開始分析」📊",
                uid
            )
            return

        if text.startswith("開通") or text.lower().startswith("activate"):
            code = text.split(" ",1)[1].strip() if " " in text else ""
            if validate_activation_code(code):
                sess["premium"] = True
                safe_reply(event.reply_token, "✅ 已開通成功！🎉", uid)
            else:
                safe_reply(event.reply_token, "❌ 密碼錯誤，請向管理員索取。", uid)
            return

        # 本金設定
        if text.isdigit():
            sess["bankroll"] = int(text)
            ladder = bet_ladder_text(sess["bankroll"])
            safe_reply(event.reply_token, f"👍 已設定本金：{int(text):,}\n{ladder}", uid)
            return

        # 結果回報（在線自適應）
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
                safe_reply(event.reply_token, "🔥 已記錄結果", uid)
            else:
                safe_reply(event.reply_token, "ℹ️ 用法：結果 莊/閒/和", uid)
            return

        # 歷史/單手輸入
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

        # 分析預測
        if ("開始分析" in text) or (text in ["分析", "開始", "GO", "go"]):
            sseq: List[int] = sess.get("seq", [])
            bankroll: int = int(sess.get("bankroll", 0) or 0)
            
            # 使用改進的預測系統
            p_avg, vote_labels, vote_counts = enhanced_ensemble(sseq)
            lab, edge, bet_pct = flexible_decision(p_avg, sseq)
            
            sess["last_suggestion"] = lab if lab in ("莊","閒","和") else None
            sess["last_edge"] = float(edge); sess["last_bet_pct"] = float(bet_pct)
            sess["last_probs"] = [float(p_avg[0]), float(p_avg[1]), float(p_avg[2])]
            
            reply = simple_reply(len(sseq), lab, edge, p_avg, bankroll, bet_pct)
            
            # 若未設定本金，提醒配注說明
            if not bankroll or bankroll <= 0:
                reply += "\n💴 請先輸入本金以顯示 5%｜10%｜15%｜25% 配注金額"
            safe_reply(event.reply_token, reply, uid); return

        # 說明
        safe_reply(event.reply_token, "🧭 指令：設定本金→貼歷史→『開始分析』｜『返回』撤回｜『結束分析』清空｜『結果 ⋯』回報上一手", uid)

# ---------- LINE webhook endpoint ----------
@app.post("/line-webhook")
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

# ---------- 主程式 ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
