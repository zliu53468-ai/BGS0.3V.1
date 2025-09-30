# server.py - 改進版百家樂預測系統（修正版平衡決策邏輯）
# server.py — 純算牌 + 粒子濾波（ONLY 莊/閒建議｜EV含抽水｜¼-Kelly｜試用制｜卡片輸出｜支援和局回報）
# Author: 親愛的 x GPT-5 Thinking

import os, logging, time
from typing import List, Tuple, Optional, Dict
import os, logging, time, csv, pathlib, re
from typing import List, Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

log = logging.getLogger("liveboot-server")
# ==== 基本設定 ====
VERSION = "bgs-deplete-pf-2025-09-15"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

app = Flask(__name__)
CORS(app) # 允許跨來源請求
CORS(app)

# ---------- 配置參數 ----------
def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    if v == "1/0": return 1
    try: return 1 if int(float(v)) != 0 else 0
    except: return 1 if default else 0

# 基礎參數
FEAT_WIN   = int(os.getenv("FEAT_WIN", "40"))
GRID_ROWS  = int(os.getenv("GRID_ROWS", "6"))
GRID_COLS  = int(os.getenv("GRID_COLS", "20"))
CLIP_T_MIN = float(os.getenv("CLIP_T_MIN", "0.06"))
CLIP_T_MAX = float(os.getenv("CLIP_T_MAX", "0.20"))
SEED       = int(os.getenv("SEED", "42"))
np.random.seed(SEED)

# 權重平衡
USE_FULL_SHOE = env_flag("USE_FULL_SHOE", 1)
LOCAL_WEIGHT  = float(os.getenv("LOCAL_WEIGHT", "0.5"))
GLOBAL_WEIGHT = float(os.getenv("GLOBAL_WEIGHT", "0.5"))
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

# 場態控制 (市場結構分析)
REGIME_CTRL   = int(os.getenv("REGIME_CTRL", "1"))
REG_WIN       = int(os.getenv("REG_WIN", "25"))
REG_STREAK_TH = float(os.getenv("REG_STREAK_TH", "0.65")) # 長龍
REG_CHOP_TH   = float(os.getenv("REG_CHOP_TH", "0.60"))   # 單跳
REG_DCHOP_TH  = float(os.getenv("REG_DCHOP_TH", "0.66")) # 雙跳
REG_DCHOP_WIN = int(os.getenv("REG_DCHOP_WIN", "6"))
REG_PATTERN_WIN = int(os.getenv("REG_PATTERN_WIN", "4")) # 一房兩廳
REG_EVENFEET_WIN = int(os.getenv("REG_EVENFEET_WIN", "3")) # 齊腳

# 歷史圖形比對控制
HISTORICAL_MATCH_CTRL  = int(os.getenv("HISTORICAL_MATCH_CTRL", "1"))
HISTORICAL_MATCH_BONUS = float(os.getenv("HISTORICAL_MATCH_BONUS", "0.05"))

# 場態調整參數
REG_ALIGN_EDGE_BONUS      = float(os.getenv("REG_ALIGN_EDGE_BONUS", "0.02"))
REG_MISMATCH_EDGE_PENALTY = float(os.getenv("REG_MISMATCH_EDGE_PENALTY", "0.04"))

# 動能衰竭分析
SAME_SIDE_SOFT_CAP = int(os.getenv("SAME_SIDE_SOFT_CAP", "4"))
SAME_SIDE_PENALTY  = float(os.getenv("SAME_SIDE_PENALTY", "0.015"))

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
# ==== 牌路/顯示 ====
MAP = {"B":0, "P":1, "T":2, "莊":0, "閒":1, "和":2}
INV = {0:"B", 1:"P", 2:"T"}
INV = {0:"莊", 1:"閒", 2:"和"}

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
    if DISABLE_RNN == 1: log.info("[MODEL] RNN disabled by env"); return
    try:
        import torch, torch.nn as nn
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS","1")))
        class TinyRNN(nn.Module):
            def __init__(self, in_dim=3, hid=int(os.getenv("RNN_HIDDEN","32")), out_dim=3):
                super().__init__(); self.gru = nn.GRU(in_dim, hid, 1, batch_first=True); self.fc = nn.Linear(hid, out_dim)
            def forward(self, x): o,_ = self.gru(x); return self.fc(o[:,-1,:])
        path = os.getenv("RNN_OUT_PATH", "data/models/rnn.pt")
        if os.path.exists(path):
            RNN_MODEL = TinyRNN(); state = __import__("torch").load(path, map_location="cpu")
            RNN_MODEL.load_state_dict(state); RNN_MODEL.eval(); log.info("[MODEL] RNN loaded: %s", path)
        else: log.warning("[MODEL] RNN file not found at %s", path)
    except Exception as e: log.warning("[MODEL] RNN load failed: %s", e)
_load_xgb(); _load_lgb(); _load_rnn()

# ---------- 基礎功能 & 特徵工程 ----------
def parse_history(s: str) -> List[int]:
    s = (s or "").strip().upper()
    if not s: return []
    s = s.replace("，"," ").replace("、"," ").replace("\u3000"," ")
    toks = s.split()
    seq = list(s) if len(toks) == 1 else toks
    seq = list(s) if (len(toks) == 1 and len(s) <= 12) else toks
    out = []
    for ch in seq:
        ch = ch.strip().upper()
        if ch in MAP:
            out.append(MAP[ch])
        if ch in MAP: out.append(MAP[ch])
    return out

def big_road_grid(seq: List[int], rows:int=6, cols:int=20):
    gs = np.zeros((rows, cols), dtype=np.int8); gt = np.zeros((rows, cols), dtype=np.int16)
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
            r,c=nr,nc
            if 0<=r<rows and 0<=c<cols: gs[r,c]=cur
        else:
            c=c+1; r=0; last_bp=cur
            if c<cols:
                gs[r,c]=cur
    return gs, gt, (r,c)

def _parse_weights(spec: str) -> Dict[str, float]:
    out={"XGB":0.33,"LGBM":0.33,"RNN":0.34}
    try:
        tmp={}
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

def big_road_features(seq: List[int], rows:int=6, cols:int=20, win:int=40) -> np.ndarray:
    local=_local_bigroad_feat(seq, rows, cols, win).astype(np.float32)
    if USE_FULL_SHOE:
        glob=_global_aggregates(seq).astype(np.float32)
        lw=max(0.0, LOCAL_WEIGHT); gw=max(0.0, GLOBAL_WEIGHT); s=lw+gw
        if s==0: lw,gw=1.0,0.0
        else: lw,gw=lw/s,gw/s
        return np.concatenate([local*lw, glob*gw], axis=0).astype(np.float32)
    else: return local

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
    return np.array([freq[0],freq[1],freq[2], altern,1.0-altern,b_mean,b_var,p_mean,p_var, B2B,P2P,B2P,P2B, tie_rate], dtype=np.float32)

def _local_bigroad_feat(seq: List[int], rows:int, cols:int, win:int) -> np.ndarray:
    sub = seq[-win:] if len(seq)>win else seq[:]
    gs, gt, (r,c) = big_road_grid(sub, rows, cols)
    grid_sign_flat = gs.flatten().astype(np.float32); grid_tie_flat  = np.clip(gt.flatten(),0,3).astype(np.float32)/3.0
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
    cur_col_height=float((gs[:,c]!=0).sum())/rows if 0<=c<cols else 0.0; cur_col_side=float(gs[0,c]) if 0<=c<cols else 0.0
    cnt=np.bincount(sub, minlength=3).astype(np.float32); freq=cnt/max(1,len(sub))
    return np.concatenate([grid_sign_flat, grid_tie_flat, np.array([streak_len/rows, streak_side], dtype=np.float32), col_heights, np.array([cur_col_height, cur_col_side], dtype=np.float32), freq], axis=0)

def softmax_log(p: np.ndarray, temp: float=1.0) -> np.ndarray:
    x = np.log(np.clip(p,1e-9,None)) / max(1e-9, temp); x = x - x.max(); e = np.exp(x); return e / e.sum()

# ---------- 場態分析 (市場結構分析) ----------
def detect_regime(seq: List[int]) -> Tuple[str, List[int], List[int]]:
    bp = [x for x in seq if x != 2]; runs, run_sides = [], []
    if bp:
        current_run, current_side = 1, bp[0]
        for i in range(1, len(bp)):
            if bp[i] == bp[i-1]: current_run += 1
            else: runs.append(current_run); run_sides.append(current_side); current_run, current_side = 1, bp[i]
        runs.append(current_run); run_sides.append(current_side)
    if len(bp) < 10: return "NORMAL", runs, run_sides
    sub = bp[-REG_WIN:] if len(bp) >= REG_WIN else bp
    if len(sub) > 1:
        if sum(1 for i in range(1, len(sub)) if sub[i] != sub[i-1]) / (len(sub) - 1) >= REG_CHOP_TH: return "CHOP", runs, run_sides
        if sum(1 for i in range(1, len(sub)) if sub[i] == sub[i-1]) / (len(sub) - 1) >= REG_STREAK_TH: return "STREAK", runs, run_sides
    if len(runs) >= REG_DCHOP_WIN and runs[-REG_DCHOP_WIN:].count(2) / REG_DCHOP_WIN >= REG_DCHOP_TH: return "DOUBLE_CHOP", runs, run_sides
    if len(runs) >= REG_PATTERN_WIN:
        if (runs[-4:] == [2, 1, 2, 1] and run_sides[-4:] == [0, 1, 0, 1]): return "PATTERN_BBP", runs, run_sides
        if (runs[-4:] == [2, 1, 2, 1] and run_sides[-4:] == [1, 0, 1, 0]): return "PATTERN_PPB", runs, run_sides
    if len(runs) >= REG_EVENFEET_WIN and len(set(runs[-REG_EVENFEET_WIN:])) == 1 and runs[-1] > 1: return "EVEN_FEET", runs, run_sides
    b_count, p_count = sub.count(0), sub.count(1)
    if b_count / len(sub) >= 0.70 or p_count / len(sub) >= 0.70: return "SIDE_BIAS", runs, run_sides
    return "NORMAL", runs, run_sides
def find_historical_pattern_match(seq: List[int]) -> Optional[int]:
    if len(seq) < 10: return None
    gs, _, (r, c) = big_road_grid(seq, GRID_ROWS, GRID_COLS)
    if c == 0 and r == 0: return None
    current_col_raw = gs[:, c]; current_col_pattern = current_col_raw[current_col_raw != 0]
    if current_col_pattern.size == 0: return None
    for prev_c in range(c - 1, -1, -1):
        prev_col_raw = gs[:, prev_c]; prev_col_pattern = prev_col_raw[prev_col_raw != 0]
        if np.array_equal(current_col_pattern, prev_col_pattern):
            next_outcome_side = gs[0, prev_c + 1]
            if next_outcome_side == 1: return 0
            elif next_outcome_side == -1: return 1
# 解析「上局結果：閒6 莊8 / 和9 / TIE / DRAW」等；回 (P_total, B_total) 或 None
def parse_last_hand_points(text: str):
    if not text: return None
    s = text.strip().upper().replace("：", ":")
    s = re.sub(r"\s+", "", s)

    # 明確和局：和9 / TIE9 / DRAW9 → 視為 P_total=B_total=9
    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:和|TIE|DRAW)\s*:?(\d)', s)
    if m:
        d = int(m.group(1)); return (d, d)

    # 單獨和（無點數）：不扣牌，但後續照常預測
    if re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:和|TIE|DRAW)\b', s):
        return None

    # 一般格式
    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:閒|P)\s*:?(\d)\s*(?:[,/]|)?\s*(?:莊|B)\s*:?(\d)', s)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:莊|B)\s*:?(\d)\s*(?:[,/]|)?\s*(?:閒|P)\s*:?(\d)', s)
    if m: return (int(m.group(2)), int(m.group(1)))
    m = re.search(r'(?:PLAYER|P)\s*:?(\d)\s*(?:[,/]|)?\s*(?:BANKER|B)\s*:?(\d)', s)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'(?:BANKER|B)\s*:?(\d)\s*(?:[,/]|)?\s*(?:PLAYER|P)\s*:?(\d)', s)
    if m: return (int(m.group(2)), int(m.group(1)))
    return None

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
    # Simplified one_hot_seq to avoid potential issues with old unused code paths
    sub = seq[-MAX_RNN_LEN:] if len(seq) > MAX_RNN_LEN else seq[:]
    L=len(sub); oh=np.zeros((1,L,3), dtype=np.float32)
    for i,v in enumerate(sub):
        if v in (0,1,2): oh[0,i,v]=1.0
    x = torch.from_numpy(oh)
    with torch.no_grad():
        logits=RNN_MODEL(x)
        p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return p.astype(np.float32)

# ---------- 模型預測 ----------
def enhanced_ensemble(seq: List[int]) -> Tuple[np.ndarray, str, List[int], List[int], Optional[int]]:
    weights=_parse_weights(ENSEMBLE_WEIGHTS); preds=[]; names=[];
    px = None if DEEP_ONLY==1 else xgb_probs(seq)
    if px is not None: preds.append(softmax_log(px, TEMP_XGB)); names.append("XGB")
    pl = None if DEEP_ONLY==1 else lgb_probs(seq)
    if pl is not None: preds.append(softmax_log(pl, TEMP_LGB)); names.append("LGBM")
    pr = rnn_probs(seq)
    if pr is not None: preds.append(softmax_log(pr, TEMP_RNN)); names.append("RNN")
    regime, runs, run_sides = detect_regime(seq) if REGIME_CTRL else ("DISABLED", [], [])
    hist_sug = find_historical_pattern_match(seq) if HISTORICAL_MATCH_CTRL else None
    if not preds: return np.array([0.45,0.45,0.1]), regime, runs, run_sides, hist_sug
    W=np.array([weights.get(n,0.0) for n in names]); W=W/W.sum() if W.sum()>0 else np.ones_like(W)/len(W)
    P=np.stack(preds, axis=0); p_avg=(P*W[:,None]).sum(axis=0)
    p_avg[2]=np.clip(p_avg[2], CLIP_T_MIN, CLIP_T_MAX); p_avg=np.clip(p_avg,1e-6,None); p_avg=p_avg/p_avg.sum()
    return p_avg, regime, runs, run_sides, hist_sug

# ---------- [核心修正] 嚴謹的平衡決策邏輯 ----------
def flexible_decision(p: np.ndarray, seq: List[int], regime: str, runs: List[int], run_sides: List[int], historical_suggestion: Optional[int]) -> Tuple[str, float, float, str]:
    best_idx = int(np.argmax(p)); sorted_probs = np.sort(p)[::-1]
    edge = float(sorted_probs[0] - sorted_probs[1]); adjusted_edge = edge
    decision_reason = "模型綜合分析"

    if historical_suggestion is not None and HISTORICAL_MATCH_CTRL:
        best_idx = historical_suggestion
        decision_reason = "歷史圖形匹配"
        adjusted_edge = p[best_idx] - (sorted_probs[0] if p.argmax() != best_idx else sorted_probs[1]) + HISTORICAL_MATCH_BONUS
    elif REGIME_CTRL and seq and runs:
        last_event = run_sides[-1]; streak_len = runs[-1]
        predicted_side = best_idx
        
        is_aligned = (predicted_side == last_event if regime == "STREAK" else predicted_side != last_event)
        
        if regime in ["STREAK", "CHOP"]:
            if is_aligned: adjusted_edge += REG_ALIGN_EDGE_BONUS; decision_reason = f"順應{'長龍' if regime=='STREAK' else '單跳'}"
            else: adjusted_edge -= REG_MISMATCH_EDGE_PENALTY; decision_reason = f"{'長龍' if regime=='STREAK' else '單跳'}結構警示"
        
        elif regime == "DOUBLE_CHOP":
            if (streak_len == 1 and predicted_side == last_event) or (streak_len == 2 and predicted_side != last_event):
                adjusted_edge += REG_ALIGN_EDGE_BONUS; decision_reason = "順應雙跳"
            else: adjusted_edge -= REG_MISMATCH_EDGE_PENALTY; decision_reason = "雙跳結構警示"

        elif regime == "EVEN_FEET" and len(runs) >= 2:
            last_run_height = runs[-2]
            if (streak_len < last_run_height and predicted_side == last_event) or (streak_len == last_run_height and predicted_side != last_event):
                adjusted_edge += REG_ALIGN_EDGE_BONUS; decision_reason = f"順應齊腳(高{last_run_height})"
            else: adjusted_edge -= REG_MISMATCH_EDGE_PENALTY; decision_reason = "齊腳結構警示"
        
        elif regime == "PATTERN_BBP":
            if streak_len == 2 and last_event == 0:
                if predicted_side == 1: adjusted_edge += REG_ALIGN_EDGE_BONUS; decision_reason = "順應一房兩廳(莊莊->閒)"
                else: adjusted_edge -= REG_MISMATCH_EDGE_PENALTY
            elif streak_len == 1 and last_event == 1:
                if predicted_side == 0: adjusted_edge += REG_ALIGN_EDGE_BONUS; decision_reason = "順應一房兩廳(閒->莊)"
                else: adjusted_edge -= REG_MISMATCH_EDGE_PENALTY
        
        elif regime == "PATTERN_PPB":
            if streak_len == 2 and last_event == 1:
                if predicted_side == 0: adjusted_edge += REG_ALIGN_EDGE_BONUS; decision_reason = "順應一房兩廳(閒閒->莊)"
                else: adjusted_edge -= REG_MISMATCH_EDGE_PENALTY
            elif streak_len == 1 and last_event == 0:
                if predicted_side == 1: adjusted_edge += REG_ALIGN_EDGE_BONUS; decision_reason = "順應一房兩廳(莊->閒)"
                else: adjusted_edge -= REG_MISMATCH_EDGE_PENALTY

    if runs and run_sides:
        if runs[-1] >= SAME_SIDE_SOFT_CAP and best_idx == run_sides[-1]:
            adjusted_edge -= (runs[-1] - SAME_SIDE_SOFT_CAP + 1) * SAME_SIDE_PENALTY
            decision_reason = "動能衰竭警示"

    final_label = "莊" if best_idx==0 else ("閒" if best_idx==1 else "和")
    if adjusted_edge < EDGE_ENTER:
        return f"{final_label} (觀)" if FORCE_DIRECTION_WHEN_UNDEREDGE else "觀望", edge, 0.0, "優勢不足"
    
    if adjusted_edge >= 0.10: bet_pct = 0.25
    elif adjusted_edge >= 0.07: bet_pct = 0.15
    elif adjusted_edge >= 0.04: bet_pct = 0.10
    else: bet_pct = 0.05
    return final_label, edge, bet_pct, decision_reason

# ---------- LINE回覆格式 ----------
def bet_amount(bankroll:int, pct:float) -> int:
    if not bankroll or bankroll<=0 or pct<=0: return 0
    return int(round(bankroll*pct))
def bet_ladder_text(bankroll: int) -> str:
    if not bankroll or bankroll <= 0: return "💴 配注：5%｜10%｜15%｜25%（先輸入本金）"
    a=int(round(bankroll*0.05)); b=int(round(bankroll*0.10)); c=int(round(bankroll*0.15)); d=int(round(bankroll*0.25))
    return f"💴 配注 5% {a:,}｜10% {b:,}｜15% {c:,}｜25% {d:,}"
def simple_reply(n_hand:int, lab:str, edge:float, p:np.ndarray, bankroll:int, bet_pct:float, regime:str, reason:str) -> str:
    conf = int(round(100*p[np.argmax(p)])); amt = bet_amount(bankroll, bet_pct)
    b_pct, p_pct, t_pct = int(round(100*p[0])), int(round(100*p[1])), int(round(100*p[2]))
    prob_display = f"莊{b_pct}%｜閒{p_pct}%｜和{t_pct}%"
    regime_map = { "NORMAL": "標準", "STREAK": "長龍趨勢", "CHOP": "單跳盤整", "DOUBLE_CHOP": "雙跳結構", "PATTERN_BBP": "一房兩廳(莊)", "PATTERN_PPB": "一房兩廳(閒)", "EVEN_FEET": "齊腳結構", "SIDE_BIAS": "單邊優勢", "DISABLED": "關閉" }
    regime_text = regime_map.get(regime, regime); reason_icon = "🧠"
    if reason == "歷史圖形匹配": reason_icon = "⚠️"
    elif "警示" in reason: reason_icon = "🔔"
    reason_display = f"{reason_icon} 決策依據：{reason} (現況：{regime_text})"
    suggestion_line = f"🎯 下一局 ({n_hand+1})：{lab} ({conf}%)\n💰 建議注額：{amt:,}" if bet_pct > 0 and amt > 0 else f"👁️ 下一局 ({n_hand+1})：{lab} ({conf}%)\n💰 建議注額：⚪"
    return f"{suggestion_line}\n📊 {prob_display}\n{reason_display}"
def trial_over_text() -> str:
    return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"

# ---------- LINE功能 & HTTP API (保留原始架構) ----------
def quick_reply_buttons():
    try:
        return QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="莊", text="莊")), QuickReplyButton(action=MessageAction(label="閒", text="閒")),
            QuickReplyButton(action=MessageAction(label="和", text="和")), QuickReplyButton(action=MessageAction(label="開始分析", text="開始分析")),
            QuickReplyButton(action=MessageAction(label="返回 ⬅️", text="返回")), QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
        ])
    except Exception: return None
# ==== 試用 / 營運 ====
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")

SESS: Dict[str, Dict[str, object]] = {}
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
    SESS[uid] = {"bankroll": 0, "seq": [], "trial_start": now, "premium": False}

def validate_activation_code(code: str) -> bool:
    return bool(ADMIN_ACTIVATION_SECRET) and bool(code) and (code == ADMIN_ACTIVATION_SECRET)
def trial_guard(uid:str, reply_token:str) -> bool:

def trial_guard(uid:str) -> Optional[str]:
    sess = SESS.get(uid) or {}
    if sess.get("premium", False): return False
    if sess.get("premium", False): return None
    now = int(time.time()); start = int(sess.get("trial_start", now))
    if (now - start) // 60 >= TRIAL_MINUTES: safe_reply(reply_token, trial_over_text(), uid); return True
    return False
    if (now - start) // 60 >= TRIAL_MINUTES:
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"
    return None

# ==== 算牌引擎 ====
from bgs.deplete import DepleteMC
from bgs.pfilter import OutcomePF

SEED = int(os.getenv("SEED","42"))
DEPL_DECKS  = int(os.getenv("DEPL_DECKS", "8"))
DEPL_SIMS   = int(os.getenv("DEPL_SIMS", "30000"))

# 粒子濾波（只用輸贏也能學）
PF_N        = int(os.getenv("PF_N", "200"))
PF_UPD_SIMS = int(os.getenv("PF_UPD_SIMS", "80"))
PF_PRED_SIMS= int(os.getenv("PF_PRED_SIMS", "220"))
PF_RESAMPLE = float(os.getenv("PF_RESAMPLE", "0.5"))
PF_DIR_ALPHA= float(os.getenv("PF_DIR_ALPHA", "0.8"))  # Dirichlet 先驗強度
PF_USE_EXACT= int(os.getenv("PF_USE_EXACT", "0"))      # 0=MC 前向；1=Exact-lite 前向

DEPL = DepleteMC(decks=DEPL_DECKS, seed=SEED)
PF   = OutcomePF(
        decks=DEPL_DECKS, seed=SEED, n_particles=PF_N,
        sims_lik=PF_UPD_SIMS, resample_thr=PF_RES_SAMPLE if (PF_RESAMPLE:=PF_RESAMPLE) else 0.5
      ) if False else OutcomePF(decks=DEPL_DECKS, seed=SEED, n_particles=PF_N,
                                sims_lik=PF_UPD_SIMS, resample_thr=PF_RESAMPLE,
                                dirichlet_alpha=PF_DIR_ALPHA, use_exact=bool(PF_USE_EXACT))

# ==== 決策（僅莊/閒）====
EDGE_ENTER  = float(os.getenv("EDGE_ENTER", "0.03"))  # 觀望門檻
USE_KELLY   = env_flag("USE_KELLY", 1)
KELLY_FACTOR= float(os.getenv("KELLY_FACTOR", "0.25"))  # ¼-Kelly
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.015"))  # 單注上限 1.5%

LOG_DIR     = os.getenv("LOG_DIR", "logs")
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
PRED_CSV    = os.path.join(LOG_DIR, "predictions.csv")
if not os.path.exists(PRED_CSV):
    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["ts","version","hands","pB","pP","pT","choice","edge","bet_pct","bankroll","bet_amt","engine","reason"])

def banker_ev(pB, pP):  # tie 退回
    return 0.95*pB - pP
def player_ev(pB, pP):
    return pP - pB
def kelly_fraction(p_win: float, payoff: float):
    q = 1.0 - p_win
    edge = p_win*payoff - q
    return max(0.0, edge / payoff)
def bet_amount(bankroll:int, pct:float) -> int:
    if not bankroll or bankroll<=0 or pct<=0: return 0
    return int(round(bankroll*pct))

def decide_only_bp(prob):
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    evB, evP = banker_ev(pB, pP), player_ev(pB, pP)
    side = 0 if evB > evP else 1
    edge_prob = abs(pB - pP)
    final_edge = max(edge_prob, abs(evB - evP))
    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, f"⚪ 優勢不足（門檻 {EDGE_ENTER:.2f}）")
    if USE_KELLY:
        f = KELLY_FACTOR * (kelly_fraction(pB, 0.95) if side==0 else kelly_fraction(pP, 1.0))
        bet_pct = min(MAX_BET_PCT, float(max(0.0, f)))
        reason = "🧠 純算牌｜📐 ¼-Kelly"
    else:
        if final_edge >= 0.10: bet_pct = 0.25
        elif final_edge >= 0.07: bet_pct = 0.15
        elif final_edge >= 0.04: bet_pct = 0.10
        else: bet_pct = 0.05
        reason = "🧠 純算牌｜🪜 階梯式配注"
    return (INV[side], final_edge, bet_pct, reason)

def log_prediction(hands:int, p, choice:str, edge:float, bankroll:int, bet_pct:float, engine:str, reason:str):
    try:
        bet_amt = bet_amount(bankroll, bet_pct)
        with open(PRED_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([int(time.time()), VERSION, hands, float(p[0]), float(p[1]), float(p[2]), choice, float(edge), float(bet_pct), int(bankroll), int(bet_amt), engine, reason])
    except Exception as e:
        log.warning("log_prediction failed: %s", e)

# ===== 卡片輸出 =====
def format_card_output(prob, choice, last_pts_text: Optional[str]):
    b_pct = f"{prob[0]*100:.2f}%"
    p_pct = f"{prob[1]*100:.2f}%"
    header = []
    if last_pts_text:
        header = ["讀取完成", last_pts_text, "開始分析下局....", ""]
    block = [
        "【預測結果】",
        f"閒：{p_pct}",
        f"莊：{b_pct}",
        f"本次預測結果：{choice if choice!='觀望' else '觀'}"
    ]
    return "\n".join(header + block)

# ==== 健康檢查 ====
@app.get("/")
def root(): return "LiveBoot Enhanced (Balanced Logic) ok", 200
def root(): return f"✅ BGS Deplete+PF Server OK ({VERSION})", 200

@app.get("/healthz")
def healthz(): return jsonify(ok=True, ts=time.time(), version=VERSION), 200

@app.get("/health")
def health():
    return jsonify(status="ok"), 200
# ==== API：/update-hand（有點數時可用；只有輸贏不必呼叫）====
@app.post("/update-hand")
def update_hand_api():
    obs = request.get_json(silent=True) or {}
    try:
        # Deplete 扣牌（有點數最準）
        if "p_total" in obs and "b_total" in obs:
            DEPL.update_hand(obs)
            last_outcome = 1 if int(obs["p_total"]) > int(obs["b_total"]) else (0 if int(obs["b_total"]) > int(obs["p_total"]) else 2)
            PF.update_outcome(last_outcome)
        return jsonify(ok=True), 200
    except Exception as e:
        log.warning("update_hand failed: %s", e)
        return jsonify(ok=False, msg=str(e)), 400

# ==== API：/predict（只回傳莊/閒建議；可回卡片）====
@app.post("/predict")
def predict_api():
    data = request.get_json(silent=True) or {}; bankroll = int(data.get("bankroll") or 0)
    seq = parse_history(data.get("history", ""))
    p_avg, regime, runs, run_sides, hist_sug = enhanced_ensemble(seq)
    lab, edge, bet_pct, reason = flexible_decision(p_avg, seq, regime, runs, run_sides, hist_sug)
    display_regime = "HISTORICAL_MATCH" if hist_sug is not None else regime
    text = simple_reply(len(seq), lab, edge, p_avg, bankroll, bet_pct, display_regime, reason)
    return jsonify(message=text, hands=len(seq), suggestion=lab, regime=display_regime, reason=reason, bet_pct=float(bet_pct), bet_amount=bet_amount(bankroll, bet_pct), probabilities={"banker": float(p_avg[0]), "player": float(p_avg[1]), "tie": float(p_avg[2])}), 200

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
            if sess.get("seq", []): sess["seq"].pop(-1); safe_reply(event.reply_token, f"↩️ 已撤回上一手，共 {len(sess['seq'])} 手。", uid)
            else: safe_reply(event.reply_token, "ℹ️ 沒有可撤回的紀錄。", uid)
            return
        if text in ["結束分析", "清空", "reset"]:
            sess["seq"], sess["bankroll"], sess["last_suggestion"] = [], 0, None
            safe_reply(event.reply_token, "🧹 已清空。\n請輸入本金（例：5000）💵\n貼歷史後輸入「開始分析」📊", uid)
            return
        if text.startswith("開通") or text.lower().startswith("activate"):
            code = text.split(" ",1)[1].strip() if " " in text else ""
            if validate_activation_code(code): sess["premium"] = True; safe_reply(event.reply_token, "✅ 已開通成功！🎉", uid)
            else: safe_reply(event.reply_token, "❌ 密碼錯誤，請向管理員索取。", uid)
            return
        if text.isdigit():
            sess["bankroll"] = int(text)
            safe_reply(event.reply_token, f"👍 已設定本金：{int(text):,}\n{bet_ladder_text(sess['bankroll'])}", uid)
            return

        # [CRASH FIX] Correctly parse history from user input
        seq_in = parse_history(text)
        if seq_in and ("開始分析" not in text):
            if len(seq_in) == 1:
                sess.setdefault("seq", []).append(seq_in[0])
                safe_reply(event.reply_token, f"✅ 已記錄 {text}（共 {len(sess['seq'])} 手）", uid)
            else:
                sess["seq"] = seq_in
                safe_reply(event.reply_token, f"✅ 已覆蓋歷史：{len(seq_in)} 手", uid)
            return

        if ("開始分析" in text) or (text in ["分析", "開始", "GO", "go"]):
            sseq = sess.get("seq", []); bankroll = int(sess.get("bankroll", 0) or 0)
            p_avg, regime, runs, run_sides, hist_sug = enhanced_ensemble(sseq)
            lab, edge, bet_pct, reason = flexible_decision(p_avg, sseq, regime, runs, run_sides, hist_sug)
            sess["last_suggestion"] = lab if lab in ("莊","閒","和") else None
            display_regime = "HISTORICAL_MATCH" if hist_sug is not None else regime
            reply = simple_reply(len(sseq), lab, edge, p_avg, bankroll, bet_pct, display_regime, reason)
            if not bankroll or bankroll <= 0: reply += f"\n{bet_ladder_text(0)}"
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
    data = request.get_json(silent=True) or {}
    bankroll = int(float(data.get("bankroll") or 0))
    seq = parse_history(str(data.get("history","")))
    lp = data.get("last_pts")
    pts = None
    engine_note = None

    # 先處理 last_pts（可能是點數；也可能是「和」）
    last_text = None
    if lp:
        pts = parse_last_hand_points(lp)
        if pts is not None:
            DEPL.update_hand({"p_total": int(pts[1 if False else 0]), "b_total": int(pts[1]) if False else int(pts[1]), "trials": 400})  # 只是保留防呆；下行為正確
            try:
                DEPL.update_hand({"p_total": int(pts[0]), "b_total": int(pts[1]), "trials": 400})
                engine_note = "Deplete"
                last_text = f"上局結果: 閒 {int(pts[0])} 莊 {int(pts[1])}"
                PF.update_outcome(1 if int(pts[0])>int(pts[1]) else (0 if int(pts[1])>int(pts[0]) else 2))
            except Exception as e:
                log.warning("deplete update in /predict failed: %s", e)
        else:
            if re.search(r'(?:和|TIE|DRAW)\b', str(lp).upper()):
                PF.update_outcome(2)
                last_text = "上局結果: 和局"

    # 也可直接傳 last_outcome: "B"/"P"/"T"
    if "last_outcome" in data:
        o = str(data["last_outcome"]).strip().upper()
        if o in ("B","莊","0"): PF.update_outcome(0); last_text = "上局結果: 莊勝"
        elif o in ("P","閒","1"): PF.update_outcome(1); last_text = "上局結果: 閒勝"
        elif o in ("T","和","2"): PF.update_outcome(2); last_text = "上局結果: 和局"

    # 取得概率
    p_depl = None; p_pf = None
    try: p_depl = DEPL.predict(sims=DEPL_SIMS)
    except Exception as e: log.warning("deplete predict failed: %s", e)
    try: p_pf   = PF.predict(sims_per_particle=PF_PRED_SIMS)
    except Exception as e: log.warning("pf predict failed: %s", e)

    if (p_depl is not None) and (p_pf is not None):
        if pts is not None: w_depl, w_pf = 0.7, 0.3; engine_note = "Mix(Deplete↑)"
        else:               w_depl, w_pf = 0.3, 0.7; engine_note = "Mix(PF↑)"
        p = w_depl * p_depl + w_pf * p_pf
        p[2] = np.clip(p[2], 0.06, 0.20); p = p / p.sum()
    elif p_depl is not None:
        p = p_depl; engine_note = "Deplete"
    elif p_pf is not None:
        p = p_pf; engine_note = "PF"
    else:
        p = np.array([0.45,0.45,0.10], dtype=np.float32); engine_note = "Fallback"

    choice, edge, bet_pct, reason = decide_only_bp(p)
    amt = bet_amount(bankroll, bet_pct)

    style = str(data.get("style","")).lower()
    if style == "card":
        msg = format_card_output(p, choice, last_text)
    else:
        b_pct, p_pct = int(round(100*p[0])), int(round(100*p[1]))
        evB = banker_ev(float(p[0]), float(p[1])); evP = player_ev(float(p[0]), float(p[1]))
        msg = (
            f"🎯 下一局建議：{choice}\n"
            f"💰 建議注額：{amt:,}\n"
            f"📊 機率｜莊 {b_pct}%｜閒 {p_pct}%\n"
            f"📐 EV（抽水後）｜莊 {evB:.3f}｜閒 {evP:.3f}\n"
            f"🧭 {reason}｜引擎：{engine_note}"
        )

    log_prediction(len(seq), p, choice, edge, bankroll, bet_pct, engine_note or "NA", reason)
    return jsonify(
        message=msg, version=VERSION, hands=len(seq),
        suggestion=choice, bet_pct=float(bet_pct), bet_amount=amt,
        probabilities={"banker": float(p[0]), "player": float(p[1])}
    ), 200

# ==== （可選）LINE Webhook：未設 TOKEN 也能啟動 API ====
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None; line_handler = None
if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import MessageEvent, TextMessage, FollowEvent, TextSendMessage, QuickReply, QuickReplyButton, MessageAction
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        def quick_reply_buttons():
            try:
                return QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="開始分析", text="開始分析")),
                    QuickReplyButton(action=MessageAction(label="返回 ⬅️", text="返回")),
                    QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
                ])
            except Exception:
                return None

        def safe_reply(reply_token: str, text: str, uid: Optional[str] = None):
            try:
                line_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=quick_reply_buttons()))
            except Exception as e:
                log.warning("[LINE] reply failed, try push: %s", e)
                if uid:
                    try: line_api.push_message(uid, TextSendMessage(text=text, quick_reply=quick_reply_buttons()))
                    except Exception as e2: log.error("[LINE] push failed: %s", e2)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            uid = event.source.user_id; _init_user(uid)
            msg = (f"🤖 歡迎！提供 {TRIAL_MINUTES} 分鐘試用。\n"
                   "輸入本金（例：5000）→ 回報『上局結果：閒6 莊8』或僅回『莊/閒/和』→『開始分析』。\n"
                   "只建議『莊/閒』，不建議押『和』。\n"
                   f"到期請輸入：開通 你的密碼（向管理員索取）{ADMIN_CONTACT}")
            line_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=quick_reply_buttons()))

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            uid = event.source.user_id; text = (event.message.text or "").strip()
            if uid not in SESS: _init_user(uid)
            guard = trial_guard(uid)
            if guard: safe_reply(event.reply_token, guard, uid); return

            if text in ["結束分析", "清空", "reset"]:
                SESS[uid] = {"bankroll": 0, "seq": [], "trial_start": int(time.time()), "premium": SESS.get(uid,{}).get("premium", False)}
                safe_reply(event.reply_token, "🧹 已清空。請輸入本金（例：5000），或回報『上局結果：閒6 莊8』或『莊/閒/和』。", uid); return
            if text.startswith("開通") or text.lower().startswith("activate"):
                code = text.split(" ",1)[1].strip() if " " in text else ""
                SESS[uid]["premium"] = validate_activation_code(code)
                safe_reply(event.reply_token, "✅ 已開通成功！" if SESS[uid]["premium"] else "❌ 密碼錯誤，請向管理員索取。", uid); return
            if text.isdigit():
                SESS[uid]["bankroll"] = int(text)
                safe_reply(event.reply_token, f"👍 已設定本金：{int(text):,}", uid); return

            # 點數句子 → Deplete 扣牌 + PF 同步勝負 + 卡片抬頭
            pts = parse_last_hand_points(text)
            if pts is not None:
                p_total, b_total = pts
                try:
                    DEPL.update_hand({"p_total": p_total, "b_total": b_total, "trials": 400})
                    last_outcome = 1 if p_total > b_total else (0 if b_total > p_total else 2)
                    PF.update_outcome(last_outcome)
                except Exception as e:
                    log.warning("deplete update(line) failed: %s", e)
                SESS[uid].setdefault("seq", []).append(last_outcome)
                SESS[uid]["last_pts_text"] = f"上局結果: 閒 {p_total} 莊 {b_total}"
                safe_reply(event.reply_token, "讀取完成\n"+SESS[uid]["last_pts_text"]+"\n開始分析下局....", uid); return

            # 只回「莊/閒/和」：用 PF 更新
            single = text.strip().upper()
            if single in ("B","莊","BANKER"):
                PF.update_outcome(0); SESS[uid].setdefault("seq", []).append(0); SESS[uid]["last_pts_text"]="上局結果: 莊勝"
                safe_reply(event.reply_token, "讀取完成\n上局結果: 莊勝\n開始分析下局....", uid); return
            if single in ("P","閒","PLAYER"):
                PF.update_outcome(1); SESS[uid].setdefault("seq", []).append(1); SESS[uid]["last_pts_text"]="上局結果: 閒勝"
                safe_reply(event.reply_token, "讀取完成\n上局結果: 閒勝\n開始分析下局....", uid); return
            if single in ("T","和","TIE","DRAW"):
                PF.update_outcome(2); SESS[uid].setdefault("seq", []).append(2); SESS[uid]["last_pts_text"]="上局結果: 和局"
                safe_reply(event.reply_token, "讀取完成\n上局結果: 和局\n開始分析下局....", uid); return

            # 開始分析
            if ("開始分析" in text) or (text in ["分析","開始","GO","go"]):
                p_depl = None; p_pf = None
                try: p_depl = DEPL.predict(sims=DEPL_SIMS)
                except Exception as e: log.warning("deplete predict failed: %s", e)
                try: p_pf   = PF.predict(sims_per_particle=PF_PRED_SIMS)
                except Exception as e: log.warning("pf predict failed: %s", e)

                if (p_depl is not None) and (p_pf is not None):
                    w_depl, w_pf = (0.5, 0.5)
                    p = w_depl * p_depl + w_pf * p_pf
                    p[2] = np.clip(p[2], 0.06, 0.20); p = p / p.sum()
                    engine_note = "Mix"
                elif p_depl is not None:
                    p = p_depl; engine_note = "Deplete"
                elif p_pf is not None:
                    p = p_pf; engine_note = "PF"
                else:
                    p = np.array([0.45,0.45,0.10], dtype=np.float32); engine_note = "Fallback"

                choice, edge, bet_pct, reason = decide_only_bp(p)
                msg = format_card_output(p, choice, SESS[uid].get("last_pts_text"))
                safe_reply(event.reply_token, msg, uid); return

            safe_reply(event.reply_token, "🧭 指令：輸入本金 → 回報『上局結果：閒6 莊8』或『莊/閒/和』 → 「開始分析」", uid)

        @app.post("/line-webhook")
        def line_webhook():
            signature = request.headers.get("X-Line-Signature", ""); body = request.get_data(as_text=True)
            try: line_handler.handle(body, signature)
            except InvalidSignatureError: abort(400, "Invalid signature")
            return "OK", 200
    except Exception as e:
        log.warning("LINE not fully configured: %s", e)

# ==== 本地啟動 ====
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
