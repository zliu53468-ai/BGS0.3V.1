
# server_pro.py — BGS 百家樂（專業版）
# Deplete（扣牌）+ 粒子濾波 PF +（可選）XGB/LGB/RNN 融合
# EV-first 決策 + 1/4 Kelly 倉控；與 server_lite.py 保持相同 API 介面
# Author: 親愛的 x GPT-5 Thinking

import os, time, re, csv, logging
from typing import List, Optional, Dict, Tuple
import numpy as np
from flask import Flask, request, jsonify

log = logging.getLogger("bgs-pro")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

# ====== 環境參數 ======
SEED = int(os.getenv("SEED","42"))
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES","30"))  # 這支不做 LINE 試用控，只保留參數
np.random.seed(SEED)

# Deplete / PF 參數
DEPL_DECKS   = int(os.getenv("DEPL_DECKS", "8"))
DEPL_SIMS    = int(os.getenv("DEPL_SIMS",  "30000"))
PF_N         = int(os.getenv("PF_N", "200"))
PF_UPD_SIMS  = int(os.getenv("PF_UPD_SIMS", "80"))
PF_PRED_SIMS = int(os.getenv("PF_PRED_SIMS","220"))
PF_RESAMPLE  = float(os.getenv("PF_RESAMPLE","0.5"))
PF_DIR_ALPHA = float(os.getenv("PF_DIR_ALPHA","0.8"))
PF_USE_EXACT = int(os.getenv("PF_USE_EXACT","0"))

# Ensemble（可選）
TEMP_XGB = float(os.getenv("TEMP_XGB", "0.9"))
TEMP_LGB = float(os.getenv("TEMP_LGB", "0.9"))
TEMP_RNN = float(os.getenv("TEMP_RNN", "0.8"))
ENSEMBLE_WEIGHTS = os.getenv("ENSEMBLE_WEIGHTS", "xgb:0.3,lgb:0.3,rnn:0.4")

# 決策參數
EDGE_ENTER   = float(os.getenv("EDGE_ENTER","0.03"))   # EV 進場門檻（抽水後）
USE_KELLY    = int(os.getenv("USE_KELLY","1"))
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR","0.25")) # 1/4 Kelly
MAX_BET_PCT  = float(os.getenv("MAX_BET_PCT","0.015")) # 單注上限（1.5%）

# ====== 模組載入 ======
from bgs.deplete import DepleteMC
from bgs.pfilter import OutcomePF

# （可選）樹/深度模型：若檔案不存在就自動跳過
XGB = None; LGBM = None; RNN = None
def _load_xgb():
    global XGB
    try:
        import xgboost as xgb
        path = os.getenv("XGB_OUT_PATH", "data/models/xgb.json")
        if os.path.exists(path):
            booster = xgb.Booster(); booster.load_model(path)
            XGB = booster; log.info("[MODEL] XGB loaded")
    except Exception as e:
        log.info("[MODEL] XGB not used: %s", e)

def _load_lgb():
    global LGBM
    try:
        import lightgbm as lgb
        path = os.getenv("LGBM_OUT_PATH", "data/models/lgbm.txt")
        if os.path.exists(path):
            LGBM = lgb.Booster(model_file=path); log.info("[MODEL] LGBM loaded")
    except Exception as e:
        log.info("[MODEL] LGBM not used: %s", e)

def _load_rnn():
    global RNN
    try:
        import torch, torch.nn as nn
        class TinyRNN(nn.Module):
            def __init__(self, in_dim=3, hid=int(os.getenv("RNN_HIDDEN","32")), out_dim=3):
                super().__init__(); self.gru = nn.GRU(in_dim, hid, 1, batch_first=True); self.fc = nn.Linear(hid, out_dim)
            def forward(self, x): o,_ = self.gru(x); return self.fc(o[:,-1,:])
        path = os.getenv("RNN_OUT_PATH", "data/models/rnn.pt")
        if os.path.exists(path):
            RNN = TinyRNN(); state = __import__("torch").load(path, map_location="cpu")
            RNN.load_state_dict(state); RNN.eval(); log.info("[MODEL] RNN loaded")
    except Exception as e:
        log.info("[MODEL] RNN not used: %s", e)

_load_xgb(); _load_lgb(); _load_rnn()

# ====== 小工具 ======
MAP = {"B":0,"P":1,"T":2,"莊":0,"閒":1,"和":2}
INV = {0:"莊",1:"閒",2:"和"}

def parse_history(s: str) -> List[int]:
    s = (s or "").strip().upper()
    if not s: return []
    s = s.replace("，"," ").replace("、"," ").replace("\u3000"," ")
    toks = s.split()
    seq = list(s) if (len(toks)==1 and len(s)<=20) else toks
    out=[]
    for t in seq:
        t = t.strip().upper()
        if t in MAP: out.append(MAP[t])
    return out

def parse_last_hand_points(text: str):
    if not text: return None
    s = (text or "").strip().upper()
    s = s.replace("：",":")
    s = re.sub(r"\s+","",s)
    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*(?:和|TIE|DRAW)[:]?(\d)', s)
    if m: d=int(m.group(1)); return (d,d)
    if re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*(?:和|TIE|DRAW)\b', s): return None
    m = re.search(r'(?:閒|P)[:]?(\d).*?(?:莊|B)[:]?(\d)', s)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'(?:莊|B)[:]?(\d).*?(?:閒|P)[:]?(\d)', s)
    if m: return (int(m.group(2)), int(m.group(1)))
    return None

def softmax_log(p: np.ndarray, temp: float=1.0) -> np.ndarray:
    x = np.log(np.clip(p,1e-9,None)) / max(1e-9,temp)
    x = x - x.max(); e = np.exp(x); return e / e.sum()

def banker_ev(pB, pP): return 0.95*pB - pP  # tie 退回
def player_ev(pB, pP): return pP - pB

def kelly_fraction(p_win: float, payoff: float):
    q = 1.0 - p_win; edge = p_win*payoff - q
    return max(0.0, edge / max(1e-9,payoff))

def bet_amount(bankroll:int, pct:float) -> int:
    if bankroll<=0 or pct<=0: return 0
    return int(round(bankroll*pct))

# ====== 引擎 ======
DEPL = DepleteMC(decks=DEPL_DECKS, seed=SEED)
PF   = OutcomePF(decks=DEPL_DECKS, seed=SEED, n_particles=PF_N,
                 sims_lik=PF_UPD_SIMS, resample_thr=PF_RESAMPLE,
                 dirichlet_alpha=PF_DIR_ALPHA, use_exact=bool(PF_USE_EXACT))

def _ensemble_probs(seq: List[int]) -> np.ndarray:
    preds=[]; names=[]; W=[]
    # Deplete
    try:
        p_depl = DEPL.predict(sims=DEPL_SIMS)
        if p_depl is not None: preds.append(p_depl); names.append("DEPL"); W.append(0.5)
    except Exception as e:
        log.warning("Deplete predict failed: %s", e)
    # PF
    try:
        p_pf = PF.predict(sims_per_particle=PF_PRED_SIMS)
        if p_pf is not None: preds.append(p_pf); names.append("PF"); W.append(0.5)
    except Exception as e:
        log.warning("PF predict failed: %s", e)

    # Optional XGB/LGB/RNN
    feat = None
    if XGB or LGBM:
        from bgs.features import big_road_features as _brf  # 如果沒有此檔，請關閉或自行提供
        try:
            feat = _brf(seq).reshape(1,-1)
        except Exception:
            feat = None
    if XGB and feat is not None:
        import xgboost as xgb
        try:
            px = XGB.predict(xgb.DMatrix(feat))[0]; preds.append(softmax_log(np.array(px), TEMP_XGB)); names.append("XGB"); W.append(0.0)
        except Exception as e: log.info("XGB skip: %s", e)
    if LGBM and feat is not None:
        try:
            pl = LGBM.predict(feat)[0]; preds.append(softmax_log(np.array(pl), TEMP_LGB)); names.append("LGBM"); W.append(0.0)
        except Exception as e: log.info("LGBM skip: %s", e)
    if RNN:
        try:
            import torch
            sub = seq[-128:] if len(seq)>128 else seq[:]
            L=len(sub); oh=np.zeros((1,L,3),dtype=np.float32)
            for i,v in enumerate(sub):
                if v in (0,1,2): oh[0,i,v]=1.0
            x = torch.from_numpy(oh)
            with torch.no_grad():
                logits=RNN(x)
                pr = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            preds.append(softmax_log(pr, TEMP_RNN)); names.append("RNN"); W.append(0.0)
        except Exception as e: log.info("RNN skip: %s", e)

    if not preds:
        return np.array([0.45,0.45,0.10], dtype=np.float32)
    P = np.stack(preds, axis=0); W = np.array(W, dtype=np.float32)
    if W.sum() <= 0: W = np.ones(len(preds), dtype=np.float32)/len(preds)
    W = W / W.sum()
    p = (P * W[:,None]).sum(axis=0)
    p[2] = np.clip(p[2], 0.06, 0.20); p = p / p.sum()
    return p.astype(np.float32)

def decide_bp(p: np.ndarray) -> Tuple[str, float, float, str]:
    pB, pP, pT = float(p[0]), float(p[1]), float(p[2])
    evB, evP = banker_ev(pB,pP), player_ev(pB,pP)
    side = 0 if evB > evP else 1
    ev_edge = max(evB, evP)
    if ev_edge < EDGE_ENTER:
        return ("觀望", ev_edge, 0.0, f"⚪ 優勢不足（門檻 {EDGE_ENTER:.3f}）")
    if USE_KELLY:
        win_p = pB if side==0 else pP
        payoff = 0.95 if side==0 else 1.0
        f = KELLY_FACTOR * kelly_fraction(win_p, payoff)
        bet_pct = min(MAX_BET_PCT, float(max(0.0, f)))
        reason = "🧠 EV優先｜📐 ¼-Kelly"
    else:
        if ev_edge >= 0.10: bet_pct = 0.25
        elif ev_edge >= 0.07: bet_pct = 0.15
        elif ev_edge >= 0.04: bet_pct = 0.10
        else: bet_pct = 0.05
        reason = "🧠 EV優先｜🪜 階梯配注"
    return (INV[side], ev_edge, bet_pct, reason)

# ====== 介面 ======
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), engine="pro"), 200

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    bankroll = int(float(data.get("bankroll") or 0))
    seq = parse_history(str(data.get("history","")))

    # 可選：從 last_pts 更新 Deplete / PF（能提升準確度）
    lp = data.get("last_pts")
    if lp:
        pts = parse_last_hand_points(lp)
        if pts is not None:
            p_total, b_total = int(pts[0]), int(pts[1])
            try:
                DEPL.update_hand({"p_total": p_total, "b_total": b_total, "trials": 400})
            except Exception as e:
                log.warning("Deplete update failed: %s", e)
            PF.update_outcome(1 if p_total>b_total else (0 if b_total>p_total else 2))

    p = _ensemble_probs(seq)
    choice, ev_edge, bet_pct, reason = decide_bp(p)
    amt = bet_amount(bankroll, bet_pct)

    b_pct, p_pct = int(round(100*p[0])), int(round(100*p[1]))
    evB, evP = banker_ev(float(p[0]), float(p[1])), player_ev(float(p[0]), float(p[1]))
    msg = (
        f"🎯 下一局建議：{choice}\n"
        f"💰 建議注額：{amt:,}\n"
        f"📊 機率｜莊 {b_pct}%｜閒 {p_pct}%｜和 {int(round(100*p[2]))}%\n"
        f"📐 EV（抽水後）｜莊 {evB:.3f}｜閒 {evP:.3f}\n"
        f"{reason}"
    )

    return jsonify(
        message=msg, engine="pro",
        suggestion=choice, bet_pct=float(bet_pct), bet_amount=amt,
        probabilities={"banker": float(p[0]), "player": float(p[1]), "tie": float(p[2])},
        ev={"banker": float(evB), "player": float(evP)},
    ), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting PRO on %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
