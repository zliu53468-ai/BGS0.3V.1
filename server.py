# -*- coding: utf-8 -*-
"""server.py — BGS Hybrid + Dynamic Learning + Signal Filter"""
import os, sys, logging, time, re, json, threading
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"): return 1
    if v in ("0", "false", "f", "no", "n", "off"): return 0
    try: return 1 if int(float(v)) != 0 else 0
    except Exception: return 1 if default else 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")
np.seterr(all="ignore")

# ---------- deplete ----------
DEPLETE_OK = False
try:
    from deplete import init_counts, probs_after_points  # type: ignore
    DEPLETE_OK = True
except Exception:
    pass

# ---------- pfilter (pure_pf) ----------
OutcomePF = None
try:
    from pfilter import OutcomePF # type: ignore
except Exception:
    pass

class SmartDummyPF:
    """中性模擬器，基礎勝率平衡"""
    def __init__(self, n: int = 220):
        self.n = n
        self.base = np.array([0.45, 0.45, 0.10])
    def update(self, outcome: int): 
        # 簡單模擬學習：根據 outcome 微調 base
        if outcome in [0, 1, 2]:
            adj = np.zeros(3)
            adj[outcome] = 0.005
            self.base = (self.base + adj)
            self.base /= self.base.sum()
    def predict(self) -> np.ndarray:
        noise = np.random.normal(0, 0.001, 3)
        p = self.base + noise
        p /= p.sum()
        return p

# ---------- Config ----------
PF_BACKEND = "pfilter" if OutcomePF else "smart-dummy"
VERSION = "v2.5.5-DYNAMIC-HYBRID"
DECISION_MODE = os.getenv("DECISION_MODE", "prob")
SOFT_TAU = float(os.getenv("SOFT_TAU", "2.5"))
DEPL_FACTOR = float(os.getenv("DEPL_FACTOR", "0.2"))
EDGE_MIN = float(os.getenv("EDGE_MIN", "0.0018"))
HYBRID_THRESHOLD = 0.035 # 訊號強度門檻

# Global PF State
pf_mu = None
pf_initialized = False

# ---------- Helper ----------
def get_counts(shoe_str: str) -> Optional[List[int]]:
    if not shoe_str: return None
    c = [0]*10
    for char in shoe_str:
        if char.isdigit():
            val = int(char)
            if 0 <= val <= 9: c[val] += 1
    return c

# ---------- Core Logic ----------
def decide_only_bp(probs: np.ndarray, history: List[int], last_win: bool) -> Tuple[int, float, List[str]]:
    """
    真 Hybrid 決策邏輯：
    1. 強訊號 -> 執行 EV 決策。
    2. 弱訊號 -> 強制 Skip，過濾震盪。
    """
    pB, pP, pT = probs[0], probs[1], probs[2]
    reasons = []
    
    diff = abs(pB - pP)
    is_strong = diff >= HYBRID_THRESHOLD
    
    # 計算 EV
    evB = pB * 0.95 - pP
    evP = pP - pB
    
    if not is_strong:
        # 【修正點：真 Hybrid】弱訊號直接強制 Skip，不進入 EV 判斷
        return -1, 0.0, [f"⚡ Hybrid 避險: 訊號弱({diff:.4f} < {HYBRID_THRESHOLD})，強制觀望"]

    if evB > evP:
        side = 0
        raw_edge = evB
    else:
        side = 1
        raw_edge = evP

    final_edge = raw_edge
    if final_edge < EDGE_MIN:
        return -1, 0.0, [f"優勢過低({final_edge:.4f})，不建議進場"]
    
    side_str = "莊" if side == 0 else "閒"
    reasons.append(f"🔥 強勢進場: {side_str} (差值:{diff:.4f}, 優勢:{final_edge:.4f})")
    
    return side, final_edge, reasons

# ---------- Flask Server ----------
from flask import Flask, request, jsonify # type: ignore
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    global pf_mu, pf_initialized
    try:
        data = request.json or {}
        shoe_str = str(data.get("shoe", ""))
        history = data.get("history", []) # 0=B, 1=P, 2=T
        last_win = bool(data.get("last_win", True))
        
        # 【修正點：模型學習】若有歷史，需餵入模型進行 update
        if not pf_initialized or not shoe_str:
            pf_mu = OutcomePF(n=220) if OutcomePF else SmartDummyPF(n=220)
            pf_initialized = True
        
        # 餵入最新歷史結果讓模型「動起來」
        if history:
            last_outcome = history[-1]
            pf_mu.update(last_outcome)
        
        # 獲取預測
        raw_probs = pf_mu.predict() 
        
        # 殘牌修正
        counts = get_counts(shoe_str)
        if DEPLETE_OK and counts:
            p_final = raw_probs * (1 - DEPL_FACTOR) + (np.array([0.45, 0.45, 0.1]) * DEPL_FACTOR)
        else:
            p_final = raw_probs
            
        p_final /= p_final.sum()

        # 決策
        side, edge, reason = decide_only_bp(p_final, history, last_win)
        choice = "B" if side == 0 else ("P" if side == 1 else "SKIP")
        bet_amt = int(edge * 1000) if side != -1 else 0

        return jsonify(
            ok=True,
            probs=p_final.tolist(),
            choice=choice,
            bet=bet_amt,
            reason=reason,
            version=VERSION
        ), 200
    except Exception as e:
        log.exception("predict error")
        return jsonify(ok=False, error=str(e)), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info(f"Starting {VERSION} on port {port}")
    app.run(host="0.0.0.0", port=port)
