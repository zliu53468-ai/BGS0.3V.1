import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple
import numpy as np
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

# ---- Optional deps ----
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

VERSION = "bgs-pf-render-optimized-2025-09-17"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---- Flask app setup ----
app = Flask(__name__)
CORS(app)

# ---- Redis or in-memory session ----
REDIS_URL = os.getenv("REDIS_URL")
redis_client: Optional["redis.Redis"] = None  # type: ignore
if redis is not None and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Redis connected.")
    except Exception as e:
        log.error("Redis connect fail: %s; fallback to memory.", e)
        redis_client = None
else:
    if redis is None:
        log.warning("redis module not available; using memory session.")
    elif not REDIS_URL:
        log.warning("REDIS_URL not set; using memory session.")

SESS_FALLBACK: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE_SECONDS = 3600
DEDUPE_TTL = 60

# ---- Session handling ----
def _rget(k: str) -> Optional[str]:
    try:
        return redis_client.get(k) if redis_client else None
    except Exception as e:
        log.warning("[Redis] GET err: %s", e)
        return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if redis_client:
            redis_client.set(k, v, ex=ex)
    except Exception as e:
        log.warning("[Redis] SET err: %s", e)

def _rsetnx(k: str, v: str, ex: int) -> bool:
    try:
        if redis_client:
            return bool(redis_client.set(k, v, ex=ex, nx=True))
        if k in SESS_FALLBACK:
            return False
        SESS_FALLBACK[k] = {"v": v, "exp": time.time() + ex}
        return True
    except Exception as e:
        log.warning("[Redis] SETNX err: %s", e)
        return True

def get_session(uid: str) -> Dict[str, Any]:
    if redis_client:
        j = _rget(f"bgs_session:{uid}")
        if j:
            try:
                return json.loads(j)
            except Exception:
                pass
    else:
        now = time.time()
        for k in list(SESS_FALLBACK.keys()):
            v = SESS_FALLBACK.get(k)
            if isinstance(v, dict) and v.get("exp") and v["exp"] < now:
                del SESS_FALLBACK[k]
        if uid in SESS_FALLBACK and "phase" in SESS_FALLBACK[uid]:
            return SESS_FALLBACK[uid]
    nowi = int(time.time())
    return {
        "bankroll": 0,
        "trial_start": nowi,
        "premium": False,
        "phase": "choose_game",
        "game": None,
        "table": None,
        "last_pts_text": None,
        "table_no": None,
    }

def save_session(uid: str, data: Dict[str, Any]):
    if redis_client:
        _rset(f"bgs_session:{uid}", json.dumps(data), ex=SESSION_EXPIRE_SECONDS)
    else:
        SESS_FALLBACK[uid] = data

# ---- Betting knobs ----
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY = 1
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR", "0.25"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.015"))

# ---- Betting logic ----
def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0:
        return 0
    return int(round(bankroll * pct))

def decide_only_bp(prob: np.ndarray) -> Tuple[str, float, float, str]:
    """根據信心度（優勢）來決定下注金額（不再用勝率映射）。"""
    pB, pP = float(prob[0]), float(prob[1])
    side = 0 if pB >= pP else 1

    # 優勢（信心度）：考慮 0.95 抽水
    evB, evP = 0.95 * pB - pP, pP - pB
    final_edge = max(abs(evB), abs(evP))

    # 階梯配注：5% / 10% / 20% / 30%
    if final_edge >= 0.10:
        bet_pct, reason = 0.30, "高信心"
    elif final_edge >= 0.07:
        bet_pct, reason = 0.20, "中等信心"
    elif final_edge >= 0.04:
        bet_pct, reason = 0.10, "低信心"
    else:
        bet_pct, reason = 0.05, "非常低信心"

    # 仍保留 MAX_BET_PCT 上限（更保守）
    bet_pct = min(bet_pct, float(os.getenv("BET_MAX_PCT", str(0.40))))

    return ("莊" if side == 0 else "閒", final_edge, bet_pct, reason)

# ---- Health Routes ----
@app.get("/")
def root():
    return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION), 200

@app.get("/healthz")
def healthz():
    return jsonify(ok=True, ts=time.time(), version=VERSION), 200

# ---- Line Webhook ----
@app.post("/line-webhook")
def line_webhook():
    from linebot.exceptions import InvalidSignatureError
    try:
        signature = request.headers.get("X-Line-Signature", "")
        body = request.get_data(as_text=True)
        # You would need to process the webhook here with the handler
        return "OK", 200
    except InvalidSignatureError:
        log.error("Invalid signature on webhook")
        return "Invalid signature", 400
    except Exception as e:
        log.error("Webhook error: %s", e)
        return "Internal error", 500

# ---- Main ----
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
