"""
server.py — 連續模式修正版（Render 優化版 + 信心度→金額 + 和局穩定器）

- Render 免費版資源優化（輕量 PF）
- 依「信心度/優勢」配注金額（階梯 5% / 10% / 20% / 30%）
- 觀望不顯示金額；非觀望只顯示金額
- 和局處理：T 機率夾緊 + 和局後冷卻（可關）
- 機率平滑與溫度縮放（可關）
"""

import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple
import numpy as np

# ---- Optional deps ----
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

try:
    from flask import Flask, request, jsonify, abort  # type: ignore
    from flask_cors import CORS  # type: ignore
    _flask_available = True
except Exception:
    _flask_available = False
    Flask = None  # type: ignore
    request = None  # type: ignore
    def jsonify(*args, **kwargs):  # type: ignore
        raise RuntimeError("Flask is not available")
    def abort(*args, **kwargs):  # type: ignore
        raise RuntimeError("Flask is not available")
    def CORS(app):  # type: ignore
        return None

VERSION = "bgs-pf-render-optimized-2025-09-17"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---- Flask or dummy ----
if _flask_available and Flask is not None:
    app = Flask(__name__)
    CORS(app)
else:
    class _DummyApp:
        def get(self, *args, **kwargs):
            def _d(fn): return fn
            return _d
        def post(self, *args, **kwargs):
            def _d(fn): return fn
            return _d
        def run(self, *args, **kwargs):
            log.warning("Flask not available; dummy app cannot run.")
    app = _DummyApp()

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

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None:
        return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"): return 1
    if v in ("0", "false", "f", "no", "n", "off"): return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except Exception:
        return 1 if default else 0

# ---- Betting knobs ----
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY = env_flag("USE_KELLY", 1)
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR", "0.25"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.015"))

# 勝率→配注（線性）：環境變數可調（現行改為用信心度，保留參數不動）
USE_WINRATE_MAP = env_flag("USE_WINRATE_MAP", 1)
BET_MIN_PCT = float(os.getenv("BET_MIN_PCT", "0.05"))   # 5%
BET_MAX_PCT = float(os.getenv("BET_MAX_PCT", "0.40"))   # 40%
WINRATE_FLOOR = float(os.getenv("WINRATE_FLOOR", "0.50"))
WINRATE_CEIL  = float(os.getenv("WINRATE_CEIL",  "0.75"))

# 和局穩定器 + 機率平滑 + 溫度縮放
TIE_PROB_MIN = float(os.getenv("TIE_PROB_MIN", "0.02"))
TIE_PROB_MAX = float(os.getenv("TIE_PROB_MAX", "0.12"))
POST_TIE_COOLDOWN = int(os.getenv("POST_TIE_COOLDOWN", "1"))
PROB_SMA_ALPHA = float(os.getenv("PROB_SMA_ALPHA", os.getenv("PROB_SMA_ALPHA".lower(), "0")))
PROB_TEMP = float(os.getenv("PROB_TEMP", os.getenv("PROB_TEMP".lower(), "1.0")))

CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)
INV = {0: "莊", 1: "閒"}

# ---- Parse last hand points ----
def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s)
    s = s.replace("\u3000", " ")
    u = s.upper().strip()
    u = re.sub(r"^開始分析", "", u)

    m = re.search(r"(?:和|TIE|DRAW)\s*:?:?\s*(\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else (0, 0)

    m = re.search(r"(?:閒|闲|P)\s*:?:?\s*(\d)\D+(?:莊|庄|B)\s*:?:?\s*(\d)", u)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    m = re.search(r"(?:莊|庄|B)\s*:?:?\s*(\d)\D+(?:閒|闲|P)\s*:?:?\s*(\d)", u)
    if m:
        return (int(m.group(2)), int(m.group(1)))

    t = u.replace(" ", "").replace("\u3000", "")
    if t in ("B", "莊", "庄"): return (0, 1)
    if t in ("P", "閒", "闲"): return (1, 0)
    if t in ("T", "和"):       return (0, 0)

    if re.search(r"[A-Z]", u):
        return None

    digits = re.findall(r"\d", u)
    if len(digits) == 2:
        return (int(digits[0]), int(digits[1]))
    return None

# ---- Main ----
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s (CONTINUOUS_MODE=%s)", VERSION, port, CONTINUOUS_MODE)
    app.run(host="0.0.0.0", port=port, debug=False)
