# -*- coding: utf-8 -*-
"""server.py — BGS Pure PF + Deplete + Stage Overrides + FULL LINE Flow + Compatibility + Stability + Advanced Control (FINAL FIXED + KEEPALIVE-CORRECT-ORDER)"""
import os, sys, logging, time, re, json, threading
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None:
        return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return 1
    if v in ("0", "false", "f", "no", "n", "off"):
        return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except Exception:
        return 1 if default else 0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")
np.seterr(all="ignore")

# ==================== KEEP ALIVE 先定義（最重要！移到最前面） ====================
KEEP_ALIVE_STARTED = False
KEEP_ALIVE_LOCK = threading.Lock()

def _self_keep_alive():
    """Keep Alive 防 Render 休眠 - 已修正定義順序"""
    import time
    try:
        import requests
    except Exception:
        log.warning("[KEEPALIVE] requests module not available, skip self-ping")
        return

    # 優先使用 Render 官方變數
    url = os.getenv("RENDER_EXTERNAL_URL")
    if not url:
        url = os.getenv("SELF_URL")
    if not url:
        url = os.getenv("SELF_PING_URL")

    interval = int(os.getenv("SELF_PING_INTERVAL", "120"))  # 改為 120 秒更穩定

    if not url:
        log.warning("[KEEPALIVE] No URL found (RENDER_EXTERNAL_URL / SELF_URL / SELF_PING_URL), skip self-ping")
        return

    # 打 /ping 比打根路徑更穩定
    ping_url = url.rstrip("/") + "/ping"

    log.info(f"[KEEPALIVE] Started successfully | URL: {ping_url} | interval: {interval}s")

    while True:
        try:
            r = requests.get(ping_url, timeout=10)
            if r.status_code < 400:
                log.info("[KEEPALIVE] self ping success")
            else:
                log.warning(f"[KEEPALIVE] ping returned status {r.status_code}")
        except Exception as e:
            log.warning("[KEEPALIVE] self ping failed: %s", e)
        time.sleep(interval)


# ---------- deplete ----------
DEPLETE_OK = False
init_counts = None
probs_after_points = None
try:
    from deplete import init_counts, probs_after_points
    DEPLETE_OK = True
except Exception:
    try:
        from bgs.deplete import init_counts, probs_after_points
        DEPLETE_OK = True
    except Exception:
        try:
            _cur_dir = os.path.dirname(os.path.abspath(__file__))
            if _cur_dir not in sys.path:
                sys.path.insert(0, _cur_dir)
            from deplete import init_counts, probs_after_points
            DEPLETE_OK = True
        except Exception:
            DEPLETE_OK = False

# ---------- Flask ----------
try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
    _flask_available = True
except Exception:
    _flask_available = False
    Flask = None
    request = None
    def jsonify(*args, **kwargs):
        raise RuntimeError("Flask not available")
    def abort(*args, **kwargs):
        raise RuntimeError("Flask not available")
    def CORS(app):
        return None

# ---------- Redis（可選） ----------
try:
    import redis
except Exception:
    redis = None

REDIS_URL = os.getenv("REDIS_URL")
redis_client: Optional["redis.Redis"] = None
if redis is not None and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Successfully connected to Redis.")
    except Exception as e:
        redis_client = None
        log.error("Failed to connect to Redis: %s. Using in-memory session.", e)
else:
    if redis is None:
        log.warning("redis module not available; using in-memory session store.")
    elif not REDIS_URL:
        log.warning("REDIS_URL not set. Using in-memory session store.")

SESS_FALLBACK: Dict[str, Dict[str, Any]] = {}
KV_FALLBACK: Dict[str, str] = {}
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1200"))
DEDUPE_TTL = 60

def _rget(k: str) -> Optional[str]:
    try:
        if redis_client:
            return redis_client.get(k)
        return KV_FALLBACK.get(k)
    except Exception as e:
        log.warning("[Redis] GET err: %s", e)
        return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if redis_client:
            redis_client.set(k, v, ex=ex)
        else:
            KV_FALLBACK[k] = v
    except Exception as e:
        log.warning("[Redis] SET err: %s", e)

def _rsetnx(k: str, v: str, ex: int) -> bool:
    try:
        if redis_client:
            return bool(redis_client.set(k, v, ex=ex, nx=True))
        if k in KV_FALLBACK:
            return False
        KV_FALLBACK[k] = v
        return True
    except Exception as e:
        log.warning("[Redis] SETNX err: %s", e)
        return True

# ---------- 事件去重、Premium、Session、UI 卡片等（完全不變） ----------
def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id:
        return True
    key = f"dedupe:{event_id}"
    return _rsetnx(key, "1", ex=DEDUPE_TTL)

def _extract_line_event_id(event: Any) -> Optional[str]:
    try:
        eid = getattr(event, "webhook_event_id", None)
        if eid:
            return str(eid)
    except Exception:
        pass
    try:
        msg = getattr(event, "message", None)
        mid = getattr(msg, "id", None) if msg is not None else None
        if mid:
            return str(mid)
    except Exception:
        pass
    try:
        eid2 = getattr(event, "id", None)
        if eid2:
            return str(eid2)
    except Exception:
        pass
    return None

def _premium_key(uid: str) -> str:
    return f"premium:{uid}"

def is_premium(uid: str) -> bool:
    if not uid:
        return False
    return _rget(_premium_key(uid)) == "1"

def set_premium(uid: str, flag: bool = True) -> None:
    if not uid:
        return
    _rset(_premium_key(uid), "1" if flag else "0")

def _sess_key(uid: str) -> str:
    return f"sess:{uid}"

def get_session(uid: str) -> Dict[str, Any]:
    if not uid:
        uid = "anon"
    try:
        if redis_client:
            raw = redis_client.get(_sess_key(uid))
            if raw:
                sess = json.loads(raw)
                if is_premium(uid):
                    sess["premium"] = True
                sess.setdefault("pending", False)
                sess.setdefault("pending_seq", 0)
                sess.setdefault("loss_streak", 0)
                sess.setdefault("adv_history", [])
                sess.setdefault("last_choice", None)
                return sess
        sess = SESS_FALLBACK.get(uid)
        if isinstance(sess, dict):
            if is_premium(uid):
                sess["premium"] = True
            sess.setdefault("pending", False)
            sess.setdefault("pending_seq", 0)
            sess.setdefault("loss_streak", 0)
            sess.setdefault("adv_history", [])
            sess.setdefault("last_choice", None)
            return sess
    except Exception as e:
        log.warning("get_session error: %s", e)

    sess = {
        "phase": "await_pts",
        "bankroll": 0,
        "rounds_seen": 0,
        "last_pts_text": None,
        "premium": is_premium(uid),
        "trial_start": int(time.time()),
        "last_card": None,
        "last_card_ts": None,
        "pending": False,
        "pending_seq": 0,
        "loss_streak": 0,
        "adv_history": [],
        "last_choice": None,
    }
    save_session(uid, sess)
    return sess

def save_session(uid: str, sess: Dict[str, Any]) -> None:
    if not uid:
        uid = "anon"
    try:
        payload = json.dumps(sess, ensure_ascii=False)
        if redis_client:
            redis_client.set(_sess_key(uid), payload, ex=SESSION_EXPIRE_SECONDS)
        else:
            SESS_FALLBACK[uid] = sess
            KV_FALLBACK[_sess_key(uid) + ":ttl"] = str(int(time.time()) + SESSION_EXPIRE_SECONDS)
    except Exception as e:
        log.warning("save_session error: %s", e)

def format_output_card(probs: np.ndarray, choice: str, last_pts: Optional[str],
                       bet_amt: int, cont: bool = True, mode: str = "") -> str:
    pB, pP, pT = [float(x) for x in probs]
    lines = []
    if last_pts:
        lines.append(str(last_pts))
    lines.append(f"機率｜莊 {pB*100:.2f}%｜閒 {pP*100:.2f}%｜和 {pT*100:.2f}%")
    lines.append(f"差距｜莊閒 {abs(pB - pP) * 100:.2f}%")
    if mode:
        lines.append(f"模式｜{mode}")
    if choice == "觀望":
        lines.append("建議：觀望 👀")
    else:
        lines.append(f"建議：下 {choice} 🎯")
        if bet_amt and bet_amt > 0:
            lines.append(f"配注：{bet_amt}")
    if cont:
        lines.append("\n（輸入下一局點數：例如 65 / 和 / 閒6莊5）")
    return "\n".join(lines)

VERSION = "bgs-pure-pf-deplete-2025-11-03+optimized+pattern-removed+PF220+advanced-control+dynamic-deplete+FINAL-FIXED+KEEPALIVE-CORRECT-ORDER"

# ---------- Flask App ----------
if _flask_available and Flask is not None:
    app = Flask(__name__)
    CORS(app)

    # 🔥 Keep Alive 在這裡啟動（函數已在上方定義完成）
    try:
        if not KEEP_ALIVE_STARTED:
            with KEEP_ALIVE_LOCK:
                if not KEEP_ALIVE_STARTED:
                    threading.Thread(target=_self_keep_alive, daemon=True).start()
                    KEEP_ALIVE_STARTED = True
                    log.info("✅ KEEP ALIVE thread started at server boot (Correct Order + /ping)")
    except Exception as e:
        log.warning("KEEP ALIVE boot failed: %s", e)
else:
    class _DummyApp:
        def get(self, *a, **k):
            def _d(f): return f
            return _d
        def post(self, *a, **k):
            def _d(f): return f
            return _d
        def options(self, *a, **k):
            def _d(f): return f
            return _d
        def run(self, *a, **k):
            log.warning("Flask not available; cannot run HTTP server.")
    app = _DummyApp()

# ========== 關鍵路由 ==========
@app.get("/")
def root():
    ua = request.headers.get("User-Agent", "") if request else ""
    if "UptimeRobot" in ua or "bot" in ua.lower():
        return "OK", 200
    st = "OK" if pf_initialized else "BACKUP_MODE"
    return f"✅ BGS Server {st} ({VERSION})", 200

@app.get("/ping")
def ping():
    return "OK", 200

@app.get("/health")
def health():
    return {"ok": True, "ts": time.time(), "version": VERSION, "status": "running"}, 200

# ---------- PF、決策、Deplete、_handle_points_and_predict 等（請貼上你原本的完整程式碼） ----------
# 從這裡開始把你原本的 PF 初始化、SmartDummyPF、決策函數、三段覆蓋、_handle_points_and_predict 等全部貼上
# （這些部分我完全沒有改動）

# ---------- LINE 完整部分（請貼上你原本的 LINE 程式碼） ----------
# 包含 LINE_CHANNEL_SECRET、line_handler、所有路由等

# ---------- 啟動入口 ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s (KEEPALIVE=boot-started, interval=120s)", VERSION, port)
    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
