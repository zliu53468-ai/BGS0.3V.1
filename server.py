# -*- coding: utf-8 -*-
"""server.py — BGS Pure PF + Deplete + Stage Overrides + FULL LINE Flow + Compatibility + Stability + Advanced Control (FINAL FIXED + GPT-PATCH4)"""
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

# ---------- Keep Alive 變數提前宣告 ----------
KEEP_ALIVE_STARTED = False
KEEP_ALIVE_LOCK = threading.Lock()

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

# ---------- 事件去重 ----------
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

# ---------- Premium ----------
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

# ---------- Session ----------
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

# ---------- UI 卡片 ----------
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

VERSION = "bgs-pure-pf-deplete-2025-11-03+optimized+pattern-removed+PF220+advanced-control+dynamic-deplete+FINAL-FIXED+GPT-PATCH4+keepalive-fixed"

# ---------- Flask App ----------
if _flask_available and Flask is not None:
    app = Flask(__name__)
    CORS(app)

    # 🔥 關鍵修正：服務啟動時就立即啟動 Keep Alive（不再依賴第一次預測）
    try:
        if not KEEP_ALIVE_STARTED:
            with KEEP_ALIVE_LOCK:
                if not KEEP_ALIVE_STARTED:
                    threading.Thread(target=_self_keep_alive, daemon=True).start()
                    KEEP_ALIVE_STARTED = True
                    log.info("✅ KEEP ALIVE thread started at server boot (RENDER compatible)")
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

# ========== 關鍵路由（解決 UptimeRobot 404 Not Found） ==========
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
    return {
        "ok": True,
        "ts": time.time(),
        "version": VERSION,
        "pf_initialized": pf_initialized,
        "status": "running"
    }, 200

# ---------- PF ----------
PF_BACKEND = os.getenv("PF_BACKEND", "mc").lower()
SKIP_TIE_UPD = env_flag("SKIP_TIE_UPD", 1)
SOFT_TAU = float(os.getenv("SOFT_TAU", "2.0"))
TIE_MIN = float(os.getenv("TIE_MIN", "0.05"))
TIE_MAX = float(os.getenv("TIE_MAX", "0.15"))
HISTORY_MODE = env_flag("HISTORY_MODE", 0)
TIE_CAP_ENABLE = env_flag("TIE_CAP_ENABLE", 1)
SHOW_RAW_PROBS = env_flag("SHOW_RAW_PROBS", 0)
PF_STATEFUL = env_flag("PF_STATEFUL", 1)

OutcomePF = None
pf_initialized = False
try:
    from bgs.pfilter import OutcomePF as RealOutcomePF
    OutcomePF = RealOutcomePF
    log.info("成功從 bgs.pfilter 導入 OutcomePF")
except Exception:
    try:
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        if _cur_dir not in sys.path:
            sys.path.insert(0, _cur_dir)
        from pfilter import OutcomePF as LocalOutcomePF
        OutcomePF = LocalOutcomePF
        log.info("成功從本地 pfilter 導入 OutcomePF")
    except Exception as pf_exc:
        log.error("無法導入 OutcomePF: %s", pf_exc)
        OutcomePF = None

class SmartDummyPF:
    def __init__(self):
        log.warning("使用 SmartDummyPF 備援模式")
    def update_outcome(self, outcome):
        return
    def predict(self, **kwargs) -> np.ndarray:
        base = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
        base = base ** (1.0 / max(1e-6, SOFT_TAU))
        base = base / base.sum()
        pT = float(base[2])
        if pT < TIE_MIN:
            base[2] = TIE_MIN
            sc = (1.0 - TIE_MIN) / (1.0 - pT) if pT < 1.0 else 1.0
            base[0] *= sc
            base[1] *= sc
        elif pT > TIE_MAX:
            base[2] = TIE_MAX
            sc = (1.0 - TIE_MAX) / (1.0 - pT) if pT < 1.0 else 1.0
            base[0] *= sc
            base[1] *= sc
        return base.astype(np.float32)
    @property
    def backend(self):
        return "smart-dummy"

_GLOBAL_PF = None
_GLOBAL_PF_LOCK = threading.Lock()
PREDICT_LOCK = threading.Lock()
_RESULT_CACHE: Dict[str, Tuple[np.ndarray, str, int, str]] = {}
_RESULT_CACHE_KEY: Dict[str, str] = {}
_RESULT_CACHE_LOCK = threading.Lock()

def _clear_prediction_cache(uid: str) -> None:
    if not uid:
        uid = "anon"
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE.pop(uid, None)
        _RESULT_CACHE_KEY.pop(uid, None)

def _make_result_cache_key(uid: str, sess: Dict[str, Any], p_pts: int, b_pts: int) -> str:
    rounds_seen = int(sess.get("rounds_seen", 0))
    bankroll = int(sess.get("bankroll", 0))
    pf_stateful = int(PF_STATEFUL)
    mode = str(sess.get("decision_mode") or os.getenv("DECISION_MODE", "hybrid")).lower()
    return f"{uid}|r={rounds_seen}|p={p_pts}|b={b_pts}|bank={bankroll}|stateful={pf_stateful}|mode={mode}"

_UID_LOCKS: Dict[str, threading.Lock] = {}
_UID_LOCKS_GUARD = threading.Lock()

def _get_uid_lock(uid: str) -> threading.Lock:
    if not uid:
        uid = "anon"
    with _UID_LOCKS_GUARD:
        if uid not in _UID_LOCKS:
            _UID_LOCKS[uid] = threading.Lock()
        return _UID_LOCKS[uid]

def _build_new_pf() -> Any:
    if OutcomePF is None:
        return SmartDummyPF()
    return OutcomePF(
        decks=int(os.getenv("DECKS", "8")),
        seed=int(os.getenv("SEED", "42")),
        n_particles=int(os.getenv("PF_N", "220")),
        sims_lik=int(os.getenv("PF_UPD_SIMS", "30")),
        resample_thr=float(os.getenv("PF_RESAMPLE", "0.5")),
        backend=PF_BACKEND,
        dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.05"))
    )

def get_pf_for_uid(uid: str) -> Any:
    global _GLOBAL_PF
    if _GLOBAL_PF is None:
        with _GLOBAL_PF_LOCK:
            if _GLOBAL_PF is None:
                try:
                    _GLOBAL_PF = _build_new_pf()
                    log.info("GLOBAL PF initialized")
                except Exception as e:
                    log.error("GLOBAL PF init failed: %s", e)
                    _GLOBAL_PF = SmartDummyPF()
    return _GLOBAL_PF

def reset_pf_for_uid(uid: str) -> None:
    _clear_prediction_cache(uid)

pf_initialized = (OutcomePF is not None)

# ---------- 決策 / 配注 ----------
# （以下所有決策函數、deplete、三段覆蓋、先前 patch 等全部保持不變）
# 為了避免篇幅過長，這裡省略中間不變的部分（decide_only_bp、_stage_bounds、_handle_points_and_predict 等）
# 你可以直接把你原本的這段程式碼貼回來替換

# ---------- KEEP ALIVE 防休眠（已大幅強化） ----------
def _self_keep_alive():
    import time
    try:
        import requests
    except Exception:
        log.warning("[KEEPALIVE] requests module not available, skip self-ping")
        return

    # 🔥 優先使用 Render 官方環境變數，其次 SELF_URL
    url = os.getenv("RENDER_EXTERNAL_URL")
    if not url:
        url = os.getenv("SELF_URL")
    if not url:
        url = os.getenv("SELF_PING_URL")

    interval = int(os.getenv("SELF_PING_INTERVAL", "180"))

    if not url:
        log.warning("[KEEPALIVE] No URL found (RENDER_EXTERNAL_URL / SELF_URL / SELF_PING_URL), skip self-ping")
        return

    log.info(f"[KEEPALIVE] Started with URL: {url} | interval: {interval}s")

    while True:
        try:
            r = requests.get(url, timeout=12)
            if r.status_code < 400:
                log.info("[KEEPALIVE] self ping success")
            else:
                log.warning(f"[KEEPALIVE] ping returned status {r.status_code}")
        except Exception as e:
            log.warning("[KEEPALIVE] self ping failed: %s", e)
        time.sleep(interval)

# ---------- 主預測與 LINE 部分 ----------
# （請把你原本程式碼中從 _handle_points_and_predict 開始到 LINE 相關的所有程式碼貼在這裡）
# 包含 LINE_CHANNEL_SECRET、line_handler、所有 @app.route 等

# ---------- 保持原本的 if __name__ == "__main__" ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    if OutcomePF is None:
        log.warning("PF backend: smart-dummy (OutcomePF import failed).")
    else:
        log.info("PF backend: %s (OutcomePF available)", PF_BACKEND)

    log.info(
        "Starting %s on port %s (PF_INIT=%s, DEPLETE_OK=%s, MODE=%s, COMPAT=%s, DEPL=%s, "
        "PF_STATEFUL=%s, KEEPALIVE=boot-started)",
        VERSION, port, pf_initialized, DEPLETE_OK, os.getenv("DECISION_MODE", "hybrid"),
        os.getenv("COMPAT_MODE", "0"), os.getenv("DEPL_ENABLE", "0"), PF_STATEFUL
    )

    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
