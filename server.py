# -*- coding: utf-8 -*-
"""
server.py — BGS Pure PF + Deplete + Probability Only + FULL LINE Flow + Stability + Stat Calibrator

This version provides:
1. Per-user particle filter instances.
2. Per-user locking to avoid cross-user contamination.
3. Negative / positive PROB_BIAS_B2P support.
4. Graceful handling for multiple probs_after_points signatures.
5. Optional stat calibrator layer:
   bgs/stat_calibrator.py + bgs/calibrator_stats.json

Stat calibrator position:
PF predict
→ soft tau
→ deplete
→ theo blend
→ tie cap
→ PROB_BIAS_B2P
→ STAT_CALIBRATOR.adjust(...)
→ advanced control
→ probability-only decision
"""

import os
import sys
import logging
import time
import re
import json
import threading
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

#
# ---------- path fix ----------
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BGS_DIR = os.path.join(BASE_DIR, "bgs")
if BGS_DIR not in sys.path:
    sys.path.insert(0, BGS_DIR)


def env_flag(name: str, default: int = 1) -> int:
    """Return 1 or 0 based on a truthy environment flag."""
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

#
# ---------- deplete ----------
#
DEPLETE_OK = False
init_counts = None
probs_after_points = None
try:
    from bgs.deplete import init_counts, probs_after_points  # type: ignore
    DEPLETE_OK = True
except Exception:
    try:
        from deplete import init_counts, probs_after_points  # type: ignore
        DEPLETE_OK = True
    except Exception as e:
        log.warning("deplete import failed: %r", e)
        DEPLETE_OK = False

#
# ---------- stat calibrator ----------
#
STAT_CALIBRATOR_OK = False
StatCalibrator = None
STAT_CALIBRATOR = None

try:
    from bgs.stat_calibrator import StatCalibrator as RealStatCalibrator  # type: ignore
    StatCalibrator = RealStatCalibrator
    STAT_CALIBRATOR = StatCalibrator()
    STAT_CALIBRATOR_OK = True
    log.info("成功從 bgs.stat_calibrator 導入 StatCalibrator")
except Exception as e1:
    try:
        from stat_calibrator import StatCalibrator as LocalStatCalibrator  # type: ignore
        StatCalibrator = LocalStatCalibrator
        STAT_CALIBRATOR = StatCalibrator()
        STAT_CALIBRATOR_OK = True
        log.info("成功從本地 stat_calibrator 導入 StatCalibrator")
    except Exception as e2:
        log.warning("StatCalibrator import failed: bgs=%r | local=%r", e1, e2)
        StatCalibrator = None
        STAT_CALIBRATOR = None
        STAT_CALIBRATOR_OK = False

#
# ---------- Flask ----------
#
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    _flask_available = True
except Exception:
    _flask_available = False
    Flask = None
    request = None

    def jsonify(*args, **kwargs):  # type: ignore
        raise RuntimeError("Flask not available")

    def CORS(app):  # type: ignore
        return None

#
# ---------- Redis ----------
#
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
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1800"))
DEDUPE_TTL = 60


def _rget(k: str) -> Optional[str]:
    """Retrieve a string value from Redis or fallback store."""
    try:
        if redis_client:
            return redis_client.get(k)  # type: ignore[no-any-return]
        return KV_FALLBACK.get(k)
    except Exception as e:
        log.warning("[Redis] GET err: %s", e)
        return None


def _rset(k: str, v: str, ex: Optional[int] = None) -> None:
    """Set a string value in Redis or fallback store."""
    try:
        if redis_client:
            redis_client.set(k, v, ex=ex)  # type: ignore[call-arg]
        else:
            KV_FALLBACK[k] = v
    except Exception as e:
        log.warning("[Redis] SET err: %s", e)


def _rsetnx(k: str, v: str, ex: int) -> bool:
    """Set a string value only if the key does not exist."""
    try:
        if redis_client:
            return bool(redis_client.set(k, v, ex=ex, nx=True))  # type: ignore[call-arg]
        if k in KV_FALLBACK:
            return False
        KV_FALLBACK[k] = v
        return True
    except Exception as e:
        log.warning("[Redis] SETNX err: %s", e)
        return True


def _dedupe_event(event_id: Optional[str]) -> bool:
    """
    Use a short-lived key to deduplicate events. Returns True if the event
    should be processed, or False if it has been seen recently.
    """
    if not event_id:
        return True
    return _rsetnx(f"dedupe:{event_id}", "1", ex=DEDUPE_TTL)


def _extract_line_event_id(event: Any) -> Optional[str]:
    """Extract a unique event ID from a LINE webhook event for deduplication."""
    for attr in ("webhook_event_id", "id"):
        try:
            v = getattr(event, attr, None)
            if v:
                return str(v)
        except Exception:
            pass
    try:
        msg = getattr(event, "message", None)
        mid = getattr(msg, "id", None) if msg is not None else None
        if mid:
            return str(mid)
    except Exception:
        pass
    return None


def _premium_key(uid: str) -> str:
    return f"premium:{uid}"


def is_premium(uid: str) -> bool:
    return bool(uid) and _rget(_premium_key(uid)) == "1"


def set_premium(uid: str, flag: bool = True) -> None:
    if uid:
        _rset(_premium_key(uid), "1" if flag else "0")


def _sess_key(uid: str) -> str:
    return f"sess:{uid}"


def _ensure_session_defaults(sess: Dict[str, Any], uid: str) -> Dict[str, Any]:
    """Ensure old Redis sessions also have new fields."""
    if is_premium(uid):
        sess["premium"] = True

    sess.setdefault("phase", "await_pts")
    sess.setdefault("bankroll", 0)
    sess.setdefault("rounds_seen", 0)
    sess.setdefault("last_pts_text", None)
    sess.setdefault("premium", is_premium(uid))
    sess.setdefault("trial_start", int(time.time()))
    sess.setdefault("last_card", None)
    sess.setdefault("last_card_ts", None)
    sess.setdefault("pending", False)
    sess.setdefault("pending_seq", 0)
    sess.setdefault("loss_streak", 0)
    sess.setdefault("adv_history", [])
    sess.setdefault("last_choice", None)

    # 新增：給百萬局統計校準層使用
    sess.setdefault("outcome_streak_side", None)
    sess.setdefault("outcome_streak_len", 0)
    sess.setdefault("last_result", None)
    sess.setdefault("last_banker_point", None)
    sess.setdefault("last_player_point", None)

    return sess


def get_session(uid: str) -> Dict[str, Any]:
    """
    Retrieve the session dictionary for a given user. If no session exists,
    initialize a default session. Sessions are stored in Redis if available
    or in an in-memory fallback.
    """
    uid = uid or "anon"
    try:
        if redis_client:
            raw = redis_client.get(_sess_key(uid))  # type: ignore[call-arg]
            if raw:
                sess = json.loads(raw)
                return _ensure_session_defaults(sess, uid)

        sess = SESS_FALLBACK.get(uid)
        if isinstance(sess, dict):
            return _ensure_session_defaults(sess, uid)

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
        "outcome_streak_side": None,
        "outcome_streak_len": 0,
        "last_result": None,
        "last_banker_point": None,
        "last_player_point": None,
    }
    save_session(uid, sess)
    return sess


def save_session(uid: str, sess: Dict[str, Any]) -> None:
    """Persist a session dictionary for the given user."""
    uid = uid or "anon"
    try:
        payload = json.dumps(sess, ensure_ascii=False)
        if redis_client:
            redis_client.set(_sess_key(uid), payload, ex=SESSION_EXPIRE_SECONDS)  # type: ignore[call-arg]
        else:
            SESS_FALLBACK[uid] = sess
    except Exception as e:
        log.warning("save_session error: %s", e)


def format_output_card(probs: np.ndarray, choice: str, last_pts: Optional[str], bet_amt: int,
                       cont: bool = True, mode: str = "") -> str:
    """
    Format the prediction and bet suggestion into a human-readable message.
    """
    pB, pP, pT = [float(x) for x in probs]
    lines = []
    if last_pts:
        lines.append(str(last_pts))
    lines.append(f"機率｜莊 {pB*100:.2f}%｜閒 {pP*100:.2f}%｜和 {pT*100:.2f}%")
    lines.append(f"差距｜莊閒 {abs(pB - pP) * 100:.2f}%")
    if mode:
        lines.append(f"模式｜{mode}")
    lines.append("建議：觀望 👀" if choice == "觀望" else f"建議：下 {choice} 🎯")
    if bet_amt and bet_amt > 0:
        lines.append(f"配注：{bet_amt}")
    if cont:
        lines.append("\n（輸入下一局點數：例如 65 / 和 / 閒6莊5）")
    return "\n".join(lines)


VERSION = "bgs-prob-only-pf360-deplete-line-2026-05-08"

#
# ---------- Flask App ----------
#
if _flask_available and Flask is not None:
    app = Flask(__name__)
    CORS(app)
else:
    class _DummyApp:
        def get(self, *a, **k):  # type: ignore
            def _d(f):
                return f
            return _d

        def post(self, *a, **k):  # type: ignore
            def _d(f):
                return f
            return _d

        def route(self, *a, **k):  # type: ignore
            def _d(f):
                return f
            return _d

        def run(self, *a, **k):
            log.warning("Flask not available; cannot run HTTP server.")

    app = _DummyApp()


#
# ---------- PF ----------
#
PF_BACKEND = os.getenv("PF_BACKEND", "mc").lower()
SKIP_TIE_UPD = env_flag("SKIP_TIE_UPD", 1)
SOFT_TAU = float(os.getenv("SOFT_TAU", "2.0"))
TIE_MIN = float(os.getenv("TIE_MIN", "0.05"))
TIE_MAX = float(os.getenv("TIE_MAX", "0.15"))
TIE_CAP_ENABLE = env_flag("TIE_CAP_ENABLE", 1)
SHOW_RAW_PROBS = env_flag("SHOW_RAW_PROBS", 0)
PF_STATEFUL = env_flag("PF_STATEFUL", 1)

OutcomePF = None
pf_initialized = False

try:
    from bgs.pfilter import OutcomePF as RealOutcomePF
    OutcomePF = RealOutcomePF
    pf_initialized = True
    log.info("成功從 bgs.pfilter 導入 OutcomePF")
except Exception as e1:
    try:
        from pfilter import OutcomePF as LocalOutcomePF
        OutcomePF = LocalOutcomePF
        pf_initialized = True
        log.info("成功從本地 pfilter 導入 OutcomePF")
    except Exception as e2:
        log.error("無法導入 OutcomePF: bgs=%r | local=%r", e1, e2)
        OutcomePF = None
        pf_initialized = False


class SmartDummyPF:
    """
    A fallback particle filter implementation used when the real PF backend is unavailable.
    """
    def update_outcome(self, outcome) -> None:
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
    def backend(self) -> str:
        return "smart-dummy"


#
# Per-user particle filter storage.
#
_PF_BY_UID: Dict[str, Any] = {}
_PF_BY_UID_LOCK = threading.Lock()

#
# Caches for prediction results and locks
#
KEEP_ALIVE_STARTED = False
KEEP_ALIVE_LOCK = threading.Lock()
_RESULT_CACHE: Dict[str, Tuple[np.ndarray, str, int, str]] = {}
_RESULT_CACHE_KEY: Dict[str, str] = {}
_RESULT_CACHE_LOCK = threading.Lock()
_UID_LOCKS: Dict[str, threading.Lock] = {}
_UID_LOCKS_GUARD = threading.Lock()


def _clear_prediction_cache(uid: str) -> None:
    """Remove any cached prediction for a specific user."""
    uid = uid or "anon"
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE.pop(uid, None)
        _RESULT_CACHE_KEY.pop(uid, None)


def _make_result_cache_key(uid: str, sess: Dict[str, Any], p_pts: int, b_pts: int) -> str:
    """
    Construct a key used for caching predictions.
    """
    return (
        f"{uid}|r={int(sess.get('rounds_seen', 0))}|p={p_pts}|b={b_pts}|"
        f"bank={int(sess.get('bankroll', 0))}|stateful={int(PF_STATEFUL)}|"
        f"mode=probability_only|"
        f"stat={int(STAT_CALIBRATOR_OK)}|"
        f"stat_enable={os.getenv('STAT_CALIBRATOR_ENABLE', '1')}|"
        f"stat_blend={os.getenv('STAT_CALIBRATOR_BLEND', '0.16')}"
    )


def _get_uid_lock(uid: str) -> threading.Lock:
    """
    Retrieve a per-user threading lock.
    """
    uid = uid or "anon"
    with _UID_LOCKS_GUARD:
        if uid not in _UID_LOCKS:
            _UID_LOCKS[uid] = threading.Lock()
        return _UID_LOCKS[uid]


def _self_keep_alive() -> None:
    """
    Periodically ping the service itself to prevent idling.
    """
    try:
        import requests  # type: ignore
    except Exception:
        log.warning("[KEEPALIVE] requests not available, skip self-ping")
        return
    url = os.getenv("SELF_PING_URL")
    interval = int(os.getenv("SELF_PING_INTERVAL", "240"))
    if not url:
        log.warning("[KEEPALIVE] SELF_PING_URL not set, skip self-ping")
        return
    while True:
        try:
            requests.get(url, timeout=10)
            log.info("[KEEPALIVE] self ping success")
        except Exception as e:
            log.warning("[KEEPALIVE] self ping failed: %s", e)
        time.sleep(interval)


def _build_new_pf() -> Any:
    """
    Create a new particle filter instance using environment parameters.
    """
    global KEEP_ALIVE_STARTED
    if not KEEP_ALIVE_STARTED:
        with KEEP_ALIVE_LOCK:
            if not KEEP_ALIVE_STARTED:
                try:
                    threading.Thread(target=_self_keep_alive, daemon=True).start()
                    KEEP_ALIVE_STARTED = True
                    log.info("KEEP ALIVE thread started (ONLY ONCE)")
                except Exception as e:
                    log.warning("KEEP ALIVE failed: %s", e)

    if OutcomePF is None:
        return SmartDummyPF()

    return OutcomePF(
        decks=int(os.getenv("DECKS", "8")),
        seed=int(os.getenv("SEED", "42")),
        n_particles=int(os.getenv("PF_N", "360")),
        sims_lik=int(os.getenv("PF_UPD_SIMS", "30")),
        resample_thr=float(os.getenv("PF_RESAMPLE", "0.5")),
        dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.05")),
        backend=PF_BACKEND,
    )


def get_pf_for_uid(uid: str) -> Any:
    """
    Retrieve a particle filter instance specific to the given user.
    """
    uid = uid or "anon"
    with _PF_BY_UID_LOCK:
        pf_obj = _PF_BY_UID.get(uid)
        if pf_obj is None:
            try:
                pf_obj = _build_new_pf()
                log.info("PF initialized for uid=%s", uid)
            except Exception as e:
                log.error("PF init failed for uid=%s: %s", uid, e)
                pf_obj = SmartDummyPF()
            _PF_BY_UID[uid] = pf_obj
        return pf_obj


def reset_pf_for_uid(uid: str) -> None:
    """
    Clear prediction cache and PF state for the given user.
    """
    uid = uid or "anon"
    _clear_prediction_cache(uid)
    with _PF_BY_UID_LOCK:
        _PF_BY_UID.pop(uid, None)


#
# ---------- Probability-only decision settings ----------
#
# EV / hybrid decisions are intentionally removed.
# Final choice is based only on Banker vs Player probability gap and confidence.
DECISION_MODE = "probability_only"
PROB_MARGIN = float(os.getenv("PROB_MARGIN", "0.018"))
MIN_CONF_FOR_ENTRY = float(os.getenv("MIN_CONF_FOR_ENTRY", "0.500"))
MIN_BET_PCT_ENV = float(os.getenv("MIN_BET_PCT", "0.05"))
MAX_BET_PCT_ENV = float(os.getenv("MAX_BET_PCT", "0.40"))
MAX_EDGE_SCALE = float(os.getenv("MAX_EDGE_FOR_FULLBET", "0.15"))
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)
SHOW_CONF_DEBUG = env_flag("SHOW_CONF_DEBUG", 1)
LOG_DECISION = env_flag("LOG_DECISION", 1)
INV = {0: "莊", 1: "閒"}
COMPAT_MODE = int(os.getenv("COMPAT_MODE", "0"))
DEPL_ENABLE = int(os.getenv("DEPL_ENABLE", "0"))
DEPL_FACTOR = float(os.getenv("DEPL_FACTOR", "0.35"))
EARLY_DEPL_SCALE = float(os.getenv("EARLY_DEPL_SCALE", "0.2"))
MID_DEPL_SCALE = float(os.getenv("MID_DEPL_SCALE", "0.6"))
LATE_DEPL_SCALE = float(os.getenv("LATE_DEPL_SCALE", "0.9"))
MAX_DEPL_SHIFT = float(os.getenv("MAX_DEPL_SHIFT", "0.03"))
PROB_BIAS_B2P = float(os.getenv("PROB_BIAS_B2P", "0.0"))
THEO_BLEND_FORCE_DISABLE = env_flag("THEO_BLEND_FORCE_DISABLE", 1)


def _current_decision_mode(over: Dict[str, float]) -> str:
    """Return fixed decision mode. EV and hybrid are disabled by design."""
    return "probability_only"


def _env_float(name: str, default: float, over: Optional[Dict[str, float]] = None) -> float:
    """Read float from stage override first, then env, then default."""
    if over and name in over:
        try:
            return float(over[name])
        except Exception:
            pass
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def bet_amount(bankroll: int, pct: float) -> int:
    """Compute a discrete bet amount based on bankroll and a percentage."""
    if bankroll <= 0 or pct <= 0:
        return 0
    return int(round(bankroll * pct))


def _apply_prob_bias(prob: np.ndarray, over: Dict[str, float]) -> np.ndarray:
    """
    Apply configurable bias between Banker and Player probabilities.

    Positive values shift probability mass from Banker to Player.
    Negative values shift probability mass from Player to Banker.
    """
    b2p = PROB_BIAS_B2P
    try:
        if "PROB_BIAS_B2P" in over:
            b2p = float(over["PROB_BIAS_B2P"])
    except Exception:
        pass

    try:
        b2p = float(b2p)
    except Exception:
        b2p = 0.0

    if b2p == 0.0:
        return prob

    p = prob.copy()

    if b2p > 0.0:
        shift = min(float(p[0]), b2p)
        if shift > 0:
            p[0] -= shift
            rem_bp = max(1e-8, 1.0 - float(p[2]))
            p[1] = min(rem_bp, float(p[1]) + shift)
    else:
        shift = min(float(p[1]), -b2p)
        if shift > 0:
            p[1] -= shift
            rem_bp = max(1e-8, 1.0 - float(p[2]))
            p[0] = min(rem_bp, float(p[0]) + shift)

    s = p.sum()
    if s > 0:
        p /= s

    return p


def decide_only_bp(prob: np.ndarray, over: Dict[str, float], effective_edge_enter: float,
                   p_pts: int, b_pts: int) -> Tuple[str, float, float, str]:
    """
    Probability-only decision.

    Removed completely:
    - EV decision
    - hybrid decision
    - payout-aware Banker 0.95 decision

    Active rule:
    1. Normalize [Banker, Player, Tie].
    2. Compare Banker vs Player only.
    3. Enter only when:
       - max(Banker, Player) >= MIN_CONF_FOR_ENTRY
       - abs(Banker - Player) >= PROB_MARGIN / stage PROB_MARGIN
       - abs(Banker - Player) >= effective_edge_enter from advanced control
    """
    p = np.asarray(prob, dtype=np.float32).copy()
    if p.shape != (3,) or (not np.isfinite(p).all()) or float(p.sum()) <= 0:
        return ("觀望", 0.0, 0.0, "純機率：機率資料異常，建議觀望")

    p = np.maximum(p, 0.0)
    p = p / max(float(p.sum()), 1e-8)

    pB, pP, pT = float(p[0]), float(p[1]), float(p[2])
    edge = abs(pB - pP)
    confidence = max(pB, pP)

    prob_margin = max(0.0, _env_float("PROB_MARGIN", PROB_MARGIN, over))
    min_conf = max(0.0, min(1.0, _env_float("MIN_CONF_FOR_ENTRY", MIN_CONF_FOR_ENTRY, over)))
    dyn_edge = max(0.0, float(effective_edge_enter or 0.0))
    enter_edge = max(prob_margin, dyn_edge)

    reason: List[str] = [
        "模式=純機率",
        f"信心={confidence:.4f}",
        f"莊閒差距={edge:.4f}",
        f"進場差距門檻={enter_edge:.4f}",
        f"信心門檻={min_conf:.4f}",
    ]

    if LOG_DECISION or SHOW_CONF_DEBUG:
        log.info(
            "[DECIDE-PROB-ONLY] pB=%.4f pP=%.4f pT=%.4f edge=%.4f conf=%.4f enter_edge=%.4f min_conf=%.4f",
            pB, pP, pT, edge, confidence, enter_edge, min_conf,
        )

    if confidence < min_conf:
        reason.append(f"⚪ 信心不足 {confidence:.4f}<{min_conf:.4f}")
        return ("觀望", edge, 0.0, "; ".join(reason))

    if edge < enter_edge:
        reason.append(f"⚪ 莊閒差距不足 {edge:.4f}<{enter_edge:.4f}")
        return ("觀望", edge, 0.0, "; ".join(reason))

    side = 0 if pB >= pP else 1
    min_b = max(0.0, min(1.0, MIN_BET_PCT_ENV))
    max_b = max(min_b, min(1.0, MAX_BET_PCT_ENV))
    scale = max(1e-6, float(MAX_EDGE_SCALE))
    bet_pct = min_b + (max_b - min_b) * (edge / scale)
    bet_pct = float(min(max_b, max(min_b, bet_pct)))

    side_label = INV.get(side, "莊")
    reason.append(f"🔻 {side_label} 機率={100.0 * (pB if side == 0 else pP):.1f}%")

    return (("莊" if side == 0 else "閒"), edge, bet_pct, "; ".join(reason))


def _stage_bounds() -> Tuple[int, int]:
    """Return the hand count boundaries for early and mid stages."""
    return int(os.getenv("EARLY_HANDS", "20")), int(os.getenv("MID_HANDS", os.getenv("LATE_HANDS", "56")))


def _stage_prefix(rounds_seen: int) -> str:
    """Return the stage prefix based on the number of rounds seen."""
    e_end, m_end = _stage_bounds()
    if rounds_seen < e_end:
        return "EARLY_"
    if rounds_seen < m_end:
        return "MID_"
    return "LATE_"


def get_stage_over(rounds_seen: int) -> Dict[str, float]:
    """
    Retrieve stage-specific environment overrides.
    """
    if COMPAT_MODE == 1 or os.getenv("STAGE_MODE", "count").lower() == "disabled":
        return {}

    over: Dict[str, float] = {}
    prefix = _stage_prefix(rounds_seen)

    keys = [
        "SOFT_TAU",
        "THEO_BLEND",
        "TIE_MAX",
        "EDGE_ENTER",
        "PROB_MARGIN",
        "MIN_CONF_FOR_ENTRY",
        "PF_PRED_SIMS",
        "DEPLETEMC_SIMS",
        "PF_UPD_SIMS",
        "PROB_BIAS_B2P",
    ]

    for k in keys:
        v = os.getenv(prefix + k)
        if v not in (None, ""):
            try:
                over[k] = float(v)
            except Exception:
                pass

    return over


def _depl_stage_scale(rounds_seen: int) -> float:
    """Return depletion scaling factor based on current stage."""
    prefix = _stage_prefix(rounds_seen)
    return EARLY_DEPL_SCALE if prefix == "EARLY_" else MID_DEPL_SCALE if prefix == "MID_" else LATE_DEPL_SCALE


def _guard_shift(old_p: np.ndarray, new_p: np.ndarray, max_shift: float) -> np.ndarray:
    """
    Blend between two probability distributions while limiting max shift per component.
    """
    p_old = old_p.astype(float).copy()
    delta = np.clip(new_p.astype(float) - p_old, -max_shift, max_shift)
    p_safe = p_old + delta
    s = float(p_safe.sum())
    if s > 0:
        p_safe /= s
    return p_safe.astype(np.float32)


def _tuned_pred_sims(base: int, pf_obj: Any) -> int:
    """
    Adjust number of prediction simulations based on caps and PF particles.
    """
    try:
        cap = int(float(os.getenv("PRED_SIMS_CAP", "95")))
    except Exception:
        cap = 95

    n = max(1, min(int(base), int(cap)))

    try:
        n_particles = int(getattr(pf_obj, "n_particles", 0) or 0)
        if n_particles >= 360:
            n = min(n, int(os.getenv("PRED_GUARD_360_CAP", "22")))
        elif n_particles >= 350:
            n = min(n, int(os.getenv("PRED_GUARD_350_CAP", "25")))
        elif n_particles >= 300:
            n = min(n, int(os.getenv("PRED_GUARD_300_CAP", "35")))
    except Exception:
        pass

    return max(1, int(n))


def _safe_predict_probs(pf_obj: Any, sims_used: int, uid: str) -> Tuple[np.ndarray, int]:
    """
    Safely invoke PF.predict.
    """
    try:
        p = np.asarray(pf_obj.predict(sims_per_particle=int(sims_used)), dtype=np.float32)

        if p.shape != (3,) or (not np.isfinite(p).all()) or float(p.sum()) <= 0:
            raise ValueError(f"invalid probs shape={getattr(p, 'shape', None)} sum={float(np.sum(p))}")

        p = p / p.sum()
        return p.astype(np.float32), int(sims_used)

    except Exception as e:
        log.exception("PF.predict failed(uid=%s): %s", uid, e)
        fallback = SmartDummyPF().predict()
        fallback = fallback / fallback.sum()
        return fallback.astype(np.float32), int(sims_used)


def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
    """
    Parse user input representing last hand points.
    Returns tuple: (Player points, Banker points)
    """
    if not text:
        return None

    s = str(text).translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s).replace("\u3000", " ")
    u = s.upper().strip()

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

    if t in ("B", "莊", "庄"):
        return (0, 1)

    if t in ("P", "閒", "闲"):
        return (1, 0)

    if t in ("T", "和"):
        return (0, 0)

    if re.search(r"[A-Z]", u):
        return None

    d = re.findall(r"\d", u)
    if len(d) == 2:
        return (int(d[0]), int(d[1]))

    return None


def _result_from_points(p_pts: int, b_pts: int) -> str:
    """
    Convert points to result:
    B = banker win
    P = player win
    T = tie
    """
    if b_pts > p_pts:
        return "B"
    if p_pts > b_pts:
        return "P"
    return "T"


def _update_outcome_context(sess: Dict[str, Any], p_pts: int, b_pts: int) -> Dict[str, Any]:
    """
    Update session outcome context for stat calibrator.
    This context represents the latest finished hand and will be used to predict next hand.
    """
    result = _result_from_points(p_pts, b_pts)

    prev_side = sess.get("outcome_streak_side")
    prev_len = int(sess.get("outcome_streak_len", 0) or 0)

    # 和局通常不打斷莊閒 streak
    if result in ("B", "P"):
        if prev_side == result:
            streak_side = result
            streak_len = prev_len + 1
        else:
            streak_side = result
            streak_len = 1
        sess["outcome_streak_side"] = streak_side
        sess["outcome_streak_len"] = streak_len
    else:
        streak_side = prev_side
        streak_len = prev_len

    sess["last_result"] = result
    sess["last_banker_point"] = b_pts
    sess["last_player_point"] = p_pts

    return {
        "last_result": result,
        "banker_point": b_pts,
        "player_point": p_pts,
        "streak_side": streak_side,
        "streak_len": streak_len,
    }


def _apply_stat_calibrator(
    p: np.ndarray,
    sess: Dict[str, Any],
    rounds_seen: int,
    p_pts: int,
    b_pts: int,
) -> Tuple[np.ndarray, str]:
    """
    Apply stat calibrator after PF/deplete/bias and before final decision.
    Returns adjusted probability and reason note.
    """
    if not STAT_CALIBRATOR_OK or STAT_CALIBRATOR is None:
        return p, "StatCalibrator=OFF"

    try:
        outcome_ctx = _update_outcome_context(sess, p_pts, b_pts)

        total_round_est = int(os.getenv("TOTAL_ROUND_EST", "70"))
        shoe_pos = float(rounds_seen + 1) / max(float(total_round_est), 1.0)
        shoe_pos = max(0.0, min(1.0, shoe_pos))

        context = {
            "shoe_pos": shoe_pos,
            "round_index": rounds_seen + 1,
            "total_round_est": total_round_est,
            "last_result": outcome_ctx.get("last_result"),
            "banker_point": outcome_ctx.get("banker_point"),
            "player_point": outcome_ctx.get("player_point"),
            "streak_side": outcome_ctx.get("streak_side"),
            "streak_len": outcome_ctx.get("streak_len"),
        }

        base_dict = {
            "banker": float(p[0]),
            "player": float(p[1]),
            "tie": float(p[2]),
        }

        adjusted = STAT_CALIBRATOR.adjust(base_dict, context)

        p2 = np.array([
            float(adjusted.get("banker", p[0])),
            float(adjusted.get("player", p[1])),
            float(adjusted.get("tie", p[2])),
        ], dtype=np.float32)

        if p2.shape != (3,) or (not np.isfinite(p2).all()) or float(p2.sum()) <= 0:
            raise ValueError(f"invalid calibrated probs={p2}")

        p2 = p2 / p2.sum()

        # 再做一次 tie 安全線，避免校準後破壞原本 tie 限制
        tie_max = float(os.getenv("TIE_MAX", str(TIE_MAX)))
        if TIE_CAP_ENABLE == 1 and p2[2] > tie_max:
            sc = (1.0 - tie_max) / (1.0 - float(p2[2])) if p2[2] < 1.0 else 1.0
            p2[2] = tie_max
            p2[0] *= sc
            p2[1] *= sc
            p2 = p2 / p2.sum()

        if p2[2] < TIE_MIN:
            sc = (1.0 - TIE_MIN) / (1.0 - float(p2[2])) if p2[2] < 1.0 else 1.0
            p2[2] = TIE_MIN
            p2[0] *= sc
            p2[1] *= sc
            p2 = p2 / p2.sum()

        if LOG_DECISION or SHOW_CONF_DEBUG:
            log.info(
                "[STAT-CAL] before=(%.4f,%.4f,%.4f) after=(%.4f,%.4f,%.4f) ctx=%s",
                float(p[0]), float(p[1]), float(p[2]),
                float(p2[0]), float(p2[1]), float(p2[2]),
                context,
            )

        return p2.astype(np.float32), "StatCalibrator=ON"

    except Exception as e:
        log.warning("StatCalibrator failed: %s", e)
        return p, "StatCalibrator=ERR"


def _advanced_control(sess: Dict[str, Any], probs: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """
    Apply advanced control logic.
    """
    history = sess.get("adv_history", [])
    history.append(int(np.argmax(probs[:2])))
    if len(history) > 20:
        history.pop(0)

    sess["adv_history"] = history

    probs2 = np.array([float(probs[0]), float(probs[1]), float(probs[2])], dtype=np.float32)
    probs2 = probs2 / probs2.sum()

    return probs2, 0.002, "正常"


def _handle_points_and_predict(uid: str, sess: Dict[str, Any], p_pts: int, b_pts: int) -> Tuple[np.ndarray, str, int, str]:
    """
    Handle a single prediction request for a given user.
    """
    uid = uid or "anon"
    lk = _get_uid_lock(uid)

    with lk:
        rounds_seen = int(sess.get("rounds_seen", 0))
        over = get_stage_over(rounds_seen)
        mode = "probability_only"
        sess["decision_mode"] = mode

        cache_key = _make_result_cache_key(uid, sess, p_pts, b_pts)

        with _RESULT_CACHE_LOCK:
            if _RESULT_CACHE_KEY.get(uid) == cache_key:
                cached = _RESULT_CACHE.get(uid)
                if cached:
                    return cached[0].copy(), cached[1], int(cached[2]), str(cached[3])

        sims_base = int(over.get("PF_PRED_SIMS", float(os.getenv("PF_PRED_SIMS", "5"))))
        sims_used = sims_base

        if PF_STATEFUL == 1:
            pf_obj = get_pf_for_uid(uid)

            try:
                if hasattr(pf_obj, "update_outcome"):
                    if p_pts == b_pts:
                        if not SKIP_TIE_UPD:
                            try:
                                pf_obj.update_outcome(2)
                            except Exception:
                                pf_obj.update_outcome("T")
                    else:
                        outcome = 0 if b_pts > p_pts else 1
                        try:
                            pf_obj.update_outcome(outcome)
                        except Exception:
                            pf_obj.update_outcome("B" if outcome == 0 else "P")
            except Exception as e:
                log.warning("PF.update_outcome failed(uid=%s): %s", uid, e)

            try:
                upd_sims_val = over.get("PF_UPD_SIMS")
                if upd_sims_val is None:
                    upd_sims_val = float(os.getenv("PF_UPD_SIMS", "30"))
                if hasattr(pf_obj, "sims_lik"):
                    pf_obj.sims_lik = int(float(upd_sims_val))
            except Exception as e:
                log.warning("stage PF_UPD_SIMS apply failed(uid=%s): %s", uid, e)

            sims_used = _tuned_pred_sims(sims_base, pf_obj)

            start_time = time.time()
            p, sims_used = _safe_predict_probs(pf_obj, sims_used, uid)

            if time.time() - start_time > 1.2:
                log.warning("⚠️ PF timeout fallback triggered(uid=%s)", uid)
                p = SmartDummyPF().predict()

        else:
            pf_obj = _build_new_pf()
            sims_used = _tuned_pred_sims(sims_base, pf_obj)
            p, sims_used = _safe_predict_probs(pf_obj, sims_used, uid)

        # Soften probabilities
        soft_tau = float(over.get("SOFT_TAU", float(os.getenv("SOFT_TAU", "2.0"))))
        p = p ** (1.0 / max(1e-6, soft_tau))
        p = p / p.sum()

        # Apply deplete adjustment
        if (COMPAT_MODE == 0) and (DEPL_ENABLE == 1) and DEPLETE_OK and probs_after_points:
            try:
                stage_scale = _depl_stage_scale(rounds_seen)
                diff = abs(p_pts - b_pts)
                total = max(p_pts, b_pts)
                base = DEPL_FACTOR * stage_scale

                if diff <= 2:
                    adj = 1.65
                elif diff in (3, 4):
                    adj = 1.38
                elif diff >= 6:
                    adj = 0.85
                else:
                    adj = 1.12

                if total >= 7:
                    adj *= 0.87
                elif total <= 5:
                    adj *= 1.22

                alpha = min(0.40, base * adj)

                if alpha > 0.0:
                    dep_sims = int(over.get("DEPLETEMC_SIMS", 25))
                    dep = None

                    try:
                        # Preferred signature in current deplete.py:
                        # probs_after_points(base_counts, p_pts, b_pts, sims=..., rounds_seen=...)
                        dep = probs_after_points(None, p_pts, b_pts, sims=dep_sims, rounds_seen=rounds_seen)  # type: ignore[misc]
                    except TypeError:
                        try:
                            dep = probs_after_points(None, p_pts, b_pts, sims=dep_sims)  # type: ignore[misc]
                        except TypeError:
                            try:
                                dep = probs_after_points(None, p_pts, b_pts)  # type: ignore[misc]
                            except TypeError:
                                try:
                                    # Backward-compatible old local signatures.
                                    dep = probs_after_points(p_pts, b_pts, sims=dep_sims, rounds_seen=rounds_seen)  # type: ignore[arg-type]
                                except Exception:
                                    dep = None

                    if dep is not None:
                        dep = np.asarray(dep, dtype=np.float32)
                        if dep.sum() > 0:
                            dep = dep / dep.sum()

                    if dep is None or dep.sum() <= 0:
                        dep = p.copy()

                    mix = (1.0 - alpha) * p + alpha * dep
                    mix = mix / mix.sum()
                    p = _guard_shift(p, mix, MAX_DEPL_SHIFT)

            except Exception as e:
                log.warning("Deplete 失敗(uid=%s): %s", uid, e)

        # Optional theoretical blend
        if COMPAT_MODE == 0 and THEO_BLEND_FORCE_DISABLE != 1:
            theo_blend = float(over.get("THEO_BLEND", float(os.getenv("THEO_BLEND", "0.0"))))
            if theo_blend > 0.0:
                theo = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
                p = (1.0 - theo_blend) * p + theo_blend * theo
                p = p / p.sum()

        # Tie cap
        tie_max = float(over.get("TIE_MAX", float(os.getenv("TIE_MAX", str(TIE_MAX)))))
        if TIE_CAP_ENABLE == 1 and p[2] > tie_max:
            sc = (1.0 - tie_max) / (1.0 - float(p[2])) if p[2] < 1.0 else 1.0
            p[2] = tie_max
            p[0] *= sc
            p[1] *= sc
            p = p / p.sum()

        if p[2] < TIE_MIN:
            sc = (1.0 - TIE_MIN) / (1.0 - float(p[2])) if p[2] < 1.0 else 1.0
            p[2] = TIE_MIN
            p[0] *= sc
            p[1] *= sc
            p = p / p.sum()

        # Bias
        p = _apply_prob_bias(p, over)

        if abs(p[0] - p[1]) > 0.20:
            mid = (p[0] + p[1]) / 2
            p[0] = mid + (p[0] - mid) * 0.95
            p[1] = mid + (p[1] - mid) * 0.95
            p = p / p.sum()

        # New: Stat calibrator after PF/deplete/bias
        p, stat_reason = _apply_stat_calibrator(p, sess, rounds_seen, p_pts, b_pts)

        # Advanced control and decision
        p, dynamic_edge, ctrl_reason = _advanced_control(sess, p)
        choice, edge, bet_pct, reason = decide_only_bp(p, over, dynamic_edge, p_pts, b_pts)
        reason = f"{reason} | {ctrl_reason} | {stat_reason}"

        bet_amt = bet_amount(int(sess.get("bankroll", 0)), bet_pct)

        sess["rounds_seen"] = rounds_seen + 1
        sess["last_choice"] = choice

        with _RESULT_CACHE_LOCK:
            _RESULT_CACHE[uid] = (p.copy(), choice, bet_amt, reason)
            _RESULT_CACHE_KEY[uid] = cache_key

        return p, choice, bet_amt, reason


# ---------- LINE / Trial / Session Management ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "aaa8881688")
TRIAL_NAMESPACE = os.getenv("TRIAL_NAMESPACE", "default").strip() or "default"
LINE_PUSH_ENABLE = env_flag("LINE_PUSH_ENABLE", 1)
LINE_PUSH_COOLDOWN_SECONDS = int(os.getenv("LINE_PUSH_COOLDOWN_SECONDS", str(30 * 24 * 3600)))
_PUSH_BLOCK_UNTIL = 0
LINE_ASYNC_HEAVY = env_flag("LINE_ASYNC_HEAVY", 0)


def _can_push() -> bool:
    """Return True if push notifications can be sent based on cooldown."""
    return LINE_PUSH_ENABLE == 1 and int(time.time()) >= int(_PUSH_BLOCK_UNTIL)


def _block_push(reason: str) -> None:
    """Temporarily block push notifications."""
    global _PUSH_BLOCK_UNTIL
    _PUSH_BLOCK_UNTIL = int(time.time()) + int(LINE_PUSH_COOLDOWN_SECONDS)
    log.warning("[LINE] push disabled temporarily: %s (block_until=%s)", reason, _PUSH_BLOCK_UNTIL)


def _looks_like_429(e: Exception) -> bool:
    """Return True if an exception appears to be due to hitting a rate limit."""
    s = str(e)
    return "status_code=429" in s or "reached your monthly limit" in s.lower()


def _trial_key(uid: str, kind: str) -> str:
    return f"trial:{TRIAL_NAMESPACE}:{kind}:{uid}"


def _trial_block_key(uid: str) -> str:
    return _trial_key(uid, "blocked")


def is_trial_blocked(uid: str) -> bool:
    return _rget(_trial_block_key(uid)) == "1"


def set_trial_blocked(uid: str, flag: bool = True) -> None:
    _rset(_trial_block_key(uid), "1" if flag else "0")


def trial_persist_guard(uid: str) -> Optional[str]:
    """
    Check and update trial usage for a given user.
    """
    if is_premium(uid):
        return None

    if is_trial_blocked(uid):
        return f"⛔ 試用已到期（帳號曾被封鎖）\n🔐 如需重新啟用，請輸入：開通 你的密碼\n📞 或聯繫：{ADMIN_CONTACT}"

    now = int(time.time())
    first_ts = _rget(_trial_key(uid, "first_ts"))
    expired = _rget(_trial_key(uid, "expired"))

    if not first_ts:
        _rset(_trial_key(uid, "first_ts"), str(now))
        _rset(_trial_key(uid, "expired"), "0")
        return None

    try:
        first = int(first_ts)
    except Exception:
        first = now
        _rset(_trial_key(uid, "first_ts"), str(now))
        _rset(_trial_key(uid, "expired"), "0")
        return None

    used_min = (now - first) // 60

    if expired == "1" and used_min < TRIAL_MINUTES:
        _rset(_trial_key(uid, "expired"), "0")
        expired = None

    if used_min >= TRIAL_MINUTES:
        _rset(_trial_key(uid, "expired"), "1")
        return f"⏰ 免費試用 {TRIAL_MINUTES} 分鐘已用完\n🔐 請輸入：開通 你的專屬密碼\n📞 沒有密碼？請聯繫：{ADMIN_CONTACT}"

    if expired == "1":
        return f"⛔ 試用已到期\n🔐 請輸入：開通 你的專屬密碼\n📞 沒有密碼？請聯繫：{ADMIN_CONTACT}"

    return None


def validate_activation_code(code: str) -> bool:
    """Validate an admin activation code."""
    if not code:
        return False
    norm = str(code).replace("\u3000", " ").replace("：", ":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and norm == ADMIN_ACTIVATION_SECRET


GAMES = {
    "1": "WM",
    "2": "PM",
    "3": "DG",
    "4": "SA",
    "5": "KU",
    "6": "歐博/卡利",
    "7": "KG",
    "8": "全利",
    "9": "名人",
    "10": "MT真人",
}


def game_menu_text(left_min: int) -> str:
    """Return a menu string listing available games and remaining trial minutes."""
    lines = ["請選擇遊戲館別"]
    for k in sorted(GAMES.keys(), key=lambda x: int(x)):
        lines.append(f"{k}. {GAMES[k]}")
    lines.append("「請直接輸入數字選擇」")
    lines.append(f"⏳ 試用剩餘 {left_min} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
    return "\n".join(lines)


def _quick_buttons():  # type: ignore
    """Generate quick reply buttons for LINE."""
    try:
        from linebot.models import QuickReply, QuickReplyButton, MessageAction  # type: ignore
        return QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="遊戲設定 🎮", text="遊戲設定")),
            QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
            QuickReplyButton(action=MessageAction(label="報莊勝 🅱️", text="B")),
            QuickReplyButton(action=MessageAction(label="報閒勝 🅿️", text="P")),
            QuickReplyButton(action=MessageAction(label="報和局 ⚪", text="T")),
        ])
    except Exception:
        return None


def _reply(api, token: str, text: str) -> None:
    """Send a reply via the LINE API."""
    from linebot.models import TextSendMessage  # type: ignore
    try:
        api.reply_message(token, TextSendMessage(text=text, quick_reply=_quick_buttons()))  # type: ignore
    except Exception as e:
        if "Invalid reply token" in str(e):
            log.info("[LINE] reply skipped (invalid token, likely retry): %s", e)
        else:
            log.warning("[LINE] reply failed: %s", e)


def _push_heavy_prediction(uid: str, p_pts: int, b_pts: int, seq: int) -> None:
    """
    Asynchronous helper to push a prediction result to a LINE user.
    """
    if line_api is None:
        return

    try:
        from linebot.models import TextSendMessage  # type: ignore

        sess = get_session(uid)
        sess["last_pts_text"] = "上局結果: 和局" if (p_pts == b_pts and SKIP_TIE_UPD) else f"上局結果: 閒 {p_pts} 莊 {b_pts}"

        probs, choice, bet_amt, reason = _handle_points_and_predict(uid, sess, p_pts, b_pts)

        msg = format_output_card(
            probs,
            choice,
            sess.get("last_pts_text"),
            bet_amt,
            cont=bool(CONTINUOUS_MODE),
            mode=sess.get("decision_mode", ""),
        )

        if int(sess.get("pending_seq", 0)) == int(seq):
            sess["last_card"] = msg
            sess["last_card_ts"] = int(time.time())
            sess["pending"] = False

        save_session(uid, sess)

        if _can_push():
            try:
                line_api.push_message(uid, TextSendMessage(text=msg, quick_reply=_quick_buttons()))  # type: ignore
            except Exception as e:
                if _looks_like_429(e):
                    _block_push("429 monthly limit reached")
                try:
                    line_api.push_message(uid, TextSendMessage(text=msg))  # type: ignore
                except Exception as e2:
                    log.warning("[LINE] push fallback failed: %s", e2)

    except Exception as e:
        log.exception("[heavy] prediction failed(uid=%s): %s", uid, e)


line_api = None
line_handler = None

try:
    from linebot import LineBotApi, WebhookHandler  # type: ignore
    from linebot.models import MessageEvent, TextMessage, FollowEvent, UnfollowEvent  # type: ignore

    if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)  # type: ignore
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)  # type: ignore
    else:
        log.error("LINE env missing: secret=%s token=%s", bool(LINE_CHANNEL_SECRET), bool(LINE_CHANNEL_ACCESS_TOKEN))

    if line_handler is not None:

        @line_handler.add(UnfollowEvent)
        def on_unfollow(event):  # type: ignore
            """Handle unfollow events."""
            if not _dedupe_event(_extract_line_event_id(event)):
                return
            try:
                uid = event.source.user_id
                set_trial_blocked(uid, True)
                _rset(_trial_key(uid, "expired"), "1")
            except Exception as e:
                log.warning("[TRIAL] unfollow handler error: %s", e)

        @line_handler.add(FollowEvent)
        def on_follow(event):  # type: ignore
            """Handle follow events."""
            if not _dedupe_event(_extract_line_event_id(event)):
                return

            uid = event.source.user_id
            sess = get_session(uid)
            guard_msg = trial_persist_guard(uid)

            try:
                first_ts = _rget(_trial_key(uid, "first_ts")) or str(int(time.time()))
                sess["trial_start"] = int(first_ts)
            except Exception:
                pass

            if sess.get("premium", False) or is_premium(uid):
                msg = "👋 歡迎回來，已是永久開通用戶。\n輸入『遊戲設定』開始。"
            else:
                msg = guard_msg or f"👋 歡迎！你有 {TRIAL_MINUTES} 分鐘免費試用。\n輸入『遊戲設定』開始。"

            _reply(line_api, event.reply_token, msg)
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):  # type: ignore
            """Handle incoming text messages."""
            if not _dedupe_event(_extract_line_event_id(event)):
                return

            uid = event.source.user_id
            text = re.sub(r"\s+", " ", (event.message.text or "").replace("\u3000", " ").strip())
            up = text.upper()
            sess = get_session(uid)

            if up.startswith("開通") or up.startswith("ACTIVATE"):
                after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                ok = validate_activation_code(after)

                if ok:
                    sess["premium"] = True
                    set_premium(uid, True)
                    set_trial_blocked(uid, False)

                _reply(line_api, event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤")
                save_session(uid, sess)
                return

            guard = trial_persist_guard(uid)
            if guard and not sess.get("premium", False):
                _reply(line_api, event.reply_token, guard)
                save_session(uid, sess)
                return

            if up in ("結束分析", "清空", "RESET"):
                premium = sess.get("premium", False) or is_premium(uid)
                start_ts = sess.get("trial_start", int(time.time()))

                sess = {
                    "phase": "await_pts",
                    "bankroll": 0,
                    "rounds_seen": 0,
                    "last_pts_text": None,
                    "premium": premium,
                    "trial_start": start_ts,
                    "last_card": None,
                    "last_card_ts": None,
                    "pending": False,
                    "pending_seq": 0,
                    "loss_streak": 0,
                    "adv_history": [],
                    "last_choice": None,
                    "outcome_streak_side": None,
                    "outcome_streak_len": 0,
                    "last_result": None,
                    "last_banker_point": None,
                    "last_player_point": None,
                }

                reset_pf_for_uid(uid)
                _reply(line_api, event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。")
                save_session(uid, sess)
                return

            hist_match = re.fullmatch(r"[BPTHbpht]{6,30}", text)
            if hist_match:
                sess["rounds_seen"] = len(text)
                save_session(uid, sess)
                _reply(line_api, event.reply_token, "歷史載入完成\n請輸入下一局點數\n例如：65 / 和 / 閒6莊5")
                return

            pts = parse_last_hand_points(text)

            if text == "遊戲設定" or up == "GAME SETTINGS":
                sess["phase"] = "choose_game"
                sess["game"] = None
                sess["table"] = None
                sess["bankroll"] = 0

                first_ts = _rget(_trial_key(uid, "first_ts"))
                left = max(0, TRIAL_MINUTES - ((int(time.time()) - int(first_ts)) // 60)) if first_ts else TRIAL_MINUTES

                _reply(line_api, event.reply_token, game_menu_text(left))
                save_session(uid, sess)
                return

            if sess.get("phase") == "choose_game":
                m = re.match(r"^\s*(\d+)", text)

                if m and m.group(1) in GAMES:
                    sess["game"] = GAMES[m.group(1)]
                    sess["phase"] = "input_bankroll"
                    _reply(line_api, event.reply_token, f"🎰 已選擇：{sess['game']}，請輸入初始籌碼（金額）")
                    save_session(uid, sess)
                    return

                _reply(line_api, event.reply_token, "⚠️ 無效的選項，請輸入上列數字。")
                return

            if sess.get("phase") == "input_bankroll":
                num = re.sub(r"[^\d]", "", text)
                amt = int(num) if num else 0

                if amt <= 0:
                    _reply(line_api, event.reply_token, "⚠️ 請輸入正整數金額。")
                    return

                sess["bankroll"] = amt
                sess["phase"] = "await_pts"

                _reply(
                    line_api,
                    event.reply_token,
                    f"✅ 設定完成！館別：{sess.get('game')}，初始籌碼：{amt}。\n📌 現在輸入第一局點數（例：閒6莊5 / 65 / 和）",
                )
                save_session(uid, sess)
                return

            if pts and sess.get("bankroll", 0) > 0:
                p_pts, b_pts = pts

                if LINE_ASYNC_HEAVY == 1 and _can_push():
                    if sess.get("pending"):
                        _reply(line_api, event.reply_token, "⚠️ 上一局還在計算中，請稍後再輸入下一局。")
                        return

                    _reply(line_api, event.reply_token, "✅ 已收到上一局結果，AI 正在計算。")

                    sess["pending"] = True
                    sess["pending_seq"] = int(sess.get("pending_seq", 0)) + 1
                    seq = int(sess["pending_seq"])
                    sess["last_card"] = None
                    sess["last_card_ts"] = None

                    save_session(uid, sess)

                    threading.Thread(
                        target=_push_heavy_prediction,
                        args=(uid, p_pts, b_pts, seq),
                        daemon=True,
                    ).start()
                    return

                try:
                    sess["last_pts_text"] = "上局結果: 和局" if (p_pts == b_pts and SKIP_TIE_UPD) else f"上局結果: 閒 {p_pts} 莊 {b_pts}"

                    probs, choice, bet_amt, reason = _handle_points_and_predict(uid, sess, p_pts, b_pts)

                    msg = format_output_card(
                        probs,
                        choice,
                        sess.get("last_pts_text"),
                        bet_amt,
                        cont=bool(CONTINUOUS_MODE),
                        mode=sess.get("decision_mode", ""),
                    )

                    sess["last_card"] = msg
                    sess["last_card_ts"] = int(time.time())
                    sess["pending"] = False

                    save_session(uid, sess)
                    _reply(line_api, event.reply_token, msg)

                except Exception as e:
                    log.exception("[LINE] sync predict failed(uid=%s): %s", uid, e)
                    _reply(line_api, event.reply_token, "⚠️ 計算失敗，請稍後再試或輸入下一局點數。")

                return

            _reply(line_api, event.reply_token, "指令無法辨識。\n📌 直接輸入點數（例：65 / 和 / 閒6莊5），或輸入『遊戲設定』。")

except Exception as e:
    log.warning("LINE not fully configured: %s", e)


def _handle_line_webhook():  # type: ignore
    """
    Dispatch incoming webhook events to the LINE handler.
    """
    if "line_handler" not in globals() or line_handler is None:
        log.error("webhook called but LINE handler not ready (missing credentials?)")
        return "OK", 200

    signature = request.headers.get("X-Line-Signature", "")  # type: ignore[attr-defined]
    body = request.get_data(as_text=True)  # type: ignore[attr-defined]

    try:
        line_handler.handle(body, signature)  # type: ignore[call-arg]
    except Exception as e:
        log.error("webhook error: %s", e)
        return "OK", 200

    return "OK", 200


@app.route("/line-webhook", methods=["POST", "GET", "OPTIONS"])
def line_webhook():  # type: ignore
    """
    HTTP endpoint for LINE webhooks.
    """
    if request.method == "OPTIONS":  # type: ignore[attr-defined]
        return ("", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-Line-Signature",
        })

    if request.method == "GET":  # type: ignore[attr-defined]
        return "OK", 200

    return _handle_line_webhook()


@app.route("/callback", methods=["POST", "GET", "OPTIONS"])
def line_webhook_callback():  # type: ignore
    """
    Alternate HTTP endpoint for LINE webhooks.
    """
    if request.method == "OPTIONS":  # type: ignore[attr-defined]
        return ("", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-Line-Signature",
        })

    if request.method == "GET":  # type: ignore[attr-defined]
        return "OK", 200

    return _handle_line_webhook()


@app.get("/")
def root():  # type: ignore
    """
    Health check root endpoint.
    """
    ua = request.headers.get("User-Agent", "") if request else ""  # type: ignore[attr-defined]

    if "UptimeRobot" in ua:
        return "OK", 200

    st = "OK" if pf_initialized else "BACKUP_MODE"
    cal = "CAL_ON" if STAT_CALIBRATOR_OK else "CAL_OFF"

    return f"✅ BGS Server {st} {cal} ({VERSION})", 200


@app.get("/health")
def health():  # type: ignore
    """
    Detailed health check endpoint.
    """
    return jsonify(
        ok=True,
        ts=time.time(),
        version=VERSION,
        pf_initialized=pf_initialized,
        pf_backend=(PF_BACKEND if OutcomePF is not None else "smart-dummy"),
        pf_stateful=bool(PF_STATEFUL),
        line_async_heavy=bool(LINE_ASYNC_HEAVY),
        line_can_push=bool(_can_push()),
        decision_mode=DECISION_MODE,
        prob_margin=PROB_MARGIN,
        min_conf_for_entry=MIN_CONF_FOR_ENTRY,
        webhook_ready=bool(line_handler is not None),
        deplete_ok=bool(DEPLETE_OK),
        stat_calibrator_ok=bool(STAT_CALIBRATOR_OK),
        stat_calibrator_enable=os.getenv("STAT_CALIBRATOR_ENABLE", "1"),
        stat_calibrator_blend=os.getenv("STAT_CALIBRATOR_BLEND", "0.16"),
        stat_calibrator_max_shift=os.getenv("STAT_CALIBRATOR_MAX_SHIFT", "0.018"),
    ), 200


@app.get("/ping")
def ping():  # type: ignore
    """
    Simple ping endpoint.
    """
    return "OK", 200


@app.post("/predict")
def predict():  # type: ignore
    """
    Public API endpoint.
    Expects JSON with:
        uid
        last_text
        bankroll optional
    """
    data: Dict[str, Any] = {}

    try:
        data = request.get_json(force=True) or {}  # type: ignore[attr-defined]

        uid = str(data.get("uid") or "anon")
        last_text = str(data.get("last_text") or "")
        bankroll = data.get("bankroll")

        sess = get_session(uid)

        if isinstance(bankroll, int) and bankroll >= 0:
            sess["bankroll"] = bankroll

        pts = parse_last_hand_points(last_text)

        if not pts:
            return jsonify(ok=False, error="無法解析點數；請輸入 '閒6莊5' / '65' / '和'"), 200  # type: ignore[call-arg]

        p_pts, b_pts = pts

        sess["last_pts_text"] = "上局結果: 和局" if (p_pts == b_pts and SKIP_TIE_UPD) else f"上局結果: 閒 {p_pts} 莊 {b_pts}"

        probs, choice, bet_amt, reason = _handle_points_and_predict(uid, sess, p_pts, b_pts)

        save_session(uid, sess)

        return jsonify(
            ok=True,
            probs=[float(probs[0]), float(probs[1]), float(probs[2])],
            choice=choice,
            bet=bet_amt,
            reason=reason,
            rounds_seen=int(sess.get("rounds_seen", 0)),
            stat_calibrator_ok=bool(STAT_CALIBRATOR_OK),
            card=format_output_card(
                probs,
                choice,
                sess.get("last_pts_text"),
                bet_amt,
                cont=bool(CONTINUOUS_MODE),
                mode=sess.get("decision_mode", ""),
            ),
        ), 200

    except Exception as e:
        log.exception("predict error(uid=%s): %s", data.get("uid"), e)
        return jsonify(ok=False, error=str(e)), 500  # type: ignore[call-arg]


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))

    if OutcomePF is None:
        log.warning("PF backend: smart-dummy (OutcomePF import failed).")
    else:
        log.info("PF backend: %s (OutcomePF available)", PF_BACKEND)

    log.info(
        "Starting %s on port %s (PF_INIT=%s, DEPLETE_OK=%s, STAT_CALIBRATOR_OK=%s, webhook_ready=%s)",
        VERSION,
        port,
        pf_initialized,
        DEPLETE_OK,
        STAT_CALIBRATOR_OK,
        bool(line_handler is not None),
    )

    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
