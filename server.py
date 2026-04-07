from pathlib import Path

server_code = '''# -*- coding: utf-8 -*-
"""
server.py — BGS Pure PF + Deplete + Stage Overrides + FULL LINE Flow + Stability
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

# ---------- path fix ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BGS_DIR = os.path.join(BASE_DIR, "bgs")
if BGS_DIR not in sys.path:
    sys.path.insert(0, BGS_DIR)


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

# ---------- deplete ----------
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

# ---------- Flask ----------
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    _flask_available = True
except Exception:
    _flask_available = False
    Flask = None
    request = None

    def jsonify(*args, **kwargs):
        raise RuntimeError("Flask not available")

    def CORS(app):
        return None

# ---------- Redis ----------
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


def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id:
        return True
    return _rsetnx(f"dedupe:{event_id}", "1", ex=DEDUPE_TTL)


def _extract_line_event_id(event: Any) -> Optional[str]:
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


def get_session(uid: str) -> Dict[str, Any]:
    uid = uid or "anon"
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
    uid = uid or "anon"
    try:
        payload = json.dumps(sess, ensure_ascii=False)
        if redis_client:
            redis_client.set(_sess_key(uid), payload, ex=SESSION_EXPIRE_SECONDS)
        else:
            SESS_FALLBACK[uid] = sess
    except Exception as e:
        log.warning("save_session error: %s", e)


def format_output_card(probs: np.ndarray, choice: str, last_pts: Optional[str], bet_amt: int, cont: bool = True, mode: str = "") -> str:
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
        lines.append("\\n（輸入下一局點數：例如 65 / 和 / 閒6莊5）")
    return "\\n".join(lines)


VERSION = "bgs-clean-server-pf360-line-fix-2026-04-08"

# ---------- Flask App ----------
if _flask_available and Flask is not None:
    app = Flask(__name__)
    CORS(app)
else:
    class _DummyApp:
        def get(self, *a, **k):
            def _d(f):
                return f
            return _d

        def post(self, *a, **k):
            def _d(f):
                return f
            return _d

        def route(self, *a, **k):
            def _d(f):
                return f
            return _d

        def run(self, *a, **k):
            log.warning("Flask not available; cannot run HTTP server.")

    app = _DummyApp()

# ---------- PF ----------
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
KEEP_ALIVE_STARTED = False
KEEP_ALIVE_LOCK = threading.Lock()
_RESULT_CACHE: Dict[str, Tuple[np.ndarray, str, int, str]] = {}
_RESULT_CACHE_KEY: Dict[str, str] = {}
_RESULT_CACHE_LOCK = threading.Lock()
_UID_LOCKS: Dict[str, threading.Lock] = {}
_UID_LOCKS_GUARD = threading.Lock()


def _clear_prediction_cache(uid: str) -> None:
    uid = uid or "anon"
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE.pop(uid, None)
        _RESULT_CACHE_KEY.pop(uid, None)


def _make_result_cache_key(uid: str, sess: Dict[str, Any], p_pts: int, b_pts: int) -> str:
    return f"{uid}|r={int(sess.get('rounds_seen', 0))}|p={p_pts}|b={b_pts}|bank={int(sess.get('bankroll', 0))}|stateful={int(PF_STATEFUL)}|mode={str(sess.get('decision_mode') or os.getenv('DECISION_MODE', 'hybrid')).lower()}"


def _get_uid_lock(uid: str) -> threading.Lock:
    uid = uid or "anon"
    with _UID_LOCKS_GUARD:
        if uid not in _UID_LOCKS:
            _UID_LOCKS[uid] = threading.Lock()
        return _UID_LOCKS[uid]


def _self_keep_alive():
    try:
        import requests
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


DECISION_MODE = os.getenv("DECISION_MODE", "hybrid").lower()
BANKER_PAYOUT = float(os.getenv("BANKER_PAYOUT", "0.95"))
PROB_MARGIN = float(os.getenv("PROB_MARGIN", "0.02"))
MIN_EV_EDGE = float(os.getenv("MIN_EV_EDGE", "0.0"))
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
EV_NEUTRAL = int(os.getenv("EV_NEUTRAL", "0"))
PROB_BIAS_B2P = float(os.getenv("PROB_BIAS_B2P", "0.0"))
PROB_PURE_MODE = int(os.getenv("PROB_PURE_MODE", "0"))
PROB_FORCE_PURE_IN_PROB_MODE = env_flag("PROB_FORCE_PURE_IN_PROB_MODE", 1)
THEO_BLEND_FORCE_DISABLE = env_flag("THEO_BLEND_FORCE_DISABLE", 1)


def _current_decision_mode(over: Dict[str, float]) -> str:
    try:
        return str(over.get("DECISION_MODE", os.getenv("DECISION_MODE", "hybrid"))).strip().lower()
    except Exception:
        return "hybrid"


def bet_amount(bankroll: int, pct: float) -> int:
    if bankroll <= 0 or pct <= 0:
        return 0
    return int(round(bankroll * pct))


def _decide_side_by_ev(pB: float, pP: float) -> Tuple[int, float, float, float]:
    evB = BANKER_PAYOUT * pB - pP
    evP = pP - pB
    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))
    return side, final_edge, evB, evP


def _effective_prob_flags(over: Dict[str, float], mode: str) -> Tuple[int, int, List[str]]:
    notes: List[str] = []
    eff_prob_pure = PROB_PURE_MODE
    eff_ev_neutral = EV_NEUTRAL
    try:
        if "PROB_PURE_MODE" in over:
            eff_prob_pure = int(float(over["PROB_PURE_MODE"]))
    except Exception:
        pass
    try:
        if "EV_NEUTRAL" in over:
            eff_ev_neutral = int(float(over["EV_NEUTRAL"]))
    except Exception:
        pass
    if mode == "prob" and PROB_FORCE_PURE_IN_PROB_MODE == 1:
        if eff_prob_pure != 1:
            notes.append("FORCE_PURE(prob 模式自動純機率)")
            eff_prob_pure = 1
        if eff_ev_neutral != 0:
            notes.append("FORCE_EV_NEUTRAL_OFF(prob 純機率關閉 payout-aware)")
            eff_ev_neutral = 0
    return eff_prob_pure, eff_ev_neutral, notes


def _decide_side_by_prob(pB: float, pP: float, eff_prob_pure: int, eff_ev_neutral: int) -> int:
    if int(eff_prob_pure) == 1:
        return 0 if pB >= pP else 1
    if int(eff_ev_neutral) == 1:
        return 0 if (BANKER_PAYOUT * pB) >= pP else 1
    return 0 if pB >= pP else 1


def _apply_prob_bias(prob: np.ndarray, over: Dict[str, float]) -> np.ndarray:
    b2p = PROB_BIAS_B2P
    try:
        if "PROB_BIAS_B2P" in over:
            b2p = float(over["PROB_BIAS_B2P"])
    except Exception:
        pass
    b2p = max(0.0, float(b2p))
    if b2p <= 0.0:
        return prob
    p = prob.copy()
    shift = min(float(p[0]), b2p)
    if shift > 0:
        p[0] -= shift
        remBP = max(1e-8, 1.0 - float(p[2]))
        p[1] = min(remBP, float(p[1]) + shift)
    s = p.sum()
    if s > 0:
        p /= s
    return p


def decide_only_bp(prob: np.ndarray, over: Dict[str, float], effective_edge_enter: float, p_pts: int, b_pts: int) -> Tuple[str, float, float, str]:
    mode = _current_decision_mode(over)
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    reason: List[str] = []
    eff_prob_pure, eff_ev_neutral, notes = _effective_prob_flags(over, mode)
    if notes:
        reason.extend(notes)
    point_diff = abs(p_pts - b_pts)

    if mode == "prob":
        side = _decide_side_by_prob(pB, pP, eff_prob_pure, eff_ev_neutral)
        _, _, evB, evP = _decide_side_by_ev(pB, pP)
        final_edge = max(abs(evB), abs(evP))
        reason.append(f"模式=prob(pure={eff_prob_pure},ev_neutral={eff_ev_neutral})")
    elif mode == "hybrid":
        edge = abs(pB - pP)
        if edge >= PROB_MARGIN * 1.15:
            side = _decide_side_by_prob(pB, pP, eff_prob_pure, eff_ev_neutral)
            _, _, evB, evP = _decide_side_by_ev(pB, pP)
            final_edge = max(abs(evB), abs(evP))
            reason.append(f"模式=hybrid→prob (大差距 {edge:.4f})")
        else:
            s2, _, evB, evP = _decide_side_by_ev(pB, pP)
            final_edge = max(abs(evB), abs(evP))
            if edge < 0.065 and point_diff <= 5:
                if final_edge < 0.014:
                    return ("觀望", final_edge, 0.0, f"點數接近(diff={point_diff}) + 差距小 → 強制觀望")
                if evB > evP + MIN_EV_EDGE + 0.001:
                    side = s2
                    reason.append("模式=hybrid→ev (Banker 有優勢)")
                else:
                    return ("觀望", final_edge, 0.0, "hybrid → 觀望 (EV不足 + 小差距)")
            else:
                side, final_edge, evB, evP = _decide_side_by_ev(pB, pP)
                reason.append("模式=ev")
    else:
        side, final_edge, evB, evP = _decide_side_by_ev(pB, pP)
        reason.append("模式=ev")

    edge = abs(pB - pP)
    if LOG_DECISION or SHOW_CONF_DEBUG:
        log.info("[DECIDE-DBG] pB=%.4f pP=%.4f pT=%.4f edge=%.4f final_edge=%.4f point_diff=%d mode=%s", pB, pP, pT, edge, final_edge, point_diff, mode)
    if final_edge < effective_edge_enter:
        reason.append(f"⚪ 優勢不足 final_edge={final_edge:.4f}<{effective_edge_enter:.4f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason))

    min_b = max(0.0, min(1.0, MIN_BET_PCT_ENV))
    max_b = max(min_b, min(1.0, MAX_BET_PCT_ENV))
    bet_pct = min_b + (max_b - min_b) * (edge / MAX_EDGE_SCALE)
    bet_pct = float(min(max_b, max(min_b, bet_pct)))
    side_label = INV.get(side, "莊")
    reason.append(f"🔻 {side_label} 勝率={100.0 * (pB if side == 0 else pP):.1f}%")
    return (("莊" if side == 0 else "閒"), final_edge, bet_pct, "; ".join(reason))


def _stage_bounds():
    return int(os.getenv("EARLY_HANDS", "20")), int(os.getenv("MID_HANDS", os.getenv("LATE_HANDS", "56")))


def _stage_prefix(rounds_seen: int) -> str:
    e_end, m_end = _stage_bounds()
    if rounds_seen < e_end:
        return "EARLY_"
    if rounds_seen < m_end:
        return "MID_"
    return "LATE_"


def get_stage_over(rounds_seen: int) -> Dict[str, float]:
    if COMPAT_MODE == 1 or os.getenv("STAGE_MODE", "count").lower() == "disabled":
        return {}
    over: Dict[str, float] = {}
    prefix = _stage_prefix(rounds_seen)
    keys = ["SOFT_TAU", "THEO_BLEND", "TIE_MAX", "EDGE_ENTER", "PROB_MARGIN", "PF_PRED_SIMS", "DEPLETEMC_SIMS", "PF_UPD_SIMS", "PROB_PURE_MODE", "EV_NEUTRAL", "PROB_BIAS_B2P"]
    for k in keys:
        v = os.getenv(prefix + k)
        if v not in (None, ""):
            try:
                over[k] = float(v)
            except Exception:
                pass
    return over


def _depl_stage_scale(rounds_seen: int) -> float:
    prefix = _stage_prefix(rounds_seen)
    return EARLY_DEPL_SCALE if prefix == "EARLY_" else MID_DEPL_SCALE if prefix == "MID_" else LATE_DEPL_SCALE


def _guard_shift(old_p: np.ndarray, new_p: np.ndarray, max_shift: float) -> np.ndarray:
    p_old = old_p.astype(float).copy()
    delta = np.clip(new_p.astype(float) - p_old, -max_shift, max_shift)
    p_safe = p_old + delta
    s = float(p_safe.sum())
    if s > 0:
        p_safe /= s
    return p_safe.astype(np.float32)


def _tuned_pred_sims(base: int, pf_obj: Any) -> int:
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
    if not text:
        return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))
    s = re.sub(r"[\\u200b-\\u200f\\u202a-\\u202e\\u2060-\\u206f\\ufeff\\r\\n\\t]", "", s).replace("\\u3000", " ")
    u = s.upper().strip()

    m = re.search(r"(?:和|TIE|DRAW)\\s*:?:?\\s*(\\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else (0, 0)
    m = re.search(r"(?:閒|闲|P)\\s*:?:?\\s*(\\d)\\D+(?:莊|庄|B)\\s*:?:?\\s*(\\d)", u)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|庄|B)\\s*:?:?\\s*(\\d)\\D+(?:閒|闲|P)\\s*:?:?\\s*(\\d)", u)
    if m:
        return (int(m.group(2)), int(m.group(1)))

    t = u.replace(" ", "").replace("\\u3000", "")
    if t in ("B", "莊", "庄"):
        return (0, 1)
    if t in ("P", "閒", "闲"):
        return (1, 0)
    if t in ("T", "和"):
        return (0, 0)
    if re.search(r"[A-Z]", u):
        return None
    d = re.findall(r"\\d", u)
    if len(d) == 2:
        return (int(d[0]), int(d[1]))
    return None


def _advanced_control(sess: Dict[str, Any], probs: np.ndarray):
    history = sess.get("adv_history", [])
    history.append(int(np.argmax(probs[:2])))
    if len(history) > 20:
        history.pop(0)
    sess["adv_history"] = history
    probs2 = np.array([float(probs[0]), float(probs[1]), float(probs[2])], dtype=np.float32)
    probs2 = probs2 / probs2.sum()
    return probs2, 0.006, "正常"


def _handle_points_and_predict(uid: str, sess: Dict[str, Any], p_pts: int, b_pts: int) -> Tuple[np.ndarray, str, int, str]:
    with PREDICT_LOCK:
        rounds_seen = int(sess.get("rounds_seen", 0))
        over = get_stage_over(rounds_seen)
        mode = _current_decision_mode(over)
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
            lk = _get_uid_lock(uid)
            with lk:
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
                    log.warning("PF.update_outcome failed: %s", e)

                try:
                    upd_sims_val = over.get("PF_UPD_SIMS")
                    if upd_sims_val is None:
                        upd_sims_val = float(os.getenv("PF_UPD_SIMS", "30"))
                    if hasattr(pf_obj, "sims_lik"):
                        pf_obj.sims_lik = int(float(upd_sims_val))
                except Exception as e:
                    log.warning("stage PF_UPD_SIMS apply failed: %s", e)

                sims_used = _tuned_pred_sims(sims_base, pf_obj)
                start_time = time.time()
                p, sims_used = _safe_predict_probs(pf_obj, sims_used, uid)
                if time.time() - start_time > 1.2:
                    log.warning("⚠️ PF timeout fallback triggered")
                    p = SmartDummyPF().predict()
        else:
            pf_obj = _build_new_pf()
            sims_used = _tuned_pred_sims(sims_base, pf_obj)
            p, sims_used = _safe_predict_probs(pf_obj, sims_used, uid)

        soft_tau = float(over.get("SOFT_TAU", float(os.getenv("SOFT_TAU", "2.0"))))
        p = p ** (1.0 / max(1e-6, soft_tau))
        p = p / p.sum()

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
                    dep = probs_after_points(p_pts, b_pts, sims=dep_sims, rounds_seen=rounds_seen)
                    dep = np.asarray(dep, dtype=np.float32)
                    dep = dep / dep.sum()
                    mix = (1.0 - alpha) * p + alpha * dep
                    mix = mix / mix.sum()
                    p = _guard_shift(p, mix, MAX_DEPL_SHIFT)
            except Exception as e:
                log.warning("Deplete 失敗: %s", e)

        if COMPAT_MODE == 0 and THEO_BLEND_FORCE_DISABLE != 1:
            theo_blend = float(over.get("THEO_BLEND", float(os.getenv("THEO_BLEND", "0.0"))))
            if theo_blend > 0.0:
                theo = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
                p = (1.0 - theo_blend) * p + theo_blend * theo
                p = p / p.sum()

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

        p = _apply_prob_bias(p, over)
        if abs(p[0] - p[1]) > 0.20:
            mid = (p[0] + p[1]) / 2
            p[0] = mid + (p[0] - mid) * 0.95
            p[1] = mid + (p[1] - mid) * 0.95
            p = p / p.sum()

        p, dynamic_edge, ctrl_reason = _advanced_control(sess, p)
        choice, edge, bet_pct, reason = decide_only_bp(p, over, dynamic_edge, p_pts, b_pts)
        reason = f"{reason} | {ctrl_reason}"
        bet_amt = bet_amount(int(sess.get("bankroll", 0)), bet_pct)
        sess["rounds_seen"] = rounds_seen + 1
        sess["last_choice"] = choice

        with _RESULT_CACHE_LOCK:
            _RESULT_CACHE[uid] = (p.copy(), choice, bet_amt, reason)
            _RESULT_CACHE_KEY[uid] = cache_key

        return p, choice, bet_amt, reason


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
    return LINE_PUSH_ENABLE == 1 and int(time.time()) >= int(_PUSH_BLOCK_UNTIL)


def _block_push(reason: str):
    global _PUSH_BLOCK_UNTIL
    _PUSH_BLOCK_UNTIL = int(time.time()) + int(LINE_PUSH_COOLDOWN_SECONDS)
    log.warning("[LINE] push disabled temporarily: %s (block_until=%s)", reason, _PUSH_BLOCK_UNTIL)


def _looks_like_429(e: Exception) -> bool:
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
    if is_premium(uid):
        return None
    if is_trial_blocked(uid):
        return f"⛔ 試用已到期（帳號曾被封鎖）\\n🔐 如需重新啟用，請輸入：開通 你的密碼\\n📞 或聯繫：{ADMIN_CONTACT}"
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
        return f"⏰ 免費試用 {TRIAL_MINUTES} 分鐘已用完\\n🔐 請輸入：開通 你的專屬密碼\\n📞 沒有密碼？請聯繫：{ADMIN_CONTACT}"
    if expired == "1":
        return f"⛔ 試用已到期\\n🔐 請輸入：開通 你的專屬密碼\\n📞 沒有密碼？請聯繫：{ADMIN_CONTACT}"
    return None


def validate_activation_code(code: str) -> bool:
    if not code:
        return False
    norm = str(code).replace("\\u3000", " ").replace("：", ":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and norm == ADMIN_ACTIVATION_SECRET


GAMES = {"1": "WM", "2": "PM", "3": "DG", "4": "SA", "5": "KU", "6": "歐博/卡利", "7": "KG", "8": "全利", "9": "名人", "10": "MT真人"}


def game_menu_text(left_min: int) -> str:
    lines = ["請選擇遊戲館別"]
    for k in sorted(GAMES.keys(), key=lambda x: int(x)):
        lines.append(f"{k}. {GAMES[k]}")
    lines.append("「請直接輸入數字選擇」")
    lines.append(f"⏳ 試用剩餘 {left_min} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
    return "\\n".join(lines)


def _quick_buttons():
    try:
        from linebot.models import QuickReply, QuickReplyButton, MessageAction
        return QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="遊戲設定 🎮", text="遊戲設定")),
            QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
            QuickReplyButton(action=MessageAction(label="報莊勝 🅱️", text="B")),
            QuickReplyButton(action=MessageAction(label="報閒勝 🅿️", text="P")),
            QuickReplyButton(action=MessageAction(label="報和局 ⚪", text="T")),
        ])
    except Exception:
        return None


def _reply(api, token: str, text: str):
    from linebot.models import TextSendMessage
    try:
        api.reply_message(token, TextSendMessage(text=text, quick_reply=_quick_buttons()))
    except Exception as e:
        if "Invalid reply token" in str(e):
            log.info("[LINE] reply skipped (invalid token, likely retry): %s", e)
        else:
            log.warning("[LINE] reply failed: %s", e)


def _push_heavy_prediction(uid: str, p_pts: int, b_pts: int, seq: int):
    if line_api is None:
        return
    try:
        from linebot.models import TextSendMessage
        sess = get_session(uid)
        sess["last_pts_text"] = "上局結果: 和局" if (p_pts == b_pts and SKIP_TIE_UPD) else f"上局結果: 閒 {p_pts} 莊 {b_pts}"
        probs, choice, bet_amt, reason = _handle_points_and_predict(uid, sess, p_pts, b_pts)
        msg = format_output_card(probs, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE), mode=sess.get("decision_mode", ""))
        if int(sess.get("pending_seq", 0)) == int(seq):
            sess["last_card"] = msg
            sess["last_card_ts"] = int(time.time())
            sess["pending"] = False
        save_session(uid, sess)
        if _can_push():
            try:
                line_api.push_message(uid, TextSendMessage(text=msg, quick_reply=_quick_buttons()))
            except Exception as e:
                if _looks_like_429(e):
                    _block_push("429 monthly limit reached")
                try:
                    line_api.push_message(uid, TextSendMessage(text=msg))
                except Exception as e2:
                    log.warning("[LINE] push fallback failed: %s", e2)
    except Exception as e:
        log.exception("[heavy] prediction failed: %s", e)


line_api = None
line_handler = None
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import MessageEvent, TextMessage, FollowEvent, UnfollowEvent

    if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)
    else:
        log.error("LINE env missing: secret=%s token=%s", bool(LINE_CHANNEL_SECRET), bool(LINE_CHANNEL_ACCESS_TOKEN))

    if line_handler is not None:

        @line_handler.add(UnfollowEvent)
        def on_unfollow(event):
            if not _dedupe_event(_extract_line_event_id(event)):
                return
            try:
                uid = event.source.user_id
                set_trial_blocked(uid, True)
                _rset(_trial_key(uid, "expired"), "1")
            except Exception as e:
                log.warning("[TRIAL] unfollow handler error: %s", e)

        @line_handler.add(FollowEvent)
        def on_follow(event):
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
                msg = "👋 歡迎回來，已是永久開通用戶。\\n輸入『遊戲設定』開始。"
            else:
                msg = guard_msg or f"👋 歡迎！你有 {TRIAL_MINUTES} 分鐘免費試用。\\n輸入『遊戲設定』開始。"
            _reply(line_api, event.reply_token, msg)
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            if not _dedupe_event(_extract_line_event_id(event)):
                return
            uid = event.source.user_id
            text = re.sub(r"\\s+", " ", (event.message.text or "").replace("\\u3000", " ").strip())
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
                    "phase": "await_pts", "bankroll": 0, "rounds_seen": 0, "last_pts_text": None,
                    "premium": premium, "trial_start": start_ts, "last_card": None, "last_card_ts": None,
                    "pending": False, "pending_seq": 0, "loss_streak": 0, "adv_history": [], "last_choice": None,
                }
                reset_pf_for_uid(uid)
                _reply(line_api, event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。")
                save_session(uid, sess)
                return

            hist_match = re.fullmatch(r"[BPTHbpht]{6,30}", text)
            if hist_match:
                sess["rounds_seen"] = len(text)
                save_session(uid, sess)
                _reply(line_api, event.reply_token, "歷史載入完成\\n請輸入下一局點數\\n例如：65 / 和 / 閒6莊5")
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
                m = re.match(r"^\\s*(\\d+)", text)
                if m and m.group(1) in GAMES:
                    sess["game"] = GAMES[m.group(1)]
                    sess["phase"] = "input_bankroll"
                    _reply(line_api, event.reply_token, f"🎰 已選擇：{sess['game']}，請輸入初始籌碼（金額）")
                    save_session(uid, sess)
                    return
                _reply(line_api, event.reply_token, "⚠️ 無效的選項，請輸入上列數字。")
                return

            if sess.get("phase") == "input_bankroll":
                num = re.sub(r"[^\\d]", "", text)
                amt = int(num) if num else 0
                if amt <= 0:
                    _reply(line_api, event.reply_token, "⚠️ 請輸入正整數金額。")
                    return
                sess["bankroll"] = amt
                sess["phase"] = "await_pts"
                _reply(line_api, event.reply_token, f"✅ 設定完成！館別：{sess.get('game')}，初始籌碼：{amt}。\\n📌 現在輸入第一局點數（例：閒6莊5 / 65 / 和）")
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
                    threading.Thread(target=_push_heavy_prediction, args=(uid, p_pts, b_pts, seq), daemon=True).start()
                    return
                try:
                    sess["last_pts_text"] = "上局結果: 和局" if (p_pts == b_pts and SKIP_TIE_UPD) else f"上局結果: 閒 {p_pts} 莊 {b_pts}"
                    probs, choice, bet_amt, reason = _handle_points_and_predict(uid, sess, p_pts, b_pts)
                    msg = format_output_card(probs, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE), mode=sess.get("decision_mode", ""))
                    sess["last_card"] = msg
                    sess["last_card_ts"] = int(time.time())
                    sess["pending"] = False
                    save_session(uid, sess)
                    _reply(line_api, event.reply_token, msg)
                except Exception as e:
                    log.exception("[LINE] sync predict failed: %s", e)
                    _reply(line_api, event.reply_token, "⚠️ 計算失敗，請稍後再試或輸入下一局點數。")
                return

            _reply(line_api, event.reply_token, "指令無法辨識。\\n📌 直接輸入點數（例：65 / 和 / 閒6莊5），或輸入『遊戲設定』。")
except Exception as e:
    log.warning("LINE not fully configured: %s", e)


def _handle_line_webhook():
    if "line_handler" not in globals() or line_handler is None:
        log.error("webhook called but LINE handler not ready (missing credentials?)")
        return "OK", 200
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except Exception as e:
        log.error("webhook error: %s", e)
        return "OK", 200
    return "OK", 200


@app.route("/line-webhook", methods=["POST", "GET", "OPTIONS"])
def line_webhook():
    if request.method == "OPTIONS":
        return ("", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-Line-Signature",
        })
    if request.method == "GET":
        return "OK", 200
    return _handle_line_webhook()


@app.route("/callback", methods=["POST", "GET", "OPTIONS"])
def line_webhook_callback():
    if request.method == "OPTIONS":
        return ("", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-Line-Signature",
        })
    if request.method == "GET":
        return "OK", 200
    return _handle_line_webhook()


@app.get("/")
def root():
    ua = request.headers.get("User-Agent", "") if request else ""
    if "UptimeRobot" in ua:
        return "OK", 200
    st = "OK" if pf_initialized else "BACKUP_MODE"
    return f"✅ BGS Server {st} ({VERSION})", 200


@app.get("/health")
def health():
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
        webhook_ready=bool(line_handler is not None),
    ), 200


@app.get("/ping")
def ping():
    return "OK", 200


@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True) or {}
        uid = str(data.get("uid") or "anon")
        last_text = str(data.get("last_text") or "")
        bankroll = data.get("bankroll")
        sess = get_session(uid)
        if isinstance(bankroll, int) and bankroll >= 0:
            sess["bankroll"] = bankroll
        pts = parse_last_hand_points(last_text)
        if not pts:
            return jsonify(ok=False, error="無法解析點數；請輸入 '閒6莊5' / '65' / '和'"), 400
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
            card=format_output_card(probs, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE), mode=sess.get("decision_mode", "")),
        ), 200
    except Exception as e:
        log.exception("predict error: %s", e)
        return jsonify(ok=False, error=str(e)), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    if OutcomePF is None:
        log.warning("PF backend: smart-dummy (OutcomePF import failed).")
    else:
        log.info("PF backend: %s (OutcomePF available)", PF_BACKEND)
    log.info("Starting %s on port %s (PF_INIT=%s, DEPLETE_OK=%s, webhook_ready=%s)", VERSION, port, pf_initialized, DEPLETE_OK, bool(line_handler is not None))
    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
