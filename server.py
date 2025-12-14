# -*- coding: utf-8 -*-
"""server.py — BGS Independent + Stage Overrides + FULL LINE Flow + Compatibility (2025-11-03+perf-guard)

這版做了什麼（僅小幅補丁，不動你原本流程/介面）
- 保留「完整 LINE 互動流程」與所有既有開關
- 維持 /line-webhook 永遠註冊（未配置回 400）
- ✦ 補丁：預測「效能保護」與「安全上限」
  * 新增 PRED_SIMS_CAP（預設 10）→ 對 PF_PRED_SIMS 做上限，避免卡死
  * 依 PF_N 自動下修 sims（PF_N≥300→至多7；PF_N≥350→至多5）
  * 允許 __OPTIONS__ /line-webhook（避免外部探測造成雜訊）
- ✦ 補丁：回覆失敗（Invalid reply token）降噪記錄，不中斷流程
- ✦ 補丁：numpy 設為忽略 underflow/overflow 警示
- ✦ 補丁：LINE 重運算改成「快速 reply + 背景 thread push_message」
- ✦ 補丁：新增 /ping 給 UptimeRobot 專用

# ★ 2025-12-12 PATCH
- ✦ 補丁：LINE 429（月額度到達）「自動停推播」+ heavy 結果改存 session（原本可用「查詢」拿結果）

# ★ 2025-12-13 PATCH (FIX)
- ✦ 修正：PF 狀態不應全域共用（會導致方向黏住/多用戶互相污染）
  * 改為「每個 UID 一個 PF 狀態」（記憶體 store）
  * 同一 UID 的 update+predict 加鎖，避免 thread race
  * RESET 時同步清除該 UID PF 狀態

# ★ 2025-12-13 PATCH (FIX-REPEAT)
- ✦ 修正：LINE dedupe 取錯欄位導致同事件被重複處理（重送 webhook 會重算同一把）
  * 使用 webhook_event_id / message.id 作為 dedupe key

# ★ 2025-12-13 PATCH (REMOVE-QUERY)
- ✦ 移除：「查詢」按鈕與指令回覆（避免浪費 LINE 用量）
  * Quick Reply 不再顯示「查詢」
  * 不再處理「查詢 / QUERY」文字指令
  * 收到點數的提示文字不再引導使用者「查詢」

# ★ 2025-12-13 PATCH (TRIAL-FIX)
- ✦ 修正：新加入好友不該直接顯示「試用已到期」
  * FollowEvent 若 first_ts 不存在 → 強制寫入 first_ts 並清 expired
  * expired=1 但 first_ts 不存在/或其實未到期 → 視為髒資料，自動清掉 expired
  * 新增 TRIAL_NAMESPACE（預設 default）避免不同 Bot/不同專案共用同一組 trial key

# ★ 2025-12-13 PATCH (BLOCK-TRIAL + PROB-BIAS-DISPLAY-FIX + TIE-CAP-CTRL + PROB-PURE)
- ✦ 新增：封鎖(Unfollow) 即永久失效試用（blocked=1）
  * UnfollowEvent → 寫入 trial:blocked=1 並標記 expired=1
  * blocked=1 且非 premium → 直接視為到期必須開通
  * 開通成功 → 自動解除 blocked
- ✦ 修正：顯示機率與決策機率不一致（造成「莊較高卻叫下閒」）
  * PROB_BIAS_B2P 改為在 _handle_points_and_predict() 一次套用
  * decide_only_bp 不再二次套用 bias
- ✦ 新增：TIE_CAP_ENABLE / SHOW_RAW_PROBS
  * 可關閉「和局封頂」以避免長期卡在 15%
  * 可在 log 印出封頂前後機率方便抓問題
- ✦ 新增：PROB_PURE_MODE
  * DECISION_MODE=prob 時可強制「純機率」(pB>=pP 選莊)，避免 payout-aware 造成看起來反直覺

# ★ 2025-12-14 PATCH (STATELESS-PF)
- ✦ 新增：PF_STATEFUL
  * PF_STATEFUL=1（預設）：維持「每 UID 一個 PF 狀態」並 update_outcome → 連續學習
  * PF_STATEFUL=0：每次預測都 new 一個 PF（不 update_outcome）→ 每一局/每一次請求完全獨立（不記憶）

# ★ 2025-12-14 PATCH (PROB-DECIDE-SAFETY)
- ✦ 修正：避免「DECISION_MODE=prob 但沒真的純機率」導致 莊勝率較高卻叫下閒
  * 新增 PROB_FORCE_PURE_IN_PROB_MODE（預設 1）
    - 當 DECISION_MODE=prob 時，若你沒設定 PROB_PURE_MODE，會自動強制純機率
    - 並在決策時計算層面自動關閉 EV_NEUTRAL（避免 payout-aware 介入）
  * 決策層增加一致性自檢：若「純機率」模式仍出現 pB>pP 但選閒 → 強制改選莊並打警告 log
"""

import os, sys, logging, time, re, json, threading
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# 安靜數值警示（避免 PF 大量運算噪聲）
np.seterr(all="ignore")

# ---------- 安全導入 deplete ----------
DEPLETE_OK = False
init_counts = None
probs_after_points = None
try:
    from deplete import init_counts, probs_after_points  # type: ignore
    DEPLETE_OK = True
except Exception:
    try:
        from bgs.deplete import init_counts, probs_after_points  # type: ignore
        DEPLETE_OK = True
    except Exception:
        try:
            _cur_dir = os.path.dirname(os.path.abspath(__file__))
            if _cur_dir not in sys.path:
                sys.path.insert(0, _cur_dir)
            from deplete import init_counts, probs_after_points  # type: ignore
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


# ---------- 事件去重 ----------
def _dedupe_event(event_id: Optional[str]) -> bool:
    """避免 LINE webhook 同一事件重複處理；True=首次處理。"""
    if not event_id:
        return True
    key = f"dedupe:{event_id}"
    return _rsetnx(key, "1", ex=DEDUPE_TTL)


# ===== PATCH: 正確抽取 LINE event id（避免 dedupe 失效） =====
def _extract_line_event_id(event: Any) -> Optional[str]:
    """
    LINE SDK 常見可用：
    - event.webhook_event_id（最準）
    - event.message.id（MessageEvent）
    - event.id（有時候會有）
    """
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
# ===== PATCH END =====


# ---------- Premium（永久開通） ----------
def _premium_key(uid: str) -> str:
    return f"premium:{uid}"


def is_premium(uid: str) -> bool:
    """檢查此 UID 是否已永久開通。"""
    if not uid:
        return False
    val = _rget(_premium_key(uid))
    return val == "1"


def set_premium(uid: str, flag: bool = True) -> None:
    """設定永久開通狀態；flag=True 表示永久開通。"""
    if not uid:
        return
    _rset(_premium_key(uid), "1" if flag else "0")


# ---------- 簡易 Session 層 ----------
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
                if "pending" not in sess:
                    sess["pending"] = False
                if "pending_seq" not in sess:
                    sess["pending_seq"] = 0
                return sess
        sess = SESS_FALLBACK.get(uid)
        if isinstance(sess, dict):
            if is_premium(uid):
                sess["premium"] = True
            if "pending" not in sess:
                sess["pending"] = False
            if "pending_seq" not in sess:
                sess["pending_seq"] = 0
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
                       bet_amt: int, cont: bool = True) -> str:
    pB, pP, pT = [float(x) for x in probs]
    lines = []
    if last_pts:
        lines.append(str(last_pts))
    lines.append(f"機率｜莊 {pB*100:.1f}%｜閒 {pP*100:.1f}%｜和 {pT*100:.1f}%")
    if choice == "觀望":
        lines.append("建議：觀望 👀")
    else:
        lines.append(f"建議：下 {choice} 🎯")
        if bet_amt and bet_amt > 0:
            lines.append(f"配注：{bet_amt}")
    if cont:
        lines.append("\n（輸入下一局點數：例如 65 / 和 / 閒6莊5）")
    return "\n".join(lines)


# ---------- 版本 ----------
VERSION = "bgs-independent-2025-11-03+stage+LINE+compat+perfguard+bgpush+429patch+trialfix+blocktrial+probdisplayfix+tiecapprobpure+statelesspf+probdecidesafety"

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

        def options(self, *a, **k):
            def _d(f):
                return f
            return _d

        def run(self, *a, **k):
            log.warning("Flask not available; cannot run HTTP server.")

    app = _DummyApp()

# ---------- PF（Outcome PF） ----------
PF_BACKEND = os.getenv("PF_BACKEND", "mc").lower()
SKIP_TIE_UPD = env_flag("SKIP_TIE_UPD", 1)
SOFT_TAU = float(os.getenv("SOFT_TAU", "2.0"))
TIE_MIN = float(os.getenv("TIE_MIN", "0.05"))
TIE_MAX = float(os.getenv("TIE_MAX", "0.15"))
HISTORY_MODE = env_flag("HISTORY_MODE", 0)

# ★ 可控的和局封頂 + debug
TIE_CAP_ENABLE = env_flag("TIE_CAP_ENABLE", 1)   # 1=維持封頂，0=不封頂（避免卡 15%）
SHOW_RAW_PROBS = env_flag("SHOW_RAW_PROBS", 0)   # 1=log 印封頂前後機率

# ★ PF 是否有狀態（是否記憶上一局）
PF_STATEFUL = env_flag("PF_STATEFUL", 1)         # 1=per-uid stateful；0=每次 new PF（完全獨立）

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
        log.warning("⚠️ OutcomePF unavailable → SmartDummyPF fallback (PROBS MAY LOOK STATIC)")

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


# ===== PF per-UID store + lock (thread-safe) =====
_PF_STORE: Dict[str, Any] = {}
_PF_LOCKS: Dict[str, threading.Lock] = {}
_PF_STORE_GUARD = threading.Lock()


def _get_uid_lock(uid: str) -> threading.Lock:
    if not uid:
        uid = "anon"
    with _PF_STORE_GUARD:
        lk = _PF_LOCKS.get(uid)
        if lk is None:
            lk = threading.Lock()
            _PF_LOCKS[uid] = lk
        return lk


def _build_new_pf() -> Any:
    if OutcomePF is None:
        return SmartDummyPF()
    return OutcomePF(
        decks=int(os.getenv("DECKS", "8")),
        seed=int(os.getenv("SEED", "42")),
        n_particles=int(os.getenv("PF_N", "50")),
        sims_lik=int(os.getenv("PF_UPD_SIMS", "30")),
        resample_thr=float(os.getenv("PF_RESAMPLE", "0.5")),
        backend=PF_BACKEND,
        dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.05"))
    )


def get_pf_for_uid(uid: str) -> Any:
    if not uid:
        uid = "anon"
    with _PF_STORE_GUARD:
        pf = _PF_STORE.get(uid)
        if pf is None:
            try:
                pf = _build_new_pf()
            except Exception as e:
                log.error("PF 初始化失敗(per-uid): %s", e)
                pf = SmartDummyPF()
            _PF_STORE[uid] = pf
        return pf


def reset_pf_for_uid(uid: str) -> None:
    if not uid:
        uid = "anon"
    with _PF_STORE_GUARD:
        if uid in _PF_STORE:
            _PF_STORE.pop(uid, None)
# ===== END =====

pf_initialized = True if (OutcomePF is not None) else True


# ---------- 決策 / 配注 ----------
DECISION_MODE = os.getenv("DECISION_MODE", "ev").lower()  # ev | prob | hybrid
BANKER_PAYOUT = float(os.getenv("BANKER_PAYOUT", "0.95"))
PROB_MARGIN = float(os.getenv("PROB_MARGIN", "0.02"))
MIN_EV_EDGE = float(os.getenv("MIN_EV_EDGE", "0.0"))

MIN_CONF_FOR_ENTRY = float(os.getenv("MIN_CONF_FOR_ENTRY", "0.56"))
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
QUIET_SMALLEdge = env_flag("QUIET_SMALLEdge", 0)

MIN_BET_PCT_ENV = float(os.getenv("MIN_BET_PCT", "0.05"))
MAX_BET_PCT_ENV = float(os.getenv("MAX_BET_PCT", "0.40"))
MAX_EDGE_SCALE = float(os.getenv("MAX_EDGE_FOR_FULLBET", "0.15"))

USE_KELLY = env_flag("USE_KELLY", 0)
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)

SHOW_CONF_DEBUG = env_flag("SHOW_CONF_DEBUG", 1)
LOG_DECISION = env_flag("LOG_DECISION", 1)

INV = {0: "莊", 1: "閒"}

COMPAT_MODE = int(os.getenv("COMPAT_MODE", "0"))
DEPL_ENABLE = int(os.getenv("DEPL_ENABLE", "0"))

DEPL_FACTOR = float(os.getenv("DEPL_FACTOR", "0.60"))
DEPL_STAGE_MODE = os.getenv("DEPL_STAGE_MODE", "depth").lower()

EARLY_DEPL_SCALE = float(os.getenv("EARLY_DEPL_SCALE", "0.2"))
MID_DEPL_SCALE = float(os.getenv("MID_DEPL_SCALE", "0.6"))
LATE_DEPL_SCALE = float(os.getenv("LATE_DEPL_SCALE", "0.9"))

MAX_DEPL_SHIFT = float(os.getenv("MAX_DEPL_SHIFT", "0.10"))

EV_NEUTRAL = int(os.getenv("EV_NEUTRAL", "0"))
PROB_BIAS_B2P = float(os.getenv("PROB_BIAS_B2P", "0.0"))

# ★ DECISION_MODE=prob 時可強制「純機率」
PROB_PURE_MODE = int(os.getenv("PROB_PURE_MODE", "0"))  # 1=純機率(pB>=pP選莊)，0=沿用既有邏輯

# ★ PATCH: prob 模式自動強制純機率（避免你忘了設 PROB_PURE_MODE）
# - 預設 1：DECISION_MODE=prob 時，若你沒設 PROB_PURE_MODE → 自動視為純機率
PROB_FORCE_PURE_IN_PROB_MODE = env_flag("PROB_FORCE_PURE_IN_PROB_MODE", 1)


def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0:
        return 0
    return int(round(bankroll * pct))


def _decide_side_by_ev(pB: float, pP: float) -> Tuple[int, float, float, float]:
    evB = BANKER_PAYOUT * pB - pP
    evP = pP - pB
    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))
    return side, final_edge, evB, evP


def _effective_prob_flags(over: Dict[str, float]) -> Tuple[int, int, List[str]]:
    """
    回傳 (eff_prob_pure, eff_ev_neutral, notes[])
    - 若 DECISION_MODE=prob 且 PROB_FORCE_PURE_IN_PROB_MODE=1：
        * 當 PROB_PURE_MODE 沒明確設定為 1/0（或為 0）時，也會強制用純機率（eff_prob_pure=1）
        * 並且決策層面關閉 payout-aware（eff_ev_neutral=0）
    """
    notes: List[str] = []

    # 允許三段覆蓋（若有）
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

    if DECISION_MODE == "prob" and PROB_FORCE_PURE_IN_PROB_MODE == 1:
        # 只要 prob 模式，就把決策層固定為「純機率」
        if eff_prob_pure != 1:
            notes.append("FORCE_PURE(prob 模式自動純機率)")
        eff_prob_pure = 1

        # 並在決策層面直接關掉 payout-aware（避免莊勝率高卻叫下閒）
        if eff_ev_neutral != 0:
            notes.append("FORCE_EV_NEUTRAL_OFF(prob 純機率關閉 payout-aware)")
        eff_ev_neutral = 0

    return eff_prob_pure, eff_ev_neutral, notes


def _decide_side_by_prob(pB: float, pP: float, eff_prob_pure: int, eff_ev_neutral: int) -> int:
    # eff_prob_pure=1：永遠用純機率比較
    if int(eff_prob_pure) == 1:
        return 0 if pB >= pP else 1
    # payout-aware（莊被打折）
    if int(eff_ev_neutral) == 1:
        return 0 if (BANKER_PAYOUT * pB) >= pP else 1
    return 0 if pB >= pP else 1


def _apply_prob_bias(prob: np.ndarray, over: Dict[str, float]) -> np.ndarray:
    # 允許三段覆蓋 PROB_BIAS_B2P
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


def decide_only_bp(prob: np.ndarray, over: Dict[str, float]) -> Tuple[str, float, float, str]:
    # 不在這裡套用 bias（避免顯示與決策不一致 / 避免雙重偏移）
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    reason: List[str] = []

    eff_prob_pure, eff_ev_neutral, notes = _effective_prob_flags(over)
    if notes:
        reason.extend(notes)

    if DECISION_MODE == "prob":
        side = _decide_side_by_prob(pB, pP, eff_prob_pure, eff_ev_neutral)
        _, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
        final_edge = max(abs(evB), abs(evP))
        reason.append(f"模式=prob(pure={eff_prob_pure},ev_neutral={eff_ev_neutral})")

        # ★ 自檢防呆：純機率下不應出現 pB>pP 但選閒
        if int(eff_prob_pure) == 1 and pB > pP and side == 1:
            side = 0
            reason.append("⚠️ FIX: pure_prob 但選到閒→強制改莊")
            log.warning("[DECIDE-FIX] pure_prob conflict detected (pB=%.4f>pP=%.4f) forced to BANKER", pB, pP)

    elif DECISION_MODE == "hybrid":
        if abs(pB - pP) >= PROB_MARGIN:
            side = _decide_side_by_prob(pB, pP, eff_prob_pure, eff_ev_neutral)
            _, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
            final_edge = max(abs(evB), abs(evP))
            reason.append(f"模式=hybrid→prob(pure={eff_prob_pure},ev_neutral={eff_ev_neutral})")
        else:
            s2, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
            if edge_ev >= MIN_EV_EDGE:
                side = s2
                final_edge = edge_ev
                reason.append("模式=hybrid→ev")
            else:
                side = _decide_side_by_prob(pB, pP, eff_prob_pure, eff_ev_neutral)
                final_edge = edge_ev
                reason.append(f"模式=hybrid→prob(EV不足)(pure={eff_prob_pure},ev_neutral={eff_ev_neutral})")
    else:
        side, final_edge, evB, evP = _decide_side_by_ev(pB, pP)
        reason.append("模式=ev")

    conf = max(pB, pP)
    if conf < MIN_CONF_FOR_ENTRY:
        reason.append(f"⚪ 信心不足 conf={conf:.3f}<{MIN_CONF_FOR_ENTRY:.3f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason))
    if final_edge < EDGE_ENTER:
        reason.append(f"⚪ 優勢不足 edge={final_edge:.4f}<{EDGE_ENTER:.4f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason))
    if QUIET_SMALLEdge and final_edge < (EDGE_ENTER * 1.2):
        reason.append("⚪ 邊際略優(quiet)")
        return ("觀望", final_edge, 0.0, "; ".join(reason))

    min_b = max(0.0, min(1.0, MIN_BET_PCT_ENV))
    max_b = max(min_b, min(1.0, MAX_BET_PCT_ENV))
    max_edge = max(EDGE_ENTER + 1e-6, MAX_EDGE_SCALE)
    bet_pct = min_b + (max_b - min_b) * (final_edge - EDGE_ENTER) / (max_edge - EDGE_ENTER)
    bet_pct = float(min(max_b, max(min_b, bet_pct)))
    side_label = INV.get(side, "莊")
    reason.append(f"🔻 {side_label} 勝率={100.0 * (pB if side==0 else pP):.1f}%")
    return (("莊" if side == 0 else "閒"), final_edge, bet_pct, "; ".join(reason))


# ---------- 三段覆蓋 ----------
def _stage_bounds():
    early_end = int(os.getenv("EARLY_HANDS", "20"))
    mid_end = int(os.getenv("MID_HANDS", os.getenv("LATE_HANDS", "56")))
    return early_end, mid_end


def _stage_prefix(rounds_seen: int) -> str:
    e_end, m_end = _stage_bounds()
    if rounds_seen < e_end:
        return "EARLY_"
    elif rounds_seen < m_end:
        return "MID_"
    else:
        return "LATE_"


def get_stage_over(rounds_seen: int) -> Dict[str, float]:
    if COMPAT_MODE == 1:
        return {}
    if os.getenv("STAGE_MODE", "count").lower() == "disabled":
        return {}

    over: Dict[str, float] = {}
    prefix = _stage_prefix(rounds_seen)

    # ★ PATCH：允許三段覆蓋決策關鍵參數（可選）
    keys = [
        "SOFT_TAU", "THEO_BLEND", "TIE_MAX",
        "MIN_CONF_FOR_ENTRY", "EDGE_ENTER", "PROB_MARGIN",
        "PF_PRED_SIMS", "DEPLETEMC_SIMS", "PF_UPD_SIMS",
        "PROB_PURE_MODE", "EV_NEUTRAL", "PROB_BIAS_B2P",
    ]

    for k in keys:
        v = os.getenv(prefix + k)
        if v not in (None, ""):
            try:
                over[k] = float(v)
            except Exception:
                pass

    if prefix == "LATE_":
        late_dep = os.getenv("LATE_DEPLETEMC_SIMS")
        if late_dep:
            try:
                over["DEPLETEMC_SIMS"] = float(late_dep)
            except Exception:
                pass
    return over


def _depl_stage_scale(rounds_seen: int) -> float:
    prefix = _stage_prefix(rounds_seen)
    if prefix == "EARLY_":
        return EARLY_DEPL_SCALE
    elif prefix == "MID_":
        return MID_DEPL_SCALE
    else:
        return LATE_DEPL_SCALE


def _guard_shift(old_p: np.ndarray, new_p: np.ndarray, max_shift: float) -> np.ndarray:
    max_shift = max(0.0, float(max_shift))
    p_old = old_p.astype(float).copy()
    p_new = new_p.astype(float).copy()
    delta = p_new - p_old
    delta = np.clip(delta, -max_shift, max_shift)
    p_safe = p_old + delta
    s = float(p_safe.sum())
    if s > 0:
        p_safe /= s
    return p_safe.astype(np.float32)


# ---------- 預測效能保護 ----------
def _tuned_pred_sims(base: int, pf_obj: Any) -> int:
    try:
        cap = int(float(os.getenv("PRED_SIMS_CAP", "10")))
    except Exception:
        cap = 10
    n = max(1, min(int(base), cap))
    try:
        n_particles = int(getattr(pf_obj, 'n_particles', 200))
        if n_particles >= 350 and n > 5:
            n = 5
        elif n_particles >= 300 and n > 7:
            n = 7
    except Exception:
        pass
    return max(1, n)


# ---------- 解析點數 ----------
def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
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

# --------------------------------------------------
# Debug/Test utilities
# --------------------------------------------------
def test_deplete_biases() -> None:
    """
    Utility function to examine potential biases introduced by the deplete
    simulation. This function prints out banker/ player/ tie probabilities
    for a series of preset point combinations along with the difference
    (banker minus player) and which side is higher. The number of Monte
    Carlo simulations is deliberately large to reduce variance (defaults
    to 10,000 but can be overridden via the environment variable
    DEPLETEMC_SIMS). You can invoke this function manually from a
    Python REPL or within your own diagnostic scripts.
    """
    if not DEPLETE_OK or init_counts is None or probs_after_points is None:
        log.warning("test_deplete_biases called but deplete support is unavailable")
        return
    try:
        decks = int(os.getenv("DECKS", "8"))
        counts = init_counts(decks)
        sims_env = os.getenv("DEPLETEMC_SIMS")
        # If DEPLETEMC_SIMS is not set in the environment, use a large default
        sims = int(float(sims_env)) if sims_env else 10000
        deplete_factor = float(os.getenv("DEPL_FACTOR", "0.60"))
        scenarios = [
            ("開局", 0, 0),
            ("閒贏1點", 1, 0),
            ("莊贏1點", 0, 1),
            ("平手1點", 1, 1),
            ("閒贏6點", 6, 0),
            ("莊贏6點", 0, 6),
        ]
        log.info("=== Deplete 偏差測試 (sims=%d, factor=%.2f) ===", sims, deplete_factor)
        for name, p_pts, b_pts in scenarios:
            try:
                probs = probs_after_points(counts, p_pts, b_pts, sims=sims, deplete_factor=deplete_factor)
                if not isinstance(probs, (list, tuple, np.ndarray)) or len(probs) < 2:
                    log.info("%s: unexpected deplete result %s", name, probs)
                    continue
                pB, pP, pT = float(probs[0]), float(probs[1]), float(probs[2] if len(probs) > 2 else 0.0)
                diff = pB - pP
                bias = "莊高" if diff > 0 else ("閒高" if diff < 0 else "平手")
                log.info(
                    "%s: 莊=%.4f 閒=%.4f 和=%.4f | 差值=%.4f (%s)",
                    name, pB, pP, pT, diff, bias
                )
            except Exception as ex:
                log.warning("test_deplete_biases scenario %s failed: %s", name, ex)
    except Exception as ex:
        log.warning("test_deplete_biases error: %s", ex)


def debug_card_distribution() -> None:
    """
    Utility function to print out the distribution of card values within
    the current shoe. This can help verify that the initial deck setup
    is balanced and does not inadvertently favour banker or player. The
    function groups cards by their Baccarat point values (with J/Q/K
    treated as 10) and logs both the absolute counts and the percentage
    of the total deck. Invoke this manually from a REPL or diagnostic
    script. It does not modify any state.
    """
    if not DEPLETE_OK or init_counts is None:
        log.warning("debug_card_distribution called but deplete support is unavailable")
        return
    try:
        decks = int(os.getenv("DECKS", "8"))
        counts = init_counts(decks)
        # counts may be dict-like or list-like depending on implementation
        total_cards = sum(counts.values()) if isinstance(counts, dict) else sum(counts)
        point_cards: Dict[int, int] = {}
        # iterate through card counts and accumulate by point value
        if isinstance(counts, dict):
            iterable = counts.items()
        else:
            # list/tuple of counts indexed by card value (1-13 or 0-12)
            iterable = enumerate(counts)
        for card_value, count in iterable:
            # Convert card faces to Baccarat points (J/Q/K count as 10)
            # Note: Some implementations index from 0; ensure values > 0 map correctly
            try:
                val = int(card_value)
            except Exception:
                continue
            point = min(10, val if val > 0 else 10)
            point_cards[point] = point_cards.get(point, 0) + int(count)
        log.info("牌組分布:")
        for point in sorted(point_cards.keys()):
            cnt = point_cards[point]
            pct = (cnt / total_cards * 100.0) if total_cards else 0.0
            log.info("  點數 %s: %s 張 (%.1f%%)", point, cnt, pct)
    except Exception as ex:
        log.warning("debug_card_distribution error: %s", ex)


# ---------- 主預測 ----------
def _handle_points_and_predict(uid: str, sess: Dict[str, Any], p_pts: int, b_pts: int) -> Tuple[np.ndarray, str, int, str]:
    rounds_seen = int(sess.get("rounds_seen", 0))
    over = get_stage_over(rounds_seen)
    # ----- Debug: prepare holders for probability snapshots -----
    # pf_probs will hold the raw PF prediction probabilities (before softening)
    # soft_probs will hold probabilities after softening by SOFT_TAU
    pf_probs: Optional[np.ndarray] = None
    soft_probs: Optional[np.ndarray] = None

    # ★ STATELESS-PF：PF_STATEFUL=0 → 每次都 new，一次預測一次丟（不記憶）
    if PF_STATEFUL == 1:
        pf_obj = get_pf_for_uid(uid)
        lk = _get_uid_lock(uid)
        with lk:
            try:
                if hasattr(pf_obj, "update_outcome"):
                    if (p_pts == b_pts):
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

            sims_per_particle = int(over.get("PF_PRED_SIMS", float(os.getenv("PF_PRED_SIMS", "5"))))
            sims_per_particle = _tuned_pred_sims(sims_per_particle, pf_obj)
            p = np.asarray(pf_obj.predict(sims_per_particle=sims_per_particle), dtype=np.float32)
            # Capture raw PF prediction probabilities for debugging
            pf_probs = p.copy()
    else:
        # 完全獨立：不 update_outcome、不用 per-uid store、不需要 lock
        try:
            pf_obj = _build_new_pf()
        except Exception as e:
            log.error("PF 初始化失敗(stateless): %s", e)
            pf_obj = SmartDummyPF()

        try:
            upd_sims_val = over.get("PF_UPD_SIMS")
            if upd_sims_val is None:
                upd_sims_val = float(os.getenv("PF_UPD_SIMS", "30"))
            if hasattr(pf_obj, "sims_lik"):
                pf_obj.sims_lik = int(float(upd_sims_val))
        except Exception as e:
            log.warning("stage PF_UPD_SIMS apply failed(stateless): %s", e)

        sims_per_particle = int(over.get("PF_PRED_SIMS", float(os.getenv("PF_PRED_SIMS", "5"))))
        sims_per_particle = _tuned_pred_sims(sims_per_particle, pf_obj)
        p = np.asarray(pf_obj.predict(sims_per_particle=sims_per_particle), dtype=np.float32)
        # Capture raw PF prediction probabilities for debugging
        pf_probs = p.copy()

    # 後處理：不需要鎖
    soft_tau = float(over.get("SOFT_TAU", float(os.getenv("SOFT_TAU", "2.0"))))
    p = p ** (1.0 / max(1e-6, soft_tau))
    p = p / p.sum()
    # Capture softened probabilities and emit debug logs
    soft_probs = p.copy()
    if SHOW_RAW_PROBS:
        try:
            if pf_probs is not None:
                log.info("[DEBUG-PF] PF原始: 莊=%.4f, 閒=%.4f", float(pf_probs[0]), float(pf_probs[1]))
                log.info("[DEBUG-SOFT] 軟化後: 莊=%.4f, 閒=%.4f", float(soft_probs[0]), float(soft_probs[1]))
        except Exception:
            pass

    if (COMPAT_MODE == 0) and (DEPL_ENABLE == 1) and DEPLETE_OK and init_counts and probs_after_points:
        try:
            stage_scale = _depl_stage_scale(rounds_seen)
            raw_alpha = DEPL_FACTOR * stage_scale
            alpha = max(0.0, min(0.55, float(raw_alpha)))

            if alpha > 0.0:
                # Capture probabilities before applying deplete for debugging
                before_deplete = p.copy()
                if SHOW_RAW_PROBS:
                    log.info("[DEBUG-B4-DEPL] Deplete前: 莊=%.4f, 閒=%.4f", float(before_deplete[0]), float(before_deplete[1]))
                counts = init_counts(int(os.getenv("DECKS", "8")))
                dep_sims = int(over.get("DEPLETEMC_SIMS", float(os.getenv("DEPLETEMC_SIMS", "18"))))

                dep = probs_after_points(
                    counts,
                    p_pts,
                    b_pts,
                    sims=dep_sims,
                    deplete_factor=alpha
                )
                dep = np.asarray(dep, dtype=np.float32)

                depT = float(dep[2])
                if depT < TIE_MIN:
                    dep[2] = TIE_MIN
                    sc = (1.0 - TIE_MIN) / (1.0 - depT) if depT < 1.0 else 1.0
                    dep[0] *= sc
                    dep[1] *= sc
                elif depT > TIE_MAX:
                    dep[2] = TIE_MAX
                    sc = (1.0 - TIE_MAX) / (1.0 - depT) if depT < 1.0 else 1.0
                    dep[0] *= sc
                    dep[1] *= sc
                dep = dep / dep.sum()

                mix = (1.0 - alpha) * p + alpha * dep
                mix = mix / mix.sum()
                p = _guard_shift(p, mix, MAX_DEPL_SHIFT)
                # Capture probabilities after deplete and log effect
                after_deplete = p.copy()
                if SHOW_RAW_PROBS:
                    log.info("[DEBUG-AFT-DEPL] Deplete後: 莊=%.4f, 閒=%.4f", float(after_deplete[0]), float(after_deplete[1]))
                    delta_B = float(after_deplete[0] - before_deplete[0])
                    delta_P = float(after_deplete[1] - before_deplete[1])
                    log.info("[DEPLETE-EFFECT] 莊變化: %+.4f, 閒變化: %+.4f", delta_B, delta_P)
                    log.info("[DEPLETE-EFFECT] 使莊 %s了機率", "增加" if delta_B > 0 else "減少")
        except Exception as e:
            log.warning("Deplete 失敗，改 PF 單模：%s", e)

    if COMPAT_MODE == 0:
        theo_blend = float(over.get("THEO_BLEND", float(os.getenv("THEO_BLEND", "0.0"))))
        if theo_blend > 0.0:
            if SHOW_RAW_PROBS:
                before_theo = p.copy()
            theo = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
            p = (1.0 - theo_blend) * p + theo_blend * theo
            p = p / p.sum()
            if SHOW_RAW_PROBS:
                after_theo = p.copy()
                log.info("[DEBUG-B4-THEO] 理論混合前: 莊=%.4f, 閒=%.4f", float(before_theo[0]), float(before_theo[1]))
                log.info("[DEBUG-AFT-THEO] 理論混合後: 莊=%.4f, 閒=%.4f", float(after_theo[0]), float(after_theo[1]))

        if SHOW_RAW_PROBS:
            log.info("[PROBS] raw(after mix/theo) B=%.4f P=%.4f T=%.4f (uid=%s rounds=%s stateful=%s)",
                     float(p[0]), float(p[1]), float(p[2]), uid, rounds_seen, PF_STATEFUL)

        # TIE_CAP_ENABLE：可關閉「和局封頂」避免卡 15%
        tie_max = float(over.get("TIE_MAX", float(os.getenv("TIE_MAX", str(TIE_MAX)))))
        if TIE_CAP_ENABLE == 1:
            if p[2] > tie_max:
                # Capture probabilities before tie cap for debugging
                if SHOW_RAW_PROBS:
                    before_tiecap = p.copy()
                sc = (1.0 - tie_max) / (1.0 - float(p[2])) if p[2] < 1.0 else 1.0
                p[2] = tie_max
                p[0] *= sc
                p[1] *= sc
                p = p / p.sum()
                # Capture probabilities after tie cap and log
                if SHOW_RAW_PROBS:
                    after_tiecap = p.copy()
                    log.info("[DEBUG-B4-TIECAP] 和局封頂前: 莊=%.4f, 閒=%.4f", float(before_tiecap[0]), float(before_tiecap[1]))
                    log.info("[DEBUG-AFT-TIECAP] 和局封頂後: 莊=%.4f, 閒=%.4f", float(after_tiecap[0]), float(after_tiecap[1]))

        if p[2] < TIE_MIN:
            sc = (1.0 - TIE_MIN) / (1.0 - float(p[2])) if p[2] < 1.0 else 1.0
            p[2] = TIE_MIN
            p[0] *= sc
            p[1] *= sc
            p = p / p.sum()

        if SHOW_RAW_PROBS:
            log.info("[PROBS] final(after tie clamp) B=%.4f P=%.4f T=%.4f (uid=%s rounds=%s stateful=%s)",
                     float(p[0]), float(p[1]), float(p[2]), uid, rounds_seen, PF_STATEFUL)

    # ★ 在這裡一次套用 bias，讓「顯示」與「決策」一致
    p = _apply_prob_bias(p, over)

    _MIN_CONF, _EDGE_ENTER, _PROB_MARGIN = MIN_CONF_FOR_ENTRY, EDGE_ENTER, PROB_MARGIN
    try:
        if COMPAT_MODE == 0:
            if "MIN_CONF_FOR_ENTRY" in over:
                globals()["MIN_CONF_FOR_ENTRY"] = float(over["MIN_CONF_FOR_ENTRY"])
            if "EDGE_ENTER" in over:
                globals()["EDGE_ENTER"] = float(over["EDGE_ENTER"])
            if "PROB_MARGIN" in over:
                globals()["PROB_MARGIN"] = float(over["PROB_MARGIN"])

        choice, edge, bet_pct, reason = decide_only_bp(p, over)
    finally:
        globals()["MIN_CONF_FOR_ENTRY"] = _MIN_CONF
        globals()["EDGE_ENTER"] = _EDGE_ENTER
        globals()["PROB_MARGIN"] = _PROB_MARGIN

    bet_amt = bet_amount(int(sess.get("bankroll", 0)), bet_pct)
    sess["rounds_seen"] = rounds_seen + 1

    if LOG_DECISION or SHOW_CONF_DEBUG:
        log.info(
            "決策: %s edge=%.4f pct=%.2f%% rounds=%d sims=%d uid=%s stateful=%s | %s",
            choice, edge, bet_pct * 100, sess["rounds_seen"],
            int(over.get("PF_PRED_SIMS", float(os.getenv("PF_PRED_SIMS", "5")))),
            uid, PF_STATEFUL, reason
        )
    return p, choice, bet_amt, reason


# ---------- LINE：完整互動 ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "aaa8881688")

# ★ TRIAL namespace（避免不同 bot 共用 trial key）
TRIAL_NAMESPACE = os.getenv("TRIAL_NAMESPACE", "default").strip() or "default"

LINE_PUSH_ENABLE = env_flag("LINE_PUSH_ENABLE", 1)
LINE_PUSH_COOLDOWN_SECONDS = int(os.getenv("LINE_PUSH_COOLDOWN_SECONDS", str(30 * 24 * 3600)))
_PUSH_BLOCK_UNTIL = 0


def _can_push() -> bool:
    global _PUSH_BLOCK_UNTIL
    if LINE_PUSH_ENABLE != 1:
        return False
    return int(time.time()) >= int(_PUSH_BLOCK_UNTIL)


def _block_push(reason: str):
    global _PUSH_BLOCK_UNTIL
    _PUSH_BLOCK_UNTIL = int(time.time()) + int(LINE_PUSH_COOLDOWN_SECONDS)
    log.warning("[LINE] push disabled temporarily: %s (block_until=%s)", reason, _PUSH_BLOCK_UNTIL)


def _looks_like_429(e: Exception) -> bool:
    s = str(e)
    if "status_code=429" in s:
        return True
    if "reached your monthly limit" in s.lower():
        return True
    if "You have reached your monthly limit" in s:
        return True
    return False


def _trial_key(uid: str, kind: str) -> str:
    # ★ namespace：避免不同 Bot/不同部署共用同一組 trial keys
    return f"trial:{TRIAL_NAMESPACE}:{kind}:{uid}"


# ★ BLOCK-TRIAL：封鎖即永久失效試用
def _trial_block_key(uid: str) -> str:
    return _trial_key(uid, "blocked")


def is_trial_blocked(uid: str) -> bool:
    return _rget(_trial_block_key(uid)) == "1"


def set_trial_blocked(uid: str, flag: bool = True) -> None:
    _rset(_trial_block_key(uid), "1" if flag else "0")


def trial_persist_guard(uid: str) -> Optional[str]:
    """
    檢查試用狀態並返回適當的提示訊息。

    - 若用戶已開通，返回 None。
    - 若曾封鎖，回傳強調帳號曾被封鎖並說明如何重新啟用。
    - 若試用時間已用完，提示試用已結束並提供開通方式與正確格式示例。
    - 若有 expired 標記但未到期或資料髒，會自動修正。
    - 若尚在試用期內或沒有任何限制，返回 None。
    """
    if is_premium(uid):
        return None

    # ★ BLOCK-TRIAL：曾封鎖 → 永久失效（除非開通）
    if is_trial_blocked(uid):
        return (
            f"⛔ 試用已到期（帳號曾被封鎖）\n"
            f"🔐 如需重新啟用，請輸入：開通 你的密碼\n"
            f"👉 範例：開通 abc123\n"
            f"📞 或聯繫：{ADMIN_CONTACT}"
        )

    now = int(time.time())
    first_ts = _rget(_trial_key(uid, "first_ts"))
    expired = _rget(_trial_key(uid, "expired"))

    # TRIAL-FIX: expired=1 但 first_ts 不存在 → 視為髒資料，清掉 expired
    if expired == "1" and not first_ts:
        _rset(_trial_key(uid, "expired"), "0")
        expired = None

    if not first_ts:
        # 新用戶：寫入 first_ts 並確保 expired 被清掉
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

    # TRIAL-FIX: expired=1 但其實還沒到期 → 自動修正
    if expired == "1" and used_min < TRIAL_MINUTES:
        _rset(_trial_key(uid, "expired"), "0")
        expired = None

    if used_min >= TRIAL_MINUTES:
        _rset(_trial_key(uid, "expired"), "1")
        return (
            f"⏰ 免費試用 {TRIAL_MINUTES} 分鐘已用完\n"
            f"🎯 想繼續使用嗎？\n"
            f"🔐 請輸入：開通 你的專屬密碼\n"
            f"👉 正確格式：開通 [密碼]\n"
            f"📞 沒有密碼？請聯繫：{ADMIN_CONTACT}"
        )

    if expired == "1":
        return (
            f"⛔ 試用已到期\n"
            f"🔐 請輸入：開通 你的專屬密碼\n"
            f"👉 正確格式：開通 [密碼]\n"
            f"📞 沒有密碼？請聯繫：{ADMIN_CONTACT}"
        )

    return None


def validate_activation_code(code: str) -> bool:
    if not code:
        return False
    norm = str(code).replace("\u3000", " ").replace("：", ":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and (norm == ADMIN_ACTIVATION_SECRET)


GAMES = {"1": "WM", "2": "PM", "3": "DG", "4": "SA", "5": "KU", "6": "歐博/卡利", "7": "KG", "8": "全利",
         "9": "名人", "10": "MT真人"}


def game_menu_text(left_min: int) -> str:
    lines = ["請選擇遊戲館別"]
    for k in sorted(GAMES.keys(), key=lambda x: int(x)):
        lines.append(f"{k}. {GAMES[k]}")
    lines.append("「請直接輸入數字選擇」")
    lines.append(f"⏳ 試用剩餘 {left_min} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
    return "\n".join(lines)


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
        log.warning("[heavy] line_api is None, skip heavy prediction.")
        return

    start = time.time()
    try:
        from linebot.models import TextSendMessage

        sess = get_session(uid)
        if (p_pts == b_pts and SKIP_TIE_UPD):
            sess["last_pts_text"] = "上局結果: 和局"
        else:
            sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"

        probs, choice, bet_amt, reason = _handle_points_and_predict(uid, sess, p_pts, b_pts)

        msg = format_output_card(probs, choice, sess.get("last_pts_text"), bet_amt,
                                 cont=bool(CONTINUOUS_MODE))

        cur_seq = int(sess.get("pending_seq", 0))
        if cur_seq == int(seq):
            sess["last_card"] = msg
            sess["last_card_ts"] = int(time.time())
            sess["pending"] = False
        else:
            log.info("[heavy] stale seq=%s (cur_seq=%s) skip write-back", seq, cur_seq)

        save_session(uid, sess)

        if _can_push():
            try:
                line_api.push_message(
                    uid,
                    TextSendMessage(text=msg, quick_reply=_quick_buttons())
                )
            except Exception as e:
                if _looks_like_429(e):
                    _block_push("429 monthly limit reached")
                log.warning("[LINE] push failed (heavy): %s", e)
        else:
            log.info("[LINE] push skipped (disabled/blocked).")

    except Exception as e:
        log.exception("[heavy] prediction failed: %s", e)
    finally:
        elapsed = time.time() - start
        log.info("[heavy] prediction done in %.2fs (uid=%s, seq=%s)", elapsed, uid, seq)


line_api = None
line_handler = None
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import MessageEvent, TextMessage, FollowEvent, UnfollowEvent
    if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        @line_handler.add(UnfollowEvent)
        def on_unfollow(event):
            # ★ BLOCK-TRIAL：封鎖/取消好友 → 永久失效試用
            if not _dedupe_event(_extract_line_event_id(event)):
                return
            try:
                uid = event.source.user_id
                set_trial_blocked(uid, True)
                _rset(_trial_key(uid, "expired"), "1")
                log.info("[TRIAL] user unfollowed -> blocked=1 expired=1 uid=%s", uid)
            except Exception as e:
                log.warning("[TRIAL] unfollow handler error: %s", e)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            if not _dedupe_event(_extract_line_event_id(event)):
                return
            uid = event.source.user_id

            # ★ 若曾封鎖 → 直接視為到期（不重給試用）
            if (not is_premium(uid)) and is_trial_blocked(uid):
                sess = get_session(uid)
                # 取得更詳盡的 trial 提示訊息（包括範例格式）
                guard_msg = trial_persist_guard(uid)
                msg = guard_msg if guard_msg else (
                    f"⛔ 試用已到期\n"
                    f"🔐 請輸入：開通 你的密碼\n"
                    f"👉 正確格式：開通 [密碼]\n"
                    f"📞 沒有密碼？請聯繫：{ADMIN_CONTACT}"
                )
                _reply(line_api, event.reply_token, msg)
                save_session(uid, sess)
                return

            # TRIAL-FIX: 加好友當下，若 first_ts 不存在 → 強制建立並清 expired
            now = int(time.time())
            ft_key = _trial_key(uid, "first_ts")
            ex_key = _trial_key(uid, "expired")
            first_ts = _rget(ft_key)
            if not first_ts:
                _rset(ft_key, str(now))
                _rset(ex_key, "0")
                first_ts = str(now)
            else:
                try:
                    first = int(first_ts)
                    used_min = (now - first) // 60
                    if _rget(ex_key) == "1" and used_min < TRIAL_MINUTES:
                        _rset(ex_key, "0")
                except Exception:
                    _rset(ft_key, str(now))
                    _rset(ex_key, "0")
                    first_ts = str(now)

            guard_msg = trial_persist_guard(uid)
            sess = get_session(uid)

            try:
                sess["trial_start"] = int(first_ts) if first_ts else int(time.time())
            except Exception:
                pass

            if sess.get("premium", False) or is_premium(uid):
                msg = (
                    "👋 歡迎回來，已是永久開通用戶。\n"
                    "輸入『遊戲設定』開始；連續模式啟動後只需輸入點數（例：65 / 和 / 閒6莊5）即可預測。"
                )
            else:
                if guard_msg:
                    msg = guard_msg
                else:
                    try:
                        ft = int(first_ts) if first_ts else int(time.time())
                        used_min = max(0, (int(time.time()) - ft) // 60)
                        left = max(0, TRIAL_MINUTES - used_min)
                    except Exception:
                        left = TRIAL_MINUTES
                    msg = (
                        f"👋 歡迎！你有 {left} 分鐘免費試用（共 {TRIAL_MINUTES} 分鐘）。\n"
                        "輸入『遊戲設定』開始；連續模式啟動後只需輸入點數（例：65 / 和 / 閒6莊5）即可預測。"
                    )

            _reply(line_api, event.reply_token, msg)
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            if not _dedupe_event(_extract_line_event_id(event)):
                return
            uid = event.source.user_id
            raw = (event.message.text or "")
            text = re.sub(r"\s+", " ", raw.replace("\u3000", " ").strip())
            sess = get_session(uid)
            up = text.upper()

            if up.startswith("開通") or up.startswith("ACTIVATE"):
                after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                ok = validate_activation_code(after)
                if ok:
                    sess["premium"] = True
                    set_premium(uid, True)
                    # ★ BLOCK-TRIAL：開通成功 → 解除 blocked
                    try:
                        set_trial_blocked(uid, False)
                    except Exception:
                        pass
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
                sess = {"phase": "await_pts", "bankroll": 0, "rounds_seen": 0,
                        "last_pts_text": None, "premium": premium, "trial_start": start_ts,
                        "last_card": None, "last_card_ts": None,
                        "pending": False, "pending_seq": 0}
                try:
                    reset_pf_for_uid(uid)
                except Exception:
                    pass
                _reply(line_api, event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。")
                save_session(uid, sess)
                return

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
                if m and (m.group(1) in GAMES):
                    sess["game"] = GAMES[m.group(1)]
                    sess["phase"] = "input_bankroll"
                    _reply(line_api, event.reply_token,
                           f"🎰 已選擇：{sess['game']}，請輸入初始籌碼（金額）")
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
                    f"✅ 設定完成！館別：{sess.get('game')}，初始籌碼：{amt}。\n📌 連續模式：現在輸入第一局點數（例：閒6莊5 / 65 / 和）"
                )
                save_session(uid, sess)
                return

            pts = parse_last_hand_points(text)
            if pts and sess.get("bankroll", 0) >= 0:
                p_pts, b_pts = pts

                _reply(
                    line_api,
                    event.reply_token,
                    "✅ 已收到上一局結果，AI 正在計算。"
                )

                sess["pending"] = True
                sess["pending_seq"] = int(sess.get("pending_seq", 0)) + 1
                seq = int(sess["pending_seq"])
                sess["last_card"] = None
                sess["last_card_ts"] = None
                save_session(uid, sess)

                try:
                    threading.Thread(
                        target=_push_heavy_prediction,
                        args=(uid, p_pts, b_pts, seq),
                        daemon=True,
                    ).start()
                except Exception as e:
                    log.exception("failed to spawn heavy prediction thread: %s", e)
                return

            _reply(
                line_api,
                event.reply_token,
                "指令無法辨識。\n📌 直接輸入點數（例：65 / 和 / 閒6莊5），或輸入『遊戲設定』。"
            )
except Exception as e:
    log.warning("LINE not fully configured: %s", e)


def _handle_line_webhook():
    if 'line_handler' not in globals() or line_handler is None:
        log.error("webhook called but LINE handler not ready (missing credentials?)")
        abort(400, "LINE handler not ready")

    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except Exception as e:
        log.error("webhook error: %s", e)
        abort(500)
    return "OK", 200


@app.post("/line-webhook")
def line_webhook():
    return _handle_line_webhook()


@app.route("/line-webhook", methods=["OPTIONS"])
def line_webhook_options():
    return ("", 204, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Line-Signature",
    })


@app.post("/callback")
def line_webhook_callback():
    return _handle_line_webhook()


@app.route("/callback", methods=["OPTIONS"])
def line_webhook_callback_options():
    return ("", 204, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Line-Signature",
    })


@app.get("/")
def root():
    ua = request.headers.get("User-Agent", "")
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
        prob_force_pure_in_prob_mode=bool(PROB_FORCE_PURE_IN_PROB_MODE),
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
        card = format_output_card(probs, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE))
        return jsonify(
            ok=True,
            probs=[float(probs[0]), float(probs[1]), float(probs[2])],
            choice=choice, bet=bet_amt, reason=reason, card=card
        ), 200
    except Exception as e:
        log.exception("predict error: %s", e)
        return jsonify(ok=False, error=str(e)), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    if OutcomePF is None:
        log.warning("PF backend: smart-dummy (OutcomePF import failed). If probs look repeated, check deployment paths.")
    else:
        log.info("PF backend: %s (OutcomePF available)", PF_BACKEND)

    log.info(
        "Starting %s on port %s (PF_INIT=%s, DEPLETE_OK=%s, MODE=%s, COMPAT=%s, DEPL=%s, TRIAL_NS=%s, "
        "PF_STATEFUL=%s, TIE_CAP_ENABLE=%s, PROB_FORCE_PURE_IN_PROB_MODE=%s, PROB_PURE_MODE=%s, EV_NEUTRAL=%s, PROB_BIAS_B2P=%.6f)",
        VERSION, port, pf_initialized, DEPLETE_OK, DECISION_MODE, COMPAT_MODE, DEPL_ENABLE, TRIAL_NAMESPACE,
        PF_STATEFUL, TIE_CAP_ENABLE, PROB_FORCE_PURE_IN_PROB_MODE, PROB_PURE_MODE, EV_NEUTRAL, float(PROB_BIAS_B2P)
    )

    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
