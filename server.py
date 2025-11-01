# -*- coding: utf-8 -*-
"""server.py — BGS Independent PF + Deplete + Debug (2025-11-02)
- PF_DEBUG：詳細列印 PF/Deplete/混合/平滑的機率向量
- 自動 Dep 順序對齊；可用 DEPLETE_RETURNS_PBT 強制
- 全路徑 Tie 夾限（CLAMP_TIE）
- prob 模式加入 PROB_GAP_ENTER 最小差距門檻
"""
import os, sys, logging, time, re, json
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

# ---------------- Env switches ----------------
THEO_BLEND       = float(os.getenv("THEO_BLEND", "0.0"))
PF_DEBUG         = 1 if str(os.getenv("PF_DEBUG", "0")).strip().lower() in ("1","t","true","yes","on") else 0
CLAMP_TIE        = 1 if str(os.getenv("CLAMP_TIE", "1")).strip().lower() in ("1","t","true","yes","on") else 0
TIE_MIN          = float(os.getenv("TIE_MIN", "0.05"))
TIE_MAX          = float(os.getenv("TIE_MAX", "0.15"))
PROB_GAP_ENTER   = float(os.getenv("PROB_GAP_ENTER", "0.0035"))  # 只在 DECISION_MODE=prob 生效
SOFT_TAU         = float(os.getenv("SOFT_TAU", "2.0"))

def _nv(a): return np.asarray(a, dtype=np.float32)

def _soft_tau(p: np.ndarray) -> np.ndarray:
    q = p.astype(np.float32)
    q = q ** (1.0 / max(1e-6, SOFT_TAU))
    q = q / max(1e-12, q.sum())
    return q

def _clamp_tie(p: np.ndarray) -> np.ndarray:
    if not CLAMP_TIE: return p
    b, pl, t = float(p[0]), float(p[1]), float(p[2])
    t0 = t
    if t < TIE_MIN:
        t = TIE_MIN
        scale = (1.0 - t) / max(1e-12, 1.0 - t0)
        b, pl = b*scale, pl*scale
    elif t > TIE_MAX:
        t = TIE_MAX
        scale = (1.0 - t) / max(1e-12, 1.0 - t0)
        b, pl = b*scale, pl*scale
    out = np.array([b, pl, t], dtype=np.float32)
    out = out / max(1e-12, out.sum())
    return out

def smooth_probs(prob: np.ndarray) -> np.ndarray:
    if THEO_BLEND <= 0.0:
        return prob
    theo = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
    sm = (1.0 - THEO_BLEND) * prob + THEO_BLEND * theo
    sm = sm / max(1e-12, sm.sum())
    return sm

# --------- Safe import deplete ----------
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
            if _cur_dir not in sys.path: sys.path.insert(0, _cur_dir)
            from deplete import init_counts, probs_after_points  # type: ignore
            DEPLETE_OK = True
        except Exception:
            DEPLETE_OK = False

try:
    import redis
except Exception:
    redis = None

try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
    _flask_available = True
except Exception:
    _flask_available = False
    Flask = None; request = None
    def jsonify(*args, **kwargs): raise RuntimeError("Flask not available")
    def abort(*args, **kwargs): raise RuntimeError("Flask not available")
    def CORS(app): return None

VERSION = "bgs-independent-2025-11-02+pfdebug+tieclamp+probgap"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

if not DEPLETE_OK:
    log.warning("deplete 模組未找到；以 PF 單模預測運行。")

# ------------- Flask -------------
if _flask_available and Flask is not None:
    app = Flask(__name__); CORS(app)
else:
    class _DummyApp:
        def get(self, *a, **k):
            def _d(f): return f
            return _d
        def post(self, *a, **k):
            def _d(f): return f
            return _d
        def run(self, *a, **k): log.warning("Dummy app running is not supported")
    app = _DummyApp()

# ------------- Redis / Session -------------
REDIS_URL = os.getenv("REDIS_URL")
redis_client: Optional["redis.Redis"] = None
if redis is not None and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Connected Redis.")
    except Exception as e:
        log.error("Redis connect fail: %s -> in-memory", e)
        redis_client = None
else:
    if redis is None: log.warning("redis module missing -> in-memory")
    elif not REDIS_URL: log.warning("REDIS_URL not set -> in-memory")

SESS_FALLBACK: Dict[str, Dict[str, Any]] = {}
KV_FALLBACK: Dict[str, str] = {}
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1200"))
DEDUPE_TTL = 60

def _rget(k: str) -> Optional[str]:
    try:
        if redis_client: return redis_client.get(k)
        return KV_FALLBACK.get(k)
    except Exception as e:
        log.warning("[Redis] GET err: %s", e); return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if redis_client: redis_client.set(k, v, ex=ex)
        else: KV_FALLBACK[k] = v
    except Exception as e:
        log.warning("[Redis] SET err: %s", e)

def _rsetnx(k: str, v: str, ex: int) -> bool:
    try:
        if redis_client: return bool(redis_client.set(k, v, ex=ex, nx=True))
        if k in KV_FALLBACK: return False
        KV_FALLBACK[k] = v; return True
    except Exception as e:
        log.warning("[Redis] SETNX err: %s", e); return True

def get_session(uid: str) -> Dict[str, Any]:
    if redis_client:
        j = _rget(f"bgs_session:{uid}")
        if j:
            try: return json.loads(j)
            except Exception: pass
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
        "bankroll": 0, "trial_start": nowi, "premium": False,
        "phase": "choose_game", "game": None, "table": None,
        "last_pts_text": None, "table_no": None, "streak_count": 0,
        "last_outcome": None, "hand_count": 0, "prob_sma": None,
    }

def save_session(uid: str, data: Dict[str, Any]):
    if redis_client: _rset(f"bgs_session:{uid}", json.dumps(data), ex=SESSION_EXPIRE_SECONDS)
    else: SESS_FALLBACK[uid] = data

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except Exception:
        return 1 if default else 0

# -------- 階段參數 --------
def get_stage_params(hand_count: int) -> Dict[str, str]:
    if hand_count <= 20: stage = "EARLY"
    elif hand_count <= 60: stage = "MID"
    else: stage = "LATE"
    params = {}
    for param in ["PF_PRED_SIMS","PF_UPD_SIMS","PF_RESAMPLE","PF_DECAY","PF_NOISE","PF_DIR_EPS","DEPLETEMC_SIMS","DEPL_FACTOR"]:
        params[param] = os.getenv(f"{stage}_{param}", os.getenv(param, ""))
    return params

def apply_session_ema_smoothing(current_prob: np.ndarray, session: Dict[str, Any], outcome: int) -> np.ndarray:
    PROB_SMA_ALPHA = float(os.getenv("PROB_SMA_ALPHA", "0.0"))  # 預設 0.0，避免未設造成黏性
    SKIP_TIE_UPD = env_flag("SKIP_TIE_UPD", 1)
    if PROB_SMA_ALPHA <= 0.0: return current_prob
    prev = session.get("prob_sma")
    if outcome == 2 and SKIP_TIE_UPD and prev is not None:
        return np.array(prev, dtype=np.float32)
    if prev is None: sm = current_prob
    else:
        prev = np.array(prev, dtype=np.float32)
        sm = PROB_SMA_ALPHA * current_prob + (1 - PROB_SMA_ALPHA) * prev
        sm = sm / max(1e-12, sm.sum())
    session["prob_sma"] = sm.tolist()
    return sm

# -------- 解析點數 --------
def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
    if not text: return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s).replace("\u3000", " ")
    u = re.sub(r"^開始分析", "", s.upper().strip())
    m = re.search(r"(?:和|TIE|DRAW)\s*:?:?\s*(\d)?", u)
    if m:
        d = m.group(1); return (int(d), int(d)) if d else (0, 0)
    m = re.search(r"(?:閒|闲|P)\s*:?:?\s*(\d)\D+(?:莊|庄|B)\s*:?:?\s*(\d)", u)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|庄|B)\s*:?:?\s*(\d)\D+(?:閒|闲|P)\s*:?:?\s*(\d)", u)
    if m: return (int(m.group(2)), int(m.group(1)))
    t = u.replace(" ", "").replace("\u3000", "")
    if t in ("B","莊","庄"): return (0,1)
    if t in ("P","閒","闲"): return (1,0)
    if t in ("T","和"): return (0,0)
    if re.search(r"[A-Z]", u): return None
    d = re.findall(r"\d", u)
    if len(d) == 2: return (int(d[0]), int(d[1]))
    return None

# -------- 試用鎖 --------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "aaa8881688")
def _trial_key(uid: str, kind: str) -> str: return f"trial:{kind}:{uid}"
def trial_persist_guard(uid: str) -> Optional[str]:
    now = int(time.time()); first_ts = _rget(_trial_key(uid, "first_ts")); expired = _rget(_trial_key(uid, "expired"))
    if expired == "1": return f"⛔ 試用已到期\n📬 管理員：{ADMIN_CONTACT}\n🔐 在此輸入：開通 你的密碼"
    if not first_ts: _rset(_trial_key(uid, "first_ts"), str(now)); return None
    try: first = int(first_ts)
    except: first = now; _rset(_trial_key(uid, "first_ts"), str(now))
    used_min = (now - first) // 60
    if used_min >= TRIAL_MINUTES:
        _rset(_trial_key(uid, "expired"), "1")
        return f"⛔ 試用已到期\n📬 管理員：{ADMIN_CONTACT}\n🔐 在此輸入：開通 你的密碼"
    return None
def validate_activation_code(code: str) -> bool:
    if not code: return False
    norm = str(code).replace("\u3000"," ").replace("：",":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and (norm == ADMIN_ACTIVATION_SECRET)

# -------- PF ----------
log.info("載入 PF 參數: PF_N=%s, PF_UPD_SIMS=%s, PF_PRED_SIMS=%s, DECKS=%s",
         os.getenv("PF_N", "50"), os.getenv("PF_UPD_SIMS", "30"),
         os.getenv("PF_PRED_SIMS", "5"), os.getenv("DECKS", "8"))

PF_BACKEND = (os.getenv("PF_BACKEND", "mc") or "mc").lower()
SKIP_TIE_UPD = env_flag("SKIP_TIE_UPD", 1)
HISTORY_MODE = env_flag("HISTORY_MODE", 0)

OutcomePF = None; PF = None; pf_initialized = False
try:
    from bgs.pfilter import OutcomePF as RealOutcomePF
    OutcomePF = RealOutcomePF; log.info("成功從 bgs.pfilter 導入 OutcomePF")
except Exception:
    try:
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        if _cur_dir not in sys.path: sys.path.insert(0, _cur_dir)
        from pfilter import OutcomePF as LocalOutcomePF
        OutcomePF = LocalOutcomePF; log.info("成功從本地 pfilter 導入 OutcomePF")
    except Exception as pf_exc:
        log.error("無法導入 OutcomePF: %s", pf_exc); OutcomePF = None

if OutcomePF:
    try:
        PF = OutcomePF(
            decks=int(os.getenv("DECKS", "8")), seed=int(os.getenv("SEED", "42")),
            n_particles=int(os.getenv("PF_N", "50")), sims_lik=int(os.getenv("PF_UPD_SIMS", "30")),
            resample_thr=float(os.getenv("PF_RESAMPLE", "0.5")), backend=PF_BACKEND,
            dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.05"))
        )
        pf_initialized = True
        log.info("PF 初始化成功 backend=%s n=%s sims_lik=%s", getattr(PF,'backend','?'),
                 getattr(PF,'n_particles','?'), getattr(PF,'sims_lik','?'))
    except Exception as e:
        log.error("PF 初始化失敗: %s", e); pf_initialized = False; OutcomePF = None

if not pf_initialized:
    class SmartDummyPF:
        def __init__(self): log.warning("使用 SmartDummyPF 備援模式")
        def update_outcome(self, outcome): return
        def update_points(self, p, b): return
        def predict(self, **kwargs) -> np.ndarray:
            base = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
            return _soft_tau(_clamp_tie(base))
        @property
        def backend(self): return "smart-dummy"
    PF = SmartDummyPF(); pf_initialized = True

# -------- 決策參數 --------
EDGE_ENTER        = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY         = env_flag("USE_KELLY", 0)
CONTINUOUS_MODE   = env_flag("CONTINUOUS_MODE", 1)
DECISION_MODE     = (os.getenv("DECISION_MODE", "ev") or "ev").lower()
BANKER_PAYOUT     = float(os.getenv("BANKER_PAYOUT", "0.95"))
PROB_MARGIN       = float(os.getenv("PROB_MARGIN", "0.02"))
MIN_EV_EDGE       = float(os.getenv("MIN_EV_EDGE", "0.0"))
MIN_CONF_FOR_ENTRY= float(os.getenv("MIN_CONF_FOR_ENTRY", "0.56"))
QUIET_SMALLEdge   = env_flag("QUIET_SMALLEdge", 0)
MIN_BET_PCT_ENV   = float(os.getenv("MIN_BET_PCT", "0.05"))
MAX_BET_PCT_ENV   = float(os.getenv("MAX_BET_PCT", "0.40"))
MAX_EDGE_SCALE    = float(os.getenv("MAX_EDGE_FOR_FULLBET", "0.15"))
INV = {0: "莊", 1: "閒"}

def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0: return 0
    return int(round(bankroll * pct))

def _decide_side_by_ev(pB: float, pP: float) -> Tuple[int, float, float, float]:
    evB = BANKER_PAYOUT * pB - pP
    evP = pP - pB
    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))
    return side, final_edge, evB, evP

def _decide_side_by_prob(pB: float, pP: float) -> int:
    return 0 if pB >= pP else 1

def decide_only_bp(prob: np.ndarray) -> Tuple[str, float, float, str]:
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    reason: List[str] = []
    if DECISION_MODE == "prob":
        gap = abs(pB - pP)
        if gap < PROB_GAP_ENTER:
            reason.append(f"模式=prob, gap={gap:.4f} < PROB_GAP_ENTER={PROB_GAP_ENTER:.4f}")
            conf = max(pB, pP)
            return ("觀望", gap, 0.0, "; ".join(reason + [f"conf={conf:.3f}"]))
        side = _decide_side_by_prob(pB, pP)
        _, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
        final_edge = max(abs(evB), abs(evP))  # 用 EV 的幅度當 edge
        reason.append(f"模式=prob (pB={pB:.4f}, pP={pP:.4f}, gap={gap:.4f})")
    elif DECISION_MODE == "hybrid":
        if abs(pB - pP) >= PROB_MARGIN:
            side = _decide_side_by_prob(pB, pP)
            _, final_edge, evB, evP = _decide_side_by_ev(pB, pP)
            reason.append(f"模式=hybrid→prob (Δ={abs(pB-pP):.4f}≥{PROB_MARGIN})")
        else:
            side, final_edge, evB, evP = _decide_side_by_ev(pB, pP)
            if final_edge < MIN_EV_EDGE:
                side = _decide_side_by_prob(pB, pP)
                reason.append(f"模式=hybrid→prob (EV不足 {final_edge:.4f}<{MIN_EV_EDGE})")
    else:  # ev
        side, final_edge, evB, evP = _decide_side_by_ev(pB, pP)
        reason.append(f"模式=ev (EV_B={evB:.4f}, EV_P={evP:.4f}, payout={BANKER_PAYOUT})")
    conf = max(pB, pP)
    if conf < MIN_CONF_FOR_ENTRY:
        reason.append(f"⚪ 信心不足 conf={conf:.3f}<{MIN_CONF_FOR_ENTRY:.2f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason))
    if final_edge < EDGE_ENTER:
        reason.append(f"⚪ 優勢不足 edge={final_edge:.4f}<{EDGE_ENTER:.4f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason))
    if QUIET_SMALLEdge and final_edge < (EDGE_ENTER * 1.2):
        reason.append(f"⚪ 邊際略優(quiet) edge={final_edge:.4f}<{EDGE_ENTER*1.2:.4f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason))
    min_b = max(0.0, min(1.0, MIN_BET_PCT_ENV))
    max_b = max(min_b, min(1.0, MAX_BET_PCT_ENV))
    max_edge = max(EDGE_ENTER + 1e-6, MAX_EDGE_SCALE)
    bet_pct = min_b + (max_b - min_b) * (final_edge - EDGE_ENTER) / (max_edge - EDGE_ENTER)
    bet_pct = float(min(max_b, max(min_b, bet_pct)))
    reason.append(f"信心度配注({int(min_b*100)}%~{int(max_b*100)}%), conf={conf:.3f}")
    return (INV[side], final_edge, bet_pct, "; ".join(reason))

def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int, cont: bool) -> str:
    b_pct_txt = f"{prob[0] * 100:.2f}%"; p_pct_txt = f"{prob[1] * 100:.2f}%"
    header: List[str] = []; 
    if last_pts_text: header.append(last_pts_text)
    header.append("開始分析下局....")
    block = ["預測結果", f"閒：{p_pct_txt}", f"莊：{b_pct_txt}", f"和：{prob[2]*100:.2f}%"]
    if choice == "觀望": block += ["本次預測結果：觀望","建議觀望（不下注）"]
    else:
        block += [f"本次預測結果：{choice}", f"建議下注：{bet_amt:,}"]
    if cont: block.append("\n📌 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）")
    return "\n".join(header + [""] + block)

# ---------- Deplete 順序自動校正 ----------
def _align_dep_order(pf_bpt: np.ndarray, dep_any: np.ndarray) -> np.ndarray:
    def _cos2(a, b):
        a2 = np.asarray([a[0], a[1]], dtype=np.float32); b2 = np.asarray([b[0], b[1]], dtype=np.float32)
        na = np.linalg.norm(a2); nb = np.linalg.norm(b2)
        if na <= 1e-12 or nb <= 1e-12: return -1.0
        return float(np.dot(a2/na, b2/nb))
    cand1 = dep_any
    cand2 = dep_any[[1,0,2]]
    s1 = _cos2(pf_bpt, cand1); s2 = _cos2(pf_bpt, cand2)
    chosen = cand1 if s1 >= s2 else cand2
    if PF_DEBUG:
        log.info("Dep對齊評分: keep=%.4f swap=%.4f -> %s", s1, s2, "swap" if s2>s1 else "keep")
    return chosen

# ---------- 核心處理 ----------
def _handle_points_and_predict(sess: Dict[str, Any], p_pts: int, b_pts: int, reply_token: str):
    start_time = time.time()
    outcome = 2 if p_pts == b_pts else (1 if p_pts > b_pts else 0)
    if outcome != 2: sess["hand_count"] = int(sess.get("hand_count", 0)) + 1
    sess["last_pts_text"] = "上局結果: 和局" if outcome == 2 else f"上局結果: 閒 {p_pts} 莊 {b_pts}"
    sess["last_outcome"] = outcome; sess["streak_count"] = 1 if outcome in (0,1) else 0; sess["phase"] = "ready"

    try:
        hand_count = int(sess.get("hand_count", 0))
        stage_params = get_stage_params(hand_count)
        feed_mode = (os.getenv("PF_FEED_MODE", "points") or "points").lower()
        pf_w = float(os.getenv("PF_WEIGHT", "0.7")); dep_w = float(os.getenv("DEPLETE_WEIGHT", "0.3"))
        s = max(1e-9, pf_w + dep_w); pf_w, dep_w = pf_w/s, dep_w/s

        # PF 餵資料
        try:
            if feed_mode == "points" and hasattr(PF, "update_points"):
                PF.update_points(int(p_pts), int(b_pts))
                if PF_DEBUG: log.info("PF.update_points(P=%d,B=%d)", p_pts, b_pts)
            elif feed_mode == "outcome" and hasattr(PF, "update_outcome"):
                PF.update_outcome(outcome)
                if PF_DEBUG: log.info("PF.update_outcome(outcome=%d)", outcome)
            else:
                if PF_DEBUG: log.info("PF feed skipped (mode=%s)", feed_mode)
        except Exception as e:
            log.warning("PF 更新狀態失敗(feed=%s)：%s", feed_mode, e)

        # PF 預測
        try:
            pf_pred_sims = int(stage_params.get("PF_PRED_SIMS", os.getenv("PF_PRED_SIMS", "5")) or "5")
        except Exception:
            pf_pred_sims = int(os.getenv("PF_PRED_SIMS", "5"))
        try:
            pf_preds = PF.predict(sims_per_particle=pf_pred_sims, obs_pts=(int(p_pts), int(b_pts)))
        except TypeError:
            pf_preds = PF.predict(sims_per_particle=pf_pred_sims)
        except Exception as e:
            log.warning("PF.predict 失敗，使用先驗：%s", e)
            pf_preds = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
        pf_preds = _nv(pf_preds); pf_preds /= max(1e-12, pf_preds.sum())
        pf_preds = _soft_tau(_clamp_tie(pf_preds))  # 和路徑一致：先夾 Tie 再 SoftTau
        if PF_DEBUG: log.info("PF機率: B=%.4f P=%.4f T=%.4f (sims=%d, backend=%s)", pf_preds[0], pf_preds[1], pf_preds[2], pf_pred_sims, getattr(PF,'backend','?'))

        # Deplete
        mix = pf_preds.copy()
        if DEPLETE_OK and init_counts and probs_after_points and dep_w > 1e-6:
            try:
                counts = init_counts(int(os.getenv("DECKS", "8")))
                deplete_sims = int(stage_params.get("DEPLETEMC_SIMS", os.getenv("DEPLETEMC_SIMS", "1000")) or "1000")
                deplete_factor = float(stage_params.get("DEPL_FACTOR", os.getenv("DEPL_FACTOR", "1.0")) or "1.0")
                dep_raw = _nv(probs_after_points(counts, int(p_pts), int(b_pts), sims=deplete_sims, deplete_factor=deplete_factor))
                dep_flag = (os.getenv("DEPLETE_RETURNS_PBT", "auto") or "auto").lower()
                if dep_flag in ("1","true","yes","on"): dep = dep_raw[[1,0,2]]
                elif dep_flag in ("0","false","no","off"): dep = dep_raw
                else: dep = _align_dep_order(pf_preds, dep_raw)
                dep = dep / max(1e-12, dep.sum())
                dep = _soft_tau(_clamp_tie(dep))
                if PF_DEBUG: log.info("Deplete機率: B=%.4f P=%.4f T=%.4f (sims=%d, factor=%.2f)", dep[0], dep[1], dep[2], deplete_sims, deplete_factor)
                mix = pf_w * pf_preds + dep_w * dep
                mix = mix / max(1e-12, mix.sum())
                if PF_DEBUG: log.info("PF/Dep混合(%.2f/%.2f): B=%.4f P=%.4f T=%.4f", pf_w, dep_w, mix[0], mix[1], mix[2])
            except Exception as e:
                log.warning("Deplete 失敗，使用 PF 單模：%s", e)

        # 後處理：理論混合 + 會話 EMA
        p_theo  = smooth_probs(mix)
        p_final = apply_session_ema_smoothing(p_theo, sess, outcome)
        if PF_DEBUG:
            log.info("後處理: theo B=%.4f P=%.4f T=%.4f | final B=%.4f P=%.4f T=%.4f",
                     p_theo[0], p_theo[1], p_theo[2], p_final[0], p_final[1], p_final[2])

        # 決策
        choice, edge, bet_pct, reason = decide_only_bp(p_final)
        bankroll_now = int(sess.get("bankroll", 0))
        bet_amt = bet_amount(bankroll_now, bet_pct)
        from linebot.models import TextSendMessage  # lazy import for typing
        msg = format_output_card(p_final, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE))
        _reply(reply_token, msg)
        log.info("決策: %s edge=%.4f pct=%.2f%% | %s", choice, edge, bet_pct*100, reason)
        log.info("完整處理完成, 耗時: %.2fs (手數: %d)", time.time() - start_time, hand_count)
    except Exception as e:
        log.error("預測過程錯誤: %s", e); _reply(reply_token, "⚠️ 預計算錯誤，請稍後再試")
    if CONTINUOUS_MODE: sess["phase"] = "await_pts"

# ---- LINE handler / webhook ----
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None; line_handler = None
if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import MessageEvent, TextMessage, FollowEvent, QuickReply, QuickReplyButton, MessageAction, TextSendMessage
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        GAMES = {"1":"WM","2":"PM","3":"DG","4":"SA","5":"KU","6":"歐博/卡利","7":"KG","8":"全利","9":"名人","10":"MT真人"}
        def _quick_buttons():
            try:
                items = [
                    QuickReplyButton(action=MessageAction(label="遊戲設定 🎮", text="遊戲設定")),
                    QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
                    QuickReplyButton(action=MessageAction(label="報莊勝 🅱️", text="B")),
                    QuickReplyButton(action=MessageAction(label="報閒勝 🅿️", text="P")),
                    QuickReplyButton(action=MessageAction(label="報和局 ⚪", text="T")),
                ]
                if CONTINUOUS_MODE == 0:
                    items.insert(0, QuickReplyButton(action=MessageAction(label="開始分析 ▶️", text="開始分析")))
                return QuickReply(items=items)
            except Exception:
                return None

        def _dedupe_event(event_id: Optional[str]) -> bool:
            if not event_id: return True
            return _rsetnx(f"dedupe:{event_id}", "1", DEDUPE_TTL)

        def _reply(token: str, text: str):
            try: line_api.reply_message(token, TextSendMessage(text=text, quick_reply=_quick_buttons()))
            except Exception as e: log.warning("[LINE] reply failed: %s", e)

        # expose _reply for core
        globals()['_reply'] = _reply

        @line_handler.add(FollowEvent)
        def on_follow(event):
            if not _dedupe_event(getattr(event, "id", None)): return
            uid = event.source.user_id
            _ = trial_persist_guard(uid)
            sess = get_session(uid)
            _reply(event.reply_token, "👋 歡迎！輸入『遊戲設定』開始；已啟用連續模式（直接輸入點數：65 / 和 / 閒6莊5）。")
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            if not _dedupe_event(getattr(event, "id", None)): return
            uid = event.source.user_id
            raw = (event.message.text or "")
            text = re.sub(r"\s+", " ", raw.replace("\u3000", " ").strip())
            sess = get_session(uid)
            try:
                up = text.upper()
                if up.startswith("開通") or up.startswith("ACTIVATE"):
                    after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                    ok = validate_activation_code(after)
                    if ok: _rset(_trial_key(uid, "expired"), "0")
                    sess["premium"] = bool(ok)
                    _reply(event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤"); save_session(uid, sess); return

                guard = trial_persist_guard(uid)
                if guard and not sess.get("premium", False):
                    _reply(event.reply_token, guard); save_session(uid, sess); return

                if up in ("結束分析","清空","RESET"):
                    premium = sess.get("premium", False); start_ts = sess.get("trial_start", int(time.time()))
                    sess = get_session(uid); sess["premium"]=premium; sess["trial_start"]=start_ts
                    sess["hand_count"]=0; sess["prob_sma"]=None
                    _reply(event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。"); save_session(uid, sess); return

                if text == "遊戲設定" or up == "GAME SETTINGS":
                    sess["phase"]="choose_game"; sess["game"]=None; sess["table"]=None; sess["table_no"]=None
                    sess["bankroll"]=0; sess["streak_count"]=0; sess["last_outcome"]=None; sess["last_pts_text"]=None
                    sess["hand_count"]=0; sess["prob_sma"]=None
                    first_ts = _rget(_trial_key(uid, "first_ts"))
                    if first_ts: used = (int(time.time())-int(first_ts))//60; left = max(0, TRIAL_MINUTES-used)
                    else: left = TRIAL_MINUTES
                    lines=["請選擇遊戲館別"]+[f"{k}. {v}" for k,v in sorted(GAMES.items(), key=lambda kv:int(kv[0]))]+["「請直接輸入數字選擇」",f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）"]
                    _reply(event.reply_token, "\n".join(lines)); save_session(uid, sess); return

                if sess.get("phase") == "choose_game":
                    m = re.match(r"^\s*(\d+)", text)
                    if m:
                        c = m.group(1)
                        if c in GAMES:
                            sess["game"]=GAMES[c]; sess["phase"]="input_bankroll"
                            _reply(event.reply_token, f"🎰 已選擇遊戲館：{sess['game']}\n請輸入初始籌碼（金額）")
                            save_session(uid, sess); return
                        else:
                            _reply(event.reply_token, "⚠️ 無效的選項，請輸入上列數字。"); return
                    else:
                        _reply(event.reply_token, "⚠️ 請直接輸入提供的數字來選擇遊戲館別。"); return

                if sess.get("phase") == "input_bankroll":
                    amount_str = re.sub(r"[^\d]", "", text); amount = int(amount_str) if amount_str else 0
                    if amount <= 0: _reply(event.reply_token, "⚠️ 請輸入正確的數字金額。"); return
                    sess["bankroll"]=amount; sess["phase"]="await_pts"; sess["hand_count"]=0; sess["prob_sma"]=None
                    _reply(event.reply_token, f"✅ 設定完成！遊戲館：{sess.get('game')}，初始籌碼：{amount}。\n📌 連續模式啟動：輸入第一局點數（例：閒6莊5 或 65）。")
                    save_session(uid, sess); return

                pts = parse_last_hand_points(text)
                if pts and sess.get("bankroll"):
                    _handle_points_and_predict(sess, pts[0], pts[1], event.reply_token); save_session(uid, sess); return

                _reply(event.reply_token, "指令無法辨識。\n📌 直接輸入點數（例：65 / 和 / 閒6莊5），或輸入『遊戲設定』。")
            except Exception as e:
                log.exception("on_text err: %s", e)
                try: _reply(event.reply_token, "⚠️ 系統錯誤，稍後再試。")
                except Exception: pass

        @app.post("/line-webhook")
        def line_webhook():
            signature = request.headers.get("X-Line-Signature", ""); body = request.get_data(as_text=True)
            try: line_handler.handle(body, signature)
            except InvalidSignatureError: abort(400, "Invalid signature")
            except Exception as e: log.error("webhook error: %s", e); abort(500)
            return "OK", 200
    except Exception as e:
        log.warning("LINE not fully configured: %s", e)
        def _reply(token: str, text: str): log.info("[noLINE] %s", text)  # fallback
        globals()['_reply'] = _reply
else:
    log.warning("LINE credentials not set. LINE webhook will not be active.")
    def _reply(token: str, text: str): log.info("[noLINE] %s", text)
    globals()['_reply'] = _reply

# ---------- Main ----------
@app.get("/")
def root():
    ua = request.headers.get("User-Agent", "")
    if "UptimeRobot" in ua: return "OK", 200
    status = "OK" if pf_initialized else "BACKUP_MODE"
    return f"✅ BGS Server {status} ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION,
                   pf_initialized=pf_initialized, pf_backend=getattr(PF, 'backend', 'unknown')), 200

@app.get("/healthz")
def healthz():
    return jsonify(ok=True, ts=time.time(), version=VERSION, pf_initialized=pf_initialized), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on %s | MODE=%s PF_DEBUG=%s CLAMP_TIE=%s THEO_BLEND=%.3f",
             VERSION, port, DECISION_MODE, PF_DEBUG, CLAMP_TIE, THEO_BLEND)
    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
