# -*- coding: utf-8 -*-
"""server.py — BGS Independent + Stage Overrides + FULL LINE Flow + Compatibility (2025-11-03)"""

import os, sys, logging, time, re, json
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

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
    def jsonify(*args, **kwargs): raise RuntimeError("Flask not available")
    def abort(*args, **kwargs): raise RuntimeError("Flask not available")
    def CORS(app): return None

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
        if redis_client: return redis_client.get(k)
        return KV_FALLBACK.get(k)
    except Exception as e:
        log.warning("[Redis] GET err: %s", e)
        return None

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

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try: return 1 if int(float(v)) != 0 else 0
    except Exception: return 1 if default else 0

# ---------- 事件去重 ----------
def _dedupe_event(event_id: Optional[str]) -> bool:
    """避免 LINE webhook 同一事件重複處理；True=首次處理。"""
    if not event_id: return True
    key = f"dedupe:{event_id}"
    return _rsetnx(key, "1", ex=DEDUPE_TTL)

# ---------- 簡易 Session 層（補上 get_session/save_session） ----------
def _sess_key(uid: str) -> str:
    return f"sess:{uid}"

def get_session(uid: str) -> Dict[str, Any]:
    """讀取使用者 session；若不存在則建立預設值。"""
    if not uid: uid = "anon"
    try:
        if redis_client:
            raw = redis_client.get(_sess_key(uid))
            if raw:
                return json.loads(raw)
        sess = SESS_FALLBACK.get(uid)
        if isinstance(sess, dict):
            return sess
    except Exception as e:
        log.warning("get_session error: %s", e)

    # 預設 session
    sess = {
        "phase": "await_pts",
        "bankroll": 0,
        "rounds_seen": 0,
        "last_pts_text": None,
        "premium": False,
        "trial_start": int(time.time())
    }
    save_session(uid, sess)
    return sess

def save_session(uid: str, sess: Dict[str, Any]) -> None:
    """儲存使用者 session，帶過期時間。"""
    if not uid: uid = "anon"
    try:
        payload = json.dumps(sess, ensure_ascii=False)
        if redis_client:
            redis_client.set(_sess_key(uid), payload, ex=SESSION_EXPIRE_SECONDS)
        else:
            SESS_FALLBACK[uid] = sess
            # 在記憶體層做簡單 TTL：寫一個 shadow key 用於 GC
            KV_FALLBACK[_sess_key(uid)+":ttl"] = str(int(time.time()) + SESSION_EXPIRE_SECONDS)
    except Exception as e:
        log.warning("save_session error: %s", e)

# ---------- 輸出卡片（LINE 文字） ----------
def format_output_card(probs: np.ndarray, choice: str, last_pts: Optional[str],
                       bet_amt: int, cont: bool = True) -> str:
    pB, pP, pT = [float(x) for x in probs]
    lines = []
    if last_pts: lines.append(str(last_pts))
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
VERSION = "bgs-independent-2025-11-03+stage+LINE+compat"

# ---------- Flask App ----------
if _flask_available and Flask is not None:
    app = Flask(__name__)
    CORS(app)
else:
    class _DummyApp:
        def get(self, *a, **k):
            def _d(f): return f
            return _d
        def post(self, *a, **k):
            def _d(f): return f
            return _d
        def run(self, *a, **k):
            log.warning("Flask not available; cannot run HTTP server.")
    app = _DummyApp()

# ---------- PF（Outcome PF） ----------
PF_BACKEND = os.getenv("PF_BACKEND", "mc").lower()
SKIP_TIE_UPD = env_flag("SKIP_TIE_UPD", 1)
SOFT_TAU = float(os.getenv("SOFT_TAU", "2.0"))
TIE_MIN = float(os.getenv("TIE_MIN", "0.05"))
TIE_MAX = float(os.getenv("TIE_MAX", "0.15"))  # 全域預設；分段可覆蓋
HISTORY_MODE = env_flag("HISTORY_MODE", 0)

OutcomePF = None
PF = None
pf_initialized = False

try:
    from bgs.pfilter import OutcomePF as RealOutcomePF
    OutcomePF = RealOutcomePF
    log.info("成功從 bgs.pfilter 導入 OutcomePF")
except Exception:
    try:
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        if _cur_dir not in sys.path: sys.path.insert(0, _cur_dir)
        from pfilter import OutcomePF as LocalOutcomePF
        OutcomePF = LocalOutcomePF
        log.info("成功從本地 pfilter 導入 OutcomePF")
    except Exception as pf_exc:
        log.error("無法導入 OutcomePF: %s", pf_exc)
        OutcomePF = None

if OutcomePF:
    try:
        PF = OutcomePF(
            decks=int(os.getenv("DECKS", "8")),
            seed=int(os.getenv("SEED", "42")),
            n_particles=int(os.getenv("PF_N", "50")),
            sims_lik=int(os.getenv("PF_UPD_SIMS", "30")),
            resample_thr=float(os.getenv("PF_RESAMPLE", "0.5")),
            backend=PF_BACKEND,
            dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.05"))
        )
        pf_initialized = True
        log.info("PF 初始化成功: n_particles=%s, sims_lik=%s, decks=%s (backend=%s)",
                 getattr(PF, 'n_particles', 'N/A'),
                 getattr(PF, 'sims_lik', 'N/A'),
                 getattr(PF, 'decks', 'N/A'),
                 getattr(PF, 'backend', 'unknown'))
    except Exception as e:
        log.error("PF 初始化失敗: %s", e)
        pf_initialized = False
        OutcomePF = None

if not pf_initialized:
    class SmartDummyPF:
        def __init__(self): log.warning("使用 SmartDummyPF 備援模式")
        def update_outcome(self, outcome): return
        def predict(self, **kwargs) -> np.ndarray:
            base = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
            base = base ** (1.0 / max(1e-6, SOFT_TAU)); base = base / base.sum()
            pT = float(base[2])
            if pT < TIE_MIN:
                base[2] = TIE_MIN
                sc = (1.0 - TIE_MIN) / (1.0 - pT) if pT < 1.0 else 1.0
                base[0] *= sc; base[1] *= sc
            elif pT > TIE_MAX:
                base[2] = TIE_MAX
                sc = (1.0 - TIE_MAX) / (1.0 - pT) if pT < 1.0 else 1.0
                base[0] *= sc; base[1] *= sc
            return base.astype(np.float32)
        @property
        def backend(self): return "smart-dummy"
    PF = SmartDummyPF(); pf_initialized = True

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

# ---- Compatibility switches ----
COMPAT_MODE = int(os.getenv("COMPAT_MODE", "0"))  # 1 = 回到純 PF
DEPL_ENABLE = int(os.getenv("DEPL_ENABLE", "0"))  # 1 = 允許 deplete；0 = 禁用

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
        side = _decide_side_by_prob(pB, pP)
        _, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
        final_edge = max(abs(evB), abs(evP)); reason.append("模式=prob")
    elif DECISION_MODE == "hybrid":
        if abs(pB - pP) >= PROB_MARGIN:
            side = _decide_side_by_prob(pB, pP)
            _, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
            final_edge = max(abs(evB), abs(evP)); reason.append("模式=hybrid→prob")
        else:
            s2, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
            if edge_ev >= MIN_EV_EDGE:
                side = s2; final_edge = edge_ev; reason.append("模式=hybrid→ev")
            else:
                side = _decide_side_by_prob(pB, pP); final_edge = edge_ev; reason.append("模式=hybrid→prob(EV不足)")
    else:
        side, final_edge, evB, evP = _decide_side_by_ev(pB, pP); reason.append("模式=ev")

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

# ---------- 三段覆蓋（EARLY_/MID_/LATE_） ----------
def _stage_bounds():
    early_end = int(os.getenv("EARLY_HANDS", "20"))  # [0, early_end)
    mid_end   = int(os.getenv("MID_HANDS",   os.getenv("LATE_HANDS", "56")))
    return early_end, mid_end

def _stage_prefix(rounds_seen: int) -> str:
    e_end, m_end = _stage_bounds()
    if rounds_seen < e_end: return "EARLY_"
    elif rounds_seen < m_end: return "MID_"
    else: return "LATE_"

def get_stage_over(rounds_seen: int) -> Dict[str, float]:
    if COMPAT_MODE == 1: return {}
    if os.getenv("STAGE_MODE","count").lower() == "disabled": return {}
    over: Dict[str,float] = {}
    prefix = _stage_prefix(rounds_seen)
    keys = ["SOFT_TAU","THEO_BLEND","TIE_MAX",
            "MIN_CONF_FOR_ENTRY","EDGE_ENTER",
            "PF_PRED_SIMS","DEPLETEMC_SIMS"]
    for k in keys:
        v = os.getenv(prefix + k)
        if v not in (None, ""):
            try: over[k] = float(v)
            except: pass
    if prefix == "LATE_":
        late_dep = os.getenv("LATE_DEPLETEMC_SIMS")
        if late_dep:
            try: over["DEPLETEMC_SIMS"] = float(late_dep)
            except: pass
    return over

# ---------- 解析點數 ----------
def parse_last_hand_points(text: str) -> Optional[Tuple[int,int]]:
    if not text: return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９：","0123456789:"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]","",s).replace("\u3000"," ")
    u = s.upper().strip()
    m = re.search(r"(?:和|TIE|DRAW)\s*:?:?\s*(\d)?", u)
    if m: d = m.group(1); return (int(d),int(d)) if d else (0,0)
    m = re.search(r"(?:閒|闲|P)\s*:?:?\s*(\d)\D+(?:莊|庄|B)\s*:?:?\s*(\d)", u)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|庄|B)\s*:?:?\s*(\d)\D+(?:閒|闲|P)\s*:?:?\s*(\d)", u)
    if m: return (int(m.group(2)), int(m.group(1)))
    t = u.replace(" ","").replace("\u3000","")
    if t in ("B","莊","庄"): return (0,1)
    if t in ("P","閒","闲"): return (1,0)
    if t in ("T","和"): return (0,0)
    if re.search(r"[A-Z]", u): return None
    d = re.findall(r"\d", u)
    if len(d)==2: return (int(d[0]), int(d[1]))
    return None

# ---------- 主預測 ----------
def _handle_points_and_predict(sess: Dict[str,Any], p_pts:int, b_pts:int) -> Tuple[np.ndarray,str,int,str]:
    rounds_seen = int(sess.get("rounds_seen", 0))
    over = get_stage_over(rounds_seen)

    sims_per_particle = int(over.get("PF_PRED_SIMS", float(os.getenv("PF_PRED_SIMS","5"))))
    p = np.asarray(PF.predict(sims_per_particle=sims_per_particle), dtype=np.float32)

    soft_tau = float(over.get("SOFT_TAU", float(os.getenv("SOFT_TAU","2.0"))))
    p = p ** (1.0/max(1e-6, soft_tau)); p = p / p.sum()

    if (COMPAT_MODE == 0) and (DEPL_ENABLE == 1) and DEPLETE_OK and init_counts and probs_after_points:
        try:
            counts = init_counts(int(os.getenv("DECKS","8")))
            dep_sims = int(over.get("DEPLETEMC_SIMS", float(os.getenv("DEPLETEMC_SIMS","1000"))))
            dep = probs_after_points(counts, p_pts, b_pts, sims=dep_sims, deplete_factor=1.0)
            p = (p + dep) * 0.5; p = p / p.sum()
        except Exception as e:
            log.warning("Deplete 失敗，改 PF 單模：%s", e)

    if COMPAT_MODE == 0:
        theo_blend = float(over.get("THEO_BLEND", float(os.getenv("THEO_BLEND","0.0"))))
        if theo_blend > 0.0:
            theo = np.array([0.4586,0.4462,0.0952], dtype=np.float32)
            p = (1.0 - theo_blend)*p + theo_blend*theo; p = p / p.sum()

        tie_max = float(over.get("TIE_MAX", float(os.getenv("TIE_MAX", str(TIE_MAX)))))
        if p[2] > tie_max:
            sc = (1.0 - tie_max) / (1.0 - float(p[2])) if p[2] < 1.0 else 1.0
            p[2] = tie_max; p[0] *= sc; p[1] *= sc; p = p / p.sum()
        if p[2] < TIE_MIN:
            sc = (1.0 - TIE_MIN) / (1.0 - float(p[2])) if p[2] < 1.0 else 1.0
            p[2] = TIE_MIN; p[0] *= sc; p[1] *= sc; p = p / p.sum()

    _MIN_CONF, _EDGE_ENTER = MIN_CONF_FOR_ENTRY, EDGE_ENTER
    try:
        if COMPAT_MODE == 0:
            if "MIN_CONF_FOR_ENTRY" in over: globals()["MIN_CONF_FOR_ENTRY"] = float(over["MIN_CONF_FOR_ENTRY"])
            if "EDGE_ENTER" in over:        globals()["EDGE_ENTER"] = float(over["EDGE_ENTER"])
        choice, edge, bet_pct, reason = decide_only_bp(p)
    finally:
        globals()["MIN_CONF_FOR_ENTRY"] = _MIN_CONF
        globals()["EDGE_ENTER"] = _EDGE_ENTER

    bet_amt = bet_amount(int(sess.get("bankroll",0)), bet_pct)
    sess["rounds_seen"] = rounds_seen + 1

    if LOG_DECISION or SHOW_CONF_DEBUG:
        log.info("決策: %s edge=%.4f pct=%.2f%% rounds=%d | %s",
                 choice, edge, bet_pct*100, sess["rounds_seen"], reason)
    return p, choice, bet_amt, reason

# ---------- 簡易 HTTP ----------
@app.get("/")
def root():
    ua = request.headers.get("User-Agent","")
    if "UptimeRobot" in ua: return "OK", 200
    st = "OK" if pf_initialized else "BACKUP_MODE"
    return f"✅ BGS Server {st} ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION,
                   pf_initialized=pf_initialized,
                   pf_backend=getattr(PF,'backend','unknown')), 200

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True) or {}
        uid = str(data.get("uid") or "anon")
        last_text = str(data.get("last_text") or "")
        bankroll = data.get("bankroll")
        sess = get_session(uid)
        if isinstance(bankroll,int) and bankroll >= 0: sess["bankroll"] = bankroll

        pts = parse_last_hand_points(last_text)
        if not pts: return jsonify(ok=False, error="無法解析點數；請輸入 '閒6莊5' / '65' / '和'"), 400

        p_pts, b_pts = pts
        sess["last_pts_text"] = "上局結果: 和局" if (p_pts == b_pts and SKIP_TIE_UPD) else f"上局結果: 閒 {p_pts} 莊 {b_pts}"
        probs, choice, bet_amt, reason = _handle_points_and_predict(sess, p_pts, b_pts)
        save_session(uid, sess)
        card = format_output_card(probs, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE))
        return jsonify(ok=True,
                       probs=[float(probs[0]), float(probs[1]), float(probs[2])],
                       choice=choice, bet=bet_amt, reason=reason, card=card), 200
    except Exception as e:
        log.exception("predict error: %s", e)
        return jsonify(ok=False, error=str(e)), 500

# ---------- LINE：完整互動 ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET","")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN","")
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES","30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT","@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET","aaa8881688")

def _trial_key(uid: str, kind: str) -> str: return f"trial:{kind}:{uid}"

def trial_persist_guard(uid: str) -> Optional[str]:
    now = int(time.time())
    first_ts = _rget(_trial_key(uid,"first_ts"))
    expired = _rget(_trial_key(uid,"expired"))
    if expired == "1":
        return f"⛔ 試用已到期\n📬 請聯繫：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"
    if not first_ts:
        _rset(_trial_key(uid,"first_ts"), str(now)); return None
    try: first = int(first_ts)
    except: first = now; _rset(_trial_key(uid,"first_ts"), str(now))
    used_min = (now - first) // 60
    if used_min >= TRIAL_MINUTES:
        _rset(_trial_key(uid,"expired"), "1")
        return f"⛔ 試用已到期\n📬 請聯繫：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"
    return None

def validate_activation_code(code: str) -> bool:
    if not code: return False
    norm = str(code).replace("\u3000"," ").replace("：",":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and (norm == ADMIN_ACTIVATION_SECRET)

GAMES = {"1":"WM","2":"PM","3":"DG","4":"SA","5":"KU","6":"歐博/卡利","7":"KG","8":"全利","9":"名人","10":"MT真人"}
def game_menu_text(left_min: int) -> str:
    lines = ["請選擇遊戲館別"]
    for k in sorted(GAMES.keys(), key=lambda x: int(x)): lines.append(f"{k}. {GAMES[k]}")
    lines.append("「請直接輸入數字選擇」"); lines.append(f"⏳ 試用剩餘 {left_min} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
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
        log.warning("[LINE] reply failed: %s", e)

line_api = None; line_handler = None
if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import MessageEvent, TextMessage, FollowEvent

        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            if not _dedupe_event(getattr(event,"id",None)): return
            uid = event.source.user_id
            _ = trial_persist_guard(uid)
            sess = get_session(uid)
            _reply(line_api, event.reply_token,
                   "👋 歡迎！輸入『遊戲設定』開始；連續模式啟動後只需輸入點數（例：65 / 和 / 閒6莊5）即可預測。")
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            if not _dedupe_event(getattr(event,"id",None)): return
            uid = event.source.user_id
            raw = (event.message.text or "")
            text = re.sub(r"\s+"," ", raw.replace("\u3000"," ").strip())
            sess = get_session(uid)
            up = text.upper()

            if up.startswith("開通") or up.startswith("ACTIVATE"):
                after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                ok = validate_activation_code(after)
                sess["premium"] = bool(ok)
                _reply(line_api, event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤")
                save_session(uid, sess); return

            guard = trial_persist_guard(uid)
            if guard and not sess.get("premium", False):
                _reply(line_api, event.reply_token, guard); save_session(uid, sess); return

            if up in ("結束分析","清空","RESET"):
                premium = sess.get("premium", False)
                start_ts = sess.get("trial_start", int(time.time()))
                sess = {"phase":"await_pts","bankroll":0,"rounds_seen":0,"last_pts_text":None,"premium":premium,"trial_start":start_ts}
                _reply(line_api, event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。")
                save_session(uid, sess); return

            if text == "遊戲設定" or up == "GAME SETTINGS":
                sess["phase"] = "choose_game"; sess["game"] = None; sess["table"] = None; sess["bankroll"] = 0
                first_ts = _rget(_trial_key(uid,"first_ts"))
                left = max(0, TRIAL_MINUTES - ((int(time.time())-int(first_ts))//60)) if first_ts else TRIAL_MINUTES
                _reply(line_api, event.reply_token, game_menu_text(left)); save_session(uid, sess); return

            if sess.get("phase") == "choose_game":
                m = re.match(r"^\s*(\d+)", text)
                if m and (m.group(1) in GAMES):
                    sess["game"] = GAMES[m.group(1)]; sess["phase"] = "input_bankroll"
                    _reply(line_api, event.reply_token, f"🎰 已選擇：{sess['game']}，請輸入初始籌碼（金額）"); save_session(uid, sess); return
                _reply(line_api, event.reply_token, "⚠️ 無效的選項，請輸入上列數字。"); return

            if sess.get("phase") == "input_bankroll":
                num = re.sub(r"[^\d]","", text)
                amt = int(num) if num else 0
                if amt <= 0: _reply(line_api, event.reply_token, "⚠️ 請輸入正整數金額。"); return
                sess["bankroll"] = amt; sess["phase"] = "await_pts"
                _reply(line_api, event.reply_token,
                       f"✅ 設定完成！館別：{sess.get('game')}，初始籌碼：{amt}。\n📌 連續模式：現在輸入第一局點數（例：閒6莊5 / 65 / 和）")
                save_session(uid, sess); return

            pts = parse_last_hand_points(text)
            if pts and sess.get("bankroll",0) >= 0:
                p_pts, b_pts = pts
                sess["last_pts_text"] = "上局結果: 和局" if (p_pts == b_pts and SKIP_TIE_UPD) else f"上局結果: 閒 {p_pts} 莊 {b_pts}"
                probs, choice, bet_amt, reason = _handle_points_and_predict(sess, p_pts, b_pts)
                save_session(uid, sess)
                msg = format_output_card(probs, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE))
                _reply(line_api, event.reply_token, msg); return

            _reply(line_api, event.reply_token,
                   "指令無法辨識。\n📌 直接輸入點數（例：65 / 和 / 閒6莊5），或輸入『遊戲設定』。")

        @app.post("/line-webhook")
        def line_webhook():
            signature = request.headers.get("X-Line-Signature","")
            body = request.get_data(as_text=True)
            try:
                line_handler.handle(body, signature)
            except InvalidSignatureError:
                abort(400, "Invalid signature")
            except Exception as e:
                log.error("webhook error: %s", e); abort(500)
            return "OK", 200

        log.info("LINE webhook enabled (FULL flow)")
    except Exception as e:
        log.warning("LINE not fully configured: %s", e)
else:
    log.warning("LINE credentials missing; LINE webhook disabled.")

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s (PF_INIT=%s, DEPLETE_OK=%s, MODE=%s, COMPAT=%s, DEPL=%s)",
             VERSION, port, pf_initialized, DEPLETE_OK, DECISION_MODE, COMPAT_MODE, DEPL_ENABLE)
    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
