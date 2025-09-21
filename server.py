# -*- coding: utf-8 -*-
"""
server.py — Render free-friendly build
- Split "analysis confidence (display only)" vs "bet sizing (affects money)"
- Online stats: bets/wins, accuracy (ex-tie), recent-N accuracy, avg edge, P&L
- PF health check; graceful dummy fallback notice
- Keep original LINE webhook flow, trial gate, quick replies

This file uses only ASCII in comments (Chinese may appear in user-facing strings).
"""

import os, sys, re, time, json, math, random, logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

# ---------- Optional deps (Flask/LINE/Redis) ----------
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    _has_flask = True
except Exception:
    _has_flask = False
    Flask = None  # type: ignore
    def jsonify(*_, **__): raise RuntimeError("Flask not available")
    def CORS(*_, **__): pass

try:
    import redis
except Exception:
    redis = None

# ---------- Version & logging ----------
VERSION = "pf-render-free-2025-09-21-conf-split+stats+diag"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---------- Flask ----------
if _has_flask:
    app = Flask(__name__)
    CORS(app)
else:
    class _DummyApp:
        def get(self,*a,**k):
            def deco(f): return f
            return deco
        def post(self,*a,**k):
            def deco(f): return f
            return deco
        def run(self,*a,**k):
            log.warning("Flask not installed; dummy app.")
    app = _DummyApp()

# ---------- Redis / Fallback ----------
REDIS_URL = os.getenv("REDIS_URL", "")
rcli = None
if redis and REDIS_URL:
    try:
        rcli = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        rcli.ping()
        log.info("Redis connected.")
    except Exception as e:
        rcli = None
        log.warning("Redis connect fail: %s => fallback memory store", e)

SESS: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE = 3600

def _rget(k: str) -> Optional[str]:
    try:
        if rcli: return rcli.get(k)
    except Exception as e:
        log.warning("Redis GET err: %s", e)
    return None

def _rset(k: str, v: str, ex: Optional[int]=None):
    try:
        if rcli: rcli.set(k, v, ex=ex)
    except Exception as e:
        log.warning("Redis SET err: %s", e)

# ---------- Flags ----------
def env_flag(name: str, default: int=0) -> int:
    v = os.getenv(name)
    if v is None: return 1 if default else 0
    v = v.strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try: return 1 if int(float(v))!=0 else 0
    except: return 1 if default else 0

# ---------- Trial / Activation ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "").strip()
ADMIN_CONTACT_LINK = os.getenv("ADMIN_CONTACT_LINK", "").strip()

def _trial_key(uid: str) -> str:
    return f"trialstart:{uid}"

def get_trial_start(uid: str) -> int:
    nowi = int(time.time())
    if rcli:
        v = _rget(_trial_key(uid))
        if v and v.isdigit():
            return int(v)
        _rset(_trial_key(uid), str(nowi), ex=None)
        return nowi
    s = SESS.get(_trial_key(uid))
    if s and isinstance(s, dict) and "ts" in s:
        return int(s["ts"])
    SESS[_trial_key(uid)] = {"ts": nowi}
    return nowi

def trial_left_minutes(sess: Dict[str, Any], uid: str) -> int:
    if sess.get("premium", False):
        return 9999
    ts = sess.get("trial_start")
    if not ts:
        ts = get_trial_start(uid)
        sess["trial_start"] = ts
    used = (int(time.time()) - int(ts)) // 60
    return max(0, TRIAL_MINUTES - used)

def validate_activation_code(code: str) -> bool:
    if not ADMIN_ACTIVATION_SECRET or not code:
        return False
    norm = str(code).replace("\u3000"," ").replace("：",":").strip().lstrip(":").strip()
    return norm == ADMIN_ACTIVATION_SECRET

def trial_guard_or_none(sess: Dict[str,Any], uid: str) -> Optional[str]:
    left = trial_left_minutes(sess, uid)
    if left > 0: return None
    link_line = f"\n👉 加入官方 LINE：{ADMIN_CONTACT_LINK}" if ADMIN_CONTACT_LINK else ""
    return f"⛔ 試用期已到\n📬 請聯繫管理員開通登入帳號{link_line}\n🔐 可輸入：開通 你的密碼"

# ---------- Session ----------
GAMES = {
    "1": "WM","2":"PM","3":"DG","4":"SA","5":"KU",
    "6":"歐博/卡利","7":"KG","8":"全利","9":"名人","10":"MT真人",
}

# ---------- PF import ----------
OutcomePF = None
try:
    from bgs.pfilter import OutcomePF  # type: ignore
except Exception:
    try:
        cur = os.path.dirname(os.path.abspath(__file__))
        if cur not in sys.path: sys.path.insert(0, cur)
        from pfilter import OutcomePF  # type: ignore
        log.info("OutcomePF from local pfilter.py")
    except Exception as e:
        OutcomePF = None  # type: ignore
        log.error("OutcomePF import failed: %s", e)

class _DummyPF:
    def update_outcome(self, outcome): pass
    def predict(self, **k): return np.array([0.48,0.47,0.05], dtype=np.float32)
    def update_point_history(self, p_pts, b_pts): pass

def _get_pf_from_sess(sess: Dict[str, Any]) -> Any:
    if OutcomePF:
        if sess.get("pf") is None:
            try:
                sess["pf"] = OutcomePF(
                    decks=int(os.getenv("DECKS","6")),
                    seed=int(os.getenv("SEED","42")),
                    n_particles=int(os.getenv("PF_N","60")),
                    sims_lik=max(1,int(os.getenv("PF_UPD_SIMS","30"))),
                    resample_thr=float(os.getenv("PF_RESAMPLE","0.7")),
                    backend=os.getenv("PF_BACKEND","mc"),
                    dirichlet_eps=float(os.getenv("PF_DIR_EPS","0.003")),
                )
                log.info("Per-session PF init ok")
            except Exception as e:
                log.error("Per-session PF init fail: %s", e)
                sess["pf"] = _DummyPF()
        return sess["pf"]
    return _DummyPF()

def now_sess(uid: str) -> Dict[str, Any]:
    if rcli:
        j = _rget(f"sess:{uid}")
        if j:
            try: return json.loads(j)
            except: pass
    s = SESS.get(uid)
    if s: return s
    s = {
        "bankroll": 0,
        "phase": "choose_game",
        "game": None, "table": None,
        "trial_start": get_trial_start(uid),
        "premium": False,
        "welcomed": False,
        "last_pts_text": None,
        "last_prob_gap": 0.0,
        "hand_idx": 0,
        "pf": None,
        # online stats
        "stats": {
            "bets": 0,
            "wins": 0,
            "push": 0,
            "sum_edge": 0.0,
            "payout": 0,
        },
        # history for accuracy excluding tie
        "hist_pred": [],  # '莊' / '閒' / '觀望'
        "hist_real": [],  # '莊' / '閒' / '和'
    }
    SESS[uid] = s
    return s

def save_sess(uid: str, s: Dict[str, Any]):
    if rcli:
        _rset(f"sess:{uid}", json.dumps(s), ex=SESSION_EXPIRE)
    else:
        SESS[uid] = s

# ---------- Utils ----------
def _norm(s: str) -> str:
    if not s: return ""
    s = s.translate(str.maketrans("　０１２３４５６７８９：－—～〜", " 0123456789:----"))
    s = re.sub(r"\s+", " ", s.strip())
    return s.upper()

def _quick_reply():
    try:
        from linebot.models import QuickReply, QuickReplyButton, MessageAction
        items = [
            QuickReplyButton(action=MessageAction(label="遊戲設定 🎮", text="遊戲設定")),
            QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
            QuickReplyButton(action=MessageAction(label="報莊 🅱️", text="B")),
            QuickReplyButton(action=MessageAction(label="報閒 🅿️", text="P")),
            QuickReplyButton(action=MessageAction(label="報和 ⚪", text="T")),
        ]
        if ADMIN_CONTACT_LINK:
            items.append(QuickReplyButton(action=MessageAction(label="聯繫管理員 📩", text="聯繫管理員")))
        return QuickReply(items=items)
    except Exception:
        return None

def welcome_text(left_min: int) -> str:
    menu = ["【請選擇遊戲館別】"]
    for k in sorted(GAMES.keys(), key=lambda x: int(x)):
        menu.append(f"{k}. {GAMES[k]}")
    menu.append("（請直接輸入數字 1-10）")
    left_line = "" if left_min>=9999 else f"\n⏳ 試用剩餘 {left_min} 分鐘"
    return (
        "👋 歡迎使用 BGS AI 預測分析！\n"
        "使用步驟：\n"
        "1️⃣ 選擇館別（輸入 1~10）\n"
        "2️⃣ 輸入桌號（例：DG01）\n"
        "3️⃣ 輸入本金（例：5000）\n"
        "4️⃣ 每局回報點數（例：65 / 和 / 閒6莊5）即可連續預測\n\n"
        + "\n".join(menu)
        + left_line
    )

# ---------- Parse points (tolerant) ----------
def parse_last_hand_points(text: str) -> Optional[Tuple[int,int]]:
    if not text: return None
    s = str(text)
    s = s.translate(str.maketrans("０１２３４５６７８９：－—～〜", "0123456789:----"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s)
    u = re.sub(r"\s+", " ", s).strip().upper()

    if re.search(r"\b(和|TIE|DRAW|^T$)\b", u):
        m = re.search(r"(?:和|TIE|DRAW|T)\s*:?\s*([0-9])", u)
        if m:
            d = int(m.group(1)); return (d, d)
        return (0,0)

    m = re.search(r"(閒|P)\s*:?\s*([0-9]).*?(莊|B)\s*:?\s*([0-9])", u)
    if m: return (int(m.group(2)), int(m.group(4)))
    m = re.search(r"(莊|B)\s*:?\s*([0-9]).*?(閒|P)\s*:?\s*([0-9])", u)
    if m: return (int(m.group(4)), int(m.group(2)))

    m = re.search(r"B\D*([0-9])\D*P\D*([0-9])", u)
    if m: return (int(m.group(2)), int(m.group(1)))
    m = re.search(r"P\D*([0-9])\D*B\D*([0-9])", u)
    if m: return (int(m.group(1)), int(m.group(2)))

    digits = re.findall(r"[0-9]", u)
    if len(digits) >= 2:
        return (int(digits[-2]), int(digits[-1]))
    return None

# ---------- Render-safe defaults ----------
os.environ.setdefault("PF_BACKEND", "mc")
os.environ.setdefault("DECKS", "6")
os.environ.setdefault("PF_N", "60")
os.environ.setdefault("PF_UPD_SIMS", "30")
os.environ.setdefault("PF_PRED_SIMS", "20")
os.environ.setdefault("PF_RESAMPLE", "0.7")
os.environ.setdefault("PF_DIR_EPS", "0.003")

# ---------- PF health diag ----------
PF_DIAG = int(os.getenv("PF_DIAG", "1"))
def _check_pf_health() -> Dict[str, Any]:
    try:
        if OutcomePF:
            pf_test = OutcomePF(n_particles=10, backend=os.getenv("PF_BACKEND","mc"))
            pred0 = np.array(pf_test.predict(sims_per_particle=10), dtype=float)
        else:
            pred0 = np.array([0.48,0.47,0.05], dtype=float)
        ok_sum = abs(pred0.sum() - 1.0) < 1e-3
        ok_range = np.all((pred0 >= -1e-6) & (pred0 <= 1+1e-6))
        is_dummy = OutcomePF is None
        info = {"pred0": pred0.tolist(), "ok_sum": ok_sum, "ok_range": ok_range, "is_dummy": is_dummy}
        if is_dummy or not ok_sum or not ok_range:
            log.error("PF health check FAIL: %s", info)
        else:
            log.info("PF health OK: %s", info)
        return info
    except Exception as e:
        log.exception("PF health check exception: %s", e)
        return {"pred0": None, "ok_sum": False, "ok_range": False, "is_dummy": True}

PF_HEALTH = {"pred0": None, "ok_sum": False, "ok_range": False, "is_dummy": True}
if PF_DIAG:
    PF_HEALTH = _check_pf_health()

# ---------- Decision / bet sizing & watch rules ----------
DECIDE_MODE = os.getenv("DECIDE_MODE", "prob").strip().lower()
BANKER_COMMISSION = float(os.getenv("BANKER_COMMISSION", "0.95"))

MIN_BET_PCT = float(os.getenv("MIN_BET_PCT","0.05"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT","0.40"))
PROB_SMA_ALPHA = float(os.getenv("PROB_SMA_ALPHA","0.45"))
PROB_TEMP = float(os.getenv("PROB_TEMP","1.0"))

TS_EN = env_flag("TS_EN", 0)
TS_ALPHA = float(os.getenv("TS_ALPHA","2"))
TS_BETA  = float(os.getenv("TS_BETA","2"))

EDGE_ENTER = float(os.getenv("EDGE_ENTER","0.005"))
TIE_PROB_MAX = float(os.getenv("TIE_PROB_MAX","0.18"))
WATCH_EN = env_flag("WATCH_EN", 1)
WATCH_INSTAB_THRESH = float(os.getenv("WATCH_INSTAB_THRESH","0.08"))

UNCERT_PENALTY_EN = env_flag("UNCERT_PENALTY_EN", 1)
UNCERT_MARGIN_MAX = int(os.getenv("UNCERT_MARGIN_MAX","2"))
UNCERT_RATIO = float(os.getenv("UNCERT_RATIO","0.25"))

W_BASE = float(os.getenv("W_BASE","1.0"))
W_MIN  = float(os.getenv("W_MIN","0.5"))
W_MAX  = float(os.getenv("W_MAX","3.0"))
W_ALPHA= float(os.getenv("W_ALPHA","0.95"))
W_SIG_K= float(os.getenv("W_SIG_K","1.10"))
W_SIG_MID=float(os.getenv("W_SIG_MID","1.8"))
W_GAMMA= float(os.getenv("W_GAMMA","1.0"))
W_GAP_CAP=float(os.getenv("W_GAP_CAP","0.06"))

DEPTH_W_EN  = env_flag("DEPTH_W_EN", 1)
DEPTH_W_MAX = float(os.getenv("DEPTH_W_MAX","1.5"))

INV = {0:"莊", 1:"閒"}

def softmax_temp(p: np.ndarray, t: float) -> np.ndarray:
    t = max(1e-6, float(t))
    x = np.log(np.clip(p,1e-9,1.0)) / t
    x = np.exp(x - np.max(x))
    x = x / np.sum(x)
    return x

def ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None: return cur
    return alpha*cur + (1-alpha)*prev

def calc_margin_weight(p_pts: int, b_pts: int, last_prob_gap: float) -> float:
    margin = abs(int(p_pts) - int(b_pts))
    sig = 1.0/(1.0 + math.exp(-W_SIG_K * (margin - W_SIG_MID)))
    part_m = W_ALPHA * sig
    gap_norm = min(max(float(last_prob_gap),0.0), W_GAP_CAP) / max(W_GAP_CAP,1e-6)
    part_g = W_GAMMA * gap_norm
    w = W_BASE + part_m + part_g
    return max(W_MIN, min(W_MAX, w))

def decide_bp(prob: np.ndarray) -> Tuple[str, float, float, float]:
    pB, pP = float(prob[0]), float(prob[1])
    gap = abs(pB - pP)
    if DECIDE_MODE == "ev":
        evB, evP = BANKER_COMMISSION * pB - pP, pP - pB
        side = 0 if evB > evP else 1
        edge = max(abs(evB), abs(evP))
        return (INV[side], edge, max(pB, pP), gap)
    side = 0 if pB >= pP else 1
    edge = gap
    return (INV[side], edge, max(pB, pP), gap)

def bet_amount(bankroll: int, pct: float) -> int:
    if bankroll<=0 or pct<=0: return 0
    return int(round(bankroll * pct))

# ----- Confidence (display) vs bet sizing (money) -----
CONF_MIN_PCT = float(os.getenv("CONF_MIN_PCT", "0.10"))
CONF_MAX_PCT = float(os.getenv("CONF_MAX_PCT", "0.90"))
RISK_LEVEL   = float(os.getenv("RISK_LEVEL", "1.0"))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def analysis_confidence(prob: np.ndarray, prev_gap: float, cur_gap: float) -> float:
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    maxp = max(pB, pP)
    c_prob = clamp((maxp - 0.50) * 4.0, 0.0, 1.0)  # 0.50->0, 0.75->1
    gap_delta = abs(cur_gap - prev_gap)
    c_stab = clamp(1.0 - (gap_delta / max(1e-6, WATCH_INSTAB_THRESH*2.0)), 0.0, 1.0)
    t_pen = clamp((pT - TIE_PROB_MAX) * 6.0, 0.0, 1.0)
    c = 0.55*c_prob + 0.35*c_stab - 0.30*t_pen
    return clamp(c, 0.0, 1.0)

def map_conf_to_display(c: float) -> float:
    return clamp(CONF_MIN_PCT + c * (CONF_MAX_PCT - CONF_MIN_PCT), CONF_MIN_PCT, CONF_MAX_PCT)

def base_bet_pct(edge: float, max_prob: float) -> float:
    e = clamp(edge / 0.06, 0.0, 1.0)              # 6% -> 1
    m = clamp((max_prob - 0.50) / 0.20, 0.0, 1.0) # 70% -> 1
    raw = (0.6*e + 0.4*m)
    raw = raw**0.9
    pct = MIN_BET_PCT + raw * (MAX_BET_PCT - MIN_BET_PCT)
    pct *= clamp(RISK_LEVEL, 0.3, 2.0)
    return clamp(pct, MIN_BET_PCT, MAX_BET_PCT)

def thompson_scale_pct(pct: float) -> float:
    if not TS_EN:
        return pct
    a = max(1e-3, TS_ALPHA); b = max(1e-3, TS_BETA)
    s = np.random.beta(a, b)
    return clamp(pct * s, MIN_BET_PCT, MAX_BET_PCT)

# history helpers
HIST_MAX = int(os.getenv("HIST_MAX", "200"))
def _append_hist(sess: Dict[str,Any], pred: str, real: str):
    try:
        sess["hist_pred"].append(pred)
        sess["hist_real"].append(real)
        if len(sess["hist_pred"]) > HIST_MAX:
            sess["hist_pred"] = sess["hist_pred"][-HIST_MAX:]
            sess["hist_real"] = sess["hist_real"][-HIST_MAX:]
    except Exception:
        pass

def _acc_ex_tie(sess: Dict[str,Any], last_n: Optional[int]=None) -> Tuple[int,int,float]:
    pred = sess.get("hist_pred", [])
    real = sess.get("hist_real", [])
    if last_n is not None:
        pred = pred[-last_n:]
        real = real[-last_n:]
    pairs = [(p,r) for p,r in zip(pred,real) if r in ("莊","閒") and p in ("莊","閒")]
    if not pairs:
        return (0,0,0.0)
    hit = sum(1 for p,r in pairs if p==r)
    tot = len(pairs)
    return (hit, tot, 100.0*hit/tot)

# input validation
def validate_input_data(p_pts: int, b_pts: int) -> bool:
    if not (0 <= int(p_pts) <= 9 and 0 <= int(b_pts) <= 9):
        log.error("Invalid points: P=%s B=%s", p_pts, b_pts)
        return False
    return True

_prev_prob_sma: Optional[np.ndarray] = None

def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    global _prev_prob_sma

    # validate points
    if not validate_input_data(p_pts, b_pts):
        return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    pf = _get_pf_from_sess(sess)
    pf.update_point_history(p_pts, b_pts)

    sess["hand_idx"] = int(sess.get("hand_idx", 0)) + 1
    margin = abs(p_pts - b_pts)

    # 1) PF update with weights
    last_gap = float(sess.get("last_prob_gap", 0.0))
    w = calc_margin_weight(p_pts, b_pts, last_gap)
    if DEPTH_W_EN and sess["hand_idx"]>0:
        depth_boost = 1.0 + min(sess["hand_idx"]/70.0, (DEPTH_W_MAX-1.0))
        w *= depth_boost
    REP_CAP = int(os.getenv("PF_REP_CAP", "3"))
    rep = max(1, min(REP_CAP, int(round(w))))

    if p_pts == b_pts:
        try: pf.update_outcome(2)
        except Exception as e: log.warning("PF tie update err: %s", e)
    else:
        outcome = 1 if p_pts > b_pts else 0
        for _ in range(rep):
            try: pf.update_outcome(outcome)
            except Exception as e: log.warning("PF update err: %s", e)
        if UNCERT_PENALTY_EN and margin <= UNCERT_MARGIN_MAX:
            rev = 0 if outcome==1 else 1
            if random.random() < UNCERT_RATIO:
                try: pf.update_outcome(rev)
                except Exception as e: log.warning("PF uncert reverse update err: %s", e)

    # 2) predict & smooth
    sims_pred = max(0, int(os.getenv("PF_PRED_SIMS","20")))
    p_raw = pf.predict(sims_per_particle=sims_pred)
    p_adj = p_raw.copy()
    p_adj = p_adj / np.sum(p_adj)
    p_temp = softmax_temp(p_adj, PROB_TEMP)
    _prev_prob_sma = ema(_prev_prob_sma, p_temp, PROB_SMA_ALPHA)
    p_final = _prev_prob_sma if _prev_prob_sma is not None else p_temp

    # 3) decide direction
    choice, edge, maxp, prob_gap = decide_bp(p_final)

    # 4) watch rules
    watch = False
    reasons = []
    if WATCH_EN:
        if prob_gap < EDGE_ENTER:
            watch = True; reasons.append("機率差過小")
        if float(p_final[2]) > TIE_PROB_MAX:
            watch = True; reasons.append("和局風險偏高")
        if abs(prob_gap - last_gap) > WATCH_INSTAB_THRESH:
            watch = True; reasons.append("勝率波動大")

    # 5) confidence (display) and bet sizing (money)
    bankroll = int(sess.get("bankroll", 0))
    c0 = analysis_confidence(p_final, last_gap, prob_gap)
    conf_pct_disp = map_conf_to_display(c0)

    if watch:
        bet_pct = 0.0
        bet_amt = 0
        choice_text = "觀望"
        strat = f"⚠️ 觀望（{'、'.join(reasons)}）"
    else:
        pct_base = base_bet_pct(edge, maxp)
        bet_pct  = thompson_scale_pct(pct_base)
        bet_amt  = bet_amount(bankroll, bet_pct)
        choice_text = choice
        if bet_pct < 0.28:  strat = f"🟡 低信心配注 {bet_pct*100:.1f}%"
        elif bet_pct < 0.34: strat = f"🟠 中信心配注 {bet_pct*100:.1f}%"
        else:                 strat = f"🟢 高信心配注 {bet_pct*100:.1f}%"

    # 6) stats update using previous result
    st = sess["stats"]
    if p_pts == b_pts:
        st["push"] += 1
        real_label = "和"
    else:
        real_label = "閒" if p_pts > b_pts else "莊"
        if not watch:
            st["bets"] += 1
            st["sum_edge"] += float(edge)
            if choice_text == real_label:
                if real_label == "莊":
                    st["payout"] += int(round(bet_amt * BANKER_COMMISSION))
                else:
                    st["payout"] += int(bet_amt)
                st["wins"] += 1
            else:
                st["payout"] -= int(bet_amt)

    pred_label = "觀望" if watch else choice
    _append_hist(sess, pred_label, real_label)

    # last result text & save gap
    if p_pts == b_pts:
        sess["last_pts_text"] = f"上局結果: 和 {p_pts}"
    else:
        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
    sess["last_prob_gap"] = prob_gap

    # 7) message
    mode_note = "（以勝率決策）" if DECIDE_MODE=="prob" else f"（以期望值決策，comm={BANKER_COMMISSION:.3f}）"
    msg = [
        sess["last_pts_text"],
        "開始分析下局....",
        "",
        "【預測結果】",
        f"閒：{p_final[1]*100:.2f}%",
        f"莊：{p_final[0]*100:.2f}%",
        f"和：{p_final[2]*100:.2f}%",
        f"本次預測結果：{choice_text} {mode_note}",
        f"分析信心度（不影響下注）：{conf_pct_disp*100:.1f}%",
        f"建議配注比例：{bet_pct*100:.1f}%",
        f"建議下注金額：{bet_amt:,}",
        f"配注策略：{strat} (優勢: {edge*100:.1f}%)",
    ]

    # stats summary
    bets = st["bets"]
    avg_edge = (st["sum_edge"]/bets*100.0) if bets>0 else 0.0
    roi = st["payout"]
    NEAR_N = int(os.getenv("NEAR_N", "50"))
    _, _, acc_all = _acc_ex_tie(sess, None)
    _, _, acc_recent = _acc_ex_tie(sess, NEAR_N)
    msg.extend([
        "—",
        f"📈 線上統計：出手 {bets}｜命中率(排和) {acc_all:.1f}%｜近{NEAR_N}手 {acc_recent:.1f}%｜平均優勢 {avg_edge:.2f}%｜盈虧 {roi:,}",
        "",
        "🔁 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）",
    ])

    # PF dummy notice
    if PF_HEALTH.get("is_dummy", True):
        msg.insert(0, "⚠️ 模型載入為簡化版（Dummy）。請確認 bgs.pfilter 是否可用。")

    return "\n".join(msg)

# ---------- REST ----------
@app.get("/")
def root():
    return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(
        ok=True,
        ts=time.time(),
        version=VERSION,
        pf_ok=(not PF_HEALTH.get("is_dummy", True) and PF_HEALTH.get("ok_sum", False) and PF_HEALTH.get("ok_range", False)),
        pf_probe=PF_HEALTH.get("pred0"),
    ), 200

@app.post("/predict")
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    uid = str(data.get("uid","guest"))
    text = str(data.get("text","")).strip()
    sess = now_sess(uid)

    # activation
    up = text.upper()
    if up.startswith("開通") or up.startswith("ACTIVATE"):
        after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
        ok = validate_activation_code(after)
        sess["premium"] = bool(ok)
        save_sess(uid, sess)
        return jsonify(ok=ok, msg=("✅ 已開通成功！" if ok else "❌ 密碼錯誤")), (200 if ok else 403)

    # trial gate
    guard = trial_guard_or_none(sess, uid)
    if guard:
        return jsonify(ok=False, err=guard), 402

    if "bankroll" in data:
        try:
            bk = int(data["bankroll"])
            if bk>0:
                sess["bankroll"] = bk
                sess["phase"] = "await_pts"
        except: pass

    pts = parse_last_hand_points(text)
    if pts is None:
        return jsonify(ok=False, err="無法解析點數（例：閒6莊5 / 65 / 和）"), 400

    msg = handle_points_and_predict(sess, pts[0], pts[1])
    save_sess(uid, sess)
    return jsonify(ok=True, msg=msg), 200

# ---------- LINE webhook ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
    _has_line = True
except Exception:
    _has_line = False

DEDUPE_TTL = 90
def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id: return True
    key = f"dedupe:{event_id}"
    try:
        if rcli:
            ok = rcli.set(key, "1", nx=True, ex=DEDUPE_TTL)
            return bool(ok)
    except Exception:
        pass
    v = SESS.get(key)
    if v and isinstance(v, dict) and v.get("exp", 0) > time.time():
        return False
    SESS[key] = {"exp": time.time() + DEDUPE_TTL}
    return True

def reply_text(token: str, text: str, user_id: Optional[str]=None):
    try:
        from linebot.models import TextSendMessage
        line_api.reply_message(token, TextSendMessage(text=text, quick_reply=_quick_reply()))
    except Exception as e:
        try:
            if user_id:
                line_api.push_message(user_id, TextSendMessage(text=text, quick_reply=_quick_reply()))
            else:
                log.warning("reply err(no uid): %s", e)
        except Exception as e2:
            log.warning("push fallback failed: %s", e2)

if _has_line and LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

    @app.post("/line-webhook")
    def line_webhook():
        signature = request.headers.get("X-Line-Signature", "")
        body = request.get_data(as_text=True)
        try:
            line_handler.handle(body, signature)
        except InvalidSignatureError:
            return "invalid signature", 400
        except Exception as e:
            log.warning("line webhook err: %s", e)
            return "ok", 200
        return "ok", 200

    def _maybe_welcome(uid: str, sess: Dict[str,Any], reply_token: str):
        if not sess.get("welcomed", False):
            left = trial_left_minutes(sess, uid)
            reply_text(reply_token, welcome_text(left), user_id=uid)
            sess["welcomed"] = True
            save_sess(uid, sess)

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event):
        try:
            dc = getattr(event, "delivery_context", None)
            if dc and getattr(dc, "is_redelivery", False):
                return
        except Exception:
            pass
        if not _dedupe_event(getattr(event, "id", None)):
            return

        uid = getattr(event.source, "user_id", "guest")
        raw = (event.message.text or "")
        norm = _norm(raw)
        sess = now_sess(uid)
        phase = sess.get("phase","choose_game")
        log.info("[LINE] uid=%s phase=%s text=%r norm=%r", uid, phase, raw, norm)

        _maybe_welcome(uid, sess, event.reply_token)

        if norm.startswith("開通") or norm.startswith("ACTIVATE"):
            after = raw[2:] if norm.startswith("開通") else raw[len("ACTIVATE"):]
            ok = validate_activation_code(after)
            sess["premium"] = bool(ok); save_sess(uid, sess)
            reply_text(event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤", user_id=uid); return
        if norm in ("聯繫管理員","CONTACT","ADMIN","客服"):
            link_line = f"\n👉 加入官方 LINE：{ADMIN_CONTACT_LINK}" if ADMIN_CONTACT_LINK else ""
            reply_text(event.reply_token, f"📬 請聯繫管理員開通登入帳號{link_line}", user_id=uid); return

        guard = trial_guard_or_none(sess, uid)
        if guard:
            reply_text(event.reply_token, guard, user_id=uid); return

        if norm in ("結束分析", "清空", "RESET"):
            keep_premium = bool(sess.get("premium", False))
            keep_trial   = int(get_trial_start(uid))
            SESS.pop(uid, None)
            sess = now_sess(uid)
            sess["premium"] = keep_premium
            sess["trial_start"] = keep_trial
            sess["phase"] = "choose_game"
            sess["pf"] = None  # reset PF
            save_sess(uid, sess)
            reply_text(event.reply_token, "🧹 已清空。輸入『遊戲設定』開始。", user_id=uid)
            return

        if norm in ("遊戲設定", "設定", "SETUP", "GAME"):
            sess["phase"] = "choose_game"
            sess["game"] = None
            sess["table"] = None
            sess["bankroll"] = 0
            sess["pf"] = None  # reset PF
            save_sess(uid, sess)
            left = trial_left_minutes(sess, uid)
            reply_text(event.reply_token, welcome_text(left), user_id=uid)
            return

        if phase == "choose_game":
            if re.fullmatch(r"(10|[1-9])", norm):
                sess["game"] = GAMES[norm]
                sess["phase"] = "choose_table"
                save_sess(uid, sess)
                reply_text(event.reply_token, f"✅ 已選【{sess['game']}】\n請輸入桌號（例：DG01，格式：2字母+2數字）", user_id=uid)
                return
            reply_text(event.reply_token, "請先選擇館別（輸入數字 1-10）。\n" + welcome_text(trial_left_minutes(sess, uid)), user_id=uid)
            return

        if phase == "choose_table":
            t = re.sub(r"\s+","", norm).upper()
            if re.fullmatch(r"[A-Z]{2}\d{2}", t):
                sess["table"] = t
                sess["phase"] = "await_bankroll"
                save_sess(uid, sess)
                reply_text(event.reply_token, f"✅ 已設桌號【{t}】\n請輸入您的本金（例：5000）", user_id=uid)
                return
            reply_text(event.reply_token, "❌ 桌號格式錯誤，請輸入 2 英文 + 2 數字（例：DG01）", user_id=uid)
            return

        if phase == "await_bankroll":
            if norm.isdigit():
                try:
                    bk = int(norm)
                    if bk>0:
                        sess["bankroll"] = bk
                        sess["phase"] = "await_pts"
                        save_sess(uid, sess)
                        reply_text(event.reply_token, f"👍 已設定本金：{bk:,}\n請輸入上一局點數（例：65 / 和 / 閒6莊5），之後進入連續模式。", user_id=uid)
                        return
                except: pass
            reply_text(event.reply_token, "請輸入本金（純數字，如 5000）。", user_id=uid)
            return

        pts = parse_last_hand_points(raw)
        if pts is None:
            reply_text(event.reply_token, "指令無法辨識。\n請輸入上一局點數（例：65 / 和 / 閒6莊5）。\n或輸入『結束分析』、『遊戲設定』。", user_id=uid)
            return

        if not int(sess.get("bankroll",0)):
            sess["phase"] = "await_bankroll"; save_sess(uid, sess)
            reply_text(event.reply_token, "請先輸入本金（例：5000），再回報點數。", user_id=uid); return

        msg = handle_points_and_predict(sess, int(pts[0]), int(pts[1]))
        sess["phase"] = "await_pts"; save_sess(uid, sess)
        reply_text(event.reply_token, msg, user_id=uid)

else:
    @app.post("/line-webhook")
    def line_webhook_min():
        return "ok", 200

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s (DECIDE_MODE=%s, COMM=%.3f)", VERSION, port, os.getenv("DECIDE_MODE","prob"), BANKER_COMMISSION)
    app.run(host="0.0.0.0", port=port, debug=False)
