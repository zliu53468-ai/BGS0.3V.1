# -*- coding: utf-8 -*-
"""
server.py — Render 免費版優化 + 點差連續加權 + 不確定性懲罰
附加：手數深度權重、點差可靠度表(5桶)、Thompson 配注縮放（皆可用環境變數開關）

說明：
- 不改變你既有的主流程（LINE/Redis/連續模式/EV 決策/下注映射）
- 只在 PF.update_outcome() 前後加「權重/懲罰」與「可靠度/深度/配注縮放」的強化層
- 依環境變數可逐項開/關，方便 A/B 測

作者：for Render Free
"""

import os, sys, time, json, re, logging, math, random
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

# ---------- 可選依賴（Flask/LINE/Redis） ----------
try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
    _has_flask = True
except Exception:
    _has_flask = False
    Flask = None  # type: ignore
    def jsonify(*_, **__): raise RuntimeError("Flask not available")
    def abort(*_, **__): raise RuntimeError("Flask not available")
    def CORS(*_, **__): pass

try:
    import redis
except Exception:
    redis = None

# ---------- 版本 & 日誌 ----------
VERSION = "pf-adv-render-free-2025-09-19"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---------- Flask 初始化 ----------
if _has_flask:
    app = Flask(__name__)
    CORS(app)
else:
    class _DummyApp:
        def get(self, *a, **k):
            def deco(f): return f
            return deco
        def post(self, *a, **k):
            def deco(f): return f
            return deco
        def run(self, *a, **k):
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

SESS: Dict[str, Dict[str, Any]] = {}  # fallback

SESSION_EXPIRE = 3600
def _rget(k: str) -> Optional[str]:
    try:
        if rcli: return rcli.get(k)
    except Exception as e:
        log.warning("Redis GET err: %s", e)
    return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if rcli: rcli.set(k, v, ex=ex)
    except Exception as e:
        log.warning("Redis SET err: %s", e)

# ---------- 環境旗標 ----------
def env_flag(name: str, default: int=0) -> int:
    v = os.getenv(name)
    if v is None: return 1 if default else 0
    v = v.strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try: return 1 if int(float(v))!=0 else 0
    except: return 1 if default else 0

# ---------- 會話 ----------
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
        "trial_start": int(time.time()),
        "premium": True,  # 省略試用流程
        "last_pts_text": None,
        "last_prob_gap": 0.0,
        "hand_idx": 0,  # 每桌手數索引
        # margin reliability (5桶: 0,1,2,3,4+): alpha/beta
        "mrel": {"a":[1.0,1.0,1.0,1.0,1.0], "b":[1.0,1.0,1.0,1.0,1.0]},
    }
    SESS[uid] = s
    return s

def save_sess(uid: str, s: Dict[str, Any]):
    if rcli:
        _rset(f"sess:{uid}", json.dumps(s), ex=SESSION_EXPIRE)
    else:
        SESS[uid] = s

# ---------- 解析點數 ----------
def parse_last_hand_points(text: str) -> Optional[Tuple[int,int]]:
    if not text: return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９：","0123456789:"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s)
    s = s.replace("\u3000"," ")
    u = re.sub(r"^開始分析","", s.strip().upper())

    m = re.search(r"(?:和|TIE|DRAW)\s*:?:?\s*(\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else (0,0)

    m = re.search(r"(?:閒|闲|P)\s*:?:?\s*(\d)\D+(?:莊|庄|B)\s*:?:?\s*(\d)", u)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|庄|B)\s*:?:?\s*(\d)\D+(?:閒|闲|P)\s*:?:?\s*(\d)", u)
    if m: return (int(m.group(2)), int(m.group(1)))

    t = u.replace(" ","")
    if t in ("B","莊","庄"): return (0,1)
    if t in ("P","閒","闲"): return (1,0)
    if t in ("T","和"): return (0,0)

    if re.search(r"[A-Z]", u): return None
    digits = re.findall(r"\d", u)
    if len(digits)==2: return (int(digits[0]), int(digits[1]))
    return None

# ---------- PF 匯入（本地 pfilter 或套件） ----------
try:
    from bgs.pfilter import OutcomePF  # type: ignore
except Exception:
    try:
        cur = os.path.dirname(os.path.abspath(__file__))
        if cur not in sys.path: sys.path.insert(0, cur)
        from pfilter import OutcomePF  # type: ignore
        log.info("OutcomePF from local pfilter.py")
    except Exception as e:
        OutcomePF = None # type: ignore
        log.error("OutcomePF import failed: %s", e)

# ---------- Render 安全預設 ----------
os.environ.setdefault("PF_BACKEND", "mc")
os.environ.setdefault("DECKS", "6")
os.environ.setdefault("PF_N", "60")
os.environ.setdefault("PF_UPD_SIMS", "30")
os.environ.setdefault("PF_PRED_SIMS", "20")
os.environ.setdefault("PF_RESAMPLE", "0.7")
os.environ.setdefault("PF_DIR_EPS", "0.003")

# ---------- 初始化 PF ----------
if OutcomePF:
    try:
        PF = OutcomePF(
            decks=int(os.getenv("DECKS","6")),
            seed=int(os.getenv("SEED","42")),
            n_particles=int(os.getenv("PF_N","60")),
            sims_lik=max(1,int(os.getenv("PF_UPD_SIMS","30"))),
            resample_thr=float(os.getenv("PF_RESAMPLE","0.7")),
            backend=os.getenv("PF_BACKEND","mc"),
            dirichlet_eps=float(os.getenv("PF_DIR_EPS","0.003")),
        )
        log.info("PF init ok: n=%s backend=%s", PF.n_particles, getattr(PF,"backend","?"))
    except Exception as e:
        log.error("PF init fail: %s", e)
        class _Dummy:  # very minimal fallback
            def update_outcome(self, outcome): pass
            def predict(self, **k): return np.array([0.48,0.47,0.05], dtype=np.float32)
        PF = _Dummy()
else:
    class _Dummy:
        def update_outcome(self, outcome): pass
        def predict(self, **k): return np.array([0.48,0.47,0.05], dtype=np.float32)
    PF = _Dummy()

# ---------- 決策/配注 ----------
EDGE_ENTER = float(os.getenv("EDGE_ENTER","0.03"))
MIN_BET_PCT = float(os.getenv("MIN_BET_PCT","0.05"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT","0.40"))
PROB_SMA_ALPHA = float(os.getenv("PROB_SMA_ALPHA","0.45"))
PROB_TEMP = float(os.getenv("PROB_TEMP","1.0"))

# Thompson Scaling（只縮放配注，不改選邊）
TS_EN = env_flag("TS_EN", 0)
TS_ALPHA = float(os.getenv("TS_ALPHA","2"))
TS_BETA  = float(os.getenv("TS_BETA","2"))

# 不確定性懲罰
UNCERT_PENALTY_EN = env_flag("UNCERT_PENALTY_EN", 1)
UNCERT_MARGIN_MAX = int(os.getenv("UNCERT_MARGIN_MAX","1"))
UNCERT_RATIO = float(os.getenv("UNCERT_RATIO","0.33"))

# 連續點差加權（含上一手機率差）
W_BASE = float(os.getenv("W_BASE","1.0"))
W_MIN  = float(os.getenv("W_MIN","0.5"))
W_MAX  = float(os.getenv("W_MAX","2.8"))
W_ALPHA= float(os.getenv("W_ALPHA","0.95"))
W_SIG_K= float(os.getenv("W_SIG_K","1.10"))
W_SIG_MID=float(os.getenv("W_SIG_MID","1.8"))
W_GAMMA= float(os.getenv("W_GAMMA","1.0"))
W_GAP_CAP=float(os.getenv("W_GAP_CAP","0.06"))

# 手數深度權重
DEPTH_W_EN  = env_flag("DEPTH_W_EN", 1)
DEPTH_W_MAX = float(os.getenv("DEPTH_W_MAX","1.3"))

# 點差可靠度表（5桶）
MREL_EN = env_flag("MREL_EN", 1)
MREL_LR = float(os.getenv("MREL_LR","0.02"))

INV = {0:"莊", 1:"閒"}

# ---------- 平滑工具 ----------
def softmax_temp(p: np.ndarray, t: float) -> np.ndarray:
    t = max(1e-6, float(t))
    x = np.log(np.clip(p,1e-9,1.0)) / t
    x = np.exp(x - np.max(x))
    x = x / np.sum(x)
    return x

def ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None: return cur
    return alpha*cur + (1-alpha)*prev

# ---------- 連續權重：點差 + 上一手機率差 ----------
def calc_margin_weight(p_pts: int, b_pts: int, last_prob_gap: float) -> float:
    margin = abs(int(p_pts) - int(b_pts))
    sig = 1.0/(1.0 + math.exp(-W_SIG_K * (margin - W_SIG_MID)))
    part_m = W_ALPHA * sig
    gap_norm = min(max(float(last_prob_gap),0.0), W_GAP_CAP) / max(W_GAP_CAP,1e-6)
    part_g = W_GAMMA * gap_norm
    w = W_BASE + part_m + part_g
    return max(W_MIN, min(W_MAX, w))

# ---------- 可靠度表（5桶：0,1,2,3,4+） ----------
def margin_bucket(margin: int) -> int:
    return 4 if margin>=4 else margin

def mrel_score(sess: Dict[str,Any], margin: int) -> float:
    if not MREL_EN: return 1.0
    b = margin_bucket(margin)
    a = sess["mrel"]["a"][b]; bb = sess["mrel"]["b"][b]
    return (a)/(a+bb)  # Beta 均值

def mrel_update(sess: Dict[str,Any], margin: int, correct: bool):
    if not MREL_EN: return
    b = margin_bucket(margin)
    # 輕量增減，避免失控
    if correct:
        sess["mrel"]["a"][b] = max(1.0, sess["mrel"]["a"][b] + MREL_LR)
    else:
        sess["mrel"]["b"][b] = max(1.0, sess["mrel"]["b"][b] + MREL_LR)

# ---------- 決策：僅莊/閒（含 5% 抽水） ----------
def decide_bp(prob: np.ndarray) -> Tuple[str, float, float]:
    pB, pP = float(prob[0]), float(prob[1])
    evB, evP = 0.95*pB - pP, pP - pB
    side = 0 if evB>evP else 1
    edge = max(abs(evB), abs(evP))
    return (INV[side], edge, max(pB,pP))

def bet_amount(bankroll: int, pct: float) -> int:
    if bankroll<=0 or pct<=0: return 0
    return int(round(bankroll * pct))

def confidence_to_pct(edge: float, max_prob: float) -> float:
    base_conf = min(1.0, edge*15.0)
    prob_conf = max(0.0, (max_prob-0.45)*2.5)
    total = 0.5*base_conf + 0.5*prob_conf
    pct = MIN_BET_PCT + (total**0.8) * (MAX_BET_PCT-MIN_BET_PCT)
    return max(MIN_BET_PCT, min(MAX_BET_PCT, pct))

def thompson_scale(pct: float) -> float:
    if not TS_EN: return pct
    # 僅縮放，不改方向
    a = max(1e-3, TS_ALPHA); b = max(1e-3, TS_BETA)
    s = np.random.beta(a, b)
    return max(MIN_BET_PCT, min(MAX_BET_PCT, pct*s))

# ---------- 主流程：收到上一局點數，更新並預測 ----------
_prev_prob_sma: Optional[np.ndarray] = None

def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    global _prev_prob_sma

    # 手數+1（每桌綁定）
    sess["hand_idx"] = int(sess.get("hand_idx", 0)) + 1
    margin = abs(p_pts - b_pts)

    # 1) 更新 PF（連續權重 + 不確定懲罰 + 深度權重）
    last_gap = float(sess.get("last_prob_gap", 0.0))
    w = calc_margin_weight(p_pts, b_pts, last_gap)

    if DEPTH_W_EN and sess["hand_idx"]>0:
        depth_boost = 1.0 + min(sess["hand_idx"]/70.0, (DEPTH_W_MAX-1.0))
        w *= depth_boost

    rep = max(1, min(3, int(round(w))))

    # 和局：更新一次 outcome=2；不吃懲罰
    if p_pts == b_pts:
        try:
            PF.update_outcome(2)
        except Exception as e:
            log.warning("PF tie update err: %s", e)
    else:
        outcome = 1 if p_pts > b_pts else 0
        for _ in range(rep):
            try:
                PF.update_outcome(outcome)
            except Exception as e:
                log.warning("PF update err: %s", e)

        # 不確定性懲罰：margin <= UNCERT_MARGIN_MAX 時，反向微更新一次
        if UNCERT_PENALTY_EN and margin <= UNCERT_MARGIN_MAX:
            rev = 0 if outcome==1 else 1
            # 以 UNCERT_RATIO 決定是否執行一次微更新（機率式）
            if random.random() < UNCERT_RATIO:
                try:
                    PF.update_outcome(rev)
                except Exception as e:
                    log.warning("PF uncert reverse update err: %s", e)

    # 2) 預測
    sims_pred = max(0, int(os.getenv("PF_PRED_SIMS","20")))
    p_raw = PF.predict(sims_per_particle=sims_pred)
    # 可靠度表加權（基於 margin 桶）
    rel = mrel_score(sess, margin)
    p_adj = np.array([p_raw[0]*rel, p_raw[1]*rel, p_raw[2]], dtype=np.float32)
    p_adj = p_adj / np.sum(p_adj)

    # 溫度 & 平滑
    p_temp = softmax_temp(p_adj, PROB_TEMP)
    _prev_prob_sma = ema(_prev_prob_sma, p_temp, PROB_SMA_ALPHA)
    p_final = _prev_prob_sma if _prev_prob_sma is not None else p_temp

    # 記錄給下一手用
    sess["last_prob_gap"] = abs(float(p_final[0]) - float(p_final[1]))

    # 3) 決策與配注
    choice, edge, maxp = decide_bp(p_final)
    bankroll = int(sess.get("bankroll", 0))
    bet_pct = confidence_to_pct(edge, maxp)
    bet_pct = thompson_scale(bet_pct)  # 只縮放，不改方向
    bet_amt = bet_amount(bankroll, bet_pct)

    # 4) 更新可靠度表（用實際 outcome 相對於「上一手」預測）
    #   這裡我們用當前 margin 對「剛完成的 outcome」做線上評分：
    #   如果 p_pts>b_pts 且我們上一手偏向閒，就算 correct；反之亦然；和局不更新。
    if p_pts != b_pts and _prev_prob_sma is not None:
        prev_choice = 1 if _prev_prob_sma[1] >= _prev_prob_sma[0] else 0
        correct = (prev_choice == (1 if p_pts>b_pts else 0))
        mrel_update(sess, margin, correct)

    # 5) 輸出
    msg = []
    if p_pts == b_pts:
        sess["last_pts_text"] = f"上局結果: 和局 (閒{p_pts} 莊{b_pts})"
    else:
        sess["last_pts_text"] = f"上局結果: 閒{p_pts} 莊{b_pts}"
    msg.append(sess["last_pts_text"])
    msg.append("開始分析下局....")
    msg.append("【預測結果】")
    msg.append(f"閒：{p_final[1]*100:.2f}%")
    msg.append(f"莊：{p_final[0]*100:.2f}%")
    msg.append(f"本次預測結果：{choice}")
    msg.append(f"建議下注：{bet_amt:,}")
    msg.append(f"(edge={edge*100:.1f}%, maxp={maxp*100:.1f}%, rep={rep}, rel={rel:.2f})")
    return "\n".join(msg)

# ---------- 路由（簡版） ----------
@app.get("/")
def root():
    return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION), 200

@app.post("/predict")
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    uid = str(data.get("uid","guest"))
    text = str(data.get("text","")).strip()
    sess = now_sess(uid)

    # 設定本金（若傳入）
    if "bankroll" in data:
        try:
            bk = int(data["bankroll"])
            if bk>0: sess["bankroll"] = bk
        except: pass

    pts = parse_last_hand_points(text)
    if pts is None:
        return jsonify(ok=False, err="無法解析點數（例：閒6莊5 / 65 / 和）"), 400

    msg = handle_points_and_predict(sess, pts[0], pts[1])
    save_sess(uid, sess)
    return jsonify(ok=True, msg=msg), 200

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
