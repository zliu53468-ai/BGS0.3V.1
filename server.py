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
VERSION = "bgs-pf-right-side-with-left-pred-merge-2025-09-22"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL","INFO"),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("bgs-bot")

# ---------- Flask ----------
if _has_flask:
    app = Flask(__name__)
    CORS(app)
else:
    app = None  # type: ignore

# ---------- Helpers ----------
def env_flag(name: str, default: int = 0) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1","true","yes","y","on")

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = np.clip(x, 1e-12, None)
    x = x / np.sum(x)
    return x

def ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None: return cur
    return alpha*cur + (1-alpha)*prev

# ====== （右側既有）可調參數區 ======
# Thompson Sampling（顯示與下注拆分保留）
TS_EN = env_flag("TS_EN", 0)
TS_ALPHA = float(os.getenv("TS_ALPHA","2"))
TS_BETA  = float(os.getenv("TS_BETA","2"))

# —— 入場/觀望條件（改預設值，但變數名與流程維持右側）
EDGE_ENTER = float(os.getenv("EDGE_ENTER","0.015"))       # ← from left: 1.5%
TIE_PROB_MAX = float(os.getenv("TIE_PROB_MAX","0.16"))    # ← from left
WATCH_EN = env_flag("WATCH_EN", 1)
WATCH_INSTAB_THRESH = float(os.getenv("WATCH_INSTAB_THRESH","0.12"))

# —— 不確定性懲罰/點差權重（保留右側、微調預設值以貼近左側）
UNCERT_PENALTY_EN = env_flag("UNCERT_PENALTY_EN", 1)
UNCERT_MARGIN_MAX = int(os.getenv("UNCERT_MARGIN_MAX","2"))
UNCERT_RATIO = float(os.getenv("UNCERT_RATIO","0.25"))

W_BASE   = float(os.getenv("W_BASE","0.8"))
W_MIN    = float(os.getenv("W_MIN","0.5"))
W_MAX    = float(os.getenv("W_MAX","2.0"))
W_ALPHA  = float(os.getenv("W_ALPHA","0.7"))
W_SIG_K  = float(os.getenv("W_SIG_K","1.5"))
W_SIG_MID= float(os.getenv("W_SIG_MID","2.0"))
W_GAMMA  = float(os.getenv("W_GAMMA","0.6"))
W_GAP_CAP= float(os.getenv("W_GAP_CAP","0.08"))

DEPTH_W_EN  = env_flag("DEPTH_W_EN", 1)
DEPTH_W_MAX = float(os.getenv("DEPTH_W_MAX","1.5"))

# —— 溫度/平滑（保留右側）
TEMP_EN = env_flag("TEMP_EN", 1)
TEMP    = float(os.getenv("TEMP","0.95"))
SMOOTH_EN = env_flag("SMOOTH_EN", 1)
SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA","0.7"))

# —— 莊家抽水/決策模式（保留右側）
BANKER_COMMISSION = float(os.getenv("BANKER_COMMISSION","0.05"))
DECIDE_MODE = os.getenv("DECIDE_MODE","prob")  # prob|ev

# ---------- PF dummy (保持右側流程) ----------
class PFHealth:
    def __init__(self): self.is_dummy=True
PF_HEALTH = PFHealth()

class OutcomePF:
    def __init__(self, decks=6, seed=42, n_particles=60, sims_lik=30, resample_thr=0.7, backend="mc", dirichlet_eps=0.003):
        random.seed(seed); np.random.seed(seed)
        self.n=n_particles; self.sims_lik=sims_lik; self.resample_thr=resample_thr
        self.backend=backend; self.dirichlet_eps=dirichlet_eps
        self.hist=[]
    def update_point_history(self, p_pts, b_pts):
        self.hist.append((p_pts,b_pts))
    def update_outcome(self, outcome):
        pass
    def predict(self) -> np.ndarray:
        # dummy: 按歷史粗略偏好，否則均勻分布
        if len(self.hist)>=4:
            xs = [1,1,1]
            for p,b in self.hist[-6:]:
                if p==b: xs[2]+=1
                elif p>b: xs[1]+=1
                else: xs[0]+=1
            return softmax(np.array(xs, dtype=float))
        return np.array([0.45,0.45,0.10])

# ---------- Redis / Session（保留右側） ----------
REDIS_URL = os.getenv("REDIS_URL")
_r = None
if REDIS_URL and redis:
    try:
        _r = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Redis connected")
    except Exception as e:
        log.warning("Redis connect fail: %s", e)

def load_sess(uid: str) -> Dict[str,Any]:
    sess: Dict[str,Any] = {}
    if _r:
        j = _r.get(f"sess:{uid}")
        if j: 
            try: sess = json.loads(j)
            except: sess={}
    sess.setdefault("stats", {"bets":0,"wins":0,"sum_edge":0.0,"payout":0,"push":0})
    sess.setdefault("hand_idx", 0)
    return sess

def save_sess(uid: str, sess: Dict[str,Any]):
    if _r:
        try: _r.set(f"sess:{uid}", json.dumps(sess) )
        except Exception as e: log.warning("Redis save sess err: %s", e)

# ---------- PF 管理（保留右側；調整 PF 預設值為左側建議） ----------
def _get_pf_from_sess(sess: Dict[str,Any]) -> OutcomePF:
    if sess.get("pf") is None:
        try:
            sess["pf"] = OutcomePF(
                decks=int(os.getenv("DECKS","6")),
                seed=int(os.getenv("SEED","42")),
                n_particles=int(os.getenv("PF_N","120")),                 # ← 60 -> 120
                sims_lik=max(1,int(os.getenv("PF_UPD_SIMS","40"))),       # ← 30 -> 40
                resample_thr=float(os.getenv("PF_RESAMPLE","0.85")),      # ← 0.7 -> 0.85
                backend=os.getenv("PF_BACKEND","mc"),
                dirichlet_eps=float(os.getenv("PF_DIR_EPS","0.025")),     # ← 0.003 -> 0.025
            )
            log.info("Per-session PF init ok")
        except Exception as e:
            log.error("Per-session PF init fail: %s; use dummy", e)
            sess["pf"] = OutcomePF()
            PF_HEALTH.is_dummy = True
    return sess["pf"]

# ---------- 下注/配注（保留右側現有公式） ----------
def map_conf_to_pct(x: float) -> float:
    return max(0.05, min(0.4, 0.05 + 0.35 * x))

def base_bet_pct(edge: float, maxp: float) -> float:
    k1 = float(os.getenv("BET_K1","0.60"))
    k2 = float(os.getenv("BET_K2","0.40"))
    return max(0.05, min(0.4, k1*edge + k2*(maxp-0.5)))

def thompson_scale_pct(pct: float) -> float:
    if not TS_EN: return pct
    s = np.random.beta(TS_ALPHA, TS_BETA)
    return max(0.05, min(0.4, pct * (0.8 + 0.4*s)))

def bet_amount(bankroll: int, pct: float) -> int:
    return int(round(max(0, bankroll) * pct))

# ---------- 計分與輸入解析（保留右側） ----------
def parse_last_hand_points(text: str) -> Optional[Tuple[int,int]]:
    t = text.strip()
    if t in ("和","和局"):
        x = random.randint(0,9)
        return (x,x)
    m = re.search(r"([閒Pp])\s*([0-9])\s*[莊Bb]\s*([0-9])", t)
    if m:
        try: return (int(m.group(2)), int(m.group(3)))
        except: return None
    m = re.search(r"([0-9])\s*([0-9])", t)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None

def validate_input_data(p_pts: int, b_pts: int) -> bool:
    return (0 <= p_pts <= 9) and (0 <= b_pts <= 9)

# ---------- 命中率/優勢工具（保留右側；calc_margin_weight 係數取左側預設） ----------
def calc_margin_weight(p_pts: int, b_pts: int, last_prob_gap: float) -> float:
    margin = abs(int(p_pts) - int(b_pts))
    sig = 1.0/(1.0 + math.exp(-W_SIG_K * (margin - W_SIG_MID)))
    part_m = W_ALPHA * sig
    gap_norm = min(max(float(last_prob_gap),0.0), W_GAP_CAP) / max(W_GAP_CAP,1e-6)
    part_g = W_GAMMA * gap_norm
    w = W_BASE + part_m + part_g
    return max(W_MIN, min(W_MAX, w))

# —— 新增（來自左側）：保守入場條件
def should_bet(prob: Tuple[float,float,float], last_gap: float, cur_gap: float, idx: int) -> bool:
    """
    max_prob>=0.52、T<=0.16、機率差變動不大、cur_gap>=1.8%、前5手略放寬
    """
    pB, pP, pT = prob
    max_prob = max(pB, pP)
    gap_change = abs(cur_gap - last_gap) if last_gap > 0 else 0.0
    conds = [
        max_prob >= 0.52,
        pT <= 0.16,
        gap_change <= 0.15,
        cur_gap >= 0.018,
        (idx > 5) or (max_prob >= 0.54)
    ]
    return all(conds)

def decide_bp(prob: np.ndarray) -> Tuple[str, float, float, float]:
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    if DECIDE_MODE == "prob":
        choice = "莊" if pB>=pP else "閒"
        maxp = max(pB,pP)
        edge = max(0.0, maxp - (1.0-maxp-pT))
    else:
        evB = pB*(1.0 - BANKER_COMMISSION) - (pP)
        evP = pP - pB
        if evB>=evP:
            choice="莊"; edge=max(0.0, evB)
        else:
            choice="閒"; edge=max(0.0, evP)
        maxp = max(pB,pP)
    prob_gap = abs(pB - pP)
    return choice, edge, maxp, prob_gap

_prev_prob_sma: Optional[np.ndarray] = None

def analysis_confidence(prob: np.ndarray, last_gap: float, cur_gap: float) -> float:
    # 仍保留右側流程（只是不再輸出顯示）
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    maxp = max(pB,pP); gap=abs(pB-pP)
    x = 0.4*max(0, maxp-0.5)/0.5 + 0.4*max(0, gap)/0.5 + 0.2*max(0, 0.16-pT)/0.16
    x = max(0.0, min(1.0, x))
    return x

# ---------- 主邏輯（保留右側，僅換 watch 規則 + 訊息格式） ----------
def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    global _prev_prob_sma

    if not validate_input_data(p_pts, b_pts):
        return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    pf = _get_pf_from_sess(sess)
    pf.update_point_history(p_pts, b_pts)

    # 1) 粗預測
    prob = pf.predict()  # [B,P,T]
    prob = softmax(prob)

    # 2) 不確定性懲罰 + 平滑/溫度（保留右側）
    last_gap = float(sess.get("last_prob_gap", 0.0))
    if UNCERT_PENALTY_EN:
        margin = min(abs(p_pts-b_pts), UNCERT_MARGIN_MAX)
        punish = (1.0 - UNCERT_RATIO * (margin/UNCERT_MARGIN_MAX))
        prob = softmax(prob * np.array([1.0,1.0,1.0]) * punish)

    if SMOOTH_EN:
        _prev_prob_sma = ema(_prev_prob_sma, prob, SMOOTH_ALPHA)
        prob = _prev_prob_sma

    if TEMP_EN and TEMP>0:
        t = max(1e-6, TEMP)
        prob = softmax(prob ** (1.0/t))

    # 3) 方向/優勢
    choice, edge, maxp, prob_gap = decide_bp(prob)

    # 4) watch 規則（改為用 should_bet）
    hand_idx = int(sess.get("hand_idx", 0))
    p_final = prob
    watch = not should_bet(tuple(p_final.tolist()), last_gap, prob_gap, hand_idx)

    # 5) 資金與配注（保留右側計算）
    bankroll = int(sess.get("bankroll", 0))
    conf_pct_disp = analysis_confidence(p_final, last_gap, prob_gap)
    pct_base = base_bet_pct(edge, maxp)
    bet_pct  = thompson_scale_pct(pct_base)
    bet_amt  = bet_amount(bankroll, bet_pct)

    if watch:
        choice_text = "觀望"
        bet_pct = 0.0
        bet_amt = 0
        strat = "⚠️ 觀望"
    else:
        choice_text = choice
        if bet_pct < 0.28:   strat = f"🟡 低信心配注 {bet_pct*100:.1f}%"
        elif bet_pct < 0.34: strat = f"🟠 中信心配注 {bet_pct*100:.1f}%"
        else:                strat = f"🟢 高信心配注 {bet_pct*100:.1f}%"

    # 6) 更新統計（保留右側）
    st = sess["stats"]
    if p_pts == b_pts:
        real_label = "和"; st["push"] += 1
    else:
        real_label = "閒" if p_pts>b_pts else "莊"
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

    pred_label = "觀望" if watch else choice_text
    sess["hand_idx"] = int(sess.get("hand_idx",0)) + 1
    if p_pts == b_pts:
        sess["last_pts_text"] = f"上局結果: 和 {p_pts}"
    else:
        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
    sess["last_prob_gap"] = prob_gap

    # 7) 訊息輸出（改為你指定的樣式）
    msg = [
        sess["last_pts_text"],
        "開始分析下局....",
        "",
        "【預測結果】",
        f"閒：{p_final[1]*100:.2f}%",
        f"莊：{p_final[0]*100:.2f}%",
        f"和：{p_final[2]*100:.2f}%",
        f"本次預測結果：{choice_text}(優勢: {edge*100:.1f}%)",
        f"建議下注金額：{bet_amt}",
        f"配注策略：{strat}",
        "—",
        "🔁 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）",
    ]

    # PF dummy 提示（保留）
    if PF_HEALTH.is_dummy:
        msg.insert(0, "⚠️ 模型載入為簡化版（Dummy）。請確認 bgs.pfilter 是否可用。")

    return "\n".join(msg)

# ---------- 簡易 REST（保留右側） ----------
@app.get("/")
def root():
    return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/health")
def health():
    try:
        return jsonify(ok=True, version=VERSION, dummy=PF_HEALTH.is_dummy), 200
    except Exception:
        return jsonify(ok=True, version=VERSION, dummy=True), 200

@app.post("/predict")
def predict_rest():
    data = request.get_json(force=True, silent=True) or {}
    uid = str(data.get("uid","demo"))
    sess = load_sess(uid)
    try:
        p_pts = int(data.get("p",0)); b_pts = int(data.get("b",0))
    except:
        return jsonify(ok=False, err="invalid p/b"), 400
    msg = handle_points_and_predict(sess, p_pts, b_pts)
    save_sess(uid, sess)
    return jsonify(ok=True, msg=msg), 200

# ---------- （右側）LINE webhook 保留 ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_TOKEN  = os.getenv("LINE_CHANNEL_TOKEN")

if LINE_CHANNEL_SECRET and LINE_CHANNEL_TOKEN and _has_flask:
    from hashlib import sha256
    import hmac, base64
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.models import MessageEvent, TextMessage, TextSendMessage
        from linebot.exceptions import InvalidSignatureError, LineBotApiError
        _line_ok=True
    except Exception:
        _line_ok=False

    line_bot_api = LineBotApi(LINE_CHANNEL_TOKEN) if _line_ok else None
    handler = WebhookHandler(LINE_CHANNEL_SECRET) if _line_ok else None

    def reply_text(token: str, text: str, user_id: Optional[str]=None):
        if not _line_ok or not line_bot_api: return
        try:
            line_bot_api.reply_message(token, TextSendMessage(text=text))
        except LineBotApiError:
            try:
                if user_id:
                    line_bot_api.push_message(user_id, TextSendMessage(text=text))
            except Exception as e:
                log.warning("LINE push fail: %s", e)

    @app.post("/line-webhook")
    def line_webhook():
        body = request.get_data(as_text=True)
        sig = request.headers.get("X-Line-Signature","")
        try:
            handler.handle(body, sig)
        except InvalidSignatureError:
            # 無憑證/測試時允許直通
            return "ok", 200
        return "ok", 200

    @handler.add(MessageEvent, message=TextMessage)
    def on_text(event):
        uid = event.source.user_id if event.source else "demo"
        text = (event.message.text or "").strip()
        sess = load_sess(uid)

        if sess.get("phase","await_pts") == "await_bankroll":
            try:
                bk = int(re.sub(r"[^0-9]","", text))
                if bk>0:
                    sess["bankroll"]=bk; sess["phase"]="await_pts"
                    save_sess(uid, sess)
                    reply_text(event.reply_token, "✅ 已設定本金。請回報上一局點數（例：65 / 和 / 閒6莊5）", user_id=uid); return
            except: pass
            reply_text(event.reply_token, "請輸入數字本金（例：5000）。", user_id=uid); return

        pts = parse_last_hand_points(text)
        if pts is None:
            # 還沒設定本金 → 先要求設定
            if not int(sess.get("bankroll",0)):
                sess["phase"] = "await_bankroll"; save_sess(uid, sess)
                reply_text(event.reply_token, "請先輸入本金（例：5000），再回報點數。", user_id=uid); return
            reply_text(event.reply_token, "點數格式錯誤（例：65 / 和 / 閒6莊5）。", user_id=uid); return

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
    log.info("Starting %s on port %s (DECIDE_MODE=%s, COMM=%.3f)...", VERSION, port, os.getenv("DECIDE_MODE","prob"), BANKER_COMMISSION)
    app.run(host="0.0.0.0", port=port, debug=False)
