# -*- coding: utf-8 -*-
"""
server.py — BGS百家樂AI 多步驟/館別桌號/本金/試用/永久帳號
改：觀望判斷由「單條件觸發」改為「加權評分 + 遲滯 + 冷卻 + Soft Bet」。
不動 OutcomePF/pfilter 內核與預測流程。
"""

import os, sys, re, time, json, logging, math
from typing import Dict, Any, List
import numpy as np

# ----------------- Flask -----------------
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    _has_flask = True
except Exception:
    _has_flask = False
    Flask = None
    def jsonify(*_, **__): raise RuntimeError("Flask not available")
    def CORS(*_, **__): pass

if _has_flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/")
    def root():
        return "✅ BGS PF Server OK", 200

    @app.get("/health")
    def health():
        return jsonify(ok=True, ts=time.time(), msg="API normal"), 200
else:
    class _DummyApp:
        def get(self, *a, **k):
            def deco(f): return f
            return deco
        def post(self, *a, **k):
            def deco(f): return f
            return deco
        def run(self, *a, **k): print("Flask not installed; dummy app.")
    app = _DummyApp()

# ----------------- Redis (optional) -----------------
try:
    import redis
except Exception:
    redis = None

REDIS_URL = os.getenv("REDIS_URL", "")
rcli = None
if redis and REDIS_URL:
    try:
        rcli = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        rcli.ping()
    except Exception:
        rcli = None

# ----------------- Session/metrics -----------------
SESS: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE = 3600
PF_LOAD_FAILS = 0

# ---------- Tunables / Defaults ----------
# 全域預設（可被每館別覆寫）
os.environ.setdefault("BANKER_COMMISSION", "0.05")
os.environ.setdefault("EDGE_ENTER_EV", "0.004")
os.environ.setdefault("ENTER_GAP_MIN", "0.03")
os.environ.setdefault("WATCH_INSTAB_THRESH", "0.04")
os.environ.setdefault("TIE_PROB_MAX", "0.20")
os.environ.setdefault("STATS_DISPLAY", "smart")

# 配注上下限與形狀
os.environ.setdefault("MIN_BET_PCT_BASE", "0.03")
os.environ.setdefault("MAX_BET_PCT", "0.30")
os.environ.setdefault("BET_UNIT", "100")
os.environ.setdefault("CONF_GAMMA", "1.25")

# —— 新增：觀望評分與策略 —— #
# 權重（越高代表該風險更重要）
os.environ.setdefault("W_EV_LOW_W", "2.0")
os.environ.setdefault("W_GAP_LOW_W", "1.0")
os.environ.setdefault("W_INSTAB_W", "1.0")
os.environ.setdefault("W_TIE_RISK_W", "1.0")
# 進入/維持觀望門檻（遲滯；進入門檻較高可降低「一下就觀望」）
os.environ.setdefault("WATCH_SCORE_ENTER", "2.0")
os.environ.setdefault("WATCH_SCORE_STAY", "2.0")
# 下注後冷卻：連續 N 手提高門檻、同時忽略一次性波動（instab）
os.environ.setdefault("WATCH_COOLDOWN", "1")      # 下注後 N 手
os.environ.setdefault("WATCH_COOLDOWN_BONUS", "1.0")  # 門檻加值
os.environ.setdefault("WATCH_IGNORE_INSTAB_IN_COOLDOWN", "1")  # 1=忽略 instab
# Soft Bet：未達觀望門檻但有風險 → 縮小配注
os.environ.setdefault("SOFT_BET_ENABLE", "1")
os.environ.setdefault("SOFT_BET_MIN_SCORE", "1.0")  # 風險分≥此值且<門檻 → Soft Bet
os.environ.setdefault("SOFT_BET_MULT", "0.5")       # 乘上配注比例

# 信心計算（沿用你現有邏輯；未動 OutcomePF）
os.environ.setdefault("CONF_EV_MID", "0.012")
os.environ.setdefault("CONF_EV_SPREAD", "0.006")
os.environ.setdefault("PROB_BONUS_K", "2.0")
os.environ.setdefault("CONF_PROB_WEIGHT", "0.40")
os.environ.setdefault("STREAK_W", "0.04")
os.environ.setdefault("STREAK_MIN", "2")
os.environ.setdefault("TIE_CONF_START", "0.18")
os.environ.setdefault("TIE_CONF_W", "0.30")

# indep 模式平滑（不動 predict 內核）
os.environ.setdefault("INDEP_SMOOTH", "1")
os.environ.setdefault("INDEP_TEMP", "1.0")
os.environ.setdefault("INDEP_EMA_ALPHA", "0.55")

# PF / 模式
os.environ.setdefault("MODEL_MODE", "indep")
os.environ.setdefault("DECKS", "6")
os.environ.setdefault("PF_N", "80")
os.environ.setdefault("PF_RESAMPLE", "0.73")
os.environ.setdefault("PF_DIR_EPS", "0.012")
os.environ.setdefault("PF_BACKEND", "mc")
os.environ.setdefault("PF_UPD_SIMS", "36")
os.environ.setdefault("PF_PRED_SIMS", "30")

# 先驗/抖動/歷史參數
os.environ.setdefault("PRIOR_B", "0.452")
os.environ.setdefault("PRIOR_P", "0.452")
os.environ.setdefault("PRIOR_T", "0.096")
os.environ.setdefault("PRIOR_STRENGTH", "40")
os.environ.setdefault("PF_DECAY", "0.985")
os.environ.setdefault("PROB_JITTER", "0.006")
os.environ.setdefault("HISTORICAL_WEIGHT", "0.2")

# 大分差學習權重
os.environ.setdefault("OUTCOME_REP_CAP", "3")
os.environ.setdefault("OUTCOME_GAP_T1", "6")
os.environ.setdefault("OUTCOME_GAP_T2", "8")

# 歷史保留
os.environ.setdefault("HIST_KEEP", "400")

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1","true","yes","on")

def _per_hall_val(name: str, hall_id: int, default_val: float) -> float:
    try:
        if hall_id:
            v = os.getenv(f"{name}_{hall_id}")
            if v is not None: return float(v)
    except Exception:
        pass
    return float(os.getenv(name, str(default_val)))

# ----------------- PF Loader -----------------
OutcomePF = None
try:
    from bgs.pfilter import OutcomePF
except Exception:
    try:
        cur = os.path.dirname(os.path.abspath(__file__))
        if cur not in sys.path: sys.path.insert(0, cur)
        from pfilter import OutcomePF
    except Exception:
        OutcomePF = None

class _DummyPF:
    def update_outcome(self, outcome): pass
    def predict(self, **k): return np.array([0.458, 0.446, 0.096], dtype=np.float32)
    def update_point_history(self, p_pts, b_pts): pass

def _get_pf_from_sess(sess: Dict[str, Any]) -> Any:
    global PF_LOAD_FAILS
    if OutcomePF:
        if sess.get("pf") is None:
            try:
                sess["pf"] = OutcomePF(
                    decks=int(os.getenv("DECKS","6")),
                    seed=int(os.getenv("SEED","42")) + int(time.time() % 1000),
                    n_particles=int(os.getenv("PF_N","80")),
                    sims_lik=max(1,int(os.getenv("PF_UPD_SIMS","36"))),
                    resample_thr=float(os.getenv("PF_RESAMPLE","0.73")),
                    backend=os.getenv("PF_BACKEND","mc"),
                    dirichlet_eps=float(os.getenv("PF_DIR_EPS","0.012")),
                )
                sess["pf_backend"] = "real"
            except Exception:
                PF_LOAD_FAILS += 1
                sess["pf"] = _DummyPF()
                sess["pf_backend"] = "dummy"
        return sess["pf"]
    PF_LOAD_FAILS += 1
    sess["pf_backend"] = "dummy"
    return _DummyPF()

# ----------------- Trial / Open -----------------
TRIAL_SECONDS = int(os.getenv("TRIAL_SECONDS", "1800"))
OPENCODE = os.getenv("OPENCODE", "aaa8881688")
ADMIN_LINE = os.getenv("ADMIN_LINE", "https://lin.ee/Dlm6Y3u")

def _now(): return int(time.time())

def _get_user_info(user_id):
    k = f"bgsu:{user_id}"
    if rcli:
        s = rcli.get(k)
        if s: return json.loads(s)
    return SESS.get(user_id, {})

def _set_user_info(user_id, info):
    k = f"bgsu:{user_id}"
    if rcli: rcli.set(k, json.dumps(info), ex=86400)
    SESS[user_id] = info

def _is_trial_valid(user_id):
    info = _get_user_info(user_id)
    if info.get("is_opened"): return True
    if not info.get("trial_start"): return False
    return (_now() - int(info["trial_start"])) < TRIAL_SECONDS

def _start_trial(user_id):
    info = _get_user_info(user_id)
    if info.get("is_opened"): return
    if not info.get("trial_start"):
        info["trial_start"] = _now()
        _set_user_info(user_id, info)

def _set_opened(user_id):
    info = _get_user_info(user_id)
    info["is_opened"] = True
    _set_user_info(user_id, info)

def _left_trial_sec(user_id):
    info = _get_user_info(user_id)
    if info.get("is_opened"): return "永久"
    if not info.get("trial_start"): return "尚未啟動"
    left = TRIAL_SECONDS - (_now() - int(info["trial_start"]))
    return f"{left//60} 分 {left%60} 秒" if left > 0 else "已到期"

# ----------------- Confidence helpers -----------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _streak_info(labels: List[str]) -> (str, int):
    if not labels: return "", 0
    last = labels[-1]
    if last not in ("莊","閒"): return "", 0
    cnt = 0
    for v in reversed(labels):
        if v == last: cnt += 1
        else: break
    return last, cnt

def calculate_adjusted_confidence(ev_b, ev_p, pB, pP, pT, choice, sess):
    sel_ev = max(0.0, ev_b if choice == "莊" else ev_p)
    ev_mid    = float(os.getenv("CONF_EV_MID", "0.012"))
    ev_spread = max(1e-9, float(os.getenv("CONF_EV_SPREAD", "0.006")))
    k_prob    = float(os.getenv("PROB_BONUS_K", "2.0"))
    w_prob    = float(os.getenv("CONF_PROB_WEIGHT", "0.40"))
    w_ev = 1.0 - w_prob

    ev_conf = _sigmoid((sel_ev - ev_mid) / ev_spread)
    prob_conf = 1.0 - math.exp(-k_prob * abs(pB - pP))
    conf = w_ev * ev_conf + w_prob * prob_conf

    streak_w = float(os.getenv("STREAK_W","0.04"))
    streak_min = int(os.getenv("STREAK_MIN","2"))
    d, n = _streak_info(sess.get("hist_real", []))
    if n >= streak_min and d in ("莊","閒"):
        adj = math.log1p(n) * streak_w
        if d == choice: conf += adj
        else:           conf -= adj

    tie_start = float(os.getenv("TIE_CONF_START","0.18"))
    tie_w     = float(os.getenv("TIE_CONF_W","0.30"))
    if pT > tie_start:
        conf *= max(0.0, 1.0 - (pT - tie_start) * tie_w)

    return max(0.0, min(1.0, conf))

def get_stats_display(sess):
    mode = os.getenv("STATS_DISPLAY", "smart").strip().lower()
    if mode == "none": return None
    pred, real = sess.get("hist_pred", []), sess.get("hist_real", [])
    if not pred or not real: return "📊 數據收集中..."
    bet_pairs = [(p,r) for p,r in zip(pred,real) if r in ("莊","閒") and p in ("莊","閒")]
    if not bet_pairs: return "📊 尚未進行下注"
    hit = sum(1 for p,r in bet_pairs if p==r)
    total = len(bet_pairs)
    acc = 100.0 * hit / total
    if mode == "smart":
        if total >= 15: return f"🎯 近期勝率：{acc:.1f}%"
        if total >= 5:  return f"🎯 當前勝率：{acc:.1f}% ({hit}/{total})"
        return f"🎯 初始勝率：{acc:.1f}% ({hit}/{total})"
    else:
        total_hands = len([r for r in real if r in ("莊","閒")])
        watched = total_hands - total
        base = f"📊 下注勝率：{acc:.1f}% ({hit}/{total})"
        return f"{base} | 觀望：{watched}手" if watched>0 else base

def _format_pts_text(p_pts, b_pts):
    if p_pts == b_pts: return f"上局結果: 和 {p_pts}"
    return f"上局結果: 閒 {p_pts} 莊 {b_pts}"

# ----------------- LINE SDK -----------------
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction
)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_TIMEOUT = float(os.getenv("LINE_TIMEOUT", "2.0"))

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN, timeout=LINE_TIMEOUT)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def _qr_btn(label, text):
    return QuickReplyButton(action=MessageAction(label=label, text=text))

def _reply(token, text, quick=None):
    try:
        if quick:
            line_bot_api.reply_message(
                token,
                TextSendMessage(text=text, quick_reply=QuickReply(items=quick))
            )
        else:
            line_bot_api.reply_message(token, TextSendMessage(text=text))
    except Exception as e:
        print("LINE reply_message error:", e)

def welcome_text(uid):
    left = _left_trial_sec(uid)
    return (
        "👋 歡迎使用 BGS AI 預測分析！\n"
        "【使用步驟】\n"
        "1️⃣ 選擇館別（輸入 1~10）\n"
        "2️⃣ 輸入桌號（例：DG01）\n"
        "3️⃣ 輸入本金（例：5000）\n"
        "4️⃣ 每局回報點數（例：65 / 和 / 閒6莊5）\n"
        f"💾 試用剩餘：{left}\n\n"
        "【請選擇遊戲館別】\n"
        "1. WM\n2. PM\n3. DG\n4. SA\n5. KU\n6. 歐博/卡利\n7. KG\n8. 金利\n9. 名人\n10. MT真人\n"
        "(請直接輸入數字1-10)"
    )

def settings_quickreply(sess) -> list:
    return [
        _qr_btn("選館別", "設定 館別"),
        _qr_btn("查看統計", "查看統計"),
        _qr_btn("試用剩餘", "試用剩餘"),
        _qr_btn("顯示模式 smart", "顯示模式 smart"),
        _qr_btn("顯示模式 basic", "顯示模式 basic"),
        _qr_btn("顯示模式 none", "顯示模式 none"),
        _qr_btn("重設流程", "重設"),
    ]

def halls_quickreply() -> list:
    return [_qr_btn(f"{i}", f"{i}") for i in range(1, 11)]

# ----------------- Core Predict Flow -----------------
def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    if not (p_pts == 0 and b_pts == 0):
        if not (0 <= int(p_pts) <= 9 and 0 <= int(b_pts) <= 9):
            return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    pf = _get_pf_from_sess(sess)

    # ===== 記錄 outcome =====
    if p_pts == b_pts and not (p_pts == 0 and b_pts == 0):
        try: pf.update_outcome(2)
        except Exception: pass
        real_label = "和"
    elif p_pts == 0 and b_pts == 0:
        try: pf.update_outcome(2)
        except Exception: pass
        real_label = "和"
    else:
        pf.update_point_history(p_pts, b_pts)
        sess["hand_idx"] = int(sess.get("hand_idx", 0)) + 1
        gap = abs(p_pts - b_pts)
        rep_cap = int(os.getenv("OUTCOME_REP_CAP","3"))
        t1 = int(os.getenv("OUTCOME_GAP_T1","6"))
        t2 = int(os.getenv("OUTCOME_GAP_T2","8"))
        rep = 1
        if gap >= t2: rep = min(rep_cap, 3)
        elif gap >= t1: rep = min(rep_cap, 2)
        outcome = 1 if p_pts > b_pts else 0
        real_label = "閒" if p_pts > b_pts else "莊"
        for _ in range(rep):
            try: pf.update_outcome(outcome)
            except Exception: pass

    # ===== 預測 =====
    sims_pred = int(os.getenv("PF_PRED_SIMS","30"))
    p_raw = pf.predict(sims_per_particle=sims_pred)
    p_final = p_raw / np.sum(p_raw)

    mode = os.getenv("MODEL_MODE","indep").strip().lower()
    if mode == "indep":
        if _env_flag("INDEP_SMOOTH","1"):
            temp = float(os.getenv("INDEP_TEMP","1.0"))
            p_temp = np.exp(np.log(np.clip(p_final,1e-9,1.0)) / max(1e-9,temp))
            p_temp = p_temp / np.sum(p_temp)
            a = float(os.getenv("INDEP_EMA_ALPHA","0.55"))
            prev = sess.get("indep_ema")
            sess["indep_ema"] = p_temp if prev is None else (a*p_temp + (1.0-a)*prev)
            p_final = sess["indep_ema"]
        p_final = np.clip(p_final, 0.01, 0.98); p_final = p_final / np.sum(p_final)
    else:
        p_temp = np.exp(np.log(np.clip(p_final,1e-9,1.0)) / float(os.getenv("PROB_TEMP","1.0")))
        p_temp = p_temp / np.sum(p_temp)
        alpha = float(os.getenv("PROB_SMA_ALPHA","0.60"))
        def ema(prev, cur, a): return cur if prev is None else a*cur + (1-a)*prev
        sess["prob_sma"] = ema(sess.get("prob_sma"), p_temp, alpha)
        p_final = sess["prob_sma"] if sess["prob_sma"] is not None else p_temp

    pB, pP, pT = float(p_final[0]), float(p_final[1]), float(p_final[2])

    # ===== EV（tie=0 EV）=====
    BCOMM = float(os.getenv("BANKER_COMMISSION","0.05"))
    ev_b = pB * (1.0 - BCOMM) - (1.0 - pB - pT)
    ev_p = pP * 1.0            - (1.0 - pP - pT)

    ev_choice = "莊" if ev_b > ev_p else "閒"
    edge_ev = max(ev_b, ev_p)
    if abs(ev_b - ev_p) < 0.005:
        ev_choice = "莊" if pB > pP else "閒"
        edge_ev = max(ev_b, ev_p) + 0.002
    if np.isnan(p_final).any() or np.sum(p_final) < 0.99:
        ev_choice = "莊" if pB > pP else "閒"; edge_ev = 0.015

    # ===== 觀望條件（新：評分制+遲滯+冷卻）=====
    hall_id = int(SESS.get(sess.get("user_id",""),{}).get("hall_id", sess.get("hall_id", 0)) or sess.get("hall_id",0))
    EDGE_ENTER_EV = _per_hall_val("EDGE_ENTER_EV", hall_id, float(os.getenv("EDGE_ENTER_EV","0.004")))
    ENTER_GAP_MIN = _per_hall_val("ENTER_GAP_MIN", hall_id, float(os.getenv("ENTER_GAP_MIN","0.03")))
    TIE_MAX       = _per_hall_val("TIE_PROB_MAX", hall_id, float(os.getenv("TIE_PROB_MAX","0.20")))
    INSTAB_TH     = _per_hall_val("WATCH_INSTAB_THRESH", hall_id, float(os.getenv("WATCH_INSTAB_THRESH","0.04")))

    # 個別風險旗標
    ev_low   = (edge_ev < EDGE_ENTER_EV)
    gap_low  = ((sorted([pB,pP,pT], reverse=True)[0] - sorted([pB,pP,pT], reverse=True)[1]) < ENTER_GAP_MIN)
    last_gap = float(sess.get("last_prob_gap", 0.0))
    instab   = (abs(edge_ev - last_gap) > INSTAB_TH)
    tie_risk = (pT > TIE_MAX and edge_ev < 0.015)

    # 權重評分
    score = 0.0
    score += float(os.getenv("W_EV_LOW_W","2.0"))  if ev_low   else 0.0
    score += float(os.getenv("W_GAP_LOW_W","1.0")) if gap_low  else 0.0
    score += float(os.getenv("W_INSTAB_W","1.0"))  if instab   else 0.0
    score += float(os.getenv("W_TIE_RISK_W","1.0"))if tie_risk else 0.0

    # 遲滯/冷卻
    was_watch = (len(sess.get("hist_pred", []))>0 and sess["hist_pred"][-1]=="觀望")
    th_enter  = float(os.getenv("WATCH_SCORE_ENTER","2.0"))
    th_stay   = float(os.getenv("WATCH_SCORE_STAY","2.0"))
    threshold = th_stay if was_watch else th_enter

    # 下注後冷卻：提高門檻、可忽略一次性 instab
    cd = int(sess.get("cooldown", 0))
    if cd > 0:
        threshold += float(os.getenv("WATCH_COOLDOWN_BONUS","1.0"))
        if _env_flag("WATCH_IGNORE_INSTAB_IN_COOLDOWN","1") and instab:
            score -= float(os.getenv("W_INSTAB_W","1.0"))
        sess["cooldown"] = cd - 1

    # 判斷觀望/Soft Bet
    watch = (score >= threshold)
    soft_bet = (not watch) and _env_flag("SOFT_BET_ENABLE","1") and (score >= float(os.getenv("SOFT_BET_MIN_SCORE","1.0")))
    reasons = []
    if ev_low:   reasons.append("EV優勢不足")
    if gap_low:  reasons.append("勝率差不足")
    if instab:   reasons.append("勝率波動大")
    if tie_risk: reasons.append("和局風險高")

    # ===== 配注 =====
    bankroll = int(sess.get("bankroll", 0))
    min_pct = float(os.getenv("MIN_BET_PCT_BASE","0.03"))
    max_pct = float(os.getenv("MAX_BET_PCT","0.30"))
    gamma   = max(0.5, float(os.getenv("CONF_GAMMA","1.25")))
    bet_pct = 0.0; bet_amt = 0

    if not watch:
        conf = calculate_adjusted_confidence(ev_b, ev_p, pB, pP, pT, ev_choice, sess)
        bet_pct = min_pct + (max_pct - min_pct) * (conf ** gamma)
        if soft_bet:
            bet_pct *= float(os.getenv("SOFT_BET_MULT","0.5"))
        if bankroll > 0 and bet_pct > 0:
            unit = int(os.getenv("BET_UNIT","100"))
            bet_amt = int(round(bankroll * bet_pct))
            bet_amt = max(0, int(round(bet_amt / unit)) * unit)
        # 設置冷卻（下注後才啟動）
        sess["cooldown"] = int(os.getenv("WATCH_COOLDOWN","1"))

    # ===== 統計 =====
    st = sess.setdefault("stats", {"bets":0,"wins":0,"push":0,"sum_edge":0.0,"payout":0,"watch":0,"low":0,"mid":0,"high":0})
    if real_label == "和":
        st["push"] += 1
    else:
        if watch:
            st["watch"] += 1
        else:
            st["bets"] += 1; st["sum_edge"] += float(edge_ev)
            if ev_choice == real_label:
                if real_label == "莊": st["payout"] += int(round(bet_amt * (1.0 - BCOMM)))
                else:                 st["payout"] += int(bet_amt)
                st["wins"] += 1
            else:
                st["payout"] -= int(bet_amt)

    # 信心等級標籤
    low_cut = min_pct + (max_pct - min_pct) * 0.33
    mid_cut = min_pct + (max_pct - min_pct) * 0.66
    if watch:
        strat = f"⚠️ 觀望（{'、'.join(reasons)}；score={score:.2f}/{threshold:.2f}）"
    else:
        tag = "🟡 低信心配注" if bet_pct<low_cut else ("🟠 中信心配注" if bet_pct<mid_cut else "🟢 高信心配注")
        if soft_bet: tag += "（Soft）"
        strat = f"{tag} {bet_pct*100:.1f}%"

    # 歷史
    hist_keep = int(os.getenv("HIST_KEEP","400"))
    pred_label = "觀望" if watch else ev_choice
    sess.setdefault("hist_pred", []).append(pred_label)
    sess.setdefault("hist_real", []).append(real_label)
    sess["hist_pred"] = sess["hist_pred"][-hist_keep:]
    sess["hist_real"] = sess["hist_real"][-hist_keep:]
    sess["last_pts_text"] = _format_pts_text(p_pts, b_pts) if not (p_pts==0 and b_pts==0) else "上局結果: 和"
    sess["last_prob_gap"] = edge_ev

    stats_display = get_stats_display(sess)
    msg = [
        sess["last_pts_text"],
        "開始分析下局....",
        "",
        "【預測結果】",
        f"閒：{p_final[1]*100:.2f}%",
        f"莊：{p_final[0]*100:.2f}%",
        f"和：{p_final[2]*100:.2f}%",
        f"本次預測結果：{'觀望' if watch else ev_choice} (EV優勢: {edge_ev*100:.2f}%)",
        f"建議下注金額：{bet_amt:,}",
        f"配注策略：{strat}",
    ]
    if stats_display: msg.append(stats_display)
    msg.extend([
        "—",
        "🔁 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）",
        "⚙️ 輸入「設定」可開啟功能按鈕；或點「選館別」看 1~10"
    ])
    return "\n".join(msg)

def _format_stats(sess):
    st = sess.get("stats", {"bets":0,"wins":0,"push":0,"sum_edge":0.0,"payout":0,"watch":0,"low":0,"mid":0,"high":0})
    bets, wins, push, payout = st["bets"], st["wins"], st["push"], st["payout"]
    acc = (wins / bets * 100.0) if bets>0 else 0.0
    return f"📈 累計：下注 {bets}｜命中 {wins}（{acc:.1f}%）｜和 {push}｜觀望 {st.get('watch',0)}｜盈虧 {payout}｜配注(L/M/H) {st.get('low',0)}/{st.get('mid',0)}/{st.get('high',0)}"

# ----------------- LINE webhook -----------------
@app.post("/line-webhook")
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        print("LINE webhook error:", e)
        return "bad request", 400
    return "ok", 200

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    if text.startswith("開通"):
        pwd = text[2:].strip()
        reply = "✅ 已開通成功！" if pwd == OPENCODE else "❌ 開通碼錯誤，請重新輸入。"
        if pwd == OPENCODE: _set_opened(user_id)
        _reply(event.reply_token, reply, quick=settings_quickreply(SESS.setdefault(user_id, {})))
        return

    if not _is_trial_valid(user_id):
        _reply(event.reply_token, f"⛔ 試用期已到\n📬 請聯繫管理員開通登入帳號\n👉 加入官方 LINE：{ADMIN_LINE}")
        return

    _start_trial(user_id)
    sess = SESS.setdefault(user_id, {"bankroll": 0})
    sess["user_id"] = user_id

    if text in ("設定","⋯","menu","Menu"):
        _reply(event.reply_token, "⚙️ 設定選單：", quick=settings_quickreply(sess)); return
    if text == "查看統計":
        _reply(event.reply_token, _format_stats(sess), quick=settings_quickreply(sess)); return
    if text == "試用剩餘":
        _reply(event.reply_token, f"⏳ 試用剩餘：{_left_trial_sec(user_id)}", quick=settings_quickreply(sess)); return
    if text.startswith("顯示模式"):
        mode = text.replace("顯示模式","").strip().lower()
        if mode in ("smart","basic","none"):
            os.environ["STATS_DISPLAY"] = mode
            _reply(event.reply_token, f"✅ 已切換顯示模式為 {mode}", quick=settings_quickreply(sess))
        else:
            _reply(event.reply_token, "可選：smart / basic / none", quick=settings_quickreply(sess))
        return
    if text == "重設":
        SESS[user_id] = {"bankroll": 0, "user_id": user_id}
        _reply(event.reply_token, "✅ 已重設流程，請選擇館別：", quick=halls_quickreply()); return

    if not sess.get("hall_id"):
        if text.isdigit() and 1 <= int(text) <= 10:
            sess["hall_id"] = int(text)
            hall_map = ["WM", "PM", "DG", "SA", "KU", "歐博/卡利", "KG", "金利", "名人", "MT真人"]
            hall_name = hall_map[int(text)-1]
            _reply(event.reply_token, f"✅ 已選 [{hall_name}]\n請輸入桌號（例：DG01，格式：2字母+2數字）", quick=settings_quickreply(sess))
        elif text == "設定 館別":
            _reply(event.reply_token, "請選擇館別（1-10）：", quick=halls_quickreply())
        else:
            _reply(event.reply_token, welcome_text(user_id), quick=halls_quickreply())
        return

    if not sess.get("table_id"):
        m = re.match(r"^[a-zA-Z]{2}\d{2}$", text)
        if m:
            sess["table_id"] = text.upper()
            _reply(event.reply_token, f"✅ 已設桌號 [{sess['table_id']}]\n請輸入您的本金（例：5000）", quick=settings_quickreply(sess))
        else:
            _reply(event.reply_token, "請輸入正確格式的桌號（例：DG01，格式：2字母+2數字）", quick=settings_quickreply(sess))
        return

    if not sess.get("bankroll") or sess["bankroll"] <= 0:
        m = re.match(r"^(\d{3,7})$", text)
        if m:
            sess["bankroll"] = int(text)
            _reply(event.reply_token, f"👍 已設定本金：{sess['bankroll']:,}\n請輸入上一局點數（例：65 / 和 / 閒6莊5），之後能連續傳手。", quick=settings_quickreply(sess))
        else:
            _reply(event.reply_token, "請輸入正確格式的本金（例：5000）", quick=settings_quickreply(sess))
        return

    try:
        if text.strip() == "和":
            pf = _get_pf_from_sess(sess)
            try: pf.update_outcome(2)
            except Exception: pass
            reply = handle_points_and_predict(sess, 0, 0)
        elif re.fullmatch(r"\d{2}", text):
            p_pts, b_pts = int(text[0]), int(text[1])
            reply = handle_points_and_predict(sess, p_pts, b_pts)
        elif re.search("閒(\d+).*莊(\d+)", text):
            mm = re.search("閒(\d+).*莊(\d+)", text)
            reply = handle_points_and_predict(sess, int(mm.group(1)), int(mm.group(2)))
        elif re.search("莊(\d+).*閒(\d+)", text):
            mm = re.search("莊(\d+).*閒(\d+)", text)
            reply = handle_points_and_predict(sess, int(mm.group(2)), int(mm.group(1)))
        else:
            reply = "請輸入正確格式，例如 65（閒6莊5），或『閒6莊5／莊5閒6／和』"
    except Exception as e:
        reply = f"❌ 輸入格式有誤: {e}"

    _reply(event.reply_token, reply, quick=settings_quickreply(sess))

# ----------------- Diag -----------------
@app.get("/diag")
def diag():
    return jsonify(ts=_now(), sessions=len(SESS), pf_load_fails=PF_LOAD_FAILS), 200

# ----------------- Run -----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    log = logging.getLogger("bgs-server")
    log.info("Starting BGS-PF on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=False)
