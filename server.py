# -*- coding: utf-8 -*-
"""
server.py — BGS百家樂AI（可一鍵覆蓋版本）
重點：
1) 維持你原本的流程/UI（QuickReply/卡片式文案/試用 30 分鐘）
2) 預測與配注「完全分離」
3) 修正「只押莊」：採用抽水後 EV + NEAR_EV 公平點判斷
4) 粒子濾波 OutcomePF 初始化參數修正（不再傳入 backend / dirichlet_eps / stability_factor）
5) 兩種模式：balanced(粒子濾波) / independent(單局規則)，以 .env 切換
"""

import os, sys, re, time, json, logging
from typing import Dict, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

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
        return "✅ BGS Server OK", 200
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
        log.info("Connected Redis ok")
    except Exception as e:
        log.warning("Redis disabled: %s", e)
        rcli = None

# ----------------- Session -----------------
SESS: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE = 3600

# ---------- Tunables / Defaults ----------
# 抽水
os.environ.setdefault("BANKER_COMMISSION", "0.05")

# 決策/觀望參數（與配注分離）
os.environ.setdefault("EDGE_ENTER_EV", "0.0015")   # EV 進場門檻（抽水後）
os.environ.setdefault("ENTER_GAP_MIN", "0.018")    # 勝率差門檻（top2 差距）
os.environ.setdefault("NEAR_EV", "0.0030")         # EV 公平點：EV 接近時改看 pB vs pP
os.environ.setdefault("TIE_PROB_MAX", "0.28")      # 和局風險上限（過高則觀望）

# 配注（與預測/決策分離）
os.environ.setdefault("MIN_BET_PCT_BASE", "0.02")  # 基礎最小下注比例（有進場時）
os.environ.setdefault("MAX_BET_PCT", "0.35")       # 單注上限（相對本金）
os.environ.setdefault("BET_UNIT", "100")           # 四捨五入單位

# 顯示
os.environ.setdefault("STATS_DISPLAY", "smart")

# 模式
os.environ.setdefault("MODEL_MODE", "balanced")    # balanced / independent
os.environ.setdefault("DECKS", "6")

# 粒子濾波（PF）參數（✅ 僅用 OutcomePF 真的有的參數）
os.environ.setdefault("PF_N", "80")
os.environ.setdefault("PF_RESAMPLE", "0.75")
os.environ.setdefault("PF_PRED_SIMS", "25")
os.environ.setdefault("PF_UPD_SIMS", "25")         # 更新/似然小模擬數
os.environ.setdefault("PF_DIR_ALPHA", "0.8")       # Dirichlet 先驗強度
os.environ.setdefault("PF_USE_EXACT", "0")         # 0=MC 前向；1=Exact-lite 前向

# 試用
TRIAL_SECONDS = int(os.getenv("TRIAL_SECONDS", "1800"))  # 30 分鐘
OPENCODE = os.getenv("OPENCODE", "aaa8881688")
ADMIN_LINE = os.getenv("ADMIN_LINE", "@jins888")  # 到期提示顯示（可填連結或 @ID）

# ----------------- PF Loader -----------------
OutcomePF = None
_pf_import_from = "none"
try:
    from bgs.pfilter import OutcomePF
    _pf_import_from = "bgs"
except Exception:
    try:
        cur = os.path.dirname(os.path.abspath(__file__))
        if cur not in sys.path: sys.path.insert(0, cur)
        from pfilter import OutcomePF
        _pf_import_from = "local"
    except Exception:
        OutcomePF = None
        _pf_import_from = "none"

PF_STATUS = {"ready": OutcomePF is not None, "error": None, "from": _pf_import_from}
log.info("OutcomePF import: %s", PF_STATUS)

class _DummyPF:
    def update_outcome(self, outcome): pass
    def predict(self, **k): return np.array([0.458, 0.446, 0.096], dtype=np.float32)
    def update_point_history(self, p_pts, b_pts): pass

def _get_pf_from_sess(sess: Dict[str, Any]) -> Any:
    """Get particle filter for the session"""
    global PF_STATUS

    if not OutcomePF:
        PF_STATUS = {"ready": False, "error": "OutcomePF module missing", "from": _pf_import_from}
        sess["_pf_dummy"] = True
        return _DummyPF()

    if sess.get("pf") is None and not sess.get("_pf_failed"):
        try:
            sess["pf"] = OutcomePF(
                decks=int(os.getenv("DECKS", "6")),
                seed=int(os.getenv("SEED", "42")) + int(time.time() % 1000),
                n_particles=int(os.getenv("PF_N", "80")),
                sims_lik=max(1, int(os.getenv("PF_UPD_SIMS", "25"))),
                resample_thr=float(os.getenv("PF_RESAMPLE", "0.75")),
                dirichlet_alpha=float(os.getenv("PF_DIR_ALPHA", "0.8")),
                use_exact=bool(int(os.getenv("PF_USE_EXACT", "0"))),
            )
            PF_STATUS = {"ready": True, "error": None, "from": _pf_import_from}
            sess.pop("_pf_dummy", None)
            log.info("OutcomePF initialised for user %s", sess.get("user_id", "unknown"))
        except Exception as exc:
            sess["_pf_failed"] = True
            sess["_pf_dummy"] = True
            sess["_pf_error_msg"] = str(exc)
            PF_STATUS = {"ready": False, "error": str(exc), "from": _pf_import_from}
            log.exception("Failed to initialise OutcomePF; falling back to dummy model")

    pf = sess.get("pf")
    if pf is None:
        sess["_pf_dummy"] = True
        if isinstance(PF_STATUS, dict) and PF_STATUS.get("error") and not sess.get("_pf_error_msg"):
            sess["_pf_error_msg"] = PF_STATUS["error"]
        return _DummyPF()

    sess.pop("_pf_dummy", None)
    sess.pop("_pf_error_msg", None)
    return pf

# ----------------- Trial / Open -----------------
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
    if not info.get("trial_start"): return True  # 第一次互動前：允許
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
    return f"{max(0,left)//60} 分 {max(0,left)%60} 秒" if left > 0 else "已到期"

# ----------------- Independent Predictor（單局規則） -----------------
class IndependentPredictor:
    def __init__(self): self.last = None
    def update_points(self, p_pts: int, b_pts: int): self.last = (p_pts, b_pts)
    def predict(self) -> np.ndarray:
        if not self.last: return np.array([0.458, 0.446, 0.096], dtype=np.float32)
        p, b = self.last; diff = abs(p-b); total = p+b
        if diff >= 6:
            return np.array([0.57, 0.38, 0.05], dtype=np.float32) if b>p else np.array([0.38, 0.57, 0.05], dtype=np.float32)
        if diff >= 4:
            return np.array([0.53, 0.42, 0.05], dtype=np.float32) if b>p else np.array([0.42, 0.53, 0.05], dtype=np.float32)
        if diff <= 1:
            return np.array([0.40, 0.40, 0.20], dtype=np.float32) if total<=6 else np.array([0.45, 0.45, 0.10], dtype=np.float32)
        return np.array([0.48, 0.47, 0.05], dtype=np.float32)

def _get_predictor_from_sess(sess: Dict[str, Any]) -> IndependentPredictor:
    if sess.get("predictor") is None: sess["predictor"] = IndependentPredictor()
    return sess["predictor"]

# ----------------- 顯示/統計 -----------------
def get_stats_display(sess):
    mode = os.getenv("STATS_DISPLAY", "smart").strip().lower()
    if mode == "none": return None
    pred, real = sess.get("hist_pred", []), sess.get("hist_real", [])
    if not pred or not real: return "📊 數據收集中..."
    bet_pairs = [(p,r) for p,r in zip(pred,real) if r in ("莊","閒") and p in ("莊","閒")]
    if not bet_pairs: return "📊 尚未進行下注"
    hit = sum(1 for p,r in bet_pairs if p==r)
    total = len(bet_pairs)
    acc = 100.0 * hit / total if total>0 else 0.0
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

def _format_stats(sess):
    st = sess.get("stats", {"bets":0,"wins":0,"push":0,"sum_edge":0.0,"payout":0})
    bets, wins, push, payout = st["bets"], st["wins"], st["push"], st["payout"]
    acc = (wins / bets * 100.0) if bets>0 else 0.0
    return f"📈 累計：下注 {bets}｜命中 {wins}（{acc:.1f}%）｜和 {push}｜盈虧 {payout}"

# ----------------- 決策與配注（分離） -----------------
def _safe_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, 1e-9, None)
    s = v.sum()
    return (v / s).astype(np.float32)

def _choose_side_and_conf(pB, pP, pT) -> Dict[str, Any]:
    BCOMM = float(os.getenv("BANKER_COMMISSION","0.05"))
    NEAR_EV = float(os.getenv("NEAR_EV","0.003"))
    EDGE_ENTER_EV = float(os.getenv("EDGE_ENTER_EV","0.0015"))
    ENTER_GAP_MIN = float(os.getenv("ENTER_GAP_MIN","0.018"))
    TIE_PROB_MAX = float(os.getenv("TIE_PROB_MAX","0.28"))

    ev_b = pB * (1.0 - BCOMM) - (1.0 - pB - pT)
    ev_p = pP * 1.0            - (1.0 - pP - pT)

    # 公平點處理：EV 很接近時改看機率大小（避免長期偏莊）
    if abs(ev_b - ev_p) < NEAR_EV:
        ev_choice = "莊" if pB > pP else "閒"
        edge_ev = max(ev_b, ev_p) + 0.001
    else:
        ev_choice = "莊" if ev_b > ev_p else "閒"
        edge_ev = max(ev_b, ev_p)

    # 觀望條件
    watch = False
    reasons = []
    if edge_ev < EDGE_ENTER_EV:
        watch = True; reasons.append("EV優勢不足")
    top2 = sorted([pB, pP, pT], reverse=True)[:2]
    if (top2[0] - top2[1]) < ENTER_GAP_MIN:
        watch = True; reasons.append("勝率差不足")
    if pT > TIE_PROB_MAX and edge_ev < 0.02:
        watch = True; reasons.append("和局風險")

    # 信心度 → 配注比例 (與決策分離)
    def calc_conf(ev_b, ev_p, pB, pP):
        edge = max(ev_b, ev_p)
        diff = abs(pB - pP)
        edge_term = min(1.0, edge / 0.06) ** 0.9
        prob_term = min(1.0, diff / 0.30) ** 0.85
        raw = 0.6 * edge_term + 0.4 * prob_term
        return float(max(0.0, min(1.0, raw ** 0.9)))

    conf = calc_conf(ev_b, ev_p, pB, pP)
    base_floor = float(os.getenv("MIN_BET_PCT_BASE", "0.02"))
    base_ceiling = 0.30
    bet_pct = 0.0
    if not watch:
        base_pct = base_floor + (base_ceiling - base_floor) * conf
        bet_pct = max(base_floor, min(float(os.getenv("MAX_BET_PCT", "0.35")), base_pct))

    return {
        "ev_choice": ev_choice,
        "edge_ev": float(edge_ev),
        "watch": watch,
        "reasons": reasons,
        "bet_pct": float(bet_pct),
        "ev_b": float(ev_b),
        "ev_p": float(ev_p),
    }

# ----------------- 核心流程：丟點數 → 更新 → 產生下一局預測 -----------------
def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    # 參數驗證
    if not (p_pts == 0 and b_pts == 0):
        if not (0 <= int(p_pts) <= 9 and 0 <= int(b_pts) <= 9):
            return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    model_mode = os.getenv("MODEL_MODE","balanced").strip().lower()

    # 更新實際結果（寫統計）
    if p_pts == b_pts and not (p_pts == 0 and b_pts == 0):
        real_label = "和"
    elif p_pts == 0 and b_pts == 0:
        real_label = "和"
    else:
        real_label = "閒" if p_pts > b_pts else "莊"

    # PF / 規則 更新
    if model_mode == "balanced":
        pf = _get_pf_from_sess(sess)
        if p_pts == b_pts:
            try: pf.update_outcome(2)
            except Exception: pass
        elif p_pts == 0 and b_pts == 0:
            try: pf.update_outcome(2)
            except Exception: pass
        else:
            try:
                pf.update_point_history(p_pts, b_pts)
                pf.update_outcome(1 if p_pts > b_pts else 0)
            except Exception as e:
                log.warning("PF update failed: %s", e)
    else:
        pred = _get_predictor_from_sess(sess)
        if not (p_pts == 0 and b_pts == 0):
            pred.update_points(p_pts, b_pts)

    # 處理上一局 pending 建議 → 寫統計
    st = sess.setdefault("stats", {"bets":0,"wins":0,"push":0,"sum_edge":0.0,"payout":0})
    if "pending_pred" in sess:
        prev_pred = sess.pop("pending_pred")
        prev_watch = bool(sess.pop("pending_watch", False))
        prev_edge = float(sess.pop("pending_edge_ev", 0.0))
        prev_bet_amt = int(sess.pop("pending_bet_amt", 0))
        prev_ev_choice = sess.pop("pending_ev_choice", None)

        # 歷史
        sess.setdefault("hist_pred", []).append("觀望" if prev_watch else (prev_ev_choice or prev_pred))
        sess.setdefault("hist_real", []).append(real_label)
        sess["hist_pred"] = sess["hist_pred"][-150:]
        sess["hist_real"] = sess["hist_real"][-150:]

        # 統計
        if not prev_watch and real_label in ("莊","閒"):
            st["bets"] += 1
            st["sum_edge"] += float(prev_edge)
            if (prev_ev_choice or prev_pred) == real_label:
                if prev_ev_choice == "莊":
                    BCOMM = float(os.getenv("BANKER_COMMISSION","0.05"))
                    st["payout"] += int(round(prev_bet_amt * (1.0 - BCOMM)))
                else:
                    st["payout"] += int(prev_bet_amt)
                st["wins"] += 1
            else:
                st["payout"] -= int(prev_bet_amt)
        elif real_label == "和":
            st["push"] += 1

    # 產生下一局預測機率
    try:
        if model_mode == "balanced":
            pf = _get_pf_from_sess(sess)
            p_raw = pf.predict(sims_per_particle=int(os.getenv("PF_PRED_SIMS","25")))
            p_final = _safe_norm(p_raw)
        else:
            pred = _get_predictor_from_sess(sess)
            p_final = _safe_norm(pred.predict())
    except Exception as e:
        log.warning("predict fallback due to %s", e)
        p_final = np.array([0.458, 0.446, 0.096], dtype=np.float32)

    # 輕度平滑（僅 balanced 用 / 或全域都可）
    alpha = 0.7
    prev_sma = sess.get("prob_sma")
    if prev_sma is None:
        sess["prob_sma"] = p_final
    else:
        sess["prob_sma"] = alpha * p_final + (1 - alpha) * prev_sma
    p_final = sess["prob_sma"]

    pB, pP, pT = float(p_final[0]), float(p_final[1]), float(p_final[2])

    # 決策（不含配注）
    dec = _choose_side_and_conf(pB, pP, pT)
    ev_choice = dec["ev_choice"]; edge_ev = dec["edge_ev"]; watch = dec["watch"]; reasons = dec["reasons"]; bet_pct = dec["bet_pct"]

    # 計算下注金額（與決策分離）
    bankroll = int(sess.get("bankroll", 0))
    bet_amt = 0
    if not watch and bankroll > 0 and bet_pct > 0:
        unit = int(os.getenv("BET_UNIT", "100"))
        bet_amt = int(round(bankroll * bet_pct))
        bet_amt = max(0, int(round(bet_amt / unit)) * unit)

    # 存 pending（下一次回報時配對）
    sess["pending_pred"] = "觀望" if watch else ev_choice
    sess["pending_watch"] = bool(watch)
    sess["pending_edge_ev"] = float(edge_ev)
    sess["pending_bet_amt"] = int(bet_amt)
    sess["pending_ev_choice"] = ev_choice

    sess["last_pts_text"] = _format_pts_text(p_pts, b_pts) if not (p_pts==0 and b_pts==0) else "上局結果: 和"
    sess["last_prob_gap"] = edge_ev

    stats_display = get_stats_display(sess)
    strat = f"⚠️ 觀望（{'、'.join(reasons)}）" if watch else (
        f"🟡 低信心配注 {bet_pct*100:.1f}%" if bet_pct < 0.15 else
        f"🟠 中信心配注 {bet_pct*100:.1f}%" if bet_pct < 0.25 else
        f"🟢 高信心配注 {bet_pct*100:.1f}%"
    )

    msg = [
        sess["last_pts_text"],
        f"開始{'平衡' if model_mode=='balanced' else '獨立'}分析下局....",
        "",
        "【預測結果】",
        f"閒：{p_final[1]*100:.2f}%",
        f"莊：{p_final[0]*100:.2f}%",
        f"和：{p_final[2]*100:.2f}%",
        f"本次預測：{'觀望' if watch else ev_choice} (EV優勢: {edge_ev*100:.2f}%)",
        f"建議下注金額：{bet_amt:,}",
        f"配注策略：{strat}",
    ]

    if sess.get("_pf_dummy"):
        warn = sess.get("_pf_error_msg") or (PF_STATUS.get("error") if isinstance(PF_STATUS, dict) else None)
        detail = f"（{warn}）" if warn else ""
        msg.append(f"⚠️ 預測引擎載入失敗，僅提供靜態機率{detail}".strip())

    if stats_display: msg.append(stats_display)
    msg.extend([
        "—",
        "🔁 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）",
        "⚙️ 輸入「設定」可開啟功能按鈕"
    ])
    return "\n".join(msg)

# ----------------- LINE SDK -----------------
_has_line = True
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import (
        MessageEvent, TextMessage, TextSendMessage,
        QuickReply, QuickReplyButton, MessageAction
    )
except Exception as e:
    _has_line = False
    LineBotApi = WebhookHandler = None
    MessageEvent = TextMessage = TextSendMessage = QuickReply = QuickReplyButton = MessageAction = object
    log.warning("LINE SDK not available, falling back to Dummy LINE mode: %s", e)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_TIMEOUT = float(os.getenv("LINE_TIMEOUT", "2.0"))

if _has_line and LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN, timeout=LINE_TIMEOUT)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
    LINE_MODE = "real"
else:
    LINE_MODE = "dummy"
    class _DummyHandler:
        def add(self, *a, **k):
            def deco(f): return f
            return deco
        def handle(self, body, signature):
            log.info("[DummyLINE] handle called")
    class _DummyLineAPI:
        def reply_message(self, token, message):
            try:
                txt = message.text if hasattr(message, "text") else str(message)
            except Exception:
                txt = str(message)
            log.info("[DummyLINE] reply: %s", txt)
    handler = _DummyHandler()
    line_bot_api = _DummyLineAPI()
    log.warning("LINE credentials missing or SDK unavailable; running in Dummy LINE mode.")

def _qr_btn(label, text):
    if LINE_MODE == "real":
        return QuickReplyButton(action=MessageAction(label=label, text=text))
    return {"label": label, "text": text}

def _reply(token, text, quick=None):
    try:
        if LINE_MODE == "real":
            if quick:
                line_bot_api.reply_message(
                    token,
                    TextSendMessage(text=text, quick_reply=QuickReply(items=quick))
                )
            else:
                line_bot_api.reply_message(token, TextSendMessage(text=text))
        else:
            log.info("[DummyLINE] reply%s: %s", " (with quick)" if quick else "", text)
    except Exception as e:
        log.warning("LINE reply_message error: %s", e)

# —— 歡迎文案 —— #
def welcome_text(uid):
    left = _left_trial_sec(uid)
    return (
        "👋 歡迎使用 BGS AI 系統！\n"
        "【使用步驟】\n"
        "1️⃣ 選擇館別（輸入 1~10）\n"
        "2️⃣ 輸入桌號（例：DG01）\n"
        "3️⃣ 輸入本金（例：5000）\n"
        "4️⃣ 每局回報點數（例：65 / 和 / 閒6莊5）\n"
        f"💾 試用剩餘：{left}\n\n"
        "【請選擇遊戲館別】\n"
        "1. WM\n2. PM\n3. DG\n4. SA\n5. KU\n"
        "6. 歐博/卡利\n7. KG\n8. 金利\n9. 名人\n10. MT真人\n"
        "(請直接輸入數字1-10)"
    )

def settings_quickreply(sess) -> list:
    return [
        _qr_btn("選館別", "設定 館別"),
        _qr_btn("查看統計", "查看統計"),
        _qr_btn("試用剩餘", "試用剩餘"),
        _qr_btn("重設流程", "重設"),
    ]

def halls_quickreply() -> list:
    return [_qr_btn(f"{i}", f"{i}") for i in range(1, 11)]

# ----------------- HTTP routes -----------------
if _has_flask:
    @app.get("/health")
    def health():
        return jsonify(
            ok=True,
            ts=time.time(),
            msg=f"API normal - mode={os.getenv('MODEL_MODE','balanced')}",
            pf_status=PF_STATUS,
            line_mode=("real" if (LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET and _has_line) else "dummy"),
        ), 200

    @app.get("/version")
    def version():
        return jsonify(
            version=os.getenv("RELEASE", "local"),
            commit=os.getenv("GIT_SHA", "unknown"),
            mode=os.getenv("MODEL_MODE","balanced")
        ), 200

    @app.post("/line-webhook")
    def callback():
        signature = request.headers.get('X-Line-Signature', '')
        body = request.get_data(as_text=True)
        try:
            handler.handle(body, signature)
        except Exception as e:
            log.warning("LINE webhook error: %s", e)
            return "bad request", 400
        return "ok", 200

# —— LINE 事件處理 —— #
def _handle_message_core(event):
    user_id = getattr(getattr(event, "source", None), "user_id", None)
    text = getattr(getattr(event, "message", None), "text", "")
    if user_id is None: user_id = "dummy-user"
    text = (text or "").strip()

    # 啟動試用
    _start_trial(user_id)

    # 開通
    if text.startswith("開通"):
        pwd = text[2:].strip()
        reply = "✅ 已開通成功！" if pwd == OPENCODE else "❌ 開通碼錯誤，請重新輸入。"
        if pwd == OPENCODE: _set_opened(user_id)
        _reply(event.reply_token, reply, quick=settings_quickreply(SESS.setdefault(user_id, {})))
        return

    # 試用檢查
    if not _is_trial_valid(user_id):
        _reply(event.reply_token, f"⛔ 試用期已到\n📬 請聯繫管理員開通登入帳號\n👉 官方 LINE：{ADMIN_LINE}")
        return

    sess = SESS.setdefault(user_id, {"bankroll": 0})
    sess["user_id"] = user_id

    # 快速功能
    if text in ("設定","⋯","menu","Menu"):
        _reply(event.reply_token, "⚙️ 設定選單：", quick=settings_quickreply(sess)); return
    if text == "查看統計":
        _reply(event.reply_token, _format_stats(sess), quick=settings_quickreply(sess)); return
    if text == "試用剩餘":
        _reply(event.reply_token, "⏳ 試用剩餘：{}".format(_left_trial_sec(user_id)), quick=settings_quickreply(sess)); return
    if text == "重設":
        SESS[user_id] = {"bankroll": 0, "user_id": user_id}
        _reply(event.reply_token, "✅ 已重設流程，請選擇館別：", quick=halls_quickreply()); return

    # 館別 -> 桌號 -> 本金
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
            _reply(event.reply_token, f"👍 已設定本金：{sess['bankroll']:,}\n請輸入上一局點數開始分析", quick=settings_quickreply(sess))
        else:
            _reply(event.reply_token, "請輸入正確格式的本金（例：5000）", quick=settings_quickreply(sess))
        return

    # 連續模式：回報上一局 → 輸出下一局建議
    try:
        if text.strip() == "和":
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

# 綁定 handler（真實 LINE）
if 'LINE_MODE' in globals() and LINE_MODE == "real":
    @handler.add(MessageEvent, message=TextMessage)
    def handle_message(event):
        _handle_message_core(event)

# ----------------- Run -----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting BGS on port %s (LINE_MODE=%s, MODE=%s)", port,
             "real" if (LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET and _has_line) else "dummy",
             os.getenv("MODEL_MODE","balanced"))
    if hasattr(app, "run"):
        app.run(host="0.0.0.0", port=port, debug=False)
