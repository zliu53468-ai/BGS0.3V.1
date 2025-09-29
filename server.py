# -*- coding: utf-8 -*-
"""
server.py — BGS百家樂AI 平衡預測（修正『只押莊』偏置 + 數值健壯化版）
特點：
1) OutcomePF 粒子濾波（若載入失敗自動靜態機率）
2) EV 決策含抽水、觀望門檻、近似打平觀望
3) 對『莊』加入可調反向偏置，避免長期只押莊
4) 30 分鐘試用到期提示 + LINE 快速按鈕
"""

import os, sys, re, time, json, logging
from typing import Dict, Any
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
        return "✅ BGS PF Server OK", 200
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

# ---------- Tunables ----------
os.environ.setdefault("BANKER_COMMISSION", "0.05")

# 進出場／觀望參數
os.environ.setdefault("EDGE_ENTER_EV", "0.0015")   # EV 未達此值 → 觀望
os.environ.setdefault("ENTER_GAP_MIN", "0.018")    # 三類機率Top2差距太小 → 觀望
os.environ.setdefault("TIE_PROB_MAX", "0.28")      # Tie 太高且優勢不明 → 觀望
os.environ.setdefault("STATS_DISPLAY", "smart")

# 新增：公平點修正與打平觀望（可用 .env 覆寫）
os.environ.setdefault("EV_BIAS_BANKER", "0.003")   # 對莊的反向偏置（越大越不容易選莊）
os.environ.setdefault("NEAR_DIFF_HOLD", "0.0025")  # |EV_B - EV_P| 小於此值 → 直接觀望

# 配注
os.environ.setdefault("MIN_BET_PCT_BASE", "0.02")
os.environ.setdefault("MAX_BET_PCT", "0.35")
os.environ.setdefault("BET_UNIT", "100")

# 粒子濾波器參數（小、快）
os.environ.setdefault("DECKS", "6")
os.environ.setdefault("PF_N", "80")
os.environ.setdefault("PF_RESAMPLE", "0.75")
os.environ.setdefault("PF_PRED_SIMS", "25")

# ----------------- PF Loader -----------------
OutcomePF = None
_pf_import_from = "none"
try:
    # 優先載本地 bgs.pfilter；失敗則載同目錄 pfilter.py
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
    def update_point_history(self, p_pts, b_pts): pass
    def predict(self, **k): return np.array([0.458, 0.446, 0.096], dtype=np.float32)

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
                # 注意：不要傳 backend / dirichlet_eps 等不存在的參數
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
TRIAL_SECONDS = int(os.getenv("TRIAL_SECONDS", "1800"))  # 30分鐘
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
    if not info.get("trial_start"): return True
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

# ----------------- Helpers -----------------
def _real(x: float) -> float:
    """確保是實數且有限"""
    try:
        x = np.real_if_close(x)
        if isinstance(x, np.ndarray): x = x.item()
        x = float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    except Exception:
        x = 0.0
    return x

def _safe_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = np.nan_to_num(v, nan=1e-9, posinf=1e-9, neginf=1e-9)
    v = np.clip(v, 1e-9, None)
    s = float(v.sum())
    if s <= 0 or not np.isfinite(s): v = np.array([0.458,0.446,0.096], dtype=np.float64); s = v.sum()
    v = (v / s).astype(np.float32)
    return v

def calculate_adjusted_confidence(ev_b, ev_p, pB, pP):
    edge = max(ev_b, ev_p)
    diff = abs(pB - pP)
    edge_term = min(1.0, _real(edge) / 0.06) ** 0.9
    prob_term = min(1.0, _real(diff) / 0.30) ** 0.85
    raw = 0.6 * edge_term + 0.4 * prob_term
    return float(max(0.0, min(1.0, raw ** 0.9)))

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
    if total >= 15: return f"🎯 近期勝率：{acc:.1f}%"
    if total >= 5:  return f"🎯 當前勝率：{acc:.1f}% ({hit}/{total})"
    return f"🎯 初始勝率：{acc:.1f}% ({hit}/{total})"

def _format_pts_text(p_pts, b_pts):
    if p_pts == b_pts: return f"上局結果: 和 {p_pts}"
    return f"上局結果: 閒 {p_pts} 莊 {b_pts}"

# ----------------- 核心：處理點數 + 預測 + 決策 -----------------
def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    # 驗證
    if not (p_pts == 0 and b_pts == 0):
        if not (0 <= int(p_pts) <= 9 and 0 <= int(b_pts) <= 9):
            return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    pf = _get_pf_from_sess(sess)

    # 輕度學習（用上一局輸贏＋點數）
    if p_pts == b_pts and not (p_pts == 0 and b_pts == 0):
        try: pf.update_outcome(2)
        except Exception: pass
        real_label = "和"
    elif p_pts == 0 and b_pts == 0:
        try: pf.update_outcome(2)
        except Exception: pass
        real_label = "和"
    else:
        try: pf.update_point_history(p_pts, b_pts)
        except Exception: pass
        outcome = 1 if p_pts > b_pts else 0
        real_label = "閒" if p_pts > b_pts else "莊"
        try: pf.update_outcome(outcome)
        except Exception: pass

    # 把上一輪 pending 建議對帳
    st = sess.setdefault("stats", {"bets":0,"wins":0,"push":0,"sum_edge":0.0,"payout":0})
    if "pending_pred" in sess:
        prev_pred = sess.pop("pending_pred")
        prev_watch = bool(sess.pop("pending_watch", False))
        prev_edge = float(sess.pop("pending_edge_ev", 0.0))
        prev_bet_amt = int(sess.pop("pending_bet_amt", 0))
        prev_ev_choice = sess.pop("pending_ev_choice", None)

        sess.setdefault("hist_pred", []).append("觀望" if prev_watch else (prev_ev_choice or prev_pred))
        sess.setdefault("hist_real", []).append(real_label)
        sess["hist_pred"] = sess["hist_pred"][-150:]
        sess["hist_real"] = sess["hist_real"][-150:]

        if not prev_watch and real_label in ("莊","閒"):
            st["bets"] += 1; st["sum_edge"] += float(prev_edge)
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

    # 預測下一局
    sims_pred = int(os.getenv("PF_PRED_SIMS","25"))
    try:
        p_raw = pf.predict(sims_per_particle=sims_pred)
        p_final = _safe_norm(p_raw)
    except Exception as e:
        log.warning("predict fallback due to %s", e)
        p_final = np.array([0.458, 0.446, 0.096], dtype=np.float32)

    # 輕度平滑（避免忽上忽下）
    alpha = 0.7
    prev_sma = sess.get("prob_sma")
    if prev_sma is None:
        sess["prob_sma"] = p_final
    else:
        sess["prob_sma"] = alpha * p_final + (1 - alpha) * prev_sma
    p_final = _safe_norm(sess["prob_sma"])

    # EV（含抽水、Tie退回）
    pB, pP, pT = map(_real, (p_final[0], p_final[1], p_final[2]))
    BCOMM = float(os.getenv("BANKER_COMMISSION","0.05"))
    ev_b = _real(pB * (1.0 - BCOMM) - (1.0 - pB - pT))
    ev_p = _real(pP * 1.0            - (1.0 - pP - pT))

    # ---- 修正『只押莊』：公平點偏置 + 打平觀望 ----
    EV_BIAS_BANKER = float(os.getenv("EV_BIAS_BANKER","0.003"))
    NEAR_DIFF_HOLD = float(os.getenv("NEAR_DIFF_HOLD","0.0025"))
    ev_b_eff = _real(ev_b - EV_BIAS_BANKER)
    ev_p_eff = _real(ev_p)

    diff_eff = abs(ev_b_eff - ev_p_eff)
    if diff_eff < NEAR_DIFF_HOLD:
        ev_choice = "觀望"; edge_ev = 0.0
    else:
        ev_choice = "莊" if ev_b_eff > ev_p_eff else "閒"
        edge_ev = max(ev_b, ev_p)

    # 觀望條件（再疊加）
    watch, reasons = False, []
    EDGE_ENTER_EV = float(os.getenv("EDGE_ENTER_EV","0.0015"))
    if ev_choice != "觀望" and edge_ev < EDGE_ENTER_EV:
        watch = True; reasons.append("EV優勢不足")

    TIE_PROB_MAX = float(os.getenv("TIE_PROB_MAX","0.28"))
    if pT > TIE_PROB_MAX and edge_ev < 0.02:
        watch = True; reasons.append("和局風險")

    enter_gap_min = float(os.getenv("ENTER_GAP_MIN","0.018"))
    top2 = sorted([pB, pP, pT], reverse=True)[:2]
    if (top2[0] - top2[1]) < enter_gap_min:
        watch = True; reasons.append("勝率差不足")

    if ev_choice == "觀望": watch = True

    # 配注（與預測邏輯分開）
    bankroll = int(sess.get("bankroll", 0))
    bet_pct = 0.0; bet_amt = 0
    if not watch:
        conf = calculate_adjusted_confidence(ev_b, ev_p, pB, pP)
        base_floor = float(os.getenv("MIN_BET_PCT_BASE", "0.02"))
        base_ceiling = 0.30
        base_pct = base_floor + (base_ceiling - base_floor) * conf
        bet_pct = max(base_floor, min(float(os.getenv("MAX_BET_PCT", "0.35")), base_pct))
        if bankroll > 0 and bet_pct > 0:
            unit = int(os.getenv("BET_UNIT", "100"))
            bet_amt = int(round(bankroll * bet_pct))
            bet_amt = max(0, int(round(bet_amt / unit)) * unit)

    # 存 pending
    sess["pending_pred"] = "觀望" if watch else ev_choice
    sess["pending_watch"] = bool(watch)
    sess["pending_edge_ev"] = float(edge_ev)
    sess["pending_bet_amt"] = int(bet_amt)
    sess["pending_ev_choice"] = None if watch else ev_choice

    sess["last_pts_text"] = _format_pts_text(p_pts, b_pts) if not (p_pts==0 and b_pts==0) else "上局結果: 和"
    stats_display = get_stats_display(sess)
    strat = f"⚠️ 觀望（{'、'.join(reasons)}）" if watch else (
        f"🟡 低信心配注 {bet_pct*100:.1f}%" if bet_pct < 0.15 else
        f"🟠 中信心配注 {bet_pct*100:.1f}%" if bet_pct < 0.25 else
        f"🟢 高信心配注 {bet_pct*100:.1f}%"
    )

    # 輸出訊息
    msg = [
        sess["last_pts_text"],
        "開始平衡分析下局....",
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

def _format_stats(sess):
    st = sess.get("stats", {"bets":0,"wins":0,"push":0,"sum_edge":0.0,"payout":0})
    bets, wins, push, payout = st["bets"], st["wins"], st["push"], st["payout"]
    acc = (wins / bets * 100.0) if bets>0 else 0.0
    return f"📈 累計：下注 {bets}｜命中 {wins}（{acc:.1f}%）｜和 {push}｜盈虧 {payout}"

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

def welcome_text(uid):
    left = _left_trial_sec(uid)
    return (
        "👋 歡迎使用 BGS AI 平衡預測系統！\n"
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
            msg="API normal - Balanced Mode",
            pf_status=PF_STATUS,
            line_mode=("real" if (LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET and _has_line) else "dummy"),
        ), 200

    @app.get("/version")
    def version():
        return jsonify(
            version=os.getenv("RELEASE", "local"),
            commit=os.getenv("GIT_SHA", "unknown"),
            mode="balanced"
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

    _start_trial(user_id)

    if text.startswith("開通"):
        pwd = text[2:].strip()
        reply = "✅ 已開通成功！" if pwd == OPENCODE else "❌ 開通碼錯誤，請重新輸入。"
        if pwd == OPENCODE: _set_opened(user_id)
        _reply(event.reply_token, reply, quick=settings_quickreply(SESS.setdefault(user_id, {})))
        return

    if not _is_trial_valid(user_id):
        _reply(event.reply_token, "⛔ 試用期已到\n📬 請聯繫管理員開通登入帳號\n👉 加入官方 LINE：{}".format(ADMIN_LINE))
        return

    sess = SESS.setdefault(user_id, {"bankroll": 0})
    sess["user_id"] = user_id

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
            _reply(event.reply_token, f"👍 已設定本金：{sess['bankroll']:,}\n請輸入上一局點數開始平衡預測", quick=settings_quickreply(sess))
        else:
            _reply(event.reply_token, "請輸入正確格式的本金（例：5000）", quick=settings_quickreply(sess))
        return

    # 連續模式
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

if 'LINE_MODE' in globals() and LINE_MODE == "real":
    @handler.add(MessageEvent, message=TextMessage)
    def handle_message(event):
        _handle_message_core(event)

# ----------------- Run -----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting BGS-Balanced on port %s (LINE_MODE=%s)", port,
             "real" if (LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET and _has_line) else "dummy")
    if hasattr(app, "run"):
        app.run(host="0.0.0.0", port=port, debug=False)
