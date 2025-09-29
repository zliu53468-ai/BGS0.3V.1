# -*- coding: utf-8 -*-
"""
server.py — BGS 百家樂 AI（可一鍵覆蓋版本）
特色：
1) UI 流程：選館別→桌號→本金→連續輸入點數（65/和/閒6莊5/莊5閒6）
2) 試用 30 分鐘到期，推送官方 LINE 卡片（含連結）
3) Outcome 粒子濾波器（匯入失敗自動 Dummy）做下一局機率
4) 預測邏輯 與 配注信心度 完全分離
5) Flask + LINE Webhook（未設定憑證時自動 Dummy LINE）

環境變數（可選）：
- PORT=8000
- TRIAL_MINUTES=30
- OPENCODE=aaa8881688
- ADMIN_LINE=https://lin.ee/Dlm6Y3u
- BANKER_COMMISSION=0.05
- PF_N=120 PF_RESAMPLE=0.65 PF_PRED_SIMS=80
- EDGE_ENTER_EV=0.0015 ENTER_GAP_MIN=0.018 TIE_PROB_MAX=0.28
- MIN_BET_PCT_BASE=0.02 MAX_BET_PCT=0.35 BET_UNIT=100
- LINE_CHANNEL_ACCESS_TOKEN / LINE_CHANNEL_SECRET
"""

import os, re, time, json, logging
from typing import Dict, Any
import numpy as np

# ----------------- Logging -----------------
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

# ----------------- LINE SDK（可選） -----------------
_has_line = True
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import (
        MessageEvent, TextMessage, TextSendMessage,
        QuickReply, QuickReplyButton, MessageAction, FlexSendMessage
    )
except Exception as e:
    _has_line = False
    WebhookHandler = LineBotApi = None
    MessageEvent = TextMessage = TextSendMessage = QuickReply = QuickReplyButton = MessageAction = FlexSendMessage = object
    log.warning("LINE SDK not available, Dummy LINE mode: %s", e)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_TIMEOUT = float(os.getenv("LINE_TIMEOUT", "2.0"))

if _has_line and LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET:
    try:
        line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN, timeout=LINE_TIMEOUT)
        handler = WebhookHandler(LINE_CHANNEL_SECRET)
        LINE_MODE = "real"
    except Exception as e:
        log.warning("LINE init failed -> Dummy: %s", e)
        LINE_MODE = "dummy"
else:
    LINE_MODE = "dummy"

if LINE_MODE == "dummy":
    class _DummyHandler:
        def add(self, *a, **k):
            def deco(f): return f
            return deco
        def handle(self, body, signature):
            log.info("[DummyLINE] handle called")
    class _DummyAPI:
        def reply_message(self, token, message):
            try:
                txt = message.text if hasattr(message, "text") else str(message)
            except Exception:
                txt = str(message)
            log.info("[DummyLINE] reply: %s", txt)
    handler = _DummyHandler()
    line_bot_api = _DummyAPI()

# ----------------- 參數 -----------------
SESS: Dict[str, Dict[str, Any]] = {}
BANKER_COMMISSION = float(os.getenv("BANKER_COMMISSION", "0.05"))
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
OPENCODE = os.getenv("OPENCODE", "aaa8881688")
ADMIN_LINE = os.getenv("ADMIN_LINE", "https://lin.ee/Dlm6Y3u")

EDGE_ENTER_EV = float(os.getenv("EDGE_ENTER_EV", "0.0015"))
ENTER_GAP_MIN = float(os.getenv("ENTER_GAP_MIN", "0.018"))
TIE_PROB_MAX  = float(os.getenv("TIE_PROB_MAX",  "0.28"))

MIN_BET_PCT_BASE = float(os.getenv("MIN_BET_PCT_BASE", "0.02"))
MAX_BET_PCT      = float(os.getenv("MAX_BET_PCT", "0.35"))
BET_UNIT         = int(os.getenv("BET_UNIT", "100"))

# ----------------- 粒子濾波器載入（可選） -----------------
OutcomePF = None
_pf_from = "none"
try:
    from bgs.pfilter import OutcomePF
    _pf_from = "bgs"
except Exception:
    try:
        from pfilter import OutcomePF
        _pf_from = "local"
    except Exception:
        OutcomePF = None
        _pf_from = "none"

PF_STATUS = {"ready": OutcomePF is not None, "from": _pf_from, "error": None}

class _DummyPF:
    def update_outcome(self, o): pass
    def update_point_history(self, p, b): pass
    def predict(self, **k):
        # 安全保守機率（含和）
        return np.array([0.458, 0.446, 0.096], dtype=np.float32)

def _get_pf(sess: Dict[str, Any]):
    if OutcomePF is None:
        sess["_pf_dummy"] = True
        return _DummyPF()
    if "pf" not in sess:
        try:
            sess["pf"] = OutcomePF(
                decks=int(os.getenv("DECKS", "6")),
                seed=int(os.getenv("SEED", "42")) + int(time.time()%1000),
                n_particles=int(os.getenv("PF_N", "120")),
                sims_lik=max(1, int(os.getenv("PF_UPD_SIMS", "40"))),
                resample_thr=float(os.getenv("PF_RESAMPLE","0.65")),
            )
            sess.pop("_pf_dummy", None)
        except Exception as e:
            PF_STATUS.update({"ready": False, "error": str(e)})
            sess["_pf_dummy"] = True
            return _DummyPF()
    return sess["pf"]

# ----------------- 工具 -----------------
def _now(): return int(time.time())

def _qr_btn(label, text):
    if LINE_MODE == "real":
        return QuickReplyButton(action=MessageAction(label=label, text=text))
    return {"label": label, "text": text}

def _reply(token, text, quick=None, flex=None):
    try:
        if LINE_MODE == "real":
            msgs = []
            if flex is not None:
                msgs.append(FlexSendMessage(alt_text="通知", contents=flex))
            msgs.append(TextSendMessage(text=text, quick_reply=QuickReply(items=quick) if quick else None))
            line_bot_api.reply_message(token, msgs if len(msgs)>1 else msgs[0])
        else:
            if flex is not None:
                log.info("[DummyLINE] flex sent: %s", json.dumps(flex)[:200])
            log.info("[DummyLINE] reply%s: %s", " (with quick)" if quick else "", text)
    except Exception as e:
        log.warning("LINE reply error: %s", e)

def halls_quickreply():
    return [_qr_btn(f"{i}", f"{i}") for i in range(1, 11)]

def settings_quickreply(sess):
    return [
        _qr_btn("選館別", "設定 館別"),
        _qr_btn("查看統計", "查看統計"),
        _qr_btn("試用剩餘", "試用剩餘"),
        _qr_btn("重設流程", "重設"),
    ]

def welcome_text(uid):
    left = left_trial_text(uid)
    return (
        "👋 歡迎使用 BGS AI 預測分析！\n"
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

# ----------------- 試用制 -----------------
def ensure_user(uid):
    sess = SESS.setdefault(uid, {"bankroll":0})
    if "trial_start" not in sess:
        sess["trial_start"] = _now()
    return sess

def is_trial_valid(uid) -> bool:
    s = SESS.get(uid, {})
    if s.get("is_opened"): return True
    start = int(s.get("trial_start", _now()))
    return (_now() - start) < TRIAL_MINUTES*60

def left_trial_text(uid) -> str:
    s = SESS.get(uid, {})
    if s.get("is_opened"): return "永久"
    start = int(s.get("trial_start", _now()))
    left = TRIAL_MINUTES*60 - (_now() - start)
    if left <= 0: return "已到期"
    return f"{left//60} 分 {left%60} 秒"

def push_trial_over_card(reply_token):
    # Flex Bubble 卡片（含官方 LINE 連結與圖）
    flex = {
      "type":"bubble",
      "hero":{
        "type":"image",
        "url":"https://i.imgur.com/7I0uU5k.png",  # 任意促圖，可換你的圖
        "size":"full","aspectRatio":"20:13","aspectMode":"cover"
      },
      "body":{
        "type":"box","layout":"vertical","contents":[
          {"type":"text","text":"試用期已到","weight":"bold","size":"lg","color":"#D32F2F"},
          {"type":"text","text":"請聯繫管理員開通登入帳號","wrap":True,"margin":"md"},
          {"type":"text","text":"加入官方 LINE：", "margin":"md"},
          {"type":"button","style":"link","action":{"type":"uri","label":"@ 官方 LINE","uri":ADMIN_LINE}},
        ]
      }
    }
    _reply(reply_token, "⛔ 試用期已到\n📬 請聯繫管理員開通登入帳號\n👉 加入官方 LINE：{}".format(ADMIN_LINE), flex=flex)

# ----------------- 預測與配注（分離） -----------------
def _safe_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, 1e-9, None); s = v.sum()
    if not np.isfinite(s) or s <= 0: return np.array([0.458,0.446,0.096], dtype=np.float32)
    return (v/s).astype(np.float32)

def predict_probs(sess) -> np.ndarray:
    pf = _get_pf(sess)
    try:
        p = pf.predict(sims_per_particle=int(os.getenv("PF_PRED_SIMS","80")))
        return _safe_norm(p)
    except Exception as e:
        log.warning("predict fallback: %s", e)
        return np.array([0.458,0.446,0.096], dtype=np.float32)

def decide_direction(p: np.ndarray) -> Dict[str, Any]:
    """只決定『觀望 or 入場、莊/閒方向』，不含配注"""
    pB, pP, pT = float(p[0]), float(p[1]), float(p[2])
    ev_b = pB*(1.0-BANKER_COMMISSION) - (1.0 - pB - pT)
    ev_p = pP*(1.0) - (1.0 - pP - pT)
    edge_ev = max(ev_b, ev_p)
    choice = "莊" if ev_b > ev_p else "閒"
    # 微調：若非常接近，用較高機率方
    if abs(ev_b-ev_p) < 0.004:
        choice = "莊" if pB>pP else "閒"
    # 觀望條件
    watch_reasons = []
    if edge_ev < EDGE_ENTER_EV: watch_reasons.append("EV 優勢不足")
    if pT > TIE_PROB_MAX and edge_ev < 0.02: watch_reasons.append("和局風險")
    gap_top2 = sorted([pB,pP,pT], reverse=True)[:2]
    if (gap_top2[0]-gap_top2[1]) < ENTER_GAP_MIN: watch_reasons.append("勝率差不足")
    watch = len(watch_reasons)>0
    return {
        "watch": watch,
        "choice": choice if not watch else "觀望",
        "edge_ev": float(edge_ev),
        "reasons": watch_reasons
    }

def confidence_for_betting(p: np.ndarray, decision_choice: str, edge_ev: float) -> float:
    """
    完全獨立的配注信心度：不影響是否入場與方向。
    綜合：EV 強度 + 莊閒機率差。回傳 0~1，再映射到下注比例。
    """
    pB, pP, pT = float(p[0]), float(p[1]), float(p[2])
    diff = abs(pB - pP)
    # 邊際歸一（0~0.06 取 0~1）
    edge_term = min(1.0, max(0.0, edge_ev) / 0.06) ** 0.9
    prob_term = min(1.0, diff / 0.30) ** 0.85
    conf = 0.6*edge_term + 0.4*prob_term
    return float(max(0.0, min(1.0, conf)))

def bet_pct_from_conf(conf: float) -> float:
    """把信心度轉成下注比例（不超過 MAX_BET_PCT）"""
    base_floor = MIN_BET_PCT_BASE
    base_ceiling = min(MAX_BET_PCT, 0.30)
    pct = base_floor + (base_ceiling - base_floor)*conf
    return float(max(base_floor, min(MAX_BET_PCT, pct)))

def bet_amount(bankroll:int, pct:float) -> int:
    if bankroll <= 0 or pct <= 0: return 0
    amt = int(round(bankroll * pct))
    if BET_UNIT > 0:
        amt = int(round(amt / BET_UNIT)) * BET_UNIT
    return max(0, amt)

# ----------------- 文案 -----------------
def _format_pts_text(p_pts, b_pts):
    if p_pts==b_pts:
        return f"上局結果: 和 {p_pts}"
    return f"上局結果: 閒 {p_pts} 莊 {b_pts}"

def stats_line(sess):
    st = sess.get("stats", {"bets":0,"wins":0,"push":0,"payout":0})
    bets, wins, push, payout = st["bets"], st["wins"], st["push"], st["payout"]
    acc = (wins/bets*100.0) if bets>0 else 0.0
    return f"📈 累計：下注 {bets}｜命中 {wins}（{acc:.1f}%）｜和 {push}｜盈虧 {payout}"

# ----------------- 主邏輯：處理上一局 + 給下一局建議 -----------------
def handle_points_and_predict(sess: Dict[str,Any], p_pts:int, b_pts:int) -> str:
    # 1) 更新上一局
    pf = _get_pf(sess)
    if p_pts==0 and b_pts==0:
        try: pf.update_outcome(2)
        except Exception: pass
        real_label = "和"
    else:
        try: pf.update_point_history(p_pts, b_pts)
        except Exception: pass
        out = 1 if p_pts>b_pts else 0
        real_label = "閒" if out==1 else "莊"
        try: pf.update_outcome(out)
        except Exception: pass

    # 對齊 pending 建議算戰績
    st = sess.setdefault("stats", {"bets":0,"wins":0,"push":0,"payout":0})
    if "pending_pred" in sess:
        prev_watch = bool(sess.pop("pending_watch", False))
        prev_ev_choice = sess.pop("pending_ev_choice", None)
        prev_bet_amt = int(sess.pop("pending_bet_amt", 0))
        if not prev_watch and real_label in ("莊","閒"):
            st["bets"] += 1
            if prev_ev_choice == real_label:
                if prev_ev_choice == "莊":
                    st["payout"] += int(round(prev_bet_amt*(1.0-BANKER_COMMISSION)))
                else:
                    st["payout"] += int(prev_bet_amt)
                st["wins"] += 1
            else:
                st["payout"] -= int(prev_bet_amt)
        elif real_label == "和":
            st["push"] += 1

    # 2) 產生下一局預測（純預測）
    p = predict_probs(sess)
    decision = decide_direction(p)
    watch, ev_choice, edge_ev = decision["watch"], decision["choice"], decision["edge_ev"]

    # 3) 配注（完全獨立）
    bankroll = int(sess.get("bankroll", 0))
    conf = confidence_for_betting(p, ev_choice, edge_ev)
    bet_pct = 0.0 if watch else bet_pct_from_conf(conf)
    amt = bet_amount(bankroll, bet_pct)

    # 存 pending（供下局結算）
    sess["pending_pred"] = "觀望" if watch else ev_choice
    sess["pending_watch"] = bool(watch)
    sess["pending_ev_choice"] = ev_choice
    sess["pending_bet_amt"] = int(amt)

    # 顯示
    sess["last_pts_text"] = _format_pts_text(p_pts, b_pts) if not (p_pts==0 and b_pts==0) else "上局結果: 和"
    strat = f"⚠️ 觀望（{'、'.join(decision['reasons'])}）" if watch else (
        f"🟡 低信心配注 {bet_pct*100:.1f}%" if conf < 0.5 else
        f"🟠 中信心配注 {bet_pct*100:.1f}%" if conf < 0.75 else
        f"🟢 高信心配注 {bet_pct*100:.1f}%"
    )
    msg = [
        sess["last_pts_text"],
        "開始分析下局....",
        "",
        "【預測結果】",
        f"閒：{p[1]*100:.2f}%",
        f"莊：{p[0]*100:.2f}%",
        f"和：{p[2]*100:.2f}%",
        f"本次預測：{'觀望' if watch else ev_choice} (EV優勢: {edge_ev*100:.2f}%)",
        f"建議下注金額：{amt:,}",
        f"配注策略：{strat}",
    ]
    if sess.get("_pf_dummy"):
        msg.append("⚠️ 預測引擎載入失敗，僅提供靜態機率")
    msg.extend([
        "—",
        "🔁 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）",
        "⚙️ 輸入「設定」可開啟功能按鈕"
    ])
    return "\n".join(msg)

# ----------------- LINE Event -----------------
def _handle_message_core(event):
    user_id = getattr(getattr(event, "source", None), "user_id", None) or "dummy-user"
    text = (getattr(getattr(event, "message", None), "text", "") or "").strip()

    sess = ensure_user(user_id)

    # 開通
    if text.startswith("開通"):
        pwd = text[2:].strip()
        if pwd == OPENCODE:
            sess["is_opened"] = True
            _reply(event.reply_token, "✅ 已開通成功！", quick=settings_quickreply(sess))
        else:
            _reply(event.reply_token, "❌ 開通碼錯誤，請重新輸入。", quick=settings_quickreply(sess))
        return

    # 試用守門
    if not is_trial_valid(user_id):
        push_trial_over_card(event.reply_token)
        return

    # 設定選單
    if text in ("設定","⋯","menu","Menu"):
        _reply(event.reply_token, "⚙️ 設定選單：", quick=settings_quickreply(sess)); return
    if text == "查看統計":
        _reply(event.reply_token, stats_line(sess), quick=settings_quickreply(sess)); return
    if text == "試用剩餘":
        _reply(event.reply_token, f"⏳ 試用剩餘：{left_trial_text(user_id)}", quick=settings_quickreply(sess)); return
    if text == "重設":
        SESS[user_id] = {"bankroll":0,"trial_start":_now()}
        _reply(event.reply_token, "✅ 已重設流程，請選擇館別：", quick=halls_quickreply()); return

    # 館別 -> 桌號 -> 本金
    if not sess.get("hall_id"):
        if text.isdigit() and 1 <= int(text) <= 10:
            sess["hall_id"] = int(text)
            hall_map = ["WM","PM","DG","SA","KU","歐博/卡利","KG","金利","名人","MT真人"]
            _reply(event.reply_token, f"✅ 已選 [{hall_map[int(text)-1]}]\n請輸入桌號（例：DG01，格式：2字母+2數字）", quick=settings_quickreply(sess))
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
            _reply(event.reply_token, f"👍 已設定本金：{sess['bankroll']:,}\n請輸入上一局點數開始分析（例：65 / 和 / 閒6莊5）", quick=settings_quickreply(sess))
        else:
            _reply(event.reply_token, "請輸入正確格式的本金（例：5000）", quick=settings_quickreply(sess))
        return

    # 連續模式：解析點數
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

# 綁定 LINE（若為真連線）
if LINE_MODE == "real":
    @handler.add(MessageEvent, message=TextMessage)
    def handle_message(event):
        _handle_message_core(event)

# ----------------- HTTP Routes -----------------
if _has_flask:
    @app.get("/")
    def root():
        return "✅ BGS PF Server OK", 200

    @app.get("/health")
    def health():
        return jsonify(
            ok=True,
            ts=time.time(),
            pf_status=PF_STATUS,
            line_mode=LINE_MODE,
            trial_minutes=TRIAL_MINUTES
        ), 200

    @app.get("/version")
    def version():
        return jsonify(version=os.getenv("RELEASE","local"), commit=os.getenv("GIT_SHA","unknown")), 200

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

# ----------------- Run -----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting BGS on port %s (LINE_MODE=%s)", port, LINE_MODE)
    if hasattr(app, "run"):
        app.run(host="0.0.0.0", port=port, debug=False)
