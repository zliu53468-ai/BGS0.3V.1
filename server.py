# -*- coding: utf-8 -*-
"""
server.py — BGS百家樂AI（Balanced/Independent 單檔切換）
重點：
1) MODEL_MODE=balanced / independent 兩種邏輯皆內建（.env 切換）
2) 修正「只會押莊」：加入抽水公平點判斷與近似EV觀望（B_BREAKEVEN_MULT、NEAR_EV）
3) 預測邏輯 與 配注信心 度（bet sizing）分離
4) 內建 30 分鐘試用與到期卡片（ADMIN_LINE, ADMIN_CONTACT, OPENCODE）
5) 可選 Redis 保存 Session（REDIS_URL）
"""

import os, sys, re, time, json, logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

# ------------ 基本設定 ------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

def _env_flag(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None: return bool(default)
    v = str(v).strip().lower()
    return v in ("1","true","t","yes","y","on")

# ------------ Flask ------------
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

# ------------ Redis（可選）------------
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

# ------------ 全域 Session ------------
SESS: Dict[str, Dict[str, Any]] = {}
def _now() -> int: return int(time.time())

def _get_user_info(uid: str) -> Dict[str, Any]:
    k = f"bgsu:{uid}"
    if rcli:
        try:
            s = rcli.get(k)
            if s: return json.loads(s)
        except Exception: pass
    return SESS.get(uid, {})

def _set_user_info(uid: str, info: Dict[str, Any]) -> None:
    k = f"bgsu:{uid}"
    if rcli:
        try: rcli.set(k, json.dumps(info), ex=86400)
        except Exception: pass
    SESS[uid] = info

# ------------ 試用 / 開通 ------------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
OPENCODE = os.getenv("OPENCODE", "aaa8881688")
ADMIN_LINE = os.getenv("ADMIN_LINE", "https://lin.ee/Dlm6Y3u")
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@jins888")

def _start_trial(uid: str):
    info = _get_user_info(uid)
    if not info.get("trial_start"):
        info["trial_start"] = _now()
        _set_user_info(uid, info)

def _is_trial_valid(uid: str) -> bool:
    info = _get_user_info(uid)
    if info.get("is_opened"): return True
    start = info.get("trial_start")
    if not start: return True
    return (_now() - int(start)) < TRIAL_MINUTES * 60

def _set_opened(uid: str):
    info = _get_user_info(uid)
    info["is_opened"] = True
    _set_user_info(uid, info)

def _trial_left_text(uid: str) -> str:
    info = _get_user_info(uid)
    if info.get("is_opened"): return "永久"
    start = info.get("trial_start")
    if not start: return "尚未啟動"
    left = TRIAL_MINUTES*60 - (_now() - int(start))
    return f"{max(0,left)//60} 分 {max(0,left)%60} 秒" if left > 0 else "已到期"

# ------------ PF 載入 ------------
BANKER_COMMISSION = float(os.getenv("BANKER_COMMISSION","0.05"))
MODEL_MODE = os.getenv("MODEL_MODE","balanced").strip().lower()

OutcomePF = None
_pf_import_from = "none"
try:
    from bgs.pfilter import OutcomePF
    _pf_import_from = "bgs"
except Exception:
    try:
        cur = os.path.dirname(os.path.abspath(__file__))
        if cur not in sys.path: sys.path.insert(0, cur)
        from pfilter import OutcomePF  # 本地同目錄 pfilter.py
        _pf_import_from = "local"
    except Exception:
        OutcomePF = None
PF_STATUS = {"ready": OutcomePF is not None, "from": _pf_import_from}

class _DummyPF:
    def update_outcome(self, outcome: int): pass
    def update_point_history(self, p_pts: int, b_pts: int): pass
    def predict(self, **k): return np.array([0.458, 0.446, 0.096], dtype=np.float32)

def _get_pf_from_sess(sess: Dict[str, Any]):
    if not OutcomePF:
        sess["_pf_dummy"] = True
        return _DummyPF()
    if sess.get("pf") is None and not sess.get("_pf_failed"):
        try:
            sess["pf"] = OutcomePF(
                decks=int(os.getenv("DECKS","6")),
                seed=int(os.getenv("SEED","42")) + int(time.time()%1000),
                n_particles=int(os.getenv("PF_N","80")),
                sims_lik=max(1, int(os.getenv("PF_UPD_SIMS","25"))),
                resample_thr=float(os.getenv("PF_RESAMPLE","0.75")),
                backend=os.getenv("PF_BACKEND","mc"),
                dirichlet_eps=float(os.getenv("PF_DIR_EPS","0.01")),
                stability_factor=float(os.getenv("PF_STAB_FACTOR","0.85")),
            )
            sess.pop("_pf_dummy", None)
        except Exception as e:
            sess["_pf_failed"] = True
            sess["_pf_dummy"] = True
            sess["_pf_error_msg"] = str(e)
            log.exception("OutcomePF init fail; use dummy")
    return sess.get("pf") or _DummyPF()

# ------------ 獨立預測器（independent）------------
class IndependentPredictor:
    def __init__(self): self.last: Optional[Tuple[int,int]] = None
    def update_points(self, p_pts:int, b_pts:int): self.last = (p_pts, b_pts)
    def predict(self) -> np.ndarray:
        if not self.last: return np.array([0.458,0.446,0.096], dtype=np.float32)
        p,b = self.last; diff = abs(p-b); total = p+b
        if diff >= 6:  # 延續
            return np.array([0.57,0.38,0.05], dtype=np.float32) if b>p else np.array([0.38,0.57,0.05], dtype=np.float32)
        if diff >= 4:
            return np.array([0.53,0.42,0.05], dtype=np.float32) if b>p else np.array([0.42,0.53,0.05], dtype=np.float32)
        if diff <= 1:
            return np.array([0.40,0.40,0.20], dtype=np.float32) if total<=6 else np.array([0.45,0.45,0.10], dtype=np.float32)
        return np.array([0.48,0.47,0.05], dtype=np.float32)

def _get_predictor_from_sess(sess: Dict[str, Any]) -> IndependentPredictor:
    if "predictor" not in sess: sess["predictor"] = IndependentPredictor()
    return sess["predictor"]

# ------------ 計算/顯示 ------------
def _safe_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, 1e-9, None); s = float(v.sum()); 
    return (v/s).astype(np.float32) if s>0 else np.array([0.458,0.446,0.096], dtype=np.float32)

def _format_pts_text(p_pts, b_pts):
    return "上局結果: 和" if p_pts==b_pts else f"上局結果: 閒 {p_pts} 莊 {b_pts}"

def calculate_adjusted_confidence(ev_b, ev_p, pB, pP, choice):
    # 與預測分離：只根據EV+差距算信心（決策已在外部確定）
    edge = max(ev_b, ev_p); diff = abs(pB - pP)
    edge_term = min(1.0, edge / 0.06) ** 0.9
    prob_term = min(1.0, diff / 0.30) ** 0.85
    raw = 0.6 * edge_term + 0.4 * prob_term
    return float(max(0.0, min(1.0, raw ** 0.9)))

# ------------ 決策（修正只押莊）------------
# 抽水公平點：在 5% 抽水下，莊至少要比閒多 ~2.56% 才值得
B_BREAKEVEN_MULT = float(os.getenv("B_BREAKEVEN_MULT", "1.0256"))  # pB >= 1.0256*pP
NEAR_EV = float(os.getenv("NEAR_EV", "0.004"))                      # |EV差| < 0.004 視為接近

def decide_side_with_rake(pB: float, pP: float, pT: float) -> Tuple[str, bool, list]:
    # EV（和退回）
    ev_b = pB*(1.0-BANKER_COMMISSION) - (1.0 - pB - pT)
    ev_p = pP*1.0                     - (1.0 - pP - pT)
    choice = "莊" if ev_b > ev_p else "閒"
    edge_ev = max(ev_b, ev_p)
    reasons = []
    watch = False

    # 接近EV → 用抽水公平點判斷；皆不足則觀望
    if abs(ev_b - ev_p) < NEAR_EV:
        if pB >= B_BREAKEVEN_MULT * pP:
            choice = "莊"; reasons.append("接近EV但莊超過公平點")
        elif pP >= (1.0/B_BREAKEVEN_MULT) * pB:
            choice = "閒"; reasons.append("接近EV但閒更優")
        else:
            watch = True; reasons.append("優勢不足（接近公平點）")
            edge_ev = 0.0

    return choice, watch, reasons, ev_b, ev_p, edge_ev

# ------------ 門檻與配注 ------------
EDGE_ENTER_EV = float(os.getenv("EDGE_ENTER_EV","0.0015" if MODEL_MODE=="balanced" else "0.001"))
ENTER_GAP_MIN = float(os.getenv("ENTER_GAP_MIN","0.018" if MODEL_MODE=="balanced" else "0.015"))
TIE_PROB_MAX = float(os.getenv("TIE_PROB_MAX","0.28" if MODEL_MODE=="balanced" else "0.30"))
MIN_BET_PCT_BASE = float(os.getenv("MIN_BET_PCT_BASE","0.02"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT","0.35"))
BET_UNIT = int(os.getenv("BET_UNIT","100"))

def compute_bet(bankroll:int, ev_b:float, ev_p:float, pB:float, pP:float, choice:str) -> Tuple[float, int, str]:
    conf = calculate_adjusted_confidence(ev_b, ev_p, pB, pP, choice)
    base_floor, base_ceiling = MIN_BET_PCT_BASE, 0.30
    base_pct = base_floor + (base_ceiling - base_floor) * conf
    bet_pct = max(base_floor, min(MAX_BET_PCT, base_pct))
    bet_amt = int(round(bankroll * bet_pct / BET_UNIT)) * BET_UNIT if bankroll>0 else 0
    strat = ("🟡 低信心配注" if bet_pct<0.15 else "🟠 中信心配注" if bet_pct<0.25 else "🟢 高信心配注") + f" {bet_pct*100:.1f}%"
    return bet_pct, bet_amt, strat

# ------------ 主預測流程（兩模式共用）------------
def handle_points_and_predict(sess: Dict[str,Any], p_pts:int, b_pts:int) -> str:
    # 驗證
    if not (p_pts==0 and b_pts==0):
        if not (0<=p_pts<=9 and 0<=b_pts<=9):
            return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    # 更新 trial
    _start_trial(sess["user_id"])

    # 更新引擎/資料
    if MODEL_MODE == "balanced":
        pf = _get_pf_from_sess(sess)
        if p_pts==b_pts:
            try: pf.update_outcome(2); 
            except Exception: pass
            real_label = "和"
        elif p_pts==0 and b_pts==0:
            try: pf.update_outcome(2)
            except Exception: pass
            real_label = "和"
        else:
            try: pf.update_point_history(p_pts, b_pts)
            except Exception: pass
            try: pf.update_outcome(1 if p_pts>b_pts else 0)
            except Exception: pass
            real_label = "閒" if p_pts>b_pts else "莊"
        # 取得機率
        sims_pred = int(os.getenv("PF_PRED_SIMS","25"))
        try:
            p_raw = pf.predict(sims_per_particle=sims_pred)
            p_final = _safe_norm(p_raw)
        except Exception as e:
            log.warning("PF predict fallback: %s", e)
            p_final = np.array([0.458,0.446,0.096], dtype=np.float32)

        # 輕度平滑（不影響決策公平點）
        alpha = 0.7
        prev = sess.get("prob_sma")
        sess["prob_sma"] = p_final if prev is None else alpha*p_final + (1-alpha)*prev
        p_final = sess["prob_sma"]

    else:  # independent
        ip = _get_predictor_from_sess(sess)
        if not (p_pts==0 and b_pts==0):
            ip.update_points(p_pts, b_pts)
        p_final = ip.predict()
        real_label = "和" if p_pts==b_pts or (p_pts==0 and b_pts==0) else ("閒" if p_pts>b_pts else "莊")

    # 決策（修正莊偏）
    pB, pP, pT = float(p_final[0]), float(p_final[1]), float(p_final[2])
    choice, watch, reasons, ev_b, ev_p, edge_ev = decide_side_with_rake(pB, pP, pT)

    # 額外觀望條件（勝率差小、和偏高）
    top2 = sorted([pB,pP,pT], reverse=True)[:2]
    if edge_ev < EDGE_ENTER_EV or (top2[0]-top2[1]) < ENTER_GAP_MIN or (pT>TIE_PROB_MAX and edge_ev<0.02):
        watch = True; reasons.append("EV/勝率差不足或和風險")

    # 配注（與預測分離）
    bankroll = int(sess.get("bankroll", 0))
    bet_pct=0.0; bet_amt=0; strat="⚠️ 觀望"
    if not watch:
        bet_pct, bet_amt, strat = compute_bet(bankroll, ev_b, ev_p, pB, pP, choice)

    # pending（統計配對）
    sess["pending_pred"] = "觀望" if watch else choice
    sess["pending_watch"] = bool(watch)
    sess["pending_edge_ev"] = float(edge_ev)
    sess["pending_bet_amt"] = int(bet_amt)
    sess["pending_ev_choice"] = choice
    sess["last_pts_text"] = _format_pts_text(p_pts, b_pts)
    sess.setdefault("stats", {"bets":0,"wins":0,"push":0,"payout":0})

    # 輸出
    msg = [
        sess["last_pts_text"],
        f"開始{'平衡' if MODEL_MODE=='balanced' else '獨立'}分析下局....",
        "",
        "【預測結果】",
        f"閒：{pP*100:.2f}%",
        f"莊：{pB*100:.2f}%",
        f"和：{pT*100:.2f}%",
        f"本次預測：{'觀望' if watch else choice} (EV優勢: {edge_ev*100:.2f}%)",
        f"建議下注金額：{bet_amt:,}",
        f"配注策略：{('⚠️ 觀望（'+'、'.join(reasons)+'）') if watch else strat}",
    ]
    if sess.get("_pf_dummy"):
        warn = sess.get("_pf_error_msg","PF 模組缺失")
        msg.append(f"⚠️ 預測引擎載入失敗，僅提供靜態機率（{warn}）")
    msg.extend([
        "—",
        "🔁 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）",
        "⚙️ 輸入「設定」可開啟功能按鈕"
    ])
    return "\n".join(msg)

def _format_stats(sess):
    st = sess.get("stats", {"bets":0,"wins":0,"push":0,"payout":0})
    bets, wins, push, payout = st["bets"], st["wins"], st["push"], st["payout"]
    acc = (wins/bets*100.0) if bets>0 else 0.0
    return f"📈 累計：下注 {bets}｜命中 {wins}（{acc:.1f}%）｜和 {push}｜盈虧 {payout}"

# ------------ LINE SDK ------------
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
    log.warning("LINE SDK not available, Dummy mode: %s", e)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_TIMEOUT = float(os.getenv("LINE_TIMEOUT","2.0"))

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
        def handle(self, body, signature): log.info("[DummyLINE] handle")
    class _DummyLineAPI:
        def reply_message(self, token, message):
            txt = getattr(message, "text", str(message))
            log.info("[DummyLINE] reply: %s", txt)
    handler = _DummyHandler()
    line_bot_api = _DummyLineAPI()

def _qr_btn(label, text):
    if LINE_MODE=="real": return QuickReplyButton(action=MessageAction(label=label, text=text))
    return {"label":label, "text":text}

def settings_quickreply(sess) -> list:
    return [
        _qr_btn("選館別", "設定 館別"),
        _qr_btn("查看統計", "查看統計"),
        _qr_btn("試用剩餘", "試用剩餘"),
        _qr_btn("重設流程", "重設"),
    ]

def halls_quickreply() -> list:
    return [_qr_btn(f"{i}", f"{i}") for i in range(1,11)]

def welcome_text(uid):
    left = _trial_left_text(uid)
    title = "平衡預測系統" if MODEL_MODE=="balanced" else "獨立預測系統"
    return (
        f"👋 歡迎使用 BGS AI {title}！\n"
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

def _reply(token, text, quick=None):
    try:
        if LINE_MODE=="real":
            if quick: line_bot_api.reply_message(token, TextSendMessage(text=text, quick_reply=QuickReply(items=quick)))
            else:     line_bot_api.reply_message(token, TextSendMessage(text=text))
        else:
            log.info("[DummyLINE] reply%s: %s", " (with quick)" if quick else "", text)
    except Exception as e:
        log.warning("LINE reply_message error: %s", e)

# ------------ HTTP 路由 ------------
if _has_flask:
    @app.get("/")
    def root(): return f"✅ BGS Server OK ({MODEL_MODE})", 200

    @app.get("/health")
    def health():
        return jsonify(
            ok=True, ts=time.time(),
            mode=MODEL_MODE,
            pf_status=PF_STATUS,
            line_mode=("real" if LINE_MODE=="real" else "dummy"),
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

# ------------ LINE 事件處理 ------------
def _end_trial_text():
    return (
        "⛔ 試用期已到\n"
        f"📬 請聯繫管理員開通登入帳號\n👉 加入官方 LINE：{ADMIN_LINE}\n"
        f"或搜尋：{ADMIN_CONTACT}\n"
        "（取得密碼後回覆：開通 你的密碼）"
    )

def _handle_message_core(event):
    user_id = getattr(getattr(event, "source", None), "user_id", None) or "dummy-user"
    text = (getattr(getattr(event, "message", None), "text", "") or "").strip()

    _start_trial(user_id)
    if text.startswith("開通"):
        pwd = text[2:].strip()
        reply = "✅ 已開通成功！" if pwd == OPENCODE else "❌ 開通碼錯誤，請重新輸入。"
        if pwd == OPENCODE:
            _set_opened(user_id)
        _reply(event.reply_token, reply, quick=settings_quickreply(SESS.setdefault(user_id, {})))
        return

    if not _is_trial_valid(user_id):
        _reply(event.reply_token, _end_trial_text()); return

    sess = SESS.setdefault(user_id, {"bankroll": 0})
    sess["user_id"] = user_id

    if text in ("設定","⋯","menu","Menu"):
        _reply(event.reply_token, "⚙️ 設定選單：", quick=settings_quickreply(sess)); return
    if text == "查看統計":
        _reply(event.reply_token, _format_stats(sess), quick=settings_quickreply(sess)); return
    if text == "試用剩餘":
        _reply(event.reply_token, f"⏳ 試用剩餘：{_trial_left_text(user_id)}", quick=settings_quickreply(sess)); return
    if text == "重設":
        SESS[user_id] = {"bankroll": 0, "user_id": user_id}
        _reply(event.reply_token, "✅ 已重設流程，請選擇館別：", quick=halls_quickreply()); return

    # 館別 → 桌號 → 本金
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
            _reply(event.reply_token, f"👍 已設定本金：{sess['bankroll']:,}\n請輸入上一局點數開始{('平衡' if MODEL_MODE=='balanced' else '獨立')}預測", quick=settings_quickreply(sess))
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

# ------------ 啟動 ------------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting BGS (%s) on port %s (LINE_MODE=%s)", MODEL_MODE, port, LINE_MODE)
    if hasattr(app, "run"):
        app.run(host="0.0.0.0", port=port, debug=False)
