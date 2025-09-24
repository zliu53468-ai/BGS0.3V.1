# -*- coding: utf-8 -*-
"""
server.py — BGS百家樂AI 最終修正版 - 100%解決配注區間問題
"""
import os, sys, re, time, json, math, random, logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import threading

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

SESS: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE = 3600

# === 最終參數設定 ===
os.environ.setdefault("MIN_BET_PCT", "0.10")  # 確保10%下限
os.environ.setdefault("MAX_BET_PCT", "0.30")  # 確保30%上限

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
    def predict(self, **k): return np.array([0.48,0.47,0.05], dtype=np.float32)
    def update_point_history(self, p_pts, b_pts): pass

def _get_pf_from_sess(sess: Dict[str, Any]) -> Any:
    if OutcomePF:
        if sess.get("pf") is None:
            try:
                sess["pf"] = OutcomePF(
                    decks=6,
                    seed=int(time.time() % 1000),
                    n_particles=80
                )
            except Exception:
                sess["pf"] = _DummyPF()
        return sess["pf"]
    return _DummyPF()

TRIAL_SECONDS = 1800
OPENCODE = "aaa8881688"
ADMIN_LINE = "https://lin.ee/Dlm6Y3u"

def _now(): return int(time.time())

def _get_user_info(user_id):
    k = f"bgsu:{user_id}"
    if rcli:
        s = rcli.get(k)
        if s:
            return json.loads(s)
    return SESS.get(user_id, {})

def _set_user_info(user_id, info):
    k = f"bgsu:{user_id}"
    if rcli:
        rcli.set(k, json.dumps(info), ex=86400)
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

# === 最終版配注計算 - 100%確保10%-30%區間 ===
def calculate_proper_bet_percentage(pB, pP, pT):
    """最終版配注計算，保證返回10%-30%的比例"""
    
    # 1. 基礎概率優勢計算
    prob_advantage = abs(pB - pP)
    
    # 2. 選擇優勢方向
    if pB > pP:
        base_edge = pB - pP
        main_prob = pB
    else:
        base_edge = pP - pB  
        main_prob = pP
    
    # 3. 和局影響調整 (和局概率高時略微保守)
    tie_adjust = 1.0 - min(0.5, pT * 0.8)  # 和局最高影響50%調整
    
    # 4. 核心配注算法 - 線性映射到10%-30%
    # 概率優勢0% → 10%下注
    # 概率優勢10% → 30%下注
    raw_bet_pct = 0.10 + (prob_advantage * 2.0)  # 每1%優勢增加0.2%下注
    
    # 5. 應用和局調整
    adjusted_bet_pct = raw_bet_pct * tie_adjust
    
    # 6. 主概率加成 (當主概率>52%時額外加成)
    if main_prob > 0.52:
        main_bonus = (main_prob - 0.52) * 0.5  # 每超出1%增加0.5%
        adjusted_bet_pct += main_bonus
    
    # 7. 嚴格限制在10%-30%範圍
    final_bet_pct = max(0.10, min(0.30, adjusted_bet_pct))
    
    # 8. 防呆檢查
    if final_bet_pct < 0.10 or final_bet_pct > 0.30:
        final_bet_pct = 0.15  # 預設中間值
    
    return round(final_bet_pct, 3)  # 取小數點3位

# === 最終版預測邏輯 ===
def handle_points_and_predict(sess: Dict[str,Any], p_pts: int, b_pts: int) -> str:
    if not (0 <= int(p_pts) <= 9 and 0 <= int(b_pts) <= 9):
        return "❌ 點數數據異常（僅接受 0~9）。請重新輸入，例如：65 / 和 / 閒6莊5"

    pf = _get_pf_from_sess(sess)
    pf.update_point_history(p_pts, b_pts)
    sess["hand_idx"] = int(sess.get("hand_idx", 0)) + 1

    # 結果記錄
    if p_pts == b_pts:
        try: pf.update_outcome(2)
        except Exception: pass
        real_label = "和"
    else:
        outcome = 1 if p_pts > b_pts else 0
        real_label = "閒" if p_pts > b_pts else "莊"
        try: pf.update_outcome(outcome)
        except Exception: pass

    # 獲取概率預測
    p_raw = pf.predict(sims_per_particle=30)
    p_adj = p_raw / np.sum(p_raw)
    
    # 概率處理
    p_final = np.clip(p_adj, 0.01, 0.98)
    p_final = p_final / np.sum(p_final)

    pB, pP, pT = float(p_final[0]), float(p_final[1]), float(p_final[2])
    
    # 選擇預測方向
    if pB > pP:
        choice_text = "莊"
        confidence = pB - pP
    else:
        choice_text = "閒"
        confidence = pP - pB

    # === 核心修正：使用最終版配注計算 ===
    bankroll = int(sess.get("bankroll", 0))
    bet_pct = calculate_proper_bet_percentage(pB, pP, pT)
    
    # 最終安全檢查
    bet_pct = max(0.10, min(0.30, bet_pct))
    bet_amt = int(round(bankroll * bet_pct)) if bankroll > 0 else 0

    # 統計記錄
    st = sess.setdefault("stats", {"bets": 0, "wins": 0, "push": 0, "payout": 0})
    if real_label == "和":
        st["push"] += 1
    else:
        st["bets"] += 1
        if choice_text == real_label:
            win_amt = int(round(bet_amt * 0.95)) if real_label == "莊" else bet_amt
            st["payout"] += win_amt
            st["wins"] += 1
        else:
            st["payout"] -= bet_amt

    # 歷史記錄
    if "hist_pred" not in sess: sess["hist_pred"] = []
    if "hist_real" not in sess: sess["hist_real"] = []
    sess["hist_pred"].append(choice_text)
    sess["hist_real"].append(real_label)
    
    # 命中率計算
    def calculate_accuracy(history_pred, history_real, last_n=30):
        pairs = [(p, r) for p, r in zip(history_pred[-last_n:], history_real[-last_n:]) 
                if r in ("莊", "閒") and p in ("莊", "閒")]
        if not pairs: return 0, 0, 0.0
        hits = sum(1 for p, r in pairs if p == r)
        total = len(pairs)
        accuracy = (hits / total) * 100
        return hits, total, accuracy

    hits, total, accuracy = calculate_accuracy(sess["hist_pred"], sess["hist_real"], 30)
    acc_text = f"📊 近30手命中率：{accuracy:.1f}%（{hits}/{total}）" if total > 0 else "📊 近30手命中率：尚無資料"

    # 策略分類
    if bet_pct < 0.15:
        strategy = f"🟡 低信心配注 {bet_pct*100:.1f}%"
    elif bet_pct < 0.25:
        strategy = f"🟠 中信心配注 {bet_pct*100:.1f}%"
    else:
        strategy = f"🟢 高信心配注 {bet_pct*100:.1f}%"

    # 結果訊息
    result_msg = [
        f"上局結果: {'和 ' + str(p_pts) if p_pts == b_pts else '閒 ' + str(p_pts) + ' 莊 ' + str(b_pts)}",
        "開始分析下局....",
        "",
        "【預測結果】",
        f"閒：{pP*100:.2f}%",
        f"莊：{pB*100:.2f}%", 
        f"和：{pT*100:.2f}%",
        f"本次預測：{choice_text} (信心度：{confidence*100:.1f}%)",
        f"建議下注：{bet_amt:,} 元",
        f"配注策略：{strategy}",
        acc_text,
        "",
        "💡 直接輸入下一局點數繼續（例：65 / 閒6莊5）"
    ]

    return "\n".join(result_msg)

# ========== LINE webhook ==========
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
else:
    line_bot_api = None
    handler = None

def _async_reply(token, text):
    if line_bot_api:
        try:
            line_bot_api.reply_message(token, TextSendMessage(text=text))
        except Exception as e:
            print("LINE回覆錯誤:", e)

@app.route("/line-webhook", methods=['POST'])
def callback():
    if handler:
        signature = request.headers.get('X-Line-Signature', '')
        body = request.get_data(as_text=True)
        try:
            handler.handle(body, signature)
        except Exception as e:
            print("LINE webhook錯誤:", e)
    return "OK", 200

def welcome_text(uid):
    left = _left_trial_sec(uid)
    return (
        "👋 歡迎使用 BGS AI 預測系統！\n"
        "【使用步驟】\n"
        "1️⃣ 選擇館別（輸入 1~10）\n" 
        "2️⃣ 輸入桌號（例：DG01）\n"
        "3️⃣ 輸入本金（例：5000）\n"
        "4️⃣ 回報點數（例：65 / 閒6莊5）\n"
        f"⏰ 試用剩餘：{left}\n\n"
        "【請選擇遊戲館別】\n"
        "1. WM 2. PM 3. DG 4. SA 5. KU\n"
        "6. 歐博/卡利 7. KG 8. 金利 9. 名人 10. MT\n"
        "(輸入數字 1-10)"
    )

@handler.add(MessageEvent, message=TextMessage) if handler else None
def handle_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    info = _get_user_info(user_id)

    if text.startswith("開通"):
        code = text[2:].strip()
        if code == OPENCODE:
            _set_opened(user_id)
            reply = "✅ 帳號已開通！享受永久服務"
        else:
            reply = "❌ 開通碼錯誤"
        threading.Thread(target=_async_reply, args=(event.reply_token, reply)).start()
        return

    if not _is_trial_valid(user_id):
        reply = f"⛔ 試用期已到期\n請聯繫管理員開通\n{ADMIN_LINE}"
        threading.Thread(target=_async_reply, args=(event.reply_token, reply)).start()
        return

    _start_trial(user_id)
    sess = SESS.setdefault(user_id, {})
    sess["user_id"] = user_id

    # 館別選擇
    if not sess.get("hall_id"):
        if text.isdigit() and 1 <= int(text) <= 10:
            halls = ["WM", "PM", "DG", "SA", "KU", "歐博/卡利", "KG", "金利", "名人", "MT真人"]
            sess["hall_id"] = int(text)
            reply = f"✅ 已選 {halls[int(text)-1]}\n請輸入桌號（例：DG01）"
        else:
            reply = welcome_text(user_id)
        threading.Thread(target=_async_reply, args=(event.reply_token, reply)).start()
        return

    # 桌號輸入
    if not sess.get("table_id"):
        if re.match(r"^[A-Za-z]{2}\d{2}$", text):
            sess["table_id"] = text.upper()
            reply = f"✅ 桌號 {sess['table_id']}\n請輸入本金（例：5000）"
        else:
            reply = "請輸入正確桌號格式（例：DG01）"
        threading.Thread(target=_async_reply, args=(event.reply_token, reply)).start()
        return

    # 本金設定
    if not sess.get("bankroll") or sess["bankroll"] <= 0:
        if re.match(r"^\d{3,6}$", text):
            sess["bankroll"] = int(text)
            reply = f"✅ 本金 {sess['bankroll']:,}\n請輸入上局點數（例：65）"
        else:
            reply = "請輸入正確本金（100-999999）"
        threading.Thread(target=_async_reply, args=(event.reply_token, reply)).start()
        return

    # 點數處理
    try:
        if re.match(r"^\d{2}$", text):  # 65格式
            p_pts, b_pts = int(text[0]), int(text[1])
            reply = handle_points_and_predict(sess, p_pts, b_pts)
        elif "閒" in text and "莊" in text:  # 閒6莊5格式
            p_match = re.search(r"閒(\d)", text)
            b_match = re.search(r"莊(\d)", text)
            if p_match and b_match:
                p_pts, b_pts = int(p_match.group(1)), int(b_match.group(1))
                reply = handle_points_and_predict(sess, p_pts, b_pts)
            else:
                reply = "請輸入正確格式：閒6莊5 或 65"
        else:
            reply = "請輸入點數（例：65 / 閒6莊5）"
    except Exception as e:
        reply = f"❌ 處理錯誤：{str(e)}"

    threading.Thread(target=_async_reply, args=(event.reply_token, reply)).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logging.basicConfig(level=logging.INFO)
    print(f"✅ BGS伺服器啟動於端口 {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
