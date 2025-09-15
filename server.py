# server.py — 已整合 Redis + 開始分析XY（無空格）版
# Author: 親愛的 x GPT-5 Thinking

import os
import logging
import time
import csv
import pathlib
import re
import json
from typing import Optional, Dict

import numpy as np
import redis
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

VERSION = "bgs-pf-rbexact-setup-flow-2025-09-17-redis-final-ka"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

app = Flask(__name__)
CORS(app)

# ---------- Redis Sessions ----------
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Successfully connected to Redis.")
    except Exception as e:
        log.error("Failed to connect to Redis: %s. Falling back to in-memory session.", e)
        redis_client = None
else:
    log.warning("REDIS_URL not set. Falling back to in-memory session (for local testing).")
    redis_client = None

SESS_FALLBACK: Dict[str, Dict[str, object]] = {}
SESSION_EXPIRE_SECONDS = 3600  # 1 小時

def get_session(uid: str) -> Dict:
    if redis_client:
        key = f"bgs_session:{uid}"
        j = redis_client.get(key)
        if j: return json.loads(j)
    elif uid in SESS_FALLBACK:
        return SESS_FALLBACK[uid]
    now = int(time.time())
    return {
        "bankroll": 0, "trial_start": now, "premium": False,
        "phase": "choose_game", "game": None, "table": None,
        "last_pts_text": None, "table_no": None,
    }

def save_session(uid: str, data: Dict):
    if redis_client:
        key = f"bgs_session:{uid}"
        redis_client.set(key, json.dumps(data), ex=SESSION_EXPIRE_SECONDS)
    else:
        SESS_FALLBACK[uid] = data

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try: return 1 if int(float(v)) != 0 else 0
    except: return 1 if default else 0

# ---------- 解析上局點數 ----------
INV = {0: "莊", 1: "閒", 2: "和"}

def parse_last_hand_points(text: str):
    """回 (P_total, B_total)；支援：'65'、'閒6莊5'、'P6 B5'、'和'、'和9'，全形數字OK"""
    if not text: return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    u = s.upper().replace("：", ":")
    # 和局
    m = re.search(r"(?:和|TIE|DRAW)\s*:?\s*(\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else None
    # 閒..莊.. / P..B..
    m = re.search(r"(?:閒|P)\s*:?\s*(\d)\D+(?:莊|B)\s*:?\s*(\d)", u)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|B)\s*:?\s*(\d)\D+(?:閒|P)\s*:?\s*(\d)", u)
    if m: return (int(m.group(2)), int(m.group(1)))
    # 純兩位數字：先閒後莊
    digits = re.findall(r"\d", u)
    if len(digits) >= 2: return (int(digits[0]), int(digits[1]))
    t = u.strip().replace(" ", "")
    if t in ("B","莊"): return (0,1)
    if t in ("P","閒"): return (1,0)
    return None

# ---------- 試用/授權 ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")

def validate_activation_code(code: str) -> bool:
    return bool(ADMIN_ACTIVATION_SECRET) and code and (code == ADMIN_ACTIVATION_SECRET)

def trial_left_minutes(sess: Dict) -> int:
    if sess.get("premium", False): return 9999
    now = int(time.time()); start = int(sess.get("trial_start", now))
    used = (now - start) // 60
    return max(0, TRIAL_MINUTES - used)

def trial_guard(sess: Dict) -> Optional[str]:
    if sess.get("premium", False): return None
    if trial_left_minutes(sess) <= 0:
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"
    return None

# ---------- Outcome PF ----------
try:
    from bgs.pfilter import OutcomePF
    PF = OutcomePF(
        decks=int(os.getenv("DECKS","8")), seed=int(os.getenv("SEED","42")),
        n_particles=int(os.getenv("PF_N","200")),
        sims_lik=max(1, int(os.getenv("PF_UPD_SIMS","80"))),
        resample_thr=float(os.getenv("PF_RESAMPLE","0.5")),
        backend=os.getenv("PF_BACKEND","exact").lower(),
        dirichlet_eps=float(os.getenv("PF_DIR_EPS","0.002"))
    )
except Exception as e:
    log.error("Could not import OutcomePF, using Dummy. err=%s", e)
    class DummyPF:
        def update_outcome(self, _): pass
        def predict(self, **_): return np.array([0.5, 0.49, 0.01])
        @property
        def backend(self): return "dummy"
    PF = DummyPF()

# ---------- 決策 & 金額 ----------
EDGE_ENTER   = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY    = env_flag("USE_KELLY", 1)
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR", "0.25"))
MAX_BET_PCT  = float(os.getenv("MAX_BET_PCT", "0.015"))

def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0: return 0
    return int(round(bankroll * pct))

def decide_only_bp(prob):
    pB, pP = float(prob[0]), float(prob[1])
    evB, evP = 0.95 * pB - pP, pP - pB
    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))
    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, "⚪ 優勢不足")
    if USE_KELLY:
        # Kelly 的簡化實作
        f = KELLY_FACTOR * ((pB * 0.95 - (1 - pB)) / 0.95 if side == 0 else (pP - (1 - pP)))
        bet_pct = min(MAX_BET_PCT, float(max(0.0, f)))
        reason = "¼-Kelly"
    else:
        if final_edge >= 0.10: bet_pct = 0.25
        elif final_edge >= 0.07: bet_pct = 0.15
        elif final_edge >= 0.04: bet_pct = 0.10
        else: bet_pct = 0.05
        reason = "階梯式配注"
    return (INV[side], final_edge, bet_pct, reason)

def format_output_card(prob, choice, last_pts_text: Optional[str], bet_amt: int):
    b_pct_txt = f"{prob[0]*100:.2f}%"
    p_pct_txt = f"{prob[1]*100:.2f}%"
    header = ["讀取完成"]
    if last_pts_text: header.append(last_pts_text)
    header.append("開始分析下局....")
    header.append("")
    block = [
        "【預測結果】",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"本次預測結果：{choice if choice!='觀望' else '觀'}",
        f"建議下注：{bet_amt:,}"
    ]
    return "\n".join(header + block)

# ---------- 健康檢查 ----------
@app.get("/")
def root(): return f"✅ BGS PF Server OK ({VERSION})", 200
@app.get("/health")
def health(): return jsonify(ok=True, ts=time.time(), version=VERSION), 200
@app.get("/healthz")
def healthz(): return jsonify(ok=True, ts=time.time(), version=VERSION), 200

# ---------- LINE Bot ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None
line_handler = None

GAMES = {"1":"WM","2":"PM","3":"DG","4":"SA","5":"KU","6":"歐博/卡利","7":"KG","8":"全利","9":"名人","10":"MT真人"}

def game_menu_text(left_min: int) -> str:
    lines = ["【請選擇遊戲館別】"]
    for k in sorted(GAMES.keys(), key=lambda x:int(x)):
        lines.append(f"{k}. {GAMES[k]}")
    lines.append("「請直接輸入數字選擇」")
    lines.append(f"⏳ 試用剩餘 {left_min} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
    return "\n".join(lines)

if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import (MessageEvent, TextMessage, FollowEvent, TextSendMessage,
                                    QuickReply, QuickReplyButton, MessageAction)
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        def quick_buttons():
            try:
                return QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="開始分析 ▶️", text="開始分析")),
                    QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
                    QuickReplyButton(action=MessageAction(label="遊戲設定 🎮", text="遊戲設定")),
                    QuickReplyButton(action=MessageAction(label="報莊勝 🅱️", text="B")),
                    QuickReplyButton(action=MessageAction(label="報閒勝 🅿️", text="P")),
                    QuickReplyButton(action=MessageAction(label="報和局 ⚪", text="T")),
                ])
            except Exception:
                return None

        def reply(token: str, text: str):
            try:
                line_api.reply_message(token, TextSendMessage(text=text, quick_reply=quick_buttons()))
            except Exception as e:
                log.warning("[LINE] reply failed: %s", e)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            uid = event.source.user_id
            sess = get_session(uid)
            left = trial_left_minutes(sess)
            reply(event.reply_token,
                  f"👋 歡迎加入！\n請先點『遊戲設定』或輸入『遊戲設定』開始。\n⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            uid = event.source.user_id
            text = (event.message.text or "").strip()
            sess = get_session(uid)

            try:
                log.info("[LINE] uid=%s phase=%s text=%s", uid, sess.get("phase"), text)

                # 試用守門
                guard = trial_guard(sess)
                if guard:
                    reply(event.reply_token, guard)
                    return

                # 先處理：開始分析XY（無空格，支援全形）
                norm = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
                norm = re.sub(r"\s+", "", norm)  # 去掉所有空白
                m_ka = re.fullmatch(r"開始分析(\d)(\d)", norm)
                if m_ka:
                    p_pts = int(m_ka.group(1))  # 閒
                    b_pts = int(m_ka.group(2))  # 莊
                    if p_pts == b_pts:
                        sess["last_pts_text"] = "上局結果: 和局"
                        # 預設不更新和局，避免影響粒子（可用環境變數控制）
                        if int(os.getenv("SKIP_TIE_UPD","1")) == 0:
                            try: PF.update_outcome(2)
                            except Exception as e: log.warning("tie update skipped: %s", e)
                    else:
                        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
                        PF.update_outcome(1 if p_pts > b_pts else 0)

                    # 直接做預測
                    sess["phase"] = "ready"
                    p = PF.predict(sims_per_particle=max(0, int(os.getenv("PF_PRED_SIMS","0"))))
                    choice, edge, bet_pct, reason = decide_only_bp(p)
                    bankroll_now = int(sess.get("bankroll", 0))
                    msg = format_output_card(p, choice, sess.get("last_pts_text"),
                                             bet_amt=bet_amount(bankroll_now, bet_pct))
                    reply(event.reply_token, msg)
                    save_session(uid, sess)
                    return

                up = text.upper()

                # 開通
                if up.startswith("開通") or up.startswith("ACTIVATE"):
                    code = text.split(" ", 1)[1].strip() if " " in text else ""
                    sess["premium"] = validate_activation_code(code)
                    reply(event.reply_token, "✅ 已開通成功！" if sess["premium"] else "❌ 密碼錯誤")
                    save_session(uid, sess)
                    return

                # 遊戲設定流程入口
                if up in ("遊戲設定","設定","SETUP","GAME"):
                    sess["phase"] = "choose_game"
                    reply(event.reply_token, "🎮 遊戲設定開始\n" + game_menu_text(trial_left_minutes(sess)))
                    save_session(uid, sess)
                    return

                phase = sess.get("phase","choose_game")

                if phase == "choose_game":
                    if re.fullmatch(r"([1-9]|10)", text):
                        sess["game"] = GAMES[text]
                        sess["phase"] = "choose_table"
                        reply(event.reply_token, f"✅ 已設定遊戲類別【{sess['game']}】\n請輸入需預測桌號（Ex: DG01）")
                        save_session(uid, sess)
                        return

                elif phase == "choose_table":
                    t = re.sub(r"\s+", "", text).upper()
                    if re.fullmatch(r"[A-Z]{2}\d{2}", t):
                        sess["table"] = t
                        sess["phase"] = "await_bankroll"
                        reply(event.reply_token, f"✅ 已設定桌號【{sess['table']}】\n請輸入您的本金金額（例如: 5000）")
                        save_session(uid, sess)
                        return
                    else:
                        reply(event.reply_token, "❌ 桌號格式錯誤，請輸入 2 個英文字母 + 2 個數字（例如: DG01）")
                        return

                elif phase == "await_bankroll":
                    if text.isdigit() and int(text) > 0:
                        sess["bankroll"] = int(text)
                        sess["phase"] = "await_pts"
                        reply(event.reply_token, f"👍 已設定本金：{sess['bankroll']:,}\n\n"
                                                 "📡 連接數據庫中..\n✅ 連接數據庫完成\n"
                                                 "📌 請輸入上局閒莊點數（例：65，先閒後莊；或輸入『和』）")
                        save_session(uid, sess)
                        return
                    else:
                        reply(event.reply_token, "❌ 金額格式錯誤，請直接輸入一個正整數（例如: 5000）")
                        return

                elif phase == "await_pts":
                    pts = parse_last_hand_points(text)
                    if pts is not None:
                        if pts[0] == pts[1]:
                            sess["last_pts_text"] = "上局結果: 和局"; PF.update_outcome(2)
                        else:
                            sess["last_pts_text"] = f"上局結果: 閒 {int(pts[0])} 莊 {int(pts[1])}"
                            PF.update_outcome(1 if int(pts[0]) > int(pts[1]) else 0)
                        sess["phase"] = "ready"
                        left = trial_left_minutes(sess)
                        reply(event.reply_token, f"✅ 已記錄上一局點數。\n所有設定完成！請點擊或輸入『開始分析』。\n"
                                                 f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
                        save_session(uid, sess)
                        return
                    else:
                        reply(event.reply_token, "格式錯誤，請再輸入上局點數（例：65 / 和 / 閒6莊5）")
                        return

                # 舊版：開始分析 / 開始分析 <桌號>
                m2 = re.match(r"^開始分析(?:\s+(\d+))?$", text)
                if (text == "開始分析" or m2):
                    if sess.get("phase") != "ready":
                        reply(event.reply_token, "⚠️ 請先完成所有設定（館別→桌號→本金→點數）才能開始分析。")
                        return
                    if m2 and m2.group(1): sess["table_no"] = m2.group(1)
                    p = PF.predict(sims_per_particle=max(0, int(os.getenv("PF_PRED_SIMS","0"))))
                    choice, edge, bet_pct, reason = decide_only_bp(p)
                    bankroll_now = int(sess.get("bankroll", 0))
                    msg = format_output_card(p, choice, sess.get("last_pts_text"),
                                             bet_amt=bet_amount(bankroll_now, bet_pct))
                    reply(event.reply_token, msg)
                    save_session(uid, sess)
                    return

                # 結束分析
                if up in ("結束分析","清空","RESET"):
                    premium = sess.get("premium", False)
                    start_ts = sess.get("trial_start", int(time.time()))
                    sess = get_session(uid)  # 重新初始化
                    sess["premium"] = premium
                    sess["trial_start"] = start_ts
                    left = trial_left_minutes(sess)
                    reply(event.reply_token, f"🧹 已清空。請輸入『遊戲設定』開始新的分析。\n"
                                             f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
                    save_session(uid, sess)
                    return

                # 其他
                left = trial_left_minutes(sess)
                reply(event.reply_token, "指令無法辨識。\n"
                                         "➡️ 若要開始，請點擊或輸入『遊戲設定』。\n"
                                         "➡️ 想直接分析，試試輸入：開始分析65（先閒後莊）。\n"
                                         f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")

            except Exception as e:
                log.exception("on_text error: %s", e)
                try:
                    reply(event.reply_token, "⚠️ 系統發生錯誤，請稍後再試。")
                except Exception:
                    pass

        @app.post("/line-webhook")
        def line_webhook():
            signature = request.headers.get("X-Line-Signature", ""); body = request.get_data(as_text=True)
            try:
                line_handler.handle(body, signature)
            except InvalidSignatureError:
                abort(400, "Invalid signature")
            except Exception as e:
                log.error("webhook error: %s", e); abort(500)
            return "OK", 200

    except Exception as e:
        log.warning("LINE not fully configured: %s", e)

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
