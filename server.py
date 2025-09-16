# server.py — Redis + 開始分析XY + 去重 + 健康檢查 + 開通先於試用守門 + 多格式開通
# Author: 親愛的 x GPT-5 Thinking
# Version: bgs-pf-rbexact-setup-flow-2025-09-17-redis-final-ka4

import os
import logging
import time
import re
import json
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import redis
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

VERSION = "bgs-pf-rbexact-setup-flow-2025-09-17-redis-final-ka4"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Redis Sessions ----------
REDIS_URL = os.getenv("REDIS_URL")
redis_client: Optional[redis.Redis] = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Successfully connected to Redis.")
    except Exception as e:
        log.error("Failed to connect to Redis: %s. Falling back to in-memory session.", e)
else:
    log.warning("REDIS_URL not set. Falling back to in-memory session (for local testing).")

SESS_FALLBACK: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE_SECONDS = 3600  # 1 小時
DEDUPE_TTL = 60  # 相同事件去重秒數

def _rget(key: str) -> Optional[str]:
    try:
        return redis_client.get(key) if redis_client else None
    except Exception as e:
        log.warning("[Redis] GET error: %s", e)
        return None

def _rset(key: str, val: str, ex: Optional[int] = None):
    try:
        if redis_client: redis_client.set(key, val, ex=ex)
    except Exception as e:
        log.warning("[Redis] SET error: %s", e)

def _rsetnx(key: str, val: str, ex: int) -> bool:
    try:
        if redis_client:
            ok = redis_client.set(key, val, ex=ex, nx=True)
            return bool(ok)
        else:
            # fallback 簡易去重
            if key in SESS_FALLBACK: return False
            SESS_FALLBACK[key] = {"v": val, "exp": time.time() + ex}
            return True
    except Exception as e:
        log.warning("[Redis] SETNX error: %s", e)
        return True  # Redis 掛了就不要擋

def get_session(uid: str) -> Dict[str, Any]:
    if redis_client:
        key = f"bgs_session:{uid}"
        j = _rget(key)
        if j:
            try:
                return json.loads(j)
            except Exception:
                pass
    else:
        # 清理 fallback 去重暫存過期
        now = time.time()
        for k in list(SESS_FALLBACK.keys()):
            v = SESS_FALLBACK.get(k)
            if isinstance(v, dict) and "exp" in v and v["exp"] < now:
                del SESS_FALLBACK[k]
        if uid in SESS_FALLBACK and isinstance(SESS_FALLBACK[uid], dict) and "phase" in SESS_FALLBACK[uid]:
            return SESS_FALLBACK[uid]

    now = int(time.time())
    return {
        "bankroll": 0, "trial_start": now, "premium": False,
        "phase": "choose_game", "game": None, "table": None,
        "last_pts_text": None, "table_no": None,
    }

def save_session(uid: str, data: Dict[str, Any]):
    if redis_client:
        key = f"bgs_session:{uid}"
        _rset(key, json.dumps(data), ex=SESSION_EXPIRE_SECONDS)
    else:
        SESS_FALLBACK[uid] = data

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except:
        return 1 if default else 0

# ---------- 解析上局點數 ----------
INV = {0: "莊", 1: "閒", 2: "和"}

def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
    """回 (P_total, B_total)；支援：'65'、'閒6莊5'、'P6 B5'、'和'、'和9'，全形數字OK；亦容忍混雜字串內出現兩個數字。"""
    if not text: return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    u = s.upper().replace("：", ":")

    # 和局（可含點數）
    m = re.search(r"(?:和|TIE|DRAW)\s*:?\s*(\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else (0, 0)  # 用(0,0)代表和，後續會特判

    # 閒..莊.. / P..B..
    m = re.search(r"(?:閒|P)\s*:?\s*(\d)\D+(?:莊|B)\s*:?\s*(\d)", u)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|B)\s*:?\s*(\d)\D+(?:閒|P)\s*:?\s*(\d)", u)
    if m: return (int(m.group(2)), int(m.group(1)))

    # 任意位置找兩個數字
    digits = re.findall(r"\d", u)
    if len(digits) >= 2:
        return (int(digits[0]), int(digits[1]))

    t = u.strip().replace(" ", "")
    if t in ("B","莊"): return (0, 1)
    if t in ("P","閒"): return (1, 0)
    if t in ("T","和"): return (0, 0)
    return None

# ---------- 試用/授權 ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")

# 預設密碼：若未設定環境變數，預設為 aaa8881688
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "aaa8881688")

def validate_activation_code(code: str) -> bool:
    # 允許前後空白、全形空白與冒號，多格式：開通 密碼 / 開通密碼 / 開通:密碼
    if not code: return False
    norm = str(code).replace("\u3000", " ").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and (norm == ADMIN_ACTIVATION_SECRET)

def trial_left_minutes(sess: Dict[str, Any]) -> int:
    if sess.get("premium", False): return 9999
    now = int(time.time()); start = int(sess.get("trial_start", now))
    used = (now - start) // 60
    return max(0, TRIAL_MINUTES - used)

def trial_guard(sess: Dict[str, Any]) -> Optional[str]:
    if sess.get("premium", False): return None
    if trial_left_minutes(sess) <= 0:
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"
    return None

# 啟動時印出是否載到密碼（不印明文）
try:
    log.info("Activation secret loaded? %s (len=%d)", bool(ADMIN_ACTIVATION_SECRET), len(ADMIN_ACTIVATION_SECRET))
except Exception:
    pass

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
        def predict(self, **_): return np.array([0.5, 0.49, 0.01])  # B, P, T
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

def decide_only_bp(prob: np.ndarray) -> Tuple[str, float, float, str]:
    pB, pP = float(prob[0]), float(prob[1])
    evB, evP = 0.95 * pB - pP, pP - pB
    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))
    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, "⚪ 優勢不足")
    if USE_KELLY:
        if side == 0:
            b = 0.95
            f = KELLY_FACTOR * ((pB * b - (1 - pB)) / b)
        else:
            b = 1.0
            f = KELLY_FACTOR * ((pP * b - (1 - pP)) / b)
        bet_pct = min(MAX_BET_PCT, max(0.0, float(f)))
        reason = "¼-Kelly"
    else:
        if final_edge >= 0.10: bet_pct = 0.25
        elif final_edge >= 0.07: bet_pct = 0.15
        elif final_edge >= 0.04: bet_pct = 0.10
        else: bet_pct = 0.05
        reason = "階梯式配注"
    return (INV[side], final_edge, bet_pct, reason)

def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int) -> str:
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
def root():
    ua = request.headers.get("User-Agent", "")
    if "UptimeRobot" in ua:
        return "OK", 200
    return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/health")
def health(): return jsonify(ok=True, ts=time.time(), version=VERSION), 200

@app.get("/healthz")
def healthz(): return jsonify(ok=True, ts=time.time(), version=VERSION), 200

# ---------- LINE Bot ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None
line_handler = None

GAMES = {
    "1":"WM","2":"PM","3":"DG","4":"SA","5":"KU",
    "6":"歐博/卡利","7":"KG","8":"全利","9":"名人","10":"MT真人"
}

def game_menu_text(left_min: int) -> str:
    lines = ["【請選擇遊戲館別】"]
    for k in sorted(GAMES.keys(), key=lambda x:int(x)):
        lines.append(f"{k}. {GAMES[k]}")
    lines.append("「請直接輸入數字選擇」")
    lines.append(f"⏳ 試用剩餘 {left_min} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
    return "\n".join(lines)

def _quick_buttons():
    try:
        from linebot.models import QuickReply, QuickReplyButton, MessageAction
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

def _reply(token: str, text: str):
    from linebot.models import TextSendMessage
    try:
        qr = _quick_buttons()
        line_api.reply_message(token, TextSendMessage(text=text, quick_reply=qr))
    except Exception as e:
        log.warning("[LINE] reply failed: %s", e)

def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id: return True
    key = f"dedupe:{event_id}"
    return _rsetnx(key, "1", DEDUPE_TTL)

if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import (MessageEvent, TextMessage, FollowEvent)

        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            if not _dedupe_event(getattr(event, "id", None)):
                return
            uid = event.source.user_id
            sess = get_session(uid)
            left = trial_left_minutes(sess)
            _reply(event.reply_token,
                   f"👋 歡迎加入！\n請先點『遊戲設定』或輸入『遊戲設定』開始。\n⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            if not _dedupe_event(getattr(event, "id", None)):
                return

            uid = event.source.user_id
            raw = (event.message.text or "")
            # 正規化空白：把全形空白換半形，保留單一空白給指令解析
            text = re.sub(r"\s+", " ", raw.replace("\u3000", " ")).strip()
            sess = get_session(uid)

            try:
                log.info("[LINE] uid=%s phase=%s text=%s", uid, sess.get("phase"), text)

                # --- 1) 先處理「開通」指令（避免被試用守門擋掉） ---
                up = text.upper()
                if up.startswith("開通") or up.startswith("ACTIVATE"):
                    # 支援：開通 密碼｜開通密碼｜開通:密碼（半形/全形冒號、空白）
                    code = ""
                    # 取「開通」後的字串
                    after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                    after = after.replace("\u3000", " ").replace("：", ":").strip()
                    if after:
                        if after[0] in (":", "："): after = after[1:].strip()
                        # 去掉可能的前導「:」或空白
                        after = after.lstrip(":").strip()
                        code = after
                    ok = validate_activation_code(code)
                    sess["premium"] = bool(ok)
                    _reply(event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤")
                    save_session(uid, sess)
                    return

                # --- 2) 再做試用守門 ---
                guard = trial_guard(sess)
                if guard:
                    _reply(event.reply_token, guard)
                    return

                # 先處理：開始分析XY（無空格，支援全形）
                norm = raw.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
                norm = re.sub(r"\s+", "", norm)  # 去掉所有空白
                m_ka = re.fullmatch(r"開始分析(\d)(\d)", norm)
                if m_ka:
                    p_pts = int(m_ka.group(1))  # 閒
                    b_pts = int(m_ka.group(2))  # 莊
                    if p_pts == b_pts:
                        sess["last_pts_text"] = "上局結果: 和局"
                        if int(os.getenv("SKIP_TIE_UPD","1")) == 0:
                            try: PF.update_outcome(2)
                            except Exception as e: log.warning("tie update skipped: %s", e)
                    else:
                        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
                        try: PF.update_outcome(1 if p_pts > b_pts else 0)
                        except Exception as e: log.warning("PF update err: %s", e)

                    sess["phase"] = "ready"
                    p = PF.predict(sims_per_particle=max(0, int(os.getenv("PF_PRED_SIMS","0"))))
                    choice, edge, bet_pct, reason = decide_only_bp(p)
                    bankroll_now = int(sess.get("bankroll", 0))
                    msg = format_output_card(p, choice, sess.get("last_pts_text"),
                                             bet_amt=bet_amount(bankroll_now, bet_pct))
                    _reply(event.reply_token, msg)
                    save_session(uid, sess)
                    return

                # 遊戲設定流程入口
                if up in ("遊戲設定","設定","SETUP","GAME"):
                    sess["phase"] = "choose_game"
                    _reply(event.reply_token, "🎮 遊戲設定開始\n" + game_menu_text(trial_left_minutes(sess)))
                    save_session(uid, sess)
                    return

                phase = sess.get("phase","choose_game")

                if phase == "choose_game":
                    if re.fullmatch(r"([1-9]|10)", text):
                        sess["game"] = GAMES[text]
                        sess["phase"] = "choose_table"
                        _reply(event.reply_token, f"✅ 已設定遊戲類別【{sess['game']}】\n請輸入需預測桌號（Ex: DG01）")
                        save_session(uid, sess)
                        return

                elif phase == "choose_table":
                    t = re.sub(r"\s+", "", text).upper()
                    if re.fullmatch(r"[A-Z]{2}\d{2}", t):
                        sess["table"] = t
                        sess["phase"] = "await_bankroll"
                        _reply(event.reply_token, f"✅ 已設定桌號【{sess['table']}】\n請輸入您的本金金額（例如: 5000）")
                        save_session(uid, sess)
                        return
                    else:
                        _reply(event.reply_token, "❌ 桌號格式錯誤，請輸入 2 個英文字母 + 2 個數字（例如: DG01）")
                        return

                elif phase == "await_bankroll":
                    if text.isdigit() and int(text) > 0:
                        sess["bankroll"] = int(text)
                        sess["phase"] = "await_pts"
                        _reply(event.reply_token, f"👍 已設定本金：{sess['bankroll']:,}\n\n"
                                                  "📡 連接數據庫中..\n✅ 連接數據庫完成\n"
                                                  "📌 請輸入上局閒莊點數（例：65，先閒後莊；或輸入『和』）")
                        save_session(uid, sess)
                        return
                    else:
                        _reply(event.reply_token, "❌ 金額格式錯誤，請直接輸入一個正整數（例如: 5000）")
                        return

                elif phase == "await_pts":
                    pts = parse_last_hand_points(text)
                    if pts is not None:
                        if pts[0] == pts[1]:
                            sess["last_pts_text"] = "上局結果: 和局"
                            try: PF.update_outcome(2)
                            except Exception as e: log.warning("PF tie update err: %s", e)
                        else:
                            sess["last_pts_text"] = f"上局結果: 閒 {int(pts[0])} 莊 {int(pts[1])}"
                            try: PF.update_outcome(1 if int(pts[0]) > int(pts[1]) else 0)
                            except Exception as e: log.warning("PF update err: %s", e)

                        sess["phase"] = "ready"
                        left = trial_left_minutes(sess)
                        _reply(event.reply_token, f"✅ 已記錄上一局點數。\n所有設定完成！請點擊或輸入『開始分析』。\n"
                                                  f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
                        save_session(uid, sess)
                        return
                    else:
                        _reply(event.reply_token, "格式錯誤，請再輸入上局點數（例：65 / 和 / 閒6莊5）")
                        return

                # 舊版：開始分析 / 開始分析 <桌號>
                m2 = re.match(r"^開始分析(?:\s+(\d+))?$", text)
                if (text == "開始分析" or m2):
                    if sess.get("phase") != "ready":
                        _reply(event.reply_token, "⚠️ 請先完成所有設定（館別→桌號→本金→點數）才能開始分析。")
                        return
                    if m2 and m2.group(1):
                        sess["table_no"] = m2.group(1)

                    p = PF.predict(sims_per_particle=max(0, int(os.getenv("PF_PRED_SIMS","0"))))
                    choice, edge, bet_pct, reason = decide_only_bp(p)
                    bankroll_now = int(sess.get("bankroll", 0))
                    msg = format_output_card(p, choice, sess.get("last_pts_text"),
                                             bet_amt=bet_amount(bankroll_now, bet_pct))
                    _reply(event.reply_token, msg)
                    save_session(uid, sess)
                    return

                # 結束分析 / RESET
                if up in ("結束分析","清空","RESET"):
                    premium = sess.get("premium", False)
                    start_ts = sess.get("trial_start", int(time.time()))
                    # 重設
                    sess = get_session(uid)
                    sess["premium"] = premium
                    sess["trial_start"] = start_ts
                    left = trial_left_minutes(sess)
                    _reply(event.reply_token, f"🧹 已清空。請輸入『遊戲設定』開始新的分析。\n"
                                              f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
                    save_session(uid, sess)
                    return

                # 其他：提示
                left = trial_left_minutes(sess)
                _reply(event.reply_token, "指令無法辨識。\n"
                                          "➡️ 若要開始，請點擊或輸入『遊戲設定』。\n"
                                          "➡️ 想直接分析，試試輸入：開始分析65（先閒後莊）。\n"
                                          f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")

            except Exception as e:
                log.exception("on_text error: %s", e)
                try:
                    _reply(event.reply_token, "⚠️ 系統發生錯誤，請稍後再試。")
                except Exception:
                    pass

        @app.post("/line-webhook")
        def line_webhook():
            signature = request.headers.get("X-Line-Signature", "")
            body = request.get_data(as_text=True)
            try:
                line_handler.handle(body, signature)
            except InvalidSignatureError:
                abort(400, "Invalid signature")
            except Exception as e:
                log.error("webhook error: %s", e); abort(500)
            return "OK", 200

    except Exception as e:
        log.warning("LINE not fully configured: %s", e)

else:
    log.warning("LINE credentials not set. LINE webhook will not be active.")

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
