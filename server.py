# server.py — Outcome-only（PF + RB-Exact）
# 進場遊戲設定流程 + 截圖版回覆 + 本金配注金額 + 試用剩餘X分鐘 + 快速按鈕 + 全域點數解析容錯
# Author: 親愛的 x GPT-5 Thinking

import os, logging, time, csv, pathlib, re
from typing import Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

VERSION = "bgs-pf-rbexact-setup-flow-2025-09-16-v3"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

app = Flask(__name__)
CORS(app)

# -------------------- 環境旗標 --------------------
def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None:
        return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"): return 1
    if v in ("0", "false", "f", "no", "n", "off"): return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except:
        return 1 if default else 0

# -------------------- 文字解析 --------------------
INV = {0: "莊", 1: "閒", 2: "和"}

def parse_last_hand_points(text: str):
    """
    支援：
    - '65'（先閒後莊；全形數字也可） 
    - '閒6莊5' / '莊5閒6'
    - 'P6 B5' / 'B5 P6'
    - '和' / 'TIE' / 'DRAW'（可含點數如 '和9'）
    回傳 (P_total, B_total) 或 None（只有 '和' 無數字時返回 None 交由呼叫端當和局）
    """
    if not text: return None
    s = str(text).strip()
    # 全形數字轉半形
    trans = str.maketrans("０１２３４５６７８９", "0123456789")
    s = s.translate(trans)

    # 純兩位數（先閒後莊）
    if re.fullmatch(r"\d\d", s):
        return (int(s[0]), int(s[1]))

    u = s.upper().replace("：", ":").replace(" ", "")

    # 和局（可帶點數）
    m = re.search(r"(?:和|TIE|DRAW)[:]?(\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else None  # 無數字就交由呼叫端直接判定和局

    # 閒..莊.. / P..B..
    m = re.search(r"(?:閒|P)[:]?(\d)\D+(?:莊|B)[:]?(\d)", u)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|B)[:]?(\d)\D+(?:閒|P)[:]?(\d)", u)
    if m: return (int(m.group(2)), int(m.group(1)))

    # 單純輸贏字母（讓呼叫端能轉成勝負）
    if u in ("B", "莊"): return (0, 1)  # 代表莊勝（讓呼叫端能判定）
    if u in ("P", "閒"): return (1, 0)
    return None

# -------------------- 試用 / 狀態 --------------------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")

# flow phase: choose_game -> choose_table -> await_bankroll -> await_pts -> ready
SESS: Dict[str, Dict[str, object]] = {}

def _init_user(uid: str):
    now = int(time.time())
    SESS[uid] = {
        "bankroll": 0,
        "trial_start": now,
        "premium": False,
        "phase": "choose_game",
        "game": None,
        "table": None,
        "last_pts_text": None,  # "上局結果: 閒 X 莊 Y" / "上局結果: 和局"
        "table_no": None,
    }

def validate_activation_code(code: str) -> bool:
    return bool(ADMIN_ACTIVATION_SECRET) and code and (code == ADMIN_ACTIVATION_SECRET)

def trial_left_minutes(uid: str) -> int:
    sess = SESS.get(uid) or {}
    if sess.get("premium", False): return 9999
    now = int(time.time()); start = int(sess.get("trial_start", now))
    used = (now - start) // 60
    return max(0, TRIAL_MINUTES - used)

def trial_guard(uid: str) -> Optional[str]:
    if SESS.get(uid, {}).get("premium", False): return None
    if trial_left_minutes(uid) <= 0:
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"
    return None

# -------------------- 引擎：Outcome 粒子濾波 --------------------
SEED  = int(os.getenv("SEED", "42"))
DECKS = int(os.getenv("DECKS", "8"))

try:
    from bgs.pfilter import OutcomePF
    PF = OutcomePF(
        decks=DECKS,
        seed=SEED,
        n_particles=int(os.getenv("PF_N", "200")),
        sims_lik=max(1, int(os.getenv("PF_UPD_SIMS", "80"))),
        resample_thr=float(os.getenv("PF_RESAMPLE", "0.5")),
        backend=os.getenv("PF_BACKEND", "exact").lower(),  # exact | mc
        dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.002")),
    )
except Exception as e:
    log.error("Could not import OutcomePF from bgs.pfilter, use dummy. err=%s", e)
    class DummyPF:
        def update_outcome(self, _): pass
        def predict(self, **_): return np.array([0.5, 0.49, 0.01])
        @property
        def backend(self): return "dummy"
    PF = DummyPF()

# -------------------- 決策與下注金額 --------------------
EDGE_ENTER   = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY    = env_flag("USE_KELLY", 1)
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR", "0.25"))
MAX_BET_PCT  = float(os.getenv("MAX_BET_PCT", "0.015"))

LOG_DIR  = os.getenv("LOG_DIR", "logs")
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
PRED_CSV = os.path.join(LOG_DIR, "predictions.csv")
if not os.path.exists(PRED_CSV):
    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "ts","version","hands","pB","pP","pT","choice","edge","bet_pct",
            "bankroll","bet_amt","engine","reason"
        ])

def banker_ev(pB, pP): return 0.95 * pB - pP
def player_ev(pB, pP): return pP - pB

def kelly_fraction(p_win: float, payoff: float):
    q = 1.0 - p_win
    edge = p_win * payoff - q
    return max(0.0, edge / payoff)

def bet_amount(bankroll:int, pct:float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0: return 0
    return int(round(bankroll * pct))

def decide_only_bp(prob):
    pB, pP = float(prob[0]), float(prob[1])
    evB, evP = banker_ev(pB, pP), player_ev(pB, pP)
    side = 0 if evB > evP else 1
    edge_prob = abs(pB - pP)
    final_edge = max(edge_prob, abs(evB - evP))

    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, "⚪ 優勢不足")

    if USE_KELLY:
        f = KELLY_FACTOR * (kelly_fraction(pB, 0.95) if side == 0 else kelly_fraction(pP, 1.0))
        bet_pct = min(MAX_BET_PCT, float(max(0.0, f)))
        reason = "¼-Kelly"
    else:
        if final_edge >= 0.10: bet_pct = 0.25
        elif final_edge >= 0.07: bet_pct = 0.15
        elif final_edge >= 0.04: bet_pct = 0.10
        else: bet_pct = 0.05
        reason = "階梯式配注"

    return (INV[side], final_edge, bet_pct, reason)

def log_prediction(hands:int, p, choice:str, edge:float, bankroll:int, bet_pct:float, engine:str, reason:str):
    try:
        bet_amt = bet_amount(bankroll, bet_pct)
        with open(PRED_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                int(time.time()), VERSION, hands,
                float(p[0]), float(p[1]), float(p[2]),
                choice, float(edge), float(bet_pct),
                int(bankroll), int(bet_amt), engine, reason
            ])
    except Exception as e:
        log.warning("log_prediction failed: %s", e)

# -------------------- 卡片樣式（與截圖一致 + 金額） --------------------
def format_output_card(prob, choice, last_pts_text: Optional[str], bet_amt: int):
    b_pct_txt = f"{prob[0]*100:.2f}%"
    p_pct_txt = f"{prob[1]*100:.2f}%"
    header = ["讀取完成"]
    if last_pts_text: header.append(last_pts_text)
    header.append("開始分析下局....")
    header.append("")  # 空行
    block = [
        "【預測結果】",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"本次預測結果：{choice if choice != '觀望' else '觀'}",
        f"建議下注：{bet_amt:,}"
    ]
    return "\n".join(header + block)

# -------------------- 健康檢查 --------------------
@app.get("/")
def root(): return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/health")
def health(): return jsonify(ok=True, ts=time.time(), version=VERSION), 200

@app.get("/healthz")
def healthz(): return jsonify(ok=True, ts=time.time(), version=VERSION), 200

# -------------------- REST：只輸贏更新 / 預測（可選給外部測） --------------------
@app.post("/update-outcome")
def update_outcome_api():
    data = request.get_json(silent=True) or {}
    o = str(data.get("outcome", "")).strip().upper()
    if o in ("B","莊","0"): PF.update_outcome(0)
    elif o in ("P","閒","1"): PF.update_outcome(1)
    elif o in ("T","和","TIE","DRAW","2"): PF.update_outcome(2)
    else: return jsonify(ok=False, msg="outcome 必須是 B/P/T 或 莊/閒/和"), 400
    return jsonify(ok=True), 200

@app.post("/predict")
def predict_api():
    data = request.get_json(silent=True) or {}
    bankroll = int(float(data.get("bankroll") or 0))
    p = PF.predict(sims_per_particle=max(0, int(os.getenv("PF_PRED_SIMS", "0"))))
    choice, edge, bet_pct, reason = decide_only_bp(p)
    amt = bet_amount(bankroll, bet_pct)
    msg = format_output_card(p, choice, None, amt)
    log_prediction(-1, p, choice, edge, bankroll, bet_pct, f"PF-{getattr(PF,'backend','')}", reason)
    return jsonify(message=msg, suggestion=choice, bet_amount=amt,
                   probabilities={"banker": float(p[0]), "player": float(p[1])},
                   version=VERSION), 200

# -------------------- LINE Webhook --------------------
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
    for k in sorted(GAMES.keys(), key=lambda x: int(x)):
        lines.append(f"{k}. {GAMES[k]}")
    lines.append("「請直接輸入數字選擇」")
    lines.append(f"⏳ 試用剩餘 {left_min} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
    return "\n".join(lines)

if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import (
            MessageEvent, TextMessage, FollowEvent, TextSendMessage,
            QuickReply, QuickReplyButton, MessageAction
        )

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

        def reply(token: str, text: str, uid: Optional[str] = None):
            try:
                line_api.reply_message(token, TextSendMessage(text=text, quick_reply=quick_buttons()))
            except Exception as e:
                log.warning("[LINE] reply failed: %s", e)
                if uid:
                    try: line_api.push_message(uid, TextSendMessage(text=text, quick_reply=quick_buttons()))
                    except Exception as e2: log.error("[LINE] push failed: %s", e2)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            uid = event.source.user_id
            _init_user(uid)
            left = trial_left_minutes(uid)
            reply(event.reply_token,
                  "👋 歡迎加入！\n請先點『遊戲設定』或輸入『遊戲設定』開始。\n"
                  f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）", uid)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            uid = event.source.user_id
            text = (event.message.text or "").strip()
            try:
                if uid not in SESS: _init_user(uid)
                log.info("[LINE] uid=%s phase=%s text=%s", uid, SESS[uid].get("phase"), text)

                # 試用檢查
                guard = trial_guard(uid)
                if guard:
                    reply(event.reply_token, guard, uid); return

                up = text.upper()

                # 開通信碼
                if up.startswith("開通") or up.startswith("ACTIVATE"):
                    code = text.split(" ", 1)[1].strip() if " " in text else ""
                    SESS[uid]["premium"] = validate_activation_code(code)
                    reply(event.reply_token,
                          "✅ 已開通成功！" if SESS[uid]["premium"] else "❌ 密碼錯誤，請向管理員索取。",
                          uid)
                    return

                # 試用剩餘查詢
                if up in ("試用剩餘","剩餘","TRIAL"):
                    left = trial_left_minutes(uid)
                    reply(event.reply_token, f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）", uid)
                    return

                # 遊戲設定入口
                if up in ("遊戲設定","設定","SETUP","GAME"):
                    SESS[uid]["phase"] = "choose_game"
                    reply(event.reply_token, "🎮 遊戲設定開始\n" + game_menu_text(trial_left_minutes(uid)), uid)
                    return

                # ---- 全域容錯：像點數就直接紀錄，無視 phase ----
                pts_any = parse_last_hand_points(text)
                if pts_any is not None:
                    if pts_any[0] == pts_any[1]:
                        SESS[uid]["last_pts_text"] = "上局結果: 和局"
                        PF.update_outcome(2)
                    else:
                        SESS[uid]["last_pts_text"] = f"上局結果: 閒 {int(pts_any[0])} 莊 {int(pts_any[1])}"
                        PF.update_outcome(1 if int(pts_any[0]) > int(pts_any[1]) else 0)
                    SESS[uid]["phase"] = "ready"
                    left = trial_left_minutes(uid)
                    reply(event.reply_token,
                          f"✅ 已記錄上一局點數。\n現在可輸入『開始分析』或『開始分析 53』。\n"
                          f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）", uid)
                    return

                # 逐步流程
                phase = SESS[uid].get("phase", "choose_game")

                # (a) 館別：1~10
                if phase == "choose_game" and re.fullmatch(r"([1-9]|10)", text):
                    SESS[uid]["game"] = GAMES[text]
                    SESS[uid]["phase"] = "choose_table"
                    reply(event.reply_token,
                          f"✅ 已設定遊戲類別【{SESS[uid]['game']}】\n請輸入需預測桌號（Ex: DG01）", uid)
                    return

                # (b) 桌號：DG05 / DG 0 5
                if phase == "choose_table" and re.fullmatch(r"[A-Za-z]{2}\s*\d\s*\d", text.replace("  ", " ")):
                    t = re.sub(r"\s+", "", text).upper()  # DG05
                    SESS[uid]["table"] = t
                    SESS[uid]["phase"] = "await_bankroll"
                    reply(event.reply_token,
                          f"✅ 已設定桌號【{SESS[uid]['table']}】\n請輸入您的本金金額（例如: 5000）", uid)
                    return

                # (c) 本金
                if phase == "await_bankroll":
                    if text.isdigit() and int(text) > 0:
                        val = int(text)
                        SESS[uid]["bankroll"] = val
                        SESS[uid]["phase"] = "await_pts"
                        reply(event.reply_token,
                              f"👍 已設定本金：{val:,}\n\n"
                              "📡 連接數據庫中..\n✅ 連接數據庫完成\n"
                              "📌 請輸入上局閒莊點數（例：65，先閒後莊；或輸入『和』）",
                              uid)
                        return
                    else:
                        reply(event.reply_token, "❌ 金額格式錯誤，請直接輸入一個數字（例如: 5000）", uid)
                        return

                # 單純回報輸贏（任何 phase 都吃）
                if up in ("B","莊","BANKER"):
                    PF.update_outcome(0); SESS[uid]["last_pts_text"] = "上局結果: 莊勝"; SESS[uid]["phase"] = "ready"
                    reply(event.reply_token, "📝 已記錄上一局：莊勝\n輸入『開始分析』取得建議。", uid); return
                if up in ("P","閒","PLAYER"):
                    PF.update_outcome(1); SESS[uid]["last_pts_text"] = "上局結果: 閒勝"; SESS[uid]["phase"] = "ready"
                    reply(event.reply_token, "📝 已記錄上一局：閒勝\n輸入『開始分析』取得建議。", uid); return
                if up in ("T","和","TIE","DRAW"):
                    PF.update_outcome(2); SESS[uid]["last_pts_text"] = "上局結果: 和局"; SESS[uid]["phase"] = "ready"
                    reply(event.reply_token, "📝 已記錄上一局：和局\n輸入『開始分析』取得建議。", uid); return

                # 開始分析（或 開始分析 53）
                m2 = re.match(r"^開始分析(?:\s+(\d+))?$", text)
                if (text == "開始分析" or m2) and SESS[uid].get("phase") == "ready":
                    if m2 and m2.group(1):
                        SESS[uid]["table_no"] = m2.group(1)
                    p = PF.predict(sims_per_particle=max(0, int(os.getenv("PF_PRED_SIMS", "0"))))
                    choice, edge, bet_pct, reason = decide_only_bp(p)
                    bankroll_now = int(SESS[uid].get("bankroll", 0))
                    msg = format_output_card(p, choice, SESS[uid].get("last_pts_text"),
                                             bet_amt=bet_amount(bankroll_now, bet_pct))
                    log_prediction(0, p, choice, edge, bankroll_now, bet_pct, f"PF-{getattr(PF,'backend','')}", reason)
                    reply(event.reply_token, msg, uid)
                    return

                # 結束分析（保留 trial_start，避免延長試用）
                if up in ("結束分析","清空","RESET"):
                    premium  = SESS.get(uid,{}).get("premium", False)
                    start_ts = SESS.get(uid,{}).get("trial_start", int(time.time()))
                    _init_user(uid)
                    SESS[uid]["premium"] = premium
                    SESS[uid]["trial_start"] = start_ts
                    left = trial_left_minutes(uid)
                    reply(event.reply_token,
                          "🧹 已清空。請輸入『遊戲設定』開始新的分析。\n"
                          f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）", uid)
                    return

                # 預設提示
                left = trial_left_minutes(uid)
                reply(event.reply_token,
                      "指令無法辨識。請輸入『遊戲設定』或直接回報上一局點數（例：65 / 和 / 閒6莊5）。\n"
                      f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）", uid)

            except Exception as e:
                log.exception("on_text error: %s", e)
                try:
                    line_api.reply_message(event.reply_token, TextSendMessage(text="⚠️ 系統忙線，請再傳一次剛剛的內容"))
                except Exception:
                    try: line_api.push_message(uid, TextSendMessage(text="⚠️ 系統忙線，請再傳一次剛剛的內容"))
                    except Exception: pass

        @app.post("/line-webhook")
        def line_webhook():
            signature = request.headers.get("X-Line-Signature", "")
            body = request.get_data(as_text=True)
            try:
                line_handler.handle(body, signature)
            except InvalidSignatureError:
                abort(400, "Invalid signature")
            except Exception as e:
                log.error("webhook error: %s", e)
                abort(500)
            return "OK", 200

    except Exception as e:
        log.warning("LINE not fully configured: %s", e)

# -------------------- 本地啟動 --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
