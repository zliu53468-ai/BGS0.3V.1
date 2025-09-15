# server.py — Outcome-only（PF + RB-Exact）
# 回覆格式（精準比照你的需求）＋ 快速按鈕（加入與結束皆顯示）
# Author: 親愛的 x GPT-5 Thinking

import os, logging, time, csv, pathlib, re
from typing import Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

VERSION = "bgs-pf-rbexact-screenshot-with-amount-2025-09-16"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

app = Flask(__name__)
CORS(app)

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: 
        return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except:
        return 1 if default else 0

# ========= 文字解析 =========
MAP = {"B":0, "P":1, "T":2, "莊":0, "閒":1, "和":2}
INV = {0:"莊", 1:"閒", 2:"和"}

def parse_last_hand_points(text: str):
    """上局結果：閒6 莊8 / 和9 / TIE / DRAW；回 (P_total, B_total) 或 None"""
    if not text: return None
    s = text.strip().upper().replace("：", ":")
    s = re.sub(r"\s+", "", s)
    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:和|TIE|DRAW)\s*:?(\d)', s)
    if m:
        d = int(m.group(1)); return (d, d)
    if re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:和|TIE|DRAW)\b', s):
        return None
    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:閒|P)\s*:?(\d)\s*(?:[,/]|)?\s*(?:莊|B)\s*:?(\d)', s)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:莊|B)\s*:?(\d)\s*(?:[,/]|)?\s*(?:閒|P)\s*:?(\d)', s)
    if m: return (int(m.group(2)), int(m.group(1)))
    m = re.search(r'(?:PLAYER|P)\s*:?(\d)\s*(?:[,/]|)?\s*(?:BANKER|B)\s*:?(\d)', s)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'(?:BANKER|B)\s*:?(\d)\s*(?:[,/]|)?\s*(?:PLAYER|P)\s*:?(\d)', s)
    if m: return (int(m.group(2)), int(m.group(1)))
    return None

# ========= 試用 / 營運 =========
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "")

SESS: Dict[str, Dict[str, object]] = {}
def _init_user(uid:str):
    now = int(time.time())
    SESS[uid] = {
        "bankroll": 0, 
        "seq": [], 
        "trial_start": now, 
        "premium": False, 
        "last_pts_text": None
    }

def validate_activation_code(code: str) -> bool:
    return bool(ADMIN_ACTIVATION_SECRET) and bool(code) and (code == ADMIN_ACTIVATION_SECRET)

def trial_guard(uid:str) -> Optional[str]:
    sess = SESS.get(uid) or {}
    if sess.get("premium", False): return None
    now = int(time.time()); start = int(sess.get("trial_start", now))
    if (now - start) // 60 >= TRIAL_MINUTES:
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 輸入：開通 你的密碼"
    return None

# ========= 引擎：Outcome PF（RB-Exact 前向） =========
SEED = int(os.getenv("SEED","42"))
DECKS = int(os.getenv("DECKS","8"))

from bgs.pfilter import OutcomePF
PF_N         = int(os.getenv("PF_N", "200"))
PF_UPD_SIMS  = int(os.getenv("PF_UPD_SIMS", "80"))   # backend=mc 用；exact 不用
PF_PRED_SIMS = int(os.getenv("PF_PRED_SIMS", "0"))   # backend=mc 用；exact 不用
PF_RESAMPLE  = float(os.getenv("PF_RESAMPLE", "0.5"))
PF_DIR_EPS   = float(os.getenv("PF_DIR_EPS", "0.002"))
PF_BACKEND   = os.getenv("PF_BACKEND", "exact").lower()

PF = OutcomePF(
    decks=DECKS, seed=SEED, n_particles=PF_N,
    sims_lik=max(1, PF_UPD_SIMS), resample_thr=PF_RESAMPLE,
    backend=PF_BACKEND, dirichlet_eps=PF_DIR_EPS
)

# ========= 決策（僅莊/閒） =========
EDGE_ENTER   = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY    = env_flag("USE_KELLY", 1)
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR", "0.25"))
MAX_BET_PCT  = float(os.getenv("MAX_BET_PCT", "0.015"))

LOG_DIR   = os.getenv("LOG_DIR", "logs")
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
PRED_CSV  = os.path.join(LOG_DIR, "predictions.csv")
if not os.path.exists(PRED_CSV):
    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["ts","version","hands","pB","pP","pT","choice","edge","bet_pct","bankroll","bet_amt","engine","reason"])

def banker_ev(pB, pP): return 0.95*pB - pP
def player_ev(pB, pP): return pP - pB

def kelly_fraction(p_win: float, payoff: float):
    q = 1.0 - p_win
    edge = p_win*payoff - q
    return max(0.0, edge / payoff)

def bet_amount(bankroll:int, pct:float) -> int:
    if not bankroll or bankroll<=0 or pct<=0: return 0
    return int(round(bankroll*pct))

def decide_only_bp(prob):
    pB, pP, _ = float(prob[0]), float(prob[1]), float(prob[2])
    evB, evP = banker_ev(pB, pP), player_ev(pB, pP)
    side = 0 if evB > evP else 1
    edge_prob = abs(pB - pP)
    final_edge = max(edge_prob, abs(evB - evP))
    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, f"⚪ 優勢不足（門檻 {EDGE_ENTER:.2f}）")
    if USE_KELLY:
        f = KELLY_FACTOR * (kelly_fraction(pB, 0.95) if side==0 else kelly_fraction(pP, 1.0))
        bet_pct = min(MAX_BET_PCT, float(max(0.0, f)))
        reason = "🧠 OutcomePF（RB-Exact）｜📐 ¼-Kelly"
    else:
        if final_edge >= 0.10: bet_pct = 0.25
        elif final_edge >= 0.07: bet_pct = 0.15
        elif final_edge >= 0.04: bet_pct = 0.10
        else: bet_pct = 0.05
        reason = "🧠 OutcomePF（RB-Exact）｜🪜 階梯式配注"
    return (INV[side], final_edge, bet_pct, reason)

def log_prediction(hands:int, p, choice:str, edge:float, bankroll:int, bet_pct:float, engine:str, reason:str):
    try:
        bet_amt = bet_amount(bankroll, bet_pct)
        with open(PRED_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([int(time.time()), VERSION, hands, float(p[0]), float(p[1]), float(p[2]), choice, float(edge), float(bet_pct), int(bankroll), int(bet_amt), engine, reason])
    except Exception as e:
        log.warning("log_prediction failed: %s", e)

# ========= 輸出格式（完全照你的卡片＋最後一行金額） =========
def format_output_strict(prob, choice, last_pts_text: Optional[str],
                         bet_amt: Optional[int] = None):
    """
    讀取完成
    上局結果: 閒 X 莊 Y
    開始分析下局....

    【預測結果】
    閒：PP.PP%
    莊：BB.BB%
    本次預測結果：莊/閒
    建議下注：<金額>
    """
    b_pct_txt = f"{prob[0]*100:.2f}%"
    p_pct_txt = f"{prob[1]*100:.2f}%"

    header = []
    if last_pts_text:
        header = ["讀取完成", last_pts_text, "開始分析下局....", ""]
    else:
        # 若無上局點數字串，但有輸贏，仍維持同樣版頭（讓格式不變）
        header = ["讀取完成", "開始分析下局....", ""] if True else []

    lines = [
        "【預測結果】",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"本次預測結果：{choice if choice!='觀望' else '觀'}",
        f"建議下注：{(bet_amt or 0):,}"
    ]
    return "\n".join(header + lines)

# ========= 健康檢查 =========
@app.get("/")
def root():
    return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/healthz")
def healthz():
    return jsonify(ok=True, ts=time.time(), version=VERSION, path="/healthz"), 200

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION, path="/health"), 200

# ========= API =========
@app.post("/update-outcome")
def update_outcome_api():
    data = request.get_json(silent=True) or {}
    o = str(data.get("outcome","")).strip().upper()
    if o in ("B","莊","0"): PF.update_outcome(0)
    elif o in ("P","閒","1"): PF.update_outcome(1)
    elif o in ("T","和","TIE","DRAW","2"): PF.update_outcome(2)
    else: return jsonify(ok=False, msg="outcome 必須是 B/P/T 或 莊/閒/和"), 400
    return jsonify(ok=True), 200

@app.post("/predict")
def predict_api():
    data = request.get_json(silent=True) or {}
    bankroll = int(float(data.get("bankroll") or 0))
    lp = data.get("last_pts")
    lo = str(data.get("last_outcome","")).strip().upper()

    last_text = None
    if lp:
        pts = parse_last_hand_points(lp)
        if pts is not None:
            last_outcome = 1 if int(pts[0]) > int(pts[1]) else (0 if int(pts[1]) > int(pts[0]) else 2)
            PF.update_outcome(last_outcome)
            last_text = f"上局結果: 閒 {int(pts[0])} 莊 {int(pts[1])}"
        else:
            if re.search(r'(?:和|TIE|DRAW)\b', str(lp).upper()):
                PF.update_outcome(2); last_text = "上局結果: 和局"

    if not last_text and lo:
        if lo in ("B","莊","0"): PF.update_outcome(0); last_text = "上局結果: 莊勝"
        elif lo in ("P","閒","1"): PF.update_outcome(1); last_text = "上局結果: 閒勝"
        elif lo in ("T","和","TIE","DRAW","2"): PF.update_outcome(2); last_text = "上局結果: 和局"

    p = PF.predict(sims_per_particle=max(0, PF_PRED_SIMS))
    choice, edge, bet_pct, reason = decide_only_bp(p)
    amt = bet_amount(bankroll, bet_pct)

    msg = format_output_strict(p, choice, last_text, bet_amt=amt)

    # 紀錄
    engine_note = f"PF-{PF.backend}"
    log_prediction(-1, p, choice, edge, bankroll, bet_pct, engine_note, reason)

    return jsonify(
        message=msg, version=VERSION,
        suggestion=choice,
        bet_pct=float(bet_pct), bet_amount=amt,
        probabilities={"banker": float(p[0]), "player": float(p[1])}
    ), 200

# ========= LINE Webhook（加入＆結束都有「開始分析」「結束分析」按鈕）=========
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None; line_handler = None
if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import MessageEvent, TextMessage, FollowEvent, TextSendMessage, QuickReply, QuickReplyButton, MessageAction
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        def quick_menu():
            try:
                return QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="開始分析", text="開始分析")),
                    QuickReplyButton(action=MessageAction(label="結束分析", text="結束分析")),
                    QuickReplyButton(action=MessageAction(label="報莊勝", text="B")),
                    QuickReplyButton(action=MessageAction(label="報閒勝", text="P")),
                    QuickReplyButton(action=MessageAction(label="報和局", text="T")),
                    QuickReplyButton(action=MessageAction(label="設定本金 5000", text="本金 5000")),
                ])
            except Exception:
                return None

        def safe_reply(token: str, text: str, uid: Optional[str] = None):
            try:
                line_api.reply_message(token, TextSendMessage(text=text, quick_reply=quick_menu()))
            except Exception as e:
                log.warning("[LINE] reply failed, try push: %s", e)
                if uid:
                    try:
                        line_api.push_message(uid, TextSendMessage(text=text, quick_reply=quick_menu()))
                    except Exception as e2:
                        log.error("[LINE] push failed: %s", e2)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            uid = event.source.user_id; _init_user(uid)
            safe_reply(event.reply_token,
                "歡迎加入！\n"
                "輸入『本金 5000』設定本金，回報上一局 B/P/T，然後按『開始分析』即可。"
            , uid)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            uid = event.source.user_id; text = (event.message.text or "").strip()
            if uid not in SESS: _init_user(uid)

            # 試用檢查
            guard = trial_guard(uid)
            if guard: 
                safe_reply(event.reply_token, guard, uid); return

            up = text.upper()

            # 設定本金（支援「本金 5000」或直接數字）
            m = re.match(r"^(?:本金|BAL|BANKROLL)\s+(\d+)$", text.strip(), flags=re.IGNORECASE)
            if m:
                SESS[uid]["bankroll"] = int(m.group(1))
                safe_reply(event.reply_token, f"👍 已設定本金：{int(m.group(1)):,}", uid); return
            if text.isdigit():
                SESS[uid]["bankroll"] = int(text)
                safe_reply(event.reply_token, f"👍 已設定本金：{int(text):,}", uid); return

            # 回報輸贏
            if up in ("B","莊","BANKER"):
                PF.update_outcome(0); SESS[uid]["last_pts_text"] = "上局結果: 莊勝"
                safe_reply(event.reply_token, "📝 已記錄上一局：莊勝", uid); return
            if up in ("P","閒","PLAYER"):
                PF.update_outcome(1); SESS[uid]["last_pts_text"] = "上局結果: 閒勝"
                safe_reply(event.reply_token, "📝 已記錄上一局：閒勝", uid); return
            if up in ("T","和","TIE","DRAW"):
                PF.update_outcome(2); SESS[uid]["last_pts_text"] = "上局結果: 和局"
                safe_reply(event.reply_token, "📝 已記錄上一局：和局", uid); return

            # 解析點數字串（可選）
            pts = parse_last_hand_points(text)
            if pts is not None:
                last_outcome = 1 if int(pts[0]) > int(pts[1]) else (0 if int(pts[1]) > int(pts[0]) else 2)
                PF.update_outcome(last_outcome)
                SESS[uid]["last_pts_text"] = f"上局結果: 閒 {int(pts[0])} 莊 {int(pts[1])}"
                safe_reply(event.reply_token, SESS[uid]["last_pts_text"], uid); return

            # 「開始分析」或「開始分析 23」都觸發預測
            if ("開始分析" == text) or re.match(r"^開始分析\s*\d*$", text):
                p = PF.predict()
                choice, edge, bet_pct, reason = decide_only_bp(p)
                bankroll_now = int(SESS[uid].get("bankroll", 0))
                msg = format_output_strict(p, choice, SESS[uid].get("last_pts_text"), bet_amt=bet_amount(bankroll_now, bet_pct))
                safe_reply(event.reply_token, msg, uid); return

            if up in ("結束分析","清空","RESET"):
                SESS[uid] = {"bankroll": 0, "seq": [], "trial_start": int(time.time()), "premium": SESS.get(uid,{}).get("premium", False), "last_pts_text": None}
                safe_reply(event.reply_token, "🧹 已清空。按『開始分析』可重新進行分析。", uid); return

            # 其餘情況：顯示含「開始分析」「結束分析」按鈕的提示
            safe_reply(event.reply_token, "請先設定本金並回報上一局（B/P/T），然後按『開始分析』。", uid)

        @app.post("/line-webhook")
        def line_webhook():
            signature = request.headers.get("X-Line-Signature", ""); body = request.get_data(as_text=True)
            try: 
                line_handler.handle(body, signature)
            except InvalidSignatureError: 
                abort(400, "Invalid signature")
            return "OK", 200
    except Exception as e:
        log.warning("LINE not fully configured: %s", e)

# ========= 本地啟動 =========
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
