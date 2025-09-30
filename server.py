# server.py — 純算牌 + 粒子濾波（ONLY 莊/閒建議｜EV含抽水｜¼-Kelly｜試用制｜卡片輸出｜支援和局回報｜加入導引清單｜FSM修正）
# Author: 親愛的 x GPT-5 Thinking

import os, logging, time, csv, pathlib, re
from typing import List, Optional, Dict
import numpy as np
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

# ==== 基本設定 ====
VERSION = "bgs-deplete-pf-2025-09-30-fsm"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

app = Flask(__name__)
CORS(app)

# ---- 旗標讀取 ----
def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try: return 1 if int(float(v)) != 0 else 0
    except: return 1 if default else 0

# ==== 牌路/顯示 ====
MAP = {"B":0, "P":1, "T":2, "莊":0, "閒":1, "和":2}
INV = {0:"莊", 1:"閒", 2:"和"}

def parse_history(s: str) -> List[int]:
    s = (s or "").strip().upper()
    if not s: return []
    s = s.replace("﹐","，").replace("，"," ").replace("、"," ").replace("\u3000"," ")
    toks = s.split()
    seq = list(s) if (len(toks) == 1 and len(s) <= 12) else toks
    out = []
    for ch in seq:
        ch = ch.strip().upper()
        if ch in MAP: out.append(MAP[ch])
    return out

# 解析「上局結果：閒6 莊8 / 和9 / TIE / DRAW / 兩位數 65」等；回 (P_total, B_total) 或 None
def parse_last_hand_points(text: str):
    if not text: return None
    # 先處理兩位數（含空白） ex: "65" -> (6,5)
    m2 = re.fullmatch(r"\s*(\d)\s*(\d)\s*", text)
    if m2:
        return (int(m2.group(1)), int(m2.group(2)))

    s = text.strip().upper().replace("：", ":")
    s = re.sub(r"\s+", "", s)

    # 明確和局：和9 / TIE9 / DRAW9 → 視為 P_total=B_total=9
    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:和|TIE|DRAW)\s*:?(\d)', s)
    if m:
        d = int(m.group(1)); return (d, d)

    # 單獨和（無點數）：不扣牌，但後續照常預測
    if re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:和|TIE|DRAW)\b', s):
        return None

    # 一般格式（帶 P/B 或 中英）
    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:閒|P)\s*:?(\d)\s*(?:[,/]|)?\s*(?:莊|B)\s*:?(\d)', s)
    if m: return (int(m.group(1)), int(m.group(2)))

    m = re.search(r'(?:上局結果|上局|LAST|PREV)?[:]*\s*(?:莊|B)\s*:?(\d)\s*(?:[,/]|)?\s*(?:閒|P)\s*:?(\d)', s)
    if m: return (int(m.group(2)), int(m.group(1)))

    m = re.search(r'(?:PLAYER|P)\s*:?(\d)\s*(?:[,/]|)?\s*(?:BANKER|B)\s*:?(\d)', s)
    if m: return (int(m.group(1)), int(m.group(2)))

    m = re.search(r'(?:BANKER|B)\s*:?(\d)\s*(?:[,/]|)?\s*(?:PLAYER|P)\s*:?(\d)', s)
    if m: return (int(m.group(2)), int(m.group(1)))
    return None

# ==== 試用 / 營運 ====
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
        "stage": "hall",        # FSM：hall -> table -> bankroll -> points
        "hall_name": None,
        "hall_code": None,
        "table": None,
        "last_pts_text": None,
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

# ====== 加入導引（符合你截圖） ======
def steps_menu_text():
    halls = [
        "1. WM", "2. PM", "3. DG", "4. SA", "5. KU",
        "6. 歐博/卡利", "7. KG", "8. 金利", "9. 名人", "10. MT真人"
    ]
    return (
        "👋 歡迎使用 BGS AI 系統！\n"
        "【使用步驟】\n"
        "1️⃣ 選擇館別（輸入 1~10）\n"
        "2️⃣ 輸入桌號（例：DG01）\n"
        "3️⃣ 輸入本金（例：5000）\n"
        "4️⃣ 每局回報點數（例：65 / 和 / 閒6 莊5）\n"
        "💾 試用剩餘：永久\n\n"
        "【請選擇遊戲館別】\n" + "\n".join(halls) + "\n(請直接輸入數字1-10)"
    )

HALL_MAP = {
    "1": ("WM", "WM"), "2": ("PM", "PM"), "3": ("DG", "DG"), "4": ("SA", "SA"),
    "5": ("KU", "KU"), "6": ("歐博/卡利", "OB"), "7": ("KG", "KG"),
    "8": ("金利", "JL"), "9": ("名人", "MR"), "10": ("MT真人", "MT")
}
TABLE_PATTERN = re.compile(r"^[A-Z]{2}\d{2}$")  # 例：DG01

# ==== 算牌引擎 ====
from bgs.deplete import DepleteMC
from bgs.pfilter import OutcomePF

SEED = int(os.getenv("SEED","42"))
DEPL_DECKS  = int(os.getenv("DEPL_DECKS", "8"))
DEPL_SIMS   = int(os.getenv("DEPL_SIMS", "30000"))

# 粒子濾波（只用輸贏也能學）
PF_N        = int(os.getenv("PF_N", "200"))
PF_UPD_SIMS = int(os.getenv("PF_UPD_SIMS", "80"))
PF_PRED_SIMS= int(os.getenv("PF_PRED_SIMS", "220"))
PF_RESAMPLE = float(os.getenv("PF_RESAMPLE", "0.5"))
PF_DIR_ALPHA= float(os.getenv("PF_DIR_ALPHA", "0.8"))  # Dirichlet 先驗強度
PF_USE_EXACT= int(os.getenv("PF_USE_EXACT", "0"))      # 0=MC 前向；1=Exact-lite 前向

DEPL = DepleteMC(decks=DEPL_DECKS, seed=SEED)
PF   = OutcomePF(
        decks=DEPL_DECKS,
        seed=SEED,
        n_particles=PF_N,
        sims_lik=PF_UPD_SIMS,
        resample_thr=PF_RESAMPLE,          # 正確參數
        dirichlet_alpha=PF_DIR_ALPHA,
        use_exact=bool(PF_USE_EXACT)
      )

# ==== 決策（僅莊/閒）====
EDGE_ENTER  = float(os.getenv("EDGE_ENTER", "0.03"))  # 觀望門檻
USE_KELLY   = env_flag("USE_KELLY", 1)
KELLY_FACTOR= float(os.getenv("KELLY_FACTOR", "0.25"))  # ¼-Kelly
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.015"))  # 單注上限 1.5%

LOG_DIR     = os.getenv("LOG_DIR", "logs")
pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
PRED_CSV    = os.path.join(LOG_DIR, "predictions.csv")
if not os.path.exists(PRED_CSV):
    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["ts","version","hands","pB","pP","pT","choice","edge","bet_pct","bankroll","bet_amt","engine","reason"])

# ---- EV / Kelly ----
def banker_ev(pB, pP):  # tie 退回
    return 0.95*pB - pP
def player_ev(pB, pP):
    return pP - pB
def kelly_fraction(p_win: float, payoff: float):
    q = 1.0 - p_win
    edge = p_win*payoff - q
    return max(0.0, edge / payoff)
def bet_amount(bankroll:int, pct:float) -> int:
    if not bankroll or bankroll<=0 or pct<=0: return 0
    return int(round(bankroll*pct))

def decide_only_bp(prob):
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    evB, evP = banker_ev(pB, pP), player_ev(pB, pP)
    side = 0 if evB > evP else 1
    edge_prob = abs(pB - pP)
    final_edge = max(edge_prob, abs(evB - evP))
    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, f"⚪ 優勢不足（門檻 {EDGE_ENTER:.2f}）")
    if USE_KELLY:
        f = KELLY_FACTOR * (kelly_fraction(pB, 0.95) if side==0 else kelly_fraction(pP, 1.0))
        bet_pct = min(MAX_BET_PCT, float(max(0.0, f)))
        reason = "🧠 純算牌｜📐 ¼-Kelly"
    else:
        if final_edge >= 0.10: bet_pct = 0.25
        elif final_edge >= 0.07: bet_pct = 0.15
        elif final_edge >= 0.04: bet_pct = 0.10
        else: bet_pct = 0.05
        reason = "🧠 純算牌｜🪜 階梯式配注"
    return (INV[side], final_edge, bet_pct, reason)

# ===== 卡片輸出 =====
def format_card_output(prob, choice, last_pts_text: Optional[str]):
    b_pct = f"{prob[0]*100:.2f}%"
    p_pct = f"{prob[1]*100:.2f}%"
    header = []
    if last_pts_text:
        header = ["讀取完成", last_pts_text, "開始平衡分析下局....", ""]
    block = [
        "【預測結果】",
        f"閒：{p_pct}",
        f"莊：{b_pct}",
        f"本次預測結果：{choice if choice!='觀望' else '觀'}"
    ]
    return "\n".join(header + block)

# ==== 健康檢查 ====
@app.get("/")
def root(): return f"✅ BGS Deplete+PF Server OK ({VERSION})", 200

@app.get("/healthz")
def healthz(): return jsonify(ok=True, ts=time.time(), version=VERSION), 200

# ==== API：/update-hand（有點數時可用；只有輸贏不必呼叫）====
@app.post("/update-hand")
def update_hand_api():
    obs = request.get_json(silent=True) or {}
    try:
        if "p_total" in obs and "b_total" in obs:
            DEPL.update_hand(obs)
            last_outcome = 1 if int(obs["p_total"]) > int(obs["b_total"]) else (0 if int(obs["b_total"]) > int(obs["p_total"]) else 2)
            PF.update_outcome(last_outcome)
        return jsonify(ok=True), 200
    except Exception as e:
        log.warning("update_hand failed: %s", e)
        return jsonify(ok=False, msg=str(e)), 400

# ==== API：/predict（只回傳莊/閒建議；可回卡片）====
@app.post("/predict")
def predict_api():
    data = request.get_json(silent=True) or {}
    bankroll = int(float(data.get("bankroll") or 0))
    seq = parse_history(str(data.get("history","")))
    lp = data.get("last_pts")
    pts = None
    engine_note = None

    # 先處理 last_pts（可能是點數；也可能是「和」）
    last_text = None
    if lp:
        pts = parse_last_hand_points(lp)
        if pts is not None:
            try:
                DEPL.update_hand({"p_total": int(pts[0]), "b_total": int(pts[1]), "trials": 400})
                engine_note = "Deplete"
                last_text = f"上局結果: 閒 {int(pts[0])} 莊 {int(pts[1])}"
                PF.update_outcome(1 if int(pts[0])>int(pts[1]) else (0 if int(pts[1])>int(pts[0]) else 2))
            except Exception as e:
                log.warning("deplete update in /predict failed: %s", e)
        else:
            if re.search(r'(?:和|TIE|DRAW)\b', str(lp).upper()):
                PF.update_outcome(2)
                last_text = "上局結果: 和局"

    # 也可直接傳 last_outcome: "B"/"P"/"T"
    if "last_outcome" in data:
        o = str(data["last_outcome"]).strip().upper()
        if o in ("B","莊","0"): PF.update_outcome(0); last_text = "上局結果: 莊勝"
        elif o in ("P","閒","1"): PF.update_outcome(1); last_text = "上局結果: 閒勝"
        elif o in ("T","和","2"): PF.update_outcome(2); last_text = "上局結果: 和局"

    # 取得概率
    p_depl = None; p_pf = None
    try: p_depl = DEPL.predict(sims=DEPL_SIMS)
    except Exception as e: log.warning("deplete predict failed: %s", e)
    try: p_pf   = PF.predict(sims_per_particle=PF_PRED_SIMS)
    except Exception as e: log.warning("pf predict failed: %s", e)

    if (p_depl is not None) and (p_pf is not None):
        if pts is not None: w_depl, w_pf = 0.7, 0.3; engine_note = "Mix(Deplete↑)"
        else:               w_depl, w_pf = 0.3, 0.7; engine_note = "Mix(PF↑)"
        p = w_depl * p_depl + w_pf * p_pf
        p[2] = np.clip(p[2], 0.06, 0.20); p = p / p.sum()
    elif p_depl is not None:
        p = p_depl; engine_note = "Deplete"
    elif p_pf is not None:
        p = p_pf; engine_note = "PF"
    else:
        p = np.array([0.45,0.45,0.10], dtype=np.float32); engine_note = "Fallback"

    choice, edge, bet_pct, reason = decide_only_bp(p)
    amt = bet_amount(bankroll, bet_pct)

    style = str(data.get("style","")).lower()
    if style == "card":
        msg = format_card_output(p, choice, last_text)
    else:
        b_pct, p_pct = int(round(100*p[0])), int(round(100*p[1]))
        evB = banker_ev(float(p[0]), float(p[1])); evP = player_ev(float(p[0]), float(p[1]))
        msg = (
            f"🎯 下一局建議：{choice}\n"
            f"💰 建議注額：{amt:,}\n"
            f"📊 機率｜莊 {b_pct}%｜閒 {p_pct}%\n"
            f"📐 EV（抽水後）｜莊 {evB:.3f}｜閒 {evP:.3f}\n"
            f"🧭 {reason}｜引擎：{engine_note}"
        )

    # 記錄
    try:
        bet_amt = bet_amount(bankroll, bet_pct)
        with open(PRED_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([int(time.time()), VERSION, len(seq), float(p[0]), float(p[1]), float(p[2]),
                                    choice, float(edge), float(bet_pct), int(bankroll), int(bet_amt),
                                    engine_note or "NA", reason])
    except Exception as e:
        log.warning("log_prediction failed: %s", e)

    return jsonify(
        message=msg, version=VERSION, hands=len(seq),
        suggestion=choice, bet_pct=float(bet_pct), bet_amount=amt,
        probabilities={"banker": float(p[0]), "player": float(p[1])}
    ), 200

# ==== （可選）LINE Webhook：未設 TOKEN 也能啟動 API ====
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None; line_handler = None
if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import MessageEvent, TextMessage, FollowEvent, TextSendMessage

        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            uid = event.source.user_id; _init_user(uid)
            try:
                line_api.reply_message(event.reply_token, [
                    TextSendMessage(text="✅ 已重設流程，請選擇館別："),
                    TextSendMessage(text=steps_menu_text())
                ])
            except Exception as e:
                log.warning("follow reply failed: %s", e)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            uid = event.source.user_id; text = (event.message.text or "").strip()
            if uid not in SESS: _init_user(uid)
            sess = SESS[uid]
            guard = trial_guard(uid)
            if guard:
                line_api.reply_message(event.reply_token, TextSendMessage(text=guard)); return

            # 允許隨時重設導引
            if text in ["遊戲設定", "重設流程", "reset", "清空", "結束分析"]:
                prem = sess.get("premium", False)
                _init_user(uid); SESS[uid]["premium"] = prem
                try:
                    line_api.reply_message(event.reply_token, [
                        TextSendMessage(text="✅ 已重設流程，請選擇館別："),
                        TextSendMessage(text=steps_menu_text())
                    ])
                except Exception as e:
                    log.warning("reset reply failed: %s", e)
                return

            # FSM：hall → table → bankroll → points
            stage = sess.get("stage", "hall")

            # 步驟 1：選館（輸入 1~10）
            if stage == "hall":
                if re.fullmatch(r"10|[1-9]", text):
                    name, code = HALL_MAP[text]
                    sess["hall_name"] = name; sess["hall_code"] = code; sess["stage"] = "table"
                    line_api.reply_message(event.reply_token, TextSendMessage(text=f"✅ 已選【{name}】\n請輸入桌號（例：DG01，格式：2字母+2數字）"))
                    return
                else:
                    line_api.reply_message(event.reply_token, TextSendMessage(text="請輸入 1~10 選擇館別")) ; return

            # 步驟 2：桌號（兩字母+兩數字）
            if stage == "table":
                t = text.upper()
                if TABLE_PATTERN.fullmatch(t):
                    sess["table"] = t; sess["stage"] = "bankroll"
                    line_api.reply_message(event.reply_token, TextSendMessage(text=f"✅ 已設桌號【{t}】\n請輸入您的本金（例：5000）"))
                    return
                else:
                    line_api.reply_message(event.reply_token, TextSendMessage(text="桌號格式錯誤，請輸入 2 字母 + 2 數字，例如 DG01")) ; return

            # 步驟 3：本金（只在此階段接受純數字為本金）
            if stage == "bankroll":
                if text.isdigit() and int(text) > 0:
                    sess["bankroll"] = int(text); sess["stage"] = "points"
                    line_api.reply_message(event.reply_token, TextSendMessage(
                        text=f"👍 已設定本金：{int(text):,}\n請輸入上一局點數開始分析（例如 65 / 閒6 莊5 / 和）"
                    ))
                    return
                else:
                    line_api.reply_message(event.reply_token, TextSendMessage(text="請輸入純數字本金，例如 5000")) ; return

            # 步驟 4：points（此階段的純數字不會被當作本金）
            if stage == "points":
                # 先嘗試解析點數/和局
                pts = parse_last_hand_points(text)
                if pts is not None or re.search(r'(?:和|TIE|DRAW)\b', text.upper()):
                    if pts is not None:
                        p_total, b_total = pts
                        try:
                            DEPL.update_hand({"p_total": p_total, "b_total": b_total, "trials": 400})
                            last_outcome = 1 if p_total > b_total else (0 if b_total > p_total else 2)
                            PF.update_outcome(last_outcome)
                        except Exception as e:
                            log.warning("deplete update(line) failed: %s", e)
                        sess.setdefault("seq", []).append(last_outcome)
                        sess["last_pts_text"] = f"上局結果: 閒 {p_total} 莊 {b_total}"
                        line_api.reply_message(event.reply_token, TextSendMessage(
                            text="讀取完成\n" + sess["last_pts_text"] + "\n開始平衡分析下局...."
                        ))
                        return
                    else:
                        PF.update_outcome(2)
                        sess.setdefault("seq", []).append(2)
                        sess["last_pts_text"] = "上局結果: 和局"
                        line_api.reply_message(event.reply_token, TextSendMessage(
                            text="讀取完成\n上局結果: 和局\n開始平衡分析下局...."
                        ))
                        return

                # 也接受單字「莊/閒/和」純勝負
                single = text.strip().upper()
                if single in ("B","莊","BANKER"):
                    PF.update_outcome(0); sess.setdefault("seq", []).append(0); sess["last_pts_text"]="上局結果: 莊勝"
                    line_api.reply_message(event.reply_token, TextSendMessage(text="讀取完成\n上局結果: 莊勝\n開始平衡分析下局....")); return
                if single in ("P","閒","PLAYER"):
                    PF.update_outcome(1); sess.setdefault("seq", []).append(1); sess["last_pts_text"]="上局結果: 閒勝"
                    line_api.reply_message(event.reply_token, TextSendMessage(text="讀取完成\n上局結果: 閒勝\n開始平衡分析下局....")); return
                if single in ("T","和","TIE","DRAW"):
                    PF.update_outcome(2); sess.setdefault("seq", []).append(2); sess["last_pts_text"]="上局結果: 和局"
                    line_api.reply_message(event.reply_token, TextSendMessage(text="讀取完成\n上局結果: 和局\n開始平衡分析下局....")); return

                # 觸發分析（維持原預測邏輯 & 卡片輸出）
                if ("開始分析" in text) or (text in ["分析","開始","GO","go"]):
                    p_depl = None; p_pf = None
                    try: p_depl = DEPL.predict(sims=DEPL_SIMS)
                    except Exception as e: log.warning("deplete predict failed: %s", e)
                    try: p_pf   = PF.predict(sims_per_particle=PF_PRED_SIMS)
                    except Exception as e: log.warning("pf predict failed: %s", e)

                    if (p_depl is not None) and (p_pf is not None):
                        p = 0.5 * p_depl + 0.5 * p_pf
                        p[2] = np.clip(p[2], 0.06, 0.20); p = p / p.sum(); engine_note="Mix"
                    elif p_depl is not None:
                        p = p_depl; engine_note = "Deplete"
                    elif p_pf is not None:
                        p = p_pf; engine_note = "PF"
                    else:
                        p = np.array([0.45,0.45,0.10], dtype=np.float32); engine_note = "Fallback"

                    choice, edge, bet_pct, reason = decide_only_bp(p)
                    msg = format_card_output(p, choice, sess.get("last_pts_text"))
                    line_api.reply_message(event.reply_token, TextSendMessage(text=msg))
                    return

                # 其餘情況：提示導引
                line_api.reply_message(event.reply_token, TextSendMessage(
                    text="🧭 指令：回報點數（65/和/閒6 莊5）或輸入『開始分析』"
                ))
                return

            # 萬一沒有 stage：重設
            _init_user(uid)
            line_api.reply_message(event.reply_token, TextSendMessage(text="狀態已重設，請輸入 1~10 選擇館別。"))
            return

        @app.post("/line-webhook")
        def line_webhook():
            signature = request.headers.get("X-Line-Signature", ""); body = request.get_data(as_text=True)
            try: line_handler.handle(body, signature)
            except InvalidSignatureError: abort(400, "Invalid signature")
            return "OK", 200

    except Exception as e:
        log.warning("LINE not fully configured: %s", e)

# ==== 本地啟動 ====
if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    log.info("Starting %s on port %s", VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
