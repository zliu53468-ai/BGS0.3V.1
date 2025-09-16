# server.py — 連續模式：輸入點數即自動預測（免按「開始分析」）
# Author: 親愛的 x GPT-5 Thinking
# Version: bgs-pf-continuous-2025-09-17-ka7 (fixed decorators)

import os
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import redis
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

VERSION = "bgs-pf-continuous-2025-09-17-ka7"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Redis ----------
REDIS_URL = os.getenv("REDIS_URL")
redis_client: Optional[redis.Redis] = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Successfully connected to Redis.")
    except Exception as e:
        log.error("Failed to connect to Redis: %s. Using in-memory session.", e)
else:
    log.warning("REDIS_URL not set. Using in-memory session.")

SESS_FALLBACK: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE_SECONDS = 3600
DEDUPE_TTL = 60

def _rget(k: str) -> Optional[str]:
    try:
        return redis_client.get(k) if redis_client else None
    except Exception as e:
        log.warning("[Redis] GET err: %s", e)
        return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if redis_client:
            redis_client.set(k, v, ex=ex)
    except Exception as e:
        log.warning("[Redis] SET err: %s", e)

def _rsetnx(k: str, v: str, ex: int) -> bool:
    try:
        if redis_client:
            return bool(redis_client.set(k, v, ex=ex, nx=True))
        # fallback
        if k in SESS_FALLBACK:
            return False
        SESS_FALLBACK[k] = {"v": v, "exp": time.time() + ex}
        return True
    except Exception as e:
        log.warning("[Redis] SETNX err: %s", e)
        return True

def get_session(uid: str) -> Dict[str, Any]:
    if redis_client:
        j = _rget(f"bgs_session:{uid}")
        if j:
            try:
                return json.loads(j)
            except Exception:
                pass
    else:
        now = time.time()
        for k in list(SESS_FALLBACK.keys()):
            v = SESS_FALLBACK.get(k)
            if isinstance(v, dict) and v.get("exp") and v["exp"] < now:
                del SESS_FALLBACK[k]
        if uid in SESS_FALLBACK and "phase" in SESS_FALLBACK[uid]:
            return SESS_FALLBACK[uid]
    nowi = int(time.time())
    return {
        "bankroll": 0, "trial_start": nowi, "premium": False,
        "phase": "choose_game", "game": None, "table": None,
        "last_pts_text": None, "table_no": None,
    }

def save_session(uid: str, data: Dict[str, Any]):
    if redis_client:
        _rset(f"bgs_session:{uid}", json.dumps(data), ex=SESSION_EXPIRE_SECONDS)
    else:
        SESS_FALLBACK[uid] = data

def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None:
        return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return 1
    if v in ("0", "false", "f", "no", "n", "off"):
        return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except Exception:
        return 1 if default else 0

# ---------- 解析上局點數 ----------
INV = {0: "莊", 1: "閒", 2: "和"}

def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
    """
    回 (P_total, B_total)
    支援：
      - '47' / '4 7' / '4-7' / '4,7'
      - '閒4莊7' / 'P4 B7'（順序自動）
      - '開始分析47'（自動剝掉前綴）
      - '和' / 'TIE' / 'DRAW'（回(0,0)表示和）
    會清除全形數字、全形冒號、零寬/控制字元。
    """
    if not text:
        return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s)
    u = s.upper().strip()
    u = re.sub(r"^開始分析", "", u)

    m = re.search(r"(?:和|TIE|DRAW)\s*:?\s*(\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else (0, 0)

    m = re.search(r"(?:閒|P)\s*:?\s*(\d)\D+(?:莊|B)\s*:?\s*(\d)", u)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|B)\s*:?\s*(\d)\D+(?:閒|P)\s*:?\s*(\d)", u)
    if m:
        return (int(m.group(2)), int(m.group(1)))

    d = re.findall(r"\d", u)
    if len(d) >= 2:
        return (int(d[0]), int(d[1]))

    t = u.replace(" ", "")
    if t in ("B", "莊"):
        return (0, 1)
    if t in ("P", "閒"):
        return (1, 0)
    if t in ("T", "和"):
        return (0, 0)
    return None

# ---------- 試用/授權 ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "aaa8881688")

def validate_activation_code(code: str) -> bool:
    if not code:
        return False
    norm = str(code).replace("\u3000", " ").replace("：", ":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and (norm == ADMIN_ACTIVATION_SECRET)

def trial_left_minutes(sess: Dict[str, Any]) -> int:
    if sess.get("premium", False):
        return 9999
    now = int(time.time())
    used = (now - int(sess.get("trial_start", now))) // 60
    return max(0, TRIAL_MINUTES - used)

def trial_guard(sess: Dict[str, Any]) -> Optional[str]:
    if sess.get("premium", False):
        return None
    if trial_left_minutes(sess) <= 0:
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 在此輸入：開通 你的密碼"
    return None

try:
    log.info("Activation secret loaded? %s (len=%d)", bool(ADMIN_ACTIVATION_SECRET), len(ADMIN_ACTIVATION_SECRET))
except Exception:
    pass

# ---------- Outcome PF ----------
try:
    from bgs.pfilter import OutcomePF
    PF = OutcomePF(
        decks=int(os.getenv("DECKS", "8")),
        seed=int(os.getenv("SEED", "42")),
        n_particles=int(os.getenv("PF_N", "200")),
        sims_lik=max(1, int(os.getenv("PF_UPD_SIMS", "80"))),
        resample_thr=float(os.getenv("PF_RESAMPLE", "0.5")),
        backend=os.getenv("PF_BACKEND", "exact").lower(),
        dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.002")),
    )
except Exception as e:
    log.error("Could not import OutcomePF, using Dummy. err=%s", e)

    class DummyPF:
        def update_outcome(self, _):
            pass

        def predict(self, **_):
            return np.array([0.5, 0.49, 0.01])  # B, P, T

        @property
        def backend(self):
            return "dummy"

    PF = DummyPF()

# ---------- 決策 & 金額 ----------
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY = env_flag("USE_KELLY", 1)
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR", "0.25"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.015"))
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)  # 1=連續模式；0=舊流程

def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0:
        return 0
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
        if final_edge >= 0.10:
            bet_pct = 0.25
        elif final_edge >= 0.07:
            bet_pct = 0.15
        elif final_edge >= 0.04:
            bet_pct = 0.10
        else:
            bet_pct = 0.05
        reason = "階梯式配注"
    return (INV[side], final_edge, bet_pct, reason)

def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int, cont: bool) -> str:
    b_pct_txt = f"{prob[0] * 100:.2f}%"
    p_pct_txt = f"{prob[1] * 100:.2f}%"
    header = []
    if last_pts_text:
        header.append(last_pts_text)
    header.append("開始分析下局....")
    block = [
        "【預測結果】",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"本次預測結果：{choice if choice != '觀望' else '觀'}",
        f"建議下注：{bet_amt:,}",
    ]
    if cont:
        block.append("\n📌 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）")
    return "\n".join(header + [""] + block)

# ---------- 健康檢查 ----------
@app.get("/")
def root():
    ua = request.headers.get("User-Agent", "")
    if "UptimeRobot" in ua:
        return "OK", 200
    return f"✅ BGS PF Server OK ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION), 200

@app.get("/healthz")
def healthz():
    return jsonify(ok=True, ts=time.time(), version=VERSION), 200

# ---------- LINE ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None
line_handler = None

GAMES = {"1": "WM", "2": "PM", "3": "DG", "4": "SA", "5": "KU", "6": "歐博/卡利", "7": "KG", "8": "全利", "9": "名人", "10": "MT真人"}

def game_menu_text(left_min: int) -> str:
    lines = ["【請選擇遊戲館別】"] + [f"{k}. {GAMES[k]}" for k in sorted(GAMES.keys(), key=lambda x: int(x))]
    lines.append("「請直接輸入數字選擇」")
    lines.append(f"⏳ 試用剩餘 {left_min} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
    return "\n".join(lines)

def _quick_buttons():
    try:
        from linebot.models import QuickReply, QuickReplyButton, MessageAction
        items = [
            QuickReplyButton(action=MessageAction(label="遊戲設定 🎮", text="遊戲設定")),
            QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析")),
            QuickReplyButton(action=MessageAction(label="報莊勝 🅱️", text="B")),
            QuickReplyButton(action=MessageAction(label="報閒勝 🅿️", text="P")),
            QuickReplyButton(action=MessageAction(label="報和局 ⚪", text="T")),
        ]
        if CONTINUOUS_MODE == 0:
            items.insert(0, QuickReplyButton(action=MessageAction(label="開始分析 ▶️", text="開始分析")))
        return QuickReply(items=items)
    except Exception:
        return None

def _reply(token: str, text: str):
    from linebot.models import TextSendMessage
    try:
        line_api.reply_message(token, TextSendMessage(text=text, quick_reply=_quick_buttons()))
    except Exception as e:
        log.warning("[LINE] reply failed: %s", e)

def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id:
        return True
    return _rsetnx(f"dedupe:{event_id}", "1", DEDUPE_TTL)

def _handle_points_and_predict(sess: Dict[str, Any], p_pts: int, b_pts: int, reply_token: str):
    # 更新上一局
    if p_pts == b_pts:
        sess["last_pts_text"] = "上局結果: 和局"
        try:
            if int(os.getenv("SKIP_TIE_UPD", "0")) == 0:
                PF.update_outcome(2)
        except Exception as e:
            log.warning("PF tie update err: %s", e)
    else:
        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
        try:
            PF.update_outcome(1 if p_pts > b_pts else 0)
        except Exception as e:
            log.warning("PF update err: %s", e)

    # 直接預測
    sess["phase"] = "ready"
    p = PF.predict(sims_per_particle=max(0, int(os.getenv("PF_PRED_SIMS", "0"))))
    choice, edge, bet_pct, reason = decide_only_bp(p)
    bankroll_now = int(sess.get("bankroll", 0))
    msg = format_output_card(p, choice, sess.get("last_pts_text"), bet_amt=bet_amount(bankroll_now, bet_pct), cont=bool(CONTINUOUS_MODE))
    _reply(reply_token, msg)

    # 連續模式：保持在 await_pts，方便下一局直接輸入
    if CONTINUOUS_MODE:
        sess["phase"] = "await_pts"

if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import MessageEvent, TextMessage, FollowEvent

        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            if not _dedupe_event(getattr(event, "id", None)):
                return
            uid = event.source.user_id
            sess = get_session(uid)
            _reply(
                event.reply_token,
                "👋 歡迎！請輸入『遊戲設定』開始；已啟用連續模式，之後只需輸入點數（例：65 / 和 / 閒6莊5）即可自動預測。",
            )
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            if not _dedupe_event(getattr(event, "id", None)):
                return

            uid = event.source.user_id
            raw = (event.message.text or "")
            text = re.sub(r"\s+", " ", raw.replace("\u3000", " ")).strip()
            sess = get_session(uid)

            try:
                log.info("[LINE] uid=%s phase=%s text=%s", uid, sess.get("phase"), text)

                # --- 開通優先（避免被試用守門擋掉） ---
                up = text.upper()
                if up.startswith("開通") or up.startswith("ACTIVATE"):
                    after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                    ok = validate_activation_code(after)
                    sess["premium"] = bool(ok)
                    _reply(event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤")
                    save_session(uid, sess)
                    return

                # --- 試用守門 ---
                guard = trial_guard(sess)
                if guard:
                    _reply(event.reply_token, guard)
                    return

                # --- 連續模式/點數輸入（任何階段皆嘗試解析點數） ---
                pts = parse_last_hand_points(raw)
                if pts is not None:
                    if not sess.get("bankroll"):
                        _reply(event.reply_token, "請先完成『遊戲設定』與『本金設定』（例如輸入 5000），再回報點數。")
                        save_session(uid, sess)
                        return
                    _handle_points_and_predict(sess, int(pts[0]), int(pts[1]), event.reply_token)
                    save_session(uid, sess)
                    return

                # --- 遊戲設定入口 ---
                if up in ("遊戲設定", "設定", "SETUP", "GAME"):
                    sess["phase"] = "choose_game"
                    left = trial_left_minutes(sess)
                    menu = ["【請選擇遊戲館別】"]
                    for k in sorted(GAMES.keys(), key=lambda x: int(x)):
                        menu.append(f"{k}. {GAMES[k]}")
                    menu.append("「請直接輸入數字選擇」")
                    menu.append(f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
                    _reply(event.reply_token, "\n".join(menu))
                    save_session(uid, sess)
                    return

                phase = sess.get("phase", "choose_game")

                if phase == "choose_game":
                    if re.fullmatch(r"([1-9]|10)", text):
                        sess["game"] = GAMES[text]
                        sess["phase"] = "choose_table"
                        _reply(event.reply_token, f"✅ 已設定館別【{sess['game']}】\n請輸入桌號（例：DG01）")
                        save_session(uid, sess)
                        return

                elif phase == "choose_table":
                    t = re.sub(r"\s+", "", text).upper()
                    if re.fullmatch(r"[A-Z]{2}\d{2}", t):
                        sess["table"] = t
                        sess["phase"] = "await_bankroll"
                        _reply(event.reply_token, f"✅ 已設定桌號【{sess['table']}】\n請輸入您的本金（例：5000）")
                        save_session(uid, sess)
                        return
                    else:
                        _reply(event.reply_token, "❌ 桌號格式錯誤，請輸入 2 英文 + 2 數字（例：DG01）")
                        return

                elif phase == "await_bankroll":
                    if text.isdigit() and int(text) > 0:
                        sess["bankroll"] = int(text)
                        sess["phase"] = "await_pts"
                        _reply(
                            event.reply_token,
                            f"👍 已設定本金：{sess['bankroll']:,}\n📌 連續模式開啟：現在直接輸入上局點數（例：65 / 和 / 閒6莊5）即可自動預測。",
                        )
                        save_session(uid, sess)
                        return
                    else:
                        _reply(event.reply_token, "❌ 金額格式錯誤，請直接輸入正整數（例：5000）")
                        return

                # 舊流程的『開始分析XY』仍兼容
                norm = raw.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
                norm = re.sub(r"\s+", "", norm)
                m_ka = re.fullmatch(r"開始分析(\d)(\d)", norm)
                if m_ka and sess.get("bankroll"):
                    _handle_points_and_predict(sess, int(m_ka.group(1)), int(m_ka.group(2)), event.reply_token)
                    save_session(uid, sess)
                    return

                # 結束分析 / RESET
                if up in ("結束分析", "清空", "RESET"):
                    premium = sess.get("premium", False)
                    start_ts = sess.get("trial_start", int(time.time()))
                    sess = get_session(uid)
                    sess["premium"] = premium
                    sess["trial_start"] = start_ts
                    _reply(event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。")
                    save_session(uid, sess)
                    return

                # 提示
                _reply(
                    event.reply_token,
                    "指令無法辨識。\n📌 已啟用連續模式：直接輸入點數即可（例：65 / 和 / 閒6莊5）。\n或輸入『遊戲設定』。",
                )
            except Exception as e:
                log.exception("on_text err: %s", e)
                try:
                    _reply(event.reply_token, "⚠️ 系統錯誤，稍後再試。")
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
                log.error("webhook error: %s", e)
                abort(500)
            return "OK", 200

    except Exception as e:
        log.warning("LINE not fully configured: %s", e)
else:
    log.warning("LINE credentials not set. LINE webhook will not be active.")

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s (CONTINUOUS_MODE=%s)", VERSION, port, CONTINUOUS_MODE)
    app.run(host="0.0.0.0", port=port, debug=False)
