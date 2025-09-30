# -*- coding: utf-8 -*-
"""
server.py — BGS 百家樂 AI（狀態獨立・點數證據版）
- 預測核心：bgs.pfilter.OutcomePF = DirichletFeaturePF（不吃趨勢，點數→證據）
- 平滑：smoothed = 0.4 * pred + 0.6 * theo  （你指定）
- 連莊懲罰：門檻 2 局、懲罰 8%（你指定）
- EV 決策：Tie 為 0EV；下注只在 B/P；信心度→5%~40%本金
- LINE 流程、試用、Redis/in-memory session 與路由維持原樣（僅補上 update_points 呼叫）
"""

import os, sys, re, time, json, logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

# ---------- Optional deps (Flask/LINE/Redis) ----------
try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
    _has_flask = True
except Exception:
    _has_flask = False
    Flask = None  # type: ignore
    def jsonify(*_, **__): raise RuntimeError("Flask not available")
    def CORS(*_, **__): pass

try:
    import redis
except Exception:
    redis = None

VERSION = "bgs-pf-dirichlet-feature-2025-10-01"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ====== 你指定的常數（其餘設定不動） ======
SMOOTH_ALPHA = 0.4
THEO_ALPHA   = 0.6
STREAK_THRESH  = 2
STREAK_PENALTY = 0.08
# ========================================

# ---------- Flask ----------
if _has_flask and Flask is not None:
    app = Flask(__name__)
    CORS(app)
else:
    class _Dummy:
        def get(self, *_, **__):
            def _f(fn): return fn
            return _f
        def post(self, *_, **__):
            def _f(fn): return fn
            return _f
        def run(self, *_, **__):
            log.warning("Flask not available")
    app = _Dummy()

# ---------- Redis / Session ----------
REDIS_URL = os.getenv("REDIS_URL")
redis_client: Optional["redis.Redis"] = None
if redis is not None and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Connected to Redis")
    except Exception as e:
        log.error("Redis connect fail: %s", e)

SESS_FALLBACK: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1200"))
DEDUPE_TTL = 60


def _rget(k: str) -> Optional[str]:
    try:
        return redis_client.get(k) if redis_client else None
    except Exception:
        return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if redis_client:
            redis_client.set(k, v, ex=ex)
    except Exception:
        pass

def _rsetnx(k: str, v: str, ex: int) -> bool:
    try:
        if redis_client:
            return bool(redis_client.set(k, v, nx=True, ex=ex))
        if k not in SESS_FALLBACK:
            SESS_FALLBACK[k] = {"v": v, "exp": time.time() + ex}
            return True
        return False
    except Exception:
        return True


def get_session(uid: str) -> Dict[str, Any]:
    if redis_client:
        j = _rget(f"bgs_session:{uid}")
        if j:
            try:
                return json.loads(j)
            except Exception:
                pass
    nowi = int(time.time())
    return {
        "bankroll": 0,
        "trial_start": nowi,
        "premium": False,
        "phase": "choose_game",
        "game": None,
        "table": None,
        "last_pts_text": None,
        "streak_count": 0,
        "last_outcome": None,
    }

def save_session(uid: str, data: Dict[str, Any]):
    if redis_client:
        _rset(f"bgs_session:{uid}", json.dumps(data), ex=SESSION_EXPIRE_SECONDS)
    else:
        SESS_FALLBACK[uid] = data


def env_flag(name: str, default: int = 1) -> int:
    val = os.getenv(name)
    if val is None: return 1 if default else 0
    v = str(val).strip().lower()
    if v in ("1","true","t","yes","y","on"): return 1
    if v in ("0","false","f","no","n","off"): return 0
    try: return 1 if int(float(v)) != 0 else 0
    except Exception: return 1 if default else 0

# ---------- 解析上局點數 ----------
_deftrash = str.maketrans("０１２３４５６７８９：", "0123456789:")

def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
    if not text: return None
    s = str(text).translate(_deftrash)
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s)
    s = s.replace("\u3000", " ")
    u = s.upper().strip()

    m = re.search(r"(?:和|TIE|DRAW)\s*:?:?\s*(\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else (0, 0)

    m = re.search(r"(?:閒|P)\s*:?:?\s*(\d)\D+(?:莊|B)\s*:?:?\s*(\d)", u)
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|B)\s*:?:?\s*(\d)\D+(?:閒|P)\s*:?:?\s*(\d)", u)
    if m: return (int(m.group(2)), int(m.group(1)))

    t = u.replace(" ", "").replace("\u3000", "")
    if t in ("B","莊"): return (0,1)
    if t in ("P","閒"): return (1,0)
    if t in ("T","和"): return (0,0)

    if re.search(r"[A-Z]", u): return None
    digits = re.findall(r"\d", u)
    if len(digits) == 2: return (int(digits[0]), int(digits[1]))
    return None

# ---------- 試用/授權 ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "60"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "aaa8881688")

def validate_activation_code(code: str) -> bool:
    if not code: return False
    norm = str(code).replace("\u3000", " ").replace("：", ":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and (norm == ADMIN_ACTIVATION_SECRET)

def trial_left_minutes(sess: Dict[str, Any]) -> int:
    if sess.get("premium", False): return 9999
    now = int(time.time())
    used = (now - int(sess.get("trial_start", now))) // 60
    return max(0, TRIAL_MINUTES - used)

def trial_guard(sess: Dict[str, Any]) -> Optional[str]:
    if sess.get("premium", False): return None
    if trial_left_minutes(sess) <= 0:
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 在此輸入：開通 你的密碼"
    return None

# ---------- Outcome PF ----------
OutcomePF = None
try:
    from bgs.pfilter import OutcomePF as _OutcomePF
    OutcomePF = _OutcomePF
    log.info("Loaded OutcomePF from bgs.pfilter")
except Exception as e:
    log.error("Cannot import bgs.pfilter: %s", e)

PF = None
pf_initialized = False
if OutcomePF:
    try:
        PF = OutcomePF(
            decks=int(os.getenv("DECKS", "8")),
            seed=int(os.getenv("SEED", "42")),
            n_particles=int(os.getenv("PF_N", "50")),
            sims_lik=int(os.getenv("PF_UPD_SIMS", "30")),
            resample_thr=float(os.getenv("PF_RESAMPLE", "0.5")),
            backend=os.getenv("PF_BACKEND", "bayes"),
            dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.08")),
        )
        pf_initialized = True
    except Exception as e:
        log.error("PF init failed: %s", e)
        pf_initialized = False

if not pf_initialized:
    # 穩定備援：理論機率常數
    class _ConstPF:
        def __init__(self): self._backend = "const-theo"
        def predict(self, **_):
            return np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
        def update_outcome(self, *_): pass
        @property
        def backend(self): return self._backend
    PF = _ConstPF()

EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.05"))
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)
INV = {0: "莊", 1: "閒"}

def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0: return 0
    return int(round(bankroll * pct))


def decide_only_bp(prob: np.ndarray, streak_count: int, last_outcome: Optional[int]) -> Tuple[str, float, float, str]:
    theo_probs = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
    smoothed = SMOOTH_ALPHA * prob + THEO_ALPHA * theo_probs
    smoothed = smoothed / smoothed.sum()
    pB, pP, pT = float(smoothed[0]), float(smoothed[1]), float(smoothed[2])

    evB, evP = 0.95 * pB - pP, pP - pB

    # 連莊懲罰（你指定）
    adj = STREAK_PENALTY if (streak_count >= STREAK_THRESH and last_outcome in (0,1)) else 0.0
    if last_outcome == 0: evB -= adj
    if last_outcome == 1: evP -= adj

    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))
    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, "⚪ 優勢不足")

    # 信心度→配注（5%~40%）
    max_edge = 0.15
    min_bet_pct, max_bet_pct = 0.05, 0.40
    bet_pct = min_bet_pct + (max_bet_pct - min_bet_pct) * (final_edge - EDGE_ENTER) / (max_edge - EDGE_ENTER)
    bet_pct = min(max_bet_pct, max(min_bet_pct, float(bet_pct)))
    return (INV[side], final_edge, bet_pct, "信心度配注(5%~40%)")


def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int, cont: bool) -> str:
    b_pct_txt = f"{prob[0] * 100:.2f}%"; p_pct_txt = f"{prob[1] * 100:.2f}%"; t_pct_txt = f"{prob[2] * 100:.2f}%"
    header = []
    if last_pts_text: header.append(last_pts_text)
    header.append("開始分析下局....")
    block = [
        "【預測結果】",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"和：{t_pct_txt}",
        f"本次預測結果：{choice}",
        f"建議下注金額：{bet_amt}",
    ]
    if cont: block.append("\n📌 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）")
    return "\n".join(header + [""] + block)

# ---------- 健康檢查 ----------
@app.get("/")
def root():
    return f"✅ BGS Server ({VERSION}) backend={getattr(PF, 'backend', 'unknown')}", 200

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION, pf_backend=getattr(PF,'backend','unknown')), 200

# ---------- LINE Bot（保持你原流程） ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None; line_handler = None
GAMES = {"1":"WM","2":"PM","3":"DG","4":"SA","5":"KU","6":"歐博/卡利","7":"KG","8":"全利","9":"名人","10":"MT真人"}

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
    if not event_id: return True
    return _rsetnx(f"dedupe:{event_id}", "1", DEDUPE_TTL)


def _handle_points_and_predict(uid: str, sess: Dict[str, Any], p_pts: int, b_pts: int, reply_token: str):
    log.info("處理點數: P%d B%d", p_pts, b_pts)
    # 1) 先把『點數證據』寫入 PF（獨立於趨勢）
    try:
        if hasattr(PF, "update_points"):
            PF.update_points(int(p_pts), int(b_pts))
            log.info("PF.update_points applied")
    except Exception as e:
        log.warning("PF.update_points error: %s", e)

    # 2) 判定 outcome（只供統計與連莊懲罰；OutcomePF 預設 OUTCOME_WEIGHT=0，不吃趨勢）
    outcome = 2 if p_pts == b_pts else (1 if p_pts > b_pts else 0)
    if sess.get("last_outcome") == outcome and outcome in (0,1):
        sess["streak_count"] = sess.get("streak_count", 0) + 1
    else:
        sess["streak_count"] = 1 if outcome in (0,1) else 0
    sess["last_outcome"] = outcome

    # 3) 可選：把勝負寫入 PF（若 OUTCOME_WEIGHT>0 才會有效）
    try:
        PF.update_outcome(outcome)
    except Exception as e:
        log.warning("PF.update_outcome error: %s", e)

    sess["last_pts_text"] = "上局結果: 和局" if p_pts == b_pts else f"上局結果: 閒 {p_pts} 莊 {b_pts}"

    # 4) 做預測
    p = PF.predict(sims_per_particle=int(os.getenv("PF_PRED_SIMS", "5")))
    choice, edge, bet_pct, reason = decide_only_bp(p, sess["streak_count"], sess["last_outcome"])
    bankroll_now = int(sess.get("bankroll", 0))
    bet_amt = bet_amount(bankroll_now, bet_pct)

    msg = format_output_card(p, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE))
    _reply(reply_token, msg)
    save_session(uid, sess)


if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import MessageEvent, TextMessage, FollowEvent

        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

        @line_handler.add(FollowEvent)
        def on_follow(event):
            if not _dedupe_event(getattr(event, "id", None)): return
            uid = event.source.user_id
            sess = get_session(uid)
            _reply(event.reply_token, "👋 歡迎！輸入『遊戲設定』開始；之後直接輸入點數（例：65 / 和 / 閒6莊5）即可自動預測。")
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            if not _dedupe_event(getattr(event, "id", None)): return
            uid = event.source.user_id
            raw = (event.message.text or "")
            text = re.sub(r"\s+", " ", raw.replace("\u3000", " ").strip())
            sess = get_session(uid)

            # 開通
            up = text.upper()
            if up.startswith("開通") or up.startswith("ACTIVATE"):
                after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                ok = validate_activation_code(after)
                sess["premium"] = bool(ok)
                _reply(event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤")
                save_session(uid, sess); return

            # 試用守門
            guard = trial_guard(sess)
            if guard: _reply(event.reply_token, guard); return

            # 解析點數
            pts = parse_last_hand_points(raw)
            if pts is not None:
                if not sess.get("bankroll"):
                    _reply(event.reply_token, "請先完成『遊戲設定』與『本金設定』（例如輸入 5000），再回報點數。")
                    save_session(uid, sess); return
                _handle_points_and_predict(uid, sess, int(pts[0]), int(pts[1]), event.reply_token)
                return

            # 設定流程（館別→桌號→本金） — 省略版保留原句型
            if up in ("遊戲設定","設定","SETUP","GAME"):
                sess["phase"] = "choose_game"
                left = trial_left_minutes(sess)
                menu = ["【請選擇遊戲館別】"]
                for k in sorted(GAMES.keys(), key=lambda x: int(x)):
                    menu.append(f"{k}. {GAMES[k]}")
                menu.append("「請直接輸入數字選擇」")
                menu.append(f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
                _reply(event.reply_token, "\n".join(menu)); save_session(uid, sess); return

            phase = sess.get("phase", "choose_game")
            if phase == "choose_game":
                if re.fullmatch(r"([1-9]|10)", text):
                    sess["game"] = GAMES[text]; sess["phase"] = "choose_table"
                    _reply(event.reply_token, f"✅ 已設定館別【{sess['game']}】\n請輸入桌號（例：DG01）"); save_session(uid, sess); return
            elif phase == "choose_table":
                t = re.sub(r"\s+", "", text).upper()
                if re.fullmatch(r"[A-Z]{2}\d{2}", t):
                    sess["table"] = t; sess["phase"] = "await_bankroll"
                    _reply(event.reply_token, f"✅ 已設定桌號【{sess['table']}】\n請輸入您的本金（例：5000）"); save_session(uid, sess); return
                else:
                    _reply(event.reply_token, "❌ 桌號格式錯誤，請輸入 2 英文字母 + 2 數字（例如: DG01）"); return
            elif phase == "await_bankroll":
                if text.isdigit() and int(text) > 0:
                    sess["bankroll"] = int(text); sess["phase"] = "await_pts"
                    _reply(event.reply_token, f"👍 已設定本金：{sess['bankroll']:,}\n📌 直接輸入上局點數（例：65 / 和 / 閒6莊5）即可自動預測。"); save_session(uid, sess); return
                else:
                    _reply(event.reply_token, "❌ 金額格式錯誤，請輸入正整數（例如: 5000）"); return

            if up in ("結束分析","清空","RESET"):
                premium = sess.get("premium", False); start_ts = sess.get("trial_start", int(time.time()))
                sess = get_session(uid); sess["premium"], sess["trial_start"] = premium, start_ts
                _reply(event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。"); save_session(uid, sess); return

            _reply(event.reply_token, "指令無法辨識。\n📌 已啟用連續模式：直接輸入點數即可（例：65 / 和 / 閒6莊5）。\n或輸入『遊戲設定』。")

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

# ---------- main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s (backend=%s)", VERSION, port, getattr(PF, 'backend', 'unknown'))
    app.run(host="0.0.0.0", port=port, debug=False)
