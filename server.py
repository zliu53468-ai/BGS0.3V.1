"""
server.py — 連續模式修正版（Render 優化版 + 信心度→金額 + 和局穩定器）

- Render 免費版資源優化（輕量 PF）
- 依「信心度/優勢」配注金額（階梯 5% / 10% / 20% / 30%）
- 觀望不顯示金額；非觀望只顯示金額
- 和局處理：T 機率夾緊 + 和局後冷卻（可關）
- 機率平滑與溫度縮放（可關）
"""

import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np

# ---- Optional deps ----
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

try:
    from flask import Flask, request, jsonify, abort  # type: ignore
    from flask_cors import CORS  # type: ignore
    _flask_available = True
except Exception:
    _flask_available = False
    Flask = None  # type: ignore
    request = None  # type: ignore
    def jsonify(*args, **kwargs):  # type: ignore
        raise RuntimeError("Flask is not available")
    def abort(*args, **kwargs):  # type: ignore
        raise RuntimeError("Flask is not available")
    def CORS(app):  # type: ignore
        return None

VERSION = "bgs-pf-render-optimized-2025-09-17"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---- Flask or dummy ----
if _flask_available and Flask is not None:
    app = Flask(__name__)
    CORS(app)
else:
    class _DummyApp:
        def get(self, *args, **kwargs):
            def _d(fn): return fn
            return _d
        def post(self, *args, **kwargs):
            def _d(fn): return fn
            return _d
        def run(self, *args, **kwargs):
            log.warning("Flask not available; dummy app cannot run.")
    app = _DummyApp()

# ---- Redis or in-memory session ----
REDIS_URL = os.getenv("REDIS_URL")
redis_client: Optional["redis.Redis"] = None  # type: ignore
if redis is not None and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Redis connected.")
    except Exception as e:
        log.error("Redis connect fail: %s; fallback to memory.", e)
        redis_client = None
else:
    if redis is None:
        log.warning("redis module not available; using memory session.")
    elif not REDIS_URL:
        log.warning("REDIS_URL not set; using memory session.")

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
        "bankroll": 0,
        "trial_start": nowi,
        "premium": False,
        "phase": "choose_game",
        "game": None,
        "table": None,
        "last_pts_text": None,
        "table_no": None,
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
    if v in ("1", "true", "t", "yes", "y", "on"): return 1
    if v in ("0", "false", "f", "no", "n", "off"): return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except Exception:
        return 1 if default else 0

# ---- Betting knobs ----
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY = env_flag("USE_KELLY", 1)
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR", "0.25"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.015"))

# 勝率→配注（線性）：環境變數可調（現行改為用信心度，保留參數不動）
USE_WINRATE_MAP = env_flag("USE_WINRATE_MAP", 1)
BET_MIN_PCT = float(os.getenv("BET_MIN_PCT", "0.05"))   # 5%
BET_MAX_PCT = float(os.getenv("BET_MAX_PCT", "0.40"))   # 40%
WINRATE_FLOOR = float(os.getenv("WINRATE_FLOOR", "0.50"))
WINRATE_CEIL  = float(os.getenv("WINRATE_CEIL",  "0.75"))

# 和局穩定器 + 機率平滑 + 溫度縮放
TIE_PROB_MIN = float(os.getenv("TIE_PROB_MIN", "0.02"))
TIE_PROB_MAX = float(os.getenv("TIE_PROB_MAX", "0.12"))
POST_TIE_COOLDOWN = int(os.getenv("POST_TIE_COOLDOWN", "1"))
PROB_SMA_ALPHA = float(os.getenv("PROB_SMA_ALPHA", os.getenv("PROB_SMA_ALPHA".lower(), "0")))
PROB_TEMP = float(os.getenv("PROB_TEMP", os.getenv("PROB_TEMP".lower(), "1.0")))

CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)
INV = {0: "莊", 1: "閒"}

# ---- Parse last hand points ----
def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s)
    s = s.replace("\u3000", " ")
    u = s.upper().strip()
    u = re.sub(r"^開始分析", "", u)

    m = re.search(r"(?:和|TIE|DRAW)\s*:?:?\s*(\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else (0, 0)

    m = re.search(r"(?:閒|闲|P)\s*:?:?\s*(\d)\D+(?:莊|庄|B)\s*:?:?\s*(\d)", u)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    m = re.search(r"(?:莊|庄|B)\s*:?:?\s*(\d)\D+(?:閒|闲|P)\s*:?:?\s*(\d)", u)
    if m:
        return (int(m.group(2)), int(m.group(1)))

    t = u.replace(" ", "").replace("\u3000", "")
    if t in ("B", "莊", "庄"): return (0, 1)
    if t in ("P", "閒", "闲"): return (1, 0)
    if t in ("T", "和"):       return (0, 0)

    if re.search(r"[A-Z]", u):
        return None

    digits = re.findall(r"\d", u)
    if len(digits) == 2:
        return (int(digits[0]), int(digits[1]))
    return None

# ---- Trial / activation ----
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

# ---- PF bootstrap (Render-friendly) ----
os.environ['PF_N'] = os.getenv('PF_N', '30')
os.environ['PF_UPD_SIMS'] = os.getenv('PF_UPD_SIMS', '20')
os.environ['PF_PRED_SIMS'] = os.getenv('PF_PRED_SIMS', '0')
os.environ['DECKS'] = os.getenv('DECKS', '6')
if not os.getenv('PF_BACKEND'):
    os.environ['PF_BACKEND'] = 'mc'
log.info("強制設置 PF 參數: PF_N=%s, PF_UPD_SIMS=%s, PF_PRED_SIMS=%s, DECKS=%s",
         os.environ['PF_N'], os.environ['PF_UPD_SIMS'], os.environ['PF_PRED_SIMS'], os.environ['DECKS'])

try:
    from bgs.pfilter import OutcomePF  # type: ignore
except Exception:
    try:
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        if _cur_dir not in sys.path:
            sys.path.insert(0, _cur_dir)
        from pfilter import OutcomePF  # type: ignore
        log.info("Imported OutcomePF from local pfilter.")
    except Exception as _pf_exc:
        OutcomePF = None  # type: ignore
        log.error("Could not import OutcomePF: %s", _pf_exc)

if OutcomePF:
    try:
        PF = OutcomePF(
            decks=int(os.getenv("DECKS", "6")),
            seed=int(os.getenv("SEED", "42")),
            n_particles=int(os.getenv("PF_N", "30")),
            sims_lik=max(1, int(os.getenv("PF_UPD_SIMS", "20"))),
            resample_thr=float(os.getenv("PF_RESAMPLE", "0.6")),
            backend=os.getenv("PF_BACKEND", "mc").lower(),
            dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.003")),
        )
        log.info("PF init ok: n=%d, sims=%d, backend=%s",
                 PF.n_particles, getattr(PF, "sims_lik", 0), getattr(PF, "backend", "unknown"))
    except Exception as _e:
        log.error("PF init fail: %s", _e)
        OutcomePF = None

if not OutcomePF:
    class DummyPF:
        def update_outcome(self, outcome):
            log.info("DummyPF 更新: %s", outcome)
        def predict(self, **kwargs):
            log.info("DummyPF 預測")
            return np.array([0.48, 0.47, 0.05], dtype=np.float32)
        @property
        def backend(self):
            return "dummy"
    PF = DummyPF()
    log.info("使用 DummyPF 模式")

def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0:
        return 0
    return int(round(bankroll * pct))

def decide_only_bp(prob: np.ndarray) -> Tuple[str, float, float, str]:
    """根據信心度（優勢）來決定下注金額（不再用勝率映射）。"""
    pB, pP = float(prob[0]), float(prob[1])
    side = 0 if pB >= pP else 1

    # 優勢（信心度）：考慮 0.95 抽水
    evB, evP = 0.95 * pB - pP, pP - pB
    final_edge = max(abs(evB), abs(evP))

    # 階梯配注：5% / 10% / 20% / 30%
    if final_edge >= 0.10:
        bet_pct, reason = 0.30, "高信心"
    elif final_edge >= 0.07:
        bet_pct, reason = 0.20, "中等信心"
    elif final_edge >= 0.04:
        bet_pct, reason = 0.10, "低信心"
    else:
        bet_pct, reason = 0.05, "非常低信心"

    # 仍保留 MAX_BET_PCT 上限（更保守）
    bet_pct = min(bet_pct, float(os.getenv("BET_MAX_PCT", str(BET_MAX_PCT))))

    return (INV[side], final_edge, bet_pct, reason)

def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int, cont: bool) -> str:
    """組合回覆：只顯示金額；觀望不顯示金額。"""
    b_pct_txt = f"{prob[0] * 100:.2f}%"
    p_pct_txt = f"{prob[1] * 100:.2f}%"
    header = []
    if last_pts_text:
        header.append(last_pts_text)
    header.append("開始分析下局....")
    bet_line = "建議：觀望" if choice == "觀望" else f"建議下注：{bet_amt:,}"
    block = [
        "【預測結果】",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"本次預測結果：{choice if choice != '觀望' else '觀'}",
        bet_line,
    ]
    if cont:
        block.append("\n📌 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）")
    return "\n".join(header + [""] + block)

# ---- Health routes ----
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

# ---- LINE Bot ----
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None
line_handler = None

if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.models import MessageEvent, TextMessage, FollowEvent
        from linebot.exceptions import InvalidSignatureError
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

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
    except Exception as e:
        log.warning("LINE SDK init failed: %s", e)
        line_api, line_handler = None, None

# 若 SDK 未配置，也提供一個無害的 _quick_buttons()
if line_handler is None:
    def _quick_buttons():
        return None

def _reply(token: str, text: str):
    # 將 import 放到 try 內，避免未安裝 linebot 套件時在此爆掉
    try:
        if line_api is None:
            return
        from linebot.models import TextSendMessage
        line_api.reply_message(token, TextSendMessage(text=text, quick_reply=_quick_buttons()))
    except Exception as e:
        log.warning("[LINE] reply failed: %s", e)

def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id: return True
    return _rsetnx(f"dedupe:{event_id}", "1", DEDUPE_TTL)

def _handle_points_and_predict(sess: Dict[str, Any], p_pts: int, b_pts: int, reply_token: str):
    log.info("開始處理點數預測: 閒%d 莊%d", p_pts, b_pts)
    start_time = time.time()

    # ---- 更新上一局結果 ----
    if p_pts == b_pts:
        sess["last_pts_text"] = "上局結果: 和局"
        sess["post_tie_cooldown"] = POST_TIE_COOLDOWN
        try:
            if int(os.getenv("SKIP_TIE_UPD", "0")) == 0:
                PF.update_outcome(2)
                log.info("和局更新完成, 耗時: %.2fs", time.time() - start_time)
        except Exception as e:
            log.warning("PF tie update err: %s", e)
    else:
        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
        try:
            outcome = 1 if p_pts > b_pts else 0
            PF.update_outcome(outcome)
            log.info("勝局更新完成 (%s), 耗時: %.2fs", "閒勝" if outcome == 1 else "莊勝", time.time() - start_time)
        except Exception as e:
            log.warning("PF update err: %s", e)

    # ---- 預測 ----
    sess["phase"] = "ready"
    try:
        predict_start = time.time()
        p = PF.predict(sims_per_particle=max(0, int(os.getenv("PF_PRED_SIMS", "0"))))
        log.info("預測完成, 耗時: %.2fs", time.time() - predict_start)

        p = np.asarray(p, dtype=np.float32)

        # 機率平滑
        if PROB_SMA_ALPHA > 0:
            last_p = np.asarray(sess.get("last_prob") or p, dtype=np.float32)
            p = (1 - PROB_SMA_ALPHA) * last_p + PROB_SMA_ALPHA * p

        # 溫度縮放
        if PROB_TEMP > 0 and abs(PROB_TEMP - 1.0) > 1e-6:
            logits = np.log(np.clip(p, 1e-6, 1.0))
            p = np.exp(logits / PROB_TEMP)
            p = p / np.sum(p)

        # 和局機率夾緊
        try:
            pB, pP, pT = float(p[0]), float(p[1]), float(p[2])
            pT = min(max(pT, TIE_PROB_MIN), TIE_PROB_MAX)
            rest = max(1e-6, 1.0 - pT)
            bp_sum = max(1e-6, pB + pP)
            b_share = pB / bp_sum
            pB = rest * b_share
            pP = rest * (1.0 - b_share)
            p = np.array([pB, pP, pT], dtype=np.float32)
            p = p / np.sum(p)
        except Exception:
            pass

        sess["last_prob"] = p.tolist()

        # 決策（含和局冷卻）
        choice, edge, bet_pct, reason = decide_only_bp(p)
        cooldown = int(sess.get("post_tie_cooldown", 0) or 0)
        if cooldown > 0:
            sess["post_tie_cooldown"] = cooldown - 1
            choice, bet_pct, reason = "觀望", 0.0, "和局冷卻"

        bankroll_now = int(sess.get("bankroll", 0))
        bet_amt = bet_amount(bankroll_now, bet_pct)
        msg = format_output_card(p, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE))
        _reply(reply_token, msg)
        log.info("完整處理完成, 總耗時: %.2fs", time.time() - start_time)

    except Exception as e:
        log.error("預測過程中錯誤: %s", e)
        _reply(reply_token, "⚠️ 預計算錯誤，請稍後再試")

    if CONTINUOUS_MODE:
        sess["phase"] = "await_pts"

# ---- LINE webhook Routes（永遠註冊，避免 404） ----
@app.post("/line-webhook")
def line_webhook():
    # 若 handler 未配置，直接回 200，避免 404 造成 LINE 重試與 Render 告警
    if not (LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN and line_handler is not None):
        log.warning("LINE webhook hit but credentials not configured; returning 200 noop.")
        return "NOOP", 200

    try:
        from linebot.exceptions import InvalidSignatureError
        signature = request.headers.get("X-Line-Signature", "")
        body = request.get_data(as_text=True)
        line_handler.handle(body, signature)
        return "OK", 200
    except InvalidSignatureError:
        log.error("Invalid signature on webhook")
        return "Invalid signature", 400
    except Exception as e:
        log.error("webhook error: %s", e)
        return "Internal error", 500

# 若 credentials 存在，註冊事件處理器
if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN and line_handler is not None:
    from linebot.models import MessageEvent, TextMessage, FollowEvent

    @line_handler.add(FollowEvent)
    def on_follow(event):
        if not _dedupe_event(getattr(event, "id", None)):
            return
        uid = event.source.user_id
        sess = get_session(uid)
        _reply(event.reply_token, "👋 歡迎！輸入『遊戲設定』開始；已啟用連續模式，之後只需輸入點數（例：65 / 和 / 閒6莊5）。")
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

            # 開通
            up = text.upper()
            if up.startswith("開通") or up.startswith("ACTIVATE"):
                after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                ok = validate_activation_code(after)
                sess["premium"] = bool(ok)
                _reply(event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤")
                save_session(uid, sess)
                return

            # 試用守門
            guard = trial_guard(sess)
            if guard:
                _reply(event.reply_token, guard)
                return

            # 連續模式：先嘗試解析上局點數
            pts = parse_last_hand_points(raw)
            if pts is not None:
                if not sess.get("bankroll"):
                    _reply(event.reply_token, "請先完成『遊戲設定』與『本金設定』（例如輸入 5000），再回報點數。")
                    save_session(uid, sess)
                    return
                _handle_points_and_predict(sess, int(pts[0]), int(pts[1]), event.reply_token)
                save_session(uid, sess)
                return

            # 遊戲設定
            if up in ("遊戲設定", "設定", "SETUP", "GAME"):
                sess["phase"] = "choose_game"
                left = trial_left_minutes(sess)
                menu = ["【請選擇遊戲館別】"]
                for k in sorted({"1":"WM","2":"PM","3":"DG","4":"SA","5":"KU","6":"歐博/卡利","7":"KG","8":"全利","9":"名人","10":"MT真人"}.keys(), key=lambda x: int(x)):
                    menu.append(f"{k}. {{'1':'WM','2':'PM','3':'DG','4':'SA','5':'KU','6':'歐博/卡利','7':'KG','8':'全利','9':'名人','10':'MT真人'}[k]}")
                menu.append("「請直接輸入數字選擇」")
                menu.append(f"⏳ 試用剩餘 {left} 分鐘（共 {TRIAL_MINUTES} 分鐘）")
                _reply(event.reply_token, "\n".join(menu))
                save_session(uid, sess)
                return

            phase = sess.get("phase", "choose_game")

            if phase == "choose_game":
                if re.fullmatch(r"([1-9]|10)", text):
                    games = {"1":"WM","2":"PM","3":"DG","4":"SA","5":"KU","6":"歐博/卡利","7":"KG","8":"全利","9":"名人","10":"MT真人"}
                    sess["game"] = games[text]
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
                    _reply(event.reply_token, "❌ 桌號格式錯誤，請輸入 2 英文字母 + 2 數字（例如: DG01）")
                    return

            elif phase == "await_bankroll":
                if text.isdigit() and int(text) > 0:
                    sess["bankroll"] = int(text)
                    sess["phase"] = "await_pts"
                    _reply(event.reply_token, f"👍 已設定本金：{sess['bankroll']:,}\n📌 連續模式開啟：直接輸入上局點數（例：65 / 和 / 閒6莊5）即可自動預測。")
                    save_session(uid, sess)
                    return
                else:
                    _reply(event.reply_token, "❌ 金額格式錯誤，請直接輸入正整數（例如: 5000）")
                    return

            # 舊流程：開始分析XY
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

            # Fallback
            _reply(event.reply_token, "指令無法辨識。\n📌 已啟用連續模式：直接輸入點數即可（例：65 / 和 / 閒6莊5）。\n或輸入『遊戲設定』。")

        except Exception as e:
            log.exception("on_text err: %s", e)
            try:
                _reply(event.reply_token, "⚠️ 系統錯誤，稍後再試。")
            except Exception:
                pass

# ---- Main ----
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s (CONTINUOUS_MODE=%s)", VERSION, port, CONTINUOUS_MODE)
    app.run(host="0.0.0.0", port=port, debug=False)
