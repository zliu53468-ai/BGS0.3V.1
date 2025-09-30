"""
server.py — 最終修正版（減少趨勢追隨，信心度5%-40%本金投注，優化Render免費版）
"""

import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np

try:
    import redis
except Exception:
    redis = None

try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
    _flask_available = True
except Exception:
    _flask_available = False
    Flask = None
    request = None
    def jsonify(*args, **kwargs):
        raise RuntimeError("Flask is not available; jsonify cannot be used.")
    def abort(*args, **kwargs):
        raise RuntimeError("Flask is not available; abort cannot be used.")
    def CORS(app):
        return None

# 版本號
VERSION = "bgs-final-fixed-2025-10-01-antitrend-v2"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---------- Flask 初始化 ----------
if _flask_available and Flask is not None:
    app = Flask(__name__)
    CORS(app)
else:
    class _DummyApp:
        def get(self, *args, **kwargs):
            def _decorator(func):
                return func
            return _decorator

        def post(self, *args, **kwargs):
            def _decorator(func):
                return func
            return _decorator

        def run(self, *args, **kwargs):
            log.warning("Flask not available; dummy app cannot run a server.")

    app = _DummyApp()

# ---------- Redis 或記憶體 Session ----------
REDIS_URL = os.getenv("REDIS_URL")
redis_client: Optional["redis.Redis"] = None
if redis is not None and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Successfully connected to Redis.")
    except Exception as e:
        redis_client = None
        log.error("Failed to connect to Redis: %s. Using in-memory session.", e)
else:
    if redis is None:
        log.warning("redis module not available; using in-memory session store.")
    elif not REDIS_URL:
        log.warning("REDIS_URL not set. Using in-memory session store.")

SESS_FALLBACK: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1200"))
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
    if t in ("B", "莊", "庄"):
        return (0, 1)
    if t in ("P", "閒", "闲"):
        return (1, 0)
    if t in ("T", "和"):
        return (0, 0)

    if re.search(r"[A-Z]", u):
        return None

    digits = re.findall(r"\d", u)
    if len(digits) == 2:
        return (int(digits[0]), int(digits[1]))
    return None

# ---------- 試用/授權 ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "60"))
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

# ---------- Outcome PF (粒子過濾器) ----------
log.info("載入 PF 參數: PF_N=%s, PF_UPD_SIMS=%s, PF_PRED_SIMS=%s, DECKS=%s",
         os.getenv("PF_N", "50"), os.getenv("PF_UPD_SIMS", "30"),
         os.getenv("PF_PRED_SIMS", "5"), os.getenv("DECKS", "8"))

os.environ['PF_BACKEND'] = 'mc'
os.environ['SKIP_TIE_UPD'] = '1'

OutcomePF = None
PF = None
pf_initialized = False

try:
    from bgs.pfilter import OutcomePF as RealOutcomePF
    OutcomePF = RealOutcomePF
    log.info("成功從 bgs.pfilter 導入 OutcomePF")
except Exception as e:
    try:
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        if _cur_dir not in sys.path:
            sys.path.insert(0, _cur_dir)
        from pfilter import OutcomePF as LocalOutcomePF
        OutcomePF = LocalOutcomePF
        log.info("成功從本地 pfilter 導入 OutcomePF")
    except Exception as pf_exc:
        log.error("無法導入 OutcomePF: %s", pf_exc)
        OutcomePF = None

if OutcomePF:
    try:
        PF = OutcomePF(
            decks=int(os.getenv("DECKS", "8")),
            seed=int(os.getenv("SEED", "42")),
            n_particles=int(os.getenv("PF_N", "50")),
            sims_lik=int(os.getenv("PF_UPD_SIMS", "30")),
            resample_thr=float(os.getenv("PF_RESAMPLE", "0.5")),
            backend=os.getenv("PF_BACKEND", "mc").lower(),
            dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.05")),  # 提高平滑
        )
        pf_initialized = True
        log.info(
            "PF 初始化成功: n_particles=%s, sims_lik=%s, decks=%s (backend=%s)",
            PF.n_particles,
            getattr(PF, 'sims_lik', 'N/A'),
            getattr(PF, 'decks', 'N/A'),
            getattr(PF, 'backend', 'unknown'),
        )
    except Exception as e:
        log.error("PF 初始化失敗: %s", e)
        pf_initialized = False
        OutcomePF = None

if not pf_initialized:
    class SmartDummyPF:
        def __init__(self):
            self.win_counts = np.array([0.0, 0.0, 0.0])
            self.total_games = 0
            log.warning("使用 SmartDummyPF 備援模式 - 請檢查 OutcomePF 導入問題")

        def update_outcome(self, outcome):
            if outcome in (0, 1, 2):
                self.win_counts[outcome] += 1.0
                self.total_games += 1
                log.info("SmartDummyPF 更新: outcome=%s, total_games=%s", outcome, self.total_games)

        def predict(self, **kwargs):
            if self.total_games == 0:
                base = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
            else:
                base = self.win_counts / self.total_games
                base = 0.7 * base + 0.3 * np.array([0.4586, 0.4462, 0.0952])
                base = base / base.sum()

            log.info("SmartDummyPF 預測: %s (基於 %s 場歷史)", base, self.total_games)
            return base.astype(np.float32)

        @property
        def backend(self):
            return "smart-dummy"

    PF = SmartDummyPF()
    log.warning("PF 初始化失敗，使用 SmartDummyPF 備援模式")

# ---------- 投注決策 ----------
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.05"))
USE_KELLY = env_flag("USE_KELLY", 0)
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)

INV = {0: "莊", 1: "閒"}

def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0:
        return 0
    return int(round(bankroll * pct))

def decide_only_bp(prob: np.ndarray, streak_count: int, last_outcome: Optional[int]) -> Tuple[str, float, float, str]:
    # 平滑機率以減少趨勢偏見
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    theo_probs = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)  # 百家樂理論機率
    smoothed_probs = 0.7 * np.array([pB, pP, pT]) + 0.3 * theo_probs
    smoothed_probs = smoothed_probs / smoothed_probs.sum()  # 重新正規化
    pB, pP = smoothed_probs[0], smoothed_probs[1]

    evB, evP = 0.95 * pB - pP, pP - pB

    # 連勝檢測：3局以上降低EV，減少趨勢追隨
    streak_adjust = 0.03 if streak_count >= 3 and last_outcome is not None else 0.0
    if last_outcome == 0:
        evB -= streak_adjust
    elif last_outcome == 1:
        evP -= streak_adjust

    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))

    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, "⚪ 優勢不足")

    # 根據信心度（final_edge）計算投注比例，範圍5%到40%
    max_edge = 0.15  # 假設最大信心度為0.15
    min_bet_pct = 0.05  # 最低5%本金
    max_bet_pct = 0.40  # 最高40%本金
    bet_pct = min_bet_pct + (max_bet_pct - min_bet_pct) * (final_edge - EDGE_ENTER) / (max_edge - EDGE_ENTER)
    bet_pct = min(max_bet_pct, max(min_bet_pct, float(bet_pct)))  # 限制在5%-40%
    reason = f"信心度配注({min_bet_pct*100:.0f}%~{max_bet_pct*100:.0f}%)"

    return (INV[side], final_edge, bet_pct, reason)

def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int, cont: bool) -> str:
    b_pct_txt = f"{prob[0] * 100:.2f}%"
    p_pct_txt = f"{prob[1] * 100:.2f}%"
    header: list[str] = []
    if last_pts_text:
        header.append(last_pts_text)
    header.append("開始分析下局....")

    block = [
        "【預測結果】",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"和：{prob[2] * 100:.2f}%",
        f"本次預測結果：{choice}",
        f"建議下注：{bet_amt:,}",
    ]
    if cont:
        block.append("\n📌 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）")
    return "\n".join(header + [""] + block)

# ---------- 健康檢查路由 ----------
@app.get("/")
def root():
    ua = request.headers.get("User-Agent", "")
    if "UptimeRobot" in ua:
        return "OK", 200
    status = "OK" if pf_initialized else "BACKUP_MODE"
    return f"✅ BGS Server {status} ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(
        ok=True,
        ts=time.time(),
        version=VERSION,
        pf_initialized=pf_initialized,
        pf_backend=getattr(PF, 'backend', 'unknown')
    ), 200

@app.get("/healthz")
def healthz():
    return jsonify(
        ok=True,
        ts=time.time(),
        version=VERSION,
        pf_initialized=pf_initialized
    ), 200

# ---------- LINE Bot ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None
line_handler = None

GAMES = {
    "1": "WM", "2": "PM", "3": "DG", "4": "SA", "5": "KU",
    "6": "歐博/卡利", "7": "KG", "8": "全利", "9": "名人", "10": "MT真人",
}

def game_menu_text(left_min: int) -> str:
    lines = ["【請選擇遊戲館別】"]
    for k in sorted(GAMES.keys(), key=lambda x: int(x)):
        lines.append(f"{k}. {GAMES[k]}")
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
    log.info("開始處理點數預測: 閒%d 莊%d", p_pts, b_pts)
    start_time = time.time()

    outcome = 2 if p_pts == b_pts else (1 if p_pts > b_pts else 0)

    # 更新連勝計數
    if sess.get("last_outcome") == outcome and outcome in (0, 1):
        sess["streak_count"] = sess.get("streak_count", 0) + 1
    else:
        sess["streak_count"] = 1 if outcome in (0, 1) else 0
    sess["last_outcome"] = outcome

    if p_pts == b_pts:
        sess["last_pts_text"] = "上局結果: 和局"
    else:
        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
        try:
            PF.update_outcome(outcome)
            log.info("勝局更新完成 (%s), 耗時: %.2fs", "閒勝" if outcome == 1 else "莊勝", time.time() - start_time)
        except Exception as e:
            log.warning("PF update err: %s", e)

    sess["phase"] = "ready"
    try:
        predict_start = time.time()
        p = PF.predict(sims_per_particle=int(os.getenv("PF_PRED_SIMS", "5")))
        log.info("預測完成, 耗時: %.2fs", time.time() - predict_start)

        choice, edge, bet_pct, reason = decide_only_bp(p, sess["streak_count"], sess["last_outcome"])
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
            text = re.sub(r"\s+", " ", raw.replace("\u3000", " ").strip())
            sess = get_session(uid)
            try:
                log.info("[LINE] uid=%s phase=%s text=%s", uid, sess.get("phase"), text)

                up = text.upper()
                if up.startswith("開通") or up.startswith("ACTIVATE"):
                    after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                    ok = validate_activation_code(after)
                    sess["premium"] = bool(ok)
                    _reply(event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤")
                    save_session(uid, sess)
                    return

                guard = trial_guard(sess)
                if guard:
                    _reply(event.reply_token, guard)
                    return

                pts = parse_last_hand_points(raw)
                if pts is not None:
                    if not sess.get("bankroll"):
                        _reply(event.reply_token, "請先完成『遊戲設定』與『本金設定』（例如輸入 5000），再回報點數。")
                        save_session(uid, sess)
                        return
                    _handle_points_and_predict(sess, int(pts[0]), int(pts[1]), event.reply_token)
                    save_session(uid, sess)
                    return

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
                        _reply(event.reply_token, "❌ 桌號格式錯誤，請輸入 2 英文字母 + 2 數字（例如: DG01）")
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
                        _reply(event.reply_token, "❌ 金額格式錯誤，請直接輸入正整數（例如: 5000）")
                        return

                norm = raw.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
                norm = re.sub(r"\s+", "", norm)
                m_ka = re.fullmatch(r"開始分析(\d)(\d)", norm)
                if m_ka and sess.get("bankroll"):
                    _handle_points_and_predict(sess, int(m_ka.group(1)), int(m_ka.group(2)), event.reply_token)
                    save_session(uid, sess)
                    return

                if up in ("結束分析", "清空", "RESET"):
                    premium = sess.get("premium", False)
                    start_ts = sess.get("trial_start", int(time.time()))
                    sess = get_session(uid)
                    sess["premium"] = premium
                    sess["trial_start"] = start_ts
                    _reply(event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。")
                    save_session(uid, sess)
                    return

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
    log.info("Starting %s on port %s (CONTINUOUS_MODE=%s, PF_INIT=%s)",
             VERSION, port, CONTINUOUS_MODE, pf_initialized)
    app.run(host="0.0.0.0", port=port, debug=False)
