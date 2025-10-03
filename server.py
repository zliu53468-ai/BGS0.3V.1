# -*- coding: utf-8 -*-
"""server.py — Updated version for independent round predictions (no memory)"""
import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple

import numpy as np

# --- deplete import shim: 同時支援 bgs/deplete.py 與根目錄 deplete.py ---
try:
    from bgs.deplete import init_counts, probs_after_points
except ModuleNotFoundError:
    try:
        from deplete import init_counts, probs_after_points
    except ModuleNotFoundError as e:
        raise ImportError(
            "找不到 deplete 模組。請確認：\n"
            "1) 有 bgs/deplete.py，且 bgs/ 內存在 __init__.py（建議做法），或\n"
            "2) deplete.py 與 server.py 在同一層。"
        ) from e

# ---------- Optional deps ----------
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
VERSION = "bgs-independent-2025-10-02+webhook-fallback+line-webhook-alias"

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
            def _decorator(func): return func
            return _decorator
        def post(self, *args, **kwargs):
            def _decorator(func): return func
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
        # Clean up expired fallback sessions
        now = time.time()
        for k in list(SESS_FALLBACK.keys()):
            v = SESS_FALLBACK.get(k)
            if isinstance(v, dict) and v.get("exp") and v["exp"] < now:
                del SESS_FALLBACK[k]
        if uid in SESS_FALLBACK and "phase" in SESS_FALLBACK[uid]:
            return SESS_FALLBACK[uid]
    # Initialize a new session
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
    if v in ("1", "true", "t", "yes", "y", "on"): return 1
    if v in ("0", "false", "f", "no", "n", "off"): return 0
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

    # Parse special formats for tie or points
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
        return (0, 1)  # Banker win (player 0, banker 1)
    if t in ("P", "閒", "闲"):
        return (1, 0)  # Player win (player 1, banker 0)
    if t in ("T", "和"):
        return (0, 0)  # Tie

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

PF_BACKEND = os.getenv("PF_BACKEND", "mc").lower()
SKIP_TIE_UPD = env_flag("SKIP_TIE_UPD", 1)
SOFT_TAU = float(os.getenv("SOFT_TAU", "2.0"))
TIE_MIN = float(os.getenv("TIE_MIN", "0.05"))
TIE_MAX = float(os.getenv("TIE_MAX", "0.15"))
HISTORY_MODE = env_flag("HISTORY_MODE", 0)

OutcomePF = None
PF = None
pf_initialized = False

try:
    from bgs.pfilter import OutcomePF as RealOutcomePF
    OutcomePF = RealOutcomePF
    log.info("成功從 bgs.pfilter 導入 OutcomePF")
except Exception:
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
            backend=PF_BACKEND,
            dirichlet_eps=float(os.getenv("PF_DIR_EPS", "0.05"))
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
            log.warning("使用 SmartDummyPF 備援模式 - 請檢查 OutcomePF 導入問題")
        def update_outcome(self, outcome):
            return
        def predict(self, **kwargs) -> np.ndarray:
            base = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
            base = base ** (1.0 / SOFT_TAU)
            base = base / base.sum()
            pT = base[2]
            if pT < TIE_MIN:
                base[2] = TIE_MIN
                scale = (1.0 - TIE_MIN) / (1.0 - pT) if pT < 1.0 else 1.0
                base[0] *= scale; base[1] *= scale
            elif base[2] > TIE_MAX:
                base[2] = TIE_MAX
                scale = (1.0 - TIE_MAX) / (1.0 - (base[2] - (base[2] - TIE_MAX)))
                base[0] *= scale; base[1] *= scale
            return base.astype(np.float32)
        @property
        def backend(self): return "smart-dummy"
    PF = SmartDummyPF()
    pf_initialized = True
    log.warning("PF 初始化失敗，使用 SmartDummyPF 備援模式")

# ---------- 投注決策 ----------
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY = env_flag("USE_KELLY", 0)
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)

INV = {0: "莊", 1: "閒"}

def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0:
        return 0
    return int(round(bankroll * pct))

def decide_only_bp(prob: np.ndarray) -> Tuple[str, float, float, str]:
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    theo_probs = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
    smoothed = 0.7 * np.array([pB, pP, pT]) + 0.3 * theo_probs
    smoothed = smoothed / smoothed.sum()
    pB, pP = float(smoothed[0]), float(smoothed[1])

    evB = 0.95 * pB - pP
    evP = pP - pB
    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))

    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, "⚪ 優勢不足")

    max_edge = 0.15
    min_bet_pct = 0.05
    max_bet_pct = 0.40
    bet_pct = min_bet_pct + (max_bet_pct - min_bet_pct) * (final_edge - EDGE_ENTER) / (max_edge - EDGE_ENTER)
    bet_pct = float(min(max_bet_pct, max(min_bet_pct, bet_pct)))
    reason = f"信心度配注({int(min_bet_pct*100)}%~{int(max_bet_pct*100)}%)"
    return (INV[side], final_edge, bet_pct, reason)

def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int, cont: bool) -> str:
    b_pct_txt = f"{prob[0] * 100:.2f}%"
    p_pct_txt = f"{prob[1] * 100:.2f}%"
    header: list[str] = []
    if last_pts_text: header.append(last_pts_text)
    header.append("開始分析下局....")
    block = [
        "預測結果",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"和：{prob[2] * 100:.2f}%",
        f"本次預測結果：{choice}",
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
    status = "OK" if pf_initialized else "BACKUP_MODE"
    return f"✅ BGS Server {status} ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(
        ok=True, ts=time.time(), version=VERSION,
        pf_initialized=pf_initialized, pf_backend=getattr(PF, 'backend', 'unknown')
    ), 200

@app.get("/healthz")
def healthz():
    return jsonify(ok=True, ts=time.time(), version=VERSION, pf_initialized=pf_initialized), 200

# ---------- LINE Bot / Dummy 模式 ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_MODE = "real" if (LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN) else "dummy"

line_api = None
line_handler = None

if LINE_MODE == "real":
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.exceptions import InvalidSignatureError
        from linebot.models import MessageEvent, TextMessage
        line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        line_handler = WebhookHandler(LINE_CHANNEL_SECRET)
        log.info("LINE Webhook 啟用（real mode）")

        # --- 共用處理：讓 /callback 與 /line-webhook 都走同一支 ---
        def _handle_line_webhook_request():
            signature = request.headers.get("X-Line-Signature", "")
            body = request.get_data(as_text=True)
            try:
                line_handler.handle(body, signature)
            except InvalidSignatureError:
                log.warning("Invalid signature")
                return "Bad signature", 400
            return "OK", 200

        @app.post("/callback")
        def callback():
            return _handle_line_webhook_request()

        # 你的 LINE 後台目前打的是這條路徑 → 直接映射同一處理器
        @app.post("/line-webhook")
        def line_webhook_alias():
            return _handle_line_webhook_request()

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text_message(event):
            if not event or not event.message or not event.message.text:
                return
            uid = event.source.user_id if getattr(event, "source", None) else "no_uid"
            if not _dedupe_event(event.message.id):
                return
            sess = get_session(uid)

            txt = event.message.text.strip()
            # 試用門
            if txt.startswith("開通"):
                code = txt.split("開通", 1)[-1].strip()
                if validate_activation_code(code):
                    sess["premium"] = True
                    save_session(uid, sess)
                    _reply(event.reply_token, "✅ 已開通永久版，歡迎使用！")
                else:
                    _reply(event.reply_token, "❌ 金鑰錯誤，請重新確認")
                return

            # 解析點數並預測
            pts = parse_last_hand_points(txt)
            if pts:
                p_pts, b_pts = pts
                gate = trial_guard(sess)
                if gate:
                    _reply(event.reply_token, gate)
                    return
                _handle_points_and_predict(sess, p_pts, b_pts, event.reply_token)
                save_session(uid, sess)
                return

            # 其他指令（簡化）
            if txt in ("遊戲設定", "開始分析"):
                left = trial_left_minutes(sess)
                _reply(event.reply_token, f"請回覆上局點數（例如：閒6 莊5 / 65 / 和）\n⏳ 試用剩餘 {left} 分鐘")
                return

            _reply(event.reply_token, "請輸入上局點數（例：65 / 和 / 閒6莊5），或輸入「遊戲設定」")

    except Exception as e:
        log.error("LINE Webhook 初始化失敗，切換為 dummy 模式: %s", e)
        LINE_MODE = "dummy"

if LINE_MODE == "dummy":
    log.info("LINE Webhook 未啟用（dummy mode）；提供 /predict 做測試")

    @app.post("/predict")
    def predict_api():
        """
        測試端點：POST JSON {"uid":"test","text":"閒6莊5"} 或 {"p":6,"b":5}
        回傳與 LINE 相同的文字卡。
        """
        data = request.get_json(silent=True) or {}
        uid = str(data.get("uid") or "test")
        sess = get_session(uid)
        # 試用門
        gate = trial_guard(sess)
        if gate:
            return jsonify(ok=False, message=gate), 200

        p_pts = data.get("p"); b_pts = data.get("b")
        text = data.get("text")
        if text and (p_pts is None or b_pts is None):
            pts = parse_last_hand_points(str(text))
            if pts: p_pts, b_pts = pts
        if p_pts is None or b_pts is None:
            return jsonify(ok=False, message="請提供 {p,b} 或 text（例如 '閒6莊5' 或 '65' 或 '和'）"), 400

        try:
            pf_preds = PF.predict(sims_per_particle=int(os.getenv("PF_PRED_SIMS", "5")))
            counts = init_counts()
            dep_preds = probs_after_points(counts, int(p_pts), int(b_pts))
            p = (pf_preds + dep_preds) * 0.5

            choice, edge, bet_pct, reason = decide_only_bp(p)
            bankroll_now = int(sess.get("bankroll", 0))
            bet_amt = bet_amount(bankroll_now, bet_pct)
            msg = format_output_card(p, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE))

            # 更新顯示資訊
            if int(p_pts) == int(b_pts):
                sess["last_pts_text"] = "上局結果: 和局"
            else:
                sess["last_pts_text"] = f"上局結果: 閒 {int(p_pts)} 莊 {int(b_pts)}"
            sess["phase"] = "await_pts" if CONTINUOUS_MODE else "ready"
            save_session(uid, sess)

            return jsonify(ok=True, choice=choice, edge=edge, bet_pct=bet_pct, message=msg), 200
        except Exception as e:
            log.exception("predict error: %s", e)
            return jsonify(ok=False, message="計算錯誤"), 500

# ---- 方便本機啟動（Render 用 gunicorn 指令；本機可 python server.py）----
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))  # 本機沒有 PORT 就用 8000
    app.run(host="0.0.0.0", port=port)
