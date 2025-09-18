"""
server.py — 連續模式修正版（Render 優化版）含信心度配注

針對 Render 免費版資源限制進行優化：
  - 強制設置輕量級粒子過濾器參數
  - 添加詳細診斷日誌
  - 優化錯誤處理防止卡死
  - 備用 Dummy 模式確保基本功能
  - 新增信心度配注系統（5%-40%本金）
"""

import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
# Optional imports for optional dependencies.  Render free plans may not
# have redis or Flask installed.  Wrap the imports in try/except blocks
# and fall back to dummy objects when unavailable.
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
        raise RuntimeError("Flask is not available; jsonify cannot be used.")
    def abort(*args, **kwargs):  # type: ignore
        raise RuntimeError("Flask is not available; abort cannot be used.")
    def CORS(app):  # type: ignore
        # no‑op when Flask is absent
        return None


# 版本號
VERSION = "bgs-pf-confidence-betting-2025-09-18"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")


# ---------- Flask 初始化 ----------
if _flask_available and Flask is not None:
    # Initialise a real Flask application when Flask is installed
    app = Flask(__name__)
    CORS(app)
else:
    # Provide a dummy app object so that decorators do not raise
    class _DummyApp:
        """Fallback for when Flask is not available.

        Methods ``get`` and ``post`` return a decorator that simply
        returns the wrapped function unchanged, allowing route
        definitions to execute without a real server.  The ``run``
        method logs a warning instead of starting a server.
        """
        def get(self, *args, **kwargs):  # type: ignore
            def _decorator(func):
                return func
            return _decorator

        def post(self, *args, **kwargs):  # type: ignore
            def _decorator(func):
                return func
            return _decorator

        def run(self, *args, **kwargs):  # type: ignore
            log.warning("Flask not available; dummy app cannot run a server.")

    app = _DummyApp()


# ---------- Redis 或記憶體 Session ----------
REDIS_URL = os.getenv("REDIS_URL")
redis_client: Optional["redis.Redis"] = None  # type: ignore
if redis is not None and REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Successfully connected to Redis.")
    except Exception as e:
        # Fall back to in‑memory sessions if Redis connection fails
        redis_client = None
        log.error("Failed to connect to Redis: %s. Using in-memory session.", e)
else:
    # Either redis is not available or no URL provided
    if redis is None:
        log.warning("redis module not available; using in-memory session store.")
    elif not REDIS_URL:
        log.warning("REDIS_URL not set. Using in-memory session store.")

SESS_FALLBACK: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRE_SECONDS = 3600  # 1 小時
DEDUPE_TTL = 60  # 相同事件去重秒數


def _rget(k: str) -> Optional[str]:
    """從 Redis 取值，失敗時回傳 None。"""
    try:
        return redis_client.get(k) if redis_client else None
    except Exception as e:
        log.warning("[Redis] GET err: %s", e)
        return None


def _rset(k: str, v: str, ex: Optional[int] = None):
    """設定 Redis 的值，選擇性設定過期時間。"""
    try:
        if redis_client:
            redis_client.set(k, v, ex=ex)
    except Exception as e:
        log.warning("[Redis] SET err: %s", e)


def _rsetnx(k: str, v: str, ex: int) -> bool:
    """只在鍵不存在時設定值，並設定過期時間；失敗或例外回 True 避免阻擋。"""
    try:
        if redis_client:
            return bool(redis_client.set(k, v, ex=ex, nx=True))
        # fallback 模式下，若已存在則回 False
        if k in SESS_FALLBACK:
            return False
        SESS_FALLBACK[k] = {"v": v, "exp": time.time() + ex}
        return True
    except Exception as e:
        log.warning("[Redis] SETNX err: %s", e)
        return True


def get_session(uid: str) -> Dict[str, Any]:
    """取得或建立使用者 session。"""
    # 先嘗試從 Redis 讀取
    if redis_client:
        j = _rget(f"bgs_session:{uid}")
        if j:
            try:
                return json.loads(j)
            except Exception:
                pass
    else:
        # 清理過期的 fallback sessions
        now = time.time()
        for k in list(SESS_FALLBACK.keys()):
            v = SESS_FALLBACK.get(k)
            if isinstance(v, dict) and v.get("exp") and v["exp"] < now:
                del SESS_FALLBACK[k]
        if uid in SESS_FALLBACK and "phase" in SESS_FALLBACK[uid]:
            return SESS_FALLBACK[uid]
    # 新建 session
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
    """儲存 session 至 Redis 或 fallback。"""
    if redis_client:
        _rset(f"bgs_session:{uid}", json.dumps(data), ex=SESSION_EXPIRE_SECONDS)
    else:
        SESS_FALLBACK[uid] = data


def env_flag(name: str, default: int = 1) -> int:
    """解析環境變數為布林 flag。"""
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
    """
    將輸入文字解析為上一局點數 (P_total, B_total)。
    """
    if not text:
        return None
    # 將全形數字與冒號替換為半形
    s = str(text).translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))
    # 移除零寬字元、BOM 與換行符號等控制字元
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s)
    # 將全形空白 (\u3000) 轉為半形空白，避免影響正則匹配
    s = s.replace("\u3000", " ")
    u = s.upper().strip()
    # 剝掉前綴『開始分析』，支援「開始分析47」這類輸入
    u = re.sub(r"^開始分析", "", u)

    # 1) 判斷和局（TIE/DRAW/和9 這類）
    m = re.search(r"(?:和|TIE|DRAW)\s*:?:?\s*(\d)?", u)
    if m:
        d = m.group(1)
        return (int(d), int(d)) if d else (0, 0)

    # 2) 閒/莊格式：支援繁體/簡體以及 B/P 縮寫
    m = re.search(r"(?:閒|闲|P)\s*:?:?\s*(\d)\D+(?:莊|庄|B)\s*:?:?\s*(\d)", u)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|庄|B)\s*:?:?\s*(\d)\D+(?:閒|闲|P)\s*:?:?\s*(\d)", u)
    if m:
        return (int(m.group(2)), int(m.group(1)))

    # 3) 單字母快速回報（莊/閒/和），支援繁簡
    # 移除所有半形與全形空白後判斷
    t = u.replace(" ", "").replace("\u3000", "")
    if t in ("B", "莊", "庄"):
        return (0, 1)
    if t in ("P", "閒", "闲"):
        return (1, 0)
    if t in ("T", "和"):
        return (0, 0)

    # 4) 若包含英文字母（A-Z），視為桌號或其他指令，不解析為點數
    if re.search(r"[A-Z]", u):
        return None

    # 5) 最後僅在輸入中恰好包含兩個數字時，視為點數 (先閒後莊)
    digits = re.findall(r"\d", u)
    if len(digits) == 2:
        return (int(digits[0]), int(digits[1]))
    return None


# ---------- 試用/授權 ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "aaa8881688")


def validate_activation_code(code: str) -> bool:
    """驗證管理員提供的開通密碼。"""
    if not code:
        return False
    # 全形空白與冒號替換為半形
    norm = str(code).replace("\u3000", " ").replace("：", ":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and (norm == ADMIN_ACTIVATION_SECRET)


def trial_left_minutes(sess: Dict[str, Any]) -> int:
    """計算試用剩餘分鐘。若已開通 premium，回傳極大值。"""
    if sess.get("premium", False):
        return 9999
    now = int(time.time())
    used = (now - int(sess.get("trial_start", now))) // 60
    return max(0, TRIAL_MINUTES - used)


def trial_guard(sess: Dict[str, Any]) -> Optional[str]:
    """若試用已過期且未開通 premium，回傳警告文字。"""
    if sess.get("premium", False):
        return None
    if trial_left_minutes(sess) <= 0:
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 在此輸入：開通 你的密碼"
    return None


try:
    log.info("Activation secret loaded? %s (len=%d)", bool(ADMIN_ACTIVATION_SECRET), len(ADMIN_ACTIVATION_SECRET))
except Exception:
    pass


# ---------- Outcome PF (粒子過濾器) ----------
# 強制設置輕量級參數（針對 Render 免費版優化）
os.environ['PF_N'] = '30'
os.environ['PF_UPD_SIMS'] = '20'
os.environ['PF_PRED_SIMS'] = '0'
os.environ['DECKS'] = '6'

# Default backend to Monte‑Carlo to greatly reduce computational burden on
# resource‑constrained platforms.  If a caller explicitly sets
# ``PF_BACKEND`` in the environment it will override this value.
if not os.getenv('PF_BACKEND'):
    os.environ['PF_BACKEND'] = 'mc'

log.info("強制設置 PF 參數: PF_N=30, PF_UPD_SIMS=20, PF_PRED_SIMS=0, DECKS=6")

try:
    # Attempt to import OutcomePF from the ``bgs`` package first
    from bgs.pfilter import OutcomePF  # type: ignore
except Exception:
    try:
        # Fallback to a local ``pfilter`` module located in the same
        # directory as this file.  When running outside of a package
        # context, add the current directory to ``sys.path`` so that
        # ``import pfilter`` resolves correctly.
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        if _cur_dir not in sys.path:
            sys.path.insert(0, _cur_dir)
        from pfilter import OutcomePF  # type: ignore
        log.info("Imported OutcomePF from local pfilter module.")
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
        log.info(
            "PF 初始化成功: n_particles=%d, sims_lik=%d (backend=%s)",
            PF.n_particles,
            getattr(PF, "sims_lik", 0),
            getattr(PF, "backend", "unknown"),
        )
    except Exception as _e:
        log.error("Failed to initialise OutcomePF: %s", _e)
        OutcomePF = None

if not OutcomePF:
    # Provide a minimal dummy PF implementation as a safety net
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


# ---------- 投注決策 ----------
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY = env_flag("USE_KELLY", 0)  # 改為使用信心度配注
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR", "0.25"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.40"))  # 最高40%本金
MIN_BET_PCT = float(os.getenv("MIN_BET_PCT", "0.05"))  # 最低5%本金
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)  # 1=連續模式；0=舊流程

INV = {0: "莊", 1: "閒"}  # 添加缺失的 INV 映射


def bet_amount(bankroll: int, pct: float) -> int:
    """依本金與比例計算下注金額。"""
    if not bankroll or bankroll <= 0 or pct <= 0:
        return 0
    return int(round(bankroll * pct))


def calculate_confidence_bet_pct(edge: float, max_prob: float) -> float:
    """
    根據優勢和最大機率計算信心度配注比例
    edge: 邊際優勢 (0-1)
    max_prob: 最高機率 (0-1)
    回傳: 下注比例 (0.05-0.40)
    """
    # 基礎信心度：優勢越高，信心度越高
    base_confidence = min(1.0, edge * 10)  # 將優勢轉換為0-1的信心度
    
    # 機率信心度：機率越高，信心度越高
    prob_confidence = max(0, (max_prob - 0.5) * 2)  # 機率50%以上才有信心
    
    # 綜合信心度
    total_confidence = (base_confidence * 0.6 + prob_confidence * 0.4)
    
    # 映射到5%-40%的配注範圍
    bet_pct = MIN_BET_PCT + total_confidence * (MAX_BET_PCT - MIN_BET_PCT)
    
    return max(MIN_BET_PCT, min(MAX_BET_PCT, bet_pct))


def decide_only_bp(prob: np.ndarray) -> Tuple[str, float, float, str, float]:
    """根據閒、莊機率，決定下注方向與邊際與下注比例。"""
    pB, pP = float(prob[0]), float(prob[1])
    evB, evP = 0.95 * pB - pP, pP - pB
    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))
    
    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, "⚪ 優勢不足", 0.0)
    
    # 使用信心度配注系統
    max_prob = max(pB, pP)
    bet_pct = calculate_confidence_bet_pct(final_edge, max_prob)
    
    # 計算信心度百分比
    confidence_percent = (bet_pct - MIN_BET_PCT) / (MAX_BET_PCT - MIN_BET_PCT) * 100
    
    reason = f"信心度配注 {confidence_percent:.1f}% (優勢: {final_edge*100:.1f}%, 機率: {max_prob*100:.1f}%)"
    
    return (INV[side], final_edge, bet_pct, reason, confidence_percent)


def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], 
                      bet_amt: int, cont: bool, confidence: float, reason: str) -> str:
    """組合回覆文字。"""
    b_pct_txt = f"{prob[0] * 100:.2f}%"
    p_pct_txt = f"{prob[1] * 100:.2f}%"
    header: List[str] = []
    if last_pts_text:
        header.append(last_pts_text)
    header.append("開始分析下局....")
    block = [
        "【預測結果】",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"本次預測結果：{choice if choice != '觀望' else '觀'}",
        f"信心度：{confidence:.1f}%",
        f"建議下注：{bet_amt:,}",
        f"配注策略：{reason}",
    ]
    if cont:
        block.append("\n📌 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）")
    return "\n".join(header + [""] + block)


# ---------- 健康檢查路由 ----------
@app.get("/")
def root():
    ua = request.headers.get("User-Agent", "") if request else ""
    if "UptimeRobot" in ua:
        return "OK", 200
    return f"✅ BGS PF Server OK ({VERSION})", 200


@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION), 200


@app.get("/healthz")
def healthz():
    return jsonify(ok=True, ts=time.time(), version=VERSION), 200


# ---------- LINE Bot ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None
line_handler = None

GAMES = {
    "1": "WM",
    "2": "PM",
    "3": "DG",
    "4": "SA",
    "5": "KU",
    "6": "歐博/卡利",
    "7": "KG",
    "8": "全利",
    "9": "名人",
    "10": "MT真人",
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
    if not line_api:
        return
    from linebot.models import TextSendMessage
    try:
        line_api.reply_message(token, TextSendMessage(text=text, quick_reply=_quick_buttons()))
    except Exception as e:
        log.warning("[LINE] reply failed: %s", e)


def _dedupe_event(event_id: Optional[str]) -> bool:
    """避免處理重覆事件（LINE 會重送）。"""
    if not event_id:
        return True
    return _rsetnx(f"dedupe:{event_id}", "1", DEDUPE_TTL)


def _handle_points_and_predict(sess: Dict[str, Any], p_pts: int, b_pts: int, reply_token: str):
    """在連續模式或人工模式中處理點數並預測下一局。"""
    log.info("開始處理點數預測: 閒%d 莊%d", p_pts, b_pts)
    start_time = time.time()
    
    # 更新上一局結果
    if p_pts == b_pts:
        sess["last_pts_text"] = f"上局結果: 和局 (閒{p_pts} 莊{b_pts})"
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
    
    # 做預測
    sess["phase"] = "ready"
    try:
        predict_start = time.time()
        p = PF.predict(sims_per_particle=max(0, int(os.getenv("PF_PRED_SIMS", "0"))))
        log.info("預測完成, 耗時: %.2fs", time.time() - predict_start)
        
        choice, edge, bet_pct, reason, confidence = decide_only_bp(p)
        bankroll_now = int(sess.get("bankroll", 0))
        bet_amt = bet_amount(bankroll_now, bet_pct)
        
        msg = format_output_card(p, choice, sess.get("last_pts_text"), bet_amt, 
                               cont=bool(CONTINUOUS_MODE), confidence=confidence, reason=reason)
        _reply(reply_token, msg)
        log.info("完整處理完成, 總耗時: %.2fs", time.time() - start_time)
        
    except Exception as e:
        log.error("預測過程中錯誤: %s", e)
        _reply(reply_token, "⚠️ 預計算錯誤，請稍後再試")
    
    # 若為連續模式，保持在 await_pts 狀態，方便下一局直接輸入點數
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
            # 將全形空白變半形並合併多個空白為一個
            text = re.sub(r"\s+", " ", raw.replace("\u3000", " ")).strip()
            sess = get_session(uid)
            try:
                log.info("[LINE] uid=%s phase=%s text=%s", uid, sess.get("phase"), text)

                # --- 開通指令優先處理 ---
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

                # --- 點數輸入：連續模式下，在任何階段先嘗試解析點數 ---
                pts = parse_last_hand_points(raw)
                if pts is not None:
                    # 若尚未設定本金，提示先完成設定
                    if not sess.get("bankroll"):
                        _reply(event.reply_token, "請先完成『遊戲設定』與『本金設定』（例如輸入 5000），再回報點數。")
                        save_session(uid, sess)
                        return
                    _handle_points_and_predict(sess, int(pts[0]), int(pts[1]), event.reply_token)
                    save_session(uid, sess)
                    return

                # --- 遊戲設定流程入口 ---
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
                    # 選擇館別
                    if re.fullmatch(r"([1-9]|10)", text):
                        sess["game"] = GAMES[text]
                        sess["phase"] = "choose_table"
                        _reply(event.reply_token, f"✅ 已設定館別【{sess['game']}】\n請輸入桌號（例：DG01）")
                        save_session(uid, sess)
                        return

                elif phase == "choose_table":
                    # 設定桌號：格式為 2 英文 + 2 數字
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
                    # 設定本金
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

                # --- 兼容舊流程：開始分析XY ---
                norm = raw.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
                norm = re.sub(r"\s+", "", norm)
                m_ka = re.fullmatch(r"開始分析(\d)(\d)", norm)
                if m_ka and sess.get("bankroll"):
                    _handle_points_and_predict(sess, int(m_ka.group(1)), int(m_ka.group(2)), event.reply_token)
                    save_session(uid, sess)
                    return

                # --- 結束分析 / RESET ---
                if up in ("結束分析", "清空", "RESET"):
                    premium = sess.get("premium", False)
                    start_ts = sess.get("trial_start", int(time.time()))
                    sess = get_session(uid)
                    sess["premium"] = premium
                    sess["trial_start"] = start_ts
                    _reply(event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。")
                    save_session(uid, sess)
                    return

                # --- Fallback 提示 ---
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
