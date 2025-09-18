"""
server.py — 連續模式修正版（Render 優化版 + 勝率→金額 + 和局穩定器）

- Render 免費版資源優化（輕量 PF）
- 勝率→配注金額（5%~40% 線性）
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

# 勝率→配注（線性）：環境變數可調
USE_WINRATE_MAP = env_flag("USE_WINRATE_MAP", 1)
BET_MIN_PCT = float(os.getenv("BET_MIN_PCT", "0.05"))   # 5%
BET_MAX_PCT = float(os.getenv("BET_MAX_PCT", "0.40"))   # 40%
WINRATE_FLOOR = float(os.getenv("WINRATE_FLOOR", "0.50"))
WINRATE_CEIL  = float(os.getenv("WINRATE_CEIL",  "0.75"))

# 和局穩定器 + 機率平滑（新增）
TIE_PROB_MIN = float(os.getenv("TIE_PROB_MIN", "0.02"))   # 夾底
TIE_PROB_MAX = float(os.getenv("TIE_PROB_MAX", "0.12"))   # 夾頂
POST_TIE_COOLDOWN = int(os.getenv("POST_TIE_COOLDOWN", "1"))  # 和局後觀望 N 手；0=關閉
PROB_SMA_ALPHA = float(os.getenv("PROB_SMA_ALPHA", os.getenv("PROB_SMA_ALPHA".lower(), "0")))

CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)
INV = {0: "莊", 1: "閒"}

def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0:
        return 0
    return int(round(bankroll * pct))

def decide_only_bp(prob: np.ndarray) -> Tuple[str, float, float, str]:
    """根據信心度（優勢）來決定下注金額"""
    pB, pP = float(prob[0]), float(prob[1])
    side = 0 if pB >= pP else 1
    p_star = max(pB, pP)

    # 計算優勢（信心度）
    evB, evP = 0.95 * pB - pP, pP - pB
    final_edge = max(abs(evB), abs(evP))

    # 根據信心度（優勢）來調整下注比例
    if final_edge >= 0.10:  # 高信心
        bet_pct = 0.30  # 高信心下注30%
        reason = "高信心"
    elif final_edge >= 0.07:  # 中等信心
        bet_pct = 0.20  # 中等信心下注20%
        reason = "中等信心"
    elif final_edge >= 0.04:  # 低信心
        bet_pct = 0.10  # 低信心下注10%
        reason = "低信心"
    else:  # 非常低信心
        bet_pct = 0.05  # 非常低信心下注5%
        reason = "非常低信心"

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
            QuickReplyButton(action=MessageAction(label="結束分析 🧹", text="結束分析"))),
            QuickReplyButton(action=MessageAction(label="報莊勝 🅱️", text="B"))),
            QuickReplyButton(action=MessageAction(label="報閒勝 🅿️", text="P"))),
            QuickReplyButton(action=MessageAction(label="報和局 ⚪", text="T"))),
        ]
        if CONTINUOUS_MODE == 0:
            items.insert(0, QuickReplyButton(action=MessageAction(label="開始分析 ▶️", text="開始分析"))))
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

def _handle_points_and_predict(sess: Dict[str, Any], p_pts: int, b_pts: int, reply_token: str):
    log.info("開始處理點數預測: 閒%d 莊%d", p_pts, b_pts)
    start_time = time.time()

    # ---- 更新上一局結果 ----
    if p_pts == b_pts:
        sess["last_pts_text"] = "上局結果: 和局"
        sess["post_tie_cooldown"] = POST_TIE_COOLDOWN  # 新增：和局後冷卻
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

        # --- 機率平滑（滑動平均） ---
        p = np.asarray(p, dtype=np.float32)
        if PROB_SMA_ALPHA > 0:
            last_p = np.asarray(sess.get("last_prob") or p, dtype=np.float32)
            p = (1 - PROB_SMA_ALPHA) * last_p + PROB_SMA_ALPHA * p

        # --- 溫度縮放（讓分布更穩/尖） ---
        if PROB_TEMP > 0 and abs(PROB_TEMP - 1.0) > 1e-6:
            logits = np.log(np.clip(p, 1e-6, 1.0))
            p = np.exp(logits / PROB_TEMP)
            p = p / np.sum(p)

        # --- 和局機率夾緊（避免 T 過低/過高扭曲） ---
        try:
            pB, pP, pT = float(p[0]), float(p[1]), float(p[2])
            pT = min(max(pT, TIE_PROB_MIN), TIE_PROB_MAX)
            # 把剩餘機率按 B/P 比例回分配
            rest = max(1e-6, 1.0 - pT)
            bp_sum = max(1e-6, pB + pP)
            b_share = pB / bp_sum
            pB = rest * b_share
            pP = rest * (1.0 - b_share)
            p = np.array([pB, pP, pT], dtype=np.float32)
            p = p / np.sum(p)
        except Exception as _:
            pass

        # 存回平滑後值供下輪使用
        sess["last_prob"] = p.tolist()

        # --- 出手決策（含和局冷卻） ---
        choice, edge, bet_pct, reason = decide_only_bp(p)

        # 和局冷卻：>0 則本手觀望
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

# ---- Main ----
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s (CONTINUOUS_MODE=%s)", VERSION, port, CONTINUOUS_MODE)
    app.run(host="0.0.0.0", port=port, debug=False)
