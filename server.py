# -*- coding: utf-8 -*-
"""server.py - 更新版本：獨立局預測（無趨勢記憶）"""
import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from deplete import init_counts, probs_after_points

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
VERSION = "bgs-independent-2025-10-02"

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
        # 清理過期的 fallback session
        now = time.time()
        for k in list(SESS_FALLBACK.keys()):
            v = SESS_FALLBACK.get(k)
            if isinstance(v, dict) and v.get("exp") and v["exp"] < now:
                del SESS_FALLBACK[k]
        if uid in SESS_FALLBACK and "phase" in SESS_FALLBACK[uid]:
            return SESS_FALLBACK[uid]
    # 初始化新的 session
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

    # 解析特殊格式（和局或具體點數）
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
        return (0, 1)  # 庄家勝利（閒0，莊1）
    if t in ("P", "閒", "闲"):
        return (1, 0)  # 閒家勝利（閒1，莊0）
    if t in ("T", "和"):
        return (0, 0)  # 和局

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
    # 備援模式: SmartDummyPF (獨立模式，不記錄歷史)
    class SmartDummyPF:
        def __init__(self):
            log.warning("使用 SmartDummyPF 備援模式 - 請檢查 OutcomePF 導入問題")
        def update_outcome(self, outcome):
            # 獨立模式下不更新記憶
            return
        def predict(self, **kwargs) -> np.ndarray:
            base = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
            # 先做軟化處理和基本概率
            base = base ** (1.0 / SOFT_TAU)
            base = base / base.sum()
            pT = base[2]
            # 維護和平局機率上下限
            if pT < TIE_MIN:
                base[2] = TIE_MIN
                scale = (1.0 - TIE_MIN) / (1.0 - pT) if pT < 1.0 else 1.0
                base[0] *= scale
                base[1] *= scale
            elif pT > TIE_MAX:
                base[2] = TIE_MAX
                scale = (1.0 - TIE_MAX) / (1.0 - pT)
                base[0] *= scale
                base[1] *= scale
            return base.astype(np.float32)
        @property
        def backend(self):
            return "smart-dummy"
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
    # 平滑機率 (混合理論機率) 以減少隨機波動影響
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    theo_probs = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
    smoothed_probs = 0.7 * np.array([pB, pP, pT]) + 0.3 * theo_probs
    smoothed_probs = smoothed_probs / smoothed_probs.sum()
    pB, pP = float(smoothed_probs[0]), float(smoothed_probs[1])

    evB = 0.95 * pB - pP  # Banker 賭注的期望值
    evP = pP - pB        # Player 賭注的期望值

    side = 0 if evB > evP else 1  # 選擇期望值較大的賭邊 (莊=0, 閒=1)
    final_edge = max(abs(evB), abs(evP))

    if final_edge < EDGE_ENTER:
        return ("觀望", final_edge, 0.0, "⚪ 優勢不足")

    # 根據信心度(final_edge)計算投注比例，範圍5%到40%
    max_edge = 0.15  # 假設最大信心度為0.15
    min_bet_pct = 0.05
    max_bet_pct = 0.40
    bet_pct = min_bet_pct + (max_bet_pct - min_bet_pct) * (final_edge - EDGE_ENTER) / (max_edge - EDGE_ENTER)
    bet_pct = float(min(max_bet_pct, max(min_bet_pct, bet_pct)))  # 限制在5%-40%
    reason = f"信心度配注({int(min_bet_pct*100)}%~{int(max_bet_pct*100)}%)"
    return (INV[side], final_edge, bet_pct, reason)

def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int, cont: bool) -> str:
    b_pct_txt = f"{prob[0] * 100:.2f}%"
    p_pct_txt = f"{prob[1] * 100:.2f}%"
    header: List[str] = []
    if last_pts_text:
        header.append(last_pts_text)
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

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None
line_handler = None

GAMES = {
    "1": "WM", "2": "PM", "3": "DG", "4": "SA", "5": "KU",
    "6": "歐博/卡利", "7": "KG", "8": "全利", "9": "名人", "10": "MT真人",
}

def game_menu_text(left_min: int) -> str:
    lines = ["請選擇遊戲館別"]
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

    if outcome == 2:
        sess["last_pts_text"] = "上局結果: 和局"
    else:
        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
    sess["last_outcome"] = outcome
    sess["streak_count"] = 1 if outcome in (0, 1) else 0
    sess["phase"] = "ready"

    try:
        predict_start = time.time()
        # 每局獨立 (不更新歷史)
        pf_preds = PF.predict(sims_per_particle=int(os.getenv("PF_PRED_SIMS", "5")))
        log.info("預測完成, 耗時: %.2fs", time.time() - predict_start)
        counts = init_counts()
        dep_preds = probs_after_points(counts, p_pts, b_pts)
        p = (pf_preds + dep_preds) * 0.5

        choice, edge, bet_pct, reason = decide_only_bp(p)
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
