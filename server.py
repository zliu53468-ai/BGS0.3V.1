# -*- coding: utf-8 -*-
"""server.py — Updated version for independent round predictions (no trend memory)
Stage overrides patch:
- 新增『前/中/後期』分段切換：支援 EARLY_*/MID_*/LATE_* 覆蓋同名環境變數
- 以手數分段（預設：<=20 EARLY, <=60 MID, 其餘 LATE）；可改為剩餘牌數近似模式
- 其它原本邏輯（PF、deplete 混合 0.5/0.5、決策流程、UI…）完全保留
"""
import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

# === 原：THEO_BLEND（仍保留）; 擴充 smooth_probs 支援傳入權重 ===
THEO_BLEND = float(os.getenv("THEO_BLEND", "0.0"))

def smooth_probs(prob: np.ndarray, theo_blend: Optional[float] = None) -> np.ndarray:
    """
    依 THEO_BLEND 將模型輸出機率與理論分佈混合，並正規化。
    可傳入 theo_blend 覆蓋（供階段覆蓋用）。None 時使用全域 THEO_BLEND。
    """
    tb = THEO_BLEND if theo_blend is None else float(theo_blend)
    if tb <= 0.0:
        return prob
    theo = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
    sm = (1.0 - tb) * prob + tb * theo
    sm = sm / sm.sum()
    return sm

# --- 安全導入 deplete（有就用，沒有不會掛） ---
DEPLETE_OK = False
init_counts = None
probs_after_points = None
try:
    from deplete import init_counts, probs_after_points  # type: ignore
    DEPLETE_OK = True
except Exception:
    try:
        from bgs.deplete import init_counts, probs_after_points  # type: ignore
        DEPLETE_OK = True
    except Exception:
        try:
            _cur_dir = os.path.dirname(os.path.abspath(__file__))
            if _cur_dir not in sys.path:
                sys.path.insert(0, _cur_dir)
            from deplete import init_counts, probs_after_points  # type: ignore
            DEPLETE_OK = True
        except Exception:
            DEPLETE_OK = False

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
    def jsonify(*args, **kwargs): raise RuntimeError("Flask is not available; jsonify cannot be used.")
    def abort(*args, **kwargs): raise RuntimeError("Flask is not available; abort cannot be used.")
    def CORS(app): return None

# 版本號
VERSION = "bgs-independent-2025-10-04+stage-overrides"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

if not DEPLETE_OK:
    log.warning("deplete 模組未找到；將以 PF 單模預測運行（功能不會中斷）。")

# ---------- Flask ----------
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

# ---------- Redis / Session ----------
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
KV_FALLBACK: Dict[str, str] = {}
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "1200"))
DEDUPE_TTL = 60

def _rget(k: str) -> Optional[str]:
    try:
        if redis_client: return redis_client.get(k)
        return KV_FALLBACK.get(k)
    except Exception as e:
        log.warning("[Redis] GET err: %s", e)
        return None

def _rset(k: str, v: str, ex: Optional[int] = None):
    try:
        if redis_client:
            redis_client.set(k, v, ex=ex)
        else:
            KV_FALLBACK[k] = v
    except Exception as e:
        log.warning("[Redis] SET err: %s", e)

def _rsetnx(k: str, v: str, ex: int) -> bool:
    try:
        if redis_client:
            return bool(redis_client.set(k, v, ex=ex, nx=True))
        if k in KV_FALLBACK:
            return False
        KV_FALLBACK[k] = v
        return True
    except Exception as e:
        log.warning("[Redis] SETNX err: %s", e)
        return True

def get_session(uid: str) -> Dict[str, Any]:
    if redis_client:
        j = _rget(f"bgs_session:{uid}")
        if j:
            try: return json.loads(j)
            except Exception: pass
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
        "last_pts_text": None, "table_no": None, "streak_count": 0,
        "last_outcome": None, "hand_count": 0, "prob_sma": None,
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
    try:
        return 1 if int(float(v)) != 0 else 0
    except Exception:
        return 1 if default else 0

# ---------- PF / 決策全域（原樣） ----------
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
        log.info("PF 初始化成功: n_particles=%s, sims_lik=%s, decks=%s (backend=%s)",
                 getattr(PF, 'n_particles', 'N/A'),
                 getattr(PF, 'sims_lik', 'N/A'),
                 getattr(PF, 'decks', 'N/A'),
                 getattr(PF, 'backend', 'unknown'))
    except Exception as e:
        log.error("PF 初始化失敗: %s", e)
        pf_initialized = False
        OutcomePF = None

if not pf_initialized:
    class SmartDummyPF:
        def __init__(self):
            log.warning("使用 SmartDummyPF 備援模式 - 請檢查 OutcomePF 導入問題")
        def update_outcome(self, outcome): return
        def predict(self, **kwargs) -> np.ndarray:
            base = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
            base = base ** (1.0 / SOFT_TAU)
            base = base / base.sum()
            pT = float(base[2])
            if pT < TIE_MIN:
                base[2] = TIE_MIN
                scale = (1.0 - TIE_MIN) / (1.0 - pT) if pT < 1.0 else 1.0
                base[0] *= scale; base[1] *= scale
            elif pT > TIE_MAX:
                base[2] = TIE_MAX
                scale = (1.0 - TIE_MAX) / (1.0 - pT) if pT < 1.0 else 1.0
                base[0] *= scale; base[1] *= scale
            return base.astype(np.float32)
        @property
        def backend(self): return "smart-dummy"
    PF = SmartDummyPF()
    pf_initialized = True
    log.warning("PF 初始化失敗，使用 SmartDummyPF 備援模式")

# ---------- 決策參數（原樣） ----------
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY = env_flag("USE_KELLY", 0)
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)

DECISION_MODE = os.getenv("DECISION_MODE", "ev").lower()  # ev | prob | hybrid
BANKER_PAYOUT = float(os.getenv("BANKER_PAYOUT", "0.95"))
PROB_MARGIN = float(os.getenv("PROB_MARGIN", "0.02"))
MIN_EV_EDGE = float(os.getenv("MIN_EV_EDGE", "0.0"))

MIN_CONF_FOR_ENTRY = float(os.getenv("MIN_CONF_FOR_ENTRY", "0.56"))
QUIET_SMALLEdge   = env_flag("QUIET_SMALLEdge", 0)

MIN_BET_PCT_ENV   = float(os.getenv("MIN_BET_PCT", "0.05"))
MAX_BET_PCT_ENV   = float(os.getenv("MAX_BET_PCT", "0.40"))
MAX_EDGE_SCALE    = float(os.getenv("MAX_EDGE_FOR_FULLBET", "0.15"))

SHOW_CONF_DEBUG   = env_flag("SHOW_CONF_DEBUG", 1)
LOG_DECISION      = env_flag("LOG_DECISION", 1)

INV = {0: "莊", 1: "閒"}

def bet_amount(bankroll: int, pct: float) -> int:
    if not bankroll or bankroll <= 0 or pct <= 0: return 0
    return int(round(bankroll * pct))

def _decide_side_by_ev(pB: float, pP: float) -> Tuple[int, float, float, float]:
    evB = BANKER_PAYOUT * pB - pP
    evP = pP - pB
    side = 0 if evB > evP else 1
    final_edge = max(abs(evB), abs(evP))
    return side, final_edge, evB, evP

def _decide_side_by_prob(pB: float, pP: float) -> int:
    return 0 if pB >= pP else 1

def decide_only_bp(prob: np.ndarray) -> Tuple[str, float, float, str]:
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])
    reason_parts: List[str] = []

    if DECISION_MODE == "prob":
        side = _decide_side_by_prob(pB, pP)
        _, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
        final_edge = max(abs(evB), abs(evP))
        reason_parts.append(f"模式=prob (pB={pB:.4f}, pP={pP:.4f})")
    elif DECISION_MODE == "hybrid":
        if abs(pB - pP) >= PROB_MARGIN:
            side = _decide_side_by_prob(pB, pP)
            _, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
            final_edge = max(abs(evB), abs(evP))
            reason_parts.append(f"模式=hybrid→prob (Δ={abs(pB-pP):.4f}≥{PROB_MARGIN})")
        else:
            ev_side, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
            if edge_ev >= MIN_EV_EDGE:
                side = ev_side
                final_edge = edge_ev
                reason_parts.append(f"模式=hybrid→ev (edge={edge_ev:.4f}≥{MIN_EV_EDGE})")
            else:
                side = _decide_side_by_prob(pB, pP)
                final_edge = edge_ev
                reason_parts.append(f"模式=hybrid→prob (EV不足 {edge_ev:.4f}<{MIN_EV_EDGE})")
    else:  # ev
        side, final_edge, evB, evP = _decide_side_by_ev(pB, pP)
        reason_parts.append(f"模式=ev (EV_B={evB:.4f}, EV_P={evP:.4f}, payout={BANKER_PAYOUT})")

    conf = max(pB, pP)
    if conf < MIN_CONF_FOR_ENTRY:
        reason_parts.append(f"⚪ 信心不足 conf={conf:.3f}<{MIN_CONF_FOR_ENTRY:.2f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason_parts))

    if final_edge < EDGE_ENTER:
        reason_parts.append(f"⚪ 優勢不足 edge={final_edge:.4f}<{EDGE_ENTER:.4f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason_parts))

    if QUIET_SMALLEdge and final_edge < (EDGE_ENTER * 1.2):
        reason_parts.append(f"⚪ 邊際略優(quiet) edge={final_edge:.4f}<{EDGE_ENTER*1.2:.4f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason_parts))

    min_b = max(0.0, min(1.0, MIN_BET_PCT_ENV))
    max_b = max(min_b, min(1.0, MAX_BET_PCT_ENV))
    max_edge = max(EDGE_ENTER + 1e-6, MAX_EDGE_SCALE)

    bet_pct = min_b + (max_b - min_b) * (final_edge - EDGE_ENTER) / (max_edge - EDGE_ENTER)
    bet_pct = float(min(max_b, max(min_b, bet_pct)))

    reason_parts.append(f"信心度配注({int(min_b*100)}%~{int(max_b*100)}%), conf={conf:.3f}")
    return (INV[side], final_edge, bet_pct, "; ".join(reason_parts))

def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int, cont: bool) -> str:
    b_pct_txt = f"{prob[0] * 100:.2f}%"
    p_pct_txt = f"{prob[1] * 100:.2f}%"
    header: List[str] = []
    if last_pts_text: header.append(last_pts_text)
    header.append("開始分析下局....")
    block = [
        "預測結果",
        f"閒：{p_pct_txt}",
        f"莊：{b_pct_txt}",
        f"和：{prob[2] * 100:.2f}%",
    ]
    if choice == "觀望":
        block.append("本次預測結果：觀望")
        block.append("建議觀望（不下注）")
    else:
        block.append(f"本次預測結果：{choice}")
        block.append(f"建議下注：{bet_amt:,}")

    if cont:
        block.append("\n📌 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）")
    return "\n".join(header + [""] + block)

# ===================== 〈新增〉階段覆蓋機制 =====================

def _as_flag(v: str) -> int:
    vv = str(v).strip().lower()
    if vv in ("1","true","t","yes","y","on"): return 1
    if vv in ("0","false","f","no","n","off"): return 0
    try: return 1 if int(float(vv)) != 0 else 0
    except: return 0

def get_stage(hand_count: int) -> str:
    """
    預設用『手數』分段：
        <= EARLY_MAX_HAND(20): EARLY
        <= MID_MAX_HAND(60):   MID
        else:                  LATE
    可設 STAGE_MODE=remaining 使用『近似剩餘牌數』：
        估每局 ~4.8 張，總牌=DECKS*52。
        以 REMAINING_MID_CARDS / REMAINING_LATE_CARDS 做門檻。
    """
    mode = os.getenv("STAGE_MODE", "hands").lower()
    if mode == "remaining":
        decks = int(os.getenv("DECKS", "8"))
        total = decks * 52
        avg_cards_per_hand = float(os.getenv("AVG_CARDS_PER_HAND", "4.8"))
        used = int(round(hand_count * avg_cards_per_hand))
        remaining = max(0, total - used)
        mid_thr  = int(os.getenv("REMAINING_MID_CARDS", str(int(total*0.75))))   # 例如 >312 張算 EARLY
        late_thr = int(os.getenv("REMAINING_LATE_CARDS", str(int(total*0.35))))  # 例如 <=145 張算 LATE
        if remaining > mid_thr:  return "EARLY"
        if remaining > late_thr: return "MID"
        return "LATE"
    else:
        emax = int(os.getenv("EARLY_MAX_HAND", "20"))
        mmax = int(os.getenv("MID_MAX_HAND", "60"))
        if hand_count <= emax: return "EARLY"
        if hand_count <= mmax: return "MID"
        return "LATE"

# 會被覆蓋的鍵（同名環境變數）
_STAGE_KEYS = [
    # 決策與門檻
    "DECISION_MODE","STRICT_PROB_ONLY","DISABLE_EV","PROB_MARGIN","MIN_EV_EDGE",
    "MIN_CONF_FOR_ENTRY","EDGE_ENTER","QUIET_SMALLEdge",
    # 平滑/和/理論
    "SOFT_TAU","PROB_SMA_ALPHA","TIE_MIN","TIE_MAX","TIE_PROB_MAX","THEO_BLEND",
    # PF / deplete（不重建 PF；僅採用可動態的）
    "PF_PRED_SIMS","DEPLETEMC_SIMS","DEPL_FACTOR",
]

def get_stage_overrides(stage: str) -> Dict[str, str]:
    """
    讀取 EARLY_*/MID_*/LATE_* 對應的覆蓋值；只收在 _STAGE_KEYS 內的鍵。
    """
    prefix = f"{stage}_"
    out: Dict[str, str] = {}
    for k in _STAGE_KEYS:
        v = os.getenv(prefix + k)
        if v is not None:
            out[k] = v
    return out

class StageEnv:
    """
    暫時覆蓋全域變數供決策使用；離開時自動還原。
    僅覆蓋 _STAGE_KEYS 中屬於全域的那幾個（不含 PF_PRED_SIMS 等非全域）。
    """
    def __init__(self, over: Dict[str, str]):
        self.over = over
        self.saved: Dict[str, Any] = {}

    def __enter__(self):
        g = globals()
        for k, v in self.over.items():
            if k not in g:  # PF_PRED_SIMS 等不是全域；略過
                continue
            self.saved[k] = g[k]
            if k in ("STRICT_PROB_ONLY","DISABLE_EV","QUIET_SMALLEdge"):
                g[k] = _as_flag(v)
            elif k in ("PROB_MARGIN","MIN_EV_EDGE","MIN_CONF_FOR_ENTRY","EDGE_ENTER",
                       "SOFT_TAU","PROB_SMA_ALPHA","TIE_MIN","TIE_MAX","TIE_PROB_MAX","THEO_BLEND"):
                g[k] = float(v)
            elif k == "DECISION_MODE":
                g[k] = str(v).lower()
        # 互斥語義（如果使用者給 STRICT_PROB_ONLY=1 或 DISABLE_EV=1 → 強制 prob）
        if globals().get("STRICT_PROB_ONLY", 0) or globals().get("DISABLE_EV", 0):
            self.saved.setdefault("DECISION_MODE", globals()["DECISION_MODE"])
            globals()["DECISION_MODE"] = "prob"
        return self

    def __exit__(self, exc_type, exc, tb):
        g = globals()
        for k, v in self.saved.items():
            g[k] = v

# 會話級 EMA（擴充：可帶覆蓋）
def apply_session_ema_smoothing(current_prob: np.ndarray, session: Dict[str, Any],
                                outcome: int,
                                alpha: Optional[float] = None,
                                skip_tie_upd: Optional[int] = None) -> np.ndarray:
    PROB_SMA_ALPHA_val = float(os.getenv("PROB_SMA_ALPHA", "0.3")) if alpha is None else float(alpha)
    SKIP_TIE_UPD_val = env_flag("SKIP_TIE_UPD", 1) if skip_tie_upd is None else int(skip_tie_upd)

    if PROB_SMA_ALPHA_val <= 0.0:
        return current_prob

    prev_smoothed = session.get("prob_sma")
    if outcome == 2 and SKIP_TIE_UPD_val and prev_smoothed is not None:
        return np.array(prev_smoothed, dtype=np.float32)

    if prev_smoothed is None:
        smoothed = current_prob
    else:
        prev_smoothed = np.array(prev_smoothed, dtype=np.float32)
        smoothed = PROB_SMA_ALPHA_val * current_prob + (1 - PROB_SMA_ALPHA_val) * prev_smoothed
        smoothed = smoothed / smoothed.sum()

    session["prob_sma"] = smoothed.tolist()
    return smoothed

# ---------- 健康檢查 ----------
@app.get("/")
def root():
    ua = request.headers.get("User-Agent", "")
    if "UptimeRobot" in ua: return "OK", 200
    status = "OK" if pf_initialized else "BACKUP_MODE"
    return f"✅ BGS Server {status} ({VERSION})", 200

@app.get("/health")
def health():
    return jsonify(ok=True, ts=time.time(), version=VERSION,
                   pf_initialized=pf_initialized, pf_backend=getattr(PF, 'backend', 'unknown')), 200

@app.get("/healthz")
def healthz():
    return jsonify(ok=True, ts=time.time(), version=VERSION, pf_initialized=pf_initialized), 200

# ---------- LINE Bot ----------
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
line_api = None
line_handler = None
if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    log.error("LINE credentials missing. SECRET set? %s, TOKEN set? %s",
              bool(LINE_CHANNEL_SECRET), bool(LINE_CHANNEL_ACCESS_TOKEN))

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
    if not event_id: return True
    return _rsetnx(f"dedupe:{event_id}", "1", DEDUPE_TTL)

# ===================== 核心：套用分段覆蓋 =====================
def _handle_points_and_predict(sess: Dict[str, Any], p_pts: int, b_pts: int, reply_token: str):
    log.info("開始處理點數預測: 閒%d 莊%d (deplete=%s, mode=%s)", p_pts, b_pts, DEPLETE_OK, DECISION_MODE)
    start_time = time.time()
    outcome = 2 if p_pts == b_pts else (1 if p_pts > b_pts else 0)

    # 非和局時累計手數（供分段判定）
    if outcome != 2:
        sess["hand_count"] = int(sess.get("hand_count", 0)) + 1
    hand_count = int(sess.get("hand_count", 0))

    if outcome == 2:
        sess["last_pts_text"] = "上局結果: 和局"
    else:
        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
    sess["last_outcome"] = outcome
    sess["streak_count"] = 1 if outcome in (0, 1) else 0
    sess["phase"] = "ready"

    try:
        # 1) 取得當前階段＋覆蓋值
        stage = get_stage(hand_count)
        over = get_stage_overrides(stage)
        log.info("當前階段: %s (hand=%d) 覆蓋鍵: %s", stage, hand_count, ",".join(over.keys()) or "-")

        # 2) PF.update_points（若有）
        try:
            if hasattr(PF, "update_points"):
                PF.update_points(int(p_pts), int(b_pts))
                log.info("PF.update_points 已餵入點數: P=%d, B=%d", p_pts, b_pts)
        except Exception as e:
            log.warning("PF.update_points 失敗: %s", e)

        # 3) PF 預測（PF_PRED_SIMS 可被階段覆蓋）
        pf_pred_sims = int(over.get("PF_PRED_SIMS", os.getenv("PF_PRED_SIMS", "5")))
        t0 = time.time()
        pf_preds = PF.predict(sims_per_particle=pf_pred_sims)
        log.info("PF 預測完成, 耗時: %.2fs (PF_PRED_SIMS=%d)", time.time() - t0, pf_pred_sims)

        p = pf_preds

        # 4) deplete 混合（DEPLETEMC_SIMS/DEPL_FACTOR 可被階段覆蓋；仍維持 0.5/0.5）
        if DEPLETE_OK and init_counts and probs_after_points:
            try:
                base_decks = int(os.getenv("DECKS", "8"))
                counts = init_counts(base_decks)
                deplete_sims = int(over.get("DEPLETEMC_SIMS", os.getenv("DEPLETEMC_SIMS", "1000")))
                deplete_factor = float(over.get("DEPL_FACTOR", os.getenv("DEPL_FACTOR", "1.0")))
                dep_preds = probs_after_points(counts, p_pts, b_pts, sims=deplete_sims, deplete_factor=deplete_factor)

                # 若外部宣告 Deplete 順序為 [P,B,T] → 轉為 [B,P,T]
                if os.getenv("DEPLETE_RETURNS_PBT", "0") == "1":
                    dep_preds = [dep_preds[1], dep_preds[0], dep_preds[2]]

                p = (pf_preds + np.asarray(dep_preds, dtype=np.float32)) * 0.5
                log.info("Deplete 混合完成: sims=%d factor=%.2f -> B=%.4f P=%.4f T=%.4f",
                         deplete_sims, deplete_factor, p[0], p[1], p[2])
            except Exception as e:
                log.warning("Deplete 模擬失敗，改用 PF 單模：%s", e)

        # 5) 理論混合（THEO_BLEND 可被階段覆蓋）
        tb = over.get("THEO_BLEND")
        p_theo = smooth_probs(p, theo_blend=float(tb) if tb is not None else None)

        # 6) 會話 EMA（PROB_SMA_ALPHA / SKIP_TIE_UPD 可被階段覆蓋）
        alpha = over.get("PROB_SMA_ALPHA")
        p_final = apply_session_ema_smoothing(
            p_theo, sess, outcome,
            alpha=float(alpha) if alpha is not None else None,
            skip_tie_upd=_as_flag(over["SKIP_TIE_UPD"]) if "SKIP_TIE_UPD" in over else None
        )

        # 7) 以 StageEnv 暫時覆蓋決策相關全域 → 呼叫既有 decide_only_bp()
        with StageEnv(over):
            choice, edge, bet_pct, reason = decide_only_bp(p_final)

        bankroll_now = int(sess.get("bankroll", 0))
        bet_amt = bet_amount(bankroll_now, bet_pct)
        msg = format_output_card(p_final, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE))
        _reply(reply_token, msg)

        if LOG_DECISION or SHOW_CONF_DEBUG:
            log.info("決策: %s edge=%.4f pct=%.2f%% | %s", choice, edge, bet_pct*100, reason)

        log.info("完整處理完成, 總耗時: %.2fs (stage=%s)", time.time() - start_time, stage)

    except Exception as e:
        log.error("預測過程中錯誤: %s", e)
        _reply(reply_token, "⚠️ 預計算錯誤，請稍後再試")

    if CONTINUOUS_MODE:
        sess["phase"] = "await_pts"

# ---- LINE Handler / Webhook ----
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
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
            _ = trial_persist_guard(uid)
            sess = get_session(uid)
            _reply(event.reply_token,
                   "👋 歡迎！請輸入『遊戲設定』開始；已啟用連續模式，之後只需輸入點數（例：65 / 和 / 閒6莊5）即可自動預測。")
            save_session(uid, sess)

        @line_handler.add(MessageEvent, message=TextMessage)
        def on_text(event):
            if not _dedupe_event(getattr(event, "id", None)): return
            uid = event.source.user_id
            raw = (event.message.text or "")
            text = re.sub(r"\s+", " ", raw.replace("\u3000", " ").strip())
            sess = get_session(uid)
            try:
                log.info("[LINE] uid=%s phase=%s text=%s", uid, sess.get("phase"), text)
                up = text.upper()

                # 開通
                if up.startswith("開通") or up.startswith("ACTIVATE"):
                    after = text[2:] if up.startswith("開通") else text[len("ACTIVATE"):]
                    ok = validate_activation_code(after)
                    if ok:
                        _rset(_trial_key(uid, "expired"), "0")
                    sess["premium"] = bool(ok)
                    _reply(event.reply_token, "✅ 已開通成功！" if ok else "❌ 密碼錯誤")
                    save_session(uid, sess); return

                # 永久試用鎖
                guard = trial_persist_guard(uid)
                if guard and not sess.get("premium", False):
                    _reply(event.reply_token, guard)
                    save_session(uid, sess); return

                # 結束/清空
                if up in ("結束分析", "清空", "RESET"):
                    premium = sess.get("premium", False)
                    start_ts = sess.get("trial_start", int(time.time()))
                    sess = get_session(uid)
                    sess["premium"] = premium
                    sess["trial_start"] = start_ts
                    sess["hand_count"] = 0
                    sess["prob_sma"] = None
                    _reply(event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。")
                    save_session(uid, sess); return

                # 遊戲設定
                if text == "遊戲設定" or up == "GAME SETTINGS":
                    sess["phase"] = "choose_game"
                    sess["game"] = None; sess["table"] = None; sess["table_no"] = None
                    sess["bankroll"] = 0; sess["streak_count"] = 0
                    sess["last_outcome"] = None; sess["last_pts_text"] = None
                    sess["hand_count"] = 0; sess["prob_sma"] = None
                    first_ts = _rget(_trial_key(uid, "first_ts"))
                    if first_ts:
                        used = (int(time.time()) - int(first_ts)) // 60
                        left = max(0, TRIAL_MINUTES - used)
                    else:
                        left = TRIAL_MINUTES
                    menu = game_menu_text(left)
                    _reply(event.reply_token, menu)
                    save_session(uid, sess); return

                # 選館
                if sess.get("phase") == "choose_game":
                    m = re.match(r"^\s*(\d+)", text)
                    if m:
                        choice = m.group(1)
                        if choice in GAMES:
                            sess["game"] = GAMES[choice]
                            sess["phase"] = "input_bankroll"
                            _reply(event.reply_token, f"🎰 已選擇遊戲館：{sess['game']}\n請輸入初始籌碼（金額）")
                            save_session(uid, sess); return
                        else:
                            _reply(event.reply_token, "⚠️ 無效的選項，請輸入上列列出的數字。")
                            return
                    else:
                        _reply(event.reply_token, "⚠️ 請直接輸入提供的數字來選擇遊戲館別。")
                        return

                # 輸入籌碼
                if sess.get("phase") == "input_bankroll":
                    amount_str = re.sub(r"[^\d]", "", text)
                    amount = int(amount_str) if amount_str else 0
                    if amount <= 0:
                        _reply(event.reply_token, "⚠️ 請輸入正確的數字金額。"); return
                    sess["bankroll"] = amount
                    sess["phase"] = "await_pts"
                    sess["hand_count"] = 0
                    sess["prob_sma"] = None
                    _reply(event.reply_token,
                           f"✅ 設定完成！遊戲館：{sess.get('game')}，初始籌碼：{amount}。\n📌 連續模式已啟動：現在請直接輸入第一局點數進行分析（例：閒6莊5 或 65）。")
                    save_session(uid, sess); return

                # 解析點數並預測
                pts = parse_last_hand_points(text)
                if pts and sess.get("bankroll"):
                    _handle_points_and_predict(sess, pts[0], pts[1], event.reply_token)
                    save_session(uid, sess); return

                _reply(event.reply_token,
                       "指令無法辨識。\n📌 已啟用連續模式：直接輸入點數即可（例：65 / 和 / 閒6莊5）。\n或輸入『遊戲設定』。")
            except Exception as e:
                log.exception("on_text err: %s", e)
                try: _reply(event.reply_token, "⚠️ 系統錯誤，稍後再試。")
                except Exception: pass

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

# ---------- Trial / Activation（原樣） ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))
ADMIN_CONTACT = os.getenv("ADMIN_CONTACT", "@admin")
ADMIN_ACTIVATION_SECRET = os.getenv("ADMIN_ACTIVATION_SECRET", "aaa8881688")

def _trial_key(uid: str, kind: str) -> str:
    return f"trial:{kind}:{uid}"

def trial_persist_guard(uid: str) -> Optional[str]:
    now = int(time.time())
    first_ts = _rget(_trial_key(uid, "first_ts"))
    expired = _rget(_trial_key(uid, "expired"))
    if expired == "1":
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 在此輸入：開通 你的密碼"
    if not first_ts:
        _rset(_trial_key(uid, "first_ts"), str(now))
        return None
    try:
        first = int(first_ts)
    except:
        first = now
        _rset(_trial_key(uid, "first_ts"), str(now))
    used_min = (now - first) // 60
    if used_min >= TRIAL_MINUTES:
        _rset(_trial_key(uid, "expired"), "1")
        return f"⛔ 試用已到期\n📬 請聯繫管理員：{ADMIN_CONTACT}\n🔐 在此輸入：開通 你的密碼"
    return None

def validate_activation_code(code: str) -> bool:
    if not code: return False
    norm = str(code).replace("\u3000", " ").replace("：", ":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and (norm == ADMIN_ACTIVATION_SECRET)

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s (CONTINUOUS_MODE=%s, PF_INIT=%s, DEPLETE_OK=%s, MODE=%s)",
             VERSION, port, CONTINUOUS_MODE, pf_initialized, DEPLETE_OK, DECISION_MODE)
    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
