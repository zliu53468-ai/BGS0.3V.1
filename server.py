# -*- coding: utf-8 -*-
"""server.py — Updated version for independent round predictions (no trend memory)
Patched by ChatGPT:
- 新增 THEO_BLEND 環境變數：控制是否將模型機率與理論分佈混合（0.0=關閉）
- 新增 smooth_probs()：決策與卡片顯示統一使用同一組機率（避免顯示與下注邏輯不一致）
- 新增 Stage 分段切換（EARLY/MID/LATE），支援 <STAGE>_KEY 覆蓋
- 新增 會話級機率 EMA 平滑（PROB_SMA_ALPHA），和局可選擇不更新（SKIP_TIE_UPD）
- Deplete 讀取 DEPLETEMC_SIMS / DEPL_FACTOR 環境變數
"""
import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

# --- 新增：理論混合權重（0.0=關閉；建議 0.0~0.15）
THEO_BLEND = float(os.getenv("THEO_BLEND", "0.0"))
PROB_SMA_ALPHA = float(os.getenv("PROB_SMA_ALPHA", "0.0"))  # 0.0=關閉；建議 0.25~0.35
SKIP_TIE_UPD = 1 if str(os.getenv("SKIP_TIE_UPD", "1")).lower() in ("1","true","y","yes","on") else 0

def _theo_mix(prob: np.ndarray) -> np.ndarray:
    if THEO_BLEND <= 0.0:
        return prob
    theo = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
    sm = (1.0 - THEO_BLEND) * prob + THEO_BLEND * theo
    sm = sm / sm.sum()
    return sm

def _ema_update(prev: Optional[np.ndarray], now: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0 or prev is None:
        return now
    out = alpha * now + (1.0 - alpha) * prev
    s = out.sum()
    if s > 0:
        out = out / s
    return out.astype(np.float32)

def smooth_probs(prob: np.ndarray) -> np.ndarray:
    """
    先做 THEO_BLEND，再做會話級 EMA 平滑（若有設定 PROB_SMA_ALPHA）
    """
    sm = _theo_mix(prob.astype(np.float32))
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
VERSION = "bgs-independent-2025-11-02+blend-stage-ema"

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
KV_FALLBACK: Dict[str, str] = {}  # 持久鍵的記憶體替代（只有沒 Redis 時用）
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
        "last_outcome": None,
        # 會話級預測平滑用（上一手的平滑後機率）
        "_last_probs": None,
        "hand_idx": 0,
        "_stage": "MID",
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

# ---------- 分段切換（手數 or 剩餘牌） ----------
STAGE_MODE = os.getenv("STAGE_MODE", "count")  # count | remain
EARLY_HANDS = int(os.getenv("EARLY_HANDS", "15"))
LATE_HANDS  = int(os.getenv("LATE_HANDS",  "56"))
REMAIN_LATE = int(os.getenv("REMAIN_LATE", "120"))  # 若你有剩餘牌估計可用

def _stage_from_count(idx: int) -> str:
    if idx <= EARLY_HANDS: return "EARLY"
    if idx >= LATE_HANDS:  return "LATE"
    return "MID"

def _current_stage(sess: Dict[str, Any], remaining_cards: Optional[int] = None) -> str:
    if STAGE_MODE == "remain" and remaining_cards is not None:
        if remaining_cards <= REMAIN_LATE:
            return "LATE"
        idx = int(sess.get("hand_idx", 0))
        return _stage_from_count(idx)
    idx = int(sess.get("hand_idx", 0))
    return _stage_from_count(idx)

def _env_stage_get(sess: Dict[str, Any], key: str, default: str) -> str:
    """
    讀取 <STAGE>_KEY，若無則讀 KEY，若仍無則用 default
    """
    stage = sess.get("_stage", "MID")
    return os.getenv(f"{stage}_{key}", os.getenv(key, default))

# ---------- 解析上局點數 ----------
def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
    if not text: return None
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
    if m: return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(?:莊|庄|B)\s*:?:?\s*(\d)\D+(?:閒|闲|P)\s*:?:?\s*(\d)", u)
    if m: return (int(m.group(2)), int(m.group(1)))

    t = u.replace(" ", "").replace("\u3000", "")
    if t in ("B","莊","庄"): return (0,1)
    if t in ("P","閒","闲"): return (1,0)
    if t in ("T","和"): return (0,0)

    if re.search(r"[A-Z]", u): return None
    digits = re.findall(r"\d", u)
    if len(digits) == 2: return (int(digits[0]), int(digits[1]))
    return None

# ---------- 永久試用鎖（綁 LINE user_id） ----------
TRIAL_MINUTES = int(os.getenv("TRIAL_MINUTES", "30"))  # 預設 30 分鐘
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
    return None  # 尚未到期

def validate_activation_code(code: str) -> bool:
    if not code: return False
    norm = str(code).replace("\u3000", " ").replace("：", ":").strip().lstrip(":").strip()
    return bool(ADMIN_ACTIVATION_SECRET) and (norm == ADMIN_ACTIVATION_SECRET)

# ---------- Outcome PF ----------
log.info("載入 PF 參數: PF_N=%s, PF_UPD_SIMS=%s, PF_PRED_SIMS=%s, DECKS=%s",
         os.getenv("PF_N", "50"), os.getenv("PF_UPD_SIMS", "30"),
         os.getenv("PF_PRED_SIMS", "5"), os.getenv("DECKS", "8"))

PF_BACKEND = os.getenv("PF_BACKEND", "mc").lower()
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

# ---------- 決策參數（原有） ----------
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
USE_KELLY = env_flag("USE_KELLY", 0)
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)

# 決策模式與參數（原有）
DECISION_MODE = os.getenv("DECISION_MODE", "ev").lower()  # ev | prob | hybrid
BANKER_PAYOUT = float(os.getenv("BANKER_PAYOUT", "0.95"))  # 莊抽水
PROB_MARGIN = float(os.getenv("PROB_MARGIN", "0.02"))      # hybrid 門檻
MIN_EV_EDGE = float(os.getenv("MIN_EV_EDGE", "0.0"))       # hybrid EV 子門檻
STRICT_PROB_ONLY = env_flag("STRICT_PROB_ONLY", 0)
DISABLE_EV = env_flag("DISABLE_EV", 0)

# ---------- 新增：可用環境變數控制的守門與配注 ----------
MIN_CONF_FOR_ENTRY = float(os.getenv("MIN_CONF_FOR_ENTRY", "0.56"))  # 低於此一律觀望
QUIET_SMALLEdge   = env_flag("QUIET_SMALLEdge", 0)                    # 邊際略優也觀望

MIN_BET_PCT_ENV   = float(os.getenv("MIN_BET_PCT", "0.05"))           # 5%
MAX_BET_PCT_ENV   = float(os.getenv("MAX_BET_PCT", "0.40"))           # 40%
MAX_EDGE_SCALE    = float(os.getenv("MAX_EDGE_FOR_FULLBET", "0.15"))  # final_edge 達此給滿注

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
    """
    誰要不要混合、混多少，已在外層用平滑管線決定並傳進來。
    """
    pB, pP, pT = float(prob[0]), float(prob[1]), float(prob[2])

    reason_parts: List[str] = []

    # 三種決策模式
    mode = DECISION_MODE
    if STRICT_PROB_ONLY:  # 強制只看機率
        mode = "prob"
    if DISABLE_EV and mode != "prob":
        # 禁用 EV 時，hybrid/ev 都退化走機率
        mode = "prob"

    if mode == "prob":
        side = _decide_side_by_prob(pB, pP)
        _, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
        final_edge = max(abs(evB), abs(evP))  # 用 EV 幅度當作 edge 指標做配注尺度
        reason_parts.append(f"模式=prob (pB={pB:.4f}, pP={pP:.4f})")
    elif mode == "hybrid":
        if abs(pB - pP) >= PROB_MARGIN:
            side = _decide_side_by_prob(pB, pP)
            _, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
            final_edge = max(abs(evB), abs(evP))
            reason_parts.append(f"模式=hybrid→prob (Δ={abs(pB-pP):.4f}≥{PROB_MARGIN})")
        else:
            ev_side, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
            if edge_ev >= MIN_EV_EDGE and not DISABLE_EV:
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

    # ------- 最終觀望守門 -------
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

    # ------- 配注（線性） -------
    min_b = max(0.0, min(1.0, MIN_BET_PCT_ENV))
    max_b = max(min_b, min(1.0, MAX_BET_PCT_ENV))
    max_edge = max(EDGE_ENTER + 1e-6, MAX_EDGE_SCALE)

    bet_pct = min_b + (max_b - min_b) * (final_edge - EDGE_ENTER) / (max_edge - EDGE_ENTER)
    bet_pct = float(min(max_b, max(min_b, bet_pct)))

    reason_parts.append(f"信心度配注({int(min_b*100)}%~{int(max_b*100)}%), conf={conf:.3f}")
    return (INV[side], final_edge, bet_pct, "; ".join(reason_parts))

def format_output_card(prob: np.ndarray, choice: str, last_pts_text: Optional[str], bet_amt: int, cont: bool, stage: str) -> str:
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

    block.append(f"Stage：{stage}")  # 顯示目前分段

    if cont:
        block.append("\n📌 連續模式：請直接輸入下一局點數（例：65 / 和 / 閒6莊5）")
    return "\n".join(header + [""] + block)

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

def _handle_points_and_predict(sess: Dict[str, Any], p_pts: int, b_pts: int, reply_token: str):
    log.info("開始處理點數預測: 閒%d 莊%d (deplete=%s, mode=%s)", p_pts, b_pts, DEPLETE_OK, DECISION_MODE)
    start_time = time.time()
    outcome = 2 if p_pts == b_pts else (1 if p_pts > b_pts else 0)

    # 顯示上局文字
    if outcome == 2:
        sess["last_pts_text"] = "上局結果: 和局"
    else:
        sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"
    sess["last_outcome"] = outcome
    sess["streak_count"] = 1 if outcome in (0, 1) else 0
    sess["phase"] = "ready"

    # 手數累計 + 判定分段
    sess["hand_idx"] = int(sess.get("hand_idx", 0)) + 1
    stage = _current_stage(sess, remaining_cards=None)
    sess["_stage"] = stage

    try:
        # ====== 動態載入 Stage 參數（預測前覆寫，預測後還原） ======
        global MIN_CONF_FOR_ENTRY, EDGE_ENTER, QUIET_SMALLEdge
        global SOFT_TAU, PROB_SMA_ALPHA
        global PF_BACKEND  # 僅供 log
        global PF  # PF 實例不重建，只調用 predict 的 sims

        # 可分段的 PF 調整（僅用於 predict 的 sims_per_particle）
        pred_sims_stage = int(_env_stage_get(sess, "PF_PRED_SIMS", os.getenv("PF_PRED_SIMS", "5")))

        _bak = {
            "MIN_CONF_FOR_ENTRY": MIN_CONF_FOR_ENTRY,
            "EDGE_ENTER": EDGE_ENTER,
            "QUIET_SMALLEdge": QUIET_SMALLEdge,
            "SOFT_TAU": SOFT_TAU,
            "PROB_SMA_ALPHA": PROB_SMA_ALPHA,
        }
        MIN_CONF_FOR_ENTRY = float(_env_stage_get(sess, "MIN_CONF_FOR_ENTRY", str(MIN_CONF_FOR_ENTRY)))
        EDGE_ENTER         = float(_env_stage_get(sess, "EDGE_ENTER",         str(EDGE_ENTER)))
        QUIET_SMALLEdge    = 1 if _env_stage_get(sess, "QUIET_SMALLEdge", "0").lower() in ("1","true","y","yes","on") else 0
        SOFT_TAU           = float(_env_stage_get(sess, "SOFT_TAU",           str(SOFT_TAU)))
        PROB_SMA_ALPHA     = float(_env_stage_get(sess, "PROB_SMA_ALPHA",     str(PROB_SMA_ALPHA)))

        # ====== PF 預測 ======
        t0 = time.time()
        pf_preds = PF.predict(sims_per_particle=pred_sims_stage)
        log.info("PF 預測完成, 耗時: %.2fs (stage=%s, sims=%s)", time.time() - t0, stage, pred_sims_stage)
        p = pf_preds

        # ====== Deplete 模擬與融合（如可用） ======
        if DEPLETE_OK and init_counts and probs_after_points:
            try:
                base_decks = int(os.getenv("DECKS", "8"))
                counts = init_counts(base_decks)
                DEPLETE_SIMS   = int(os.getenv("DEPLETEMC_SIMS", "1000"))
                DEPLETE_FACTOR = float(os.getenv("DEPL_FACTOR",   "1.0"))
                dep_preds = probs_after_points(counts, p_pts, b_pts, sims=DEPLETE_SIMS, deplete_factor=DEPLETE_FACTOR)
                p = (pf_preds + dep_preds) * 0.5
            except Exception as e:
                log.warning("Deplete 模擬失敗，改用 PF 單模：%s", e)

        # ====== 機率平滑（理論混合 + 會話級 EMA） ======
        p_use = smooth_probs(p)
        # 會話級 EMA：和局後若 SKIP_TIE_UPD=1 則不上寫 _last_probs
        last_probs = sess.get("_last_probs", None)
        if isinstance(last_probs, list):
            try:
                last_probs = np.array(last_probs, dtype=np.float32)
            except Exception:
                last_probs = None
        p_use = _ema_update(last_probs, p_use, PROB_SMA_ALPHA)

        # ====== 決策與配注 ======
        choice, edge, bet_pct, reason = decide_only_bp(p_use)
        bankroll_now = int(sess.get("bankroll", 0))
        bet_amt = bet_amount(bankroll_now, bet_pct)

        # 儲存 EMA 狀態（若不是和局或允許更新）
        if not (SKIP_TIE_UPD and outcome == 2):
            sess["_last_probs"] = p_use.tolist()

        msg = format_output_card(p_use, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE), stage=stage)
        _reply(reply_token, msg)

        if LOG_DECISION or SHOW_CONF_DEBUG:
            log.info("決策: %s edge=%.4f pct=%.2f%% | stage=%s | %s",
                     choice, edge, bet_pct*100, stage, reason)

    except Exception as e:
        log.error("預測過程中錯誤: %s", e)
        try:
            _reply(reply_token, "⚠️ 預計算錯誤，請稍後再試")
        except Exception:
            pass
    finally:
        # 還原全域參數
        MIN_CONF_FOR_ENTRY = _bak["MIN_CONF_FOR_ENTRY"]
        EDGE_ENTER         = _bak["EDGE_ENTER"]
        QUIET_SMALLEdge    = _bak["QUIET_SMALLEdge"]
        SOFT_TAU           = _bak["SOFT_TAU"]
        PROB_SMA_ALPHA     = _bak["PROB_SMA_ALPHA"]

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
            _ = trial_persist_guard(uid)  # 註冊 first_ts
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
                    _reply(event.reply_token, "🧹 已清空。輸入『遊戲設定』重新開始。")
                    save_session(uid, sess); return

                # 遊戲設定
                if text == "遊戲設定" or up == "GAME SETTINGS":
                    sess["phase"] = "choose_game"
                    sess["game"] = None; sess["table"] = None; sess["table_no"] = None
                    sess["bankroll"] = 0; sess["streak_count"] = 0
                    sess["last_outcome"] = None; sess["last_pts_text"] = None
                    sess["_last_probs"] = None
                    sess["hand_idx"] = 0
                    sess["_stage"] = "MID"
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

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info(
        "Starting %s on port %s (CONTINUOUS_MODE=%s, PF_INIT=%s, DEPLETE_OK=%s, MODE=%s, THEO_BLEND=%.3f, STAGE_MODE=%s)",
        VERSION, port, CONTINUOUS_MODE, pf_initialized, DEPLETE_OK, DECISION_MODE, THEO_BLEND, STAGE_MODE
    )
    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
