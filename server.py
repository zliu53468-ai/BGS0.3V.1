# -*- coding: utf-8 -*-
"""server.py — BGS Independent Prediction + Stage Overrides (2025-11-02)

- 修正：先前語法錯誤 `sess["phase"] = "await_pts"]` → 已移除多餘 `]`
- 新增：分段覆蓋邏輯 get_stage_over()，支援 LATE_* 參數（尾段）
- 依你提供的流程嵌入：
  1) SoftTau 溫度縮放
  2) deplete MC（可調 DEPLETEMC_SIMS）
  3) 分段 THEO_BLEND（局部混合理論分布）
  4) 分段 TIE_MAX 封頂
  5) 在決策前臨時覆蓋 MIN_CONF_FOR_ENTRY / EDGE_ENTER（只本次有效）

可用環境變數（重點，與你之前一致/相容）：
- PF：PF_N, PF_UPD_SIMS, PF_PRED_SIMS, PF_RESAMPLE, PF_DIR_EPS, PF_BACKEND (mc/np)
- 基本決策：DECISION_MODE(ev|prob|hybrid), BANKER_PAYOUT, PROB_MARGIN, MIN_EV_EDGE
- 出手守門：MIN_CONF_FOR_ENTRY, EDGE_ENTER, QUIET_SMALLEdge
- 配注：MIN_BET_PCT, MAX_BET_PCT, MAX_EDGE_FOR_FULLBET
- 和局/平滑：SKIP_TIE_UPD, SOFT_TAU, TIE_MIN, TIE_MAX
- 分段：STAGE_MODE=count|disabled, EARLY_HANDS, LATE_HANDS,
         LATE_SOFT_TAU, LATE_PROB_SMA_ALPHA(保留欄位), LATE_PF_PRED_SIMS,
         LATE_MIN_CONF_FOR_ENTRY, LATE_EDGE_ENTER,
         DEPLETEMC_SIMS, DEPL_FACTOR(保留欄位), THEO_BLEND(分段覆蓋可用)
"""

import os
import sys
import logging
import time
import re
import json
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")

# ---------- 安全導入 deplete（有就用，沒有不會掛） ----------
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

# ---------- 安全導入 Flask ----------
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

# ---------- Redis（可選） ----------
try:
    import redis
except Exception:
    redis = None

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

# ---------- 版本 ----------
VERSION = "bgs-independent-2025-11-02+stage-overrides"

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

# ---------- PF（Outcome Particle Filter） ----------
PF_BACKEND = os.getenv("PF_BACKEND", "mc").lower()
SKIP_TIE_UPD = env_flag("SKIP_TIE_UPD", 1)
SOFT_TAU = float(os.getenv("SOFT_TAU", "2.0"))
TIE_MIN = float(os.getenv("TIE_MIN", "0.05"))
TIE_MAX = float(os.getenv("TIE_MAX", "0.15"))  # 作為全域預設；分段時可臨時覆蓋
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
            base = base ** (1.0 / max(1e-6, SOFT_TAU))
            base = base / base.sum()
            pT = float(base[2])
            # 保持在 TIE_MIN ~ TIE_MAX
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

# ---------- 決策 / 配注 ----------
DECISION_MODE = os.getenv("DECISION_MODE", "ev").lower()  # ev | prob | hybrid
BANKER_PAYOUT = float(os.getenv("BANKER_PAYOUT", "0.95"))
PROB_MARGIN = float(os.getenv("PROB_MARGIN", "0.02"))
MIN_EV_EDGE = float(os.getenv("MIN_EV_EDGE", "0.0"))

MIN_CONF_FOR_ENTRY = float(os.getenv("MIN_CONF_FOR_ENTRY", "0.56"))
EDGE_ENTER = float(os.getenv("EDGE_ENTER", "0.03"))
QUIET_SMALLEdge = env_flag("QUIET_SMALLEdge", 0)

MIN_BET_PCT_ENV = float(os.getenv("MIN_BET_PCT", "0.05"))
MAX_BET_PCT_ENV = float(os.getenv("MAX_BET_PCT", "0.40"))
MAX_EDGE_SCALE = float(os.getenv("MAX_EDGE_FOR_FULLBET", "0.15"))

USE_KELLY = env_flag("USE_KELLY", 0)  # 目前未使用，保留
CONTINUOUS_MODE = env_flag("CONTINUOUS_MODE", 1)

SHOW_CONF_DEBUG = env_flag("SHOW_CONF_DEBUG", 1)
LOG_DECISION = env_flag("LOG_DECISION", 1)

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
        ev_side, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
        final_edge = max(abs(evB), abs(evP))
        reason_parts.append(f"模式=prob (pB={pB:.4f}, pP={pP:.4f})")
    elif DECISION_MODE == "hybrid":
        if abs(pB - pP) >= PROB_MARGIN:
            side = _decide_side_by_prob(pB, pP)
            ev_side, edge_ev, evB, evP = _decide_side_by_ev(pB, pP)
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
    else:
        side, final_edge, evB, evP = _decide_side_by_ev(pB, pP)
        reason_parts.append(f"模式=ev (EV_B={evB:.4f}, EV_P={evP:.4f}, payout={BANKER_PAYOUT})")

    conf = max(pB, pP)
    if conf < MIN_CONF_FOR_ENTRY:
        reason_parts.append(f"⚪ 信心不足 conf={conf:.3f}<{MIN_CONF_FOR_ENTRY:.3f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason_parts))

    if final_edge < EDGE_ENTER:
        reason_parts.append(f"⚪ 優勢不足 edge={final_edge:.4f}<{EDGE_ENTER:.4f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason_parts))

    if QUIET_SMALLEdge and final_edge < (EDGE_ENTER * 1.2):
        reason_parts.append(f"⚪ 邊際略優(quiet) edge={final_edge:.4f}<{EDGE_ENTER*1.2:.4f}")
        return ("觀望", final_edge, 0.0, "; ".join(reason_parts))

    # 線性配注
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

# ---------- Session ----------
def get_session(uid: str) -> Dict[str, Any]:
    if redis_client:
        j = _rget(f"sess:{uid}")
        if j:
            try: return json.loads(j)
            except Exception: pass
    sess = SESS_FALLBACK.get(uid) or {
        "phase": "await_pts",
        "bankroll": 0,
        "rounds_seen": 0,  # 用來做分段
        "last_pts_text": None
    }
    return sess

def save_session(uid: str, data: Dict[str, Any]):
    if redis_client:
        _rset(f"sess:{uid}", json.dumps(data), ex=SESSION_EXPIRE_SECONDS)
    else:
        SESS_FALLBACK[uid] = data

def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id: return True
    return _rsetnx(f"dedupe:{event_id}", "1", DEDUPE_TTL)

# ---------- 分段覆蓋器 ----------
def get_stage_over(rounds_seen: int) -> Dict[str, float]:
    """
    依局數回傳「本次決策」臨時覆蓋的參數（overrides）。
    你可用環境變數調整（示例：尾段更穩）
    - STAGE_MODE=count|disabled
    - EARLY_HANDS (含) 以前 = early；EARLY_HANDS ~ LATE_HANDS 之間 = mid；> LATE_HANDS = late
    - 目前主要在 late 才覆蓋，保持你之前習慣。
    可用的 LATE_* 例：
      LATE_SOFT_TAU, LATE_PF_PRED_SIMS, LATE_MIN_CONF_FOR_ENTRY, LATE_EDGE_ENTER,
      THEO_BLEND（分段用此鍵即可覆蓋，沿用你 snippet 寫法),
      TIE_MAX（分段封頂）,
      DEPLETEMC_SIMS（尾段 1300~1600）
    """
    stage_mode = os.getenv("STAGE_MODE", "count").lower()
    if stage_mode == "disabled":
        return {}

    early = int(os.getenv("EARLY_HANDS", "15"))
    late  = int(os.getenv("LATE_HANDS", "56"))

    over: Dict[str, float] = {}

    if rounds_seen > late:
        # 尾段覆蓋（用你的建議預設值；沒設環境變數就用這些合理缺省）
        over["SOFT_TAU"] = float(os.getenv("LATE_SOFT_TAU", "1.92"))
        over["DEPLETEMC_SIMS"] = float(os.getenv("DEPLETEMC_SIMS", "1600"))
        over["THEO_BLEND"] = float(os.getenv("THEO_BLEND", "0.004"))
        over["TIE_MAX"] = float(os.getenv("TIE_MAX", "0.11"))
        over["MIN_CONF_FOR_ENTRY"] = float(os.getenv("LATE_MIN_CONF_FOR_ENTRY", "0.462"))
        over["EDGE_ENTER"] = float(os.getenv("LATE_EDGE_ENTER", "0.0030"))

        # 若你提供 LATE_PF_PRED_SIMS，就用它覆蓋 PF_PRED_SIMS 參數（透過 env 再讀）
        lpred = os.getenv("LATE_PF_PRED_SIMS")
        if lpred:
            try:
                over["PF_PRED_SIMS"] = float(lpred)
            except Exception:
                pass

    # 你也可以擴充 early/mid 覆蓋邏輯；此處先簡化只在 late 處理
    return over

# ---------- 點數解析 ----------
def parse_last_hand_points(text: str) -> Optional[Tuple[int, int]]:
    if not text: return None
    s = str(text).translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\r\n\t]", "", s)
    s = s.replace("\u3000", " ")
    u = s.upper().strip()

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

# ---------- 主預測處理（嵌入你的 1~5 流程） ----------
def _handle_points_and_predict(sess: Dict[str, Any], p_pts: int, b_pts: int) -> Tuple[np.ndarray, str, int, str]:
    start_time = time.time()

    # （可選）若你想要把上一手結果回灌 PF，可在這裡做：
    # if not (p_pts == b_pts and SKIP_TIE_UPD):
    #     outcome = 2 if p_pts == b_pts else (1 if p_pts > b_pts else 0)
    #     try:
    #         PF.update_outcome(outcome)
    #     except Exception as e:
    #         log.warning("PF.update_outcome failed: %s", e)

    # 取得「分段覆蓋」參數
    rounds_seen = int(sess.get("rounds_seen", 0))
    over = get_stage_over(rounds_seen)

    # 先跑 PF 預測（基礎機率）
    sims_per_particle = int(over.get("PF_PRED_SIMS", float(os.getenv("PF_PRED_SIMS", "5"))))
    pf_preds = PF.predict(sims_per_particle=sims_per_particle)
    p = np.asarray(pf_preds, dtype=np.float32)

    # 1) SoftTau 溫度縮放
    soft_tau = float(over.get("SOFT_TAU", float(os.getenv("SOFT_TAU", "2.0"))))
    p = p ** (1.0 / max(1e-6, soft_tau))
    p = p / p.sum()

    # 2) deplete MC（若可用），尾段可用更重的模擬量
    if DEPLETE_OK and init_counts and probs_after_points:
        try:
            base_decks = int(os.getenv("DECKS", "8"))
            counts = init_counts(base_decks)
            dep_sims = int(over.get("DEPLETEMC_SIMS", float(os.getenv("DEPLETEMC_SIMS", "1000"))))
            dep_preds = probs_after_points(counts, p_pts, b_pts, sims=dep_sims, deplete_factor=1.0)
            p = (p + dep_preds) * 0.5
            p = p / p.sum()
        except Exception as e:
            log.warning("Deplete 模擬失敗，改用 PF 單模：%s", e)

    # 3) 分段 THEO_BLEND 局部混合（不改全域）
    theo_blend = float(over.get("THEO_BLEND", float(os.getenv("THEO_BLEND", "0.0"))))
    if theo_blend > 0.0:
        theo = np.array([0.4586, 0.4462, 0.0952], dtype=np.float32)
        p = (1.0 - theo_blend) * p + theo_blend * theo
        p = p / p.sum()

    # 4) 分段 TIE_MAX 封頂（同時確保不低於全域 TIE_MIN）
    tie_max = float(over.get("TIE_MAX", float(os.getenv("TIE_MAX", str(TIE_MAX)))))
    pT = float(p[2])
    if pT > tie_max:
        scale = (1.0 - tie_max) / (1.0 - pT) if pT < 1.0 else 1.0
        p[2] = tie_max
        p[0] *= scale; p[1] *= scale
        p = p / p.sum()
    if float(p[2]) < TIE_MIN:  # 下限保護
        scale = (1.0 - TIE_MIN) / (1.0 - float(p[2])) if p[2] < 1.0 else 1.0
        p[2] = TIE_MIN
        p[0] *= scale; p[1] *= scale
        p = p / p.sum()

    # 5) 決策前臨時覆蓋觀望門檻（只對本次有效）
    _global_MIN_CONF = globals()["MIN_CONF_FOR_ENTRY"]
    _global_EDGE_ENTER = globals()["EDGE_ENTER"]
    try:
        if "MIN_CONF_FOR_ENTRY" in over:
            globals()["MIN_CONF_FOR_ENTRY"] = float(over["MIN_CONF_FOR_ENTRY"])
        if "EDGE_ENTER" in over:
            globals()["EDGE_ENTER"] = float(over["EDGE_ENTER"])

        choice, edge, bet_pct, reason = decide_only_bp(p)
    finally:
        globals()["MIN_CONF_FOR_ENTRY"] = _global_MIN_CONF
        globals()["EDGE_ENTER"] = _global_EDGE_ENTER

    # 配注金額（用 session bankroll，如未設定則 0）
    bankroll_now = int(sess.get("bankroll", 0))
    bet_amt = bet_amount(bankroll_now, bet_pct)

    # 更新 session（不漏增 rounds_seen）
    sess["rounds_seen"] = rounds_seen + 1

    elapsed = time.time() - start_time
    if LOG_DECISION or SHOW_CONF_DEBUG:
        log.info("決策: %s edge=%.4f pct=%.2f%% | rounds=%d | %.2fs | %s",
                 choice, edge, bet_pct*100, sess["rounds_seen"], elapsed, reason)

    return p, choice, bet_amt, reason

# ---------- 簡易 HTTP 介面 ----------
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

@app.post("/predict")
def predict():
    """
    請求 JSON 欄位：
      - uid: 使用者 id（用於 session）
      - last_text: 上局點數輸入（例：'閒6莊5' / '65' / '和'）
      - bankroll:（可選）本次籌碼；若提供會寫入 session
    回傳：
      - probs: [pB, pP, pT]
      - choice: "莊" / "閒" / "觀望"
      - bet: 建議下注金額（整數）
      - reason: 決策說明
    """
    try:
        data = request.get_json(force=True) or {}
        uid = str(data.get("uid") or "anon")
        last_text = str(data.get("last_text") or "")
        bankroll = data.get("bankroll")

        sess = get_session(uid)
        if isinstance(bankroll, int) and bankroll >= 0:
            sess["bankroll"] = bankroll

        pts = parse_last_hand_points(last_text)
        if not pts:
            return jsonify(ok=False, error="無法解析點數；請輸入如 '閒6莊5'、'65'、'和'"), 400

        p_pts, b_pts = pts[0], pts[1]
        if p_pts == b_pts and SKIP_TIE_UPD:
            sess["last_pts_text"] = "上局結果: 和局"
        else:
            sess["last_pts_text"] = f"上局結果: 閒 {p_pts} 莊 {b_pts}"

        probs, choice, bet_amt, reason = _handle_points_and_predict(sess, p_pts, b_pts)
        save_session(uid, sess)

        card = format_output_card(probs, choice, sess.get("last_pts_text"), bet_amt, cont=bool(CONTINUOUS_MODE))
        return jsonify(ok=True,
                       probs=[float(probs[0]), float(probs[1]), float(probs[2])],
                       choice=choice, bet=bet_amt, reason=reason, card=card), 200
    except Exception as e:
        log.exception("predict error: %s", e)
        return jsonify(ok=False, error=str(e)), 500

# ---------- Main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("Starting %s on port %s (PF_INIT=%s, DEPLETE_OK=%s, MODE=%s)",
             VERSION, port, pf_initialized, DEPLETE_OK, DECISION_MODE)
    if _flask_available and Flask is not None:
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        log.warning("Flask not available; cannot run HTTP server.")
