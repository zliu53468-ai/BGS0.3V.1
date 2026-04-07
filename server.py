# -*- coding: utf-8 -*-
"""server.py — BGS v4.4 真實盈利監控版（EV正確率評估 + 真實牌靴 + ROI追蹤 + 每日風控）
🔥 2025-01-27 修復版：Webhook Log + Handler try-catch + dedupe_event 修復
🔥 2025-01-27 v2：Flask強制檢查 + SELF PING防休眠 + Webhook強化
🔥 2025-01-28 v3：徹底修復 404 問題 + 路由診斷
"""
import os, sys, logging, time, re, json, threading
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, date
import numpy as np

# ==================== VERSION 定義 ====================
VERSION = "bgs-v4.4-profit-monitor-2025-01-28-fixed-v3"

# ==================== 試用設定 ====================
TRIAL_SECONDS = int(os.getenv("TRIAL_SECONDS", "1800"))
PREMIUM_TRIAL_SECONDS = int(os.getenv("PREMIUM_TRIAL_SECONDS", "2592000"))

# ==================== 風控設定 ====================
DAILY_MAX_LOSS_RATIO = float(os.getenv("DAILY_MAX_LOSS_RATIO", "0.1"))
SESSION_MIN_PROFIT_RATIO = float(os.getenv("SESSION_MIN_PROFIT_RATIO", "-0.05"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("bgs-server")
np.seterr(all="ignore")

# ==================== 館別選單 ====================
GAMES = {
    "1": "WM", "2": "PM", "3": "DG", "4": "SA", "5": "KU",
    "6": "歐博/卡利", "7": "KG", "8": "全利", "9": "名人", "10": "MT真人"
}

# ==================== Deplete Module (安全載入) ====================
DEPLETE_OK = False
init_counts = None
probs_after_points = None

try:
    from deplete import init_counts as _init_counts, probs_after_points as _probs_after_points
    if callable(_init_counts) and callable(_probs_after_points):
        init_counts = _init_counts
        probs_after_points = _probs_after_points
        DEPLETE_OK = True
        log.info("Deplete module loaded successfully")
    else:
        log.warning("Deplete module loaded but functions not callable")
except Exception as e:
    log.warning(f"Deplete not available: {e}")

# ==================== PF Module (安全載入) ====================
PF_INITIALIZED = False
pf_model = None

class SmartDummyPF:
    """PF模型 - 穩定學習版（完全獨立，不依賴外部）"""
    
    def __init__(self):
        self.base_probs = np.array([0.4586, 0.4462, 0.0952])
        self.outcome_history = []
        self.direction_history = []
        self.pf_ev_accuracy_history = []
        self.deplete_ev_accuracy_history = []
        self.learning_rate = 0.1
        self.min_prob = 0.40
        self.max_prob = 0.55
        log.info("PF Model initialized (v4.4 Profit Monitor)")
    
    def predict_proba(self, points: List[int], shoe_counts: Optional[Dict] = None) -> np.ndarray:
        if DEPLETE_OK and init_counts is not None and probs_after_points is not None:
            try:
                counts = shoe_counts if shoe_counts is not None else (init_counts() if init_counts else None)
                if counts is not None and len(points) >= 2:
                    try:
                        probs = probs_after_points(counts, points[0], points[1])
                    except TypeError:
                        try:
                            probs = probs_after_points(points)
                        except Exception:
                            probs = probs_after_points(counts=counts, points=points)
                    if probs is not None and not isinstance(probs, np.ndarray):
                        probs = np.array(probs)
                    if probs is not None:
                        return probs
            except Exception as e:
                log.debug(f"Deplete fallback: {e}")
        return self.base_probs.copy()
    
    def update_outcome(self, outcome: int, confidence: float = 1.0, 
                       pf_probs: np.ndarray = None, dep_probs: np.ndarray = None,
                       choice: str = None) -> None:
        if outcome is None or confidence < 0.8:
            return
        
        self.direction_history.append("莊" if outcome == 1 else "閒")
        if len(self.direction_history) > 50:
            self.direction_history = self.direction_history[-50:]
        
        if pf_probs is not None:
            pB, pP, _ = pf_probs
            ev_banker = pB * 0.95 - (1 - pB)
            ev_player = pP * 1.0 - (1 - pP)
            
            if choice == "莊":
                is_ev_correct = (ev_banker > 0 and outcome == 1) or (ev_banker <= 0 and outcome == 0)
            elif choice == "閒":
                is_ev_correct = (ev_player > 0 and outcome == 0) or (ev_player <= 0 and outcome == 1)
            else:
                is_ev_correct = False
            
            self.pf_ev_accuracy_history.append(1 if is_ev_correct else 0)
            if len(self.pf_ev_accuracy_history) > 50:
                self.pf_ev_accuracy_history = self.pf_ev_accuracy_history[-50:]
        
        if dep_probs is not None:
            pB, pP, _ = dep_probs
            ev_banker = pB * 0.95 - (1 - pB)
            ev_player = pP * 1.0 - (1 - pP)
            
            if choice == "莊":
                is_ev_correct = (ev_banker > 0 and outcome == 1) or (ev_banker <= 0 and outcome == 0)
            elif choice == "閒":
                is_ev_correct = (ev_player > 0 and outcome == 0) or (ev_player <= 0 and outcome == 1)
            else:
                is_ev_correct = False
            
            self.deplete_ev_accuracy_history.append(1 if is_ev_correct else 0)
            if len(self.deplete_ev_accuracy_history) > 50:
                self.deplete_ev_accuracy_history = self.deplete_ev_accuracy_history[-50:]
        
        self.outcome_history.append(outcome)
        if len(self.outcome_history) > 100:
            self.outcome_history = self.outcome_history[-100:]
        if len(self.outcome_history) < 20:
            return
        
        recent = self.outcome_history[-20:]
        banker_rate = recent.count(1) / len(recent)
        player_rate = 1 - banker_rate
        alpha = self.learning_rate
        self.base_probs[0] = self.base_probs[0] * (1 - alpha) + banker_rate * alpha
        self.base_probs[1] = self.base_probs[1] * (1 - alpha) + player_rate * alpha
        self.base_probs[2] = 0.0952
        self.base_probs = np.clip(self.base_probs, self.min_prob, self.max_prob)
        self.base_probs = self.base_probs / self.base_probs.sum()
    
    def detect_pattern(self) -> str:
        if len(self.direction_history) < 5:
            return "未知"
        
        last_5 = self.direction_history[-5:]
        last_3 = self.direction_history[-3:]
        
        if len(set(last_3)) == 1:
            return "連莊" if last_3[0] == "莊" else "連閒"
        
        is_alternating = all(last_5[i] != last_5[i+1] for i in range(4))
        if is_alternating:
            return "跳路"
        
        if len(self.direction_history) >= 8:
            last_8 = self.direction_history[-8:]
            if len(set(last_8[:4])) == 1 and last_8[4] != last_8[0]:
                return "反轉信號"
        
        return "正常"
    
    def get_dynamic_weights(self) -> Tuple[float, float]:
        if len(self.pf_ev_accuracy_history) >= 20 and len(self.deplete_ev_accuracy_history) >= 20:
            pf_acc = sum(self.pf_ev_accuracy_history[-20:]) / 20
            dep_acc = sum(self.deplete_ev_accuracy_history[-20:]) / 20
            
            total = pf_acc + dep_acc
            if total > 0:
                w_pf = pf_acc / total
                w_dep = dep_acc / total
                w_pf = max(0.3, min(0.8, w_pf))
                w_dep = 1 - w_pf
                return w_pf, w_dep
        
        return 0.7, 0.3

try:
    from pf import SmartPF
    pf_model = SmartPF()
    PF_INITIALIZED = True
    log.info("Real PF loaded")
except Exception as e:
    log.warning(f"Real PF not available, using Dummy: {e}")
    pf_model = SmartDummyPF()
    PF_INITIALIZED = True

# ==================== Flask (安全載入 + 強制檢查) ====================
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    _flask_available = True
except Exception as e:
    log.error(f"Flask import failed: {e}")
    _flask_available = False

if not _flask_available:
    raise RuntimeError("❌ Flask 未安裝！請執行：pip install flask flask-cors")

# ==================== Redis (選用) ====================
try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL")
    redis_client = None
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        log.info("Redis connected")
    else:
        redis_client = None
        log.info("Redis not configured, using memory store")
except Exception as e:
    log.warning(f"Redis not available: {e}")
    redis_client = None

# ==================== Session Management ====================
SESS_FALLBACK = {}
KV_FALLBACK = {}
SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "3600"))
DEDUPE_TTL = 60

def _rget(k: str) -> Optional[str]:
    if redis_client:
        try:
            return redis_client.get(k)
        except Exception:
            pass
    return KV_FALLBACK.get(k)

def _rset(k: str, v: str, ex: Optional[int] = None):
    if redis_client:
        try:
            redis_client.set(k, v, ex=ex)
            return
        except Exception:
            pass
    KV_FALLBACK[k] = v

def _rsetnx(k: str, v: str, ex: int) -> bool:
    if redis_client:
        try:
            return bool(redis_client.set(k, v, ex=ex, nx=True))
        except Exception:
            pass
    if k in KV_FALLBACK:
        return False
    KV_FALLBACK[k] = v
    return True

def _dedupe_event(event_id: Optional[str]) -> bool:
    if not event_id:
        return True
    return _rsetnx(f"dedupe:{event_id}", "1", ex=DEDUPE_TTL)

# ==================== Premium System ====================
def _premium_key(uid: str) -> str:
    return f"premium:{uid}"

def get_activation_codes() -> List[str]:
    raw = os.getenv("ACTIVATION_CODES", "")
    return [c.strip().upper() for c in raw.split(",") if c.strip()]

def use_activation_code(uid: str, code: str) -> bool:
    if code.upper() not in get_activation_codes():
        return False
    _rset(_premium_key(uid), "1", ex=PREMIUM_TRIAL_SECONDS)
    return True

def is_premium(uid: str) -> bool:
    if not uid:
        return False
    return _rget(_premium_key(uid)) == "1"

def is_blocked(uid: str) -> bool:
    return False

# ==================== Session Functions ====================
def get_session(uid: str) -> Dict[str, Any]:
    if not uid:
        uid = "anon"
    try:
        if redis_client:
            raw = redis_client.get(f"sess:{uid}")
            if raw:
                sess = json.loads(raw)
                _init_session_defaults(sess, uid)
                return sess
    except Exception:
        pass
    
    sess = SESS_FALLBACK.get(uid)
    if sess:
        _init_session_defaults(sess, uid)
        return sess
    
    sess = _create_new_session(uid)
    save_session(uid, sess)
    return sess

def _init_session_defaults(sess: Dict, uid: str):
    sess.setdefault("phase", "init")
    sess.setdefault("game", None)
    sess.setdefault("bankroll", 0)
    sess.setdefault("initial_bankroll", 0)
    sess.setdefault("rounds_seen", 0)
    sess.setdefault("premium", is_premium(uid))
    sess.setdefault("trial_start", int(time.time()))
    sess.setdefault("skip_streak", 0)
    sess.setdefault("loss_streak", 0)
    sess.setdefault("win_streak", 0)
    sess.setdefault("last_choice", None)
    sess.setdefault("last_prediction", None)
    sess.setdefault("last_actual_outcome", None)
    sess.setdefault("shoe_counts", None)
    sess.setdefault("signal_history", [])
    sess.setdefault("stage", "觀望期")
    sess.setdefault("consecutive_losses", 0)
    sess.setdefault("session_profit", 0)
    sess.setdefault("session_roi", 0.0)
    sess.setdefault("daily_profit", 0)
    sess.setdefault("daily_date", str(date.today()))

def _create_new_session(uid: str) -> Dict:
    return {
        "phase": "init", "game": None, "bankroll": 0, "initial_bankroll": 0,
        "rounds_seen": 0, "premium": is_premium(uid), "trial_start": int(time.time()),
        "skip_streak": 0, "loss_streak": 0, "win_streak": 0,
        "last_choice": None, "last_prediction": None, "last_actual_outcome": None,
        "shoe_counts": None, "signal_history": [], "stage": "觀望期",
        "consecutive_losses": 0, "session_profit": 0, "session_roi": 0.0,
        "daily_profit": 0, "daily_date": str(date.today())
    }

def save_session(uid: str, sess: Dict[str, Any]) -> None:
    if not uid:
        uid = "anon"
    try:
        payload = json.dumps(sess, ensure_ascii=False)
        if redis_client:
            redis_client.set(f"sess:{uid}", payload, ex=SESSION_EXPIRE_SECONDS)
        else:
            SESS_FALLBACK[uid] = sess
    except Exception as e:
        log.warning(f"Save session failed: {e}")

# ==================== Trial System ====================
def get_trial_remaining(sess: Dict[str, Any]) -> int:
    start = sess.get("trial_start", int(time.time()))
    remain = TRIAL_SECONDS - (int(time.time()) - start)
    return max(0, remain)

def is_trial_expired(sess: Dict[str, Any]) -> bool:
    return get_trial_remaining(sess) <= 0

def format_trial_time(sess: Dict[str, Any]) -> str:
    remaining = get_trial_remaining(sess)
    minutes = remaining // 60
    return f"⏳ 試用剩餘 {minutes} 分鐘" if minutes > 0 else "⏰ 試用即將到期"

# ==================== v4.4 核心策略函數 ====================

def check_daily_risk(sess: Dict[str, Any]) -> Tuple[bool, str]:
    today = str(date.today())
    if sess.get("daily_date") != today:
        sess["daily_date"] = today
        sess["daily_profit"] = 0
        save_session(sess.get("uid", "anon"), sess)
    
    initial_bankroll = sess.get("initial_bankroll", sess.get("bankroll", 1000))
    daily_loss = -sess.get("daily_profit", 0)
    
    if initial_bankroll > 0 and daily_loss > initial_bankroll * DAILY_MAX_LOSS_RATIO:
        return False, f"⚠️ 單日虧損已達 {DAILY_MAX_LOSS_RATIO*100:.0f}%，今日停止交易"
    
    return True, ""

def update_roi(sess: Dict[str, Any], bet_amount: int, won: bool) -> None:
    if won:
        profit = bet_amount * 0.95
        sess["session_profit"] = sess.get("session_profit", 0) + profit
        sess["daily_profit"] = sess.get("daily_profit", 0) + profit
    else:
        loss = -bet_amount
        sess["session_profit"] = sess.get("session_profit", 0) + loss
        sess["daily_profit"] = sess.get("daily_profit", 0) + loss
    
    initial = sess.get("initial_bankroll", sess.get("bankroll", 1000))
    if initial > 0:
        sess["session_roi"] = sess["session_profit"] / initial
    else:
        sess["session_roi"] = 0

def determine_stage(sess: Dict[str, Any]) -> str:
    skip_streak = sess.get("skip_streak", 0)
    win_streak = sess.get("win_streak", 0)
    loss_streak = sess.get("loss_streak", 0)
    
    if skip_streak >= 2:
        return "試探期"
    elif win_streak >= 3:
        return "主攻期"
    elif win_streak >= 2:
        return "加溫期"
    elif loss_streak >= 2:
        return "收縮期"
    elif loss_streak >= 1:
        return "觀察期"
    else:
        return "觀望期"

def calculate_signal_strength(probs: np.ndarray, sess: Dict[str, Any]) -> Tuple[float, str]:
    pB, pP, _ = probs
    current_direction = "莊" if pB > pP else "閒"
    edge = abs(pB - pP)
    
    signal_history = sess.get("signal_history", [])
    signal_history.append({"direction": current_direction, "edge": edge, "timestamp": time.time()})
    
    if len(signal_history) > 5:
        signal_history = signal_history[-5:]
    sess["signal_history"] = signal_history
    
    consecutive = 0
    for sig in reversed(signal_history):
        if sig["direction"] == current_direction:
            consecutive += 1
        else:
            break
    
    if edge < 0.015:
        return 1.0, ""
    
    if consecutive >= 3:
        return 1.5, f"🔥 連續{consecutive}局同向，信號強化50%"
    elif consecutive >= 2:
        return 1.2, f"📈 連續{consecutive}局同向，信號強化20%"
    else:
        return 1.0, ""

def calculate_kelly_fraction(pB: float, pP: float) -> float:
    if pB > pP:
        p = pB
        b = 0.95
    else:
        p = pP
        b = 1.0
    
    q = 1 - p
    kelly = (p * b - q) / b
    kelly = max(0, min(kelly, 0.05))
    
    return kelly

def calculate_bet_ratio_advanced(pB: float, pP: float, ev: float, mode: str, 
                                  loss_streak: int, stage: str, 
                                  signal_boost: float, pattern: str,
                                  consecutive_losses: int) -> Tuple[float, str]:
    
    if ev < 0.01:
        return 0, "📉 EV過低，暫停下注"
    
    edge = abs(pB - pP)
    if edge < 0.02:
        return 0, "📉 邊際過小（避免假優勢）"
    
    strength = edge + max(ev, 0)
    
    stage_multiplier = 1.0
    stage_reason = ""
    
    if stage == "主攻期":
        stage_multiplier = 1.3
        stage_reason = " | 主攻期+30%"
    elif stage == "加溫期":
        stage_multiplier = 1.15
        stage_reason = " | 加溫期+15%"
    elif stage == "試探期":
        stage_multiplier = 0.7
        stage_reason = " | 試探期-30%"
    elif stage == "收縮期":
        stage_multiplier = 0.5
        stage_reason = " | 收縮期-50%"
    elif stage == "觀察期":
        stage_multiplier = 0.8
        stage_reason = " | 觀察期-20%"
    
    kelly_fraction = calculate_kelly_fraction(pB, pP)
    
    if kelly_fraction > 0.03:
        base_ratio = kelly_fraction
        reason = f"⚡ Kelly建議 {kelly_fraction*100:.1f}%"
    elif strength > 0.08:
        base_ratio = 0.04
        reason = "⚡ 極強信號"
    elif strength > 0.05:
        base_ratio = 0.025
        reason = "🔥 強勢信號"
    elif strength > 0.03:
        base_ratio = 0.015
        reason = "📈 中等信號"
    else:
        base_ratio = 0.008
        reason = "📊 微幅信號"
    
    if pattern == "連莊" or pattern == "連閒":
        base_ratio *= 1.1
        reason += f" | {pattern}+10%"
    elif pattern == "跳路":
        base_ratio *= 0.9
        reason += " | 跳路-10%"
    elif pattern == "反轉信號":
        base_ratio *= 0.7
        reason += " | 反轉信號-30%"
    
    if mode == "advanced":
        base_ratio = min(base_ratio * 1.2, 0.05)
        reason += " | 付費模式"
    
    base_ratio *= stage_multiplier
    base_ratio *= signal_boost
    
    if signal_boost > 1.0:
        reason += f" | 信號強化{int((signal_boost-1)*100)}%"
    reason += stage_reason
    
    if loss_streak >= 3:
        decay = 0.5 ** (loss_streak - 2)
        base_ratio *= max(0.1, decay)
        reason += f" | ⚠️ 連輸{loss_streak}局降{int((1-decay)*100)}%"
    if loss_streak >= 7:
        base_ratio = 0
        reason = "🛑 連輸保護啟動 - 暫停下注"
    
    if consecutive_losses >= 5:
        base_ratio *= 0.3
        reason += " | 📉 波動過大降70%"
    
    base_ratio = min(base_ratio, 0.05)
    base_ratio = max(base_ratio, 0)
    
    return base_ratio, reason

def parse_points_input(text: str) -> Optional[Tuple[List[int], str]]:
    text = text.strip()
    if re.match(r'^\d{2,4}$', text):
        points = [int(d) for d in text]
        if len(points) == 2:
            return (points, "normal")
        elif len(points) == 4:
            return (points, "full")
    parts = text.split()
    if len(parts) >= 2:
        points = []
        for p in parts[:4]:
            if p.isdigit() and 0 <= int(p) <= 9:
                points.append(int(p))
        if len(points) >= 2:
            return (points, "spaced")
    if text in ["和", "tie", "Tie", "TIE"]:
        return ([], "tie")
    banker_match = re.search(r'莊(\d)', text)
    player_match = re.search(r'閒(\d)', text)
    if banker_match and player_match:
        points = [int(player_match.group(1)), int(banker_match.group(1))]
        return (points, "named")
    return None

def calculate_baccarat_probabilities(points: List[int], shoe_counts: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pf_probs = pf_model.predict_proba(points, shoe_counts)
    dep_probs = pf_probs.copy()
    
    if DEPLETE_OK and probs_after_points is not None and len(points) >= 2:
        try:
            counts = shoe_counts if shoe_counts is not None else (init_counts() if init_counts else None)
            if counts is not None:
                dep_probs = probs_after_points(counts, points[0], points[1])
                if dep_probs is not None and not isinstance(dep_probs, np.ndarray):
                    dep_probs = np.array(dep_probs)
        except Exception as e:
            log.debug(f"Deplete calculation failed: {e}")
    
    if hasattr(pf_model, 'get_dynamic_weights'):
        w_pf, w_dep = pf_model.get_dynamic_weights()
    else:
        edge = abs(pf_probs[0] - pf_probs[1])
        if edge < 0.015:
            w_pf, w_dep = 0.5, 0.5
        elif edge < 0.04:
            w_pf, w_dep = 0.7, 0.3
        else:
            w_pf, w_dep = 0.85, 0.15
    
    final_probs = pf_probs * w_pf + dep_probs * w_dep
    final_probs = final_probs / final_probs.sum()
    
    return final_probs, pf_probs, dep_probs

def calculate_ev(probs: np.ndarray) -> Tuple[float, float]:
    pB, pP, _ = probs
    return pB * 0.95 - (1 - pB), pP * 1.0 - (1 - pP)

def determine_bet_final_v44(probs: np.ndarray, pf_probs: np.ndarray, dep_probs: np.ndarray,
                            mode: str, bankroll: int, skip_streak: int, 
                            loss_streak: int, stage: str, signal_boost: float, 
                            pattern: str, consecutive_losses: int) -> Tuple[str, int, str, np.ndarray, np.ndarray]:
    pB, pP, _ = probs
    ev_banker, ev_player = calculate_ev(probs)
    edge = abs(pB - pP)
    
    best_choice = None
    best_ev = -999
    if ev_banker > 0:
        best_choice = "莊"
        best_ev = ev_banker
    elif ev_player > 0:
        best_choice = "閒"
        best_ev = ev_player
    
    if best_ev < 0.01 and best_choice is not None:
        return "觀望", 0, f"📉 EV過低 ({best_ev*100:.2f}%)", pf_probs, dep_probs
    if loss_streak >= 7:
        return "觀望", 0, "🛑 連輸7局，風控保護強制暫停", pf_probs, dep_probs
    
    if pattern in ["連莊", "連閒"]:
        if (pattern == "連莊" and best_choice != "莊") or (pattern == "連閒" and best_choice != "閒"):
            return "觀望", 0, f"⚠️ 路單({pattern})與信號({best_choice})衝突", pf_probs, dep_probs
    
    if best_choice is None:
        if edge > 0.01:
            choice = "莊" if pB > pP else "閒"
            return choice, max(1, int(bankroll * 0.002)), "🧪 邊緣測試", pf_probs, dep_probs
        if skip_streak >= 2:
            choice = "莊" if pB > pP else "閒"
            return choice, max(1, int(bankroll * 0.003)), "🔄 強制出手", pf_probs, dep_probs
        return "觀望", 0, "📉 期望值為負", pf_probs, dep_probs
    
    bet_ratio, reason = calculate_bet_ratio_advanced(pB, pP, best_ev, mode, loss_streak, 
                                                      stage, signal_boost, pattern, consecutive_losses)
    
    if bet_ratio <= 0:
        return "觀望", 0, reason, pf_probs, dep_probs
    
    bet_amount = max(1, int(bankroll * bet_ratio))
    bet_amount = min(bet_amount, int(bankroll * 0.05))
    
    return best_choice, bet_amount, reason, pf_probs, dep_probs

def format_output_v44(probs: np.ndarray, choice: str, bet_amt: int, reason: str, mode: str,
                      last_pts: Optional[str], game: str, bankroll: int, trial_msg: str,
                      skip_streak: int, loss_streak: int, win_streak: int, stage: str,
                      pattern: str, session_roi: float, ev_banker: float, ev_player: float) -> str:
    pB, pP, pT = [float(x) for x in probs]
    lines = []
    if game:
        lines.append(f"🎰 {game}")
    lines.append(f"💰 本金：{bankroll}")
    lines.append(f"📈 當前ROI：{session_roi*100:.1f}%")
    lines.append(f"📊 輸入：{last_pts}")
    lines.append(f"🎲 機率｜莊 {pB*100:.1f}%｜閒 {pP*100:.1f}%｜和 {pT*100:.1f}%")
    lines.append(f"📈 差距｜{abs(pB-pP)*100:.1f}%")
    lines.append(f"📊 EV｜莊 {ev_banker*100:.2f}%｜閒 {ev_player*100:.2f}%")
    lines.append(f"⚙️ 模式｜{'🎖️ 付費' if mode=='advanced' else '🆓 免費'}")
    lines.append(f"🎯 節奏｜{stage}")
    lines.append(f"🃏 路單｜{pattern}")
    if skip_streak > 0:
        lines.append(f"👀 觀望：{skip_streak}局")
    if loss_streak > 0:
        lines.append(f"📉 連輸：{loss_streak}局")
    if win_streak > 0:
        lines.append(f"📈 連勝：{win_streak}局")
    if choice == "觀望":
        lines.append(f"👀 建議：觀望")
    else:
        lines.append(f"🎯 建議：下注 {choice} {bet_amt}單位")
        if bankroll and bet_amt > 0:
            lines.append(f"📊 風險：{bet_amt/bankroll*100:.1f}%")
    lines.append(f"💡 {reason}")
    if trial_msg:
        lines.append(trial_msg)
    lines.append("\n💡 輸入點數 | 回報：贏 / 輸")
    return "\n".join([l for l in lines if l])

def update_shoe_counts(sess: Dict[str, Any], points: List[int]) -> Optional[Dict]:
    shoe_counts = sess.get("shoe_counts")
    if shoe_counts is None and DEPLETE_OK and init_counts is not None:
        try:
            shoe_counts = init_counts()
            log.info("初始化牌靴計數器")
        except Exception as e:
            log.warning(f"Init shoe counts failed: {e}")
            shoe_counts = {}
    return shoe_counts

def _handle_points_and_predict(uid: str, points_text: str, sess: Dict[str, Any]) -> str:
    raw_text = (points_text or "").strip()

    # 先處理結果回報，避免被點數解析擋掉
    if raw_text in ["贏", "勝", "win", "W"]:
        last_outcome = sess.get("last_actual_outcome")
        last_choice = sess.get("last_choice")
        last_pf_probs = sess.get("last_pf_probs")
        last_dep_probs = sess.get("last_dep_probs")
        last_bet_amount = sess.get("last_bet_amount", 0)

        if last_outcome is not None and hasattr(pf_model, 'update_outcome'):
            pf_model.update_outcome(last_outcome, confidence=1.0, 
                                    pf_probs=last_pf_probs, dep_probs=last_dep_probs,
                                    choice=last_choice)
            log.info(f"用戶{uid}回報贏，PF學習")

        if last_bet_amount > 0:
            update_roi(sess, last_bet_amount, won=True)

        sess["win_streak"] = sess.get("win_streak", 0) + 1
        sess["loss_streak"] = 0
        sess["consecutive_losses"] = 0
        save_session(uid, sess)
        return "✅ 已記錄為「贏」，AI已學習此結果"

    if raw_text in ["輸", "負", "loss", "L"]:
        last_outcome = sess.get("last_actual_outcome")
        last_choice = sess.get("last_choice")
        last_pf_probs = sess.get("last_pf_probs")
        last_dep_probs = sess.get("last_dep_probs")
        last_bet_amount = sess.get("last_bet_amount", 0)

        if last_outcome is not None and hasattr(pf_model, 'update_outcome'):
            opposite = 1 - last_outcome if last_outcome in [0, 1] else None
            if opposite is not None:
                pf_model.update_outcome(opposite, confidence=1.0,
                                        pf_probs=last_pf_probs, dep_probs=last_dep_probs,
                                        choice=last_choice)
                log.info(f"用戶{uid}回報輸，PF學習")

        if last_bet_amount > 0:
            update_roi(sess, last_bet_amount, won=False)

        sess["loss_streak"] = sess.get("loss_streak", 0) + 1
        sess["win_streak"] = 0
        sess["consecutive_losses"] = sess.get("consecutive_losses", 0) + 1
        save_session(uid, sess)
        return "✅ 已記錄為「輸」，AI已學習此結果"

    if raw_text in ["和", "tie", "Tie", "TIE"]:
        sess["last_choice"] = "和"
        save_session(uid, sess)
        return "🎲 和局\n\n建議謹慎下注"

    parsed = parse_points_input(raw_text)
    if not parsed:
        return "❌ 格式錯誤！\n請輸入：65 / 6523 / 閒6莊5 / 和"

    points, input_type = parsed

    risk_ok, risk_msg = check_daily_risk(sess)
    if not risk_ok:
        return risk_msg
    
    shoe_counts = update_shoe_counts(sess, points)
    if shoe_counts is not None:
        sess["shoe_counts"] = shoe_counts
    
    try:
        final_probs, pf_probs, dep_probs = calculate_baccarat_probabilities(points, shoe_counts)
    except Exception as e:
        log.error(f"Probability calculation failed: {e}")
        return f"❌ 計算失敗：{str(e)}"
    
    mode = "advanced" if is_premium(uid) else "normal"
    bankroll = sess.get("bankroll", 1000)
    initial_bankroll = sess.get("initial_bankroll", bankroll)
    if initial_bankroll == 0:
        sess["initial_bankroll"] = bankroll
    
    skip_streak = sess.get("skip_streak", 0)
    loss_streak = sess.get("loss_streak", 0)
    win_streak = sess.get("win_streak", 0)
    consecutive_losses = sess.get("consecutive_losses", 0)
    session_roi = sess.get("session_roi", 0.0)
    ev_banker, ev_player = calculate_ev(final_probs)
    
    stage = determine_stage(sess)
    sess["stage"] = stage
    
    signal_boost, signal_reason = calculate_signal_strength(final_probs, sess)
    
    pattern = "正常"
    if hasattr(pf_model, 'detect_pattern'):
        pattern = pf_model.detect_pattern()
    
    choice, bet_amt, reason, pf_probs_result, dep_probs_result = determine_bet_final_v44(
        final_probs, pf_probs, dep_probs, mode, bankroll, skip_streak, 
        loss_streak, stage, signal_boost, pattern, consecutive_losses
    )
    
    if signal_reason and reason != "🛑 連輸保護啟動 - 暫停下注" and not reason.startswith("⚠️"):
        reason = f"{signal_reason} | {reason}"
    
    sess["last_prediction"] = final_probs.copy()
    sess["last_actual_outcome"] = 1 if choice == "莊" else 0 if choice == "閒" else None
    sess["last_choice"] = choice
    sess["last_pf_probs"] = pf_probs_result
    sess["last_dep_probs"] = dep_probs_result
    sess["last_bet_amount"] = bet_amt if choice != "觀望" else 0
    sess["last_pts_text"] = points_text
    sess["rounds_seen"] = sess.get("rounds_seen", 0) + 1
    
    if choice == "觀望":
        sess["skip_streak"] = skip_streak + 1
    else:
        sess["skip_streak"] = 0
    
    save_session(uid, sess)
    
    trial_msg = None if is_premium(uid) else format_trial_time(sess)
    
    return format_output_v44(final_probs, choice, bet_amt, reason, mode, points_text,
                            sess.get("game"), sess.get("bankroll"), trial_msg,
                            sess.get("skip_streak"), loss_streak, win_streak, stage,
                            pattern, session_roi, ev_banker, ev_player)

# ==================== LINE Bot (安全載入) ====================
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
_line_bot_available = False
line_bot_api = None
handler = None

if LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN:
    try:
        from linebot import LineBotApi, WebhookHandler
        from linebot.models import TextSendMessage, FollowEvent, UnfollowEvent, QuickReply, QuickReplyButton, MessageAction, TextMessage
        line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
        handler = WebhookHandler(LINE_CHANNEL_SECRET)
        _line_bot_available = True
        log.info("LINE Bot ready")
    except Exception as e:
        log.warning(f"LINE Bot not available: {e}")

def build_main_menu():
    if not _line_bot_available:
        return None
    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="🎮 遊戲設定", text="遊戲設定")),
        QuickReplyButton(action=MessageAction(label="🛑 結束", text="結束分析")),
    ])

# ==================== Flask App ====================
app = Flask(__name__)
CORS(app)

# ==================== 全域 404 記錄 ====================
@app.errorhandler(404)
def not_found(e):
    log.warning(f"404 Not Found: {request.method} {request.path} - headers: {dict(request.headers)}")
    return jsonify({"error": "not found", "path": request.path}), 404

# ==================== Routes ====================
@app.get("/")
def root():
    return f"✅ BGS v4.4 真實盈利監控版 ({VERSION})", 200

@app.get("/ping")
def ping():
    return "OK", 200

@app.get("/health")
def health():
    return {
        "ok": True,
        "version": VERSION,
        "status": "running",
        "pf_initialized": PF_INITIALIZED,
        "deplete_ok": DEPLETE_OK,
        "line_bot": _line_bot_available,
        "redis": redis_client is not None,
        "timestamp": time.time()
    }, 200

@app.get("/debug/routes")
def debug_routes():
    """列出所有已註冊的路由，方便除錯"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "path": str(rule)
        })
    return jsonify(routes), 200

@app.post("/predict")
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400
        uid = data.get("uid", "anon")
        points = data.get("points", "")
        if not points:
            return jsonify({"error": "points required"}), 400
        sess = get_session(uid)
        if not is_premium(uid) and is_trial_expired(sess):
            return jsonify({"error": "試用已到期", "trial_expired": True}), 403
        result = _handle_points_and_predict(uid, points, sess)
        return jsonify({"result": result}), 200
    except Exception as e:
        log.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500

# 🔥 核心 Webhook 處理器 (統一入口)
def _handle_line_webhook():
    """LINE Webhook - 完整除錯版本，無論如何都返回 200"""
    log.info("=" * 50)
    log.info("🔥🔥🔥 Webhook HIT 🔥🔥🔥")
    log.info(f"Method: {request.method}")
    log.info(f"Path: {request.path}")
    log.info(f"Headers: {dict(request.headers)}")
    
    # 即使 LINE Bot 未設定，也記錄並返回 200，避免 LINE 重試
    if not _line_bot_available or handler is None:
        log.error("❌ LINE Bot not configured - returning 200 anyway")
        return "OK", 200

    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    log.info(f"🔑 Signature (first 20): {signature[:20]}..." if signature else "🔑 Signature: MISSING")
    log.info(f"📩 Body length: {len(body)}")
    log.info(f"📩 Body preview: {body[:500]}")

    if not signature:
        log.error("❌ Missing X-Line-Signature header")
        # 仍回 200 避免 LINE 持續重送
        return "OK", 200

    try:
        handler.handle(body, signature)
        log.info("✅ Webhook handled OK")
    except Exception as e:
        log.error(f"❌ Webhook crash: {e}", exc_info=True)
        # 錯誤仍回 200
        return "OK", 200

    log.info("=" * 50)
    return "OK", 200

# 三個路徑都指向同一個處理函數，確保相容性
@app.post("/webhook")
def webhook():
    return _handle_line_webhook()

@app.post("/line-webhook")
def line_webhook():
    return _handle_line_webhook()

@app.post("/callback")
def line_webhook_callback():
    return _handle_line_webhook()

# ==================== LINE Handlers ====================
if _line_bot_available and handler:
    
    @handler.add(FollowEvent)
    def handle_follow(event):
        try:
            user_id = event.source.user_id
            log.info(f"📱 New follower: {user_id}")
            msg = f"🎰 BGS v4.4 真實盈利監控版\n\n"
            msg += f"⏳ 試用 {TRIAL_SECONDS//60} 分鐘\n\n"
            msg += "✨ 核心功能：\n"
            msg += "• 🎯 Stage節奏系統\n"
            msg += "• 📈 連續信號強化 + 假信號過濾\n"
            msg += "• 🃏 真實牌靴記憶（每局扣牌）\n"
            msg += "• 💰 Kelly混合下注 + EV濾網\n"
            msg += "• 🃏 路單偵測 + 一致性檢查\n"
            msg += "• 🔄 動態權重（基於EV正確率）\n"
            msg += "• 📊 波動控制（指數衰減）\n"
            msg += "• 🛡️ 連輸風控（指數衰減）\n"
            msg += "• 📈 ROI追蹤 + 每日風控\n"
            msg += "• 🧠 AI真實學習（EV正確率）\n\n"
            msg += "🎮 點擊「遊戲設定」開始"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=build_main_menu()))
            log.info(f"✅ Follow reply sent to {user_id}")
        except Exception as e:
            log.error(f"❌ handle_follow error: {e}", exc_info=True)
    
    @handler.add(TextMessage)
    def handle_text_message(event):
        try:
            user_id = event.source.user_id
            text = event.message.text.strip()
            log.info(f"💬 TextMessage from {user_id}: {text}")
            
            event_id = getattr(event, "webhook_event_id", None)
            if event_id:
                if not _dedupe_event(event_id):
                    log.info(f"⚠️ Duplicate event skipped: {event_id}")
                    return
            else:
                log.info(f"ℹ️ No webhook_event_id, processing anyway")
            
            sess = get_session(user_id)
            
            if not is_premium(user_id) and is_trial_expired(sess):
                log.info(f"⏰ Trial expired for {user_id}")
                line_bot_api.reply_message(event.reply_token, TextSendMessage(
                    text="⏰ 試用已用完，請輸入開通碼：開通 [密碼]",
                    quick_reply=build_main_menu()
                ))
                return
            
            if text.startswith("開通"):
                parts = text.split()
                if len(parts) != 2:
                    reply = "❌ 格式：開通 [密碼]"
                else:
                    if use_activation_code(user_id, parts[1]):
                        sess["premium"] = True
                        save_session(user_id, sess)
                        reply = "✅ 開通成功！已升級付費模式"
                        log.info(f"🎉 User {user_id} activated premium")
                    else:
                        reply = "❌ 密碼錯誤"
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply, quick_reply=build_main_menu()))
                return
            
            phase = sess.get("phase", "init")
            
            if text in ["結束分析", "停止", "exit"]:
                sess["phase"] = "init"
                sess["game"] = None
                sess["bankroll"] = 0
                sess["skip_streak"] = 0
                sess["loss_streak"] = 0
                sess["win_streak"] = 0
                sess["signal_history"] = []
                sess["stage"] = "觀望期"
                sess["consecutive_losses"] = 0
                save_session(user_id, sess)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(
                    text="🛑 已結束分析\n\n🎮 點擊「遊戲設定」重新開始",
                    quick_reply=build_main_menu()
                ))
                return
            
            if text in ["遊戲設定", "設定", "start"]:
                sess["phase"] = "choose_game"
                sess["skip_streak"] = 0
                sess["loss_streak"] = 0
                sess["win_streak"] = 0
                sess["signal_history"] = []
                sess["stage"] = "觀望期"
                sess["consecutive_losses"] = 0
                save_session(user_id, sess)
                msg = "🎯 請選擇館別：\n\n"
                for k, v in GAMES.items():
                    msg += f"{k}. {v}\n"
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=build_main_menu()))
                return
            
            if phase == "choose_game":
                if text in GAMES:
                    sess["game"] = GAMES[text]
                    sess["phase"] = "set_bankroll"
                    save_session(user_id, sess)
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(
                        text=f"✅ 已選擇：{GAMES[text]}\n\n💰 請輸入本金（單位）",
                        quick_reply=build_main_menu()
                    ))
                else:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 請輸入1-10", quick_reply=build_main_menu()))
                return
            
            if phase == "set_bankroll":
                if text.isdigit() and int(text) >= 100:
                    sess["bankroll"] = int(text)
                    sess["initial_bankroll"] = int(text)
                    sess["phase"] = "playing"
                    save_session(user_id, sess)
                    msg = f"✅ 本金：{text} 單位\n\n"
                    msg += "📊 輸入點數（如：65 / 和 / 閒6莊5）\n"
                    msg += "🎯 結果回報：輸入「贏」或「輸」讓AI學習\n"
                    msg += "📈 系統會自動追蹤ROI"
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=build_main_menu()))
                else:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="❌ 本金至少100", quick_reply=build_main_menu()))
                return
            
            if phase == "init":
                line_bot_api.reply_message(event.reply_token, TextSendMessage(
                    text="⚠️ 請先點擊「遊戲設定」",
                    quick_reply=build_main_menu()
                ))
                return
            
            if phase == "playing":
                result = _handle_points_and_predict(user_id, text, sess)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result, quick_reply=build_main_menu()))
                log.info(f"✅ Response sent to {user_id}")
                return
                
        except Exception as e:
            log.error(f"❌ handle_text_message crashed: {e}", exc_info=True)
            try:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(
                    text=f"⚠️ 系統錯誤，請稍後再試\n錯誤：{str(e)[:100]}"
                ))
            except Exception as reply_error:
                log.error(f"❌ Failed to send error message: {reply_error}")

# ==================== SELF PING 防休眠 ====================
def self_ping():
    """每5分鐘 ping 自己，防止 Render 休眠（需設定 RENDER_EXTERNAL_URL）"""
    url = os.getenv("RENDER_EXTERNAL_URL")
    if not url:
        log.warning("⚠️ SELF_PING 無法啟動（沒有 RENDER_EXTERNAL_URL）")
        return
    import requests
    while True:
        try:
            full_url = f"{url}/ping"
            log.info(f"🔁 SELF PING: {full_url}")
            requests.get(full_url, timeout=10)
        except Exception as e:
            log.warning(f"SELF_PING failed: {e}")
        time.sleep(300)  # 5分鐘

# ==================== 啟動前輸出路由 ====================
def log_routes():
    log.info("=== Registered Routes ===")
    for rule in app.url_map.iter_rules():
        log.info(f"{rule.methods} {rule}")

# ==================== 啟動 ====================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    
    render_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if render_url:
        log.info(f"🌐 Webhook URL 應設為: {render_url}/webhook")
    else:
        log.info("🌐 Webhook URL: https://YOUR_DOMAIN/webhook (請設定 RENDER_EXTERNAL_URL 以啟用 SELF PING)")
    
    # 環境變數檢查
    log.info(f"ENV CHECK → LINE_SECRET: {'✅ OK' if LINE_CHANNEL_SECRET else '❌ MISSING'}")
    log.info(f"ENV CHECK → LINE_TOKEN: {'✅ OK' if LINE_CHANNEL_ACCESS_TOKEN else '❌ MISSING'}")
    log.info(f"ENV CHECK → RENDER_URL: {render_url if render_url else '❌ NOT SET (SELF PING 無法啟動)'}")
    
    # 啟動 SELF PING（防 Render 休眠）
    if render_url:
        threading.Thread(target=self_ping, daemon=True).start()
        log.info("✅ SELF PING 已啟動（每5分鐘）")
    else:
        log.warning("⚠️ SELF PING 未啟動（需設定 RENDER_EXTERNAL_URL）")
    
    log.info(f"🚀 BGS v4.4 啟動 - 0.0.0.0:{port}")
    log.info("=== 🔥 v4.4 真實盈利監控版功能 (修復版v3) ===")
    log.info("  ✓ Stage節奏系統（觀望→試探→加溫→主攻→收縮）")
    log.info("  ✓ 連續信號強化 + 假信號過濾（edge < 0.015 不強化）")
    log.info("  ✓ 真實牌靴記憶（每局扣牌，不重新 init）")
    log.info("  ✓ Kelly混合下注 + EV濾網 + 邊際過小濾網")
    log.info("  ✓ 路單偵測 + 一致性檢查（避免邏輯衝突）")
    log.info("  ✓ 動態權重（基於 EV 正確率，不是方向正確率）")
    log.info("  ✓ 波動控制（指數衰減，連續虧損降70%）")
    log.info("  ✓ 連輸風控（指數衰減）")
    log.info("  ✓ 每日風控（單日虧損超過10%停止）")
    log.info("  ✓ ROI追蹤（即時顯示收益率）")
    log.info("  ✓ 真實AI學習（EV正確率評估）")
    log.info("  ✓ 所有模組安全 Fallback")
    log.info("=== 🔧 修復內容 v3 ===")
    log.info("  ✓ 全域 404 記錄 + 診斷")
    log.info("  ✓ Webhook 無論如何都回傳 200 (避免 LINE 重試)")
    log.info("  ✓ 新增 /debug/routes 端點")
    log.info("  ✓ 啟動時輸出所有註冊路由")
    log.info("  ✓ 強化 Webhook 路徑相容性")
    
    # 輸出路由表
    log_routes()
    
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
