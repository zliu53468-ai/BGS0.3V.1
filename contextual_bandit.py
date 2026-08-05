"""BGS cMAB 主模型：LinUCB 上下文相關多臂老虎機。

主要方向只有兩個 Arm：B（莊）與 P（閒）。
本模組不使用粒子濾波、超幾何分布、蒙地卡羅或 Stacking。
每次預測保存上下文向量；使用者回報實際結果後，再以 reward 更新 Arm。
和局不更新 B/P Arm。

V1.2 動態 OOD 防禦：
- 精確使用每個 Arm 的 x^T A^-1 x 預測方差與置信區間寬度。
- 每個 UID 以歷史平均 + 1.5 個標準差建立動態未知區間門檻。
- 高方差時動作切換為觀望／0%配置，但保留最近 3 局微觀方向先驗。
- 高方差樣本以 4～5 倍觀測權重完成 Few-shot 更新。

banker_rate / player_rate 是方向分數正規化結果，不是真實開出機率。
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import json
import math
import os
import time

import numpy as np

ARMS = ("B", "P")
MODEL_VERSION = "CMAB-LINUCB-V1.2-DYNAMIC-OOD-DEFENSE"
STATE_SCHEMA_VERSION = "CMAB-UID-ISOLATED-V1"
FEATURE_NAMES = (
    "bias", "history_maturity", "global_banker_balance",
    "recent5_banker_balance", "recent10_banker_balance",
    "recent20_banker_balance", "recent40_banker_balance",
    "current_streak_direction", "current_streak_length",
    "alternation5", "alternation10", "alternation20",
    "last_outcome_direction", "previous_outcome_direction",
    "observed_tie_rate", "road_planning_balance",
    "road_recent_balance", "road_confidence",
    "road_planning_reliability", "road_recent_reliability",
    "road_agreement", "markov1_balance", "markov2_balance",
    "markov3_balance",
)
CONTEXT_DIM = len(FEATURE_NAMES)
BASE_DIR = Path(__file__).resolve().parent
_LOCK = RLock()


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


CMAB_ALPHA = _env_float("CMAB_ALPHA", 0.80, 0.0, 5.0)
CMAB_L2 = _env_float("CMAB_L2", 1.0, 0.05, 100.0)
CMAB_SCORE_TEMPERATURE = _env_float(
    "CMAB_SCORE_TEMPERATURE", 1.35, 0.10, 10.0
)

# 動態 OOD 閾值：歷史平均 + sigma × 歷史標準差。
CMAB_UNCERTAINTY_SIGMA = _env_float(
    "CMAB_UNCERTAINTY_SIGMA", 1.50, 0.50, 4.00
)
CMAB_UNCERTAINTY_MIN_SAMPLES = max(
    4,
    min(
        100,
        int(os.getenv("CMAB_UNCERTAINTY_MIN_SAMPLES", "8") or "8"),
    ),
)
CMAB_UNCERTAINTY_HISTORY_SIZE = max(
    16,
    min(
        512,
        int(os.getenv("CMAB_UNCERTAINTY_HISTORY_SIZE", "128") or "128"),
    ),
)
CMAB_DYNAMIC_THRESHOLD_FLOOR = _env_float(
    "CMAB_DYNAMIC_THRESHOLD_FLOOR", 0.25, 0.05, 5.00
)

# 冷啟動時尚無足夠歷史基準，沿用原本固定門檻作為安全 fallback。
CMAB_UNKNOWN_STD_THRESHOLD = _env_float(
    "CMAB_UNKNOWN_STD_THRESHOLD", 1.35, 0.10, 10.0
)

# 未知區間每一筆真實 B/P reward 強制視為 4～5 筆同等觀測。
CMAB_UNKNOWN_UPDATE_MULTIPLIER = _env_float(
    "CMAB_UNKNOWN_UPDATE_MULTIPLIER", 4.50, 4.00, 5.00
)
CMAB_UNKNOWN_CONFIDENCE_CAP = _env_float(
    "CMAB_UNKNOWN_CONFIDENCE_CAP", 0.28, 0.05, 0.49
)
CMAB_UNKNOWN_LONG_TERM_WEIGHT = _env_float(
    "CMAB_UNKNOWN_LONG_TERM_WEIGHT", 0.08, 0.00, 0.25
)
CMAB_MAX_EVENT_IDS = max(
    100,
    min(
        20000,
        int(os.getenv("CMAB_MAX_EVENT_IDS", "5000") or "5000"),
    ),
)


def _resolve_state_file() -> Path:
    configured = Path(os.getenv("CMAB_STATE_FILE", str(BASE_DIR / "data" / "contextual_bandit_state.json"))).expanduser()
    candidates = [configured, BASE_DIR / "data" / "contextual_bandit_state.json", Path("/tmp/bgs_contextual_bandit_state.json")]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".cmab_write_test_{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            if candidate != configured:
                print(f"CMAB_STATE_FILE fallback: {configured} -> {candidate}")
            return candidate
        except OSError as exc:
            print(f"CMAB_STATE_FILE unavailable: {candidate}: {exc}")
    raise RuntimeError("No writable CMAB_STATE_FILE path is available")


CMAB_STATE_FILE = _resolve_state_file()


def _clean_history(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-2000:]


def _clip(value: Any, minimum: float = -1.0, maximum: float = 1.0) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except Exception:
        return 0.0


def _prob_balance(probability: Any) -> float:
    try:
        return _clip((float(probability) - 0.5) * 2.0)
    except Exception:
        return 0.0


def _banker_balance(sequence: Sequence[str], size: Optional[int] = None) -> float:
    values = list(sequence[-size:] if size else sequence)
    if not values:
        return 0.0
    banker = sum(value == "B" for value in values)
    return _clip((banker / len(values) - 0.5) * 2.0)


def _alternation(sequence: Sequence[str], size: int) -> float:
    values = list(sequence[-size:])
    if len(values) < 2:
        return 0.0
    rate = sum(a != b for a, b in zip(values, values[1:])) / (len(values) - 1)
    return _clip((rate - 0.5) * 2.0)


def _streak(sequence: Sequence[str]) -> tuple[str, int]:
    if not sequence:
        return "", 0
    direction = sequence[-1]
    length = 1
    for value in reversed(sequence[:-1]):
        if value != direction:
            break
        length += 1
    return direction, length


def _model_probability(road_context: Mapping[str, Any], model_name: str, fallback: float = 0.5) -> float:
    models = road_context.get("models")
    if not isinstance(models, Mapping):
        return fallback
    model = models.get(model_name)
    if not isinstance(model, Mapping):
        return fallback
    try:
        return float(model.get("banker_probability", fallback) or fallback)
    except Exception:
        return fallback


def build_context_vector(history: Iterable[Any], *, road_context: Optional[Mapping[str, Any]] = None) -> List[float]:
    """建立固定 24 維上下文；所有值限制在 [-1, 1]。"""
    raw = _clean_history(history)
    bp = [value for value in raw if value in ARMS]
    road = dict(road_context or {})
    streak_direction, streak_length = _streak(bp)
    last_direction = 1.0 if bp and bp[-1] == "B" else -1.0 if bp else 0.0
    previous_direction = 1.0 if len(bp) >= 2 and bp[-2] == "B" else -1.0 if len(bp) >= 2 else 0.0
    streak_sign = 1.0 if streak_direction == "B" else -1.0 if streak_direction == "P" else 0.0
    tie_rate = sum(value == "T" for value in raw) / max(1, len(raw))
    confidence = _clip(road.get("confidence_score", 0.0), 0.0, 1.0)
    planning_reliability = _clip(road.get("planning_reliability", 0.0), 0.0, 1.0)
    recent_reliability = _clip(road.get("recent_reliability", 0.0), 0.0, 1.0)
    disagreement = _clip(road.get("recent_model_disagreement", road.get("model_disagreement", 0.20)), 0.0, 1.0)
    agreement = _clip(1.0 - min(1.0, disagreement / 0.20), 0.0, 1.0)
    vector = [
        1.0,
        min(1.0, len(bp) / 60.0),
        _banker_balance(bp), _banker_balance(bp, 5), _banker_balance(bp, 10),
        _banker_balance(bp, 20), _banker_balance(bp, 40),
        streak_sign, min(1.0, streak_length / 8.0),
        _alternation(bp, 5), _alternation(bp, 10), _alternation(bp, 20),
        last_direction, previous_direction, _clip(tie_rate / 0.20, 0.0, 1.0),
        _prob_balance(road.get("planning_probability", 0.5)),
        _prob_balance(road.get("recent_probability", 0.5)),
        confidence, planning_reliability, recent_reliability, agreement,
        _prob_balance(_model_probability(road, "markov1")),
        _prob_balance(_model_probability(road, "markov2")),
        _prob_balance(_model_probability(road, "markov3")),
    ]
    if len(vector) != CONTEXT_DIM:
        raise RuntimeError(f"CMAB context dimension mismatch: {len(vector)} != {CONTEXT_DIM}")
    return [round(_clip(value), 10) for value in vector]


def _uid_key(user_id: str) -> str:
    """以雜湊鍵隔離各 LINE UID，不把原始 UID 寫進模型狀態檔。"""
    normalized = str(user_id or "").strip() or "__anonymous__"
    return sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _context_fingerprint(context: Sequence[float]) -> str:
    values = np.asarray(list(context), dtype=np.float64)
    rounded = np.round(values, decimals=8)
    return sha256(rounded.tobytes()).hexdigest()[:24]


def _prediction_fingerprint(
    history: Sequence[str],
    *,
    venue: str,
    room: str,
    context: Sequence[float],
) -> str:
    payload = {
        "history": "".join(history[-2000:]),
        "venue": str(venue or "").upper().strip(),
        "room": str(room or "").strip(),
        "context": [round(float(value), 8) for value in context],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:24]


def _clean_uncertainty_values(values: Any) -> List[float]:
    result: List[float] = []
    for value in list(values or []):
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number) and number >= 0.0:
            result.append(number)
    return result[-CMAB_UNCERTAINTY_HISTORY_SIZE:]


def _baseline_summary(state: Mapping[str, Any]) -> Dict[str, Any]:
    baseline = dict(state.get("uncertainty_baseline") or {})
    values = _clean_uncertainty_values(baseline.get("values"))
    sample_count = len(values)
    mean = float(np.mean(values)) if values else 0.0
    std = (
        float(np.std(values, ddof=1))
        if sample_count >= 2
        else 0.0
    )
    dynamic_ready = sample_count >= CMAB_UNCERTAINTY_MIN_SAMPLES
    threshold = (
        max(
            CMAB_DYNAMIC_THRESHOLD_FLOOR,
            mean + CMAB_UNCERTAINTY_SIGMA * std,
        )
        if dynamic_ready
        else max(
            CMAB_DYNAMIC_THRESHOLD_FLOOR,
            CMAB_UNKNOWN_STD_THRESHOLD,
        )
    )
    return {
        "values": values,
        "sample_count": sample_count,
        "mean": mean,
        "std": std,
        "threshold": float(threshold),
        "dynamic_ready": bool(dynamic_ready),
        "sigma_multiplier": float(CMAB_UNCERTAINTY_SIGMA),
        "minimum_samples": int(CMAB_UNCERTAINTY_MIN_SAMPLES),
        "history_size": int(CMAB_UNCERTAINTY_HISTORY_SIZE),
        "fallback_threshold": float(CMAB_UNKNOWN_STD_THRESHOLD),
        "threshold_floor": float(CMAB_DYNAMIC_THRESHOLD_FLOOR),
    }


def _update_uncertainty_baseline(
    state: Dict[str, Any],
    *,
    action_space_std: float,
    unknown_region_active: bool,
    prediction_fingerprint: str,
) -> Dict[str, Any]:
    """只用已判定為分布內的樣本更新基準，避免 OOD 值污染門檻。"""
    summary = _baseline_summary(state)
    values = list(summary["values"])
    previous_fingerprint = str(
        state.get("last_uncertainty_fingerprint") or ""
    )
    is_new_prediction = (
        bool(prediction_fingerprint)
        and prediction_fingerprint != previous_fingerprint
    )

    if is_new_prediction and not unknown_region_active:
        values.append(max(0.0, float(action_space_std)))
        values = values[-CMAB_UNCERTAINTY_HISTORY_SIZE:]

    state["uncertainty_baseline"] = {
        "values": values,
        "last_observed_std": max(0.0, float(action_space_std)),
        "last_unknown_region_active": bool(unknown_region_active),
        "updated_at": int(time.time()),
    }
    state["last_uncertainty_fingerprint"] = prediction_fingerprint
    return _baseline_summary(state)


def _current_short_term_buffer(history: Sequence[str]) -> List[str]:
    return [value for value in history if value in ARMS][-3:]


def _new_state() -> Dict[str, Any]:
    """建立單一 UID 專屬的 cMAB 狀態。"""
    identity = (
        np.eye(CONTEXT_DIM, dtype=np.float64) * CMAB_L2
    ).tolist()
    zeros = np.zeros(CONTEXT_DIM, dtype=np.float64).tolist()
    return {
        "version": MODEL_VERSION,
        "context_dim": CONTEXT_DIM,
        "feature_names": list(FEATURE_NAMES),
        "alpha": CMAB_ALPHA,
        "l2": CMAB_L2,
        "arms": {
            arm: {
                "A": identity,
                "b": zeros,
                "updates": 0,
                "weighted_updates": 0.0,
                "reward_sum": 0.0,
                "weighted_reward_sum": 0.0,
            }
            for arm in ARMS
        },
        "applied_event_ids": [],
        "total_updates": 0,
        "total_weighted_updates": 0.0,
        "uncertainty_baseline": {
            "values": [],
            "last_observed_std": 0.0,
            "last_unknown_region_active": False,
            "updated_at": 0,
        },
        "short_term_buffer": [],
        "last_uncertainty_fingerprint": "",
        "last_prediction_risk": {},
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }

def _new_state_store() -> Dict[str, Any]:
    """建立全部 UID 的外層容器；每個 UID 仍持有完全獨立的 A／b。"""
    now = int(time.time())
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "version": MODEL_VERSION,
        "context_dim": CONTEXT_DIM,
        "feature_names": list(FEATURE_NAMES),
        "alpha": CMAB_ALPHA,
        "l2": CMAB_L2,
        "users": {},
        "created_at": now,
        "updated_at": now,
    }


def _normalize_user_state(data: Mapping[str, Any]) -> Dict[str, Any]:
    state = dict(data or {})
    if int(state.get("context_dim", 0) or 0) != CONTEXT_DIM:
        raise ValueError("invalid UID state context dimension")

    arms = state.get("arms")
    if not isinstance(arms, dict) or any(
        arm not in arms for arm in ARMS
    ):
        raise ValueError("missing UID state arms")

    for arm in ARMS:
        A = np.asarray(arms[arm].get("A"), dtype=np.float64)
        b = np.asarray(arms[arm].get("b"), dtype=np.float64)
        if (
            A.shape != (CONTEXT_DIM, CONTEXT_DIM)
            or b.shape != (CONTEXT_DIM,)
        ):
            raise ValueError("invalid UID state matrix shape")

    # 舊 UID 狀態直接延續，不重置已學習的 A／b。
    state["version"] = MODEL_VERSION
    state["context_dim"] = CONTEXT_DIM
    state["feature_names"] = list(FEATURE_NAMES)
    state["alpha"] = CMAB_ALPHA
    state["l2"] = CMAB_L2
    state["total_weighted_updates"] = float(
        state.get(
            "total_weighted_updates",
            state.get("total_updates", 0),
        )
        or 0.0
    )

    for arm in ARMS:
        arm_state = arms[arm]
        arm_state["weighted_updates"] = float(
            arm_state.get(
                "weighted_updates",
                arm_state.get("updates", 0),
            )
            or 0.0
        )
        arm_state["weighted_reward_sum"] = float(
            arm_state.get(
                "weighted_reward_sum",
                arm_state.get("reward_sum", 0.0),
            )
            or 0.0
        )

    state["applied_event_ids"] = list(
        state.get("applied_event_ids") or []
    )[-CMAB_MAX_EVENT_IDS:]

    baseline = dict(state.get("uncertainty_baseline") or {})
    state["uncertainty_baseline"] = {
        "values": _clean_uncertainty_values(
            baseline.get("values")
        ),
        "last_observed_std": max(
            0.0,
            float(baseline.get("last_observed_std", 0.0) or 0.0),
        ),
        "last_unknown_region_active": bool(
            baseline.get("last_unknown_region_active", False)
        ),
        "updated_at": int(baseline.get("updated_at", 0) or 0),
    }
    state["short_term_buffer"] = [
        value
        for value in list(state.get("short_term_buffer") or [])
        if value in ARMS
    ][-3:]
    state["last_uncertainty_fingerprint"] = str(
        state.get("last_uncertainty_fingerprint") or ""
    )
    state["last_prediction_risk"] = (
        dict(state.get("last_prediction_risk") or {})
        if isinstance(state.get("last_prediction_risk"), Mapping)
        else {}
    )
    state["created_at"] = int(
        state.get("created_at", time.time()) or time.time()
    )
    state["updated_at"] = int(
        state.get("updated_at", time.time()) or time.time()
    )
    return state

def _read_state_unlocked() -> Dict[str, Any]:
    try:
        data = json.loads(CMAB_STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("invalid state store")

        users = data.get("users")
        if (
            str(data.get("schema_version") or "") == STATE_SCHEMA_VERSION
            and isinstance(users, dict)
            and int(data.get("context_dim", 0) or 0) == CONTEXT_DIM
        ):
            normalized_users: Dict[str, Any] = {}
            for uid_key, raw_state in users.items():
                if not isinstance(raw_state, Mapping):
                    continue
                try:
                    normalized_users[str(uid_key)] = _normalize_user_state(raw_state)
                except Exception:
                    # 單一 UID 狀態損壞時只重置該 UID，不影響其他使用者。
                    normalized_users[str(uid_key)] = _new_state()
            data["schema_version"] = STATE_SCHEMA_VERSION
            data["version"] = MODEL_VERSION
            data["context_dim"] = CONTEXT_DIM
            data["feature_names"] = list(FEATURE_NAMES)
            data["alpha"] = CMAB_ALPHA
            data["l2"] = CMAB_L2
            data["users"] = normalized_users
            return data

        # 舊版只有一組全域 arms，無法可靠拆回各 UID。
        # 為避免把其他人的學習複製給新 UID，改版後每個 UID 從自己的空白模型開始。
        if isinstance(data.get("arms"), dict):
            store = _new_state_store()
            store["legacy_shared_state_detected"] = True
            store["legacy_shared_total_updates"] = int(
                data.get("total_updates", 0) or 0
            )
            return store

        raise ValueError("unsupported state store schema")
    except Exception:
        return _new_state_store()


def _get_user_state_unlocked(
    state_store: Dict[str, Any],
    user_id: str,
    *,
    create: bool,
) -> tuple[str, Dict[str, Any]]:
    uid_key = _uid_key(user_id)
    users = dict(state_store.get("users") or {})
    raw_state = users.get(uid_key)
    if isinstance(raw_state, Mapping):
        try:
            state = _normalize_user_state(raw_state)
        except Exception:
            state = _new_state()
    else:
        state = _new_state()

    if create:
        users[uid_key] = state
        state_store["users"] = users
    return uid_key, state


def _write_state_unlocked(state_store: Dict[str, Any]) -> None:
    state_store["schema_version"] = STATE_SCHEMA_VERSION
    state_store["version"] = MODEL_VERSION
    state_store["context_dim"] = CONTEXT_DIM
    state_store["feature_names"] = list(FEATURE_NAMES)
    state_store["alpha"] = CMAB_ALPHA
    state_store["l2"] = CMAB_L2
    state_store["updated_at"] = int(time.time())
    state_store["users"] = dict(state_store.get("users") or {})
    CMAB_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CMAB_STATE_FILE.with_suffix(CMAB_STATE_FILE.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state_store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(CMAB_STATE_FILE)

def _arm_metrics(
    state: Mapping[str, Any],
    arm: str,
    context: np.ndarray,
) -> Dict[str, float]:
    """以 solve 取代顯式矩陣反矩陣，提升穩定性與速度。"""
    arm_state = dict(
        dict(state.get("arms") or {}).get(arm) or {}
    )
    A = np.asarray(arm_state.get("A"), dtype=np.float64)
    b = np.asarray(arm_state.get("b"), dtype=np.float64)

    # 避免長期浮點累積造成 A 輕微不對稱。
    A = 0.5 * (A + A.T)

    try:
        theta = np.linalg.solve(A, b)
        solved_context = np.linalg.solve(A, context)
    except np.linalg.LinAlgError:
        regularized = A + np.eye(
            CONTEXT_DIM,
            dtype=np.float64,
        ) * 1e-8
        try:
            theta = np.linalg.solve(regularized, b)
            solved_context = np.linalg.solve(
                regularized,
                context,
            )
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(regularized)
            theta = A_inv @ b
            solved_context = A_inv @ context

    estimate = float(theta @ context)
    variance = float(
        max(0.0, float(context @ solved_context))
    )
    uncertainty = float(math.sqrt(variance))
    exploration = float(CMAB_ALPHA * uncertainty)

    return {
        "estimate": estimate,
        "variance": variance,
        "uncertainty": uncertainty,
        "confidence_interval_half_width": exploration,
        "confidence_interval_full_width": 2.0 * exploration,
        "exploration": exploration,
        "score": estimate + exploration,
        "updates": int(arm_state.get("updates", 0) or 0),
        "weighted_updates": float(
            arm_state.get(
                "weighted_updates",
                arm_state.get("updates", 0),
            )
            or 0.0
        ),
        "reward_sum": float(
            arm_state.get("reward_sum", 0.0) or 0.0
        ),
        "weighted_reward_sum": float(
            arm_state.get(
                "weighted_reward_sum",
                arm_state.get("reward_sum", 0.0),
            )
            or 0.0
        ),
    }

def _softmax_two(score_b: float, score_p: float) -> Dict[str, float]:
    values = np.asarray([score_b, score_p], dtype=np.float64) / max(0.10, CMAB_SCORE_TEMPERATURE)
    values -= float(np.max(values))
    exp_values = np.exp(np.clip(values, -40.0, 40.0))
    total = float(exp_values.sum()) or 1.0
    return {"B": float(exp_values[0] / total), "P": float(exp_values[1] / total), "T": 0.0}


def _fallback_direction(history: Sequence[str], road_context: Mapping[str, Any]) -> str:
    for key in ("direction", "planning_direction", "recent_direction"):
        value = str(road_context.get(key) or "").upper().strip()
        if value in ARMS:
            return value
    planning = float(road_context.get("planning_probability", 0.5) or 0.5)
    recent = float(road_context.get("recent_probability", 0.5) or 0.5)
    if abs(planning - 0.5) > 1e-9:
        return "B" if planning >= 0.5 else "P"
    if abs(recent - 0.5) > 1e-9:
        return "B" if recent >= 0.5 else "P"
    bp = [value for value in history if value in ARMS]
    return bp[-1] if bp else "B"



def _short_term_trend_buffer(
    history: Sequence[str],
    fallback_direction: str,
) -> Dict[str, Any]:
    """只使用最新 3 個 B/P 建立 OOD 微觀趨勢先驗。"""
    buffer = _current_short_term_buffer(history)
    fallback = (
        fallback_direction
        if fallback_direction in ARMS
        else "B"
    )

    if len(buffer) >= 2 and buffer[-1] == buffer[-2]:
        direction = buffer[-1]
        strategy = "follow_last_two_streak"
        strength = 0.60
        evidence = buffer[-2:]
    elif (
        len(buffer) >= 3
        and buffer[-3] == buffer[-1]
        and buffer[-2] != buffer[-1]
    ):
        direction = "P" if buffer[-1] == "B" else "B"
        strategy = "continue_three_step_alternation"
        strength = 0.57
        evidence = buffer[-3:]
    elif len(buffer) >= 3:
        banker = sum(value == "B" for value in buffer)
        direction = "B" if banker >= 2 else "P"
        strategy = "recent_three_majority"
        strength = 0.55
        evidence = buffer[-3:]
    else:
        direction = fallback
        strategy = "insufficient_micro_history_fallback"
        strength = 0.52
        evidence = buffer[-3:]

    opposite = "P" if direction == "B" else "B"
    probabilities = {
        direction: strength,
        opposite: 1.0 - strength,
        "T": 0.0,
    }
    return {
        "direction": direction,
        "direction_text": "莊" if direction == "B" else "閒",
        "strategy": strategy,
        "strength": strength,
        "evidence": list(evidence),
        "short_term_buffer": list(buffer),
        "probabilities": probabilities,
        "short_term_weight": round(
            1.0 - CMAB_UNKNOWN_LONG_TERM_WEIGHT,
            6,
        ),
        "long_term_weight": round(
            CMAB_UNKNOWN_LONG_TERM_WEIGHT,
            6,
        ),
        "meta_learning_takeover": True,
    }

def _uncertainty_braking_metrics(
    metrics: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    """計算 LinUCB 的精確 xᵀA⁻¹x 與 UID 動態 OOD 閾值。

    每個 Arm 的 variance 均為 xᵀA_arm⁻¹x。
    action-space variance 使用兩個 Arm variance 的幾何平均；
    這能保留對雙邊新穎度的敏感度，同時避免單一未選 Arm
    永久讓系統停留在 OOD 狀態。
    """
    variance_b = max(
        0.0,
        float(metrics["B"].get("variance", 0.0) or 0.0),
    )
    variance_p = max(
        0.0,
        float(metrics["P"].get("variance", 0.0) or 0.0),
    )
    std_b = math.sqrt(variance_b)
    std_p = math.sqrt(variance_p)

    action_space_variance = math.sqrt(
        max(0.0, variance_b * variance_p)
    )
    action_space_std = math.sqrt(action_space_variance)
    decision_gap_variance = variance_b + variance_p
    decision_gap_std = math.sqrt(decision_gap_variance)

    baseline = _baseline_summary(state)
    threshold = float(baseline["threshold"])
    active = action_space_std > threshold
    severity_ratio = (
        action_space_std / max(threshold, 1e-12)
    )

    if active and severity_ratio >= 1.50:
        uncertainty_level = "extreme"
    elif active:
        uncertainty_level = "high"
    elif severity_ratio >= 0.80:
        uncertainty_level = "elevated"
    else:
        uncertainty_level = "normal"

    return {
        "active": bool(active),
        "uncertainty_level": uncertainty_level,
        "threshold_mode": (
            "dynamic_mean_plus_1_5_std"
            if baseline["dynamic_ready"]
            else "cold_start_fallback"
        ),
        "threshold_std": threshold,
        "dynamic_threshold_std": threshold,
        "historical_mean_std": float(baseline["mean"]),
        "historical_std_of_std": float(baseline["std"]),
        "historical_sample_count": int(
            baseline["sample_count"]
        ),
        "dynamic_threshold_ready": bool(
            baseline["dynamic_ready"]
        ),
        "sigma_multiplier": float(
            baseline["sigma_multiplier"]
        ),
        "action_space_std": float(action_space_std),
        "action_space_variance": float(
            action_space_variance
        ),
        "decision_gap_std": float(decision_gap_std),
        "decision_gap_variance": float(
            decision_gap_variance
        ),
        "severity_ratio": float(severity_ratio),
        "per_arm_std": {
            "B": float(std_b),
            "P": float(std_p),
        },
        "per_arm_variance": {
            "B": float(variance_b),
            "P": float(variance_p),
        },
        "per_arm_ci_half_width": {
            "B": float(
                metrics["B"].get(
                    "confidence_interval_half_width",
                    0.0,
                )
                or 0.0
            ),
            "P": float(
                metrics["P"].get(
                    "confidence_interval_half_width",
                    0.0,
                )
                or 0.0
            ),
        },
        "few_shot_update_weight": (
            float(CMAB_UNKNOWN_UPDATE_MULTIPLIER)
            if active
            else 1.0
        ),
        "observe_required": bool(active),
        "bet_multiplier": 0.0 if active else 1.0,
        "downstream_signal_code": (
            "OOD_EXTREME_UNCERTAINTY_OBSERVE"
            if active
            else "IN_DISTRIBUTION"
        ),
    }

def predict_bandit(
    history: Iterable[Any],
    *,
    road_context: Optional[Mapping[str, Any]] = None,
    venue: str = "",
    room: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    raw_history = _clean_history(history)
    road = dict(road_context or {})
    vector = build_context_vector(
        raw_history,
        road_context=road,
    )
    context = np.asarray(vector, dtype=np.float64)
    prediction_fingerprint = _prediction_fingerprint(
        raw_history,
        venue=venue,
        room=room,
        context=vector,
    )

    with _LOCK:
        state_store = _read_state_unlocked()
        uid_key, state = _get_user_state_unlocked(
            state_store,
            user_id,
            create=True,
        )
        metrics = {
            arm: _arm_metrics(state, arm, context)
            for arm in ARMS
        }

        score_b = metrics["B"]["score"]
        score_p = metrics["P"]["score"]
        if abs(score_b - score_p) <= 1e-12:
            base_direction = _fallback_direction(
                raw_history,
                road,
            )
            tie_break = True
        else:
            base_direction = (
                "B" if score_b > score_p else "P"
            )
            tie_break = False

        base_probabilities = _softmax_two(
            score_b,
            score_p,
        )
        braking = _uncertainty_braking_metrics(
            metrics,
            state,
        )
        short_term = _short_term_trend_buffer(
            raw_history,
            base_direction,
        )

        if bool(braking["active"]):
            direction = str(short_term["direction"])
            probabilities = dict(
                short_term["probabilities"]
            )
            action_code = "O"
            action_text = "觀望／降低注碼"
            direction_source = (
                "ood_meta_learning_short_term_prior"
            )
            signal_reason = (
                "極高不確定性防禦："
                f"action-space std "
                f"{float(braking['action_space_std']):.4f} > "
                f"動態門檻 "
                f"{float(braking['threshold_std']):.4f}；"
                f"長週期權重降至 "
                f"{float(short_term['long_term_weight']) * 100.0:.1f}%，"
                f"以最近 3 局策略 "
                f"{short_term['strategy']} 建立方向先驗，"
                "但本局建議觀望／0%配置。"
            )
        else:
            direction = base_direction
            probabilities = dict(base_probabilities)
            action_code = direction
            action_text = (
                "莊" if direction == "B" else "閒"
            )
            direction_source = (
                "contextual_bandit_linu_cb"
            )
            signal_reason = (
                "LinUCB 依目前牌路上下文、歷史回饋、"
                "探索上界與 UID 動態不確定性基準選擇莊／閒 Arm"
            )

        margin = abs(score_b - score_p)
        total_updates = int(
            state.get("total_updates", 0) or 0
        )
        maturity = 1.0 - math.exp(
            -total_updates / 80.0
        )
        quality = min(
            0.95,
            0.34
            + 0.36 * maturity
            + 0.25 * math.tanh(margin),
        )
        if sum(value in ARMS for value in raw_history) < 8:
            quality = min(quality, 0.45)
        if bool(braking["active"]):
            quality = min(
                quality,
                CMAB_UNKNOWN_CONFIDENCE_CAP,
            )

        direction_edge = abs(
            probabilities["B"] - probabilities["P"]
        )
        consistency = min(
            1.0,
            0.50
            + 0.50 * math.tanh(margin * 1.5),
        )
        selected = metrics[direction]
        few_shot_weight = float(
            braking["few_shot_update_weight"]
        )

        # 更新 UID 專屬的最新 3 局 Buffer 與「分布內」不確定性基準。
        state["short_term_buffer"] = list(
            short_term["short_term_buffer"]
        )
        baseline_after = _update_uncertainty_baseline(
            state,
            action_space_std=float(
                braking["action_space_std"]
            ),
            unknown_region_active=bool(
                braking["active"]
            ),
            prediction_fingerprint=prediction_fingerprint,
        )
        context_hash = _context_fingerprint(vector)
        state["last_prediction_risk"] = {
            "prediction_fingerprint": prediction_fingerprint,
            "context_hash": context_hash,
            "unknown_region_active": bool(
                braking["active"]
            ),
            "few_shot_update_weight": few_shot_weight,
            "action_space_std": float(
                braking["action_space_std"]
            ),
            "dynamic_threshold_std": float(
                braking["threshold_std"]
            ),
            "selected_arm": direction,
            "recommended_action": action_code,
            "created_at": int(time.time()),
        }
        state["updated_at"] = int(time.time())
        state_store["users"][uid_key] = state
        _write_state_unlocked(state_store)

    confidence_label = (
        "極低"
        if braking["active"]
        else "較高"
        if quality >= 0.72
        else "中等"
        if quality >= 0.50
        else "偏低"
    )
    signal_allowed = not bool(braking["active"])
    risk_signal = {
        "code": str(
            braking["downstream_signal_code"]
        ),
        "ood_detected": bool(braking["active"]),
        "extreme_uncertainty": bool(
            braking["active"]
        ),
        "observe_required": bool(
            braking["active"]
        ),
        "bet_multiplier": (
            0.0 if braking["active"] else 1.0
        ),
        "confidence_cap": (
            float(CMAB_UNKNOWN_CONFIDENCE_CAP)
            if braking["active"]
            else 1.0
        ),
        "few_shot_update_weight": few_shot_weight,
        "meta_direction": direction,
        "meta_direction_text": (
            "莊" if direction == "B" else "閒"
        ),
    }

    return {
        "ok": True,
        "engine": (
            "CONTEXTUAL_MULTI_ARMED_BANDIT_LINUCB"
        ),
        "model_version": MODEL_VERSION,
        "model_core": (
            "contextual_multi_armed_bandit_linu_cb"
        ),
        "mode": "screen_contextual_bandit",
        "probabilities": dict(probabilities),
        "pre_braking_probabilities": dict(
            base_probabilities
        ),
        "banker_rate": round(
            probabilities["B"] * 100.0,
            2,
        ),
        "player_rate": round(
            probabilities["P"] * 100.0,
            2,
        ),
        "tie_rate": 0.0,
        # 高方差時仍保留微觀先驗方向供介面顯示與線上學習；
        # 真正動作則切換為 O，讓既有資金模組自動給 0 元。
        "recommend": direction,
        "recommend_text": (
            "莊" if direction == "B" else "閒"
        ),
        "action": action_code,
        "action_text": action_text,
        "internal_recommend": direction,
        "internal_action": action_code,
        "signal_allowed": signal_allowed,
        "signal_status_code": str(
            braking["downstream_signal_code"]
        ),
        "signal_status_text": (
            "極高不確定性：觀望／降低注碼"
            if braking["active"]
            else "cMAB 下一局方向評估"
        ),
        "signal_reason": signal_reason,
        "internal_signal_reason": signal_reason,
        "selected_arm": direction,
        "base_bandit_direction": base_direction,
        "base_bandit_direction_text": (
            "莊" if base_direction == "B" else "閒"
        ),
        "next_round_direction": direction,
        "next_round_direction_text": (
            "莊" if direction == "B" else "閒"
        ),
        "direction_source": direction_source,
        "direction_edge": float(direction_edge),
        "direction_edge_percent": round(
            direction_edge * 100.0,
            4,
        ),
        "quality_score": float(quality),
        "confidence_label": confidence_label,
        "model_consistency": float(consistency),
        "uncertainty": float(
            selected["uncertainty"]
        ),
        "prediction_variance": float(
            selected["variance"]
        ),
        "unknown_region_active": bool(
            braking["active"]
        ),
        "extreme_uncertainty_signal": bool(
            braking["active"]
        ),
        "few_shot_update_weight": few_shot_weight,
        "bet_multiplier": (
            0.0 if braking["active"] else 1.0
        ),
        "uncertainty_braking": {
            **braking,
            "base_direction": base_direction,
            "selected_direction": direction,
            "recommended_action": action_code,
            "baseline_after_prediction": {
                "sample_count": int(
                    baseline_after["sample_count"]
                ),
                "mean": float(
                    baseline_after["mean"]
                ),
                "std": float(
                    baseline_after["std"]
                ),
                "threshold": float(
                    baseline_after["threshold"]
                ),
                "dynamic_ready": bool(
                    baseline_after["dynamic_ready"]
                ),
            },
            "short_term_buffer": dict(short_term),
        },
        "short_term_trend_buffer": dict(
            short_term
        ),
        "meta_learning_takeover": {
            "active": bool(braking["active"]),
            "short_term_buffer": list(
                short_term["short_term_buffer"]
            ),
            "strategy": str(
                short_term["strategy"]
            ),
            "long_term_weight": float(
                short_term["long_term_weight"]
            ),
            "short_term_weight": float(
                short_term["short_term_weight"]
            ),
            "prior_direction": direction,
            "prior_probabilities": dict(
                short_term["probabilities"]
            ),
        },
        # predictor.py 與未來校準器可直接讀取這兩組欄位。
        "predictor_signal": dict(risk_signal),
        "online_calibrator_signal": {
            **risk_signal,
            "calibration_weight": (
                0.15 if braking["active"] else 1.0
            ),
            "confidence_scale": (
                0.25 if braking["active"] else 1.0
            ),
        },
        "bandit_context": list(vector),
        "context_vector": list(vector),
        "context_feature_names": list(
            FEATURE_NAMES
        ),
        "bandit_scores": {
            arm: {
                key: (
                    int(value)
                    if key == "updates"
                    else round(float(value), 10)
                )
                for key, value in metrics[arm].items()
            }
            for arm in ARMS
        },
        "bandit_state": {
            "total_updates": total_updates,
            "total_weighted_updates": float(
                state.get(
                    "total_weighted_updates",
                    total_updates,
                )
                or 0.0
            ),
            "arm_updates": {
                arm: int(metrics[arm]["updates"])
                for arm in ARMS
            },
            "arm_weighted_updates": {
                arm: float(
                    metrics[arm]["weighted_updates"]
                )
                for arm in ARMS
            },
            "alpha": CMAB_ALPHA,
            "l2": CMAB_L2,
            "unknown_std_threshold": (
                CMAB_UNKNOWN_STD_THRESHOLD
            ),
            "dynamic_threshold_std": float(
                braking["threshold_std"]
            ),
            "dynamic_threshold_ready": bool(
                braking["dynamic_threshold_ready"]
            ),
            "uncertainty_history_count": int(
                braking["historical_sample_count"]
            ),
            "uncertainty_sigma_multiplier": (
                CMAB_UNCERTAINTY_SIGMA
            ),
            "unknown_update_multiplier": (
                CMAB_UNKNOWN_UPDATE_MULTIPLIER
            ),
            "state_file": str(CMAB_STATE_FILE),
            "cold_start_tie_break": tie_break,
        },
        "calibration": {
            "active": total_updates > 0,
            "scope": "cmab_online_reward",
            "sample_count": total_updates,
            "reason": (
                "cMAB 直接使用每局 reward 更新；"
                "高方差時另輸出 online_calibrator_signal"
            ),
        },
        "adaptive_ensemble": {
            "active": False,
            "effective_share": 0.0,
            "sample_count": total_updates,
            "reason": (
                "已由 LinUCB 線上更新取代舊自適應 Stacking"
            ),
        },
        "venue": str(venue or ""),
        "room": str(room or ""),
        "user_id": str(user_id or ""),
        "run_seed": run_seed,
        "input_required": False,
        "disclaimer": (
            "莊／閒百分比是 cMAB 方向分數正規化結果，"
            "不是真實開出機率；高不確定性時系統會建議觀望。"
        ),
    }

def update_bandit(
    *,
    context: Sequence[float],
    selected_arm: str,
    reward: Optional[float],
    event_id: str = "",
    actual_outcome: str = "",
    update_weight: float = 1.0,
    user_id: str = "",
) -> Dict[str, Any]:
    arm = str(selected_arm or "").upper().strip()
    if arm not in ARMS:
        raise ValueError("selected_arm must be B or P")

    if reward is None:
        return {
            "updated": False,
            "reason": "tie_or_skipped_reward",
            "selected_arm": arm,
            "actual_outcome": str(
                actual_outcome or ""
            ).upper(),
        }

    x = np.asarray(list(context), dtype=np.float64)
    if x.shape != (CONTEXT_DIM,):
        raise ValueError(
            f"context must contain {CONTEXT_DIM} values, "
            f"got {x.shape}"
        )
    x = np.clip(x, -1.0, 1.0)
    context_hash = _context_fingerprint(x)
    reward_value = max(
        0.0,
        min(1.0, float(reward)),
    )
    requested_weight = max(
        0.25,
        min(12.0, float(update_weight)),
    )
    event_key = str(event_id or "").strip()

    with _LOCK:
        state_store = _read_state_unlocked()
        uid_key, state = _get_user_state_unlocked(
            state_store,
            user_id,
            create=True,
        )
        applied = list(
            state.get("applied_event_ids") or []
        )
        if event_key and event_key in applied:
            return {
                "updated": False,
                "reason": "duplicate_event",
                "event_id": event_key,
                "total_updates": int(
                    state.get("total_updates", 0) or 0
                ),
            }

        last_risk = dict(
            state.get("last_prediction_risk") or {}
        )
        same_high_variance_context = bool(
            last_risk.get("unknown_region_active")
            and str(last_risk.get("context_hash") or "")
            == context_hash
        )
        boost_requested = (
            requested_weight > 1.0
            or same_high_variance_context
        )

        # 高方差區間強制 4～5 倍；一般區間固定 1 倍。
        observation_weight = (
            max(
                4.0,
                min(
                    5.0,
                    requested_weight
                    if requested_weight > 1.0
                    else CMAB_UNKNOWN_UPDATE_MULTIPLIER,
                ),
            )
            if boost_requested
            else 1.0
        )

        arm_state = dict(state["arms"][arm])
        A = np.asarray(
            arm_state["A"],
            dtype=np.float64,
        )
        b = np.asarray(
            arm_state["b"],
            dtype=np.float64,
        )

        # Weighted Bayesian ridge / LinUCB update：
        # A += w xxᵀ，b += w r x。
        A += observation_weight * np.outer(x, x)
        b += (
            observation_weight
            * reward_value
            * x
        )
        arm_state["A"] = A.tolist()
        arm_state["b"] = b.tolist()
        arm_state["updates"] = (
            int(arm_state.get("updates", 0) or 0)
            + 1
        )
        arm_state["weighted_updates"] = float(
            arm_state.get(
                "weighted_updates",
                arm_state["updates"] - 1,
            )
            or 0.0
        ) + observation_weight
        arm_state["reward_sum"] = float(
            arm_state.get("reward_sum", 0.0) or 0.0
        ) + reward_value
        arm_state["weighted_reward_sum"] = float(
            arm_state.get(
                "weighted_reward_sum",
                arm_state["reward_sum"] - reward_value,
            )
            or 0.0
        ) + observation_weight * reward_value

        state["arms"][arm] = arm_state
        state["total_updates"] = (
            int(state.get("total_updates", 0) or 0)
            + 1
        )
        state["total_weighted_updates"] = float(
            state.get(
                "total_weighted_updates",
                state["total_updates"] - 1,
            )
            or 0.0
        ) + observation_weight

        if event_key:
            applied.append(event_key)
            state["applied_event_ids"] = applied[
                -CMAB_MAX_EVENT_IDS:
            ]

        state["last_update"] = {
            "event_id": event_key,
            "selected_arm": arm,
            "actual_outcome": str(
                actual_outcome or ""
            ).upper(),
            "reward": reward_value,
            "requested_weight": requested_weight,
            "applied_weight": observation_weight,
            "few_shot_boost_applied": bool(
                observation_weight >= 4.0
            ),
            "same_high_variance_context": bool(
                same_high_variance_context
            ),
            "context_hash": context_hash,
            "updated_at": int(time.time()),
        }
        state["updated_at"] = int(time.time())
        state_store["users"][uid_key] = state
        _write_state_unlocked(state_store)

    return {
        "updated": True,
        "event_id": event_key,
        "selected_arm": arm,
        "actual_outcome": str(
            actual_outcome or ""
        ).upper(),
        "reward": reward_value,
        "requested_update_weight": requested_weight,
        "update_weight": observation_weight,
        "few_shot_boost_applied": bool(
            observation_weight >= 4.0
        ),
        "boost_reason": (
            "high_variance_previous_prediction"
            if observation_weight >= 4.0
            else "standard_update"
        ),
        "arm_updates": int(arm_state["updates"]),
        "arm_weighted_updates": float(
            arm_state["weighted_updates"]
        ),
        "total_updates": int(
            state["total_updates"]
        ),
        "total_weighted_updates": float(
            state["total_weighted_updates"]
        ),
    }

def get_bandit_summary(
    user_id: str = "",
) -> Dict[str, Any]:
    with _LOCK:
        state_store = _read_state_unlocked()
        _, state = _get_user_state_unlocked(
            state_store,
            user_id,
            create=False,
        )
        baseline = _baseline_summary(state)

    return {
        "version": state.get(
            "version",
            MODEL_VERSION,
        ),
        "context_dim": CONTEXT_DIM,
        "feature_names": list(FEATURE_NAMES),
        "total_updates": int(
            state.get("total_updates", 0) or 0
        ),
        "total_weighted_updates": float(
            state.get(
                "total_weighted_updates",
                state.get("total_updates", 0),
            )
            or 0.0
        ),
        "unknown_std_threshold": float(
            CMAB_UNKNOWN_STD_THRESHOLD
        ),
        "dynamic_uncertainty_threshold": {
            "ready": bool(
                baseline["dynamic_ready"]
            ),
            "current_threshold": float(
                baseline["threshold"]
            ),
            "historical_mean": float(
                baseline["mean"]
            ),
            "historical_std": float(
                baseline["std"]
            ),
            "sample_count": int(
                baseline["sample_count"]
            ),
            "sigma_multiplier": float(
                CMAB_UNCERTAINTY_SIGMA
            ),
        },
        "unknown_update_multiplier": float(
            CMAB_UNKNOWN_UPDATE_MULTIPLIER
        ),
        "short_term_buffer": list(
            state.get("short_term_buffer") or []
        )[-3:],
        "last_prediction_risk": dict(
            state.get("last_prediction_risk") or {}
        ),
        "arms": {
            arm: {
                "updates": int(
                    state["arms"][arm].get(
                        "updates",
                        0,
                    )
                    or 0
                ),
                "weighted_updates": float(
                    state["arms"][arm].get(
                        "weighted_updates",
                        state["arms"][arm].get(
                            "updates",
                            0,
                        ),
                    )
                    or 0.0
                ),
                "reward_sum": float(
                    state["arms"][arm].get(
                        "reward_sum",
                        0.0,
                    )
                    or 0.0
                ),
                "weighted_reward_sum": float(
                    state["arms"][arm].get(
                        "weighted_reward_sum",
                        state["arms"][arm].get(
                            "reward_sum",
                            0.0,
                        ),
                    )
                    or 0.0
                ),
            }
            for arm in ARMS
        },
        "state_file": str(CMAB_STATE_FILE),
    }

__all__ = ["ARMS", "CONTEXT_DIM", "FEATURE_NAMES", "MODEL_VERSION",
           "build_context_vector", "get_bandit_summary", "predict_bandit", "update_bandit"]
