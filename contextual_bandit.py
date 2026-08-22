"""BGS cMAB 主模型：LinUCB 上下文相關多臂老虎機。

主要方向只有兩個 Arm：B（莊）與 P（閒）。
本模組不使用粒子濾波、超幾何分布、蒙地卡羅或 Stacking。
每次預測保存上下文向量；使用者回報實際結果後，再以 reward 更新 Arm。
和局不更新 B/P Arm。

V2.1 非平穩牌路版（移除一至三階條件轉移訊號）：
- 精確使用每個 Arm 的 x^T A^-1 x 預測方差與置信區間寬度。
- 另以共享 context information matrix 的 x^T A_ctx^-1 x 判定牌路新穎度，
  避免某一個 Arm 樣本較少時讓系統永久誤判為 OOD。
- 每個 UID 以經驗方差歷史平均 + 1.3 個標準差建立動態門檻。
- 同時計算最近 12 局排列熵、B/P 卡方與經驗方差診斷。
- 只有「模型方差超標且排列熵 > 0.95」才標記極端混沌區間。
- OOD 必須等動態方差基準與真實回饋都成熟，避免新 UID 假熔斷。
- 上下文新增最近 3 局、5 局轉換波動、斷龍、長龍尾端、單跳開端，
  以及大眼仔／小路／曱甴路的規律飽和度。
- 以遞迴 ridge forgetting factor 淡化舊鞋路；偵測斷龍時加速遺忘，
  但不複製單局樣本，避免 Few-shot 對亂數過擬合。
- reward 依「預測時信心」調整：高信心命中略增益，高信心失誤重罰。
- 完全停用 4～5 倍 Few-shot；每局一律 1 倍更新，避免放大隨機噪音。
- B/P 結果揭曉後以完整資訊同時更新兩個 Arm，消除選擇偏誤。

banker_rate / player_rate 是方向分數正規化結果，不是真實開出機率。
"""
from __future__ import annotations

from collections import Counter
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
MODEL_VERSION = "CMAB-LINUCB-V2.1-NO-TRANSITION-CHAIN"
STATE_SCHEMA_VERSION = "CMAB-UID-ISOLATED-V4"
COMPATIBLE_STATE_SCHEMA_VERSIONS = {
    "CMAB-UID-ISOLATED-V1",
    "CMAB-UID-ISOLATED-V2",
    "CMAB-UID-ISOLATED-V3",
    STATE_SCHEMA_VERSION,
}
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
    "road_agreement", "reserved_removed_sequence_slot_1",
    "reserved_removed_sequence_slot_2", "reserved_removed_sequence_slot_3",
    # V2.0 全部附加在舊 24 維之後，讓既有 A／b 可無損升維。
    "shoe_round_maturity", "recent3_banker_balance",
    "recent5_transition_volatility", "recent5_run_volatility",
    "streak_break_signal", "long_dragon_tail_pressure",
    "single_jump_onset", "big_eye_saturation",
    "small_road_saturation", "cockroach_road_saturation",
    "derived_road_consensus",
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
    "CMAB_SCORE_TEMPERATURE", 0.95, 0.10, 10.0
)
CMAB_TIE_PRIOR = 0.095156
CMAB_TIE_PRIOR_STRENGTH = _env_float(
    "CMAB_TIE_PRIOR_STRENGTH", 40.0, 10.0, 500.0
)
CMAB_MIN_SIGNAL_EDGE = _env_float(
    "CMAB_MIN_SIGNAL_EDGE", 0.012, 0.0, 0.20
)
CMAB_MIN_SIGNAL_UPDATES = max(
    0,
    min(200, int(os.getenv("CMAB_MIN_SIGNAL_UPDATES", "4") or "4")),
)

# 遞迴 ridge 衰減：A <- lambda*A + (1-lambda)*ridge*I + xx^T。
CMAB_FORGETTING_FACTOR = _env_float(
    "CMAB_FORGETTING_FACTOR", 0.985, 0.90, 1.00
)
CMAB_REVERSAL_FORGETTING_FACTOR = _env_float(
    "CMAB_REVERSAL_FORGETTING_FACTOR", 0.94, 0.85, 1.00
)

# 動態 OOD 閾值：歷史平均 + sigma × 歷史標準差。
CMAB_UNCERTAINTY_SIGMA = _env_float(
    "CMAB_UNCERTAINTY_SIGMA", 1.30, 0.50, 4.00
)
CMAB_PERMUTATION_WINDOW = 12
CMAB_PERMUTATION_ORDER = 3
CMAB_PERMUTATION_ENTROPY_THRESHOLD = 0.95
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

# V1.5 起完全停用 Few-shot 加權；保留名稱僅供既有監控欄位相容。
CMAB_UNKNOWN_UPDATE_MULTIPLIER = 1.0
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
CMAB_MAX_PENDING_OOD_CONTEXTS = max(
    8,
    min(
        128,
        int(os.getenv("CMAB_MAX_PENDING_OOD_CONTEXTS", "32") or "32"),
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


def _transition_rate(sequence: Sequence[str], size: int) -> float:
    """最近窗口的 B/P 轉換率；和局不推進牌路時間軸。"""
    values = list(sequence[-size:])
    if len(values) < 2:
        return 0.0
    return float(
        sum(a != b for a, b in zip(values, values[1:]))
        / (len(values) - 1)
    )


def _recent_run_volatility(sequence: Sequence[str], size: int = 5) -> float:
    """以窗口內 run length 的離散程度描述短週期震盪。"""
    values = list(sequence[-size:])
    if len(values) < 2:
        return 0.0
    runs: List[int] = []
    previous = ""
    for value in values:
        if runs and value == previous:
            runs[-1] += 1
        else:
            runs.append(1)
        previous = value
    if len(runs) <= 1:
        return 0.0
    return float(min(1.0, float(np.std(runs)) / 2.0))


def _streak_break_signal(sequence: Sequence[str]) -> float:
    """最新一局是否剛斷掉至少 3 顆的龍；符號代表新方向。"""
    values = list(sequence)
    if len(values) < 4 or values[-1] == values[-2]:
        return 0.0
    previous_side = values[-2]
    previous_run = 1
    for value in reversed(values[:-2]):
        if value != previous_side:
            break
        previous_run += 1
    if previous_run < 3:
        return 0.0
    sign = 1.0 if values[-1] == "B" else -1.0
    return sign * min(1.0, previous_run / 6.0)


def _long_dragon_tail_pressure(sequence: Sequence[str]) -> float:
    """長龍尾端壓力，而非宣稱能知道物理上的下一張牌。"""
    direction, length = _streak(sequence)
    if length < 4:
        return 0.0
    sign = 1.0 if direction == "B" else -1.0
    return sign * min(1.0, (length - 3) / 5.0)


def _single_jump_onset(sequence: Sequence[str]) -> float:
    """最近 4 局是否形成單跳開端；符號代表最新落點。"""
    values = list(sequence[-4:])
    if len(values) < 4 or not all(
        left != right for left, right in zip(values, values[1:])
    ):
        return 0.0
    return 1.0 if values[-1] == "B" else -1.0


def _derived_road_saturation(
    road_context: Mapping[str, Any],
    road_name: str,
) -> float:
    """下三路規律飽和度：0 為中性，1 為近期顏色／延續高度集中。"""
    planning = road_context.get("full_road_analysis")
    if not isinstance(planning, Mapping):
        models = road_context.get("models")
        full_road = models.get("full_road") if isinstance(models, Mapping) else None
        planning = full_road if isinstance(full_road, Mapping) else {}
    derived_stats = planning.get("derived_stats")
    stats = (
        derived_stats.get(road_name)
        if isinstance(derived_stats, Mapping)
        else None
    )
    if not isinstance(stats, Mapping):
        return 0.0
    balance = _clip(stats.get("balance", 0.0), 0.0, 1.0)
    continuation = _clip(
        stats.get("recent_continuation", 0.5), 0.0, 1.0
    )
    continuation_concentration = abs(2.0 * continuation - 1.0)
    return float(max(balance, continuation_concentration))


def build_context_vector(history: Iterable[Any], *, road_context: Optional[Mapping[str, Any]] = None) -> List[float]:
    """建立固定上下文；B/P 規律特徵不讓和局推進時間軸。"""
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
    big_eye_saturation = _derived_road_saturation(road, "big_eye")
    small_road_saturation = _derived_road_saturation(road, "small_road")
    cockroach_saturation = _derived_road_saturation(
        road, "cockroach_road"
    )
    derived_mean_saturation = (
        big_eye_saturation + small_road_saturation + cockroach_saturation
    ) / 3.0
    derived_consensus = _clip(
        derived_mean_saturation
        * (
            1.0
            - (
                abs(big_eye_saturation - small_road_saturation)
                + abs(small_road_saturation - cockroach_saturation)
                + abs(cockroach_saturation - big_eye_saturation)
            ) / 3.0
        ),
        0.0,
        1.0,
    )
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
        # 保留三個零值槽位以維持矩陣維度；舊欄位名稱不映射，
        # 因此既有條件轉移權重不會被帶入新版 A／b。
        0.0,
        0.0,
        0.0,
        min(1.0, len(raw) / 80.0),
        _banker_balance(bp, 3),
        _transition_rate(bp, 5),
        _recent_run_volatility(bp, 5),
        _streak_break_signal(bp),
        _long_dragon_tail_pressure(bp),
        _single_jump_onset(bp),
        big_eye_saturation,
        small_road_saturation,
        cockroach_saturation,
        derived_consensus,
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


def _rolling_randomness_diagnostics(history: Sequence[str]) -> Dict[str, Any]:
    """最近 12 局的排列熵、B/P 卡方與經驗方差。

    B/P/T 先映射為 +1/-1/0，再轉成累積隨機漫步；對該連續軌跡做
    order=3 的 Bandt-Pompe 排列熵。極小的確定性 dither 只用來處理
    累積值相同的 ordinal tie；它不改變相差至少 1 的大小關係，也不
    引入隨機數，因此二元序列仍可使用完整六種排列且回測完全可重現。
    """
    window = [
        value for value in history if value in {"B", "P", "T"}
    ][-CMAB_PERMUTATION_WINDOW:]
    ready = len(window) >= CMAB_PERMUTATION_WINDOW
    increments = np.asarray(
        [1.0 if value == "B" else -1.0 if value == "P" else 0.0 for value in window],
        dtype=np.float64,
    )
    walk = np.cumsum(increments)
    if walk.size:
        indices = np.arange(walk.size, dtype=np.float64)
        deterministic_dither = np.sin(
            (indices + 1.0) * math.sqrt(2.0)
        ) * 1e-12
        walk = walk + deterministic_dither

    patterns: Counter[tuple[int, ...]] = Counter()
    order = CMAB_PERMUTATION_ORDER
    for index in range(max(0, len(walk) - order + 1)):
        segment = walk[index:index + order]
        pattern = tuple(
            int(value)
            for value in np.argsort(segment, kind="mergesort")
        )
        patterns[pattern] += 1

    pattern_total = sum(patterns.values())
    if pattern_total:
        probabilities = [count / pattern_total for count in patterns.values()]
        raw_entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        maximum_entropy = math.log(math.factorial(order))
        permutation_entropy = raw_entropy / maximum_entropy
    else:
        permutation_entropy = 0.0
    permutation_entropy = max(0.0, min(1.0, float(permutation_entropy)))

    bp = [value for value in window if value in ARMS]
    banker_count = sum(value == "B" for value in bp)
    player_count = len(bp) - banker_count
    banker_binary = np.asarray(
        [1.0 if value == "B" else 0.0 for value in bp],
        dtype=np.float64,
    )
    empirical_variance = (
        float(np.var(banker_binary, ddof=1))
        if len(banker_binary) >= 2
        else 0.0
    )

    # 使用 B/P 排除和局後的長期基準比例；df=1 的 survival function
    # 可用 erfc(sqrt(chi2/2)) 精確表示，無需新增 scipy 相依套件。
    banker_expected_ratio = 0.458597 / (0.458597 + 0.446247)
    expected_banker = len(bp) * banker_expected_ratio
    expected_player = len(bp) - expected_banker
    chi_square = 0.0
    if expected_banker > 0.0 and expected_player > 0.0:
        chi_square = (
            (banker_count - expected_banker) ** 2 / expected_banker
            + (player_count - expected_player) ** 2 / expected_player
        )
    chi_square_p_value = math.erfc(math.sqrt(max(0.0, chi_square) / 2.0))

    return {
        "window_size": len(window),
        "required_window_size": CMAB_PERMUTATION_WINDOW,
        "ready": bool(ready),
        "outcomes": list(window),
        "permutation_order": int(order),
        "permutation_entropy": float(permutation_entropy),
        "permutation_entropy_threshold": float(
            CMAB_PERMUTATION_ENTROPY_THRESHOLD
        ),
        "entropy_indicates_white_noise": bool(
            ready
            and permutation_entropy > CMAB_PERMUTATION_ENTROPY_THRESHOLD
        ),
        "ordinal_pattern_count": int(pattern_total),
        "unique_ordinal_patterns": int(len(patterns)),
        "bp_sample_count": int(len(bp)),
        "banker_count": int(banker_count),
        "player_count": int(player_count),
        "banker_empirical_rate": float(
            banker_count / len(bp) if bp else 0.0
        ),
        "empirical_variance": float(empirical_variance),
        "chi_square_statistic": float(chi_square),
        "chi_square_p_value": float(chi_square_p_value),
    }


def _baseline_summary(state: Mapping[str, Any]) -> Dict[str, Any]:
    """共享 context 預測「方差」的歷史基準，單位保持為 variance。"""
    baseline = dict(state.get("variance_baseline") or {})
    values = _clean_uncertainty_values(baseline.get("values"))
    if not values:
        # V1.4 的 uncertainty_baseline 儲存的是 std；平方後無損遷移。
        legacy = dict(state.get("uncertainty_baseline") or {})
        legacy_std = _clean_uncertainty_values(legacy.get("values"))
        values = [float(value * value) for value in legacy_std]
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
            CMAB_DYNAMIC_THRESHOLD_FLOOR ** 2,
            mean + CMAB_UNCERTAINTY_SIGMA * std,
        )
        if dynamic_ready
        else max(
            CMAB_DYNAMIC_THRESHOLD_FLOOR ** 2,
            CMAB_UNKNOWN_STD_THRESHOLD ** 2,
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
        "unit": "variance",
        "fallback_threshold": float(CMAB_UNKNOWN_STD_THRESHOLD ** 2),
        "threshold_floor": float(CMAB_DYNAMIC_THRESHOLD_FLOOR ** 2),
    }


def _update_uncertainty_baseline(
    state: Dict[str, Any],
    *,
    action_space_variance: float,
    unknown_region_active: bool,
    prediction_fingerprint: str,
) -> Dict[str, Any]:
    """只用非極端樣本更新經驗方差基準，避免混沌值污染門檻。"""
    summary = _baseline_summary(state)
    values = list(summary["values"])
    previous_fingerprint = str(
        state.get("last_variance_fingerprint") or ""
    )
    is_new_prediction = (
        bool(prediction_fingerprint)
        and prediction_fingerprint != previous_fingerprint
    )

    if is_new_prediction and not unknown_region_active:
        values.append(max(0.0, float(action_space_variance)))
        values = values[-CMAB_UNCERTAINTY_HISTORY_SIZE:]

    state["variance_baseline"] = {
        "values": values,
        "last_observed_variance": max(0.0, float(action_space_variance)),
        "last_unknown_region_active": bool(unknown_region_active),
        "updated_at": int(time.time()),
    }
    state["last_variance_fingerprint"] = prediction_fingerprint
    return _baseline_summary(state)


def _current_short_term_buffer(history: Sequence[str]) -> List[str]:
    return [value for value in history if value in ARMS][-3:]


def _identity_information_matrix() -> np.ndarray:
    return np.eye(CONTEXT_DIM, dtype=np.float64) * CMAB_L2


def _feature_index_mapping(
    stored_dim: int,
    stored_feature_names: Any,
) -> Dict[int, int]:
    """將舊欄位映射到新版；舊版 24 維前綴可原位延續。"""
    names = (
        [str(value) for value in stored_feature_names]
        if isinstance(stored_feature_names, (list, tuple))
        else []
    )
    current = {name: index for index, name in enumerate(FEATURE_NAMES)}
    mapping: Dict[int, int] = {}
    if len(names) == stored_dim:
        for old_index, name in enumerate(names):
            if name in current:
                mapping[old_index] = current[name]
    if not mapping:
        mapping = {
            index: index
            for index in range(min(stored_dim, CONTEXT_DIM))
        }
    return mapping


def _migrate_matrix_and_vector(
    matrix_value: Any,
    vector_value: Any,
    *,
    stored_l2: float,
    stored_feature_names: Any,
) -> tuple[np.ndarray, np.ndarray]:
    old_matrix = np.asarray(matrix_value, dtype=np.float64)
    old_vector = np.asarray(vector_value, dtype=np.float64)
    if (
        old_matrix.ndim != 2
        or old_matrix.shape[0] != old_matrix.shape[1]
        or old_vector.shape != (old_matrix.shape[0],)
        or old_matrix.shape[0] <= 0
        or old_matrix.shape[0] > CONTEXT_DIM
        or not np.all(np.isfinite(old_matrix))
        or not np.all(np.isfinite(old_vector))
    ):
        raise ValueError("invalid UID state matrix shape")
    old_dim = int(old_matrix.shape[0])
    mapping = _feature_index_mapping(old_dim, stored_feature_names)
    migrated_matrix = (
        np.eye(CONTEXT_DIM, dtype=np.float64) * stored_l2
    )
    migrated_vector = np.zeros(CONTEXT_DIM, dtype=np.float64)
    symmetric_old = 0.5 * (old_matrix + old_matrix.T)
    for old_i, new_i in mapping.items():
        migrated_vector[new_i] = old_vector[old_i]
        for old_j, new_j in mapping.items():
            migrated_matrix[new_i, new_j] = symmetric_old[old_i, old_j]
    return migrated_matrix, migrated_vector


def _migrate_information_matrix(
    matrix_value: Any,
    *,
    stored_l2: float,
    stored_feature_names: Any,
) -> np.ndarray:
    old_matrix = np.asarray(matrix_value, dtype=np.float64)
    zeros = np.zeros(
        old_matrix.shape[0] if old_matrix.ndim == 2 else 0,
        dtype=np.float64,
    )
    matrix, _ = _migrate_matrix_and_vector(
        old_matrix,
        zeros,
        stored_l2=stored_l2,
        stored_feature_names=stored_feature_names,
    )
    return matrix


def _as_information_matrix(value: Any) -> np.ndarray:
    """讀取、驗證並對稱化 information matrix。"""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (CONTEXT_DIM, CONTEXT_DIM):
        raise ValueError("invalid information matrix shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("information matrix contains non-finite values")
    return 0.5 * (matrix + matrix.T)


def _reconstruct_shared_context_matrix(
    arms: Mapping[str, Any],
    *,
    prior_l2: float,
) -> np.ndarray:
    """從舊版 disjoint LinUCB matrices 無損重建共享 context matrix。

    A_B = lambda I + sum_B(w xxT)，A_P 同理，因此
    A_ctx = A_B + A_P - lambda I = lambda I + sum_all(w xxT)。
    """
    banker = _as_information_matrix(dict(arms["B"]).get("A"))
    player = _as_information_matrix(dict(arms["P"]).get("A"))
    prior = np.eye(CONTEXT_DIM, dtype=np.float64) * max(
        1e-12, float(prior_l2)
    )
    reconstructed = banker + player - prior
    return 0.5 * (reconstructed + reconstructed.T)


def _new_state() -> Dict[str, Any]:
    """建立單一 UID 專屬的 cMAB 狀態。"""
    identity = _identity_information_matrix().tolist()
    zeros = np.zeros(CONTEXT_DIM, dtype=np.float64).tolist()
    return {
        "version": MODEL_VERSION,
        "context_dim": CONTEXT_DIM,
        "feature_names": list(FEATURE_NAMES),
        "alpha": CMAB_ALPHA,
        "l2": CMAB_L2,
        "arms": {
            arm: {
                "A": [row[:] for row in identity],
                "b": list(zeros),
                "updates": 0,
                "weighted_updates": 0.0,
                "reward_sum": 0.0,
                "weighted_reward_sum": 0.0,
            }
            for arm in ARMS
        },
        # 只衡量「這個 context 看過多少」，與選擇哪個 Arm／reward 無關。
        # 這是 OOD braking 的主要 uncertainty matrix。
        "context_information": {
            "A": [row[:] for row in identity],
            "updates": 0,
            "weighted_updates": 0.0,
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
        "variance_baseline": {
            "values": [],
            "last_observed_variance": 0.0,
            "last_unknown_region_active": False,
            "updated_at": 0,
        },
        "short_term_buffer": [],
        "pending_ood_contexts": [],
        "last_uncertainty_fingerprint": "",
        "last_variance_fingerprint": "",
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
    stored_l2 = max(
        1e-12,
        float(state.get("l2", CMAB_L2) or CMAB_L2),
    )
    stored_dim = int(state.get("context_dim", 0) or 0)
    if stored_dim <= 0 or stored_dim > CONTEXT_DIM:
        raise ValueError("invalid UID state context dimension")
    stored_feature_names = state.get("feature_names")

    arms = state.get("arms")
    if not isinstance(arms, dict) or any(
        arm not in arms for arm in ARMS
    ):
        raise ValueError("missing UID state arms")

    for arm in ARMS:
        A, b = _migrate_matrix_and_vector(
            arms[arm].get("A"),
            arms[arm].get("b"),
            stored_l2=stored_l2,
            stored_feature_names=stored_feature_names,
        )
        arms[arm]["A"] = A.tolist()
        arms[arm]["b"] = b.tolist()

    # 舊 UID 狀態直接延續，不重置已學習的 A／b。
    state["version"] = MODEL_VERSION
    state["context_dim"] = CONTEXT_DIM
    state["feature_names"] = list(FEATURE_NAMES)
    state["alpha"] = CMAB_ALPHA
    # 已訓練矩陣內含建立當時的 ridge prior，不能只改 metadata 偽裝成新值。
    state["l2"] = stored_l2
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

    context_information = state.get("context_information")
    if isinstance(context_information, Mapping):
        context_matrix = _migrate_information_matrix(
            context_information.get("A"),
            stored_l2=stored_l2,
            stored_feature_names=stored_feature_names,
        )
        context_updates = int(
            context_information.get(
                "updates",
                state.get("total_updates", 0),
            )
            or 0
        )
        context_weighted_updates = float(
            context_information.get(
                "weighted_updates",
                state.get("total_weighted_updates", context_updates),
            )
            or 0.0
        )
    else:
        # V1 -> V2 就地遷移，不丟失既有 UID 的 A／b 學習成果。
        context_matrix = _reconstruct_shared_context_matrix(
            arms,
            prior_l2=stored_l2,
        )
        context_updates = int(state.get("total_updates", 0) or 0)
        context_weighted_updates = float(
            state.get("total_weighted_updates", context_updates) or 0.0
        )
    state["context_information"] = {
        "A": context_matrix.tolist(),
        "updates": max(0, context_updates),
        "weighted_updates": max(0.0, context_weighted_updates),
    }

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
    variance_baseline = dict(state.get("variance_baseline") or {})
    variance_values = _clean_uncertainty_values(
        variance_baseline.get("values")
    )
    if not variance_values:
        variance_values = [
            float(value * value)
            for value in state["uncertainty_baseline"]["values"]
        ]
    state["variance_baseline"] = {
        "values": variance_values,
        "last_observed_variance": max(
            0.0,
            float(
                variance_baseline.get("last_observed_variance", 0.0)
                or 0.0
            ),
        ),
        "last_unknown_region_active": bool(
            variance_baseline.get("last_unknown_region_active", False)
        ),
        "updated_at": int(variance_baseline.get("updated_at", 0) or 0),
    }
    state["short_term_buffer"] = [
        value
        for value in list(state.get("short_term_buffer") or [])
        if value in ARMS
    ][-3:]
    # V1.5 不再使用待加速 OOD 快取；部署後主動清除舊項目。
    state["pending_ood_contexts"] = []
    state["last_uncertainty_fingerprint"] = str(
        state.get("last_uncertainty_fingerprint") or ""
    )
    state["last_variance_fingerprint"] = str(
        state.get("last_variance_fingerprint") or ""
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
            str(data.get("schema_version") or "")
            in COMPATIBLE_STATE_SCHEMA_VERSIONS
            and isinstance(users, dict)
            and 0 < int(data.get("context_dim", 0) or 0) <= CONTEXT_DIM
        ):
            # 不在每次預測時正規化全部 UID；只在該 UID 被使用時驗證。
            # 使用者數量增加後，這可避免 O(number_of_users) 的矩陣轉換。
            retained_users: Dict[str, Any] = {}
            for uid_key, raw_state in users.items():
                if isinstance(raw_state, Mapping):
                    retained_users[str(uid_key)] = dict(raw_state)
            data["schema_version"] = STATE_SCHEMA_VERSION
            data["version"] = MODEL_VERSION
            data["context_dim"] = CONTEXT_DIM
            data["feature_names"] = list(FEATURE_NAMES)
            data["alpha"] = CMAB_ALPHA
            data["l2"] = CMAB_L2
            data["users"] = retained_users
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
    users = state_store.get("users")
    if not isinstance(users, dict):
        users = {}
        state_store["users"] = users
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
    """精確計算 disjoint LinUCB mean 與 x^T A^-1 x。

    使用同一個 solve 的雙 RHS [b, x]，只分解 A 一次；相較兩次
    solve 或顯式 inverse，速度更快且數值更穩定。
    """
    arm_state = dict(
        dict(state.get("arms") or {}).get(arm) or {}
    )
    A = np.asarray(arm_state.get("A"), dtype=np.float64)
    b = np.asarray(arm_state.get("b"), dtype=np.float64)

    # 避免長期浮點累積造成 A 輕微不對稱。
    A = 0.5 * (A + A.T)

    right_hand_sides = np.column_stack((b, context))
    solved: Optional[np.ndarray] = None
    regularized = A
    for jitter in (0.0, 1e-10, 1e-8, 1e-6):
        regularized = A + np.eye(CONTEXT_DIM, dtype=np.float64) * jitter
        try:
            solved = np.linalg.solve(regularized, right_hand_sides)
            break
        except np.linalg.LinAlgError:
            continue
    if solved is None:
        solved = np.linalg.pinv(regularized) @ right_hand_sides
    theta = solved[:, 0]
    solved_context = solved[:, 1]

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


def _shared_context_uncertainty(
    state: Mapping[str, Any],
    context: np.ndarray,
) -> Dict[str, float]:
    """計算與 Arm／reward 無關的牌路 context 新穎度。"""
    information = dict(state.get("context_information") or {})
    matrix = _as_information_matrix(information.get("A"))
    solved_context: Optional[np.ndarray] = None
    regularized = matrix
    for jitter in (0.0, 1e-10, 1e-8, 1e-6):
        regularized = matrix + np.eye(CONTEXT_DIM, dtype=np.float64) * jitter
        try:
            solved_context = np.linalg.solve(regularized, context)
            break
        except np.linalg.LinAlgError:
            continue
    if solved_context is None:
        solved_context = np.linalg.pinv(regularized) @ context

    variance = max(0.0, float(context @ solved_context))
    return {
        "variance": float(variance),
        "std": float(math.sqrt(variance)),
        "confidence_interval_half_width": float(
            CMAB_ALPHA * math.sqrt(variance)
        ),
        "updates": int(information.get("updates", 0) or 0),
        "weighted_updates": float(
            information.get(
                "weighted_updates",
                information.get("updates", 0),
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


def _smoothed_tie_probability(history: Sequence[str]) -> float:
    """以標準八副牌先驗平滑觀測和局率，避免 T=0 破壞校準指標。"""
    sample_count = len(history)
    tie_count = sum(value == "T" for value in history)
    posterior = (
        tie_count + CMAB_TIE_PRIOR * CMAB_TIE_PRIOR_STRENGTH
    ) / max(1e-12, sample_count + CMAB_TIE_PRIOR_STRENGTH)
    return float(max(0.04, min(0.18, posterior)))


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


def _blend_ood_probabilities(
    long_term: Mapping[str, Any],
    short_term: Mapping[str, Any],
) -> Dict[str, float]:
    """OOD 時將長週期降權，但不把已學資訊完全切斷。"""
    long_weight = float(CMAB_UNKNOWN_LONG_TERM_WEIGHT)
    short_weight = 1.0 - long_weight
    values = {
        arm: (
            long_weight * float(long_term.get(arm, 0.5) or 0.0)
            + short_weight * float(short_term.get(arm, 0.5) or 0.0)
        )
        for arm in ARMS
    }
    total = max(1e-12, sum(values.values()))
    return {
        "B": float(values["B"] / total),
        "P": float(values["P"] / total),
        "T": 0.0,
    }


def _uncertainty_braking_metrics(
    metrics: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    shared_context: Mapping[str, Any],
    base_direction: str,
    randomness: Mapping[str, Any],
) -> Dict[str, Any]:
    """以模型方差與 12 局排列熵的 AND gate 判定極端混沌。

    每個 Arm 的 variance 均為 xᵀA_arm⁻¹x。
    OOD braking 使用共享 A_ctx 的 xᵀA_ctx⁻¹x：A_ctx 在任何 B/P
    實際結果回報後都會更新，因此準確代表「此 context 是否見過」，
    不會被某個 Arm 被選較少所混淆。決策差方差則另行輸出為
    Var(mu_B - mu_P) = Var(mu_B) + Var(mu_P)。
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

    action_space_variance = max(
        0.0,
        float(shared_context.get("variance", 0.0) or 0.0),
    )
    action_space_std = math.sqrt(action_space_variance)
    decision_gap_variance = variance_b + variance_p
    decision_gap_std = math.sqrt(decision_gap_variance)
    selected_arm = base_direction if base_direction in ARMS else "B"
    selected_variance = variance_b if selected_arm == "B" else variance_p
    selected_std = math.sqrt(selected_variance)

    baseline = _baseline_summary(state)
    threshold_variance = float(baseline["threshold"])
    variance_above_threshold = action_space_variance > threshold_variance
    shared_context_updates = int(
        shared_context.get("updates", 0) or 0
    )
    # 「高於歷史平均 + 1.3 sigma」只有在歷史基準與真實回饋都已
    # 累積完成後才有統計意義。舊版在新 UID 的第 1～2 局便套用冷啟動
    # 固定門檻，容易把正常的新使用者誤判成極端 OOD，並讓影子熔斷
    # 長時間鎖住觀望。
    ood_detection_ready = bool(
        baseline["dynamic_ready"]
        and shared_context_updates >= CMAB_UNCERTAINTY_MIN_SAMPLES
    )
    entropy_indicates_white_noise = bool(
        randomness.get("entropy_indicates_white_noise", False)
    )
    active = bool(
        ood_detection_ready
        and variance_above_threshold
        and entropy_indicates_white_noise
    )
    severity_ratio = (
        action_space_variance / max(threshold_variance, 1e-12)
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
        # 全局聯動的穩定欄位名稱；adaptive_ensemble.py 直接讀取。
        "is_extreme_unseen": bool(active),
        "variance": float(action_space_variance),
        "uncertainty_level": uncertainty_level,
        "threshold_mode": (
            "dynamic_variance_mean_plus_1_3_std"
            if baseline["dynamic_ready"]
            else "cold_start_variance_fallback"
        ),
        "threshold_metric": (
            "shared_context_variance_AND_permutation_entropy"
        ),
        "threshold_variance": float(threshold_variance),
        "dynamic_threshold_variance": float(threshold_variance),
        # 保留舊 std 欄位供既有下游讀取，但正式判斷使用 variance。
        "threshold_std": float(math.sqrt(max(0.0, threshold_variance))),
        "dynamic_threshold_std": float(
            math.sqrt(max(0.0, threshold_variance))
        ),
        "historical_mean_variance": float(baseline["mean"]),
        "historical_std_of_variance": float(baseline["std"]),
        "historical_mean_std": float(
            math.sqrt(max(0.0, float(baseline["mean"])))
        ),
        "historical_std_of_std": 0.0,
        "variance_above_dynamic_threshold": bool(
            variance_above_threshold
        ),
        "variance_safe": bool(not variance_above_threshold),
        "permutation_entropy": float(
            randomness.get("permutation_entropy", 0.0) or 0.0
        ),
        "permutation_entropy_threshold": float(
            CMAB_PERMUTATION_ENTROPY_THRESHOLD
        ),
        "entropy_indicates_white_noise": bool(
            entropy_indicates_white_noise
        ),
        "rolling_randomness": dict(randomness),
        "historical_sample_count": int(
            baseline["sample_count"]
        ),
        "ood_detection_ready": bool(ood_detection_ready),
        "ood_minimum_observations": int(
            CMAB_UNCERTAINTY_MIN_SAMPLES
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
        "selected_arm": selected_arm,
        "selected_arm_std": float(selected_std),
        "selected_arm_variance": float(selected_variance),
        "shared_context_updates": int(
            shared_context.get("updates", 0) or 0
        ),
        "shared_context_weighted_updates": float(
            shared_context.get(
                "weighted_updates",
                shared_context.get("updates", 0),
            )
            or 0.0
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
        "few_shot_update_weight": 1.0,
        "few_shot_boost_disabled": True,
        "observe_required": bool(active),
        "bet_multiplier": 0.0 if active else 1.0,
        "downstream_signal_code": (
            "STATISTICAL_CHAOS_HARD_BRAKE"
            if active
            else "IN_DISTRIBUTION"
        ),
    }

def _predict_bandit_impl(
    history: Iterable[Any],
    *,
    road_context: Optional[Mapping[str, Any]] = None,
    venue: str = "",
    room: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    raw_history = _clean_history(history)
    randomness = _rolling_randomness_diagnostics(raw_history)
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
        shared_context = _shared_context_uncertainty(state, context)

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

        conditional_bp = _softmax_two(
            score_b,
            score_p,
        )
        tie_probability = _smoothed_tie_probability(raw_history)
        bp_mass = 1.0 - tie_probability
        base_probabilities = {
            "B": float(conditional_bp["B"] * bp_mass),
            "P": float(conditional_bp["P"] * bp_mass),
            "T": float(tie_probability),
        }
        total_updates = int(state.get("total_updates", 0) or 0)
        conditional_edge = abs(
            float(conditional_bp["B"]) - float(conditional_bp["P"])
        )
        maturity_ready = total_updates >= CMAB_MIN_SIGNAL_UPDATES
        direction_signal_ready = bool(
            maturity_ready and conditional_edge >= CMAB_MIN_SIGNAL_EDGE
        )
        braking = _uncertainty_braking_metrics(
            metrics,
            state,
            shared_context,
            base_direction,
            randomness,
        )
        short_term = _short_term_trend_buffer(
            raw_history,
            base_direction,
        )
        short_term["meta_learning_takeover"] = False

        if bool(braking["active"]):
            # 混沌區間不再以 3 局規則覆寫主模型；只保留內部 Arm
            # 供 1 倍被動學習，公開分數強制中性並交由下游硬熔斷。
            direction = base_direction
            probabilities = {
                "B": float(bp_mass * 0.5),
                "P": float(bp_mass * 0.5),
                "T": float(tie_probability),
            }
            action_code = "O"
            action_text = "觀望／絕對不下注"
            direction_source = (
                "statistical_chaos_hard_brake_candidate"
            )
            signal_reason = (
                "統計混沌硬煞車："
                f"共享方差 {float(braking['action_space_variance']):.4f} > "
                f"方差門檻 {float(braking['threshold_variance']):.4f}，且"
                f"12局排列熵 {float(braking['permutation_entropy']):.4f} > "
                f"{CMAB_PERMUTATION_ENTROPY_THRESHOLD:.2f}；"
                "本局強制觀望／0%配置。"
            )
        elif direction_signal_ready:
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
        else:
            direction = base_direction
            probabilities = dict(base_probabilities)
            action_code = "O"
            action_text = "觀望／方向優勢不足"
            direction_source = "contextual_bandit_low_edge_brake"
            signal_reason = (
                "方向訊號尚未達安全門檻："
                f"有效回饋 {total_updates}/{CMAB_MIN_SIGNAL_UPDATES}，"
                f"B/P 條件優勢 {conditional_edge:.4f}/"
                f"{CMAB_MIN_SIGNAL_EDGE:.4f}；本局不下注。"
            )

        margin = abs(score_b - score_p)
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
        elif not direction_signal_ready:
            quality = min(quality, 0.49)

        direction_edge = float(conditional_edge)
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
            action_space_variance=float(
                braking["action_space_variance"]
            ),
            unknown_region_active=bool(
                braking["active"]
            ),
            prediction_fingerprint=prediction_fingerprint,
        )
        context_hash = _context_fingerprint(vector)
        state["pending_ood_contexts"] = []
        state["last_prediction_risk"] = {
            "prediction_fingerprint": prediction_fingerprint,
            "context_hash": context_hash,
            "unknown_region_active": bool(
                braking["active"]
            ),
            "is_extreme_unseen": bool(braking["active"]),
            "variance": float(braking["action_space_variance"]),
            "few_shot_update_weight": few_shot_weight,
            "action_space_std": float(
                braking["action_space_std"]
            ),
            "dynamic_threshold_std": float(
                braking["threshold_std"]
            ),
            "dynamic_threshold_variance": float(
                braking["threshold_variance"]
            ),
            "permutation_entropy": float(
                braking["permutation_entropy"]
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
    signal_allowed = action_code in ARMS
    output_signal_code = (
        str(braking["downstream_signal_code"])
        if braking["active"]
        else "LOW_EDGE_OBSERVE"
        if not signal_allowed
        else "IN_DISTRIBUTION"
    )
    risk_signal = {
        "code": output_signal_code,
        "ood_detected": bool(braking["active"]),
        "is_extreme_unseen": bool(braking["active"]),
        "variance": float(braking["action_space_variance"]),
        "extreme_uncertainty": bool(
            braking["active"]
        ),
        "observe_required": bool(not signal_allowed),
        "bet_multiplier": 1.0 if signal_allowed else 0.0,
        "confidence_cap": (
            float(CMAB_UNKNOWN_CONFIDENCE_CAP)
            if braking["active"]
            else 1.0
        ),
        "few_shot_update_weight": few_shot_weight,
        "few_shot_boost_disabled": True,
        "permutation_entropy": float(braking["permutation_entropy"]),
        "entropy_indicates_white_noise": bool(
            braking["entropy_indicates_white_noise"]
        ),
        "variance_above_dynamic_threshold": bool(
            braking["variance_above_dynamic_threshold"]
        ),
        "variance_safe": bool(braking["variance_safe"]),
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
        "prediction_fingerprint": prediction_fingerprint,
        "mode": "screen_contextual_bandit",
        # 將真正參與本局特徵計算的牌路專家同步交給集成層。舊版直到
        # predictor 完成後才在 screenshot_predictor 補回 road_support，
        # 導致 adaptive_ensemble 在決策當下看不到任何牌路成員。
        "road_support": dict(road),
        "component_probabilities": dict(
            road.get("component_probabilities") or {}
        ),
        "probabilities": dict(probabilities),
        "pre_braking_probabilities": dict(
            base_probabilities
        ),
        "bandit_learning_probabilities": dict(base_probabilities),
        "timeline_alignment": {
            "raw_round_index": len(raw_history),
            "bp_round_index": sum(
                value in ARMS for value in raw_history
            ),
            "tie_count": sum(
                value == "T" for value in raw_history
            ),
            "structural_features_skip_ties": True,
            "prediction_uses_history_before_target": True,
        },
        "banker_rate": round(
            probabilities["B"] * 100.0,
            2,
        ),
        "player_rate": round(
            probabilities["P"] * 100.0,
            2,
        ),
        "tie_rate": round(
            probabilities["T"] * 100.0,
            2,
        ),
        # selected_arm 仍供固定 1 倍被動學習；真正資金動作由 action=O 控制。
        "recommend": direction,
        "recommend_text": (
            "莊" if direction == "B" else "閒"
        ),
        "action": action_code,
        "action_text": action_text,
        "internal_recommend": direction,
        "internal_action": action_code,
        "signal_allowed": signal_allowed,
        "signal_status_code": output_signal_code,
        "signal_status_text": (
            "統計混沌硬煞車：觀望／絕對不下注"
            if braking["active"]
            else "方向優勢或回饋樣本不足：觀望"
            if not signal_allowed
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
        # 頂層 uncertainty 是 OOD braking 真正使用的共享 context std。
        "uncertainty": float(braking["action_space_std"]),
        "state_novelty_uncertainty": float(
            braking["action_space_std"]
        ),
        "state_novelty_variance": float(
            braking["action_space_variance"]
        ),
        "selected_arm_uncertainty": float(
            selected["uncertainty"]
        ),
        "prediction_variance": float(
            selected["variance"]
        ),
        # 供全局集成層使用的固定契約：variance 是共享 context
        # x^T A_ctx^-1 x，而非某一個 Arm 的局部方差。
        "variance": float(braking["action_space_variance"]),
        "variance_threshold": float(braking["threshold_variance"]),
        "variance_safe": bool(braking["variance_safe"]),
        "permutation_entropy": float(braking["permutation_entropy"]),
        "permutation_entropy_threshold": float(
            braking["permutation_entropy_threshold"]
        ),
        "statistical_tests": dict(braking["rolling_randomness"]),
        "decision_gap_uncertainty": float(
            braking["decision_gap_std"]
        ),
        "decision_gap_variance": float(
            braking["decision_gap_variance"]
        ),
        "unknown_region_active": bool(
            braking["active"]
        ),
        "is_extreme_unseen": bool(braking["active"]),
        "extreme_uncertainty_signal": bool(
            braking["active"]
        ),
        "few_shot_update_weight": few_shot_weight,
        "bet_multiplier": 1.0 if signal_allowed else 0.0,
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
            "active": False,
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
            "reason": "V1.5 已停用即時短線接管；只允許 predictor 影子回測",
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
            "dynamic_threshold_variance": float(
                braking["threshold_variance"]
            ),
            "dynamic_threshold_ready": bool(
                braking["dynamic_threshold_ready"]
            ),
            "ood_detection_ready": bool(
                braking["ood_detection_ready"]
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
            "few_shot_boost_disabled": True,
            "forgetting_factor": CMAB_FORGETTING_FACTOR,
            "reversal_forgetting_factor": (
                CMAB_REVERSAL_FORGETTING_FACTOR
            ),
            "reward_strategy": (
                "probability_weighted_asymmetric_payoff"
            ),
            "permutation_entropy_window": CMAB_PERMUTATION_WINDOW,
            "permutation_entropy_threshold": (
                CMAB_PERMUTATION_ENTROPY_THRESHOLD
            ),
            "tie_prior": CMAB_TIE_PRIOR,
            "tie_prior_strength": CMAB_TIE_PRIOR_STRENGTH,
            "minimum_signal_edge": CMAB_MIN_SIGNAL_EDGE,
            "minimum_signal_updates": CMAB_MIN_SIGNAL_UPDATES,
            "signal_maturity_ready": bool(maturity_ready),
            "direction_signal_ready": bool(direction_signal_ready),
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

def _coerce_context_vector(context: Sequence[float]) -> np.ndarray:
    """接受目前維度與部署切換期間尚未結算的舊 24 維 prediction。"""
    values = np.asarray(list(context), dtype=np.float64)
    if values.ndim != 1 or values.size <= 0 or values.size > CONTEXT_DIM:
        raise ValueError(
            f"context must contain 1..{CONTEXT_DIM} values, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("context contains non-finite values")
    if values.size < CONTEXT_DIM:
        migrated = np.zeros(CONTEXT_DIM, dtype=np.float64)
        migrated[:values.size] = values
        values = migrated
    return np.clip(values, -1.0, 1.0)


def _conditional_bp_probabilities(values: Any) -> Dict[str, float]:
    if not isinstance(values, Mapping):
        return {"B": 0.5, "P": 0.5}
    banker = max(0.0, float(values.get("B", 0.0) or 0.0))
    player = max(0.0, float(values.get("P", 0.0) or 0.0))
    total = banker + player
    if total <= 1e-12:
        return {"B": 0.5, "P": 0.5}
    return {"B": banker / total, "P": player / total}


def _probability_weighted_arm_rewards(
    actual: str,
    prediction_probabilities: Any,
) -> Dict[str, float]:
    """方向 payoff 的信心加權版，範圍 [-2, 1]。"""
    conditional = _conditional_bp_probabilities(prediction_probabilities)
    rewards: Dict[str, float] = {}
    for candidate in ARMS:
        confidence = float(conditional[candidate])
        if candidate == actual:
            value = 0.5 + 0.5 * confidence
        else:
            excess = max(0.0, (confidence - 0.5) / 0.5)
            value = -(0.25 + 1.75 * excess * excess)
        rewards[candidate] = float(max(-2.0, min(1.0, value)))
    return rewards


def _update_bandit_impl(
    *,
    context: Sequence[float],
    selected_arm: str,
    reward: Optional[float],
    event_id: str = "",
    actual_outcome: str = "",
    update_weight: float = 1.0,
    user_id: str = "",
    prediction_probabilities: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    arm = str(selected_arm or "").upper().strip()
    actual = str(actual_outcome or "").upper().strip()
    if arm not in ARMS:
        raise ValueError("selected_arm must be B or P")

    if reward is None:
        return {
            "updated": False,
            "reason": "tie_or_skipped_reward",
            "selected_arm": arm,
            "actual_outcome": actual,
        }

    x = _coerce_context_vector(context)
    context_hash = _context_fingerprint(x)
    if not math.isfinite(float(reward)):
        raise ValueError("reward must be finite")
    reward_value = max(-2.0, min(1.0, float(reward)))
    requested_weight_value = float(update_weight)
    if not math.isfinite(requested_weight_value):
        raise ValueError("update_weight must be finite")
    requested_weight = max(0.25, min(12.0, requested_weight_value))
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

        last_risk = dict(state.get("last_prediction_risk") or {})
        same_high_variance_context = bool(
            last_risk.get(
                "is_extreme_unseen",
                last_risk.get("unknown_region_active", False),
            )
            and str(last_risk.get("context_hash") or "")
            == context_hash
        )
        # V1.5：無論呼叫端傳入多少權重，一律只視為一筆觀測。
        # 這能防止單局隨機結果以 4～5 筆證據灌入 posterior。
        observation_weight = 1.0

        # 百家樂揭曉 B/P 後，兩個 Arm 的反事實 reward 都已知：
        # B 開出即 B=1/P=0，P 開出則相反。舊版只更新「被選 Arm」，
        # 會引入 action-selection bias 並浪費一半監督訊號。
        # 有完整結果時同時更新兩 Arm；舊呼叫未提供 actual 時仍相容
        # selected-arm-only 更新。每個 Arm 仍固定 w=1，沒有 Few-shot。
        full_information_update = actual in ARMS
        arm_rewards = (
            _probability_weighted_arm_rewards(
                actual,
                prediction_probabilities,
            )
            if full_information_update
            else {arm: reward_value}
        )
        feature_index = {
            name: index for index, name in enumerate(FEATURE_NAMES)
        }
        reversal_signal = abs(
            float(x[feature_index["streak_break_signal"]])
        )
        forgetting_factor = (
            CMAB_REVERSAL_FORGETTING_FACTOR
            if reversal_signal >= 0.5
            else CMAB_FORGETTING_FACTOR
        )
        updated_arm_states: Dict[str, Dict[str, Any]] = {}
        outer_context = np.outer(x, x)
        for candidate, candidate_reward in arm_rewards.items():
            candidate_state = dict(state["arms"][candidate])
            previous_updates = int(candidate_state.get("updates", 0) or 0)
            previous_weighted_updates = float(
                candidate_state.get("weighted_updates", previous_updates)
                or 0.0
            )
            previous_reward_sum = float(
                candidate_state.get("reward_sum", 0.0) or 0.0
            )
            previous_weighted_reward_sum = float(
                candidate_state.get(
                    "weighted_reward_sum", previous_reward_sum
                )
                or 0.0
            )
            candidate_A = np.asarray(
                candidate_state["A"], dtype=np.float64
            )
            candidate_b = np.asarray(
                candidate_state["b"], dtype=np.float64
            )
            ridge = np.eye(CONTEXT_DIM, dtype=np.float64) * float(
                state.get("l2", CMAB_L2) or CMAB_L2
            )
            candidate_A = (
                forgetting_factor * candidate_A
                + (1.0 - forgetting_factor) * ridge
            )
            candidate_b *= forgetting_factor
            candidate_A += observation_weight * outer_context
            candidate_b += (
                observation_weight * float(candidate_reward) * x
            )
            candidate_state["A"] = candidate_A.tolist()
            candidate_state["b"] = candidate_b.tolist()
            candidate_state["updates"] = previous_updates + 1
            candidate_state["weighted_updates"] = (
                forgetting_factor * previous_weighted_updates
                + observation_weight
            )
            candidate_state["reward_sum"] = (
                previous_reward_sum + float(candidate_reward)
            )
            candidate_state["weighted_reward_sum"] = (
                forgetting_factor * previous_weighted_reward_sum
                + observation_weight * float(candidate_reward)
            )
            state["arms"][candidate] = candidate_state
            updated_arm_states[candidate] = candidate_state

        arm_state = updated_arm_states[arm]

        # 共享 context matrix 不看 reward，也不分 Arm；每筆已揭曉 B/P
        # 都代表這個特徵區間多了一次真實觀測。
        context_information = dict(state.get("context_information") or {})
        context_A = _as_information_matrix(context_information.get("A"))
        context_ridge = np.eye(CONTEXT_DIM, dtype=np.float64) * float(
            state.get("l2", CMAB_L2) or CMAB_L2
        )
        context_A = (
            forgetting_factor * context_A
            + (1.0 - forgetting_factor) * context_ridge
        )
        context_A += observation_weight * np.outer(x, x)
        context_information["A"] = (
            0.5 * (context_A + context_A.T)
        ).tolist()
        context_information["updates"] = (
            int(context_information.get("updates", 0) or 0) + 1
        )
        context_information["weighted_updates"] = forgetting_factor * float(
            context_information.get(
                "weighted_updates",
                context_information["updates"] - 1,
            )
            or 0.0
        ) + observation_weight
        state["context_information"] = context_information
        state["total_updates"] = (
            int(state.get("total_updates", 0) or 0)
            + 1
        )
        state["total_weighted_updates"] = forgetting_factor * float(
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

        state["pending_ood_contexts"] = []

        state["last_update"] = {
            "event_id": event_key,
            "selected_arm": arm,
            "actual_outcome": actual,
            "reward": reward_value,
            "full_information_update": bool(full_information_update),
            "updated_arms": list(updated_arm_states),
            "arm_rewards": dict(arm_rewards),
            "prediction_probabilities": dict(
                _conditional_bp_probabilities(prediction_probabilities)
            ),
            "reward_strategy": "probability_weighted_asymmetric_payoff",
            "forgetting_factor": float(forgetting_factor),
            "reversal_signal": float(reversal_signal),
            "reversal_decay_applied": bool(
                forgetting_factor == CMAB_REVERSAL_FORGETTING_FACTOR
            ),
            "requested_weight": requested_weight,
            "applied_weight": observation_weight,
            "few_shot_boost_applied": False,
            "few_shot_boost_disabled": True,
            "same_high_variance_context": bool(
                same_high_variance_context
            ),
            "matched_pending_ood_context": False,
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
        "actual_outcome": actual,
        "reward": reward_value,
        "full_information_update": bool(full_information_update),
        "updated_arms": list(updated_arm_states),
        "arm_rewards": dict(arm_rewards),
        "prediction_probabilities": dict(
            _conditional_bp_probabilities(prediction_probabilities)
        ),
        "reward_strategy": "probability_weighted_asymmetric_payoff",
        "forgetting_factor": float(forgetting_factor),
        "reversal_signal": float(reversal_signal),
        "reversal_decay_applied": bool(
            forgetting_factor == CMAB_REVERSAL_FORGETTING_FACTOR
        ),
        "requested_update_weight": requested_weight,
        "update_weight": observation_weight,
        "few_shot_boost_applied": False,
        "few_shot_boost_disabled": True,
        "reversal_forgetting_factor": float(
            CMAB_REVERSAL_FORGETTING_FACTOR
        ),
        "boost_reason": "disabled_to_prevent_random_noise_overfitting",
        "arm_updates": int(arm_state["updates"]),
        "arm_weighted_updates": float(
            arm_state["weighted_updates"]
        ),
        "per_arm_updates": {
            candidate: int(updated_arm_states[candidate]["updates"])
            for candidate in updated_arm_states
        },
        "total_updates": int(
            state["total_updates"]
        ),
        "total_weighted_updates": float(
            state["total_weighted_updates"]
        ),
        "shared_context_updates": int(
            context_information["updates"]
        ),
        "shared_context_weighted_updates": float(
            context_information["weighted_updates"]
        ),
    }


def _get_bandit_summary_impl(
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
            "unit": "variance",
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
        "few_shot_boost_disabled": True,
        "forgetting_factor": float(CMAB_FORGETTING_FACTOR),
        "reversal_forgetting_factor": float(
            CMAB_REVERSAL_FORGETTING_FACTOR
        ),
        "reward_strategy": "probability_weighted_asymmetric_payoff",
        "permutation_entropy": {
            "window": CMAB_PERMUTATION_WINDOW,
            "order": CMAB_PERMUTATION_ORDER,
            "threshold": CMAB_PERMUTATION_ENTROPY_THRESHOLD,
            "extreme_gate": (
                "variance_above_mean_plus_1.3_std_AND_entropy_above_0.95"
            ),
        },
        "short_term_buffer": list(
            state.get("short_term_buffer") or []
        )[-3:],
        "last_prediction_risk": dict(
            state.get("last_prediction_risk") or {}
        ),
        "shared_context_information": {
            "updates": int(
                dict(state.get("context_information") or {}).get(
                    "updates", 0
                )
                or 0
            ),
            "weighted_updates": float(
                dict(state.get("context_information") or {}).get(
                    "weighted_updates",
                    dict(state.get("context_information") or {}).get(
                        "updates", 0
                    ),
                )
                or 0.0
            ),
        },
        "pending_ood_context_count": len(
            list(state.get("pending_ood_contexts") or [])
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


class ContextualBanditEngine:
    """BGS LinUCB 核心類別；公開函式由此類別提供相容包裝。"""

    def predict(
        self,
        history: Iterable[Any],
        *,
        road_context: Optional[Mapping[str, Any]] = None,
        venue: str = "",
        room: str = "",
        user_id: str = "",
        run_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        return _predict_bandit_impl(
            history,
            road_context=road_context,
            venue=venue,
            room=room,
            user_id=user_id,
            run_seed=run_seed,
        )

    def update(
        self,
        *,
        context: Sequence[float],
        selected_arm: str,
        reward: Optional[float],
        event_id: str = "",
        actual_outcome: str = "",
        update_weight: float = 1.0,
        user_id: str = "",
        prediction_probabilities: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return _update_bandit_impl(
            context=context,
            selected_arm=selected_arm,
            reward=reward,
            event_id=event_id,
            actual_outcome=actual_outcome,
            update_weight=update_weight,
            user_id=user_id,
            prediction_probabilities=prediction_probabilities,
        )

    def summary(self, user_id: str = "") -> Dict[str, Any]:
        return _get_bandit_summary_impl(user_id=user_id)


_DEFAULT_ENGINE = ContextualBanditEngine()


def predict_bandit(
    history: Iterable[Any],
    *,
    road_context: Optional[Mapping[str, Any]] = None,
    venue: str = "",
    room: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """既有 predictor.py 相容入口。"""
    return _DEFAULT_ENGINE.predict(
        history,
        road_context=road_context,
        venue=venue,
        room=room,
        user_id=user_id,
        run_seed=run_seed,
    )


def update_bandit(
    *,
    context: Sequence[float],
    selected_arm: str,
    reward: Optional[float],
    event_id: str = "",
    actual_outcome: str = "",
    update_weight: float = 1.0,
    user_id: str = "",
    prediction_probabilities: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """既有 performance_tracker.py 相容入口。"""
    return _DEFAULT_ENGINE.update(
        context=context,
        selected_arm=selected_arm,
        reward=reward,
        event_id=event_id,
        actual_outcome=actual_outcome,
        update_weight=update_weight,
        user_id=user_id,
        prediction_probabilities=prediction_probabilities,
    )


def get_bandit_summary(user_id: str = "") -> Dict[str, Any]:
    return _DEFAULT_ENGINE.summary(user_id=user_id)


__all__ = [
    "ARMS",
    "CONTEXT_DIM",
    "ContextualBanditEngine",
    "FEATURE_NAMES",
    "MODEL_VERSION",
    "build_context_vector",
    "get_bandit_summary",
    "predict_bandit",
    "update_bandit",
]
