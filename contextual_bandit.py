"""Frozen-direct 32D Single-Brain Contextual LinUCB core.

Production behavior mirrors the BBB standalone web panel's
"reuse local 32D brain and predict directly" mode:

1. A new scope starts with the untouched ridge matrices (A=I, b=0).
2. Starting analysis never replays B/P/T history and never bootstraps A/b.
3. Newly appended B/P/T results only change the current 32D context. They do
   NOT resolve the previous prediction, decay the matrices, or update A/b.
4. The persisted A/b matrices remain frozen unless the explicit update_bandit
   compatibility API is called by a separate caller.

Formal Banker/Player direction remains owned only by two-arm LinUCB UCB score
comparison. OCR, screenshot parsing, API fields and money management are outside
this module and are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import os
import time

import numpy as np

from road_model import build_standard_derived_roads
from shoe_constants import AVERAGE_CARDS_PER_HAND, SHOE_DECKS

ARMS = ("P", "B")
CONTEXT_DIM = 32
CONTEXT_FEATURE_NAMES = (
    # 1-16: shoe state
    "remaining_cards_ratio",
    "penetration_ratio",
    "estimated_hands_remaining_norm",
    "shoe_maturity_ratio",
    "rank_A_relative_ratio",
    "rank_2_relative_ratio",
    "rank_3_relative_ratio",
    "rank_4_relative_ratio",
    "rank_5_relative_ratio",
    "rank_6_relative_ratio",
    "rank_7_relative_ratio",
    "rank_8_relative_ratio",
    "rank_9_relative_ratio",
    "rank_10JQK_relative_ratio",
    "physical_edge_proxy",
    "shoe_information_reliability",
    # 17-32: road state
    "current_side_banker_binary",
    "current_run_length_norm",
    "previous_run_length_norm",
    "previous2_run_length_norm",
    "recent5_banker_ratio",
    "recent8_banker_ratio",
    "recent12_banker_ratio",
    "recent5_turn_rate",
    "recent8_turn_rate",
    "recent12_turn_rate",
    "run_length_hazard_rate",
    "hsmm_stable_probability",
    "big_eye_regularity",
    "small_road_regularity",
    "cockroach_road_regularity",
    "derived_road_consensus",
)

LINUCB_ALPHA = max(0.0, float(os.getenv("LINUCB_ALPHA", "0.5") or "0.5"))
LINUCB_RIDGE = max(1e-6, float(os.getenv("LINUCB_RIDGE", "1.0") or "1.0"))
LINUCB_UPDATE_WEIGHT = max(1e-3, float(os.getenv("LINUCB_UPDATE_WEIGHT", "1.0") or "1.0"))
LINUCB_FORGETTING = max(0.70, min(1.0, float(os.getenv("LINUCB_FORGETTING", "0.90") or "0.90")))
LINUCB_ARM_ALPHA_MAX_SCALE = max(1.0, min(2.5, float(os.getenv("LINUCB_ARM_ALPHA_MAX_SCALE", "1.60") or "1.60")))
LINUCB_SCORE_TIE_EPSILON = max(1e-12, float(os.getenv("LINUCB_SCORE_TIE_EPSILON", "0.000001") or "0.000001"))
LINUCB_SCORE_TEMPERATURE = max(0.25, min(10.0, float(os.getenv("LINUCB_SCORE_TEMPERATURE", "2.0") or "2.0")))
ROAD_PRIOR_SCORE_WEIGHT = 0.0
ROAD_PRIOR_PROBABILITY_SPAN = 0.0
LINUCB_PROBABILITY_CORRECTION_SPAN = 0.0
PROBABILITY_MIN = 0.42
PROBABILITY_MAX = 0.58
STATE_VERSION = "LINUCB-2ARM-SINGLE-BRAIN-CONTEXT-16SHOE-16ROAD-32D-WEB-PARITY-V10"
ESTIMATED_CARDS_PER_ROUND = AVERAGE_CARDS_PER_HAND
_LOCK = RLock()


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _normalize_history(history: Iterable[Any] | str | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = history.replace("|", "").replace(",", "").replace(" ", "").upper()
        if compact and all(char in {"B", "P", "T"} for char in compact):
            return list(compact)[-2000:]
        items: Iterable[Any] = [part for part in history.replace("|", ",").split(",") if part.strip()]
    else:
        items = deepcopy(list(history))
    output: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            output.append(value)
    return output[-2000:]


def _bp(sequence: Sequence[str]) -> list[str]:
    return [value for value in sequence if value in {"B", "P"}]


def _runs(sequence: Sequence[str]) -> list[tuple[str, int]]:
    values = _bp(sequence)
    if not values:
        return []
    result: list[tuple[str, int]] = []
    side, length = values[0], 1
    for value in values[1:]:
        if value == side:
            length += 1
        else:
            result.append((side, length))
            side, length = value, 1
    result.append((side, length))
    return result


def _recent_banker_ratio(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-max(1, int(window)):]
    if not values:
        return 0.5
    return float(sum(value == "B" for value in values) / len(values))


def _turn_rate(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-max(2, int(window)):]
    if len(values) < 2:
        return 0.5
    turns = sum(values[index] != values[index - 1] for index in range(1, len(values)))
    return float(turns / (len(values) - 1))


def _road_regularity(values: Iterable[Any], window: int = 8) -> tuple[float, int]:
    marks = [str(value).upper().strip() for value in list(values)[-max(1, int(window)):]]
    marks = [mark for mark in marks if mark in {"R", "U"}]
    if not marks:
        return 0.5, 0
    return float(sum(mark == "R" for mark in marks) / len(marks)), len(marks)


# Panel hazard proxy ---------------------------------------------------------
def _length_bucket(length: int) -> str:
    value = max(1, int(length))
    return str(value) if value <= 5 else "6+"


def _hazard_contexts(side: str, current: int, previous_heights: Sequence[int]) -> list[tuple[str, str]]:
    previous_height = previous_heights[-1] if previous_heights else 0
    deltas = [
        "UP" if previous_heights[index] > previous_heights[index - 1]
        else "DOWN" if previous_heights[index] < previous_heights[index - 1]
        else "EQUAL"
        for index in range(1, len(previous_heights))
    ]
    delta1 = deltas[-1] if deltas else "NA"
    delta2 = deltas[-2] if len(deltas) >= 2 else "NA"
    current_bucket = _length_bucket(current)
    previous_bucket = _length_bucket(previous_height) if previous_height else "0"
    return [
        ("full", f"HZF|side={side or 'NA'}|cur={current_bucket}|prev={previous_bucket}|d1={delta1}|d2={delta2}"),
        ("structure", f"HZS|cur={current_bucket}|prev={previous_bucket}|d1={delta1}|d2={delta2}"),
        ("shape", f"HZP|cur={current_bucket}|prev={previous_bucket}|d1={delta1}"),
        ("length", f"HZL|cur={current_bucket}"),
        ("global", "HZG|GLOBAL"),
    ]


def _build_hazard_table(run_values: Sequence[tuple[str, int]]) -> dict[str, dict[str, float]]:
    completed = list(run_values[:-1])
    heights = [item[1] for item in completed]
    table: dict[str, dict[str, float]] = {}
    for index, (side, final_length) in enumerate(completed):
        previous = heights[:index]
        for at_length in range(1, max(1, final_length) + 1):
            event = "CONTINUE" if at_length < final_length else "TURN"
            for _, key in _hazard_contexts(side, at_length, previous):
                table.setdefault(key, {"CONTINUE": 0.0, "TURN": 0.0})[event] += 1.0
    return table


def _hazard_posterior(counts: Mapping[str, Any]) -> dict[str, float]:
    continued = float(counts.get("CONTINUE", 0.0) or 0.0)
    turned = float(counts.get("TURN", 0.0) or 0.0)
    denominator = continued + turned + 6.0
    if denominator <= 1e-12:
        return {"CONTINUE": 0.5, "TURN": 0.5}
    return {"CONTINUE": (continued + 3.0) / denominator, "TURN": (turned + 3.0) / denominator}


def _hazard(sequence: Sequence[str]) -> float:
    run_values = _runs(sequence)
    if not run_values:
        return 0.5
    side, current_length = run_values[-1]
    previous_heights = [item[1] for item in run_values[:-1]]
    table = _build_hazard_table(run_values)
    selected_tier = "prior"
    probabilities = {"CONTINUE": 0.5, "TURN": 0.5}
    penalty = 1.0
    contexts = _hazard_contexts(side, current_length, previous_heights)
    for index, (tier, key) in enumerate(contexts):
        counts = table.get(key, {"CONTINUE": 0.0, "TURN": 0.0})
        support = counts["CONTINUE"] + counts["TURN"]
        posterior = _hazard_posterior(counts)
        if support >= 4:
            selected_tier = tier
            probabilities = posterior
            break
        if index < len(contexts) - 1:
            penalty *= 0.75
    if selected_tier == "prior":
        global_counts = table.get("HZG|GLOBAL", {"CONTINUE": 0.0, "TURN": 0.0})
        if global_counts["CONTINUE"] + global_counts["TURN"] > 0:
            probabilities = _hazard_posterior(global_counts)
        else:
            penalty = 0.0
    continue_probability = (1.0 - penalty) * 0.5 + penalty * probabilities["CONTINUE"]
    return _clip(1.0 - continue_probability)


# Panel HSMM-stability proxy -------------------------------------------------
def _entropy(sequence: Sequence[str], window: int = 12) -> float:
    values = list(sequence[-window:])
    if not values:
        return 1.0
    entropy = 0.0
    for outcome in ("B", "P", "T"):
        probability = sum(value == outcome for value in values) / len(values)
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return _clip(entropy / math.log2(3))


def _run_volatility(sequence: Sequence[str]) -> float:
    heights = [item[1] for item in _runs(sequence)[-6:]]
    if len(heights) < 2:
        return 0.25
    mean_delta = sum(abs(heights[index] - heights[index - 1]) for index in range(1, len(heights))) / (len(heights) - 1)
    return _clip(mean_delta / 3.0)


def _hsmm_stable_proxy(sequence: Sequence[str]) -> float:
    alternation = _turn_rate(sequence, 10)
    run_values = _runs(sequence)
    current_length = run_values[-1][1] if run_values else 0
    current_normalized = _clip(current_length / 6.0)
    entropy_normalized = _entropy(sequence)
    volatility = _run_volatility(sequence)

    persistent = math.exp(
        -((alternation - 0.25) / 0.24) ** 2
        -((current_normalized - 0.70) / 0.28) ** 2
        -((entropy_normalized - 0.62) / 0.24) ** 2
        -((volatility - 0.26) / 0.24) ** 2
    )
    alternating = math.exp(
        -((alternation - 0.84) / 0.18) ** 2
        -((current_normalized - 0.18) / 0.20) ** 2
        -((entropy_normalized - 0.70) / 0.23) ** 2
        -((volatility - 0.30) / 0.24) ** 2
    )
    transition = math.exp(
        -((alternation - 0.52) / 0.28) ** 2
        -((current_normalized - 0.34) / 0.26) ** 2
        -((entropy_normalized - 0.82) / 0.18) ** 2
        -((volatility - 0.72) / 0.23) ** 2
    )
    noise = math.exp(
        -((alternation - 0.55) / 0.30) ** 2
        -((current_normalized - 0.27) / 0.24) ** 2
        -((entropy_normalized - 0.94) / 0.11) ** 2
        -((volatility - 0.55) / 0.28) ** 2
    )
    weighted = (0.25 * persistent, 0.25 * alternating, 0.20 * transition, 0.30 * noise)
    total = sum(weighted) or 1.0
    return _clip((weighted[0] + weighted[1]) / total)


def _model_x(vector: Sequence[float]) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(vector, dtype=np.float64).reshape(CONTEXT_DIM),
        nan=0.0,
        posinf=2.0,
        neginf=-1.0,
    )


@dataclass(frozen=True)
class ContextSnapshot:
    vector: np.ndarray
    metadata: dict[str, Any]


class ContextGenerator:
    def build(self, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None = None) -> ContextSnapshot:
        raw = _normalize_history(deepcopy(history))
        bp_values = _bp(raw)
        context = deepcopy(dict(shoe_context or {}))
        try:
            decks = int(context.get("decks", SHOE_DECKS) or SHOE_DECKS)
        except (TypeError, ValueError):
            decks = SHOE_DECKS
        decks = max(1, min(16, decks))
        total_cards = float(52 * decks)

        # BBB is a button-only panel and has no exact card-rank input. Keep the
        # production predictor on the same feature contract: shoe depth comes
        # only from the B/P/T round count, while rank ratios stay neutral.
        # The caller may still supply verified card data for outward API and
        # diagnostics, but it must not change this web-parity context vector.
        exact_available = False
        remaining_cards = max(0.0, total_cards - len(raw) * float(AVERAGE_CARDS_PER_HAND))
        remaining_source = "panel_history_round_estimate"
        remaining_cards = max(0.0, min(total_cards, remaining_cards))
        remaining_ratio = _clip(remaining_cards / total_cards if total_cards else 1.0)
        penetration_ratio = _clip(1.0 - remaining_ratio)
        shoe_maturity_ratio = _clip(len(raw) / 70.0)

        rank_ratios = [1.0] * 10
        group_ratios = [1.0] * 7
        rank_ratio_source = "neutral_fallback_web_panel"
        physical_edge_proxy = 0.0
        shoe_information_reliability = 0.0

        run_values = _runs(raw)
        current_side, current_run = run_values[-1] if run_values else ("", 0)
        previous_run = run_values[-2][1] if len(run_values) >= 2 else 0
        previous2_run = run_values[-3][1] if len(run_values) >= 3 else 0
        current_side_banker_binary = 1.0 if current_side == "B" else 0.0 if current_side == "P" else 0.5
        hazard_rate = _hazard(raw)
        hsmm_stable = _hsmm_stable_proxy(raw)

        derived = build_standard_derived_roads(deepcopy(bp_values))
        big_eye = list(derived.get("big_eye") or [])
        small_road = list(derived.get("small_road") or [])
        cockroach = list(derived.get("cockroach_road") or [])
        big_eye_regularity, big_eye_samples = _road_regularity(big_eye)
        small_regularity, small_samples = _road_regularity(small_road)
        cockroach_regularity, cockroach_samples = _road_regularity(cockroach)
        regularity_mean = (big_eye_regularity + small_regularity + cockroach_regularity) / 3.0
        derived_consensus = _clip(
            1.0 - (
                abs(big_eye_regularity - regularity_mean)
                + abs(small_regularity - regularity_mean)
                + abs(cockroach_regularity - regularity_mean)
            ) / 1.5
        )
        combined_samples = small_samples + cockroach_samples
        small_cockroach_regularity = (
            (small_regularity * small_samples + cockroach_regularity * cockroach_samples) / combined_samples
            if combined_samples else 0.5
        )
        latest_marks = [
            str(road[-1]).upper()
            for road in (big_eye, small_road, cockroach)
            if road and str(road[-1]).upper() in {"R", "U"}
        ]
        derived_binary = 1.0 if latest_marks and sum(mark == "R" for mark in latest_marks) * 2 >= len(latest_marks) else 0.0

        shoe_features = [
            remaining_ratio,
            penetration_ratio,
            remaining_ratio,
            shoe_maturity_ratio,
            *rank_ratios,
            physical_edge_proxy,
            shoe_information_reliability,
        ]
        road_features = [
            current_side_banker_binary,
            _clip(current_run / 8.0),
            _clip(previous_run / 8.0),
            _clip(previous2_run / 8.0),
            _recent_banker_ratio(raw, 5),
            _recent_banker_ratio(raw, 8),
            _recent_banker_ratio(raw, 12),
            _turn_rate(raw, 5),
            _turn_rate(raw, 8),
            _turn_rate(raw, 12),
            hazard_rate,
            hsmm_stable,
            big_eye_regularity,
            small_regularity,
            cockroach_regularity,
            derived_consensus,
        ]
        vector = np.nan_to_num(
            np.asarray([*shoe_features, *road_features], dtype=np.float64),
            nan=0.0,
            posinf=2.0,
            neginf=-1.0,
        )
        if vector.shape != (CONTEXT_DIM,):
            raise RuntimeError(f"context dimension mismatch: {vector.shape}")

        metadata = {
            "raw_round_count": len(raw),
            "bp_round_count": len(bp_values),
            "tie_count": sum(value == "T" for value in raw),
            "remaining_cards": remaining_cards,
            "remaining_ratio": remaining_ratio,
            "penetration_ratio": penetration_ratio,
            "estimated_hands_remaining_norm": remaining_ratio,
            "shoe_maturity_ratio": shoe_maturity_ratio,
            "remaining_cards_source": remaining_source,
            "soft_remaining_cards_ignored_for_panel_compatibility": bool(context.get("remaining_cards")),
            "exact_card_input_ignored_for_web_panel_compatibility": bool(
                "remaining_counts" in context or "observed_cards" in context
            ),
            "exact_composition_available": exact_available,
            "rank_ratio_source": rank_ratio_source,
            "rank_ratios_a_to_10jqk": [float(value) for value in rank_ratios],
            "shoe_group_ratios": {
                "A23": group_ratios[0], "45": group_ratios[1], "6": group_ratios[2],
                "7": group_ratios[3], "8": group_ratios[4], "9": group_ratios[5], "10JQK": group_ratios[6],
            },
            "physical_edge_proxy": physical_edge_proxy,
            "shoe_information_reliability": shoe_information_reliability,
            "combinatorial_advantage_offset": 0.0,
            "probabilistic_shoe_reliability": 0.0,
            "hsmm_stable_probability": hsmm_stable,
            "hazard_rate": hazard_rate,
            "hazard_formula": "panel_proxy",
            "hsmm_formula": "panel_proxy",
            "derived_road_regularity_binary": derived_binary,
            "derived_latest_marks": latest_marks,
            "run_length": current_run,
            "run_length_norm": _clip(current_run / 8.0),
            "shoe_decks": decks,
            "previous_run_length": previous_run,
            "previous2_run_length": previous2_run,
            "current_side": current_side,
            "current_side_banker_binary": current_side_banker_binary,
            "recent5_banker_ratio": _recent_banker_ratio(raw, 5),
            "recent8_banker_ratio": _recent_banker_ratio(raw, 8),
            "recent12_banker_ratio": _recent_banker_ratio(raw, 12),
            "recent5_turn_rate": _turn_rate(raw, 5),
            "recent8_turn_rate": _turn_rate(raw, 8),
            "recent12_turn_rate": _turn_rate(raw, 12),
            "big_eye_regularity": big_eye_regularity,
            "small_road_regularity": small_regularity,
            "cockroach_road_regularity": cockroach_regularity,
            "small_cockroach_regularity": small_cockroach_regularity,
            "derived_road_consensus": derived_consensus,
            "context_layout": "16_shoe_plus_16_road_32d",
            "context_compatibility": "bbb_standalone_32d_panel_frozen_direct",
            "shoe_feature_values": [float(value) for value in shoe_features],
            "road_feature_values": [float(value) for value in road_features],
            "formal_direction_source": "contextual_linucb",
            "single_brain": True,
            "external_direction_votes_enabled": False,
            "anti_echo_external_penalty": False,
        }
        return ContextSnapshot(vector=vector, metadata=metadata)


def _state_path() -> Path:
    candidates: list[Path] = []
    configured = str(os.getenv("LINUCB_STATE_FILE", "") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        Path("/var/data/contextual_linucb_state.json"),
        Path(__file__).resolve().parent / "data" / "contextual_linucb_state.json",
        Path("/tmp/contextual_linucb_state.json"),
    ])
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".linucb_write_{time.time_ns()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return Path("/tmp/contextual_linucb_state.json")


STATE_FILE = _state_path()


def _new_arm() -> dict[str, Any]:
    return {
        "A": (np.eye(CONTEXT_DIM) * LINUCB_RIDGE).tolist(),
        "b": np.zeros(CONTEXT_DIM).tolist(),
        "n": 0,
        "effective_n": 0.0,
    }


def _new_scope() -> dict[str, Any]:
    now = int(time.time())
    return {
        "arms": {arm: _new_arm() for arm in ARMS},
        "pending": {},
        "updates": 0,
        "last_selected": "",
        "selection_streak": 0,
        "direct_predict_only": True,
        "no_bootstrap_on_start": True,
        "no_feedback_update": True,
        "created_at": now,
        "updated_at": now,
    }


def _read_state() -> dict[str, Any]:
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
    except Exception:
        payload = {}
    if payload.get("version") != STATE_VERSION or payload.get("dim") != CONTEXT_DIM:
        payload = {}
    return {
        "version": STATE_VERSION,
        "dim": CONTEXT_DIM,
        "alpha": LINUCB_ALPHA,
        "ridge": LINUCB_RIDGE,
        "forgetting": LINUCB_FORGETTING,
        "scopes": payload.get("scopes") if isinstance(payload.get("scopes"), dict) else {},
    }


def _write_state(payload: Mapping[str, Any]) -> None:
    temporary = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False), encoding="utf-8")
    temporary.replace(STATE_FILE)


def make_scope_key(*, user_id: str = "", venue: str = "", room: str = "", shoe_id: str = "") -> str:
    raw = "|".join((
        str(user_id or "").strip(),
        str(venue or "").upper().strip(),
        str(room or "").strip(),
        str(shoe_id or "").strip(),
    ))
    return sha256((raw or "GLOBAL").encode("utf-8")).hexdigest()[:24]


def _history_fingerprint(history: Sequence[str]) -> str:
    return sha256("".join(history).encode("utf-8")).hexdigest()[:24]


def _arm_arrays(state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        matrix = np.asarray(state.get("A"), dtype=np.float64).reshape(CONTEXT_DIM, CONTEXT_DIM)
        vector = np.asarray(state.get("b"), dtype=np.float64).reshape(CONTEXT_DIM)
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
            raise ValueError
        return matrix, vector
    except Exception:
        return np.eye(CONTEXT_DIM) * LINUCB_RIDGE, np.zeros(CONTEXT_DIM)


class ContextualLinUCB:
    def __init__(self, alpha: float = LINUCB_ALPHA):
        self.alpha = max(0.0, float(alpha))
        self.generator = ContextGenerator()

    def _score(self, arm_state: Mapping[str, Any], context_vector: np.ndarray, alpha_scale: float) -> dict[str, float]:
        x = _model_x(context_vector)
        matrix, reward_vector = _arm_arrays(arm_state)
        try:
            theta = np.linalg.solve(matrix, reward_vector)
            solved_x = np.linalg.solve(matrix, x)
        except np.linalg.LinAlgError:
            matrix = matrix + np.eye(CONTEXT_DIM) * LINUCB_RIDGE
            theta = np.linalg.solve(matrix, reward_vector)
            solved_x = np.linalg.solve(matrix, x)
        mean = float(x @ theta)
        uncertainty = float(math.sqrt(max(0.0, x @ solved_x)))
        effective_alpha = self.alpha * max(0.5, min(2.5, float(alpha_scale)))
        return {
            "score": mean + effective_alpha * uncertainty,
            "mean": mean,
            "uncertainty": uncertainty,
            "effective_alpha": effective_alpha,
            "raw_n": float(arm_state.get("n", 0) or 0),
            "effective_n": float(arm_state.get("effective_n", arm_state.get("n", 0)) or 0.0),
        }

    def _decay(self, scope: dict[str, Any]) -> None:
        identity = np.eye(CONTEXT_DIM) * LINUCB_RIDGE
        arms = scope.setdefault("arms", {})
        for arm in ARMS:
            state = dict(arms.get(arm) or _new_arm())
            matrix, reward_vector = _arm_arrays(state)
            state["A"] = (identity + LINUCB_FORGETTING * (matrix - identity)).tolist()
            state["b"] = (LINUCB_FORGETTING * reward_vector).tolist()
            state["effective_n"] = LINUCB_FORGETTING * float(state.get("effective_n", state.get("n", 0)) or 0.0)
            arms[arm] = state

    def _update_scope(self, scope: dict[str, Any], *, action: str, context_vector: Sequence[float], actual_outcome: str) -> dict[str, Any]:
        action = str(action or "").upper().strip()
        actual = str(actual_outcome or "").upper().strip()
        if action not in ARMS or actual not in {"B", "P", "T"}:
            return {"updated": False, "reason": "invalid_feedback"}
        x = _model_x(context_vector)
        self._decay(scope)
        scope["updates"] = int(scope.get("updates", 0) or 0) + 1
        scope["updated_at"] = int(time.time())
        if actual == "T":
            return {
                "updated": True,
                "action": action,
                "actual_outcome": actual,
                "reward": 0.0,
                "directional_sample_applied": False,
                "forgetting": LINUCB_FORGETTING,
                "reason": "tie_reward_zero_no_directional_information",
            }
        reward = (0.95 if action == "B" else 1.0) if action == actual else -1.0
        state = dict(scope.get("arms", {}).get(action) or _new_arm())
        matrix, reward_vector = _arm_arrays(state)
        matrix = matrix + LINUCB_UPDATE_WEIGHT * np.outer(x, x)
        reward_vector = reward_vector + LINUCB_UPDATE_WEIGHT * reward * x
        state.update({
            "A": matrix.tolist(),
            "b": reward_vector.tolist(),
            "n": int(state.get("n", 0) or 0) + 1,
            "effective_n": float(state.get("effective_n", 0.0) or 0.0) + 1.0,
        })
        scope.setdefault("arms", {})[action] = state
        return {
            "updated": True,
            "action": action,
            "actual_outcome": actual,
            "reward": reward,
            "directional_sample_applied": True,
            "update_weight": LINUCB_UPDATE_WEIGHT,
            "forgetting": LINUCB_FORGETTING,
            "context_l2_normalized": False,
            "single_brain_update": True,
        }

    def update(self, *, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str, Any]:
        """Explicit compatibility API. Formal predict() never calls this."""
        with _LOCK:
            root = _read_state()
            scope = deepcopy(dict(root["scopes"].get(scope_key) or _new_scope()))
            result = self._update_scope(scope, action=action, context_vector=context_vector, actual_outcome=actual_outcome)
            result.update({"diagnostic_only": False, "formal_model": "contextual_linucb", "explicit_update_only": True})
            if clear_pending:
                scope["pending"] = {}
            root["scopes"][scope_key] = scope
            _write_state(root)
            return result

    def _tie_choice(self, scope: Mapping[str, Any], raw_history: Sequence[str]) -> tuple[str, str]:
        arms = dict(scope.get("arms") or {})
        banker_n = float((arms.get("B") or {}).get("effective_n", 0.0) or 0.0)
        player_n = float((arms.get("P") or {}).get("effective_n", 0.0) or 0.0)
        if abs(banker_n - player_n) > 1e-9:
            return ("B" if banker_n < player_n else "P"), "tie_less_sampled_arm"
        last = str(scope.get("last_selected") or "").upper().strip()
        if last in ARMS:
            return ("P" if last == "B" else "B"), "tie_opposite_previous_arm"
        # Match BBB app.js exactly:
        #   h = (h * 31 + token.charCodeAt(i)) >>> 0
        #   direction = h % 2 ? "B" : "P"
        token = "LOCAL_32D|" + "".join(raw_history)
        panel_hash = 0
        for char in token:
            panel_hash = (panel_hash * 31 + ord(char)) & 0xFFFFFFFF
        return ("B" if panel_hash % 2 else "P"), "tie_deterministic_history_hash"

    def _choose(self, scope: Mapping[str, Any], context_vector: np.ndarray, raw_history: Sequence[str]):
        bp_rounds = len(_bp(raw_history))
        base_scale = 1.35 if bp_rounds < 8 else 1.15 if bp_rounds < 15 else 1.0
        arms = dict(scope.get("arms") or {})
        effective_samples = {
            arm: max(0.0, float((arms.get(arm) or {}).get("effective_n", (arms.get(arm) or {}).get("n", 0)) or 0.0))
            for arm in ARMS
        }
        total_effective = sum(effective_samples.values())
        scores: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            imbalance = math.sqrt(max(1.0, total_effective + 2.0) / max(1.0, effective_samples[arm] + 1.0))
            alpha_scale = base_scale * _clip(imbalance, 0.85, LINUCB_ARM_ALPHA_MAX_SCALE)
            item = self._score(arms.get(arm, {}), context_vector, alpha_scale)
            item.update({"linucb_score": item["score"], "alpha_scale": alpha_scale, "external_score_component": 0.0})
            scores[arm] = item
        score_gap = float(scores["B"]["score"] - scores["P"]["score"])
        if abs(score_gap) <= LINUCB_SCORE_TIE_EPSILON:
            direction, reason = self._tie_choice(scope, raw_history)
        else:
            direction = "B" if score_gap > 0.0 else "P"
            reason = "linucb_ucb_score_argmax"
        return scores, effective_samples, total_effective, direction, reason, score_gap

    def _remember_selection(self, scope: dict[str, Any], direction: str) -> int:
        previous = str(scope.get("last_selected") or "").upper().strip()
        previous_streak = int(scope.get("selection_streak", 0) or 0)
        streak = previous_streak + 1 if previous == direction else 1
        scope.update({"last_selected": direction, "selection_streak": streak, "updated_at": int(time.time())})
        return streak

    def predict(self, *, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None, scope_key: str) -> dict[str, Any]:
        raw_history = _normalize_history(deepcopy(history))
        context = deepcopy(dict(shoe_context or {}))
        snapshot = self.generator.build(raw_history, context)
        raw_x = snapshot.vector.copy()
        x = _model_x(raw_x)
        fingerprint = _history_fingerprint(raw_history)

        with _LOCK:
            root = _read_state()
            scope = deepcopy(dict(root["scopes"].get(scope_key) or _new_scope()))
            bootstrap = {
                "applied": False,
                "reason": "web_panel_direct_no_bootstrap",
                "bootstrap_rounds": 0,
                "source_rounds": len(raw_history),
            }

            # BBB behavior: load the current local brain, calculate the latest
            # 32D context, predict directly, and save selection metadata only.
            # A/b, n, effective_n and updates remain untouched.
            feedback = {
                "updated": False,
                "reason": "web_panel_direct_no_feedback_update",
                "diagnostic_only": False,
                "formal_model": "contextual_linucb",
                "a_b_frozen_without_bootstrap": True,
            }

            scores, effective_samples, total_effective, direction, reason, score_gap = self._choose(scope, x, raw_history)
            raw_p_b = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, score_gap / LINUCB_SCORE_TEMPERATURE))))
            p_b = _clip(raw_p_b, PROBABILITY_MIN, PROBABILITY_MAX)
            p_p = 1.0 - p_b
            probabilities = {"B": p_b, "P": p_p, "T": 0.0}
            confidence = p_b if direction == "B" else p_p
            streak = self._remember_selection(scope, direction)

            snapshot.metadata.update({
                "selection_streak": streak,
                "linucb_direction_weight": 1.0,
                "road_prior_direction_weight": 0.0,
                "road_forecaster_direction_weight": 0.0,
                "derived_road_direction_weight": 0.0,
                "geometry_direction_weight": 0.0,
                "anti_echo_direction_weight": 0.0,
                "panel_bootstrap": deepcopy(bootstrap),
                "prediction_mode": "frozen_32d_local_brain_direct",
                "automatic_feedback_update_enabled": False,
                "a_b_frozen_without_bootstrap": True,
                "no_bootstrap_on_start": True,
            })

            # No pending prediction is stored in web-parity formal mode. The next B/P/T
            # result is simply part of the next history/context, exactly like
            # typing one more record into the test panel then pressing direct
            # predict.
            scope["pending"] = {}
            scope["frozen_direct_mode"] = True
            scope["direct_predict_only"] = True
            scope["no_bootstrap_on_start"] = True
            scope["no_feedback_update"] = True
            root["scopes"][scope_key] = scope
            _write_state(root)

        return {
            "model": "contextual_linucb_single_brain",
            "version": STATE_VERSION,
            "legacy_state_version": STATE_VERSION,
            "direction": direction,
            "selected_arm": direction,
            "arm_index": 1 if direction == "B" else 0,
            "probabilities": probabilities,
            "selected_win_probability": confidence,
            "confidence": confidence,
            "context_vector": [float(value) for value in raw_x],
            "model_context_vector": [float(value) for value in x],
            "context_feature_names": list(CONTEXT_FEATURE_NAMES),
            "context_dim": CONTEXT_DIM,
            "context_metadata": deepcopy(snapshot.metadata),
            "road_prior": {"diagnostic_only": True, "direction_weight": 0.0, "banker_probability": 0.5, "player_probability": 0.5},
            "road_prior_probability": {"B": 0.5, "P": 0.5},
            "road_forecaster": {"available": False, "diagnostic_only": True, "formal_direction_weight": 0.0},
            "features_used": dict(zip(CONTEXT_FEATURE_NAMES, [float(value) for value in raw_x])),
            "effective_support": total_effective,
            "uncertainty": scores[direction]["uncertainty"],
            "linucb_probability_correction": 0.0,
            "linucb_direction_weight": 1.0,
            "learning_reliability": _clip(total_effective / 10.0),
            "scores": scores,
            "score_gap": score_gap,
            "score_semantics": "contextual_linucb_ucb_scores_only",
            "alpha": self.alpha,
            "ridge": LINUCB_RIDGE,
            "forgetting": LINUCB_FORGETTING,
            "feedback_update": feedback,
            "bootstrap_update": deepcopy(bootstrap),
            "panel_bootstrap_applied": bool(bootstrap.get("applied")),
            "scope_key": scope_key,
            "arms": list(ARMS),
            "selection_reason": reason,
            "selection_streak": streak,
            "effective_arm_samples": effective_samples,
            "history_round_count": len(raw_history),
            "bp_history_round_count": len(_bp(raw_history)),
            "history_fingerprint": fingerprint,
            "short_shoe_target_rounds": "50-70",
            "formal_context_source": "single_brain_32d_panel_frozen_direct_context",
            "formal_direction_source": "contextual_linucb",
            "road_context_direction_weight": 0.0,
            "card_composition_direction_weight": 0.0,
            "probability_semantics": "bounded_logistic_mapping_of_linucb_ucb_score_gap",
            "cold_start_uses_road_prior": False,
            "shoe_context_used_for_formal_direction": False,
            "shoe_context_used_as_features": False,
            "history_estimated_shoe_features_used": True,
            "shoe_context_independent_vote": False,
            "external_road_vote_enabled": False,
            "anti_echo_external_penalty": False,
            "panel_compatible": True,
            "frozen_direct_mode": True,
            "direct_predict_only": True,
            "no_bootstrap_on_start": True,
            "automatic_feedback_update_enabled": False,
            "anti_lock": {
                "enabled": False,
                "method": "none_external_feedback_only",
                "tie_is_non_directional": True,
                "old_state_reused": False,
            },
        }


_DEFAULT_BANDIT = ContextualLinUCB()


def predict_bandit(*, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None, scope_key: str) -> dict[str, Any]:
    return _DEFAULT_BANDIT.predict(
        history=deepcopy(history),
        shoe_context=deepcopy(dict(shoe_context or {})),
        scope_key=str(scope_key or ""),
    )


def update_bandit(*, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str, Any]:
    return _DEFAULT_BANDIT.update(
        scope_key=str(scope_key or ""),
        action=action,
        context_vector=deepcopy(list(context_vector)),
        actual_outcome=actual_outcome,
        clear_pending=clear_pending,
    )


__all__ = [
    "ARMS",
    "CONTEXT_DIM",
    "CONTEXT_FEATURE_NAMES",
    "ContextGenerator",
    "ContextualLinUCB",
    "ESTIMATED_CARDS_PER_ROUND",
    "SHOE_DECKS",
    "LINUCB_ALPHA",
    "LINUCB_ARM_ALPHA_MAX_SCALE",
    "LINUCB_FORGETTING",
    "LINUCB_RIDGE",
    "LINUCB_SCORE_TIE_EPSILON",
    "LINUCB_UPDATE_WEIGHT",
    "PROBABILITY_MIN",
    "PROBABILITY_MAX",
    "ROAD_PRIOR_PROBABILITY_SPAN",
    "ROAD_PRIOR_SCORE_WEIGHT",
    "STATE_VERSION",
    "make_scope_key",
    "predict_bandit",
    "update_bandit",
]
