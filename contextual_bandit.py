"""Single-brain Contextual LinUCB core for BGS.

Formal Banker/Player direction has exactly one owner: the two-arm LinUCB score
comparison. The fixed 32-dimensional context is deliberately split into sixteen
shoe-state features and sixteen road-state features. No external road model,
Anti-Echo rule, geometry vote, Markov vote or LLM may override B/P.

The public state/update API and contextual_linucb_state.json persistence contract
are preserved. Changing STATE_VERSION intentionally invalidates old A/b matrices
because the contextual feature basis changed from 16D to 32D.
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

from hsmm_regime import analyze_hidden_regime
from markov_model import update_and_predict_engine
from probabilistic_shoe_estimator import estimate_probabilistic_shoe
from road_model import build_standard_derived_roads
from run_length_hazard import analyze_run_length_hazard
from shoe_composition import analyze_shoe_composition, fresh_counts
from shoe_constants import (
    AVERAGE_CARDS_PER_HAND,
    BURN_CARDS,
    REFERENCE_HANDS,
    SHOE_DECKS,
    estimate_remaining_cards,
)

ARMS = ("P", "B")
CONTEXT_DIM = 32
CONTEXT_FEATURE_NAMES = (
    # 1-16: shoe state.
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
    # 17-32: road state.
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
STATE_VERSION = "LINUCB-2ARM-SINGLE-BRAIN-CONTEXT-16SHOE-16ROAD-32D-V7"
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
        if compact and all(c in {"B", "P", "T"} for c in compact):
            return list(compact)[-2000:]
        items: Iterable[Any] = [p for p in history.replace("|", ",").split(",") if p.strip()]
    else:
        items = deepcopy(list(history))
    out: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out[-2000:]


def _bp(history: Sequence[str]) -> list[str]:
    return [v for v in history if v in {"B", "P"}]


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


def _run_length(sequence: Sequence[str]) -> tuple[str, int]:
    values = _runs(sequence)
    return values[-1] if values else ("", 0)


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


def _model_x(vector: Sequence[float]) -> np.ndarray:
    x = np.asarray(vector, dtype=np.float64).reshape(CONTEXT_DIM)
    return np.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-1.0)


@dataclass(frozen=True)
class ContextSnapshot:
    vector: np.ndarray
    metadata: dict[str, Any]


class ContextGenerator:
    def build(self, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None = None) -> ContextSnapshot:
        raw = _normalize_history(deepcopy(history))
        bp_values = _bp(raw)
        ctx = deepcopy(dict(shoe_context or {}))
        try:
            decks = int(ctx.get("decks", SHOE_DECKS) or SHOE_DECKS)
        except (TypeError, ValueError):
            decks = SHOE_DECKS
        decks = max(1, min(16, decks))
        total_shoe_cards = float(52 * decks)

        exact_shoe = analyze_shoe_composition(ctx, default_decks=decks)
        exact_available = bool(exact_shoe.get("available"))
        if exact_available:
            counts = [float(v) for v in exact_shoe.get("remaining_counts", [])]
            if len(counts) != 10:
                counts = []
            remaining_cards = float(sum(counts)) if counts else total_shoe_cards
            remaining_source = str(exact_shoe.get("remaining_cards_source") or exact_shoe.get("source") or "exact")
        else:
            try:
                remaining_cards = float(ctx.get("remaining_cards", ""))
            except (TypeError, ValueError):
                remaining_cards = 0.0
            if remaining_cards <= 0.0:
                remaining_cards = estimate_remaining_cards(
                    len(raw), decks=decks,
                    average_cards_per_hand=AVERAGE_CARDS_PER_HAND,
                    burn_cards=BURN_CARDS,
                )
            counts = []
            remaining_source = str(ctx.get("remaining_cards_source") or "round_count_estimate")

        remaining_cards = max(0.0, min(total_shoe_cards, remaining_cards))
        remaining_ratio = _clip(remaining_cards / total_shoe_cards if total_shoe_cards > 0.0 else 1.0, 0.0, 1.0)
        penetration_ratio = _clip(1.0 - remaining_ratio, 0.0, 1.0)
        estimated_hands_remaining_norm = remaining_ratio
        shoe_maturity_ratio = _clip(len(raw) / float(max(1, REFERENCE_HANDS)), 0.0, 1.0)

        rank_ratios: list[float] = []
        shoe_group_ratios: list[float] = []
        exact_counts_available = bool(exact_available and len(counts) == 10)
        if exact_counts_available:
            fresh = [float(v) for v in fresh_counts(decks)]
            point_order = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
            for point in point_order:
                expected_at_depth = fresh[point] * remaining_ratio
                ratio = 1.0 if expected_at_depth <= 1e-12 else counts[point] / expected_at_depth
                rank_ratios.append(_clip(ratio, 0.0, 2.0))

            def grouped_ratio(points: Sequence[int]) -> float:
                expected = sum(fresh[point] * remaining_ratio for point in points)
                observed = sum(counts[point] for point in points)
                return _clip(1.0 if expected <= 1e-12 else observed / expected, 0.0, 2.0)

            shoe_group_ratios = [
                grouped_ratio((1, 2, 3)), grouped_ratio((4, 5)),
                grouped_ratio((6,)), grouped_ratio((7,)), grouped_ratio((8,)),
                grouped_ratio((9,)), grouped_ratio((0,)),
            ]
            rank_ratio_source = "exact_relative_to_expected_depth"
        else:
            rank_ratios = [1.0] * 10
            shoe_group_ratios = [1.0] * 7
            rank_ratio_source = "neutral_fallback"

        # Match the 32D test panel: the physical edge dimension stays neutral
        # without exact point counts and becomes a conservative zero-centered
        # weighted residual only when exact point composition is supplied.
        edge_weights = (0.02, 0.01, 0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, -0.03)
        physical_edge_proxy = (
            _clip(sum(weight * (ratio - 1.0) for weight, ratio in zip(edge_weights, rank_ratios)), -1.0, 1.0)
            if exact_counts_available else 0.0
        )
        shoe_information_reliability = 1.0 if exact_counts_available else 0.0

        # Preserve the existing probabilistic shoe calculation as diagnostics only.
        try:
            depth_reliability = float(ctx.get("remaining_cards_reliability", 0.65) or 0.65)
        except (TypeError, ValueError):
            depth_reliability = 0.65
        probabilistic_shoe = estimate_probabilistic_shoe(
            deepcopy(raw), decks=decks, particle_count=32,
            target_remaining_cards=int(round(remaining_cards)),
            depth_reliability=_clip(depth_reliability, 0.0, 1.0),
        )
        bp_probs = dict(probabilistic_shoe.get("bp_conditional_probabilities") or {})
        p_b_physical = float(bp_probs.get("B", 0.5) or 0.5)
        p_p_physical = float(bp_probs.get("P", 0.5) or 0.5)
        physical_total = p_b_physical + p_p_physical
        if physical_total <= 1e-12:
            p_b_physical = p_p_physical = 0.5
        else:
            p_b_physical /= physical_total
            p_p_physical /= physical_total
        baseline_b, baseline_p = 0.4586, 0.4462
        baseline_total = baseline_b + baseline_p
        baseline_edge = baseline_b / baseline_total - baseline_p / baseline_total
        combinatorial_advantage = _clip((p_b_physical - p_p_physical) - baseline_edge, -1.0, 1.0)

        road_runs = _runs(raw)
        current_side, run_length = road_runs[-1] if road_runs else ("", 0)
        previous_run_length = road_runs[-2][1] if len(road_runs) >= 2 else 0
        previous2_run_length = road_runs[-3][1] if len(road_runs) >= 3 else 0
        current_side_banker_binary = 1.0 if current_side == "B" else 0.0 if current_side == "P" else 0.5
        run_length_norm = _clip(run_length / 8.0, 0.0, 1.0)
        previous_run_length_norm = _clip(previous_run_length / 8.0, 0.0, 1.0)
        previous2_run_length_norm = _clip(previous2_run_length / 8.0, 0.0, 1.0)

        recent5_banker_ratio = _recent_banker_ratio(raw, 5)
        recent8_banker_ratio = _recent_banker_ratio(raw, 8)
        recent12_banker_ratio = _recent_banker_ratio(raw, 12)
        recent5_turn_rate = _turn_rate(raw, 5)
        recent8_turn_rate = _turn_rate(raw, 8)
        recent12_turn_rate = _turn_rate(raw, 12)

        hazard = analyze_run_length_hazard(deepcopy(raw))
        hazard_rate = _clip(hazard.get("turn_probability", 0.5), 0.0, 1.0)

        regime_input = update_and_predict_engine(deepcopy(raw))
        hsmm = analyze_hidden_regime(deepcopy(regime_input))
        regime_probability = _clip(hsmm.get("stable_probability", 0.5), 0.0, 1.0)

        derived = build_standard_derived_roads(deepcopy(bp_values))
        big_eye = list(derived.get("big_eye") or [])
        small_road = list(derived.get("small_road") or [])
        cockroach_road = list(derived.get("cockroach_road") or [])
        big_eye_regularity, big_eye_samples = _road_regularity(big_eye)
        small_regularity, small_samples = _road_regularity(small_road)
        cockroach_regularity, cockroach_samples = _road_regularity(cockroach_road)
        regularity_mean = (big_eye_regularity + small_regularity + cockroach_regularity) / 3.0
        derived_road_consensus = _clip(
            1.0 - (
                abs(big_eye_regularity - regularity_mean)
                + abs(small_regularity - regularity_mean)
                + abs(cockroach_regularity - regularity_mean)
            ) / 1.5,
            0.0, 1.0,
        )
        combined_samples = small_samples + cockroach_samples
        small_cockroach_regularity = (
            (small_regularity * small_samples + cockroach_regularity * cockroach_samples) / combined_samples
            if combined_samples > 0 else 0.5
        )

        # Preserve the previous derived-road metadata fields for compatibility.
        latest_marks: list[str] = []
        for road in (big_eye, small_road, cockroach_road):
            if road:
                latest = str(road[-1]).upper()
                if latest in {"R", "U"}:
                    latest_marks.append(latest)
        if latest_marks:
            regular_count = sum(mark == "R" for mark in latest_marks)
            derived_regularity_binary = 1.0 if regular_count * 2 >= len(latest_marks) else 0.0
        else:
            derived_regularity_binary = 0.0

        shoe_features = [
            remaining_ratio,
            penetration_ratio,
            estimated_hands_remaining_norm,
            shoe_maturity_ratio,
            *rank_ratios,
            physical_edge_proxy,
            shoe_information_reliability,
        ]
        road_features = [
            current_side_banker_binary,
            run_length_norm,
            previous_run_length_norm,
            previous2_run_length_norm,
            recent5_banker_ratio,
            recent8_banker_ratio,
            recent12_banker_ratio,
            recent5_turn_rate,
            recent8_turn_rate,
            recent12_turn_rate,
            hazard_rate,
            regime_probability,
            big_eye_regularity,
            small_regularity,
            cockroach_regularity,
            derived_road_consensus,
        ]
        vector = np.asarray([*shoe_features, *road_features], dtype=np.float64)
        if vector.shape != (CONTEXT_DIM,):
            raise RuntimeError(f"context dimension mismatch: {vector.shape}; expected {(CONTEXT_DIM,)}")
        vector = np.nan_to_num(vector, nan=0.0, posinf=2.0, neginf=-1.0)
        return ContextSnapshot(vector=vector, metadata={
            "raw_round_count": len(raw), "bp_round_count": len(bp_values), "tie_count": sum(v == "T" for v in raw),
            "remaining_cards": float(remaining_cards), "remaining_ratio": float(remaining_ratio),
            "penetration_ratio": float(penetration_ratio),
            "estimated_hands_remaining_norm": float(estimated_hands_remaining_norm),
            "shoe_maturity_ratio": float(shoe_maturity_ratio),
            "remaining_cards_source": remaining_source, "exact_composition_available": exact_counts_available,
            "rank_ratio_source": rank_ratio_source, "rank_ratios_a_to_10jqk": [float(v) for v in rank_ratios],
            "shoe_group_ratios": {
                "A23": float(shoe_group_ratios[0]), "45": float(shoe_group_ratios[1]),
                "6": float(shoe_group_ratios[2]), "7": float(shoe_group_ratios[3]),
                "8": float(shoe_group_ratios[4]), "9": float(shoe_group_ratios[5]),
                "10JQK": float(shoe_group_ratios[6]),
            },
            "physical_edge_proxy": float(physical_edge_proxy),
            "shoe_information_reliability": float(shoe_information_reliability),
            "combinatorial_advantage_offset": float(combinatorial_advantage),
            "probabilistic_shoe_reliability": float(probabilistic_shoe.get("reliability", 0.0) or 0.0),
            "hsmm_stable_probability": float(regime_probability), "hazard_rate": float(hazard_rate),
            "derived_road_regularity_binary": float(derived_regularity_binary), "derived_latest_marks": list(latest_marks),
            "run_length": int(run_length), "run_length_norm": float(run_length_norm), "shoe_decks": int(decks),
            "previous_run_length": int(previous_run_length), "previous2_run_length": int(previous2_run_length),
            "current_side": current_side, "current_side_banker_binary": float(current_side_banker_binary),
            "recent5_banker_ratio": float(recent5_banker_ratio),
            "recent8_banker_ratio": float(recent8_banker_ratio),
            "recent12_banker_ratio": float(recent12_banker_ratio),
            "recent5_turn_rate": float(recent5_turn_rate),
            "recent8_turn_rate": float(recent8_turn_rate),
            "recent12_turn_rate": float(recent12_turn_rate),
            "big_eye_regularity": float(big_eye_regularity),
            "small_road_regularity": float(small_regularity),
            "cockroach_road_regularity": float(cockroach_regularity),
            "small_cockroach_regularity": float(small_cockroach_regularity),
            "derived_road_consensus": float(derived_road_consensus),
            "context_layout": "16_shoe_plus_16_road_32d",
            "shoe_feature_values": [float(v) for v in shoe_features],
            "road_feature_values": [float(v) for v in road_features],
            "formal_direction_source": "contextual_linucb", "single_brain": True,
            "external_direction_votes_enabled": False, "anti_echo_external_penalty": False,
        })


def _state_path() -> Path:
    candidates: list[Path] = []
    configured = str(os.getenv("LINUCB_STATE_FILE", "") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates += [Path("/var/data/contextual_linucb_state.json"), Path(__file__).resolve().parent / "data" / "contextual_linucb_state.json", Path("/tmp/contextual_linucb_state.json")]
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
    return {"A": (np.eye(CONTEXT_DIM) * LINUCB_RIDGE).tolist(), "b": np.zeros(CONTEXT_DIM).tolist(), "n": 0, "effective_n": 0.0}


def _new_scope_state() -> dict[str, Any]:
    now = int(time.time())
    return {"arms": {a: _new_arm() for a in ARMS}, "pending": {}, "updates": 0, "last_selected": "", "selection_streak": 0, "created_at": now, "updated_at": now}


def _read_state() -> dict[str, Any]:
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
    except Exception:
        payload = {}
    if payload.get("version") != STATE_VERSION or payload.get("dim") != CONTEXT_DIM:
        payload = {}
    return {"version": STATE_VERSION, "dim": CONTEXT_DIM, "alpha": LINUCB_ALPHA, "ridge": LINUCB_RIDGE, "forgetting": LINUCB_FORGETTING, "scopes": payload.get("scopes") if isinstance(payload.get("scopes"), dict) else {}}


def _write_state(payload: Mapping[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def make_scope_key(*, user_id: str = "", venue: str = "", room: str = "", shoe_id: str = "") -> str:
    raw = "|".join((str(user_id or "").strip(), str(venue or "").upper().strip(), str(room or "").strip(), str(shoe_id or "").strip()))
    return sha256((raw or "GLOBAL").encode("utf-8")).hexdigest()[:24]


def _history_fingerprint(history: Sequence[str]) -> str:
    return sha256("".join(history).encode("utf-8")).hexdigest()[:24]


def _arm_arrays(state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        A = np.asarray(state.get("A"), dtype=np.float64).reshape(CONTEXT_DIM, CONTEXT_DIM)
        b = np.asarray(state.get("b"), dtype=np.float64).reshape(CONTEXT_DIM)
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError
        return A, b
    except Exception:
        return np.eye(CONTEXT_DIM) * LINUCB_RIDGE, np.zeros(CONTEXT_DIM)


class ContextualLinUCB:
    def __init__(self, alpha: float = LINUCB_ALPHA):
        self.alpha = max(0.0, float(alpha))
        self.generator = ContextGenerator()

    def _score(self, arm_state: Mapping[str, Any], x: np.ndarray, alpha_scale: float) -> dict[str, float]:
        x = _model_x(x)
        A, b = _arm_arrays(arm_state)
        try:
            theta = np.linalg.solve(A, b)
            solved_x = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            A = A + np.eye(CONTEXT_DIM) * LINUCB_RIDGE
            theta = np.linalg.solve(A, b)
            solved_x = np.linalg.solve(A, x)
        mean = float(x @ theta)
        uncertainty = float(math.sqrt(max(0.0, x @ solved_x)))
        effective_alpha = self.alpha * max(0.5, min(2.5, float(alpha_scale)))
        return {"score": float(mean + effective_alpha * uncertainty), "mean": mean, "uncertainty": uncertainty, "effective_alpha": effective_alpha, "raw_n": float(arm_state.get("n", 0) or 0), "effective_n": float(arm_state.get("effective_n", arm_state.get("n", 0)) or 0.0)}

    def _decay_arms(self, scope: dict[str, Any]) -> None:
        identity = np.eye(CONTEXT_DIM) * LINUCB_RIDGE
        arms = scope.setdefault("arms", {})
        for arm in ARMS:
            state = dict(arms.get(arm) or _new_arm())
            A, b = _arm_arrays(state)
            state["A"] = (identity + LINUCB_FORGETTING * (A - identity)).tolist()
            state["b"] = (LINUCB_FORGETTING * b).tolist()
            state["effective_n"] = LINUCB_FORGETTING * float(state.get("effective_n", state.get("n", 0)) or 0.0)
            arms[arm] = state

    def _update_scope(self, scope: dict[str, Any], *, action: str, context_vector: Sequence[float], actual_outcome: str) -> dict[str, Any]:
        action = str(action or "").upper().strip()
        actual = str(actual_outcome or "").upper().strip()
        if action not in ARMS or actual not in {"B", "P", "T"}:
            return {"updated": False, "reason": "invalid_feedback"}
        x = _model_x(deepcopy(list(context_vector)))
        self._decay_arms(scope)
        scope["updates"] = int(scope.get("updates", 0) or 0) + 1
        scope["updated_at"] = int(time.time())
        if actual == "T":
            return {"updated": True, "action": action, "actual_outcome": actual, "reward": 0.0, "directional_sample_applied": False, "forgetting": LINUCB_FORGETTING, "reason": "tie_reward_zero_no_directional_information"}
        reward = (0.95 if action == "B" else 1.0) if action == actual else -1.0
        state = dict(scope.get("arms", {}).get(action) or _new_arm())
        A, b = _arm_arrays(state)
        A = A + LINUCB_UPDATE_WEIGHT * np.outer(x, x)
        b = b + LINUCB_UPDATE_WEIGHT * reward * x
        state.update({"A": A.tolist(), "b": b.tolist(), "n": int(state.get("n", 0) or 0) + 1, "effective_n": float(state.get("effective_n", 0.0) or 0.0) + 1.0})
        scope.setdefault("arms", {})[action] = state
        return {"updated": True, "action": action, "actual_outcome": actual, "reward": reward, "directional_sample_applied": True, "update_weight": LINUCB_UPDATE_WEIGHT, "forgetting": LINUCB_FORGETTING, "context_l2_normalized": False, "single_brain_update": True}

    def update(self, *, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str, Any]:
        with _LOCK:
            root = _read_state()
            scope = deepcopy(dict(root["scopes"].get(scope_key) or _new_scope_state()))
            result = self._update_scope(scope, action=action, context_vector=deepcopy(list(context_vector)), actual_outcome=actual_outcome)
            result["diagnostic_only"] = False
            result["formal_model"] = "contextual_linucb"
            if clear_pending:
                scope["pending"] = {}
            root["scopes"][scope_key] = scope
            _write_state(root)
            return result

    def _apply_pending(self, scope: dict[str, Any], raw_history: Sequence[str]) -> dict[str, Any]:
        pending = deepcopy(dict(scope.get("pending") or {}))
        if not pending:
            return {"updated": False, "reason": "no_pending_prediction"}
        previous_len = int(pending.get("raw_round_count", 0) or 0)
        if len(raw_history) <= previous_len:
            return {"updated": False, "reason": "no_new_resolved_round"}
        if _history_fingerprint(list(raw_history[:previous_len])) != str(pending.get("history_fingerprint") or ""):
            scope["pending"] = {}
            return {"updated": False, "reason": "history_reset_or_misaligned", "previous_len": previous_len, "current_len": len(raw_history)}
        result = self._update_scope(scope, action=str(pending.get("action") or ""), context_vector=deepcopy(list(pending.get("context_vector") or [])), actual_outcome=raw_history[previous_len])
        scope["pending"] = {}
        result.update({"history_aligned": True, "resolved_history_index": previous_len, "history_rounds_after_append": len(raw_history)})
        return result

    def _tie_choice(self, scope: Mapping[str, Any], scope_key: str, fingerprint: str) -> tuple[str, str]:
        arms = dict(scope.get("arms") or {})
        b_n = float((arms.get("B") or {}).get("effective_n", 0.0) or 0.0)
        p_n = float((arms.get("P") or {}).get("effective_n", 0.0) or 0.0)
        if abs(b_n - p_n) > 1e-9:
            return ("B" if b_n < p_n else "P"), "tie_less_sampled_arm"
        last = str(scope.get("last_selected") or "").upper().strip()
        if last in ARMS:
            return ("P" if last == "B" else "B"), "tie_opposite_previous_arm"
        token = sha256(f"{scope_key}|{fingerprint}".encode("utf-8")).digest()[0]
        return ("B" if token % 2 else "P"), "tie_deterministic_balanced_hash"

    def predict(self, *, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None, scope_key: str) -> dict[str, Any]:
        raw = _normalize_history(deepcopy(history))
        snapshot = self.generator.build(deepcopy(raw), deepcopy(dict(shoe_context or {})))
        raw_x = snapshot.vector.copy()
        x = _model_x(raw_x)
        fingerprint = _history_fingerprint(raw)
        with _LOCK:
            root = _read_state()
            scope = deepcopy(dict(root["scopes"].get(scope_key) or _new_scope_state()))
            pending = deepcopy(dict(scope.get("pending") or {}))
            repeat_prediction = pending.get("history_fingerprint") == fingerprint and pending.get("raw_round_count") == len(raw)
            feedback = self._apply_pending(scope, raw)
            feedback["diagnostic_only"] = False
            feedback["formal_model"] = "contextual_linucb"
            n_bp = len(_bp(raw))
            base_scale = 1.35 if n_bp < 8 else 1.15 if n_bp < 15 else 1.0
            arms = dict(scope.get("arms") or {})
            eff = {arm: max(0.0, float((arms.get(arm) or {}).get("effective_n", (arms.get(arm) or {}).get("n", 0)) or 0.0)) for arm in ARMS}
            total_eff = sum(eff.values())
            scores: dict[str, dict[str, float]] = {}
            for arm in ARMS:
                imbalance = math.sqrt(max(1.0, total_eff + 2.0) / max(1.0, eff[arm] + 1.0))
                alpha_scale = base_scale * _clip(imbalance, 0.85, LINUCB_ARM_ALPHA_MAX_SCALE)
                item = self._score(arms.get(arm, {}), x, alpha_scale)
                item["linucb_score"] = float(item["score"])
                item["alpha_scale"] = float(alpha_scale)
                item["external_score_component"] = 0.0
                scores[arm] = item
            score_b = float(scores["B"]["score"])
            score_p = float(scores["P"]["score"])
            score_gap = score_b - score_p
            if abs(score_gap) <= LINUCB_SCORE_TIE_EPSILON:
                direction, selection_reason = self._tie_choice(scope, scope_key, fingerprint)
            else:
                direction = "B" if score_gap > 0.0 else "P"
                selection_reason = "linucb_ucb_score_argmax"
            raw_p_b = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, score_gap / LINUCB_SCORE_TEMPERATURE))))
            p_b = _clip(raw_p_b, PROBABILITY_MIN, PROBABILITY_MAX)
            p_p = 1.0 - p_b
            probabilities = {"B": float(p_b), "P": float(p_p), "T": 0.0}
            selected_probability = p_b if direction == "B" else p_p
            previous = str(scope.get("last_selected") or "").upper().strip()
            previous_streak = int(scope.get("selection_streak", 0) or 0)
            streak = previous_streak if repeat_prediction and previous == direction else previous_streak + 1 if previous == direction else 1
            snapshot.metadata.update({"selection_streak": int(streak), "linucb_direction_weight": 1.0, "road_prior_direction_weight": 0.0, "road_forecaster_direction_weight": 0.0, "derived_road_direction_weight": 0.0, "geometry_direction_weight": 0.0, "anti_echo_direction_weight": 0.0})
            scope.update({"last_selected": direction, "selection_streak": streak, "updated_at": int(time.time()), "pending": {"action": direction, "context_vector": [float(v) for v in raw_x], "raw_round_count": len(raw), "history_fingerprint": fingerprint, "created_at": int(time.time())}})
            root["scopes"][scope_key] = scope
            _write_state(root)
        return {
            "model": "contextual_linucb_single_brain", "version": STATE_VERSION, "legacy_state_version": STATE_VERSION,
            "direction": direction, "selected_arm": direction, "arm_index": 1 if direction == "B" else 0,
            "probabilities": probabilities, "selected_win_probability": float(selected_probability), "confidence": float(selected_probability),
            "context_vector": [float(v) for v in raw_x], "model_context_vector": [float(v) for v in x],
            "context_feature_names": list(CONTEXT_FEATURE_NAMES), "context_dim": CONTEXT_DIM,
            "context_metadata": deepcopy(snapshot.metadata),
            "road_prior": {"diagnostic_only": True, "direction_weight": 0.0, "banker_probability": 0.5, "player_probability": 0.5},
            "road_prior_probability": {"B": 0.5, "P": 0.5},
            "road_forecaster": {"available": False, "diagnostic_only": True, "formal_direction_weight": 0.0},
            "features_used": dict(zip(CONTEXT_FEATURE_NAMES, [float(v) for v in raw_x])),
            "effective_support": float(total_eff), "uncertainty": float(scores[direction]["uncertainty"]),
            "linucb_probability_correction": 0.0, "linucb_direction_weight": 1.0,
            "learning_reliability": _clip(total_eff / 10.0, 0.0, 1.0),
            "scores": scores, "score_gap": float(score_gap), "score_semantics": "contextual_linucb_ucb_scores_only",
            "alpha": self.alpha, "ridge": LINUCB_RIDGE, "forgetting": LINUCB_FORGETTING,
            "feedback_update": feedback, "scope_key": scope_key, "arms": list(ARMS),
            "selection_reason": selection_reason, "selection_streak": int(streak), "effective_arm_samples": eff,
            "history_round_count": len(raw), "bp_history_round_count": n_bp, "history_fingerprint": fingerprint,
            "short_shoe_target_rounds": "50-70", "formal_context_source": "single_brain_32d_context",
            "formal_direction_source": "contextual_linucb", "road_context_direction_weight": 0.0, "card_composition_direction_weight": 0.0,
            "probability_semantics": "bounded_logistic_mapping_of_linucb_ucb_score_gap",
            "cold_start_uses_road_prior": False, "shoe_context_used_for_formal_direction": True, "shoe_context_used_as_features": True,
            "shoe_context_independent_vote": False, "external_road_vote_enabled": False, "anti_echo_external_penalty": False,
            "anti_lock": {"enabled": False, "method": "none_external_feedback_only", "tie_is_non_directional": True, "old_v1_v2_v3_v4_state_reused": False},
        }


_DEFAULT_BANDIT = ContextualLinUCB()


def predict_bandit(*, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None, scope_key: str) -> dict[str, Any]:
    return _DEFAULT_BANDIT.predict(history=deepcopy(history), shoe_context=deepcopy(dict(shoe_context or {})), scope_key=str(scope_key or ""))


def update_bandit(*, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str, Any]:
    return _DEFAULT_BANDIT.update(scope_key=str(scope_key or ""), action=action, context_vector=deepcopy(list(context_vector)), actual_outcome=actual_outcome, clear_pending=clear_pending)


__all__ = [
    "ARMS", "CONTEXT_DIM", "CONTEXT_FEATURE_NAMES", "ContextGenerator", "ContextualLinUCB",
    "ESTIMATED_CARDS_PER_ROUND", "SHOE_DECKS", "LINUCB_ALPHA", "LINUCB_ARM_ALPHA_MAX_SCALE", "LINUCB_FORGETTING",
    "LINUCB_RIDGE", "LINUCB_SCORE_TIE_EPSILON", "LINUCB_UPDATE_WEIGHT", "PROBABILITY_MIN", "PROBABILITY_MAX",
    "ROAD_PRIOR_PROBABILITY_SPAN", "ROAD_PRIOR_SCORE_WEIGHT", "STATE_VERSION", "make_scope_key", "predict_bandit", "update_bandit",
]
