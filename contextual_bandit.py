"""BGS 正式下一手 forecaster 的兩向相容入口。

方向與機率只由 road_forecaster 對已揭曉 B/P 歷史的因果式線上訓練決定。
V4 LinUCB 狀態、16 維 context、update API 與 Road Prior 保留作相容診斷，
方向權重為零，不能覆蓋 forecaster 的 argmax。OCR 與 LLM 不在本模組內。
"""
from __future__ import annotations

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

from road_forecaster import forecast_next
from shoe_constants import (
    AVERAGE_CARDS_PER_HAND,
    BURN_CARDS,
    SHOE_DECKS,
    estimate_remaining_cards,
)

ARMS = ("P", "B")
CONTEXT_DIM = 16
CONTEXT_FEATURE_NAMES = (
    "recent4_banker_centered",
    "recent8_banker_centered",
    "recent12_banker_centered",
    "recent20_banker_centered",
    "whole_shoe_banker_centered",
    "momentum_recent4_vs_recent12",
    "last_side_signed",
    "signed_run_length_norm",
    "recent6_switch_centered",
    "recent12_switch_centered",
    "switch_acceleration",
    "order1_transition_banker_edge",
    "order2_transition_banker_edge",
    "local8_cumulative_slope",
    "global_cumulative_slope",
    "local_global_slope_gap",
)

LINUCB_ALPHA = max(0.0, float(os.getenv("LINUCB_ALPHA", "0.5") or "0.5"))
LINUCB_RIDGE = max(1e-6, float(os.getenv("LINUCB_RIDGE", "1.0") or "1.0"))
LINUCB_UPDATE_WEIGHT = max(1e-3, float(os.getenv("LINUCB_UPDATE_WEIGHT", "1.0") or "1.0"))
LINUCB_FORGETTING = max(0.70, min(1.0, float(os.getenv("LINUCB_FORGETTING", "0.90") or "0.90")))
LINUCB_ARM_ALPHA_MAX_SCALE = max(1.0, min(2.5, float(os.getenv("LINUCB_ARM_ALPHA_MAX_SCALE", "1.60") or "1.60")))
LINUCB_SCORE_TIE_EPSILON = max(1e-12, float(os.getenv("LINUCB_SCORE_TIE_EPSILON", "0.000001") or "0.000001"))
ROAD_PRIOR_SCORE_WEIGHT = max(0.05, min(1.0, float(os.getenv("ROAD_PRIOR_SCORE_WEIGHT", "0.35") or "0.35")))
ROAD_PRIOR_PROBABILITY_SPAN = max(0.01, min(0.08, float(os.getenv("ROAD_PRIOR_PROBABILITY_SPAN", "0.055") or "0.055")))
LINUCB_PROBABILITY_CORRECTION_SPAN = max(0.0, min(0.04, float(os.getenv("LINUCB_PROBABILITY_CORRECTION_SPAN", "0.018") or "0.018")))
PROBABILITY_MIN = 0.42
PROBABILITY_MAX = 0.58
STATE_VERSION = "LINUCB-2ARM-ROAD-PRIMARY-DYNAMIC-V4"
_ANTI_CHASE_FEATURE_SCALE = 0.20
_LOCK = RLock()


def _clip(value: Any, lo: float, hi: float) -> float:
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
        items = history
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


def _centered_banker_ratio(sequence: Sequence[str], window: int | None = None) -> float:
    values = _bp(sequence)
    if window is not None:
        values = values[-max(1, int(window)):]
    if not values:
        return 0.0
    ratio = sum(v == "B" for v in values) / len(values)
    return _clip((ratio - 0.5) * 2.0, -1.0, 1.0)


def _run_length(sequence: Sequence[str]) -> tuple[str, int]:
    values = _bp(sequence)
    if not values:
        return "", 0
    last = values[-1]
    length = 0
    for value in reversed(values):
        if value != last:
            break
        length += 1
    return last, length


def _switch_rate(sequence: Sequence[str], window: int) -> float:
    values = _bp(sequence)[-max(2, int(window)):]
    if len(values) < 2:
        return 0.5
    switches = sum(values[i] != values[i - 1] for i in range(1, len(values)))
    return switches / (len(values) - 1)


def _transition_edge(sequence: Sequence[str], order: int) -> tuple[float, int]:
    values = _bp(sequence)
    order = max(1, min(2, int(order)))
    if len(values) <= order:
        return 0.0, 0
    key = tuple(values[-order:])
    banker = player = 0
    for i in range(order, len(values)):
        if tuple(values[i - order:i]) != key:
            continue
        if values[i] == "B":
            banker += 1
        elif values[i] == "P":
            player += 1
    support = banker + player
    return _clip((banker - player) / float(support + 2), -1.0, 1.0), support


def _cumulative_slope(sequence: Sequence[str], window: int | None = None) -> float:
    values = _bp(sequence)
    if window is not None:
        values = values[-max(2, int(window)):]
    if len(values) < 2:
        return 0.0
    encoded = np.asarray([1.0 if v == "B" else -1.0 for v in values], dtype=np.float64)
    cumulative = np.cumsum(encoded)
    x = np.arange(len(values), dtype=np.float64)
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-12:
        return 0.0
    slope = float(np.dot(x_centered, cumulative - float(np.mean(cumulative))) / denom)
    return float(math.tanh(slope))


def _model_x(vector: Sequence[float]) -> np.ndarray:
    x = np.asarray(vector, dtype=np.float64).reshape(CONTEXT_DIM)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    norm = float(np.linalg.norm(x))
    return x / norm if norm > 1e-12 else x


def _anti_chase_x(vector: Sequence[float]) -> np.ndarray:
    """Discount last-side/run evidence for scoring without changing the V4 state basis."""
    x = np.asarray(vector, dtype=np.float64).copy()
    x[6:8] *= _ANTI_CHASE_FEATURE_SCALE  # last_side_signed, signed_run_length_norm
    # Do not renormalize: doing so would undo the discount for sparse contexts.
    # Raw/exported contexts and update-time A/b/pending vectors remain unchanged.
    return x


def _attenuate_continuation_edge(edge: float, metadata: Mapping[str, Any]) -> tuple[float, float]:
    """Shrink only evidence continuing a 3+ B/P run; never force a reversal."""
    run_length = int(metadata.get("run_length", 0) or 0)
    run_side = str(metadata.get("run_side") or "")
    run_sign = 1.0 if run_side == "B" else -1.0 if run_side == "P" else 0.0
    if run_length < 3 or edge * run_sign <= 0.0:
        return edge, 1.0
    factor = 0.60 if run_length == 3 else 0.40 if run_length == 4 else 0.25
    return edge * factor, factor


@dataclass(frozen=True)
class ContextSnapshot:
    vector: np.ndarray
    metadata: dict[str, Any]


class ContextGenerator:
    def build(self, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None = None) -> ContextSnapshot:
        raw = _normalize_history(history)
        values = _bp(raw)
        r4 = _centered_banker_ratio(raw, 4)
        r8 = _centered_banker_ratio(raw, 8)
        r12 = _centered_banker_ratio(raw, 12)
        r20 = _centered_banker_ratio(raw, 20)
        whole = _centered_banker_ratio(raw, None)
        momentum = _clip(r4 - r12, -1.0, 1.0)
        last_side = 1.0 if values and values[-1] == "B" else -1.0 if values else 0.0
        run_side, run_length = _run_length(raw)
        run_sign = 1.0 if run_side == "B" else -1.0 if run_side == "P" else 0.0
        signed_run = run_sign * _clip(run_length / 8.0, 0.0, 1.0)
        switch6_raw = _switch_rate(raw, 6)
        switch12_raw = _switch_rate(raw, 12)
        switch6 = _clip((switch6_raw - 0.5) * 2.0, -1.0, 1.0)
        switch12 = _clip((switch12_raw - 0.5) * 2.0, -1.0, 1.0)
        switch_acceleration = _clip((switch6_raw - switch12_raw) * 2.0, -1.0, 1.0)
        order1, order1_support = _transition_edge(raw, 1)
        order2, order2_support = _transition_edge(raw, 2)
        local_slope = _cumulative_slope(raw, 8)
        global_slope = _cumulative_slope(raw, None)
        slope_gap = _clip(local_slope - global_slope, -1.0, 1.0)
        vector = np.asarray([r4, r8, r12, r20, whole, momentum, last_side, signed_run, switch6, switch12, switch_acceleration, order1, order2, local_slope, global_slope, slope_gap], dtype=np.float64)
        if vector.shape != (CONTEXT_DIM,):
            raise RuntimeError(f"context dimension mismatch: {vector.shape}")
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
        ctx = dict(shoe_context or {})
        remaining_hint = ctx.get("remaining_cards")
        if remaining_hint is None or remaining_hint == "":
            ctx["remaining_cards"] = estimate_remaining_cards(
                len(raw), decks=SHOE_DECKS,
                average_cards_per_hand=AVERAGE_CARDS_PER_HAND,
                burn_cards=BURN_CARDS,
            )
            ctx["remaining_cards_source"] = "round_count_estimate"
        try:
            diagnostic_remaining = max(0.0, float(ctx.get("remaining_cards", 0.0) or 0.0))
        except (TypeError, ValueError):
            diagnostic_remaining = estimate_remaining_cards(len(raw), decks=SHOE_DECKS)
            ctx["remaining_cards_source"] = "round_count_estimate"
        return ContextSnapshot(vector=vector, metadata={
            "raw_round_count": len(raw), "bp_round_count": len(values), "tie_count": sum(v == "T" for v in raw),
            "recent4_banker_centered": r4, "recent8_banker_centered": r8, "recent12_banker_centered": r12,
            "recent20_banker_centered": r20, "whole_shoe_banker_centered": whole,
            "momentum_recent4_vs_recent12": momentum, "last_side": values[-1] if values else "",
            "run_side": run_side, "run_length": run_length, "switch6": switch6_raw, "switch12": switch12_raw,
            "order1_transition_support": order1_support, "order2_transition_support": order2_support,
            "local8_cumulative_slope": local_slope, "global_cumulative_slope": global_slope,
            "raw_context_l2_norm": float(np.linalg.norm(vector)), "model_context_l2_norm": float(np.linalg.norm(_model_x(vector))),
            "context_scaling": "centered_road_features_plus_l2_internal",
            "feature_priority": "screenshot_big_road_plus_manual_history_primary",
            "formal_direction_source": "road_history_only", "shoe_context_used_for_formal_direction": False,
            "remaining_cards": diagnostic_remaining,
            "remaining_cards_source": str(ctx.get("remaining_cards_source") or "round_count_estimate"),
            "average_cards_per_hand": float(AVERAGE_CARDS_PER_HAND),
            "shoe_decks": int(SHOE_DECKS),
            "burn_cards": int(BURN_CARDS),
            "remaining_cards_semantics": "maturity_depth_estimate_not_exact_composition",
            "estimated_remaining_counts_0_to_9": [], "rank_ratio_source": "not_used_road_primary", "rank_ratios_a_to_10jqk": [],
        })


def _road_prior(snapshot: ContextSnapshot) -> dict[str, float | int | bool]:
    x = _anti_chase_x(snapshot.vector)
    meta = snapshot.metadata
    n = int(meta.get("bp_round_count", 0) or 0)
    support_conf = _clip(n / 12.0, 0.20 if n > 0 else 0.0, 1.0)
    order1_support = int(meta.get("order1_transition_support", 0) or 0)
    order2_support = int(meta.get("order2_transition_support", 0) or 0)
    trans1_conf = _clip(order1_support / 5.0, 0.0, 1.0)
    trans2_conf = _clip(order2_support / 4.0, 0.0, 1.0)
    # Halve transition/run coefficients; last_side_signed has no direct prior term.
    raw_edge = (
        0.12 * float(x[12]) * trans2_conf + 0.09 * float(x[11]) * trans1_conf + 0.15 * float(x[5])
        + 0.13 * float(x[13]) + 0.08 * float(x[15]) + 0.06 * float(x[14]) + 0.05 * float(x[0])
        + 0.04 * float(x[1]) + 0.04 * float(x[10]) + 0.015 * float(x[7]) * (1.0 - abs(float(x[8])))
    )
    edge_before_anti_chase = float(math.tanh(1.8 * raw_edge) * support_conf)
    edge, anti_chase_factor = _attenuate_continuation_edge(edge_before_anti_chase, meta)
    p_b = _clip(0.5 + ROAD_PRIOR_PROBABILITY_SPAN * edge, PROBABILITY_MIN, PROBABILITY_MAX)
    return {
        "edge": edge, "raw_edge": raw_edge, "banker_probability": p_b, "player_probability": 1.0 - p_b,
        "support_confidence": support_conf, "order1_support": order1_support, "order2_support": order2_support,
        "edge_before_anti_chase": edge_before_anti_chase, "anti_chase_factor": anti_chase_factor,
        "anti_chase_applied": anti_chase_factor < 1.0,
    }


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
        x = _anti_chase_x(x)
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
        return {"score": mean + effective_alpha * uncertainty, "mean": mean, "uncertainty": uncertainty, "effective_alpha": effective_alpha, "raw_n": float(arm_state.get("n", 0) or 0), "effective_n": float(arm_state.get("effective_n", arm_state.get("n", 0)) or 0.0)}

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
        x = _model_x(context_vector)
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
        return {"updated": True, "action": action, "actual_outcome": actual, "reward": reward, "directional_sample_applied": True, "update_weight": LINUCB_UPDATE_WEIGHT, "forgetting": LINUCB_FORGETTING, "context_l2_normalized": True}

    def update(self, *, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str, Any]:
        with _LOCK:
            root = _read_state()
            scope = dict(root["scopes"].get(scope_key) or _new_scope_state())
            result = self._update_scope(scope, action=action, context_vector=context_vector, actual_outcome=actual_outcome)
            result["diagnostic_only"] = True
            result["forecaster_update_mode"] = "replay_from_resolved_history_on_next_prediction"
            if clear_pending:
                scope["pending"] = {}
            root["scopes"][scope_key] = scope
            _write_state(root)
            return result

    def _apply_pending(self, scope: dict[str, Any], raw_history: Sequence[str]) -> dict[str, Any]:
        pending = dict(scope.get("pending") or {})
        if not pending:
            return {"updated": False, "reason": "no_pending_prediction"}
        previous_len = int(pending.get("raw_round_count", 0) or 0)
        if len(raw_history) <= previous_len:
            return {"updated": False, "reason": "no_new_resolved_round"}
        if _history_fingerprint(list(raw_history[:previous_len])) != str(pending.get("history_fingerprint") or ""):
            scope["pending"] = {}
            return {"updated": False, "reason": "history_reset_or_misaligned", "previous_len": previous_len, "current_len": len(raw_history)}
        result = self._update_scope(scope, action=str(pending.get("action") or ""), context_vector=list(pending.get("context_vector") or []), actual_outcome=raw_history[previous_len])
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
        raw = _normalize_history(history)
        forecast = forecast_next(raw)
        snapshot = self.generator.build(raw, shoe_context)
        prior = dict(_road_prior(snapshot), diagnostic_only=True, direction_weight=0.0)
        raw_x = snapshot.vector
        x = _model_x(raw_x)
        fingerprint = _history_fingerprint(raw)
        p_b, p_p = float(forecast["p_b"]), float(forecast["p_p"])
        # Strict forecaster argmax, with B as the deterministic exact-tie choice.
        # No LinUCB/streak/road-diagnostic rule may override these probabilities.
        direction = "B" if p_b >= p_p else "P"
        selected_probability = p_b if direction == "B" else p_p
        with _LOCK:
            root = _read_state()
            scope = dict(root["scopes"].get(scope_key) or _new_scope_state())
            pending = dict(scope.get("pending") or {})
            repeat_prediction = pending.get("history_fingerprint") == fingerprint and pending.get("raw_round_count") == len(raw)
            feedback = self._apply_pending(scope, raw)
            feedback["diagnostic_only"] = True
            feedback["forecaster_update_mode"] = "replay_from_resolved_history_only"
            n_bp = len(_bp(raw))
            base_scale = 1.35 if n_bp < 8 else 1.15 if n_bp < 15 else 1.0
            arms = dict(scope.get("arms") or {})
            eff = {arm: max(0.0, float((arms.get(arm) or {}).get("effective_n", (arms.get(arm) or {}).get("n", 0)) or 0.0)) for arm in ARMS}
            total_eff = sum(eff.values())
            scores: dict[str, dict[str, float]] = {}
            for arm in ARMS:
                imbalance = math.sqrt(max(1.0, total_eff + 2.0) / max(1.0, eff[arm] + 1.0))
                scale = base_scale * _clip(imbalance, 0.85, LINUCB_ARM_ALPHA_MAX_SCALE)
                item = self._score(arms.get(arm, {}), x, scale)
                item["linucb_score"] = float(item["score"])
                item["road_prior_component"] = 0.0
                item["forecaster_probability"] = p_b if arm == "B" else p_p
                item["score"] = item["forecaster_probability"]
                item["alpha_scale"] = scale
                item["anti_chase_score_adjustment"] = 0.0
                scores[arm] = item
            previous = str(scope.get("last_selected") or "").upper().strip()
            previous_streak = int(scope.get("selection_streak", 0) or 0)
            streak = previous_streak if repeat_prediction and previous == direction else previous_streak + 1 if previous == direction else 1
            snapshot.metadata.update({
                "formal_direction_source": "road_forecaster",
                "feature_priority": "forecaster_run_switch_supported_transitions_only",
                "legacy_context_diagnostic_only": True,
                "forecaster_features_used": dict(forecast["features_used"]),
                "anti_chase_enabled": True,
                "anti_chase_applied": bool(forecast["metadata"]["anti_chase_applied"]),
                "anti_chase_feature_factors": dict(forecast["metadata"]["feature_decay_factors"]),
                # Preserve old metadata keys; the legacy prior/feature scales
                # are diagnostics only and no extra final-score decay is used.
                "anti_chase_feature_scales": {"last_side_signed": _ANTI_CHASE_FEATURE_SCALE, "signed_run_length_norm": _ANTI_CHASE_FEATURE_SCALE},
                "anti_chase_prior_factor": float(prior["anti_chase_factor"]),
                "anti_chase_score_factor": 1.0,
                "anti_chase_probability_factor": 1.0,
                "selection_streak": streak,
                "linucb_direction_weight": 0.0,
                "road_prior_direction_weight": 0.0,
            })
            scope.update({"last_selected": direction, "selection_streak": streak, "updated_at": int(time.time()), "pending": {
                "action": direction, "context_vector": [float(v) for v in raw_x],
                "raw_round_count": len(raw), "history_fingerprint": fingerprint, "created_at": int(time.time()),
            }})
            root["scopes"][scope_key] = scope
            _write_state(root)
        return {
            "model": forecast["model_id"], "version": forecast["version"], "legacy_state_version": STATE_VERSION,
            "direction": direction, "selected_arm": direction, "arm_index": 1 if direction == "B" else 0,
            "probabilities": {"B": p_b, "P": p_p, "T": 0.0},
            "selected_win_probability": selected_probability, "confidence": selected_probability,
            "context_vector": [float(v) for v in raw_x], "model_context_vector": [float(v) for v in x],
            "context_feature_names": list(CONTEXT_FEATURE_NAMES), "context_dim": CONTEXT_DIM,
            "context_metadata": snapshot.metadata, "road_prior": prior,
            "road_prior_probability": {"B": float(prior["banker_probability"]), "P": float(prior["player_probability"])},
            "road_forecaster": forecast, "features_used": dict(forecast["features_used"]),
            "effective_support": forecast["effective_support"], "uncertainty": forecast["uncertainty"],
            "linucb_probability_correction": 0.0, "linucb_direction_weight": 0.0,
            "learning_reliability": _clip(total_eff / 10.0, 0.0, 1.0),
            "scores": scores, "score_semantics": "forecaster_probabilities; linucb_score_is_diagnostic_only",
            "alpha": self.alpha, "ridge": LINUCB_RIDGE, "forgetting": LINUCB_FORGETTING,
            "feedback_update": feedback, "scope_key": scope_key, "arms": list(ARMS),
            "selection_reason": "road_forecaster_probability_argmax", "selection_streak": streak, "effective_arm_samples": eff,
            "history_round_count": len(raw), "bp_history_round_count": n_bp, "history_fingerprint": fingerprint,
            "short_shoe_target_rounds": "50-70", "formal_context_source": "screenshot_big_road_plus_manual_history",
            "road_context_direction_weight": 1.0, "card_composition_direction_weight": 0.0,
            "probability_semantics": "causal_online_logistic_next_resolved_BP_probability",
            "cold_start_uses_road_prior": False, "shoe_context_used_for_formal_direction": False,
            "anti_lock": {"enabled": True, "method": "supervised_forecaster_causal_features_contribution_decay",
                          "fixed_player_tie_bias_removed": True, "tie_is_non_directional": True,
                          "old_v1_v2_v3_state_reused": False, "winner_probability_staircase_removed": True},
        }

_DEFAULT_BANDIT = ContextualLinUCB()


def predict_bandit(*, history: Iterable[Any] | str | None, shoe_context: Mapping[str, Any] | None, scope_key: str) -> dict[str, Any]:
    return _DEFAULT_BANDIT.predict(history=history, shoe_context=shoe_context, scope_key=scope_key)


def update_bandit(*, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str, clear_pending: bool = True) -> dict[str, Any]:
    return _DEFAULT_BANDIT.update(scope_key=scope_key, action=action, context_vector=context_vector, actual_outcome=actual_outcome, clear_pending=clear_pending)

__all__ = [
    "ARMS", "CONTEXT_DIM", "CONTEXT_FEATURE_NAMES", "ContextGenerator", "ContextualLinUCB",
    "ESTIMATED_CARDS_PER_ROUND", "SHOE_DECKS", "LINUCB_ALPHA", "LINUCB_ARM_ALPHA_MAX_SCALE", "LINUCB_FORGETTING",
    "LINUCB_RIDGE", "LINUCB_SCORE_TIE_EPSILON", "LINUCB_UPDATE_WEIGHT", "PROBABILITY_MIN", "PROBABILITY_MAX",
    "ROAD_PRIOR_PROBABILITY_SPAN", "ROAD_PRIOR_SCORE_WEIGHT", "STATE_VERSION", "make_scope_key", "predict_bandit", "update_bandit",
]
