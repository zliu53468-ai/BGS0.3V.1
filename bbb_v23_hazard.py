"""BBB V23_SHORT_X_DYNAMIC_HAZARD parity layer for BGS0.3V.1.

Only the formal prediction layer is ported here. Existing LINE/OCR/LIFF/UI,
public predictor interfaces, 256D context, 6x15 Big Road, Frozen LinUCB base,
probability bounds, and selection-state persistence remain intact.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence
import math
import time

import numpy as np
import contextual_bandit as cb

VERSION = "V23_SHORT_X_DYNAMIC_HAZARD"
GAP_SCALE = 0.30


def _transition_sequence(sequence: Sequence[str]) -> list[str]:
    values = cb._bp(sequence)
    return ["S" if values[i] == values[i - 1] else "X" for i in range(1, len(values))]


def _transition_motif_order(tokens: Sequence[str], order: int) -> dict[str, Any] | None:
    if order < 1 or len(tokens) <= order:
        return None
    context = "".join(tokens[-order:])
    same = switch = total = 0.0
    start = max(0, len(tokens) - 120)
    for i in range(start, len(tokens) - order):
        if "".join(tokens[i:i + order]) != context:
            continue
        age = len(tokens) - 1 - (i + order)
        weight = 0.965 ** age
        total += weight
        if tokens[i + order] == "S":
            same += weight
        else:
            switch += weight
    if total <= 0:
        return None
    prior = 0.62
    denom = total + 2.0 * prior
    return {
        "order": order,
        "context": context,
        "pSame": cb._clip((same + prior) / denom),
        "pSwitch": cb._clip((switch + prior) / denom),
        "support": cb._clip(total / 3.2),
        "weightedSamples": total,
    }


def _transition_motif_backoff(sequence: Sequence[str]) -> dict[str, Any]:
    tokens = _transition_sequence(sequence)
    if not tokens:
        return {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0, "order": 0, "agreement": 0.0, "details": []}
    details: list[dict[str, Any]] = []
    for order in range(min(5, len(tokens)), 0, -1):
        item = _transition_motif_order(tokens, order)
        if item:
            details.append(item)
    if not details:
        return {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0, "order": 0, "agreement": 0.0, "details": []}

    order_weights = {5: 1.00, 4: 0.84, 3: 0.68, 2: 0.52, 1: 0.36}
    same_sum = weight_sum = support_sum = 0.0
    directions: list[int] = []
    for item in details:
        ow = order_weights.get(int(item["order"]), 0.30)
        weight = ow * (0.35 + 0.65 * float(item["support"]))
        same_sum += float(item["pSame"]) * weight
        weight_sum += weight
        support_sum += ow * float(item["support"])
        if abs(float(item["pSame"]) - 0.5) >= 0.04:
            directions.append(1 if float(item["pSame"]) > 0.5 else -1)
    raw = same_sum / weight_sum if weight_sum else 0.5
    support = cb._clip(support_sum / 2.5)
    agreement = abs(sum(directions)) / len(directions) if directions else 0.0
    shrink = 0.48 + 0.52 * support
    p_same = cb._clip(0.5 + (raw - 0.5) * shrink, 0.12, 0.88)
    return {
        "pSame": p_same,
        "pSwitch": 1.0 - p_same,
        "support": support,
        "order": details[0].get("order", 0),
        "agreement": agreement,
        "details": details,
    }


def _transition_depth_forecast(sequence: Sequence[str]) -> dict[str, Any]:
    tokens = _transition_sequence(sequence)
    if not tokens:
        return {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0, "token": "", "depth": 0, "pPersist": 0.5}
    token = tokens[-1]
    depth = 1
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i] != token:
            break
        depth += 1

    completed_runs: list[dict[str, Any]] = []
    current = tokens[0]
    run_length = 1
    for value in tokens[1:]:
        if value == current:
            run_length += 1
        else:
            completed_runs.append({"token": current, "length": run_length})
            current = value
            run_length = 1

    items = [item for item in completed_runs if item["token"] == token][-24:]
    reached = persisted = ended = 0.0
    for i, item in enumerate(items):
        if int(item["length"]) < depth:
            continue
        weight = 0.95 ** (len(items) - 1 - i)
        reached += weight
        if int(item["length"]) > depth:
            persisted += weight
        else:
            ended += weight
    prior = 0.70
    denom = reached + 2.0 * prior
    raw_persist = cb._clip((persisted + prior) / denom) if denom > 0 else 0.5
    support = cb._clip(reached / 4.0)
    p_persist = cb._clip(0.5 + (raw_persist - 0.5) * (0.42 + 0.58 * support), 0.12, 0.88)
    p_same = p_persist if token == "S" else 1.0 - p_persist
    return {
        "pSame": p_same,
        "pSwitch": 1.0 - p_same,
        "support": support,
        "token": token,
        "depth": depth,
        "pPersist": p_persist,
        "reached": reached,
        "persisted": persisted,
        "ended": ended,
    }


def _current_road_state(sequence: Sequence[str]) -> dict[str, Any]:
    road = cb._build_big_road(sequence)
    current = road.get("currentStreak")
    streaks = list(road.get("streaks") or [])
    completed = streaks[:-1] if len(streaks) > 1 else []
    side = str(current.get("side") or "") if current else ""
    length = int(current.get("logicalLength", 0) or 0) if current else 0
    opposite = "P" if side == "B" else "B"
    return {
        "road": road,
        "current": current,
        "completed": completed,
        "side": side,
        "length": length,
        "opposite": opposite,
        "sign": cb._side_sign(side),
    }


def _stage_curve(completed: Sequence[Mapping[str, Any]], side: str, max_stage: int) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for stage in range(1, max_stage + 1):
        value = cb._stage_survival(completed, side, stage)
        curve.append({
            "stage": stage,
            "pContinue": value["cont"],
            "pTurn": value["turn"],
            "support": value["support"],
            "reached": value.get("reached", 0.0),
        })
    return curve


def _conditional_stage_evidence(completed: Sequence[Mapping[str, Any]], side: str, length: int) -> dict[str, Any]:
    if not side or not length:
        return {
            "pSame": 0.5, "pSwitch": 0.5, "support": 0.0,
            "contextPSame": 0.5, "contextPSwitch": 0.5, "contextSupport": 0.0,
            "nextPSame": 0.5, "nextSupport": 0.0,
            "cliff": 0.0, "cliffSupport": 0.0, "curve": [],
        }
    now = cb._stage_survival(completed, side, length)
    next_stage = cb._stage_survival(completed, side, length + 1)
    context = cb._contextual_stage_stats(completed, len(completed), side, length)
    curve = _stage_curve(completed, side, max(6, min(8, length + 2)))
    cliff = cb._clip(float(now["cont"]) - float(next_stage["cont"]), 0.0, 1.0)
    cliff_support = cb._clip(min(float(now.get("support", 0.0)), max(0.20, float(next_stage.get("support", 0.0)))))
    return {
        "pSame": float(now["cont"]),
        "pSwitch": float(now["turn"]),
        "support": float(now.get("support", 0.0)),
        "contextPSame": float(context["cont"]),
        "contextPSwitch": float(context["turn"]),
        "contextSupport": float(context.get("support", 0.0)),
        "nextPSame": float(next_stage["cont"]),
        "nextSupport": float(next_stage.get("support", 0.0)),
        "cliff": cliff,
        "cliffSupport": cliff_support,
        "curve": curve,
    }


def _candidate_evidence(sequence: Sequence[str], state: Mapping[str, Any], base_prediction: Mapping[str, Any]) -> dict[str, Any]:
    cand = base_prediction.get("candidates") or cb._big_road_candidates(sequence)
    if not state.get("side"):
        return {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0, "sameScore": 0.5, "switchScore": 0.5}
    same_score = cb._clip(cand.get(state["side"], 0.5))
    switch_score = cb._clip(cand.get(state["opposite"], 0.5))
    diff = same_score - switch_score
    p_same = cb._clip(1.0 / (1.0 + math.exp(-diff / 0.12)), 0.18, 0.82)
    return {
        "pSame": p_same,
        "pSwitch": 1.0 - p_same,
        "support": cb._clip(cand.get("support", 0.0)),
        "sameScore": same_score,
        "switchScore": switch_score,
        "diff": diff,
    }


def _base_background_evidence(base_prediction: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    if not state.get("side"):
        return {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0}
    gap = float(base_prediction.get("gap", 0.0) or 0.0)
    side_gap = float(state.get("sign", 0.0)) * gap
    p_same = cb._clip(1.0 / (1.0 + math.exp(-side_gap / 0.42)), 0.32, 0.68)
    support = cb._clip(abs(gap) / 0.24)
    return {"pSame": p_same, "pSwitch": 1.0 - p_same, "support": support, "gap": gap, "sideGap": side_gap}


def _single_hazard_signals(sequence: Sequence[str], base_prediction: Mapping[str, Any]) -> dict[str, Any]:
    state = _current_road_state(sequence)
    motif = _transition_motif_backoff(sequence)
    depth = _transition_depth_forecast(sequence)
    if not state["side"]:
        return {
            "state": state, "pSame": 0.5, "pSwitch": 0.5, "sameEdge": 0.0, "support": 0.0, "hazardSupport": 0.0,
            "motif": motif, "depth": depth,
            "stage": _conditional_stage_evidence([], "", 0),
            "candidate": {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0},
            "background": {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0},
            "cliffEvidence": 0.0, "formationBoost": 0.0,
            "shortSwitchPhase": False, "switchConsensus": False, "consensusStrength": 0.0,
            "staleScale": 1.0, "shortSwitchBoost": 0.0,
            "weights": {"stageWeight": 0.0, "contextWeight": 0.0, "motifWeight": 0.0, "depthWeight": 0.0, "candidateWeight": 0.0, "backgroundWeight": 0.0, "neutralWeight": 0.0},
        }

    stage = _conditional_stage_evidence(state["completed"], state["side"], state["length"])
    candidate = _candidate_evidence(sequence, state, base_prediction)
    background = _base_background_evidence(base_prediction, state)

    short_switch_phase = state["length"] == 1 and depth.get("token") == "X"
    switch_consensus = bool(short_switch_phase and motif["pSwitch"] >= 0.54 and depth["pSwitch"] >= 0.54)
    consensus_strength = cb._clip(((motif["pSwitch"] - 0.50) + (depth["pSwitch"] - 0.50)) / 0.28) if switch_consensus else 0.0
    stale_scale = cb._clip(0.60 - 0.18 * consensus_strength, 0.42, 0.60) if switch_consensus else 1.0

    stage_base = 0.17 if short_switch_phase else 0.30
    context_base = 0.12 if short_switch_phase else 0.20
    motif_base = 0.28 if short_switch_phase else 0.20
    depth_base = 0.22 if short_switch_phase else 0.14
    candidate_base = 0.07 if short_switch_phase else 0.09

    stage_weight = stage_base * (0.28 + 0.72 * stage["support"]) * stale_scale
    context_weight = context_base * (0.28 + 0.72 * stage["contextSupport"]) * stale_scale
    motif_weight = motif_base * (0.30 + 0.70 * motif["support"]) * (0.78 + 0.22 * motif["agreement"])
    depth_weight = depth_base * (0.30 + 0.70 * depth["support"])
    candidate_weight = candidate_base * (0.30 + 0.70 * candidate["support"])

    if short_switch_phase:
        hazard_support = cb._clip(
            0.18 * stage["support"] +
            0.12 * stage["contextSupport"] +
            0.30 * motif["support"] +
            0.24 * depth["support"] +
            0.16 * candidate["support"]
        )
    else:
        hazard_support = cb._clip(
            0.30 * stage["support"] +
            0.20 * stage["contextSupport"] +
            0.22 * motif["support"] +
            0.14 * depth["support"] +
            0.14 * candidate["support"]
        )

    background_weight = (0.16 * (1.0 - 0.72 * hazard_support) + 0.03) if short_switch_phase else (0.22 * (1.0 - 0.68 * hazard_support) + 0.04)
    neutral_weight = 0.34 * (1.0 - 0.55 * hazard_support)
    if short_switch_phase:
        neutral_weight *= 0.72

    numerator = 0.5 * neutral_weight
    denominator = neutral_weight
    for p_value, weight in (
        (stage["pSame"], stage_weight),
        (stage["contextPSame"], context_weight),
        (motif["pSame"], motif_weight),
        (depth["pSame"], depth_weight),
        (candidate["pSame"], candidate_weight),
        (background["pSame"], background_weight),
    ):
        numerator += cb._clip(p_value) * weight
        denominator += weight
    p_same = numerator / denominator if denominator > 0 else 0.5

    cliff_gate = cb._clip((state["length"] - 2) / 2.0) if state["length"] >= 3 else 0.0
    cliff_evidence = stage["cliff"] * stage["cliffSupport"] * cliff_gate
    p_same -= 0.22 * cliff_evidence

    short_switch_boost = 0.0
    if switch_consensus:
        evidence = cb._clip(
            0.52 * cb._clip((motif["pSwitch"] - 0.50) / 0.22) * (0.35 + 0.65 * motif["support"]) +
            0.48 * cb._clip((depth["pSwitch"] - 0.50) / 0.22) * (0.35 + 0.65 * depth["support"])
        )
        short_switch_boost = 0.045 * evidence
        p_same -= short_switch_boost

    formation_boost = 0.0
    if state["length"] == 2:
        agreement = min(stage["pSame"], stage["contextPSame"])
        formation_support = min(1.0, 0.55 * stage["support"] + 0.45 * stage["contextSupport"])
        formation_boost = cb._clip((agreement - 0.5) / 0.32) * formation_support * 0.055
        p_same += formation_boost

    p_same = cb._clip(p_same, 0.16, 0.84)
    same_edge = cb._signed((p_same - 0.5) * 2.0)
    support = cb._clip(0.72 * hazard_support + 0.18 * background["support"] + 0.10 * abs(same_edge))
    return {
        "state": state,
        "pSame": p_same,
        "pSwitch": 1.0 - p_same,
        "sameEdge": same_edge,
        "support": support,
        "hazardSupport": hazard_support,
        "motif": motif,
        "depth": depth,
        "stage": stage,
        "candidate": candidate,
        "background": background,
        "cliffEvidence": cliff_evidence,
        "formationBoost": formation_boost,
        "shortSwitchPhase": short_switch_phase,
        "switchConsensus": switch_consensus,
        "consensusStrength": consensus_strength,
        "staleScale": stale_scale,
        "shortSwitchBoost": short_switch_boost,
        "weights": {
            "stageWeight": stage_weight,
            "contextWeight": context_weight,
            "motifWeight": motif_weight,
            "depthWeight": depth_weight,
            "candidateWeight": candidate_weight,
            "backgroundWeight": background_weight,
            "neutralWeight": neutral_weight,
        },
    }


def _hazard_choose(sequence: Sequence[str], base_prediction: Mapping[str, Any]) -> dict[str, Any]:
    signal = _single_hazard_signals(sequence, base_prediction)
    current_side = signal["state"].get("side", "")
    base = dict(base_prediction)
    if not current_side:
        return {**base, "singleHazard": signal, "continuationSignals": signal}

    direction = current_side if signal["pSame"] >= 0.5 else signal["state"]["opposite"]
    p_b_unclipped = signal["pSame"] if current_side == "B" else 1.0 - signal["pSame"]
    p_b = cb._clip(p_b_unclipped, cb.PROBABILITY_MIN, cb.PROBABILITY_MAX)
    p_p = 1.0 - p_b
    confidence = p_b if direction == "B" else p_p
    gap = float(signal["state"]["sign"]) * float(signal["sameEdge"]) * GAP_SCALE

    regime = "S/X平衡"
    if signal["shortSwitchPhase"] and signal["switchConsensus"] and signal["pSame"] < 0.50:
        regime = "短交錯切換"
    elif signal["state"]["length"] >= 3 and signal["cliffEvidence"] >= 0.08 and signal["pSame"] < 0.50:
        regime = "條件斷點"
    elif signal["pSame"] >= 0.56:
        regime = "條件延續"
    elif signal["pSame"] <= 0.44:
        regime = "條件切換"

    strength = cb._clip(
        0.42 + 0.36 * signal["support"] + 0.18 * abs(signal["sameEdge"]) +
        0.04 * min(1.0, signal["state"]["length"] / 4.0)
    )
    diag = {
        "version": VERSION,
        "baseGap": float(base.get("gap", 0.0) or 0.0),
        "finalGap": gap,
        "pSame": signal["pSame"],
        "pSwitch": signal["pSwitch"],
        "currentSide": current_side,
        "currentLength": signal["state"]["length"],
        "transitionToken": signal["depth"].get("token", ""),
        "transitionDepth": signal["depth"].get("depth", 0),
        "motifPSame": signal["motif"].get("pSame", 0.5),
        "motifSupport": signal["motif"].get("support", 0.0),
        "depthPSame": signal["depth"].get("pSame", 0.5),
        "depthSupport": signal["depth"].get("support", 0.0),
        "stagePSame": signal["stage"].get("pSame", 0.5),
        "nextStagePSame": signal["stage"].get("nextPSame", 0.5),
        "stageSupport": signal["stage"].get("support", 0.0),
        "contextPSame": signal["stage"].get("contextPSame", 0.5),
        "contextSupport": signal["stage"].get("contextSupport", 0.0),
        "survivalCliff": signal["stage"].get("cliff", 0.0),
        "cliffEvidence": signal["cliffEvidence"],
        "formationBoost": signal["formationBoost"],
        "hazardSupport": signal["hazardSupport"],
        "backgroundWeight": signal["weights"].get("backgroundWeight", 0.0),
        "shortSwitchPhase": signal["shortSwitchPhase"],
        "switchConsensus": signal["switchConsensus"],
        "staleScale": signal["staleScale"],
        "shortSwitchBoost": signal["shortSwitchBoost"],
    }
    adjustment = gap - float(base.get("gap", 0.0) or 0.0)
    compatibility = {
        "version": VERSION,
        "baseGap": diag["baseGap"],
        "adjustment": adjustment,
        "startSignal": max(0.0, signal["sameEdge"]),
        "breakSignal": max(0.0, -signal["sameEdge"]),
        "pSame": signal["pSame"],
        "pSwitch": signal["pSwitch"],
    }
    return {
        **base,
        "direction": direction,
        "gap": gap,
        "confidence": confidence,
        "probabilities": {"B": p_b, "P": p_p, "T": 0.0},
        "regime": regime,
        "strength": strength,
        "singleHazard": signal,
        "continuationSignals": signal,
        "v19": compatibility,
        "v22": diag,
        "v23": diag,
    }


def predict_v23_bandit(*, history: Any, shoe_context: Mapping[str, Any] | None, scope_key: str) -> dict[str, Any]:
    """Return the same public policy shape as contextual_bandit.predict_bandit,
    but formal direction/probability are the BBB V23 hazard result over the
    untouched V18 frozen 256D base.
    """
    raw_history = cb._normalize_history(deepcopy(history))
    snapshot = cb.ContextGenerator().build(raw_history, deepcopy(dict(shoe_context or {})))
    x = np.nan_to_num(snapshot.vector.copy(), nan=0.0, posinf=2.0, neginf=-1.0)
    base = cb._base_choose(raw_history, x, snapshot.feature_names, snapshot.metadata)
    chosen = _hazard_choose(raw_history, base)
    direction = chosen["direction"]
    probabilities = dict(chosen["probabilities"])
    confidence = float(chosen["confidence"])
    fingerprint = cb._history_fingerprint(raw_history)

    with cb._LOCK:
        root = cb._read_state()
        scope = deepcopy(dict(root["scopes"].get(scope_key) or cb._new_scope()))
        previous = str(scope.get("last_selected") or "").upper().strip()
        streak = int(scope.get("selection_streak", 0) or 0) + 1 if previous == direction else 1
        scope.update({
            "last_selected": direction,
            "selection_streak": streak,
            "updated_at": int(time.time()),
            "pending": {},
            "frozen_direct_mode": True,
            "direct_predict_only": True,
            "no_bootstrap_on_start": True,
            "no_feedback_update": True,
            "no_ab_update": True,
            "no_decay": True,
        })
        root["scopes"][scope_key] = scope
        cb._write_state(root)

    bootstrap = {"applied": False, "reason": "bbb_v23_parity_no_bootstrap", "bootstrap_rounds": 0, "source_rounds": len(raw_history)}
    feedback = {"updated": False, "reason": "bbb_v23_parity_no_feedback_update", "diagnostic_only": False, "formal_model": "contextual_linucb", "a_b_frozen_without_bootstrap": True, "no_settlement": True, "no_decay": True}
    metadata = deepcopy(snapshot.metadata)
    metadata.update({
        "selection_streak": streak,
        "linucb_direction_weight": 1.0,
        "prediction_mode": "bbb_v18_v23_short_x_dynamic_frozen_256d_6x15_parity",
        "automatic_feedback_update_enabled": False,
        "a_b_frozen_without_bootstrap": True,
        "no_bootstrap_on_start": True,
        "no_replay": True,
        "no_decay": True,
        "continuation_signals": deepcopy(chosen.get("continuationSignals") or {}),
        "v23": deepcopy(chosen.get("v23") or {}),
        "regime": chosen["regime"],
        "structure_strength": chosen["strength"],
    })
    scores = deepcopy(chosen["scores"])
    v23 = deepcopy(chosen.get("v23") or {})
    base_gap = float(v23.get("baseGap", base.get("gap", 0.0)) or 0.0)
    adjustment = float(chosen["gap"]) - base_gap
    candidates = chosen["candidates"]
    road = candidates["road"]

    return {
        "model": "contextual_linucb_single_brain",
        "version": VERSION,
        "legacy_state_version": cb.STATE_VERSION,
        "direction": direction,
        "selected_arm": direction,
        "arm_index": 1 if direction == "B" else 0,
        "probabilities": probabilities,
        "selected_win_probability": confidence,
        "confidence": confidence,
        "context_vector": [float(v) for v in snapshot.vector],
        "model_context_vector": [float(v) for v in x],
        "context_feature_names": list(snapshot.feature_names),
        "context_dim": cb.CONTEXT_DIM,
        "context_metadata": metadata,
        "road_prior": {"diagnostic_only": True, "direction_weight": 0.0, "banker_probability": 0.5, "player_probability": 0.5},
        "road_prior_probability": {"B": 0.5, "P": 0.5},
        "road_forecaster": {"available": False, "diagnostic_only": True, "formal_direction_weight": 0.0},
        "features_used": dict(zip(snapshot.feature_names, [float(v) for v in snapshot.vector])),
        "effective_support": 0.0,
        "uncertainty": scores[direction]["uncertainty"],
        "linucb_probability_correction": 0.0,
        "linucb_direction_weight": 1.0,
        "learning_reliability": 0.0,
        "scores": scores,
        "score_gap": float(chosen["gap"]),
        "base_score_gap": base_gap,
        "v19_adjustment": adjustment,
        "score_semantics": "bbb_v18_frozen_linucb_base_plus_v23_single_sx_conditional_hazard",
        "alpha": cb.LINUCB_ALPHA,
        "ridge": cb.LINUCB_RIDGE,
        "forgetting": 1.0,
        "feedback_update": feedback,
        "bootstrap_update": bootstrap,
        "panel_bootstrap_applied": False,
        "scope_key": scope_key,
        "arms": list(cb.ARMS),
        "selection_reason": "bbb_v23_same_switch_hazard_to_bp",
        "selection_streak": streak,
        "effective_arm_samples": {"B": 0.0, "P": 0.0},
        "history_round_count": len(raw_history),
        "bp_history_round_count": len(cb._bp(raw_history)),
        "history_fingerprint": fingerprint,
        "short_shoe_target_rounds": "50-70",
        "formal_context_source": "bbb_v18_256d_128shoe_128road_6x15_frozen_context",
        "formal_direction_source": "contextual_linucb",
        "road_context_direction_weight": 0.0,
        "card_composition_direction_weight": 0.0,
        "probability_semantics": "bbb_v23_same_switch_probability_bounded_42_58",
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
        "no_replay": True,
        "no_previous_settlement": True,
        "no_ab_update": True,
        "no_decay": True,
        "regime": chosen["regime"],
        "structure_strength": chosen["strength"],
        "continuation_signals": deepcopy(chosen.get("continuationSignals") or {}),
        "v19": deepcopy(chosen.get("v19") or {}),
        "v23": v23,
        "big_road": {"rows": cb.BIG_ROAD_ROWS, "cols": cb.BIG_ROAD_COLS, "viewStartCol": road["viewStartCol"], "viewEndCol": road["viewEndCol"], "maxCol": road["maxCol"], "grid": road["grid"]},
        "anti_lock": {"enabled": False, "method": "none_external_feedback_only", "tie_is_non_directional": True, "old_state_reused": False},
    }
