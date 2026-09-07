"""BBB V23.2 SAME/SWITCH hazard parity layer for BGS0.3V.1.

This module contains only the prediction-layer calibration used by BBB
app256continuation.js V23_2_SIDE_CONDITIONED_REFERENCE.  It deliberately
receives the existing BGS V18 256D/6x15 primitives through an API mapping so
LINE/OCR/LIFF/public interfaces and the frozen LinUCB base remain untouched.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

VERSION = "V23_2_SIDE_CONDITIONED_REFERENCE"
GAP_SCALE = 0.30


def _transition_sequence(sequence: Sequence[str], api: Mapping[str, Any]) -> list[str]:
    values = api["bp"](sequence)
    return ["S" if values[i] == values[i - 1] else "X" for i in range(1, len(values))]


def _transition_motif_order(tokens: Sequence[str], order: int, api: Mapping[str, Any]) -> dict[str, Any] | None:
    clip = api["clip"]
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
        "pSame": clip((same + prior) / denom),
        "pSwitch": clip((switch + prior) / denom),
        "support": clip(total / 3.2),
        "weightedSamples": total,
    }


def _transition_motif_backoff(sequence: Sequence[str], api: Mapping[str, Any]) -> dict[str, Any]:
    clip = api["clip"]
    tokens = _transition_sequence(sequence, api)
    if not tokens:
        return {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0, "order": 0, "agreement": 0.0, "details": []}
    details: list[dict[str, Any]] = []
    for order in range(min(5, len(tokens)), 0, -1):
        item = _transition_motif_order(tokens, order, api)
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
    support = clip(support_sum / 2.5)
    agreement = abs(sum(directions)) / len(directions) if directions else 0.0
    shrink = 0.48 + 0.52 * support
    p_same = clip(0.5 + (raw - 0.5) * shrink, 0.12, 0.88)
    return {
        "pSame": p_same,
        "pSwitch": 1.0 - p_same,
        "support": support,
        "order": details[0].get("order", 0),
        "agreement": agreement,
        "details": details,
    }


def _transition_depth_forecast(sequence: Sequence[str], api: Mapping[str, Any]) -> dict[str, Any]:
    clip = api["clip"]
    tokens = _transition_sequence(sequence, api)
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
    raw_persist = clip((persisted + prior) / denom) if denom > 0 else 0.5
    support = clip(reached / 4.0)
    p_persist = clip(0.5 + (raw_persist - 0.5) * (0.42 + 0.58 * support), 0.12, 0.88)
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


def _current_road_state(sequence: Sequence[str], api: Mapping[str, Any]) -> dict[str, Any]:
    road = api["build_big_road"](sequence)
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
        "sign": api["side_sign"](side),
    }


def _stage_curve(completed: Sequence[Mapping[str, Any]], side: str, max_stage: int, api: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage in range(1, max_stage + 1):
        signal = api["stage_survival"](completed, side, stage)
        out.append({
            "stage": stage,
            "pContinue": signal["cont"],
            "pTurn": signal["turn"],
            "support": signal["support"],
            "reached": signal.get("reached", 0.0),
        })
    return out


def _conditional_stage_evidence(completed: Sequence[Mapping[str, Any]], side: str, length: int, api: Mapping[str, Any]) -> dict[str, Any]:
    clip = api["clip"]
    if not side or not length:
        return {
            "pSame": 0.5, "pSwitch": 0.5, "support": 0.0,
            "contextPSame": 0.5, "contextPSwitch": 0.5, "contextSupport": 0.0,
            "nextPSame": 0.5, "nextSupport": 0.0,
            "cliff": 0.0, "cliffSupport": 0.0, "curve": [],
        }
    now = api["stage_survival"](completed, side, length)
    next_stage = api["stage_survival"](completed, side, length + 1)
    context = api["contextual_stage_stats"](completed, len(completed), side, length)
    curve = _stage_curve(completed, side, max(6, min(8, length + 2)), api)
    cliff = clip(float(now["cont"]) - float(next_stage["cont"]), 0.0, 1.0)
    cliff_support = clip(min(float(now.get("support", 0.0)), max(0.20, float(next_stage.get("support", 0.0)))))
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


def _candidate_evidence(sequence: Sequence[str], state: Mapping[str, Any], base_prediction: Mapping[str, Any], api: Mapping[str, Any]) -> dict[str, Any]:
    clip = api["clip"]
    cand = base_prediction.get("candidates") or api["big_road_candidates"](sequence)
    if not state.get("side"):
        return {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0, "sameScore": 0.5, "switchScore": 0.5}
    same_score = clip(cand.get(state["side"], 0.5))
    switch_score = clip(cand.get(state["opposite"], 0.5))
    diff = same_score - switch_score
    p_same = clip(1.0 / (1.0 + math.exp(-diff / 0.12)), 0.18, 0.82)
    return {
        "pSame": p_same,
        "pSwitch": 1.0 - p_same,
        "support": clip(cand.get("support", 0.0)),
        "sameScore": same_score,
        "switchScore": switch_score,
        "diff": diff,
    }


def _base_background_evidence(base_prediction: Mapping[str, Any], state: Mapping[str, Any], api: Mapping[str, Any]) -> dict[str, Any]:
    clip = api["clip"]
    if not state.get("side"):
        return {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0}
    gap = float(base_prediction.get("gap", 0.0) or 0.0)
    side_gap = float(state.get("sign", 0.0)) * gap
    p_same = clip(1.0 / (1.0 + math.exp(-side_gap / 0.42)), 0.32, 0.68)
    support = clip(abs(gap) / 0.24)
    return {"pSame": p_same, "pSwitch": 1.0 - p_same, "support": support, "gap": gap, "sideGap": side_gap}


def _side_conditioned_reference(sequence: Sequence[str], current_side: str, api: Mapping[str, Any]) -> dict[str, Any]:
    clip = api["clip"]
    outcomes = api["bp"](sequence)
    tokens = _transition_sequence(sequence, api)
    empty = {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0, "order": 0, "agreement": 0.0, "weightedSamples": 0.0, "details": []}
    if not current_side or len(outcomes) < 4 or len(tokens) < 2:
        return empty

    order_weights = {4: 1.00, 3: 0.78, 2: 0.56, 1: 0.36}
    details: list[dict[str, Any]] = []
    max_order = min(4, len(tokens))
    for order in range(max_order, 0, -1):
        context = "".join(tokens[-order:])
        same = switch = total = 0.0
        start = max(0, len(tokens) - 140)
        for i in range(start, len(tokens) - order):
            if "".join(tokens[i:i + order]) != context:
                continue
            anchor_side = outcomes[i + order]
            if anchor_side != current_side:
                continue
            age = len(tokens) - 1 - (i + order)
            weight = 0.965 ** age
            total += weight
            if tokens[i + order] == "S":
                same += weight
            else:
                switch += weight
        if total <= 0:
            continue
        prior = 0.75
        denom = total + 2.0 * prior
        raw_p_same = clip((same + prior) / denom)
        support = clip(total / 3.5)
        p_same = clip(0.5 + (raw_p_same - 0.5) * (0.40 + 0.60 * support), 0.16, 0.84)
        details.append({
            "order": order,
            "context": context,
            "pSame": p_same,
            "pSwitch": 1.0 - p_same,
            "support": support,
            "weightedSamples": total,
        })
    if not details:
        return empty

    same_sum = weight_sum = support_sum = weighted_samples = 0.0
    directions: list[int] = []
    for item in details:
        ow = order_weights.get(int(item["order"]), 0.30)
        weight = ow * (0.35 + 0.65 * float(item["support"]))
        same_sum += float(item["pSame"]) * weight
        weight_sum += weight
        support_sum += ow * float(item["support"])
        weighted_samples += float(item["weightedSamples"])
        if abs(float(item["pSame"]) - 0.5) >= 0.04:
            directions.append(1 if float(item["pSame"]) > 0.5 else -1)
    raw = same_sum / weight_sum if weight_sum else 0.5
    support = clip(support_sum / 2.2)
    agreement = abs(sum(directions)) / len(directions) if directions else 0.0
    p_same = clip(0.5 + (raw - 0.5) * (0.55 + 0.45 * support), 0.18, 0.82)
    return {
        "pSame": p_same,
        "pSwitch": 1.0 - p_same,
        "support": support,
        "order": details[0].get("order", 0),
        "agreement": agreement,
        "weightedSamples": weighted_samples,
        "currentSide": current_side,
        "details": details,
    }


def _low_evidence_calibration(stage: Mapping[str, Any], motif: Mapping[str, Any], depth: Mapping[str, Any], candidate: Mapping[str, Any], background: Mapping[str, Any], evidence_quality: float, api: Mapping[str, Any]) -> dict[str, Any]:
    clip = api["clip"]
    signed = api["signed"]
    quality = clip(evidence_quality)
    unknown_pressure = clip((0.46 - quality) / 0.28)
    components = [
        (stage["pSame"], stage.get("support", 0.0), 0.34),
        (stage["contextPSame"], stage.get("contextSupport", 0.0), 0.24),
        (candidate["pSame"], candidate.get("support", 0.0), 0.16),
        (background["pSame"], background.get("support", 0.0), 0.26),
    ]
    fallback_numerator = fallback_denominator = 0.0
    for p_value, support, base in components:
        weight = base * (0.35 + 0.65 * float(support))
        fallback_numerator += clip(p_value) * weight
        fallback_denominator += weight
    fallback_p_same = fallback_numerator / fallback_denominator if fallback_denominator else 0.5

    votes = [
        (stage["pSame"], stage.get("support", 0.0)),
        (stage["contextPSame"], stage.get("contextSupport", 0.0)),
        (motif["pSame"], motif.get("support", 0.0)),
        (depth["pSame"], depth.get("support", 0.0)),
        (candidate["pSame"], candidate.get("support", 0.0)),
        (background["pSame"], background.get("support", 0.0)),
    ]
    signed_vote = absolute_vote = 0.0
    for p_value, support in votes:
        edge = signed((clip(p_value) - 0.5) * 2.0)
        weight = 0.25 + 0.75 * clip(support)
        signed_vote += edge * weight
        absolute_vote += abs(edge) * weight
    agreement = clip(abs(signed_vote) / absolute_vote) if absolute_vote > 1e-9 else 0.0
    fallback_blend = 0.32 * unknown_pressure * (0.85 + 0.15 * (1.0 - agreement))
    conflict_shrink = 1.0 - 0.16 * unknown_pressure * (1.0 - agreement)
    return {
        "quality": quality,
        "unknownPressure": unknown_pressure,
        "agreement": agreement,
        "fallbackPSame": clip(fallback_p_same, 0.22, 0.78),
        "fallbackBlend": clip(fallback_blend, 0.0, 0.32),
        "conflictShrink": clip(conflict_shrink, 0.84, 1.0),
    }


def _single_hazard_signals(sequence: Sequence[str], base_prediction: Mapping[str, Any], api: Mapping[str, Any]) -> dict[str, Any]:
    clip = api["clip"]
    signed = api["signed"]
    state = _current_road_state(sequence, api)
    motif = _transition_motif_backoff(sequence, api)
    depth = _transition_depth_forecast(sequence, api)
    if not state["side"]:
        return {
            "state": state, "pSame": 0.5, "pSwitch": 0.5, "sameEdge": 0.0, "support": 0.0, "hazardSupport": 0.0,
            "motif": motif, "depth": depth,
            "stage": _conditional_stage_evidence([], "", 0, api),
            "candidate": {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0},
            "background": {"pSame": 0.5, "pSwitch": 0.5, "support": 0.0},
            "sideReference": _side_conditioned_reference(sequence, "", api),
            "cliffEvidence": 0.0, "formationBoost": 0.0,
            "shortSwitchPhase": False, "switchConsensus": False, "consensusStrength": 0.0, "staleScale": 1.0, "shortSwitchBoost": 0.0,
            "evidenceQuality": 0.0, "unknownPressure": 1.0, "evidenceAgreement": 0.0,
            "fallbackPSame": 0.5, "fallbackBlend": 0.0, "conflictShrink": 1.0,
            "sideReferenceWeight": 0.0, "sideReferenceAligned": True,
            "weights": {"stageWeight": 0.0, "contextWeight": 0.0, "motifWeight": 0.0, "depthWeight": 0.0, "candidateWeight": 0.0, "backgroundWeight": 0.0, "neutralWeight": 0.0},
        }

    stage = _conditional_stage_evidence(state["completed"], state["side"], state["length"], api)
    candidate = _candidate_evidence(sequence, state, base_prediction, api)
    background = _base_background_evidence(base_prediction, state, api)
    side_reference = _side_conditioned_reference(sequence, state["side"], api)

    short_switch_phase = state["length"] == 1 and depth.get("token") == "X"
    switch_consensus = bool(short_switch_phase and motif["pSwitch"] >= 0.54 and depth["pSwitch"] >= 0.54)
    consensus_strength = clip(((motif["pSwitch"] - 0.50) + (depth["pSwitch"] - 0.50)) / 0.28) if switch_consensus else 0.0
    stale_scale = clip(0.60 - 0.18 * consensus_strength, 0.42, 0.60) if switch_consensus else 1.0

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
        hazard_support = clip(
            0.18 * stage["support"] + 0.12 * stage["contextSupport"] + 0.30 * motif["support"] +
            0.24 * depth["support"] + 0.16 * candidate["support"]
        )
    else:
        hazard_support = clip(
            0.30 * stage["support"] + 0.20 * stage["contextSupport"] + 0.22 * motif["support"] +
            0.14 * depth["support"] + 0.14 * candidate["support"]
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
        numerator += clip(p_value) * weight
        denominator += weight
    p_same = numerator / denominator if denominator > 0 else 0.5

    cliff_gate = clip((state["length"] - 2) / 2.0) if state["length"] >= 3 else 0.0
    cliff_evidence = stage["cliff"] * stage["cliffSupport"] * cliff_gate
    p_same -= 0.22 * cliff_evidence

    short_switch_boost = 0.0
    if switch_consensus:
        evidence = clip(
            0.52 * clip((motif["pSwitch"] - 0.50) / 0.22) * (0.35 + 0.65 * motif["support"]) +
            0.48 * clip((depth["pSwitch"] - 0.50) / 0.22) * (0.35 + 0.65 * depth["support"])
        )
        short_switch_boost = 0.045 * evidence
        p_same -= short_switch_boost

    formation_boost = 0.0
    if state["length"] == 2:
        agreement = min(stage["pSame"], stage["contextPSame"])
        formation_support = min(1.0, 0.55 * stage["support"] + 0.45 * stage["contextSupport"])
        formation_boost = clip((agreement - 0.5) / 0.32) * formation_support * 0.055
        p_same += formation_boost

    calibration = _low_evidence_calibration(stage, motif, depth, candidate, background, hazard_support, api)
    if calibration["unknownPressure"] > 0:
        blend = calibration["fallbackBlend"]
        p_same = p_same * (1.0 - blend) + calibration["fallbackPSame"] * blend
        p_same = 0.5 + (p_same - 0.5) * calibration["conflictShrink"]

    side_gate = clip((side_reference["support"] - 0.24) / 0.56)
    side_edge = signed((side_reference["pSame"] - 0.5) * 2.0)
    core_edge_before_side = signed((p_same - 0.5) * 2.0)
    side_aligned = side_edge == 0 or core_edge_before_side == 0 or (side_edge > 0) == (core_edge_before_side > 0)
    disagreement_scale = 1.0 if side_aligned else (0.75 if abs(core_edge_before_side) < 0.10 else 0.45)
    side_reference_weight = (
        clip(0.10 * side_gate * (0.70 + 0.30 * side_reference["agreement"]) * disagreement_scale, 0.0, 0.10)
        if side_gate > 0 and abs(side_edge) >= 0.08 else 0.0
    )
    if side_reference_weight > 0:
        p_same = p_same * (1.0 - side_reference_weight) + side_reference["pSame"] * side_reference_weight

    p_same = clip(p_same, 0.16, 0.84)
    same_edge = signed((p_same - 0.5) * 2.0)
    support = clip(0.72 * hazard_support + 0.18 * background["support"] + 0.10 * abs(same_edge))
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
        "sideReference": side_reference,
        "cliffEvidence": cliff_evidence,
        "formationBoost": formation_boost,
        "shortSwitchPhase": short_switch_phase,
        "switchConsensus": switch_consensus,
        "consensusStrength": consensus_strength,
        "staleScale": stale_scale,
        "shortSwitchBoost": short_switch_boost,
        "evidenceQuality": calibration["quality"],
        "unknownPressure": calibration["unknownPressure"],
        "evidenceAgreement": calibration["agreement"],
        "fallbackPSame": calibration["fallbackPSame"],
        "fallbackBlend": calibration["fallbackBlend"],
        "conflictShrink": calibration["conflictShrink"],
        "sideReferenceWeight": side_reference_weight,
        "sideReferenceAligned": side_aligned,
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


def _diag(base_prediction: Mapping[str, Any], signal: Mapping[str, Any]) -> dict[str, Any]:
    state = signal["state"]
    stage = signal["stage"]
    depth = signal["depth"]
    motif = signal["motif"]
    side_reference = signal["sideReference"]
    return {
        "version": VERSION,
        "baseGap": float(base_prediction.get("gap", 0.0) or 0.0),
        "finalGap": float(state.get("sign", 0.0)) * float(signal["sameEdge"]) * GAP_SCALE,
        "pSame": signal["pSame"],
        "pSwitch": signal["pSwitch"],
        "currentSide": state.get("side", ""),
        "currentLength": state.get("length", 0),
        "transitionToken": depth.get("token", ""),
        "transitionDepth": depth.get("depth", 0),
        "motifPSame": motif.get("pSame", 0.5),
        "motifSupport": motif.get("support", 0.0),
        "depthPSame": depth.get("pSame", 0.5),
        "depthSupport": depth.get("support", 0.0),
        "stagePSame": stage.get("pSame", 0.5),
        "nextStagePSame": stage.get("nextPSame", 0.5),
        "stageSupport": stage.get("support", 0.0),
        "contextPSame": stage.get("contextPSame", 0.5),
        "contextSupport": stage.get("contextSupport", 0.0),
        "survivalCliff": stage.get("cliff", 0.0),
        "cliffEvidence": signal.get("cliffEvidence", 0.0),
        "formationBoost": signal.get("formationBoost", 0.0),
        "hazardSupport": signal.get("hazardSupport", 0.0),
        "evidenceQuality": signal.get("evidenceQuality", 0.0),
        "unknownPressure": signal.get("unknownPressure", 0.0),
        "evidenceAgreement": signal.get("evidenceAgreement", 0.0),
        "fallbackPSame": signal.get("fallbackPSame", 0.5),
        "fallbackBlend": signal.get("fallbackBlend", 0.0),
        "conflictShrink": signal.get("conflictShrink", 1.0),
        "sideReferencePSame": side_reference.get("pSame", 0.5),
        "sideReferenceSupport": side_reference.get("support", 0.0),
        "sideReferenceAgreement": side_reference.get("agreement", 0.0),
        "sideReferenceOrder": side_reference.get("order", 0),
        "sideReferenceWeight": signal.get("sideReferenceWeight", 0.0),
        "sideReferenceAligned": signal.get("sideReferenceAligned", True),
        "backgroundWeight": signal.get("weights", {}).get("backgroundWeight", 0.0),
        "shortSwitchPhase": signal.get("shortSwitchPhase", False),
        "switchConsensus": signal.get("switchConsensus", False),
        "staleScale": signal.get("staleScale", 1.0),
        "shortSwitchBoost": signal.get("shortSwitchBoost", 0.0),
    }


def hazard_choose(sequence: Sequence[str], base_prediction: Mapping[str, Any], api: Mapping[str, Any]) -> dict[str, Any]:
    clip = api["clip"]
    signal = _single_hazard_signals(sequence, base_prediction, api)
    current_side = signal["state"].get("side", "")
    base = dict(base_prediction)
    base_gap = float(base.get("gap", 0.0) or 0.0)

    if not current_side:
        diag = _diag(base, signal)
        v19_compat = {
            "version": VERSION,
            "baseGap": base_gap,
            "adjustment": 0.0,
            "startSignal": 0.0,
            "breakSignal": 0.0,
            "pSame": 0.5,
            "pSwitch": 0.5,
        }
        return {
            **base,
            "singleHazard": signal,
            "continuationSignals": signal,
            "v19": v19_compat,
            "v22": diag,
            "v23": diag,
            "v23_1": diag,
            "v23_2": diag,
        }

    direction = current_side if signal["pSame"] >= 0.5 else signal["state"]["opposite"]
    p_b_unclipped = signal["pSame"] if current_side == "B" else 1.0 - signal["pSame"]
    p_b = clip(p_b_unclipped, api["probability_min"], api["probability_max"])
    p_p = 1.0 - p_b
    confidence = p_b if direction == "B" else p_p
    gap = float(signal["state"]["sign"]) * float(signal["sameEdge"]) * GAP_SCALE

    regime = "S/X平衡"
    if signal["unknownPressure"] >= 0.55:
        regime = "低樣本校準"
    elif signal["shortSwitchPhase"] and signal["switchConsensus"] and signal["pSame"] < 0.50:
        regime = "短交錯切換"
    elif signal["state"]["length"] >= 3 and signal["cliffEvidence"] >= 0.08 and signal["pSame"] < 0.50:
        regime = "條件斷點"
    elif signal["pSame"] >= 0.56:
        regime = "條件延續"
    elif signal["pSame"] <= 0.44:
        regime = "條件切換"

    strength = clip(
        0.42 + 0.36 * signal["support"] + 0.18 * abs(signal["sameEdge"]) +
        0.04 * min(1.0, signal["state"]["length"] / 4.0)
    )
    diag = _diag(base, signal)
    diag["finalGap"] = gap
    adjustment = gap - base_gap
    v19_compat = {
        "version": VERSION,
        "baseGap": base_gap,
        "adjustment": adjustment,
        "startSignal": max(0.0, float(signal["sameEdge"])),
        "breakSignal": max(0.0, -float(signal["sameEdge"])),
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
        "v19": v19_compat,
        "v22": diag,
        "v23": diag,
        "v23_1": diag,
        "v23_2": diag,
    }
