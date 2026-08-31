from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# run_length_hazard.py: softer backoff, higher usable hazard reliability,
# preserve context posterior instead of pulling sparse backoff toward 0.5.
replace_once(
    "run_length_hazard.py",
    "HAZARD_TRANSITION_SUPPORT_BOOST = 2\nHAZARD_BACKOFF_ALPHA = 0.82\nHAZARD_PRIOR_STRENGTH = 6.0\nMAX_HAZARD_RELIABILITY = 0.15",
    "HAZARD_TRANSITION_SUPPORT_BOOST = 1\nHAZARD_BACKOFF_ALPHA = 0.88\nHAZARD_PRIOR_STRENGTH = 6.0\nMAX_HAZARD_RELIABILITY = 0.25",
)
replace_once(
    "run_length_hazard.py",
    "LENGTH_SMOOTH_BLEND_MIN = 0.22\nLENGTH_SMOOTH_BLEND_MAX = 0.50",
    "LENGTH_SMOOTH_BLEND_MIN = 0.18\nLENGTH_SMOOTH_BLEND_MAX = 0.42",
)
replace_once(
    "run_length_hazard.py",
    "    raw_continue_probability = (\n        (1.0 - penalty) * 0.5\n        + penalty * float(selected_probability[\"CONTINUE\"])\n    )\n    raw_turn_probability = 1.0 - raw_continue_probability\n",
    "    # Backoff now changes reliability, not the probability itself. Sparse\n    # contexts are blended with the global parent below instead of being pulled\n    # directly toward 0.5, which preserves useful transition structure.\n    raw_continue_probability = float(selected_probability[\"CONTINUE\"])\n    raw_turn_probability = float(selected_probability[\"TURN\"])\n",
)
replace_once(
    "run_length_hazard.py",
    "    support_factor = (\n        support / (support + effective_support_threshold)\n        if support > 0.0 else 0.0\n    )",
    "    support_factor = (\n        support / (support + 0.75 * effective_support_threshold)\n        if support > 0.0 else 0.0\n    )",
)
replace_once(
    "run_length_hazard.py",
    "        + 0.16 * critical_proximity\n        + 0.08 * (1.0 - support_factor)\n        + 0.04 * (1.0 - transition_stability),",
    "        + 0.14 * critical_proximity\n        + 0.06 * (1.0 - support_factor)\n        + 0.03 * (1.0 - transition_stability),",
)
replace_once(
    "run_length_hazard.py",
    "    stability_factor = 0.82 + 0.18 * transition_stability\n    reliability = min(\n        MAX_HAZARD_RELIABILITY,\n        MAX_HAZARD_RELIABILITY\n        * support_factor\n        * maturity\n        * penalty\n        * stability_factor\n        * (0.65 + 0.35 * separation),\n    )",
    "    stability_factor = 0.90 + 0.10 * transition_stability\n    backoff_reliability = 0.85 + 0.15 * penalty\n    reliability = min(\n        MAX_HAZARD_RELIABILITY,\n        MAX_HAZARD_RELIABILITY\n        * support_factor\n        * maturity\n        * backoff_reliability\n        * stability_factor\n        * (0.75 + 0.25 * separation),\n    )",
)
replace_once(
    "run_length_hazard.py",
    "        \"backoff_penalty\": float(penalty),\n        \"reliability\": float(reliability),",
    "        \"backoff_penalty\": float(penalty),\n        \"backoff_reliability_factor\": float(backoff_reliability),\n        \"reliability\": float(reliability),",
)

# hsmm_regime.py: keep transition informative and make entry evidence gradual.
replace_once(
    "hsmm_regime.py",
    "        \"duration_mean\": 3.2,\n        \"pattern_factor\": 0.66,\n        \"markov_factor\": 0.68,\n        \"road_factor\": 0.62,\n        \"hazard_factor\": 0.88,",
    "        \"duration_mean\": 3.4,\n        \"pattern_factor\": 0.72,\n        \"markov_factor\": 0.70,\n        \"road_factor\": 0.70,\n        \"hazard_factor\": 0.84,",
)
replace_once(
    "hsmm_regime.py",
    "    sigma = 0.62",
    "    sigma = 0.68",
)
replace_once(
    "hsmm_regime.py",
    "    event_strength = _clip(\n        0.62 * (1.0 if change_point else 0.0)\n        + 0.38 * (1.0 if pattern_break else 0.0)\n    )\n    transition_evidence = _clip(\n        (\n            0.72 * event_strength\n            + 0.28 * volatility\n        )\n        * (0.72 + 0.28 * (1.0 - transition_stability))\n    )\n    if not (change_point or pattern_break):\n        transition_evidence *= 0.60",
    "    event_strength = _clip(\n        0.50 * (1.0 if change_point else 0.0)\n        + 0.30 * (1.0 if pattern_break else 0.0)\n    )\n    transition_evidence = _clip(\n        0.60 * event_strength\n        + 0.25 * volatility\n        + 0.15 * (1.0 - transition_stability)\n    )\n    if not (change_point or pattern_break):\n        transition_evidence *= 0.55",
)
replace_once(
    "hsmm_regime.py",
    "        duration_weight = 0.55 if hidden_state == \"S2_TRANSITION\" else 0.35",
    "        duration_weight = 0.50 if hidden_state == \"S2_TRANSITION\" else 0.35",
)
replace_once(
    "hsmm_regime.py",
    "            log_score += math.log(1.0 + 1.35 * transition_evidence)\n        elif hidden_state in {\"S0_PERSISTENT\", \"S1_ALTERNATING\"}:\n            # Stable regimes are faded gradually rather than hard-cut to 0.45x.\n            log_score += math.log(max(0.62, 1.0 - 0.34 * transition_evidence))",
    "            log_score += math.log(1.0 + 0.80 * transition_evidence)\n        elif hidden_state in {\"S0_PERSISTENT\", \"S1_ALTERNATING\"}:\n            # Stable regimes fade gently, so one or two noisy hands cannot force\n            # an abrupt transition posterior.\n            log_score += math.log(max(0.75, 1.0 - 0.22 * transition_evidence))",
)

# pattern_survival.py: soften transition attenuation and eliminate double shoe penalty.
replace_once(
    "pattern_survival.py",
    "SHOE_STAGE_FACTORS = {\n    \"OPENING\": 0.45,\n    \"DEVELOPING\": 0.75,\n    \"MATURE\": 1.00,\n    \"LATE\": 0.80,\n    \"UNKNOWN\": 0.70,\n}",
    "SHOE_STAGE_FACTORS = {\n    \"OPENING\": 0.92,\n    \"DEVELOPING\": 0.97,\n    \"MATURE\": 1.00,\n    \"LATE\": 0.90,\n    \"UNKNOWN\": 0.95,\n}",
)
replace_once(
    "pattern_survival.py",
    "TRANSITION_SHOE_STAGE_FACTORS = {\n    \"OPENING\": 0.82,\n    \"DEVELOPING\": 0.93,\n    \"MATURE\": 1.00,\n    \"LATE\": 0.72,\n    \"UNKNOWN\": 0.84,\n}\n\n# Mild transition attenuation. This replaces the former hard 0.25 multiplier.\nTRANSITION_CHANGE_FACTOR = 0.62\nTRANSITION_CHANGE_FACTOR_MIN = 0.58\nTRANSITION_CHANGE_FACTOR_MAX = 0.70\nTRANSITION_HAZARD_RETAIN = 0.12",
    "TRANSITION_SHOE_STAGE_FACTORS = {\n    \"OPENING\": 0.95,\n    \"DEVELOPING\": 0.98,\n    \"MATURE\": 1.00,\n    \"LATE\": 0.88,\n    \"UNKNOWN\": 0.95,\n}\n\n# Soft transition attenuation. The old 0.25 hard cut is intentionally gone.\nTRANSITION_CHANGE_FACTOR = 0.70\nTRANSITION_CHANGE_FACTOR_MIN = 0.66\nTRANSITION_CHANGE_FACTOR_MAX = 0.78\nTRANSITION_HAZARD_RETAIN = 0.16",
)
replace_once(
    "pattern_survival.py",
    "    stage_application = 1.0 - reliability * (1.0 - stage_anchor)\n    reliability_quality = 0.86 + 0.14 * reliability\n    return _clip(reliability_quality * stage_application, 0.65, 1.0)",
    "    stage_application = 1.0 - reliability * (1.0 - stage_anchor)\n    return _clip(stage_application, 0.88, 1.0)",
)
replace_once(
    "pattern_survival.py",
    "    hazard_pressure_strength = _clip(abs(turn_pressure) * 2.0)\n    hazard_transition_signal = _clip(\n        hazard_pressure_strength * (0.40 + 0.60 * hazard_support)\n    )",
    "    # Only positive turn pressure boosts change confidence. Negative pressure\n    # still remains visible diagnostically but cannot manufacture a turn signal.\n    hazard_pressure_strength = _clip(max(0.0, turn_pressure) * 2.0)\n    hazard_transition_signal = _clip(\n        hazard_pressure_strength * (0.45 + 0.55 * hazard_support)\n    )",
)
replace_once(
    "pattern_survival.py",
    "            0.28\n            + 0.22 * transition_stability\n            + 0.15 * hazard_transition_signal",
    "            0.34\n            + 0.20 * transition_stability\n            + 0.18 * hazard_transition_signal",
)
replace_once(
    "pattern_survival.py",
    "            TRANSITION_CHANGE_FACTOR\n            + 0.05 * transition_stability\n            + 0.03 * hazard_transition_signal\n            - 0.02 * (1.0 if (change_point and pattern_break) else 0.0),",
    "            TRANSITION_CHANGE_FACTOR\n            + 0.04 * transition_stability\n            + 0.04 * hazard_transition_signal\n            - 0.01 * (1.0 if (change_point and pattern_break) else 0.0),",
)
replace_once(
    "pattern_survival.py",
    "    pre_hidden_score = _clip(\n        raw_score\n        * stage_factor\n        * change_factor\n        * transition_shoe_factor\n    )",
    "    # Apply shoe depth once. Transition uses its reliability-gated soft stage\n    # factor instead of multiplying the general stage factor a second time.\n    effective_stage_factor = (\n        transition_shoe_factor if in_transition else stage_factor\n    )\n    pre_hidden_score = _clip(\n        raw_score\n        * effective_stage_factor\n        * change_factor\n    )",
)
replace_once(
    "pattern_survival.py",
    "        \"transition_shoe_factor\": float(transition_shoe_factor),\n        \"transition_confidence_factor\": float(transition_confidence_factor),",
    "        \"transition_shoe_factor\": float(transition_shoe_factor),\n        \"effective_stage_factor\": float(effective_stage_factor),\n        \"transition_confidence_factor\": float(transition_confidence_factor),",
)
replace_once(
    "pattern_survival.py",
    "            \"transition_shoe_factor\": float(transition_shoe_factor),\n            \"hidden_regime_factor\": float(hidden_factor),",
    "            \"transition_shoe_factor\": float(transition_shoe_factor),\n            \"effective_stage_factor\": float(effective_stage_factor),\n            \"hidden_regime_factor\": float(hidden_factor),",
)

print("soft transition v2 patch applied")
