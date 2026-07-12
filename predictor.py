#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}. "
            "Your predictor.py may differ from the supplied version."
        )
    return text.replace(old, new, 1)


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "predictor.py").resolve()
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")

    original = target.read_text(encoding="utf-8")
    text = original

    old = """B_PRIOR = _env_float("B_PRIOR", 0.4586, 0.0001)
P_PRIOR = _env_float("P_PRIOR", 0.4462, 0.0001)
T_PRIOR = _env_float("T_PRIOR", 0.0952, 0.0001)

RANDOM_SEED = _env_int("RANDOM_SEED", 42)
"""
    new = """B_PRIOR = _env_float("B_PRIOR", 0.4586, 0.0001)
P_PRIOR = _env_float("P_PRIOR", 0.4462, 0.0001)
T_PRIOR = _env_float("T_PRIOR", 0.0952, 0.0001)

# Bias controls. These do not replace or disable any model.
FALLBACK_RECENT_WINDOW = _env_int("FALLBACK_RECENT_WINDOW", 36, minimum=6)
FALLBACK_BP_MAX_SHIFT = _env_float(
    "FALLBACK_BP_MAX_SHIFT", 0.025, 0.0, 0.15
)
FUSION_BP_MARGIN_SHRINK = _env_float(
    "FUSION_BP_MARGIN_SHRINK", 0.68, 0.0, 1.0
)
FUSION_MIN_ACTIVE_FOR_SHRINK = _env_int(
    "FUSION_MIN_ACTIVE_FOR_SHRINK", 2, minimum=1
)

RANDOM_SEED = _env_int("RANDOM_SEED", 42)
"""
    text = replace_once(text, old, new, "settings insertion")

    old = """def _empirical_fallback_probs(history: Sequence[str]) -> np.ndarray:
    prior = _prior_probs()
    pseudo = prior * FALLBACK_PRIOR_STRENGTH
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for item in history:
        counts[CLASS_TO_INDEX[item]] += 1.0
    return _normalize(counts + pseudo, fallback=prior)
"""
    new = """def _empirical_fallback_probs(history: Sequence[str]) -> np.ndarray:
    # Stable fallback that cannot become a whole-shoe majority-side model.
    prior = _prior_probs()
    recent = list(history[-FALLBACK_RECENT_WINDOW:])
    pseudo = prior * FALLBACK_PRIOR_STRENGTH

    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for item in recent:
        counts[CLASS_TO_INDEX[item]] += 1.0

    posterior = _normalize(counts + pseudo, fallback=prior)
    tie_prob = float(posterior[CLASS_TO_INDEX["T"]])
    bp_mass = max(1e-12, 1.0 - tie_prob)

    prior_bp_total = max(
        1e-12,
        float(prior[CLASS_TO_INDEX["B"]] + prior[CLASS_TO_INDEX["P"]]),
    )
    prior_b_share = float(prior[CLASS_TO_INDEX["B"]]) / prior_bp_total

    posterior_bp_total = max(
        1e-12,
        float(
            posterior[CLASS_TO_INDEX["B"]]
            + posterior[CLASS_TO_INDEX["P"]]
        ),
    )
    posterior_b_share = (
        float(posterior[CLASS_TO_INDEX["B"]]) / posterior_bp_total
    )

    bounded_b_share = _clamp(
        posterior_b_share,
        prior_b_share - FALLBACK_BP_MAX_SHIFT,
        prior_b_share + FALLBACK_BP_MAX_SHIFT,
    )

    return _normalize(
        [
            bp_mass * bounded_b_share,
            bp_mass * (1.0 - bounded_b_share),
            tie_prob,
        ],
        fallback=prior,
    )
"""
    text = replace_once(text, old, new, "fallback replacement")

    old = """        side, streak_len = _current_streak(window)
        features.extend(
            [
                1.0 if side == "B" else 0.0,
                1.0 if side == "P" else 0.0,
                1.0 if side == "T" else 0.0,
                min(1.0, streak_len / max(1.0, float(window_size))),
            ]
        )
"""
    new = """        side, streak_len = _current_streak(window)
        # Keep feature width unchanged, but remove repeated current B/P identity.
        # Positional one-hot data above already contains the latest side.
        features.extend(
            [
                0.0,
                0.0,
                1.0 if side == "T" else 0.0,
                min(1.0, streak_len / max(1.0, float(window_size))),
            ]
        )
"""
    text = replace_once(text, old, new, "GBM context streak replacement")

    old = """    side, streak_len = _current_streak(recent)
    features.extend(
        [
            min(1.0, len(history) / 100.0),
            1.0 if side == "B" else 0.0,
            1.0 if side == "P" else 0.0,
            1.0 if side == "T" else 0.0,
            min(1.0, streak_len / 10.0),
            stats["runs"] / norm,
            stats["longest_b"] / norm,
            stats["longest_p"] / norm,
            stats["longest_t"] / norm,
            stats["mean_run"] / norm,
"""
    new = """    side, streak_len = _current_streak(recent)
    features.extend(
        [
            min(1.0, len(history) / 100.0),
            0.0,
            0.0,
            1.0 if side == "T" else 0.0,
            min(1.0, streak_len / 10.0),
            stats["runs"] / norm,
            max(stats["longest_b"], stats["longest_p"]) / norm,
            min(stats["longest_b"], stats["longest_p"]) / norm,
            stats["longest_t"] / norm,
            stats["mean_run"] / norm,
"""
    text = replace_once(text, old, new, "GBM global streak replacement")

    anchor = """def _fusion(
    component_probs: Mapping[str, Sequence[float]],
    availability: Mapping[str, bool],
    fallback: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, float]]:
"""
    helper = """def _shrink_correlated_bp_margin(
    values: Sequence[float],
    active_model_count: int,
    fallback: Sequence[float],
) -> np.ndarray:
    # Shrink duplicated B/P confidence from models trained on the same road.
    probs = _normalize(values, fallback=fallback)
    if active_model_count < FUSION_MIN_ACTIVE_FOR_SHRINK:
        return probs

    b_index = CLASS_TO_INDEX["B"]
    p_index = CLASS_TO_INDEX["P"]
    t_index = CLASS_TO_INDEX["T"]

    tie_prob = float(probs[t_index])
    bp_mass = max(1e-12, 1.0 - tie_prob)
    b_share = float(probs[b_index]) / bp_mass
    shrunk_b_share = 0.5 + (b_share - 0.5) * FUSION_BP_MARGIN_SHRINK

    result = probs.copy()
    result[b_index] = bp_mass * shrunk_b_share
    result[p_index] = bp_mass * (1.0 - shrunk_b_share)
    result[t_index] = tie_prob
    return _normalize(result, fallback=fallback)


def _neutral_bp_tie_break(history: Sequence[str]) -> str:
    # Avoid always selecting Banker when B and P are exactly equal.
    non_ties = [item for item in history if item in {"B", "P"}]
    if not non_ties:
        return "B" if len(history) % 2 == 0 else "P"

    digest = hashlib.sha1(
        "".join(non_ties[-24:]).encode("utf-8")
    ).digest()
    return "B" if digest[0] % 2 == 0 else "P"


""" + anchor
    text = replace_once(text, anchor, helper, "fusion helper insertion")

    old = """        final_probs, effective_weights = _fusion(
            component_probs=all_components,
            availability=availability,
            fallback=fallback,
        )
        final_probs = _cap_probability_vector(
            final_probs, FINAL_MAX_PROB, fallback
        )
"""
    new = """        final_probs, effective_weights = _fusion(
            component_probs=all_components,
            availability=availability,
            fallback=fallback,
        )
        active_model_count = sum(
            1
            for name in ("lstm", "gru", "tcn", "gbm", "deepseek")
            if availability.get(name, False)
            and effective_weights.get(name, 0.0) > 0.0
        )
        final_probs = _shrink_correlated_bp_margin(
            final_probs,
            active_model_count=active_model_count,
            fallback=fallback,
        )
        final_probs = _cap_probability_vector(
            final_probs, FINAL_MAX_PROB, fallback
        )
"""
    text = replace_once(text, old, new, "post-fusion shrink insertion")

    old = """        # Compatibility: direction remains B or P; T is displayed only.
        recommend = "B" if b_prob >= p_prob else "P"
        recommend_text = "莊" if recommend == "B" else "閒"
"""
    new = """        # Compatibility: direction remains B or P; T is displayed only.
        # Exact equality no longer defaults permanently to Banker.
        if abs(b_prob - p_prob) <= 1e-12:
            recommend = _neutral_bp_tie_break(cleaned)
        else:
            recommend = "B" if b_prob > p_prob else "P"
        recommend_text = "莊" if recommend == "B" else "閒"
"""
    text = replace_once(text, old, new, "main recommendation tie-break")

    old = """        recommend = "B" if b_prob >= p_prob else "P"
        bp_total = max(1e-12, b_prob + p_prob)
"""
    new = """        if abs(b_prob - p_prob) <= 1e-12:
            recommend = _neutral_bp_tie_break(cleaned)
        else:
            recommend = "B" if b_prob > p_prob else "P"
        bp_total = max(1e-12, b_prob + p_prob)
"""
    text = replace_once(text, old, new, "exception fallback tie-break")

    old = """            "decision_edge": round(edge, 6),
            "signal_level": signal_level,
"""
    new = """            "decision_edge": round(edge, 6),
            "signal_level": signal_level,
            "bias_control": {
                "fallback_recent_window": FALLBACK_RECENT_WINDOW,
                "fallback_bp_max_shift": FALLBACK_BP_MAX_SHIFT,
                "fusion_bp_margin_shrink": FUSION_BP_MARGIN_SHRINK,
                "active_model_count": active_model_count,
                "gbm_directional_streak_identity_removed": True,
                "hard_banker_tie_break_removed": True,
            },
"""
    text = replace_once(text, old, new, "diagnostic insertion")

    backup = target.with_suffix(target.suffix + ".bak")
    shutil.copy2(target, backup)
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8")

    print(f"Patched successfully: {target}")
    print(f"Backup created:      {backup}")
    print("Syntax check:        OK")
    print("")
    print("Optional environment variables:")
    print("FALLBACK_RECENT_WINDOW=36")
    print("FALLBACK_BP_MAX_SHIFT=0.025")
    print("FUSION_BP_MARGIN_SHRINK=0.68")
    print("FUSION_MIN_ACTIVE_FOR_SHRINK=2")


if __name__ == "__main__":
    main()
