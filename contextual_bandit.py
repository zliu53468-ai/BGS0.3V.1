"""Tie-freeze + exploit-only sizing bridge for the Single-Brain LinUCB core.

The stable V5 implementation lives in contextual_bandit_base_v5.py. This bridge
keeps its public API intact while enforcing two invariants:
1) Ties freeze the LinUCB memory clock completely (no decay, no A/b update).
2) UCB scores choose direction, while pure arm means are exposed separately for
   downstream probability calibration and Kelly sizing.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence
import math

import contextual_bandit_base_v5 as _base

# Re-export the stable V5 public surface plus compatibility symbols used by the
# existing regression suite and any external integrations that imported them.
for _name in getattr(_base, "__all__", []):
    globals()[_name] = getattr(_base, _name)

ARMS = _base.ARMS
CONTEXT_DIM = _base.CONTEXT_DIM
CONTEXT_FEATURE_NAMES = _base.CONTEXT_FEATURE_NAMES
ContextSnapshot = _base.ContextSnapshot
ContextGenerator = _base.ContextGenerator
ContextualLinUCB = _base.ContextualLinUCB
ESTIMATED_CARDS_PER_ROUND = _base.ESTIMATED_CARDS_PER_ROUND
SHOE_DECKS = _base.SHOE_DECKS
LINUCB_ALPHA = _base.LINUCB_ALPHA
LINUCB_ARM_ALPHA_MAX_SCALE = _base.LINUCB_ARM_ALPHA_MAX_SCALE
LINUCB_FORGETTING = _base.LINUCB_FORGETTING
LINUCB_RIDGE = _base.LINUCB_RIDGE
LINUCB_SCORE_TIE_EPSILON = _base.LINUCB_SCORE_TIE_EPSILON
LINUCB_UPDATE_WEIGHT = _base.LINUCB_UPDATE_WEIGHT
PROBABILITY_MIN = _base.PROBABILITY_MIN
PROBABILITY_MAX = _base.PROBABILITY_MAX
ROAD_PRIOR_PROBABILITY_SPAN = _base.ROAD_PRIOR_PROBABILITY_SPAN
ROAD_PRIOR_SCORE_WEIGHT = _base.ROAD_PRIOR_SCORE_WEIGHT
STATE_VERSION = _base.STATE_VERSION
make_scope_key = _base.make_scope_key

# Compatibility hook aliases. Regression tests patch these names on this module;
# _sync_base_hooks propagates those patches into the stable implementation module.
build_standard_derived_roads = _base.build_standard_derived_roads
analyze_run_length_hazard = _base.analyze_run_length_hazard
analyze_hidden_regime = _base.analyze_hidden_regime
update_and_predict_engine = _base.update_and_predict_engine
estimate_probabilistic_shoe = _base.estimate_probabilistic_shoe
fresh_counts = _base.fresh_counts
analyze_shoe_composition = _base.analyze_shoe_composition
_read_state = _base._read_state
_write_state = _base._write_state

_ORIGINAL_CONTEXT_BUILD = ContextGenerator.build
_ORIGINAL_UPDATE_SCOPE = ContextualLinUCB._update_scope
_ORIGINAL_APPLY_PENDING = ContextualLinUCB._apply_pending
_ORIGINAL_PREDICT = ContextualLinUCB.predict


def _sync_base_hooks() -> None:
    for name in (
        "build_standard_derived_roads",
        "analyze_run_length_hazard",
        "analyze_hidden_regime",
        "update_and_predict_engine",
        "estimate_probabilistic_shoe",
        "fresh_counts",
        "analyze_shoe_composition",
        "_read_state",
        "_write_state",
    ):
        setattr(_base, name, globals()[name])


def _context_build_synced(self, history, shoe_context=None):
    _sync_base_hooks()
    return _ORIGINAL_CONTEXT_BUILD(self, history, shoe_context)


def _tie_freeze_result(*, action: str, actual_outcome: str, reason: str = "TIE_FREEZE_BRAIN") -> dict[str, Any]:
    return {
        "updated": True,
        "action": str(action or "").upper().strip(),
        "actual_outcome": str(actual_outcome or "T").upper().strip(),
        "reward": 0.0,
        "directional_sample_applied": False,
        "memory_decay_applied": False,
        "forgetting": 1.0,
        "reason": reason,
        "formal_model": "contextual_linucb",
        "diagnostic_only": False,
    }


def _update_scope_tie_freeze(
    self: ContextualLinUCB,
    scope: dict[str, Any],
    *,
    action: str,
    context_vector: Sequence[float],
    actual_outcome: str,
) -> dict[str, Any]:
    actual = str(actual_outcome or "").upper().strip()
    if actual == "T":
        # Hard exit before the V5 implementation can call _decay_arms().
        return _tie_freeze_result(action=action, actual_outcome=actual)
    return _ORIGINAL_UPDATE_SCOPE(
        self,
        scope,
        action=action,
        context_vector=deepcopy(list(context_vector)),
        actual_outcome=actual,
    )


def _apply_pending_tie_freeze(
    self: ContextualLinUCB,
    scope: dict[str, Any],
    raw_history: Sequence[str],
) -> dict[str, Any]:
    pending = deepcopy(dict(scope.get("pending") or {}))
    if not pending:
        return {"updated": False, "reason": "no_pending_prediction"}
    previous_len = int(pending.get("raw_round_count", 0) or 0)
    if len(raw_history) <= previous_len:
        return {"updated": False, "reason": "no_new_resolved_round"}

    if _base._history_fingerprint(list(raw_history[:previous_len])) != str(
        pending.get("history_fingerprint") or ""
    ):
        scope["pending"] = {}
        return {
            "updated": False,
            "reason": "history_reset_or_misaligned",
            "previous_len": previous_len,
            "current_len": len(raw_history),
        }

    resolved_outcome = str(raw_history[previous_len] or "").upper().strip()
    if resolved_outcome == "T":
        scope["pending"] = {}
        result = _tie_freeze_result(
            action=str(pending.get("action") or ""),
            actual_outcome=resolved_outcome,
        )
        result.update(
            {
                "history_aligned": True,
                "resolved_history_index": previous_len,
                "history_rounds_after_append": len(raw_history),
            }
        )
        return result

    return _ORIGINAL_APPLY_PENDING(self, scope, raw_history)


def _softmax_two(means: Mapping[str, Any]) -> dict[str, float]:
    mean_b = float(means.get("B", 0.0) or 0.0)
    mean_p = float(means.get("P", 0.0) or 0.0)
    shift = max(mean_b, mean_p)
    exp_b = math.exp(max(-40.0, min(40.0, mean_b - shift)))
    exp_p = math.exp(max(-40.0, min(40.0, mean_p - shift)))
    total = exp_b + exp_p
    if total <= 1e-12:
        return {"B": 0.5, "P": 0.5, "T": 0.0}
    return {"B": exp_b / total, "P": exp_p / total, "T": 0.0}


def _predict_with_exploit_means(
    self: ContextualLinUCB,
    *,
    history,
    shoe_context,
    scope_key: str,
) -> dict[str, Any]:
    _sync_base_hooks()
    result = _ORIGINAL_PREDICT(
        self,
        history=deepcopy(history),
        shoe_context=deepcopy(dict(shoe_context or {})),
        scope_key=str(scope_key or ""),
    )
    scores = dict(result.get("scores") or {})
    mean_scores = {
        "B": float((scores.get("B") or {}).get("mean", 0.0) or 0.0),
        "P": float((scores.get("P") or {}).get("mean", 0.0) or 0.0),
    }
    exploit_probabilities = _softmax_two(mean_scores)
    result["mean_scores"] = mean_scores
    result["exploit_probabilities"] = exploit_probabilities
    result["direction_probabilities_ucb"] = deepcopy(
        dict(result.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0})
    )
    result["direction_score_semantics"] = "ucb_mean_plus_exploration_for_argmax_only"
    result["sizing_probability_semantics"] = "softmax_of_exploit_only_mean_scores"
    result["tie_memory_semantics"] = "TIE_FREEZE_BRAIN_no_decay_no_A_b_update"
    return result


# Patch the stable implementation in its defining module. Existing V5 functions
# resolve these methods dynamically, so the public API remains unchanged.
ContextGenerator.build = _context_build_synced
ContextualLinUCB._update_scope = _update_scope_tie_freeze
ContextualLinUCB._apply_pending = _apply_pending_tie_freeze
ContextualLinUCB.predict = _predict_with_exploit_means

predict_bandit = _base.predict_bandit
update_bandit = _base.update_bandit

__all__ = list(dict.fromkeys(list(getattr(_base, "__all__", [])) + ["ContextSnapshot"]))
