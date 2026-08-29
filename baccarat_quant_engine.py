"""Unified baccarat dynamic prediction and capital-allocation engine.

BaccaratQuantEngine combines:
1) support-aware variable-order B/P/T Markov evidence,
2) remaining-card-aware Pattern Survival calibration,
3) physical-baseline-corrected probabilistic shoe evidence,
4) bounded derived-road Markov likelihood,
5) bounded run-length structural turn-hazard likelihood,
6) positive-Edge-only bankroll sizing.

Direction and bankroll math deliberately use two related posteriors:
- direction posterior: Banker-neutral B/P baseline, used to choose B vs P;
- economic posterior: baccarat physical prior + the same evidence, used for Edge.

This version also applies conservative evidence controls before fusion:
- nested Markov orders that merely re-express the same active run are treated as
  correlated evidence rather than independent confirmation;
- a single latest hand may reduce confidence when it causes a large Markov swing,
  but it is never mechanically reversed or ignored;
- run-length Hazard is downweighted when it duplicates the same current-run
  continuation direction already carried by Markov.

These guards reduce evidence duplication. They do not create an anti-streak rule
and do not imply deterministic baccarat patterns or guaranteed outcomes.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from bankroll_display_bridge import install_legacy_app_bankroll_adapter
from markov_model import update_and_predict_engine
from money_management import MoneyManagementModel
from pattern_survival import (
    PHYSICAL_PRIOR,
    calculate_pattern_survival,
    calibrate_markov_probabilities,
)
from run_length_hazard import analyze_run_length_hazard

_DIRECTION_EPSILON = 1e-12
_LAST_HAND_MIN_HISTORY = 6
_LAST_HAND_SHIFT_THRESHOLD = 0.10
_LAST_HAND_MAX_PENALTY = 0.35
_SAME_RUN_PER_NESTED_ORDER_PENALTY = 0.10
_SAME_RUN_MARKOV_FACTOR_FLOOR = 0.70
_HAZARD_CURRENT_RUN_CORRELATION_FACTOR = 0.55
_HAZARD_LAST_HAND_SENSITIVITY_FACTOR = 0.75


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize(
    values: Mapping[str, float],
    keys: Sequence[str],
) -> dict[str, float]:
    raw = {
        key: max(1e-12, float(values.get(key, 0.0) or 0.0))
        for key in keys
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: raw[key] / total for key in keys}


def _clean_history_values(history: str | Iterable[Any]) -> list[str]:
    values: Iterable[Any]
    if isinstance(history, str):
        values = list(history)
    else:
        values = history

    result: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = (
                item.get("outcome")
                or item.get("actual")
                or item.get("actual_outcome")
                or item.get("virtual_outcome")
            )
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-2000:]


def _bp_conditional_margin(probabilities: Mapping[str, Any]) -> float:
    b = max(0.0, float(probabilities.get("B", 0.0) or 0.0))
    p = max(0.0, float(probabilities.get("P", 0.0) or 0.0))
    mass = b + p
    if mass <= 1e-12:
        return 0.0
    return float((b - p) / mass)


def _bp_direction(probabilities: Mapping[str, Any]) -> str:
    return "B" if _bp_conditional_margin(probabilities) >= 0.0 else "P"


def _shrink_bp_toward_neutral(
    probabilities: Mapping[str, Any],
    factor: float,
) -> dict[str, float]:
    """Shrink only B/P direction strength while preserving the Tie mass."""
    normalized = _normalize(
        {
            "B": float(probabilities.get("B", 0.0) or 0.0),
            "P": float(probabilities.get("P", 0.0) or 0.0),
            "T": float(probabilities.get("T", 0.0) or 0.0),
        },
        ("B", "P", "T"),
    )
    bp_mass = normalized["B"] + normalized["P"]
    if bp_mass <= 1e-12:
        return normalized

    strength = _clip(factor)
    current_b = normalized["B"] / bp_mass
    guarded_b = 0.5 + (current_b - 0.5) * strength
    guarded = {
        "B": bp_mass * guarded_b,
        "P": bp_mass * (1.0 - guarded_b),
        "T": normalized["T"],
    }
    return _normalize(guarded, ("B", "P", "T"))


def _markov_evidence_guard(
    history: Sequence[str],
    markov: Mapping[str, Any],
) -> dict[str, Any]:
    """Return multiplicative evidence factors without choosing B or P.

    Same-run factor:
      O1..O4 are nested contexts. During a run such as BBBB, O1=B, O2=BB,
      O3=BBB and O4=BBBB are correlated descriptions of the same active run.
      We therefore reduce directional strength as more nested same-run orders
      become simultaneously available. This is symmetric for Banker and Player.

    Last-hand factor:
      Compare the current Markov posterior with the posterior before the latest
      observed hand. If one additional hand creates an unusually large B/P swing
      without an explicit structural change-point, reduce only the swing strength.
      The latest hand remains included and no opposite-side rule is introduced.
    """
    state = dict(markov.get("state") or {})
    run_length = max(0, int(state.get("current_run_length", 0) or 0))
    order_weights = dict(markov.get("order_weights") or {})
    positive_orders = sum(
        1
        for order in range(1, 5)
        if float(order_weights.get(f"order_{order}", 0.0) or 0.0) > 1e-12
    )
    same_run_nested_orders = min(run_length, positive_orders)
    same_run_factor = 1.0
    if same_run_nested_orders >= 2:
        same_run_factor = max(
            _SAME_RUN_MARKOV_FACTOR_FLOOR,
            1.0
            - _SAME_RUN_PER_NESTED_ORDER_PENALTY
            * (same_run_nested_orders - 1),
        )

    current_probs = dict(markov.get("probabilities") or {})
    current_margin = _bp_conditional_margin(current_probs)
    previous_margin = current_margin
    shift = 0.0
    direction_flip = False
    last_hand_factor = 1.0
    previous_available = False

    regime_profile = dict(markov.get("regime_profile") or {})
    structural_break = bool(
        regime_profile.get("change_point")
        or regime_profile.get("pattern_break")
    )

    if len(history) >= _LAST_HAND_MIN_HISTORY:
        previous_markov = update_and_predict_engine(history[:-1])
        previous_probs = dict(previous_markov.get("probabilities") or {})
        previous_margin = _bp_conditional_margin(previous_probs)
        shift = abs(current_margin - previous_margin)
        direction_flip = bool(
            current_margin * previous_margin < 0.0
            and abs(current_margin) > 1e-9
            and abs(previous_margin) > 1e-9
        )
        previous_available = True

        if not structural_break and shift > _LAST_HAND_SHIFT_THRESHOLD:
            excess = (shift - _LAST_HAND_SHIFT_THRESHOLD) / max(
                1e-9,
                0.40 - _LAST_HAND_SHIFT_THRESHOLD,
            )
            penalty = min(_LAST_HAND_MAX_PENALTY, max(0.0, excess) * _LAST_HAND_MAX_PENALTY)
            if direction_flip:
                penalty = max(penalty, 0.22)
            last_hand_factor = max(1.0 - _LAST_HAND_MAX_PENALTY, 1.0 - penalty)

    combined_factor = _clip(same_run_factor * last_hand_factor, 0.50, 1.0)
    return {
        "combined_factor": float(combined_factor),
        "same_run_factor": float(same_run_factor),
        "same_run_nested_orders": int(same_run_nested_orders),
        "current_run_length": int(run_length),
        "positive_order_count": int(positive_orders),
        "last_hand_factor": float(last_hand_factor),
        "last_hand_guard_applied": bool(last_hand_factor < 0.999999),
        "previous_available": bool(previous_available),
        "current_bp_margin": float(current_margin),
        "previous_bp_margin": float(previous_margin),
        "last_hand_margin_shift": float(shift),
        "last_hand_direction_flip": bool(direction_flip),
        "structural_break_exemption": bool(structural_break),
        "semantics": (
            "confidence_only_correlation_and_last_hand_sensitivity_guard_"
            "never_forces_opposite_direction"
        ),
    }


class BaccaratQuantEngine:
    """Dynamic predictor with neutral direction selection and physical EV math."""

    def __init__(self) -> None:
        self.money = MoneyManagementModel()

    @staticmethod
    def _coerce_probs(
        probabilities: Mapping[str, Any] | Sequence[float] | None,
    ) -> tuple[dict[str, float] | None, bool]:
        if probabilities is None:
            return None, False

        if isinstance(probabilities, Mapping):
            has_tie = "T" in probabilities or "t" in probabilities
            if has_tie:
                values = {
                    "B": float(probabilities.get("B", probabilities.get("b", 0.0)) or 0.0),
                    "P": float(probabilities.get("P", probabilities.get("p", 0.0)) or 0.0),
                    "T": float(probabilities.get("T", probabilities.get("t", 0.0)) or 0.0),
                }
                return _normalize(values, ("B", "P", "T")), True
            values = {
                "B": float(probabilities.get("B", probabilities.get("b", 0.0)) or 0.0),
                "P": float(probabilities.get("P", probabilities.get("p", 0.0)) or 0.0),
            }
            return _normalize(values, ("B", "P")), False

        values = list(probabilities)
        if len(values) >= 3:
            return _normalize(
                {"B": float(values[0]), "P": float(values[1]), "T": float(values[2])},
                ("B", "P", "T"),
            ), True
        if len(values) >= 2:
            return _normalize({"B": float(values[0]), "P": float(values[1])}, ("B", "P")), False
        return None, False

    @classmethod
    def _physical_evidence_ratio(
        cls,
        probabilities: Mapping[str, Any] | Sequence[float] | None,
    ) -> tuple[dict[str, float] | None, bool]:
        """Convert a model posterior into evidence relative to baccarat baseline."""
        probs, has_tie = cls._coerce_probs(probabilities)
        if probs is None:
            return None, False

        if has_tie:
            return {
                outcome: max(1e-12, float(probs[outcome]) / max(1e-12, PHYSICAL_PRIOR[outcome]))
                for outcome in ("B", "P", "T")
            }, True

        physical_bp_mass = PHYSICAL_PRIOR["B"] + PHYSICAL_PRIOR["P"]
        physical_bp = {
            "B": PHYSICAL_PRIOR["B"] / physical_bp_mass,
            "P": PHYSICAL_PRIOR["P"] / physical_bp_mass,
        }
        return {
            side: max(1e-12, float(probs[side]) / max(1e-12, physical_bp[side]))
            for side in ("B", "P")
        }, False

    @classmethod
    def _apply_tempered_likelihood(
        cls,
        prior: Mapping[str, Any],
        likelihood_input: Mapping[str, Any] | Sequence[float] | None,
        *,
        reliability: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        normalized_prior = _normalize(
            {
                "B": float(prior.get("B", 0.0) or 0.0),
                "P": float(prior.get("P", 0.0) or 0.0),
                "T": float(prior.get("T", 0.0) or 0.0),
            },
            ("B", "P", "T"),
        )
        likelihood, has_tie = cls._coerce_probs(likelihood_input)
        w = _clip(reliability)

        if likelihood is None or w <= 0.0:
            return normalized_prior, {
                "applied": False,
                "reliability": float(w if likelihood is not None else 0.0),
                "likelihood": likelihood,
                "has_tie": bool(has_tie),
            }

        if has_tie:
            scores = {
                outcome: normalized_prior[outcome] * max(1e-12, float(likelihood[outcome])) ** w
                for outcome in ("B", "P", "T")
            }
            posterior = _normalize(scores, ("B", "P", "T"))
            method = "tempered_bayes_threeway"
        else:
            bp_mass = normalized_prior["B"] + normalized_prior["P"]
            if bp_mass <= 1e-12:
                return normalized_prior, {
                    "applied": False,
                    "reliability": float(w),
                    "likelihood": likelihood,
                    "has_tie": False,
                    "reason": "no_resolved_bp_mass",
                }
            prior_bp = {"B": normalized_prior["B"] / bp_mass, "P": normalized_prior["P"] / bp_mass}
            scores = {
                side: prior_bp[side] * max(1e-12, float(likelihood[side])) ** w
                for side in ("B", "P")
            }
            posterior_bp = _normalize(scores, ("B", "P"))
            posterior = {
                "B": bp_mass * posterior_bp["B"],
                "P": bp_mass * posterior_bp["P"],
                "T": normalized_prior["T"],
            }
            posterior = _normalize(posterior, ("B", "P", "T"))
            method = "tempered_bayes_bp_conditional_preserve_tie"

        return posterior, {
            "applied": True,
            "reliability": float(w),
            "likelihood": likelihood,
            "has_tie": bool(has_tie),
            "method": method,
            "prior": normalized_prior,
            "posterior": posterior,
        }

    @classmethod
    def bayesian_fuse(
        cls,
        markov_probs: Mapping[str, Any],
        shoe_probs: Mapping[str, Any] | Sequence[float] | None,
        *,
        shoe_reliability: float = 1.0,
        road_probs: Mapping[str, Any] | Sequence[float] | None = None,
        road_reliability: float = 0.0,
        hazard_probs: Mapping[str, Any] | Sequence[float] | None = None,
        hazard_reliability: float = 0.0,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        prior = _normalize(
            {
                "B": float(markov_probs.get("B", 0.0) or 0.0),
                "P": float(markov_probs.get("P", 0.0) or 0.0),
                "T": float(markov_probs.get("T", 0.0) or 0.0),
            },
            ("B", "P", "T"),
        )

        shoe_evidence, shoe_has_tie = cls._physical_evidence_ratio(shoe_probs)
        after_shoe, shoe_detail = cls._apply_tempered_likelihood(
            prior,
            shoe_evidence,
            reliability=shoe_reliability,
        )
        shoe_detail["raw_probabilities"] = shoe_probs
        shoe_detail["physical_baseline_corrected"] = bool(shoe_evidence is not None)
        shoe_detail["raw_has_tie"] = bool(shoe_has_tie)

        after_road, road_detail = cls._apply_tempered_likelihood(
            after_shoe,
            road_probs,
            reliability=road_reliability,
        )
        final, hazard_detail = cls._apply_tempered_likelihood(
            after_road,
            hazard_probs,
            reliability=hazard_reliability,
        )

        return final, {
            "method": "banker_neutral_direction_tempered_bayes_with_physical_ev_split",
            "markov_prior": prior,
            "shoe": shoe_detail,
            "road": road_detail,
            "hazard": hazard_detail,
            "shoe_reliability": float(shoe_detail.get("reliability", 0.0) or 0.0),
            "road_reliability": float(road_detail.get("reliability", 0.0) or 0.0),
            "hazard_reliability": float(hazard_detail.get("reliability", 0.0) or 0.0),
            "likelihood": shoe_detail.get("likelihood"),
            "raw_shoe_probabilities": shoe_probs,
            "road_likelihood": road_detail.get("likelihood"),
            "hazard_likelihood": hazard_detail.get("likelihood"),
            "posterior_after_shoe": after_shoe,
            "posterior_after_road": after_road,
            "posterior": final,
        }

    @classmethod
    def _select_direction(
        cls,
        final_probs: Mapping[str, Any],
        *,
        calibrated_markov_probs: Mapping[str, Any],
        hazard_probs: Mapping[str, Any] | None,
        road_probs: Mapping[str, Any] | Sequence[float] | None,
        shoe_probs: Mapping[str, Any] | Sequence[float] | None,
    ) -> tuple[str, dict[str, Any]]:
        margin = float(final_probs.get("B", 0.0) or 0.0) - float(final_probs.get("P", 0.0) or 0.0)
        if margin > _DIRECTION_EPSILON:
            return "B", {"source": "final_direction_posterior", "margin": margin}
        if margin < -_DIRECTION_EPSILON:
            return "P", {"source": "final_direction_posterior", "margin": margin}

        candidates: list[tuple[str, Any]] = [
            ("run_length_hazard", hazard_probs),
            ("derived_road", road_probs),
            ("markov_direction_signal", calibrated_markov_probs),
        ]
        shoe_evidence, _ = cls._physical_evidence_ratio(shoe_probs)
        candidates.append(("shoe_excess_evidence", shoe_evidence))

        for source, candidate in candidates:
            probs, _ = cls._coerce_probs(candidate)
            if not probs or "B" not in probs or "P" not in probs:
                continue
            local_margin = float(probs["B"]) - float(probs["P"])
            if local_margin > _DIRECTION_EPSILON:
                return "B", {"source": source, "margin": margin, "tie_break_margin": local_margin}
            if local_margin < -_DIRECTION_EPSILON:
                return "P", {"source": source, "margin": margin, "tie_break_margin": local_margin}

        return "B", {
            "source": "unresolved_exact_tie_compatibility",
            "margin": margin,
            "unresolved": True,
        }

    def predict(
        self,
        history: str | Iterable[Any],
        *,
        shoe_probs: Mapping[str, Any] | Sequence[float] | None = None,
        shoe_reliability: float = 1.0,
        road_probs: Mapping[str, Any] | Sequence[float] | None = None,
        road_reliability: float = 0.0,
        remaining_card_state: Mapping[str, Any] | None = None,
        bankroll: float = 0.0,
    ) -> dict[str, Any]:
        install_legacy_app_bankroll_adapter()

        history_values = _clean_history_values(history)
        markov = update_and_predict_engine(history_values)
        markov_probs = dict(markov["probabilities"])
        evidence_guard = _markov_evidence_guard(history_values, markov)
        guarded_raw_markov_probs = _shrink_bp_toward_neutral(
            markov_probs,
            float(evidence_guard["combined_factor"]),
        )

        road_analysis: dict[str, Any] = {}
        if road_probs is None:
            try:
                from road_model import build_road_context
                road_analysis = dict(build_road_context(history_values))
                candidate = road_analysis.get("derived_markov_likelihood")
                if isinstance(candidate, Mapping):
                    road_probs = candidate
                    road_reliability = float(road_analysis.get("derived_markov_reliability", 0.0) or 0.0)
            except Exception:
                road_analysis = {}

        pattern_survival = calculate_pattern_survival(markov, road_analysis, remaining_card_state)
        survival_score = float(pattern_survival.get("score", 0.0) or 0.0)
        calibrated_markov_probs = calibrate_markov_probabilities(
            guarded_raw_markov_probs,
            survival_score,
        )

        raw_road_reliability = _clip(float(road_reliability or 0.0))
        effective_road_reliability = _clip(raw_road_reliability * survival_score)

        hazard_analysis = analyze_run_length_hazard(history_values)
        hazard_probs = dict(hazard_analysis.get("likelihood") or {})
        raw_hazard_reliability = _clip(float(hazard_analysis.get("reliability", 0.0) or 0.0))
        remaining_state = dict(remaining_card_state or {})
        shoe_stage_factor = _clip(float(remaining_state.get("shoe_stage_factor", 0.70) or 0.70))

        hazard_correlation_factor = 1.0
        current_side = str(hazard_analysis.get("current_side") or "").upper().strip()
        current_run_length = int(hazard_analysis.get("current_run_length", 0) or 0)
        hazard_direction = _bp_direction(hazard_probs) if hazard_probs else ""
        markov_direction = _bp_direction(calibrated_markov_probs)
        duplicate_current_run_direction = bool(
            current_run_length >= 2
            and current_side in {"B", "P"}
            and hazard_direction == current_side
            and markov_direction == current_side
        )
        if duplicate_current_run_direction:
            hazard_correlation_factor *= _HAZARD_CURRENT_RUN_CORRELATION_FACTOR
        if bool(evidence_guard.get("last_hand_guard_applied")):
            hazard_correlation_factor *= _HAZARD_LAST_HAND_SENSITIVITY_FACTOR
        hazard_correlation_factor = _clip(hazard_correlation_factor, 0.25, 1.0)

        effective_hazard_reliability = _clip(
            raw_hazard_reliability
            * shoe_stage_factor
            * hazard_correlation_factor
        )

        direction_probs, fusion = self.bayesian_fuse(
            calibrated_markov_probs,
            shoe_probs,
            shoe_reliability=shoe_reliability,
            road_probs=road_probs,
            road_reliability=effective_road_reliability,
            hazard_probs=hazard_probs,
            hazard_reliability=effective_hazard_reliability,
        )

        direction, direction_detail = self._select_direction(
            direction_probs,
            calibrated_markov_probs=calibrated_markov_probs,
            hazard_probs=hazard_probs,
            road_probs=road_probs,
            shoe_probs=shoe_probs,
        )

        markov_evidence, _ = self._physical_evidence_ratio(guarded_raw_markov_probs)
        economic_markov_prior, economic_markov_detail = self._apply_tempered_likelihood(
            PHYSICAL_PRIOR,
            markov_evidence,
            reliability=survival_score,
        )
        economic_probs, economic_fusion = self.bayesian_fuse(
            economic_markov_prior,
            shoe_probs,
            shoe_reliability=shoe_reliability,
            road_probs=road_probs,
            road_reliability=effective_road_reliability,
            hazard_probs=hazard_probs,
            hazard_reliability=effective_hazard_reliability,
        )

        pattern_calibrated_weight = _clip(
            float(markov["final_weight"])
            * float(evidence_guard["combined_factor"])
            * (0.35 + 0.65 * survival_score)
        )
        money = self.money.allocate(
            direction=direction,
            probabilities=economic_probs,
            final_weight=pattern_calibrated_weight,
            bankroll=float(bankroll or 0.0),
        )

        if bool(direction_detail.get("unresolved")):
            money = dict(money)
            money.update({
                "bet_allowed": False,
                "bet_percentage": 0.0,
                "bet_amount": 0.0,
                "final_bet_ratio": 0.0,
                "adjusted_ratio": 0.0,
                "reason": "unresolved_exact_direction_tie_no_bet",
            })

        fusion.update({
            "pattern_survival": pattern_survival,
            "raw_markov_prior": dict(markov_probs),
            "guarded_raw_markov_prior": dict(guarded_raw_markov_probs),
            "markov_evidence_guard": dict(evidence_guard),
            "pattern_calibrated_markov_prior": dict(calibrated_markov_probs),
            "raw_road_reliability": float(raw_road_reliability),
            "pattern_calibrated_road_reliability": float(effective_road_reliability),
            "run_length_hazard": hazard_analysis,
            "raw_hazard_reliability": float(raw_hazard_reliability),
            "hazard_correlation_factor": float(hazard_correlation_factor),
            "duplicate_current_run_direction": bool(duplicate_current_run_direction),
            "shoe_stage_calibrated_hazard_reliability": float(effective_hazard_reliability),
            "direction_selection": direction_detail,
            "economic_markov_evidence": economic_markov_detail,
            "economic_posterior": economic_probs,
            "economic_fusion": economic_fusion,
            "direction_semantics": "banker_neutral_bp_direction_posterior_with_correlated_evidence_guards",
            "economic_semantics": "physical_baccarat_prior_plus_guarded_excess_model_evidence_for_edge_only",
        })

        return {
            "direction": direction,
            "direction_text": "莊" if direction == "B" else "閒",
            "direction_margin": float(direction_probs["B"] - direction_probs["P"]),
            "direction_selection": direction_detail,
            "markov_probs": markov_probs,
            "guarded_markov_probs": guarded_raw_markov_probs,
            "markov_evidence_guard": evidence_guard,
            "pattern_calibrated_markov_probs": calibrated_markov_probs,
            "final_probs": direction_probs,
            "direction_probs": direction_probs,
            "economic_probs": economic_probs,
            "final_probability": float(direction_probs[direction]),
            "economic_probability_for_direction": float(economic_probs[direction]),
            "bet_allowed": bool(money["bet_allowed"]),
            "bet_percentage": float(money["bet_percentage"]),
            "bet_amount": float(money["bet_amount"]),
            "suggested_bet_amount": float(money["bet_amount"]),
            "edge": float(money["edge"]),
            "edge_percent": float(money["edge_percent"]),
            "money_management": money,
            "markov": markov,
            "pattern_survival": pattern_survival,
            "pattern_calibrated_final_weight": float(pattern_calibrated_weight),
            "remaining_card_state": remaining_state,
            "run_length_hazard": hazard_analysis,
            "hazard_likelihood": hazard_probs,
            "hazard_reliability": float(effective_hazard_reliability),
            "hazard_correlation_factor": float(hazard_correlation_factor),
            "fusion": fusion,
            "road_analysis": road_analysis,
        }


if __name__ == "__main__":
    engine = BaccaratQuantEngine()
    example = engine.predict(
        "BPPPBBPPBBBPPPPBPPB",
        shoe_probs=[0.506, 0.494],
        shoe_reliability=0.25,
        remaining_card_state={
            "available": True,
            "mean_remaining_cards": 300.0,
            "shoe_stage": "DEVELOPING",
            "shoe_stage_factor": 0.75,
            "reliability": 0.55,
        },
        bankroll=10000,
    )
    print({
        "prediction_direction": example["direction"],
        "direction_probs": example["direction_probs"],
        "economic_probs": example["economic_probs"],
        "run_length_hazard": example["run_length_hazard"],
        "markov_evidence_guard": example["markov_evidence_guard"],
        "pattern_survival": example["pattern_survival"],
        "dynamic_bet_percentage": example["bet_percentage"],
        "suggested_bet_amount": example["suggested_bet_amount"],
        "edge_percent": example["edge_percent"],
        "bet_allowed": example["bet_allowed"],
    })


__all__ = ["BaccaratQuantEngine"]
