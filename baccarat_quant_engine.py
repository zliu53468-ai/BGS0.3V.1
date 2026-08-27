"""Unified baccarat dynamic prediction and capital-allocation engine.

BaccaratQuantEngine combines:
1) support-aware variable-order B/P/T Markov prior,
2) remaining-card-aware Pattern Survival calibration,
3) tempered Bayesian shoe likelihood,
4) bounded derived-road Markov likelihood,
5) positive-Edge-only bankroll sizing.

Pattern Survival is a calibration score, not a baccarat win probability.  It can
only reduce unsupported historical-pattern influence; it never creates a new
B/P signal from remaining-card depth by itself.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from markov_model import update_and_predict_engine
from money_management import MoneyManagementModel
from pattern_survival import (
    calculate_pattern_survival,
    calibrate_markov_probabilities,
)


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


class BaccaratQuantEngine:
    """Dynamic predictor with pattern calibration + Bayesian likelihood fusion."""

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
                    "B": float(
                        probabilities.get("B", probabilities.get("b", 0.0)) or 0.0
                    ),
                    "P": float(
                        probabilities.get("P", probabilities.get("p", 0.0)) or 0.0
                    ),
                    "T": float(
                        probabilities.get("T", probabilities.get("t", 0.0)) or 0.0
                    ),
                }
                return _normalize(values, ("B", "P", "T")), True
            values = {
                "B": float(
                    probabilities.get("B", probabilities.get("b", 0.0)) or 0.0
                ),
                "P": float(
                    probabilities.get("P", probabilities.get("p", 0.0)) or 0.0
                ),
            }
            return _normalize(values, ("B", "P")), False

        values = list(probabilities)
        if len(values) >= 3:
            return _normalize(
                {
                    "B": float(values[0]),
                    "P": float(values[1]),
                    "T": float(values[2]),
                },
                ("B", "P", "T"),
            ), True
        if len(values) >= 2:
            return _normalize(
                {"B": float(values[0]), "P": float(values[1])},
                ("B", "P"),
            ), False
        return None, False

    @classmethod
    def _apply_tempered_likelihood(
        cls,
        prior: Mapping[str, Any],
        likelihood_input: Mapping[str, Any] | Sequence[float] | None,
        *,
        reliability: float,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Apply one likelihood channel.

        Three-way:
            posterior(y) proportional to prior(y) * L(y)^w

        B/P-only:
            preserve Tie mass and update conditional B/P odds only.
        """
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
                outcome: (
                    normalized_prior[outcome]
                    * max(1e-12, float(likelihood[outcome])) ** w
                )
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
            prior_bp = {
                "B": normalized_prior["B"] / bp_mass,
                "P": normalized_prior["P"] / bp_mass,
            }
            scores = {
                side: prior_bp[side] * max(
                    1e-12,
                    float(likelihood[side]),
                ) ** w
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
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Sequential tempered-Bayes fusion.

        The supplied Markov prior may already be Pattern-Survival calibrated:

            P_final(y) proportional to
                P_M_cal(y) * L_shoe(y)^w_shoe * L_road(y)^w_road

        B/P-only likelihoods preserve Tie mass.  w_road remains deliberately
        small because all derived roads come from the same Big Road history.
        """
        prior = _normalize(
            {
                "B": float(markov_probs.get("B", 0.0) or 0.0),
                "P": float(markov_probs.get("P", 0.0) or 0.0),
                "T": float(markov_probs.get("T", 0.0) or 0.0),
            },
            ("B", "P", "T"),
        )

        after_shoe, shoe_detail = cls._apply_tempered_likelihood(
            prior,
            shoe_probs,
            reliability=shoe_reliability,
        )
        final, road_detail = cls._apply_tempered_likelihood(
            after_shoe,
            road_probs,
            reliability=road_reliability,
        )

        return final, {
            "method": "pattern_calibrated_sequential_tempered_bayes",
            "markov_prior": prior,
            "shoe": shoe_detail,
            "road": road_detail,
            "shoe_reliability": float(shoe_detail.get("reliability", 0.0) or 0.0),
            "road_reliability": float(road_detail.get("reliability", 0.0) or 0.0),
            "likelihood": shoe_detail.get("likelihood"),
            "road_likelihood": road_detail.get("likelihood"),
            "posterior_after_shoe": after_shoe,
            "posterior": final,
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
        markov = update_and_predict_engine(history)
        markov_probs = dict(markov["probabilities"])

        # If no explicit road likelihood is provided, rebuild the canonical
        # derived roads from the same history and activate the bounded
        # derived-road Markov ask-road channel automatically.
        road_analysis: dict[str, Any] = {}
        if road_probs is None:
            try:
                from road_model import build_road_context
                road_analysis = dict(build_road_context(history))
                candidate = road_analysis.get("derived_markov_likelihood")
                if isinstance(candidate, Mapping):
                    road_probs = candidate
                    road_reliability = float(
                        road_analysis.get("derived_markov_reliability", 0.0) or 0.0
                    )
            except Exception:
                # Auxiliary road failure must not break Markov + shoe output.
                road_analysis = {}

        pattern_survival = calculate_pattern_survival(
            markov,
            road_analysis,
            remaining_card_state,
        )
        survival_score = float(pattern_survival.get("score", 0.0) or 0.0)
        calibrated_markov_probs = calibrate_markov_probabilities(
            markov_probs,
            survival_score,
        )

        # Derived-road evidence is deterministic from the same Big Road, so the
        # same survival score can only reduce its already-capped likelihood power.
        raw_road_reliability = _clip(float(road_reliability or 0.0))
        effective_road_reliability = _clip(
            raw_road_reliability * survival_score
        )

        final_probs, fusion = self.bayesian_fuse(
            calibrated_markov_probs,
            shoe_probs,
            shoe_reliability=shoe_reliability,
            road_probs=road_probs,
            road_reliability=effective_road_reliability,
        )

        direction = "B" if final_probs["B"] >= final_probs["P"] else "P"

        # Pattern calibration never increases the original Markov confidence.
        # The 0.35 floor keeps the money layer compatible when shoe evidence is
        # useful even while the historical pattern itself is weak.
        pattern_calibrated_weight = _clip(
            float(markov["final_weight"]) * (0.35 + 0.65 * survival_score)
        )
        money = self.money.allocate(
            direction=direction,
            probabilities=final_probs,
            final_weight=pattern_calibrated_weight,
            bankroll=float(bankroll or 0.0),
        )

        fusion.update({
            "pattern_survival": pattern_survival,
            "raw_markov_prior": dict(markov_probs),
            "pattern_calibrated_markov_prior": dict(calibrated_markov_probs),
            "raw_road_reliability": float(raw_road_reliability),
            "pattern_calibrated_road_reliability": float(
                effective_road_reliability
            ),
        })

        return {
            "direction": direction,
            "direction_text": "莊" if direction == "B" else "閒",
            "markov_probs": markov_probs,
            "pattern_calibrated_markov_probs": calibrated_markov_probs,
            "final_probs": final_probs,
            "final_probability": float(final_probs[direction]),
            "bet_allowed": bool(money["bet_allowed"]),
            "bet_percentage": float(money["bet_percentage"]),
            "bet_amount": float(money["bet_amount"]),
            "edge": float(money["edge"]),
            "edge_percent": float(money["edge_percent"]),
            "money_management": money,
            "markov": markov,
            "pattern_survival": pattern_survival,
            "pattern_calibrated_final_weight": float(pattern_calibrated_weight),
            "remaining_card_state": dict(remaining_card_state or {}),
            "fusion": fusion,
            "road_analysis": road_analysis,
        }


if __name__ == "__main__":
    engine = BaccaratQuantEngine()
    example = engine.predict(
        "BBPBPTBPPB",
        shoe_probs=[0.506, 0.494],
        shoe_reliability=1.0,
        remaining_card_state={
            "available": True,
            "mean_remaining_cards": 365.0,
            "shoe_stage": "DEVELOPING",
            "shoe_stage_factor": 0.75,
            "reliability": 0.55,
        },
        bankroll=10000,
    )
    print({
        "prediction_direction": example["direction"],
        "final_probs": example["final_probs"],
        "pattern_survival": example["pattern_survival"],
        "dynamic_bet_percentage": example["bet_percentage"],
        "edge_percent": example["edge_percent"],
        "bet_allowed": example["bet_allowed"],
    })


__all__ = ["BaccaratQuantEngine"]
