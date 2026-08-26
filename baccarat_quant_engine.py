"""Unified baccarat dynamic prediction and capital-allocation engine.

The public BaccaratQuantEngine accepts a B/P/T history plus optional shoe
probabilities. It combines:
1) support-aware variable-order Markov prior,
2) Bayesian/tempered shoe likelihood,
3) Edge-gated dynamic bankroll sizing.

The class is intentionally dependency-light and preserves B/P/T probabilities.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from markov_model import update_and_predict_engine
from money_management import MoneyManagementModel


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalize(values: Mapping[str, float], keys: Sequence[str]) -> dict[str, float]:
    raw = {key: max(1e-12, float(values.get(key, 0.0) or 0.0)) for key in keys}
    total = sum(raw.values())
    if total <= 1e-12:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: raw[key] / total for key in keys}


class BaccaratQuantEngine:
    """Baccarat dynamic predictor with Bayesian shoe fusion and Edge sizing."""

    def __init__(self) -> None:
        self.money = MoneyManagementModel()

    @staticmethod
    def _coerce_shoe_probs(
        shoe_probs: Mapping[str, Any] | Sequence[float] | None,
    ) -> tuple[dict[str, float] | None, bool]:
        if shoe_probs is None:
            return None, False

        if isinstance(shoe_probs, Mapping):
            has_tie = "T" in shoe_probs or "t" in shoe_probs
            if has_tie:
                values = {
                    "B": float(shoe_probs.get("B", shoe_probs.get("b", 0.0)) or 0.0),
                    "P": float(shoe_probs.get("P", shoe_probs.get("p", 0.0)) or 0.0),
                    "T": float(shoe_probs.get("T", shoe_probs.get("t", 0.0)) or 0.0),
                }
                return _normalize(values, ("B", "P", "T")), True
            values = {
                "B": float(shoe_probs.get("B", shoe_probs.get("b", 0.0)) or 0.0),
                "P": float(shoe_probs.get("P", shoe_probs.get("p", 0.0)) or 0.0),
            }
            return _normalize(values, ("B", "P")), False

        values = list(shoe_probs)
        if len(values) >= 3:
            return _normalize(
                {"B": float(values[0]), "P": float(values[1]), "T": float(values[2])},
                ("B", "P", "T"),
            ), True
        if len(values) >= 2:
            return _normalize(
                {"B": float(values[0]), "P": float(values[1])},
                ("B", "P"),
            ), False
        return None, False

    @classmethod
    def bayesian_fuse(
        cls,
        markov_probs: Mapping[str, Any],
        shoe_probs: Mapping[str, Any] | Sequence[float] | None,
        *,
        shoe_reliability: float = 1.0,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Bayesian fusion.

        Full 3-way likelihood:
            P(y|H,S) proportional to P_M(y|H) * L_S(y)^w

        If shoe_probs only contains [B,P], it is interpreted as likelihood
        conditional on a non-tie result. The Markov tie mass is preserved and
        Bayes is applied only to the resolved B/P odds:

            P(B|resolved,H,S) proportional to
                P_M(B|resolved,H) * L_S(B)^w

        w in [0,1] tempers weak inferred-shoe evidence.
        """
        prior = _normalize(
            {
                "B": float(markov_probs.get("B", 0.0) or 0.0),
                "P": float(markov_probs.get("P", 0.0) or 0.0),
                "T": float(markov_probs.get("T", 0.0) or 0.0),
            },
            ("B", "P", "T"),
        )
        likelihood, has_tie = cls._coerce_shoe_probs(shoe_probs)
        if likelihood is None:
            return prior, {
                "method": "markov_only_no_shoe_likelihood",
                "shoe_reliability": 0.0,
                "likelihood": None,
            }

        w = _clip(shoe_reliability)
        if w <= 0.0:
            return prior, {
                "method": "markov_only_zero_shoe_reliability",
                "shoe_reliability": 0.0,
                "likelihood": likelihood,
            }

        if has_tie:
            scores = {
                outcome: prior[outcome] * (max(1e-12, likelihood[outcome]) ** w)
                for outcome in ("B", "P", "T")
            }
            posterior = _normalize(scores, ("B", "P", "T"))
            method = "tempered_bayes_threeway"
        else:
            bp_mass = prior["B"] + prior["P"]
            if bp_mass <= 1e-12:
                return prior, {
                    "method": "markov_only_no_bp_mass",
                    "shoe_reliability": w,
                    "likelihood": likelihood,
                }
            prior_bp = {
                "B": prior["B"] / bp_mass,
                "P": prior["P"] / bp_mass,
            }
            scores = {
                side: prior_bp[side] * (max(1e-12, likelihood[side]) ** w)
                for side in ("B", "P")
            }
            posterior_bp = _normalize(scores, ("B", "P"))
            posterior = {
                "B": bp_mass * posterior_bp["B"],
                "P": bp_mass * posterior_bp["P"],
                "T": prior["T"],
            }
            posterior = _normalize(posterior, ("B", "P", "T"))
            method = "tempered_bayes_bp_conditional_preserve_tie"

        return posterior, {
            "method": method,
            "shoe_reliability": float(w),
            "likelihood": likelihood,
            "markov_prior": prior,
            "posterior": posterior,
        }

    def predict(
        self,
        history: str | Iterable[Any],
        *,
        shoe_probs: Mapping[str, Any] | Sequence[float] | None = None,
        shoe_reliability: float = 1.0,
        bankroll: float = 0.0,
    ) -> dict[str, Any]:
        markov = update_and_predict_engine(history)
        markov_probs = dict(markov["probabilities"])
        final_probs, fusion = self.bayesian_fuse(
            markov_probs,
            shoe_probs,
            shoe_reliability=shoe_reliability,
        )

        direction = "B" if final_probs["B"] >= final_probs["P"] else "P"
        money = self.money.allocate(
            direction=direction,
            probabilities=final_probs,
            final_weight=float(markov["final_weight"]),
            bankroll=float(bankroll or 0.0),
        )

        return {
            "direction": direction,
            "direction_text": "莊" if direction == "B" else "閒",
            "markov_probs": markov_probs,
            "final_probs": final_probs,
            "final_probability": float(final_probs[direction]),
            "bet_allowed": bool(money["bet_allowed"]),
            "bet_percentage": float(money["bet_percentage"]),
            "bet_amount": float(money["bet_amount"]),
            "edge": float(money["edge"]),
            "edge_percent": float(money["edge_percent"]),
            "money_management": money,
            "markov": markov,
            "fusion": fusion,
        }


if __name__ == "__main__":
    # Simulation example requested by the architecture specification.
    engine = BaccaratQuantEngine()
    example = engine.predict(
        "BBPBPTBPPB",
        shoe_probs=[0.506, 0.494],
        shoe_reliability=1.0,
        bankroll=10000,
    )
    print(
        {
            "prediction_direction": example["direction"],
            "final_probs": example["final_probs"],
            "dynamic_bet_percentage": example["bet_percentage"],
            "edge_percent": example["edge_percent"],
            "bet_allowed": example["bet_allowed"],
        }
    )


__all__ = ["BaccaratQuantEngine"]
