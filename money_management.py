"""Money management for the three-way Markov predictor.

Tie is modeled as a push for B/P wagers. Kelly is computed on B/P resolved mass,
then multiplied by entropy/shoe-depth confidence. A high predicted tie rate reduces
exposure but never creates a tie wager.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

HIGH_TIE_THRESHOLD = 0.15
HIGH_TIE_BET_MULTIPLIER = 0.65
MIN_FINAL_WEIGHT = 0.15
MIN_BET_RATIO = 0.05
MAX_BET_RATIO = 0.30


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


class MoneyManagementModel:
    @staticmethod
    def kelly_fraction(
        *,
        side: str,
        probabilities: Mapping[str, Any],
    ) -> float:
        side = str(side or "").upper().strip()
        if side not in {"B", "P"}:
            return 0.0

        p_b = _clip(probabilities.get("B", 0.0))
        p_p = _clip(probabilities.get("P", 0.0))
        p_win = p_b if side == "B" else p_p
        p_loss = p_p if side == "B" else p_b
        resolved_mass = p_win + p_loss
        if resolved_mass <= 1e-12:
            return 0.0

        # Banker pays 0.95 net; Player pays 1.00. T is a push, not a loss.
        b = 0.95 if side == "B" else 1.0
        full_kelly = (b * p_win - p_loss) / (b * resolved_mass)
        return max(0.0, float(full_kelly))

    def allocate(
        self,
        *,
        direction: str,
        probabilities: Mapping[str, Any],
        final_weight: float,
        bankroll: float = 0.0,
    ) -> dict[str, Any]:
        direction = str(direction or "").upper().strip()
        final_weight = _clip(final_weight)
        bankroll = max(0.0, float(bankroll or 0.0))
        p_tie = _clip(probabilities.get("T", 0.0))

        kelly = self.kelly_fraction(
            side=direction,
            probabilities=probabilities,
        )
        pre_tie_adjusted_ratio = kelly * final_weight
        tie_risk_active = p_tie > HIGH_TIE_THRESHOLD
        tie_multiplier = (
            HIGH_TIE_BET_MULTIPLIER if tie_risk_active else 1.0
        )
        adjusted_ratio = pre_tie_adjusted_ratio * tie_multiplier

        if adjusted_ratio <= 0.0 or final_weight < MIN_FINAL_WEIGHT:
            final_bet_ratio = 0.0
            bet_allowed = False
            if final_weight < MIN_FINAL_WEIGHT:
                reason = "entropy_or_shoe_maturity_gate"
            else:
                reason = "kelly_non_positive"
        else:
            final_bet_ratio = max(
                MIN_BET_RATIO,
                min(adjusted_ratio, MAX_BET_RATIO),
            )
            bet_allowed = True
            reason = (
                "kelly_entropy_shoe_depth_with_high_tie_reduction"
                if tie_risk_active
                else "kelly_entropy_shoe_depth"
            )

        bet_amount = bankroll * final_bet_ratio if bet_allowed else 0.0
        return {
            "direction": direction,
            "bankroll": bankroll,
            "kelly_fraction": float(kelly),
            "final_weight": float(final_weight),
            "pre_tie_adjusted_ratio": float(pre_tie_adjusted_ratio),
            "tie_probability": float(p_tie),
            "tie_risk_active": bool(tie_risk_active),
            "tie_risk_threshold": float(HIGH_TIE_THRESHOLD),
            "tie_bet_multiplier": float(tie_multiplier),
            "adjusted_ratio": float(adjusted_ratio),
            "final_bet_ratio": float(final_bet_ratio),
            "bet_percentage": float(final_bet_ratio * 100.0),
            "bet_amount": float(bet_amount),
            "bet_allowed": bool(bet_allowed),
            "reason": reason,
            "min_final_weight": float(MIN_FINAL_WEIGHT),
            "min_bet_ratio": float(MIN_BET_RATIO),
            "max_bet_ratio": float(MAX_BET_RATIO),
        }


__all__ = [
    "MoneyManagementModel",
    "HIGH_TIE_THRESHOLD",
    "HIGH_TIE_BET_MULTIPLIER",
    "MIN_FINAL_WEIGHT",
    "MIN_BET_RATIO",
    "MAX_BET_RATIO",
]
