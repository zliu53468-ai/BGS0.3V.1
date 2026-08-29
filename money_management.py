"""Quarter-Kelly money management for road-model baccarat probabilities.

The public ``MoneyManagementModel`` interface is unchanged.  The model receives
B/P/T probabilities, computes the selected side's virtual EV after banker
commission, and sizes only positive-EV signals with one-quarter Kelly.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

HIGH_TIE_THRESHOLD = 0.15
MIN_BET_RATIO = 0.0
MAX_BET_RATIO = 0.30
BANKER_NET_PAYOUT = 0.95
PLAYER_NET_PAYOUT = 1.00
KELLY_FRACTION = 0.25
MIN_POSITIVE_EV = 0.002


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
    def _resolved_probability(
        *,
        side: str,
        probabilities: Mapping[str, Any],
    ) -> tuple[float, float, float]:
        p_b = _clip(probabilities.get("B", 0.0))
        p_p = _clip(probabilities.get("P", 0.0))
        resolved = p_b + p_p
        if resolved <= 1e-12:
            return 0.5, p_b, p_p
        p_win = (p_b if side == "B" else p_p) / resolved
        return float(p_win), p_b, p_p

    @staticmethod
    def break_even_probability(side: str) -> float:
        side = str(side or "").upper().strip()
        payout = BANKER_NET_PAYOUT if side == "B" else PLAYER_NET_PAYOUT
        return float(1.0 / (1.0 + payout))

    @classmethod
    def edge_probability(
        cls,
        *,
        side: str,
        probabilities: Mapping[str, Any],
    ) -> float:
        side = str(side or "").upper().strip()
        if side not in {"B", "P"}:
            return 0.0
        p_win, _, _ = cls._resolved_probability(
            side=side,
            probabilities=probabilities,
        )
        return float(p_win - cls.break_even_probability(side))

    @classmethod
    def expected_value(
        cls,
        *,
        side: str,
        probabilities: Mapping[str, Any],
    ) -> float:
        side = str(side or "").upper().strip()
        if side not in {"B", "P"}:
            return 0.0
        p_win, _, _ = cls._resolved_probability(
            side=side,
            probabilities=probabilities,
        )
        payout = BANKER_NET_PAYOUT if side == "B" else PLAYER_NET_PAYOUT
        return float(payout * p_win - (1.0 - p_win))

    @classmethod
    def full_kelly_fraction(
        cls,
        *,
        side: str,
        probabilities: Mapping[str, Any],
    ) -> float:
        side = str(side or "").upper().strip()
        if side not in {"B", "P"}:
            return 0.0
        p_win, _, _ = cls._resolved_probability(
            side=side,
            probabilities=probabilities,
        )
        q = 1.0 - p_win
        b = BANKER_NET_PAYOUT if side == "B" else PLAYER_NET_PAYOUT
        if b <= 0.0:
            return 0.0
        return max(0.0, float((b * p_win - q) / b))

    @classmethod
    def kelly_fraction(
        cls,
        *,
        side: str,
        probabilities: Mapping[str, Any],
    ) -> float:
        """Return one-quarter Kelly, capped by the existing hard max ratio."""
        full = cls.full_kelly_fraction(
            side=side,
            probabilities=probabilities,
        )
        return min(MAX_BET_RATIO, max(0.0, full * KELLY_FRACTION))

    @staticmethod
    def _volatility_adjustment(
        *,
        probabilities: Mapping[str, Any],
        final_weight: float,
    ) -> float:
        # Compatibility diagnostic: quarter Kelly is the sizing rule, therefore
        # this factor no longer changes the capital fraction.
        del probabilities, final_weight
        return 1.0

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

        p_win_resolved, _, _ = self._resolved_probability(
            side=direction,
            probabilities=probabilities,
        )
        break_even = self.break_even_probability(direction)
        edge = self.edge_probability(
            side=direction,
            probabilities=probabilities,
        )
        expected_value_per_unit = self.expected_value(
            side=direction,
            probabilities=probabilities,
        )
        full_kelly = self.full_kelly_fraction(
            side=direction,
            probabilities=probabilities,
        )
        quarter_kelly = self.kelly_fraction(
            side=direction,
            probabilities=probabilities,
        )

        if (
            direction not in {"B", "P"}
            or expected_value_per_unit < MIN_POSITIVE_EV
            or quarter_kelly <= 0.0
        ):
            final_bet_ratio = 0.0
            bet_allowed = False
            reason = "virtual_ev_below_threshold_no_bet"
        else:
            final_bet_ratio = min(MAX_BET_RATIO, quarter_kelly)
            bet_allowed = True
            reason = "positive_virtual_ev_quarter_kelly"

        bet_amount = bankroll * final_bet_ratio
        tie_risk_active = p_tie > HIGH_TIE_THRESHOLD

        return {
            "direction": direction,
            "bankroll": bankroll,
            "resolved_win_probability": float(p_win_resolved),
            "break_even_probability": float(break_even),
            "edge": float(edge),
            "edge_percent": float(edge * 100.0),
            "expected_value_per_unit": float(expected_value_per_unit),
            "virtual_ev": float(expected_value_per_unit),
            "virtual_ev_percent": float(expected_value_per_unit * 100.0),
            "full_kelly_fraction": float(full_kelly),
            "kelly_fraction": float(quarter_kelly),
            "applied_kelly_multiplier": float(KELLY_FRACTION),
            "volatility_adjustment": 1.0,
            "edge_target_ratio": float(quarter_kelly),
            "base_ratio": float(quarter_kelly),
            "final_weight": float(final_weight),
            "pre_tie_adjusted_ratio": float(quarter_kelly),
            "adjusted_ratio": float(quarter_kelly),
            "tie_probability": float(p_tie),
            "tie_risk_active": bool(tie_risk_active),
            "tie_risk_threshold": float(HIGH_TIE_THRESHOLD),
            "final_bet_ratio": float(final_bet_ratio),
            "bet_percentage": float(final_bet_ratio * 100.0),
            "bet_amount": float(bet_amount),
            "bet_allowed": bool(bet_allowed),
            "mandatory_bet": False,
            "reason": reason,
            "min_bet_ratio": float(MIN_BET_RATIO),
            "max_bet_ratio": float(MAX_BET_RATIO),
            "minimum_positive_ev": float(MIN_POSITIVE_EV),
            "banker_net_payout": float(BANKER_NET_PAYOUT),
            "player_net_payout": float(PLAYER_NET_PAYOUT),
            "sizing_method": "quarter_kelly_from_model_probability_virtual_ev",
        }


__all__ = [
    "MoneyManagementModel",
    "HIGH_TIE_THRESHOLD",
    "MIN_BET_RATIO",
    "MAX_BET_RATIO",
    "BANKER_NET_PAYOUT",
    "PLAYER_NET_PAYOUT",
    "KELLY_FRACTION",
    "MIN_POSITIVE_EV",
]
