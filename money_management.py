"""Edge-gated money management for the baccarat quant engine.

A direction is still predicted every hand, but capital is deployed only when the
resolved B/P posterior clears the break-even threshold after commission.

If Edge <= 0:
    bet_ratio = 0

If Edge > 0:
    target_ratio = Edge * volatility_adjustment
    final_ratio = clip(target_ratio, 5%, 30%)

The full Kelly fraction is retained as a diagnostic/risk ceiling reference.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

HIGH_TIE_THRESHOLD = 0.15
MIN_BET_RATIO = 0.05
MAX_BET_RATIO = 0.30
BANKER_NET_PAYOUT = 0.95
PLAYER_NET_PAYOUT = 1.00


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
        """Break-even p* from b*p - (1-p)=0 => p*=1/(1+b)."""
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
    def kelly_fraction(
        cls,
        *,
        side: str,
        probabilities: Mapping[str, Any],
    ) -> float:
        """Full Kelly: f*=(b*p-q)/b, with ties removed as push mass."""
        side = str(side or "").upper().strip()
        if side not in {"B", "P"}:
            return 0.0

        p_win, _, _ = cls._resolved_probability(
            side=side,
            probabilities=probabilities,
        )
        q = 1.0 - p_win
        b = BANKER_NET_PAYOUT if side == "B" else PLAYER_NET_PAYOUT
        full_kelly = (b * p_win - q) / b
        return max(0.0, float(full_kelly))

    @staticmethod
    def _volatility_adjustment(
        *,
        probabilities: Mapping[str, Any],
        final_weight: float,
    ) -> float:
        """Scale exposure down under weak/noisy evidence.

        margin_strength is the B/P resolved separation in [0,1].
        adjustment is deliberately bounded so Edge remains the primary driver.
        """
        p_b = _clip(probabilities.get("B", 0.0))
        p_p = _clip(probabilities.get("P", 0.0))
        resolved = p_b + p_p
        margin_strength = (
            abs(p_b - p_p) / resolved if resolved > 1e-12 else 0.0
        )
        confidence = _clip(final_weight)
        return float(
            max(
                0.50,
                min(
                    3.00,
                    0.50 + 1.50 * confidence + 1.00 * margin_strength,
                ),
            )
        )

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
        raw_kelly = self.kelly_fraction(
            side=direction,
            probabilities=probabilities,
        )
        volatility_adjustment = self._volatility_adjustment(
            probabilities=probabilities,
            final_weight=final_weight,
        )

        # Requested capital rule:
        #   target bankroll fraction = Edge * volatility_adjustment.
        edge_target_ratio = max(0.0, edge) * volatility_adjustment

        # Keep Kelly as a diagnostic. The requested 5%-30% floor/cap applies
        # only after the Edge gate has opened.
        if edge <= 0.0 or direction not in {"B", "P"}:
            final_bet_ratio = 0.0
            bet_allowed = False
            reason = "negative_or_zero_edge_no_bet"
        else:
            final_bet_ratio = max(
                MIN_BET_RATIO,
                min(edge_target_ratio, MAX_BET_RATIO),
            )
            bet_allowed = True
            if edge_target_ratio <= MIN_BET_RATIO:
                reason = "positive_edge_minimum_5pct"
            elif edge_target_ratio >= MAX_BET_RATIO:
                reason = "positive_edge_cap_30pct"
            else:
                reason = "edge_volatility_dynamic"

        bet_amount = bankroll * final_bet_ratio
        tie_risk_active = p_tie > HIGH_TIE_THRESHOLD
        payout = BANKER_NET_PAYOUT if direction == "B" else PLAYER_NET_PAYOUT
        expected_value_per_unit = (
            payout * p_win_resolved - (1.0 - p_win_resolved)
            if direction in {"B", "P"} else 0.0
        )

        return {
            "direction": direction,
            "bankroll": bankroll,
            "resolved_win_probability": float(p_win_resolved),
            "break_even_probability": float(break_even),
            "edge": float(edge),
            "edge_percent": float(edge * 100.0),
            "expected_value_per_unit": float(expected_value_per_unit),
            "kelly_fraction": float(raw_kelly),
            "volatility_adjustment": float(volatility_adjustment),
            "edge_target_ratio": float(edge_target_ratio),
            "base_ratio": float(max(0.0, edge)),
            "final_weight": float(final_weight),
            # Compatibility aliases retained for predictor / LINE output.
            "pre_tie_adjusted_ratio": float(edge_target_ratio),
            "adjusted_ratio": float(edge_target_ratio),
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
            "banker_net_payout": float(BANKER_NET_PAYOUT),
            "player_net_payout": float(PLAYER_NET_PAYOUT),
        }


__all__ = [
    "MoneyManagementModel",
    "HIGH_TIE_THRESHOLD",
    "MIN_BET_RATIO",
    "MAX_BET_RATIO",
    "BANKER_NET_PAYOUT",
    "PLAYER_NET_PAYOUT",
]
