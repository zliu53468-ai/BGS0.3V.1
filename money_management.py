"""BGS Kelly 資金管理核心。

正式方向永遠是 P/B 二選一。模型勝率會先保守限制在 48%～58%，
再套用分數 Kelly，最後將下注比例硬性限制在本金 5%～30%。
"""
from __future__ import annotations

from typing import Any, Mapping
import math
import os

HIGH_TIE_THRESHOLD = 0.15
MIN_BET_RATIO = max(0.05, float(os.getenv("MIN_BET_FRACTION", "0.05") or "0.05"))
MAX_BET_RATIO = min(0.30, float(os.getenv("MAX_BET_FRACTION", "0.30") or "0.30"))
if MAX_BET_RATIO < MIN_BET_RATIO:
    MAX_BET_RATIO = MIN_BET_RATIO
BANKER_NET_PAYOUT = 0.95
PLAYER_NET_PAYOUT = 1.00
KELLY_FRACTION = max(0.0, float(os.getenv("KELLY_FRACTION", "0.25") or "0.25"))
MIN_POSITIVE_EV = 0.0  # 相容欄位；新版不再以 EV gate 禁止下注。
PROBABILITY_MIN = 0.48
PROBABILITY_MAX = 0.58


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
        *, side: str, probabilities: Mapping[str, Any]
    ) -> tuple[float, float, float]:
        p_b = _clip(probabilities.get("B", 0.5))
        p_p = _clip(probabilities.get("P", 0.5))
        resolved = p_b + p_p
        if resolved <= 1e-12:
            raw_p_win = 0.5
            p_b, p_p = 0.5, 0.5
        else:
            raw_p_win = (p_b if side == "B" else p_p) / resolved
        # 短靴只有 50～70 局，避免少量樣本令模型過度自信；正式 Kelly
        # 使用的勝率一律保守收斂到 48%～58%。
        p_win = _clip(raw_p_win, PROBABILITY_MIN, PROBABILITY_MAX)
        return float(p_win), float(p_b), float(p_p)

    @staticmethod
    def break_even_probability(side: str) -> float:
        side = str(side or "").upper().strip()
        payout = BANKER_NET_PAYOUT if side == "B" else PLAYER_NET_PAYOUT
        return float(1.0 / (1.0 + payout))

    @classmethod
    def edge_probability(cls, *, side: str, probabilities: Mapping[str, Any]) -> float:
        side = "B" if str(side or "").upper().strip() == "B" else "P"
        p_win, _, _ = cls._resolved_probability(side=side, probabilities=probabilities)
        return float(p_win - cls.break_even_probability(side))

    @classmethod
    def expected_value(cls, *, side: str, probabilities: Mapping[str, Any]) -> float:
        side = "B" if str(side or "").upper().strip() == "B" else "P"
        p_win, _, _ = cls._resolved_probability(side=side, probabilities=probabilities)
        payout = BANKER_NET_PAYOUT if side == "B" else PLAYER_NET_PAYOUT
        return float(payout * p_win - (1.0 - p_win))

    @classmethod
    def raw_full_kelly_fraction(
        cls, *, side: str, probabilities: Mapping[str, Any]
    ) -> float:
        side = "B" if str(side or "").upper().strip() == "B" else "P"
        p_win, _, _ = cls._resolved_probability(side=side, probabilities=probabilities)
        q = 1.0 - p_win
        b = BANKER_NET_PAYOUT if side == "B" else PLAYER_NET_PAYOUT
        return float((p_win * b - q) / b) if b > 0.0 else 0.0

    @classmethod
    def full_kelly_fraction(
        cls, *, side: str, probabilities: Mapping[str, Any]
    ) -> float:
        # 保留舊 API 語意：對外 full Kelly 不回傳負比例。
        return max(0.0, cls.raw_full_kelly_fraction(side=side, probabilities=probabilities))

    @classmethod
    def kelly_fraction(cls, *, side: str, probabilities: Mapping[str, Any]) -> float:
        raw_full = cls.raw_full_kelly_fraction(side=side, probabilities=probabilities)
        scaled = raw_full * KELLY_FRACTION
        # 強制 5%～30% 的原因：產品規格要求每局都有明確方向與實際下注比例，
        # 因此即使 Kelly 原始值接近 0 或為負，最終執行比例仍以 5% 為硬下限；
        # 同時用 30% 硬上限避免短靴少樣本造成資金暴露失控。
        return _clip(scaled, MIN_BET_RATIO, MAX_BET_RATIO)

    @staticmethod
    def _volatility_adjustment(
        *, probabilities: Mapping[str, Any], final_weight: float
    ) -> float:
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
        direction = "B" if str(direction or "").upper().strip() == "B" else "P"
        final_weight = _clip(final_weight)
        bankroll = max(0.0, float(bankroll or 0.0))
        p_tie = _clip(probabilities.get("T", 0.0))

        p_win_resolved, _, _ = self._resolved_probability(
            side=direction, probabilities=probabilities
        )
        break_even = self.break_even_probability(direction)
        edge = self.edge_probability(side=direction, probabilities=probabilities)
        expected_value_per_unit = self.expected_value(
            side=direction, probabilities=probabilities
        )
        raw_full_kelly = self.raw_full_kelly_fraction(
            side=direction, probabilities=probabilities
        )
        full_kelly = max(0.0, raw_full_kelly)
        final_bet_ratio = self.kelly_fraction(
            side=direction, probabilities=probabilities
        )
        raw_fractional_kelly = raw_full_kelly * KELLY_FRACTION
        bet_amount = bankroll * final_bet_ratio
        tie_risk_active = p_tie > HIGH_TIE_THRESHOLD

        return {
            "direction": direction,
            "bankroll": bankroll,
            "resolved_win_probability": float(p_win_resolved),
            "clamped_win_probability": float(p_win_resolved),
            "probability_min": PROBABILITY_MIN,
            "probability_max": PROBABILITY_MAX,
            "break_even_probability": float(break_even),
            "edge": float(edge),
            "edge_percent": float(edge * 100.0),
            "expected_value_per_unit": float(expected_value_per_unit),
            "virtual_ev": float(expected_value_per_unit),
            "virtual_ev_percent": float(expected_value_per_unit * 100.0),
            "raw_full_kelly_fraction": float(raw_full_kelly),
            "full_kelly_fraction": float(full_kelly),
            "raw_fractional_kelly": float(raw_fractional_kelly),
            "kelly_fraction": float(final_bet_ratio),
            "applied_kelly_multiplier": float(KELLY_FRACTION),
            "volatility_adjustment": 1.0,
            "edge_target_ratio": float(final_bet_ratio),
            "base_ratio": float(final_bet_ratio),
            "final_weight": float(final_weight),
            "pre_tie_adjusted_ratio": float(final_bet_ratio),
            "adjusted_ratio": float(final_bet_ratio),
            "tie_probability": float(p_tie),
            "tie_risk_active": bool(tie_risk_active),
            "tie_risk_threshold": float(HIGH_TIE_THRESHOLD),
            "final_bet_ratio": float(final_bet_ratio),
            "bet_percentage": float(final_bet_ratio * 100.0),
            "bet_amount": float(bet_amount),
            "bet_allowed": True,
            "mandatory_bet": True,
            "reason": "two_arm_fractional_kelly_clipped_5_to_30_percent",
            "min_bet_ratio": float(MIN_BET_RATIO),
            "max_bet_ratio": float(MAX_BET_RATIO),
            "minimum_positive_ev": float(MIN_POSITIVE_EV),
            "banker_net_payout": float(BANKER_NET_PAYOUT),
            "player_net_payout": float(PLAYER_NET_PAYOUT),
            "sizing_method": "fractional_kelly_forced_clip_5_to_30_percent",
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
    "PROBABILITY_MIN",
    "PROBABILITY_MAX",
]
