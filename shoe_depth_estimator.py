"""Shoe depth estimator based on observed B/P/T round count.

This is a maturity / risk-control estimate only. It does not reconstruct the exact
remaining card composition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

TOTAL_CARDS = 416.0
AVERAGE_CARDS_PER_HAND = 4.89
REFERENCE_HANDS = 70.0


def _clean_threeway(values: Iterable[Any]) -> list[str]:
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
    return result


@dataclass(frozen=True)
class ShoeDepthEstimate:
    hand_count: int
    estimated_cards_used: float
    raw_remaining_cards: float
    remaining_cards: float
    shoe_progress: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "hand_count": int(self.hand_count),
            "estimated_cards_used": float(self.estimated_cards_used),
            "raw_remaining_cards": float(self.raw_remaining_cards),
            "remaining_cards": float(self.remaining_cards),
            "shoe_progress": float(self.shoe_progress),
            "cards_per_hand_assumption": float(AVERAGE_CARDS_PER_HAND),
            "starting_cards_assumption": float(TOTAL_CARDS),
            "reference_hands": float(REFERENCE_HANDS),
            "semantics": "round_count_maturity_estimate_not_exact_card_composition",
        }


class ShoeDepthEstimator:
    def __init__(
        self,
        *,
        total_cards: float = TOTAL_CARDS,
        average_cards_per_hand: float = AVERAGE_CARDS_PER_HAND,
        reference_hands: float = REFERENCE_HANDS,
    ) -> None:
        self.total_cards = max(1.0, float(total_cards))
        self.average_cards_per_hand = max(0.01, float(average_cards_per_hand))
        self.reference_hands = max(1.0, float(reference_hands))

    def estimate(self, history: Iterable[Any]) -> ShoeDepthEstimate:
        hand_count = len(_clean_threeway(history))
        used = hand_count * self.average_cards_per_hand
        raw_remaining = self.total_cards - used
        remaining = max(0.0, raw_remaining)
        progress = min(1.0, max(0.0, hand_count / self.reference_hands))
        return ShoeDepthEstimate(
            hand_count=hand_count,
            estimated_cards_used=used,
            raw_remaining_cards=raw_remaining,
            remaining_cards=remaining,
            shoe_progress=progress,
        )


__all__ = [
    "TOTAL_CARDS",
    "AVERAGE_CARDS_PER_HAND",
    "REFERENCE_HANDS",
    "ShoeDepthEstimate",
    "ShoeDepthEstimator",
]
