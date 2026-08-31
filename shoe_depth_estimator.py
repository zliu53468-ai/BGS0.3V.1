"""Round-count baccarat shoe depth estimator.

This module is strictly the *estimated* path used when exact remaining point
counts or observed card faces are unavailable. It never fabricates
``remaining_counts`` and never claims exact composition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from shoe_constants import (
    AVERAGE_CARDS_PER_HAND,
    BURN_CARDS,
    REFERENCE_HANDS,
    SHOE_DECKS,
    TOTAL_SHOE_CARDS,
    estimate_cards_used,
    estimate_remaining_cards,
    total_cards_for_decks,
)

# Compatibility aliases. Values are authoritative imports from shoe_constants;
# there is no independent definition in this module.
TOTAL_CARDS = TOTAL_SHOE_CARDS


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
    shoe_decks: int
    total_cards: float
    average_cards_per_hand: float
    burn_cards: int
    reference_hands: float

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "hand_count": int(self.hand_count),
            "estimated_cards_used": float(self.estimated_cards_used),
            "raw_remaining_cards": float(self.raw_remaining_cards),
            "remaining_cards": float(self.remaining_cards),
            "remaining_cards_source": "round_count_estimate",
            "shoe_progress": float(self.shoe_progress),
            "average_cards_per_hand": float(self.average_cards_per_hand),
            "cards_per_hand_assumption": float(self.average_cards_per_hand),
            "shoe_decks": int(self.shoe_decks),
            "starting_cards_assumption": float(self.total_cards),
            "burn_cards": int(self.burn_cards),
            "reference_hands": float(self.reference_hands),
            "reference_hands_semantics": "product_maturity_reference_not_cut_card_position",
            "exact_composition": False,
            "semantics": "round_count_maturity_depth_estimate_not_exact_card_composition",
        }


class ShoeDepthEstimator:
    def __init__(
        self,
        *,
        total_cards: float | None = None,
        average_cards_per_hand: float = AVERAGE_CARDS_PER_HAND,
        reference_hands: float = REFERENCE_HANDS,
        burn_cards: int = BURN_CARDS,
        shoe_decks: int = SHOE_DECKS,
    ) -> None:
        self.shoe_decks = max(1, min(16, int(shoe_decks)))
        authoritative_total = float(total_cards_for_decks(self.shoe_decks))
        self.total_cards = (
            authoritative_total
            if total_cards is None
            else max(1.0, float(total_cards))
        )
        self.average_cards_per_hand = max(0.01, float(average_cards_per_hand))
        self.reference_hands = max(1.0, float(reference_hands))
        self.burn_cards = max(0, int(burn_cards))

    def estimate(self, history: Iterable[Any]) -> ShoeDepthEstimate:
        hand_count = len(_clean_threeway(history))
        used = estimate_cards_used(
            hand_count,
            average_cards_per_hand=self.average_cards_per_hand,
            burn_cards=self.burn_cards,
        )
        raw_remaining = self.total_cards - used
        if self.total_cards == float(total_cards_for_decks(self.shoe_decks)):
            remaining = estimate_remaining_cards(
                hand_count,
                decks=self.shoe_decks,
                average_cards_per_hand=self.average_cards_per_hand,
                burn_cards=self.burn_cards,
            )
        else:
            remaining = max(0.0, raw_remaining)
        progress = min(1.0, max(0.0, hand_count / self.reference_hands))
        return ShoeDepthEstimate(
            hand_count=hand_count,
            estimated_cards_used=used,
            raw_remaining_cards=raw_remaining,
            remaining_cards=remaining,
            shoe_progress=progress,
            shoe_decks=self.shoe_decks,
            total_cards=self.total_cards,
            average_cards_per_hand=self.average_cards_per_hand,
            burn_cards=self.burn_cards,
            reference_hands=self.reference_hands,
        )


__all__ = [
    "TOTAL_CARDS",
    "AVERAGE_CARDS_PER_HAND",
    "BURN_CARDS",
    "REFERENCE_HANDS",
    "SHOE_DECKS",
    "ShoeDepthEstimate",
    "ShoeDepthEstimator",
]
