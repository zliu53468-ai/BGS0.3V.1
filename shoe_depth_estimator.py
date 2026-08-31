"""Baccarat shoe depth and cut-card features for the production LSTM fusion.

The target table is a short 50-70 hand shoe.  Remaining-card totals and cut-card
position are physical depth features; by themselves they never invent Banker or
Player direction.  Exact remaining composition can contribute directional
information in ``lstm_road_model`` through the existing non-replacement shoe
calculator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import os

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

TOTAL_CARDS = TOTAL_SHOE_CARDS
TARGET_HANDS_MIN = max(40, min(60, int(os.getenv("BGS_TARGET_HANDS_MIN", "50") or "50")))
TARGET_HANDS_MAX = max(
    TARGET_HANDS_MIN + 5,
    min(90, int(os.getenv("BGS_TARGET_HANDS_MAX", "70") or "70")),
)
# 70 cards remaining on an eight-deck shoe is close to a 69-hand endpoint when
# the project-wide 4.9 cards/hand and burn assumptions are used.  The value can
# be overridden by request ``shoe_context.cut_card_remaining_cards`` or env.
DEFAULT_CUT_CARD_REMAINING = max(
    6,
    int(os.getenv("BGS_CUT_CARD_REMAINING", "70") or "70"),
)
SHOE_STAGE_CONFIDENCE_FACTORS = {
    "OPENING": 1.00,
    "DEVELOPING": 1.00,
    "MATURE": 0.98,
    "LATE": 0.92,
    "CUT_ZONE": 0.88,
    "UNKNOWN": 0.95,
}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _validated_cut_remaining(
    cut_card_remaining_cards: float | None,
    *,
    shoe_decks: int,
) -> float:
    total = float(total_cards_for_decks(shoe_decks))
    raw = DEFAULT_CUT_CARD_REMAINING if cut_card_remaining_cards is None else cut_card_remaining_cards
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(DEFAULT_CUT_CARD_REMAINING)
    # A cut point must still leave enough cards to complete a baccarat hand.
    return max(6.0, min(max(6.0, total - 6.0), value))


def build_cut_card_features(
    remaining_cards: float,
    *,
    hand_count: int = 0,
    shoe_decks: int = SHOE_DECKS,
    cut_card_remaining_cards: float | None = None,
    average_cards_per_hand: float = AVERAGE_CARDS_PER_HAND,
) -> dict[str, float | int | bool]:
    """Return continuous physical depth relative to the configured cut card.

    ``cut_progress`` is 0 at a fresh shoe and 1 at/after the cut point.  It is a
    weighting/maturity feature, not a Banker/Player vote.
    """
    decks = max(1, min(16, int(shoe_decks)))
    total = float(total_cards_for_decks(decks))
    remaining = max(0.0, min(total, float(remaining_cards)))
    cut_remaining = _validated_cut_remaining(
        cut_card_remaining_cards,
        shoe_decks=decks,
    )
    playable_span = max(1.0, total - cut_remaining)
    cards_used = max(0.0, total - remaining)
    cut_progress = _clip(cards_used / playable_span)
    cards_until_cut = max(0.0, remaining - cut_remaining)
    hands_until_cut = cards_until_cut / max(0.01, float(average_cards_per_hand))
    count = max(0, int(hand_count or 0))
    target_window_progress = _clip(
        (count - TARGET_HANDS_MIN) / max(1.0, float(TARGET_HANDS_MAX - TARGET_HANDS_MIN))
    )
    return {
        "target_hands_min": int(TARGET_HANDS_MIN),
        "target_hands_max": int(TARGET_HANDS_MAX),
        "cut_card_remaining_cards": float(cut_remaining),
        "cards_until_cut": float(cards_until_cut),
        "estimated_hands_until_cut": float(hands_until_cut),
        "cut_progress": float(cut_progress),
        "cut_proximity": float(cut_progress),
        "cut_reached": bool(remaining <= cut_remaining),
        "within_50_70_hand_window": bool(TARGET_HANDS_MIN <= count <= TARGET_HANDS_MAX),
        "target_window_progress": float(target_window_progress),
    }


def classify_shoe_stage(
    remaining_ratio: float,
    *,
    cut_progress: float = 0.0,
) -> str:
    ratio = _clip(remaining_ratio)
    cut = _clip(cut_progress)
    if cut >= 0.92:
        return "CUT_ZONE"
    if ratio >= 0.82:
        return "OPENING"
    if ratio >= 0.64:
        return "DEVELOPING"
    if ratio >= 0.43:
        return "MATURE"
    return "LATE"


def build_shoe_depth_features(
    remaining_cards: float,
    *,
    shoe_decks: int = SHOE_DECKS,
    reliability: float = 1.0,
    source: str = "",
    hand_count: int = 0,
    cut_card_remaining_cards: float | None = None,
) -> dict[str, float | str | int | bool]:
    decks = max(1, min(16, int(shoe_decks)))
    total = float(total_cards_for_decks(decks))
    remaining = max(0.0, min(total, float(remaining_cards)))
    ratio = _clip(remaining / max(1.0, total))
    penetration = _clip(1.0 - ratio)
    cut = build_cut_card_features(
        remaining,
        hand_count=hand_count,
        shoe_decks=decks,
        cut_card_remaining_cards=cut_card_remaining_cards,
    )
    stage = classify_shoe_stage(ratio, cut_progress=float(cut["cut_progress"]))
    reliability_value = _clip(reliability)
    anchor = float(SHOE_STAGE_CONFIDENCE_FACTORS[stage])
    applied_factor = _clip(1.0 - reliability_value * (1.0 - anchor), 0.88, 1.0)
    return {
        "remaining_ratio": float(ratio),
        "penetration": float(penetration),
        "shoe_stage": stage,
        "shoe_confidence_factor": float(applied_factor),
        "shoe_stage_anchor": float(anchor),
        "remaining_cards_reliability": float(reliability_value),
        "depth_feature_source": str(source or "unknown"),
        "direction_authority": False,
        **cut,
    }


def _clean_threeway(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in values:
        raw = (
            item.get("outcome")
            or item.get("actual")
            or item.get("actual_outcome")
            or item.get("virtual_outcome")
            if isinstance(item, Mapping)
            else item
        )
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
    remaining_ratio: float
    penetration: float
    shoe_stage: str
    shoe_confidence_factor: float
    remaining_cards_reliability: float
    cut_card_remaining_cards: float
    cut_progress: float
    cards_until_cut: float
    estimated_hands_until_cut: float

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "hand_count": int(self.hand_count),
            "estimated_cards_used": float(self.estimated_cards_used),
            "raw_remaining_cards": float(self.raw_remaining_cards),
            "remaining_cards": float(self.remaining_cards),
            "remaining_cards_source": "round_count_estimate",
            "shoe_progress": float(self.shoe_progress),
            "remaining_ratio": float(self.remaining_ratio),
            "penetration": float(self.penetration),
            "shoe_stage": str(self.shoe_stage),
            "shoe_confidence_factor": float(self.shoe_confidence_factor),
            "remaining_cards_reliability": float(self.remaining_cards_reliability),
            "cut_card_remaining_cards": float(self.cut_card_remaining_cards),
            "cut_progress": float(self.cut_progress),
            "cut_proximity": float(self.cut_progress),
            "cards_until_cut": float(self.cards_until_cut),
            "estimated_hands_until_cut": float(self.estimated_hands_until_cut),
            "target_hands_min": int(TARGET_HANDS_MIN),
            "target_hands_max": int(TARGET_HANDS_MAX),
            "average_cards_per_hand": float(self.average_cards_per_hand),
            "cards_per_hand_assumption": float(self.average_cards_per_hand),
            "shoe_decks": int(self.shoe_decks),
            "starting_cards_assumption": float(self.total_cards),
            "burn_cards": int(self.burn_cards),
            "reference_hands": float(self.reference_hands),
            "reference_hands_semantics": "product_maturity_reference_not_cut_card_position",
            "exact_composition": False,
            "direction_authority": False,
            "semantics": "50_70_hand_shoe_depth_and_cut_features_for_lstm_fusion",
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
        cut_card_remaining_cards: float | None = None,
    ) -> None:
        self.shoe_decks = max(1, min(16, int(shoe_decks)))
        authoritative_total = float(total_cards_for_decks(self.shoe_decks))
        self.total_cards = authoritative_total if total_cards is None else max(1.0, float(total_cards))
        self.average_cards_per_hand = max(0.01, float(average_cards_per_hand))
        self.reference_hands = max(1.0, float(reference_hands))
        self.burn_cards = max(0, int(burn_cards))
        self.cut_card_remaining_cards = cut_card_remaining_cards

    def estimate(self, history: Iterable[Any]) -> ShoeDepthEstimate:
        hand_count = len(_clean_threeway(history))
        used = estimate_cards_used(
            hand_count,
            average_cards_per_hand=self.average_cards_per_hand,
            burn_cards=self.burn_cards,
        )
        raw_remaining = self.total_cards - used
        remaining = (
            estimate_remaining_cards(
                hand_count,
                decks=self.shoe_decks,
                average_cards_per_hand=self.average_cards_per_hand,
                burn_cards=self.burn_cards,
            )
            if self.total_cards == float(total_cards_for_decks(self.shoe_decks))
            else max(0.0, raw_remaining)
        )
        progress = min(1.0, max(0.0, hand_count / max(1.0, float(TARGET_HANDS_MAX))))
        depth_reliability = min(0.82, 0.55 + 0.27 * progress)
        features = build_shoe_depth_features(
            remaining,
            shoe_decks=self.shoe_decks,
            reliability=depth_reliability,
            source="round_count_estimate",
            hand_count=hand_count,
            cut_card_remaining_cards=self.cut_card_remaining_cards,
        )
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
            remaining_ratio=float(features["remaining_ratio"]),
            penetration=float(features["penetration"]),
            shoe_stage=str(features["shoe_stage"]),
            shoe_confidence_factor=float(features["shoe_confidence_factor"]),
            remaining_cards_reliability=float(features["remaining_cards_reliability"]),
            cut_card_remaining_cards=float(features["cut_card_remaining_cards"]),
            cut_progress=float(features["cut_progress"]),
            cards_until_cut=float(features["cards_until_cut"]),
            estimated_hands_until_cut=float(features["estimated_hands_until_cut"]),
        )


__all__ = [
    "TOTAL_CARDS",
    "AVERAGE_CARDS_PER_HAND",
    "BURN_CARDS",
    "REFERENCE_HANDS",
    "SHOE_DECKS",
    "TARGET_HANDS_MIN",
    "TARGET_HANDS_MAX",
    "DEFAULT_CUT_CARD_REMAINING",
    "SHOE_STAGE_CONFIDENCE_FACTORS",
    "build_cut_card_features",
    "classify_shoe_stage",
    "build_shoe_depth_features",
    "ShoeDepthEstimate",
    "ShoeDepthEstimator",
]
