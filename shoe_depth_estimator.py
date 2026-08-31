"""Round-count baccarat shoe depth estimator and confidence features.

This is an estimated depth path only. It never fabricates remaining_counts and
never creates a B/P direction. LSTM uses remaining ratio / penetration / stage
only as confidence and sizing metadata.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from shoe_constants import AVERAGE_CARDS_PER_HAND, BURN_CARDS, REFERENCE_HANDS, SHOE_DECKS, TOTAL_SHOE_CARDS, estimate_cards_used, estimate_remaining_cards, total_cards_for_decks

TOTAL_CARDS = TOTAL_SHOE_CARDS
SHOE_STAGE_CONFIDENCE_FACTORS = {"OPENING": 1.00, "DEVELOPING": 1.00, "MATURE": 0.97, "LATE": 0.88, "UNKNOWN": 0.95}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def classify_shoe_stage(remaining_ratio: float) -> str:
    ratio = _clip(remaining_ratio)
    if ratio >= 0.84: return "OPENING"
    if ratio >= 0.67: return "DEVELOPING"
    if ratio >= 0.48: return "MATURE"
    return "LATE"


def build_shoe_depth_features(remaining_cards: float, *, shoe_decks: int = SHOE_DECKS, reliability: float = 1.0, source: str = "") -> dict[str, float | str]:
    decks = max(1, min(16, int(shoe_decks))); total = float(total_cards_for_decks(decks)); remaining = max(0.0, min(total, float(remaining_cards)))
    ratio = _clip(remaining / max(1.0, total)); penetration = _clip(1.0 - ratio); stage = classify_shoe_stage(ratio); reliability_value = _clip(reliability)
    anchor = float(SHOE_STAGE_CONFIDENCE_FACTORS[stage]); applied_factor = _clip(1.0 - reliability_value * (1.0 - anchor), 0.88, 1.0)
    return {"remaining_ratio": float(ratio), "penetration": float(penetration), "shoe_stage": stage, "shoe_confidence_factor": float(applied_factor),
            "shoe_stage_anchor": float(anchor), "remaining_cards_reliability": float(reliability_value), "depth_feature_source": str(source or "unknown"), "direction_authority": False}


def _clean_threeway(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in values:
        raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome") if isinstance(item, Mapping) else item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}: result.append(value)
    return result


@dataclass(frozen=True)
class ShoeDepthEstimate:
    hand_count: int; estimated_cards_used: float; raw_remaining_cards: float; remaining_cards: float; shoe_progress: float; shoe_decks: int; total_cards: float
    average_cards_per_hand: float; burn_cards: int; reference_hands: float; remaining_ratio: float; penetration: float; shoe_stage: str; shoe_confidence_factor: float; remaining_cards_reliability: float

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {"hand_count": int(self.hand_count), "estimated_cards_used": float(self.estimated_cards_used), "raw_remaining_cards": float(self.raw_remaining_cards),
                "remaining_cards": float(self.remaining_cards), "remaining_cards_source": "round_count_estimate", "shoe_progress": float(self.shoe_progress),
                "remaining_ratio": float(self.remaining_ratio), "penetration": float(self.penetration), "shoe_stage": str(self.shoe_stage),
                "shoe_confidence_factor": float(self.shoe_confidence_factor), "remaining_cards_reliability": float(self.remaining_cards_reliability),
                "average_cards_per_hand": float(self.average_cards_per_hand), "cards_per_hand_assumption": float(self.average_cards_per_hand), "shoe_decks": int(self.shoe_decks),
                "starting_cards_assumption": float(self.total_cards), "burn_cards": int(self.burn_cards), "reference_hands": float(self.reference_hands),
                "reference_hands_semantics": "product_maturity_reference_not_cut_card_position", "exact_composition": False, "direction_authority": False,
                "semantics": "round_count_maturity_depth_estimate_for_confidence_only_not_exact_card_composition_not_BP_direction"}


class ShoeDepthEstimator:
    def __init__(self, *, total_cards: float | None = None, average_cards_per_hand: float = AVERAGE_CARDS_PER_HAND, reference_hands: float = REFERENCE_HANDS, burn_cards: int = BURN_CARDS, shoe_decks: int = SHOE_DECKS) -> None:
        self.shoe_decks = max(1, min(16, int(shoe_decks))); authoritative_total = float(total_cards_for_decks(self.shoe_decks))
        self.total_cards = authoritative_total if total_cards is None else max(1.0, float(total_cards)); self.average_cards_per_hand = max(0.01, float(average_cards_per_hand))
        self.reference_hands = max(1.0, float(reference_hands)); self.burn_cards = max(0, int(burn_cards))

    def estimate(self, history: Iterable[Any]) -> ShoeDepthEstimate:
        hand_count = len(_clean_threeway(history)); used = estimate_cards_used(hand_count, average_cards_per_hand=self.average_cards_per_hand, burn_cards=self.burn_cards); raw_remaining = self.total_cards - used
        remaining = estimate_remaining_cards(hand_count, decks=self.shoe_decks, average_cards_per_hand=self.average_cards_per_hand, burn_cards=self.burn_cards) if self.total_cards == float(total_cards_for_decks(self.shoe_decks)) else max(0.0, raw_remaining)
        progress = min(1.0, max(0.0, hand_count / self.reference_hands)); depth_reliability = min(0.80, 0.55 + 0.25 * progress)
        features = build_shoe_depth_features(remaining, shoe_decks=self.shoe_decks, reliability=depth_reliability, source="round_count_estimate")
        return ShoeDepthEstimate(hand_count=hand_count, estimated_cards_used=used, raw_remaining_cards=raw_remaining, remaining_cards=remaining, shoe_progress=progress,
                                 shoe_decks=self.shoe_decks, total_cards=self.total_cards, average_cards_per_hand=self.average_cards_per_hand, burn_cards=self.burn_cards,
                                 reference_hands=self.reference_hands, remaining_ratio=float(features["remaining_ratio"]), penetration=float(features["penetration"]),
                                 shoe_stage=str(features["shoe_stage"]), shoe_confidence_factor=float(features["shoe_confidence_factor"]),
                                 remaining_cards_reliability=float(features["remaining_cards_reliability"]))


__all__ = ["TOTAL_CARDS", "AVERAGE_CARDS_PER_HAND", "BURN_CARDS", "REFERENCE_HANDS", "SHOE_DECKS", "SHOE_STAGE_CONFIDENCE_FACTORS", "classify_shoe_stage", "build_shoe_depth_features", "ShoeDepthEstimate", "ShoeDepthEstimator"]
