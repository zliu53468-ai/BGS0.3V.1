"""Short-shoe depth / cut-card estimator for the LSTM-primary BGS pipeline.

The target product assumption is a 50-70 hand baccarat shoe, default 60 hands.
Depth and cut-card state only calibrate confidence / sizing. They never choose B/P.
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
TARGET_HANDS_MIN = max(40, min(60, int(os.getenv("BGS_SHOE_MIN_HANDS", "50") or "50")))
TARGET_HANDS_MAX = max(TARGET_HANDS_MIN, min(80, int(os.getenv("BGS_SHOE_MAX_HANDS", "70") or "70")))
TARGET_HANDS_DEFAULT = max(TARGET_HANDS_MIN, min(TARGET_HANDS_MAX, int(os.getenv("BGS_SHOE_TARGET_HANDS", "60") or "60")))
CUT_CARD_REMAINING_OVERRIDE = max(0.0, float(os.getenv("BGS_CUT_CARD_REMAINING", "0") or "0"))
SHOE_STAGE_CONFIDENCE_FACTORS = {"OPENING": 1.00, "DEVELOPING": 1.00, "MATURE": 0.96, "LATE": 0.88, "UNKNOWN": 0.95}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _target_hands(value: int | float | None = None) -> int:
    raw = TARGET_HANDS_DEFAULT if value is None else int(value)
    return max(TARGET_HANDS_MIN, min(TARGET_HANDS_MAX, raw))


def auto_cut_card_remaining(*, shoe_decks: int = SHOE_DECKS, target_hands: int = TARGET_HANDS_DEFAULT, average_cards_per_hand: float = AVERAGE_CARDS_PER_HAND, burn_cards: int = BURN_CARDS) -> float:
    decks = max(1, min(16, int(shoe_decks)))
    total = float(total_cards_for_decks(decks))
    hands = _target_hands(target_hands)
    reserve = total - float(max(0, burn_cards)) - hands * max(0.01, float(average_cards_per_hand))
    return max(6.0, min(total * 0.60, reserve))


def resolve_cut_card_remaining(*, shoe_decks: int = SHOE_DECKS, target_hands: int = TARGET_HANDS_DEFAULT, cut_card_remaining: float | None = None) -> float:
    decks = max(1, min(16, int(shoe_decks)))
    total = float(total_cards_for_decks(decks))
    if cut_card_remaining is not None and float(cut_card_remaining) > 0.0:
        return max(6.0, min(total * 0.60, float(cut_card_remaining)))
    if CUT_CARD_REMAINING_OVERRIDE > 0.0:
        return max(6.0, min(total * 0.60, CUT_CARD_REMAINING_OVERRIDE))
    return auto_cut_card_remaining(shoe_decks=decks, target_hands=target_hands)


def classify_shoe_stage(progress_to_cut: float) -> str:
    progress = _clip(progress_to_cut)
    if progress < 0.25:
        return "OPENING"
    if progress < 0.55:
        return "DEVELOPING"
    if progress < 0.80:
        return "MATURE"
    return "LATE"


def build_shoe_depth_features(remaining_cards: float, *, shoe_decks: int = SHOE_DECKS, reliability: float = 1.0, source: str = "", target_hands: int = TARGET_HANDS_DEFAULT, cut_card_remaining: float | None = None, average_cards_per_hand: float = AVERAGE_CARDS_PER_HAND, burn_cards: int = BURN_CARDS) -> dict[str, float | int | str | bool]:
    decks = max(1, min(16, int(shoe_decks)))
    total = float(total_cards_for_decks(decks))
    remaining = max(0.0, min(total, float(remaining_cards)))
    target = _target_hands(target_hands)
    cut_reserve = resolve_cut_card_remaining(shoe_decks=decks, target_hands=target, cut_card_remaining=cut_card_remaining)
    avg = max(0.01, float(average_cards_per_hand))
    estimated_used_after_burn = max(0.0, total - float(max(0, burn_cards)) - remaining)
    estimated_hands_played = max(0.0, estimated_used_after_burn / avg)
    hands_remaining_to_cut = max(0.0, (remaining - cut_reserve) / avg)
    projected_total_hands = estimated_hands_played + hands_remaining_to_cut
    if projected_total_hands <= 1e-9:
        projected_total_hands = float(target)
    projected_total_hands = max(float(TARGET_HANDS_MIN), min(float(TARGET_HANDS_MAX), projected_total_hands))
    progress_to_cut = _clip(estimated_hands_played / max(1.0, projected_total_hands))
    remaining_ratio = _clip(remaining / max(1.0, total))
    penetration = _clip(1.0 - remaining_ratio)
    stage = classify_shoe_stage(progress_to_cut)
    reliability_value = _clip(reliability)
    anchor = float(SHOE_STAGE_CONFIDENCE_FACTORS[stage])
    applied_factor = _clip(1.0 - reliability_value * (1.0 - anchor), 0.88, 1.0)
    return {"remaining_ratio": float(remaining_ratio), "penetration": float(penetration), "shoe_stage": stage, "shoe_confidence_factor": float(applied_factor), "shoe_stage_anchor": float(anchor), "remaining_cards_reliability": float(reliability_value), "depth_feature_source": str(source or "unknown"), "direction_authority": False, "target_hands": int(target), "target_hands_min": int(TARGET_HANDS_MIN), "target_hands_max": int(TARGET_HANDS_MAX), "cut_card_remaining_cards": float(cut_reserve), "estimated_hands_played": float(estimated_hands_played), "hands_remaining_to_cut": float(hands_remaining_to_cut), "projected_total_hands": float(projected_total_hands), "short_shoe_progress": float(progress_to_cut), "cut_card_semantics": "confidence_and_sizing_only_never_BP_direction"}


def _clean_threeway(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
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
    remaining_ratio: float
    penetration: float
    shoe_stage: str
    shoe_confidence_factor: float
    remaining_cards_reliability: float
    target_hands: int
    cut_card_remaining_cards: float
    projected_total_hands: float
    hands_remaining_to_cut: float
    short_shoe_progress: float

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {"hand_count": int(self.hand_count), "estimated_cards_used": float(self.estimated_cards_used), "raw_remaining_cards": float(self.raw_remaining_cards), "remaining_cards": float(self.remaining_cards), "remaining_cards_source": "round_count_estimate", "shoe_progress": float(self.shoe_progress), "remaining_ratio": float(self.remaining_ratio), "penetration": float(self.penetration), "shoe_stage": str(self.shoe_stage), "shoe_confidence_factor": float(self.shoe_confidence_factor), "remaining_cards_reliability": float(self.remaining_cards_reliability), "average_cards_per_hand": float(self.average_cards_per_hand), "cards_per_hand_assumption": float(self.average_cards_per_hand), "shoe_decks": int(self.shoe_decks), "starting_cards_assumption": float(self.total_cards), "burn_cards": int(self.burn_cards), "reference_hands": float(self.reference_hands), "reference_hands_semantics": "legacy_product_maturity_reference_unchanged", "target_hands": int(self.target_hands), "target_hands_min": int(TARGET_HANDS_MIN), "target_hands_max": int(TARGET_HANDS_MAX), "cut_card_remaining_cards": float(self.cut_card_remaining_cards), "projected_total_hands": float(self.projected_total_hands), "hands_remaining_to_cut": float(self.hands_remaining_to_cut), "short_shoe_progress": float(self.short_shoe_progress), "exact_composition": False, "direction_authority": False, "semantics": "50_to_70_hand_short_shoe_depth_for_confidence_only_not_BP_direction"}


class ShoeDepthEstimator:
    def __init__(self, *, total_cards: float | None = None, average_cards_per_hand: float = AVERAGE_CARDS_PER_HAND, reference_hands: float = REFERENCE_HANDS, burn_cards: int = BURN_CARDS, shoe_decks: int = SHOE_DECKS, target_hands: int = TARGET_HANDS_DEFAULT, cut_card_remaining: float | None = None) -> None:
        self.shoe_decks = max(1, min(16, int(shoe_decks)))
        authoritative_total = float(total_cards_for_decks(self.shoe_decks))
        self.total_cards = authoritative_total if total_cards is None else max(1.0, float(total_cards))
        self.average_cards_per_hand = max(0.01, float(average_cards_per_hand))
        self.reference_hands = max(1.0, float(reference_hands))
        self.burn_cards = max(0, int(burn_cards))
        self.target_hands = _target_hands(target_hands)
        self.cut_card_remaining = resolve_cut_card_remaining(shoe_decks=self.shoe_decks, target_hands=self.target_hands, cut_card_remaining=cut_card_remaining)

    def estimate(self, history: Iterable[Any]) -> ShoeDepthEstimate:
        hand_count = len(_clean_threeway(history))
        used = estimate_cards_used(hand_count, average_cards_per_hand=self.average_cards_per_hand, burn_cards=self.burn_cards)
        raw_remaining = self.total_cards - used
        remaining = estimate_remaining_cards(hand_count, decks=self.shoe_decks, average_cards_per_hand=self.average_cards_per_hand, burn_cards=self.burn_cards) if self.total_cards == float(total_cards_for_decks(self.shoe_decks)) else max(0.0, raw_remaining)
        legacy_progress = min(1.0, max(0.0, hand_count / self.reference_hands))
        short_progress = min(1.0, max(0.0, hand_count / max(1.0, float(self.target_hands))))
        depth_reliability = min(0.85, 0.55 + 0.30 * short_progress)
        features = build_shoe_depth_features(remaining, shoe_decks=self.shoe_decks, reliability=depth_reliability, source="round_count_estimate", target_hands=self.target_hands, cut_card_remaining=self.cut_card_remaining, average_cards_per_hand=self.average_cards_per_hand, burn_cards=self.burn_cards)
        return ShoeDepthEstimate(hand_count=hand_count, estimated_cards_used=used, raw_remaining_cards=raw_remaining, remaining_cards=remaining, shoe_progress=legacy_progress, shoe_decks=self.shoe_decks, total_cards=self.total_cards, average_cards_per_hand=self.average_cards_per_hand, burn_cards=self.burn_cards, reference_hands=self.reference_hands, remaining_ratio=float(features["remaining_ratio"]), penetration=float(features["penetration"]), shoe_stage=str(features["shoe_stage"]), shoe_confidence_factor=float(features["shoe_confidence_factor"]), remaining_cards_reliability=float(features["remaining_cards_reliability"]), target_hands=int(features["target_hands"]), cut_card_remaining_cards=float(features["cut_card_remaining_cards"]), projected_total_hands=float(features["projected_total_hands"]), hands_remaining_to_cut=float(features["hands_remaining_to_cut"]), short_shoe_progress=float(features["short_shoe_progress"]))


__all__ = ["TOTAL_CARDS", "AVERAGE_CARDS_PER_HAND", "BURN_CARDS", "REFERENCE_HANDS", "SHOE_DECKS", "TARGET_HANDS_MIN", "TARGET_HANDS_MAX", "TARGET_HANDS_DEFAULT", "SHOE_STAGE_CONFIDENCE_FACTORS", "auto_cut_card_remaining", "resolve_cut_card_remaining", "classify_shoe_stage", "build_shoe_depth_features", "ShoeDepthEstimate", "ShoeDepthEstimator"]
