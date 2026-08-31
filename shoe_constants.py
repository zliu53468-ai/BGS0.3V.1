"""Authoritative baccarat shoe configuration and depth helpers.

All modules that need shoe deck count, total cards, average cards consumed per
resolved hand, reference maturity hands, or burn-card assumptions must import
from this module instead of defining their own magic numbers.

`REFERENCE_HANDS` is a product maturity/progress reference only. It is NOT a
physical cut-card position and must never be interpreted as "the shoe ends on
hand 70".
"""
from __future__ import annotations

import os
from typing import List

CARDS_PER_DECK = 52
POINT_COUNTS_PER_DECK = (16, 4, 4, 4, 4, 4, 4, 4, 4, 4)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


SHOE_DECKS = _env_int("SHOE_DECKS", 8, 1, 16)
TOTAL_SHOE_CARDS = CARDS_PER_DECK * SHOE_DECKS
AVERAGE_CARDS_PER_HAND = _env_float("AVERAGE_CARDS_PER_HAND", 4.9, 4.0, 6.0)
REFERENCE_HANDS = _env_int("REFERENCE_HANDS", 70, 1, 200)
BURN_CARDS = _env_int("BURN_CARDS", 0, 0, 20)


def normalize_decks(decks: int | None = None) -> int:
    value = SHOE_DECKS if decks is None else int(decks)
    return max(1, min(16, value))


def total_cards_for_decks(decks: int | None = None) -> int:
    return CARDS_PER_DECK * normalize_decks(decks)


def fresh_point_counts(decks: int | None = None) -> List[int]:
    """Return baccarat point-value counts 0..9 for a fresh shoe.

    Point 0 contains 10/J/Q/K: 16 cards per deck. Points 1..9 contain four
    cards per deck each.
    """
    count = normalize_decks(decks)
    return [per_deck * count for per_deck in POINT_COUNTS_PER_DECK]


def estimate_cards_used(
    hand_count: int,
    *,
    average_cards_per_hand: float = AVERAGE_CARDS_PER_HAND,
    burn_cards: int = BURN_CARDS,
) -> float:
    """Maturity/depth estimate only; never an exact composition calculation."""
    hands = max(0, int(hand_count))
    average = max(0.0, float(average_cards_per_hand))
    burn = max(0, int(burn_cards))
    return float(burn + hands * average)


def estimate_remaining_cards(
    hand_count: int,
    *,
    decks: int = SHOE_DECKS,
    average_cards_per_hand: float = AVERAGE_CARDS_PER_HAND,
    burn_cards: int = BURN_CARDS,
) -> float:
    """Estimate remaining total cards from round count only.

    Formula: max(0, total_cards - burn - hand_count * average_cards_per_hand).
    This result is a depth/maturity estimate and must not be promoted into
    `remaining_counts` or treated as exact card composition.
    """
    total = float(total_cards_for_decks(decks))
    used = estimate_cards_used(
        hand_count,
        average_cards_per_hand=average_cards_per_hand,
        burn_cards=burn_cards,
    )
    return max(0.0, total - used)


__all__ = [
    "AVERAGE_CARDS_PER_HAND",
    "BURN_CARDS",
    "CARDS_PER_DECK",
    "POINT_COUNTS_PER_DECK",
    "REFERENCE_HANDS",
    "SHOE_DECKS",
    "TOTAL_SHOE_CARDS",
    "estimate_cards_used",
    "estimate_remaining_cards",
    "fresh_point_counts",
    "normalize_decks",
    "total_cards_for_decks",
]
