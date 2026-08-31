"""Virtual-shoe compatibility module for BGS V6.

This file keeps virtual-shoe operations in one import location while reusing
the tested baccarat rules and particle engine from ``particle_filter_points``.
It is a simulation utility only and has no connection to an external live table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import os
import secrets

from shoe_constants import SHOE_DECKS

from particle_filter_points import (
    EngineSettings,
    HandResult,
    VirtualShoeParticleEngine,
    baccarat_total,
    banker_should_draw,
    counts_from_shoe,
    create_virtual_shoe,
    fresh_counts,
    simulate_one_from_counts,
)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


DEFAULT_DECKS = SHOE_DECKS
DEFAULT_CUT_CARD_MIN = _env_int("VIRTUAL_CUT_CARD_MIN", 60, 12, 160)
DEFAULT_CUT_CARD_MAX = _env_int("VIRTUAL_CUT_CARD_MAX", 85, 12, 180)


@dataclass
class VirtualShoe:
    """Mutable hidden-order virtual shoe.

    ``cards`` contains point values only: 0 represents 10/J/Q/K and 1-9
    represent their normal baccarat point values.
    """

    cards: List[int]
    decks: int = DEFAULT_DECKS
    cut_card_remaining: int = 75
    hand_number: int = 0

    @classmethod
    def new(
        cls,
        decks: int = DEFAULT_DECKS,
        *,
        seed: Optional[int] = None,
        cut_card_remaining: Optional[int] = None,
    ) -> "VirtualShoe":
        decks = max(1, min(16, int(decks)))
        if cut_card_remaining is None:
            low = min(DEFAULT_CUT_CARD_MIN, DEFAULT_CUT_CARD_MAX)
            high = max(DEFAULT_CUT_CARD_MIN, DEFAULT_CUT_CARD_MAX)
            cut_card_remaining = low + secrets.randbelow(high - low + 1)
        return cls(
            cards=create_virtual_shoe(decks=decks, seed=seed),
            decks=decks,
            cut_card_remaining=max(12, int(cut_card_remaining)),
            hand_number=0,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VirtualShoe":
        cards = [int(card) for card in list(value.get("cards") or [])]
        if not cards or any(card < 0 or card > 9 for card in cards):
            raise ValueError("無效的虛擬牌靴資料。")
        decks = max(1, min(16, int(value.get("decks") or DEFAULT_DECKS)))
        return cls(
            cards=cards,
            decks=decks,
            cut_card_remaining=max(
                12,
                int(value.get("cut_card_remaining") or 75),
            ),
            hand_number=max(0, int(value.get("hand_number") or 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cards": list(self.cards),
            "decks": int(self.decks),
            "cut_card_remaining": int(self.cut_card_remaining),
            "hand_number": int(self.hand_number),
            "remaining_counts": self.remaining_counts,
            "remaining_cards": self.remaining_cards,
        }

    @property
    def remaining_cards(self) -> int:
        return len(self.cards)

    @property
    def remaining_counts(self) -> List[int]:
        return counts_from_shoe(self.cards)

    @property
    def needs_shuffle(self) -> bool:
        return len(self.cards) <= max(12, int(self.cut_card_remaining))

    def reshuffle(self, *, seed: Optional[int] = None) -> None:
        replacement = VirtualShoe.new(
            decks=self.decks,
            seed=seed,
        )
        self.cards = replacement.cards
        self.cut_card_remaining = replacement.cut_card_remaining
        self.hand_number = 0

    def deal(self) -> HandResult:
        """Deal one hand from the hidden order using baccarat drawing rules."""
        if len(self.cards) < 6:
            raise ValueError("虛擬牌靴剩餘牌數不足。")

        index = 0

        def draw_card() -> int:
            nonlocal index
            if index >= len(self.cards):
                raise ValueError("虛擬牌靴在發牌途中用盡。")
            card = int(self.cards[index])
            index += 1
            return card

        player = [draw_card(), draw_card()]
        banker = [draw_card(), draw_card()]

        player_total = baccarat_total(player)
        banker_total = baccarat_total(banker)
        player_drew = False
        banker_drew = False

        if player_total not in {8, 9} and banker_total not in {8, 9}:
            player_third: Optional[int] = None
            if player_total <= 5:
                player_third = draw_card()
                player.append(player_third)
                player_drew = True

            if banker_should_draw(banker_total, player_third):
                banker.append(draw_card())
                banker_drew = True

        player_total = baccarat_total(player)
        banker_total = baccarat_total(banker)
        outcome = (
            "B"
            if banker_total > player_total
            else "P"
            if player_total > banker_total
            else "T"
        )
        draw_path = (
            "D"
            if player_drew and banker_drew
            else "P"
            if player_drew
            else "B"
            if banker_drew
            else "N"
        )

        self.cards = self.cards[index:]
        self.hand_number += 1

        return HandResult(
            player_cards=tuple(player),
            banker_cards=tuple(banker),
            player_total=player_total,
            banker_total=banker_total,
            outcome=outcome,
            draw_path=draw_path,
            cards_used=index,
        )


def deal_ordered_hand(shoe: Sequence[int]) -> Tuple[HandResult, List[int]]:
    """Compatibility helper: deal one hand and return the remaining ordered shoe."""
    state = VirtualShoe(
        cards=[int(card) for card in list(shoe)],
        decks=DEFAULT_DECKS,
        cut_card_remaining=12,
        hand_number=0,
    )
    hand = state.deal()
    return hand, list(state.cards)


def build_particle_engine() -> VirtualShoeParticleEngine:
    """Create the configured particle/Monte Carlo analysis engine."""
    return VirtualShoeParticleEngine(
        EngineSettings(
            decks=DEFAULT_DECKS,
            particles=_env_int("PF_PARTICLES", 500, 64, 4000),
            replicas=_env_int("PF_REPLICAS", 5, 3, 11),
            simulations_per_replica=_env_int(
                "PF_PREDICT_SIMULATIONS_PER_REPLICA",
                1200,
                200,
                20_000,
            ),
            particle_draws_per_particle=_env_int(
                "PF_DRAWS_PER_PARTICLE",
                2,
                1,
                12,
            ),
        )
    )


def analyze_virtual_shoe(
    shoe: VirtualShoe,
    *,
    outcome_history: Optional[Sequence[str]] = None,
    draw_path_history: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    engine: Optional[VirtualShoeParticleEngine] = None,
) -> Dict[str, Any]:
    """Estimate the next-hand probabilities without revealing hidden order."""
    selected_engine = engine or build_particle_engine()
    return selected_engine.analyze(
        remaining_counts=shoe.remaining_counts,
        history=list(outcome_history or []),
        draw_path_history=list(draw_path_history or []),
        seed=(int(seed) & 0xFFFFFFFF) if seed is not None else secrets.randbits(32),
    )


def analyze_then_deal(
    shoe: VirtualShoe,
    *,
    outcome_history: Optional[Sequence[str]] = None,
    draw_path_history: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    engine: Optional[VirtualShoeParticleEngine] = None,
) -> Dict[str, Any]:
    """Analyze first, then consume one hidden virtual hand."""
    prediction = analyze_virtual_shoe(
        shoe,
        outcome_history=outcome_history,
        draw_path_history=draw_path_history,
        seed=seed,
        engine=engine,
    )
    hand = shoe.deal()
    return {
        "prediction": prediction,
        "hand": hand.as_dict(),
        "shoe": shoe.to_dict(),
        "needs_shuffle": shoe.needs_shuffle,
    }


__all__ = [
    "DEFAULT_DECKS",
    "EngineSettings",
    "HandResult",
    "VirtualShoe",
    "VirtualShoeParticleEngine",
    "analyze_then_deal",
    "analyze_virtual_shoe",
    "banker_should_draw",
    "baccarat_total",
    "build_particle_engine",
    "counts_from_shoe",
    "create_virtual_shoe",
    "deal_ordered_hand",
    "fresh_counts",
    "simulate_one_from_counts",
]
