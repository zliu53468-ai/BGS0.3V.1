"""BGS 輕量牌靴相容工具（粒子／超幾何／蒙地卡羅模型已移除）。

只保留 store.py 與既有虛擬牌靴介面需要的建靴、點數統計與發牌規則。
舊 ``VirtualShoeParticleEngine`` 名稱仍可匯入，但其 analyze 已轉接 cMAB。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import random
import secrets

from contextual_bandit import predict_bandit

DEFAULT_BASELINE = (0.458597, 0.446247, 0.095156)
OUTCOME_NAMES = ("B", "P", "T")
PATH_NAMES = ("none", "player_only", "banker_only", "both")
PATH_SUFFIXES = ("N", "P", "B", "D")
DB_HOLDOUT: Dict[str, Any] = {"status": "removed", "replacement": "CMAB-LINUCB-V1"}
DECKS = 8


def baccarat_total(cards: Iterable[int]) -> int:
    return sum(int(card) for card in cards) % 10


def fresh_counts(decks: int = DECKS) -> List[int]:
    count = max(1, min(16, int(decks)))
    return [16 * count] + [4 * count] * 9


def counts_from_shoe(shoe: Sequence[int]) -> List[int]:
    counts = [0] * 10
    for card in shoe:
        value = int(card)
        if not 0 <= value <= 9:
            raise ValueError("shoe card values must be between 0 and 9")
        counts[value] += 1
    return counts


def create_virtual_shoe(decks: int = DECKS, *, seed: Optional[int] = None) -> List[int]:
    cards: List[int] = []
    for value, count in enumerate(fresh_counts(decks)):
        cards.extend([value] * count)
    rng = random.Random(seed if seed is not None else secrets.randbits(64))
    rng.shuffle(cards)
    return cards


def player_should_draw(player_total: int) -> bool:
    return int(player_total) <= 5


def banker_should_draw(banker_total: int, player_third_card: Optional[int]) -> bool:
    total = int(banker_total)
    if player_third_card is None:
        return total <= 5
    third = int(player_third_card)
    if total <= 2:
        return True
    if total == 3:
        return third != 8
    if total == 4:
        return 2 <= third <= 7
    if total == 5:
        return 4 <= third <= 7
    if total == 6:
        return 6 <= third <= 7
    return False


@dataclass(frozen=True)
class HandResult:
    player_cards: List[int]
    banker_cards: List[int]
    player_total: int
    banker_total: int
    outcome: str
    draw_path: str
    natural: bool
    cards_used: int

    @property
    def outcome_text(self) -> str:
        return {"B": "莊", "P": "閒", "T": "和"}[self.outcome]

    @property
    def draw_path_text(self) -> str:
        return {"N": "雙方不補牌", "P": "閒補牌", "B": "莊補牌", "D": "莊閒皆補牌"}[self.draw_path]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "player_cards": list(self.player_cards),
            "banker_cards": list(self.banker_cards),
            "player_total": int(self.player_total),
            "banker_total": int(self.banker_total),
            "outcome": self.outcome,
            "outcome_text": self.outcome_text,
            "draw_path": self.draw_path,
            "draw_path_text": self.draw_path_text,
            "natural": bool(self.natural),
            "cards_used": int(self.cards_used),
            "player_third_card": self.player_cards[2] if len(self.player_cards) >= 3 else None,
            "banker_third_card": self.banker_cards[2] if len(self.banker_cards) >= 3 else None,
        }


def deal_ordered_hand(shoe: Sequence[int]) -> Tuple[HandResult, List[int]]:
    remaining = [int(card) for card in list(shoe)]
    if len(remaining) < 6:
        raise ValueError("shoe requires at least 6 cards")

    def draw() -> int:
        return int(remaining.pop(0))

    player = [draw()]
    banker = [draw()]
    player.append(draw())
    banker.append(draw())
    player_total = baccarat_total(player)
    banker_total = baccarat_total(banker)
    natural = player_total in {8, 9} or banker_total in {8, 9}
    player_drew = banker_drew = False
    player_third: Optional[int] = None

    if not natural:
        if player_should_draw(player_total):
            player_third = draw()
            player.append(player_third)
            player_drew = True
            player_total = baccarat_total(player)
        if banker_should_draw(banker_total, player_third):
            banker.append(draw())
            banker_drew = True
            banker_total = baccarat_total(banker)

    outcome = "P" if player_total > banker_total else "B" if banker_total > player_total else "T"
    draw_path = "D" if player_drew and banker_drew else "P" if player_drew else "B" if banker_drew else "N"
    result = HandResult(
        player_cards=player,
        banker_cards=banker,
        player_total=player_total,
        banker_total=banker_total,
        outcome=outcome,
        draw_path=draw_path,
        natural=natural,
        cards_used=len(player) + len(banker),
    )
    return result, remaining


def simulate_one_from_counts(remaining_counts: Sequence[int], *, seed: Optional[int] = None) -> HandResult:
    if len(remaining_counts) != 10:
        raise ValueError("remaining_counts must contain 10 values")
    shoe: List[int] = []
    for value, count in enumerate(remaining_counts):
        shoe.extend([value] * max(0, int(count)))
    rng = random.Random(seed if seed is not None else secrets.randbits(64))
    rng.shuffle(shoe)
    hand, _ = deal_ordered_hand(shoe)
    return hand


@dataclass(frozen=True)
class EngineSettings:
    decks: int = 8
    particles: int = 0
    replicas: int = 0
    simulations_per_replica: int = 0
    particle_draws_per_particle: int = 0


class VirtualShoeParticleEngine:
    """舊類別名稱相容層；內部已改呼叫 cMAB。"""

    def __init__(self, settings: Optional[EngineSettings] = None) -> None:
        self.settings = settings or EngineSettings()

    def analyze(self, *, remaining_counts: Optional[Sequence[int]] = None,
                history: Optional[Iterable[Any]] = None,
                draw_path_history: Optional[Iterable[Any]] = None,
                seed: Optional[int] = None,
                road_context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        result = predict_bandit(list(history or []), road_context=dict(road_context or {}), run_seed=seed)
        result.update({
            "compatibility_adapter": True,
            "legacy_class_name": "VirtualShoeParticleEngine",
            "remaining_counts_ignored": bool(remaining_counts),
            "draw_path_history_ignored": bool(draw_path_history),
            "particle_model_active": False,
            "hypergeometric_model_active": False,
            "monte_carlo_model_active": False,
        })
        return result


__all__ = [
    "DB_HOLDOUT", "DECKS", "EngineSettings", "HandResult", "VirtualShoeParticleEngine",
    "baccarat_total", "banker_should_draw", "counts_from_shoe", "create_virtual_shoe",
    "deal_ordered_hand", "fresh_counts", "player_should_draw", "simulate_one_from_counts",
]
