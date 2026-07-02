"""
baccarat_simulator.py

百家樂 8 副牌模擬牌局引擎。
- 8 decks = 416 張，不放回抽取
- 實作標準 Player / Banker Third Card Rule
- 輸出每靴結果 list[str]，元素為 "B" / "P" / "T"

用法：
    from baccarat_simulator import simulate_shoe, simulate_many_shoes
    shoe = simulate_shoe()
    shoes = simulate_many_shoes(2000)
"""

from __future__ import annotations

import random
from typing import List, Optional

Result = str  # "B" / "P" / "T"


def build_shoe(n_decks: int = 8, rng: Optional[random.Random] = None) -> List[int]:
    """建立並洗牌。牌值規則：A=1, 2~9=原值, 10/J/Q/K=0。"""
    if n_decks <= 0:
        raise ValueError("n_decks must be positive")

    # 單副牌：A~9 各 4 張，10/J/Q/K 共 16 張且點數都算 0。
    one_deck = []
    for value in range(1, 10):
        one_deck.extend([value] * 4)
    one_deck.extend([0] * 16)

    shoe = one_deck * n_decks
    (rng or random).shuffle(shoe)
    return shoe


def hand_total(cards: List[int]) -> int:
    """百家樂點數只看個位數。"""
    return sum(cards) % 10


def banker_should_draw(banker_total: int, player_third_card: Optional[int]) -> bool:
    """標準莊家補牌規則。player_third_card=None 代表閒家未補第三張。"""
    if player_third_card is None:
        # 閒家停牌時，莊家 0~5 補，6~7 停。
        return banker_total <= 5

    p3 = player_third_card

    if banker_total <= 2:
        return True
    if banker_total == 3:
        return p3 != 8
    if banker_total == 4:
        return 2 <= p3 <= 7
    if banker_total == 5:
        return 4 <= p3 <= 7
    if banker_total == 6:
        return 6 <= p3 <= 7
    return False  # banker_total == 7 stands


def deal_one_round(shoe: List[int]) -> Result:
    """從 shoe 原地抽牌打一局，回傳 B/P/T。"""
    if len(shoe) < 6:
        raise ValueError("Not enough cards to safely deal one baccarat round")

    # 發牌順序：閒、莊、閒、莊。
    player = [shoe.pop(), shoe.pop()]
    banker = [shoe.pop(), shoe.pop()]

    p_total = hand_total(player)
    b_total = hand_total(banker)

    # Natural：任一方 8/9，雙方停牌。
    if p_total not in (8, 9) and b_total not in (8, 9):
        player_third: Optional[int] = None

        # 閒家 0~5 補，6~7 停。
        if p_total <= 5:
            player_third = shoe.pop()
            player.append(player_third)
            p_total = hand_total(player)

        # 莊家根據閒家是否補牌與第三張補牌規則判斷。
        if banker_should_draw(b_total, player_third):
            banker.append(shoe.pop())
            b_total = hand_total(banker)

    if b_total > p_total:
        return "B"
    if p_total > b_total:
        return "P"
    return "T"


def simulate_shoe(
    n_decks: int = 8,
    cut_card_remaining: int = 14,
    seed: Optional[int] = None,
) -> List[Result]:
    """
    回傳一整靴結果序列，通常約 60~80 局。

    cut_card_remaining：剩餘牌數小於等於此值就停止，避免最後牌不夠補牌。
    seed：可重現測試用；大量回測通常不用傳。
    """
    rng = random.Random(seed) if seed is not None else random
    shoe = build_shoe(n_decks=n_decks, rng=rng)
    results: List[Result] = []

    # 一局最多可能用 6 張，這裡留 cut_card_remaining 作為 cut card 安全區。
    while len(shoe) > max(6, cut_card_remaining):
        try:
            results.append(deal_one_round(shoe))
        except ValueError:
            break

    return results


def simulate_many_shoes(
    n_shoes: int,
    n_decks: int = 8,
    cut_card_remaining: int = 14,
    seed: Optional[int] = None,
) -> List[List[Result]]:
    """回傳多靴序列。seed 有傳入時，每次結果可重現。"""
    if n_shoes <= 0:
        raise ValueError("n_shoes must be positive")

    master_rng = random.Random(seed) if seed is not None else random
    shoes: List[List[Result]] = []
    for _ in range(n_shoes):
        shoe_seed = master_rng.randrange(0, 2**32) if seed is not None else None
        shoes.append(
            simulate_shoe(
                n_decks=n_decks,
                cut_card_remaining=cut_card_remaining,
                seed=shoe_seed,
            )
        )
    return shoes


if __name__ == "__main__":
    demo = simulate_shoe(seed=42)
    print(f"rounds={len(demo)}")
    print("".join(demo[:80]))
