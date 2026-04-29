# -*- coding: utf-8 -*-
"""
bgs/sim_dataset_builder.py

百萬局百家樂模擬資料產生器
用途：
1. 模擬 8 副牌百家樂牌靴
2. 依照百家樂補牌規則產生莊 / 閒 / 和結果
3. 統計不同牌靴階段、上一局結果、連莊連閒、點數後的下一局分布
4. 輸出 bgs/calibrator_stats.json 給 stat_calibrator.py 使用

執行方式：
    python bgs/sim_dataset_builder.py

可調環境變數：
    SIM_TARGET_ROUNDS=1000000
    SIM_DECKS=8
    SIM_SEED=42
"""

import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calibrator_stats.json"
)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def card_value(card: int) -> int:
    """
    百家樂點數：
    A = 1
    2~9 = 2~9
    10/J/Q/K = 0
    """
    if card == 1:
        return 1
    if 2 <= card <= 9:
        return card
    return 0


def make_shoe(decks: int = 8) -> List[int]:
    """
    建立 8 副牌牌靴
    用 1~13 表示 A~K
    """
    shoe = []
    for _ in range(decks):
        for rank in range(1, 14):
            shoe.extend([rank] * 4)
    random.shuffle(shoe)
    return shoe


def hand_total(cards: List[int]) -> int:
    return sum(card_value(c) for c in cards) % 10


def draw_card(shoe: List[int]) -> int:
    return shoe.pop()


def simulate_round(shoe: List[int]) -> Dict[str, Any]:
    """
    模擬一局百家樂
    回傳：
        result: B / P / T
        banker_point
        player_point
        cards_used
    """

    start_len = len(shoe)

    player_cards = [draw_card(shoe), draw_card(shoe)]
    banker_cards = [draw_card(shoe), draw_card(shoe)]

    player_total = hand_total(player_cards)
    banker_total = hand_total(banker_cards)

    # 天牌：任一方 8 / 9，雙方不補牌
    if player_total in (8, 9) or banker_total in (8, 9):
        pass
    else:
        player_third_value: Optional[int] = None

        # 閒家補牌
        if player_total <= 5:
            third = draw_card(shoe)
            player_cards.append(third)
            player_third_value = card_value(third)
            player_total = hand_total(player_cards)

        # 莊家補牌
        banker_total_before = banker_total

        if player_third_value is None:
            # 閒家未補牌，莊家 0~5 補，6~7 停
            if banker_total_before <= 5:
                banker_cards.append(draw_card(shoe))
        else:
            pt = player_third_value

            if banker_total_before <= 2:
                banker_cards.append(draw_card(shoe))
            elif banker_total_before == 3 and pt != 8:
                banker_cards.append(draw_card(shoe))
            elif banker_total_before == 4 and pt in (2, 3, 4, 5, 6, 7):
                banker_cards.append(draw_card(shoe))
            elif banker_total_before == 5 and pt in (4, 5, 6, 7):
                banker_cards.append(draw_card(shoe))
            elif banker_total_before == 6 and pt in (6, 7):
                banker_cards.append(draw_card(shoe))

        banker_total = hand_total(banker_cards)

    player_total = hand_total(player_cards)
    banker_total = hand_total(banker_cards)

    if banker_total > player_total:
        result = "B"
    elif player_total > banker_total:
        result = "P"
    else:
        result = "T"

    return {
        "result": result,
        "banker_point": banker_total,
        "player_point": player_total,
        "cards_used": start_len - len(shoe),
    }


class StatCounter:
    def __init__(self):
        self.banker = 0
        self.player = 0
        self.tie = 0
        self.samples = 0

    def add(self, result: str):
        self.samples += 1
        if result == "B":
            self.banker += 1
        elif result == "P":
            self.player += 1
        elif result == "T":
            self.tie += 1

    def to_dict(self) -> Dict[str, Any]:
        if self.samples <= 0:
            return {
                "banker": 0.0,
                "player": 0.0,
                "tie": 0.0,
                "samples": 0,
            }

        return {
            "banker": round(self.banker / self.samples, 8),
            "player": round(self.player / self.samples, 8),
            "tie": round(self.tie / self.samples, 8),
            "samples": self.samples,
        }


def stage_from_shoe_pos(shoe_pos: float) -> str:
    if shoe_pos < 0.33:
        return "early"
    if shoe_pos < 0.70:
        return "mid"
    return "late"


def streak_key(side: Optional[str], length: int) -> Optional[str]:
    if side not in ("B", "P"):
        return None

    if length < 2:
        return None

    if length >= 4:
        return f"{side}4+"

    return f"{side}{length}"


def point_key(last_result: str, banker_point: int, player_point: int) -> str:
    return f"{last_result}_B{banker_point}_P{player_point}"


def build_stats(
    target_rounds: int = 1_000_000,
    decks: int = 8,
    seed: int = 42,
    min_cards_left: int = 20,
) -> Dict[str, Any]:
    """
    建立統計資料。

    注意：
    這裡統計的是「上一局狀態」對「下一局結果」的分布。
    也就是你前端輸入上一局點數後，機器人要預測下一局時可以查的校準表。
    """

    random.seed(seed)

    global_counter = StatCounter()
    stage_counter = defaultdict(StatCounter)
    last_result_counter = defaultdict(StatCounter)
    streak_counter = defaultdict(StatCounter)
    points_counter = defaultdict(StatCounter)

    generated_rounds = 0
    shoe_count = 0

    while generated_rounds < target_rounds:
        shoe = make_shoe(decks)
        shoe_count += 1
        full_shoe_cards = len(shoe)

        previous_context: Optional[Dict[str, Any]] = None

        current_streak_side: Optional[str] = None
        current_streak_len = 0

        while len(shoe) > min_cards_left and generated_rounds < target_rounds:
            before_cards_left = len(shoe)
            shoe_pos = 1.0 - (before_cards_left / full_shoe_cards)
            stage = stage_from_shoe_pos(shoe_pos)

            current_round = simulate_round(shoe)
            current_result = current_round["result"]

            # 如果已經有上一局狀態，就用「上一局狀態」統計「這一局結果」
            if previous_context is not None:
                global_counter.add(current_result)

                stage_counter[previous_context["stage"]].add(current_result)

                last_result_counter[previous_context["last_result"]].add(current_result)

                sk = previous_context.get("streak_key")
                if sk:
                    streak_counter[sk].add(current_result)

                pk = previous_context.get("point_key")
                if pk:
                    points_counter[pk].add(current_result)

                generated_rounds += 1

                if generated_rounds % 100000 == 0:
                    print(f"[sim] generated {generated_rounds:,} rounds... shoes={shoe_count}")

            # 更新目前連莊 / 連閒狀態
            if current_result in ("B", "P"):
                if current_result == current_streak_side:
                    current_streak_len += 1
                else:
                    current_streak_side = current_result
                    current_streak_len = 1

            # 和局通常不打斷莊閒 streak，這邊維持不變
            skey = streak_key(current_streak_side, current_streak_len)

            # 這一局會變成下一局的 previous_context
            previous_context = {
                "stage": stage,
                "last_result": current_result,
                "banker_point": current_round["banker_point"],
                "player_point": current_round["player_point"],
                "streak_key": skey,
                "point_key": point_key(
                    current_result,
                    current_round["banker_point"],
                    current_round["player_point"],
                ),
            }

    stats = {
        "meta": {
            "target_rounds": target_rounds,
            "actual_samples": global_counter.samples,
            "decks": decks,
            "seed": seed,
            "note": "Stats are next-round distributions conditioned on previous round context.",
        },
        "global": global_counter.to_dict(),
        "stage": {
            "early": stage_counter["early"].to_dict(),
            "mid": stage_counter["mid"].to_dict(),
            "late": stage_counter["late"].to_dict(),
        },
        "last_result": {
            "B": last_result_counter["B"].to_dict(),
            "P": last_result_counter["P"].to_dict(),
            "T": last_result_counter["T"].to_dict(),
        },
        "streak": {
            key: counter.to_dict()
            for key, counter in sorted(streak_counter.items())
        },
        "points": {
            key: counter.to_dict()
            for key, counter in sorted(points_counter.items())
        },
    }

    return stats


def save_stats(stats: Dict[str, Any], output_path: str = DEFAULT_OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[sim] saved stats to: {output_path}")


def main():
    target_rounds = env_int("SIM_TARGET_ROUNDS", 1_000_000)
    decks = env_int("SIM_DECKS", 8)
    seed = env_int("SIM_SEED", 42)

    print("[sim] start baccarat simulation")
    print(f"[sim] target_rounds={target_rounds:,}, decks={decks}, seed={seed}")

    stats = build_stats(
        target_rounds=target_rounds,
        decks=decks,
        seed=seed,
    )

    save_stats(stats)

    print("[sim] done")
    print(json.dumps(stats.get("global", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
