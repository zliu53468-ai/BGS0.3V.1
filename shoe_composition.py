"""精確牌靴組成、下一手不放回機率、抽水後 EV 與 B/P-only Kelly。

資料優先順序：
1. remaining_counts：點數 0..9 的精確剩餘張數。
2. observed_cards：本靴已揭曉牌面，從新牌靴逐張扣除。
3. 兩者皆無或輸入無效：available=False，交由上層 road fallback。

本模組不產生 O/觀望臂；有精確牌組時永遠回傳 EV 較佳的 B/P。
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import os

from shoe_constants import (
    AVERAGE_CARDS_PER_HAND,
    SHOE_DECKS,
    fresh_point_counts,
)

# Compatibility alias; authoritative value lives in shoe_constants.py.
DECKS = SHOE_DECKS
BANKER_COMMISSION = min(
    0.20,
    max(0.0, float(os.getenv("BANKER_COMMISSION", "0.05") or "0.05")),
)
MIN_POSITIVE_EV = max(
    0.0,
    float(os.getenv("PHYSICAL_MIN_EV", "0.002") or "0.002"),
)
KELLY_FRACTION = min(
    1.0,
    max(0.0, float(os.getenv("KELLY_FRACTION", "0.25") or "0.25")),
)
WEAK_EV_UPPER_BOUND = max(
    MIN_POSITIVE_EV,
    float(os.getenv("WEAK_EV_UPPER_BOUND", "0.005") or "0.005"),
)
WEAK_EV_KELLY_FRACTION = min(
    KELLY_FRACTION,
    max(
        0.0,
        float(os.getenv("WEAK_EV_KELLY_FRACTION", "0.125") or "0.125"),
    ),
)
MIN_BET_FRACTION = max(
    0.05,
    min(0.30, float(os.getenv("MIN_BET_FRACTION", "0.05") or "0.05")),
)
MAX_BET_FRACTION = min(
    0.30,
    max(MIN_BET_FRACTION, float(os.getenv("MAX_BET_FRACTION", "0.30") or "0.30")),
)

STANDARD_EIGHT_DECK_BASELINE = {
    "B": 0.458597,
    "P": 0.446247,
    "T": 0.095156,
}


def fresh_counts(decks: int = DECKS) -> List[int]:
    """回傳點數 0..9 的新牌靴張數；權威張數表由 shoe_constants 提供。"""
    return fresh_point_counts(decks)


def parse_card_value(value: Any) -> int:
    """把實際牌面轉為百家樂點數。"""
    if isinstance(value, bool):
        raise ValueError("布林值不是合法牌面")
    if isinstance(value, int):
        point = value
    elif isinstance(value, float) and value.is_integer():
        point = int(value)
    else:
        raw = str(value or "").strip().upper()
        aliases = {
            "A": 1,
            "ACE": 1,
            "T": 0,
            "10": 0,
            "J": 0,
            "Q": 0,
            "K": 0,
        }
        if raw in aliases:
            return aliases[raw]
        if not raw.isdigit():
            raise ValueError(f"無法辨識牌面：{value!r}")
        point = int(raw)
    if point == 10:
        return 0
    if not 0 <= point <= 9:
        raise ValueError("牌點必須是 0..9，或 A/10/J/Q/K")
    return point


def remaining_counts_from_observed(
    observed_cards: Iterable[Any],
    *,
    decks: int = DECKS,
) -> List[int]:
    """由每張已觀察牌面從新牌靴精確扣除。"""
    remaining = fresh_counts(decks)
    for raw in observed_cards:
        point = parse_card_value(raw)
        remaining[point] -= 1
        if remaining[point] < 0:
            raise ValueError(
                f"牌點 {point} 的已輸入張數超過 {decks} 副牌上限"
            )
    return remaining


def _mapping_counts(values: Mapping[Any, Any]) -> List[Any]:
    result: List[Any] = []
    for point in range(10):
        if point in values:
            result.append(values[point])
        elif str(point) in values:
            result.append(values[str(point)])
        else:
            raise ValueError("remaining_counts mapping 必須包含 0..9 全部鍵")
    return result


def validate_remaining_counts(
    values: Sequence[Any] | Mapping[Any, Any],
    *,
    decks: int = DECKS,
) -> Tuple[int, ...]:
    if isinstance(values, Mapping):
        raw_values = _mapping_counts(values)
    else:
        raw_values = list(values)
    if len(raw_values) != 10:
        raise ValueError("remaining_counts 必須依序包含點數 0..9 共 10 個數值")
    maximum = fresh_counts(decks)
    counts: List[int] = []
    for point, raw in enumerate(raw_values):
        if isinstance(raw, bool):
            raise ValueError("remaining_counts 必須是非負整數")
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            raise ValueError("remaining_counts 必須是非負整數") from None
        if not numeric.is_integer():
            raise ValueError("remaining_counts 必須是非負整數")
        count = int(numeric)
        if count < 0 or count > maximum[point]:
            raise ValueError(
                f"點數 {point} 的剩餘張數超出 {decks} 副牌範圍"
            )
        counts.append(count)
    if sum(counts) < 6:
        raise ValueError("剩餘牌不足以安全完成一手百家樂")
    return tuple(counts)


def _banker_should_draw(
    banker_total: int,
    player_third_card: Optional[int],
) -> bool:
    if player_third_card is None:
        return banker_total <= 5
    if banker_total <= 2:
        return True
    if banker_total == 3:
        return player_third_card != 8
    if banker_total == 4:
        return 2 <= player_third_card <= 7
    if banker_total == 5:
        return 4 <= player_third_card <= 7
    if banker_total == 6:
        return 6 <= player_third_card <= 7
    return False


def _draw_branches(counts: Tuple[int, ...]):
    total = sum(counts)
    if total <= 0:
        return
    for point, count in enumerate(counts):
        if count <= 0:
            continue
        next_counts = list(counts)
        next_counts[point] -= 1
        yield point, count / total, tuple(next_counts)


@lru_cache(maxsize=1024)
def exact_next_hand_probabilities(
    counts: Tuple[int, ...],
) -> Tuple[float, float, float]:
    """依正式補牌規則完整枚舉下一手不放回 B/P/T 機率。"""
    if (
        len(counts) != 10
        or sum(counts) < 6
        or any(value < 0 for value in counts)
    ):
        raise ValueError("無效的剩餘牌組")

    states: Dict[
        Tuple[Tuple[int, ...], int, int],
        float,
    ] = {(counts, 0, 0): 1.0}
    for owner in ("P", "B", "P", "B"):
        next_states: DefaultDict[
            Tuple[Tuple[int, ...], int, int],
            float,
        ] = defaultdict(float)
        for (
            state_counts,
            player_total,
            banker_total,
        ), state_probability in states.items():
            for point, draw_probability, after_draw in _draw_branches(
                state_counts
            ):
                if owner == "P":
                    key = (
                        after_draw,
                        (player_total + point) % 10,
                        banker_total,
                    )
                else:
                    key = (
                        after_draw,
                        player_total,
                        (banker_total + point) % 10,
                    )
                next_states[key] += (
                    state_probability * draw_probability
                )
        states = dict(next_states)

    outcome: DefaultDict[str, float] = defaultdict(float)

    def settle(
        player_total: int,
        banker_total: int,
        probability: float,
    ) -> None:
        code = (
            "P"
            if player_total > banker_total
            else "B"
            if banker_total > player_total
            else "T"
        )
        outcome[code] += probability

    for (
        state_counts,
        player_total,
        banker_total,
    ), state_probability in states.items():
        if player_total in {8, 9} or banker_total in {8, 9}:
            settle(
                player_total,
                banker_total,
                state_probability,
            )
            continue

        if player_total <= 5:
            for (
                player_third,
                p_draw,
                after_player,
            ) in _draw_branches(state_counts):
                new_player_total = (
                    player_total + player_third
                ) % 10
                branch_probability = (
                    state_probability * p_draw
                )
                if _banker_should_draw(
                    banker_total,
                    player_third,
                ):
                    for (
                        banker_third,
                        b_draw,
                        _,
                    ) in _draw_branches(after_player):
                        settle(
                            new_player_total,
                            (
                                banker_total
                                + banker_third
                            )
                            % 10,
                            branch_probability * b_draw,
                        )
                else:
                    settle(
                        new_player_total,
                        banker_total,
                        branch_probability,
                    )
        elif _banker_should_draw(
            banker_total,
            None,
        ):
            for (
                banker_third,
                b_draw,
                _,
            ) in _draw_branches(state_counts):
                settle(
                    player_total,
                    (
                        banker_total + banker_third
                    )
                    % 10,
                    state_probability * b_draw,
                )
        else:
            settle(
                player_total,
                banker_total,
                state_probability,
            )

    total_probability = sum(outcome.values())
    if total_probability <= 0.0:
        raise ValueError("無法計算下一手機率")
    return (
        outcome["B"] / total_probability,
        outcome["P"] / total_probability,
        outcome["T"] / total_probability,
    )


def expected_returns(
    probabilities: Mapping[str, float],
) -> Dict[str, float]:
    """單位下注 EV；和局為 push，莊注計 5% 抽水。"""
    banker_probability = float(
        probabilities.get("B", 0.0) or 0.0
    )
    player_probability = float(
        probabilities.get("P", 0.0) or 0.0
    )
    banker_net_win = 1.0 - BANKER_COMMISSION
    return {
        "B": (
            banker_net_win * banker_probability
            - player_probability
        ),
        "P": player_probability - banker_probability,
        "T": None,
    }


def _full_kelly_fraction(
    *,
    side: str,
    probabilities: Mapping[str, float],
) -> float:
    code = str(side or "").upper()
    if code not in {"B", "P"}:
        raise ValueError("Kelly side 必須是 B 或 P")
    p_win = float(probabilities.get(code, 0.0) or 0.0)
    p_loss = float(
        probabilities.get(
            "P" if code == "B" else "B",
            0.0,
        )
        or 0.0
    )
    odds = (
        1.0 - BANKER_COMMISSION
        if code == "B"
        else 1.0
    )
    resolved_probability = p_win + p_loss
    if resolved_probability <= 0.0:
        return 0.0
    return (
        odds * p_win - p_loss
    ) / (odds * resolved_probability)


def _applied_kelly_fraction(
    *,
    expected_return: float,
) -> float:
    value = float(expected_return)
    if (
        MIN_POSITIVE_EV
        <= value
        <= WEAK_EV_UPPER_BOUND
    ):
        return float(WEAK_EV_KELLY_FRACTION)
    return float(KELLY_FRACTION)


def kelly_fraction(
    *,
    side: str,
    probabilities: Mapping[str, float],
) -> float:
    """分數 Kelly，正式產品規格硬限制在 5%～30%。"""
    returns = expected_returns(probabilities)
    selected_ev = float(returns[str(side).upper()])
    full_kelly = max(
        0.0,
        _full_kelly_fraction(
            side=side,
            probabilities=probabilities,
        ),
    )
    applied = _applied_kelly_fraction(
        expected_return=selected_ev
    )
    raw_fractional = full_kelly * applied
    return min(
        MAX_BET_FRACTION,
        max(MIN_BET_FRACTION, raw_fractional),
    )


def _composition_summary(
    counts: Sequence[int],
    decks: int,
) -> Dict[str, Any]:
    baseline = fresh_counts(decks)
    total = sum(counts)
    zero = counts[0]
    small = sum(counts[1:5])
    large = sum(counts[5:10])
    baseline_total = sum(baseline)
    return {
        "remaining_cards": total,
        "zero_value_count": zero,
        "small_1_to_4_count": small,
        "large_5_to_9_count": large,
        "zero_value_ratio": zero / total,
        "small_1_to_4_ratio": small / total,
        "large_5_to_9_ratio": large / total,
        "small_ratio_delta_from_fresh": (
            small / total
            - sum(baseline[1:5]) / baseline_total
        ),
        "large_ratio_delta_from_fresh": (
            large / total
            - sum(baseline[5:10]) / baseline_total
        ),
    }


def _unavailable(
    *,
    source: str,
    reason_code: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "available": False,
        "action": None,
        "action_text": "牌靴資料不可用，交由牌路模型",
        "formal_no_observe_arm": True,
        "source": source,
        "reason_code": reason_code,
        "reason": reason,
        "expected_returns": {
            "B": None,
            "P": None,
            "T": None,
        },
        "selected_side_by_ev": None,
        "selected_expected_return": None,
        "best_raw_expected_return": None,
        "kelly_fraction": 0.0,
        "recommended_bet_percentage": 0.0,
        "minimum_positive_ev": MIN_POSITIVE_EV,
    }


def analyze_shoe_composition(
    shoe_context: Optional[Mapping[str, Any]],
    *,
    default_decks: int = DECKS,
) -> Dict[str, Any]:
    """解析精確牌組並回傳 B/P-only 正式物理方向。"""
    context = dict(shoe_context or {})
    decks = max(
        1,
        min(
            16,
            int(
                context.get("decks", default_decks)
                or default_decks
            ),
        ),
    )
    source = "none"
    try:
        raw_counts = context.get("remaining_counts")
        observed = context.get("observed_cards")

        if isinstance(
            raw_counts,
            (Sequence, Mapping),
        ) and not isinstance(
            raw_counts,
            (str, bytes),
        ):
            counts = validate_remaining_counts(
                raw_counts,
                decks=decks,
            )
            source = "remaining_counts"
        elif isinstance(
            observed,
            Iterable,
        ) and not isinstance(
            observed,
            (str, bytes, Mapping),
        ):
            observed_list = list(observed)
            if not observed_list:
                return _unavailable(
                    source="none",
                    reason_code="NO_CARD_COMPOSITION",
                    reason="observed_cards 為空，正式方向回退牌路模型。",
                )
            counts = validate_remaining_counts(
                remaining_counts_from_observed(
                    observed_list,
                    decks=decks,
                ),
                decks=decks,
            )
            source = "observed_cards"
        else:
            return _unavailable(
                source="none",
                reason_code="NO_CARD_COMPOSITION",
                reason=(
                    "未提供 remaining_counts 或 observed_cards；"
                    "正式方向回退牌路模型。"
                ),
            )

        banker, player, tie = (
            exact_next_hand_probabilities(counts)
        )
        probabilities = {
            "B": banker,
            "P": player,
            "T": tie,
        }
        returns = expected_returns(probabilities)
        selected_side = max(
            ("B", "P"),
            key=lambda side: float(returns[side]),
        )
        selected_ev = float(returns[selected_side])
        full_kelly_raw = _full_kelly_fraction(
            side=selected_side,
            probabilities=probabilities,
        )
        full_kelly = max(0.0, full_kelly_raw)
        applied_kelly = _applied_kelly_fraction(
            expected_return=selected_ev
        )
        fraction = kelly_fraction(
            side=selected_side,
            probabilities=probabilities,
        )
        baseline = STANDARD_EIGHT_DECK_BASELINE

        return {
            "available": True,
            "action": selected_side,
            "action_text": (
                "莊"
                if selected_side == "B"
                else "閒"
            ),
            "formal_direction": selected_side,
            "formal_no_observe_arm": True,
            "source": source,
            "input_source_label": str(
                context.get("source") or source
            ),
            "decks": decks,
            "shoe_decks": decks,
            "remaining_cards": int(sum(counts)),
            "remaining_cards_source": (
                "exact_counts" if source == "remaining_counts" else "observed_cards"
            ),
            "average_cards_per_hand": float(AVERAGE_CARDS_PER_HAND),
            "remaining_cards_semantics": "exact_from_card_composition",
            "remaining_counts": list(counts),
            "probabilities": probabilities,
            "probability_delta_from_standard": {
                side: (
                    probabilities[side]
                    - baseline[side]
                )
                for side in ("B", "P", "T")
            },
            "composition": _composition_summary(
                counts,
                decks,
            ),
            "banker_commission": BANKER_COMMISSION,
            "expected_returns": returns,
            "banker_ev": float(returns["B"]),
            "player_ev": float(returns["P"]),
            "selected_side_by_ev": selected_side,
            "selected_expected_return": selected_ev,
            "best_raw_expected_return": selected_ev,
            "minimum_positive_ev": MIN_POSITIVE_EV,
            "ev_is_positive": selected_ev > 0.0,
            "ev_meets_legacy_threshold": (
                selected_ev >= MIN_POSITIVE_EV
            ),
            "kelly_method": (
                "fractional_kelly_forced_clip_5_to_30_percent"
            ),
            "applied_kelly_fraction": applied_kelly,
            "weak_ev_upper_bound": WEAK_EV_UPPER_BOUND,
            "weak_ev_kelly_fraction": (
                WEAK_EV_KELLY_FRACTION
            ),
            "full_kelly_before_fraction": full_kelly,
            "raw_full_kelly_before_floor": (
                full_kelly_raw
            ),
            "fractional_kelly_before_cap": (
                full_kelly * applied_kelly
            ),
            "kelly_fraction": fraction,
            "recommended_bet_percentage": (
                fraction * 100.0
            ),
            "risk_gate_open": True,
            "reason_code": "EXACT_SHOE_EV_BP_ONLY",
            "reason": (
                "精確不放回牌組可計算；"
                "正式方向永遠選抽水後 EV 較佳的 B/P，"
                "低或負 EV 僅使 Kelly 靠近 5% 下限，不改成觀望。"
            ),
        }
    except (TypeError, ValueError) as exc:
        return _unavailable(
            source="none",
            reason_code="INVALID_CARD_COMPOSITION",
            reason=str(exc),
        )


__all__ = [
    "BANKER_COMMISSION",
    "DECKS",
    "KELLY_FRACTION",
    "MAX_BET_FRACTION",
    "MIN_BET_FRACTION",
    "MIN_POSITIVE_EV",
    "WEAK_EV_KELLY_FRACTION",
    "WEAK_EV_UPPER_BOUND",
    "analyze_shoe_composition",
    "exact_next_hand_probabilities",
    "expected_returns",
    "fresh_counts",
    "kelly_fraction",
    "parse_card_value",
    "remaining_counts_from_observed",
    "validate_remaining_counts",
]
