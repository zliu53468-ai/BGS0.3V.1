"""精確牌靴組成、下一手不放回機率、EV 與分數凱利。

重要限制：B/P/T 結果本身無法唯一反推出已移除的牌點。本模組只在收到
實際剩餘點數計數，或每一張已觀察到的牌點時啟用資金訊號；否則回傳
``available=False``，由 predictor 強制 No Bet，避免把路單當成物理算牌。

點數編碼使用百家樂點數：A=1、2..9 原值、10/J/Q/K=0。
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import os


DECKS = max(1, min(16, int(os.getenv("SHOE_DECKS", "8") or "8")))
BANKER_COMMISSION = min(0.20, max(0.0, float(os.getenv("BANKER_COMMISSION", "0.05") or "0.05")))
MIN_POSITIVE_EV = max(0.0, float(os.getenv("PHYSICAL_MIN_EV", "0.002") or "0.002"))
KELLY_FRACTION = min(1.0, max(0.0, float(os.getenv("KELLY_FRACTION", "0.25") or "0.25")))
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
MAX_BET_FRACTION = min(0.10, max(0.0, float(os.getenv("MAX_BET_FRACTION", "0.02") or "0.02")))

# 8 副牌標準牌靴的公開理論基準；僅用於比較，不會在未知牌組時產生下注訊號。
STANDARD_EIGHT_DECK_BASELINE = {"B": 0.458597, "P": 0.446247, "T": 0.095156}


def fresh_counts(decks: int = DECKS) -> List[int]:
    """回傳點數 0..9 的新牌靴張數；0 包含 10/J/Q/K。"""
    count = max(1, min(16, int(decks)))
    return [16 * count] + [4 * count] * 9


def parse_card_value(value: Any) -> int:
    """把實際牌面轉為百家樂點數；拒絕模糊或越界輸入。"""
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
    """由每張實際已出牌面，精確扣除新牌靴點數計數。"""
    remaining = fresh_counts(decks)
    for raw in observed_cards:
        point = parse_card_value(raw)
        remaining[point] -= 1
        if remaining[point] < 0:
            raise ValueError(f"牌點 {point} 的已輸入張數超過 {decks} 副牌上限")
    return remaining


def validate_remaining_counts(values: Sequence[Any], *, decks: int = DECKS) -> Tuple[int, ...]:
    if len(values) != 10:
        raise ValueError("remaining_counts 必須依序包含點數 0..9 共 10 個數值")
    maximum = fresh_counts(decks)
    counts: List[int] = []
    for point, raw in enumerate(values):
        if isinstance(raw, bool) or int(raw) != float(raw):
            raise ValueError("remaining_counts 必須是非負整數")
        count = int(raw)
        if count < 0 or count > maximum[point]:
            raise ValueError(f"點數 {point} 的剩餘張數超出 {decks} 副牌範圍")
        counts.append(count)
    if sum(counts) < 6:
        raise ValueError("剩餘牌不足以安全完成一手百家樂")
    return tuple(counts)


def _banker_should_draw(banker_total: int, player_third_card: Optional[int]) -> bool:
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
def exact_next_hand_probabilities(counts: Tuple[int, ...]) -> Tuple[float, float, float]:
    """依正式補牌規則，完整枚舉下一手的精確不放回 B/P/T 機率。"""
    if len(counts) != 10 or sum(counts) < 6 or any(value < 0 for value in counts):
        raise ValueError("無效的剩餘牌組")

    # 先聚合前四張 P1/B1/P2/B2；相同剩餘狀態與點數總和可合併。
    states: Dict[Tuple[Tuple[int, ...], int, int], float] = {(counts, 0, 0): 1.0}
    for owner in ("P", "B", "P", "B"):
        next_states: DefaultDict[Tuple[Tuple[int, ...], int, int], float] = defaultdict(float)
        for (state_counts, player_total, banker_total), state_probability in states.items():
            for point, draw_probability, after_draw in _draw_branches(state_counts):
                if owner == "P":
                    key = (after_draw, (player_total + point) % 10, banker_total)
                else:
                    key = (after_draw, player_total, (banker_total + point) % 10)
                next_states[key] += state_probability * draw_probability
        states = dict(next_states)

    outcome: DefaultDict[str, float] = defaultdict(float)

    def settle(player_total: int, banker_total: int, probability: float) -> None:
        code = "P" if player_total > banker_total else "B" if banker_total > player_total else "T"
        outcome[code] += probability

    for (state_counts, player_total, banker_total), state_probability in states.items():
        if player_total in {8, 9} or banker_total in {8, 9}:
            settle(player_total, banker_total, state_probability)
            continue

        if player_total <= 5:
            for player_third, p_draw, after_player in _draw_branches(state_counts):
                new_player_total = (player_total + player_third) % 10
                branch_probability = state_probability * p_draw
                if _banker_should_draw(banker_total, player_third):
                    for banker_third, b_draw, _ in _draw_branches(after_player):
                        settle(
                            new_player_total,
                            (banker_total + banker_third) % 10,
                            branch_probability * b_draw,
                        )
                else:
                    settle(new_player_total, banker_total, branch_probability)
        elif _banker_should_draw(banker_total, None):
            for banker_third, b_draw, _ in _draw_branches(state_counts):
                settle(
                    player_total,
                    (banker_total + banker_third) % 10,
                    state_probability * b_draw,
                )
        else:
            settle(player_total, banker_total, state_probability)

    total_probability = sum(outcome.values())
    if total_probability <= 0.0:
        raise ValueError("無法計算下一手機率")
    return (
        outcome["B"] / total_probability,
        outcome["P"] / total_probability,
        outcome["T"] / total_probability,
    )


def expected_returns(probabilities: Mapping[str, float]) -> Dict[str, float]:
    """單位下注 EV；和局視為 B/P 注的 push，莊注計 5% 抽水。"""
    banker_probability = float(probabilities.get("B", 0.0) or 0.0)
    player_probability = float(probabilities.get("P", 0.0) or 0.0)
    banker_net_win = 1.0 - BANKER_COMMISSION
    return {
        "B": banker_net_win * banker_probability - player_probability,
        "P": player_probability - banker_probability,
        # 不建議和局投注；保留 None 可防止下游誤把未知賠率當 8:1。
        "T": None,
    }


def kelly_fraction(
    *,
    side: str,
    probabilities: Mapping[str, float],
) -> float:
    """含和局 push 的 full Kelly，再套用動態分數凱利與硬上限。

    最佳淨 EV 剛跨過出手門檻、但尚未超過弱優勢上限時，只採用
    八分之一 Kelly。EV 較明顯時才恢復一般分數 Kelly，避免為了
    提高出手率而同步放大本金波動。
    """
    code = str(side or "").upper()
    if code not in {"B", "P"}:
        return 0.0
    p_win = float(probabilities.get(code, 0.0) or 0.0)
    p_loss = float(probabilities.get("P" if code == "B" else "B", 0.0) or 0.0)
    odds = 1.0 - BANKER_COMMISSION if code == "B" else 1.0
    resolved_probability = p_win + p_loss
    if resolved_probability <= 0.0 or odds <= 0.0:
        return 0.0
    raw_ev = odds * p_win - p_loss
    full_kelly = raw_ev / (odds * resolved_probability)
    applied_fraction = (
        WEAK_EV_KELLY_FRACTION
        if MIN_POSITIVE_EV <= raw_ev <= WEAK_EV_UPPER_BOUND
        else KELLY_FRACTION
    )
    return min(MAX_BET_FRACTION, max(0.0, full_kelly) * applied_fraction)


def _applied_kelly_fraction(*, expected_return: float) -> float:
    """回傳本次使用的 Kelly 比例，供 API 與回測完整稽核。"""
    value = float(expected_return)
    if MIN_POSITIVE_EV <= value <= WEAK_EV_UPPER_BOUND:
        return float(WEAK_EV_KELLY_FRACTION)
    return float(KELLY_FRACTION)


def _full_kelly_fraction(*, side: str, probabilities: Mapping[str, float]) -> float:
    code = str(side or "").upper()
    if code not in {"B", "P"}:
        return 0.0
    p_win = float(probabilities.get(code, 0.0) or 0.0)
    p_loss = float(probabilities.get("P" if code == "B" else "B", 0.0) or 0.0)
    odds = 1.0 - BANKER_COMMISSION if code == "B" else 1.0
    resolved_probability = p_win + p_loss
    if resolved_probability <= 0.0:
        return 0.0
    return max(0.0, (odds * p_win - p_loss) / (odds * resolved_probability))


def _composition_summary(counts: Sequence[int], decks: int) -> Dict[str, Any]:
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
        "small_ratio_delta_from_fresh": small / total - sum(baseline[1:5]) / baseline_total,
        "large_ratio_delta_from_fresh": large / total - sum(baseline[5:10]) / baseline_total,
    }


def analyze_shoe_composition(
    shoe_context: Optional[Mapping[str, Any]],
    *,
    default_decks: int = DECKS,
) -> Dict[str, Any]:
    """解析精確組成並產生最終物理訊號；沒有牌面資料即明確 No Bet。"""
    context = dict(shoe_context or {})
    decks = max(1, min(16, int(context.get("decks", default_decks) or default_decks)))
    source = ""
    try:
        raw_counts = context.get("remaining_counts")
        observed = context.get("observed_cards")
        if isinstance(raw_counts, Sequence) and not isinstance(raw_counts, (str, bytes)):
            counts = validate_remaining_counts(raw_counts, decks=decks)
            source = str(context.get("source") or "exact_remaining_counts")
        elif isinstance(observed, Iterable) and not isinstance(observed, (str, bytes, Mapping)):
            observed_list = list(observed)
            if not observed_list:
                raise ValueError("尚未輸入任何實際牌面")
            counts = validate_remaining_counts(
                remaining_counts_from_observed(observed_list, decks=decks),
                decks=decks,
            )
            source = str(context.get("source") or "observed_card_values")
        else:
            return {
                "available": False,
                "action": "O",
                "action_text": "觀望／無物理優勢",
                "source": "outcome_history_only",
                "reason_code": "CARD_COMPOSITION_NOT_IDENTIFIABLE_FROM_BPT",
                "reason": (
                    "只有莊／閒／和結果，無法反推出已移除的牌點；"
                    "需輸入實際牌面或精確剩餘點數計數才可啟用算牌。"
                ),
                "expected_returns": {"B": None, "P": None, "T": None},
                "selected_expected_return": 0.0,
                "kelly_fraction": 0.0,
                "recommended_bet_percentage": 0.0,
                "minimum_positive_ev": MIN_POSITIVE_EV,
            }

        banker, player, tie = exact_next_hand_probabilities(counts)
        probabilities = {"B": banker, "P": player, "T": tie}
        returns = expected_returns(probabilities)
        selected_side = max(("B", "P"), key=lambda side: float(returns[side]))
        selected_ev = float(returns[selected_side])
        action = selected_side if selected_ev >= MIN_POSITIVE_EV else "O"
        full_kelly = _full_kelly_fraction(side=action, probabilities=probabilities)
        applied_kelly = _applied_kelly_fraction(expected_return=selected_ev)
        fraction = kelly_fraction(side=action, probabilities=probabilities)
        baseline = STANDARD_EIGHT_DECK_BASELINE
        return {
            "available": True,
            "action": action,
            "action_text": "莊" if action == "B" else "閒" if action == "P" else "觀望／無正期望優勢",
            "source": source,
            "decks": decks,
            "remaining_counts": list(counts),
            "probabilities": probabilities,
            "probability_delta_from_standard": {
                side: probabilities[side] - baseline[side] for side in ("B", "P", "T")
            },
            "composition": _composition_summary(counts, decks),
            "banker_commission": BANKER_COMMISSION,
            "expected_returns": returns,
            "selected_side_by_ev": selected_side,
            "selected_expected_return": selected_ev if action in {"B", "P"} else 0.0,
            "best_raw_expected_return": selected_ev,
            "minimum_positive_ev": MIN_POSITIVE_EV,
            "kelly_method": (
                "eighth_kelly_for_weak_positive_ev_with_tie_push_and_hard_cap"
                if action in {"B", "P"}
                and selected_ev <= WEAK_EV_UPPER_BOUND
                else "fractional_kelly_with_tie_push_and_hard_cap"
            ),
            "applied_kelly_fraction": applied_kelly if action in {"B", "P"} else 0.0,
            "weak_ev_upper_bound": WEAK_EV_UPPER_BOUND,
            "weak_ev_kelly_fraction": WEAK_EV_KELLY_FRACTION,
            "full_kelly_before_fraction": full_kelly,
            "fractional_kelly_before_cap": full_kelly * applied_kelly,
            "kelly_fraction": fraction,
            "recommended_bet_percentage": fraction * 100.0,
            "risk_gate_open": action in {"B", "P"} and fraction > 0.0,
            "reason_code": "POSITIVE_PHYSICAL_EV" if action in {"B", "P"} else "NO_POSITIVE_PHYSICAL_EV",
            "reason": (
                "精確不放回牌組的淨期望報酬達到門檻。"
                if action in {"B", "P"}
                else "精確牌組雖可計算，但莊／閒在抽水後皆未達正期望門檻。"
            ),
        }
    except (TypeError, ValueError) as exc:
        return {
            "available": False,
            "action": "O",
            "action_text": "觀望／牌組輸入無效",
            "source": source or "invalid_shoe_context",
            "reason_code": "INVALID_CARD_COMPOSITION",
            "reason": str(exc),
            "expected_returns": {"B": None, "P": None, "T": None},
            "selected_expected_return": 0.0,
            "kelly_fraction": 0.0,
            "recommended_bet_percentage": 0.0,
            "minimum_positive_ev": MIN_POSITIVE_EV,
        }


__all__ = [
    "BANKER_COMMISSION",
    "DECKS",
    "KELLY_FRACTION",
    "MAX_BET_FRACTION",
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
