"""BGS cMAB 統一預測入口。

正式圖片／真人桌預測使用完整 B/P/T 歷史、牌路上下文與 LinUCB cMAB。
最後階段會比較「固定牌路規則」與 cMAB 的最近真實 B/P 命中率，
只輸出一個最終 B/P 方向；不使用算牌、EV 或觀望方向。
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from adaptive_ensemble import adapt_prediction
from contextual_bandit import predict_bandit
from road_model import build_road_context

DB_HOLDOUT: Dict[str, Any] = {
    "status": "removed",
    "replacement": "CMAB-LINUCB-V1",
    "note": "舊粒子／有限牌組驗證層已從主要預測流程移除",
}

SHADOW_REQUIRED_CONSECUTIVE_HITS = 2
SHADOW_MAX_STREAMS = 2048

# 正式方向選擇器只保留最近已經真正開出的 B/P 結果。它是
# predictor.py 內的短期記憶，不會碰 cMAB 的 A/b 矩陣，也不會把
# 牌路先行的輸贏誤回灌成 cMAB 的 reward。
DIRECTION_SOURCE_WINDOW = 20
DIRECTION_SOURCE_MIN_SAMPLES = 12
DIRECTION_SOURCE_ADVANTAGE = 0.04
DIRECTION_SOURCE_MAX_STREAMS = 2048


def _bandit_learning_scope(
    *,
    user_id: str,
    venue: str,
    room: str,
) -> str:
    """建立可跨鞋泛化、但不跨使用者／場館／桌污染的 cMAB 範圍。

    回傳不可逆摘要，避免把外部 user id 直接寫進模型狀態檔。
    鞋號仍用於預測去重與影子狀態隔離，但不再重置長期學習矩陣；
    否則每次換鞋都會永久停留在牌路冷啟動與近期順勢退化。
    """
    raw = "|".join((
        str(user_id or "__anonymous__"),
        str(venue or "").upper().strip(),
        str(room or "").strip(),
    ))
    return "__cmab_scope__:" + sha256(
        raw.encode("utf-8")
    ).hexdigest()[:32]


def _normalize_outcome_history(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-2000:]


def _fallback_direction(prediction: Mapping[str, Any]) -> str:
    for key in (
        "internal_recommend",
        "selected_arm",
        "recommend",
        "base_bandit_direction",
    ):
        direction = str(prediction.get(key) or "").upper().strip()
        if direction in {"B", "P"}:
            return direction
    banker = float(prediction.get("banker_rate", 0.0) or 0.0)
    player = float(prediction.get("player_rate", 0.0) or 0.0)
    return "B" if banker >= player else "P"


def _short_term_trend_prior(
    buffer: List[str],
    fallback_direction: str,
) -> Dict[str, Any]:
    """只依最近 3 個 B/P 建立微觀順勢先驗；T 不改變 B/P 走勢。"""
    fallback = fallback_direction if fallback_direction in {"B", "P"} else "B"
    if len(buffer) >= 2 and buffer[-1] == buffer[-2]:
        direction = buffer[-1]
        strategy = "follow_last_two_streak"
        strength = 0.60
        evidence = buffer[-2:]
    elif (
        len(buffer) >= 3
        and buffer[-3] == buffer[-1]
        and buffer[-2] != buffer[-1]
    ):
        direction = "P" if buffer[-1] == "B" else "B"
        strategy = "continue_three_step_alternation"
        strength = 0.57
        evidence = buffer[-3:]
    elif len(buffer) >= 3:
        banker_count = sum(value == "B" for value in buffer)
        direction = "B" if banker_count >= 2 else "P"
        strategy = "recent_three_majority"
        strength = 0.55
        evidence = buffer[-3:]
    else:
        direction = fallback
        strategy = "insufficient_micro_history_fallback"
        strength = 0.52
        evidence = buffer[-3:]

    opposite = "P" if direction == "B" else "B"
    return {
        "direction": direction,
        "direction_text": "莊" if direction == "B" else "閒",
        "strategy": strategy,
        "strength": float(strength),
        "evidence": list(evidence),
        "short_term_buffer": list(buffer[-3:]),
        "probabilities": {
            direction: float(strength),
            opposite: float(1.0 - strength),
            "T": 0.0,
        },
    }


def _bp_history(values: Iterable[Any]) -> List[str]:
    """保留 B/P 時序；和局不推進兩個方向來源的命中統計。"""
    return [
        value for value in _normalize_outcome_history(values)
        if value in {"B", "P"}
    ]


def _road_prior_direction(
    history: Iterable[Any],
    fallback_direction: str,
) -> Dict[str, Any]:
    """以固定、可解釋的牌路規則產生第二個候選方向。

    規則刻意不讀取牌面、EV、算牌或外部模型：
    1. 末段同向至少 3 手時，優先延續長龍；
    2. 最近 3 手為單跳，或最近 8 手轉換率很高時，反向最後一手；
    3. 其餘以最近 5～8 手多數決，平手時才回到 Bandit 候選。
    """
    bp = _bp_history(history)
    fallback = fallback_direction if fallback_direction in {"B", "P"} else "B"
    if not bp:
        return {
            "direction": fallback,
            "rule": "insufficient_history_bandit_fallback",
            "streak_length": 0,
            "switch_rate": 0.0,
            "window": [],
        }

    last = bp[-1]
    streak = 1
    for value in reversed(bp[:-1]):
        if value != last:
            break
        streak += 1

    recent = bp[-min(8, len(bp)):]
    switch_rate = (
        sum(left != right for left, right in zip(recent, recent[1:]))
        / max(1, len(recent) - 1)
    )
    alternating = (
        len(bp) >= 3
        and bp[-3] == bp[-1]
        and bp[-2] != bp[-1]
    )
    opposite = "P" if last == "B" else "B"

    if streak >= 3:
        direction, rule = last, "streak_follow"
    elif alternating:
        direction, rule = opposite, "single_chop_reverse"
    elif len(recent) >= 5 and switch_rate >= 0.65:
        direction, rule = opposite, "high_switch_rate_reverse"
    else:
        majority_window = bp[-min(8, len(bp)):]
        banker_count = sum(value == "B" for value in majority_window)
        player_count = len(majority_window) - banker_count
        if banker_count == player_count:
            direction, rule = fallback, "recent_majority_tie_bandit_fallback"
        else:
            direction = "B" if banker_count > player_count else "P"
            rule = "recent_majority"

    return {
        "direction": direction,
        "rule": rule,
        "streak_length": int(streak),
        "switch_rate": round(float(switch_rate), 4),
        "window": list(recent),
    }


class DirectionSourceSelector:
    """比較牌路先行與 cMAB 最近真實命中率的輕量選擇器。

    每個 user／場館／桌／靴各自隔離。只有新的 B/P 結果出現時，才會
    結算上一筆兩個候選方向；同一張圖重按預測不會重複計算。
    """

    def __init__(self) -> None:
        self._states: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def stream_key(
        *,
        user_id: str,
        venue: str,
        room: str,
        shoe_id: str,
    ) -> str:
        raw = "|".join((
            str(user_id or "__anonymous__"),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "__unspecified_shoe__").strip(),
        ))
        return sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _new_state(bp: List[str]) -> Dict[str, Any]:
        return {
            "last_bp_history": list(bp),
            "pending_bandit_direction": "",
            "pending_road_direction": "",
            "bandit_hits": [],
            "road_hits": [],
        }

    @staticmethod
    def _performance(values: Iterable[Any]) -> Dict[str, Any]:
        hits = [bool(value) for value in list(values)[-DIRECTION_SOURCE_WINDOW:]]
        sample_count = len(hits)
        correct_count = sum(hits)
        return {
            "sample_count": sample_count,
            "correct_count": correct_count,
            "accuracy": (
                round(correct_count / sample_count, 4)
                if sample_count else None
            ),
            "window": DIRECTION_SOURCE_WINDOW,
        }

    def select(
        self,
        history: Iterable[Any],
        *,
        bandit_direction: str,
        road_direction: str,
        stream_key: str,
    ) -> Dict[str, Any]:
        """以當下之前已結算的績效決定這一局使用哪個來源。"""
        bp = _bp_history(history)
        bandit = bandit_direction if bandit_direction in {"B", "P"} else "B"
        road = road_direction if road_direction in {"B", "P"} else bandit

        with self._lock:
            state = self._states.get(stream_key)
            if state is None:
                state = self._new_state([])
                self._states[stream_key] = state
            self._states.move_to_end(stream_key)
            while len(self._states) > DIRECTION_SOURCE_MAX_STREAMS:
                self._states.popitem(last=False)

            previous_bp = list(state.get("last_bp_history") or [])
            history_replaced = bool(
                len(bp) < len(previous_bp)
                or bp[:len(previous_bp)] != previous_bp
            )
            resolved_actual = ""
            if history_replaced:
                # 換靴、換桌或截圖從不同位置開始時，不拿舊預測結算新歷史。
                state = self._new_state(bp)
                self._states[stream_key] = state
            elif len(bp) > len(previous_bp):
                # 一筆 pending 只結算「下一個」新 B/P；若呼叫端一次補了多局，
                # 其餘局沒有對應預測，絕不虛構命中率。
                resolved_actual = bp[len(previous_bp)]
                old_bandit = str(state.get("pending_bandit_direction") or "")
                old_road = str(state.get("pending_road_direction") or "")
                if old_bandit in {"B", "P"}:
                    state["bandit_hits"] = list(
                        state.get("bandit_hits") or []
                    )[-(DIRECTION_SOURCE_WINDOW - 1):] + [
                        old_bandit == resolved_actual
                    ]
                if old_road in {"B", "P"}:
                    state["road_hits"] = list(
                        state.get("road_hits") or []
                    )[-(DIRECTION_SOURCE_WINDOW - 1):] + [
                        old_road == resolved_actual
                    ]
                state["last_bp_history"] = list(bp)
            else:
                state["last_bp_history"] = list(bp)

            bandit_performance = self._performance(
                state.get("bandit_hits") or []
            )
            road_performance = self._performance(
                state.get("road_hits") or []
            )
            bandit_accuracy = bandit_performance["accuracy"]
            road_accuracy = road_performance["accuracy"]
            enough_samples = bool(
                bandit_performance["sample_count"] >= DIRECTION_SOURCE_MIN_SAMPLES
                and road_performance["sample_count"] >= DIRECTION_SOURCE_MIN_SAMPLES
            )

            if bandit == road:
                final_direction = bandit
                direction_source = "road_bandit_agree"
            elif (
                enough_samples
                and road_accuracy is not None
                and bandit_accuracy is not None
                and road_accuracy > bandit_accuracy + DIRECTION_SOURCE_ADVANTAGE
            ):
                final_direction = road
                direction_source = "road_prior_better"
            elif (
                enough_samples
                and road_accuracy is not None
                and bandit_accuracy is not None
                and bandit_accuracy > road_accuracy + DIRECTION_SOURCE_ADVANTAGE
            ):
                final_direction = bandit
                direction_source = "bandit_better"
            else:
                final_direction = bandit
                direction_source = "bandit_default"

            state["pending_bandit_direction"] = bandit
            state["pending_road_direction"] = road
            self._states[stream_key] = state

        return {
            "direction": final_direction,
            "direction_source": direction_source,
            "bandit_performance": bandit_performance,
            "road_performance": road_performance,
            "enough_samples": enough_samples,
            "resolved_actual": resolved_actual,
            "history_replaced": history_replaced,
        }


_DIRECTION_SOURCE_SELECTOR = DirectionSourceSelector()


class ShadowBacktestController:
    """No Bet 期間的三局影子回測與雙條件解鎖狀態機。"""

    def __init__(self) -> None:
        self.shadow_buffer: List[str] = []
        self._states: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def stream_key(
        *,
        user_id: str,
        venue: str,
        room: str,
        shoe_id: str,
    ) -> str:
        raw = "|".join((
            str(user_id or "__anonymous__"),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
        ))
        return sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _new_state(history: List[str]) -> Dict[str, Any]:
        return {
            "hard_brake_latched": False,
            "pending_direction": "",
            "consecutive_hits": 0,
            "total_shadow_bets": 0,
            "shadow_hits": 0,
            "shadow_misses": 0,
            "ties_skipped": 0,
            "last_history_length": len(history),
            "last_history_hash": sha256(
                "".join(history).encode("utf-8")
            ).hexdigest()[:24],
        }

    @staticmethod
    def _variance_status(prediction: Mapping[str, Any]) -> tuple[float, float, bool]:
        braking = dict(prediction.get("uncertainty_braking") or {})
        variance = float(
            prediction.get(
                "variance",
                braking.get("action_space_variance", 0.0),
            )
            or 0.0
        )
        threshold = float(
            prediction.get(
                "variance_threshold",
                braking.get("threshold_variance", float("inf")),
            )
            or float("inf")
        )
        variance_safe = bool(
            braking.get("variance_safe", variance <= threshold)
        )
        return variance, threshold, variance_safe

    @staticmethod
    def _force_no_bet(
        result: Dict[str, Any],
        *,
        reason: str,
    ) -> None:
        result.setdefault(
            "pre_hard_brake_probabilities",
            dict(result.get("probabilities") or {}),
        )
        current_probabilities = dict(result.get("probabilities") or {})
        tie_probability = max(
            0.0,
            min(0.30, float(current_probabilities.get("T", 0.0) or 0.0)),
        )
        neutral_bp = (1.0 - tie_probability) * 0.5
        result["probabilities"] = {
            "B": neutral_bp,
            "P": neutral_bp,
            "T": tie_probability,
        }
        result["banker_rate"] = round(neutral_bp * 100.0, 2)
        result["player_rate"] = round(neutral_bp * 100.0, 2)
        result["tie_rate"] = round(tie_probability * 100.0, 2)
        result["recommend"] = "O"
        result["recommend_text"] = "觀望"
        result["action"] = "O"
        result["action_text"] = "觀望／絕對不下注"
        result["internal_recommend"] = "O"
        result["internal_action"] = "O"
        result["next_round_direction"] = "O"
        result["next_round_direction_text"] = "觀望"
        result["signal_allowed"] = False
        result["signal_status_code"] = "HARD_BRAKE_NO_BET"
        result["signal_status_text"] = "硬熔斷鎖定：觀望／絕對不下注"
        result["signal_reason"] = reason
        result["internal_signal_reason"] = reason
        result["direction_source"] = "shadow_backtest_hard_brake"
        result["ensemble_confidence"] = 0.0
        result["confidence"] = 0.0
        result["quality_score"] = 0.0
        result["confidence_label"] = "零信心／硬熔斷"
        result["bet_multiplier"] = 0.0
        result["hard_brake_active"] = True

    def apply(
        self,
        history: List[str],
        prediction: Mapping[str, Any],
        *,
        stream_key: str = "__default__",
    ) -> Dict[str, Any]:
        result = dict(prediction or {})
        local_buffer = [value for value in history if value in {"B", "P"}][-3:]
        history_hash = sha256("".join(history).encode("utf-8")).hexdigest()[:24]
        current_extreme = bool(
            result.get("is_extreme_unseen")
            or result.get("hard_brake_active")
        )
        variance, variance_threshold, variance_safe = self._variance_status(result)

        with self._lock:
            self.shadow_buffer = list(local_buffer)
            state = self._states.get(stream_key)
            if state is None:
                state = self._new_state(history)
                self._states[stream_key] = state
            self._states.move_to_end(stream_key)
            while len(self._states) > SHADOW_MAX_STREAMS:
                self._states.popitem(last=False)

            previous_length = int(state.get("last_history_length", len(history)))
            previous_hash = str(state.get("last_history_hash") or "")
            pending_direction = str(state.get("pending_direction") or "")
            shadow_result = "PENDING"
            resolved_actual = ""

            current_prefix_hash = sha256(
                "".join(history[:previous_length]).encode("utf-8")
            ).hexdigest()[:24]
            history_replaced = bool(
                len(history) < previous_length
                or (
                    len(history) >= previous_length
                    and previous_hash
                    and previous_hash != current_prefix_hash
                )
            )
            if history_replaced:
                state = self._new_state(history)
                self._states[stream_key] = state
                pending_direction = ""
                shadow_result = "STREAM_RESET"
            elif len(history) > previous_length and pending_direction in {"B", "P"}:
                new_outcomes = history[previous_length:]
                resolved_actual = next(
                    (value for value in new_outcomes if value in {"B", "P"}),
                    "",
                )
                if resolved_actual:
                    state["total_shadow_bets"] = int(
                        state.get("total_shadow_bets", 0) or 0
                    ) + 1
                    if pending_direction == resolved_actual:
                        state["consecutive_hits"] = int(
                            state.get("consecutive_hits", 0) or 0
                        ) + 1
                        state["shadow_hits"] = int(
                            state.get("shadow_hits", 0) or 0
                        ) + 1
                        shadow_result = "HIT"
                    else:
                        state["consecutive_hits"] = 0
                        state["shadow_misses"] = int(
                            state.get("shadow_misses", 0) or 0
                        ) + 1
                        shadow_result = "MISS"
                    state["pending_direction"] = ""
                    pending_direction = ""
                elif any(value == "T" for value in new_outcomes):
                    state["ties_skipped"] = int(
                        state.get("ties_skipped", 0) or 0
                    ) + 1
                    shadow_result = "TIE_SKIPPED"

            if current_extreme:
                state["hard_brake_latched"] = True

            consecutive_hits = int(state.get("consecutive_hits", 0) or 0)
            release_allowed = bool(
                state.get("hard_brake_latched")
                and not current_extreme
                and variance_safe
                and consecutive_hits >= SHADOW_REQUIRED_CONSECUTIVE_HITS
            )
            released_this_round = False
            if release_allowed:
                state["hard_brake_latched"] = False
                state["pending_direction"] = ""
                pending_direction = ""
                released_this_round = True

            hard_brake_latched = bool(state.get("hard_brake_latched"))
            shadow_prior: Dict[str, Any] = {}
            if hard_brake_latched:
                if pending_direction not in {"B", "P"}:
                    shadow_prior = _short_term_trend_prior(
                        local_buffer,
                        _fallback_direction(result),
                    )
                    pending_direction = str(shadow_prior["direction"])
                    state["pending_direction"] = pending_direction
                else:
                    shadow_prior = _short_term_trend_prior(
                        local_buffer,
                        pending_direction,
                    )

            state["last_history_length"] = len(history)
            state["last_history_hash"] = history_hash
            self._states[stream_key] = state

            shadow_payload = {
                "active": bool(hard_brake_latched),
                "hard_brake_latched": bool(hard_brake_latched),
                "released_this_round": bool(released_this_round),
                "shadow_buffer": list(local_buffer),
                "pending_direction": pending_direction,
                "pending_direction_text": (
                    "莊" if pending_direction == "B"
                    else "閒" if pending_direction == "P"
                    else ""
                ),
                "strategy": str(shadow_prior.get("strategy") or ""),
                "last_shadow_result": shadow_result,
                "last_resolved_actual": resolved_actual,
                "consecutive_hits": int(state.get("consecutive_hits", 0) or 0),
                "required_consecutive_hits": SHADOW_REQUIRED_CONSECUTIVE_HITS,
                "total_shadow_bets": int(state.get("total_shadow_bets", 0) or 0),
                "shadow_hits": int(state.get("shadow_hits", 0) or 0),
                "shadow_misses": int(state.get("shadow_misses", 0) or 0),
                "ties_skipped": int(state.get("ties_skipped", 0) or 0),
                "variance": float(variance),
                "variance_threshold": float(variance_threshold),
                "variance_safe": bool(variance_safe),
                "model_is_extreme_unseen": bool(current_extreme),
                "release_allowed": bool(release_allowed),
            }

        result["shadow_backtest"] = shadow_payload
        result["shadow_buffer"] = list(local_buffer)
        if hard_brake_latched:
            self._force_no_bet(
                result,
                reason=(
                    "統計混沌熔斷仍鎖定；只有影子回測連中 2 局且"
                    "模型方差降回安全門檻，才允許恢復正式方向。"
                ),
            )
            adaptive = dict(result.get("adaptive_ensemble") or {})
            adaptive.update({
                "active": True,
                "circuit_breaker_active": True,
                "hard_brake_active": True,
                "mode": "shadow_backtest_latched_hard_brake",
                "overall_confidence": 0.0,
                "final_action": "O",
                "bet_multiplier": 0.0,
                "shadow_backtest_required": True,
                "shadow_consecutive_hits": shadow_payload["consecutive_hits"],
                "variance_safe": bool(variance_safe),
            })
            result["adaptive_ensemble"] = adaptive
        elif released_this_round:
            result["hard_brake_released"] = True
            result["signal_reason"] = (
                "影子回測已連中 2 局且模型方差回到安全門檻，解除硬熔斷。"
            )
        return result


_SHADOW_CONTROLLER = ShadowBacktestController()
# 保留前一版類別／單例名稱，避免外部整合在部署過渡期 import 失敗；
# 行為已統一改為 No Bet 影子回測，不會恢復舊的實盤短線接管。
ShortTermTakeoverController = ShadowBacktestController
_SHORT_TERM_CONTROLLER = _SHADOW_CONTROLLER


def predict(history: Union[str, Iterable[Any], None] = None, venue: str = "", room: str = "",
            shoe_id: str = "", user_id: str = "", run_seed: Optional[int] = None,
            shoe_context: Optional[Mapping[str, Any]] = None,
            road_context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """統一預測 API；保留舊參數名稱以相容 app.py。"""
    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        history_values = [part for part in history.replace("|", ",").split(",") if part.strip()]
    else:
        history_values = list(history)
    cleaned = _normalize_outcome_history(history_values)
    # 第一層：Full Road Pattern Model。截圖入口若已附上完整 road_context
    # 則直接沿用；否則從目前 B/P/T 歷史建構，確保不會跳過全路圖分析。
    road = dict(road_context or {})
    if not isinstance(road.get("models"), Mapping):
        road = build_road_context(cleaned, seed=run_seed)

    # 第二層所需的 cMAB 先獨立評估。它保留自己的 Arm 與 state，只提供
    # context 輔助訊號，不能先覆寫 Full Road／Adaptive 的最終方向。
    bandit_learning_user_id = _bandit_learning_scope(
        user_id=str(user_id or ""),
        venue=str(venue or ""),
        room=str(room or ""),
    )
    result = predict_bandit(
        cleaned,
        road_context=road,
        venue=venue,
        room=room,
        user_id=bandit_learning_user_id,
        run_seed=run_seed,
    )
    bandit_direction = _fallback_direction(result)

    # 第二層：Adaptive Ensemble 只讀取 Full Road 與既有路子專家的機率。
    # particle、算牌、EV、外部模型都不會透過這裡注入。
    result["road_support"] = dict(road)
    result["component_probabilities"] = dict(
        road.get("component_probabilities") or {}
    )
    result["decision_pipeline"] = (
        "full_road_pattern_to_adaptive_ensemble_with_contextual_bandit_auxiliary"
    )
    result["bandit_direction"] = bandit_direction
    result["bandit_original_direction"] = bandit_direction
    result["bandit_learning_direction"] = bandit_direction
    result = adapt_prediction(
        result,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )

    # 產品契約：不論路圖樣本、模型衝突或混沌診斷，都對外強制一致輸出 B/P。
    final_direction = _fallback_direction(result)
    if final_direction not in {"B", "P"}:
        final_direction = bandit_direction if bandit_direction in {"B", "P"} else "B"
    final_text = "莊" if final_direction == "B" else "閒"
    road_direction = str(road.get("direction") or "").upper().strip()
    if road_direction not in {"B", "P"}:
        road_direction = final_direction

    result.update({
        "forced_bp_direction": True,
        "bandit_direction": bandit_direction,
        "bandit_original_direction": bandit_direction,
        "bandit_learning_direction": bandit_direction,
        "road_direction": road_direction,
        "road_prior": {
            "direction": road_direction,
            "rule": "full_road_pattern_model",
            "regime": dict(road.get("regime") or {}),
        },
        "recommend": final_direction,
        "recommend_text": final_text,
        "action": final_direction,
        "action_text": final_text,
        "next_round_direction": final_direction,
        "next_round_direction_text": final_text,
        "internal_recommend": final_direction,
        "internal_action": final_direction,
        "signal_allowed": True,
        "signal_status_code": "ROAD_PRIMARY_ADAPTIVE_DIRECTION",
        "signal_status_text": "全路圖 → Adaptive Ensemble → cMAB 輔助：已輸出 B/P",
    })
    model_fingerprint = str(
        result.get("prediction_fingerprint") or ""
    ).strip()
    result["model_prediction_fingerprint"] = model_fingerprint
    result["prediction_fingerprint"] = sha256(
        "|".join((
            model_fingerprint,
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "__unspecified_shoe__").strip(),
        )).encode("utf-8")
    ).hexdigest()[:24]
    result.update({
        "shoe_id": str(shoe_id or ""),
        "bandit_learning_user_id": bandit_learning_user_id,
        "bandit_scope_mode": "user_venue_room_long_term",
        "bandit_shoe_isolated": False,
        "shoe_event_isolated": True,
        "composition_quality": "not_applicable_cmab",
        "remaining_counts_source": "not_used",
        "shoe_context_ignored": bool(shoe_context),
        "road_quality_ok": bool(road.get(
            "quality_ok", road.get("recognition_quality_ok", True)
        )),
        "input_required": False,
    })
    return result


def run_virtual_round(session: Mapping[str, Any], run_seed: Optional[int] = None) -> Dict[str, Any]:
    """保留舊虛擬牌靴介面，但方向同樣由 cMAB 產生。"""
    # 虛擬牌靴僅供舊相容入口使用；真人圖片預測不載入粒子／算牌模組，
    # 避免其環境參數或初始化副作用混入正式 cMAB 路徑。
    from particle_filter_points import counts_from_shoe, deal_ordered_hand

    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")
    outcome_history = _normalize_outcome_history(list(session.get("round_history") or []))
    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    prediction = predict(
        history=outcome_history,
        venue=str(session.get("venue") or ""),
        room=str(session.get("room") or ""),
        shoe_id=str(session.get("shoe_id") or ""),
        user_id=str(session.get("user_id") or ""),
        run_seed=seed,
        road_context=None,
    )
    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted_side = str(
        prediction.get("action")
        or prediction.get("recommend")
        or ""
    ).upper()
    actual = str(hand.outcome or "").upper()
    verdict = (
        "OBSERVE"
        if predicted_side == "O"
        else "TIE_SKIPPED"
        if actual == "T"
        else "HIT"
        if predicted_side == actual
        else "MISS"
    )
    prediction.update({
        "ok": True,
        "mode": "virtual_shoe_cmab_compatibility",
        "model_version": "CMAB-LINUCB-V1-VIRTUAL-COMPAT",
        "virtual_hand": hand_data,
        "virtual_outcome": actual,
        "virtual_outcome_text": hand_data["outcome_text"],
        "verdict": verdict,
        "verdict_text": {
            "HIT": "命中",
            "MISS": "未命中",
            "TIE_SKIPPED": "和局不計",
            "OBSERVE": "觀望／不計勝負",
        }[verdict],
        "cards_consumed": int(hand.cards_used),
        "remaining_cards_after": len(remaining_shoe),
        "remaining_counts_after": counts_from_shoe(remaining_shoe),
        "round_number": int(session.get("hand_number", 0) or 0) + 1,
        "warmup_rounds": int(session.get("warmup_rounds", 0) or 0),
        "bandit_learning_applied": False,
        "disclaimer": "虛擬相容模式方向由 cMAB 產生；虛擬結果不回寫正式 cMAB。",
    })
    return {"prediction": prediction, "hand": hand_data, "remaining_shoe": remaining_shoe}


def parse_point_observation(value: Any) -> None:
    return None


__all__ = [
    "DB_HOLDOUT",
    "ShadowBacktestController",
    "ShortTermTakeoverController",
    "parse_point_observation",
    "predict",
    "run_virtual_round",
]
