"""BGS cMAB 統一預測入口。

正式圖片／真人桌預測只使用完整 B/P/T 歷史、牌路上下文與 LinUCB cMAB。
不再匯入或執行粒子、超幾何、蒙地卡羅、Stacking、舊自適應集成或舊校準器。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from contextual_bandit import predict_bandit
from particle_filter_points import counts_from_shoe, deal_ordered_hand

DB_HOLDOUT: Dict[str, Any] = {
    "status": "removed",
    "replacement": "CMAB-LINUCB-V1",
    "note": "舊粒子／有限牌組驗證層已從主要預測流程移除",
}


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
    result = predict_bandit(
        cleaned,
        road_context=dict(road_context or {}),
        venue=venue,
        room=room,
        user_id=user_id,
        run_seed=run_seed,
    )
    result.update({
        "shoe_id": str(shoe_id or ""),
        "composition_quality": "not_applicable_cmab",
        "remaining_counts_source": "not_used",
        "shoe_context_ignored": bool(shoe_context),
        "road_quality_ok": bool(dict(road_context or {}).get(
            "quality_ok", dict(road_context or {}).get("recognition_quality_ok", True)
        )),
        "input_required": False,
    })
    return result


def run_virtual_round(session: Mapping[str, Any], run_seed: Optional[int] = None) -> Dict[str, Any]:
    """保留舊虛擬牌靴介面，但方向同樣由 cMAB 產生。"""
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
    predicted_side = str(prediction.get("recommend") or "").upper()
    actual = str(hand.outcome or "").upper()
    verdict = "TIE_SKIPPED" if actual == "T" else "HIT" if predicted_side == actual else "MISS"
    prediction.update({
        "ok": True,
        "mode": "virtual_shoe_cmab_compatibility",
        "model_version": "CMAB-LINUCB-V1-VIRTUAL-COMPAT",
        "virtual_hand": hand_data,
        "virtual_outcome": actual,
        "virtual_outcome_text": hand_data["outcome_text"],
        "verdict": verdict,
        "verdict_text": {"HIT": "命中", "MISS": "未命中", "TIE_SKIPPED": "和局不計"}[verdict],
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


__all__ = ["DB_HOLDOUT", "parse_point_observation", "predict", "run_virtual_round"]
