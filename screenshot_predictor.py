"""遊戲截圖 cMAB 預測轉接層。

完整 B/P/T 歷史與牌路上下文送入 cMAB；若呼叫端明確提供經驗證的 10 維
``shoe_context.remaining_counts``，同時交由物理 EV／策略 Bandit 管線。
``prior_counts`` 與 ``observed_cards`` 仍只保留舊呼叫端相容，不可自動升格。

人工回報時以 session 保存的 prediction_id 精確結算上一筆 reward，
再把本局結果加入歷史並產生下一局預測，避免 latest-pending 競態與重複結算。
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from performance_tracker import record_prediction, resolve_latest_prediction
from predictor import predict
from road_model import build_road_context

# 正式畫面模型固定啟用「上一筆 prediction_id 精準結算」。不再讓舊的
# PERFORMANCE_TRACKING_ENABLED 環境變數意外關閉線上標記與時間軸防呆。
PERFORMANCE_TRACKING_ENABLED = True


def _clean_raw(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for item in values:
        raw = item.get("outcome") if isinstance(item, Mapping) else item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out[-2000:]


def _clean_bp(values: Iterable[Any]) -> list[str]:
    return [value for value in _clean_raw(values) if value in {"B", "P"}][-1000:]


def predict_from_screenshot(
    sequence: Iterable[Any], *, remaining_cards: Optional[int],
    raw_outcomes: Optional[Iterable[Any]] = None,
    tie_markers: Optional[Mapping[str, Any]] = None,
    prior_counts: Optional[Sequence[int]] = None,
    observed_cards: Optional[Iterable[Any]] = None,
    shoe_context: Optional[Mapping[str, Any]] = None,
    venue: str = "", room: str = "", shoe_id: str = "", user_id: str = "",
    run_seed: Optional[int] = None,
    road_context: Optional[Mapping[str, Any]] = None,
    screen_metadata: Optional[Mapping[str, Any]] = None,
    initial_grid_cells: Optional[Sequence[Mapping[str, Any]]] = None,
    initial_image_history: Optional[Iterable[Any]] = None,
    manual_outcome_history: Optional[Iterable[Any]] = None,
    previous_prediction_id: str = "",
    latest_actual_outcome: str = "",
    record_for_learning: bool = True,
) -> Dict[str, Any]:
    initial_raw = _clean_raw(initial_image_history or [])
    manual_raw = _clean_raw(manual_outcome_history or [])
    supplied_raw = _clean_raw(raw_outcomes or [])
    combined_raw = initial_raw + manual_raw if (initial_raw or manual_raw) else supplied_raw
    if not combined_raw:
        combined_raw = _clean_raw(sequence)
    cleaned = _clean_bp(combined_raw)

    if run_seed is None:
        seed_payload = "|".join((
            "".join(combined_raw),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
        ))
        seed = int.from_bytes(
            sha256(seed_payload.encode("utf-8")).digest()[:4],
            byteorder="big",
            signed=False,
        )
    else:
        seed = int(run_seed) & 0xFFFFFFFF
    context = dict(road_context or build_road_context(
        combined_raw,
        seed=(seed ^ 0x9E3779B9) & 0xFFFFFFFF,
        grid_cells=list(initial_grid_cells or []),
        initial_image_count=len(initial_raw),
        manual_count=len(manual_raw),
    ))
    metadata = dict(screen_metadata or {})

    # 只有呼叫端「明確告知」這次確實新增加了一局結果時，才能結算上一筆
    # pending prediction。絕不能以 manual_raw[-1] 推論：同一份歷史在
    # 使用者重按「開始 AI 判斷」時仍會有最後一局，若直接拿它當答案，會
    # 把已包含於特徵的資料回灌模型，造成目標洩漏與錯誤的線上學習。
    latest_actual = str(latest_actual_outcome or "").upper().strip()
    latest_actual_is_new = latest_actual in {"B", "P", "T"}
    previous_resolution = None
    if (
        PERFORMANCE_TRACKING_ENABLED
        and record_for_learning
        and user_id
        and latest_actual_is_new
    ):
        previous_resolution = resolve_latest_prediction(
            user_id,
            latest_actual,
            venue=venue,
            room=room,
            prediction_id=str(previous_prediction_id or ""),
        )

    # 只有呼叫端明確帶入的 ``remaining_counts`` 可以進物理決策；截圖
    # OCR 的剩餘總張數、prior_counts 與 observed_cards 都不能替代它。
    prior_counts_ignored = bool(prior_counts)
    observed_cards_ignored = bool(observed_cards)
    physical_context = dict(shoe_context or {})

    result = predict(
        history=combined_raw,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
        user_id=user_id,
        run_seed=seed,
        shoe_context=physical_context,
        road_context=context,
    )
    result.update({
        # 保留 contextual_bandit 回傳的真實模型版本，避免績效資料把
        # V1.5 誤標成舊 V1；此欄只描述畫面輸入管線。
        "screen_pipeline_version": "CMAB-PHYSICAL-STRATEGY-GRID-AWARE-V1",
        "mode": "screen_full_history_strategy_bandit",
        "shoe_id": str(shoe_id or ""),
        "screen_remaining_cards": int(remaining_cards or 0),
        "estimated_remaining_counts": [],
        "composition_source": str(
            dict(result.get("physical_signal") or {}).get("source") or "not_supplied"
        ),
        "composition_quality": str(
            result.get("composition_quality") or "unavailable_or_untrusted"
        ),
        "exact_remaining_counts_supplied": bool(
            isinstance(physical_context.get("remaining_counts"), (list, tuple))
            and len(physical_context.get("remaining_counts") or []) == 10
        ),
        "prior_counts_ignored": prior_counts_ignored,
        "observed_cards_ignored": observed_cards_ignored,
        "road_sequence_length": len(cleaned),
        "raw_outcome_length": len(combined_raw),
        "initial_image_count": len(initial_raw),
        "manual_round_count": len(manual_raw),
        "combined_round_count": len(combined_raw),
        "full_history_used_count": len(combined_raw),
        "initial_grid_cells": list(initial_grid_cells or []),
        "tie_count": sum(value == "T" for value in combined_raw),
        "tie_markers": dict(tie_markers or {}),
        "road_support": context,
        "road_pipeline_completed": True,
        "screen_metadata": metadata,
        "screen_input_type": str(metadata.get("input_type") or "unknown"),
        "virtual_only": False,
        "external_screen_input": True,
        "previous_prediction_resolved_before_next": bool(previous_resolution),
        "previous_prediction_id": str(previous_prediction_id or ""),
        "learning_update_triggered_by_new_actual": latest_actual_is_new,
        "deterministic_feature_seed": True,
    })
    result["road_fusion"] = {
        "applied": True,
        "mode": "context_features_only",
        "reason": "牌路資訊作為 cMAB 上下文，不再以 Stacking 機率融合",
    }

    if PERFORMANCE_TRACKING_ENABLED and record_for_learning and user_id:
        result["prediction_id"] = record_prediction(
            user_id,
            result,
            venue=venue,
            room=room,
            metadata={
                **metadata,
                "initial_image_count": len(initial_raw),
                "manual_round_count": len(manual_raw),
                "combined_round_count": len(combined_raw),
                "shoe_id": str(shoe_id or ""),
            },
        )
        result["performance_tracking"] = True
    else:
        result["performance_tracking"] = False
    return result


__all__ = ["predict_from_screenshot"]
