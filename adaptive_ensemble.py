"""牌路主導的自適應集成。

決策順序固定為：Full Road Pattern Model 建立完整路圖與牌路專家 →
Adaptive Ensemble 整合牌路專家並輸出正式方向 → Contextual Bandit 只保留
影子方向與線上學習。這裡不混入算牌、EV、粒子或外部模型，並永遠維持 B/P。

輸出的 B/P 是方向分數，不是未來開出機率或勝率保證。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping
import math

from performance_tracker import get_performance_summary


OUTCOMES = ("B", "P", "T")
# 新的正式融合／影子比較使用獨立版本標籤。舊版績效不會再影響這一版的
# component Brier 權重，避免部署後看似「又回到舊模型」。
ADAPTIVE_MODEL_VARIANT = "V34.2_ADAPTIVE_ROAD_PRIMARY_UCB_SHADOW"
# 方向架構參數固定在程式內，避免 Render 遺留的 ADAPTIVE_* 環境變數讓
# 同一張截圖在不同部署得到不同結果。
ADAPTIVE_ENABLED = True
ADAPTIVE_MIN_SAMPLES = 12
ADAPTIVE_MAX_SHARE = 1.0
ADAPTIVE_TEMPERATURE = 0.30
ADAPTIVE_CHAMPION_ENABLED = True
ADAPTIVE_CHAMPION_MIN_HISTORY = 8
ADAPTIVE_CHAMPION_MIN_EDGE = 0.015
ADAPTIVE_CHAMPION_MIN_RELIABILITY = 0.20
ADAPTIVE_CHAMPION_PERFORMANCE_SAMPLES = 30
ADAPTIVE_CHAMPION_OVERRIDE_BASE_EDGE = 0.035
ADAPTIVE_CHAMPION_TEMPERATURE = 0.85
ADAPTIVE_FUSION_TEMPERATURE = 0.80
ADAPTIVE_PLURALITY_STRENGTH = 0.12
ADAPTIVE_LOGIT_CAP = 2.20
ADAPTIVE_SHOE_MIN_SAMPLES = 12

# Adaptive 的正式成員只允許 Full Road 與既有五個路子專家。V34.2 起，
# cMAB 不再參與正式融合；它只保留自己的原始方向，供同一筆預測的影子
# 比對與線上學習使用。
ROAD_PRIMARY_COMPONENTS = (
    "structural_regime", "full_road", "short", "mid", "long", "pattern", "analogue",
)
FULL_ROAD_PRIMARY_MULTIPLIER = 1.40
STRUCTURAL_REGIME_PRIMARY_MULTIPLIER = 2.20
STRUCTURAL_RECENCY_REDUCTION = 0.55
# 保留常數名稱只為讀取舊診斷資料相容；正式融合固定為 0%，不可被 UCB
# 翻轉。這不是 Render 可覆寫的環境變數。
CONTEXTUAL_BANDIT_AUXILIARY_MAX_SHARE = 0.0

# V34.1 穩定性微調：同一牌靴剛好連中幾次時，不讓短期成績把某個成員
# 放大到 1.30 倍。12 手以上才採用，且權重只允許在 0.92～1.08 間小幅
# 校準；50% 命中率剛好是 1.00，不改變原本的基準融合。
CURRENT_SHOE_WEIGHT_MIN = 0.92
CURRENT_SHOE_WEIGHT_MAX = 1.08
CURRENT_SHOE_SELECTION_MIN = 0.95
CURRENT_SHOE_SELECTION_MAX = 1.05

# 結構規則與全牌路模型若方向衝突，保留 V34 的結構優先，但先扣掉一小段
# 結構權重，避免「剛形成三連」單獨壓過完整牌路的長視窗證據。
STRUCTURAL_FULL_ROAD_CONFLICT_FACTOR = 0.82


def _normalize(values: Mapping[str, Any]) -> Dict[str, float]:
    data = {key: max(1e-12, float(values.get(key, 0.0) or 0.0)) for key in OUTCOMES}
    total = sum(data.values())
    return {key: data[key] / total for key in OUTCOMES}


def _components(prediction: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """只收 Full Road Pattern Model 與既有牌路專家。

    舊版會讓 particle、Monte Carlo、算牌或其他遺留欄位混入候選，造成
    Adaptive 的主輸出不再純粹代表牌路。現在白名單只保留 road_model
    已產生、且由 Full Road Pattern Model 帶頭的六個成員。
    """
    result: Dict[str, Dict[str, float]] = {}
    nested = prediction.get("component_probabilities")
    if isinstance(nested, Mapping):
        for name, values in nested.items():
            normalized_name = str(name)
            if (
                normalized_name not in ROAD_PRIMARY_COMPONENTS
                or not isinstance(values, Mapping)
            ):
                continue
            try:
                result[normalized_name] = _normalize(values)
            except Exception:
                pass
    return result


def _select_component_champion(
    prediction: Mapping[str, Any],
    components: Mapping[str, Mapping[str, float]],
    historical_scores: Mapping[str, Any],
    component_sample_counts: Mapping[str, Any],
) -> Dict[str, Any]:
    """模型互相抵消時，選出一個可稽核的核心模型，不做盲目平均。"""
    road = prediction.get("road_support")
    road_data = dict(road) if isinstance(road, Mapping) else {}
    history_count = int(road_data.get("sample_count", 0) or 0)
    if history_count < ADAPTIVE_CHAMPION_MIN_HISTORY:
        return {}

    raw_models = road_data.get("models")
    model_metadata = dict(raw_models) if isinstance(raw_models, Mapping) else {}
    candidates = []
    for name, values in components.items():
        b = max(0.0, float(values.get("B", 0.0) or 0.0))
        p = max(0.0, float(values.get("P", 0.0) or 0.0))
        bp_total = b + p
        if bp_total <= 1e-12:
            continue
        banker = b / bp_total
        edge = abs(2.0 * banker - 1.0)
        direction = "B" if banker >= 0.5 else "P"

        raw_meta = model_metadata.get(name)
        meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        model_active = bool(
            meta.get("active", meta.get("ok", True))
        )
        reliability = max(
            0.0,
            min(1.0, float(meta.get("reliability", 0.0) or 0.0)),
        )
        if not model_active:
            continue
        if edge < ADAPTIVE_CHAMPION_MIN_EDGE:
            continue
        if reliability < ADAPTIVE_CHAMPION_MIN_RELIABILITY:
            continue

        performance_samples = int(component_sample_counts.get(name, 0) or 0)
        historical_brier = historical_scores.get(name)
        historical_quality = None
        if performance_samples >= ADAPTIVE_CHAMPION_PERFORMANCE_SAMPLES:
            try:
                score = max(0.0, float(historical_brier))
                historical_quality = math.exp(-score / 0.50)
            except Exception:
                historical_quality = None

        evidence_quality = (
            float(historical_quality)
            if historical_quality is not None
            else reliability
        )
        selection_score = edge * (0.45 + 0.55 * evidence_quality)
        candidates.append({
            "name": str(name),
            "direction": direction,
            "banker_probability": float(banker),
            "edge": float(edge),
            "reliability": float(reliability),
            "support": int(meta.get("support", 0) or 0),
            "performance_samples": performance_samples,
            "historical_brier": (
                float(historical_brier)
                if historical_quality is not None
                else None
            ),
            "historical_performance_used": historical_quality is not None,
            "selection_score": float(selection_score),
        })

    if not candidates:
        return {}
    candidates.sort(
        key=lambda item: (
            float(item["selection_score"]),
            int(item["performance_samples"]),
            float(item["reliability"]),
            float(item["edge"]),
            str(item["name"]),
        ),
        reverse=True,
    )
    winner = dict(candidates[0])
    probability = max(
        1e-6,
        min(1.0 - 1e-6, float(winner["banker_probability"])),
    )
    logit = math.log(probability / (1.0 - probability))
    sharpened = 1.0 / (
        1.0 + math.exp(-logit / ADAPTIVE_CHAMPION_TEMPERATURE)
    )
    winner.update({
        "tempered_banker_probability": float(sharpened),
        "tempered_edge": float(abs(2.0 * sharpened - 1.0)),
        "models_conflict": len({item["direction"] for item in candidates}) > 1,
        "candidate_count": len(candidates),
        "history_count": history_count,
        "candidates": candidates,
    })
    return winner


def _is_extreme_unseen(prediction: Mapping[str, Any]) -> bool:
    """接受新舊欄位，避免部署期間因版本差異再次造成訊號斷層。"""
    def flagged(model: Mapping[str, Any]) -> bool:
        braking = model.get("uncertainty_braking")
        predictor_signal = model.get("predictor_signal")
        return bool(
            model.get("is_extreme_unseen")
            or model.get("extreme_uncertainty_signal")
            or model.get("unknown_region_active")
            or (
                isinstance(braking, Mapping)
                and (
                    braking.get("is_extreme_unseen")
                    or braking.get("active")
                )
            )
            or (
                isinstance(predictor_signal, Mapping)
                and (
                    predictor_signal.get("is_extreme_unseen")
                    or predictor_signal.get("extreme_uncertainty")
                )
            )
        )

    if flagged(prediction):
        return True

    # 預留其他模型以巢狀結果接入的契約：任一成員亮起極端標籤，
    # 都優先於正常加權與樣本成熟度，立即觸發全局硬熔斷。
    for collection_name in (
        "model_predictions",
        "component_predictions",
        "ensemble_members",
        "models",
    ):
        collection = prediction.get(collection_name)
        if isinstance(collection, Mapping):
            members = collection.values()
        elif isinstance(collection, (list, tuple)):
            members = collection
        else:
            continue
        if any(
            isinstance(member, Mapping) and flagged(member)
            for member in members
        ):
            return True
    return False


def _base_confidence(prediction: Mapping[str, Any], base: Mapping[str, float]) -> float:
    """統一為 [0, 1]；優先沿用 cMAB 已有 quality_score。"""
    try:
        quality = float(prediction.get("quality_score", 0.0) or 0.0)
    except Exception:
        quality = 0.0
    if quality > 0.0:
        return max(0.0, min(1.0, quality))
    return max(0.0, min(1.0, abs(float(base["B"]) - float(base["P"]))))


def _conditional_banker(values: Mapping[str, Any]) -> float:
    banker = max(0.0, float(values.get("B", 0.0) or 0.0))
    player = max(0.0, float(values.get("P", 0.0) or 0.0))
    total = banker + player
    return 0.5 if total <= 1e-12 else banker / total


def _logit(probability: float) -> float:
    value = max(1e-6, min(1.0 - 1e-6, float(probability)))
    return max(
        -ADAPTIVE_LOGIT_CAP,
        min(ADAPTIVE_LOGIT_CAP, math.log(value / (1.0 - value))),
    )


def _current_shoe_performance_factor(
    posterior_accuracy: float,
    *,
    for_selection: bool = False,
) -> float:
    """把同鞋短期表現限制為小幅校準，而不是正回饋放大。

    後驗命中率 0.50 對應 1.00；即使短期全中或全錯，也不會讓某個模型
    被放大／削弱到足以取代完整牌路與結構證據。
    """
    accuracy = max(0.0, min(1.0, float(posterior_accuracy)))
    if for_selection:
        return (
            CURRENT_SHOE_SELECTION_MIN
            + (CURRENT_SHOE_SELECTION_MAX - CURRENT_SHOE_SELECTION_MIN)
            * accuracy
        )
    return (
        CURRENT_SHOE_WEIGHT_MIN
        + (CURRENT_SHOE_WEIGHT_MAX - CURRENT_SHOE_WEIGHT_MIN) * accuracy
    )


def _road_primary_fallback(
    prediction: Mapping[str, Any],
    components: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    """純牌路候選不足時的最後 B/P 決定，不讀取 cMAB 方向。

    正常情況下 ``full_road``、結構規律與其他牌路專家都會有候選列。只有
    新鞋初期、OCR 輸入很短或所有專家暫時 inactive 時才會進入此分支。
    此時仍優先讀 Full Road 的原始機率；完全沒有牌路資訊才給中性 B 作為
    產品強制 B/P 契約的固定冷啟動值。UCB 只會被標記為影子，不可接管。
    """
    for name in (
        "full_road", "structural_regime", "long", "mid", "short",
        "pattern", "analogue",
    ):
        values = components.get(name)
        if not isinstance(values, Mapping):
            continue
        banker = _conditional_banker(values)
        return {
            "direction": "B" if banker >= 0.5 else "P",
            "banker_probability": float(banker),
            "source": f"road_component_{name}_fallback",
        }

    road = prediction.get("road_support")
    road_data = dict(road) if isinstance(road, Mapping) else {}
    direction = str(road_data.get("direction") or "").upper().strip()
    if direction in {"B", "P"}:
        return {
            "direction": direction,
            "banker_probability": 0.51 if direction == "B" else 0.49,
            "source": "road_context_direction_fallback",
        }
    return {
        "direction": "B",
        "banker_probability": 0.50,
        "source": "empty_road_neutral_bp_fallback",
    }


def _fusion_candidates(
    prediction: Mapping[str, Any],
    base: Mapping[str, float],
    components: Mapping[str, Mapping[str, float]],
    historical_scores: Mapping[str, Any],
    component_sample_counts: Mapping[str, Any],
    shoe_direction_performance: Mapping[str, Any],
    base_confidence: float,
    *,
    include_contextual_bandit: bool = False,
) -> list[Dict[str, Any]]:
    """建立純牌路融合候選。

    Full Road Pattern Model 的完整歷史結論與五個既有路子專家是唯一主要
    成員。正式呼叫固定不加入 LinUCB；參數僅保留給離線舊版比對，不能從
    正式預測入口啟用。
    """
    road = prediction.get("road_support")
    road_data = dict(road) if isinstance(road, Mapping) else {}
    raw_models = road_data.get("models")
    metadata = dict(raw_models) if isinstance(raw_models, Mapping) else {}
    rows: list[Dict[str, Any]] = []
    for name in ROAD_PRIMARY_COMPONENTS:
        values = components.get(name)
        if not isinstance(values, Mapping):
            continue
        banker = _conditional_banker(values)
        edge = abs(2.0 * banker - 1.0)
        raw_meta = metadata.get(name)
        meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        if not bool(meta.get("active", meta.get("ok", True))):
            continue
        reliability = max(
            0.05,
            min(1.0, float(meta.get("reliability", 0.20) or 0.20)),
        )
        samples = int(component_sample_counts.get(name, 0) or 0)
        historical_brier = historical_scores.get(name)
        performance_used = samples >= ADAPTIVE_CHAMPION_PERFORMANCE_SAMPLES
        if performance_used:
            try:
                performance_quality = math.exp(
                    -max(0.0, float(historical_brier)) / 0.50
                )
            except Exception:
                performance_quality = 0.50
                performance_used = False
        else:
            performance_quality = 0.50
        weight = (
            (0.25 + 0.75 * reliability)
            * (0.55 + 0.45 * performance_quality)
            * (0.80 + 0.20 * edge)
        )
        if name == "full_road":
            # 完整路圖是第一層產物，給予溫和優先權；不是硬鎖方向，仍需
            # 接受其他近期牌路專家與 cMAB context 的交叉驗證。
            weight *= FULL_ROAD_PRIMARY_MULTIPLIER
        elif name == "structural_regime":
            # 已通過 run-length 確認的單跳／雙跳／跳跳龍，不能再被三個
            # 「最後一顆權重較高」的近期比例模型平均回原方向。
            weight *= STRUCTURAL_REGIME_PRIMARY_MULTIPLIER
        support = int(meta.get("support", 0) or 0)
        raw_shoe_performance = shoe_direction_performance.get(name)
        shoe_performance = (
            dict(raw_shoe_performance)
            if isinstance(raw_shoe_performance, Mapping)
            else {}
        )
        shoe_samples = int(shoe_performance.get("sample_count", 0) or 0)
        shoe_accuracy = float(
            shoe_performance.get("posterior_accuracy", 0.50) or 0.50
        )
        if shoe_samples >= ADAPTIVE_SHOE_MIN_SAMPLES:
            weight *= _current_shoe_performance_factor(shoe_accuracy)
        selection_score = weight * (
            0.55 + 0.25 * min(1.0, support / 12.0) + 0.20 * edge
        )
        if shoe_samples >= ADAPTIVE_SHOE_MIN_SAMPLES:
            selection_score *= _current_shoe_performance_factor(
                shoe_accuracy, for_selection=True
            )
        candidate_logit = _logit(banker)
        rows.append({
            "name": name,
            "banker_probability": float(banker),
            "direction": (
                "B" if candidate_logit > 0.0 else "P" if candidate_logit < 0.0 else ""
            ),
            "edge": float(edge),
            "logit": float(candidate_logit),
            "weight": float(weight),
            "reliability": float(reliability),
            "support": support,
            "performance_samples": samples,
            "historical_brier": (
                float(historical_brier) if performance_used else None
            ),
            "historical_performance_used": bool(performance_used),
            "selection_score": float(selection_score),
            "current_shoe_samples": shoe_samples,
            "current_shoe_posterior_accuracy": float(shoe_accuracy),
            "current_shoe_performance_used": bool(
                shoe_samples >= ADAPTIVE_SHOE_MIN_SAMPLES
            ),
            "role": "road_primary",
        })

    # Full Road 和結構規則同時存在卻不同向時，代表「近期局型」和「完整
    # 路紙」正出現分歧。只溫和收斂結構權重，並非取消 V34 的結構優先。
    structural_row = next(
        (row for row in rows if row["name"] == "structural_regime"), None
    )
    full_road_row = next(
        (row for row in rows if row["name"] == "full_road"), None
    )
    if structural_row is not None and full_road_row is not None:
        structural_direction = str(structural_row.get("direction") or "")
        full_road_direction = str(full_road_row.get("direction") or "")
        aligned = (
            structural_direction in {"B", "P"}
            and structural_direction == full_road_direction
        )
        structural_row["full_road_aligned"] = aligned
        if (
            structural_direction in {"B", "P"}
            and full_road_direction in {"B", "P"}
            and not aligned
        ):
            structural_row["weight"] = float(structural_row["weight"]) * (
                STRUCTURAL_FULL_ROAD_CONFLICT_FACTOR
            )
            structural_row["selection_score"] = float(
                structural_row["selection_score"]
            ) * STRUCTURAL_FULL_ROAD_CONFLICT_FACTOR
            structural_row["full_road_conflict_downweighted"] = True

    # 結構元件一旦啟用，近期比例仍保留參考，但降低其合計影響力。這個
    # 動作只對已確認結構生效；混合盤仍完全使用原本的多專家融合。
    structural_active = any(
        row["name"] == "structural_regime" for row in rows
    )
    if structural_active:
        for row in rows:
            if row["name"] in {"short", "mid", "long", "pattern", "analogue"}:
                row["weight"] = float(row["weight"]) * STRUCTURAL_RECENCY_REDUCTION

    primary_weight = sum(float(row["weight"]) for row in rows)
    bandit_banker = _conditional_banker(base)
    bandit_edge = abs(2.0 * bandit_banker - 1.0)
    bandit_logit = _logit(bandit_banker)
    bandit_direction = (
        "B" if bandit_logit > 0.0 else "P" if bandit_logit < 0.0 else ""
    )
    if include_contextual_bandit and primary_weight > 1e-12:
        # aux / (primary + aux) <= max_share，因此 cMAB 永遠是輔助，
        # 而不是把長龍或跳路重新平均成單純機率。
        auxiliary_weight = primary_weight * (
            CONTEXTUAL_BANDIT_AUXILIARY_MAX_SHARE
            / (1.0 - CONTEXTUAL_BANDIT_AUXILIARY_MAX_SHARE)
        )
        rows.append({
            "name": "contextual_bandit_auxiliary",
            "banker_probability": float(bandit_banker),
            "direction": bandit_direction,
            "edge": float(bandit_edge),
            "logit": float(bandit_logit),
            "weight": float(auxiliary_weight),
            "reliability": max(0.20, min(1.0, base_confidence)),
            "support": 0,
            "performance_samples": 0,
            "historical_brier": None,
            "historical_performance_used": False,
            "selection_score": float(auxiliary_weight),
            "current_shoe_samples": 0,
            "current_shoe_posterior_accuracy": 0.50,
            "current_shoe_performance_used": False,
            "role": "contextual_auxiliary",
        })
    elif include_contextual_bandit:
        # 路紙剛開始或 ROI 沒有可用路子成員時，仍強制輸出 B/P；這是唯一
        # 允許 cMAB 直接當方向來源的冷啟動例外。
        rows.append({
            "name": "contextual_bandit_fallback",
            "banker_probability": float(bandit_banker),
            "direction": bandit_direction,
            "edge": float(bandit_edge),
            "logit": float(bandit_logit),
            "weight": 1.0,
            "reliability": max(0.20, min(1.0, base_confidence)),
            "support": 0,
            "performance_samples": 0,
            "historical_brier": None,
            "historical_performance_used": False,
            "selection_score": float(max(0.01, bandit_edge)),
            "current_shoe_samples": 0,
            "current_shoe_posterior_accuracy": 0.50,
            "current_shoe_performance_used": False,
            "role": "cold_start_bandit_fallback",
        })
    return rows


def _plurality_logit_fusion(candidates: list[Dict[str, Any]]) -> Dict[str, Any]:
    active = [row for row in candidates if float(row["weight"]) > 0.0]
    if not active:
        return {}
    weight_total = sum(float(row["weight"]) for row in active) or 1.0
    pooled_logit = sum(
        float(row["weight"]) * float(row["logit"])
        for row in active
    ) / weight_total
    plurality_margin = sum(
        float(row["weight"])
        * (1.0 if float(row["logit"]) > 0.0 else -1.0 if float(row["logit"]) < 0.0 else 0.0)
        for row in active
    ) / weight_total
    plurality_evidence = sum(
        float(row["weight"])
        * math.sqrt(max(0.0, float(row["edge"])))
        * (1.0 if float(row["logit"]) > 0.0 else -1.0 if float(row["logit"]) < 0.0 else 0.0)
        for row in active
    ) / weight_total
    combined_logit = (
        pooled_logit
        + ADAPTIVE_PLURALITY_STRENGTH * plurality_evidence
    )
    tiebreaker: Dict[str, Any] = {}
    if abs(combined_logit) <= 1e-12:
        directional = [
            row for row in active if str(row["direction"]) in {"B", "P"}
        ]
        if directional:
            directional.sort(
                key=lambda row: (
                    float(row["selection_score"]),
                    int(row["performance_samples"]),
                    float(row["reliability"]),
                    float(row["edge"]),
                    str(row["name"]),
                ),
                reverse=True,
            )
            tiebreaker = dict(directional[0])
            combined_logit = float(tiebreaker["logit"])
    sharpened_logit = combined_logit / ADAPTIVE_FUSION_TEMPERATURE
    conditional_banker = 1.0 / (1.0 + math.exp(-sharpened_logit))
    winning_weight = sum(
        float(row["weight"])
        for row in active
        if (
            float(row["logit"]) > 0.0 and combined_logit > 0.0
        ) or (
            float(row["logit"]) < 0.0 and combined_logit < 0.0
        )
    )
    agreement = winning_weight / weight_total if abs(combined_logit) > 1e-12 else 0.0
    return {
        "conditional_banker": float(conditional_banker),
        "pooled_logit": float(pooled_logit),
        "plurality_margin": float(plurality_margin),
        "plurality_evidence": float(plurality_evidence),
        "combined_logit": float(combined_logit),
        "agreement": float(agreement),
        "models_conflict": len(
            {str(row["direction"]) for row in active if row["direction"]}
        ) > 1,
        "candidate_count": len(active),
        "tiebreaker": tiebreaker,
        "weights": {
            str(row["name"]): float(row["weight"]) / weight_total
            for row in active
        },
        "candidates": active,
    }


def _apply_plurality_decision(
    result: Dict[str, Any],
    *,
    base: Mapping[str, float],
    components: Mapping[str, Mapping[str, float]],
    historical_scores: Mapping[str, Any],
    component_sample_counts: Mapping[str, Any],
    shoe_direction_performance: Mapping[str, Any],
    shoe_sample_count: int,
    sample_count: int,
    base_confidence: float,
) -> Dict[str, Any]:
    if not ADAPTIVE_ENABLED:
        result["ensemble_confidence"] = float(base_confidence)
        result["confidence"] = float(base_confidence)
        result["hard_brake_active"] = False
        result["adaptive_ensemble"] = {
            "active": False,
            "mode": "disabled_by_configuration",
            "hard_brake_active": False,
            "sample_count": sample_count,
            "overall_confidence": float(base_confidence),
        }
        return result

    candidates = _fusion_candidates(
        result, base, components, historical_scores,
        component_sample_counts, shoe_direction_performance,
        base_confidence, include_contextual_bandit=False,
    )
    fusion = _plurality_logit_fusion(candidates)
    result["pre_adaptive_probabilities"] = dict(base)
    base_action = str(result.get("action") or "O").upper().strip()
    road_fallback = _road_primary_fallback(result, components)
    road_fallback_direction = str(road_fallback["direction"])
    road_fallback_banker = float(road_fallback["banker_probability"])

    # UCB 保留為同一手的影子候選，讓績效紀錄器可以和正式 Adaptive
    # 方向做公平比較；這些欄位絕不參與以下 B/P 的選擇或權重。
    ucb_shadow_direction = str(
        result.get("selected_arm")
        or result.get("base_bandit_direction")
        or base_action
        or "B"
    ).upper().strip()
    if ucb_shadow_direction not in {"B", "P"}:
        ucb_shadow_direction = "B"
    ucb_shadow_banker = _conditional_banker(base)
    combined_logit = float(fusion.get("combined_logit", 0.0) or 0.0)
    exact_tie = not fusion or abs(combined_logit) <= 1e-12
    road_tie_values = [
        max(0.0, float(dict(components.get(str(row["name"])) or {}).get("T", 0.0) or 0.0))
        for row in candidates
        if isinstance(components.get(str(row["name"])), Mapping)
    ]
    tie_probability = max(
        0.0,
        min(0.30, sum(road_tie_values) / len(road_tie_values))
    ) if road_tie_values else 0.0
    bp_mass = 1.0 - tie_probability
    # 牌路專家完全抵消時仍強制 B/P，但回到 Full Road 優先的牌路 fallback；
    # 不能退回 cMAB，否則 UCB 又會在冷啟動或衝突局偷偷接管正式方向。
    conditional_banker = (
        road_fallback_banker
        if exact_tie
        else float(
            fusion.get("conditional_banker", road_fallback_banker)
            or road_fallback_banker
        )
    )
    banker = bp_mass * conditional_banker
    player = bp_mass * (1.0 - conditional_banker)
    direction = (
        road_fallback_direction
        if exact_tie
        else "B" if combined_logit > 0.0 else "P"
    )
    edge = abs(2.0 * conditional_banker - 1.0)
    agreement = float(fusion.get("agreement", 0.0) or 0.0)
    maturity = min(1.0, sample_count / max(1.0, float(ADAPTIVE_MIN_SAMPLES)))
    confidence = (
        min(
            0.78,
            0.36 + 0.22 * agreement + 0.10 * maturity
            + 0.10 * min(1.0, abs(combined_logit))
            + 0.10 * min(1.0, edge),
        )
    )
    tiebreaker = dict(fusion.get("tiebreaker") or {})
    if tiebreaker:
        reason = (
            f"加權意見完全抵消，改由證據品質最高的 {tiebreaker['name']} "
            "依可靠度、support 與已結算表現破局。"
        )
    elif exact_tie:
        reason = (
            "牌路專家暫時完全抵消，依 Full Road 優先的牌路 fallback "
            f"（{road_fallback['source']}）強制輸出 B/P；UCB 僅留作影子比較。"
        )
    else:
        reason = (
            "以 reliability／Brier 權重進行 logit pooling，"
            "再以 plurality evidence 放大非零方向優勢。"
        )

    result.update({
        "probabilities": {"B": banker, "P": player, "T": tie_probability},
        "banker_rate": round(banker * 100.0, 2),
        "player_rate": round(player * 100.0, 2),
        "tie_rate": round(tie_probability * 100.0, 2),
        "recommend": direction,
        "recommend_text": "莊" if direction == "B" else "閒",
        "action": direction,
        "action_text": "莊" if direction == "B" else "閒",
        "internal_recommend": direction,
        "internal_action": direction,
        "next_round_direction": direction,
        "next_round_direction_text": "莊" if direction == "B" else "閒",
        "signal_allowed": True,
        "signal_status_code": "ROAD_PRIMARY_ADAPTIVE_DIRECTION",
        "signal_status_text": "牌路主導 Adaptive Ensemble：正式方向已啟用",
        "signal_reason": reason,
        "internal_signal_reason": reason,
        "direction_source": (
            "adaptive_road_fallback"
            if exact_tie
            else "adaptive_ensemble_road_primary"
        ),
        "adaptive_only_direction": direction,
        "adaptive_only_probabilities": {
            "B": float(banker), "P": float(player), "T": float(tie_probability),
        },
        "ucb_shadow_direction": ucb_shadow_direction,
        "ucb_shadow_banker_probability": float(ucb_shadow_banker),
        "ucb_shadow_agrees": bool(ucb_shadow_direction == direction),
        "ucb_influenced_final_direction": False,
        "direction_edge": float(edge),
        "direction_edge_percent": round(edge * 100.0, 4),
        "ensemble_confidence": float(confidence),
        "confidence": float(confidence),
        "quality_score": float(confidence),
        "confidence_label": "較高" if confidence >= 0.72 else "中等" if confidence >= 0.56 else "偏低" if confidence > 0.0 else "零方向",
        "bet_multiplier": 1.0,
        "hard_brake_active": False,
        "is_extreme_unseen": False,
        "probability_semantics": "softmax_direction_score_not_guaranteed_outcome_probability",
    })
    if tiebreaker:
        result["component_champion"] = tiebreaker
    result["adaptive_ensemble"] = {
        "active": True,
        "mode": "adaptive_ensemble_road_primary_ucb_shadow",
        "circuit_breaker_active": False,
        "hard_brake_active": False,
        "is_extreme_unseen": False,
        "sample_count": sample_count,
        "current_shoe_sample_count": shoe_sample_count,
        "base_action_before": base_action,
        "candidate_count": int(fusion.get("candidate_count", 0) or 0),
        "models_conflict": bool(fusion.get("models_conflict", False)),
        "pooled_logit": float(fusion.get("pooled_logit", 0.0) or 0.0),
        "plurality_margin": float(fusion.get("plurality_margin", 0.0) or 0.0),
        "plurality_evidence": float(fusion.get("plurality_evidence", 0.0) or 0.0),
        "combined_logit": combined_logit,
        "temperature": ADAPTIVE_FUSION_TEMPERATURE,
        "plurality_strength": ADAPTIVE_PLURALITY_STRENGTH,
        "exact_50_50": bool(exact_tie),
        "tie_breaker_used": bool(tiebreaker),
        "tie_breaker_model": str(tiebreaker.get("name") or ""),
        "agreement": agreement,
        "weights": {name: round(float(value), 8) for name, value in dict(fusion.get("weights") or {}).items()},
        "candidates": list(fusion.get("candidates") or []),
        "overall_confidence": float(confidence),
        "final_action": direction,
        "bet_multiplier": 1.0,
        "road_primary": True,
        "contextual_bandit_role": "shadow_only_no_final_weight",
        "adaptive_only_direction": direction,
        "ucb_shadow": {
            "direction": ucb_shadow_direction,
            "conditional_banker_probability": float(ucb_shadow_banker),
            "agrees_with_adaptive": bool(ucb_shadow_direction == direction),
            "can_change_final_direction": False,
        },
        "fallback_required": False,
        "probability_semantics": result["probability_semantics"],
        "reason": reason,
    }
    return result


def adapt_prediction(
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> Dict[str, Any]:
    result = dict(prediction or {})
    base = _normalize(
        result.get("probabilities")
        if isinstance(result.get("probabilities"), Mapping)
        else {
            "B": float(result.get("banker_rate", 0.0) or 0.0) / 100.0,
            "P": float(result.get("player_rate", 0.0) or 0.0) / 100.0,
            "T": float(result.get("tie_rate", 0.0) or 0.0) / 100.0,
        }
    )
    result.setdefault("raw_probabilities", dict(base))
    components = _components(result)
    summary = get_performance_summary(
        venue=venue,
        room=room,
        model_variant=ADAPTIVE_MODEL_VARIANT,
        limit=5000,
    )
    sample_count = int(summary.get("sample_count", 0) or 0)
    historical_scores = dict(summary.get("component_brier_scores") or {})
    component_sample_counts = dict(
        summary.get("component_sample_counts") or {}
    )
    shoe_summary = (
        get_performance_summary(
            venue=venue,
            room=room,
            shoe_id=shoe_id,
            model_variant=ADAPTIVE_MODEL_VARIANT,
            limit=500,
        )
        if str(shoe_id or "").strip()
        else {}
    )
    shoe_direction_performance = dict(
        shoe_summary.get("component_direction_performance") or {}
    )
    extreme_unseen = _is_extreme_unseen(result)
    base_confidence = _base_confidence(result, base)

    if extreme_unseen:
        # 保留極端未知診斷，但產品契約要求每局皆輸出 B/P。V34.2 的正式
        # fallback 仍必須來自牌路：UCB 原始 Arm 只寫入影子欄位，不能在
        # 混沌局反而取得最終方向控制權。
        learning_arm = str(result.get("selected_arm") or "").upper().strip()
        if learning_arm not in {"B", "P"}:
            learning_arm = "B" if float(base.get("B", 0.5)) >= float(base.get("P", 0.5)) else "P"
        road_fallback = _road_primary_fallback(result, components)
        road_direction = str(road_fallback["direction"])
        road_banker = float(road_fallback["banker_probability"])
        result["pre_hard_brake_probabilities"] = dict(base)
        result["pre_hard_brake_recommend"] = str(
            result.get("recommend") or learning_arm
        )
        road_tie_values = [
            max(0.0, float(dict(values).get("T", 0.0) or 0.0))
            for values in components.values() if isinstance(values, Mapping)
        ]
        tie_probability = max(
            0.0,
            min(0.30, sum(road_tie_values) / len(road_tie_values))
        ) if road_tie_values else 0.0
        conditional_banker = road_banker
        neutral_bp = 1.0 - tie_probability
        result["probabilities"] = {
            "B": neutral_bp * conditional_banker,
            "P": neutral_bp * (1.0 - conditional_banker),
            "T": tie_probability,
        }
        result["banker_rate"] = round(result["probabilities"]["B"] * 100.0, 2)
        result["player_rate"] = round(result["probabilities"]["P"] * 100.0, 2)
        result["tie_rate"] = round(tie_probability * 100.0, 2)
        result["recommend"] = road_direction
        result["recommend_text"] = "莊" if road_direction == "B" else "閒"
        result["action"] = road_direction
        result["action_text"] = result["recommend_text"]
        result["internal_recommend"] = road_direction
        result["internal_action"] = road_direction
        result["next_round_direction"] = road_direction
        result["next_round_direction_text"] = result["recommend_text"]
        result["signal_allowed"] = True
        result["signal_status_code"] = "CHAOS_ADAPTIVE_ROAD_FALLBACK"
        result["signal_status_text"] = "統計混沌：Adaptive 牌路 fallback 已啟用"
        result["signal_reason"] = (
            "牌路模型回報極端未知區間；保留診斷標記，但正式方向仍由"
            f"牌路 fallback（{road_fallback['source']}）輸出，UCB 僅做影子比較。"
        )
        result["internal_signal_reason"] = result["signal_reason"]
        result["direction_source"] = "adaptive_road_chaos_fallback"
        result["adaptive_only_direction"] = road_direction
        result["adaptive_only_probabilities"] = dict(result["probabilities"])
        result["ucb_shadow_direction"] = learning_arm
        result["ucb_shadow_banker_probability"] = float(
            _conditional_banker(base)
        )
        result["ucb_shadow_agrees"] = bool(learning_arm == road_direction)
        result["ucb_influenced_final_direction"] = False
        result["ensemble_confidence"] = min(0.45, base_confidence)
        result["confidence"] = min(0.45, base_confidence)
        result["quality_score"] = min(0.45, base_confidence)
        result["confidence_label"] = "偏低"
        result["bet_multiplier"] = 1.0
        result["hard_brake_active"] = False
        result["chaos_diagnostic_active"] = True
        # selected_arm 不覆寫：即使不下注，實際結果仍可用固定 1 倍
        # 被動更新模型與共享 context 方差。
        if learning_arm in {"B", "P"}:
            result["selected_arm"] = learning_arm
        result["is_extreme_unseen"] = True
        result["adaptive_ensemble"] = {
            "active": True,
            "mode": "statistical_chaos_adaptive_road_ucb_shadow",
            "circuit_breaker_active": False,
            "hard_brake_active": False,
            "sample_count": sample_count,
            "minimum_samples_bypassed_for_safety": True,
            "is_extreme_unseen": True,
            "variance": float(result.get("variance", 0.0) or 0.0),
            "bandit_weight_before": 0.0,
            "bandit_weight_after": 0.0,
            "weight_reduction_ratio": 0.0,
            "weight_reduction_percent": 0.0,
            "alternative_model_weight": 0.0,
            "alternative_components_available": sorted(components),
            "alternative_fusion_attempted": False,
            "fallback_required": True,
            "shadow_backtest_required": True,
            "overall_confidence": min(0.45, base_confidence),
            "final_action": road_direction,
            "bet_multiplier": 1.0,
            "road_primary": True,
            "contextual_bandit_role": "shadow_only_no_final_weight",
            "adaptive_only_direction": road_direction,
            "ucb_shadow": {
                "direction": learning_arm,
                "conditional_banker_probability": float(
                    _conditional_banker(base)
                ),
                "agrees_with_adaptive": bool(learning_arm == road_direction),
                "can_change_final_direction": False,
            },
            "reason": "極端未知區間保留診斷，正式方向仍由 Adaptive 牌路 fallback 輸出",
        }
        return result

    # 正常區間一律走新版 logit/plurality 決策；下方舊版分支保留於
    # 部署過渡期方便比對，但不再進入，避免再次平均回 50/50。
    return _apply_plurality_decision(
        result,
        base=base,
        components=components,
        historical_scores=historical_scores,
        component_sample_counts=component_sample_counts,
        shoe_direction_performance=shoe_direction_performance,
        shoe_sample_count=int(shoe_summary.get("sample_count", 0) or 0),
        sample_count=sample_count,
        base_confidence=base_confidence,
    )

    champion = _select_component_champion(
        result,
        components,
        historical_scores,
        component_sample_counts,
    ) if ADAPTIVE_CHAMPION_ENABLED else {}
    base_action = str(result.get("action") or "O").upper().strip()
    try:
        base_edge = max(0.0, float(result.get("direction_edge", 0.0) or 0.0))
    except Exception:
        base_edge = 0.0
    champion_takeover = bool(
        champion
        and (
            base_action not in {"B", "P"}
            or (
                bool(champion.get("models_conflict"))
                and base_edge < ADAPTIVE_CHAMPION_OVERRIDE_BASE_EDGE
            )
        )
    )
    if champion_takeover:
        # 低優勢或模型衝突時不再把所有方向平均回 50/50；改由當前
        # 可靠度最高、或已有足夠歷史 Brier 證據的單一核心模型接管。
        # 真正 extreme_unseen 已在上方返回，因此此分支絕不繞過硬熔斷。
        result["pre_champion_probabilities"] = dict(base)
        tie_probability = max(0.0, min(0.30, float(base.get("T", 0.0))))
        bp_mass = 1.0 - tie_probability
        conditional_banker = float(
            champion["tempered_banker_probability"]
        )
        banker = bp_mass * conditional_banker
        player = bp_mass * (1.0 - conditional_banker)
        direction = str(champion["direction"])
        champion_edge = float(champion["tempered_edge"])
        champion_confidence = max(
            0.50,
            min(
                0.78,
                0.44
                + 0.24 * float(champion["reliability"])
                + 0.22 * min(1.0, champion_edge),
            ),
        )
        result["probabilities"] = {
            "B": float(banker),
            "P": float(player),
            "T": float(tie_probability),
        }
        result["banker_rate"] = round(banker * 100.0, 2)
        result["player_rate"] = round(player * 100.0, 2)
        result["tie_rate"] = round(tie_probability * 100.0, 2)
        result["recommend"] = direction
        result["recommend_text"] = "莊" if direction == "B" else "閒"
        result["action"] = direction
        result["action_text"] = result["recommend_text"]
        result["internal_recommend"] = direction
        result["internal_action"] = direction
        result["next_round_direction"] = direction
        result["next_round_direction_text"] = result["recommend_text"]
        result["signal_allowed"] = True
        result["signal_status_code"] = "DYNAMIC_CHAMPION_DIRECTION"
        result["signal_status_text"] = "核心模型優先：正式方向已啟用"
        result["signal_reason"] = (
            f"多模型低優勢／衝突時停止平均抵消，改由核心模型 "
            f"{champion['name']} 依"
            + (
                f" {champion['performance_samples']} 局歷史 Brier 表現"
                if champion["historical_performance_used"]
                else "當前牌路可靠度與方向差"
            )
            + "接管莊／閒方向。"
        )
        result["internal_signal_reason"] = result["signal_reason"]
        result["direction_source"] = "adaptive_ensemble_dynamic_champion"
        result["direction_edge"] = champion_edge
        result["direction_edge_percent"] = round(champion_edge * 100.0, 4)
        result["ensemble_confidence"] = champion_confidence
        result["confidence"] = champion_confidence
        result["quality_score"] = max(
            float(result.get("quality_score", 0.0) or 0.0),
            champion_confidence,
        )
        result["confidence_label"] = (
            "較高"
            if champion_confidence >= 0.72
            else "中等"
            if champion_confidence >= 0.56
            else "偏低"
        )
        result["bet_multiplier"] = 1.0
        result["hard_brake_active"] = False
        result["is_extreme_unseen"] = False
        result["component_champion"] = dict(champion)
        result["adaptive_ensemble"] = {
            "active": True,
            "mode": "dynamic_component_champion",
            "circuit_breaker_active": False,
            "hard_brake_active": False,
            "is_extreme_unseen": False,
            "sample_count": sample_count,
            "base_action_before": base_action,
            "base_edge_before": base_edge,
            "models_conflict": bool(champion["models_conflict"]),
            "champion_model": str(champion["name"]),
            "champion_direction": direction,
            "champion_edge": champion_edge,
            "champion_reliability": float(champion["reliability"]),
            "historical_performance_used": bool(
                champion["historical_performance_used"]
            ),
            "overall_confidence": champion_confidence,
            "final_action": direction,
            "bet_multiplier": 1.0,
            "fallback_required": False,
            "reason": "低優勢／衝突時由單一核心模型接管，避免平均抵消",
        }
        return result

    eligible = {
        name: probabilities
        for name, probabilities in components.items()
        if name in historical_scores
        and int(component_sample_counts.get(name, 0) or 0) >= ADAPTIVE_MIN_SAMPLES
    }
    active = bool(
        ADAPTIVE_ENABLED
        and sample_count >= ADAPTIVE_MIN_SAMPLES
        and len(eligible) >= 2
        and ADAPTIVE_MAX_SHARE > 0
    )

    if not active:
        result["ensemble_confidence"] = float(base_confidence)
        result["confidence"] = float(base_confidence)
        result["hard_brake_active"] = False
        result["adaptive_ensemble"] = {
            "active": False,
            "circuit_breaker_active": False,
            "hard_brake_active": False,
            "is_extreme_unseen": False,
            "sample_count": sample_count,
            "minimum_samples": ADAPTIVE_MIN_SAMPLES,
            "reason": "已結算樣本不足或可比較子模型不足",
            "effective_share": 0.0,
            "bandit_weight_before": 1.0,
            "bandit_weight_after": 1.0,
            "weight_reduction_ratio": 0.0,
            "fallback_required": False,
            "overall_confidence": float(base_confidence),
        }
        return result

    raw_weights = {
        name: math.exp(-float(historical_scores[name]) / ADAPTIVE_TEMPERATURE)
        for name in eligible
    }
    weight_total = sum(raw_weights.values()) or 1.0
    weights = {name: value / weight_total for name, value in raw_weights.items()}
    adaptive = {
        outcome: sum(weights[name] * eligible[name][outcome] for name in eligible)
        for outcome in OUTCOMES
    }
    maturity = min(1.0, (sample_count - ADAPTIVE_MIN_SAMPLES + 1) / max(1.0, ADAPTIVE_MIN_SAMPLES * 3.0))
    share = ADAPTIVE_MAX_SHARE * maturity
    blended = _normalize({
        outcome: base[outcome] * (1.0 - share) + adaptive[outcome] * share
        for outcome in OUTCOMES
    })
    result["probabilities"] = blended
    result["banker_rate"] = round(blended["B"] * 100.0, 2)
    result["player_rate"] = round(blended["P"] * 100.0, 2)
    result["tie_rate"] = round(blended["T"] * 100.0, 2)
    result["ensemble_confidence"] = float(base_confidence)
    result["confidence"] = float(base_confidence)
    result["hard_brake_active"] = False
    result["adaptive_ensemble"] = {
        "active": True,
        "circuit_breaker_active": False,
        "hard_brake_active": False,
        "is_extreme_unseen": False,
        "sample_count": sample_count,
        "minimum_samples": ADAPTIVE_MIN_SAMPLES,
        "effective_share": round(share, 6),
        "bandit_weight_before": 1.0,
        "bandit_weight_after": round(1.0 - share, 6),
        "weight_reduction_ratio": round(share, 6),
        "fallback_required": False,
        "overall_confidence": float(base_confidence),
        "weights": {name: round(value, 6) for name, value in weights.items()},
        "component_brier_scores": {
            name: round(float(historical_scores[name]), 6) for name in eligible
        },
    }
    return result


__all__ = ["adapt_prediction"]
