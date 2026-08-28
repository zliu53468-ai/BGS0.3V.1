"""Compatibility bridge for displaying 5%-30% bankroll sizing in the legacy LINE panel.

The prediction engine may keep a stricter Edge gate for its internal economic
diagnostics. The LINE panel, however, should always show a bankroll amount when
there is a formal B/P recommendation and the user has set a bankroll.

Display sizing rules:
1) If MoneyManagementModel already produced a positive 5%-30% ratio, reuse it.
2) Otherwise derive a conservative display-only ratio from the model confidence:
       5% + confidence * 25%
   and clamp it to 5%-30%.
3) Do not alter Markov/HSMM/shoe/road/hazard predictions or the underlying
   MoneyManagementModel formula.
"""
from __future__ import annotations

from typing import Any, Mapping
import sys

MIN_DISPLAY_RATIO = 0.05
MAX_DISPLAY_RATIO = 0.30


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _model_confidence(result: Mapping[str, Any]) -> float:
    """Return the best available bounded model-confidence value."""
    for key in (
        "pattern_calibrated_confidence",
        "confidence",
        "ensemble_confidence",
        "quality_score",
    ):
        try:
            value = float(result.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return _clip(value)
    return 0.0


def apply_edge_bankroll_advice(
    prediction: Mapping[str, Any],
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose a visible 5%-30% bankroll amount for every formal B/P signal."""
    result = dict(prediction or {})
    money = dict(result.get("money_management") or {})

    try:
        bankroll = max(0.0, float(session.get("bankroll", 0.0) or 0.0))
    except (TypeError, ValueError):
        bankroll = 0.0

    action = str(
        result.get("action")
        or result.get("direction")
        or result.get("recommend")
        or ""
    ).upper().strip()
    formal_bp_signal = action in {"B", "P"}

    try:
        core_ratio = float(
            money.get(
                "final_bet_ratio",
                result.get("final_bet_ratio", 0.0),
            )
            or 0.0
        )
    except (TypeError, ValueError):
        core_ratio = 0.0

    core_ratio = _clip(core_ratio, 0.0, MAX_DISPLAY_RATIO)
    core_bet_allowed = bool(
        money.get(
            "bet_allowed",
            result.get("bet_allowed", result.get("risk_gate_open", False)),
        )
    )

    if bankroll > 0.0 and formal_bp_signal:
        if core_bet_allowed and core_ratio > 0.0:
            ratio = _clip(core_ratio, MIN_DISPLAY_RATIO, MAX_DISPLAY_RATIO)
            source = "existing_money_management_ratio"
            level = "模型動態配注"
        else:
            confidence = _model_confidence(result)
            ratio = _clip(
                MIN_DISPLAY_RATIO
                + confidence * (MAX_DISPLAY_RATIO - MIN_DISPLAY_RATIO),
                MIN_DISPLAY_RATIO,
                MAX_DISPLAY_RATIO,
            )
            source = "confidence_fallback_5_to_30"
            level = "模型信心配注"

        percentage = ratio * 100.0
        amount = bankroll * ratio
        reason = (
            f"正式推薦{action}；依模型狀態動態配置 "
            f"{percentage:.1f}%（5%-30% 範圍）。"
        )
    else:
        ratio = 0.0
        percentage = 0.0
        amount = 0.0
        source = "no_bankroll_or_no_formal_bp_signal"
        if bankroll <= 0.0:
            level = "尚未設定"
            reason = "請先設定本次分析本金"
        else:
            level = "無正式莊閒方向"
            reason = "目前沒有可用的 B/P 正式推薦"

    result.update(
        {
            "bankroll": int(round(bankroll)),
            "final_bet_ratio": float(ratio),
            "bet_percentage": float(percentage),
            "suggested_bet_amount": float(amount),
            "bet_amount": float(amount),
            "bet_level_text": level,
            "bet_reason": reason,
            "kelly_percentage_applied": float(percentage),
            "bankroll_display_source": source,
        }
    )
    return result


def install_legacy_app_bankroll_adapter() -> bool:
    """Patch only the already-loaded legacy app display adapter."""
    app_module = sys.modules.get("app")
    if app_module is None or not hasattr(app_module, "_attach_bankroll_advice"):
        return False
    current = getattr(app_module, "_attach_bankroll_advice", None)
    if current is not apply_edge_bankroll_advice:
        setattr(app_module, "_attach_bankroll_advice", apply_edge_bankroll_advice)
    return True


__all__ = [
    "MIN_DISPLAY_RATIO",
    "MAX_DISPLAY_RATIO",
    "apply_edge_bankroll_advice",
    "install_legacy_app_bankroll_adapter",
]
