"""Compatibility bridge between current Edge sizing and the legacy LINE panel.

The current MoneyManagementModel already computes ``final_bet_ratio`` and
``bet_amount`` from positive Edge.  Older app.py code still expects a historical
``trusted_exact_counts``/Kelly gate and otherwise overwrites the displayed amount
with zero.  This bridge is installed at prediction runtime, after app.py has
finished importing, so the LINE panel consumes the current Edge-based sizing
without rewriting unrelated LINE/OCR code.
"""
from __future__ import annotations

from typing import Any, Mapping
import sys


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def apply_edge_bankroll_advice(
    prediction: Mapping[str, Any],
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose MoneyManagementModel sizing as a user-visible currency amount.

    Display rule:
      amount = bankroll * final_bet_ratio, only when bet_allowed and B/P.

    A non-positive Edge remains a strict no-bet and therefore displays 0.
    This function does not alter the sizing formula itself.
    """
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
    bet_allowed = bool(
        money.get(
            "bet_allowed",
            result.get("bet_allowed", result.get("risk_gate_open", False)),
        )
    ) and action in {"B", "P"}

    try:
        ratio = _clip(
            float(
                money.get(
                    "final_bet_ratio",
                    result.get("final_bet_ratio", 0.0),
                )
                or 0.0
            )
        )
    except (TypeError, ValueError):
        ratio = 0.0

    try:
        percentage = max(
            0.0,
            float(
                money.get(
                    "bet_percentage",
                    result.get("bet_percentage", ratio * 100.0),
                )
                or 0.0
            ),
        )
    except (TypeError, ValueError):
        percentage = ratio * 100.0

    if ratio <= 0.0 and percentage > 0.0:
        ratio = _clip(percentage / 100.0)
    if percentage <= 0.0 and ratio > 0.0:
        percentage = ratio * 100.0

    try:
        edge = float(
            money.get("edge", result.get("direction_edge", 0.0)) or 0.0
        )
    except (TypeError, ValueError):
        edge = 0.0

    sizing_reason = str(money.get("reason") or "")
    if bankroll > 0.0 and bet_allowed and ratio > 0.0 and edge > 0.0:
        amount = bankroll * ratio
        level = {
            "positive_edge_minimum_5pct": "正 Edge｜最低 5% 配置",
            "positive_edge_cap_30pct": "正 Edge｜上限 30% 配置",
            "edge_volatility_dynamic": "正 Edge｜動態配置",
        }.get(sizing_reason, "正 Edge｜動態配置")
        reason = str(
            result.get("signal_reason")
            or money.get("reason")
            or "正 Edge 已通過資金配置閘門"
        )
    else:
        amount = 0.0
        ratio = 0.0
        percentage = 0.0
        bet_allowed = False
        if bankroll <= 0.0:
            level = "尚未設定"
            reason = "請先設定本次分析本金"
        else:
            level = "Edge ≤ 0｜不下注"
            reason = str(
                result.get("signal_reason")
                or "目前 Edge 未通過資金配置閘門"
            )

    result.update(
        {
            "bankroll": int(round(bankroll)),
            "bet_allowed": bool(bet_allowed),
            "risk_gate_open": bool(bet_allowed),
            "final_bet_ratio": float(ratio),
            "bet_percentage": float(percentage),
            "suggested_bet_amount": float(amount),
            "bet_amount": float(amount),
            "bet_level_text": level,
            "bet_reason": reason,
            "screen_edge": round(float(edge), 6),
            "kelly_percentage_applied": float(percentage),
            "bankroll_display_source": "edge_based_money_management",
        }
    )
    return result


def install_legacy_app_bankroll_adapter() -> bool:
    """Patch only the already-loaded legacy app display adapter.

    Called during prediction runtime, so ``app`` has normally completed module
    initialization.  Independent model usage (without the LINE app imported) is
    unaffected.
    """
    app_module = sys.modules.get("app")
    if app_module is None or not hasattr(app_module, "_attach_bankroll_advice"):
        return False
    current = getattr(app_module, "_attach_bankroll_advice", None)
    if current is not apply_edge_bankroll_advice:
        setattr(app_module, "_attach_bankroll_advice", apply_edge_bankroll_advice)
    return True


__all__ = [
    "apply_edge_bankroll_advice",
    "install_legacy_app_bankroll_adapter",
]
