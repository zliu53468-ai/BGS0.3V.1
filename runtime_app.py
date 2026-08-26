"""Runtime entrypoint that preserves the existing FastAPI app while installing
three-way Markov money-management advice.

This avoids rewriting the large LINE/OCR application module. Route functions in
app.py resolve `_attach_bankroll_advice` at runtime, so replacing that module global
updates every existing image/manual prediction path.
"""
from __future__ import annotations

from typing import Any, Mapping

import app as legacy_app


def _attach_threeway_bankroll_advice(
    prediction: Mapping[str, Any],
    session: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(prediction or {})
    bankroll = max(0.0, float(session.get("bankroll", 0.0) or 0.0))
    ratio = max(0.0, min(0.30, float(result.get("final_bet_ratio", 0.0) or 0.0)))
    bet_allowed = bool(result.get("risk_gate_open")) and ratio > 0.0
    amount = bankroll * ratio if bet_allowed else 0.0

    money = dict(result.get("money_management") or {})
    reason_code = str(money.get("reason") or "threeway_markov_risk_gate")
    if bankroll <= 0.0:
        level = "尚未設定"
        reason = "請先設定本次分析本金"
        amount = 0.0
    elif not bet_allowed:
        level = "不下注"
        reason = (
            "三元 Markov 仍提供下一局方向，但資訊熵／牌靴進度／Kelly "
            f"風控未通過（{reason_code}）。"
        )
    else:
        level = "三元 Markov／Kelly 風控"
        reason = (
            f"final_weight={float(result.get('confidence', 0.0) or 0.0):.3f}；"
            f"Kelly={float(result.get('kelly_fraction', 0.0) or 0.0):.4f}；"
            f"final_ratio={ratio:.4f}。"
        )
        if bool(result.get("tie_risk_active")):
            reason += " 預測和局機率偏高，下注比例已額外降低。"

    result.update({
        "bankroll": int(bankroll) if bankroll.is_integer() else bankroll,
        "suggested_bet_amount": float(amount),
        "bet_percentage": float(ratio * 100.0 if bet_allowed else 0.0),
        "bet_level_text": level,
        "bet_reason": reason,
        "screen_edge": round(float(result.get("direction_edge", 0.0) or 0.0), 6),
    })
    if isinstance(result.get("money_management"), Mapping):
        result["money_management"] = {
            **dict(result["money_management"]),
            "bankroll": bankroll,
            "bet_amount": float(amount),
        }
    return result


legacy_app._attach_bankroll_advice = _attach_threeway_bankroll_advice
app = legacy_app.app

__all__ = ["app"]
