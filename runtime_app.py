"""Runtime entrypoint that preserves the existing FastAPI app while installing
mandatory 5%-30% three-way Markov bankroll sizing.

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
    ratio = max(
        0.05,
        min(0.30, float(result.get("final_bet_ratio", 0.05) or 0.05)),
    )
    amount = bankroll * ratio

    money = dict(result.get("money_management") or {})
    reason_code = str(money.get("reason") or "mandatory_5_30_sizing")

    if bankroll <= 0.0:
        level = "尚未設定"
        reason = "請先設定本次分析本金；比例仍固定輸出於 5%-30% 區間。"
    else:
        level = "每局必下／5%-30% 動態"
        reason = (
            f"final_weight={float(result.get('confidence', 0.0) or 0.0):.3f}；"
            f"Kelly={float(result.get('kelly_fraction', 0.0) or 0.0):.4f}；"
            f"final_ratio={ratio:.4f}；"
            f"mode={reason_code}。"
        )
        if bool(result.get("tie_risk_active")):
            reason += " 和局風險偏高已反映在三元機率與資訊熵，但不再觸發觀望。"

    result.update({
        "bankroll": int(bankroll) if bankroll.is_integer() else bankroll,
        "risk_gate_open": True,
        "mandatory_bet": True,
        "suggested_bet_amount": float(amount),
        "bet_percentage": float(ratio * 100.0),
        "bet_level_text": level,
        "bet_reason": reason,
        "screen_edge": round(float(result.get("direction_edge", 0.0) or 0.0), 6),
    })
    if isinstance(result.get("money_management"), Mapping):
        result["money_management"] = {
            **dict(result["money_management"]),
            "bankroll": bankroll,
            "final_bet_ratio": float(ratio),
            "bet_percentage": float(ratio * 100.0),
            "bet_amount": float(amount),
            "bet_allowed": True,
            "mandatory_bet": True,
        }
    return result


legacy_app._attach_bankroll_advice = _attach_threeway_bankroll_advice
app = legacy_app.app

__all__ = ["app"]
