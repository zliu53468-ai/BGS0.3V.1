"""Runtime entrypoint for the BGS LINE/FastAPI application.

The runtime installs the dynamic prediction policy before loading the legacy app.
The prediction core is road-only and the bankroll display bridge now preserves
its exact quarter-Kelly ratio instead of imposing the previous 5% minimum.

OCR, screenshot upload, LINE transport and UI layout remain unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping

from dynamic_prediction_policy import install_dynamic_prediction_policy

# 必須在 app 載入前安裝，讓 predictor 使用同一套政策。
install_dynamic_prediction_policy()

import app as legacy_app


def _attach_threeway_bankroll_advice(
    prediction: Mapping[str, Any],
    session: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(prediction or {})
    bankroll = max(0.0, float(session.get("bankroll", 0.0) or 0.0))
    money = dict(result.get("money_management") or {})

    action = str(
        result.get("action")
        or result.get("recommend")
        or result.get("direction")
        or ""
    ).upper().strip()
    core_bet_allowed = bool(
        money.get(
            "bet_allowed",
            result.get("bet_allowed", result.get("signal_allowed", False)),
        )
    )
    core_ratio = max(
        0.0,
        min(
            0.30,
            float(
                result.get(
                    "final_bet_ratio",
                    money.get("final_bet_ratio", 0.0),
                )
                or 0.0
            ),
        ),
    )
    skip_active = action in {"SKIP", "O"} or not core_bet_allowed or core_ratio <= 0.0

    if skip_active:
        ratio = 0.0
        amount = 0.0
        level = "觀望 / SKIP"
        reason = str(
            result.get("signal_status_text")
            or result.get("skip_reason")
            or money.get("reason")
            or "目前未通過牌路模型／虛擬 EV 決策閘門。"
        )
        risk_gate_open = False
    else:
        # Preserve the exact ratio produced by MoneyManagementModel.  Do not
        # re-introduce the legacy mandatory 5% floor here.
        ratio = core_ratio
        amount = bankroll * ratio
        risk_gate_open = True
        if bankroll <= 0.0:
            level = "尚未設定"
            reason = "請先設定本次分析本金；方向訊號已通過牌路模型與虛擬 EV 閘門。"
        else:
            level = "+EV 通過／1/4 Kelly"
            reason = (
                f"confidence={float(result.get('confidence', 0.0) or 0.0):.3f}；"
                f"Kelly={float(result.get('kelly_fraction', 0.0) or 0.0):.4f}；"
                f"final_ratio={ratio:.4f}；"
                f"mode={str(money.get('reason') or 'positive_virtual_ev_quarter_kelly')}。"
            )

    result.update({
        "bankroll": int(bankroll) if bankroll.is_integer() else bankroll,
        "risk_gate_open": bool(risk_gate_open),
        "mandatory_bet": False,
        "suggested_bet_amount": float(amount),
        "bet_amount": float(amount),
        "bet_percentage": float(ratio * 100.0),
        "final_bet_ratio": float(ratio),
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
            "bet_allowed": bool(risk_gate_open),
            "mandatory_bet": False,
        }
    return result


legacy_app._attach_bankroll_advice = _attach_threeway_bankroll_advice
app = legacy_app.app

__all__ = ["app"]
