"""Runtime entrypoint for the BGS LINE/FastAPI application.

The runtime installs the dynamic policy before loading the legacy app, then
keeps the formal predictor contract intact at the UI boundary:
- formal action is always B/P;
- shoe evidence priority is remaining_counts > observed_cards > none;
- exact shoe EV owns direction when exact composition is available;
- road forecaster is the fallback when it is not;
- the core fractional-Kelly ratio is preserved inside the 5%..30% product band.

OCR, screenshot recognition, LINE transport and UI layout remain unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping

from dynamic_prediction_policy import install_dynamic_prediction_policy
from shoe_constants import SHOE_DECKS

# 必須在 app 載入前安裝，讓 predictor 使用同一套政策。
install_dynamic_prediction_policy()

import app as legacy_app


MIN_FORMAL_BET_RATIO = 0.05
MAX_FORMAL_BET_RATIO = 0.30


def _formal_bp_action(result: Mapping[str, Any]) -> str:
    action = str(
        result.get("action")
        or result.get("recommend")
        or result.get("direction")
        or ""
    ).upper().strip()
    if action in {"B", "P"}:
        return action
    try:
        banker = float(result.get("banker_rate", 0.0) or 0.0)
        player = float(result.get("player_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        banker = player = 0.0
    return "B" if banker >= player else "P"


def _formal_shoe_context(session: Mapping[str, Any]) -> dict[str, Any]:
    """Read verified shoe evidence from Session using the formal priority order."""
    context: dict[str, Any] = {
        "bankroll": max(0.0, float(session.get("bankroll", 0.0) or 0.0)),
    }

    # Priority 1: explicitly supplied remaining point counts (0..9).
    counts = session.get("exact_remaining_counts")
    if isinstance(counts, list) and len(counts) == 10:
        context.update(
            {
                "remaining_counts": list(counts),
                "decks": int(session.get("exact_remaining_decks", SHOE_DECKS) or SHOE_DECKS),
                "source": "remaining_counts",
            }
        )
        return context

    # Priority 2: actual observed card values saved by /api/shoe/cards.
    observed = session.get("observed_cards")
    if isinstance(observed, list) and observed:
        context.update(
            {
                "observed_cards": list(observed),
                "decks": int(session.get("exact_remaining_decks", SHOE_DECKS) or SHOE_DECKS),
                "source": "observed_cards",
            }
        )
    return context


def _attach_formal_bankroll_advice(
    prediction: Mapping[str, Any],
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the formal B/P + 5%..30% Kelly contract at the runtime UI edge."""
    result = dict(prediction or {})
    bankroll = max(0.0, float(session.get("bankroll", 0.0) or 0.0))
    money = dict(result.get("money_management") or {})
    action = _formal_bp_action(result)

    try:
        core_ratio = float(
            result.get(
                "final_bet_ratio",
                money.get("final_bet_ratio", MIN_FORMAL_BET_RATIO),
            )
            or MIN_FORMAL_BET_RATIO
        )
    except (TypeError, ValueError):
        core_ratio = MIN_FORMAL_BET_RATIO
    ratio = max(MIN_FORMAL_BET_RATIO, min(MAX_FORMAL_BET_RATIO, core_ratio))
    amount = bankroll * ratio
    text = "莊" if action == "B" else "閒"

    result.update(
        {
            "action": action,
            "recommend": action,
            "next_round_direction": action,
            "direction": action,
            "action_text": text,
            "recommend_text": text,
            "next_round_direction_text": text,
            "direction_text": text,
            "bankroll": int(bankroll) if bankroll.is_integer() else bankroll,
            "risk_gate_open": True,
            "signal_allowed": True,
            "bet_allowed": True,
            "mandatory_bet": True,
            "skip": False,
            "skip_reason": "",
            "suggested_bet_amount": float(amount),
            "bet_amount": float(amount),
            "bet_percentage": float(ratio * 100.0),
            "final_bet_ratio": float(ratio),
            "bet_level_text": (
                "精確牌靴 EV／Fractional Kelly"
                if bool(result.get("shoe_context_used_for_formal_direction"))
                else "牌路方向／Fractional Kelly"
            ),
            "bet_reason": str(
                result.get("signal_reason")
                or money.get("reason")
                or "B/P two-arm fractional Kelly"
            ),
            "screen_edge": round(
                float(result.get("direction_edge", 0.0) or 0.0),
                6,
            ),
        }
    )

    if isinstance(result.get("money_management"), Mapping):
        result["money_management"] = {
            **dict(result["money_management"]),
            "direction": action,
            "bankroll": bankroll,
            "final_bet_ratio": float(ratio),
            "bet_percentage": float(ratio * 100.0),
            "bet_amount": float(amount),
            "bet_allowed": True,
            "mandatory_bet": True,
        }
    return result


# app.py still contains legacy compatibility helpers. Override only these two
# runtime seams instead of modifying OCR/screenshot/LINE code in the large file.
legacy_app._exact_shoe_context = _formal_shoe_context
legacy_app._attach_bankroll_advice = _attach_formal_bankroll_advice
app = legacy_app.app

__all__ = ["app"]
