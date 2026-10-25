"""Runtime entrypoint for the BGS LINE/FastAPI application.

The runtime installs the BBB Frozen Direct 32D policy before loading the legacy
app, then keeps the formal predictor contract intact at the UI boundary:
- formal BBB action is B/P/Skip; Skip remains zero-stake at this boundary;
- B/P/T history builds the same 16-shoe + 16-road context as the web panel;
- exact card evidence stays diagnostic and does not alter the formal context;
- prediction never bootstraps or automatically updates A/b;
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
    # Prediction boundary only: a BBB Skip must not be converted into a B/P bet.
    final = dict(result.get("bbb_final57") or {})
    if final.get("active") and final.get("direction") == "Skip":
        result.update({
            "action": "Skip", "recommend": "Skip", "direction": "Skip", "next_round_direction": "Skip",
            "internal_action": "Skip", "internal_recommend": "Skip",
            "action_text": "觀望", "recommend_text": "觀望", "direction_text": "觀望", "next_round_direction_text": "觀望",
            "bankroll": int(bankroll) if bankroll.is_integer() else bankroll,
            "risk_gate_open": False, "signal_allowed": False, "bet_allowed": False,
            "mandatory_bet": False, "skip": True, "force_observe": True,
            "skip_reason": "bbb_physics_particle500_skip",
            "suggested_bet_amount": 0.0, "bet_amount": 0.0, "bet_percentage": 0.0,
            "final_bet_ratio": 0.0, "kelly_fraction": 0.0,
            "kelly_percentage_applied": 0.0, "bet_multiplier": 0.0,
        })
        money.update({"direction": "Skip", "bankroll": bankroll, "final_bet_ratio": 0.0,
                      "kelly_fraction": 0.0, "bet_percentage": 0.0, "bet_amount": 0.0,
                      "bet_allowed": False, "mandatory_bet": False})
        result["money_management"] = money
        return result
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


def _fixed_forecast_panel(user_id, session):
    """Display the fixed 2pp forecast without changing wager/session state."""
    prediction = dict(session.get("screen_last_prediction") or {})
    final = dict(prediction.get("bbb_final57") or {})
    forecast = final.get("next_hand_forecast") or {}
    direction = final.get("forecast_direction")
    if final.get("active") and forecast.get("available") and direction in ("B", "P", "Skip"):
        prediction.update({
            "internal_action": direction, "internal_recommend": direction,
            "action": direction, "recommend": direction, "skip": direction == "Skip",
            "banker_rate": float(forecast["directional"]["B"]) * 100.0,
            "player_rate": float(forecast["directional"]["P"]) * 100.0,
            "tie_rate": float(forecast["probabilities"]["T"]) * 100.0,
        })
    return _ORIGINAL_SCREEN_RESULT_PANEL(user_id, {**dict(session), "screen_last_prediction": prediction})


# app.py still contains legacy compatibility helpers. Override only these two
# runtime seams instead of modifying OCR/screenshot/LINE code in the large file.
legacy_app._exact_shoe_context = _formal_shoe_context
legacy_app._attach_bankroll_advice = _attach_formal_bankroll_advice
_ORIGINAL_SCREEN_RESULT_PANEL = legacy_app.screen_result_panel
legacy_app.screen_result_panel = _fixed_forecast_panel
app = legacy_app.app

__all__ = ["app"]
