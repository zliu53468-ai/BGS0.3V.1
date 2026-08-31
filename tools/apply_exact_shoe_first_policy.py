from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}: {old[:120]!r}")
    write(path, text.replace(old, new, 1))


# predictor.py: exact shoe must be evaluated before road fallback.
text = read("predictor.py")
if "    road_only_policy,\n" not in text:
    text = text.replace(
        "    linucb_policy,\n    normalize_big_road,\n",
        "    linucb_policy,\n    normalize_big_road,\n    road_only_policy,\n",
        1,
    )

start_marker = "    # Road path is always evaluated so diagnostics/online feedback/public interfaces\n"
end_marker = "\n    action = direction\n"
start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise RuntimeError("predictor.py: formal routing block markers not found")

new_block = '''    # Formal routing is intentionally shoe-first. Exact card composition is
    # validated before any road fallback is allowed to own the public B/P decision.
    # The road model may still run after an exact shoe decision for diagnostics only.
    shoe_analysis = dict(analyze_shoe_composition(context))
    shoe_available = bool(shoe_analysis.get("available"))
    composition_source = _shoe_source(context, shoe_analysis)

    if shoe_available:
        probabilities = dict(shoe_analysis.get("probabilities") or {})
        returns = dict(shoe_analysis.get("expected_returns") or {})
        b_ev = float(returns.get("B", 0.0) or 0.0)
        p_ev = float(returns.get("P", 0.0) or 0.0)
        # No EV gate and no third arm: even if both sides are negative, choose the
        # mathematically better B/P side. Low/negative edge is handled by Kelly floor.
        direction = "B" if b_ev >= p_ev else "P"
        confidence = _resolved_confidence(probabilities, direction)
        formal_source = "exact_shoe_ev"
        card_weight = 1.0
        road_weight = 0.0
        selected_physical_ev = b_ev if direction == "B" else p_ev
        shoe_analysis["action"] = direction
        shoe_analysis["action_text"] = "莊" if direction == "B" else "閒"
        shoe_analysis["formal_direction"] = direction
        shoe_analysis["formal_no_observe_arm"] = True

        # Compatibility/feedback diagnostics only. The caller's complete
        # shoe_context is forwarded and cannot change the already-selected shoe EV side.
        policy = linucb_policy(
            raw_history,
            shoe_context=context,
            user_id=user_id,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
        )
        road_probabilities = dict(policy["probabilities"])
        road_direction = str(policy["direction"])
        road_confidence = float(policy["selected_win_probability"])
    else:
        # Missing or invalid exact composition is non-fatal. Only here may the
        # road fallback own the formal B/P direction.
        policy = road_only_policy(
            raw_history,
            shoe_context=context,
            user_id=user_id,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
        )
        road_probabilities = dict(policy["probabilities"])
        road_direction = str(policy["direction"])
        road_confidence = float(policy["selected_win_probability"])
        probabilities = dict(road_probabilities)
        direction = road_direction if road_direction in {"B", "P"} else "B"
        confidence = road_confidence
        formal_source = "road_forecaster"
        card_weight = 0.0
        road_weight = 1.0
        selected_physical_ev = None
        shoe_analysis["action"] = None
        shoe_analysis["action_text"] = "牌靴資料不可用，正式方向回退牌路"
        shoe_analysis["formal_direction"] = direction
        shoe_analysis["formal_no_observe_arm"] = True
'''
text = text[:start] + new_block + text[end:]

for old, new in (
    (
        '            "road_context_direction_weight": road_weight,\n            "card_composition_source": composition_source,\n',
        '            "road_context_direction_weight": road_weight,\n            "road_direction_weight": road_weight,\n            "card_composition_source": composition_source,\n',
    ),
    (
        '    policy["road_context_direction_weight"] = road_weight\n    policy["card_composition_direction_weight"] = card_weight\n',
        '    policy["road_context_direction_weight"] = road_weight\n    policy["road_direction_weight"] = road_weight\n    policy["card_composition_direction_weight"] = card_weight\n',
    ),
    (
        '        "road_context_direction_weight": road_weight,\n        "card_composition_source": composition_source,\n',
        '        "road_context_direction_weight": road_weight,\n        "road_direction_weight": road_weight,\n        "card_composition_source": composition_source,\n',
    ),
):
    if text.count(old) != 1:
        raise RuntimeError(f"predictor.py: alias target count={text.count(old)} for {old!r}")
    text = text.replace(old, new, 1)
write("predictor.py", text)


# dynamic_prediction_policy.py: road_only_policy is fallback-only and forwards context.
text = read("dynamic_prediction_policy.py")
start = text.find("def road_only_policy(")
end = text.find("\ndef shoe_progress_policy", start)
if start < 0 or end < 0:
    raise RuntimeError("dynamic_prediction_policy.py: road_only_policy markers not found")
road_only_block = '''def road_only_policy(
    history: str | Iterable[Any] | None,
    *,
    shoe_context: Mapping[str, Any] | None = None,
    user_id: str = "",
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> dict[str, Any]:
    """無精確牌組時的 B/P fallback 相容入口。

    `shoe_context` 只作 depth/metadata 相容並完整往下傳，不會在此被清成 `{}`。
    本函式不做 shoe-composition EV，因此不能覆蓋 predictor 的 exact-shoe-first
    正式路徑。
    """
    return linucb_policy(
        history,
        shoe_context=shoe_context,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )

'''
text = text[:start] + road_only_block + text[end + 1:]
write("dynamic_prediction_policy.py", text)


# shoe_composition.py: clear stable API aliases for direction and EV.
replace_once(
    "shoe_composition.py",
    '        "available": False,\n        "action": None,\n',
    '        "available": False,\n        "direction": None,\n        "action": None,\n',
)
replace_once(
    "shoe_composition.py",
    '        "expected_returns": {\n            "B": None,\n            "P": None,\n            "T": None,\n        },\n',
    '        "probabilities": {"B": None, "P": None, "T": None},\n        "expected_returns": {\n            "B": None,\n            "P": None,\n            "T": None,\n        },\n        "ev": {"B": None, "P": None, "T": None},\n',
)
replace_once(
    "shoe_composition.py",
    '            "available": True,\n            "action": selected_side,\n',
    '            "available": True,\n            "direction": selected_side,\n            "action": selected_side,\n',
)
replace_once(
    "shoe_composition.py",
    '            "expected_returns": returns,\n            "banker_ev": float(returns["B"]),\n',
    '            "expected_returns": returns,\n            "ev": {\n                "B": float(returns["B"]),\n                "P": float(returns["P"]),\n                "T": None,\n            },\n            "banker_ev": float(returns["B"]),\n',
)


TEST_CONTENT = '''"""Formal routing regression: exact shoe first, road fallback second, B/P only."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from dynamic_prediction_policy import road_only_policy
from predictor import predict
from shoe_composition import analyze_shoe_composition


BANKER_BETTER_1D = [13, 2, 0, 3, 0, 4, 2, 1, 4, 1]
PLAYER_BETTER_1D = [16, 2, 1, 1, 4, 1, 0, 0, 0, 3]


def _road_policy(direction: str = "B") -> dict:
    p_b, p_p = (0.56, 0.44) if direction == "B" else (0.44, 0.56)
    return {
        "direction": direction,
        "selected_arm": direction,
        "probabilities": {"B": p_b, "P": p_p, "T": 0.0},
        "selected_win_probability": max(p_b, p_p),
        "confidence": max(p_b, p_p),
        "context_vector": [0.0] * 16,
        "context_feature_names": [f"road_{index}" for index in range(16)],
        "context_metadata": {
            "remaining_cards": 0.0,
            "remaining_cards_source": "round_count_estimate",
            "estimated_remaining_counts_0_to_9": [],
        },
        "scores": {"B": {"uncertainty": 0.0}, "P": {"uncertainty": 0.0}},
        "feedback_update": {},
        "ridge": 1.0,
        "road_forecaster": {"model_id": "test-road", "effective_support": 20.0},
        "effective_support": 20.0,
        "regression_analysis": {},
        "scope_key": "test-scope",
    }


class FormalShoeDecisionTests(unittest.TestCase):
    def _predict(self, shoe_context: dict):
        with patch("predictor.linucb_policy", return_value=_road_policy("B")) as diagnostic, patch(
            "predictor.road_only_policy", return_value=_road_policy("B")
        ) as fallback, patch("predictor.recent_user_direction_feedback", return_value={}):
            result = predict(
                history="BPBPBBPP",
                venue="DG",
                room="1",
                shoe_id="TEST",
                user_id="unit-test-user",
                shoe_context={"bankroll": 10000, **shoe_context},
            )
        return result, diagnostic, fallback

    def assert_bp_contract(self, result: dict) -> None:
        self.assertIn(result["recommend"], {"B", "P"})
        self.assertIn(result["action"], {"B", "P"})
        self.assertIn(result["next_round_direction"], {"B", "P"})
        self.assertEqual(result["recommend"], result["action"])
        self.assertEqual(result["action"], result["next_round_direction"])
        self.assertTrue(result["bet_allowed"])
        self.assertGreaterEqual(result["bet_percentage"], 5.0)
        self.assertLessEqual(result["bet_percentage"], 30.0)
        self.assertFalse(result["skip"])

    def test_no_composition_uses_road_fallback_and_forwards_context(self) -> None:
        result, diagnostic, fallback = self._predict({"remaining_cards": 300})
        self.assert_bp_contract(result)
        self.assertEqual(result["action"], "B")
        self.assertEqual(result["formal_direction_source"], "road_forecaster")
        self.assertFalse(result["shoe_context_used_for_formal_direction"])
        self.assertEqual(result["card_composition_direction_weight"], 0.0)
        self.assertEqual(result["road_direction_weight"], 1.0)
        self.assertEqual(result["road_context_direction_weight"], 1.0)
        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.kwargs["shoe_context"]["remaining_cards"], 300)
        self.assertEqual(fallback.call_args.kwargs["shoe_context"]["bankroll"], 10000)
        diagnostic.assert_not_called()

    def test_remaining_counts_own_formal_direction_before_road(self) -> None:
        banker, banker_diag, banker_fallback = self._predict(
            {"remaining_counts": BANKER_BETTER_1D, "decks": 1}
        )
        player, player_diag, player_fallback = self._predict(
            {"remaining_counts": PLAYER_BETTER_1D, "decks": 1}
        )
        self.assert_bp_contract(banker)
        self.assert_bp_contract(player)
        self.assertEqual(banker["action"], "B")
        self.assertEqual(player["action"], "P")
        self.assertEqual(player["formal_direction_source"], "exact_shoe_ev")
        self.assertTrue(player["shoe_context_used_for_formal_direction"])
        self.assertEqual(player["card_composition_direction_weight"], 1.0)
        self.assertEqual(player["road_direction_weight"], 0.0)
        self.assertEqual(player["road_context_direction_weight"], 0.0)
        self.assertEqual(player["card_composition_source"], "remaining_counts")
        banker_fallback.assert_not_called()
        player_fallback.assert_not_called()
        banker_diag.assert_called_once()
        player_diag.assert_called_once()
        self.assertEqual(
            player_diag.call_args.kwargs["shoe_context"]["remaining_counts"],
            PLAYER_BETTER_1D,
        )
        self.assertIsNotNone(player["banker_ev"])
        self.assertIsNotNone(player["player_ev"])
        self.assertEqual(
            player["dynamic_prediction_policy"]["forecast"]["road_direction_before_shoe_override"],
            "B",
        )

    def test_observed_cards_are_second_priority_exact_source(self) -> None:
        result, diagnostic, fallback = self._predict(
            {"observed_cards": ["A", 8, "K", 3], "decks": 8}
        )
        self.assert_bp_contract(result)
        self.assertEqual(result["formal_direction_source"], "exact_shoe_ev")
        self.assertTrue(result["shoe_context_used_for_formal_direction"])
        self.assertEqual(result["card_composition_source"], "observed_cards")
        self.assertEqual(result["remaining_cards_source"], "observed_cards")
        fallback.assert_not_called()
        diagnostic.assert_called_once()

    def test_remaining_counts_have_priority_over_observed_cards(self) -> None:
        result, _, fallback = self._predict(
            {
                "remaining_counts": PLAYER_BETTER_1D,
                "observed_cards": ["A", 8, "K", 3],
                "decks": 1,
            }
        )
        self.assertEqual(result["card_composition_source"], "remaining_counts")
        self.assertEqual(result["action"], "P")
        fallback.assert_not_called()

    def test_invalid_composition_falls_back_without_error(self) -> None:
        result, diagnostic, fallback = self._predict(
            {"remaining_counts": [999] * 10, "decks": 8}
        )
        self.assert_bp_contract(result)
        self.assertEqual(result["formal_direction_source"], "road_forecaster")
        self.assertFalse(result["shoe_context_used_for_formal_direction"])
        self.assertEqual(
            result["shoe_composition"]["reason_code"],
            "INVALID_CARD_COMPOSITION",
        )
        fallback.assert_called_once()
        diagnostic.assert_not_called()

    def test_shoe_composition_has_clear_direction_and_ev_api(self) -> None:
        exact = analyze_shoe_composition(
            {"remaining_counts": BANKER_BETTER_1D, "decks": 1}
        )
        self.assertTrue(exact["available"])
        self.assertEqual(exact["direction"], exact["action"])
        self.assertIn(exact["direction"], {"B", "P"})
        self.assertEqual(set(exact["probabilities"]), {"B", "P", "T"})
        self.assertAlmostEqual(exact["ev"]["B"], exact["banker_ev"], places=12)
        self.assertAlmostEqual(exact["ev"]["P"], exact["player_ev"], places=12)

    def test_road_only_policy_does_not_clear_shoe_context(self) -> None:
        context = {
            "remaining_cards": 250,
            "remaining_cards_source": "round_count_estimate",
        }
        with patch(
            "dynamic_prediction_policy.predict_bandit",
            return_value=_road_policy("B"),
        ) as mocked:
            result = road_only_policy(
                "BPBP",
                shoe_context=context,
                user_id="U",
                venue="DG",
                room="1",
                shoe_id="S",
            )
        self.assertIn(result["direction"], {"B", "P"})
        self.assertEqual(mocked.call_args.kwargs["shoe_context"], context)


if __name__ == "__main__":
    unittest.main()
'''

(ROOT / "test_formal_shoe_decision.py").write_text(TEST_CONTENT, encoding="utf-8")
print("Exact-shoe-first formal policy patch applied.")
