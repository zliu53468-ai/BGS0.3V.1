"""正式決策接線回歸：精確牌靴優先，缺資料回退 road，且永遠 B/P。"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from predictor import predict


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
            "remaining_cards_source": "estimated",
            "estimated_remaining_counts_0_to_9": [],
        },
        "scores": {
            "B": {"uncertainty": 0.0},
            "P": {"uncertainty": 0.0},
        },
        "feedback_update": {},
        "ridge": 1.0,
        "road_forecaster": {
            "model_id": "test-road",
            "effective_support": 20.0,
        },
        "effective_support": 20.0,
        "regression_analysis": {},
        "scope_key": "test-scope",
    }


class FormalShoeDecisionTests(unittest.TestCase):
    def _predict(self, shoe_context: dict) -> dict:
        with patch("predictor.linucb_policy", return_value=_road_policy("B")), patch(
            "predictor.recent_user_direction_feedback", return_value={}
        ):
            return predict(
                history="BPBPBBPP",
                venue="DG",
                room="1",
                shoe_id="TEST",
                user_id="unit-test-user",
                shoe_context={"bankroll": 10000, **shoe_context},
            )

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

    def test_no_composition_falls_back_to_road_without_observe(self) -> None:
        result = self._predict({})
        self.assert_bp_contract(result)
        self.assertEqual(result["action"], "B")
        self.assertFalse(result["shoe_context_used_for_formal_direction"])
        self.assertEqual(result["card_composition_direction_weight"], 0.0)
        self.assertEqual(result["road_context_direction_weight"], 1.0)
        self.assertEqual(result["card_composition_source"], "none")
        self.assertEqual(
            result["direction_source"], "road_forecaster_probability_argmax"
        )

    def test_remaining_counts_override_same_road_with_ev_argmax(self) -> None:
        banker = self._predict(
            {"remaining_counts": BANKER_BETTER_1D, "decks": 1}
        )
        player = self._predict(
            {"remaining_counts": PLAYER_BETTER_1D, "decks": 1}
        )
        self.assert_bp_contract(banker)
        self.assert_bp_contract(player)
        self.assertEqual(banker["action"], "B")
        self.assertEqual(player["action"], "P")
        self.assertTrue(player["shoe_context_used_for_formal_direction"])
        self.assertEqual(player["card_composition_direction_weight"], 1.0)
        self.assertEqual(player["road_context_direction_weight"], 0.0)
        self.assertEqual(player["card_composition_source"], "remaining_counts")
        self.assertEqual(player["direction_source"], "shoe_composition_ev_argmax")
        self.assertIsNotNone(player["banker_ev"])
        self.assertIsNotNone(player["player_ev"])
        # The mocked road always says B, so P proves the exact shoe owns formal direction.
        self.assertEqual(
            player["dynamic_prediction_policy"]["forecast"][
                "road_direction_before_shoe_override"
            ],
            "B",
        )

    def test_observed_cards_are_second_priority_source(self) -> None:
        result = self._predict(
            {"observed_cards": ["A", 8, "K", 3], "decks": 8}
        )
        self.assert_bp_contract(result)
        self.assertTrue(result["shoe_context_used_for_formal_direction"])
        self.assertEqual(result["card_composition_source"], "observed_cards")
        self.assertEqual(result["remaining_counts_source"], "observed_cards")

    def test_invalid_composition_never_breaks_main_flow(self) -> None:
        result = self._predict(
            {"remaining_counts": [999] * 10, "decks": 8}
        )
        self.assert_bp_contract(result)
        self.assertEqual(result["action"], "B")
        self.assertFalse(result["shoe_context_used_for_formal_direction"])
        self.assertEqual(result["card_composition_source"], "none")
        self.assertEqual(
            result["shoe_composition"]["reason_code"],
            "INVALID_CARD_COMPOSITION",
        )


if __name__ == "__main__":
    unittest.main()
