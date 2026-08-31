"""Compatibility regression tests for the formal Road-Primary decision path."""
from __future__ import annotations

import unittest

from dynamic_prediction_policy import lstm_primary_policy, road_only_policy
from predictor import predict


class FormalRoadDecisionCompatibilityTests(unittest.TestCase):
    def test_road_only_policy_is_formal(self) -> None:
        result = road_only_policy(
            "BPBPBBPPBPBPBBPP",
            shoe_context={"remaining_cards": 180, "cut_card_remaining_cards": 70},
            user_id="U",
            venue="DG",
            room="1",
            shoe_id="S",
        )
        self.assertIn(result["direction"], {"B", "P"})
        self.assertEqual(result["formal_direction_source"], "road_pattern_core")
        self.assertTrue(result["road_primary"])
        self.assertEqual(result["formal_direction_weight"], 1.0)
        self.assertEqual(result["lstm_direction_weight"], 0.0)
        self.assertEqual(result["card_composition_direction_weight"], 0.0)

    def test_legacy_lstm_policy_name_routes_to_road_primary(self) -> None:
        road = road_only_policy("BPBPBBPPBPBPBBPP", shoe_id="S")
        legacy = lstm_primary_policy("BPBPBBPPBPBPBBPP", shoe_id="S")
        self.assertEqual(road["direction"], legacy["direction"])
        self.assertEqual(road["probabilities"], legacy["probabilities"])
        self.assertEqual(legacy["formal_direction_source"], "road_pattern_core")
        self.assertFalse(legacy["lstm_primary"])

    def test_predict_public_fields_remain_compatible(self) -> None:
        result = predict(
            history="BPBPBBPPBPBPBBPP",
            venue="DG",
            room="1",
            shoe_id="S",
            user_id="U",
            shoe_context={"bankroll": 10000, "cut_card_remaining_cards": 70},
        )
        for field in (
            "recommend",
            "action",
            "next_round_direction",
            "confidence",
            "bet_amount",
            "bet_percentage",
            "bet_allowed",
            "money_management",
            "markov",
            "road_predict",
            "shoe_composition",
        ):
            self.assertIn(field, result)
        self.assertIn(result["recommend"], {"B", "P"})
        self.assertEqual(result["recommend"], result["action"])
        self.assertEqual(result["action"], result["next_round_direction"])
        self.assertTrue(result["bet_allowed"])
        self.assertGreaterEqual(result["bet_percentage"], 5.0)
        self.assertLessEqual(result["bet_percentage"], 30.0)
        self.assertEqual(result["formal_direction_source"], "road_pattern_core")
        self.assertFalse(result["lstm_enabled"])

    def test_shoe_composition_is_diagnostic_only(self) -> None:
        result = predict(
            history="BPBPBBPPBPBPBBPP",
            shoe_context={
                "bankroll": 10000,
                "decks": 8,
                "remaining_counts": [100, 24, 24, 24, 24, 24, 24, 24, 24, 24],
            },
        )
        self.assertEqual(result["card_composition_direction_weight"], 0.0)
        self.assertFalse(result["shoe_context_used_for_formal_direction"])
        self.assertFalse(result["shoe_composition"]["formal_direction_authority"])


if __name__ == "__main__":
    unittest.main()
