"""Regression tests for the Road-Primary production core."""
from __future__ import annotations

import unittest

from predictor import predict
from road_pattern_core import forecast_road_pattern, normalize_bp


def _swap_bp(history: str) -> str:
    return "".join("P" if ch == "B" else "B" if ch == "P" else ch for ch in history)


class RoadPatternCoreTests(unittest.TestCase):
    def test_ties_are_skipped_from_formal_sequence(self) -> None:
        self.assertEqual(normalize_bp("BTPPTBT"), ["B", "P", "P", "B"])

    def test_orientation_mirror_swaps_probability(self) -> None:
        history = "BPBPBBPPBPBBPBPBBPPBP"
        mirrored = _swap_bp(history)
        left = forecast_road_pattern(history)
        right = forecast_road_pattern(mirrored)
        self.assertAlmostEqual(
            float(left["probabilities"]["B"]),
            float(right["probabilities"]["P"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(left["probabilities"]["P"]),
            float(right["probabilities"]["B"]),
            places=12,
        )

    def test_formal_components_exist_and_sum_to_bp(self) -> None:
        result = forecast_road_pattern("BPBPBPBBPPBPBPBBPPBP")
        self.assertIn(result["direction"], {"B", "P"})
        self.assertAlmostEqual(
            result["probabilities"]["B"] + result["probabilities"]["P"],
            1.0,
            places=12,
        )
        self.assertEqual(
            set(result["components"]),
            {"multi_window", "pattern_replay", "ngram", "pattern_survival"},
        )
        self.assertEqual(result["direction_authority"], "road_pattern_core_only")
        self.assertEqual(result["shoe_direction_weight"], 0.0)
        self.assertEqual(result["lstm_direction_weight"], 0.0)

    def test_single_jump_is_detected_as_pattern_not_forced_global_rule(self) -> None:
        result = forecast_road_pattern("BPBPBPBP")
        survival = result["components"]["pattern_survival"]
        self.assertEqual(survival["pattern"], "SINGLE_JUMP")
        self.assertEqual(survival["desired_relation"], "SWITCH")
        self.assertGreater(survival["reliability"], 0.0)

    def test_long_run_can_remain_same_side_when_evidence_supports_it(self) -> None:
        result = forecast_road_pattern("BBBBBBBB")
        self.assertEqual(result["direction"], "B")
        self.assertGreaterEqual(result["probabilities"]["B"], 0.5)
        self.assertNotIn("forced", result["semantics"].lower())


class PredictorRoadPrimaryContractTests(unittest.TestCase):
    def assert_bp_contract(self, result: dict) -> None:
        self.assertIn(result["recommend"], {"B", "P"})
        self.assertEqual(result["recommend"], result["action"])
        self.assertEqual(result["action"], result["next_round_direction"])
        self.assertTrue(result["bet_allowed"])
        self.assertFalse(result["skip"])
        self.assertGreaterEqual(result["bet_percentage"], 5.0)
        self.assertLessEqual(result["bet_percentage"], 30.0)

    def test_predictor_uses_road_pattern_as_only_formal_direction(self) -> None:
        result = predict(
            history="BPBPBBPPBPBPBBPPBPBP",
            venue="DG",
            room="1",
            shoe_id="ROAD-TEST",
            user_id="test-user",
            shoe_context={"bankroll": 10000, "cut_card_remaining_cards": 70},
        )
        self.assert_bp_contract(result)
        self.assertEqual(result["engine"], "ROAD_PATTERN_PRIMARY_BP")
        self.assertEqual(result["formal_direction_source"], "road_pattern_core")
        self.assertEqual(result["road_direction_weight"], 1.0)
        self.assertEqual(result["card_composition_direction_weight"], 0.0)
        self.assertEqual(result["lstm_direction_weight"], 0.0)
        self.assertEqual(result["fallback_markov_direction_weight"], 0.0)
        self.assertFalse(result["shoe_context_used_for_formal_direction"])
        self.assertFalse(result["lstm_enabled"])
        self.assertTrue(result["dynamic_prediction_policy"]["road_primary"])

    def test_exact_shoe_composition_cannot_change_direction(self) -> None:
        history = "BPBPBBPPBPBPBBPPBPBP"
        # Both are valid 8-deck-ish value-count vectors but intentionally very different.
        context_a = {
            "bankroll": 10000,
            "decks": 8,
            "remaining_counts": [120, 20, 20, 20, 20, 20, 20, 20, 20, 20],
            "cut_card_remaining_cards": 70,
        }
        context_b = {
            "bankroll": 10000,
            "decks": 8,
            "remaining_counts": [32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
            "cut_card_remaining_cards": 70,
        }
        a = predict(history=history, shoe_id="SAME", shoe_context=context_a)
        b = predict(history=history, shoe_id="SAME", shoe_context=context_b)
        self.assertEqual(a["action"], b["action"])
        self.assertEqual(a["raw_direction_probabilities"], b["raw_direction_probabilities"])
        self.assertEqual(a["card_composition_direction_weight"], 0.0)
        self.assertEqual(b["card_composition_direction_weight"], 0.0)
        self.assertFalse(a["shoe_composition"]["formal_direction_authority"])
        self.assertFalse(b["shoe_composition"]["formal_direction_authority"])

    def test_cut_setting_never_flips_raw_road_direction(self) -> None:
        history = "BPBPBBPPBPBPBBPPBPBP"
        early_cut = predict(
            history=history,
            shoe_context={"bankroll": 10000, "cut_card_remaining_cards": 40},
        )
        deep_cut = predict(
            history=history,
            shoe_context={"bankroll": 10000, "cut_card_remaining_cards": 120},
        )
        self.assertEqual(early_cut["action"], deep_cut["action"])
        self.assertEqual(
            early_cut["raw_direction_probabilities"],
            deep_cut["raw_direction_probabilities"],
        )
        self.assertFalse(early_cut["confidence_calibration"]["direction_override"])
        self.assertFalse(deep_cut["confidence_calibration"]["direction_override"])

    def test_empty_history_still_returns_bp(self) -> None:
        result = predict(history="", shoe_context={"bankroll": 1000})
        self.assert_bp_contract(result)
        self.assertEqual(result["formal_direction_source"], "road_pattern_core")


if __name__ == "__main__":
    unittest.main()
