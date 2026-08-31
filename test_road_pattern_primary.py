"""Regression tests for V1 Big-Road + human derived-road final candidate."""
from __future__ import annotations

import unittest

from predictor import predict
from road_pattern_core import (
    COMPONENT_WEIGHTS,
    V1_VERSION,
    forecast_road_pattern,
    fuse_v1_with_derived,
    normalize_bp,
)
from road_pattern_v1_core import forecast_road_pattern as forecast_v1_only


def _swap_bp(history: str) -> str:
    return "".join("P" if ch == "B" else "B" if ch == "P" else ch for ch in history)


class RoadPatternCoreTests(unittest.TestCase):
    def test_ties_are_skipped_from_formal_sequence(self) -> None:
        self.assertEqual(normalize_bp("BTPPTBT"), ["B", "P", "P", "B"])

    def test_v1_core_is_restored_exactly_before_derived_fusion(self) -> None:
        history = "BPBPBBPPBPBBPBPBBPPBP"
        base = forecast_v1_only(history)
        wrapped = forecast_road_pattern(history)
        self.assertEqual(wrapped["v1_version"], V1_VERSION)
        self.assertEqual(wrapped["v1_model_id"], "ROAD-PATTERN-PRIMARY-V1")
        self.assertEqual(wrapped["v1_probabilities"], base["probabilities"])
        self.assertEqual(
            set(wrapped["components"]),
            {"multi_window", "pattern_replay", "ngram", "pattern_survival"},
        )
        self.assertEqual(
            COMPONENT_WEIGHTS,
            {
                "multi_window": 0.30,
                "pattern_replay": 0.30,
                "ngram": 0.25,
                "pattern_survival": 0.15,
            },
        )
        self.assertNotIn("probabilistic_turning", wrapped["components"])
        self.assertNotIn("change_point_probability", wrapped)

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

    def test_derived_signal_is_capped_auxiliary(self) -> None:
        result = forecast_road_pattern("BPBPBBPPBPBBPBPBBPPBPBPBBP")
        fusion = result["derived_ask_road_fusion"]
        self.assertLessEqual(fusion["derived_reliability"], 0.18)
        self.assertLessEqual(fusion["derived_effective_weight"], 0.18)
        self.assertEqual(result["derived_direction_authority"], "capped_auxiliary_only")
        self.assertEqual(
            result["direction_authority"],
            "road_pattern_v1_plus_human_derived_ask_road",
        )

    def test_strong_v1_cannot_be_overpowered_by_opposite_derived_signal(self) -> None:
        fusion = fuse_v1_with_derived(
            0.58,
            {
                "likelihood": {"B": 0.25, "P": 0.75},
                "reliability": 0.18,
                "active_road_count": 3,
            },
        )
        self.assertGreater(fusion["final_p_b"], 0.5)
        self.assertFalse(fusion["direction_override"])

    def test_weak_v1_can_be_corrected_by_mature_opposite_derived_signal(self) -> None:
        fusion = fuse_v1_with_derived(
            0.505,
            {
                "likelihood": {"B": 0.25, "P": 0.75},
                "reliability": 0.18,
                "active_road_count": 3,
            },
        )
        self.assertLess(fusion["final_p_b"], 0.5)
        self.assertTrue(fusion["direction_override"])

    def test_single_jump_v1_pattern_remains_available(self) -> None:
        result = forecast_road_pattern("BPBPBPBP")
        survival = result["components"]["pattern_survival"]
        self.assertEqual(survival["pattern"], "SINGLE_JUMP")
        self.assertEqual(survival["desired_relation"], "SWITCH")

    def test_long_run_can_remain_same_side(self) -> None:
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

    def test_predictor_uses_v1_plus_derived_road_core(self) -> None:
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
        road_pattern = result["road_pattern_model"]
        self.assertEqual(road_pattern["v1_model_id"], "ROAD-PATTERN-PRIMARY-V1")
        self.assertIn("derived_ask_road", road_pattern)
        self.assertNotIn("probabilistic_turning", road_pattern["components"])

    def test_exact_shoe_composition_cannot_change_direction(self) -> None:
        history = "BPBPBBPPBPBPBBPPBPBP"
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
