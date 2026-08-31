"""Regression tests for the capped HSMM + run-hazard Road residual gate."""
from __future__ import annotations

import unittest

from road_pattern_core import forecast_road_pattern, fuse_with_regime_gate
from road_regime_gate import MAX_REGIME_DIRECTION_WEIGHT, analyze_road_regime_gate


def _swap_bp(history: str) -> str:
    return "".join("P" if ch == "B" else "B" if ch == "P" else ch for ch in history)


class RoadRegimeGateUnitTests(unittest.TestCase):
    def test_short_history_has_no_formal_regime_weight(self) -> None:
        gate = analyze_road_regime_gate(list("BPBPBP"), derived_analysis={})
        self.assertEqual(gate["reliability"], 0.0)
        self.assertFalse(gate["available"])

    def test_state_posterior_is_normalized_and_named(self) -> None:
        gate = analyze_road_regime_gate(
            list("BBPBBPBBPBBPBBPBBPBBP"),
            derived_analysis={"likelihood": {"B": 0.5, "P": 0.5}},
        )
        posterior = gate["state_posterior"]
        self.assertEqual(
            set(posterior),
            {"S0_PERSISTENT", "S1_ALTERNATING", "S2_TRANSITION", "S3_NOISE"},
        )
        self.assertAlmostEqual(sum(posterior.values()), 1.0, places=12)
        self.assertIn(gate["dominant_state"], posterior)

    def test_regime_reliability_is_capped(self) -> None:
        gate = analyze_road_regime_gate(
            list("BPBBPBPBBPPBPBBPBPBBPPBPBBPBP"),
            derived_analysis={
                "likelihood": {"B": 0.58, "P": 0.42},
                "reliability": 0.18,
                "active_road_count": 3,
                "cross_road_agreement": 0.8,
            },
        )
        self.assertLessEqual(gate["reliability"], MAX_REGIME_DIRECTION_WEIGHT)
        self.assertLessEqual(MAX_REGIME_DIRECTION_WEIGHT, 0.12)

    def test_hsmm_never_invents_direction_without_hazard_support(self) -> None:
        gate = analyze_road_regime_gate(list("BBBBBBBBBBBB"), derived_analysis={})
        self.assertEqual(gate["reliability"], 0.0)
        self.assertFalse(gate["available"])

    def test_strong_existing_road_probability_is_protected(self) -> None:
        result = fuse_with_regime_gate(
            0.58,
            {
                "available": True,
                "likelihood": {"B": 0.20, "P": 0.80},
                "reliability": 0.12,
            },
        )
        self.assertGreater(result["final_p_b"], 0.5)
        self.assertFalse(result["direction_override"])
        self.assertLessEqual(result["regime_effective_weight"], 0.12)

    def test_neutral_existing_probability_can_be_residually_corrected(self) -> None:
        result = fuse_with_regime_gate(
            0.502,
            {
                "available": True,
                "likelihood": {"B": 0.20, "P": 0.80},
                "reliability": 0.12,
            },
        )
        self.assertLess(result["final_p_b"], 0.5)
        self.assertTrue(result["direction_override"])


class RoadRegimeGateIntegrationTests(unittest.TestCase):
    def test_regime_gate_is_exposed_without_replacing_v1_components(self) -> None:
        result = forecast_road_pattern("BPBBPBPBBPPBPBBPBPBBPPBPBBPBP")
        self.assertIn("regime_gate", result)
        self.assertIn("regime_gate_fusion", result)
        self.assertIn("pre_regime_probabilities", result)
        self.assertLessEqual(result["regime_direction_weight"], 0.12)
        self.assertEqual(
            set(result["components"]),
            {"multi_window", "pattern_replay", "ngram", "pattern_survival"},
        )
        self.assertEqual(
            result["direction_authority"],
            "road_pattern_v1_plus_human_derived_ask_road",
        )

    def test_orientation_mirror_still_swaps_final_probability(self) -> None:
        history = "BPBBPBPBBPPBPBBPBPBBPPBPBBPBP"
        left = forecast_road_pattern(history)
        right = forecast_road_pattern(_swap_bp(history))
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

    def test_long_dragon_is_not_forced_to_turn(self) -> None:
        result = forecast_road_pattern("BBBBBBBBBBBB")
        self.assertEqual(result["direction"], "B")
        self.assertGreaterEqual(result["probabilities"]["B"], 0.5)
        self.assertEqual(result["regime_gate"]["reliability"], 0.0)


if __name__ == "__main__":
    unittest.main()
