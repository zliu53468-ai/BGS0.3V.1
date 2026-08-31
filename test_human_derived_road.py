"""Focused tests for the human-style derived-road Ask-Road model."""
from __future__ import annotations

import unittest

from derived_road_markov import (
    MAX_DERIVED_ROAD_RELIABILITY,
    predict_next_derived_mark,
    score_ask_road_scenarios,
)
from road_model import build_standard_derived_roads


class HumanDerivedRoadPatternTests(unittest.TestCase):
    def test_single_jump_pattern(self) -> None:
        result = predict_next_derived_mark(list("RURURURU"))
        self.assertEqual(result["pattern"], "SINGLE_JUMP")
        self.assertEqual(
            result["components"]["run_rhythm"]["desired_relation"],
            "SWITCH",
        )

    def test_double_jump_pattern(self) -> None:
        result = predict_next_derived_mark(list("RRUURRUU"))
        self.assertEqual(result["pattern"], "DOUBLE_JUMP")

    def test_colour_dragon_pattern(self) -> None:
        result = predict_next_derived_mark(list("RRRRRR"))
        self.assertEqual(result["pattern"], "COLOR_DRAGON")
        self.assertGreaterEqual(result["current_run_length"], 3)

    def test_human_components_exist(self) -> None:
        result = predict_next_derived_mark(list("RRUURURRRUURUR"))
        self.assertEqual(
            set(result["components"]),
            {"recent", "pattern_replay", "ngram", "run_rhythm"},
        )
        self.assertAlmostEqual(
            result["probabilities"]["R"] + result["probabilities"]["U"],
            1.0,
            places=12,
        )


class AskRoadScenarioTests(unittest.TestCase):
    def test_ask_road_requires_multiple_mature_roads_for_formal_reliability(self) -> None:
        roads = {
            "big_eye": list("RURURURURURU"),
            "small_road": [],
            "cockroach_road": [],
        }
        scenarios = {
            "B": {"big_eye": "U", "small_road": "", "cockroach_road": ""},
            "P": {"big_eye": "R", "small_road": "", "cockroach_road": ""},
        }
        result = score_ask_road_scenarios(roads, scenarios)
        self.assertLess(result["active_road_count"], 2)
        self.assertEqual(result["reliability"], 0.0)

    def test_three_road_ask_likelihood_is_capped(self) -> None:
        roads = {
            "big_eye": list("RURURURURURU"),
            "small_road": list("RURURURURU"),
            "cockroach_road": list("RURURURU"),
        }
        scenarios = {
            "B": {"big_eye": "U", "small_road": "U", "cockroach_road": "U"},
            "P": {"big_eye": "R", "small_road": "R", "cockroach_road": "R"},
        }
        result = score_ask_road_scenarios(roads, scenarios)
        self.assertEqual(result["active_road_count"], 3)
        self.assertLessEqual(result["reliability"], MAX_DERIVED_ROAD_RELIABILITY)
        self.assertAlmostEqual(
            result["likelihood"]["B"] + result["likelihood"]["P"],
            1.0,
            places=12,
        )
        self.assertGreaterEqual(result["cross_road_agreement"], 0.0)
        self.assertLessEqual(result["cross_road_agreement"], 1.0)

    def test_standard_derived_road_builder_still_works(self) -> None:
        standard = build_standard_derived_roads(list("BPBPBBPPBPBPBBPP"))
        self.assertIn("big_eye", standard)
        self.assertIn("small_road", standard)
        self.assertIn("cockroach_road", standard)
        for name in ("big_eye", "small_road", "cockroach_road"):
            self.assertTrue(all(mark in {"R", "U"} for mark in standard[name]))


if __name__ == "__main__":
    unittest.main()
