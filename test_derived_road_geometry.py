"""Regression tests for the derived-road six-row geometry patch."""
from __future__ import annotations

import unittest

from derived_road_geometry import (
    MAX_GEOMETRY_RELIABILITY,
    build_derived_road_geometry,
    predict_next_geometry_mark,
    score_geometry_ask_road_scenarios,
)
from road_pattern_core import forecast_road_pattern
from road_pattern_v1_core import forecast_road_pattern as forecast_v1


class DerivedRoadGeometryTests(unittest.TestCase):
    def test_five_same_colours_drop_vertically(self) -> None:
        geometry = build_derived_road_geometry(list("UUUUU"))
        positions = geometry["positions"]
        self.assertEqual([p["column"] for p in positions], [1, 1, 1, 1, 1])
        self.assertEqual([p["row"] for p in positions], [1, 2, 3, 4, 5])
        self.assertFalse(geometry["horizontal_tail_active"])

    def test_seventh_same_colour_creates_bottom_horizontal_tail(self) -> None:
        geometry = build_derived_road_geometry(list("UUUUUUU"))
        positions = geometry["positions"]
        self.assertEqual((positions[5]["row"], positions[5]["column"]), (6, 1))
        self.assertEqual((positions[6]["row"], positions[6]["column"]), (6, 2))
        self.assertEqual(positions[6]["placement"], "RIGHT_BOTTOM")
        self.assertTrue(geometry["horizontal_tail_active"])
        self.assertEqual(geometry["shape_family"], "HORIZONTAL_TAIL")

    def test_previous_tail_can_force_early_collision_turn(self) -> None:
        # First run leaves a tail in row 6 / column 2. The second run begins at
        # row 1 / column 2 and therefore turns right before reaching row 6.
        geometry = build_derived_road_geometry(list("RRRRRRRUUUUUU"))
        second_run = geometry["runs"][1]
        self.assertTrue(second_run["collision_turn"])
        self.assertTrue(second_run["has_horizontal_tail"])
        self.assertLess(second_run["end_row"], 6)

    def test_column_rhythm_is_exposed(self) -> None:
        geometry = build_derived_road_geometry(list("RRUURRUUR"))
        self.assertIn(
            geometry["shape_family"],
            {"DOUBLE_COLUMN_RHYTHM", "COLUMN_RHYTHM_2_2", "GENERIC"},
        )
        self.assertEqual(geometry["column_heights"][:4], [2, 2, 2, 2])

    def test_geometry_prediction_is_probabilistic_not_forced(self) -> None:
        result = predict_next_geometry_mark(list("RURURURU"))
        self.assertIn(result["direction"], {"R", "U"})
        self.assertGreaterEqual(result["probabilities"]["R"], 0.0)
        self.assertLessEqual(result["probabilities"]["R"], 1.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIn("geometry", result)

    def test_geometry_ask_road_requires_multiple_active_roads_and_is_capped(self) -> None:
        roads = {
            "big_eye": list("RURURURURURU"),
            "small_road": list("UURUURUURUUR"),
            "cockroach_road": list("RRUURRUURRUU"),
        }
        scenarios = {
            "B": {"big_eye": "R", "small_road": "U", "cockroach_road": "R"},
            "P": {"big_eye": "U", "small_road": "R", "cockroach_road": "U"},
        }
        result = score_geometry_ask_road_scenarios(roads, scenarios)
        self.assertLessEqual(result["reliability"], MAX_GEOMETRY_RELIABILITY)
        self.assertGreaterEqual(result["active_road_count"], 2)
        self.assertAlmostEqual(
            result["likelihood"]["B"] + result["likelihood"]["P"],
            1.0,
            places=12,
        )


class GeometryIntegrationTests(unittest.TestCase):
    def test_exact_v1_probability_is_preserved_before_auxiliary_fusion(self) -> None:
        history = "BPBPBBPPBPBPBBPPBPBP"
        v1 = forecast_v1(history)
        final = forecast_road_pattern(history)
        self.assertAlmostEqual(
            float(v1["probabilities"]["B"]),
            float(final["v1_probabilities"]["B"]),
            places=12,
        )
        self.assertEqual(v1["components"], final["components"])

    def test_geometry_layer_is_present_inside_derived_ask_road(self) -> None:
        result = forecast_road_pattern("BPBPBBPPBPBPBBPPBPBPBBPP")
        derived = result["derived_ask_road"]
        self.assertIn("sequence_layer", derived)
        self.assertIn("geometry_layer", derived)
        self.assertIn("models", derived["geometry_layer"])
        self.assertLessEqual(result["derived_direction_weight"], 0.18)
        self.assertNotIn("probabilistic_turning", result["components"])
        self.assertNotIn("change_point_probability", result)


if __name__ == "__main__":
    unittest.main()
