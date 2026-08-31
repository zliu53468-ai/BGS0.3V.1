"""Regression tests for unified shoe configuration and depth semantics."""
from __future__ import annotations

import unittest

from shoe_constants import (
    AVERAGE_CARDS_PER_HAND,
    BURN_CARDS,
    REFERENCE_HANDS,
    SHOE_DECKS,
    estimate_remaining_cards,
    fresh_point_counts,
    total_cards_for_decks,
)
from shoe_composition import analyze_shoe_composition, remaining_counts_from_observed
from shoe_depth_estimator import ShoeDepthEstimator


class UnifiedShoeConstantsTests(unittest.TestCase):
    def test_eight_deck_geometry(self) -> None:
        self.assertEqual(total_cards_for_decks(8), 416)
        self.assertEqual(fresh_point_counts(8), [128] + [32] * 9)

    def test_round_count_estimate_20_hands(self) -> None:
        remaining = estimate_remaining_cards(
            20,
            decks=8,
            average_cards_per_hand=4.9,
            burn_cards=0,
        )
        self.assertAlmostEqual(remaining, 318.0, places=9)
        estimate = ShoeDepthEstimator(
            shoe_decks=8,
            average_cards_per_hand=4.9,
            burn_cards=0,
            reference_hands=70,
        ).estimate(["B", "P"] * 10).as_dict()
        self.assertEqual(estimate["remaining_cards_source"], "round_count_estimate")
        self.assertFalse(estimate["exact_composition"])
        self.assertAlmostEqual(float(estimate["remaining_cards"]), 318.0, places=9)
        self.assertAlmostEqual(float(estimate["shoe_progress"]), 20.0 / 70.0, places=9)
        self.assertEqual(
            estimate["reference_hands_semantics"],
            "product_maturity_reference_not_cut_card_position",
        )

    def test_observed_cards_are_exact_not_round_estimate(self) -> None:
        observed = ["A", 8, "K", 3]
        counts = remaining_counts_from_observed(observed, decks=8)
        self.assertEqual(sum(counts), 412)
        result = analyze_shoe_composition({"observed_cards": observed, "decks": 8})
        self.assertTrue(result["available"])
        self.assertEqual(result["remaining_cards_source"], "observed_cards")
        self.assertIn(result["action"], {"B", "P"})
        self.assertEqual(sum(result["remaining_counts"]), 412)

    def test_authoritative_defaults_are_exposed(self) -> None:
        self.assertGreaterEqual(SHOE_DECKS, 1)
        self.assertGreaterEqual(AVERAGE_CARDS_PER_HAND, 4.0)
        self.assertGreaterEqual(REFERENCE_HANDS, 1)
        self.assertGreaterEqual(BURN_CARDS, 0)


if __name__ == "__main__":
    unittest.main()
