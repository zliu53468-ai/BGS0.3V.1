"""無外部套件的核心回歸測試：python -m unittest -v。"""
from __future__ import annotations

import unittest

from shoe_composition import (
    analyze_shoe_composition,
    exact_next_hand_probabilities,
    fresh_counts,
    parse_card_value,
    remaining_counts_from_observed,
)


class ShoeCompositionTests(unittest.TestCase):
    def test_standard_eight_deck_probabilities(self) -> None:
        banker, player, tie = exact_next_hand_probabilities(tuple(fresh_counts(8)))
        self.assertAlmostEqual(banker, 0.4585974226, places=9)
        self.assertAlmostEqual(player, 0.4462466093, places=9)
        self.assertAlmostEqual(tie, 0.0951559680, places=9)
        self.assertAlmostEqual(banker + player + tie, 1.0, places=12)

    def test_outcome_history_is_not_treated_as_card_composition(self) -> None:
        result = analyze_shoe_composition({"outcomes": ["B", "P", "T"]})
        self.assertFalse(result["available"])
        self.assertEqual(result["action"], "O")
        self.assertEqual(result["recommended_bet_percentage"], 0.0)

    def test_fresh_shoe_is_negative_ev_after_commission(self) -> None:
        result = analyze_shoe_composition({"remaining_counts": fresh_counts(8)})
        self.assertTrue(result["available"])
        self.assertEqual(result["action"], "O")
        self.assertLess(result["expected_returns"]["B"], 0.0)
        self.assertLess(result["expected_returns"]["P"], 0.0)

    def test_positive_ev_case_opens_only_capped_kelly(self) -> None:
        # 極端但合法的測試牌組：移除所有 5..9，驗證正 EV 與硬風險上限。
        counts = fresh_counts(8)[:5] + [0, 0, 0, 0, 0]
        result = analyze_shoe_composition({"remaining_counts": counts})
        self.assertEqual(result["action"], "P")
        self.assertGreater(result["selected_expected_return"], 0.0)
        self.assertGreater(result["recommended_bet_percentage"], 0.0)
        self.assertLessEqual(result["recommended_bet_percentage"], 2.0)

    def test_card_parser_and_exact_removal(self) -> None:
        self.assertEqual(parse_card_value("A"), 1)
        for face in ("10", 10, "J", "Q", "K", 0):
            self.assertEqual(parse_card_value(face), 0)
        counts = remaining_counts_from_observed(["A", 8, "K", 3], decks=8)
        self.assertEqual(counts[0], 127)
        self.assertEqual(counts[1], 31)
        self.assertEqual(counts[3], 31)
        self.assertEqual(counts[8], 31)


if __name__ == "__main__":
    unittest.main()
