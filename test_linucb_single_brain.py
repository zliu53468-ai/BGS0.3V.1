from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import contextual_bandit as cb
from contextual_bandit import (
    CONTEXT_DIM,
    CONTEXT_FEATURE_NAMES,
    ContextGenerator,
    ContextSnapshot,
    ContextualLinUCB,
    STATE_VERSION,
    make_scope_key,
)
from validated_decision_layer import apply_strategy_decision

EXPECTED_FEATURES = (
    "remaining_cards_ratio",
    "rank_A_relative_ratio",
    "rank_2_relative_ratio",
    "rank_3_relative_ratio",
    "rank_4_relative_ratio",
    "rank_5_relative_ratio",
    "rank_6_relative_ratio",
    "rank_7_relative_ratio",
    "rank_8_relative_ratio",
    "rank_9_relative_ratio",
    "rank_10JQK_relative_ratio",
    "combinatorial_advantage_offset",
    "hsmm_stable_probability",
    "run_length_hazard_rate",
    "derived_road_regularity_binary",
    "current_run_length_norm",
)


class SingleBrainContextTests(unittest.TestCase):
    def test_context_dimension_and_order_are_fixed(self) -> None:
        self.assertEqual(CONTEXT_DIM, 16)
        self.assertEqual(CONTEXT_FEATURE_NAMES, EXPECTED_FEATURES)
        self.assertEqual(STATE_VERSION, "LINUCB-2ARM-SINGLE-BRAIN-CONTEXT-V5")

    @patch.object(cb, "build_standard_derived_roads", return_value={"big_eye": ["R"], "small_road": ["R"], "cockroach_road": ["U"]})
    @patch.object(cb, "analyze_run_length_hazard", return_value={"turn_probability": 0.60})
    @patch.object(cb, "analyze_hidden_regime", return_value={"stable_probability": 0.70})
    @patch.object(cb, "update_and_predict_engine", return_value={})
    @patch.object(cb, "estimate_probabilistic_shoe", return_value={"bp_conditional_probabilities": {"B": 0.55, "P": 0.45}, "reliability": 0.20})
    def test_context_values_are_16d_and_no_last_hand_feature(self, *_mocks) -> None:
        fresh = cb.fresh_counts(8)
        snapshot = ContextGenerator().build("BBB", {"decks": 8, "remaining_counts": fresh, "remaining_cards": sum(fresh)})
        vector = snapshot.vector
        self.assertEqual(vector.shape, (16,))
        self.assertAlmostEqual(vector[0], 1.0, places=12)
        for index in range(1, 11):
            self.assertAlmostEqual(vector[index], 1.0, places=12)
        self.assertAlmostEqual(vector[12], 0.70, places=12)
        self.assertAlmostEqual(vector[13], 0.60, places=12)
        self.assertEqual(vector[14], 1.0)
        self.assertAlmostEqual(vector[15], 3.0 / 8.0, places=12)
        self.assertNotIn("last_side_signed", CONTEXT_FEATURE_NAMES)
        self.assertNotIn("recent8_banker_centered", CONTEXT_FEATURE_NAMES)

    def test_scope_keys_isolate_uid_room_and_shoe(self) -> None:
        base = make_scope_key(user_id="u1", venue="DG", room="1", shoe_id="s1")
        self.assertNotEqual(base, make_scope_key(user_id="u2", venue="DG", room="1", shoe_id="s1"))
        self.assertNotEqual(base, make_scope_key(user_id="u1", venue="DG", room="2", shoe_id="s1"))
        self.assertNotEqual(base, make_scope_key(user_id="u1", venue="DG", room="1", shoe_id="s2"))


class SingleBrainDecisionTests(unittest.TestCase):
    def _root(self) -> dict:
        b = np.ones(CONTEXT_DIM, dtype=float)
        return {"version": STATE_VERSION, "dim": CONTEXT_DIM, "alpha": cb.LINUCB_ALPHA, "ridge": cb.LINUCB_RIDGE, "forgetting": cb.LINUCB_FORGETTING, "scopes": {"scope": {"arms": {"B": {"A": np.eye(CONTEXT_DIM).tolist(), "b": b.tolist(), "n": 5, "effective_n": 5.0}, "P": {"A": np.eye(CONTEXT_DIM).tolist(), "b": (-b).tolist(), "n": 5, "effective_n": 5.0}}, "pending": {}, "updates": 10, "last_selected": "", "selection_streak": 0}}}

    def test_final_direction_is_linucb_ucb_argmax(self) -> None:
        bandit = ContextualLinUCB()
        bandit.generator.build = lambda *_args, **_kwargs: ContextSnapshot(vector=np.ones(CONTEXT_DIM, dtype=float), metadata={})
        root = self._root()
        with patch.object(cb, "_read_state", return_value=root), patch.object(cb, "_write_state", return_value=None):
            result = bandit.predict(history="BPBP", shoe_context={}, scope_key="scope")
        self.assertGreater(result["scores"]["B"]["score"], result["scores"]["P"]["score"])
        self.assertEqual(result["direction"], "B")
        self.assertEqual(result["selected_arm"], "B")
        self.assertEqual(result["linucb_direction_weight"], 1.0)
        self.assertFalse(result["external_road_vote_enabled"])
        self.assertFalse(result["anti_echo_external_penalty"])

    def test_strategy_cannot_reduce_bet_below_five_percent(self) -> None:
        prediction = {"direction": "B", "action": "B", "recommend": "B", "probabilities": {"B": 0.58, "P": 0.42, "T": 0.0}}
        result = apply_strategy_decision(prediction, strategy_selection={"selected_arm": "conservative", "profile": {"kelly_multiplier": 0.01}}, bankroll=10000.0)
        self.assertTrue(result["bet_allowed"])
        self.assertEqual(result["direction"], "B")
        self.assertGreaterEqual(result["bet_percentage"], 5.0)
        self.assertLessEqual(result["bet_percentage"], 30.0)

    def test_validation_does_not_change_linucb_direction(self) -> None:
        from validated_decision_layer import apply_validated_decision
        prediction = {"direction": "P", "action": "P", "recommend": "P", "probabilities": {"B": 0.45, "P": 0.55, "T": 0.0}, "dynamic_prediction_policy": {"forecast": {"direction": "B", "probabilities": {"B": 0.99, "P": 0.01, "T": 0.0}}}}
        result = apply_validated_decision(prediction)
        self.assertEqual(result["direction"], "P")
        self.assertEqual(result["action"], "P")
        self.assertEqual(result["recommend"], "P")
        self.assertTrue(result["bet_allowed"])
        self.assertGreaterEqual(result["bet_percentage"], 5.0)
        self.assertLessEqual(result["bet_percentage"], 30.0)


if __name__ == "__main__":
    unittest.main()
