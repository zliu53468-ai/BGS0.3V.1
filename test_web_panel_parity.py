from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

import contextual_bandit as cb


class WebPanelContextParityTests(unittest.TestCase):
    def test_exact_card_input_does_not_change_web_panel_context(self) -> None:
        history = "BBPPTBPBTBBP"
        plain = cb.ContextGenerator().build(history, {"decks": 8})
        supplied = cb.ContextGenerator().build(
            history,
            {
                "decks": 8,
                "remaining_counts": [80, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                "remaining_cards": 296,
            },
        )

        np.testing.assert_allclose(plain.vector, supplied.vector, rtol=0.0, atol=0.0)
        self.assertEqual(supplied.vector.shape, (32,))
        np.testing.assert_allclose(supplied.vector[4:14], np.ones(10), rtol=0.0, atol=0.0)
        self.assertEqual(float(supplied.vector[14]), 0.0)
        self.assertEqual(float(supplied.vector[15]), 0.0)
        self.assertTrue(
            supplied.metadata["exact_card_input_ignored_for_web_panel_compatibility"]
        )


class WebPanelDecisionParityTests(unittest.TestCase):
    def _memory_state(self) -> tuple[dict, object, object]:
        state = {
            "version": cb.STATE_VERSION,
            "dim": cb.CONTEXT_DIM,
            "alpha": cb.LINUCB_ALPHA,
            "ridge": cb.LINUCB_RIDGE,
            "forgetting": cb.LINUCB_FORGETTING,
            "scopes": {},
        }

        def read_state() -> dict:
            return deepcopy(state)

        def write_state(payload: dict) -> None:
            state.clear()
            state.update(deepcopy(payload))

        return state, read_state, write_state

    def test_new_scope_never_bootstraps_or_updates_arm_matrices(self) -> None:
        state, read_state, write_state = self._memory_state()
        bandit = cb.ContextualLinUCB()

        with patch.object(cb, "_read_state", side_effect=read_state), patch.object(
            cb, "_write_state", side_effect=write_state
        ):
            result = bandit.predict(
                history="BBPPTBPBTBBPPBTBP",
                shoe_context={"remaining_counts": [40] * 10},
                scope_key="web-parity-scope",
            )

        self.assertFalse(result["panel_bootstrap_applied"])
        self.assertEqual(result["bootstrap_update"]["reason"], "web_panel_direct_no_bootstrap")
        self.assertEqual(result["score_gap"], 0.0)
        self.assertEqual(result["direction"], "B")

        scope = state["scopes"]["web-parity-scope"]
        self.assertEqual(scope["updates"], 0)
        for arm in cb.ARMS:
            np.testing.assert_allclose(
                scope["arms"][arm]["A"],
                np.eye(cb.CONTEXT_DIM) * cb.LINUCB_RIDGE,
            )
            np.testing.assert_allclose(scope["arms"][arm]["b"], np.zeros(cb.CONTEXT_DIM))
            self.assertEqual(scope["arms"][arm]["n"], 0)
            self.assertEqual(scope["arms"][arm]["effective_n"], 0.0)

    def test_equal_scores_follow_bbb_last_selected_sequence(self) -> None:
        _, read_state, write_state = self._memory_state()
        bandit = cb.ContextualLinUCB()

        with patch.object(cb, "_read_state", side_effect=read_state), patch.object(
            cb, "_write_state", side_effect=write_state
        ):
            first = bandit.predict(history="BPPBBP", shoe_context={}, scope_key="same")
            second = bandit.predict(history="BPPBBP", shoe_context={}, scope_key="same")
            third = bandit.predict(history="BPPBBPB", shoe_context={}, scope_key="same")

        self.assertEqual(first["direction"], "B")
        self.assertEqual(first["selection_reason"], "tie_deterministic_history_hash")
        self.assertEqual(second["direction"], "P")
        self.assertEqual(second["selection_reason"], "tie_opposite_previous_arm")
        self.assertEqual(third["direction"], "B")
        self.assertEqual(third["selection_reason"], "tie_opposite_previous_arm")


if __name__ == "__main__":
    unittest.main()
