from __future__ import annotations

import unittest

from bbb_final57_runtime import (
    FEATURE_DIM,
    HISTORY_INPUT_DIM,
    PHYSICS_DIM,
    apply_final_prediction,
    history_vector,
    model_status,
)
from dynamic_prediction_policy import road_only_policy
from predictor import predict


HISTORY = list("BPPBBPBPBBPPBPTBPBBPPBPB")


class BBBFinal57RuntimeTests(unittest.TestCase):
    def test_model_bundles_and_dimensions(self) -> None:
        status = model_status()
        self.assertTrue(status["physics"])
        self.assertTrue(status["final57"])
        self.assertEqual(status["feature_dim"], FEATURE_DIM)
        self.assertEqual(HISTORY_INPUT_DIM, 213)
        self.assertEqual(PHYSICS_DIM, 48)
        self.assertEqual(FEATURE_DIM, 57)

    def test_history_vector_is_213d(self) -> None:
        vector = history_vector(HISTORY)
        self.assertEqual(vector.shape, (HISTORY_INPUT_DIM,))

    def test_v23_r1_to_final57_shape(self) -> None:
        core = road_only_policy(
            HISTORY,
            shoe_context={},
            user_id="ci-final57",
            venue="DG",
            room="R1",
            shoe_id="ci-shoe",
        )
        self.assertEqual(core.get("version"), "V23_SHORT_X_DYNAMIC_HAZARD_R1")
        final = apply_final_prediction(HISTORY, core)
        self.assertIn(final["direction"], {"B", "P", "Skip"})
        self.assertEqual(len(final["physics_48d"]), PHYSICS_DIM)
        self.assertEqual(len(final["features_57d"]), FEATURE_DIM)
        self.assertAlmostEqual(final["probabilities"]["B"] + final["probabilities"]["P"], 1.0, places=8)
        self.assertGreaterEqual(final["final_p_b"], final["probability_bounds"]["min"])
        self.assertLessEqual(final["final_p_b"], final["probability_bounds"]["max"])

    def test_public_predictor_uses_final57(self) -> None:
        output = predict(
            history=HISTORY,
            venue="DG",
            room="R1",
            shoe_id="ci-public-shoe",
            user_id="ci-public",
            shoe_context={"bankroll": 100000},
        )
        self.assertEqual(output["engine"], "BBB_V23_R1_PHYSICS48_FINAL57_EV")
        self.assertIn(output["action"], {"B", "P", "Skip"})
        self.assertTrue(output["bbb_final57"]["active"])
        self.assertEqual(len(output["bbb_final57"]["features_57d"]), FEATURE_DIM)
        if output["action"] == "Skip":
            self.assertFalse(output["bet_allowed"])
            self.assertEqual(output["suggested_bet_amount"], 0.0)


if __name__ == "__main__":
    unittest.main()
