"""Formal routing regression for the LSTM-primary BGS predictor."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from dynamic_prediction_policy import road_only_policy, time_decay_markov_fallback
from lstm_road_model import LSTMRoadModel
from pattern_survival import calibrate_lstm_confidence
from predictor import predict
from shoe_composition import analyze_shoe_composition

BANKER_BETTER_1D = [13, 2, 0, 3, 0, 4, 2, 1, 4, 1]
PLAYER_BETTER_1D = [16, 2, 1, 1, 4, 1, 0, 0, 0, 3]


def _diagnostic_policy(direction: str = "B") -> dict:
    p_b, p_p = (0.56, 0.44) if direction == "B" else (0.44, 0.56)
    return {"direction": direction, "selected_arm": direction, "probabilities": {"B": p_b, "P": p_p, "T": 0.0}, "selected_win_probability": max(p_b, p_p), "confidence": max(p_b, p_p), "context_vector": [0.0] * 16, "context_feature_names": [f"road_{i}" for i in range(16)], "context_metadata": {}, "scores": {"B": {"uncertainty": 0.0}, "P": {"uncertainty": 0.0}}, "feedback_update": {}, "ridge": 1.0, "road_forecaster": {"model_id": "test-road", "effective_support": 20.0}, "effective_support": 20.0, "regression_analysis": {}, "scope_key": "test-scope"}


def _primary_policy(direction: str = "P", source: str = "lstm_road_model", confidence: float = 0.60) -> dict:
    p_b, p_p = ((confidence, 1.0 - confidence) if direction == "B" else (1.0 - confidence, confidence))
    fallback = {"direction": "B", "probabilities": {"B": 0.54, "P": 0.46, "T": 0.0}, "selected_win_probability": 0.54, "context_support": 4.0, "selected_order": 1, "decay": 0.93}
    return {**_diagnostic_policy("B"), "direction": direction, "selected_arm": direction, "action": direction, "probabilities": {"B": p_b, "P": p_p, "T": 0.0}, "selected_win_probability": confidence, "confidence": confidence, "formal_direction_source": source, "policy_source": source, "lstm": {"available": source == "lstm_road_model", "direction": direction if source == "lstm_road_model" else None, "probabilities": {"B": p_b, "P": p_p}, "raw_confidence": confidence, "reason": "test"}, "fallback_markov": fallback, "fallback_active": source != "lstm_road_model", "fallback_reason": "cold_start" if source != "lstm_road_model" else "", "road_forecaster_diagnostic": {"direction": "B", "probabilities": {"B": 0.56, "P": 0.44, "T": 0.0}, "selected_win_probability": 0.56}}


class LSTMPrimaryFormalDecisionTests(unittest.TestCase):
    def _predict(self, shoe_context: dict, *, direction: str = "P", source: str = "lstm_road_model", confidence: float = 0.60, history: str = "BPBPBBPPBPBP"):
        with patch("predictor.lstm_primary_policy", return_value=_primary_policy(direction, source, confidence)), patch("predictor.recent_user_direction_feedback", return_value={}):
            return predict(history=history, venue="DG", room="1", shoe_id="TEST", user_id="unit-test-user", shoe_context={"bankroll": 10000, **shoe_context})

    def assert_bp_contract(self, result: dict) -> None:
        self.assertIn(result["recommend"], {"B", "P"}); self.assertIn(result["action"], {"B", "P"}); self.assertIn(result["next_round_direction"], {"B", "P"})
        self.assertEqual(result["recommend"], result["action"]); self.assertEqual(result["action"], result["next_round_direction"]); self.assertTrue(result["bet_allowed"])
        self.assertGreaterEqual(result["bet_percentage"], 5.0); self.assertLessEqual(result["bet_percentage"], 30.0); self.assertFalse(result["skip"])

    def test_exact_shoe_does_not_override_lstm_direction(self) -> None:
        result = self._predict({"remaining_counts": BANKER_BETTER_1D, "decks": 1}, direction="P")
        self.assert_bp_contract(result); self.assertEqual(result["action"], "P"); self.assertEqual(result["formal_direction_source"], "lstm_road_model")
        self.assertFalse(result["shoe_context_used_for_formal_direction"]); self.assertEqual(result["card_composition_direction_weight"], 0.0); self.assertEqual(result["lstm_direction_weight"], 1.0)
        self.assertEqual(result["card_composition_source"], "remaining_counts"); self.assertIsNotNone(result["banker_ev"]); self.assertIsNotNone(result["player_ev"]); self.assertFalse(result["shoe_composition"]["formal_direction_authority"])

    def test_two_exact_compositions_keep_same_lstm_direction(self) -> None:
        banker = self._predict({"remaining_counts": BANKER_BETTER_1D, "decks": 1}, direction="P"); player = self._predict({"remaining_counts": PLAYER_BETTER_1D, "decks": 1}, direction="P")
        self.assertEqual(banker["action"], "P"); self.assertEqual(player["action"], "P"); self.assertEqual(banker["formal_direction_source"], "lstm_road_model"); self.assertEqual(player["formal_direction_source"], "lstm_road_model")

    def test_markov_fallback_owns_direction_during_cold_start(self) -> None:
        result = self._predict({}, direction="B", source="time_decay_markov_fallback", confidence=0.54, history="BPBPBB")
        self.assert_bp_contract(result); self.assertEqual(result["action"], "B"); self.assertEqual(result["formal_direction_source"], "time_decay_markov_fallback")
        self.assertEqual(result["lstm_direction_weight"], 0.0); self.assertEqual(result["fallback_markov_direction_weight"], 1.0)

    def test_late_shoe_reduces_confidence_without_flip(self) -> None:
        opening = calibrate_lstm_confidence(direction="P", raw_confidence=0.64, remaining_card_state={"shoe_stage": "OPENING", "reliability": 1.0, "lstm_shoe_stage_factor": 1.0})
        late = calibrate_lstm_confidence(direction="P", raw_confidence=0.64, remaining_card_state={"shoe_stage": "LATE", "reliability": 1.0, "lstm_shoe_stage_factor": 0.88})
        self.assertEqual(late["direction"], "P"); self.assertGreater(late["confidence"], 0.5); self.assertLess(late["confidence"], opening["confidence"]); self.assertGreater(late["probabilities"]["P"], late["probabilities"]["B"])

    def test_shoe_composition_api_remains_available_for_diagnostics(self) -> None:
        exact = analyze_shoe_composition({"remaining_counts": BANKER_BETTER_1D, "decks": 1})
        self.assertTrue(exact["available"]); self.assertIn(exact["direction"], {"B", "P"}); self.assertEqual(set(exact["probabilities"]), {"B", "P", "T"})
        self.assertAlmostEqual(exact["ev"]["B"], exact["banker_ev"], places=12); self.assertAlmostEqual(exact["ev"]["P"], exact["player_ev"], places=12)

    def test_lstm_model_cold_start_does_not_require_tensorflow(self) -> None:
        result = LSTMRoadModel(scope_key="unit-test").predict("BPBPBP")
        self.assertFalse(result["available"]); self.assertEqual(result["reason"], "cold_start_insufficient_history"); self.assertEqual(result["probabilities"], {"B": 0.5, "P": 0.5})

    def test_time_decay_markov_fallback_is_deterministic_bp(self) -> None:
        first = time_decay_markov_fallback("BPBPBBPPBP"); second = time_decay_markov_fallback("BPBPBBPPBP")
        self.assertEqual(first["direction"], second["direction"]); self.assertIn(first["direction"], {"B", "P"}); self.assertAlmostEqual(first["probabilities"]["B"] + first["probabilities"]["P"], 1.0, places=12)

    def test_road_only_policy_forwards_shoe_context_but_shoe_has_no_vote(self) -> None:
        context = {"remaining_cards": 250, "remaining_cards_source": "round_count_estimate"}
        with patch("dynamic_prediction_policy.predict_bandit", return_value=_diagnostic_policy("B")) as diagnostic, patch("dynamic_prediction_policy.predict_lstm_road", return_value={"available": False, "direction": None, "probabilities": {"B": 0.5, "P": 0.5}, "confidence": 0.5, "reason": "cold_start_insufficient_history"}):
            result = road_only_policy("BPBP", shoe_context=context, user_id="U", venue="DG", room="1", shoe_id="S")
        self.assertIn(result["direction"], {"B", "P"}); self.assertEqual(result["formal_direction_source"], "time_decay_markov_fallback"); self.assertEqual(diagnostic.call_args.kwargs["shoe_context"], context)


if __name__ == "__main__":
    unittest.main()
