"""Regression tests for the production LSTM + shoe + cut-card fusion."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import lstm_road_model
from dynamic_prediction_policy import lstm_primary_policy
from lstm_road_model import LSTMRoadModel
from predictor import predict
from shoe_depth_estimator import (
    TARGET_HANDS_MAX,
    TARGET_HANDS_MIN,
    build_cut_card_features,
)


def _diagnostic_policy() -> dict:
    return {
        "direction": "P",
        "selected_arm": "P",
        "probabilities": {"B": 0.45, "P": 0.55, "T": 0.0},
        "selected_win_probability": 0.55,
        "confidence": 0.55,
        "context_vector": [0.0] * 16,
        "context_feature_names": [f"road_{i}" for i in range(16)],
        "context_metadata": {},
        "scores": {
            "B": {"uncertainty": 0.0},
            "P": {"uncertainty": 0.0},
        },
        "feedback_update": {},
        "ridge": 1.0,
        "road_forecaster": {
            "model_id": "test-road",
            "effective_support": 20.0,
        },
        "effective_support": 20.0,
        "regression_analysis": {},
        "scope_key": "test-scope",
    }


def _fused_stub(direction: str = "B", confidence: float = 0.57) -> dict:
    p_b, p_p = (
        (confidence, 1.0 - confidence)
        if direction == "B"
        else (1.0 - confidence, confidence)
    )
    return {
        **_diagnostic_policy(),
        "direction": direction,
        "selected_arm": direction,
        "action": direction,
        "probabilities": {"B": p_b, "P": p_p, "T": 0.0},
        "selected_win_probability": confidence,
        "confidence": confidence,
        "formal_direction_source": "lstm_road_model",
        "policy_source": "lstm_road_model",
        "fallback_markov": {
            "direction": "P",
            "probabilities": {"B": 0.46, "P": 0.54, "T": 0.0},
            "selected_win_probability": 0.54,
            "context_support": 5.0,
            "selected_order": 1,
            "diagnostic_only": True,
        },
        "fallback_active": False,
        "fallback_reason": "",
        "road_forecaster_diagnostic": {
            "direction": "P",
            "probabilities": {"B": 0.45, "P": 0.55, "T": 0.0},
            "selected_win_probability": 0.55,
        },
        "lstm": {
            "available": True,
            "direction": direction,
            "probabilities": {"B": p_b, "P": p_p},
            "raw_confidence": confidence,
            "neural": {
                "available": True,
                "probability_b": 0.55,
                "probability_p": 0.45,
                "maturity": 0.65,
            },
            "shoe_fusion": {
                "remaining_cards": 180.0,
                "remaining_ratio": 180.0 / 416.0,
                "penetration": 1.0 - 180.0 / 416.0,
                "shoe_stage": "MATURE",
                "remaining_cards_reliability": 1.0,
                "cut_card_remaining_cards": 70.0,
                "cut_progress": 0.68,
                "cards_until_cut": 110.0,
                "estimated_hands_until_cut": 22.4,
                "exact_composition_available": True,
                "shoe_direction": direction,
                "depth_feature_source": "remaining_counts",
            },
            "fusion": {
                "lstm_weight": 0.55,
                "shoe_weight": 0.62,
                "structure_weight": 0.18,
                "cut_progress": 0.68,
                "direction": direction,
            },
            "training_balance": {
                "b_samples": 10,
                "p_samples": 10,
                "balanced": True,
            },
        },
    }


def _exact_analysis(context: dict) -> dict:
    side = str(context.get("test_side") or "B").upper()
    if side == "B":
        p_b, p_p = 0.70, 0.25
    else:
        p_b, p_p = 0.25, 0.70
    remaining = float(context.get("remaining_cards", 180.0) or 180.0)
    return {
        "available": True,
        "source": "remaining_counts",
        "remaining_cards": remaining,
        "remaining_counts": [int(max(1, remaining // 10))] * 10,
        "probabilities": {"B": p_b, "P": p_p, "T": 0.05},
        "expected_returns": {"B": 0.0, "P": 0.0, "T": None},
    }


class LSTMProductionFusionTests(unittest.TestCase):
    def assert_bp_contract(self, result: dict) -> None:
        self.assertIn(result["recommend"], {"B", "P"})
        self.assertIn(result["action"], {"B", "P"})
        self.assertIn(result["next_round_direction"], {"B", "P"})
        self.assertEqual(result["recommend"], result["action"])
        self.assertEqual(result["action"], result["next_round_direction"])
        self.assertTrue(result["bet_allowed"])
        self.assertGreaterEqual(result["bet_percentage"], 5.0)
        self.assertLessEqual(result["bet_percentage"], 30.0)
        self.assertFalse(result["skip"])

    def test_cold_start_without_tensorflow_still_returns_fused_bp(self) -> None:
        model = LSTMRoadModel(scope_key="cold-start")
        with patch("lstm_road_model._tensorflow", return_value=None):
            result = model.predict(
                "BPBPBPBP",
                shoe_context={"remaining_cards": 360},
            )
        self.assertTrue(result["available"])
        self.assertIn(result["direction"], {"B", "P"})
        self.assertFalse(result["neural"]["available"])
        self.assertFalse(result["fallback_required"])
        self.assertEqual(result["formal_direction_source"], "lstm_road_model")

    def test_exact_shoe_can_change_fused_direction(self) -> None:
        history = "BPBPBPBPBPBPBPBP"
        with patch("lstm_road_model._tensorflow", return_value=None), patch(
            "lstm_road_model.analyze_shoe_composition",
            side_effect=_exact_analysis,
        ):
            banker = LSTMRoadModel(scope_key="shoe-b").predict(
                history,
                shoe_context={
                    "remaining_cards": 180,
                    "test_side": "B",
                    "cut_card_remaining_cards": 70,
                },
            )
            player = LSTMRoadModel(scope_key="shoe-p").predict(
                history,
                shoe_context={
                    "remaining_cards": 180,
                    "test_side": "P",
                    "cut_card_remaining_cards": 70,
                },
            )
        self.assertEqual(banker["direction"], "B")
        self.assertEqual(player["direction"], "P")
        self.assertGreater(banker["fusion"]["shoe_weight"], 0.0)
        self.assertGreater(player["fusion"]["shoe_weight"], 0.0)

    def test_cut_depth_changes_lstm_and_shoe_fusion_weights(self) -> None:
        history = "BPBPBPBPBPBPBPBP"

        def analysis(context: dict) -> dict:
            return _exact_analysis({**context, "test_side": "B"})

        with patch("lstm_road_model._tensorflow", return_value=None), patch(
            "lstm_road_model.analyze_shoe_composition",
            side_effect=analysis,
        ):
            early = LSTMRoadModel(scope_key="cut-early").predict(
                history,
                shoe_context={
                    "remaining_cards": 360,
                    "cut_card_remaining_cards": 70,
                },
            )
            late = LSTMRoadModel(scope_key="cut-late").predict(
                history,
                shoe_context={
                    "remaining_cards": 90,
                    "cut_card_remaining_cards": 70,
                },
            )
        self.assertLess(
            early["shoe_fusion"]["cut_progress"],
            late["shoe_fusion"]["cut_progress"],
        )
        self.assertLess(
            early["fusion"]["shoe_weight"],
            late["fusion"]["shoe_weight"],
        )

    def test_history_prefix_change_resets_per_shoe_model(self) -> None:
        model = LSTMRoadModel(scope_key="prefix-reset")
        with model._lock:
            model._ensure_history_alignment(list("BPBPBP"))
            before = model._reset_count
            model._ensure_history_alignment(list("PPBBPP"))
        self.assertGreater(model._reset_count, before)
        self.assertEqual(
            model._last_reset_reason,
            "history_prefix_changed_or_new_shoe",
        )

    def test_single_formal_policy_never_uses_markov_fallback(self) -> None:
        fused = {
            "available": True,
            "direction": "B",
            "probabilities": {"B": 0.57, "P": 0.43},
            "confidence": 0.57,
            "formal_direction_source": "lstm_road_model",
            "fusion": {"lstm_weight": 0.5, "shoe_weight": 0.5},
        }
        with patch(
            "dynamic_prediction_policy.linucb_policy",
            return_value=_diagnostic_policy(),
        ), patch(
            "dynamic_prediction_policy.predict_lstm_road",
            return_value=fused,
        ):
            result = lstm_primary_policy(
                "BPBPBPBP",
                shoe_context={"remaining_cards": 300},
                user_id="U",
                venue="DG",
                room="1",
                shoe_id="S",
            )
        self.assertEqual(result["direction"], "B")
        self.assertEqual(result["formal_direction_source"], "lstm_road_model")
        self.assertFalse(result["fallback_active"])
        self.assertEqual(result["formal_direction_weight"], 1.0)
        self.assertTrue(result["fallback_markov"]["diagnostic_only"])

    def test_predictor_exposes_production_fusion_and_bp_kelly_contract(self) -> None:
        with patch(
            "predictor.lstm_primary_policy",
            return_value=_fused_stub("B", 0.57),
        ), patch(
            "predictor.recent_user_direction_feedback",
            return_value={},
        ):
            result = predict(
                history="BPBPBBPPBPBPBPBP",
                venue="DG",
                room="1",
                shoe_id="TEST",
                user_id="unit-test-user",
                shoe_context={
                    "bankroll": 10000,
                    "remaining_cards": 180,
                    "cut_card_remaining_cards": 70,
                },
            )
        self.assert_bp_contract(result)
        self.assertEqual(result["engine"], "LSTM_SHOE_CUT_FUSION_BP")
        self.assertEqual(result["formal_direction_source"], "lstm_road_model")
        self.assertTrue(result["shoe_context_used_for_formal_direction"])
        self.assertEqual(result["fallback_markov_direction_weight"], 0.0)
        self.assertEqual(result["road_direction_weight"], 0.0)
        self.assertEqual(result["run_length_hazard_weight"], 0.0)
        self.assertFalse(result["run_length_hazard_fusion"]["applied"])
        self.assertTrue(result["lstm_shoe_cut_fusion"])

    def test_cut_feature_targets_50_to_70_hand_shoe(self) -> None:
        features = build_cut_card_features(
            70,
            hand_count=TARGET_HANDS_MAX,
            shoe_decks=8,
            cut_card_remaining_cards=70,
        )
        self.assertEqual(features["target_hands_min"], TARGET_HANDS_MIN)
        self.assertEqual(features["target_hands_max"], TARGET_HANDS_MAX)
        self.assertTrue(features["cut_reached"])
        self.assertAlmostEqual(features["cut_progress"], 1.0, places=12)
        self.assertTrue(features["within_50_70_hand_window"])


if __name__ == "__main__":
    unittest.main()
