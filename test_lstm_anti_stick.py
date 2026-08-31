"""Focused regressions for the V3 LSTM anti-stick production core."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import lstm_road_model
from lstm_road_model import LSTMRoadModel


class _ArrayResult:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self._values


class _ConstantBiasedModel:
    """Returns the same class bias for original and B/P-swapped inputs."""

    def __call__(self, values, training=False):
        del values, training
        return _ArrayResult([[0.80, 0.20]])


def _exact_analysis(context: dict) -> dict:
    remaining = int(context.get("remaining_cards", 180) or 180)
    unit = max(1, remaining // 10)
    counts = [unit] * 10
    side = str(context.get("test_side") or "P").upper()
    if side == "B":
        probabilities = {"B": 0.70, "P": 0.25, "T": 0.05}
    else:
        probabilities = {"B": 0.25, "P": 0.70, "T": 0.05}
    return {
        "available": True,
        "source": "remaining_counts",
        "remaining_cards": float(sum(counts)),
        "remaining_counts": counts,
        "probabilities": probabilities,
        "expected_returns": {"B": 0.0, "P": 0.0, "T": None},
    }


class LSTMAntiStickTests(unittest.TestCase):
    def test_handcrafted_structure_has_zero_formal_vote(self) -> None:
        with patch("lstm_road_model._tensorflow", return_value=None):
            result = LSTMRoadModel(scope_key="no-structure").predict(
                "PPPPPPPPPPPP",
                shoe_context={"remaining_cards": 330},
            )
        self.assertEqual(result["features"]["structure_logit"], 0.0)
        self.assertEqual(result["features"]["formal_structure_weight"], 0.0)
        self.assertEqual(result["fusion"]["structure_logit"], 0.0)
        self.assertEqual(result["fusion"]["structure_weight"], 0.0)

    def test_single_side_real_replay_is_symmetry_balanced_not_skipped(self) -> None:
        x_real, y_real = lstm_road_model._training_examples(
            "PPPPPPPPPPPP",
            min_context=3,
            replay_window=20,
        )
        self.assertGreater(len(y_real), 0)
        self.assertEqual(int(np.sum(y_real == 0)), 0)
        x_aug, y_aug, weights, meta = lstm_road_model._symmetry_augmented_replay(
            x_real,
            y_real,
        )
        self.assertEqual(len(x_aug), 2 * len(x_real))
        self.assertEqual(len(y_aug), 2 * len(y_real))
        self.assertEqual(len(weights), len(y_aug))
        self.assertTrue(meta["symmetry_augmented"])
        self.assertEqual(int(np.sum(y_aug == 0)), int(np.sum(y_aug == 1)))

    def test_paired_inference_cancels_constant_class_bias(self) -> None:
        model = LSTMRoadModel(scope_key="paired-bias")
        model._model = _ConstantBiasedModel()
        model._bootstrap_done = True
        model._trained_rounds = 999
        neural = model._neural_state(
            list("BPBPBBPPBPBP"),
            allow_online_update=False,
        )
        self.assertTrue(neural["available"])
        self.assertAlmostEqual(neural["raw_probability_b"], 0.80, places=5)
        self.assertAlmostEqual(neural["swapped_probability_b"], 0.80, places=5)
        self.assertAlmostEqual(neural["symmetry_projected_logit"], 0.0, places=10)
        self.assertAlmostEqual(neural["probability_b"], 0.5, places=10)
        self.assertAlmostEqual(neural["probability_p"], 0.5, places=10)

    def test_unchanged_exact_counts_lose_direction_authority_after_any_new_hand(self) -> None:
        model = LSTMRoadModel(scope_key="stale-exact")
        context = {
            "remaining_cards": 180,
            "test_side": "P",
            "cut_card_remaining_cards": 70,
        }
        with patch("lstm_road_model._tensorflow", return_value=None), patch(
            "lstm_road_model.analyze_shoe_composition",
            side_effect=_exact_analysis,
        ):
            first = model.predict("BPBPBPBPBP", shoe_context=context)
            # T consumes physical cards but does not change the B/P LSTM sequence.
            # Freshness must therefore use full B/P/T round count, not B/P count.
            second = model.predict("BPBPBPBPBPT", shoe_context=context)

        self.assertTrue(first["shoe_fusion"]["exact_composition_fresh"])
        self.assertGreater(first["fusion"]["shoe_weight"], 0.0)
        self.assertFalse(second["shoe_fusion"]["exact_composition_fresh"])
        self.assertTrue(second["shoe_fusion"]["stale_exact_composition"])
        self.assertEqual(second["fusion"]["shoe_weight"], 0.0)
        self.assertEqual(second["fusion"]["shoe_composition_logit_deviation"], 0.0)
        self.assertEqual(
            second["shoe_fusion"]["stale_exact_reason"],
            "unchanged_exact_counts_after_history_advanced",
        )
        self.assertEqual(
            second["shoe_fusion"]["depth_feature_source"],
            "round_count_estimate_after_stale_exact",
        )

    def test_refreshed_exact_counts_regain_direction_authority(self) -> None:
        model = LSTMRoadModel(scope_key="fresh-exact")
        with patch("lstm_road_model._tensorflow", return_value=None), patch(
            "lstm_road_model.analyze_shoe_composition",
            side_effect=_exact_analysis,
        ):
            first = model.predict(
                "BPBPBPBPBP",
                shoe_context={
                    "remaining_cards": 180,
                    "test_side": "P",
                    "cut_card_remaining_cards": 70,
                },
            )
            second = model.predict(
                "BPBPBPBPBPP",
                shoe_context={
                    "remaining_cards": 170,
                    "test_side": "B",
                    "cut_card_remaining_cards": 70,
                },
            )

        self.assertTrue(first["shoe_fusion"]["exact_composition_fresh"])
        self.assertTrue(second["shoe_fusion"]["exact_composition_fresh"])
        self.assertFalse(second["shoe_fusion"]["stale_exact_composition"])
        self.assertGreater(second["fusion"]["shoe_weight"], 0.0)

    def test_repeated_prediction_same_round_does_not_invalidate_exact_counts(self) -> None:
        model = LSTMRoadModel(scope_key="same-round")
        context = {
            "remaining_cards": 180,
            "test_side": "B",
            "cut_card_remaining_cards": 70,
        }
        with patch("lstm_road_model._tensorflow", return_value=None), patch(
            "lstm_road_model.analyze_shoe_composition",
            side_effect=_exact_analysis,
        ):
            first = model.predict("BPBPBPBPBP", shoe_context=context)
            second = model.predict("BPBPBPBPBP", shoe_context=context)
        self.assertTrue(first["shoe_fusion"]["exact_composition_fresh"])
        self.assertTrue(second["shoe_fusion"]["exact_composition_fresh"])
        self.assertGreater(second["fusion"]["shoe_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
