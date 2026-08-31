from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}: {old[:120]!r}")
    write(path, text.replace(old, new, 1))


# predictor.py: formal routing is exact-shoe-first; road is fallback only.
replace_once(
    "predictor.py",
    "    linucb_policy,\n    normalize_big_road,\n",
    "    linucb_policy,\n    normalize_big_road,\n    road_only_policy,\n",
)

old_predictor_block = '''    # Road path is always evaluated so diagnostics/online feedback/public interfaces
    # remain compatible. Exact shoe composition, when available, owns formal direction.
    policy = linucb_policy(
        raw_history,
        shoe_context=context,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )
    road_probabilities = dict(policy["probabilities"])
    road_direction = str(policy["direction"])
    road_confidence = float(policy["selected_win_probability"])

    shoe_analysis = dict(analyze_shoe_composition(context))
    shoe_available = bool(shoe_analysis.get("available"))
    composition_source = _shoe_source(context, shoe_analysis)

    if shoe_available:
        probabilities = dict(shoe_analysis.get("probabilities") or {})
        returns = dict(shoe_analysis.get("expected_returns") or {})
        b_ev = float(returns.get("B", 0.0) or 0.0)
        p_ev = float(returns.get("P", 0.0) or 0.0)
        # No EV gate and no third arm: even if both sides are negative, choose the
        # mathematically better B/P side. Low/negative edge is handled by Kelly floor.
        direction = "B" if b_ev >= p_ev else "P"
        confidence = _resolved_confidence(probabilities, direction)
        formal_source = "shoe_composition_ev_argmax"
        card_weight = 1.0
        road_weight = 0.0
        selected_physical_ev = b_ev if direction == "B" else p_ev
        shoe_analysis["action"] = direction
        shoe_analysis["action_text"] = "莊" if direction == "B" else "閒"
        shoe_analysis["formal_direction"] = direction
        shoe_analysis["formal_no_observe_arm"] = True
    else:
        probabilities = dict(road_probabilities)
        direction = road_direction if road_direction in {"B", "P"} else "B"
        confidence = road_confidence
        formal_source = "road_forecaster_probability_argmax"
        card_weight = 0.0
        road_weight = 1.0
        selected_physical_ev = None
        shoe_analysis["action"] = None
        shoe_analysis["action_text"] = "牌靴資料不可用，正式方向回退牌路"
        shoe_analysis["formal_direction"] = direction
        shoe_analysis["formal_no_observe_arm"] = True
'''

new_predictor_block = '''    # Formal routing is intentionally shoe-first. Exact card composition is
    # validated before any road fallback is allowed to own the public B/P decision.
    # The road model may still run after an exact shoe decision for diagnostics only.
    shoe_analysis = dict(analyze_shoe_composition(context))
    shoe_available = bool(shoe_analysis.get("available"))
    composition_source = _shoe_source(context, shoe_analysis)

    if shoe_available:
        probabilities = dict(shoe_analysis.get("probabilities") or {})
        returns = dict(shoe_analysis.get("expected_returns") or {})
        b_ev = float(returns.get("B", 0.0) or 0.0)
        p_ev = float(returns.get("P", 0.0) or 0.0)
        # No EV gate and no third arm: even if both sides are negative, choose the
        # mathematically better B/P side. Low/negative edge is handled by Kelly floor.
        direction = "B" if b_ev >= p_ev else "P"
        confidence = _resolved_confidence(probabilities, direction)
        formal_source = "exact_shoe_ev"
        card_weight = 1.0
        road_weight = 0.0
        selected_physical_ev = b_ev if direction == "B" else p_ev
        shoe_analysis["action"] = direction
        shoe_analysis["action_text"] = "莊" if direction == "B" else "閒"
        shoe_analysis["formal_direction"] = direction
        shoe_analysis["formal_no_observe_arm"] = True

        # Compatibility/feedback diagnostics only. Crucially this happens after
        # exact shoe EV has already selected the formal direction, and the caller's
        # complete shoe_context is forwarded instead of being replaced by {}.
        policy = linucb_policy(
            raw_history,
            shoe_context=context,
            user_id=user_id,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
        )
        road_probabilities = dict(policy["probabilities"])
        road_direction = str(policy["direction"])
        road_confidence = float(policy["selected_win_probability"])
    else:
        # Missing/invalid exact composition is non-fatal: only here may the legacy
        # road-only compatibility entry own the formal B/P direction.
        policy = road_only_policy(
            raw_history,
            shoe_context=context,
            user_id=user_id,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
        )
        road_probabilities = dict(policy["probabilities"])
        road_direction = str(policy["direction"])
        road_confidence = float(policy["selected_win_probability"])
        probabilities = dict(road_probabilities)
        direction = road_direction if road_direction in {"B", "P"} else "B"
        confidence = road_confidence
        formal_source = "road_forecaster"
        card_weight = 0.0
        road_weight = 1.0
        selected_physical_ev = None
        shoe_analysis["action"] = None
        shoe_analysis["action_text"] = "牌靴資料不可用，正式方向回退牌路"
        shoe_analysis["formal_direction"] = direction
        shoe_analysis["formal_no_observe_arm"] = True
'''
replace_once("predictor.py", old_predictor_block, new_predictor_block)

# Keep the existing compatibility key and add the exact product-spec alias.
replace_once(
    "predictor.py",
    '            "road_context_direction_weight": road_weight,\n            "card_composition_source": composition_source,\n',
    '            "road_context_direction_weight": road_weight,\n            "road_direction_weight": road_weight,\n            "card_composition_source": composition_source,\n',
)
replace_once(
    "predictor.py",
    '    policy["road_context_direction_weight"] = road_weight\n    policy["card_composition_direction_weight"] = card_weight\n',
    '    policy["road_context_direction_weight"] = road_weight\n    policy["road_direction_weight"] = road_weight\n    policy["card_composition_direction_weight"] = card_weight\n',
)
replace_once(
    "predictor.py",
    '        "road_context_direction_weight": road_weight,\n        "card_composition_source": composition_source,\n',
    '        "road_context_direction_weight": road_weight,\n        "road_direction_weight": road_weight,\n        "card_composition_source": composition_source,\n',
)

# dynamic_prediction_policy.py: road_only_policy is explicitly fallback-only and
# must never erase caller-provided shoe_context. Add optional kwargs compatibly.
replace_once(
    "dynamic_prediction_policy.py",
    '"""BGS 動態決策相容層：正式方向由因果式 road_forecaster 產生。\n\n保留舊模組常用 helper 名稱，避免其他程式 import 失效；但所有正式 P/B\n方向經既有 contextual_bandit 入口轉接 forecaster 機率 argmax。\nLinUCB 與事後迴歸僅供診斷，不能覆蓋正式下一手方向。\n"""',
    '"""BGS 牌路 fallback 相容層。\n\n正式入口在 predictor：有精確牌組時由 shoe composition EV 先決定 B/P；\n只有缺少或無效牌組時才使用本模組的 road_forecaster fallback。保留舊 helper\n名稱避免外部 import 失效；LinUCB 與事後迴歸只作牌路診斷/相容。\n"""',
)
replace_once(
    "dynamic_prediction_policy.py",
    '''def road_only_policy(history: str | Iterable[Any] | None) -> dict[str, Any]:
    """保留舊入口名稱與簽名；呼叫正式下一手 forecaster。"""
    return linucb_policy(history, shoe_context={})
''',
    '''def road_only_policy(
    history: str | Iterable[Any] | None,
    *,
    shoe_context: Mapping[str, Any] | None = None,
    user_id: str = "",
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> dict[str, Any]:
    """無精確牌組時的 B/P fallback 相容入口。

    `shoe_context` 只作 depth/metadata 相容並完整往下傳，不會在此被清成 `{}`。
    本函式不做 shoe-composition EV，因此不能覆蓋 predictor 的 exact-shoe-first
    正式路徑。
    """
    return linucb_policy(
        history,
        shoe_context=shoe_context,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )
''',
)

# shoe_composition.py: expose an unambiguous available/direction/probabilities/ev API.
replace_once(
    "shoe_composition.py",
    '        "available": False,\n        "action": None,\n',
    '        "available": False,\n        "direction": None,\n        "action": None,\n',
)
replace_once(
    "shoe_composition.py",
    '        "expected_returns": {\n            "B": None,\n            "P": None,\n            "T": None,\n        },\n',
    '        "probabilities": {"B": None, "P": None, "T": None},\n        "expected_returns": {\n            "B": None,\n            "P": None,\n            "T": None,\n        },\n        "ev": {"B": None, "P": None, "T": None},\n',
)
replace_once(
    "shoe_composition.py",
    '            "available": True,\n            "action": selected_side,\n',
    '            "available": True,\n            "direction": selected_side,\n            "action": selected_side,\n',
)
replace_once(
    "shoe_composition.py",
    '            "expected_returns": returns,\n            "banker_ev": float(returns["B"]),\n',
    '            "expected_returns": returns,\n            "ev": {\n                "B": float(returns["B"]),\n                "P": float(returns["P"]),\n                "T": None,\n            },\n            "banker_ev": float(returns["B"]),\n',
)

# Replace the formal wiring regression test with exact acceptance checks.
(ROOT / "test_formal_shoe_decision.py").write_text(
    '''"""Formal routing regression: exact shoe first, road fallback second, B/P only."""\n'
    'from __future__ import annotations\n\n'
    'import unittest\n'
    'from unittest.mock import patch\n\n'
    'from dynamic_prediction_policy import road_only_policy\n'
    'from predictor import predict\n'
    'from shoe_composition import analyze_shoe_composition\n\n\n'
    'BANKER_BETTER_1D = [13, 2, 0, 3, 0, 4, 2, 1, 4, 1]\n'
    'PLAYER_BETTER_1D = [16, 2, 1, 1, 4, 1, 0, 0, 0, 3]\n\n\n'
    'def _road_policy(direction: str = "B") -> dict:\n'
    '    p_b, p_p = (0.56, 0.44) if direction == "B" else (0.44, 0.56)\n'
    '    return {\n'
    '        "direction": direction, "selected_arm": direction,\n'
    '        "probabilities": {"B": p_b, "P": p_p, "T": 0.0},\n'
    '        "selected_win_probability": max(p_b, p_p), "confidence": max(p_b, p_p),\n'
    '        "context_vector": [0.0] * 16,\n'
    '        "context_feature_names": [f"road_{index}" for index in range(16)],\n'
    '        "context_metadata": {"remaining_cards": 0.0, "remaining_cards_source": "round_count_estimate", "estimated_remaining_counts_0_to_9": []},\n'
    '        "scores": {"B": {"uncertainty": 0.0}, "P": {"uncertainty": 0.0}},\n'
    '        "feedback_update": {}, "ridge": 1.0,\n'
    '        "road_forecaster": {"model_id": "test-road", "effective_support": 20.0},\n'
    '        "effective_support": 20.0, "regression_analysis": {}, "scope_key": "test-scope",\n'
    '    }\n\n\n'
    'class FormalShoeDecisionTests(unittest.TestCase):\n'
    '    def _predict(self, shoe_context: dict):\n'
    '        with patch("predictor.linucb_policy", return_value=_road_policy("B")) as diagnostic, patch(\n'
    '            "predictor.road_only_policy", return_value=_road_policy("B")\n'
    '        ) as fallback, patch("predictor.recent_user_direction_feedback", return_value={}):\n'
    '            result = predict(\n'
    '                history="BPBPBBPP", venue="DG", room="1", shoe_id="TEST",\n'
    '                user_id="unit-test-user", shoe_context={"bankroll": 10000, **shoe_context},\n'
    '            )\n'
    '        return result, diagnostic, fallback\n\n'
    '    def assert_bp_contract(self, result: dict) -> None:\n'
    '        self.assertIn(result["recommend"], {"B", "P"})\n'
    '        self.assertIn(result["action"], {"B", "P"})\n'
    '        self.assertIn(result["next_round_direction"], {"B", "P"})\n'
    '        self.assertEqual(result["recommend"], result["action"])\n'
    '        self.assertEqual(result["action"], result["next_round_direction"])\n'
    '        self.assertTrue(result["bet_allowed"])\n'
    '        self.assertGreaterEqual(result["bet_percentage"], 5.0)\n'
    '        self.assertLessEqual(result["bet_percentage"], 30.0)\n'
    '        self.assertFalse(result["skip"])\n\n'
    '    def test_no_composition_uses_road_fallback_and_forwards_context(self) -> None:\n'
    '        result, diagnostic, fallback = self._predict({"remaining_cards": 300})\n'
    '        self.assert_bp_contract(result)\n'
    '        self.assertEqual(result["action"], "B")\n'
    '        self.assertEqual(result["formal_direction_source"], "road_forecaster")\n'
    '        self.assertFalse(result["shoe_context_used_for_formal_direction"])\n'
    '        self.assertEqual(result["card_composition_direction_weight"], 0.0)\n'
    '        self.assertEqual(result["road_direction_weight"], 1.0)\n'
    '        self.assertEqual(result["road_context_direction_weight"], 1.0)\n'
    '        fallback.assert_called_once()\n'
    '        self.assertEqual(fallback.call_args.kwargs["shoe_context"]["remaining_cards"], 300)\n'
    '        self.assertEqual(fallback.call_args.kwargs["shoe_context"]["bankroll"], 10000)\n'
    '        diagnostic.assert_not_called()\n\n'
    '    def test_remaining_counts_own_formal_direction_before_road(self) -> None:\n'
    '        banker, banker_diag, banker_fallback = self._predict({"remaining_counts": BANKER_BETTER_1D, "decks": 1})\n'
    '        player, player_diag, player_fallback = self._predict({"remaining_counts": PLAYER_BETTER_1D, "decks": 1})\n'
    '        self.assert_bp_contract(banker); self.assert_bp_contract(player)\n'
    '        self.assertEqual(banker["action"], "B")\n'
    '        self.assertEqual(player["action"], "P")\n'
    '        self.assertEqual(player["formal_direction_source"], "exact_shoe_ev")\n'
    '        self.assertTrue(player["shoe_context_used_for_formal_direction"])\n'
    '        self.assertEqual(player["card_composition_direction_weight"], 1.0)\n'
    '        self.assertEqual(player["road_direction_weight"], 0.0)\n'
    '        self.assertEqual(player["road_context_direction_weight"], 0.0)\n'
    '        self.assertEqual(player["card_composition_source"], "remaining_counts")\n'
    '        banker_fallback.assert_not_called(); player_fallback.assert_not_called()\n'
    '        banker_diag.assert_called_once(); player_diag.assert_called_once()\n'
    '        self.assertEqual(player_diag.call_args.kwargs["shoe_context"]["remaining_counts"], PLAYER_BETTER_1D)\n'
    '        self.assertIsNotNone(player["banker_ev"]); self.assertIsNotNone(player["player_ev"])\n'
    '        self.assertEqual(player["dynamic_prediction_policy"]["forecast"]["road_direction_before_shoe_override"], "B")\n\n'
    '    def test_observed_cards_are_second_priority_exact_source(self) -> None:\n'
    '        result, diagnostic, fallback = self._predict({"observed_cards": ["A", 8, "K", 3], "decks": 8})\n'
    '        self.assert_bp_contract(result)\n'
    '        self.assertEqual(result["formal_direction_source"], "exact_shoe_ev")\n'
    '        self.assertTrue(result["shoe_context_used_for_formal_direction"])\n'
    '        self.assertEqual(result["card_composition_source"], "observed_cards")\n'
    '        self.assertEqual(result["remaining_cards_source"], "observed_cards")\n'
    '        fallback.assert_not_called(); diagnostic.assert_called_once()\n\n'
    '    def test_remaining_counts_have_priority_over_observed_cards(self) -> None:\n'
    '        result, _, fallback = self._predict({\n'
    '            "remaining_counts": PLAYER_BETTER_1D, "observed_cards": ["A", 8, "K", 3], "decks": 1\n'
    '        })\n'
    '        self.assertEqual(result["card_composition_source"], "remaining_counts")\n'
    '        self.assertEqual(result["action"], "P")\n'
    '        fallback.assert_not_called()\n\n'
    '    def test_invalid_composition_falls_back_without_error(self) -> None:\n'
    '        result, diagnostic, fallback = self._predict({"remaining_counts": [999] * 10, "decks": 8})\n'
    '        self.assert_bp_contract(result)\n'
    '        self.assertEqual(result["formal_direction_source"], "road_forecaster")\n'
    '        self.assertFalse(result["shoe_context_used_for_formal_direction"])\n'
    '        self.assertEqual(result["shoe_composition"]["reason_code"], "INVALID_CARD_COMPOSITION")\n'
    '        fallback.assert_called_once(); diagnostic.assert_not_called()\n\n'
    '    def test_shoe_composition_has_clear_direction_and_ev_api(self) -> None:\n'
    '        exact = analyze_shoe_composition({"remaining_counts": BANKER_BETTER_1D, "decks": 1})\n'
    '        self.assertTrue(exact["available"])\n'
    '        self.assertEqual(exact["direction"], exact["action"])\n'
    '        self.assertIn(exact["direction"], {"B", "P"})\n'
    '        self.assertEqual(set(exact["probabilities"]), {"B", "P", "T"})\n'
    '        self.assertAlmostEqual(exact["ev"]["B"], exact["banker_ev"], places=12)\n'
    '        self.assertAlmostEqual(exact["ev"]["P"], exact["player_ev"], places=12)\n\n'
    '    def test_road_only_policy_does_not_clear_shoe_context(self) -> None:\n'
    '        context = {"remaining_cards": 250, "remaining_cards_source": "round_count_estimate"}\n'
    '        with patch("dynamic_prediction_policy.predict_bandit", return_value=_road_policy("B")) as mocked:\n'
    '            result = road_only_policy("BPBP", shoe_context=context, user_id="U", venue="DG", room="1", shoe_id="S")\n'
    '        self.assertIn(result["direction"], {"B", "P"})\n'
    '        self.assertEqual(mocked.call_args.kwargs["shoe_context"], context)\n\n\n'
    'if __name__ == "__main__":\n'
    '    unittest.main()\n'
    ,
    encoding="utf-8",
)

print("Exact-shoe-first formal policy patch applied.")
