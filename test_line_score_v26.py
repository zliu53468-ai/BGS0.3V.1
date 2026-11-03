"""BGS LINE exact BBB V2.6 independent final-score parity & UID isolation."""
from __future__ import annotations
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bbb_v26_points_runtime as bridge
import line_score_mode
import store

ROOT = Path(__file__).resolve().parent


class LineScoreV26Tests(unittest.TestCase):
    def test_immutable_bbb_v26_modules_and_config(self):
        bridge.verify_source()
        self.assertEqual(bridge.SOURCE_COMMIT, "a9e25b42d003f1a721007d6ae68127082ab86a29")
        for player, banker in ((8, 4), (9, 7), (7, 8), (0, 0)):
            r = bridge.predict_final_score(player, banker)
            self.assertEqual(r["config"], {
                "N": 96, "L": 256, "M": 96,
                "ess_resampling_ratio": .5,
                "physics_weight": 5, "road_weight": .1,
                "independentFinalScoreMode": True,
            })
            self.assertEqual(r["history_length"], 1)
            self.assertEqual(r["point_score_observation_count"], 1)
            self.assertEqual(r["physical_ev"]["samples"], 9216)
            self.assertEqual(r["raw_direction"], "B" if r["ev_banker"] >= r["ev_player"] else "P")
            self.assertIn(r["direction"], ("B", "P", "SKIP"))
            self.assertIn(r["display_signal"]["reason"], ("small_ev_gap_or_uncertain", "relative_ev_gap_pass"))
            self.assertEqual(r["display_signal"]["version"], "BBB_V26_SELECTIVE_ENTRY_V3")
            self.assertTrue(r["display_signal"]["moderate_entry_enabled"])
            self.assertEqual(r["display_signal"]["entry_profile"], "moderate_high")
            self.assertGreater(r["display_signal"]["threshold_addition"], 0)
            self.assertFalse(r["display_signal"]["verified_edge"])
            if r["direction"] != "SKIP":
                self.assertEqual(r["direction"], r["raw_direction"])
            self.assertEqual(r["early_shoe_guard"]["mode"], "independent_final_score_quality")
            self.assertEqual(r["action"], "SKIP")
            self.assertFalse(r["bet_allowed"])
            self.assertFalse(r["final_scores_are_observed_ranks"])
            self.assertAlmostEqual(sum(r["physical_ev"]["simulated_probability"].values()), 1.0, places=10)

    def test_deterministic_single_score_no_historical_particle_reuse(self):
        first = bridge.predict_final_score(8, 4)
        bridge.predict_final_score(1, 6)
        again = bridge.predict_final_score(8, 4)
        self.assertEqual(first, again)

    def test_all_100_pairs_match_bbb_v26_ci_distribution(self):
        counts = {"B": 0, "P": 0, "SKIP": 0}
        raw_counts = {"B": 0, "P": 0}
        for player in range(10):
            for banker in range(10):
                r = bridge.predict_final_score(player, banker)
                draw = r["physical_ev"]["draw_length_diagnostics"]
                self.assertEqual(draw["version"], "BBB_V26_TWO_HAND_DRAW_LENGTH_V1")
                self.assertTrue(draw["diagnostic_only"])
                self.assertFalse(draw["actual_current_length_known"])
                self.assertFalse(draw["affects_ev_or_direction"])
                self.assertEqual(draw["samples"], 9216)
                self.assertAlmostEqual(sum(draw["current_length_probability"].values()), 1, places=9)
                self.assertAlmostEqual(sum(draw["next_length_probability"].values()), 1, places=9)
                self.assertAlmostEqual(sum(
                    value for row in draw["joint_probability"].values()
                    for value in row.values()), 1, places=9)
                for current in ("4", "5", "6"):
                    self.assertAlmostEqual(
                        sum(draw["joint_probability"][current].values()),
                        draw["current_length_probability"][current], places=9)
                for upcoming in ("4", "5", "6"):
                    self.assertAlmostEqual(
                        sum(draw["joint_probability"][current][upcoming]
                            for current in ("4", "5", "6")),
                        draw["next_length_probability"][upcoming], places=9)
                counts[r["direction"]] += 1
                raw_counts[r["raw_direction"]] += 1
                self.assertEqual(r["display_signal"]["raw_direction"], r["raw_direction"])
        self.assertEqual(sum(raw_counts.values()), 100)
        self.assertGreater(raw_counts["B"], 0)
        self.assertGreater(raw_counts["P"], 0)
        self.assertEqual(sum(counts.values()), 100)
        self.assertEqual(raw_counts, {"B": 56, "P": 44})
        self.assertEqual(counts, {"B": 36, "P": 34, "SKIP": 30},
                         "LINE V3 100-score benchmark must match BBB")
        self.assertEqual(raw_counts, {"B": 56, "P": 44})
        print("BGS V2.6 V3 selective N96/L256/M96 display:", counts, "raw:", raw_counts)

    def test_text_parsing_is_unambiguous(self):
        self.assertEqual(line_score_mode.parse_score_pairs("閒8莊4"), [{"player": 8, "banker": 4}])
        self.assertEqual(line_score_mode.parse_score_pairs("莊4 閒8"), [{"player": 8, "banker": 4}])
        self.assertEqual(line_score_mode.parse_score_pairs("8,4"), [{"player": 8, "banker": 4}])
        self.assertEqual(line_score_mode.parse_score_pairs("53"), [{"player": 5, "banker": 3}])
        self.assertEqual(line_score_mode.parse_score_pairs("５３"), [{"player": 5, "banker": 3}])
        self.assertEqual(line_score_mode.parse_score_pairs("09"), [{"player": 0, "banker": 9}])
        self.assertEqual(line_score_mode.parse_score_pairs("53\n97"), [
            {"player": 5, "banker": 3}, {"player": 9, "banker": 7}
        ])

        self.assertEqual(line_score_mode.parse_score_pairs("閒9莊7\n閒8莊4"),
                         [{"player": 9, "banker": 7}, {"player": 8, "banker": 4}])
        for bad in ("開始分析", "8", "10,9", "本局閒八莊九", "abc", "11223344", "0053", "1234"):
            self.assertIsNone(line_score_mode.parse_score_pairs(bad), bad)

    def test_two_UIDs_persist_separately_and_stale_run_cannot_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(store, "SESSION_DATA_FILE", Path(tmp) / "sessions.json"):
                store.select_venue("UID-A", "DG")
                store.select_venue("UID-B", "MT")
                a = line_score_mode.begin("UID-A")
                b = line_score_mode.begin("UID-B")
                self.assertNotEqual(a["analysis_run_id"], b["analysis_run_id"])
                self.assertEqual(a["venue"], "DG")
                self.assertEqual(b["venue"], "MT")
                self.assertFalse(a["awaiting_screenshot"])
                self.assertFalse(a["awaiting_bankroll"])
                run_a = a["analysis_run_id"]
                done_a = line_score_mode.evaluate_and_save(
                    "UID-A", run_a, [{"player": 8, "banker": 4}], "line-message-1")
                self.assertEqual(done_a["point_score_audit"], [{"player": 8, "banker": 4}])
                self.assertEqual(done_a["screen_last_prediction"]["bet_amount"], 0)
                self.assertEqual(done_a["screen_last_prediction"]["point_score_observation_count"], 1)
                self.assertEqual(store.get_session("UID-B").get("point_score_audit"), [])
                duplicate = line_score_mode.evaluate_and_save(
                    "UID-A", run_a, [{"player": 8, "banker": 4}], "line-message-1")
                self.assertEqual(len(duplicate["point_score_audit"]), 1)
                # Different account, same score: identical physical prediction, separate audit.
                done_b = line_score_mode.evaluate_and_save(
                    "UID-B", b["analysis_run_id"], [{"player": 8, "banker": 4}], "line-message-2")
                self.assertEqual(done_a["screen_last_prediction"]["bbb_v26_score"],
                                 done_b["screen_last_prediction"]["bbb_v26_score"])
                store.end_session("UID-A")
                with self.assertRaisesRegex(RuntimeError, "已結束"):
                    line_score_mode.evaluate_and_save(
                        "UID-A", run_a, [{"player": 3, "banker": 5}], "late-message")
                reopened = line_score_mode.begin("UID-A")
                self.assertEqual(reopened["point_score_audit"], [])
                self.assertNotEqual(reopened["analysis_run_id"], run_a)
                self.assertEqual(len(store.get_session("UID-B")["point_score_audit"]), 1)
                with self.assertRaisesRegex(RuntimeError, "已結束"):
                    line_score_mode.evaluate_and_save(
                        "UID-A", run_a, [{"player": 4, "banker": 3}], "stale-message")

    def test_original_result_card_only_removes_manual_buttons(self):
        """Exercise actual card builders without importing the FastAPI/OCR process."""
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fns = {
            node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        # Compile only presentation functions, not the API, OCR or predictor.
        from typing import Any, Dict, Mapping
        api = {
            "Any": Any, "Dict": Dict, "Mapping": Mapping,
            "_postback_button": lambda label, action, **options:
                {"label": label, "action": action, **options},
            "_yellow_white_panel": lambda title, alt, contents:
                {"title": title, "alt": alt, "body": contents},
            "_format_money": lambda amount: format(int(amount), ","),
            "point_input_panel": lambda session: {"title": "請輸入閒莊歷史點數"},
        }
        module = ast.Module(
            body=[fns["screen_result_panel"], fns["score_result_panel"]],
            type_ignores=[],
        )
        exec(compile(module, str(ROOT / "app.py"), "exec"), api)

        def result_card(direction, point_mode=True, diagnostic=None):
            result = {"direction": direction}
            if diagnostic is not None:
                result["physical_ev"] = {"draw_length_diagnostics": diagnostic}
            pred = {
                "bbb_v26_score": result,
                "action": direction, "recommend": direction,
                "banker_rate": 45.89, "player_rate": 44.93,
                "tie_rate": 9.18,
                "suggested_bet_amount": 0, "bet_percentage": 0,
            }
            session = {
                "user_id": "UID-A", "venue": "DG", "bankroll": 30000,
                "point_score_mode": point_mode,
                "screen_last_prediction": pred,
            }
            return api["score_result_panel"](session) if point_mode else (
                api["screen_result_panel"]("UID-A", session)
            )

        for direction, text in (("B", "莊"), ("P", "閒"), ("SKIP", "觀望")):
            card = result_card(direction)
            self.assertEqual(card["title"], "分析數據")
            contents = card["body"]
            self.assertEqual(contents[0]["contents"][1]["text"], "45.89%")
            self.assertEqual(contents[1]["contents"][1]["text"], "44.93%")
            self.assertEqual(contents[2]["contents"][1]["text"], "9.18%")
            self.assertEqual(contents[3]["contents"][0]["text"], "推薦")
            self.assertEqual(contents[3]["contents"][1]["text"], text)
            self.assertIn("分析本金：30,000 元", contents[5]["text"])
            buttons = contents[-1]["contents"]
            self.assertEqual([button["action"] for button in buttons], ["end"])
            self.assertEqual(buttons[0]["label"], "結束分析")

        diag = bridge.predict_final_score(5, 3)["physical_ev"]["draw_length_diagnostics"]
        detailed = result_card("B", diagnostic=diag)
        labels = [row.get("text", "") for row in detailed["body"]]
        self.assertIn("4／5／6 張聯合模擬（診斷）", labels)
        self.assertTrue(any("當局 4 張 → 下局 4／5／6 張" in label for label in labels))
        self.assertEqual(detailed["body"][-1]["contents"][0]["action"], "end")
        self.assertEqual(detailed["body"][3]["contents"][1]["text"], "莊")
        legacy = result_card("P", point_mode=False)
        self.assertEqual(legacy["title"], "分析數據")
        self.assertEqual(
            [button["action"] for button in legacy["body"][-1]["contents"]],
            ["road_append", "road_append", "road_append", "end"],
            "Legacy/OCR card remains unchanged",
        )

        self.assertEqual(
            api["score_result_panel"]({
                "screen_last_prediction": {}, "point_score_mode": True
            })["title"],
            "請輸入閒莊歷史點數",
        )
        self.assertIn('LINE_OCR_ENABLED = os.getenv("BGS_LINE_OCR_ENABLED", "0")', source)
        self.assertIn("if not LINE_OCR_ENABLED:", source)
        self.assertIn("def _process_screen_image_sync(", source)
        self.assertIn("def _process_manual_outcome_via_reply(", source)


if __name__ == "__main__":
    unittest.main()
