"""BBB exact web Soft Shoe V2 / BGS OCR integration regression contract."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
import unittest
from typing import Any, Mapping

import bbb_soft_shoe_runtime as soft
from bbb_final57_runtime import _node_command

ROOT = Path(__file__).resolve().parent
SAMPLES = list("BPPBTBBPPT")


class SoftShoeOCRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        soft.verify_source()

    def test_sources_and_config_match_bbb(self):
        self.assertEqual(soft.SOURCE_COMMIT, "3cc2e27b8cf8b1eb935addc46708d056b332b4a8")
        self.assertEqual(soft.VERSION, "BBB_SOFT_SHOE_BPT_EV_V2")
        self.assertEqual(set(soft.SOURCE_BLOBS), {
            "soft_shoe_decision_runtime.js", "soft_shoe_model_bundle.js",
            "soft_shoe_frozen_lstm_runtime.js", "soft_shoe_frozen_model_bundle.js",
        })
        for name, sha in soft.SOURCE_BLOBS.items():
            data = (ROOT / "bbb_engine" / name).read_bytes()
            self.assertEqual(
                hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
                sha
            )
        report = soft.analyze_history(SAMPLES[:3])
        cfg = report["config"]
        self.assertEqual((cfg["PARTICLE_COUNT"], cfg["LIKELIHOOD_DRAWS"], cfg["NEXT_HAND_DRAWS"]), (128, 14, 14))
        self.assertEqual((cfg["MIN_DIRECTION_ROUNDS"], cfg["MIN_DIRECTION_EV_GAP"],
                          cfg["MIN_DIRECTION_GAP_SNR"]), (3, .020, .70))
        self.assertEqual(report["observations"], 3)
        self.assertEqual(report["feature_schema"], "soft_shoe_bpt_48d_v2")
        self.assertEqual(report["feature_dimension"], 48)
        self.assertEqual(len(report["features48"]), 48)
        self.assertEqual(report["sequence_weight_applied"], 0)
        self.assertEqual(report["frozen_core"]["validated"], False)
        self.assertEqual(report["frozen_core"]["trained"], False)
        self.assertEqual(report["model"], "physics_only_untrained_decision_fallback")
        self.assertEqual(report["action"], "Skip")
        self.assertEqual(report["bankroll_fraction"], 0)
        self.assertFalse(report["validated_edge"])

    def test_research_direction_exact_web_parity_and_no_cross_uid_memory(self):
        for history in ([], ["B"], ["T"], list("BPPBT"), SAMPLES * 2):
            actual = soft.analyze_history(history)
            source = (
                "global.self=global;const core=require('./bbb_engine/soft_shoe_decision_runtime.js');"
                "require('./bbb_engine/soft_shoe_model_bundle.js');"
                "const frozen=require('./bbb_engine/soft_shoe_frozen_lstm_runtime.js');"
                "require('./bbb_engine/soft_shoe_frozen_model_bundle.js');"
                "process.stdout.write(JSON.stringify(core.run(JSON.parse(process.argv[1]),{},"
                "global.BBB_SOFT_SHOE_XGB_MODEL,frozen,global.BBB_SOFT_SHOE_LSTM_MODEL)));"
            )
            direct = subprocess.run(
                _node_command() + ["-e", source, json.dumps(history)],
                cwd=ROOT, capture_output=True, text=True, check=True, timeout=8.0
            )
            self.assertEqual(actual, json.loads(direct.stdout))
            self.assertEqual(actual["bankroll_fraction"], 0)
            self.assertEqual(actual["sequence_weight_applied"], 0)
            self.assertEqual(actual["feature_schema"], "soft_shoe_bpt_48d_v2")
            self.assertEqual(len(actual["features48"]), 48)
        a = soft.predict_screen(SAMPLES, user_id="UID-A", venue="DG", room="1", shoe_id="shoe1")
        b = soft.predict_screen(SAMPLES, user_id="UID-B", venue="DG", room="1", shoe_id="shoe2")
        self.assertEqual(a["bbb_soft_shoe"]["report"], b["bbb_soft_shoe"]["report"])
        self.assertEqual(a["action"], b["action"])
        self.assertEqual(a["bet_percentage"], 0)
        self.assertTrue(a["bbb_soft_shoe"]["active"])
        self.assertFalse(a["bbb_soft_shoe"]["official_bet_allowed"])
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0)
        p = soft.predict_screen(SAMPLES[:-1],user_id="UID-A")
        self.assertEqual(p["bbb_soft_shoe"]["report"]["observations"],len(SAMPLES)-1)
        self.assertEqual(soft.predict_screen(SAMPLES,user_id="UID-A")["bbb_soft_shoe"]["report"],
                         a["bbb_soft_shoe"]["report"])

    def test_v2_frozen_sequence_and_xgb_schema_fail_closed(self):
        # Includes production untrained LSTM bundle, no fake learned weights.
        out = soft.analyze_history(list("BBPTBBPT"))
        self.assertEqual(out["feature_schema"], "soft_shoe_bpt_48d_v2")
        self.assertEqual(out["frozen_core"]["reason"], "validated_frozen_lstm_unavailable")
        self.assertFalse(out["validated_edge"])
        self.assertEqual(out["sequence_weight_applied"], 0)
        self.assertEqual(out["bankroll_fraction"], 0)
        self.assertIn(out["research_view"]["direction"], ("B","P","SKIP"))
        js = (
            "global.self=global;"
            "const core=require('./bbb_engine/soft_shoe_decision_runtime.js');"
            "const frozen=require('./bbb_engine/soft_shoe_frozen_lstm_runtime.js');"
            "const mock={trained:true,calibrated:true,feature_dimension:48,"
            "schema:'soft_shoe_bpt_48d_v1',"
            "validation:{real_shoe_data:true,out_of_shoe_holdout:true},"
            "predict:()=>({B:.7,P:.2,T:.1})};"
            "process.stdout.write(core.modelDecision({B:.45,P:.45,T:.1},"
            "Array(48).fill(0),core.frozenCore(['B'],frozen,null),mock).model);"
        )
        result = subprocess.run(_node_command()+["-e",js],cwd=ROOT,
                                capture_output=True,text=True,check=True,timeout=8)
        self.assertEqual(result.stdout,"physics_only_untrained_decision_fallback")

    def test_shoe_depth_and_invalid_input_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "72"):
            soft.analyze_history(["B"] * 73)
        with self.assertRaises(ValueError):
            soft.analyze_history(["B", "X"])
        no_bet = soft.predict_screen(["B", "T", "P"])
        self.assertIn(no_bet["action"], ("B", "P", "SKIP"))
        self.assertEqual(no_bet["suggested_bet_amount"], 0)

    def test_cold_fifty_round_ocr_import_under_model_budget(self):
        began = time.monotonic()
        report = soft.analyze_history(SAMPLES * 5)
        seconds = time.monotonic() - began
        self.assertEqual(report["observations"], 50)
        self.assertLess(seconds, 8.5, "A cold OCR snapshot must not exhaust LINE model budget")

    def test_line_result_card_and_zero_stake_boundary(self):
        src = (ROOT/"runtime_app.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and
               x.name in ("_formal_bp_action","_attach_formal_bankroll_advice")]
        env={"Mapping":Mapping,"Any":Any,"MIN_FORMAL_BET_RATIO":.05,
             "MAX_FORMAL_BET_RATIO":.30}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),"runtime_boundary","exec"),env)
        attach=env["_attach_formal_bankroll_advice"]
        for candidate in ("B", "P", "SKIP"):
            result = attach({
                "action":candidate,"bet_allowed":True,"bet_amount":5000,
                "money_management":{"bet_allowed":True,"bet_amount":5000},
                "bbb_soft_shoe":{"active":True,"report":{"action":"Skip",
                   "validated_edge":False,"bankroll_fraction":0,
                   "research_view":{"direction":candidate,"reason":"test_case"}}}
            },{"bankroll":100000,"permanent_access":True,"trial_started_at":123})
            self.assertEqual(result["action"],"Skip" if candidate=="SKIP" else candidate)
            self.assertEqual(result["bet_amount"],0)
            self.assertEqual(result["suggested_bet_amount"],0)
            self.assertEqual(result["bet_percentage"],0)
            self.assertFalse(result["bet_allowed"])
            self.assertFalse(result["money_management"]["bet_allowed"])
        with self.assertRaises(RuntimeError):
            attach({"bbb_soft_shoe":{"active":True,"report":{
                "action":"Bet Banker","validated_edge":True,"bankroll_fraction":.02,
                "research_view":{"direction":"B"}}}},{"bankroll":100000})
        self.assertIn("_screen_adapter.predict = _bbb_soft_shoe_predict",src)

    def test_latest_ocr_panel_from_pr79_instead_of_point_score_ui(self):
        text=(ROOT/"app.py").read_text(encoding="utf-8")
        self.assertIn("LINE_OCR_ENABLED = True",text)
        self.assertIn("def _process_screen_image_sync(",text)
        self.assertIn("請開始上傳最新完整遊戲畫面",text)
        self.assertIn("首次辨識完成後，每局只需回報莊／閒／和",text)
        self.assertIn("session = _start_new_uid_analysis(user_id)",text)
        self.assertIn("return {\"panel\": upload_request_panel(user_id, session), \"state\": \"image\"}",text)
        self.assertIn("session = _request_bankroll(user_id)",text)
        self.assertIn('_postback_button("🔴 本局結果：莊"',text)
        self.assertIn('_postback_button("🔵 本局結果：閒"',text)
        self.assertIn('_postback_button("🟢 本局結果：和"',text)
        self.assertIn('"point_score_mode": False',text)
        self.assertIn("store.update_screen_analysis(user_id",text)
        self.assertIn("expected_run_id=str(expected_run_id or \"\")",text)
        self.assertIn("expected_data_version=expected_version",text)
        self.assertIn("if score_pairs is not None and session.get(\"point_score_mode\"):",text)
        self.assertIn("if not LINE_OCR_ENABLED:",text)  # guarded fallback retained

    def test_render_and_license_files_unchanged(self):
        doc=(ROOT/"BBB_SOFT_SHOE_OCR_RELEASE.md").read_text(encoding="utf-8")
        self.assertIn("PR #79",doc)
        self.assertIn("ededd462",doc)
        source=(ROOT/"bbb_engine/manifest.json").read_text(encoding="utf-8")
        manifest=json.loads(source)
        for path in ("store.py","render.yaml","room_ocr.py","screen_pipeline.py",
                     "road_detector.py","static/liff.html","screenshot_predictor.py"):
            actual=hashlib.sha256((ROOT/path).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
            self.assertEqual(actual,manifest["protected_files_sha256"][path])


if __name__=="__main__":
    unittest.main()
