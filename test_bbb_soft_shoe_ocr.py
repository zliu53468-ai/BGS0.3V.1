"""BBB exact web Soft Shoe V5.1 / BGS OCR integration regression contract."""
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
        self.assertEqual(soft.SOURCE_COMMIT, "975c5f0a9d348691618e2dc76bde66db985ddcde")
        self.assertEqual(soft.VERSION, "BBB_SOFT_SHOE_BPT_EV_V5_1")
        self.assertEqual(set(soft.SOURCE_BLOBS), {
            "soft_shoe_decision_runtime.js", "soft_shoe_model_bundle.js",
            "soft_shoe_frozen_lstm_runtime.js", "soft_shoe_frozen_model_bundle.js",
            "derived_road_aux.js",
        })
        for name, sha in soft.SOURCE_BLOBS.items():
            data = (ROOT / "bbb_engine" / name).read_bytes()
            self.assertEqual(
                hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
                sha
            )
        report = soft.analyze_history(SAMPLES[:3])
        cfg = report["config"]
        self.assertEqual((cfg["PARTICLE_COUNT"], cfg["LIKELIHOOD_DRAWS"], cfg["NEXT_HAND_DRAWS"]), (448, 80, 128))
        self.assertEqual((cfg["MIN_DIRECTION_ROUNDS"], cfg["MIN_DIRECTION_EV_GAP"],
                          cfg["MIN_DIRECTION_GAP_SNR"]), (3, .010, 1.05))
        self.assertEqual((cfg["MIN_DIRECTION_ESS_RATIO"], cfg["MIN_DIRECTION_DIVERSITY"],
                          cfg["MAX_DIRECTION_GAP_SE"]), (.40, .25, .070))
        self.assertTrue(cfg["RESEARCH_DIRECTION_RELAXED"])
        self.assertIs(cfg["ROAD_AUX_ENABLED"], True)
        self.assertIs(cfg["ROAD_AUX_ADAPTIVE_ENABLED"], True)
        self.assertEqual(cfg["ROAD_AUX_STABILITY_WINDOW"], 8)
        self.assertEqual(cfg["ROAD_AUX_STABILITY_MIN"], .35)
        self.assertEqual(cfg["ROAD_AUX_LEARN_WINDOW"], 30)
        self.assertEqual(cfg["ROAD_AUX_LEARN_MIN_SAMPLES"], 8)
        self.assertEqual(cfg["ROAD_AUX_LEARN_FLOOR"], .40)
        self.assertEqual(cfg["ROAD_AUX_LEARN_CEIL"], 1.0)
        self.assertEqual(cfg["ROAD_AUX_LEARN_NEUTRAL"], .70)
        self.assertEqual(cfg["ROAD_AUX_MIN_HISTORY"], 12)
        self.assertEqual(cfg["ROAD_AUX_AGREEMENT_MIN"], .67)
        self.assertEqual(cfg["ROAD_AUX_CONFIDENCE_MIN"], .45)
        self.assertEqual(cfg["ROAD_AUX_MAX_BOOST"], .08)
        self.assertEqual(cfg["ROAD_AUX_CONFLICT_PENALTY"], .06)
        self.assertIs(cfg["ROAD_AUX_REQUIRE_SOFT_ELIGIBLE"], True)
        self.assertEqual(report["observations"], 3)
        self.assertEqual(cfg["ADAPTIVE_EXTRA_DRAWS"], 32)
        self.assertIn(report.get("adaptive_sampling", {}).get("extra_draws"), (0, 32))
        self.assertEqual(cfg["MIN_EV_THRESHOLD"], .0030)
        self.assertEqual(cfg["MID_SHOE_MIN_EV_THRESHOLD"], .00235)
        self.assertEqual(cfg["LATE_SHOE_MIN_EV_THRESHOLD"], .0019)
        self.assertEqual(cfg["MIN_CONFIDENCE"], .75)
        self.assertEqual(cfg["LATE_MIN_CONFIDENCE"], .80)
        self.assertEqual(cfg["MIN_BET_EV_GAP"], .0015)
        self.assertEqual(cfg["MIN_BET_ESS_RATIO"], .45)
        self.assertEqual(cfg["MAX_BET_EV_SE"], .060)
        self.assertEqual(cfg["FRACTIONAL_KELLY"], .225)
        self.assertEqual(cfg["MAX_BET_FRACTION"], .0175)
        self.assertEqual(cfg["MIN_REMAINING_CARDS_FOR_BET"], 25)
        self.assertEqual(cfg["END_SHOE_OVERRIDE_MIN_EV"], .025)
        self.assertEqual(cfg["END_SHOE_OVERRIDE_CONFIDENCE"], .90)
        self.assertEqual(cfg["END_SHOE_OVERRIDE_MAX_EV_SE"], .030)
        self.assertEqual(report["wager_gate"]["min_ev_threshold"], .0030)
        self.assertFalse(report["wager_gate"]["model_verified"])
        self.assertFalse(report["wager_gate"]["eligible"])
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
        self.assertEqual(report["road_aux"]["adaptive"]["mode"], "adaptive")
        self.assertEqual(report["road_aux"]["adaptive"]["quality"], 0)
        self.assertEqual(report["road_aux"]["adaptive"]["w_applied"], 0)


    def test_three_histories_against_byte_locked_bbb_main_engine(self):
        # The four local JS blobs are checked against BBB/main in
        # test_sources_and_config_match_bbb; run() must remain behavior-identical.
        cases = [list("BPT"), list("BPPBTBBPPT"), list("BBPPPTTBBPBPTPPBBTP")]
        script = (
            "global.self=global;"
            "const core=require('./bbb_engine/soft_shoe_decision_runtime.js');"
            "require('./bbb_engine/soft_shoe_model_bundle.js');"
            "const frozen=require('./bbb_engine/soft_shoe_frozen_lstm_runtime.js');"
            "require('./bbb_engine/soft_shoe_frozen_model_bundle.js');"
            "const histories=JSON.parse(process.argv[1]);"
            "process.stdout.write(JSON.stringify(histories.map(h=>core.run(h,{},"
            "global.BBB_SOFT_SHOE_XGB_MODEL,frozen,global.BBB_SOFT_SHOE_LSTM_MODEL))));"
        )
        direct = subprocess.run(
            _node_command() + ["-e", script, json.dumps(cases)],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=12.0
        )
        expected = json.loads(direct.stdout)
        self.assertEqual(len(expected), len(cases))
        for history, web in zip(cases, expected):
            line = soft.analyze_history(history)
            self.assertEqual(line["version"], web["version"])
            for outcome in ("B", "P", "T"):
                self.assertAlmostEqual(line["probabilities"][outcome],
                                       web["probabilities"][outcome], places=12)
            self.assertEqual(line["research_view"]["direction"],
                             web["research_view"]["direction"])
            self.assertEqual(line["research_view"]["eligible"],
                             web["research_view"]["eligible"])
            self.assertEqual(line, web)
            self.assertEqual(line["action"], "Skip")
            self.assertEqual(line["bankroll_fraction"], 0)
            self.assertIs(line["validated_edge"], False)
            cfg = line["config"]
            self.assertEqual((cfg["PARTICLE_COUNT"], cfg["LIKELIHOOD_DRAWS"],
                              cfg["NEXT_HAND_DRAWS"]), (448, 80, 128))
            self.assertIs(cfg["RESEARCH_DIRECTION_RELAXED"], True)
            self.assertFalse(soft.predict_screen(history)["bet_allowed"])


    def test_bbb_main_road_aux_parity_and_zero_stakes(self):
        # Direct BBB.run on the EXACT five source blobs. All history prefixes
        # are B/P/T only; neither OCR nor older road feature input is passed.
        cases = [
            [],
            ["B", "P", "T", "B"],
            list("BP" * 12),  # alternating road; derived marks available
            list("BBBBBBPPPPPPBBBBPPPTTTBPB"),  # long streak and ties
            list("BBPPPTTBBPBPTPPBBTP"),  # EV-sensitive synthetic sample
        ]
        script = (
            "global.self=global;"
            "const aux=require('./bbb_engine/derived_road_aux.js');"
            "const core=require('./bbb_engine/soft_shoe_decision_runtime.js');"
            "require('./bbb_engine/soft_shoe_model_bundle.js');"
            "const frozen=require('./bbb_engine/soft_shoe_frozen_lstm_runtime.js');"
            "require('./bbb_engine/soft_shoe_frozen_model_bundle.js');"
            "const histories=JSON.parse(process.argv[1]);"
            "process.stdout.write(JSON.stringify(histories.map(h=>core.run(h,{},"
            "global.BBB_SOFT_SHOE_XGB_MODEL,frozen,global.BBB_SOFT_SHOE_LSTM_MODEL))));"
        )
        direct = subprocess.run(
            _node_command() + ["-e", script, json.dumps(cases)],
            cwd=ROOT, capture_output=True, text=True,
            check=True, timeout=20,
        )
        expected = json.loads(direct.stdout)
        for history, original in zip(cases, expected):
            line = soft.analyze_history(history)
            # Structural equality includes future_path, adaptive_sampling,
            # EV, confidence and the original research gate, not just VERSION.
            self.assertEqual(line, original)
            self.assertEqual(line["version"], soft.VERSION)
            cfg = line["config"]
            self.assertEqual(
                (cfg["PARTICLE_COUNT"], cfg["LIKELIHOOD_DRAWS"],
                 cfg["NEXT_HAND_DRAWS"]), (448, 80, 128)
            )
            self.assertIs(cfg["ROAD_AUX_ENABLED"], True)
            road = line["road_aux"]
            self.assertIs(road["estimated_only"], True)
            self.assertIs(road["validated_betting_edge"], False)
            self.assertEqual(road, original["road_aux"])
            self.assertEqual(line.get("research_view"), original.get("research_view"))
            self.assertEqual(line.get("future_path"), original.get("future_path"))
            self.assertEqual(line.get("adaptive_sampling"), original.get("adaptive_sampling"))
            self.assertEqual(line["action"], "Skip")
            self.assertEqual(line["bankroll_fraction"], 0)
            self.assertIs(line["validated_edge"], False)
            if line.get("probabilities"):
                for k in ("B", "P", "T"):
                    self.assertAlmostEqual(line["probabilities"][k],
                                           original["probabilities"][k], places=12)
                self.assertAlmostEqual(line["ev_banker"], original["ev_banker"], places=12)
                self.assertAlmostEqual(line["ev_player"], original["ev_player"], places=12)
        self.assertTrue(expected[2]["road_aux"]["available"])
        self.assertGreaterEqual(expected[2]["road_aux"]["big_road_depth"], 12)

    def test_adaptive_switch_is_research_only_and_skip_is_unchanged(self):
        history = list("BBBBBBPPPPPP") * 4
        script = (
            "global.self=global;"
            "const core=require('./bbb_engine/soft_shoe_decision_runtime.js');"
            "const h=JSON.parse(process.argv[1]);"
            "const adaptive=core.run(h);"
            "const fixed=core.run(h,{ROAD_AUX_ADAPTIVE_ENABLED:false});"
            "process.stdout.write(JSON.stringify({adaptive,fixed}));"
        )
        direct = subprocess.run(
            _node_command() + ["-e", script, json.dumps(history)],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=12.0
        )
        pair = json.loads(direct.stdout)
        adaptive, fixed = pair["adaptive"], pair["fixed"]
        self.assertIs(adaptive["config"]["ROAD_AUX_ADAPTIVE_ENABLED"], True)
        self.assertIs(fixed["config"]["ROAD_AUX_ADAPTIVE_ENABLED"], False)
        self.assertEqual(fixed["road_aux"]["adaptive"]["mode"], "fixed")
        for key in ("probabilities", "physical_probabilities", "candidate",
                    "action", "wager_gate", "bankroll_fraction",
                    "validated_edge"):
            self.assertEqual(adaptive[key], fixed[key], key)
        self.assertEqual(adaptive["action"], "Skip")
        self.assertEqual(adaptive["bankroll_fraction"], 0)
        self.assertIs(adaptive["validated_edge"], False)
        self.assertLessEqual(adaptive["road_aux"]["adaptive"]["w_applied"], .08)

    def test_road_aux_disabled_preserves_soft_physics_and_wager(self):
        history = list("BBPPPTTBBPBPTPPBBTP")
        script = (
            "global.self=global;"
            "const core=require('./bbb_engine/soft_shoe_decision_runtime.js');"
            "const h=JSON.parse(process.argv[1]);"
            "const on=core.run(h),off=core.run(h,{ROAD_AUX_ENABLED:false});"
            "process.stdout.write(JSON.stringify({on,off}));"
        )
        direct = subprocess.run(
            _node_command() + ["-e", script, json.dumps(history)],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=12,
        )
        both = json.loads(direct.stdout)
        on, off = both["on"], both["off"]
        self.assertIs(on["config"]["ROAD_AUX_ENABLED"], True)
        self.assertIs(off["config"]["ROAD_AUX_ENABLED"], False)
        self.assertEqual(off["road_aux"]["reason"], "disabled")
        self.assertEqual(off["road_aux"]["adaptive"]["mode"], "off")
        for field in (
            "version", "candidate", "probabilities",
            "physical_probabilities", "ev_banker", "ev_player",
            "ev_tie", "particle_ess", "ancestor_diversity", "features48",
            "adaptive_sampling", "wager_gate", "action",
            "bankroll_fraction", "validated_edge",
        ):
            self.assertEqual(on[field], off[field], field)
        self.assertEqual(on["research_view"]["direction"],
                         off["research_view"]["direction"])
        self.assertEqual(on["research_view"]["eligible"],
                         off["research_view"]["eligible"])

    def test_line_adapter_exposes_road_without_authorizing_bets(self):
        history = list("BP" * 12)
        first = soft.predict_screen(history, user_id="ROAD-UID-A")
        second = soft.predict_screen(history, user_id="ROAD-UID-B")
        self.assertEqual(first["bbb_soft_shoe"]["report"],
                         second["bbb_soft_shoe"]["report"])
        report = first["bbb_soft_shoe"]["report"]
        self.assertEqual(first["research_view"], report["research_view"])
        self.assertEqual(first["future_path"], report["future_path"])
        self.assertEqual(first["adaptive_sampling"], report["adaptive_sampling"])
        self.assertEqual(first["road_aux"], report["road_aux"])
        self.assertEqual(first["road_summary"]["road_side"],
                         report["road_aux"]["road_side"])
        self.assertEqual(first["road_summary"]["agreement"],
                         report["road_aux"]["road_agreement"])
        self.assertIs(first["road_confirmed"],
                      report["research_view"]["road_confirmed"])
        self.assertIs(first["road_conflict"],
                      report["research_view"]["road_conflict"])
        self.assertEqual(first["research_direction"],
                         report["research_view"]["direction"])
        self.assertFalse(first["bet_allowed"])
        self.assertIs(first["official_bet_allowed"], False)
        self.assertIs(first["bbb_soft_shoe"]["official_bet_allowed"], False)
        self.assertFalse(first["money_management"]["bet_allowed"])
        self.assertEqual(first["bet_amount"], 0)
        self.assertIn("非保證勝率", first["road_signal_note"])

    def test_full_shoe_worker_uses_same_bounded_eight_second_transport(self):
        # Synthetic-only latency/integration check: not real-shoe hit-rate evidence.
        history = list("BPPBTBBPPT") * 7
        started = time.perf_counter()
        result = soft.analyze_history(history)
        elapsed = time.perf_counter() - started
        self.assertEqual(result["observations"], len(history))
        self.assertEqual(result["config"]["PARTICLE_COUNT"], 448)
        self.assertEqual(result["config"]["LIKELIHOOD_DRAWS"], 80)
        self.assertEqual(result["config"]["NEXT_HAND_DRAWS"], 128)
        self.assertLess(elapsed, soft.TIMEOUT_SECONDS)
        print(f"BGS_SYNTHETIC_70_HAND_WORKER_SECONDS={elapsed:.3f}")

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
            if actual.get("wager_gate"):
                self.assertEqual(actual["wager_gate"]["model_verified"], False)
                self.assertEqual(actual["wager_gate"]["eligible"], False)
                self.assertEqual(actual["wager_gate"]["max_ev_se"], .060)
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

    def test_v5_1_frozen_sequence_and_xgb_schema_fail_closed(self):
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

    def test_dynamic_wager_policy_exact_boundaries_and_fail_closed(self):
        # A direct V5.1 policy test. Does not change the OCR/Flex/session path.
        source = (
            "const s=require('./bbb_engine/soft_shoe_decision_runtime.js');"
            "const c=s.DEFAULT;"
            "const base={cfg:c,remainingCards:416*.7,ess:230,quality:.82,"
            "ev:{B:.05,P:.01,T:0},candidate:'B',confidence:.90,se:.02,"
            "lower:.01,enoughDepth:true,observations:10,model:'validated_external_xgb'};"
            "const gate=x=>s.wagerGate(Object.assign({},base,x));"
            "const out={tiers:[.61,.60,.30,.2999].map(p=>s.wagerPolicy(416*p,c)),"
            "essRejected:gate({ess:.45*448-.001}).reasons.includes('low_particle_ess'),"
            "lateConfidenceRejected:gate({remainingCards:416*.34,confidence:.79})"
            ".reasons.includes('confidence_below_dynamic_gate'),"
            "tinyGapRejected:gate({ev:{B:.05,P:.049,T:0}})"
            ".reasons.includes('ev_gap_too_small'),"
            "endShoeRejected:gate({remainingCards:20,confidence:.85})"
            ".reasons.includes('shoe_end_no_bet'),"
            "endShoeException:gate({remainingCards:20,ev:{B:.025,P:0,T:0},"
            "confidence:.9,se:.03,lower:.01}).eligible,"
            "untrainedRejected:gate({model:'physics_only_untrained_decision_fallback'}).eligible};"
            "process.stdout.write(JSON.stringify(out));"
        )
        completed = subprocess.run(
            _node_command()+["-e",source],cwd=ROOT,capture_output=True,
            text=True,check=True,timeout=8
        )
        result=json.loads(completed.stdout)
        self.assertEqual([r["min_ev_threshold"] for r in result["tiers"]],
                         [.003,.00235,.00235,.0019])
        self.assertEqual([r["min_confidence"] for r in result["tiers"]],
                         [.75,.75,.80,.80])
        self.assertTrue(result["essRejected"])
        self.assertTrue(result["lateConfidenceRejected"])
        self.assertTrue(result["tinyGapRejected"])
        self.assertTrue(result["endShoeRejected"])
        self.assertTrue(result["endShoeException"])
        self.assertFalse(result["untrainedRejected"])

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
