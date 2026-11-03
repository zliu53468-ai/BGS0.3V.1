"""BBB exact V2 PF256 smoke/parity contract for BGS LINE (prediction only)."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import subprocess
import unittest
from typing import Any, Mapping

import bbb_v2_runtime as bridge
from bbb_v2_overlay import overlay_v2

ROOT = Path(__file__).resolve().parent
H = list("BPPBT")


class BBBV2LINEBridgeTests(unittest.TestCase):
    def test_original_js_blobs_unmodified(self):
        bridge.verify_source()
        self.assertEqual(bridge.SOURCE_COMMIT, "7f49802f6f514eb7bf44aa3a61bdf4b1ffcc9920")

    def test_identical_sampling_and_probability_contract(self):
        r = bridge.predict_v2(H[:1], scope_key="unit/user1/room1/shoe1")
        self.assertEqual(r["config"], {
            "N": 256, "L": 384, "M": 160,
            "ess_resampling_ratio": .5, "physics_weight": 5, "road_weight": .1,
        })
        self.assertEqual(r["version"], bridge.VERSION)
        self.assertEqual(r["action"], "SKIP")
        self.assertFalse(r["bet_allowed"])
        self.assertTrue(r["proxy_only"])
        self.assertFalse(r["validated_edge"])
        self.assertEqual(r["physical_ev"]["samples"], 40960)
        probs = r["physical_ev"]["simulated_probability"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=12)
        self.assertEqual(r["research_signal"]["direction"], r["display_direction"])

    def test_incremental_idempotent_scope_isolation_and_undo(self):
        a = bridge.predict_v2(H[:3], scope_key="user-A/table-A/shoe-A")
        again = bridge.predict_v2(H[:3], scope_key="user-A/table-A/shoe-A")
        self.assertEqual(a, again)
        b = bridge.predict_v2(H[:3], scope_key="user-B/table-A/shoe-A")
        self.assertEqual(a, b)
        moved = bridge.predict_v2(H, scope_key="user-A/table-A/shoe-A")
        self.assertEqual(moved["history_length"], len(H))
        rewind = bridge.predict_v2(H[:3], scope_key="user-A/table-A/shoe-A")
        self.assertEqual(rewind, a)
        self.assertEqual(bridge.predict_v2(H, scope_key="user-A/table-A/shoe-A"), moved)

    def test_research_display_always_zero_stake(self):
        baseline = {
            "probabilities": {"B": .5, "P": .5, "T": 0.0},
            "money_management": {"final_bet_ratio": .30, "bet_amount": 3000},
            "component_probabilities": {}
        }
        got = overlay_v2(
            baseline, history=H[:2], core_policy={"scope_key": "ci/overlay/1"},
            bankroll=10000
        )
        self.assertEqual(got["engine"], "BBB_PHYSICS_FORECAST_FIRST_V2")
        self.assertEqual(got["formal_action"], "Skip")
        self.assertIn(got["action"], ("B", "P", "Skip"))
        self.assertEqual(got["bet_amount"], 0)
        self.assertEqual(got["bet_percentage"], 0)
        self.assertFalse(got["bet_allowed"])
        self.assertFalse(got["money_management"]["bet_allowed"])
        self.assertTrue(got["bbb_v2"]["proxy_only"])
        self.assertFalse(got["bbb_v2"]["validated_edge"])
        # The read-only probability display includes ties; it isn't a win rate.
        self.assertAlmostEqual(sum(got["probabilities"].values()), 1.0, places=12)

    def test_runtime_boundary_never_converts_v2_to_bet(self):
        tree = ast.parse((ROOT / "runtime_app.py").read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ("_formal_bp_action", "_attach_formal_bankroll_advice")]
        env = {
            "Mapping": Mapping, "Any": Any,
            "MIN_FORMAL_BET_RATIO": .05, "MAX_FORMAL_BET_RATIO": .30
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "runtime_boundary", "exec"), env)
        attach = env["_attach_formal_bankroll_advice"]
        for signal, direction in ((True, "B"), (False, "P")):
            result = attach({
                "action": direction, "bet_allowed": True, "bet_amount": 3000,
                "bbb_v2": {
                    "active": True, "action": "SKIP",
                    "official_bet_allowed": False, "display_direction": direction,
                    "research_signal": {"available": signal}
                }
            }, {"bankroll": 10000, "permanent_access": True, "trial_started_at": 123})
            self.assertEqual(result["action"], direction if signal else "Skip")
            self.assertEqual(result["bet_amount"], 0)
            self.assertEqual(result["bet_percentage"], 0)
            self.assertFalse(result["bet_allowed"])
            self.assertFalse(result["money_management"]["bet_allowed"])

    def test_protected_service_files_match_original_manifest(self):
        import json
        manifest = json.loads((ROOT / "bbb_engine/manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["protected_files_sha256"].items():
            payload = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest, name)


if __name__ == "__main__":
    unittest.main()
