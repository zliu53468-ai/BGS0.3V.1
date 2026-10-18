from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from typing import Mapping
from unittest.mock import patch

import bbb_final57_runtime as runtime

ROOT = Path(__file__).resolve().parent


class BBBMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(runtime._node_command()+[str(ROOT/"tests/bbb_reference.cjs")], capture_output=True,
                                text=True, encoding="utf-8", check=True, timeout=20)
        cls.reference = json.loads(result.stdout)

    def assertParity(self, actual, expected):
        for key in ("direction", "confidence", "final_p_b", "raw_p_b", "ev_banker", "ev_player",
                    "smoothing_alpha", "execution_order", "physics_48d", "features_57d"):
            self.assertEqual(actual[key], expected[key], key)
        self.assertEqual(actual["particle_diagnostics"]["particle_count"], 500)
        self.assertEqual(actual["source_commit"], self.reference["source_commit"])

    def test_browser_parity_cold_and_late_shoes(self):
        for row in self.reference["standalone"]:
            with self.subTest(rounds=len(row["history"])):
                actual = runtime.apply_final_prediction(row["history"], {})
                self.assertParity(actual, row["expected"])

    def test_scope_isolation_ema_undo_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"BGS_BBB_STATE_FILE": str(Path(tmp)/"predictions.db")}):
            for row in self.reference["sequence"]:
                actual = runtime.apply_final_prediction(row["history"], {"scope_key": "alice/DG/1/shoe1"})
                self.assertParity(actual, row["expected"])
                # Same-history retry must not settle or double-smooth a prediction.
                self.assertParity(runtime.apply_final_prediction(row["history"], {"scope_key": "alice/DG/1/shoe1"}), row["expected"])
            base = self.reference["sequence"][0]
            # No user, room or shoe can read the other scope's settled predictions.
            for scope in ("bob/DG/1/shoe1", "alice/DG/2/shoe1", "alice/DG/1/shoe2"):
                self.assertParity(runtime.apply_final_prediction(base["history"], {"scope_key": scope}), base["expected"])
            # Rewind truncates later settled rows; return to the earlier prediction.
            self.assertParity(runtime.apply_final_prediction(base["history"], {"scope_key": "alice/DG/1/shoe1"}), base["expected"])
            # Disk state survives a new worker/process invocation (each call is new).
            next_row = self.reference["sequence"][1]
            self.assertParity(runtime.apply_final_prediction(next_row["history"], {"scope_key": "alice/DG/1/shoe1"}), next_row["expected"])

    def test_auth_ocr_line_static_and_deployment_unchanged(self):
        manifest = json.loads((ROOT/"bbb_engine/manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["protected_files_sha256"].items():
            with self.subTest(file=name):
                content = (ROOT/name).read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected)
        self.assertGreater(len(manifest["protected_files_sha256"]), 60)

    def test_skip_survives_real_runtime_bankroll_boundary(self):
        # Compile ONLY these prediction functions; no app/LINE/OCR initialization.
        tree = ast.parse((ROOT/"runtime_app.py").read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 ("_formal_bp_action", "_attach_formal_bankroll_advice")]
        namespace = {"Mapping": Mapping, "Any": object, "MIN_FORMAL_BET_RATIO": .05, "MAX_FORMAL_BET_RATIO": .30}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "runtime_prediction_boundary", "exec"), namespace)
        attach = namespace["_attach_formal_bankroll_advice"]
        skipped = attach({"action": "Skip", "bbb_final57": {"active": True, "direction": "Skip"}}, {"bankroll": 10000})
        self.assertEqual(skipped["action"], "Skip")
        self.assertFalse(skipped["bet_allowed"])
        self.assertEqual(skipped["bet_amount"], 0)
        for direction in ("B", "P"):
            active = attach({"action": direction, "final_bet_ratio": .10}, {"bankroll": 10000})
            self.assertEqual(active["action"], direction)
            self.assertEqual(active["bet_amount"], 1000)

    def test_failure_never_falls_back_to_old_direction(self):
        with patch.object(runtime, "_run", side_effect=RuntimeError("engine unavailable")):
            with self.assertRaisesRegex(RuntimeError, "engine unavailable"):
                runtime.apply_final_prediction("BPPBT", {"direction": "B"})


if __name__ == "__main__":
    unittest.main()
