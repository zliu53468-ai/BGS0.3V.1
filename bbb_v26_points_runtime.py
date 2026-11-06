"""Transport-only Python bridge for the current BBB V4 independent-score engine.

All eight v4 modules are byte-identical to SOURCE_COMMIT in BBB.
The historical module name remains for LINE import compatibility.
BGS user/session state and LINE credentials never enter the Node predictor.
"""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import json
import os
import select
import subprocess
import threading
import time
from typing import Any

from bbb_final57_runtime import _node_command

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "bbb_engine"
SOURCE_COMMIT = "aec8124dd01d98b2dc2a4823a5304823a6f9d06b"
VERSION = "BBB_V41_ADAPTIVE_BATCH_STABILITY_V1"
SOURCE_BLOBS = {
    "quant256_online_runtime.js": "ed9e076f6a91d6a8068ee4d1909bacde287b2595",
    "shoe_distribution_guard.js": "b7c34f045952e9ffe7883b670e0314565c557890",
    "markov_residual_runtime.js": "8fbee10200b067c5ef1ac4632e63b2ec18da2ab8",
    "physics_forecast_first_runtime.js": "f53989bc0df86c4c1d7cf8c5161c63c146d7e5b3",
    "deplete_mc_runtime.js": "096b6a537105f3f80ec6d38f855714595deaf048",
    "pf_eor_v4_runtime.js": "52b2d2726e1ebc340046629d318ae8862fb7b7be",
    "stability_reference.js": "8b0db63f12560984440646ebdc5e7fe70a63ae08",
    "stability_runtime.js": "01fdec104e3c02961c231e1448b05614c14c7672"
}
TIMEOUT_SECONDS = 15.0
_LOCK = threading.RLock()
_PROCESS: subprocess.Popen[bytes] | None = None
_BUFFER = b""

def verify_source() -> None:
    for name, expected in SOURCE_BLOBS.items():
        payload = (ENGINE / "v4" / name).read_bytes()
        digest = sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if digest != expected:
            raise RuntimeError("BBB V4 source mismatch: " + name)

def _drop_worker() -> None:
    global _PROCESS, _BUFFER
    proc, _PROCESS, _BUFFER = _PROCESS, None, b""
    if proc is not None:
        try:
            proc.kill()
            proc.wait(timeout=1)
        except (OSError, subprocess.SubprocessError):
            pass

def predict_final_score(player: int, banker: int) -> dict[str, Any]:
    """The entire input to the BBB engine is exactly one resolved score pair."""
    if type(player) is not int or type(banker) is not int or not (
        0 <= player <= 9 and 0 <= banker <= 9
    ):
        raise ValueError("請輸入閒、莊各 0 到 9 點")
    global _PROCESS, _BUFFER
    with _LOCK:
        if _PROCESS is None or _PROCESS.poll() is not None:
            _drop_worker()
            verify_source()
            _PROCESS = subprocess.Popen(
                _node_command() + [str(ENGINE / "worker_v26.cjs")],
                cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0,
            )
        proc = _PROCESS
        if proc.stdin is None or proc.stdout is None:
            _drop_worker()
            raise RuntimeError("BBB V4 worker unavailable")
        try:
            request = json.dumps({"player": player, "banker": banker}, separators=(",", ":"))
            proc.stdin.write((request + "\n").encode())
            proc.stdin.flush()
            deadline = time.monotonic() + TIMEOUT_SECONDS
            while b"\n" not in _BUFFER:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("BBB V4 single-score calculation timeout")
                readable, _, _ = select.select([proc.stdout], [], [], remaining)
                if not readable:
                    raise TimeoutError("BBB V4 single-score calculation timeout")
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("BBB V4 worker closed its stdout")
                _BUFFER += chunk
                if len(_BUFFER) > 1_000_000:
                    raise RuntimeError("BBB V4 response too large")
            line, _BUFFER = _BUFFER.split(b"\n", 1)
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError("BBB V4: " + str(response.get("error", "unknown")))
            result = response["result"]
            if (result.get("source_commit") != SOURCE_COMMIT or
                    result.get("version") != VERSION):
                raise RuntimeError("BBB V4 source/version mismatch")
            if result.get("config") != {
                "particles": 192, "simulationsPerParticle": 512,
                "eorParticleCount": 24, "eorSamples": 96,
                "minEvGap": .004, "noiseGuardMultiplier": .65,
                "minPosteriorQuality": .5, "eorFragilityFloor": .025,
                "crossCheckZ": 2.5, "crossCheckFloor": .008,
                "crossCheckDirectionFloor": .005,
            }:
                raise RuntimeError("BBB V4 configuration mismatch")
            if (result.get("v4", {}).get("direction") != result.get("direction") or
                    result.get("v4", {}).get("user_entry_policy") != {
                        "minEstimatorSE": .0055, "minAbsoluteEVGap": 0,
                        "minBatchAgreement": .55,
                    }):
                raise RuntimeError("BBB V4 entry policy mismatch")
            if (result.get("history_length") != 1 or
                    result.get("point_score_observation_count") != 1 or
                    result.get("bet_allowed") is not False or
                    result.get("action") != "SKIP"):
                raise RuntimeError("BBB V4 single-score contract mismatch")
            return result
        except Exception:
            _drop_worker()
            raise
