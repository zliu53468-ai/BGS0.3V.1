"""Transport-only Python bridge for BBB V2.6 final-score independent mode.

Physics/particle modules are byte-identical to the matching BBB N96/L256/M96 deployment.\nOther vendored BBB V2.6 modules retain their pinned original Git objects.
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
SOURCE_COMMIT = "a9e25b42d003f1a721007d6ae68127082ab86a29"
VERSION = "BBB_PHYSICS_FORECAST_FIRST_V2_DYNAMIC_GATE"
SOURCE_BLOBS = {
    "quant256_online_runtime.js": "9b8b6fef7037bd64ded026aeff5bab6d547f0a76",
    "shoe_distribution_guard.js": "b7c34f045952e9ffe7883b670e0314565c557890",
    "markov_residual_runtime.js": "8fbee10200b067c5ef1ac4632e63b2ec18da2ab8",
    "physics_forecast_first_runtime.js": "a2a1263b9844092c986ad0eed93568a7615fbcec",
    "point_signal_quality_guard.js": "f39b94ebb14f493afc402e2f32b116b551c27a2a",
}
TIMEOUT_SECONDS = 9.0
_LOCK = threading.RLock()
_PROCESS: subprocess.Popen[bytes] | None = None
_BUFFER = b""

def verify_source() -> None:
    for name, expected in SOURCE_BLOBS.items():
        payload = (ENGINE / "v26" / name).read_bytes()
        digest = sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if digest != expected:
            raise RuntimeError("BBB V2.6 source mismatch: " + name)

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
            raise RuntimeError("BBB V2.6 worker unavailable")
        try:
            request = json.dumps({"player": player, "banker": banker}, separators=(",", ":"))
            proc.stdin.write((request + "\n").encode())
            proc.stdin.flush()
            deadline = time.monotonic() + TIMEOUT_SECONDS
            while b"\n" not in _BUFFER:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("BBB V2.6 single-score calculation timeout")
                readable, _, _ = select.select([proc.stdout], [], [], remaining)
                if not readable:
                    raise TimeoutError("BBB V2.6 single-score calculation timeout")
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("BBB V2.6 worker closed its stdout")
                _BUFFER += chunk
                if len(_BUFFER) > 1_000_000:
                    raise RuntimeError("BBB V2.6 response too large")
            line, _BUFFER = _BUFFER.split(b"\n", 1)
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError("BBB V2.6: " + str(response.get("error", "unknown")))
            result = response["result"]
            if (result.get("source_commit") != SOURCE_COMMIT or
                    result.get("version") != VERSION):
                raise RuntimeError("BBB V2.6 source/version mismatch")
            if result.get("config") != {
                "N": 96, "L": 256, "M": 96,
                "ess_resampling_ratio": .5, "physics_weight": 5, "road_weight": .1,
                "independentFinalScoreMode": True,
            }:
                raise RuntimeError("BBB V2.6 N/L/M contract mismatch")
            if (result.get("history_length") != 1 or
                    result.get("point_score_observation_count") != 1 or
                    result.get("bet_allowed") is not False or
                    result.get("action") != "SKIP"):
                raise RuntimeError("BBB V2.6 single-score contract mismatch")
            return result
        except Exception:
            _drop_worker()
            raise
