"""Isolated BGS-to-BBB V2 prediction bridge.

Runs *unchanged* BBB PhysicsForecastFirst JS modules at source commit 7f49802f.
No access to LINE, license, trials, OCR or persisted member session data.
"""
from __future__ import annotations

from hashlib import sha1, sha256
from pathlib import Path
import json
import os
import select
import subprocess
import threading
import time
from typing import Any, Iterable

from bbb_final57_runtime import _node_command

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "bbb_engine"
SOURCE_COMMIT = "7f49802f6f514eb7bf44aa3a61bdf4b1ffcc9920"
VERSION = "BBB_PHYSICS_FORECAST_FIRST_V2_DYNAMIC_GATE"
SOURCE_BLOBS = {
    "quant256_online_runtime.js": "4370e242156ceb408530ab8424fedf5f07b9489a",
    "physics_forecast_first_runtime.js": "04103fda59fe1b8b658a5ebd23ec0e9f736dbf8b",
    "shoe_distribution_guard.js": "b7c34f045952e9ffe7883b670e0314565c557890",
}
_LOCK = threading.RLock()
_PROCESS: subprocess.Popen[bytes] | None = None
_BUFFER = b""
# The LINE image pipeline has a separate 12-second bound; keep a safe margin.
TIMEOUT_SECONDS = 9.0


def verify_source() -> None:
    """Compare Git object hashes against the BBB merge's exact source blobs."""
    for name, expected in SOURCE_BLOBS.items():
        payload = (ENGINE / name).read_bytes()
        git_blob = sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if git_blob != expected:
            raise RuntimeError("BBB V2 source blob hash mismatch: " + name)


def _drop_process() -> None:
    global _PROCESS, _BUFFER
    proc, _PROCESS, _BUFFER = _PROCESS, None, b""
    if proc is not None:
        try:
            proc.kill()
            proc.wait(timeout=1)
        except (OSError, subprocess.SubprocessError):
            pass


def _request(history: list[str], scope: str) -> dict[str, Any]:
    global _PROCESS, _BUFFER
    with _LOCK:
        if _PROCESS is None or _PROCESS.poll() is not None:
            _drop_process()
            verify_source()
            _PROCESS = subprocess.Popen(
                _node_command() + [str(ENGINE / "worker_v2.cjs")],
                cwd=str(ROOT),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0,
            )
        proc = _PROCESS
        if proc.stdin is None or proc.stdout is None:
            _drop_process()
            raise RuntimeError("BBB V2 worker pipes unavailable")
        try:
            payload = json.dumps({"history": history, "scope": scope},
                                 ensure_ascii=True, separators=(",", ":"))
            proc.stdin.write((payload + "\n").encode("utf-8"))
            proc.stdin.flush()
            deadline = time.monotonic() + TIMEOUT_SECONDS
            while b"\n" not in _BUFFER:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("BBB V2 prediction exceeded LINE time budget")
                readable, _, _ = select.select([proc.stdout], [], [], remaining)
                if not readable:
                    raise TimeoutError("BBB V2 prediction exceeded LINE time budget")
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("BBB V2 worker exited")
                _BUFFER += chunk
                if len(_BUFFER) > 1_000_000:
                    raise RuntimeError("BBB V2 response is too large")
            raw, _BUFFER = _BUFFER.split(b"\n", 1)
            response = json.loads(raw)
            if not response.get("ok"):
                raise RuntimeError("BBB V2: " + str(response.get("error", "unknown")))
            result = response["result"]
            if result.get("source_commit") != SOURCE_COMMIT or result.get("version") != VERSION:
                raise RuntimeError("BBB V2 worker/version mismatch")
            if (result.get("config") or {}) != {
                "N": 256, "L": 384, "M": 160, "ess_resampling_ratio": .5,
                "physics_weight": 5, "road_weight": .1
            }:
                raise RuntimeError("BBB V2 sampling configuration mismatch")
            return result
        except Exception:
            # Fail closed. Never silently fallback to the older Final57 policy.
            _drop_process()
            raise


def predict_v2(history: Iterable[str], *, scope_key: str = "") -> dict[str, Any]:
    seq = [str(x).strip().upper() for x in history]
    if any(x not in ("B", "P", "T") for x in seq):
        raise ValueError("invalid history")
    # Match the browser's 500-round maximum and keep UID/scope out of worker logs.
    seq = seq[-500:]
    scope = sha256((str(scope_key) or "anonymous").encode("utf-8")).hexdigest()
    return _request(seq, scope)
