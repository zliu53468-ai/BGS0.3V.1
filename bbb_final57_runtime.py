"""BGS prediction-only adapter to the pinned, unmodified BBB JavaScript engine.

No LINE, activation, session, OCR or screenshot code is implemented here.
"""
from __future__ import annotations
import hashlib
from contextlib import closing
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ENGINE_DIR = BASE_DIR / "bbb_engine"
PHYSICS_MODEL_PATH = BASE_DIR / "physics_multitask_model.json"
FINAL_MODEL_PATH = BASE_DIR / "final_probability_model.json"
VERSION = "PHYSICS_57D_V14_8_MACRO_EMA_GUARD"
HISTORY_INPUT_DIM, PHYSICS_DIM, FEATURE_DIM = 213, 48, 57
_LOCK = threading.RLock()


def normalize_history(history: str | Iterable[Any] | None) -> list[str]:
    result = []
    for item in history or []:
        if isinstance(item, Mapping):
            item = item.get("outcome") or item.get("actual") or item.get("actual_outcome")
        token = str(item or "").upper().strip()
        if token in ("B", "P", "T"):
            result.append(token)
    return result[-2000:]


@lru_cache(maxsize=1)
def _verify_source() -> dict:
    manifest = json.loads((ENGINE_DIR / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        target = BASE_DIR / name if name.endswith(".json") else ENGINE_DIR / name
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError("BBB prediction source checksum mismatch: " + name)
    return manifest


def _node_command() -> list[str]:
    if importlib.util.find_spec("nodejs_wheel") is not None:
        return [sys.executable, "-m", "nodejs_wheel"]
    node = shutil.which("node")
    if node:
        return [node]
    raise RuntimeError("BBB prediction runtime missing: install nodejs-wheel==24.19.0")


def _run(payload: Mapping[str, Any]) -> dict:
    _verify_source()
    try:
        completed = subprocess.run(
            _node_command() + [str(ENGINE_DIR / "worker.cjs")],
            input=json.dumps(payload, ensure_ascii=True, allow_nan=False),
            capture_output=True, text=True, encoding="utf-8", timeout=2.8,
            cwd=str(BASE_DIR), check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("BBB prediction timed out") from error
    if completed.returncode:
        raise RuntimeError("BBB prediction failed: " + completed.stderr.strip()[:300])
    return json.loads(completed.stdout)


def _state_path() -> Path:
    explicit = os.environ.get("BGS_BBB_STATE_FILE")
    if explicit:
        return Path(explicit)
    session_path = Path(os.environ.get("SESSION_DATA_FILE", str(BASE_DIR / "sessions.json")))
    return session_path.parent / "bbb_prediction_state.sqlite3"


def _predict_scoped(seq: list[str], scope: str) -> dict:
    if not scope:
        return _run({"history": seq})
    # Separate prediction-only DB. Never reads/writes license or session contents.
    key = hashlib.sha256(scope.encode("utf-8")).hexdigest()
    state_path = _state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, closing(sqlite3.connect(state_path, timeout=5)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS bbb_state (scope TEXT PRIMARY KEY, payload TEXT NOT NULL, touched REAL NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT payload FROM bbb_state WHERE scope=?", (key,)).fetchone()
        old = json.loads(row[0]) if row else {}
        out = _run({"history": seq, "previous_history": old.get("history", []), "storage": old.get("storage", {})})
        state = json.dumps({"history": out["history"], "storage": out["storage"]}, allow_nan=False)
        db.execute("INSERT OR REPLACE INTO bbb_state VALUES (?, ?, julianday('now'))", (key, state))
        db.execute("DELETE FROM bbb_state WHERE scope IN (SELECT scope FROM bbb_state ORDER BY touched DESC LIMIT -1 OFFSET 256)")
        return out


def apply_final_prediction(history, core_prediction=None, *, previous_final_pb=None, estimated_total_hands=60.0):
    # Core/EMA arithmetic comes from BBB itself. The old Python core is not a
    # substitute for the vendored JS Frozen Core; only its scope identity is used.
    del previous_final_pb, estimated_total_hands
    core_prediction = dict(core_prediction or {})
    out = _predict_scoped(normalize_history(history), str(core_prediction.get("scope_key") or ""))
    p, r = out["prediction"], out["prediction"]["finalProbability"]
    direction = p["direction"]
    probability = p["probabilities"][direction] if direction in ("B", "P") else max(p["probabilities"].values())
    return {
        "version": r["version"], "active": r["active"], "mode": r["mode"],
        "direction": direction, "final_direction": p["final_direction"],
        "core_direction": r["coreDirection"], "flipped": r["flipped_from_physics"],
        "confidence": p["confidence"], "direction_probability": probability,
        "probabilities": p["probabilities"], "core_p_b": r["core_aux_p_b"],
        "raw_p_b": r["rawPB"], "clipped_p_b": r["clippedPB"],
        "smoothed_p_b": r["smoothedPB"], "final_p_b": r["finalPB"],
        "p_tie": r["p_tie"], "p_player": r["p_player"],
        "ev_banker": p["ev_banker"], "ev_player": p["ev_player"],
        "min_ev": r["min_ev"], "activation_ev": r["activation_ev"],
        "soft_band": r["soft_band"], "decision_policy_enabled": r["decision_policy_enabled"],
        "probability_bounds": {"min": r["bounds"]["low"], "max": r["bounds"]["high"]},
        "smoothing_alpha": r["smoothingAlpha"], "smoothing_strength": r["smoothingStrength"], "smoothing_profile": r["smoothingProfile"],
        "shoe_progress_weight": r["extended"][1], "physics_noise_score": r["extended"][-1],
        "original7": r["original7"], "physics_48d": r["physics"], "features_57d": r["extended"],
        "physics_forecast": r["physicsForecast"], "particle_diagnostics": r["particleDiagnostics"],
        "execution_order": r["execution_order"], "source_commit": out["source_commit"],
        "bbb_runtime": r,
    }


def load_models(force=False):
    if force:
        _verify_source.cache_clear()
    _verify_source()
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in (PHYSICS_MODEL_PATH, FINAL_MODEL_PATH))


def model_status():
    out = _run({"operation": "status"})
    return {"version": out["version"], "physics": out["status"]["physics"], "final57": out["status"]["final56"], "feature_dim": FEATURE_DIM}


def history_vector(history):
    return np.asarray(_run({"operation": "history_vector", "history": normalize_history(history)})["value"], dtype=np.float64)


def predict_physics(history, bundle=None):
    if bundle is not None and bundle != load_models()[0]:
        raise ValueError("BBB uses the pinned Physics bundle")
    return np.asarray(_run({"operation": "physics", "history": normalize_history(history)})["value"], dtype=np.float64)


def build_original7(history, core_prediction, estimated_total_hands=60.0):
    del estimated_total_hands
    return _run({"operation": "original7", "history": normalize_history(history), "core": dict(core_prediction)})["value"]


def build_features(core_pb, original7, physics):
    return np.asarray(_run({"operation": "features", "core_pb": float(core_pb), "original7": dict(original7), "physics": list(map(float, physics))})["value"], dtype=np.float64)


def physics_noise_score(physics, round_index=70.0):
    return _run({"operation": "noise", "physics": list(map(float, physics)), "round_index": float(round_index)})["value"]


def predict_raw_probability(bundle, vector):
    return _run({"operation": "raw_probability", "bundle": dict(bundle), "features": list(map(float, vector))})["value"]
