"""Bounded exact BBB Soft Shoe (B/P/T-only) adapter for LINE OCR / manual road results.

The imported BBB JavaScript engine and model bundle are immutable, matched to
the web release by their Git blob SHAs. This adapter keeps all LINE UIDs out of
the JS engine: the full current shoe's B/P/T history is passed per request.
"""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from bbb_final57_runtime import _node_command

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "bbb_engine"
SOURCE_COMMIT = "2be54a70982f7cb3d9e5d6e55bb6636764283077"
SOURCE_BLOBS = {
    "soft_shoe_decision_runtime.js": "83dc2c4ca539c7c9c1228042888a1b1e09c4e4bc",
    "soft_shoe_model_bundle.js": "59afcff8391ba26d2f4bcc6bbc5f868cb7de9120",
    "soft_shoe_frozen_lstm_runtime.js": "032c53395d7c285b0dfeb915a5317b6a564416b7",
    "soft_shoe_frozen_model_bundle.js": "68728cc22c5c3780432827de33e737e7cfcd8343",
}
VERSION = "BBB_SOFT_SHOE_BPT_EV_V2"
MAX_HANDS = 72
TIMEOUT_SECONDS = 8.0


def verify_source() -> None:
    for name, expected in SOURCE_BLOBS.items():
        data = (ENGINE / name).read_bytes()
        got = sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        if got != expected:
            raise RuntimeError(f"BBB Soft Shoe source blob mismatch: {name}")


def analyze_history(history: Iterable[str]) -> dict[str, Any]:
    seq = [str(v).upper().strip() for v in history]
    if any(x not in {"B", "P", "T"} for x in seq):
        raise ValueError("invalid_bpt_history")
    # No silent truncation: doing so would invent a new shoe at its middle.
    if len(seq) > MAX_HANDS:
        raise ValueError("本靴已超過 72 局，超出目前 BBB 模型深度上限；請換新靴再開始分析。")
    verify_source()
    try:
        completed = subprocess.run(
            _node_command() + [str(ENGINE / "worker_soft_shoe.cjs")],
            input=json.dumps({"history": seq}, separators=(",", ":"), allow_nan=False),
            capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT_SECONDS,
            cwd=str(ROOT), check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("BBB Soft Shoe 計算超過 8 秒，未更新預測結果。") from exc
    if completed.returncode:
        raise RuntimeError("BBB Soft Shoe worker failed: " + completed.stderr[-350:])
    try:
        response = json.loads(completed.stdout.strip())
    except (ValueError, TypeError) as exc:
        raise RuntimeError("BBB Soft Shoe worker invalid response") from exc
    if not response.get("ok"):
        raise RuntimeError("BBB Soft Shoe error: " + str(response.get("error", "unknown")))
    result = response["result"]
    if result.get("source_commit") != SOURCE_COMMIT or result.get("source_blob") != SOURCE_BLOBS["soft_shoe_decision_runtime.js"]:
        raise RuntimeError("BBB Soft Shoe worker source mismatch")
    report = result.get("report") or {}
    if report.get("version") != VERSION or report.get("observations") != len(seq):
        raise RuntimeError("BBB Soft Shoe observation/version mismatch")
    if (report.get("config") or {}).get("PARTICLE_COUNT") != 448 or (
        (report.get("config") or {}).get("LIKELIHOOD_DRAWS") != 80 or
        (report.get("config") or {}).get("NEXT_HAND_DRAWS") != 128
    ):
        raise RuntimeError("BBB Soft Shoe N/L/M configuration mismatch")
    if report.get("action") != "Skip" or report.get("bankroll_fraction") != 0 or report.get("validated_edge") is not False:
        raise RuntimeError("BBB unvalidated betting gate changed")
    return report


def predict_screen(
    history: Iterable[str] | None = None, *, venue: str = "", room: str = "",
    shoe_id: str = "", user_id: str = "", run_seed: int | None = None,
    shoe_context: Mapping[str, Any] | None = None,
    road_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Screenshot predictor seam; visual context remains diagnostic only.

    No actual cards, OCR image pixels, estimated remaining counts or road feature
    arrays are sent as if they were observed ranks to BBB's BPT-only model.
    """
    del run_seed, shoe_context, road_context
    seq = [str(x.get("outcome") if isinstance(x, Mapping) else x).upper().strip()
           for x in (history or [])]
    report = analyze_history(seq)
    view = dict(report.get("research_view") or {})
    display = view.get("direction") if view.get("direction") in {"B", "P"} else "SKIP"
    label = {"B": "莊", "P": "閒", "SKIP": "觀望"}[display]
    prob = report.get("probabilities") or {}
    if not all(isinstance(prob.get(x), (float, int)) for x in ("B", "P", "T")):
        # An unsuccessful posterior is not a 0% probability prediction.
        raise ValueError("BBB Soft Shoe 尚無有效機率估計：" + str(report.get("reason") or "unknown"))
    return {
        "ok": True, "engine": VERSION, "model_version": VERSION,
        "system_model_version": VERSION, "bbb_soft_shoe": {
            "active": True, "source_commit": SOURCE_COMMIT,
            "report": report, "research_signal": view,
            "display_direction": display, "official_bet_allowed": False,
        },
        "action": display, "recommend": display, "direction": display,
        "internal_action": display, "internal_recommend": display,
        "next_round_direction": display,
        "action_text": label, "recommend_text": label, "direction_text": label,
        "next_round_direction_text": label,
        "skip": display == "SKIP", "force_observe": display == "SKIP",
        "skip_reason": str(view.get("reason") or report.get("reason") or ""),
        "banker_rate": float(prob["B"]) * 100,
        "player_rate": float(prob["P"]) * 100,
        "tie_rate": float(prob["T"]) * 100,
        "probabilities": dict(prob),
        "ev_banker": report.get("ev_banker"), "ev_player": report.get("ev_player"),
        "confidence": report.get("confidence"),
        "confidence_label": "模型估計，非命中率",
        "bet_allowed": False, "mandatory_bet": False, "risk_gate_open": False,
        "signal_allowed": view.get("eligible") is True,
        "suggested_bet_amount": 0, "bet_amount": 0,
        "bet_percentage": 0, "final_bet_ratio": 0,
        "kelly_fraction": 0, "kelly_percentage_applied": 0,
        "money_management": {
            "bet_allowed": False, "mandatory_bet": False, "bet_amount": 0,
            "bet_percentage": 0, "final_bet_ratio": 0, "kelly_fraction": 0,
        },
        "shoe_id": str(shoe_id or ""), "venue": str(venue or ""),
        "room": str(room or ""), "user_scope_in_model": False,
        "observed_actual_cards": False, "validated_edge": False,
    }
