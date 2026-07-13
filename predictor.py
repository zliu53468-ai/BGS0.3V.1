"""Fast baccarat predictor: indexed shape DB + Monte Carlo + light shoe calibration.

No TensorFlow and no synchronous DeepSeek call are used in the critical path.
This deliberately avoids the previous latency from local/global LSTM training
and external API waits.

The database lookup supplies a conditional continue/switch/tie distribution.
Monte Carlo samples that distribution. A small, symmetric shoe calibration
adjusts continuation/switch using recent shape stability without counting which
absolute side is currently ahead.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from pattern_database_shape import PatternDatabase, encode_shape, last_bp_side

B_PRIOR = float(os.getenv("B_PRIOR", "0.4586"))
P_PRIOR = float(os.getenv("P_PRIOR", "0.4462"))
T_PRIOR = float(os.getenv("T_PRIOR", "0.0952"))
PATTERN_DB_PATH = os.getenv("PATTERN_DB_PATH", "pattern_10m.sqlite3")
PATTERN_DB_MAX_ORDER = int(os.getenv("PATTERN_DB_MAX_ORDER", "16"))
PATTERN_DB_MIN_MATCHES = int(os.getenv("PATTERN_DB_MIN_MATCHES", "20"))
PATTERN_DB_SMOOTHING = float(os.getenv("PATTERN_DB_SMOOTHING", "6"))
MC_SIMULATIONS = int(os.getenv("MC_SIMULATIONS", "5000"))
MC_SEED = int(os.getenv("MC_SEED", "20260713"))
SHOE_CALIBRATION_WEIGHT = float(os.getenv("SHOE_CALIBRATION_WEIGHT", "0.18"))
SHOE_SHORT_WINDOW = int(os.getenv("SHOE_SHORT_WINDOW", "6"))
SHOE_LONG_WINDOW = int(os.getenv("SHOE_LONG_WINDOW", "16"))
MAX_DIRECTION_PROB = float(os.getenv("MAX_DIRECTION_PROB", "0.66"))
RECOMMEND_TIE = os.getenv("RECOMMEND_TIE", "0") == "1"

_PATTERN_DB = PatternDatabase(
    PATTERN_DB_PATH,
    max_order=PATTERN_DB_MAX_ORDER,
    min_matches=PATTERN_DB_MIN_MATCHES,
    smoothing=PATTERN_DB_SMOOTHING,
    b_prior=B_PRIOR,
    p_prior=P_PRIOR,
    t_prior=T_PRIOR,
)


def _clean(history: Union[str, Iterable[Any], None]) -> List[str]:
    if history is None:
        return []
    if isinstance(history, str):
        values = [ch for ch in history.upper() if ch in {"B", "P", "T"}]
    else:
        values = [str(x).strip().upper() for x in history]
    return [x for x in values if x in {"B", "P", "T"}]


def _normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(arr.sum())
    return arr / total if total > 0 else np.ones(len(arr)) / len(arr)


def _shape_rates(history: Sequence[str], window: int) -> Tuple[float, float]:
    shape = encode_shape(history)[-max(1, window):]
    decisions = [x for x in shape if x in {"S", "C"}]
    if not decisions:
        return 0.5, 0.5
    same = decisions.count("S") / len(decisions)
    return same, 1.0 - same


def _calibrate_shape(
    history: Sequence[str],
    continue_prob: float,
    switch_prob: float,
    tie_prob: float,
) -> Tuple[float, float, float, Dict[str, float]]:
    short_c, short_s = _shape_rates(history, SHOE_SHORT_WINDOW)
    long_c, long_s = _shape_rates(history, SHOE_LONG_WINDOW)
    # Use shape stability only; no absolute B/P counts.
    local_c = short_c * 0.65 + long_c * 0.35
    local_s = 1.0 - local_c
    reliability = min(1.0, len(encode_shape(history)) / max(1.0, SHOE_LONG_WINDOW))
    weight = max(0.0, min(0.45, SHOE_CALIBRATION_WEIGHT * reliability))
    non_tie = max(1e-12, continue_prob + switch_prob)
    db_c, db_s = continue_prob / non_tie, switch_prob / non_tie
    adjusted_c = db_c * (1.0 - weight) + local_c * weight
    adjusted_s = 1.0 - adjusted_c
    tie = max(0.02, min(0.20, tie_prob))
    remaining = 1.0 - tie
    return (
        adjusted_c * remaining,
        adjusted_s * remaining,
        tie,
        {
            "weight": round(weight, 6),
            "short_continue_rate": round(short_c, 6),
            "long_continue_rate": round(long_c, 6),
        },
    )


def _mc(
    probs: Sequence[float],
    history: Sequence[str],
    simulations: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    base = _normalize(probs)
    fingerprint = hashlib.sha1("".join(history).encode()).hexdigest()[:16]
    seed = (int(fingerprint, 16) + MC_SEED) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    simulations = max(100, min(100_000, int(simulations)))
    samples = rng.multinomial(simulations, base)
    result = samples / simulations
    std = np.sqrt(base * (1.0 - base) / simulations)
    return result, {
        "simulations": simulations,
        "counts": {"B": int(samples[0]), "P": int(samples[1]), "T": int(samples[2])},
        "standard_error": {"B": float(std[0]), "P": float(std[1]), "T": float(std[2])},
    }


def _cap_bp(probs: np.ndarray) -> np.ndarray:
    result = probs.copy()
    tie = float(result[2])
    non_tie = max(1e-12, float(result[0] + result[1]))
    bp = result[:2] / non_tie
    winner = int(np.argmax(bp))
    maximum = max(0.50, min(0.80, MAX_DIRECTION_PROB))
    if bp[winner] > maximum:
        bp[winner] = maximum
        bp[1 - winner] = 1.0 - maximum
    result[0], result[1] = bp[0] * (1.0 - tie), bp[1] * (1.0 - tie)
    return _normalize(result)


def predict(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
) -> Dict[str, Any]:
    cleaned = _clean(history)
    lookup = _PATTERN_DB.lookup(cleaned)

    c, s, t, calibration = _calibrate_shape(
        cleaned,
        lookup.continue_prob,
        lookup.switch_prob,
        lookup.tie_prob,
    )
    last_side = last_bp_side(cleaned)
    if last_side == "B":
        base = np.asarray([c, s, t], dtype=float)
    elif last_side == "P":
        base = np.asarray([s, c, t], dtype=float)
    else:
        non_tie = 1.0 - t
        b_ratio = B_PRIOR / max(1e-12, B_PRIOR + P_PRIOR)
        base = np.asarray([non_tie * b_ratio, non_tie * (1.0 - b_ratio), t], dtype=float)

    mc_probs, mc_info = _mc(base, cleaned, MC_SIMULATIONS)
    final = _cap_bp(mc_probs)
    names = ["B", "P", "T"]
    allowed = [0, 1, 2] if RECOMMEND_TIE else [0, 1]
    best_index = max(allowed, key=lambda i: float(final[i]))
    recommend = names[best_index]
    recommend_text = {"B": "莊", "P": "閒", "T": "和"}[recommend]
    bp_total = max(1e-12, float(final[0] + final[1]))
    confidence = max(float(final[0]), float(final[1])) / bp_total
    edge = abs(float(final[0] - final[1])) / bp_total
    signal = "HIGH" if confidence >= 0.60 else "MEDIUM" if confidence >= 0.54 else "LOW"

    reason = (
        f"PatternDB={lookup.status}; shape_order={lookup.order}; "
        f"matches={lookup.matches}; MonteCarlo={mc_info['simulations']}; "
        f"鞋內形狀校正={calibration['weight']:.3f}"
    )
    return {
        "ok": True,
        "engine": "PATTERN_DB_CONDITIONAL_MONTE_CARLO_V3",
        "user_id": str(user_id or ""),
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "round_no": len(cleaned) + 1,
        "history_len": len(cleaned),
        "history_tail": "".join(cleaned[-36:]),
        "banker_rate": round(float(final[0]) * 100, 1),
        "player_rate": round(float(final[1]) * 100, 1),
        "tie_rate": round(float(final[2]) * 100, 1),
        "probabilities": {"B": float(final[0]), "P": float(final[1]), "T": float(final[2])},
        "recommend": recommend,
        "recommend_text": recommend_text,
        "is_observe": False,
        "observe_reason": "",
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100, 1),
        "decision_edge": round(edge, 6),
        "signal_level": signal,
        "pattern_label": "牌路形狀條件機率＋Monte Carlo",
        "regime": "SHAPE_DB_MC",
        "reason": reason,
        "pattern_database": {
            "status": lookup.status,
            "available": lookup.available,
            "context": lookup.context,
            "order": lookup.order,
            "matches": lookup.matches,
            "continue_count": lookup.continue_count,
            "switch_count": lookup.switch_count,
            "tie_count": lookup.tie_count,
            "continue_prob": lookup.continue_prob,
            "switch_prob": lookup.switch_prob,
            "tie_prob": lookup.tie_prob,
            "last_side": lookup.last_side,
        },
        "monte_carlo": mc_info,
        "shoe_calibration": calibration,
        # Compatibility fields used by older UI.
        "ai_used": False,
        "ai_status": "disabled_fast_path",
        "ml_trained": False,
        "ml_samples": lookup.matches,
        "tf_available": False,
        "lstm_status": "disabled_fast_path",
        "global_lstm_status": "disabled_fast_path",
        "gru_status": "disabled",
        "tcn_status": "disabled",
        "gbm_status": "disabled",
        "gbm_backend": "DISABLED",
        "configured_weights": {"pattern_db": 1.0, "lstm": 0.0, "deepseek": 0.0},
        "effective_weights": {"pattern_db": 1.0, "lstm": 0.0, "deepseek": 0.0},
        "component_probs": {"pattern_db": {"B": float(base[0]), "P": float(base[1]), "T": float(base[2])}, "final": {"B": float(final[0]), "P": float(final[1]), "T": float(final[2])}},
        "debug": None,
    }


def fit_history(
    history: Union[str, Iterable[Any]],
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    force: bool = True,
) -> Dict[str, Any]:
    cleaned = _clean(history)
    return {
        "ok": True,
        "history_len": len(cleaned),
        "training_key": f"{user_id}|{venue}|{room}|{shoe_id}",
        "models": {"pattern_db": {"status": "indexed_static_database", "trained": True}},
        "lstm": {"status": "disabled_fast_path", "trained": False},
    }


def clear_model_cache(user_id: Optional[str] = None) -> Dict[str, Any]:
    return {"ok": True, "removed": 0, "user_id": user_id}


def reset_uid_model(
    user_id: str,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> Dict[str, Any]:
    return {"ok": True, "removed": 0, "training_key": f"{user_id}|{venue}|{room}|{shoe_id}"}


def get_model_cache_info() -> Dict[str, Any]:
    return {"size": 0, "engine": "PATTERN_DB_CONDITIONAL_MONTE_CARLO_V3"}
