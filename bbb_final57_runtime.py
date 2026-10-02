"""BBB main Final57 runtime port for the LINE predictor.

Prediction-only bridge:
BBB V23 R1 Core P(B)
    -> 213D B/P/T history vector
    -> frozen 48D Physics MLP
    -> fixed 57D direct-classification XGBoost
    -> dynamic probability clip / optional post-clip EMA
    -> Banker / Player / Skip EV decision

No OCR, LIFF, webhook, session mutation, or image-recognition code lives here.
The model JSON files are copied verbatim from BBB/main.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PHYSICS_MODEL_PATH = BASE_DIR / "physics_multitask_model.json"
FINAL_MODEL_PATH = BASE_DIR / "final_probability_model.json"

VERSION = "PHYSICS_57D_FINAL_PROBABILITY_V5"
HISTORY_WINDOW = 64
HISTORY_INPUT_DIM = 213
PHYSICS_DIM = 48
FEATURE_DIM = 57
OUTCOMES = ("B", "P", "T")
DEFAULT_BOUNDS = (0.40, 0.60)
EARLY_BOUNDS = (0.45, 0.55)
LATE_CLEAN_BOUNDS = (0.35, 0.65)
PHYSICS_NOISE_LOW_THRESHOLD = 0.78
DEFAULT_MIN_EV = {"early": 0.020, "middle": 0.010, "late": 0.005}

ORIGINAL7_NAMES = (
    "core_p_b", "round_index", "estimated_total_hands", "remaining_ratio",
    "sx_markov_p_same", "stage", "depth",
)
PHYSICS_NAMES = (
    ("cards_p4", "cards_p5", "cards_p6")
    + tuple(f"player_point_p{i}" for i in range(10))
    + tuple(f"banker_point_p{i}" for i in range(10))
    + ("winner_p_b", "winner_p_p", "winner_p_t")
    + tuple(f"next_rank_expected_{x}" for x in ("A","2","3","4","5","6","7","8","9","10","J","Q","K"))
    + ("next_suit_ratio_spades","next_suit_ratio_hearts","next_suit_ratio_diamonds","next_suit_ratio_clubs")
    + ("shoe_consumed_cards","remaining_low_rank_density","remaining_high_rank_density")
    + ("expected_point_diff_norm","expected_abs_point_diff_norm")
)
PHYSICS_INDEX = {name: i for i, name in enumerate(PHYSICS_NAMES)}
FEATURE_NAMES = (
    ("core_p_b_external", "shoe_progress_weight")
    + tuple(f"original7_{name}" for name in ORIGINAL7_NAMES[1:])
    + PHYSICS_NAMES
    + ("physics_noise_score",)
)
assert len(PHYSICS_NAMES) == PHYSICS_DIM
assert len(FEATURE_NAMES) == FEATURE_DIM

_PHYSICS_BUNDLE: dict[str, Any] | None = None
_FINAL_BUNDLE: dict[str, Any] | None = None


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid model bundle: {path.name}")
    return payload


def load_models(force: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    global _PHYSICS_BUNDLE, _FINAL_BUNDLE
    if force or _PHYSICS_BUNDLE is None:
        _PHYSICS_BUNDLE = _load_json(PHYSICS_MODEL_PATH)
    if force or _FINAL_BUNDLE is None:
        _FINAL_BUNDLE = _load_json(FINAL_MODEL_PATH)
    if _PHYSICS_BUNDLE.get("model_type") != "baccarat_physics_multitask_mlp":
        raise ValueError("physics model type mismatch")
    if not bool(_PHYSICS_BUNDLE.get("trained")):
        raise ValueError("physics model is not trained")
    if int(_PHYSICS_BUNDLE.get("history_input_dim", 0)) != HISTORY_INPUT_DIM:
        raise ValueError("physics history dimension mismatch")
    if int(_PHYSICS_BUNDLE.get("physics_dim", 0)) != PHYSICS_DIM:
        raise ValueError("physics output dimension mismatch")
    if _FINAL_BUNDLE.get("model_type") != "xgb_final_probability_classifier":
        raise ValueError("final probability model type mismatch")
    if not bool(_FINAL_BUNDLE.get("trained")):
        raise ValueError("final probability model is not trained")
    if len(_FINAL_BUNDLE.get("feature_names") or []) != FEATURE_DIM:
        raise ValueError("final probability feature dimension mismatch")
    return _PHYSICS_BUNDLE, _FINAL_BUNDLE


def normalize_history(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    raw = list(history) if not isinstance(history, str) else list(history.upper())
    out: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            item = item.get("outcome") or item.get("actual") or item.get("actual_outcome")
        token = str(item or "").upper().strip()
        if token in OUTCOMES:
            out.append(token)
    return out[-2000:]


def _bp(seq: Sequence[str]) -> list[str]:
    return [x for x in seq if x in {"B", "P"}]


def _transition_sequence(seq: Sequence[str]) -> list[str]:
    values = _bp(seq)
    return ["S" if values[i] == values[i - 1] else "X" for i in range(1, len(values))]


def _current_stage(seq: Sequence[str]) -> int:
    values = _bp(seq)
    if not values:
        return 0
    side, count = values[-1], 1
    for token in reversed(values[:-1]):
        if token != side:
            break
        count += 1
    return count


def _current_depth(seq: Sequence[str]) -> int:
    tokens = _transition_sequence(seq)
    if not tokens:
        return 0
    token, count = tokens[-1], 1
    for value in reversed(tokens[:-1]):
        if value != token:
            break
        count += 1
    return count


def _sx_markov_p_same(seq: Sequence[str], window: int = 24, prior: float = 1.0) -> float:
    tokens = _transition_sequence(seq)
    if not tokens:
        return 0.5
    current = tokens[-1]
    start = max(0, len(tokens) - 1 - max(2, window))
    same = switch = 0
    for i in range(start, len(tokens) - 1):
        if tokens[i] != current:
            continue
        if tokens[i + 1] == "S":
            same += 1
        elif tokens[i + 1] == "X":
            switch += 1
    return _clip((same + prior) / (same + switch + 2.0 * prior))


def _ratios(seq: Sequence[str], n: int) -> tuple[float, float, float]:
    block = list(seq[-n:]) if n else list(seq)
    if not block:
        return 0.0, 0.0, 0.0
    size = float(len(block))
    return block.count("B") / size, block.count("P") / size, block.count("T") / size


def _entropy3(probabilities: Sequence[float]) -> float:
    total = 0.0
    for value in probabilities:
        if value > 0:
            total -= value * math.log(value)
    return total / math.log(3.0)


def history_vector(history: str | Sequence[str]) -> np.ndarray:
    seq = normalize_history(history)
    one_hot = np.zeros(HISTORY_WINDOW * 3, dtype=np.float64)
    tail = seq[-HISTORY_WINDOW:]
    offset = HISTORY_WINDOW - len(tail)
    index = {"B": 0, "P": 1, "T": 2}
    for j, token in enumerate(tail):
        one_hot[(offset + j) * 3 + index[token]] = 1.0

    bp_values = _bp(seq)
    turns = sum(bp_values[i] != bp_values[i - 1] for i in range(1, len(bp_values)))
    r8, r16, r32 = _ratios(seq, 8), _ratios(seq, 16), _ratios(seq, 32)
    rall = _ratios(seq, max(1, len(seq)))
    summary = np.asarray([
        min(len(seq), 90) / 90.0,
        min(len(bp_values), 90) / 90.0,
        min(_current_stage(seq), 12) / 12.0,
        turns / max(1, len(bp_values) - 1),
        *r8, *r16, *r32, *rall,
        _entropy3(r8), _entropy3(r16), _entropy3(r32),
        1.0 if bp_values and bp_values[-1] == "B" else 0.0,
        1.0 if bp_values and bp_values[-1] == "P" else 0.0,
    ], dtype=np.float64)
    out = np.hstack((one_hot, summary))
    if out.size != HISTORY_INPUT_DIM:
        raise RuntimeError(f"history vector mismatch: {out.size}")
    return out


def _normalise(block: Sequence[float], fallback: Sequence[float]) -> np.ndarray:
    values = np.maximum(0.0, np.asarray(block, dtype=np.float64))
    total = float(values.sum())
    return values / total if total > 1e-12 else np.asarray(fallback, dtype=np.float64)


def _temperature_norm(block: Sequence[float], fallback: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    probabilities = np.maximum(1e-8, _normalise(block, fallback))
    logits = np.log(probabilities) / max(0.25, float(temperature or 1.0))
    logits -= float(np.max(logits))
    exp = np.exp(logits)
    return exp / float(exp.sum())


def _sanitize_physics(raw: Sequence[float], temperatures: Mapping[str, Any] | None = None) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != PHYSICS_DIM:
        raise ValueError("physics dim mismatch")
    temperatures = dict(temperatures or {})
    out = np.zeros(PHYSICS_DIM, dtype=np.float64)
    src = dst = 0

    def copy_norm(size: int, fallback: Sequence[float], key: str) -> None:
        nonlocal src, dst
        block = _temperature_norm(values[src:src + size], fallback, float(temperatures.get(key, 1.0) or 1.0))
        src += size
        out[dst:dst + size] = block
        dst += size

    copy_norm(3, (0.58, 0.34, 0.08), "card_count")
    copy_norm(10, (0.1,) * 10, "player_points")
    copy_norm(10, (0.1,) * 10, "banker_points")
    copy_norm(3, (0.4586, 0.4462, 0.0952), "winner")
    out[dst:dst + 13] = np.clip(values[src:src + 13], 0.0, 6.0); src += 13; dst += 13
    copy_norm(4, (0.25,) * 4, "suit")
    out[dst] = _clip(values[src], 0.0, 416.0); src += 1; dst += 1
    out[dst] = _clip(values[src]); src += 1; dst += 1
    out[dst] = _clip(values[src]); src += 1; dst += 1
    out[dst] = _clip(values[src], -1.0, 1.0); src += 1; dst += 1
    out[dst] = _clip(values[src]); src += 1; dst += 1
    return out


def predict_physics(history: str | Sequence[str], bundle: Mapping[str, Any] | None = None) -> np.ndarray:
    if bundle is None:
        bundle, _ = load_models()
    x = history_vector(history)
    scaler = dict(bundle.get("scaler") or {})
    mean = np.asarray(scaler.get("mean") or np.zeros(HISTORY_INPUT_DIM), dtype=np.float64)
    scale = np.asarray(scaler.get("scale") or np.ones(HISTORY_INPUT_DIM), dtype=np.float64)
    h = (x - mean) / np.maximum(1e-12, scale)

    coefs = list(bundle.get("coefs") or [])
    intercepts = list(bundle.get("intercepts") or [])
    if not coefs or len(coefs) != len(intercepts):
        raise ValueError("invalid physics model")
    for layer, (weights, bias) in enumerate(zip(coefs, intercepts)):
        h = h @ np.asarray(weights, dtype=np.float64) + np.asarray(bias, dtype=np.float64)
        if layer < len(coefs) - 1:
            h = np.maximum(0.0, h)

    loss_scale = np.asarray(bundle.get("loss_scale") or np.ones(PHYSICS_DIM), dtype=np.float64)
    if loss_scale.size == PHYSICS_DIM:
        h = h / np.maximum(1e-12, loss_scale)
    target_scaler = dict(bundle.get("target_scaler") or {})
    t_mean = np.asarray(target_scaler.get("mean") or np.zeros(PHYSICS_DIM), dtype=np.float64)
    t_scale = np.asarray(target_scaler.get("scale") or np.ones(PHYSICS_DIM), dtype=np.float64)
    h = h * np.maximum(1e-12, t_scale) + t_mean

    calibration = dict(bundle.get("output_calibration") or {})
    slope = np.asarray(calibration.get("slope") or np.ones(PHYSICS_DIM), dtype=np.float64)
    intercept = np.asarray(calibration.get("intercept") or np.zeros(PHYSICS_DIM), dtype=np.float64)
    if slope.size == PHYSICS_DIM and intercept.size == PHYSICS_DIM:
        h = h * slope + intercept
    return _sanitize_physics(h, bundle.get("calibration_temperatures") or {})


def _normalised_entropy(block: Sequence[float]) -> float:
    p = _normalise(block, (1.0 / len(block),) * len(block))
    return float(-sum(v * math.log(v) for v in p if v > 0) / math.log(len(p)))


def physics_noise_score(physics: Sequence[float], round_index: float = 70.0) -> float:
    values = np.asarray(physics, dtype=np.float64)
    winner = sorted(values[23:26], reverse=True)
    gap = float(winner[0] - winner[1])
    density_gap = abs(_clip(values[44]) - _clip(values[45]))
    density_ambiguity = 1.0 - min(1.0, density_gap / 0.25)
    raw = _clip(
        0.10 * _normalised_entropy(values[0:3])
        + 0.075 * _normalised_entropy(values[3:13])
        + 0.075 * _normalised_entropy(values[13:23])
        + 0.15 * _normalised_entropy(values[23:26])
        + 0.25 * _normalised_entropy(values[26:39])
        + 0.15 * _normalised_entropy(values[39:43])
        + 0.075 * (1.0 - gap)
        + 0.125 * density_ambiguity
    )
    compressed = 0.50 + 0.35 * math.tanh((raw - 0.75) / 0.20)
    influence = 0.35 if round_index <= 40 else 0.65 if round_index <= 50 else 1.0
    return _clip(0.50 + (compressed - 0.50) * influence)


def build_original7(history: Sequence[str], core_prediction: Mapping[str, Any], estimated_total_hands: float = 60.0) -> dict[str, float]:
    seq = normalize_history(history)
    probabilities = dict(core_prediction.get("probabilities") or {})
    core_pb = _clip(probabilities.get("B", 0.5))
    round_index = float(max(1, min(70, len(seq) + 1)))
    total_hands = float(estimated_total_hands) if 40 <= float(estimated_total_hands) <= 90 else 60.0
    remaining_ratio = _clip((total_hands - (round_index - 1.0)) / max(1.0, total_hands))
    signal = dict(core_prediction.get("singleHazard") or {})
    state = dict(signal.get("state") or {})
    depth_info = dict(signal.get("depth") or {})
    return {
        "core_p_b": core_pb,
        "round_index": round_index,
        "estimated_total_hands": total_hands,
        "remaining_ratio": remaining_ratio,
        "sx_markov_p_same": _sx_markov_p_same(seq),
        "stage": float(state.get("length", _current_stage(seq)) or 0.0),
        "depth": float(depth_info.get("depth", _current_depth(seq)) or 0.0),
    }


def build_features(core_pb: float, original7: Mapping[str, Any], physics: Sequence[float]) -> np.ndarray:
    original = np.asarray([float(original7[name]) for name in ORIGINAL7_NAMES], dtype=np.float32)
    physics_array = np.asarray(physics, dtype=np.float32).reshape(-1)
    progress = np.float32((float(original[1]) / 70.0) ** 3)
    noise = np.float32(physics_noise_score(physics_array, float(original[1])))
    out = np.hstack((np.asarray([core_pb, progress], dtype=np.float32), original[1:], physics_array, np.asarray([noise], dtype=np.float32)))
    if out.size != FEATURE_DIM:
        raise RuntimeError(f"extended dim mismatch: {out.size}")
    return out.astype(np.float32, copy=False)


def _tree_child(node: Mapping[str, Any], node_id: Any) -> Mapping[str, Any] | None:
    target = int(node_id)
    for child in node.get("children") or []:
        if int(child.get("nodeid", -1)) == target:
            return child
    return None


def _split_index(split: Any, names: Sequence[str]) -> int:
    token = str(split or "")
    if token.startswith("f") and token[1:].isdigit():
        return int(token[1:])
    try:
        return names.index(token)
    except ValueError:
        return -1


def _evaluate_tree(tree: Mapping[str, Any], vector: np.ndarray, names: Sequence[str]) -> float:
    node: Mapping[str, Any] | None = tree
    guard = 0
    while node is not None and guard < 256:
        guard += 1
        if "leaf" in node:
            return float(node.get("leaf", 0.0) or 0.0)
        index = _split_index(node.get("split"), names)
        value = np.float32(vector[index]) if 0 <= index < vector.size else np.float32(np.nan)
        threshold = np.float32(float(node.get("split_condition", 0.0) or 0.0))
        if not np.isfinite(value):
            next_id = node.get("missing")
        else:
            next_id = node.get("yes") if value < threshold else node.get("no")
        node = _tree_child(node, next_id)
    return 0.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def predict_raw_probability(bundle: Mapping[str, Any], vector: np.ndarray) -> float:
    names = list(bundle.get("feature_names") or FEATURE_NAMES)
    margin = float(bundle.get("base_margin", 0.0) or 0.0)
    for tree in bundle.get("trees") or []:
        margin += _evaluate_tree(tree, vector, names)
    probability = _clip(_sigmoid(margin), 1e-7, 1.0 - 1e-7)
    calibration = dict(bundle.get("calibration") or {})
    method = str(calibration.get("method") or "").lower()
    if method == "platt":
        logit = math.log(probability / (1.0 - probability))
        probability = _sigmoid(float(calibration.get("slope", 1.0) or 1.0) * logit + float(calibration.get("intercept", 0.0) or 0.0))
    elif method == "isotonic":
        xs = list(calibration.get("x_thresholds") or [])
        ys = list(calibration.get("y_thresholds") or [])
        if len(xs) > 1 and len(xs) == len(ys):
            probability = float(np.interp(probability, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
    return _clip(probability)


def dynamic_bounds(round_index: float, noise_score: float) -> tuple[float, float]:
    if round_index <= 40:
        return EARLY_BOUNDS
    if round_index > 50 and noise_score <= PHYSICS_NOISE_LOW_THRESHOLD:
        return LATE_CLEAN_BOUNDS
    return DEFAULT_BOUNDS


def _dynamic_ema_alpha(round_index: float, noise_score: float, config: Mapping[str, Any]) -> float:
    if round_index <= 40:
        stage, bounds = "early", (0.35, 0.45)
    elif round_index > 50:
        stage, bounds = "late", (0.65, 0.75)
    else:
        stage, bounds = "middle", (0.50, 0.60)
    base = float(config.get(f"{stage}_alpha", sum(bounds) / 2.0) or sum(bounds) / 2.0)
    gain = max(0.0, float(config.get("noise_gain", 0.05) or 0.05))
    return _clip(base + gain * (0.5 - _clip(noise_score)), bounds[0], bounds[1])


def _apply_smoothing(clipped_pb: float, round_index: float, noise_score: float, bounds: tuple[float, float], bundle: Mapping[str, Any], previous_final_pb: float | None) -> tuple[float, float, float, str]:
    config = dict(bundle.get("smoothing") or {})
    dynamic = config.get("method") == "dynamic_post_clip_ema" and config.get("enabled") is True
    legacy = config.get("method") == "causal_ema" and float(config.get("strength", 0.0) or 0.0) > 0.0
    if previous_final_pb is None or (not dynamic and not legacy):
        return clipped_pb, 1.0, 0.0, str(config.get("profile") or "off")
    if dynamic:
        alpha = _dynamic_ema_alpha(round_index, noise_score, config)
    else:
        alpha = 1.0 - _clip(config.get("strength", 0.0), 0.0, 0.15)
    value = _clip(alpha * clipped_pb + (1.0 - alpha) * float(previous_final_pb), bounds[0], bounds[1])
    return value, alpha, 1.0 - alpha, str(config.get("profile") or ("legacy" if legacy else "custom"))


def _decision_policy(round_index: float, noise_score: float, bundle: Mapping[str, Any]) -> tuple[float, float, float, bool]:
    thresholds = dict(DEFAULT_MIN_EV)
    thresholds.update(bundle.get("ev_thresholds") or {})
    base = float(thresholds["early"] if round_index <= 40 else thresholds["late"] if round_index > 50 else thresholds["middle"])
    config = dict(bundle.get("decision_policy") or {})
    enabled = config.get("enabled") is True
    if not enabled:
        return base, base, 0.0, False
    noise_threshold = float(config.get("noise_threshold", PHYSICS_NOISE_LOW_THRESHOLD) or PHYSICS_NOISE_LOW_THRESHOLD)
    low_noise = noise_score <= noise_threshold
    middle, late = 40 < round_index <= 50, round_index > 50
    relief = float(config.get("middle_relief", 0.001) if middle else config.get("late_relief", 0.001) if late else 0.0) if low_noise else 0.0
    max_penalty = max(0.0, float(config.get("max_noise_ev_penalty", 0.001) or 0.001))
    penalty = min(max_penalty, max(0.0, noise_score - noise_threshold) * (max_penalty / max(1e-9, 1.0 - noise_threshold)))
    min_ev = max(0.0, base - relief + penalty)
    soft_band = float(config.get("middle_soft_band", 0.001) if middle else config.get("late_soft_band", 0.0015) if late else 0.0) if low_noise else 0.0
    return min_ev, max(0.0, min_ev - soft_band), soft_band, True


def _soft_confidence(edge: float, min_ev: float, activation_ev: float, soft_band: float, bundle: Mapping[str, Any]) -> float:
    if edge <= activation_ev:
        return 0.0
    premium = max(0.0, edge - min_ev)
    if soft_band <= 0.0:
        return premium
    config = dict(bundle.get("decision_policy") or {})
    minimum = max(0.0, float(config.get("min_confidence", 0.0005) or 0.0005))
    return max(minimum, premium, 0.5 * min(soft_band, edge - activation_ev))


def unpack_physics(physics: Sequence[float]) -> dict[str, Any]:
    p = np.asarray(physics, dtype=np.float64)
    card_count = {"4_cards": float(p[0]), "5_cards": float(p[1]), "6_cards": float(p[2])}
    expected_cards = 4.0 * p[0] + 5.0 * p[1] + 6.0 * p[2]
    return {
        "next_card_count_probabilities": card_count,
        "expected_next_card_count": float(expected_cards),
        "winner_probabilities": {"B": float(p[23]), "P": float(p[24]), "T": float(p[25])},
    }


def apply_final_prediction(
    history: str | Sequence[str],
    core_prediction: Mapping[str, Any],
    *,
    previous_final_pb: float | None = None,
    estimated_total_hands: float = 60.0,
) -> dict[str, Any]:
    seq = normalize_history(history)
    physics_bundle, final_bundle = load_models()
    original7 = build_original7(seq, core_prediction, estimated_total_hands)
    core_pb = float(original7["core_p_b"])
    physics = predict_physics(seq, physics_bundle)
    features = build_features(core_pb, original7, physics)
    raw_pb = predict_raw_probability(final_bundle, features)
    round_index = float(original7["round_index"])
    noise = float(features[-1])
    probability_bounds = dynamic_bounds(round_index, noise)
    clipped_pb = _clip(raw_pb, *probability_bounds)
    final_pb, smoothing_alpha, smoothing_strength, smoothing_profile = _apply_smoothing(
        clipped_pb, round_index, noise, probability_bounds, final_bundle, previous_final_pb
    )

    p_player = 1.0 - final_pb
    p_tie = _clip(physics[PHYSICS_INDEX["winner_p_t"]])
    ev_banker = final_pb * 0.95 - p_player
    ev_player = p_player - final_pb
    min_ev, activation_ev, soft_band, policy_enabled = _decision_policy(round_index, noise, final_bundle)
    if ev_banker > activation_ev and ev_banker > ev_player:
        direction = "B"
        edge = ev_banker
    elif ev_player > activation_ev and ev_player > ev_banker:
        direction = "P"
        edge = ev_player
    else:
        direction = "Skip"
        edge = 0.0
    confidence = 0.0 if direction == "Skip" else _soft_confidence(edge, min_ev, activation_ev, soft_band, final_bundle)

    core_direction = str(core_prediction.get("direction") or "").upper().strip()
    return {
        "version": VERSION,
        "active": True,
        "mode": "final56",
        "direction": direction,
        "final_direction": {"B": "莊 B", "P": "閒 P", "Skip": "觀望 Skip"}[direction],
        "core_direction": core_direction,
        "flipped": direction != core_direction,
        "confidence": float(confidence),
        "direction_probability": float(final_pb if direction == "B" else p_player if direction == "P" else max(final_pb, p_player)),
        "probabilities": {"B": float(final_pb), "P": float(p_player), "T": 0.0},
        "core_p_b": core_pb,
        "raw_p_b": float(raw_pb),
        "clipped_p_b": float(clipped_pb),
        "smoothed_p_b": float(final_pb),
        "final_p_b": float(final_pb),
        "p_tie": float(p_tie),
        "p_player": float(p_player),
        "ev_banker": float(ev_banker),
        "ev_player": float(ev_player),
        "min_ev": float(min_ev),
        "activation_ev": float(activation_ev),
        "soft_band": float(soft_band),
        "decision_policy_enabled": bool(policy_enabled),
        "probability_bounds": {"min": float(probability_bounds[0]), "max": float(probability_bounds[1])},
        "smoothing_alpha": float(smoothing_alpha),
        "smoothing_strength": float(smoothing_strength),
        "smoothing_profile": smoothing_profile,
        "shoe_progress_weight": float(features[1]),
        "physics_noise_score": float(noise),
        "original7": original7,
        "physics_48d": [float(v) for v in physics],
        "features_57d": [float(v) for v in features],
        "physics_forecast": unpack_physics(physics),
    }


def model_status() -> dict[str, Any]:
    physics, final = load_models()
    return {
        "version": VERSION,
        "physics": bool(physics.get("trained")),
        "final57": bool(final.get("trained")),
        "physics_schema": physics.get("schema_version"),
        "final_schema": final.get("schema_version"),
        "feature_dim": FEATURE_DIM,
    }


__all__ = [
    "VERSION", "FEATURE_DIM", "HISTORY_INPUT_DIM", "PHYSICS_DIM",
    "apply_final_prediction", "build_features", "build_original7",
    "history_vector", "load_models", "model_status", "physics_noise_score",
    "predict_physics", "predict_raw_probability",
]
