"""Production LSTM + fresh exact-shoe + cut-depth fusion for baccarat B/P.

The formal direction is produced by one symmetric LSTM/shoe fusion. Legacy
road/Markov/LinUCB/HSMM/hazard models do not participate here.

Anti-stick rules:
* no hand-written recent-side/persistence term has formal direction weight;
* training is B/P-symmetry augmented instead of skipping one-class windows;
* inference projects the neural output through a B/P-swapped paired pass, which
  removes learned class bias while retaining sequence-pattern evidence;
* exact remaining composition loses direction authority immediately when the
  B/P/T history advances but the supplied composition does not change;
* cut depth changes the relative LSTM/shoe trust but never invents a B/P sign.

This is a probabilistic model and cannot guarantee profitable baccarat results.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence
import math
import os

import numpy as np

from shoe_composition import STANDARD_EIGHT_DECK_BASELINE, analyze_shoe_composition
from shoe_constants import SHOE_DECKS
from shoe_depth_estimator import (
    DEFAULT_CUT_CARD_REMAINING,
    TARGET_HANDS_MAX,
    TARGET_HANDS_MIN,
    ShoeDepthEstimator,
    build_shoe_depth_features,
)

MODEL_ID = "LSTM-SHOE-CUT-FUSION-V3-ANTI-STICK"
PAD_TOKEN = 0
B_TOKEN = 1
P_TOKEN = 2
VOCAB_SIZE = 3
WINDOW_SIZE = max(18, min(32, int(os.getenv("LSTM_ROAD_WINDOW", "24") or "24")))
MIN_HISTORY = max(8, min(16, int(os.getenv("LSTM_ROAD_MIN_HISTORY", "10") or "10")))
MIN_CONTEXT = max(3, min(8, int(os.getenv("LSTM_ROAD_MIN_CONTEXT", "5") or "5")))
MIN_ONLINE_SAMPLES = max(4, min(16, int(os.getenv("LSTM_ROAD_MIN_ONLINE_SAMPLES", "5") or "5")))
FULL_MATURITY_ROUNDS = max(
    MIN_HISTORY + 10,
    min(50, int(os.getenv("LSTM_ROAD_FULL_MATURITY", "32") or "32")),
)
REPLAY_WINDOW = max(20, min(56, int(os.getenv("LSTM_ROAD_REPLAY_WINDOW", "36") or "36")))
ONLINE_LEARNING_RATE = max(
    1e-6,
    min(1e-3, float(os.getenv("LSTM_ROAD_ONLINE_LR", "0.00010") or "0.00010")),
)
L2_REGULARIZATION = max(
    0.0,
    min(1e-2, float(os.getenv("LSTM_ROAD_L2", "0.00025") or "0.00025")),
)
RECENCY_DECAY = max(
    0.90,
    min(0.999, float(os.getenv("LSTM_ROAD_RECENCY_DECAY", "0.97") or "0.97")),
)
BOOTSTRAP_EPOCHS = max(1, min(3, int(os.getenv("LSTM_ROAD_BOOTSTRAP_EPOCHS", "2") or "2")))
ONLINE_EPOCHS = 1
MAX_SCOPE_MODELS = max(4, min(128, int(os.getenv("LSTM_ROAD_MAX_SCOPES", "32") or "32")))
MAX_NEURAL_LOGIT = 0.62
MAX_SHOE_DEVIATION_LOGIT = 0.48
MAX_FINAL_LOGIT = 0.45

_BASE_B = float(STANDARD_EIGHT_DECK_BASELINE["B"])
_BASE_P = float(STANDARD_EIGHT_DECK_BASELINE["P"])
_BASE_RESOLVED_B = _BASE_B / (_BASE_B + _BASE_P)
BASE_RESOLVED_LOGIT = math.log(_BASE_RESOLVED_B / (1.0 - _BASE_RESOLVED_B))

_TF = None
_TF_IMPORT_ERROR: str | None = None
_CACHE_LOCK = RLock()
_MODEL_CACHE: "OrderedDict[str, LSTMRoadModel]" = OrderedDict()


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _safe_logit(probability: float) -> float:
    p = _clip(probability, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(logit: float) -> float:
    value = max(-20.0, min(20.0, float(logit)))
    return 1.0 / (1.0 + math.exp(-value))


def normalize_bpt(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = history.replace("|", "").replace(",", "").replace(" ", "").upper()
        if compact and all(char in {"B", "P", "T"} for char in compact):
            return list(compact[-2000:])
        items: Iterable[Any] = [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    else:
        items = history
    result: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            raw = (
                item.get("outcome")
                or item.get("actual")
                or item.get("actual_outcome")
                or item.get("virtual_outcome")
            )
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-2000:]


def normalize_bp(history: str | Iterable[Any] | None) -> list[str]:
    return [value for value in normalize_bpt(history) if value in {"B", "P"}]


def _token(side: str) -> int:
    return B_TOKEN if str(side).upper() == "B" else P_TOKEN


def _encode_window(sequence: Sequence[str], window_size: int = WINDOW_SIZE) -> np.ndarray:
    width = max(8, int(window_size))
    tokens = [_token(side) for side in sequence[-width:]]
    padded = [PAD_TOKEN] * max(0, width - len(tokens)) + tokens
    return np.asarray(padded[-width:], dtype=np.int32)


def _swap_bp_tokens(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.int32).copy()
    swapped = data.copy()
    swapped[data == B_TOKEN] = P_TOKEN
    swapped[data == P_TOKEN] = B_TOKEN
    return swapped


def _training_examples(
    sequence: Sequence[str],
    *,
    window_size: int = WINDOW_SIZE,
    min_context: int = MIN_CONTEXT,
    replay_window: int = REPLAY_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    cleaned = [side for side in sequence if side in {"B", "P"}]
    first_target = max(1, int(min_context))
    start_target = max(first_target, len(cleaned) - max(8, int(replay_window)))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for target_index in range(start_target, len(cleaned)):
        xs.append(_encode_window(cleaned[:target_index], window_size))
        ys.append(0 if cleaned[target_index] == "B" else 1)
    if not xs:
        return (
            np.empty((0, max(8, int(window_size))), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    return np.stack(xs).astype(np.int32), np.asarray(ys, dtype=np.int32)


def _symmetry_augmented_replay(
    x_real: np.ndarray,
    y_real: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Mirror every sample so a one-sided opening cannot create class bias."""
    x = np.asarray(x_real, dtype=np.int32)
    y = np.asarray(y_real, dtype=np.int32)
    if y.size == 0:
        return (
            x,
            y,
            np.empty((0,), dtype=np.float32),
            {
                "real_samples": 0,
                "augmented_samples": 0,
                "real_b_samples": 0,
                "real_p_samples": 0,
                "symmetry_augmented": False,
            },
        )

    x_swap = _swap_bp_tokens(x)
    y_swap = 1 - y
    ages = np.arange(len(y) - 1, -1, -1, dtype=np.float64)
    recency = np.power(float(RECENCY_DECAY), ages).astype(np.float32)
    x_aug = np.concatenate((x, x_swap), axis=0)
    y_aug = np.concatenate((y, y_swap), axis=0)
    w_aug = np.concatenate((recency, recency), axis=0)
    mean = float(np.mean(w_aug)) if w_aug.size else 1.0
    if mean > 1e-12:
        w_aug = w_aug / mean

    return (
        x_aug,
        y_aug,
        w_aug.astype(np.float32),
        {
            "real_samples": int(len(y)),
            "augmented_samples": int(len(y_aug)),
            "real_b_samples": int(np.sum(y == 0)),
            "real_p_samples": int(np.sum(y == 1)),
            "augmented_b_samples": int(np.sum(y_aug == 0)),
            "augmented_p_samples": int(np.sum(y_aug == 1)),
            "symmetry_augmented": True,
            "recency_decay": float(RECENCY_DECAY),
        },
    )


def sequence_features(history: str | Iterable[Any] | None) -> dict[str, float | int | str]:
    """Diagnostics only; handcrafted road structure has zero formal vote."""
    seq = normalize_bp(history)
    if not seq:
        return {
            "rounds": 0,
            "current_run_length": 0,
            "recent_switch_rate": 0.5,
            "recent_b_ratio": 0.5,
            "recent_p_ratio": 0.5,
            "last_side": "",
            "structure_logit": 0.0,
            "formal_structure_weight": 0.0,
        }
    current = seq[-1]
    run = 1
    for side in reversed(seq[:-1]):
        if side != current:
            break
        run += 1
    recent = seq[-12:]
    switches = sum(left != right for left, right in zip(recent, recent[1:]))
    switch_rate = switches / max(1, len(recent) - 1)
    b_ratio = sum(side == "B" for side in recent) / max(1, len(recent))
    return {
        "rounds": len(seq),
        "current_run_length": int(run),
        "recent_switch_rate": float(switch_rate),
        "recent_b_ratio": float(b_ratio),
        "recent_p_ratio": float(1.0 - b_ratio),
        "last_side": current,
        "structure_logit": 0.0,
        "formal_structure_weight": 0.0,
    }


def _tensorflow():
    global _TF, _TF_IMPORT_ERROR
    if _TF is not None:
        return _TF
    if _TF_IMPORT_ERROR is not None:
        return None
    try:
        import tensorflow as tf
    except Exception as exc:
        _TF_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    _TF = tf
    return _TF


def tensorflow_status() -> dict[str, Any]:
    tf = _tensorflow()
    return {
        "available": tf is not None,
        "error": _TF_IMPORT_ERROR,
        "backend": "tensorflow.keras" if tf is not None else "unavailable",
    }


def _shoe_fusion_state(
    depth_history: Sequence[str],
    shoe_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    context = dict(shoe_context or {})
    try:
        decks = max(1, min(16, int(context.get("decks", SHOE_DECKS) or SHOE_DECKS)))
    except (TypeError, ValueError):
        decks = int(SHOE_DECKS)
    cut_override = context.get("cut_card_remaining_cards")
    exact = dict(analyze_shoe_composition(context))
    exact_available = bool(exact.get("available"))

    if exact_available:
        remaining = float(exact.get("remaining_cards", 0.0) or 0.0)
        reliability = 1.0
        depth_source = str(exact.get("source") or "exact_composition")
    else:
        supplied = context.get("remaining_cards")
        try:
            supplied_remaining = float(supplied) if supplied is not None else None
        except (TypeError, ValueError):
            supplied_remaining = None
        if supplied_remaining is not None and supplied_remaining >= 0.0:
            remaining = supplied_remaining
            reliability = _clip(context.get("remaining_cards_reliability", 0.70))
            depth_source = str(
                context.get("remaining_cards_source") or "supplied_remaining_cards"
            )
        else:
            estimate = ShoeDepthEstimator(
                shoe_decks=decks,
                cut_card_remaining_cards=cut_override,
            ).estimate(depth_history).as_dict()
            remaining = float(estimate["remaining_cards"])
            reliability = _clip(estimate.get("remaining_cards_reliability", 0.60))
            depth_source = "round_count_estimate"

    depth = build_shoe_depth_features(
        remaining,
        shoe_decks=decks,
        reliability=reliability,
        source=depth_source,
        hand_count=len(depth_history),
        cut_card_remaining_cards=cut_override,
    )

    shoe_logit = 0.0
    shoe_probability_b = None
    shoe_probability_p = None
    shoe_direction = None
    exact_counts = list(exact.get("remaining_counts") or []) if exact_available else []
    exact_signature = (
        sha256(",".join(str(int(value)) for value in exact_counts).encode("utf-8")).hexdigest()[:20]
        if len(exact_counts) == 10
        else ""
    )
    if exact_available:
        probabilities = dict(exact.get("probabilities") or {})
        p_b = max(1e-9, float(probabilities.get("B", 0.0) or 0.0))
        p_p = max(1e-9, float(probabilities.get("P", 0.0) or 0.0))
        bp_mass = p_b + p_p
        if bp_mass > 1e-12:
            resolved_b = p_b / bp_mass
            raw_logit = _safe_logit(resolved_b)
            shoe_logit = max(
                -MAX_SHOE_DEVIATION_LOGIT,
                min(MAX_SHOE_DEVIATION_LOGIT, raw_logit - BASE_RESOLVED_LOGIT),
            )
            shoe_probability_b = float(resolved_b)
            shoe_probability_p = float(1.0 - resolved_b)
            shoe_direction = "B" if resolved_b >= 0.5 else "P"

    return {
        "exact_composition_available": exact_available,
        "exact_composition_fresh": exact_available,
        "exact_composition_direction_authority": exact_available,
        "exact_composition_source": str(exact.get("source") or "none"),
        "exact_composition_signature": exact_signature,
        "stale_exact_composition": False,
        "stale_exact_reason": "",
        "shoe_decks": int(decks),
        "remaining_cards": float(remaining),
        "remaining_ratio": float(depth["remaining_ratio"]),
        "penetration": float(depth["penetration"]),
        "shoe_stage": str(depth["shoe_stage"]),
        "remaining_cards_reliability": float(reliability),
        "cut_card_remaining_cards": float(depth["cut_card_remaining_cards"]),
        "cut_progress": float(depth["cut_progress"]),
        "cut_proximity": float(depth["cut_proximity"]),
        "cards_until_cut": float(depth["cards_until_cut"]),
        "estimated_hands_until_cut": float(depth["estimated_hands_until_cut"]),
        "target_hands_min": int(TARGET_HANDS_MIN),
        "target_hands_max": int(TARGET_HANDS_MAX),
        "shoe_resolved_probability_b": shoe_probability_b,
        "shoe_resolved_probability_p": shoe_probability_p,
        "shoe_direction": shoe_direction,
        "shoe_logit_deviation": float(shoe_logit),
        "raw_shoe_logit_deviation": float(shoe_logit),
        "shoe_analysis": exact,
        "depth_feature_source": depth_source,
    }


class LSTMRoadModel:
    """Per-shoe compact LSTM with fresh exact-shoe late fusion."""

    def __init__(
        self,
        *,
        scope_key: str,
        window_size: int = WINDOW_SIZE,
        min_history: int = MIN_HISTORY,
        weight_path: str | None = None,
    ) -> None:
        self.scope_key = str(scope_key or "GLOBAL")
        self.window_size = max(18, min(32, int(window_size)))
        self.min_history = max(8, min(16, int(min_history)))
        self.weight_path = str(
            weight_path or os.getenv("LSTM_ROAD_MODEL_PATH", "") or ""
        ).strip()
        self._lock = RLock()
        self._model = None
        self._pretrained_loaded = False
        self._bootstrap_done = False
        self._trained_rounds = 0
        self._online_updates = 0
        self._load_error: str | None = None
        self._last_sequence: list[str] = []
        self._reset_count = 0
        self._last_reset_reason = "initial"
        self._training_balance: dict[str, Any] = {}
        self._exact_signature = ""
        self._exact_signature_round_count = -1

    def _seed(self) -> int:
        return int(sha256(self.scope_key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF

    def _build_model(self):
        tf = _tensorflow()
        if tf is None:
            return None
        seed = self._seed()
        keras = tf.keras
        inputs = keras.Input(shape=(self.window_size,), dtype="int32", name="bp_window")
        x = keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=6,
            mask_zero=True,
            embeddings_initializer=keras.initializers.GlorotUniform(seed=seed),
            name="bp_embedding",
        )(inputs)
        x = keras.layers.LSTM(
            20,
            dropout=0.10,
            recurrent_dropout=0.0,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 1),
            recurrent_initializer=keras.initializers.Orthogonal(seed=seed + 2),
            bias_initializer="zeros",
            name="road_lstm",
        )(x)
        x = keras.layers.Dense(
            10,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION),
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 3),
            name="road_dense",
        )(x)
        outputs = keras.layers.Dense(
            2,
            activation="softmax",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="bp_softmax",
        )(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="bgs_lstm_shoe_cut_anti_stick")
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=ONLINE_LEARNING_RATE,
                clipnorm=1.0,
            ),
            loss="sparse_categorical_crossentropy",
        )
        return model

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        self._model = self._build_model()
        if self._model is None:
            return None
        if self.weight_path:
            try:
                if os.path.exists(self.weight_path):
                    self._model.load_weights(self.weight_path)
                    self._pretrained_loaded = True
                    self._bootstrap_done = True
                else:
                    self._load_error = f"weights_not_found:{self.weight_path}"
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
        return self._model

    def _reset_for_history(self, reason: str) -> None:
        self._model = None
        self._pretrained_loaded = False
        self._bootstrap_done = False
        self._trained_rounds = 0
        self._online_updates = 0
        self._last_sequence = []
        self._training_balance = {}
        self._exact_signature = ""
        self._exact_signature_round_count = -1
        self._reset_count += 1
        self._last_reset_reason = str(reason)

    def _ensure_history_alignment(self, sequence: Sequence[str]) -> None:
        current = list(sequence)
        previous = list(self._last_sequence)
        if previous:
            same_prefix = len(current) >= len(previous) and current[: len(previous)] == previous
            same_exact = current == previous
            if not same_exact and not same_prefix:
                self._reset_for_history("history_prefix_changed_or_new_shoe")
        self._last_sequence = current

    def _validate_exact_freshness(
        self,
        shoe: Mapping[str, Any],
        *,
        depth_history: Sequence[str],
    ) -> dict[str, Any]:
        state = dict(shoe)
        if not bool(state.get("exact_composition_available")):
            state["exact_composition_fresh"] = False
            state["exact_composition_direction_authority"] = False
            return state

        signature = str(state.get("exact_composition_signature") or "")
        round_count = len(depth_history)
        if not signature:
            fresh = False
            stale_reason = "exact_counts_signature_missing"
        elif not self._exact_signature or signature != self._exact_signature:
            self._exact_signature = signature
            self._exact_signature_round_count = round_count
            fresh = True
            stale_reason = ""
        elif round_count <= self._exact_signature_round_count:
            fresh = True
            stale_reason = ""
        else:
            fresh = False
            stale_reason = "unchanged_exact_counts_after_history_advanced"

        state["exact_composition_fresh"] = bool(fresh)
        state["exact_composition_direction_authority"] = bool(fresh)
        state["stale_exact_composition"] = not bool(fresh)
        state["stale_exact_reason"] = stale_reason
        if fresh:
            return state

        state["shoe_logit_deviation"] = 0.0
        decks = int(state.get("shoe_decks", SHOE_DECKS) or SHOE_DECKS)
        estimate = ShoeDepthEstimator(
            shoe_decks=decks,
            cut_card_remaining_cards=state.get("cut_card_remaining_cards"),
        ).estimate(depth_history).as_dict()
        state.update(
            {
                "remaining_cards": float(estimate["remaining_cards"]),
                "remaining_ratio": float(estimate["remaining_ratio"]),
                "penetration": float(estimate["penetration"]),
                "shoe_stage": str(estimate["shoe_stage"]),
                "remaining_cards_reliability": float(
                    estimate.get("remaining_cards_reliability", 0.60)
                ),
                "cut_card_remaining_cards": float(estimate["cut_card_remaining_cards"]),
                "cut_progress": float(estimate["cut_progress"]),
                "cut_proximity": float(estimate["cut_progress"]),
                "cards_until_cut": float(estimate["cards_until_cut"]),
                "estimated_hands_until_cut": float(estimate["estimated_hands_until_cut"]),
                "depth_feature_source": "round_count_estimate_after_stale_exact",
            }
        )
        return state

    def _fit_replay(self, sequence: Sequence[str], *, epochs: int) -> bool:
        model = self._ensure_model()
        if model is None:
            return False
        x_real, y_real = _training_examples(
            sequence,
            window_size=self.window_size,
            min_context=MIN_CONTEXT,
            replay_window=REPLAY_WINDOW,
        )
        if len(y_real) < MIN_ONLINE_SAMPLES:
            self._training_balance = {
                "real_samples": int(len(y_real)),
                "augmented_samples": 0,
                "symmetry_augmented": False,
                "reason": "insufficient_samples",
            }
            return False
        x_train, y_train, sample_weights, balance = _symmetry_augmented_replay(x_real, y_real)
        self._training_balance = {**balance, "reason": "symmetry_augmented_replay"}
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
            epochs=max(1, int(epochs)),
            batch_size=min(16, len(y_train)),
            shuffle=False,
            verbose=0,
        )
        self._online_updates += int(len(y_real))
        self._trained_rounds = len(sequence)
        return True

    def _neural_state(
        self,
        sequence: Sequence[str],
        *,
        allow_online_update: bool,
    ) -> dict[str, Any]:
        n = len(sequence)
        model = self._ensure_model()
        if model is None:
            return {
                "available": False,
                "probability_b": 0.5,
                "probability_p": 0.5,
                "logit": 0.0,
                "maturity": 0.0,
                "reason": "tensorflow_unavailable",
            }

        if n >= self.min_history and not self._bootstrap_done:
            trained = self._fit_replay(sequence, epochs=BOOTSTRAP_EPOCHS)
            self._bootstrap_done = bool(trained or self._pretrained_loaded)
        elif allow_online_update and self._bootstrap_done and n > self._trained_rounds:
            self._fit_replay(sequence, epochs=ONLINE_EPOCHS)

        if not self._bootstrap_done:
            return {
                "available": False,
                "probability_b": 0.5,
                "probability_p": 0.5,
                "logit": 0.0,
                "maturity": 0.0,
                "reason": (
                    "cold_start_symmetry_replay_not_ready"
                    if n < self.min_history
                    else "symmetry_replay_not_ready"
                ),
            }

        encoded = _encode_window(sequence, self.window_size)
        raw = model(encoded[None, :], training=False).numpy()[0]
        swapped = model(_swap_bp_tokens(encoded)[None, :], training=False).numpy()[0]
        raw_b = _clip(raw[0], 1e-6, 1.0 - 1e-6)
        swapped_b = _clip(swapped[0], 1e-6, 1.0 - 1e-6)
        raw_logit = _safe_logit(raw_b)
        swapped_logit = _safe_logit(swapped_b)
        symmetric_logit_raw = 0.5 * (raw_logit - swapped_logit)
        neural_logit = max(
            -MAX_NEURAL_LOGIT,
            min(MAX_NEURAL_LOGIT, symmetric_logit_raw),
        )
        p_b = _sigmoid(neural_logit)
        p_p = 1.0 - p_b
        maturity = _clip(
            (n - self.min_history + 1)
            / max(1.0, float(FULL_MATURITY_ROUNDS - self.min_history + 1))
        )
        if self._pretrained_loaded:
            maturity = max(0.65, maturity)
        return {
            "available": True,
            "probability_b": float(p_b),
            "probability_p": float(p_p),
            "raw_probability_b": float(raw_b),
            "swapped_probability_b": float(swapped_b),
            "raw_logit": float(raw_logit),
            "swapped_logit": float(swapped_logit),
            "symmetry_projected_logit": float(neural_logit),
            "logit": float(neural_logit),
            "maturity": float(maturity),
            "reason": "symmetry_augmented_online_lstm_available",
        }

    def predict(
        self,
        history: str | Iterable[Any] | None,
        *,
        shoe_context: Mapping[str, Any] | None = None,
        allow_online_update: bool = True,
    ) -> dict[str, Any]:
        depth_history = normalize_bpt(history)
        sequence = [value for value in depth_history if value in {"B", "P"}]
        features = sequence_features(sequence)
        shoe_raw = _shoe_fusion_state(depth_history, shoe_context)

        with self._lock:
            self._ensure_history_alignment(sequence)
            shoe = self._validate_exact_freshness(shoe_raw, depth_history=depth_history)
            neural = self._neural_state(sequence, allow_online_update=allow_online_update)

        maturity = float(neural.get("maturity", 0.0) or 0.0)
        cut = _clip(shoe.get("cut_progress", 0.0))
        exact_fresh = bool(shoe.get("exact_composition_direction_authority"))
        neural_logit = float(neural.get("logit", 0.0) or 0.0)
        shoe_logit = (
            float(shoe.get("shoe_logit_deviation", 0.0) or 0.0)
            if exact_fresh
            else 0.0
        )

        lstm_weight = maturity * (0.90 - 0.15 * cut)
        shoe_weight = (
            (0.30 + 0.20 * cut)
            * float(shoe.get("remaining_cards_reliability", 0.0) or 0.0)
            if exact_fresh
            else 0.0
        )
        structure_weight = 0.0
        structure_logit = 0.0

        fused_logit_unclipped = (
            BASE_RESOLVED_LOGIT
            + lstm_weight * neural_logit
            + shoe_weight * shoe_logit
        )
        fused_logit = max(
            -MAX_FINAL_LOGIT,
            min(MAX_FINAL_LOGIT, fused_logit_unclipped),
        )
        p_b = _sigmoid(fused_logit)
        p_p = 1.0 - p_b
        direction = "B" if p_b >= p_p else "P"
        confidence = p_b if direction == "B" else p_p

        return {
            "model_id": MODEL_ID,
            "available": True,
            "direction": direction,
            "probabilities": {"B": float(p_b), "P": float(p_p)},
            "confidence": float(confidence),
            "raw_confidence": float(confidence),
            "window_size": self.window_size,
            "min_history": self.min_history,
            "full_maturity_rounds": int(FULL_MATURITY_ROUNDS),
            "sequence_length": len(sequence),
            "raw_round_count": len(depth_history),
            "pretrained_loaded": bool(self._pretrained_loaded),
            "online_updates": int(self._online_updates),
            "trained_rounds": int(self._trained_rounds),
            "weight_path": self.weight_path or None,
            "weight_load_error": self._load_error,
            "features": features,
            "neural": neural,
            "shoe_fusion": shoe,
            "fusion": {
                "base_resolved_banker_logit": float(BASE_RESOLVED_LOGIT),
                "lstm_logit": float(neural_logit),
                "shoe_composition_logit_deviation": float(shoe_logit),
                "structure_logit": float(structure_logit),
                "lstm_weight": float(lstm_weight),
                "shoe_weight": float(shoe_weight),
                "structure_weight": 0.0,
                "cut_progress": float(cut),
                "exact_composition_fresh": bool(exact_fresh),
                "stale_exact_composition": bool(
                    shoe.get("stale_exact_composition", False)
                ),
                "fused_logit_unclipped": float(fused_logit_unclipped),
                "fused_logit": float(fused_logit),
                "direction": direction,
                "direction_authority": (
                    "single_symmetric_lstm_plus_fresh_shoe_cut_fusion"
                ),
            },
            "training_balance": dict(self._training_balance),
            "reset_count": int(self._reset_count),
            "last_reset_reason": self._last_reset_reason,
            "tensorflow": tensorflow_status(),
            "reason": (
                "symmetric_lstm_plus_fresh_exact_shoe_cut_fusion"
                if bool(neural.get("available"))
                else "cold_start_fresh_shoe_plus_physical_prior_until_lstm_ready"
            ),
            "formal_direction_source": "lstm_road_model",
            "fallback_required": False,
            "target_hand_window": {
                "min": int(TARGET_HANDS_MIN),
                "max": int(TARGET_HANDS_MAX),
            },
            "default_cut_card_remaining_cards": int(DEFAULT_CUT_CARD_REMAINING),
            "semantics": (
                "single_BP_decision_from_BP_symmetric_masked_LSTM_plus_fresh_exact_"
                "nonreplacement_shoe_logit_with_50_70_cut_weighting"
            ),
        }


def _cache_key(scope_key: str) -> str:
    return sha256(str(scope_key or "GLOBAL").encode("utf-8")).hexdigest()[:24]


def get_lstm_road_model(scope_key: str = "") -> LSTMRoadModel:
    key = _cache_key(scope_key)
    with _CACHE_LOCK:
        model = _MODEL_CACHE.pop(key, None)
        if model is None:
            model = LSTMRoadModel(scope_key=scope_key or key)
        _MODEL_CACHE[key] = model
        while len(_MODEL_CACHE) > MAX_SCOPE_MODELS:
            _MODEL_CACHE.popitem(last=False)
        return model


def predict_lstm_road(
    history: str | Iterable[Any] | None,
    *,
    scope_key: str = "",
    shoe_context: Mapping[str, Any] | None = None,
    allow_online_update: bool = True,
) -> dict[str, Any]:
    return get_lstm_road_model(scope_key).predict(
        history,
        shoe_context=shoe_context,
        allow_online_update=allow_online_update,
    )


def clear_lstm_model_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


__all__ = [
    "MODEL_ID",
    "WINDOW_SIZE",
    "MIN_HISTORY",
    "MIN_CONTEXT",
    "MIN_ONLINE_SAMPLES",
    "FULL_MATURITY_ROUNDS",
    "REPLAY_WINDOW",
    "ONLINE_LEARNING_RATE",
    "L2_REGULARIZATION",
    "RECENCY_DECAY",
    "BASE_RESOLVED_LOGIT",
    "normalize_bpt",
    "normalize_bp",
    "sequence_features",
    "tensorflow_status",
    "LSTMRoadModel",
    "get_lstm_road_model",
    "predict_lstm_road",
    "clear_lstm_model_cache",
]
