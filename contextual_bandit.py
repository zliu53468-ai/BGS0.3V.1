"""BGS 大路優先的兩臂 Dynamic Contextual LinUCB 決策核心。

正式方向只由「首次截圖辨識的大路歷史 + 後續每局莊/閒/和按鈕紀錄」
所形成的完整 chronology 產生。OCR/截圖辨識本身不在此模組修改。

設計重點：
- 固定 16 維 Road Context，全部由 B/P/T 歷史衍生。
- Player(P) / Banker(B) 只有兩臂，絕不輸出觀望臂。
- Dynamic forgetting 避免 50～70 局短靴被早期單次 reward 鎖死。
- 內部 L2 normalization 降低不同尺度特徵造成的單邊偏置。
- 和局 reward=0，但不當成 B/P 方向樣本更新 A。
- 每次新歷史比 pending prediction 多一局時，先結算上一筆再預測下一局。
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import os
import time

import numpy as np

ARMS = ("P", "B")
CONTEXT_DIM = 16
CONTEXT_FEATURE_NAMES = (
    "recent_4_banker_centered",
    "recent_8_banker_centered",
    "recent_12_banker_centered",
    "recent_20_banker_centered",
    "global_banker_centered",
    "momentum_4_vs_12",
    "last_side_signed",
    "signed_run_length",
    "switch_rate_6_centered",
    "switch_rate_12_centered",
    "switch_acceleration",
    "transition_order1_tendency",
    "transition_order2_tendency",
    "local_cumulative_slope_8",
    "global_cumulative_slope",
    "local_global_slope_gap",
)

# 保留既有匯出常數，避免其他模組 import 失效；正式方向不再使用估計牌組。
SHOE_DECKS = max(1, int(os.getenv("SHOE_DECKS", "8") or "8"))
ESTIMATED_CARDS_PER_ROUND = max(
    4.0,
    min(6.0, float(os.getenv("ESTIMATED_CARDS_PER_ROUND", "4.8") or "4.8")),
)

LINUCB_ALPHA = max(0.0, float(os.getenv("LINUCB_ALPHA", "0.5") or "0.5"))
LINUCB_RIDGE = max(1e-6, float(os.getenv("LINUCB_RIDGE", "1.0") or "1.0"))
LINUCB_UPDATE_WEIGHT = max(
    1e-3, float(os.getenv("LINUCB_UPDATE_WEIGHT", "1.0") or "1.0")
)
LINUCB_FORGETTING = max(
    0.70, min(1.0, float(os.getenv("LINUCB_FORGETTING", "0.90") or "0.90"))
)
LINUCB_ARM_ALPHA_MAX_SCALE = max(
    1.0,
    min(
        2.5,
        float(os.getenv("LINUCB_ARM_ALPHA_MAX_SCALE", "1.60") or "1.60"),
    ),
)
LINUCB_SCORE_TIE_EPSILON = max(
    1e-12,
    float(os.getenv("LINUCB_SCORE_TIE_EPSILON", "0.000001") or "0.000001"),
)

PROBABILITY_MIN = 0.48
PROBABILITY_MAX = 0.58

# V4 明確切換成 Road-primary Context，部署後不沿用 V1/V2/V3 牌組型狀態。
STATE_VERSION = "LINUCB-2ARM-ROAD-PRIMARY-DYNAMIC-V4"
_LOCK = RLock()


def _clip(value: Any, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _normalize_history(history: Iterable[Any] | str | None) -> list[str]:
    """保留 B/P/T 完整時間序；和局不新增大路側別，但仍保留在 chronology。"""
    if history is None:
        return []
    if isinstance(history, str):
        compact = (
            history.replace("|", "")
            .replace(",", "")
            .replace(" ", "")
            .upper()
        )
        if compact and all(char in {"B", "P", "T"} for char in compact):
            return list(compact)[-2000:]
        items: Iterable[Any] = [
            part
            for part in history.replace("|", ",").split(",")
            if part.strip()
        ]
    else:
        items = history

    out: list[str] = []
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
            out.append(value)
    return out[-2000:]


def _bp_sequence(raw: Sequence[str]) -> list[str]:
    return [value for value in raw if value in {"B", "P"}]


def _banker_ratio(bp: Sequence[str], window: int | None = None) -> float:
    values = list(bp[-window:]) if window else list(bp)
    if not values:
        return 0.5
    return float(sum(value == "B" for value in values) / len(values))


def _center_ratio(ratio: float) -> float:
    """將 0..1 比例中心化到 -1..+1，避免全正值造成線性模型偏置。"""
    return _clip((float(ratio) - 0.5) * 2.0, -1.0, 1.0)


def _run_state(bp: Sequence[str]) -> tuple[str, int]:
    if not bp:
        return "", 0
    side = bp[-1]
    length = 0
    for value in reversed(bp):
        if value != side:
            break
        length += 1
    return side, length


def _switch_rate(bp: Sequence[str], window: int) -> float:
    values = list(bp[-window:])
    if len(values) < 2:
        return 0.5
    switches = sum(
        values[index] != values[index - 1]
        for index in range(1, len(values))
    )
    return float(switches / (len(values) - 1))


def _transition_tendency(bp: Sequence[str], order: int) -> float:
    """估計目前 state 之後開 B/P 的平滑差值，輸出 -1..+1。

    +1 越偏向 B，-1 越偏向 P。只有歷史中相同 state 的後繼樣本會被計數。
    使用 Beta(1,1) 平滑，避免小樣本直接變成極端 0/1。
    """
    order = max(1, min(2, int(order)))
    if len(bp) <= order:
        return 0.0
    state = tuple(bp[-order:])
    b_count = 1.0
    p_count = 1.0
    matches = 0
    for index in range(order, len(bp)):
        if tuple(bp[index - order:index]) != state:
            continue
        next_side = bp[index]
        if next_side == "B":
            b_count += 1.0
        else:
            p_count += 1.0
        matches += 1
    if matches <= 0:
        return 0.0
    probability_b = b_count / (b_count + p_count)
    return _center_ratio(probability_b)


def _cumulative_slope(bp: Sequence[str], window: int | None = None) -> float:
    """對 B=+1/P=-1 的累積路徑做線性斜率，再壓到 -1..+1。"""
    values = list(bp[-window:]) if window else list(bp)
    n = len(values)
    if n < 2:
        return 0.0
    encoded = np.asarray(
        [1.0 if value == "B" else -1.0 for value in values],
        dtype=np.float64,
    )
    cumulative = np.cumsum(encoded)
    x = np.arange(n, dtype=np.float64)
    x_center = x - float(np.mean(x))
    denominator = float(np.dot(x_center, x_center))
    if denominator <= 1e-12:
        return 0.0
    slope = float(
        np.dot(x_center, cumulative - float(np.mean(cumulative)))
        / denominator
    )
    # 理論上長時間單邊斜率接近 +/-1；tanh 抑制小樣本極端。
    return float(math.tanh(1.35 * slope))


def _model_x(vector: Sequence[float]) -> np.ndarray:
    """API 保留原始 16 維 Road Context；LinUCB 內部只做 L2 normalization。"""
    x = np.asarray(vector, dtype=np.float64).reshape(CONTEXT_DIM)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    norm = float(np.linalg.norm(x))
    return x / norm if norm > 1e-12 else x


@dataclass(frozen=True)
class ContextSnapshot:
    vector: np.ndarray
    metadata: dict[str, Any]


class ContextGenerator:
    """由完整截圖大路歷史產生固定 16 維 Road Context。

    shoe_context 仍接受，純粹為舊 API 相容與診斷；正式方向特徵不使用
    remaining_counts / observed_cards / remaining_cards。
    """

    def __init__(self, decks: int = SHOE_DECKS):
        self.decks = max(1, int(decks or SHOE_DECKS))

    def build(
        self,
        history: Iterable[Any] | str | None,
        shoe_context: Mapping[str, Any] | None = None,
    ) -> ContextSnapshot:
        raw = _normalize_history(history)
        bp = _bp_sequence(raw)
        context = dict(shoe_context or {})

        recent4 = _center_ratio(_banker_ratio(bp, 4))
        recent8 = _center_ratio(_banker_ratio(bp, 8))
        recent12 = _center_ratio(_banker_ratio(bp, 12))
        recent20 = _center_ratio(_banker_ratio(bp, 20))
        global_ratio = _center_ratio(_banker_ratio(bp, None))

        # 短期相對中期的方向變化；正值代表最近更往 B，負值代表更往 P。
        momentum = _clip(recent4 - recent12, -1.0, 1.0)

        last_side = bp[-1] if bp else ""
        last_signed = 1.0 if last_side == "B" else -1.0 if last_side == "P" else 0.0

        run_side, run_length = _run_state(bp)
        signed_run = min(run_length, 8) / 8.0
        if run_side == "P":
            signed_run *= -1.0
        elif not run_side:
            signed_run = 0.0

        switch6_raw = _switch_rate(bp, 6)
        switch12_raw = _switch_rate(bp, 12)
        switch6 = _center_ratio(switch6_raw)
        switch12 = _center_ratio(switch12_raw)
        switch_acceleration = _clip(
            (switch6_raw - switch12_raw) * 2.0,
            -1.0,
            1.0,
        )

        transition1 = _transition_tendency(bp, 1)
        transition2 = _transition_tendency(bp, 2)

        local_slope = _cumulative_slope(bp, 8)
        global_slope = _cumulative_slope(bp, None)
        slope_gap = _clip(local_slope - global_slope, -1.0, 1.0)

        vector = np.asarray(
            [
                recent4,
                recent8,
                recent12,
                recent20,
                global_ratio,
                momentum,
                last_signed,
                signed_run,
                switch6,
                switch12,
                switch_acceleration,
                transition1,
                transition2,
                local_slope,
                global_slope,
                slope_gap,
            ],
            dtype=np.float64,
        )
        if vector.shape != (CONTEXT_DIM,):
            raise RuntimeError(f"context dimension mismatch: {vector.shape}")
        vector = np.nan_to_num(
            vector,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        try:
            diagnostic_remaining = max(
                0.0, float(context.get("remaining_cards", 0.0) or 0.0)
            )
        except (TypeError, ValueError):
            diagnostic_remaining = 0.0

        return ContextSnapshot(
            vector=vector,
            metadata={
                "raw_round_count": len(raw),
                "bp_round_count": len(bp),
                "tie_count": sum(value == "T" for value in raw),
                "recent4_banker_ratio": _banker_ratio(bp, 4),
                "recent8_banker_ratio": _banker_ratio(bp, 8),
                "recent12_banker_ratio": _banker_ratio(bp, 12),
                "recent20_banker_ratio": _banker_ratio(bp, 20),
                "global_banker_ratio": _banker_ratio(bp, None),
                "last_side": last_side,
                "run_side": run_side,
                "run_length": run_length,
                "switch6": switch6_raw,
                "switch12": switch12_raw,
                "transition_order1_tendency": transition1,
                "transition_order2_tendency": transition2,
                "local_cumulative_slope_8": local_slope,
                "global_cumulative_slope": global_slope,
                "local_global_slope_gap": slope_gap,
                "raw_context_l2_norm": float(np.linalg.norm(vector)),
                "model_context_l2_norm": float(
                    np.linalg.norm(_model_x(vector))
                ),
                "context_scaling": "l2_internal_raw_api_preserved",
                "feature_priority": "screenshot_big_road_history_primary",
                "formal_direction_data_source": (
                    "initial_image_history_plus_manual_outcome_history"
                ),
                "card_composition_used_for_direction": False,
                # 舊 predictor 診斷欄位相容：保留鍵但不參與正式方向。
                "remaining_cards": diagnostic_remaining,
                "remaining_cards_ratio": 0.0,
                "remaining_cards_source": "diagnostic_only_not_in_road_context",
                "rank_ratio_source": "disabled_for_formal_direction",
                "rank_ratios_a_to_10jqk": [],
                "estimated_remaining_counts_0_to_9": [],
                "decks": max(
                    1,
                    int(context.get("decks", self.decks) or self.decks),
                ),
                "full_cards": 52
                * max(1, int(context.get("decks", self.decks) or self.decks)),
            },
        )


def _state_path() -> Path:
    candidates: list[Path] = []
    configured = str(os.getenv("LINUCB_STATE_FILE", "") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path("/var/data/contextual_linucb_state.json"),
            Path(__file__).resolve().parent
            / "data"
            / "contextual_linucb_state.json",
            Path("/tmp/contextual_linucb_state.json"),
        ]
    )
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            probe = candidate.parent / f".linucb_write_{time.time_ns()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    return Path("/tmp/contextual_linucb_state.json")


STATE_FILE = _state_path()


def _new_arm() -> dict[str, Any]:
    return {
        "A": (np.eye(CONTEXT_DIM) * LINUCB_RIDGE).tolist(),
        "b": np.zeros(CONTEXT_DIM).tolist(),
        "n": 0,
        "effective_n": 0.0,
    }


def _new_scope_state() -> dict[str, Any]:
    now = int(time.time())
    return {
        "arms": {arm: _new_arm() for arm in ARMS},
        "pending": {},
        "updates": 0,
        "last_selected": "",
        "selection_streak": 0,
        "created_at": now,
        "updated_at": now,
    }


def _read_state() -> dict[str, Any]:
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
    except Exception:
        payload = {}

    # Road-primary V4 不沿用舊牌組 Context 參數。
    if (
        payload.get("version") != STATE_VERSION
        or payload.get("dim") != CONTEXT_DIM
    ):
        payload = {}

    return {
        "version": STATE_VERSION,
        "dim": CONTEXT_DIM,
        "alpha": LINUCB_ALPHA,
        "ridge": LINUCB_RIDGE,
        "forgetting": LINUCB_FORGETTING,
        "scopes": (
            payload.get("scopes")
            if isinstance(payload.get("scopes"), dict)
            else {}
        ),
    }


def _write_state(payload: Mapping[str, Any]) -> None:
    temporary = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(STATE_FILE)


def make_scope_key(
    *,
    user_id: str = "",
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> str:
    raw = "|".join(
        (
            str(user_id or "").strip(),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
        )
    )
    return sha256((raw or "GLOBAL").encode("utf-8")).hexdigest()[:24]


def _history_fingerprint(history: Sequence[str]) -> str:
    return sha256("".join(history).encode("utf-8")).hexdigest()[:24]


def _arm_arrays(
    state: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        A = np.asarray(
            state.get("A"),
            dtype=np.float64,
        ).reshape(CONTEXT_DIM, CONTEXT_DIM)
        b = np.asarray(
            state.get("b"),
            dtype=np.float64,
        ).reshape(CONTEXT_DIM)
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError
        return A, b
    except Exception:
        return (
            np.eye(CONTEXT_DIM) * LINUCB_RIDGE,
            np.zeros(CONTEXT_DIM),
        )


class ContextualLinUCB:
    """兩臂 Dynamic Ridge LinUCB，專門處理 50～70 局短靴。"""

    def __init__(self, alpha: float = LINUCB_ALPHA):
        self.alpha = max(0.0, float(alpha))
        self.generator = ContextGenerator()

    def _score(
        self,
        arm_state: Mapping[str, Any],
        x: np.ndarray,
        alpha_scale: float,
    ) -> dict[str, float]:
        A, b = _arm_arrays(arm_state)
        try:
            theta = np.linalg.solve(A, b)
            solved_x = np.linalg.solve(A, x)
        except np.linalg.LinAlgError:
            A = A + np.eye(CONTEXT_DIM) * LINUCB_RIDGE
            theta = np.linalg.solve(A, b)
            solved_x = np.linalg.solve(A, x)

        mean = float(x @ theta)
        uncertainty = float(
            math.sqrt(max(0.0, float(x @ solved_x)))
        )
        effective_alpha = self.alpha * max(
            0.5, min(2.5, float(alpha_scale))
        )
        return {
            "score": mean + effective_alpha * uncertainty,
            "mean": mean,
            "uncertainty": uncertainty,
            "effective_alpha": effective_alpha,
            "raw_n": float(arm_state.get("n", 0) or 0),
            "effective_n": float(
                arm_state.get(
                    "effective_n",
                    arm_state.get("n", 0),
                )
                or 0.0
            ),
        }

    def _decay_arms(self, scope: dict[str, Any]) -> None:
        """兩臂舊統計一起衰減，避免靴前單次輸贏永久鎖邊。"""
        identity = np.eye(CONTEXT_DIM) * LINUCB_RIDGE
        arms = scope.setdefault("arms", {})
        for arm in ARMS:
            state = dict(arms.get(arm) or _new_arm())
            A, b = _arm_arrays(state)
            state["A"] = (
                identity + LINUCB_FORGETTING * (A - identity)
            ).tolist()
            state["b"] = (LINUCB_FORGETTING * b).tolist()
            state["effective_n"] = LINUCB_FORGETTING * float(
                state.get(
                    "effective_n",
                    state.get("n", 0),
                )
                or 0.0
            )
            arms[arm] = state

    def _update_scope(
        self,
        scope: dict[str, Any],
        *,
        action: str,
        context_vector: Sequence[float],
        actual_outcome: str,
    ) -> dict[str, Any]:
        action = str(action or "").upper().strip()
        actual = str(actual_outcome or "").upper().strip()
        if action not in ARMS or actual not in {"B", "P", "T"}:
            return {"updated": False, "reason": "invalid_feedback"}

        x = _model_x(context_vector)
        self._decay_arms(scope)
        scope["updates"] = int(scope.get("updates", 0) or 0) + 1
        scope["updated_at"] = int(time.time())

        if actual == "T":
            # 和局沒有 B/P 方向資訊，不把零 reward 樣本加入 A。
            return {
                "updated": True,
                "action": action,
                "actual_outcome": actual,
                "reward": 0.0,
                "directional_sample_applied": False,
                "forgetting": LINUCB_FORGETTING,
                "reason": "tie_reward_zero_no_directional_information",
            }

        reward = (
            (0.95 if action == "B" else 1.0)
            if action == actual
            else -1.0
        )

        state = dict(scope.get("arms", {}).get(action) or _new_arm())
        A, b = _arm_arrays(state)
        A = A + LINUCB_UPDATE_WEIGHT * np.outer(x, x)
        b = b + LINUCB_UPDATE_WEIGHT * reward * x

        state.update(
            {
                "A": A.tolist(),
                "b": b.tolist(),
                "n": int(state.get("n", 0) or 0) + 1,
                "effective_n": float(
                    state.get("effective_n", 0.0) or 0.0
                )
                + 1.0,
            }
        )
        scope.setdefault("arms", {})[action] = state

        return {
            "updated": True,
            "action": action,
            "actual_outcome": actual,
            "reward": reward,
            "directional_sample_applied": True,
            "update_weight": LINUCB_UPDATE_WEIGHT,
            "forgetting": LINUCB_FORGETTING,
            "context_l2_normalized": True,
        }

    def update(
        self,
        *,
        scope_key: str,
        action: str,
        context_vector: Sequence[float],
        actual_outcome: str,
        clear_pending: bool = True,
    ) -> dict[str, Any]:
        with _LOCK:
            root = _read_state()
            scope = dict(
                root["scopes"].get(scope_key)
                or _new_scope_state()
            )
            result = self._update_scope(
                scope,
                action=action,
                context_vector=context_vector,
                actual_outcome=actual_outcome,
            )
            if clear_pending:
                scope["pending"] = {}
            root["scopes"][scope_key] = scope
            _write_state(root)
            return result

    def _apply_pending(
        self,
        scope: dict[str, Any],
        raw_history: Sequence[str],
    ) -> dict[str, Any]:
        """最新歷史多一局時，用那一局結算上一筆 prediction。"""
        pending = dict(scope.get("pending") or {})
        if not pending:
            return {
                "updated": False,
                "reason": "no_pending_prediction",
            }

        previous_len = int(
            pending.get("raw_round_count", 0) or 0
        )
        if len(raw_history) <= previous_len:
            return {
                "updated": False,
                "reason": "no_new_resolved_round",
            }

        prefix = list(raw_history[:previous_len])
        if _history_fingerprint(prefix) != str(
            pending.get("history_fingerprint") or ""
        ):
            scope["pending"] = {}
            return {
                "updated": False,
                "reason": "history_reset_or_misaligned",
                "previous_len": previous_len,
                "current_len": len(raw_history),
            }

        result = self._update_scope(
            scope,
            action=str(pending.get("action") or ""),
            context_vector=list(
                pending.get("context_vector") or []
            ),
            actual_outcome=raw_history[previous_len],
        )
        scope["pending"] = {}
        result.update(
            {
                "history_aligned": True,
                "resolved_history_index": previous_len,
                "history_rounds_after_append": len(raw_history),
            }
        )
        return result

    def _choose(
        self,
        scope: Mapping[str, Any],
        scores: Mapping[str, Mapping[str, float]],
        scope_key: str,
        fingerprint: str,
    ) -> tuple[str, str]:
        gap = float(scores["B"]["score"]) - float(
            scores["P"]["score"]
        )
        if abs(gap) > LINUCB_SCORE_TIE_EPSILON:
            return (
                ("B" if gap > 0 else "P"),
                "max_ucb_score",
            )

        # 真正數值平手時才做 deterministic tie-break，
        # 不會覆寫有意義的 UCB 分數差。
        arms = dict(scope.get("arms") or {})
        b_n = float(
            (arms.get("B") or {}).get(
                "effective_n", 0.0
            )
            or 0.0
        )
        p_n = float(
            (arms.get("P") or {}).get(
                "effective_n", 0.0
            )
            or 0.0
        )
        if abs(b_n - p_n) > 1e-9:
            return (
                ("B" if b_n < p_n else "P"),
                "tie_less_sampled_arm",
            )

        last = str(
            scope.get("last_selected") or ""
        ).upper().strip()
        if last in ARMS:
            return (
                ("P" if last == "B" else "B"),
                "tie_opposite_previous_arm",
            )

        token = sha256(
            f"{scope_key}|{fingerprint}".encode("utf-8")
        ).digest()[0]
        return (
            ("B" if token % 2 else "P"),
            "tie_deterministic_balanced_hash",
        )

    def predict(
        self,
        *,
        history: Iterable[Any] | str | None,
        shoe_context: Mapping[str, Any] | None,
        scope_key: str,
    ) -> dict[str, Any]:
        raw = _normalize_history(history)
        snapshot = self.generator.build(raw, shoe_context)
        raw_x = snapshot.vector
        x = _model_x(raw_x)
        fingerprint = _history_fingerprint(raw)

        with _LOCK:
            root = _read_state()
            scope = dict(
                root["scopes"].get(scope_key)
                or _new_scope_state()
            )

            feedback = self._apply_pending(scope, raw)

            n_bp = sum(
                value in {"B", "P"} for value in raw
            )
            if n_bp < 8:
                base_scale = 1.35
            elif n_bp < 15:
                base_scale = 1.15
            else:
                base_scale = 1.0

            arms = dict(scope.get("arms") or {})
            effective_samples = {
                arm: max(
                    0.0,
                    float(
                        (arms.get(arm) or {}).get(
                            "effective_n",
                            (arms.get(arm) or {}).get(
                                "n", 0
                            ),
                        )
                        or 0.0
                    ),
                )
                for arm in ARMS
            }
            total_effective = sum(
                effective_samples.values()
            )

            scores: dict[str, dict[str, float]] = {}
            for arm in ARMS:
                imbalance = math.sqrt(
                    max(1.0, total_effective + 2.0)
                    / max(
                        1.0,
                        effective_samples[arm] + 1.0,
                    )
                )
                scale = base_scale * _clip(
                    imbalance,
                    0.85,
                    LINUCB_ARM_ALPHA_MAX_SCALE,
                )
                scores[arm] = self._score(
                    arms.get(arm, {}),
                    x,
                    scale,
                )
                scores[arm]["alpha_scale"] = scale

            direction, selection_reason = self._choose(
                scope,
                scores,
                scope_key,
                fingerprint,
            )
            other = "P" if direction == "B" else "B"
            margin = max(
                0.0,
                float(scores[direction]["score"])
                - float(scores[other]["score"]),
            )
            probability = _clip(
                0.50 + 0.08 * math.tanh(margin),
                PROBABILITY_MIN,
                PROBABILITY_MAX,
            )
            probabilities = (
                {
                    "B": probability,
                    "P": 1.0 - probability,
                    "T": 0.0,
                }
                if direction == "B"
                else {
                    "P": probability,
                    "B": 1.0 - probability,
                    "T": 0.0,
                }
            )

            previous = str(
                scope.get("last_selected") or ""
            ).upper().strip()
            streak = (
                int(scope.get("selection_streak", 0) or 0)
                + 1
                if previous == direction
                else 1
            )

            scope.update(
                {
                    "last_selected": direction,
                    "selection_streak": streak,
                    "updated_at": int(time.time()),
                    "pending": {
                        "action": direction,
                        "context_vector": [
                            float(value)
                            for value in raw_x
                        ],
                        "raw_round_count": len(raw),
                        "history_fingerprint": fingerprint,
                        "created_at": int(time.time()),
                    },
                }
            )
            root["scopes"][scope_key] = scope
            _write_state(root)

        return {
            "model": "road_primary_two_arm_dynamic_contextual_linucb",
            "version": STATE_VERSION,
            "direction": direction,
            "selected_arm": direction,
            "arm_index": 1 if direction == "B" else 0,
            "probabilities": probabilities,
            "selected_win_probability": probability,
            "confidence": probability,
            "context_vector": [
                float(value) for value in raw_x
            ],
            "model_context_vector": [
                float(value) for value in x
            ],
            "context_feature_names": list(
                CONTEXT_FEATURE_NAMES
            ),
            "context_dim": CONTEXT_DIM,
            "context_metadata": snapshot.metadata,
            "scores": scores,
            "alpha": self.alpha,
            "ridge": LINUCB_RIDGE,
            "forgetting": LINUCB_FORGETTING,
            "feedback_update": feedback,
            "scope_key": scope_key,
            "arms": list(ARMS),
            "selection_reason": selection_reason,
            "selection_streak": streak,
            "effective_arm_samples": effective_samples,
            "short_shoe_target_rounds": "50-70",
            "history_round_count": len(raw),
            "bp_history_round_count": n_bp,
            "history_fingerprint": fingerprint,
            "formal_context_source": (
                "screenshot_big_road_plus_manual_history"
            ),
            "card_composition_direction_weight": 0.0,
            "road_context_direction_weight": 1.0,
            "anti_lock": {
                "enabled": True,
                "method": (
                    "dynamic_forgetting_l2_context_"
                    "adaptive_exploration"
                ),
                "fixed_player_tie_bias_removed": True,
                "tie_is_non_directional": True,
                "old_v1_v2_v3_state_reused": False,
            },
        }


_DEFAULT_BANDIT = ContextualLinUCB()


def predict_bandit(
    *,
    history: Iterable[Any] | str | None,
    shoe_context: Mapping[str, Any] | None,
    scope_key: str,
) -> dict[str, Any]:
    return _DEFAULT_BANDIT.predict(
        history=history,
        shoe_context=shoe_context,
        scope_key=scope_key,
    )


def update_bandit(
    *,
    scope_key: str,
    action: str,
    context_vector: Sequence[float],
    actual_outcome: str,
    clear_pending: bool = True,
) -> dict[str, Any]:
    return _DEFAULT_BANDIT.update(
        scope_key=scope_key,
        action=action,
        context_vector=context_vector,
        actual_outcome=actual_outcome,
        clear_pending=clear_pending,
    )


__all__ = [
    "ARMS",
    "CONTEXT_DIM",
    "CONTEXT_FEATURE_NAMES",
    "ContextGenerator",
    "ContextualLinUCB",
    "ESTIMATED_CARDS_PER_ROUND",
    "LINUCB_ALPHA",
    "LINUCB_ARM_ALPHA_MAX_SCALE",
    "LINUCB_FORGETTING",
    "LINUCB_RIDGE",
    "LINUCB_SCORE_TIE_EPSILON",
    "LINUCB_UPDATE_WEIGHT",
    "PROBABILITY_MIN",
    "PROBABILITY_MAX",
    "SHOE_DECKS",
    "STATE_VERSION",
    "make_scope_key",
    "predict_bandit",
    "update_bandit",
]
