"""BGS CUSUM-LinUCB with 36D road/Markov/structural context.

The original 21D road context and 8D Laplace-smoothed Markov block are preserved.
Seven structural road features are appended. They summarize the latest 5-6 logical
Big-Road columns (runs), historical look-alike continuation/break behavior, column
rhythm/trend, and Big-Eye/Small/Cockroach resonance. LinUCB/CUSUM formulas and the
formal B/P-only output contract remain unchanged.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import json
import math
import time

import numpy as np
from road_model import build_road_context

ARMS = ("B", "P")
MODEL_VERSION = "CUSUM-LINUCB-V1.4-STRUCTURAL36-DYNAMIC-RESET-NO-OBSERVE"
STATE_SCHEMA_VERSION = "CUSUM-LINUCB-STATE-V5-STRUCTURAL36"

ROAD_FEATURE_NAMES = (
    "bias", "history_maturity", "global_banker_balance", "recent3_banker_balance",
    "recent8_banker_balance", "current_streak_direction", "current_streak_length",
    "alternation6", "alternation12", "transition_acceleration", "streak_break_signal",
    "long_dragon_tail_pressure", "observed_tie_rate", "road_planning_balance",
    "road_recent_balance", "road_confidence", "road_agreement", "big_eye_saturation",
    "small_road_saturation", "cockroach_road_saturation", "derived_road_consensus",
)
MARKOV_FEATURE_NAMES = (
    "markov_p_b_given_b", "markov_p_p_given_b",
    "markov_p_b_given_p", "markov_p_p_given_p",
    "markov_p_b_given_bb", "markov_p_b_given_pp",
    "markov_p_p_given_bb", "markov_p_p_given_pp",
)
STRUCTURAL_FEATURE_NAMES = (
    "road5_pattern_similarity",
    "road5_next_direction_edge",
    "road5_pattern_support",
    "road6_column_height_trend",
    "road6_rhythm_strength",
    "derived_road_resonance",
    "road6_regime_stability",
)
FEATURE_NAMES = ROAD_FEATURE_NAMES + MARKOV_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES
CONTEXT_DIM = len(FEATURE_NAMES)

CUSUM_ALPHA = 0.65
CUSUM_L2 = 1.0
CUSUM_FORGETTING_FACTOR = 0.985
CUSUM_DRIFT_V = 0.15
CUSUM_THRESHOLD_H = 4.50
CUSUM_MIN_OBSERVATIONS = 8
CUSUM_VACUUM_HANDS = 5
CUSUM_FORCE_OBSERVE_HANDS = 0
PREQUENTIAL_WARMUP_BP = 6
HISTORY_REPLAY_LIMIT = 120
MARKOV_ALPHA = 1.0
MARKOV_WINDOW_SIZE = 36
STRUCTURAL_SIGNATURE_COLUMNS = 5
STRUCTURAL_WINDOW_COLUMNS = 6
STRUCTURAL_NEIGHBORS = 18
STRUCTURAL_MIN_SIMILARITY = 0.48
STRUCTURAL_SUPPORT_TARGET = 8
TIE_PRIOR = 0.095156
TIE_PRIOR_STRENGTH = 40.0
_LOCK = RLock()
BASE_DIR = Path(__file__).resolve().parent


def _state_file() -> Path:
    for p in (
        Path("/var/data/contextual_bandit_state_cusum_v1.json"),
        BASE_DIR / "data" / "contextual_bandit_state_cusum_v1.json",
        Path("/tmp/bgs_contextual_bandit_state_cusum_v1.json"),
    ):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            q = p.parent / f".cusum_probe_{time.time_ns()}"
            q.write_text("ok", encoding="utf-8")
            q.unlink(missing_ok=True)
            return p
        except OSError:
            pass
    raise RuntimeError("No writable CUSUM state path")


CMAB_STATE_FILE = _state_file()


def _clip(v: Any, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(lo, min(hi, x)) if math.isfinite(x) else 0.0


def _clean(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in values:
        raw = item.get("outcome") if isinstance(item, Mapping) else item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out[-2000:]


class MarkovFeatureExtractor:
    """Original B/P first/second-order Markov features; semantics intentionally unchanged."""

    FEATURE_NAMES = MARKOV_FEATURE_NAMES

    def __init__(self, *, alpha: float = MARKOV_ALPHA, window_size: int = MARKOV_WINDOW_SIZE) -> None:
        self.alpha = max(1e-9, float(alpha))
        self.window_size = max(3, int(window_size))
        self._values: List[int] = []

    @staticmethod
    def _encode(value: Any) -> Optional[int]:
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return int(value)
        text = str(value or "").upper().strip()
        if text == "B":
            return 1
        if text == "P":
            return 0
        return None

    def reset(self) -> None:
        self._values = []

    def update(self, value: Any) -> None:
        encoded = self._encode(value)
        if encoded is None:
            return
        self._values.append(encoded)
        if len(self._values) > self.window_size:
            self._values = self._values[-self.window_size:]

    def extend(self, values: Iterable[Any]) -> None:
        for value in values:
            self.update(value)

    def _smoothed_pair(self, positive: int, negative: int) -> tuple[float, float]:
        denominator = float(positive + negative) + 2.0 * self.alpha
        return (
            (float(positive) + self.alpha) / denominator,
            (float(negative) + self.alpha) / denominator,
        )

    def extract_features(self) -> List[float]:
        values = list(self._values)
        bb = bp = pb = pp = 0
        for previous, current in zip(values, values[1:]):
            if previous == 1 and current == 1:
                bb += 1
            elif previous == 1 and current == 0:
                bp += 1
            elif previous == 0 and current == 1:
                pb += 1
            else:
                pp += 1
        p_b_given_b, p_p_given_b = self._smoothed_pair(bb, bp)
        p_b_given_p, p_p_given_p = self._smoothed_pair(pb, pp)

        b_after_bb = p_after_bb = b_after_pp = p_after_pp = 0
        for first, second, current in zip(values, values[1:], values[2:]):
            if first == 1 and second == 1:
                if current == 1:
                    b_after_bb += 1
                else:
                    p_after_bb += 1
            elif first == 0 and second == 0:
                if current == 1:
                    b_after_pp += 1
                else:
                    p_after_pp += 1
        p_b_given_bb, p_p_given_bb = self._smoothed_pair(b_after_bb, p_after_bb)
        p_b_given_pp, p_p_given_pp = self._smoothed_pair(b_after_pp, p_after_pp)
        return [
            p_b_given_b, p_p_given_b, p_b_given_p, p_p_given_p,
            p_b_given_bb, p_b_given_pp, p_p_given_bb, p_p_given_pp,
        ]

    def feature_dict(self) -> Dict[str, float]:
        return dict(zip(self.FEATURE_NAMES, self.extract_features()))

    def to_state(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha, "window_size": self.window_size,
            "values": list(self._values), "sample_count": len(self._values),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "MarkovFeatureExtractor":
        extractor = cls(
            alpha=state.get("alpha", MARKOV_ALPHA),
            window_size=state.get("window_size", MARKOV_WINDOW_SIZE),
        )
        try:
            extractor.extend(list(state.get("values") or []))
        except Exception:
            extractor.reset()
        return extractor

    @classmethod
    def from_sequence(
        cls, values: Iterable[Any], *, alpha: float = MARKOV_ALPHA,
        window_size: int = MARKOV_WINDOW_SIZE,
    ) -> "MarkovFeatureExtractor":
        extractor = cls(alpha=alpha, window_size=window_size)
        extractor.extend(values)
        return extractor


def _balance(seq: Sequence[str], n: Optional[int] = None) -> float:
    s = list(seq[-n:] if n else seq)
    return 0.0 if not s else _clip((sum(x == "B" for x in s) / len(s) - 0.5) * 2.0)


def _transition_rate(seq: Sequence[str], n: int) -> float:
    s = list(seq[-n:])
    return 0.0 if len(s) < 2 else sum(a != b for a, b in zip(s, s[1:])) / (len(s) - 1)


def _alternation(seq: Sequence[str], n: int) -> float:
    return _clip((_transition_rate(seq, n) - 0.5) * 2.0)


def _streak(seq: Sequence[str]) -> tuple[str, int]:
    if not seq:
        return "", 0
    side, length = seq[-1], 1
    for x in reversed(seq[:-1]):
        if x != side:
            break
        length += 1
    return side, length


def _streak_break(seq: Sequence[str]) -> float:
    s = list(seq)
    if len(s) < 4 or s[-1] == s[-2]:
        return 0.0
    old, run = s[-2], 1
    for x in reversed(s[:-2]):
        if x != old:
            break
        run += 1
    if run < 3:
        return 0.0
    return (1.0 if s[-1] == "B" else -1.0) * min(1.0, run / 6.0)


def _road_saturation(road: Mapping[str, Any], name: str) -> float:
    planning = road.get("full_road_analysis")
    if not isinstance(planning, Mapping):
        models = road.get("models")
        planning = models.get("full_road") if isinstance(models, Mapping) else {}
    stats = dict(planning.get("derived_stats") or {}).get(name) if isinstance(planning, Mapping) else None
    if not isinstance(stats, Mapping):
        return 0.0
    balance = _clip(stats.get("balance", 0.0), 0.0, 1.0)
    cont = _clip(stats.get("recent_continuation", 0.5), 0.0, 1.0)
    return max(balance, abs(2.0 * cont - 1.0))


def _prob_balance(v: Any) -> float:
    try:
        p = float(v)
    except (TypeError, ValueError):
        p = 0.5
    return _clip((p - 0.5) * 2.0)


def _logical_runs(sequence: Sequence[str]) -> List[List[Any]]:
    runs: List[List[Any]] = []
    for value in sequence:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return runs


def _run_signature(runs: Sequence[Sequence[Any]], count: int = STRUCTURAL_SIGNATURE_COLUMNS) -> np.ndarray:
    tail = [min(6, int(item[1])) / 6.0 for item in list(runs)[-count:]]
    if len(tail) < count:
        tail = [0.0] * (count - len(tail)) + tail
    return np.asarray(tail, dtype=np.float64)


def _derived_road_resonance(road: Mapping[str, Any]) -> float:
    planning = road.get("full_road_analysis")
    if not isinstance(planning, Mapping):
        models = road.get("models")
        planning = models.get("full_road") if isinstance(models, Mapping) else {}
    derived = planning.get("derived_roads") if isinstance(planning, Mapping) else {}
    derived = dict(derived) if isinstance(derived, Mapping) else {}
    signed_strengths: List[float] = []
    for name in ("big_eye", "small_road", "cockroach_road"):
        values = list(derived.get(name) or [])[-5:]
        values = [str(v).upper() for v in values if str(v).upper() in {"R", "U"}]
        if len(values) < 2:
            continue
        r_ratio = values.count("R") / len(values)
        signed_strengths.append(2.0 * r_ratio - 1.0)
    if not signed_strengths:
        return 0.0
    mean_abs = sum(abs(v) for v in signed_strengths) / len(signed_strengths)
    directional_agreement = abs(sum(signed_strengths)) / max(1e-9, sum(abs(v) for v in signed_strengths))
    depth = min(1.0, len(signed_strengths) / 3.0)
    return _clip(mean_abs * directional_agreement * depth, 0.0, 1.0)


def _structural_road_features(sequence: Sequence[str], road: Mapping[str, Any]) -> tuple[List[float], Dict[str, Any]]:
    """Extract seven non-redundant 5/6-column structural features.

    Logical columns are consecutive B/P runs. The latest five column heights form the
    matching signature; the sixth column supplies trend/rhythm context. Historical
    matches are strictly prequential: a prefix signature is compared with the current
    signature and labelled by the next already-observed hand.
    """
    seq = [x for x in sequence if x in ARMS]
    runs = _logical_runs(seq)
    if len(runs) < STRUCTURAL_SIGNATURE_COLUMNS:
        zeros = [0.0] * len(STRUCTURAL_FEATURE_NAMES)
        return zeros, {
            "ready": False, "logical_column_count": len(runs),
            "signature_columns": STRUCTURAL_SIGNATURE_COLUMNS,
            "window_columns": STRUCTURAL_WINDOW_COLUMNS,
            "continuation_probability": 0.5, "break_probability": 0.5,
            "candidate_count": 0, "support_count": 0,
        }

    target = _run_signature(runs)
    candidates: List[tuple[float, bool]] = []
    prefix_runs: List[List[Any]] = []
    for index, actual_next in enumerate(seq):
        if index > 0 and len(prefix_runs) >= STRUCTURAL_SIGNATURE_COLUMNS:
            signature = _run_signature(prefix_runs)
            distance = float(np.sqrt(np.mean(np.square(target - signature))))
            similarity = float(math.exp(-5.2 * distance))
            continued = actual_next == seq[index - 1]
            candidates.append((similarity, continued))
        if prefix_runs and prefix_runs[-1][0] == actual_next:
            prefix_runs[-1][1] += 1
        else:
            prefix_runs.append([actual_next, 1])

    candidates.sort(key=lambda item: item[0], reverse=True)
    neighbors = candidates[:STRUCTURAL_NEIGHBORS]
    continue_mass = 2.0
    break_mass = 2.0
    strong = 0
    similarity_total = 0.0
    for similarity, continued in neighbors:
        similarity_total += similarity
        if similarity >= STRUCTURAL_MIN_SIMILARITY:
            strong += 1
        if continued:
            continue_mass += similarity
        else:
            break_mass += similarity
    continuation_probability = continue_mass / max(1e-9, continue_mass + break_mass)
    break_probability = 1.0 - continuation_probability
    mean_similarity = similarity_total / max(1, len(neighbors))
    support = min(1.0, strong / float(STRUCTURAL_SUPPORT_TARGET))
    pattern_similarity = _clip(mean_similarity * (0.45 + 0.55 * support), 0.0, 1.0)

    last_side = seq[-1] if seq else ""
    p_b = continuation_probability if last_side == "B" else 1.0 - continuation_probability
    next_direction_edge = _clip((2.0 * p_b - 1.0) * support)

    heights = [min(6, int(item[1])) for item in runs[-STRUCTURAL_WINDOW_COLUMNS:]]
    if len(heights) >= 2:
        trend = _clip((heights[-1] - heights[0]) / 5.0)
    else:
        trend = 0.0

    if len(heights) >= 4:
        period2 = sum(1.0 - min(1.0, abs(heights[i] - heights[i - 2]) / 5.0) for i in range(2, len(heights))) / (len(heights) - 2)
        single = sum(h == 1 for h in heights) / len(heights)
        double = sum(h == 2 for h in heights) / len(heights)
        rhythm = _clip(max(period2, single, double), 0.0, 1.0)
    else:
        rhythm = 0.0

    resonance = _derived_road_resonance(road)
    regime_stability = _clip(0.42 * rhythm + 0.33 * resonance + 0.25 * support, 0.0, 1.0)
    features = [pattern_similarity, next_direction_edge, support, trend, rhythm, resonance, regime_stability]
    diagnostics = {
        "ready": True,
        "logical_column_count": len(runs),
        "recent_column_heights": heights,
        "signature_heights": [int(item[1]) for item in runs[-STRUCTURAL_SIGNATURE_COLUMNS:]],
        "signature_columns": STRUCTURAL_SIGNATURE_COLUMNS,
        "window_columns": STRUCTURAL_WINDOW_COLUMNS,
        "candidate_count": len(candidates),
        "neighbor_count": len(neighbors),
        "support_count": strong,
        "pattern_similarity": pattern_similarity,
        "continuation_probability": continuation_probability,
        "break_probability": break_probability,
        "next_direction_edge": next_direction_edge,
        "column_height_trend": trend,
        "rhythm_strength": rhythm,
        "derived_road_resonance": resonance,
        "regime_stability": regime_stability,
    }
    return features, diagnostics


def build_context_vector(
    history: Iterable[Any], *, road_context: Optional[Mapping[str, Any]] = None,
    markov_features: Optional[Sequence[float]] = None,
) -> List[float]:
    raw = _clean(history)
    bp = [x for x in raw if x in ARMS]
    road = dict(road_context or {})
    side, run = _streak(bp)
    streak_sign = 1.0 if side == "B" else -1.0 if side == "P" else 0.0
    tie_rate = sum(x == "T" for x in raw) / max(1, len(raw))
    disagreement = _clip(road.get("recent_model_disagreement", road.get("model_disagreement", 0.20)), 0.0, 1.0)
    big = _road_saturation(road, "big_eye")
    small = _road_saturation(road, "small_road")
    cock = _road_saturation(road, "cockroach_road")
    mean = (big + small + cock) / 3.0
    consensus = _clip(mean * (1.0 - (abs(big-small)+abs(small-cock)+abs(cock-big))/3.0), 0.0, 1.0)

    road_values = [
        1.0, min(1.0, len(bp)/60.0), _balance(bp), _balance(bp,3), _balance(bp,8),
        streak_sign, min(1.0, run/8.0), _alternation(bp,6), _alternation(bp,12),
        _clip(_transition_rate(bp,6)-_transition_rate(bp,14)), _streak_break(bp),
        streak_sign * min(1.0, max(0, run-3)/5.0), _clip(tie_rate/0.20, 0.0, 1.0),
        _prob_balance(road.get("planning_probability",0.5)), _prob_balance(road.get("recent_probability",0.5)),
        _clip(road.get("confidence_score",0.0),0.0,1.0), _clip(1.0-min(1.0,disagreement/0.20),0.0,1.0),
        big, small, cock, consensus,
    ]
    if len(road_values) != len(ROAD_FEATURE_NAMES):
        raise RuntimeError("Road context dimension mismatch")

    if markov_features is None:
        markov_values = MarkovFeatureExtractor.from_sequence(bp).extract_features()
    else:
        markov_values = [float(value) for value in markov_features]
    if len(markov_values) != len(MARKOV_FEATURE_NAMES):
        raise ValueError("markov_features must contain exactly 8 values")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in markov_values):
        raise ValueError("markov_features must be finite probabilities in [0, 1]")

    structural_values, _ = _structural_road_features(bp, road)
    values = road_values + markov_values + structural_values
    if len(values) != CONTEXT_DIM:
        raise RuntimeError("CUSUM context dimension mismatch")
    return [round(_clip(v), 10) for v in values]


def _vec(context: Sequence[float]) -> np.ndarray:
    x = np.asarray(list(context), dtype=np.float64)
    if x.shape != (CONTEXT_DIM,) or not np.all(np.isfinite(x)):
        raise ValueError(f"context must be finite {CONTEXT_DIM}-vector")
    return np.clip(x, -1.0, 1.0)


def _softmax(b: float, p: float) -> Dict[str, float]:
    z = np.asarray([b,p], dtype=np.float64) / 0.85
    z -= np.max(z)
    e = np.exp(np.clip(z,-40,40))
    e /= max(1e-12, float(e.sum()))
    return {"B": float(e[0]), "P": float(e[1])}


def _tie_prob(raw: Sequence[str]) -> float:
    p = (sum(x=="T" for x in raw) + TIE_PRIOR*TIE_PRIOR_STRENGTH) / (len(raw)+TIE_PRIOR_STRENGTH)
    return max(0.04, min(0.18, float(p)))


class CUSUMLinUCB:
    """Dynamic two-arm LinUCB with two-sided CUSUM and active hard reset."""
    def __init__(self, *, alpha: float=CUSUM_ALPHA, l2: float=CUSUM_L2,
                 forgetting_factor: float=CUSUM_FORGETTING_FACTOR,
                 cusum_h: float=CUSUM_THRESHOLD_H, cusum_v: float=CUSUM_DRIFT_V,
                 min_cusum_observations: int=CUSUM_MIN_OBSERVATIONS,
                 vacuum_hands: int=CUSUM_VACUUM_HANDS) -> None:
        self.alpha=float(alpha)
        self.l2=max(1e-9,float(l2))
        self.forgetting_factor=max(0.80,min(1.0,float(forgetting_factor)))
        self.cusum_h=max(0.5,float(cusum_h))
        self.cusum_v=max(0.0,float(cusum_v))
        self.min_cusum_observations=max(2,int(min_cusum_observations))
        self.vacuum_hands=max(1,int(vacuum_hands))
        self.total_observations=0
        self.observations_since_reset=0
        self.reset_count=0
        self.g_plus=0.0
        self.g_minus=0.0
        self.last_residual=0.0
        self.last_expected_reward=0.0
        self.last_observed_reward=0.0
        self.last_reset: Dict[str,Any]={}
        self.applied_event_ids: List[str]=[]
        self.markov_extractor = MarkovFeatureExtractor()
        self._fresh_matrices()

    def _fresh_matrices(self) -> None:
        I=np.eye(CONTEXT_DIM,dtype=np.float64)*self.l2
        z=np.zeros(CONTEXT_DIM,dtype=np.float64)
        self.arms={a:{"A":I.copy(),"b":z.copy(),"updates":0,"reward_sum":0.0} for a in ARMS}
        self.context_information={"A":I.copy(),"updates":0}

    @staticmethod
    def _pinv(A: np.ndarray) -> np.ndarray:
        A=0.5*(A+A.T)
        try:
            inv=np.linalg.pinv(A,rcond=1e-10,hermitian=True)
        except TypeError:
            inv=np.linalg.pinv(A,rcond=1e-10)
        if not np.all(np.isfinite(inv)):
            inv=np.linalg.pinv(A+np.eye(A.shape[0])*1e-6,rcond=1e-8)
        return inv

    def arm_metrics(self, arm: str, context: Sequence[float]) -> Dict[str,float]:
        x=_vec(context)
        s=self.arms[arm]
        inv=self._pinv(np.asarray(s["A"],dtype=float))
        theta=inv@np.asarray(s["b"],dtype=float)
        mean=float(theta@x)
        var=max(0.0,float(x@inv@x))
        std=math.sqrt(var)
        bonus=self.alpha*std
        return {"expected_reward":mean,"mean_reward":mean,"variance":var,"uncertainty":std,
                "ucb_bonus":bonus,"ucb_score":mean+bonus,"updates":int(s["updates"]),"reward_sum":float(s["reward_sum"])}

    def predict_context(self, context: Sequence[float]) -> Dict[str,Any]:
        x=_vec(context)
        m={a:self.arm_metrics(a,x) for a in ARMS}
        b,p=m["B"]["ucb_score"],m["P"]["ucb_score"]
        selected="B" if b>=p else "P"
        inv=self._pinv(np.asarray(self.context_information["A"],dtype=float))
        var=max(0.0,float(x@inv@x))
        return {"selected_arm":selected,"metrics":m,"conditional_probabilities":_softmax(b,p),
                "shared_uncertainty":{"variance":var,"uncertainty":math.sqrt(var),"updates":int(self.context_information["updates"])}}

    def _cusum(self, observed: float, expected: float) -> Dict[str,Any]:
        residual=float(observed-expected)
        self.last_residual=residual
        self.last_observed_reward=float(observed)
        self.last_expected_reward=float(expected)
        self.g_plus=max(0.0,self.g_plus+residual-self.cusum_v)
        self.g_minus=max(0.0,self.g_minus-residual-self.cusum_v)
        ready=self.observations_since_reset>=self.min_cusum_observations and self.total_observations>=self.min_cusum_observations
        plus=bool(ready and self.g_plus>self.cusum_h)
        minus=bool(ready and self.g_minus>self.cusum_h)
        return {"residual":residual,"g_plus":self.g_plus,"g_minus":self.g_minus,"ready":ready,
                "alarm":plus or minus,"alarm_side":"positive" if plus else "negative" if minus else "",
                "threshold_h":self.cusum_h,"drift_v":self.cusum_v}

    def reset_model(self, *, reason: str="cusum_change_point", alarm_side: str="", residual: Optional[float]=None) -> Dict[str,Any]:
        self.reset_count+=1
        event={"triggered":True,"reason":reason,"alarm_side":alarm_side,
               "residual":float(self.last_residual if residual is None else residual),
               "g_plus_before_reset":float(self.g_plus),"g_minus_before_reset":float(self.g_minus),
               "threshold_h":self.cusum_h,"drift_v":self.cusum_v,
               "at_total_observation":self.total_observations,"reset_count":self.reset_count,"timestamp":int(time.time())}
        self._fresh_matrices()
        self.markov_extractor.reset()
        self.observations_since_reset=0
        self.g_plus=0.0
        self.g_minus=0.0
        self.last_reset=event
        return dict(event)

    def _update(self, x: np.ndarray, rewards: Mapping[str,float]) -> None:
        I=np.eye(CONTEXT_DIM)*self.l2
        outer=np.outer(x,x)
        lam=self.forgetting_factor
        for arm,reward in rewards.items():
            s=self.arms[arm]
            A=np.asarray(s["A"],dtype=float)
            b=np.asarray(s["b"],dtype=float)
            A=lam*(0.5*(A+A.T))+(1-lam)*I+outer
            b=lam*b+float(reward)*x
            s.update(A=0.5*(A+A.T),b=b,updates=int(s["updates"])+1,reward_sum=float(s["reward_sum"])+float(reward))
        info=self.context_information
        A=np.asarray(info["A"],dtype=float)
        A=lam*(0.5*(A+A.T))+(1-lam)*I+outer
        info["A"]=0.5*(A+A.T)
        info["updates"]=int(info["updates"])+1

    def observe(self, context: Sequence[float], actual_outcome: str, *, selected_arm: str="") -> Dict[str,Any]:
        actual=str(actual_outcome or "").upper().strip()
        if actual not in ARMS:
            return {"updated":False,"reason":"tie_or_invalid_outcome"}
        x=_vec(context)
        pred=self.predict_context(x)
        chosen=str(selected_arm or pred["selected_arm"]).upper()
        if chosen not in ARMS:
            chosen=pred["selected_arm"]
        expected=_clip(pred["metrics"][chosen]["expected_reward"])
        observed=1.0 if chosen==actual else -1.0
        c=self._cusum(observed,expected)
        reset={}
        if c["alarm"]:
            reset=self.reset_model(reason="cusum_residual_change_point",alarm_side=c["alarm_side"],residual=c["residual"])
        self._update(x,{a:(1.0 if a==actual else -1.0) for a in ARMS})
        self.markov_extractor.update(actual)
        self.total_observations+=1
        self.observations_since_reset+=1
        return {"updated":True,"actual_outcome":actual,"selected_arm":chosen,"observed_reward":observed,
                "expected_reward":expected,"cusum":c,"reset_triggered":bool(reset),"reset_event":reset}

    def observe_reward(self, context: Sequence[float], *, selected_arm: str, reward: float) -> Dict[str,Any]:
        arm=str(selected_arm).upper()
        x=_vec(context)
        expected=_clip(self.arm_metrics(arm,x)["expected_reward"])
        r=_clip(reward)
        c=self._cusum(r,expected)
        reset={}
        if c["alarm"]:
            reset=self.reset_model(reason="cusum_selected_arm_reward_change_point",alarm_side=c["alarm_side"],residual=c["residual"])
        self._update(x,{arm:r})
        self.total_observations+=1
        self.observations_since_reset+=1
        return {"updated":True,"selected_arm":arm,"reward":r,"expected_reward":expected,"cusum":c,"reset_triggered":bool(reset),"reset_event":reset}

    def risk_status(self, context: Sequence[float]) -> Dict[str,Any]:
        shared=self.predict_context(context)["shared_uncertainty"]
        uncertainty=float(shared["uncertainty"])
        info=1.0/(1.0+uncertainty)
        ref=self.observations_since_reset if self.reset_count else self.total_observations
        maturity=min(1.0,ref/12.0)
        confidence=_clip(0.10+0.55*info+0.35*maturity,0.0,0.90)
        vacuum=bool(self.reset_count and self.observations_since_reset<=self.vacuum_hands)
        if vacuum:
            confidence=min(confidence,min(0.48,0.08+0.08*self.observations_since_reset))
            weight=min(0.18,0.04+0.03*self.observations_since_reset)
            if self.observations_since_reset <= 2:
                bet=0.35
            elif self.observations_since_reset == 3:
                bet=0.45
            elif self.observations_since_reset == 4:
                bet=0.50
            else:
                bet=0.60
        elif self.total_observations<PREQUENTIAL_WARMUP_BP:
            weight,bet=0.08,0.50
        else:
            weight,bet=min(0.45,0.15+0.35*confidence),min(1.0,0.55+0.50*confidence)
        return {"confidence_score":confidence,"post_reset_vacuum_active":vacuum,"vacuum_hands_required":self.vacuum_hands,
                "observations_since_reset":self.observations_since_reset,"force_observe":False,"bet_multiplier":bet,
                "ensemble_weight_suggestion":weight,"uncertainty":uncertainty,"variance":float(shared["variance"]),"maturity":maturity}

    def to_state(self) -> Dict[str,Any]:
        return {"version":MODEL_VERSION,"context_dim":CONTEXT_DIM,"alpha":self.alpha,"l2":self.l2,
                "forgetting_factor":self.forgetting_factor,"cusum_h":self.cusum_h,"cusum_v":self.cusum_v,
                "min_cusum_observations":self.min_cusum_observations,"vacuum_hands":self.vacuum_hands,
                "total_observations":self.total_observations,"observations_since_reset":self.observations_since_reset,
                "reset_count":self.reset_count,"g_plus":self.g_plus,"g_minus":self.g_minus,"last_residual":self.last_residual,
                "last_expected_reward":self.last_expected_reward,"last_observed_reward":self.last_observed_reward,
                "last_reset":self.last_reset,"applied_event_ids":self.applied_event_ids[-5000:],
                "markov_extractor":self.markov_extractor.to_state(),
                "arms":{a:{"A":np.asarray(self.arms[a]["A"]).tolist(),"b":np.asarray(self.arms[a]["b"]).tolist(),
                            "updates":int(self.arms[a]["updates"]),"reward_sum":float(self.arms[a]["reward_sum"])} for a in ARMS},
                "context_information":{"A":np.asarray(self.context_information["A"]).tolist(),"updates":int(self.context_information["updates"])}}

    @classmethod
    def from_state(cls, state: Mapping[str,Any]) -> "CUSUMLinUCB":
        if not isinstance(state,Mapping) or int(state.get("context_dim",0) or 0)!=CONTEXT_DIM:
            return cls()
        m=cls(alpha=state.get("alpha",CUSUM_ALPHA),l2=state.get("l2",CUSUM_L2),forgetting_factor=state.get("forgetting_factor",CUSUM_FORGETTING_FACTOR),
              cusum_h=state.get("cusum_h",CUSUM_THRESHOLD_H),cusum_v=state.get("cusum_v",CUSUM_DRIFT_V),
              min_cusum_observations=state.get("min_cusum_observations",CUSUM_MIN_OBSERVATIONS),vacuum_hands=state.get("vacuum_hands",CUSUM_VACUUM_HANDS))
        try:
            for a in ARMS:
                A=np.asarray(state["arms"][a]["A"],dtype=float)
                b=np.asarray(state["arms"][a]["b"],dtype=float)
                if A.shape!=(CONTEXT_DIM,CONTEXT_DIM) or b.shape!=(CONTEXT_DIM,):
                    raise ValueError
                m.arms[a]={"A":0.5*(A+A.T),"b":b,"updates":int(state["arms"][a].get("updates",0)),"reward_sum":float(state["arms"][a].get("reward_sum",0.0))}
            A=np.asarray(state["context_information"]["A"],dtype=float)
            if A.shape!=(CONTEXT_DIM,CONTEXT_DIM):
                raise ValueError
            m.context_information={"A":0.5*(A+A.T),"updates":int(state["context_information"].get("updates",0))}
            m.total_observations=int(state.get("total_observations",0))
            m.observations_since_reset=int(state.get("observations_since_reset",0))
            m.reset_count=int(state.get("reset_count",0))
            m.g_plus=float(state.get("g_plus",0.0))
            m.g_minus=float(state.get("g_minus",0.0))
            m.last_residual=float(state.get("last_residual",0.0))
            m.last_expected_reward=float(state.get("last_expected_reward",0.0))
            m.last_observed_reward=float(state.get("last_observed_reward",0.0))
            m.last_reset=dict(state.get("last_reset") or {})
            m.applied_event_ids=[str(x) for x in list(state.get("applied_event_ids") or [])][-5000:]
            m.markov_extractor=MarkovFeatureExtractor.from_state(dict(state.get("markov_extractor") or {}))
        except Exception:
            return cls()
        return m


def _safe_road(history: Sequence[str]) -> Dict[str,Any]:
    try:
        return dict(build_road_context(history,initial_image_count=len(history),manual_count=0) or {})
    except Exception:
        return {}


def _replay(raw: Sequence[str]) -> tuple[CUSUMLinUCB,Dict[str,Any]]:
    history=list(raw)[-HISTORY_REPLAY_LIMIT:]
    model=CUSUMLinUCB()
    prefix: List[str]=[]
    bp_before=0
    resets=[]
    replayed=0
    for actual in history:
        if actual in ARMS:
            if bp_before>=PREQUENTIAL_WARMUP_BP:
                context=build_context_vector(prefix,road_context=_safe_road(prefix),markov_features=model.markov_extractor.extract_features())
                update=model.observe(context,actual)
                if update.get("reset_triggered"):
                    resets.append(dict(update.get("reset_event") or {}))
                replayed+=1
            else:
                model.markov_extractor.update(actual)
        prefix.append(actual)
        bp_before+=int(actual in ARMS)
    return model,{"raw_round_count":len(history),"bp_training_samples":replayed,"reset_count":model.reset_count,
                  "reset_events":resets[-20:],"history_fingerprint":sha256("".join(history).encode()).hexdigest()[:24],
                  "mode":"prequential_cusum_dynamic_linucb_markov29_structural36","markov_window_size":model.markov_extractor.window_size,
                  "markov_sample_count":len(model.markov_extractor.to_state()["values"]),
                  "structural_signature_columns":STRUCTURAL_SIGNATURE_COLUMNS,"structural_window_columns":STRUCTURAL_WINDOW_COLUMNS}


def predict_bandit(history: Iterable[Any], *, road_context: Optional[Mapping[str,Any]]=None, venue: str="", room: str="", user_id: str="", run_seed: Optional[int]=None) -> Dict[str,Any]:
    del run_seed
    raw=_clean(history)
    road=dict(road_context or {}) or _safe_road(raw)
    model,replay=_replay(raw)
    markov_values=model.markov_extractor.extract_features()
    structural_values, structural_state = _structural_road_features([x for x in raw if x in ARMS], road)
    context=build_context_vector(raw,road_context=road,markov_features=markov_values)
    pred=model.predict_context(context)
    risk=model.risk_status(context)
    selected=pred["selected_arm"]
    conditional=pred["conditional_probabilities"]
    tie=_tie_prob(raw)
    bp=1-tie
    probs={"B":conditional["B"]*bp,"P":conditional["P"]*bp,"T":tie}
    action=selected
    metrics=pred["metrics"]
    shared=pred["shared_uncertainty"]
    fingerprint=sha256(json.dumps({"history":"".join(raw),"venue":venue.upper(),"room":room,"context":context},sort_keys=True).encode()).hexdigest()[:24]
    cusum={"g_plus":model.g_plus,"g_minus":model.g_minus,"h":model.cusum_h,"v":model.cusum_v,"last_residual":model.last_residual,
           "last_expected_reward":model.last_expected_reward,"last_observed_reward":model.last_observed_reward,"reset_count":model.reset_count,"last_reset":model.last_reset}
    return {"ok":True,"engine":"CUSUM_LINUCB_DYNAMIC_CONTEXTUAL_BANDIT","model_version":MODEL_VERSION,"model_core":"cusum_linucb_markov29_plus_structural7_36d_dynamic_reset_no_observe",
            "prediction_fingerprint":fingerprint,"road_support":road,"probabilities":probs,"bandit_learning_probabilities":probs,
            "banker_rate":round(probs["B"]*100,2),"player_rate":round(probs["P"]*100,2),"tie_rate":round(probs["T"]*100,2),
            "selected_arm":selected,"base_bandit_direction":selected,"recommend":action,"recommend_text":"莊" if selected=="B" else "閒",
            "action":action,"action_text":"莊" if selected=="B" else "閒","internal_recommend":selected,"internal_action":selected,
            "next_round_direction":selected,"next_round_direction_text":"莊" if selected=="B" else "閒","signal_allowed":True,
            "signal_status_code":"CUSUM_LINUCB_DIRECTION","direction_source":"cusum_linucb",
            "direction_edge":abs(conditional["B"]-conditional["P"]),"confidence_score":risk["confidence_score"],"confidence":risk["confidence_score"],"quality_score":risk["confidence_score"],
            "post_reset_vacuum_active":risk["post_reset_vacuum_active"],"force_observe":False,"hands_since_reset":risk["observations_since_reset"],
            "ensemble_weight_suggestion":risk["ensemble_weight_suggestion"],"bet_multiplier":risk["bet_multiplier"],"risk_control":risk,"cusum":cusum,"reset_triggered":bool(model.last_reset),
            "uncertainty":shared["uncertainty"],"variance":shared["variance"],"variance_safe":True,"unknown_region_active":risk["post_reset_vacuum_active"],
            "is_extreme_unseen":False,"hard_brake_active":False,"uncertainty_braking":{"active":False,"is_extreme_unseen":False,"variance":shared["variance"],
            "action_space_variance":shared["variance"],"action_space_std":shared["uncertainty"],"variance_safe":True,"post_reset_vacuum_active":risk["post_reset_vacuum_active"],"confidence_score":risk["confidence_score"],"bet_multiplier":risk["bet_multiplier"],"observe_required":False,"cusum":cusum},
            "markov_features":dict(zip(MARKOV_FEATURE_NAMES,markov_values)),"markov_state":model.markov_extractor.to_state(),
            "structural_features":dict(zip(STRUCTURAL_FEATURE_NAMES,structural_values)),"structural_state":structural_state,
            "bandit_context":context,"context_vector":context,"context_feature_names":list(FEATURE_NAMES),"bandit_scores":metrics,
            "bandit_state":{"alpha":CUSUM_ALPHA,"l2":CUSUM_L2,"forgetting_factor":CUSUM_FORGETTING_FACTOR,"context_dim":CONTEXT_DIM,"total_updates":model.total_observations,
            "observations_since_reset":model.observations_since_reset,"reset_count":model.reset_count,"cusum_h":CUSUM_THRESHOLD_H,"cusum_v":CUSUM_DRIFT_V,
            "vacuum_hands":CUSUM_VACUUM_HANDS,"force_observe_hands":CUSUM_FORCE_OBSERVE_HANDS,"history_replay":replay,"state_file":str(CMAB_STATE_FILE),
            "markov_alpha":MARKOV_ALPHA,"markov_window_size":MARKOV_WINDOW_SIZE,"structural_signature_columns":STRUCTURAL_SIGNATURE_COLUMNS,"structural_window_columns":STRUCTURAL_WINDOW_COLUMNS},
            "adaptive_ensemble":{"active":False,"suggested_share":risk["ensemble_weight_suggestion"],"reason":"predictor.py performs final fusion"},
            "venue":venue,"room":room,"user_id":user_id,"input_required":False,"probability_semantics":"normalized_model_score_not_guaranteed_outcome_probability"}


def _uid(user_id: str) -> str:
    return sha256((str(user_id or "").strip() or "__anonymous__").encode()).hexdigest()[:24]


def _read_store() -> Dict[str,Any]:
    try:
        d=json.loads(CMAB_STATE_FILE.read_text(encoding="utf-8"))
        if d.get("schema_version")==STATE_SCHEMA_VERSION and isinstance(d.get("users"),dict):
            return d
    except Exception:
        pass
    return {"schema_version":STATE_SCHEMA_VERSION,"version":MODEL_VERSION,"context_dim":CONTEXT_DIM,"users":{}}


def _write_store(d: Mapping[str,Any]) -> None:
    x=dict(d)
    x.update(schema_version=STATE_SCHEMA_VERSION,version=MODEL_VERSION,context_dim=CONTEXT_DIM,updated_at=int(time.time()))
    tmp=CMAB_STATE_FILE.with_suffix(CMAB_STATE_FILE.suffix+".tmp")
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(CMAB_STATE_FILE)


def update_bandit(*, context: Sequence[float], selected_arm: str, reward: Optional[float], event_id: str="", actual_outcome: str="", update_weight: float=1.0,
                  user_id: str="", prediction_probabilities: Optional[Mapping[str,Any]]=None) -> Dict[str,Any]:
    del update_weight,prediction_probabilities
    arm=str(selected_arm).upper()
    event=str(event_id or "")
    if arm not in ARMS:
        raise ValueError("selected_arm must be B or P")
    with _LOCK:
        store=_read_store()
        key=_uid(user_id)
        model=CUSUMLinUCB.from_state(dict(store["users"].get(key) or {}))
        if event and event in model.applied_event_ids:
            return {"updated":False,"reason":"duplicate_event","event_id":event}
        actual=str(actual_outcome).upper()
        if actual in ARMS:
            result=model.observe(context,actual,selected_arm=arm)
        elif reward is not None and math.isfinite(float(reward)):
            result=model.observe_reward(context,selected_arm=arm,reward=float(reward))
        else:
            return {"updated":False,"reason":"tie_or_skipped_reward","event_id":event}
        if event:
            model.applied_event_ids=(model.applied_event_ids+[event])[-5000:]
        store["users"][key]=model.to_state()
        _write_store(store)
    return {**result,"event_id":event,"model_version":MODEL_VERSION,"reset_count":model.reset_count,"hands_since_reset":model.observations_since_reset}


def get_bandit_summary(user_id: str="") -> Dict[str,Any]:
    with _LOCK:
        model=CUSUMLinUCB.from_state(dict(_read_store()["users"].get(_uid(user_id)) or {}))
    return {"version":MODEL_VERSION,"context_dim":CONTEXT_DIM,"feature_names":list(FEATURE_NAMES),"total_updates":model.total_observations,
            "observations_since_reset":model.observations_since_reset,"reset_count":model.reset_count,
            "markov_features":model.markov_extractor.feature_dict(),"markov_state":model.markov_extractor.to_state(),
            "structural_feature_names":list(STRUCTURAL_FEATURE_NAMES),
            "cusum":{"g_plus":model.g_plus,"g_minus":model.g_minus,"h":model.cusum_h,"v":model.cusum_v,"last_residual":model.last_residual,"last_reset":model.last_reset},
            "arms":{a:{"updates":model.arms[a]["updates"],"reward_sum":model.arms[a]["reward_sum"]} for a in ARMS},"state_file":str(CMAB_STATE_FILE)}


class ContextualBanditEngine:
    def predict(self, history: Iterable[Any], **kwargs: Any) -> Dict[str,Any]:
        return predict_bandit(history,**kwargs)
    def update(self, **kwargs: Any) -> Dict[str,Any]:
        return update_bandit(**kwargs)
    def summary(self, user_id: str="") -> Dict[str,Any]:
        return get_bandit_summary(user_id)
    def reset_model(self, user_id: str="", reason: str="manual_reset") -> Dict[str,Any]:
        with _LOCK:
            store=_read_store()
            key=_uid(user_id)
            m=CUSUMLinUCB.from_state(dict(store["users"].get(key) or {}))
            event=m.reset_model(reason=reason)
            store["users"][key]=m.to_state()
            _write_store(store)
            return event


DECISION_STRATEGY_ARMS=("math_only","ev_road_blend","conservative")
DECISION_STRATEGY_CONTEXT_DIM=34


def build_decision_strategy_context(history: Iterable[Any], **kwargs: Any) -> List[float]:
    del kwargs
    return [1.0,min(1.0,len(_clean(history))/80.0)]+[0.0]*(DECISION_STRATEGY_CONTEXT_DIM-2)


def select_decision_strategy(history: Iterable[Any], **kwargs: Any) -> Dict[str,Any]:
    return {"version":"DECISION-STRATEGY-COMPAT-CUSUM-V1","selected_arm":"conservative","profile":{"kelly_multiplier":0.5},
            "context":build_decision_strategy_context(history),"eligible_exact_composition":False,"reason":"compatibility only"}


def update_decision_strategy(**kwargs: Any) -> Dict[str,Any]:
    return {"updated":False,"reason":"legacy_strategy_disabled","event_id":str(kwargs.get("event_id") or "")}


class DecisionStrategyBanditEngine:
    def select(self, history: Iterable[Any], **kwargs: Any) -> Dict[str,Any]:
        return select_decision_strategy(history,**kwargs)
    def update(self, **kwargs: Any) -> Dict[str,Any]:
        return update_decision_strategy(**kwargs)


__all__=["ARMS","MODEL_VERSION","ROAD_FEATURE_NAMES","MARKOV_FEATURE_NAMES","STRUCTURAL_FEATURE_NAMES","FEATURE_NAMES","CONTEXT_DIM","MARKOV_ALPHA","MARKOV_WINDOW_SIZE",
         "STRUCTURAL_SIGNATURE_COLUMNS","STRUCTURAL_WINDOW_COLUMNS","CUSUM_ALPHA","CUSUM_L2","CUSUM_FORGETTING_FACTOR","CUSUM_DRIFT_V","CUSUM_THRESHOLD_H","CUSUM_MIN_OBSERVATIONS","CUSUM_VACUUM_HANDS",
         "CUSUM_FORCE_OBSERVE_HANDS","MarkovFeatureExtractor","CUSUMLinUCB","ContextualBanditEngine","DecisionStrategyBanditEngine",
         "DECISION_STRATEGY_ARMS","DECISION_STRATEGY_CONTEXT_DIM","build_context_vector","build_decision_strategy_context","predict_bandit","update_bandit",
         "get_bandit_summary","select_decision_strategy","update_decision_strategy"]
