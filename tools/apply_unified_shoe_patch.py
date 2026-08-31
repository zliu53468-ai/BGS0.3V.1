from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


def regex_once(path: str, pattern: str, replacement: str, flags: int = 0) -> None:
    text = read(path)
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{path}: regex expected one match, found {count}: {pattern!r}")
    write(path, updated)


# contextual_bandit.py: authoritative constants + round-count depth estimate only.
replace_once(
    "contextual_bandit.py",
    "from road_forecaster import forecast_next\n",
    "from road_forecaster import forecast_next\nfrom shoe_constants import (\n"
    "    AVERAGE_CARDS_PER_HAND,\n"
    "    BURN_CARDS,\n"
    "    SHOE_DECKS,\n"
    "    estimate_remaining_cards,\n"
    ")\n",
)
regex_once(
    "contextual_bandit.py",
    r"\nSHOE_DECKS = max\(1, int\(os\.getenv\(\"SHOE_DECKS\", \"8\"\) or \"8\"\)\)\n"
    r"ESTIMATED_CARDS_PER_ROUND = max\(4\.0, min\(6\.0, float\(os\.getenv\(\"ESTIMATED_CARDS_PER_ROUND\", \"4\.8\"\) or \"4\.8\"\)\)\)\n",
    "\n",
)
replace_once(
    "contextual_bandit.py",
    "        ctx = dict(shoe_context or {})\n        try:\n            diagnostic_remaining = max(0.0, float(ctx.get(\"remaining_cards\", 0.0) or 0.0))\n        except (TypeError, ValueError):\n            diagnostic_remaining = 0.0\n",
    "        ctx = dict(shoe_context or {})\n"
    "        if ctx.get(\"remaining_cards\") in {None, \"\"}:\n"
    "            ctx[\"remaining_cards\"] = estimate_remaining_cards(\n"
    "                len(raw), decks=SHOE_DECKS,\n"
    "                average_cards_per_hand=AVERAGE_CARDS_PER_HAND,\n"
    "                burn_cards=BURN_CARDS,\n"
    "            )\n"
    "            ctx[\"remaining_cards_source\"] = \"round_count_estimate\"\n"
    "        try:\n"
    "            diagnostic_remaining = max(0.0, float(ctx.get(\"remaining_cards\", 0.0) or 0.0))\n"
    "        except (TypeError, ValueError):\n"
    "            diagnostic_remaining = estimate_remaining_cards(len(raw), decks=SHOE_DECKS)\n"
    "            ctx[\"remaining_cards_source\"] = \"round_count_estimate\"\n",
)
replace_once(
    "contextual_bandit.py",
    "            \"remaining_cards\": diagnostic_remaining, \"remaining_cards_source\": str(ctx.get(\"remaining_cards_source\") or \"diagnostic_only\"),\n            \"estimated_remaining_counts_0_to_9\": [], \"rank_ratio_source\": \"not_used_road_primary\", \"rank_ratios_a_to_10jqk\": [],\n",
    "            \"remaining_cards\": diagnostic_remaining,\n"
    "            \"remaining_cards_source\": str(ctx.get(\"remaining_cards_source\") or \"round_count_estimate\"),\n"
    "            \"average_cards_per_hand\": float(AVERAGE_CARDS_PER_HAND),\n"
    "            \"shoe_decks\": int(SHOE_DECKS),\n"
    "            \"burn_cards\": int(BURN_CARDS),\n"
    "            \"remaining_cards_semantics\": \"maturity_depth_estimate_not_exact_composition\",\n"
    "            \"estimated_remaining_counts_0_to_9\": [], \"rank_ratio_source\": \"not_used_road_primary\", \"rank_ratios_a_to_10jqk\": [],\n",
)

# shoe_composition.py: exact path imports authoritative deck/count table only.
regex_once(
    "shoe_composition.py",
    r"import os\n\n\nDECKS = max\(1, min\(16, int\(os\.getenv\(\"SHOE_DECKS\", \"8\"\) or \"8\"\)\)\)\n",
    "import os\n\nfrom shoe_constants import (\n"
    "    AVERAGE_CARDS_PER_HAND,\n"
    "    SHOE_DECKS,\n"
    "    fresh_point_counts,\n"
    ")\n\n# Compatibility alias; authoritative value lives in shoe_constants.py.\n"
    "DECKS = SHOE_DECKS\n",
)
replace_once(
    "shoe_composition.py",
    "def fresh_counts(decks: int = DECKS) -> List[int]:\n    \"\"\"回傳點數 0..9 的新牌靴張數；0 包含 10/J/Q/K。\"\"\"\n    count = max(1, min(16, int(decks)))\n    return [16 * count] + [4 * count] * 9\n",
    "def fresh_counts(decks: int = DECKS) -> List[int]:\n"
    "    \"\"\"回傳點數 0..9 的新牌靴張數；權威張數表由 shoe_constants 提供。\"\"\"\n"
    "    return fresh_point_counts(decks)\n",
)
replace_once(
    "shoe_composition.py",
    "            \"decks\": decks,\n            \"remaining_counts\": list(counts),\n",
    "            \"decks\": decks,\n"
    "            \"shoe_decks\": decks,\n"
    "            \"remaining_cards\": int(sum(counts)),\n"
    "            \"remaining_cards_source\": (\n"
    "                \"exact_counts\" if source == \"remaining_counts\" else \"observed_cards\"\n"
    "            ),\n"
    "            \"average_cards_per_hand\": float(AVERAGE_CARDS_PER_HAND),\n"
    "            \"remaining_cards_semantics\": \"exact_from_card_composition\",\n"
    "            \"remaining_counts\": list(counts),\n",
)

# predictor.py: exact counts/observed cards stay exact; road-only fallback gets a
# round-count maturity estimate and never fabricated remaining_counts.
replace_once(
    "predictor.py",
    "from shoe_composition import analyze_shoe_composition\n",
    "from shoe_composition import analyze_shoe_composition\n"
    "from shoe_constants import (\n"
    "    AVERAGE_CARDS_PER_HAND,\n"
    "    BURN_CARDS,\n"
    "    CARDS_PER_DECK,\n"
    "    REFERENCE_HANDS,\n"
    "    SHOE_DECKS,\n"
    ")\n"
    "from shoe_depth_estimator import ShoeDepthEstimator\n",
)
replace_once(
    "predictor.py",
    "    raw_history = _normalize_outcome_history(_history_values(history))\n    big_road = normalize_big_road(raw_history)\n",
    "    raw_history = _normalize_outcome_history(_history_values(history))\n"
    "    big_road = normalize_big_road(raw_history)\n"
    "    depth_estimate = ShoeDepthEstimator(\n"
    "        shoe_decks=SHOE_DECKS,\n"
    "        average_cards_per_hand=AVERAGE_CARDS_PER_HAND,\n"
    "        reference_hands=REFERENCE_HANDS,\n"
    "        burn_cards=BURN_CARDS,\n"
    "    ).estimate(raw_history).as_dict()\n",
)
replace_once(
    "predictor.py",
    "    exact_counts = list(shoe_analysis.get(\"remaining_counts\") or []) if shoe_available else []\n    remaining_cards_value = (\n        float(sum(exact_counts))\n        if exact_counts\n        else float(context_meta.get(\"remaining_cards\", 0.0) or 0.0)\n    )\n    estimated_counts = exact_counts or list(context_meta.get(\"estimated_remaining_counts_0_to_9\") or [])\n    if not remaining_cards_value:\n        try:\n            remaining_cards_value = max(0.0, float(context.get(\"remaining_cards\", 0.0) or 0.0))\n        except (TypeError, ValueError):\n            remaining_cards_value = 0.0\n",
    "    exact_counts = list(shoe_analysis.get(\"remaining_counts\") or []) if shoe_available else []\n"
    "    if shoe_available:\n"
    "        remaining_cards_value = float(sum(exact_counts))\n"
    "        remaining_cards_source = (\n"
    "            \"exact_counts\" if composition_source == \"remaining_counts\" else \"observed_cards\"\n"
    "        )\n"
    "        estimated_counts = exact_counts\n"
    "    else:\n"
    "        remaining_cards_value = float(depth_estimate[\"remaining_cards\"])\n"
    "        remaining_cards_source = \"round_count_estimate\"\n"
    "        estimated_counts = []\n",
)
replace_once(
    "predictor.py",
    "            \"remaining_cards_source\": composition_source if shoe_available else str(\n                context_meta.get(\"remaining_cards_source\") or \"estimated\"\n            ),\n            \"estimated_remaining_counts_0_to_9\": estimated_counts,\n",
    "            \"remaining_cards_source\": remaining_cards_source,\n"
    "            \"average_cards_per_hand\": float(AVERAGE_CARDS_PER_HAND),\n"
    "            \"shoe_decks\": int(SHOE_DECKS),\n"
    "            \"burn_cards\": int(BURN_CARDS),\n"
    "            \"reference_hands\": int(REFERENCE_HANDS),\n"
    "            \"remaining_cards_semantics\": (\n"
    "                \"exact_from_card_composition\" if shoe_available\n"
    "                else \"maturity_depth_estimate_not_exact_composition\"\n"
    "            ),\n"
    "            \"estimated_remaining_counts_0_to_9\": estimated_counts,\n",
)
replace_once(
    "predictor.py",
    "    depth_constraint = {\n        \"applied\": shoe_available or supplied_remaining,\n        \"reason\": (\n            f\"{composition_source}_used_for_exact_shoe_ev\"\n            if shoe_available\n            else \"remaining_depth_estimated_or_diagnostic_only\"\n        ),\n        \"target_remaining_cards\": remaining_cards_value,\n    }\n",
    "    depth_constraint = {\n"
    "        \"applied\": True,\n"
    "        \"reason\": (\n"
    "            f\"{composition_source}_used_for_exact_shoe_ev\"\n"
    "            if shoe_available else \"round_count_maturity_depth_estimate\"\n"
    "        ),\n"
    "        \"target_remaining_cards\": remaining_cards_value,\n"
    "        \"source\": remaining_cards_source,\n"
    "        \"semantics\": (\n"
    "            \"exact_total_from_composition\" if shoe_available\n"
    "            else \"burn_plus_hand_count_times_average_not_exact_composition\"\n"
    "        ),\n"
    "    }\n",
)
replace_once(
    "predictor.py",
    "        \"expected_remaining_decks\": remaining_cards_value / 52.0,\n",
    "        \"expected_remaining_decks\": remaining_cards_value / float(CARDS_PER_DECK),\n",
)
replace_once(
    "predictor.py",
    "            \"available\": shoe_available,\n            \"conditioned_rounds\": len(big_road),\n            \"source\": composition_source if shoe_available else \"none\",\n",
    "            \"available\": True,\n"
    "            \"conditioned_rounds\": len(raw_history),\n"
    "            \"source\": remaining_cards_source,\n"
    "            \"exact_composition\": bool(shoe_available),\n",
)
replace_once(
    "predictor.py",
    "        \"conditioned_rounds\": len(big_road),\n        \"particle_count\": 0,\n        \"reliability\": 1.0 if shoe_available else 0.0,\n",
    "        \"conditioned_rounds\": len(raw_history),\n"
    "        \"particle_count\": 0,\n"
    "        \"reliability\": 1.0 if shoe_available else 0.5,\n",
)
replace_once(
    "predictor.py",
    "        \"depth_constraint_applied\": shoe_available,\n",
    "        \"depth_constraint_applied\": True,\n",
)
replace_once(
    "predictor.py",
    "            else \"unavailable_road_fallback\"\n",
    "            else \"round_count_maturity_depth_estimate_road_fallback\"\n",
)
replace_once(
    "predictor.py",
    "        \"remaining_counts_source\": composition_source,\n",
    "        \"remaining_counts_source\": composition_source,\n"
    "        \"remaining_cards_source\": remaining_cards_source,\n"
    "        \"average_cards_per_hand\": float(AVERAGE_CARDS_PER_HAND),\n"
    "        \"shoe_decks\": int(SHOE_DECKS),\n"
    "        \"burn_cards\": int(BURN_CARDS),\n"
    "        \"reference_hands\": int(REFERENCE_HANDS),\n",
)
replace_once(
    "predictor.py",
    "        \"shoe_progress\": float(min(1.0, len(big_road) / 70.0)),\n        \"shoe_depth_estimate\": {\n            \"rounds\": len(big_road),\n            \"remaining_cards\": remaining_cards_value,\n            \"source\": composition_source if shoe_available else str(context_meta.get(\"remaining_cards_source\") or \"estimated\"),\n        },\n",
    "        \"shoe_progress\": float(depth_estimate[\"shoe_progress\"]),\n"
    "        \"shoe_depth_estimate\": {\n"
    "            **dict(depth_estimate),\n"
    "            \"rounds\": len(raw_history),\n"
    "            \"remaining_cards\": remaining_cards_value,\n"
    "            \"remaining_cards_source\": remaining_cards_source,\n"
    "            \"source\": remaining_cards_source,\n"
    "            \"exact_composition\": bool(shoe_available),\n"
    "        },\n",
)
replace_once(
    "predictor.py",
    "            \"remaining_cards\": len(hidden_shoe),\n            \"remaining_cards_reliability\": 1.0,\n            \"remaining_cards_source\": \"virtual_shoe_exact_total\",\n",
    "            \"remaining_counts\": counts_from_shoe(hidden_shoe),\n"
    "            \"remaining_cards\": len(hidden_shoe),\n"
    "            \"remaining_cards_reliability\": 1.0,\n"
    "            \"remaining_cards_source\": \"exact_counts\",\n"
    "            \"decks\": SHOE_DECKS,\n",
)

# probabilistic_shoe_estimator.py: use shared deck/count constants. Keep particle
# composition explicitly probabilistic; round-count depth is reported separately.
replace_once(
    "probabilistic_shoe_estimator.py",
    "import statistics\n\n# Keep MODEL_VERSION stable because it is part of the deterministic RNG seed.\n",
    "import statistics\n\nfrom shoe_constants import (\n"
    "    AVERAGE_CARDS_PER_HAND,\n"
    "    BURN_CARDS,\n"
    "    CARDS_PER_DECK,\n"
    "    SHOE_DECKS,\n"
    "    estimate_cards_used,\n"
    "    estimate_remaining_cards,\n"
    "    fresh_point_counts,\n"
    "    total_cards_for_decks,\n"
    ")\n\n# Keep MODEL_VERSION stable because it is part of the deterministic RNG seed.\n",
)
replace_once(
    "probabilistic_shoe_estimator.py",
    "DECKS = 8\n",
    "DECKS = SHOE_DECKS  # compatibility alias; authoritative value is centralized\n",
)
replace_once(
    "probabilistic_shoe_estimator.py",
    "def _fresh_counts(decks: int = DECKS) -> list[int]:\n    decks = max(1, min(16, int(decks)))\n    return [16 * decks] + [4 * decks] * 9\n",
    "def _fresh_counts(decks: int = DECKS) -> list[int]:\n"
    "    return fresh_point_counts(decks)\n",
)
replace_once(
    "probabilistic_shoe_estimator.py",
    "    start_cards = 52 * decks\n",
    "    start_cards = total_cards_for_decks(decks)\n"
    "    round_count_remaining_cards = estimate_remaining_cards(\n"
    "        len(full_sequence), decks=decks,\n"
    "        average_cards_per_hand=AVERAGE_CARDS_PER_HAND,\n"
    "        burn_cards=BURN_CARDS,\n"
    "    )\n"
    "    round_count_cards_used = estimate_cards_used(\n"
    "        len(full_sequence),\n"
    "        average_cards_per_hand=AVERAGE_CARDS_PER_HAND,\n"
    "        burn_cards=BURN_CARDS,\n"
    "    )\n",
)
replace_once(
    "probabilistic_shoe_estimator.py",
    "        \"expected_remaining_decks\": expected_remaining_cards / 52.0,\n        \"expected_cards_used\": expected_cards_used,\n",
    "        \"expected_remaining_decks\": expected_remaining_cards / float(CARDS_PER_DECK),\n"
    "        \"expected_cards_used\": expected_cards_used,\n"
    "        \"round_count_estimated_remaining_cards\": float(round_count_remaining_cards),\n"
    "        \"round_count_estimated_cards_used\": float(round_count_cards_used),\n"
    "        \"remaining_cards_source\": \"round_count_estimate\",\n"
    "        \"average_cards_per_hand\": float(AVERAGE_CARDS_PER_HAND),\n"
    "        \"shoe_decks\": int(decks),\n"
    "        \"burn_cards\": int(BURN_CARDS),\n"
    "        \"depth_estimate_semantics\": \"maturity_depth_estimate_not_exact_composition\",\n"
    "        \"posterior_composition_semantics\": \"probabilistic_particle_composition_not_exact_remaining_counts\",\n",
)

# particle_filter_points.py: virtual/exact helper uses same deck/count table.
replace_once(
    "particle_filter_points.py",
    "import secrets\n\nDEFAULT_BASELINE",
    "import secrets\n\nfrom shoe_constants import SHOE_DECKS, fresh_point_counts\n\nDEFAULT_BASELINE",
)
replace_once(
    "particle_filter_points.py",
    "DECKS = 8\n",
    "DECKS = SHOE_DECKS  # compatibility alias\n",
)
replace_once(
    "particle_filter_points.py",
    "def fresh_counts(decks: int = DECKS) -> List[int]:\n    count = max(1, min(16, int(decks)))\n    return [16 * count] + [4 * count] * 9\n",
    "def fresh_counts(decks: int = DECKS) -> List[int]:\n"
    "    return fresh_point_counts(decks)\n",
)

# store.py: virtual shoe deck count comes from SHOE_DECKS, not a second PF_DECKS env.
replace_once(
    "store.py",
    "from shoe_composition import (\n    DECKS as PHYSICAL_SHOE_DECKS,\n    parse_card_value,\n    remaining_counts_from_observed,\n)\n",
    "from shoe_composition import parse_card_value, remaining_counts_from_observed\n"
    "from shoe_constants import SHOE_DECKS, total_cards_for_decks\n\n"
    "PHYSICAL_SHOE_DECKS = SHOE_DECKS\n",
)
regex_once(
    "store.py",
    r"PF_DECKS = max\(1, min\(16, int\(os\.getenv\(\"PF_DECKS\", \"8\"\) or \"8\"\)\)\)\n",
    "PF_DECKS = SHOE_DECKS  # compatibility alias; one authoritative shoe deck count\n",
)
replace_once(
    "store.py",
    "    total_cards = 52 * max(1, decks)\n",
    "    total_cards = total_cards_for_decks(decks)\n",
)

# virtual_shoe.py: default deck count follows SHOE_DECKS.
replace_once(
    "virtual_shoe.py",
    "from particle_filter_points import (\n",
    "from shoe_constants import SHOE_DECKS\n\nfrom particle_filter_points import (\n",
)
replace_once(
    "virtual_shoe.py",
    "DEFAULT_DECKS = _env_int(\"PF_DECKS\", 8, 1, 16)\n",
    "DEFAULT_DECKS = SHOE_DECKS\n",
)

# shoe_state_db.py: centralize deck total math.
replace_once(
    "shoe_state_db.py",
    "import numpy as np\n\nDEFAULT_BASELINE",
    "import numpy as np\n\nfrom shoe_constants import SHOE_DECKS, total_cards_for_decks\n\nDEFAULT_BASELINE",
)
replace_once(
    "shoe_state_db.py",
    "def state_key_from_counts(counts: np.ndarray, decks: int = 8) -> tuple[int, int, int, int, int]:\n",
    "def state_key_from_counts(counts: np.ndarray, decks: int = SHOE_DECKS) -> tuple[int, int, int, int, int]:\n",
)
replace_once(
    "shoe_state_db.py",
    "    removed = 52 * decks - total\n",
    "    removed = total_cards_for_decks(decks) - total\n",
)

# runtime_app.py: exact session context defaults to the same shared deck count.
replace_once(
    "runtime_app.py",
    "from dynamic_prediction_policy import install_dynamic_prediction_policy\n",
    "from dynamic_prediction_policy import install_dynamic_prediction_policy\n"
    "from shoe_constants import SHOE_DECKS\n",
)
replace_once(
    "runtime_app.py",
    "                \"decks\": int(session.get(\"exact_remaining_decks\", 8) or 8),\n",
    "                \"decks\": int(session.get(\"exact_remaining_decks\", SHOE_DECKS) or SHOE_DECKS),\n",
)
replace_once(
    "runtime_app.py",
    "                \"decks\": int(session.get(\"exact_remaining_decks\", 8) or 8),\n",
    "                \"decks\": int(session.get(\"exact_remaining_decks\", SHOE_DECKS) or SHOE_DECKS),\n",
)

# app.py: only configuration/depth glue changes; OCR recognition code is untouched.
replace_once(
    "app.py",
    "from shoe_composition import validate_remaining_counts\n",
    "from shoe_composition import validate_remaining_counts\n"
    "from shoe_constants import (\n"
    "    AVERAGE_CARDS_PER_HAND,\n"
    "    SHOE_DECKS,\n"
    "    TOTAL_SHOE_CARDS,\n"
    "    estimate_remaining_cards,\n"
    ")\n",
)
regex_once(
    "app.py",
    r"SCREEN_ESTIMATED_CARDS_PER_ROUND = max\(\n    0,\n    min\(6, int\(os\.getenv\(\"SCREEN_ESTIMATED_CARDS_PER_ROUND\", \"5\"\) or \"5\"\)\),\n\)\n",
    "SCREEN_ESTIMATED_CARDS_PER_ROUND = AVERAGE_CARDS_PER_HAND  # compatibility alias\n",
)
replace_once(
    "app.py",
    "    decks: int = Field(default=8, ge=1, le=16)\n",
    "    decks: int = Field(default=SHOE_DECKS, ge=1, le=16)\n",
)
# Normalize legacy hard-coded full-shoe totals without touching OCR algorithms.
text = read("app.py")
text = text.replace(" or 416", " or TOTAL_SHOE_CARDS")
text = text.replace(" or 8),\n            \"source\": \"user_exact_remaining_counts\"", " or SHOE_DECKS),\n            \"source\": \"user_exact_remaining_counts\"")
write("app.py", text)
replace_once(
    "app.py",
    "        remaining = max(6, current_remaining - SCREEN_ESTIMATED_CARDS_PER_ROUND)\n",
    "        remaining = max(\n"
    "            6,\n"
    "            int(round(estimate_remaining_cards(len(raw_history), decks=SHOE_DECKS))),\n"
    "        )\n",
)

# Focused regression tests for central constants, exact-vs-estimate separation and B/P-only contract.
(ROOT / "test_shoe_constants.py").write_text(
    '''"""Regression tests for unified shoe configuration and depth semantics."""\n'
    'from __future__ import annotations\n\n'
    'import unittest\n\n'
    'from shoe_constants import (\n'
    '    AVERAGE_CARDS_PER_HAND, BURN_CARDS, REFERENCE_HANDS, SHOE_DECKS,\n'
    '    fresh_point_counts, total_cards_for_decks, estimate_remaining_cards,\n'
    ')\n'
    'from shoe_composition import analyze_shoe_composition, remaining_counts_from_observed\n'
    'from shoe_depth_estimator import ShoeDepthEstimator\n\n\n'
    'class UnifiedShoeConstantsTests(unittest.TestCase):\n'
    '    def test_eight_deck_geometry(self) -> None:\n'
    '        self.assertEqual(total_cards_for_decks(8), 416)\n'
    '        self.assertEqual(fresh_point_counts(8), [128] + [32] * 9)\n\n'
    '    def test_round_count_estimate_20_hands(self) -> None:\n'
    '        remaining = estimate_remaining_cards(20, decks=8, average_cards_per_hand=4.9, burn_cards=0)\n'
    '        self.assertAlmostEqual(remaining, 318.0, places=9)\n'
    '        estimate = ShoeDepthEstimator(shoe_decks=8, average_cards_per_hand=4.9, burn_cards=0, reference_hands=70).estimate(["B", "P"] * 10).as_dict()\n'
    '        self.assertEqual(estimate["remaining_cards_source"], "round_count_estimate")\n'
    '        self.assertFalse(estimate["exact_composition"])\n'
    '        self.assertAlmostEqual(float(estimate["remaining_cards"]), 318.0, places=9)\n'
    '        self.assertAlmostEqual(float(estimate["shoe_progress"]), 20.0 / 70.0, places=9)\n\n'
    '    def test_observed_cards_are_exact_not_round_estimate(self) -> None:\n'
    '        observed = ["A", 8, "K", 3]\n'
    '        counts = remaining_counts_from_observed(observed, decks=8)\n'
    '        self.assertEqual(sum(counts), 412)\n'
    '        result = analyze_shoe_composition({"observed_cards": observed, "decks": 8})\n'
    '        self.assertTrue(result["available"])\n'
    '        self.assertEqual(result["remaining_cards_source"], "observed_cards")\n'
    '        self.assertEqual(result["action"] in {"B", "P"}, True)\n'
    '        self.assertEqual(sum(result["remaining_counts"]), 412)\n\n'
    '    def test_authoritative_defaults_are_exposed(self) -> None:\n'
    '        self.assertGreaterEqual(SHOE_DECKS, 1)\n'
    '        self.assertGreaterEqual(AVERAGE_CARDS_PER_HAND, 4.0)\n'
    '        self.assertGreaterEqual(REFERENCE_HANDS, 1)\n'
    '        self.assertGreaterEqual(BURN_CARDS, 0)\n\n\n'
    'if __name__ == "__main__":\n'
    '    unittest.main()\n''',
    encoding="utf-8",
)

print("Unified shoe configuration patch applied successfully.")
