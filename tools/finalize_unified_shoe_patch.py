from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# In the B/P/T-only probabilistic path, the authoritative depth estimate must be
# burn + hand_count * AVERAGE_CARDS_PER_HAND. Particle-derived totals remain
# diagnostics only and are explicitly exposed under posterior_* names.
replace_once(
    "probabilistic_shoe_estimator.py",
    "    expected_remaining_cards = float(sum(expected_counts))\n"
    "    expected_cards_used = max(0.0, float(start_cards) - expected_remaining_cards)\n"
    "    mean_cards_per_round = (\n"
    "        expected_cards_used / conditioned_rounds\n"
    "        if conditioned_rounds > 0 else 0.0\n"
    "    )\n",
    "    posterior_expected_remaining_cards = float(sum(expected_counts))\n"
    "    posterior_expected_cards_used = max(\n"
    "        0.0, float(start_cards) - posterior_expected_remaining_cards\n"
    "    )\n"
    "    mean_cards_per_round = (\n"
    "        posterior_expected_cards_used / conditioned_rounds\n"
    "        if conditioned_rounds > 0 else 0.0\n"
    "    )\n",
)
replace_once(
    "probabilistic_shoe_estimator.py",
    "    remaining_cards_interval = _card_interval(remaining_totals)\n"
    "    remaining_cards_interval[\"semantics\"] = (\n"
    "        \"posterior_particle_interval_not_exact_remaining_card_count\"\n"
    "    )\n",
    "    posterior_remaining_cards_interval = _card_interval(remaining_totals)\n"
    "    posterior_remaining_cards_interval[\"semantics\"] = (\n"
    "        \"posterior_particle_interval_not_exact_remaining_card_count\"\n"
    "    )\n"
    "    round_count_remaining_interval = {\n"
    "        \"p10\": float(round_count_remaining_cards),\n"
    "        \"p50\": float(round_count_remaining_cards),\n"
    "        \"p90\": float(round_count_remaining_cards),\n"
    "        \"width_p90_p10\": 0.0,\n"
    "        \"semantics\": \"round_count_maturity_depth_point_estimate_not_exact_composition\",\n"
    "    }\n",
)
replace_once(
    "probabilistic_shoe_estimator.py",
    "    phase = _shoe_phase(start_cards, expected_remaining_cards)\n",
    "    phase = _shoe_phase(start_cards, round_count_remaining_cards)\n",
)
replace_once(
    "probabilistic_shoe_estimator.py",
    "        \"expected_remaining_counts\": [float(x) for x in expected_counts],\n"
    "        \"remaining_count_std\": [float(x) for x in std_counts],\n"
    "        \"expected_remaining_cards\": expected_remaining_cards,\n"
    "        \"remaining_cards_interval\": remaining_cards_interval,\n"
    "        \"expected_remaining_decks\": expected_remaining_cards / float(CARDS_PER_DECK),\n"
    "        \"expected_cards_used\": expected_cards_used,\n"
    "        \"round_count_estimated_remaining_cards\": float(round_count_remaining_cards),\n"
    "        \"round_count_estimated_cards_used\": float(round_count_cards_used),\n",
    "        \"expected_remaining_counts\": [float(x) for x in expected_counts],\n"
    "        \"remaining_count_std\": [float(x) for x in std_counts],\n"
    "        \"expected_remaining_counts_source\": \"probabilistic_particle_posterior_not_exact\",\n"
    "        \"expected_remaining_cards\": float(round_count_remaining_cards),\n"
    "        \"remaining_cards_interval\": round_count_remaining_interval,\n"
    "        \"expected_remaining_decks\": float(round_count_remaining_cards) / float(CARDS_PER_DECK),\n"
    "        \"expected_cards_used\": float(round_count_cards_used),\n"
    "        \"round_count_estimated_remaining_cards\": float(round_count_remaining_cards),\n"
    "        \"round_count_estimated_cards_used\": float(round_count_cards_used),\n"
    "        \"posterior_expected_remaining_cards\": float(posterior_expected_remaining_cards),\n"
    "        \"posterior_remaining_cards_interval\": posterior_remaining_cards_interval,\n"
    "        \"posterior_expected_remaining_decks\": float(posterior_expected_remaining_cards) / float(CARDS_PER_DECK),\n"
    "        \"posterior_expected_cards_used\": float(posterior_expected_cards_used),\n",
)

# Make contextual metadata fallback robust without changing its formal B/P logic.
replace_once(
    "contextual_bandit.py",
    "        ctx = dict(shoe_context or {})\n"
    "        if ctx.get(\"remaining_cards\") in {None, \"\"}:\n",
    "        ctx = dict(shoe_context or {})\n"
    "        remaining_hint = ctx.get(\"remaining_cards\")\n"
    "        if remaining_hint is None or remaining_hint == \"\":\n",
)

print("Final unified shoe depth semantics applied.")
