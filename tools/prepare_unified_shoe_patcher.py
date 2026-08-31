from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("apply_unified_shoe_patch.py")
text = path.read_text(encoding="utf-8")
old = '''replace_once(
    "predictor.py",
    "        \\\"remaining_counts_source\\\": composition_source,\\n",
    "        \\\"remaining_counts_source\\\": composition_source,\\n"
    "        \\\"remaining_cards_source\\\": remaining_cards_source,\\n"
    "        \\\"average_cards_per_hand\\\": float(AVERAGE_CARDS_PER_HAND),\\n"
    "        \\\"shoe_decks\\\": int(SHOE_DECKS),\\n"
    "        \\\"burn_cards\\\": int(BURN_CARDS),\\n"
    "        \\\"reference_hands\\\": int(REFERENCE_HANDS),\\n",
)
'''
new = '''replace_once(
    "predictor.py",
    "        \\\"card_composition_source\\\": composition_source,\\n"
    "        \\\"remaining_counts_source\\\": composition_source,\\n"
    "        \\\"banker_expected_return\\\": banker_ev,\\n",
    "        \\\"card_composition_source\\\": composition_source,\\n"
    "        \\\"remaining_counts_source\\\": composition_source,\\n"
    "        \\\"remaining_cards_source\\\": remaining_cards_source,\\n"
    "        \\\"average_cards_per_hand\\\": float(AVERAGE_CARDS_PER_HAND),\\n"
    "        \\\"shoe_decks\\\": int(SHOE_DECKS),\\n"
    "        \\\"burn_cards\\\": int(BURN_CARDS),\\n"
    "        \\\"reference_hands\\\": int(REFERENCE_HANDS),\\n"
    "        \\\"banker_expected_return\\\": banker_ev,\\n",
)
'''
if old not in text:
    raise RuntimeError("target patcher block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Prepared unified shoe patcher v2.")
