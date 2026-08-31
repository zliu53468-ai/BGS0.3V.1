from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("apply_unified_shoe_patch.py")
text = path.read_text(encoding="utf-8")

# predictor.py has two remaining_counts_source entries; target the public result block.
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
    raise RuntimeError("target predictor metadata patcher block not found")
text = text.replace(old, new, 1)

# The virtual shoe path already supplies exact remaining_counts on current main.
obsolete = '''replace_once(
    "predictor.py",
    "            \\\"remaining_cards\\\": len(hidden_shoe),\\n            \\\"remaining_cards_reliability\\\": 1.0,\\n            \\\"remaining_cards_source\\\": \\\"virtual_shoe_exact_total\\\",\\n",
    "            \\\"remaining_counts\\\": counts_from_shoe(hidden_shoe),\\n"
    "            \\\"remaining_cards\\\": len(hidden_shoe),\\n"
    "            \\\"remaining_cards_reliability\\\": 1.0,\\n"
    "            \\\"remaining_cards_source\\\": \\\"exact_counts\\\",\\n"
    "            \\\"decks\\\": SHOE_DECKS,\\n",
)
'''
if obsolete not in text:
    raise RuntimeError("obsolete virtual-shoe patcher block not found")
text = text.replace(obsolete, "", 1)

# runtime_app.py legitimately has the same default-deck expression twice.
runtime_old = '''replace_once(
    "runtime_app.py",
    "                \\\"decks\\\": int(session.get(\\\"exact_remaining_decks\\\", 8) or 8),\\n",
    "                \\\"decks\\\": int(session.get(\\\"exact_remaining_decks\\\", SHOE_DECKS) or SHOE_DECKS),\\n",
)
replace_once(
    "runtime_app.py",
    "                \\\"decks\\\": int(session.get(\\\"exact_remaining_decks\\\", 8) or 8),\\n",
    "                \\\"decks\\\": int(session.get(\\\"exact_remaining_decks\\\", SHOE_DECKS) or SHOE_DECKS),\\n",
)
'''
runtime_new = '''runtime_text = read("runtime_app.py")
runtime_old_value = "                \\\"decks\\\": int(session.get(\\\"exact_remaining_decks\\\", 8) or 8),\\n"
runtime_new_value = "                \\\"decks\\\": int(session.get(\\\"exact_remaining_decks\\\", SHOE_DECKS) or SHOE_DECKS),\\n"
if runtime_text.count(runtime_old_value) != 2:
    raise RuntimeError(
        f"runtime_app.py: expected 2 exact deck defaults, found {runtime_text.count(runtime_old_value)}"
    )
write("runtime_app.py", runtime_text.replace(runtime_old_value, runtime_new_value))
'''
if runtime_old not in text:
    raise RuntimeError("runtime duplicate-deck patcher block not found")
text = text.replace(runtime_old, runtime_new, 1)

# test_shoe_constants.py is committed directly, so do not generate it from a string.
marker = "# Focused regression tests for central constants, exact-vs-estimate separation and B/P-only contract.\n"
end_marker = 'print("Unified shoe configuration patch applied successfully.")\n'
start = text.find(marker)
end = text.find(end_marker)
if start < 0 or end < 0 or end <= start:
    raise RuntimeError("test generation block markers not found")
text = (
    text[:start]
    + "# Focused regression tests are maintained as test_shoe_constants.py.\n\n"
    + text[end:]
)

path.write_text(text, encoding="utf-8")
print("Prepared unified shoe patcher v4.")
