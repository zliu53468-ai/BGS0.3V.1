from __future__ import annotations

from pathlib import Path

path = Path(__file__).resolve().parents[1] / "app.py"
text = path.read_text(encoding="utf-8")

old_decks = '            "decks": int(session.get("exact_remaining_decks", 8) or SHOE_DECKS),\n'
new_decks = '            "decks": int(session.get("exact_remaining_decks", SHOE_DECKS) or SHOE_DECKS),\n'
if text.count(old_decks) != 1:
    raise RuntimeError(f"expected one legacy exact_remaining_decks default, found {text.count(old_decks)}")
text = text.replace(old_decks, new_decks, 1)

old_remaining = '''        current_remaining = int(
            session.get("screen_remaining_cards")
            or ocr.get("remaining_cards")
            or TOTAL_SHOE_CARDS
        )
        remaining = max(
            6,
            int(round(estimate_remaining_cards(len(raw_history), decks=SHOE_DECKS))),
        )
'''
new_remaining = '''        remaining = int(
            round(estimate_remaining_cards(len(raw_history), decks=SHOE_DECKS))
        )
'''
if text.count(old_remaining) != 1:
    raise RuntimeError(f"expected one manual round-count estimate block, found {text.count(old_remaining)}")
text = text.replace(old_remaining, new_remaining, 1)

path.write_text(text, encoding="utf-8")
print("app.py shoe depth consistency finalized")
