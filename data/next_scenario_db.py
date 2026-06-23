import json, os
from typing import Dict, Optional

NEXT_SCENARIO_DB_PATH = os.getenv("NEXT_SCENARIO_DB_PATH", "data/next_scenario_db.json")

def load_next_scenario_db() -> Dict[str, Dict[str, float]]:
    if not os.path.exists(NEXT_SCENARIO_DB_PATH):
        return {}
    with open(NEXT_SCENARIO_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_next_scenario_probs(player_point: int, banker_point: int) -> Dict[str, float]:
    db = load_next_scenario_db()
    key = f"P{player_point}_B{banker_point}"
    return db.get(key, {})
