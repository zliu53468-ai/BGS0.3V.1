import os


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


APP_NAME = os.getenv("APP_NAME", "BGS_DUAL_3M_DB_LINE_BOT")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

SESSION_EXPIRE_SECONDS = _get_int("SESSION_EXPIRE_SECONDS", 1800)
ENABLE_SIGNATURE_VERIFY = _get_bool("ENABLE_SIGNATURE_VERIFY", True)

DEFAULT_GAME = os.getenv("DEFAULT_GAME", "DG")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "DG05")
PLAYER_INPUT_FIRST = _get_bool("PLAYER_INPUT_FIRST", True)
NO_OBSERVE = _get_bool("NO_OBSERVE", True)
REPLY_STYLE = os.getenv("REPLY_STYLE", "COOL").upper()

# ============================================================
# DB 路徑
# ============================================================

POINT_DB_PATH = os.getenv("POINT_DB_PATH", "data/point_db_3m.json")
RESULT_PATTERN_DB_PATH = os.getenv("RESULT_PATTERN_DB_PATH", "data/result_pattern_db_3m.json")
COMBO_DB_PATH = os.getenv("COMBO_DB_PATH", "data/combo_db_3m.json")

USE_POINT_DB = _get_bool("USE_POINT_DB", True)
USE_PATTERN_DB = _get_bool("USE_PATTERN_DB", True)
USE_COMBO_DB = _get_bool("USE_COMBO_DB", True)

# ============================================================
# 權重
# ============================================================
# 建議第一版：combo 為細資料主修正，point 保底，pattern 小權重，AI 小修正。

POINT_WEIGHT = _get_float("POINT_WEIGHT", 0.55)
COMBO_WEIGHT = _get_float("COMBO_WEIGHT", 0.30)
PATTERN_WEIGHT = _get_float("PATTERN_WEIGHT", 0.03)
SIM_WEIGHT = _get_float("SIM_WEIGHT", 0.12)

COMBO_MIN_SAMPLE = _get_int("COMBO_MIN_SAMPLE", 80)

MIN_OUTPUT_PROB = _get_float("MIN_OUTPUT_PROB", 0.38)
MAX_OUTPUT_PROB = _get_float("MAX_OUTPUT_PROB", 0.62)
PERCENT_DECIMALS = _get_int("PERCENT_DECIMALS", 2)

# ============================================================
# 進場與風控
# ============================================================

MIN_GAP_FOR_ENTRY = _get_float("MIN_GAP_FOR_ENTRY", 0.060)
STRONG_GAP_FOR_ENTRY = _get_float("STRONG_GAP_FOR_ENTRY", 0.090)

TIE_AI_MAX_WEIGHT = _get_float("TIE_AI_MAX_WEIGHT", 0.01)
TIE_SHRINK = _get_float("TIE_SHRINK", 0.18)
TIE_MIN_GAP_FOR_ENTRY = _get_float("TIE_MIN_GAP_FOR_ENTRY", 0.12)

USE_MONTE_CARLO = _get_bool("USE_MONTE_CARLO", True)
MC_SIMULATIONS = _get_int("MC_SIMULATIONS", 600)
MC_MIN_SIMULATIONS = _get_int("MC_MIN_SIMULATIONS", 100)
MC_MAX_SIMULATIONS = _get_int("MC_MAX_SIMULATIONS", 900)
MC_SEED = _get_int("MC_SEED", 42)
MC_MAX_NOISE = _get_float("MC_MAX_NOISE", 0.018)
MC_BLOCK_LOW_GAP = _get_bool("MC_BLOCK_LOW_GAP", True)
MC_MIN_GAP_FOR_ENTRY = _get_float("MC_MIN_GAP_FOR_ENTRY", 0.055)
MC_DIRECTION_MISMATCH_BLOCK = _get_bool("MC_DIRECTION_MISMATCH_BLOCK", True)

AI_NOISE_SCALE = _get_float("AI_NOISE_SCALE", 0.008)
AI_HISTORY_WINDOW = _get_int("AI_HISTORY_WINDOW", 8)
AI_TREND_STRENGTH = _get_float("AI_TREND_STRENGTH", 0.010)
AI_DIFF_MOMENTUM_STRENGTH = _get_float("AI_DIFF_MOMENTUM_STRENGTH", 0.009)
AI_REVERSAL_STRENGTH = _get_float("AI_REVERSAL_STRENGTH", 0.016)
AI_HISTORY_MAX_ADJUST = _get_float("AI_HISTORY_MAX_ADJUST", 0.018)

GEMINI_ENABLE = _get_bool("GEMINI_ENABLE", False)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
AI_TEXT_ENABLE = _get_bool("AI_TEXT_ENABLE", True)

GAME_MAP = {
    "1": "WM",
    "2": "PM",
    "3": "DG",
    "4": "SA",
    "5": "KU",
    "6": "歐博/卡利",
    "7": "KG",
    "8": "全利",
    "9": "名人",
    "10": "MT真人",
}
