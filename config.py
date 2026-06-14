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

POINT_DB_PATH = os.getenv("POINT_DB_PATH", "data/point_db_3m.json")
RESULT_PATTERN_DB_PATH = os.getenv("RESULT_PATTERN_DB_PATH", "data/result_pattern_db_3m.json")

USE_POINT_DB = _get_bool("USE_POINT_DB", True)
USE_PATTERN_DB = _get_bool("USE_PATTERN_DB", True)

# 預設值只在 Render 沒設定環境變數時生效。
# 如果你要測其他比例，直接在 Render Environment Variables 改即可。
POINT_WEIGHT = _get_float("POINT_WEIGHT", 0.80)
PATTERN_WEIGHT = _get_float("PATTERN_WEIGHT", 0.05)
SIM_WEIGHT = _get_float("SIM_WEIGHT", 0.15)

MIN_OUTPUT_PROB = _get_float("MIN_OUTPUT_PROB", 0.05)
MAX_OUTPUT_PROB = _get_float("MAX_OUTPUT_PROB", 0.95)
PERCENT_DECIMALS = _get_int("PERCENT_DECIMALS", 2)

# predictor.py 會直接讀環境變數；這裡保留預設值給 getattr 相容。
BASE_BANKER_NO_TIE = _get_float("BASE_BANKER_NO_TIE", 0.5068)

MIN_GAP_FOR_ENTRY = _get_float("MIN_GAP_FOR_ENTRY", 0.035)
STRONG_GAP_FOR_ENTRY = _get_float("STRONG_GAP_FOR_ENTRY", 0.065)

TIE_AI_MAX_WEIGHT = _get_float("TIE_AI_MAX_WEIGHT", 0.02)
TIE_SHRINK = _get_float("TIE_SHRINK", 0.30)
TIE_MIN_GAP_FOR_ENTRY = _get_float("TIE_MIN_GAP_FOR_ENTRY", 0.08)

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
