import math
import os

# Render / CPU 環境穩定設定：避免 TensorFlow 佔用過多執行緒
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import json
import numpy as np
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

# LSTM / TensorFlow 在 Render CPU 上很容易造成冷啟動慢、記憶體爆掉或每局重訓卡住。
# 因此改成可開關：預設 USE_LSTM=0，穩定優先；若你升級 Render 規格再打開。
USE_LSTM = os.getenv("USE_LSTM", "0").strip() == "1"
if USE_LSTM:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam

        TF_AVAILABLE = True
        TF_IMPORT_ERROR = ""
    except Exception as e:
        tf = None
        Sequential = None
        LSTM = Dense = Dropout = Input = None
        Adam = None
        TF_AVAILABLE = False
        TF_IMPORT_ERROR = str(e)
else:
    tf = None
    Sequential = None
    LSTM = Dense = Dropout = Input = None
    Adam = None
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = "USE_LSTM=0，Render 穩定模式下略過 TensorFlow 匯入"

try:
    from deepseek_client import DeepSeekClient
except Exception as e:
    class DeepSeekClient:  # type: ignore
        """DeepSeek client fallback：回測或本機缺少 deepseek_client.py 時不讓 predictor 掛掉。"""
        def __init__(self, *args, **kwargs):
            self.import_error = str(e)

        def calibrate(self, payload):
            return {"error": True, "message": f"DeepSeekClient unavailable: {self.import_error}"}

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not TF_AVAILABLE:
    logger.warning(f"TensorFlow 未啟用，LSTM 會暫時回傳 0.5。原因：{TF_IMPORT_ERROR}")
else:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

# ============ 環境變數 ============
B_PRIOR = float(os.getenv("B_PRIOR", "0.4586"))
P_PRIOR = float(os.getenv("P_PRIOR", "0.4462"))
T_PRIOR = float(os.getenv("T_PRIOR", "0.0952"))

# 模型權重（固定權重模式會使用；動態模式會依牌路型態自動調整）
# 本版核心：大路 + 下三路為主，Markov / NGram / ML / DeepSeek 為輔
MARKOV_WEIGHT = float(os.getenv("MARKOV_WEIGHT", "0.055"))
ROAD_WEIGHT = float(os.getenv("ROAD_WEIGHT", "0.055"))
STREAK_WEIGHT = float(os.getenv("STREAK_WEIGHT", "0.030"))
BALANCE_WEIGHT = float(os.getenv("BALANCE_WEIGHT", "0.015"))
RECENT_WEIGHT = float(os.getenv("RECENT_WEIGHT", "0.040"))
NGRAM_WEIGHT = float(os.getenv("NGRAM_WEIGHT", "0.060"))

# 四路明細權重（實際融合會把三條下路合併成一個家族權重）
BIG_ROAD_WEIGHT = float(os.getenv("BIG_ROAD_WEIGHT", "0.30"))
BIG_EYE_WEIGHT = float(os.getenv("BIG_EYE_WEIGHT", "0.16"))
SMALL_ROAD_WEIGHT = float(os.getenv("SMALL_ROAD_WEIGHT", "0.13"))
COCKROACH_WEIGHT = float(os.getenv("COCKROACH_WEIGHT", "0.10"))

# 舊版 RoadEngine 權重保留相容用；新版不再把下三路合成單一主權重
ROAD_ENGINE_WEIGHT = float(os.getenv("ROAD_ENGINE_WEIGHT", "0.00"))
TIE_WEIGHT = float(os.getenv("TIE_WEIGHT", "0.04"))
# DeepSeek 硬開關：預設關閉，避免環境變數未吃到時誤呼叫 API
USE_DEEPSEEK = os.getenv("USE_DEEPSEEK", "0").strip() == "1"
AI_BLEND = float(os.getenv("AI_BLEND", "0")) if USE_DEEPSEEK else 0.0

# 動態權重開關：調整融合比例；觀望由下方 ALLOW_OBSERVE 控制
USE_DYNAMIC_REGIME_WEIGHTS = os.getenv("USE_DYNAMIC_REGIME_WEIGHTS", "1") == "1"
USE_ONLINE_WEIGHTING = os.getenv("USE_ONLINE_WEIGHTING", "1") == "1"
USE_ROAD_ENGINE = os.getenv("USE_ROAD_ENGINE", "1") == "1"
ONLINE_WEIGHT_WINDOW = int(os.getenv("ONLINE_WEIGHT_WINDOW", "24"))
ONLINE_WEIGHT_MIN_COUNT = int(os.getenv("ONLINE_WEIGHT_MIN_COUNT", "12"))
ONLINE_WEIGHT_ALPHA = float(os.getenv("ONLINE_WEIGHT_ALPHA", "0.22"))
ONLINE_BAYES_ALPHA = float(os.getenv("ONLINE_BAYES_ALPHA", "6.0"))
ONLINE_DISABLE_BELOW = float(os.getenv("ONLINE_DISABLE_BELOW", "0.42"))
ONLINE_BOOST_ABOVE = float(os.getenv("ONLINE_BOOST_ABOVE", "0.58"))

# Walk-forward Learning：逐局前推學習 / 回測 / 每個 LINE UID 獨立權重
# 目的：每個模型只能用當下以前的資料，並依每個使用者 / 場館 / 房間 / 靴號各自累積準度。
USE_WALK_FORWARD_LEARNING = os.getenv("USE_WALK_FORWARD_LEARNING", "1") == "1"
WALK_FORWARD_LIVE_PER_UID = os.getenv("WALK_FORWARD_LIVE_PER_UID", "1") == "1"
WALK_FORWARD_WINDOW = int(os.getenv("WALK_FORWARD_WINDOW", "24"))
WALK_FORWARD_MIN_COUNT = int(os.getenv("WALK_FORWARD_MIN_COUNT", "3"))
WALK_FORWARD_ALPHA = float(os.getenv("WALK_FORWARD_ALPHA", "0.35"))
WALK_FORWARD_BAYES_ALPHA = float(os.getenv("WALK_FORWARD_BAYES_ALPHA", "2.0"))
WALK_FORWARD_MIN_FACTOR = float(os.getenv("WALK_FORWARD_MIN_FACTOR", "0.70"))
WALK_FORWARD_MAX_FACTOR = float(os.getenv("WALK_FORWARD_MAX_FACTOR", "1.25"))
WALK_FORWARD_DISABLE_BELOW = float(os.getenv("WALK_FORWARD_DISABLE_BELOW", "0.42"))
WALK_FORWARD_BOOST_ABOVE = float(os.getenv("WALK_FORWARD_BOOST_ABOVE", "0.56"))
WALK_FORWARD_MIN_EDGE = float(os.getenv("WALK_FORWARD_MIN_EDGE", "0.006"))
WALK_FORWARD_ML_MIN_EDGE = float(os.getenv("WALK_FORWARD_ML_MIN_EDGE", "0.010"))
WALK_FORWARD_AI_MIN_EDGE = float(os.getenv("WALK_FORWARD_AI_MIN_EDGE", "0.010"))
WALK_FORWARD_APPLY_TO_ML = os.getenv("WALK_FORWARD_APPLY_TO_ML", "1") == "1"
WALK_FORWARD_APPLY_TO_AI = os.getenv("WALK_FORWARD_APPLY_TO_AI", "1") == "1"
WALK_FORWARD_STORE_PENDING = os.getenv("WALK_FORWARD_STORE_PENDING", "1") == "1"
WALK_FORWARD_DEBUG = os.getenv("WALK_FORWARD_DEBUG", "0") == "1"

# Pattern Replay Memory：全靴逐局前推相似規律回放
# 目的：不是只看最近幾口，而是用目前整靴已知資料，回找前面相似片段，判斷下一局延續/轉折。
USE_PATTERN_REPLAY_MEMORY = os.getenv("USE_PATTERN_REPLAY_MEMORY", "1") == "1"
PATTERN_REPLAY_MIN_HISTORY = int(os.getenv("PATTERN_REPLAY_MIN_HISTORY", "14"))
PATTERN_REPLAY_LOOKBACK = int(os.getenv("PATTERN_REPLAY_LOOKBACK", "80"))
PATTERN_REPLAY_FULL_SHOE = os.getenv("PATTERN_REPLAY_FULL_SHOE", "0") == "1"
PATTERN_REPLAY_WINDOWS = os.getenv("PATTERN_REPLAY_WINDOWS", "5,6,8,10,12,16")
PATTERN_REPLAY_MIN_MATCHES = int(os.getenv("PATTERN_REPLAY_MIN_MATCHES", "2"))
PATTERN_REPLAY_MIN_SIMILARITY = float(os.getenv("PATTERN_REPLAY_MIN_SIMILARITY", "0.72"))
PATTERN_REPLAY_EXACT_WEIGHT = float(os.getenv("PATTERN_REPLAY_EXACT_WEIGHT", "0.15"))
PATTERN_REPLAY_SHAPE_WEIGHT = float(os.getenv("PATTERN_REPLAY_SHAPE_WEIGHT", "0.55"))
PATTERN_REPLAY_TRANSITION_WEIGHT = float(os.getenv("PATTERN_REPLAY_TRANSITION_WEIGHT", "0.30"))
PATTERN_REPLAY_RECENCY_WEIGHT = float(os.getenv("PATTERN_REPLAY_RECENCY_WEIGHT", "0.30"))
PATTERN_REPLAY_LONG_WINDOW_WEIGHT = float(os.getenv("PATTERN_REPLAY_LONG_WINDOW_WEIGHT", "0.22"))
PATTERN_REPLAY_BAYES_ALPHA = float(os.getenv("PATTERN_REPLAY_BAYES_ALPHA", "1.2"))
PATTERN_REPLAY_MIN_EDGE = float(os.getenv("PATTERN_REPLAY_MIN_EDGE", "0.025"))
PATTERN_REPLAY_MAX_BIAS = float(os.getenv("PATTERN_REPLAY_MAX_BIAS", "0.075"))
PATTERN_REPLAY_WEIGHT = float(os.getenv("PATTERN_REPLAY_WEIGHT", "0.24"))
PATTERN_REPLAY_APPLY_WF = os.getenv("PATTERN_REPLAY_APPLY_WF", "1") == "1"
PATTERN_REPLAY_MAX_MATCHES = int(os.getenv("PATTERN_REPLAY_MAX_MATCHES", "40"))
# Render 防當機保護：單次預測最多掃描多少個候選片段，避免牌靴變長時 O(n*w) 卡住。
PATTERN_REPLAY_MAX_SCAN = int(os.getenv("PATTERN_REPLAY_MAX_SCAN", "360"))
PATTERN_REPLAY_CACHE_SIZE = int(os.getenv("PATTERN_REPLAY_CACHE_SIZE", "200"))
PATTERN_REPLAY_DEBUG = os.getenv("PATTERN_REPLAY_DEBUG", "0") == "1"


# RoadEngine / 下三路路紙引擎參數
ROAD_ENGINE_ROWS = int(os.getenv("ROAD_ENGINE_ROWS", "6"))
ROAD_ENGINE_MIN_HISTORY = int(os.getenv("ROAD_ENGINE_MIN_HISTORY", "6"))
ROAD_ENGINE_BREAK_STREAK = int(os.getenv("ROAD_ENGINE_BREAK_STREAK", "5"))
ROAD_ENGINE_DERIVED_LOOKBACK = int(os.getenv("ROAD_ENGINE_DERIVED_LOOKBACK", "10"))
ROAD_ENGINE_BLUE_BREAK_BIAS = float(os.getenv("ROAD_ENGINE_BLUE_BREAK_BIAS", "0.024"))
ROAD_ENGINE_RED_CONT_BIAS = float(os.getenv("ROAD_ENGINE_RED_CONT_BIAS", "0.016"))
DERIVED_ROAD_MIN_COUNT = int(os.getenv("DERIVED_ROAD_MIN_COUNT", "3"))
# Candidate Down-Road Simulation：下三路候選模擬參數
DERIVED_CANDIDATE_LOOKBACK = int(os.getenv("DERIVED_CANDIDATE_LOOKBACK", str(ROAD_ENGINE_DERIVED_LOOKBACK)))
DERIVED_CANDIDATE_MAX_EDGE = float(os.getenv("DERIVED_CANDIDATE_MAX_EDGE", "0.078"))
DERIVED_CANDIDATE_MIN_EDGE = float(os.getenv("DERIVED_CANDIDATE_MIN_EDGE", "0.008"))
DERIVED_COLOR_JUMP_RATE = float(os.getenv("DERIVED_COLOR_JUMP_RATE", "0.68"))
DERIVED_COLOR_STREAK_MIN = int(os.getenv("DERIVED_COLOR_STREAK_MIN", "3"))
DERIVED_COLOR_RATIO_GAP = float(os.getenv("DERIVED_COLOR_RATIO_GAP", "0.22"))
DERIVED_COLOR_NGRAM_MAX = int(os.getenv("DERIVED_COLOR_NGRAM_MAX", "5"))
FUHAO_DOWN3_MIN_DIFF = float(os.getenv("FUHAO_DOWN3_MIN_DIFF", "0.030"))

# Down-Road Structure：下三路齊整 / 有無 / 直落結構分
DERIVED_COLOR_SCORE_WEIGHT = float(os.getenv("DERIVED_COLOR_SCORE_WEIGHT", "0.78"))
DERIVED_STRUCTURE_SCORE_WEIGHT = float(os.getenv("DERIVED_STRUCTURE_SCORE_WEIGHT", "0.22"))
DERIVED_STRUCTURE_NEAT_BONUS = float(os.getenv("DERIVED_STRUCTURE_NEAT_BONUS", "0.055"))
DERIVED_STRUCTURE_MISMATCH_PENALTY = float(os.getenv("DERIVED_STRUCTURE_MISMATCH_PENALTY", "0.045"))
DERIVED_STRUCTURE_DROP_BONUS = float(os.getenv("DERIVED_STRUCTURE_DROP_BONUS", "0.035"))
DERIVED_STRUCTURE_SIDE_DRAG_PENALTY = float(os.getenv("DERIVED_STRUCTURE_SIDE_DRAG_PENALTY", "0.020"))
DERIVED_STRUCTURE_NEWCOL_BONUS = float(os.getenv("DERIVED_STRUCTURE_NEWCOL_BONUS", "0.025"))
DERIVED_STRUCTURE_MAX_EDGE = float(os.getenv("DERIVED_STRUCTURE_MAX_EDGE", "0.090"))
# End Candidate Down-Road Simulation
# Ask Road Hit Memory：問路命中率記憶
# 目的：讓每一靴依照最近實際命中率，微調大眼仔 / 小路 / 蟑螂路的可信度。
USE_ASK_ROAD_MEMORY = os.getenv("USE_ASK_ROAD_MEMORY", "1") == "1"
ASK_ROAD_MEMORY_WINDOW = int(os.getenv("ASK_ROAD_MEMORY_WINDOW", "24"))
ASK_ROAD_MEMORY_MIN_COUNT = max(8, int(os.getenv("ASK_ROAD_MEMORY_MIN_COUNT", "10")))
ASK_ROAD_MEMORY_ALPHA = float(os.getenv("ASK_ROAD_MEMORY_ALPHA", "0.35"))
ASK_ROAD_MEMORY_BAYES_ALPHA = float(os.getenv("ASK_ROAD_MEMORY_BAYES_ALPHA", "2.0"))
ASK_ROAD_MEMORY_MIN_FACTOR = float(os.getenv("ASK_ROAD_MEMORY_MIN_FACTOR", "0.72"))
ASK_ROAD_MEMORY_MAX_FACTOR = float(os.getenv("ASK_ROAD_MEMORY_MAX_FACTOR", "1.28"))
ASK_ROAD_MEMORY_DISABLE_BELOW = float(os.getenv("ASK_ROAD_MEMORY_DISABLE_BELOW", "0.43"))
ASK_ROAD_MEMORY_BOOST_ABOVE = float(os.getenv("ASK_ROAD_MEMORY_BOOST_ABOVE", "0.57"))
ASK_ROAD_MEMORY_APPLY_TO_HYBRID = os.getenv("ASK_ROAD_MEMORY_APPLY_TO_HYBRID", "1") == "1"
ASK_ROAD_MEMORY_APPLY_TO_FUHAO = os.getenv("ASK_ROAD_MEMORY_APPLY_TO_FUHAO", "1") == "1"
ASK_ROAD_MEMORY_DROP_BAD_VOTE = os.getenv("ASK_ROAD_MEMORY_DROP_BAD_VOTE", "1") == "1"
ASK_ROAD_MEMORY_BAD_VOTE_ACC = float(os.getenv("ASK_ROAD_MEMORY_BAD_VOTE_ACC", "0.40"))
ASK_ROAD_MEMORY_DEBUG = os.getenv("ASK_ROAD_MEMORY_DEBUG", "0") == "1"

# Column Shape Score：前排大路欄型分
# 目的：讓問路不只看紅藍與有無，也看候選落點是否符合前排欄高節奏。
USE_DERIVED_COLUMN_SHAPE = os.getenv("USE_DERIVED_COLUMN_SHAPE", "1") == "1"
DERIVED_COLUMN_SHAPE_WEIGHT = float(os.getenv("DERIVED_COLUMN_SHAPE_WEIGHT", "0.18"))
DERIVED_COLUMN_SHAPE_LOOKBACK = int(os.getenv("DERIVED_COLUMN_SHAPE_LOOKBACK", "5"))
DERIVED_COLUMN_NEAT_BONUS = float(os.getenv("DERIVED_COLUMN_NEAT_BONUS", "0.040"))
DERIVED_COLUMN_BREAK_PENALTY = float(os.getenv("DERIVED_COLUMN_BREAK_PENALTY", "0.035"))
DERIVED_COLUMN_DRAG_PENALTY = float(os.getenv("DERIVED_COLUMN_DRAG_PENALTY", "0.020"))
DERIVED_COLUMN_MAX_EDGE = float(os.getenv("DERIVED_COLUMN_MAX_EDGE", "0.070"))

# Down-Three Family：大眼仔 / 小路 / 蟑螂路先在家族內整合，對外最多只算一票。
# 避免三條同源衍生路被當成三個獨立模型，造成假共識與重複加權。
USE_DOWN3_FAMILY = os.getenv("USE_DOWN3_FAMILY", "1") == "1"
DOWN3_FAMILY_MIN_VALID_ROADS = int(os.getenv("DOWN3_FAMILY_MIN_VALID_ROADS", "2"))
DOWN3_FAMILY_MIN_AGREE = int(os.getenv("DOWN3_FAMILY_MIN_AGREE", "2"))
DOWN3_FAMILY_ROAD_MIN_GAP = float(os.getenv("DOWN3_FAMILY_ROAD_MIN_GAP", "0.010"))
DOWN3_FAMILY_MIN_GAP = max(0.030, float(os.getenv("DOWN3_FAMILY_MIN_GAP", "0.030")))
DOWN3_FAMILY_STRONG_GAP = float(os.getenv("DOWN3_FAMILY_STRONG_GAP", "0.055"))
DOWN3_FAMILY_MAX_GAP = float(os.getenv("DOWN3_FAMILY_MAX_GAP", "0.100"))
DOWN3_FAMILY_MAX_WEIGHT = float(os.getenv("DOWN3_FAMILY_MAX_WEIGHT", "0.24"))
DOWN3_FAMILY_COLUMN_SCALE = float(os.getenv("DOWN3_FAMILY_COLUMN_SCALE", "0.10"))
DOWN3_FAMILY_DISAGREE_SHRINK = float(os.getenv("DOWN3_FAMILY_DISAGREE_SHRINK", "0.62"))
DOWN3_FAMILY_DENSE_SHRINK = float(os.getenv("DOWN3_FAMILY_DENSE_SHRINK", "0.72"))
DOWN3_FAMILY_BIG_EYE_FACTOR = float(os.getenv("DOWN3_FAMILY_BIG_EYE_FACTOR", "1.00"))
DOWN3_FAMILY_SMALL_ROAD_FACTOR = float(os.getenv("DOWN3_FAMILY_SMALL_ROAD_FACTOR", "0.85"))
DOWN3_FAMILY_COCKROACH_FACTOR = float(os.getenv("DOWN3_FAMILY_COCKROACH_FACTOR", "0.70"))

# Dense Board Guard：短欄密集、欄高變化大時，下三路與大路衝突會提高最終確認門檻。
USE_DENSE_BOARD_GUARD = os.getenv("USE_DENSE_BOARD_GUARD", "1") == "1"
DENSE_BOARD_MIN_HISTORY = int(os.getenv("DENSE_BOARD_MIN_HISTORY", "12"))
DENSE_BOARD_MIN_COLUMNS = int(os.getenv("DENSE_BOARD_MIN_COLUMNS", "6"))
DENSE_BOARD_MAX_AVG_HEIGHT = float(os.getenv("DENSE_BOARD_MAX_AVG_HEIGHT", "3.20"))
DENSE_BOARD_SHORT_COLUMN_RATIO = float(os.getenv("DENSE_BOARD_SHORT_COLUMN_RATIO", "0.65"))
DENSE_BOARD_HEIGHT_CHANGE_RATE = float(os.getenv("DENSE_BOARD_HEIGHT_CHANGE_RATE", "0.45"))
DENSE_BOARD_SWITCH_LOW = float(os.getenv("DENSE_BOARD_SWITCH_LOW", "0.30"))
DENSE_BOARD_SWITCH_HIGH = float(os.getenv("DENSE_BOARD_SWITCH_HIGH", "0.80"))
ASK_ROAD_MEMORY_NO_BOOST_DENSE = os.getenv("ASK_ROAD_MEMORY_NO_BOOST_DENSE", "1") == "1"

# Final Confirmation Gate：下三路家族只提供候選，至少需要一個獨立來源確認。
FINAL_CONFIRM_MIN_SOURCES = int(os.getenv("FINAL_CONFIRM_MIN_SOURCES", "1"))
FINAL_CONFIRM_SCORE_GAP = float(os.getenv("FINAL_CONFIRM_SCORE_GAP", "0.018"))
FINAL_CONFIRM_PATTERN_CONF = float(os.getenv("FINAL_CONFIRM_PATTERN_CONF", "0.40"))
FINAL_CONFIRM_PATTERN_EDGE = float(os.getenv("FINAL_CONFIRM_PATTERN_EDGE", "0.025"))
DENSE_CONFLICT_REQUIRE_NON_ROAD_CONFIRM = os.getenv("DENSE_CONFLICT_REQUIRE_NON_ROAD_CONFIRM", "1") == "1"
ROAD_CONSENSUS_BOOST = float(os.getenv("ROAD_CONSENSUS_BOOST", "0.020"))
ROAD_CONFLICT_SHRINK = float(os.getenv("ROAD_CONFLICT_SHRINK", "0.055"))

# Road Lifecycle：用大路 + 下三路判斷「規律健康度 / 疲乏 / 斷點壓力」
# 這層不是觀望/下注決策，而是讓程式知道規律該跟、該降權、還是偏反邊。
USE_ROAD_LIFECYCLE = os.getenv("USE_ROAD_LIFECYCLE", "1") == "1"
ROAD_LIFECYCLE_WEIGHT = float(os.getenv("ROAD_LIFECYCLE_WEIGHT", "0.26"))
FOLLOW_SCORE_MIN = float(os.getenv("FOLLOW_SCORE_MIN", "0.61"))
BREAK_SCORE_MIN = float(os.getenv("BREAK_SCORE_MIN", "0.64"))
BREAK_FORCE_SCORE = float(os.getenv("BREAK_FORCE_SCORE", "0.78"))
FOLLOW_BOOST = float(os.getenv("FOLLOW_BOOST", "0.060"))
FATIGUE_SHRINK = float(os.getenv("FATIGUE_SHRINK", "0.045"))
BREAK_REVERSE_BIAS = float(os.getenv("BREAK_REVERSE_BIAS", "0.070"))
RED_HEALTH_WEIGHT = float(os.getenv("RED_HEALTH_WEIGHT", "0.36"))
BLUE_BREAK_WEIGHT = float(os.getenv("BLUE_BREAK_WEIGHT", "0.38"))
ROAD_CONFLICT_WEIGHT = float(os.getenv("ROAD_CONFLICT_WEIGHT", "0.20"))
DRAGON_FATIGUE_WEIGHT = float(os.getenv("DRAGON_FATIGUE_WEIGHT", "0.14"))
LIFECYCLE_PROTECT_MIN_CONF = float(os.getenv("LIFECYCLE_PROTECT_MIN_CONF", "0.66"))
LIFECYCLE_ML_SHRINK = float(os.getenv("LIFECYCLE_ML_SHRINK", "0.45"))
LIFECYCLE_AI_SHRINK = float(os.getenv("LIFECYCLE_AI_SHRINK", "0.40"))

# ML模型權重（在規律模型之後進行二次校準）
ML_WEIGHT = float(os.getenv("ML_WEIGHT", "0.12"))
ML_LR_WEIGHT = float(os.getenv("ML_LR_WEIGHT", "0.40"))
ML_RF_WEIGHT = float(os.getenv("ML_RF_WEIGHT", "0.45"))
ML_LSTM_WEIGHT = float(os.getenv("ML_LSTM_WEIGHT", "0.15"))

TIE_SHRINK = float(os.getenv("TIE_SHRINK", "0.30"))
TIE_MAX_PROB = float(os.getenv("TIE_MAX_PROB", "0.16"))
ALLOW_TIE_RECOMMEND = os.getenv("ALLOW_TIE_RECOMMEND", "0") == "1"
TIE_RECOMMEND_MIN = float(os.getenv("TIE_RECOMMEND_MIN", "0.165"))
MIN_HISTORY_FOR_AI = int(os.getenv("MIN_HISTORY_FOR_AI", "6"))
MIN_HISTORY_FOR_SIGNAL = int(os.getenv("MIN_HISTORY_FOR_SIGNAL", "4"))

# 決策彈性：放寬主方向機率鎖，並支援混亂/弱訊號時輸出觀望
SIDE_CLAMP_MIN = float(os.getenv("SIDE_CLAMP_MIN", "0.20"))
SIDE_CLAMP_MAX = float(os.getenv("SIDE_CLAMP_MAX", "0.80"))
ALLOW_OBSERVE = os.getenv("ALLOW_OBSERVE", "1") == "1"
OBSERVE_EDGE_MIN = float(os.getenv("OBSERVE_EDGE_MIN", "0.015"))
OBSERVE_CONF_MAX = float(os.getenv("OBSERVE_CONF_MAX", "0.45"))
OBSERVE_CONFLICT_MIN = float(os.getenv("OBSERVE_CONFLICT_MIN", "0.48"))
OBSERVE_CONFLICT_CONF_MAX = float(os.getenv("OBSERVE_CONFLICT_CONF_MAX", "0.52"))
OBSERVE_LIFECYCLE_STATES = set(
    x.strip().upper()
    for x in os.getenv("OBSERVE_LIFECYCLE_STATES", "CHAOS").split(",")
    if x.strip()
)

# Adaptive Road Memory：本靴內相似牌路狀態回測記憶
# 目的：不要只靠固定規則，而是看「目前這種類似路型」在本靴過去是跟路準，還是斷路準。
USE_ADAPTIVE_ROAD_MEMORY = os.getenv("USE_ADAPTIVE_ROAD_MEMORY", "1") == "1"
ROAD_MEMORY_LOOKBACK = int(os.getenv("ROAD_MEMORY_LOOKBACK", "32"))
ROAD_MEMORY_MIN_SAMPLE = int(os.getenv("ROAD_MEMORY_MIN_SAMPLE", "6"))
ROAD_MEMORY_FULL_SAMPLE = int(os.getenv("ROAD_MEMORY_FULL_SAMPLE", "24"))
ROAD_MEMORY_ALPHA = float(os.getenv("ROAD_MEMORY_ALPHA", "3.0"))
ROAD_MEMORY_MIN_MATCH_SCORE = float(os.getenv("ROAD_MEMORY_MIN_MATCH_SCORE", "4.0"))
ROAD_MEMORY_EXACT_BONUS = float(os.getenv("ROAD_MEMORY_EXACT_BONUS", "1.0"))
ROAD_MEMORY_RECENCY_BONUS = float(os.getenv("ROAD_MEMORY_RECENCY_BONUS", "0.35"))
ROAD_MEMORY_WEIGHT = float(os.getenv("ROAD_MEMORY_WEIGHT", "0.22"))
ROAD_MEMORY_MAX_BIAS = float(os.getenv("ROAD_MEMORY_MAX_BIAS", "0.055"))
ROAD_MEMORY_FOLLOW_THRESHOLD = float(os.getenv("ROAD_MEMORY_FOLLOW_THRESHOLD", "0.58"))
ROAD_MEMORY_BREAK_THRESHOLD = float(os.getenv("ROAD_MEMORY_BREAK_THRESHOLD", "0.58"))
ROAD_MEMORY_MIN_ADVANTAGE = float(os.getenv("ROAD_MEMORY_MIN_ADVANTAGE", "0.12"))
ROAD_MEMORY_PROTECT_MIN_CONF = float(os.getenv("ROAD_MEMORY_PROTECT_MIN_CONF", "0.62"))
ROAD_MEMORY_ML_SHRINK = float(os.getenv("ROAD_MEMORY_ML_SHRINK", "0.45"))
ROAD_MEMORY_AI_SHRINK = float(os.getenv("ROAD_MEMORY_AI_SHRINK", "0.40"))

# Road Rhythm Controller：多週期牌路節奏控制器
# 目的：不要太看當前一兩口，而是分辨「短暫波動 / 假斷」與「節奏真的轉折」。
USE_ROAD_RHYTHM = os.getenv("USE_ROAD_RHYTHM", "1") == "1"
ROAD_RHYTHM_MIN_HISTORY = int(os.getenv("ROAD_RHYTHM_MIN_HISTORY", "18"))
ROAD_RHYTHM_SHORT_WINDOW = int(os.getenv("ROAD_RHYTHM_SHORT_WINDOW", "8"))
ROAD_RHYTHM_MID_WINDOW = int(os.getenv("ROAD_RHYTHM_MID_WINDOW", "18"))
ROAD_RHYTHM_LONG_WINDOW = int(os.getenv("ROAD_RHYTHM_LONG_WINDOW", "36"))
ROAD_RHYTHM_WEIGHT = float(os.getenv("ROAD_RHYTHM_WEIGHT", "0.20"))
ROAD_RHYTHM_MAX_BIAS = float(os.getenv("ROAD_RHYTHM_MAX_BIAS", "0.050"))
ROAD_RHYTHM_INERTIA = float(os.getenv("ROAD_RHYTHM_INERTIA", "0.62"))
ROAD_RHYTHM_FALSE_BREAK_GUARD = float(os.getenv("ROAD_RHYTHM_FALSE_BREAK_GUARD", "0.58"))
ROAD_RHYTHM_TURN_CONFIRM = float(os.getenv("ROAD_RHYTHM_TURN_CONFIRM", "0.60"))
ROAD_RHYTHM_BLUE_RISE_MIN = float(os.getenv("ROAD_RHYTHM_BLUE_RISE_MIN", "0.08"))
ROAD_RHYTHM_ML_SHRINK = float(os.getenv("ROAD_RHYTHM_ML_SHRINK", "0.35"))
ROAD_RHYTHM_AI_SHRINK = float(os.getenv("ROAD_RHYTHM_AI_SHRINK", "0.32"))

# Strict Turn Confirm：轉折二次確認層
# 目的：避免 Rhythm 單層分數把短暫假斷誤判成真轉折。
# 啟用後，RHYTHM_TURN_CONFIRM 需要至少 N 個確認來源同意。
USE_STRICT_TURN_CONFIRM = os.getenv("USE_STRICT_TURN_CONFIRM", "1") == "1"
TURN_CONFIRM_MIN_VOTES = int(os.getenv("TURN_CONFIRM_MIN_VOTES", "2"))
TURN_CONFIRM_GAP = float(os.getenv("TURN_CONFIRM_GAP", "0.05"))
TURN_CONFIRM_CONSENSUS_MIN = float(os.getenv("TURN_CONFIRM_CONSENSUS_MIN", "0.66"))
TURN_CONFIRM_BLUE_PRESSURE_MIN = float(os.getenv("TURN_CONFIRM_BLUE_PRESSURE_MIN", "0.55"))
TURN_CONFIRM_LIFECYCLE_BREAK_MIN = float(os.getenv("TURN_CONFIRM_LIFECYCLE_BREAK_MIN", "0.58"))
TURN_CONFIRM_MEMORY_CONF_MIN = float(os.getenv("TURN_CONFIRM_MEMORY_CONF_MIN", "0.50"))

# Long Anchor Guard：長週期錨定層
# 目的：降低系統太看當局/短線雜訊；短線要反向時必須被中長週期或嚴格轉折確認。
USE_LONG_ANCHOR_GUARD = os.getenv("USE_LONG_ANCHOR_GUARD", "1") == "1"
LONG_ANCHOR_MIN_HISTORY = int(os.getenv("LONG_ANCHOR_MIN_HISTORY", "32"))
LONG_ANCHOR_WINDOW = int(os.getenv("LONG_ANCHOR_WINDOW", "54"))
LONG_ANCHOR_WEIGHT = float(os.getenv("LONG_ANCHOR_WEIGHT", "0.22"))
LONG_ANCHOR_MAX_PULL = float(os.getenv("LONG_ANCHOR_MAX_PULL", "0.055"))
LONG_ANCHOR_MAX_OPPOSITE_EDGE = float(os.getenv("LONG_ANCHOR_MAX_OPPOSITE_EDGE", "0.035"))
LONG_ANCHOR_CONF_MIN = float(os.getenv("LONG_ANCHOR_CONF_MIN", "0.52"))
LONG_ANCHOR_CONSENSUS_MIN = float(os.getenv("LONG_ANCHOR_CONSENSUS_MIN", "0.64"))
LONG_ANCHOR_TURN_BYPASS_VOTES = int(os.getenv("LONG_ANCHOR_TURN_BYPASS_VOTES", "3"))
LONG_ANCHOR_BREAK_BYPASS_SCORE = float(os.getenv("LONG_ANCHOR_BREAK_BYPASS_SCORE", "0.70"))

# LSTM參數：預設改保守，避免單靴資料少時過擬合
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", "8"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "2"))
LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "8"))
ML_RETRAIN_INTERVAL = int(os.getenv("ML_RETRAIN_INTERVAL", "8"))


# ============ 富濠式保守牌路多數決引擎 ============
# 預設啟用 FUHAO_CLONE，讓覆蓋後不用再改程式碼即可使用富濠式邏輯。
# 若要切回原本混合模型，Render 環境變數改成 PREDICT_ENGINE=HYBRID 即可。
PREDICT_ENGINE = os.getenv("PREDICT_ENGINE", "HYBRID").strip().upper()
FUHAO_HISTORY_LIMIT = int(os.getenv("FUHAO_HISTORY_LIMIT", "100"))
FUHAO_MIN_VALID_ROUNDS = int(os.getenv("FUHAO_MIN_VALID_ROUNDS", "6"))
FUHAO_IGNORE_TIE_FOR_PREDICT = os.getenv("FUHAO_IGNORE_TIE_FOR_PREDICT", "1") == "1"
FUHAO_KEEP_TIE_COUNT = os.getenv("FUHAO_KEEP_TIE_COUNT", "1") == "1"
FUHAO_LONG_THRESHOLD = int(os.getenv("FUHAO_LONG_THRESHOLD", "4"))
FUHAO_FENG_LOOKBACK_COLS = int(os.getenv("FUHAO_FENG_LOOKBACK_COLS", "6"))
FUHAO_DOWN3_TIE_BIAS = os.getenv("FUHAO_DOWN3_TIE_BIAS", "BANKER").strip().upper()
FUHAO_USE_BIG_ROAD = os.getenv("FUHAO_USE_BIG_ROAD", "1") == "1"
FUHAO_USE_BIG_EYE = os.getenv("FUHAO_USE_BIG_EYE", "1") == "1"
FUHAO_USE_SMALL_ROAD = os.getenv("FUHAO_USE_SMALL_ROAD", "1") == "1"
FUHAO_USE_COCKROACH = os.getenv("FUHAO_USE_COCKROACH", "1") == "1"
FUHAO_USE_DEEP_PARITY = os.getenv("FUHAO_USE_DEEP_PARITY", "0") == "1"
FUHAO_USE_LENGTH_PARITY = os.getenv("FUHAO_USE_LENGTH_PARITY", "0") == "1"
FUHAO_USE_BANKER_RATE = os.getenv("FUHAO_USE_BANKER_RATE", "0") == "1"
FUHAO_FINAL_METHOD = os.getenv("FUHAO_FINAL_METHOD", "MAJORITY").strip().upper()
FUHAO_FINAL_TIE_BREAKER = os.getenv("FUHAO_FINAL_TIE_BREAKER", "BIGROAD").strip().upper()
FUHAO_CONFIDENCE_MODE = os.getenv("FUHAO_CONFIDENCE_MODE", "VOTE_RATIO").strip().upper()
FUHAO_REQUIRE_ROAD_AND_ADVANCED_SAME = os.getenv("FUHAO_REQUIRE_ROAD_AND_ADVANCED_SAME", "1") == "1"
FUHAO_MIN_VOTE_AGREE = int(os.getenv("FUHAO_MIN_VOTE_AGREE", "2"))
FUHAO_OBSERVE_ON_CONFLICT = os.getenv("FUHAO_OBSERVE_ON_CONFLICT", "1") == "1"
FUHAO_OBSERVE_ON_UNKNOWN = os.getenv("FUHAO_OBSERVE_ON_UNKNOWN", "1") == "1"
FUHAO_OBSERVE_ON_TIE_ONLY = os.getenv("FUHAO_OBSERVE_ON_TIE_ONLY", "1") == "1"
FUHAO_PROB_EDGE = float(os.getenv("FUHAO_PROB_EDGE", "0.060"))
FUHAO_MAX_EDGE = float(os.getenv("FUHAO_MAX_EDGE", "0.100"))
FUHAO_TIE_BASE = float(os.getenv("FUHAO_TIE_BASE", "0.095"))
FUHAO_TIE_SHRINK = float(os.getenv("FUHAO_TIE_SHRINK", "0.30"))
FUHAO_DEBUG = os.getenv("FUHAO_DEBUG", "0") == "1"

# 富濠式 DeepSeek 輔助確認層：主模型仍以牌路多數決為主，AI 只做確認/校準。
# USE_DEEPSEEK=1 且 FUHAO_USE_DEEPSEEK=1 時才會啟用。
FUHAO_USE_DEEPSEEK = os.getenv("FUHAO_USE_DEEPSEEK", "1" if USE_DEEPSEEK else "0") == "1"
FUHAO_DEEPSEEK_MODE = os.getenv("FUHAO_DEEPSEEK_MODE", "CONFIRM").strip().upper()
FUHAO_DEEPSEEK_WEIGHT = float(os.getenv("FUHAO_DEEPSEEK_WEIGHT", "0.10"))
FUHAO_DEEPSEEK_MIN_HISTORY = int(os.getenv("FUHAO_DEEPSEEK_MIN_HISTORY", "8"))
FUHAO_DEEPSEEK_OBSERVE_ON_CONFLICT = os.getenv("FUHAO_DEEPSEEK_OBSERVE_ON_CONFLICT", "1") == "1"
FUHAO_DEEPSEEK_MAX_ADJUST = float(os.getenv("FUHAO_DEEPSEEK_MAX_ADJUST", "0.035"))
FUHAO_DEEPSEEK_TIE_MAX_ADJUST = float(os.getenv("FUHAO_DEEPSEEK_TIE_MAX_ADJUST", "0.020"))
FUHAO_DEEPSEEK_MIN_CONFIDENCE = float(os.getenv("FUHAO_DEEPSEEK_MIN_CONFIDENCE", "0.45"))
FUHAO_DEEPSEEK_CONFIDENCE_BOOST = float(os.getenv("FUHAO_DEEPSEEK_CONFIDENCE_BOOST", "0.060"))
FUHAO_DEEPSEEK_CONFIDENCE_SHRINK = float(os.getenv("FUHAO_DEEPSEEK_CONFIDENCE_SHRINK", "0.080"))
FUHAO_DEEPSEEK_INCLUDE_PAYLOAD = os.getenv("FUHAO_DEEPSEEK_INCLUDE_PAYLOAD", "0") == "1"


# 富濠式假規律 / 轉折保護模型：只修正「一直押多數方、假規律、假斷、路單轉折慢」問題。
# 這層不取代主模型，只在主模型出方向後判斷該方向是否屬於假規律風險。
FUHAO_USE_FAKE_PATTERN_DETECTOR = os.getenv("FUHAO_USE_FAKE_PATTERN_DETECTOR", "1") == "1"
FUHAO_FAKE_PATTERN_MIN_HISTORY = int(os.getenv("FUHAO_FAKE_PATTERN_MIN_HISTORY", "10"))
FUHAO_FAKE_PATTERN_SHORT_WINDOW = int(os.getenv("FUHAO_FAKE_PATTERN_SHORT_WINDOW", "8"))
FUHAO_FAKE_PATTERN_MID_WINDOW = int(os.getenv("FUHAO_FAKE_PATTERN_MID_WINDOW", "16"))
FUHAO_FAKE_PATTERN_LONG_WINDOW = int(os.getenv("FUHAO_FAKE_PATTERN_LONG_WINDOW", "32"))
FUHAO_FAKE_PATTERN_OBSERVE_SCORE = float(os.getenv("FUHAO_FAKE_PATTERN_OBSERVE_SCORE", "0.58"))
FUHAO_FAKE_PATTERN_HARD_OBSERVE_SCORE = float(os.getenv("FUHAO_FAKE_PATTERN_HARD_OBSERVE_SCORE", "0.72"))
FUHAO_FAKE_PATTERN_SHRINK_SCORE = float(os.getenv("FUHAO_FAKE_PATTERN_SHRINK_SCORE", "0.46"))
FUHAO_FAKE_PATTERN_CONF_SHRINK = float(os.getenv("FUHAO_FAKE_PATTERN_CONF_SHRINK", "0.42"))
FUHAO_FAKE_PATTERN_OBSERVE_ON_SCORE = os.getenv("FUHAO_FAKE_PATTERN_OBSERVE_ON_SCORE", "1") == "1"
FUHAO_FAKE_PATTERN_OBSERVE_ON_TURN = os.getenv("FUHAO_FAKE_PATTERN_OBSERVE_ON_TURN", "1") == "1"
FUHAO_FAKE_PATTERN_OBSERVE_ON_FALSE_BREAK = os.getenv("FUHAO_FAKE_PATTERN_OBSERVE_ON_FALSE_BREAK", "1") == "1"
FUHAO_FAKE_PATTERN_TURN_SCORE = float(os.getenv("FUHAO_FAKE_PATTERN_TURN_SCORE", "0.62"))
FUHAO_FAKE_PATTERN_FALSE_BREAK_SCORE = float(os.getenv("FUHAO_FAKE_PATTERN_FALSE_BREAK_SCORE", "0.64"))
FUHAO_FAKE_PATTERN_FALSE_BREAK_MIN_STREAK = int(os.getenv("FUHAO_FAKE_PATTERN_FALSE_BREAK_MIN_STREAK", "4"))
FUHAO_FAKE_PATTERN_FALSE_BREAK_CONFIRM_ROUNDS = int(os.getenv("FUHAO_FAKE_PATTERN_FALSE_BREAK_CONFIRM_ROUNDS", "2"))
FUHAO_FAKE_PATTERN_DERIVED_MIN_AGREE = int(os.getenv("FUHAO_FAKE_PATTERN_DERIVED_MIN_AGREE", "2"))
FUHAO_FAKE_PATTERN_REQUIRE_DERIVED_CONFIRM = os.getenv("FUHAO_FAKE_PATTERN_REQUIRE_DERIVED_CONFIRM", "1") == "1"
FUHAO_FAKE_PATTERN_OBSERVE_ON_DERIVED_CONFLICT = os.getenv("FUHAO_FAKE_PATTERN_OBSERVE_ON_DERIVED_CONFLICT", "1") == "1"
FUHAO_FAKE_PATTERN_CHAOS_SWITCH_RATE = float(os.getenv("FUHAO_FAKE_PATTERN_CHAOS_SWITCH_RATE", "0.72"))
FUHAO_FAKE_PATTERN_DENSE_SWITCH_LOW = float(os.getenv("FUHAO_FAKE_PATTERN_DENSE_SWITCH_LOW", "0.38"))
FUHAO_FAKE_PATTERN_DENSE_SWITCH_HIGH = float(os.getenv("FUHAO_FAKE_PATTERN_DENSE_SWITCH_HIGH", "0.62"))
FUHAO_FAKE_PATTERN_MIN_VOTE_RATIO = float(os.getenv("FUHAO_FAKE_PATTERN_MIN_VOTE_RATIO", "0.64"))
FUHAO_FAKE_PATTERN_DEBUG = os.getenv("FUHAO_FAKE_PATTERN_DEBUG", "0") == "1"

# 假規律裁決層：讓假規律/轉折模型可以真正蓋掉主模型，而不只是降低信心。
# 目的：避免 final_pick 已經被主模型決定後，畫面機率仍偏向多數方。
FUHAO_FAKE_PATTERN_HARD_DECISION = os.getenv("FUHAO_FAKE_PATTERN_HARD_DECISION", "1") == "1"
FUHAO_FAKE_PATTERN_NEUTRALIZE_ON_OBSERVE = os.getenv("FUHAO_FAKE_PATTERN_NEUTRALIZE_ON_OBSERVE", "1") == "1"
FUHAO_FAKE_PATTERN_ALLOW_REVERSE = os.getenv("FUHAO_FAKE_PATTERN_ALLOW_REVERSE", "1") == "1"
FUHAO_FAKE_PATTERN_REVERSE_SCORE = float(os.getenv("FUHAO_FAKE_PATTERN_REVERSE_SCORE", "0.72"))
FUHAO_FAKE_PATTERN_REVERSE_DERIVED_MIN = int(os.getenv("FUHAO_FAKE_PATTERN_REVERSE_DERIVED_MIN", "2"))
FUHAO_FAKE_PATTERN_REVERSE_REQUIRE_ROAD = os.getenv("FUHAO_FAKE_PATTERN_REVERSE_REQUIRE_ROAD", "1") == "1"
FUHAO_FAKE_PATTERN_REVERSE_MIN_RATIO = float(os.getenv("FUHAO_FAKE_PATTERN_REVERSE_MIN_RATIO", "0.62"))
FUHAO_FAKE_PATTERN_OBSERVE_RESETS_EDGE = os.getenv("FUHAO_FAKE_PATTERN_OBSERVE_RESETS_EDGE", "1") == "1"

# 最後整合裁決層：把大路 / 四路多數決降級成候選訊號。
# 目的：避免 road_majority / big_road_pick 自己決定方向，造成仍然偏向多數方。
FUHAO_ROAD_MAJORITY_AS_CANDIDATE_ONLY = os.getenv("FUHAO_ROAD_MAJORITY_AS_CANDIDATE_ONLY", "1") == "1"
FUHAO_DISABLE_BIGROAD_FALLBACK = os.getenv("FUHAO_DISABLE_BIGROAD_FALLBACK", "1") == "1"
FUHAO_REQUIRE_DERIVED_FOR_FINAL = os.getenv("FUHAO_REQUIRE_DERIVED_FOR_FINAL", "1") == "1"
FUHAO_OBSERVE_ON_BIGROAD_ONLY = os.getenv("FUHAO_OBSERVE_ON_BIGROAD_ONLY", "1") == "1"
FUHAO_FINAL_REQUIRE_NON_BIGROAD_VOTES = int(os.getenv("FUHAO_FINAL_REQUIRE_NON_BIGROAD_VOTES", "2"))
FUHAO_FINAL_REQUIRE_DERIVED_RATIO = float(os.getenv("FUHAO_FINAL_REQUIRE_DERIVED_RATIO", "0.62"))
FUHAO_NEUTRALIZE_ON_FINAL_GATE_OBSERVE = os.getenv("FUHAO_NEUTRALIZE_ON_FINAL_GATE_OBSERVE", "1") == "1"

# ============ 全局模型實例（單例模式） ============
class MLModels:
    """機器學習模型容器：每個 user_id / 場館 / 房間 / 靴號 可建立獨立實例"""

    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1
        )
        self.lr = LogisticRegression(
            max_iter=300,
            random_state=42,
            C=1.0
        )
        self.lstm = None
        self.scaler = StandardScaler()

        self.is_trained = False
        self.training_samples = 0
        self.last_training_history = []
        self.last_training_key = ""

        # Render 啟動穩定版：不在 import 時建立 LSTM，避免服務啟動卡住。
        # LSTM 會在資料足夠並進入 train() 時才建立與訓練。

    def _build_lstm(self):
        """建立 LSTM 模型架構（權重需訓練）"""
        if not TF_AVAILABLE:
            self.lstm = None
            return None

        self.lstm = Sequential([
            Input(shape=(LSTM_SEQUENCE_LENGTH, 1)),
            LSTM(48, return_sequences=True),
            Dropout(0.20),
            LSTM(24, return_sequences=False),
            Dropout(0.20),
            Dense(12, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.lstm.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.lstm

    def _encode_sequence(self, non_tie: List[str]) -> np.ndarray:
        """編碼牌路序列為數值"""
        mapping = {'B': 1, 'P': 0}
        return np.array([mapping.get(x, 0) for x in non_tie]).reshape(-1, 1)

    def _extract_features(self, non_tie: List[str]) -> np.ndarray:
        """提取ML特徵（無資料洩漏版本）。維持原本 12 維，避免舊模型流程被大改。"""
        if len(non_tie) < 6:
            return np.zeros((1, 12))

        n = len(non_tie)
        b_count = non_tie.count('B')
        p_count = n - b_count
        b_rate = b_count / n if n > 0 else 0.5

        recent = non_tie[-10:] if n >= 10 else non_tie
        recent_b_rate = recent.count('B') / len(recent) if len(recent) > 0 else 0.5

        if n >= 2:
            switches = sum(1 for i in range(1, n) if non_tie[i] != non_tie[i - 1])
            switch_rate = switches / (n - 1)
        else:
            switch_rate = 0.5

        current_streak = 1
        if n >= 2:
            for i in range(n - 2, -1, -1):
                if non_tie[i] == non_tie[-1]:
                    current_streak += 1
                else:
                    break

        max_streak = 1
        current = 1
        for i in range(1, n):
            if non_tie[i] == non_tie[i - 1]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1

        last_5 = non_tie[-5:] if n >= 5 else non_tie
        last_5_b = last_5.count('B') / len(last_5) if len(last_5) > 0 else 0.5

        last_3 = non_tie[-3:] if n >= 3 else non_tie
        last_3_b = last_3.count('B') / len(last_3) if len(last_3) > 0 else 0.5

        features = np.array([[
            b_rate,
            recent_b_rate,
            switch_rate,
            current_streak / max(10, n),
            max_streak / max(10, n),
            last_5_b,
            last_3_b,
            b_count / max(10, n),
            p_count / max(10, n),
            1 if non_tie[-1] == 'B' else 0,
            (b_count - p_count) / max(10, n),
            n / 100
        ]])

        return features

    def _prepare_lstm_data(self, non_tie: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """準備LSTM序列資料"""
        if len(non_tie) < LSTM_SEQUENCE_LENGTH + 1:
            return np.array([]), np.array([])

        encoded = self._encode_sequence(non_tie)
        X, y = [], []

        for i in range(LSTM_SEQUENCE_LENGTH, len(encoded)):
            X.append(encoded[i - LSTM_SEQUENCE_LENGTH:i, 0])
            y.append(encoded[i, 0])

        if len(X) == 0:
            return np.array([]), np.array([])

        return np.array(X).reshape(-1, LSTM_SEQUENCE_LENGTH, 1), np.array(y)

    def train(self, non_tie: List[str], training_key: str = "") -> Dict[str, Any]:
        """訓練所有 ML 模型：LR + RF + LSTM（有 TensorFlow 才啟用）"""
        if len(non_tie) < 30:
            return {
                "status": "error",
                "message": f"需要至少30局歷史資料，目前{len(non_tie)}局"
            }

        try:
            X_features = []
            y_labels = []

            for i in range(12, len(non_tie)):
                features = self._extract_features(non_tie[:i])
                X_features.append(features[0])
                y_labels.append(1 if non_tie[i] == 'B' else 0)

            X_features = np.array(X_features)
            y_labels = np.array(y_labels)

            if len(X_features) < 10:
                return {"status": "error", "message": "有效訓練樣本不足"}

            if len(set(y_labels.tolist())) < 2:
                return {"status": "error", "message": "訓練資料只有單一類別，暫不訓練 ML"}

            X_scaled = self.scaler.fit_transform(X_features)
            self.lr.fit(X_scaled, y_labels)
            self.rf.fit(X_scaled, y_labels)

            lstm_status = "disabled"
            if TF_AVAILABLE:
                X_lstm, y_lstm = self._prepare_lstm_data(non_tie)
                if len(X_lstm) > 10 and len(set(y_lstm.tolist())) >= 2:
                    self._build_lstm()
                    callbacks = [
                        tf.keras.callbacks.EarlyStopping(
                            patience=3,
                            restore_best_weights=True
                        )
                    ]
                    self.lstm.fit(
                        X_lstm,
                        y_lstm,
                        epochs=LSTM_EPOCHS,
                        batch_size=LSTM_BATCH_SIZE,
                        verbose=0,
                        validation_split=0.2,
                        callbacks=callbacks
                    )
                    lstm_status = "trained"
                else:
                    lstm_status = "not_enough_sequence"
            else:
                self.lstm = None
                lstm_status = f"tensorflow_unavailable: {TF_IMPORT_ERROR}"

            self.is_trained = True
            self.training_samples = len(X_features)
            self.last_training_history = list(non_tie)
            self.last_training_key = training_key

            return {
                "status": "success",
                "samples": self.training_samples,
                "lstm_status": lstm_status,
                "message": "ML模型訓練完成"
            }

        except Exception as e:
            logger.error(f"ML訓練錯誤: {e}")
            return {"status": "error", "message": str(e)}

    def predict(self, non_tie: List[str]) -> Dict[str, float]:
        """使用ML模型預測"""
        default_result = {
            'lr': 0.5,
            'rf': 0.5,
            'lstm': 0.5,
            'ensemble': 0.5
        }

        if len(non_tie) < 12 or not self.is_trained:
            return default_result

        try:
            features = self._extract_features(non_tie)
            features_scaled = self.scaler.transform(features)

            predictions = {}

            try:
                lr_prob = self.lr.predict_proba(features_scaled)[0][1]
                predictions['lr'] = float(lr_prob)
            except Exception:
                predictions['lr'] = 0.5

            try:
                rf_prob = self.rf.predict_proba(features_scaled)[0][1]
                predictions['rf'] = float(rf_prob)
            except Exception:
                predictions['rf'] = 0.5

            try:
                if self.lstm is not None and len(non_tie) >= LSTM_SEQUENCE_LENGTH:
                    encoded = self._encode_sequence(non_tie[-LSTM_SEQUENCE_LENGTH:])
                    X_lstm = np.array(encoded).reshape(1, LSTM_SEQUENCE_LENGTH, 1)
                    lstm_prob = float(self.lstm.predict(X_lstm, verbose=0)[0][0])
                    predictions['lstm'] = lstm_prob
                else:
                    predictions['lstm'] = 0.5
            except Exception:
                predictions['lstm'] = 0.5

            total_model_w = max(0.0001, ML_LR_WEIGHT + ML_RF_WEIGHT + ML_LSTM_WEIGHT)
            weights = {
                'lr': ML_LR_WEIGHT / total_model_w,
                'rf': ML_RF_WEIGHT / total_model_w,
                'lstm': ML_LSTM_WEIGHT / total_model_w,
            }
            ensemble = sum(predictions[k] * weights[k] for k in weights)
            predictions['ensemble'] = float(ensemble)

            return predictions

        except Exception as e:
            logger.error(f"ML預測錯誤: {e}")
            return default_result

# ============ 模型快取池 ============
MAX_MODEL_CACHE = int(os.getenv("MAX_MODEL_CACHE", "12"))
_MODEL_CACHE: Dict[str, MLModels] = {}
_MODEL_CACHE_ORDER: List[str] = []


# 每個 LINE UID / 場館 / 房間 / 靴號 的逐局前推狀態。
# 只保存「上一局預測下一局」的 pending，以及最近 N 次各模型是否命中的紀錄。
_WALK_FORWARD_STATE: Dict[str, Dict[str, Any]] = {}


# Ask Road Hit Memory：每個 LINE UID / 場館 / 房間 / 靴號 的問路命中率記憶。
# 只保存上一輪問路票 pending，以及最近 N 次問路票是否命中的紀錄。
_ASK_ROAD_STATE: Dict[str, Dict[str, Any]] = {}

# Pattern Replay 快取：同一個 LINE UID/房間/靴號、同一段歷史重複按「開始分析」時，不重新掃描整靴。
_PATTERN_REPLAY_CACHE: Dict[str, Dict[str, Any]] = {}
_PATTERN_REPLAY_CACHE_ORDER: List[str] = []


def _get_ml_models(training_key: str) -> MLModels:
    key = training_key or "global"

    if key in _MODEL_CACHE:
        try:
            _MODEL_CACHE_ORDER.remove(key)
        except ValueError:
            pass
        _MODEL_CACHE_ORDER.append(key)
        return _MODEL_CACHE[key]

    while len(_MODEL_CACHE) >= MAX_MODEL_CACHE and _MODEL_CACHE_ORDER:
        old_key = _MODEL_CACHE_ORDER.pop(0)
        _MODEL_CACHE.pop(old_key, None)

    model = MLModels()
    _MODEL_CACHE[key] = model
    _MODEL_CACHE_ORDER.append(key)
    return model


def clear_model_cache() -> Dict[str, Any]:
    """清空 ML 模型快取。

    用途：回測、重新開始新測試、或你想確保模型不沿用上一批資料時呼叫。
    正常 LINE 使用流程不需要主動呼叫。
    """
    removed = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    _MODEL_CACHE_ORDER.clear()
    return {"ok": True, "removed": removed}


def get_model_cache_info() -> Dict[str, Any]:
    """回傳目前 ML 模型快取狀態，方便 debug / backtest 檢查。"""
    return {
        "size": len(_MODEL_CACHE),
        "max_size": MAX_MODEL_CACHE,
        "keys": list(_MODEL_CACHE_ORDER),
    }

# ============ 輔助函數 ============
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def _normalize_three(b: float, p: float, t: float) -> Tuple[float, float, float]:
    b = max(0.001, b)
    p = max(0.001, p)
    t = max(0.001, min(TIE_MAX_PROB, t))
    s = b + p + t
    return b / s, p / s, t / s


def _last_non_tie(history: List[str]) -> List[str]:
    return [x for x in history if x in {"B", "P"}]


def _streak(non_tie: List[str]) -> Tuple[str, int]:
    if not non_tie:
        return "", 0
    last = non_tie[-1]
    n = 1
    for x in reversed(non_tie[:-1]):
        if x == last:
            n += 1
        else:
            break
    return last, n


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        n = max(1, len(clean))
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


def _pick_from_score(score: Dict[str, Any], min_edge: float = 0.001) -> str:
    b = float(score.get("B", 0.5))
    p = float(score.get("P", 0.5))
    if abs(b - p) < min_edge:
        return ""
    return "B" if b > p else "P"

# ============ 規律 / 牌路模型 ============
def _transition_prob(non_tie: List[str]) -> Dict[str, float]:
    counts = defaultdict(lambda: Counter())
    for a, b in zip(non_tie, non_tie[1:]):
        counts[a][b] += 1
    if not non_tie:
        return {"B": 0.5, "P": 0.5, "sample": 0}
    last = non_tie[-1]
    c = counts[last]
    sample = c["B"] + c["P"]
    alpha = float(os.getenv("MARKOV_ALPHA", "2.4"))
    b = (c["B"] + alpha) / (sample + 2 * alpha)
    p = (c["P"] + alpha) / (sample + 2 * alpha)
    shrink = min(1.0, sample / float(os.getenv("MARKOV_FULL_SAMPLE", "14")))
    b = 0.5 * (1 - shrink) + b * shrink
    p = 0.5 * (1 - shrink) + p * shrink
    return {"B": b, "P": p, "sample": sample}


def _ngram_score(non_tie: List[str], max_k: int = 6) -> Dict[str, Any]:
    """通用 N-Gram 回測：尋找最近 k 碼在本靴過去出現後，下一手較常接 B 或 P。"""
    if len(non_tie) < 10:
        return {"B": 0.5, "P": 0.5, "label": "NGram資料不足", "sample": 0, "strength": 0.0, "key": ""}

    seq = "".join(non_tie)
    upper_k = min(max_k, len(non_tie) - 1)

    for k in range(upper_k, 1, -1):
        key = seq[-k:]
        follows = []

        for i in range(0, len(seq) - k):
            if seq[i:i + k] == key and i + k < len(seq):
                follows.append(seq[i + k])

        if len(follows) >= 2:
            c = Counter(follows)
            total = c["B"] + c["P"]
            alpha = float(os.getenv("NGRAM_ALPHA", "1.6"))
            b_raw = (c["B"] + alpha) / (total + 2 * alpha)
            shrink = min(0.80, total / float(os.getenv("NGRAM_FULL_SAMPLE", "8")))
            b = 0.5 * (1 - shrink) + b_raw * shrink
            p = 1 - b
            return {
                "B": b,
                "P": p,
                "label": f"NGram{k}碼:{key}",
                "sample": total,
                "strength": min(0.22, 0.08 + total * 0.02),
                "key": key,
            }

    return {"B": 0.5, "P": 0.5, "label": "NGram無重複", "sample": 0, "strength": 0.0, "key": ""}


def _road_pattern_score(non_tie: List[str]) -> Dict[str, Any]:
    if len(non_tie) < 3:
        return {"B": 0.5, "P": 0.5, "label": "資料不足", "strength": 0.0}

    last, streak_n = _streak(non_tie)
    opp = "P" if last == "B" else "B"
    recent = non_tie[-12:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)

    b = p = 0.5
    label = "混合盤"
    strength = 0.08

    if streak_n >= 5:
        cont = 0.53 + min(0.05, (streak_n - 5) * 0.008)
        b = cont if last == "B" else 1 - cont
        p = cont if last == "P" else 1 - cont
        label = f"長龍{last}{streak_n}"
        strength = 0.18
    elif switch_rate >= 0.72 and len(recent) >= 6:
        b = 0.57 if opp == "B" else 0.43
        p = 0.57 if opp == "P" else 0.43
        label = "跳路偏強"
        strength = 0.16
    elif len(non_tie) >= 6 and non_tie[-6:] in [list("BBPPBB"), list("PPBBPP")]:
        next_side = non_tie[-2]
        b = 0.56 if next_side == "B" else 0.44
        p = 0.56 if next_side == "P" else 0.44
        label = "雙跳/兩房型"
        strength = 0.15
    elif len(non_tie) >= 8:
        key = "".join(non_tie[-4:])
        follows = []
        seq = "".join(non_tie)
        for i in range(0, len(seq) - 4):
            if seq[i:i + 4] == key and i + 4 < len(seq):
                follows.append(seq[i + 4])
        if follows:
            c = Counter(follows)
            total = c["B"] + c["P"]
            b_raw = c["B"] / total
            shrink = min(0.75, total / 10)
            b = 0.5 * (1 - shrink) + b_raw * shrink
            p = 1 - b
            label = f"四碼回測{key}"
            strength = min(0.20, 0.08 + total * 0.015)
        else:
            b_count = recent.count("B")
            p_count = recent.count("P")
            if abs(b_count - p_count) >= 4:
                scarce = "B" if b_count < p_count else "P"
                b = 0.54 if scarce == "B" else 0.46
                p = 0.54 if scarce == "P" else 0.46
                label = "短窗均衡修正"
                strength = 0.10

    return {"B": b, "P": p, "label": label, "strength": strength, "switch_rate": switch_rate, "streak": streak_n}


def _recent_score(non_tie: List[str]) -> Dict[str, float]:
    if not non_tie:
        return {"B": 0.5, "P": 0.5}
    recent = non_tie[-10:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)
    last, n = _streak(non_tie)
    opp = "P" if last == "B" else "B"
    if switch_rate > 0.66:
        side = opp
        edge = 0.055
    elif n >= 3:
        side = last
        edge = 0.045 + min(0.025, (n - 3) * 0.008)
    else:
        b_count = recent.count("B")
        p_count = recent.count("P")
        side = "B" if b_count < p_count else "P"
        edge = min(0.035, abs(b_count - p_count) * 0.006)
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}


def _balance_score(non_tie: List[str]) -> Dict[str, float]:
    if len(non_tie) < 8:
        return {"B": 0.5, "P": 0.5}
    b = non_tie.count("B")
    p = non_tie.count("P")
    diff = b - p
    edge = min(0.055, abs(diff) / max(1, len(non_tie)) * 0.16)
    side = "B" if diff < 0 else "P"
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}


def _streak_score(non_tie: List[str]) -> Dict[str, float]:
    last, n = _streak(non_tie)
    if not last:
        return {"B": 0.5, "P": 0.5}
    opp = "P" if last == "B" else "B"
    if n == 1:
        side, edge = opp, 0.025
    elif n == 2:
        side, edge = last, 0.030
    elif n == 3:
        side, edge = last, 0.045
    elif n >= 4:
        side, edge = last, min(0.075, 0.050 + (n - 4) * 0.008)
    else:
        side, edge = last, 0.0
    return {"B": 0.5 + edge if side == "B" else 0.5 - edge, "P": 0.5 + edge if side == "P" else 0.5 - edge}

# ============ RoadEngine：大路 / 下三路獨立主模型 ============
def _build_big_road(non_tie: List[str], rows: int = ROAD_ENGINE_ROWS) -> Dict[str, Any]:
    # 百家樂大路盤面：同邊直落；到底/卡位橫拖；換邊開新欄。
    rows = max(3, int(rows or 6))
    sequence = [x for x in non_tie if x in {"B", "P"}]
    grid: Dict[Tuple[int, int], str] = {}
    positions: List[Dict[str, Any]] = []
    last_side = ""
    row = 0
    col = 0

    for idx, side in enumerate(sequence):
        if idx == 0:
            row, col, move_type = 0, 0, "START"
        elif side != last_side:
            target_row, target_col = 0, col + 1
            while (target_row, target_col) in grid:
                target_col += 1
            row, col, move_type = target_row, target_col, "NEW_COLUMN"
        else:
            target_row, target_col = row + 1, col
            if target_row < rows and (target_row, target_col) not in grid:
                row, col, move_type = target_row, target_col, "VERTICAL_DROP"
            else:
                target_row, target_col = row, col + 1
                while (target_row, target_col) in grid:
                    target_col += 1
                row, col, move_type = target_row, target_col, "SIDE_DRAG"

        grid[(row, col)] = side
        positions.append({"i": idx, "side": side, "row": row, "col": col, "move_type": move_type})
        last_side = side

    col_heights = Counter()
    col_sides: Dict[int, str] = {}
    for (r, c), side in grid.items():
        col_heights[c] += 1
        if r == 0:
            col_sides[c] = side

    max_col = max([p["col"] for p in positions], default=0)
    last_pos = positions[-1] if positions else {"i": -1, "side": "", "row": 0, "col": 0, "move_type": "NONE"}
    return {
        "rows": rows,
        "sequence": sequence,
        "grid": grid,
        "positions": positions,
        "col_heights": dict(col_heights),
        "col_sides": col_sides,
        "max_col": max_col,
        "last": last_pos,
    }


def _derived_color_at(layout: Dict[str, Any], pos: Dict[str, Any], offset: int) -> int:
    # 下三路紅藍：offset=1 大眼仔；2 小路；3 蟑螂路。1=紅，-1=藍，0=資料不足。
    col = int(pos.get("col", 0))
    row = int(pos.get("row", 0))
    offset = int(offset)
    grid = layout.get("grid", {}) or {}
    heights = layout.get("col_heights", {}) or {}
    if col <= offset:
        return 0
    if row == 0:
        left_h = int(heights.get(col - 1, 0))
        compare_h = int(heights.get(col - 1 - offset, 0))
        if left_h <= 0 or compare_h <= 0:
            return 0
        return 1 if left_h == compare_h else -1
    compare_col = col - offset
    has_same_row = ((row, compare_col) in grid)
    has_prev_row = ((row - 1, compare_col) in grid)
    return 1 if has_same_row == has_prev_row else -1


def _derived_series(layout: Dict[str, Any], offset: int) -> List[int]:
    # 下三路必須逐局生成，不能用最後版面倒回去重算，避免未來格子污染過去紅藍。
    seq = layout.get("sequence")
    if seq:
        clean_seq = [x for x in seq if x in {"B", "P"}]
        cache_key = (int(offset), "".join(clean_seq))
        cache = getattr(_derived_series, "_cache", None)
        if cache is None:
            cache = {}
            setattr(_derived_series, "_cache", cache)
        if cache_key in cache:
            return list(cache[cache_key])
        series: List[int] = []
        for i in range(1, len(clean_seq) + 1):
            partial_layout = _build_big_road(clean_seq[:i])
            positions = partial_layout.get("positions", []) or []
            if not positions:
                continue
            color = _derived_color_at(partial_layout, positions[-1], offset)
            if color != 0:
                series.append(color)
        if len(cache) > 500:
            cache.clear()
        cache[cache_key] = list(series)
        return series

    series = []
    for pos in layout.get("positions", []):
        color = _derived_color_at(layout, pos, offset)
        if color != 0:
            series.append(color)
    return series


def _color_stats(series: List[int], lookback: int = ROAD_ENGINE_DERIVED_LOOKBACK) -> Dict[str, Any]:
    tail = series[-lookback:] if series else []
    if not tail:
        return {"last": 0, "red_rate": 0.5, "blue_rate": 0.5, "count": 0, "tail": ""}
    red = tail.count(1)
    blue = tail.count(-1)
    total = red + blue
    return {
        "last": tail[-1],
        "red_rate": round(red / total, 4) if total else 0.5,
        "blue_rate": round(blue / total, 4) if total else 0.5,
        "count": total,
        "tail": "".join("R" if x == 1 else "B" for x in tail),
    }


def _classify_bigroad_move(before_layout: Dict[str, Any], after_layout: Dict[str, Any], candidate: str) -> Dict[str, Any]:
    before_positions = before_layout.get("positions", []) or []
    after_positions = after_layout.get("positions", []) or []
    if not after_positions:
        return {"move_type": "NONE", "before": {}, "after": {}}
    after_pos = after_positions[-1]
    before_pos = before_positions[-1] if before_positions else {}
    if not before_pos:
        return {"move_type": "START", "before": before_pos, "after": after_pos}
    before_side = before_pos.get("side", "")
    before_row = int(before_pos.get("row", 0))
    before_col = int(before_pos.get("col", 0))
    after_row = int(after_pos.get("row", 0))
    after_col = int(after_pos.get("col", 0))
    if candidate != before_side:
        move_type = "NEW_COLUMN"
    elif after_col == before_col and after_row == before_row + 1:
        move_type = "VERTICAL_DROP"
    elif after_col > before_col and after_row == before_row:
        move_type = "SIDE_DRAG"
    else:
        move_type = after_pos.get("move_type", "CONTINUE_OTHER")
    return {"move_type": move_type, "before": before_pos, "after": after_pos,
            "before_row": before_row, "before_col": before_col, "after_row": after_row, "after_col": after_col}


def _candidate_derived_color_info(non_tie: List[str], candidate: str, offset: int) -> Dict[str, Any]:
    if candidate not in {"B", "P"}:
        return {"candidate": candidate, "new_color": 0, "new_color_text": "N", "before_len": 0, "after_len": 0, "pos": {}, "move": {}, "structure": {}}

    before_layout = _build_big_road(non_tie)
    before_series = _derived_series(before_layout, offset)
    after_layout = _build_big_road(non_tie + [candidate])
    after_series = _derived_series(after_layout, offset)
    new_color = after_series[-1] if len(after_series) > len(before_series) else 0
    move_info = _classify_bigroad_move(before_layout, after_layout, candidate)
    pos = move_info.get("after", {}) or {}
    row = int(pos.get("row", 0))
    col = int(pos.get("col", 0))
    grid = after_layout.get("grid", {}) or {}
    heights = after_layout.get("col_heights", {}) or {}
    structure: Dict[str, Any] = {
        "move_type": move_info.get("move_type", "NONE"), "row": row, "col": col, "offset": offset,
        "is_new_column": row == 0, "is_vertical_drop": move_info.get("move_type") == "VERTICAL_DROP",
        "is_side_drag": move_info.get("move_type") == "SIDE_DRAG", "is_neat": False,
        "has_same_row": None, "has_prev_row": None, "left_height": None, "compare_height": None, "relation": "",
    }
    if col <= offset:
        structure["relation"] = "資料不足"
    elif row == 0:
        left_col = col - 1
        compare_col = col - 1 - offset
        left_h = int(heights.get(left_col, 0))
        compare_h = int(heights.get(compare_col, 0))
        is_neat = bool(left_h > 0 and compare_h > 0 and left_h == compare_h)
        structure.update({"left_col": left_col, "compare_col": compare_col, "left_height": left_h,
                          "compare_height": compare_h, "is_neat": is_neat,
                          "relation": f"新欄高度{'齊整' if is_neat else '不齊'}:{left_h}/{compare_h}"})
    else:
        compare_col = col - offset
        has_same_row = ((row, compare_col) in grid)
        has_prev_row = ((row - 1, compare_col) in grid)
        is_neat = bool(has_same_row == has_prev_row)
        relation = ("有" if has_same_row else "無") + "/" + ("有" if has_prev_row else "無")
        structure.update({"compare_col": compare_col, "has_same_row": has_same_row, "has_prev_row": has_prev_row,
                          "is_neat": is_neat, "relation": f"有無{relation}:{'齊整' if is_neat else '不齊'}"})
    return {"candidate": candidate, "new_color": new_color, "new_color_text": "R" if new_color == 1 else "B" if new_color == -1 else "N",
            "before_len": len(before_series), "after_len": len(after_series), "pos": pos, "move": move_info, "structure": structure}


def _score_candidate_color_pattern(series: List[int], candidate_color: int, lookback: Optional[int] = None) -> Dict[str, Any]:
    if lookback is None:
        lookback = DERIVED_CANDIDATE_LOOKBACK
    if candidate_color not in {1, -1}:
        return {"score": 0.5, "confidence": 0.0, "expected_color": 0, "expected_color_text": "N", "candidate_color_text": "N", "label": "候選無新色"}
    tail = series[-lookback:] if series else []
    if len(tail) < 3:
        return {"score": 0.5, "confidence": 0.0, "expected_color": 0, "expected_color_text": "N", "candidate_color_text": "R" if candidate_color == 1 else "B", "label": "紅藍樣本不足"}
    last_color = tail[-1]
    color_streak = 1
    for x in reversed(tail[:-1]):
        if x == last_color:
            color_streak += 1
        else:
            break
    switches = sum(1 for a, b in zip(tail, tail[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(tail) - 1), 0.5)
    red_rate = tail.count(1) / len(tail)
    blue_rate = tail.count(-1) / len(tail)
    expected_color, edge, label = 0, 0.0, "紅藍中性"
    if switch_rate >= DERIVED_COLOR_JUMP_RATE and len(tail) >= 6:
        expected_color = -last_color
        edge = min(0.16, 0.09 + (switch_rate - DERIVED_COLOR_JUMP_RATE) * 0.28)
        label = "下三路紅藍單跳"
    elif color_streak >= DERIVED_COLOR_STREAK_MIN:
        expected_color = last_color
        edge = min(0.17, 0.09 + (color_streak - DERIVED_COLOR_STREAK_MIN) * 0.025)
        label = f"下三路{'紅' if last_color == 1 else '藍'}連{color_streak}"
    elif abs(red_rate - blue_rate) >= DERIVED_COLOR_RATIO_GAP:
        expected_color = 1 if red_rate > blue_rate else -1
        edge = min(0.11, abs(red_rate - blue_rate) * 0.22)
        label = "下三路紅藍比例偏態"
    else:
        found = False
        max_k = min(max(2, DERIVED_COLOR_NGRAM_MAX), len(tail) - 1)
        for k in range(max_k, 1, -1):
            key = tail[-k:]
            follows = [tail[i + k] for i in range(0, len(tail) - k) if tail[i:i + k] == key and i + k < len(tail)]
            if len(follows) >= 2:
                red_follow = follows.count(1)
                blue_follow = follows.count(-1)
                if red_follow != blue_follow:
                    expected_color = 1 if red_follow > blue_follow else -1
                    edge = min(0.12, abs(red_follow - blue_follow) / len(follows) * 0.16)
                    label = f"下三路紅藍NGram{k}"
                    found = True
                    break
        if not found:
            expected_color, edge, label = last_color, 0.035, "下三路弱續勢"
    score = 0.5 + edge if candidate_color == expected_color else 0.5 - edge
    return {"score": round(score, 5), "confidence": round(min(1.0, abs(score - 0.5) * 2.8), 4),
            "expected_color": expected_color, "expected_color_text": "R" if expected_color == 1 else "B" if expected_color == -1 else "N",
            "candidate_color_text": "R" if candidate_color == 1 else "B", "label": label,
            "switch_rate": round(switch_rate, 4), "color_streak": color_streak, "red_rate": round(red_rate, 4),
            "blue_rate": round(blue_rate, 4), "tail": "".join("R" if x == 1 else "B" for x in tail)}


def _score_candidate_structure(info: Dict[str, Any], series: List[int]) -> Dict[str, Any]:
    """只評估「候選落點移動風險」，不重複把紅藍的齊整/不齊再算一次。

    下三路紅藍本身已由欄高、有無與齊整關係推導；舊版再次對 is_neat
    加減分，等於同一特徵重複計分。這裡只保留較獨立的落點型態：
    直落 / 新欄 / 黏邊橫拖，且幅度刻意限制在小範圍。
    """
    structure = info.get("structure", {}) or {}
    move_type = structure.get("move_type", "NONE")
    new_color = int(info.get("new_color", 0) or 0)

    edge = 0.0
    reasons: List[str] = []

    # 不再使用 DERIVED_STRUCTURE_NEAT_BONUS / MISMATCH_PENALTY，避免與紅藍重複。
    if move_type == "VERTICAL_DROP":
        edge += min(0.015, max(0.0, DERIVED_STRUCTURE_DROP_BONUS))
        reasons.append("直落移動")
    elif move_type == "NEW_COLUMN":
        edge += min(0.010, max(0.0, DERIVED_STRUCTURE_NEWCOL_BONUS))
        reasons.append("新欄移動")
    elif move_type == "SIDE_DRAG":
        edge -= min(0.018, max(0.0, DERIVED_STRUCTURE_SIDE_DRAG_PENALTY))
        reasons.append("黏邊橫拖")
    else:
        reasons.append("移動中性")

    # 沒有產生新下路顏色時，結構訊號只保留一半，避免空資料被硬加分。
    if new_color not in {1, -1}:
        edge *= 0.5
        reasons.append("無新色降權")

    edge = _clamp(edge, -0.025, 0.025)
    return {
        "score": round(0.5 + edge, 5),
        "edge": round(edge, 5),
        "label": "+".join(reasons),
        "move_type": move_type,
        "is_neat": bool(structure.get("is_neat", False)),
        "relation": structure.get("relation", ""),
        "structure": structure,
        "deduplicated": True,
    }

def _score_column_shape(non_tie: List[str], candidate: str) -> Dict[str, Any]:
    # 前排大路欄型分：
    # 讓候選判斷不只看下三路紅藍，也看下一口落點是否符合最近欄高節奏。
    if not USE_DERIVED_COLUMN_SHAPE or candidate not in {"B", "P"} or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return {"score": 0.5, "edge": 0.0, "label": "欄型關閉或資料不足", "tail_heights": []}

    try:
        before_layout = _build_big_road(non_tie)
        after_layout = _build_big_road(non_tie + [candidate])
        move = _classify_bigroad_move(before_layout, after_layout, candidate)
        move_type = move.get("move_type", "NONE")

        after_last = after_layout.get("last", {}) or {}
        col = int(after_last.get("col", 0))
        row = int(after_last.get("row", 0))
        heights = after_layout.get("col_heights", {}) or {}

        lookback = max(3, DERIVED_COLUMN_SHAPE_LOOKBACK)
        start_col = max(0, col - lookback)
        prev_heights = [int(heights.get(c, 0)) for c in range(start_col, col) if int(heights.get(c, 0)) > 0]
        tail_heights = prev_heights[-lookback:]
        current_height = int(heights.get(col, 0))

        if not tail_heights:
            return {"score": 0.5, "edge": 0.0, "label": "欄型樣本不足", "tail_heights": [], "current_height": current_height, "move_type": move_type}

        avg_h = sum(tail_heights) / len(tail_heights)
        max_h = max(tail_heights)
        min_h = min(tail_heights)
        last_h = tail_heights[-1]
        repeated = tail_heights.count(last_h) >= max(2, len(tail_heights) // 2)

        edge = 0.0
        reasons = []

        if move_type == "VERTICAL_DROP":
            # 直落如果仍在前排欄高範圍內，視為欄型延續；若超出太多，視為可能疲乏。
            if current_height <= max_h + 1:
                edge += DERIVED_COLUMN_NEAT_BONUS
                reasons.append("直落貼近前排欄高")
            else:
                edge -= DERIVED_COLUMN_BREAK_PENALTY
                reasons.append("直落超出前排欄高")
        elif move_type == "NEW_COLUMN":
            # 換邊開新欄：如果前一欄高度接近近期欄型，代表上一欄完成得較漂亮。
            prev_col_h = int(heights.get(col - 1, 0))
            if repeated and prev_col_h == last_h:
                edge += DERIVED_COLUMN_NEAT_BONUS * 0.85
                reasons.append("新欄承接重複欄型")
            elif abs(prev_col_h - avg_h) <= 1.0:
                edge += DERIVED_COLUMN_NEAT_BONUS * 0.55
                reasons.append("新欄承接平均欄高")
            else:
                edge -= DERIVED_COLUMN_BREAK_PENALTY * 0.60
                reasons.append("新欄前欄破欄型")
        elif move_type == "SIDE_DRAG":
            # 橫拖代表到底/卡位，通常要保守一點；若近期欄高本來就很高，扣分較小。
            if max_h >= ROAD_ENGINE_ROWS - 1:
                edge -= DERIVED_COLUMN_DRAG_PENALTY * 0.55
                reasons.append("高欄橫拖")
            else:
                edge -= DERIVED_COLUMN_DRAG_PENALTY
                reasons.append("黏邊橫拖")
        else:
            if abs(current_height - avg_h) <= 1.0:
                edge += DERIVED_COLUMN_NEAT_BONUS * 0.35
                reasons.append("欄高接近平均")

        # 如果最近欄型很整齊，候選造成明顯偏離就扣分。
        if repeated and current_height not in {last_h, last_h + 1, 1}:
            edge -= DERIVED_COLUMN_BREAK_PENALTY * 0.45
            reasons.append("偏離重複欄型")

        edge = _clamp(edge, -DERIVED_COLUMN_MAX_EDGE, DERIVED_COLUMN_MAX_EDGE)

        return {
            "score": round(0.5 + edge, 5),
            "edge": round(edge, 5),
            "label": "+".join(reasons) if reasons else "欄型中性",
            "move_type": move_type,
            "row": row,
            "col": col,
            "current_height": current_height,
            "tail_heights": tail_heights,
            "avg_height": round(avg_h, 3),
            "max_height": max_h,
            "min_height": min_h,
        }
    except Exception as e:
        return {"score": 0.5, "edge": 0.0, "label": f"欄型錯誤:{e}", "tail_heights": []}


def _combine_candidate_scores(color_score: float, structure_score: float, column_score: Optional[float] = None) -> float:
    """合併單一衍生路候選分數。

    單一路只合併紅藍節奏與小幅移動風險；欄型分不在此函數重複加入。
    欄型分只會在 _down3_family_score() 對整個下三路家族加入一次。
    即使 Render 還留著舊的 0.55/0.45，這裡仍限制結構權重最多 28%。
    """
    color_w = max(0.72, float(DERIVED_COLOR_SCORE_WEIGHT))
    struct_w = min(0.28, max(0.0, float(DERIVED_STRUCTURE_SCORE_WEIGHT)))
    total = max(0.0001, color_w + struct_w)
    result = float(color_score) * (color_w / total) + float(structure_score) * (struct_w / total)

    # 相容舊呼叫，但欄型最多只給極小權重；新版正常流程不會在單一路傳入。
    if column_score is not None:
        col_w = min(0.08, max(0.0, float(DERIVED_COLUMN_SHAPE_WEIGHT)))
        result = result * (1.0 - col_w) + float(column_score) * col_w
    return float(result)


def _detect_dense_board(non_tie: List[str]) -> Dict[str, Any]:
    """偵測短欄密集且欄高變化大的盤面。

    這不是預測方向，只用來降低高度相關的下三路訊號，並在與大路衝突時
    提高最後確認門檻。
    """
    default = {
        "enabled": USE_DENSE_BOARD_GUARD,
        "is_dense": False,
        "score": 0.0,
        "label": "密集盤資料不足",
        "columns": 0,
        "avg_height": 0.0,
        "short_ratio": 0.0,
        "height_change_rate": 0.0,
        "switch_rate": 0.5,
        "tail_heights": [],
    }
    if not USE_DENSE_BOARD_GUARD or len(non_tie) < DENSE_BOARD_MIN_HISTORY:
        return default

    layout = _build_big_road(non_tie)
    heights_map = layout.get("col_heights", {}) or {}
    ordered_cols = sorted(int(c) for c in heights_map.keys())
    heights = [int(heights_map.get(c, 0)) for c in ordered_cols if int(heights_map.get(c, 0)) > 0]
    tail = heights[-10:]
    if len(tail) < DENSE_BOARD_MIN_COLUMNS:
        return {**default, "columns": len(tail), "tail_heights": tail, "label": "密集盤欄數不足"}

    avg_h = sum(tail) / len(tail)
    short_ratio = sum(1 for h in tail if h <= 3) / len(tail)
    height_changes = sum(1 for a, b in zip(tail, tail[1:]) if a != b)
    height_change_rate = _safe_div(height_changes, max(1, len(tail) - 1), 0.0)
    recent = non_tie[-16:]
    switch_rate = _safe_div(sum(1 for a, b in zip(recent, recent[1:]) if a != b), max(1, len(recent) - 1), 0.5)

    avg_component = _clamp((DENSE_BOARD_MAX_AVG_HEIGHT - avg_h + 1.0) / max(1.0, DENSE_BOARD_MAX_AVG_HEIGHT), 0.0, 1.0)
    short_component = _clamp(short_ratio, 0.0, 1.0)
    change_component = _clamp(height_change_rate, 0.0, 1.0)
    switch_component = 1.0 if DENSE_BOARD_SWITCH_LOW <= switch_rate <= DENSE_BOARD_SWITCH_HIGH else 0.35
    score = _clamp(avg_component * 0.25 + short_component * 0.30 + change_component * 0.30 + switch_component * 0.15, 0.0, 1.0)

    is_dense = bool(
        len(tail) >= DENSE_BOARD_MIN_COLUMNS
        and avg_h <= DENSE_BOARD_MAX_AVG_HEIGHT
        and short_ratio >= DENSE_BOARD_SHORT_COLUMN_RATIO
        and height_change_rate >= DENSE_BOARD_HEIGHT_CHANGE_RATE
        and DENSE_BOARD_SWITCH_LOW <= switch_rate <= DENSE_BOARD_SWITCH_HIGH
    )
    label = (
        f"密集盤保護:欄{len(tail)} 均高{avg_h:.2f} 短欄{short_ratio:.0%} 變高{height_change_rate:.0%}"
        if is_dense else
        f"非密集盤:均高{avg_h:.2f} 短欄{short_ratio:.0%} 變高{height_change_rate:.0%}"
    )
    return {
        "enabled": True,
        "is_dense": is_dense,
        "score": round(score, 4),
        "label": label,
        "columns": len(tail),
        "avg_height": round(avg_h, 4),
        "short_ratio": round(short_ratio, 4),
        "height_change_rate": round(height_change_rate, 4),
        "switch_rate": round(switch_rate, 4),
        "tail_heights": tail,
    }


def _limit_ask_road_performance_for_dense(performance: Dict[str, Any], dense_board: Dict[str, Any]) -> Dict[str, Any]:
    """密集盤仍保留問路記憶降權，但禁止短樣本把下三路放大。"""
    if not (
        USE_ASK_ROAD_MEMORY
        and ASK_ROAD_MEMORY_NO_BOOST_DENSE
        and dense_board.get("is_dense")
        and isinstance(performance, dict)
    ):
        return performance

    cloned = dict(performance)
    models = {}
    for name, stat in (performance.get("models") or {}).items():
        new_stat = dict(stat or {})
        old_factor = float(new_stat.get("factor", 1.0) or 1.0)
        new_stat["factor"] = round(min(1.0, old_factor), 4)
        if old_factor > 1.0:
            new_stat["dense_boost_blocked"] = True
        models[name] = new_stat
    cloned["models"] = models
    cloned["dense_boost_blocked"] = True
    cloned["label"] = f"{performance.get('label', '問路記憶')}|密集盤禁止加權"
    return cloned


def _down3_family_score(non_tie: List[str], road_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """將大眼仔、小路、蟑螂路整合成單一家庭訊號。

    三條路仍各自保留紅藍與命中記憶，但對外最多只產生一個方向。
    欄型分在這裡對 B/P 候選各計算一次，不再在三條路重複加入。
    """
    default = {
        "B": 0.5,
        "P": 0.5,
        "pick": "",
        "label": "下三路家族資料不足",
        "confidence": 0.0,
        "strength": 0.0,
        "valid_roads": 0,
        "agree_count": 0,
        "agreement_ratio": 0.0,
        "gap": 0.0,
        "raw_gap": 0.0,
        "column_gap": 0.0,
        "details": {},
        "dense_board": _detect_dense_board(non_tie),
    }
    if not USE_DOWN3_FAMILY or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return default

    weights = {
        "big_eye": max(0.0, DOWN3_FAMILY_BIG_EYE_FACTOR),
        "small_road": max(0.0, DOWN3_FAMILY_SMALL_ROAD_FACTOR),
        "cockroach": max(0.0, DOWN3_FAMILY_COCKROACH_FACTOR),
    }
    valid = []
    details: Dict[str, Any] = {}
    for name, base_w in weights.items():
        score = road_scores.get(name, {}) or {}
        stats_count = int((score.get("stats") or {}).get("count", 0) or 0)
        b = float(score.get("B", 0.5) or 0.5)
        p = float(score.get("P", 0.5) or 0.5)
        gap = b - p
        candidate = score.get("candidate") or {}
        has_candidate = bool(candidate) or stats_count >= DERIVED_ROAD_MIN_COUNT
        directional = abs(gap) >= DOWN3_FAMILY_ROAD_MIN_GAP
        pick = "B" if gap > 0 else "P" if gap < 0 else ""
        details[name] = {
            "B": round(b, 5),
            "P": round(p, 5),
            "gap": round(gap, 5),
            "pick": pick if directional else "",
            "directional": directional,
            "weight": round(base_w, 4),
            "count": stats_count,
            "label": score.get("label", ""),
        }
        if has_candidate and stats_count >= DERIVED_ROAD_MIN_COUNT and base_w > 0:
            valid.append((name, base_w, gap, directional, pick))

    if len(valid) < DOWN3_FAMILY_MIN_VALID_ROADS:
        return {**default, "details": details, "valid_roads": len(valid), "label": f"下三路家族有效路僅{len(valid)}條"}

    total_w = sum(x[1] for x in valid) or 1.0
    raw_gap = sum(w * gap for _, w, gap, _, _ in valid) / total_w
    directional = [x for x in valid if x[3] and x[4] in {"B", "P"}]
    b_agree = sum(w for _, w, _, _, pick in directional if pick == "B")
    p_agree = sum(w for _, w, _, _, pick in directional if pick == "P")
    agree_side = "B" if b_agree > p_agree else "P" if p_agree > b_agree else ""
    agree_count = sum(1 for _, _, _, _, pick in directional if pick == agree_side) if agree_side else 0
    directional_w = b_agree + p_agree
    agreement_ratio = max(b_agree, p_agree) / directional_w if directional_w > 0 else 0.0

    # 三路不同向時只保留柔性偏移，不讓弱多數變成完整一票。
    if directional and agreement_ratio < 0.999:
        raw_gap *= _clamp(DOWN3_FAMILY_DISAGREE_SHRINK + (agreement_ratio - 0.5) * 0.35, 0.35, 0.90)

    # 全局大路欄型只加入一次。
    b_column = _score_column_shape(non_tie, "B")
    p_column = _score_column_shape(non_tie, "P")
    column_raw_diff = float(b_column.get("score", 0.5)) - float(p_column.get("score", 0.5))
    column_gap = _clamp(column_raw_diff * DOWN3_FAMILY_COLUMN_SCALE, -0.014, 0.014)

    dense_board = _detect_dense_board(non_tie)
    family_gap = raw_gap + column_gap
    if dense_board.get("is_dense"):
        family_gap *= _clamp(DOWN3_FAMILY_DENSE_SHRINK, 0.30, 1.0)

    family_gap = _clamp(family_gap, -DOWN3_FAMILY_MAX_GAP, DOWN3_FAMILY_MAX_GAP)
    enough_agree = agree_count >= DOWN3_FAMILY_MIN_AGREE
    enough_gap = abs(family_gap) >= DOWN3_FAMILY_MIN_GAP
    pick = ("B" if family_gap > 0 else "P") if enough_agree and enough_gap else ""

    # 未過門檻仍保留少量柔性偏移供融合，但不投方向票。
    output_gap = family_gap if pick else family_gap * 0.35
    b = _clamp(0.5 + output_gap / 2.0, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)
    p = 1.0 - b
    confidence = 0.0
    if pick:
        gap_strength = _clamp((abs(family_gap) - DOWN3_FAMILY_MIN_GAP) / max(0.001, DOWN3_FAMILY_STRONG_GAP - DOWN3_FAMILY_MIN_GAP), 0.0, 1.0)
        confidence = _clamp(0.46 + agreement_ratio * 0.22 + gap_strength * 0.22, 0.0, 0.88)
        label = f"下三路家族偏{'莊' if pick == 'B' else '閒'} 路{agree_count}/{len(valid)} 差{abs(family_gap)*100:.1f}%"
    else:
        confidence = _clamp(0.20 + agreement_ratio * 0.18 + abs(family_gap) * 2.0, 0.0, 0.48)
        if not enough_agree:
            label = f"下三路家族分歧 路{agree_count}/{len(valid)}"
        else:
            label = f"下三路家族差距不足 {abs(family_gap)*100:.1f}%<{DOWN3_FAMILY_MIN_GAP*100:.1f}%"

    return {
        "B": round(b, 5),
        "P": round(p, 5),
        "pick": pick,
        "label": label,
        "confidence": round(confidence, 4),
        "strength": round(confidence * 0.25, 4),
        "valid_roads": len(valid),
        "agree_count": agree_count,
        "agreement_ratio": round(agreement_ratio, 4),
        "gap": round(family_gap, 5),
        "raw_gap": round(raw_gap, 5),
        "column_gap": round(column_gap, 5),
        "column_once": {"B": b_column, "P": p_column},
        "details": details,
        "dense_board": dense_board,
    }


def _collapse_down3_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """把三條下路權重合併成一個家族權重，並限制家族總影響。"""
    adjusted = dict(weights or {})
    original_total = sum(max(0.0, float(v)) for v in adjusted.values())
    down_total = sum(max(0.0, float(adjusted.get(k, 0.0))) for k in ("big_eye", "small_road", "cockroach"))
    for k in ("big_eye", "small_road", "cockroach"):
        adjusted[k] = 0.0
    adjusted["down3_family"] = min(max(0.0, down_total), min(0.28, max(0.0, DOWN3_FAMILY_MAX_WEIGHT))) if USE_DOWN3_FAMILY else 0.0

    # 重新正規化會把被移除的重複權重自然分配給大路與其他獨立模型。
    if original_total <= 0:
        return _normalize_weights(adjusted)
    return _normalize_weights(adjusted)


def _strong_pick_from_score(score: Dict[str, Any], min_gap: float = FINAL_CONFIRM_SCORE_GAP) -> str:
    try:
        b = float(score.get("B", 0.5))
        p = float(score.get("P", 0.5))
        if abs(b - p) < min_gap:
            return ""
        return "B" if b > p else "P"
    except Exception:
        return ""


def _final_confirmation_summary(
    target: str,
    big_road: Dict[str, Any],
    pattern_replay: Dict[str, Any],
    independent_scores: Optional[Dict[str, Dict[str, Any]]] = None,
    ml_pick: str = "",
    ml_gap: float = 0.0,
) -> Dict[str, Any]:
    """計算下三路家族候選是否取得真正較獨立來源確認。"""
    sources: Dict[str, str] = {}
    if target not in {"B", "P"}:
        return {"target": target, "count": 0, "confirmed": False, "sources": sources, "non_road_count": 0}

    br_pick = _strong_pick_from_score(big_road, min_gap=FINAL_CONFIRM_SCORE_GAP)
    if br_pick == target:
        sources["big_road"] = br_pick

    pr_pick = ""
    if (
        pattern_replay.get("state") == "REPLAY_MATCH"
        and float(pattern_replay.get("confidence", 0.0) or 0.0) >= FINAL_CONFIRM_PATTERN_CONF
        and float(pattern_replay.get("edge", 0.0) or 0.0) >= FINAL_CONFIRM_PATTERN_EDGE
    ):
        pr_pick = str(pattern_replay.get("bias_side", ""))
        if pr_pick == target:
            sources["pattern_replay"] = pr_pick

    for name, score in (independent_scores or {}).items():
        pick = _strong_pick_from_score(score, min_gap=FINAL_CONFIRM_SCORE_GAP)
        if pick == target:
            sources[name] = pick

    if ml_pick == target and ml_gap >= FINAL_CONFIRM_SCORE_GAP:
        sources["ml"] = ml_pick

    non_road_count = sum(1 for name in sources if name != "big_road")
    return {
        "target": target,
        "count": len(sources),
        "non_road_count": non_road_count,
        "confirmed": len(sources) >= max(1, FINAL_CONFIRM_MIN_SOURCES),
        "sources": sources,
        "pattern_replay_pick": pr_pick,
        "big_road_pick": br_pick,
    }

def _candidate_scores_to_side_prob(b_score: float, p_score: float, max_edge: Optional[float] = None) -> Tuple[float, float, float]:
    if max_edge is None:
        max_edge = DERIVED_CANDIDATE_MAX_EDGE
    edge = _clamp((float(b_score) - float(p_score)) * 0.18, -max_edge, max_edge)
    return 0.5 + edge, 0.5 - edge, abs(edge)


def _roadmap_ask_road_debug(non_tie: List[str]) -> Dict[str, Any]:
    layout = _build_big_road(non_tie)
    result: Dict[str, Any] = {"current_big_road": {"last": layout.get("last", {}), "max_col": layout.get("max_col", 0), "col_heights": layout.get("col_heights", {})}}
    for candidate in ["B", "P"]:
        result[f"ask_{candidate}"] = {"candidate": candidate, "candidate_text": "莊" if candidate == "B" else "閒", "roads": {}}
        for offset, road_key in {1: "big_eye", 2: "small_road", 3: "cockroach"}.items():
            info = _candidate_derived_color_info(non_tie, candidate, offset)
            result[f"ask_{candidate}"]["roads"][road_key] = {"color": info.get("new_color_text", "N"), "pos": info.get("pos", {}), "move_type": info.get("move", {}).get("move_type", ""), "structure": info.get("structure", {})}
    return result


def _big_road_score(non_tie: List[str]) -> Dict[str, Any]:
    """大路獨立模型：負責長龍、跳路、欄高、黏邊、斷龍壓力。"""
    default = {
        "B": 0.5, "P": 0.5, "label": "大路資料不足", "strength": 0.0,
        "break_risk": 0.0, "big_road": {}, "red_pressure": 0.5, "blue_pressure": 0.5,
    }
    if not USE_ROAD_ENGINE or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return default

    layout = _build_big_road(non_tie)
    last_side, streak_n = _streak(non_tie)
    if not last_side:
        return default
    opp = "P" if last_side == "B" else "B"

    recent = non_tie[-16:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)

    last = layout.get("last", {})
    last_col = int(last.get("col", 0))
    last_row = int(last.get("row", 0))
    col_heights = layout.get("col_heights", {})
    current_col_height = int(col_heights.get(last_col, 0))

    big_eye_stats = _color_stats(_derived_series(layout, 1))
    small_road_stats = _color_stats(_derived_series(layout, 2))
    cockroach_stats = _color_stats(_derived_series(layout, 3))

    red_rates = [float(x.get("red_rate", 0.5)) for x in [big_eye_stats, small_road_stats, cockroach_stats] if x.get("count", 0) > 0]
    blue_rates = [float(x.get("blue_rate", 0.5)) for x in [big_eye_stats, small_road_stats, cockroach_stats] if x.get("count", 0) > 0]
    red_pressure = sum(red_rates) / len(red_rates) if red_rates else 0.5
    blue_pressure = sum(blue_rates) / len(blue_rates) if blue_rates else 0.5

    break_risk = 0.0
    if streak_n >= ROAD_ENGINE_BREAK_STREAK:
        break_risk += 0.24
    if last_row >= ROAD_ENGINE_ROWS - 1:
        break_risk += 0.16
    if blue_pressure >= 0.58:
        break_risk += min(0.24, (blue_pressure - 0.5) * 0.70)
    if switch_rate >= 0.72:
        break_risk += 0.10
    break_risk = _clamp(break_risk, 0.0, 0.85)

    label = "大路混合"
    strength = 0.10
    side = last_side
    edge = 0.022

    if switch_rate >= 0.72:
        side = opp
        edge = 0.050 + min(0.018, (switch_rate - 0.72) * 0.12)
        label = "大路單跳"
        strength = 0.16
    elif streak_n >= 4:
        cont_edge = 0.050 + min(0.030, (streak_n - 4) * 0.007)
        label = "大路長龍延續"
        if blue_pressure >= 0.60 or break_risk >= 0.62:
            side = opp
            edge = min(0.052, cont_edge * 0.68 + ROAD_ENGINE_BLUE_BREAK_BIAS * 0.50)
            label = "大路斷龍壓力"
        else:
            side = last_side
            edge = cont_edge + (ROAD_ENGINE_RED_CONT_BIAS if red_pressure >= 0.58 else 0.0)
        edge = _clamp(edge, 0.025, 0.085)
        strength = 0.18 + min(0.06, streak_n * 0.006)
    elif current_col_height >= 3:
        side = last_side
        edge = 0.038
        label = "大路欄高延續"
        strength = 0.14
    elif blue_pressure >= 0.64:
        side = opp
        edge = 0.034 + ROAD_ENGINE_BLUE_BREAK_BIAS * 0.35
        label = "大路藍路變化"
        strength = 0.13
    elif red_pressure >= 0.64:
        side = last_side
        edge = 0.034 + ROAD_ENGINE_RED_CONT_BIAS * 0.35
        label = "大路紅路整齊"
        strength = 0.13

    b = 0.5 + edge if side == "B" else 0.5 - edge
    p = 1 - b
    return {
        "B": b,
        "P": p,
        "label": label,
        "strength": round(strength, 4),
        "break_risk": round(break_risk, 4),
        "red_pressure": round(red_pressure, 4),
        "blue_pressure": round(blue_pressure, 4),
        "big_road": {
            "last_side": last_side,
            "last_col": last_col,
            "last_row": last_row,
            "current_col_height": current_col_height,
            "max_col": layout.get("max_col", 0),
            "is_dragon": streak_n >= 4,
            "streak": streak_n,
            "switch_rate_16": round(switch_rate, 4),
        },
    }


def _derived_road_score(non_tie: List[str], offset: int, road_key: str, display_name: str) -> Dict[str, Any]:
    """單一衍生路：只評估該路紅藍節奏 + 小幅落點移動風險。

    欄型分不在這裡加入；三路完成後由 _down3_family_score() 全局只算一次。
    """
    default = {
        "B": 0.5,
        "P": 0.5,
        "label": f"{display_name}資料不足",
        "strength": 0.0,
        "road_key": road_key,
        "stats": {"last": 0, "red_rate": 0.5, "blue_rate": 0.5, "count": 0, "tail": ""},
        "red_pressure": 0.5,
        "blue_pressure": 0.5,
        "candidate": {},
    }
    if not USE_ROAD_ENGINE or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return default

    layout = _build_big_road(non_tie)
    series = _derived_series(layout, offset=offset)
    stats = _color_stats(series)
    count = int(stats.get("count", 0))
    if count < DERIVED_ROAD_MIN_COUNT:
        return {**default, "stats": stats, "label": f"{display_name}樣本不足"}

    b_info = _candidate_derived_color_info(non_tie, "B", offset)
    p_info = _candidate_derived_color_info(non_tie, "P", offset)
    b_color_eval = _score_candidate_color_pattern(series, int(b_info.get("new_color", 0)))
    p_color_eval = _score_candidate_color_pattern(series, int(p_info.get("new_color", 0)))
    b_struct_eval = _score_candidate_structure(b_info, series)
    p_struct_eval = _score_candidate_structure(p_info, series)

    b_score = _combine_candidate_scores(float(b_color_eval.get("score", 0.5)), float(b_struct_eval.get("score", 0.5)))
    p_score = _combine_candidate_scores(float(p_color_eval.get("score", 0.5)), float(p_struct_eval.get("score", 0.5)))
    b, p, edge = _candidate_scores_to_side_prob(b_score, p_score, max_edge=DERIVED_CANDIDATE_MAX_EDGE)

    if edge < DERIVED_CANDIDATE_MIN_EDGE:
        label = f"{display_name}候選接近"
        strength = 0.05
    else:
        pick_text = "莊" if b > p else "閒"
        label = f"{display_name}候選偏{pick_text}"
        strength = 0.08 + min(0.12, edge * 1.8)

    return {
        "B": round(b, 5),
        "P": round(p, 5),
        "label": label,
        "strength": round(strength, 4),
        "road_key": road_key,
        "stats": stats,
        "red_pressure": round(float(stats.get("red_rate", 0.5)), 4),
        "blue_pressure": round(float(stats.get("blue_rate", 0.5)), 4),
        "tail": stats.get("tail", ""),
        "candidate": {
            "B": {
                "new_color": b_info.get("new_color_text", "N"),
                "color_score": round(float(b_color_eval.get("score", 0.5)), 5),
                "structure_score": round(float(b_struct_eval.get("score", 0.5)), 5),
                "column_score": 0.5,
                "score": round(b_score, 5),
                "color_eval": b_color_eval,
                "structure_eval": b_struct_eval,
                "column_eval": {"score": 0.5, "label": "欄型由家族層只算一次"},
                "pos": b_info.get("pos", {}),
                "structure": b_info.get("structure", {}),
            },
            "P": {
                "new_color": p_info.get("new_color_text", "N"),
                "color_score": round(float(p_color_eval.get("score", 0.5)), 5),
                "structure_score": round(float(p_struct_eval.get("score", 0.5)), 5),
                "column_score": 0.5,
                "score": round(p_score, 5),
                "color_eval": p_color_eval,
                "structure_eval": p_struct_eval,
                "column_eval": {"score": 0.5, "label": "欄型由家族層只算一次"},
                "pos": p_info.get("pos", {}),
                "structure": p_info.get("structure", {}),
            },
            "edge": round(edge, 5),
            "diff": round(b_score - p_score, 5),
            "column_applied_here": False,
        },
    }

def _big_eye_score(non_tie: List[str]) -> Dict[str, Any]:
    return _derived_road_score(non_tie, offset=1, road_key="big_eye", display_name="大眼仔")


def _small_road_score(non_tie: List[str]) -> Dict[str, Any]:
    return _derived_road_score(non_tie, offset=2, road_key="small_road", display_name="小路")


def _cockroach_score(non_tie: List[str]) -> Dict[str, Any]:
    return _derived_road_score(non_tie, offset=3, road_key="cockroach", display_name="蟑螂路")


def _road_consensus_score(road_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """大路 + 下三路家族的二來源共識。

    大眼仔、小路、蟑螂路只保留為家族內部明細，不再各自對外投票。
    """
    big_road = road_scores.get("big_road", {}) or {}
    family = road_scores.get("down3_family", {}) or {}
    source_scores = {"big_road": big_road, "down3_family": family}
    source_weights = {"big_road": 0.56, "down3_family": 0.44}

    details: Dict[str, Any] = {}
    valid_picks: List[Tuple[str, str, float]] = []
    for name, score in source_scores.items():
        pick = score.get("pick", "") if name == "down3_family" else _strong_pick_from_score(score, min_gap=0.012)
        confidence = float(score.get("confidence", score.get("strength", 0.0)) or 0.0)
        details[name] = {
            "pick": pick,
            "weight": source_weights[name],
            "label": score.get("label", ""),
            "B": round(float(score.get("B", 0.5)), 4),
            "P": round(float(score.get("P", 0.5)), 4),
            "confidence": round(confidence, 4),
        }
        if pick in {"B", "P"}:
            valid_picks.append((name, pick, source_weights[name]))

    b_raw = sum(float(source_scores[k].get("B", 0.5)) * source_weights[k] for k in source_scores)
    p_raw = sum(float(source_scores[k].get("P", 0.5)) * source_weights[k] for k in source_scores)
    b, p = (0.5, 0.5) if b_raw + p_raw <= 0 else (b_raw / (b_raw + p_raw), p_raw / (b_raw + p_raw))

    if len(valid_picks) == 2 and valid_picks[0][1] == valid_picks[1][1]:
        side = valid_picks[0][1]
        consensus_ratio = _clamp(0.72 + abs(b - p) * 1.8, 0.72, 0.95)
        conflict_ratio = 1.0 - consensus_ratio
        label = f"大路/下三路家族共識:{'莊' if side == 'B' else '閒'}"
    elif len(valid_picks) == 2:
        side = ""
        consensus_ratio = 0.50
        conflict_ratio = 0.50
        label = "大路與下三路家族衝突"
    elif len(valid_picks) == 1:
        side = valid_picks[0][1]
        consensus_ratio = 0.58
        conflict_ratio = 0.42
        label = f"僅{valid_picks[0][0]}有方向"
    else:
        side = ""
        consensus_ratio = 0.50
        conflict_ratio = 0.50
        label = "大路/下三路家族皆無明確方向"

    return {
        "B": round(b, 5),
        "P": round(p, 5),
        "label": label,
        "pick": side,
        "votes": [pick for _, pick, _ in valid_picks],
        "vote_score": {
            "B": round(sum(w for _, pick, w in valid_picks if pick == "B"), 4),
            "P": round(sum(w for _, pick, w in valid_picks if pick == "P"), 4),
        },
        "consensus_ratio": round(consensus_ratio, 4),
        "conflict_ratio": round(conflict_ratio, 4),
        "details": details,
        "strength": round(0.08 + max(0.0, consensus_ratio - 0.5) * 0.28, 4),
        "source_count": len(valid_picks),
        "family_vote_counted_once": True,
    }

def _road_family_scores(non_tie: List[str]) -> Dict[str, Any]:
    """取得大路、三條下路明細、下三路家族，以及二來源共識。"""
    big_road = _big_road_score(non_tie)
    big_eye = _big_eye_score(non_tie)
    small_road = _small_road_score(non_tie)
    cockroach = _cockroach_score(non_tie)
    scores = {
        "big_road": big_road,
        "big_eye": big_eye,
        "small_road": small_road,
        "cockroach": cockroach,
    }
    down3_family = _down3_family_score(non_tie, scores)
    scores["down3_family"] = down3_family
    consensus = _road_consensus_score(scores)
    return {**scores, "consensus": consensus}

def _road_lifecycle_score(non_tie: List[str], road_family: Dict[str, Any], regime_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Road Lifecycle：判斷一段牌路規律目前是「可跟、疲乏、斷點壓力、已斷、混亂」。

    核心概念：
    - 大路負責主趨勢與龍/跳狀態。
    - 下三路紅藍負責規律健康度：紅多偏健康可跟，藍多偏變化/斷點。
    - 四路共識負責確認方向；四路分歧代表路型開始不穩。

    這個函數不直接做觀望或下注金額決策，只輸出 bias 給主融合層使用。
    """
    default = {
        "enabled": False,
        "state": "NEUTRAL",
        "trend_side": "",
        "bias_side": "",
        "follow_score": 0.5,
        "break_score": 0.0,
        "fatigue_score": 0.0,
        "health_score": 0.5,
        "confidence": 0.0,
        "label": "Lifecycle資料不足",
        "components": {},
    }

    if not USE_ROAD_LIFECYCLE or not USE_ROAD_ENGINE or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return default

    regime_info = regime_info or {}
    last_side, streak_n = _streak(non_tie)
    if not last_side:
        return default
    opp = "P" if last_side == "B" else "B"

    consensus = road_family.get("consensus", {})
    big_road = road_family.get("big_road", {})
    big_eye = road_family.get("big_eye", {})
    small_road = road_family.get("small_road", {})
    cockroach = road_family.get("cockroach", {})

    trend_side = consensus.get("pick", "") or last_side
    consensus_ratio = float(consensus.get("consensus_ratio", 0.5))
    conflict_ratio = float(consensus.get("conflict_ratio", 0.5))

    def _get_rate(model: Dict[str, Any], key: str, default_v: float = 0.5) -> float:
        return float(model.get(key, default_v))

    # 下三路紅藍健康度：大眼仔較穩，小路居中，蟑螂路較短線。
    derived_models = [big_eye, small_road, cockroach]
    valid = [m for m in derived_models if int(m.get("stats", {}).get("count", 0)) >= DERIVED_ROAD_MIN_COUNT]
    if valid:
        red_pressure = sum(_get_rate(m, "red_pressure", 0.5) for m in valid) / len(valid)
        blue_pressure = sum(_get_rate(m, "blue_pressure", 0.5) for m in valid) / len(valid)
        derived_count = sum(int(m.get("stats", {}).get("count", 0)) for m in valid)
    else:
        red_pressure = float(big_road.get("red_pressure", 0.5))
        blue_pressure = float(big_road.get("blue_pressure", 0.5))
        derived_count = 0

    big_break = float(big_road.get("break_risk", 0.0))
    big_info = big_road.get("big_road", {})
    last_row = int(big_info.get("last_row", 0))
    current_col_height = int(big_info.get("current_col_height", 0))
    switch_rate = float(regime_info.get("switch_rate", big_info.get("switch_rate_16", 0.5)))

    # 龍/規律疲乏：長龍、到底/黏邊、欄高過高、下三路轉藍、四路分歧都會提高。
    dragon_len_pressure = 0.0
    if streak_n >= max(3, ROAD_ENGINE_BREAK_STREAK - 1):
        dragon_len_pressure = _clamp((streak_n - (ROAD_ENGINE_BREAK_STREAK - 1)) / 6.0, 0.0, 1.0)

    edge_pressure = 0.0
    if last_row >= ROAD_ENGINE_ROWS - 1:
        edge_pressure += 0.45
    if current_col_height >= ROAD_ENGINE_ROWS:
        edge_pressure += 0.20
    edge_pressure = _clamp(edge_pressure, 0.0, 1.0)

    blue_shift = _clamp((blue_pressure - 0.50) * 2.0, 0.0, 1.0)
    red_health = _clamp((red_pressure - 0.50) * 2.0, 0.0, 1.0)
    conflict_pressure = _clamp((conflict_ratio - 0.25) / 0.35, 0.0, 1.0)

    fatigue_score = _clamp(
        dragon_len_pressure * DRAGON_FATIGUE_WEIGHT * 2.1
        + edge_pressure * 0.22
        + blue_shift * 0.26
        + conflict_pressure * 0.20,
        0.0,
        1.0,
    )

    follow_score = _clamp(
        0.42
        + red_health * RED_HEALTH_WEIGHT
        + (consensus_ratio - 0.50) * 0.50
        + max(0.0, 0.62 - conflict_ratio) * 0.12
        - blue_shift * 0.22
        - big_break * 0.22
        - fatigue_score * 0.18,
        0.0,
        1.0,
    )

    break_score = _clamp(
        0.18
        + blue_shift * BLUE_BREAK_WEIGHT
        + big_break * 0.42
        + conflict_pressure * ROAD_CONFLICT_WEIGHT
        + fatigue_score * 0.32
        - red_health * 0.16,
        0.0,
        1.0,
    )

    # 狀態判斷：不是「硬條件下注」，只是讓模型理解規律生命週期。
    state = "FORMING"
    bias_side = trend_side
    if conflict_ratio >= 0.48 and follow_score < 0.58 and break_score < BREAK_SCORE_MIN:
        state = "CHAOS"
        bias_side = ""
    elif break_score >= BREAK_FORCE_SCORE:
        state = "BROKEN"
        bias_side = opp if trend_side == last_side else ("P" if trend_side == "B" else "B")
    elif break_score >= BREAK_SCORE_MIN:
        state = "BREAK_RISK"
        bias_side = opp if trend_side == last_side else ("P" if trend_side == "B" else "B")
    elif follow_score >= FOLLOW_SCORE_MIN and break_score < BREAK_SCORE_MIN:
        state = "FOLLOW"
        bias_side = trend_side
    elif fatigue_score >= 0.48 or break_score >= 0.52:
        state = "FATIGUE"
        bias_side = trend_side

    confidence = _clamp(max(follow_score, break_score) * 0.65 + consensus_ratio * 0.25 + (1.0 - conflict_ratio) * 0.10, 0.0, 1.0)

    side_text = {"B": "莊", "P": "閒", "": "無"}.get(bias_side, bias_side)
    state_text = {
        "FORMING": "規律形成",
        "FOLLOW": "規律健康可跟",
        "FATIGUE": "規律疲乏降權",
        "BREAK_RISK": "斷點壓力偏反",
        "BROKEN": "規律已斷偏反",
        "CHAOS": "四路混亂",
        "NEUTRAL": "中性",
    }.get(state, state)

    return {
        "enabled": True,
        "state": state,
        "trend_side": trend_side,
        "bias_side": bias_side,
        "follow_score": round(follow_score, 4),
        "break_score": round(break_score, 4),
        "fatigue_score": round(fatigue_score, 4),
        "health_score": round(red_pressure, 4),
        "red_pressure": round(red_pressure, 4),
        "blue_pressure": round(blue_pressure, 4),
        "confidence": round(confidence, 4),
        "label": f"{state_text}:{side_text} F{int(follow_score*100)} B{int(break_score*100)}",
        "components": {
            "streak": streak_n,
            "last_side": last_side,
            "consensus_ratio": round(consensus_ratio, 4),
            "conflict_ratio": round(conflict_ratio, 4),
            "big_break": round(big_break, 4),
            "dragon_len_pressure": round(dragon_len_pressure, 4),
            "edge_pressure": round(edge_pressure, 4),
            "blue_shift": round(blue_shift, 4),
            "red_health": round(red_health, 4),
            "derived_count": derived_count,
            "switch_rate": round(switch_rate, 4),
        },
    }


def _apply_lifecycle_weighting(weights: Dict[str, float], lifecycle: Dict[str, Any]) -> Dict[str, float]:
    """依照規律生命週期微調權重：可跟時提高四路；疲乏/斷點時降低追近路與盲目跟龍。"""
    if not USE_ROAD_LIFECYCLE or not lifecycle.get("enabled"):
        return _normalize_weights(weights)

    adjusted = dict(weights)
    state = lifecycle.get("state", "NEUTRAL")
    conf = float(lifecycle.get("confidence", 0.0))
    scale = _clamp(ROAD_LIFECYCLE_WEIGHT / 0.26, 0.20, 2.00)
    road_keys = ["big_road", "big_eye", "small_road", "cockroach"]

    if state == "FOLLOW":
        boost = 1.0 + 0.22 * conf * scale
        for k in road_keys:
            adjusted[k] = adjusted.get(k, 0.0) * boost
        adjusted["balance"] = adjusted.get("balance", 0.0) * 0.80
    elif state == "FATIGUE":
        for k in ["streak", "recent"]:
            adjusted[k] = adjusted.get(k, 0.0) * 0.72
        adjusted["big_road"] = adjusted.get("big_road", 0.0) * 0.86
        for k in ["big_eye", "small_road", "cockroach"]:
            adjusted[k] = adjusted.get(k, 0.0) * (1.0 + 0.10 * conf * scale)
    elif state in {"BREAK_RISK", "BROKEN"}:
        # 斷點壓力高時，下三路比單純大路/連莊更重要。
        adjusted["streak"] = adjusted.get("streak", 0.0) * 0.45
        adjusted["recent"] = adjusted.get("recent", 0.0) * 0.70
        adjusted["big_road"] = adjusted.get("big_road", 0.0) * 0.78
        for k in ["big_eye", "small_road", "cockroach"]:
            adjusted[k] = adjusted.get(k, 0.0) * (1.0 + 0.18 * conf * scale)
    elif state == "CHAOS":
        # 混亂時不要讓任何單一路型過度主導，回到較均衡的融合。
        for k in road_keys + ["ngram", "markov", "road", "recent", "streak", "balance"]:
            adjusted[k] = adjusted.get(k, 0.0) * 0.95

    return _normalize_weights(adjusted)


def _apply_lifecycle_bias(b_side: float, lifecycle: Dict[str, Any]) -> float:
    """將生命周期狀態轉成輕量方向偏移：跟、降權、斷點偏反。"""
    if not USE_ROAD_LIFECYCLE or not lifecycle.get("enabled"):
        return b_side

    state = lifecycle.get("state", "NEUTRAL")
    bias_side = lifecycle.get("bias_side", "")
    trend_side = lifecycle.get("trend_side", "")
    follow_score = float(lifecycle.get("follow_score", 0.5))
    break_score = float(lifecycle.get("break_score", 0.0))
    fatigue_score = float(lifecycle.get("fatigue_score", 0.0))
    scale = _clamp(ROAD_LIFECYCLE_WEIGHT / 0.26, 0.20, 2.00)

    def _signed(side: str) -> int:
        return 1 if side == "B" else -1 if side == "P" else 0

    if state == "FOLLOW" and bias_side:
        b_side += _signed(bias_side) * FOLLOW_BOOST * follow_score * scale
    elif state == "FATIGUE" and trend_side:
        # 疲乏不是直接反打，而是先把原本跟路方向降權，避免傻傻續跟。
        b_side -= _signed(trend_side) * FATIGUE_SHRINK * max(0.45, fatigue_score) * scale
    elif state == "BREAK_RISK" and bias_side:
        b_side += _signed(bias_side) * BREAK_REVERSE_BIAS * break_score * scale
    elif state == "BROKEN" and bias_side:
        b_side += _signed(bias_side) * BREAK_REVERSE_BIAS * min(1.0, break_score * 1.18) * scale
    elif state == "CHAOS":
        b_side = 0.5 + (b_side - 0.5) * 0.82

    return _clamp(b_side, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)


def _bucket_value(value: float, cuts: List[float], labels: List[str]) -> str:
    """把連續數值轉成穩定桶，避免記憶匹配太死。"""
    try:
        v = float(value)
    except Exception:
        v = 0.0
    for cut, label in zip(cuts, labels):
        if v < cut:
            return label
    return labels[-1] if labels else "X"


def _road_state_fingerprint(non_tie: List[str], road_family: Dict[str, Any], lifecycle: Dict[str, Any], regime_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Road State Fingerprint：把目前路型壓成「可比較」的狀態指紋。
    重點是用相對路型，不死綁 B/P，讓程式學：同類狀態到底是跟準還是斷準。
    """
    last_side, streak_n = _streak(non_tie)
    opp = "P" if last_side == "B" else "B" if last_side == "P" else ""
    consensus = road_family.get("consensus", {}) if road_family else {}
    trend_side = lifecycle.get("trend_side", "") or consensus.get("pick", "") or last_side

    consensus_ratio = float(consensus.get("consensus_ratio", 0.5))
    conflict_ratio = float(consensus.get("conflict_ratio", 0.5))
    red_pressure = float(lifecycle.get("red_pressure", lifecycle.get("health_score", 0.5))) if lifecycle else 0.5
    blue_pressure = float(lifecycle.get("blue_pressure", 0.5)) if lifecycle else 0.5
    break_score = float(lifecycle.get("break_score", 0.0)) if lifecycle else 0.0
    follow_score = float(lifecycle.get("follow_score", 0.5)) if lifecycle else 0.5
    fatigue_score = float(lifecycle.get("fatigue_score", 0.0)) if lifecycle else 0.0
    regime = str(regime_info.get("regime", "mixed"))
    if regime.startswith("periodic"):
        regime = "periodic"

    streak_bucket = _bucket_value(streak_n, [2, 3, 4, 5, 7], ["S1", "S2", "S3", "S4", "S5_6", "S7P"])
    consensus_bucket = _bucket_value(consensus_ratio, [0.58, 0.66, 0.74, 0.84], ["C50", "C60", "C70", "C80", "C90"])
    conflict_bucket = _bucket_value(conflict_ratio, [0.28, 0.40, 0.50], ["KLOW", "KMID", "KHIGH", "KMAX"])
    red_bucket = _bucket_value(red_pressure, [0.46, 0.56, 0.66], ["RLOW", "RMID", "RHIGH", "RMAX"])
    blue_bucket = _bucket_value(blue_pressure, [0.46, 0.56, 0.66], ["BLOW", "BMID", "BHIGH", "BMAX"])
    break_bucket = _bucket_value(break_score, [0.40, 0.58, 0.70], ["BRLOW", "BRMID", "BRHIGH", "BRMAX"])
    follow_bucket = _bucket_value(follow_score, [0.52, 0.62, 0.72], ["FLOW", "FMID", "FHIGH", "FMAX"])
    fatigue_bucket = _bucket_value(fatigue_score, [0.35, 0.52, 0.68], ["TLOW", "TMID", "THIGH", "TMAX"])
    lifecycle_state = str(lifecycle.get("state", "NEUTRAL")) if lifecycle else "NEUTRAL"

    # 四路投票型態，用相對於 trend_side 的 F=跟趨勢、R=反趨勢、N=中性。
    vote_pattern = []
    details = consensus.get("details", {}) if consensus else {}
    for key in ["big_road", "big_eye", "small_road", "cockroach"]:
        pick = details.get(key, {}).get("pick", "")
        if not pick or not trend_side:
            vote_pattern.append("N")
        elif pick == trend_side:
            vote_pattern.append("F")
        else:
            vote_pattern.append("R")
    vote_pattern = "".join(vote_pattern)

    # trend_relation 讓不同 B/P 可以共用記憶：趨勢是跟最後一口，還是反最後一口。
    if trend_side and last_side and trend_side == last_side:
        trend_relation = "TREND_LAST"
    elif trend_side and opp and trend_side == opp:
        trend_relation = "TREND_OPP"
    else:
        trend_relation = "TREND_UNKNOWN"

    components = {
        "regime": regime,
        "streak_bucket": streak_bucket,
        "consensus_bucket": consensus_bucket,
        "conflict_bucket": conflict_bucket,
        "red_bucket": red_bucket,
        "blue_bucket": blue_bucket,
        "break_bucket": break_bucket,
        "follow_bucket": follow_bucket,
        "fatigue_bucket": fatigue_bucket,
        "lifecycle_state": lifecycle_state,
        "vote_pattern": vote_pattern,
        "trend_relation": trend_relation,
    }
    key = "|".join(f"{k}:{v}" for k, v in components.items())
    return {
        "key": key,
        "components": components,
        "trend_side": trend_side,
        "last_side": last_side,
        "opp_side": opp,
    }


def _memory_match_score(current_fp: Dict[str, Any], past_fp: Dict[str, Any]) -> float:
    """相似狀態分數：不要求完全一樣，避免記憶模型太死板。"""
    c = current_fp.get("components", {})
    p = past_fp.get("components", {})
    if not c or not p:
        return 0.0
    score = 0.0
    weights = {
        "regime": 1.25,
        "streak_bucket": 1.00,
        "consensus_bucket": 1.00,
        "conflict_bucket": 0.90,
        "red_bucket": 1.05,
        "blue_bucket": 1.10,
        "break_bucket": 1.10,
        "follow_bucket": 0.90,
        "fatigue_bucket": 0.80,
        "lifecycle_state": 1.00,
        "vote_pattern": 1.25,
        "trend_relation": 0.80,
    }
    for name, w in weights.items():
        if c.get(name) == p.get(name):
            score += w
    if current_fp.get("key") == past_fp.get("key"):
        score += ROAD_MEMORY_EXACT_BONUS
    return score


def _adaptive_road_memory_score(non_tie: List[str], road_family: Dict[str, Any], lifecycle: Dict[str, Any], regime_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adaptive Road Memory：本靴內相似牌路回測記憶。
    它不再只問「紅多還是藍多」，而是回看本靴過去相似狀態：
    - 當時如果跟趨勢，下一手有沒有中？
    - 當時如果斷趨勢，下一手有沒有中？
    樣本夠且偏向明顯時，才用柔性 bias 修正主方向。
    """
    default = {
        "enabled": False,
        "state": "MEMORY_COLD",
        "label": "記憶樣本不足",
        "bias_side": "",
        "trend_side": "",
        "follow_rate": 0.5,
        "break_rate": 0.5,
        "confidence": 0.0,
        "sample": 0,
        "weighted_sample": 0.0,
        "follow_weight": 0.0,
        "break_weight": 0.0,
        "current_fingerprint": {},
        "matched_examples": [],
    }

    if not USE_ADAPTIVE_ROAD_MEMORY or not USE_ROAD_ENGINE or len(non_tie) < max(ROAD_ENGINE_MIN_HISTORY + 3, 10):
        return default

    current_fp = _road_state_fingerprint(non_tie, road_family, lifecycle, regime_info)
    trend_side = current_fp.get("trend_side", "")
    opp_side = current_fp.get("opp_side", "")
    if trend_side not in {"B", "P"}:
        return {**default, "current_fingerprint": current_fp}
    if not opp_side:
        opp_side = "P" if trend_side == "B" else "B"

    start = max(ROAD_ENGINE_MIN_HISTORY, len(non_tie) - ROAD_MEMORY_LOOKBACK)
    end = len(non_tie)  # i 的 truth 是 non_tie[i]，所以 prefix 到 i 前一手
    follow_w = 0.0
    break_w = 0.0
    raw_matches = 0
    examples = []

    max_score_seen = 0.0
    for i in range(start, end):
        prefix = non_tie[:i]
        truth = non_tie[i]
        if len(prefix) < ROAD_ENGINE_MIN_HISTORY or truth not in {"B", "P"}:
            continue
        try:
            pfamily = _road_family_scores(prefix)
            pregime = _detect_regime(prefix)
            plifecycle = _road_lifecycle_score(prefix, pfamily, pregime)
            pfp = _road_state_fingerprint(prefix, pfamily, plifecycle, pregime)
            ptrend = pfp.get("trend_side", "")
            if ptrend not in {"B", "P"}:
                continue
            match_score = _memory_match_score(current_fp, pfp)
            max_score_seen = max(max_score_seen, match_score)
            if match_score < ROAD_MEMORY_MIN_MATCH_SCORE:
                continue

            # 近期相似狀態稍微加權，但不讓最近幾手完全主宰。
            recency = (i - start + 1) / max(1, end - start)
            weight = (match_score / max(ROAD_MEMORY_MIN_MATCH_SCORE, 0.0001)) * (1.0 + ROAD_MEMORY_RECENCY_BONUS * recency)
            raw_matches += 1
            if truth == ptrend:
                follow_w += weight
                outcome = "follow"
            else:
                break_w += weight
                outcome = "break"
            if len(examples) < 6:
                examples.append({
                    "round": i + 1,
                    "match_score": round(match_score, 3),
                    "trend": ptrend,
                    "truth": truth,
                    "outcome": outcome,
                    "state": plifecycle.get("state", ""),
                    "key": pfp.get("key", ""),
                })
        except Exception:
            continue

    weighted_sample = follow_w + break_w
    alpha = max(0.0, ROAD_MEMORY_ALPHA)
    denom = weighted_sample + 2 * alpha
    if denom <= 0:
        follow_rate = 0.5
    else:
        follow_rate = (follow_w + alpha) / denom
    break_rate = 1.0 - follow_rate
    advantage = abs(follow_rate - break_rate)

    sample_strength = _clamp(weighted_sample / max(ROAD_MEMORY_FULL_SAMPLE, 1), 0.0, 1.0)
    confidence = _clamp(sample_strength * (0.30 + advantage * 1.65), 0.0, 1.0)

    state = "MEMORY_COLD"
    bias_side = ""
    if raw_matches >= ROAD_MEMORY_MIN_SAMPLE and weighted_sample >= ROAD_MEMORY_MIN_SAMPLE:
        if follow_rate >= ROAD_MEMORY_FOLLOW_THRESHOLD and advantage >= ROAD_MEMORY_MIN_ADVANTAGE:
            state = "MEMORY_FOLLOW"
            bias_side = trend_side
        elif break_rate >= ROAD_MEMORY_BREAK_THRESHOLD and advantage >= ROAD_MEMORY_MIN_ADVANTAGE:
            state = "MEMORY_BREAK"
            bias_side = opp_side
        else:
            state = "MEMORY_NEUTRAL"
            bias_side = ""

    state_text = {
        "MEMORY_COLD": "記憶樣本不足",
        "MEMORY_FOLLOW": "相似路型過去偏跟",
        "MEMORY_BREAK": "相似路型過去偏斷",
        "MEMORY_NEUTRAL": "相似路型跟斷接近",
    }.get(state, state)
    side_text = {"B": "莊", "P": "閒", "": "無"}.get(bias_side, bias_side)

    return {
        "enabled": True,
        "state": state,
        "label": f"{state_text}:{side_text} 跟{int(follow_rate*100)} 斷{int(break_rate*100)} 樣本{raw_matches}",
        "bias_side": bias_side,
        "trend_side": trend_side,
        "opp_side": opp_side,
        "follow_rate": round(follow_rate, 4),
        "break_rate": round(break_rate, 4),
        "advantage": round(advantage, 4),
        "confidence": round(confidence, 4),
        "sample": raw_matches,
        "weighted_sample": round(weighted_sample, 4),
        "follow_weight": round(follow_w, 4),
        "break_weight": round(break_w, 4),
        "max_score_seen": round(max_score_seen, 4),
        "current_fingerprint": current_fp,
        "matched_examples": examples,
    }


def _apply_road_memory_weighting(weights: Dict[str, float], memory: Dict[str, Any]) -> Dict[str, float]:
    """依相似牌路記憶微調權重：偏跟時保留四路主導；偏斷時提高下三路/回測，降低盲目跟龍。"""
    if not USE_ADAPTIVE_ROAD_MEMORY or not memory.get("enabled"):
        return _normalize_weights(weights)
    state = memory.get("state", "MEMORY_COLD")
    conf = float(memory.get("confidence", 0.0))
    if state not in {"MEMORY_FOLLOW", "MEMORY_BREAK"} or conf <= 0:
        return _normalize_weights(weights)

    adjusted = dict(weights)
    scale = _clamp(ROAD_MEMORY_WEIGHT / 0.22, 0.10, 2.20)
    if state == "MEMORY_FOLLOW":
        for k in ["big_road", "big_eye", "small_road", "cockroach"]:
            adjusted[k] = adjusted.get(k, 0.0) * (1.0 + 0.16 * conf * scale)
        adjusted["recent"] = adjusted.get("recent", 0.0) * 0.92
    elif state == "MEMORY_BREAK":
        adjusted["big_road"] = adjusted.get("big_road", 0.0) * (1.0 - 0.10 * conf * scale)
        adjusted["streak"] = adjusted.get("streak", 0.0) * (1.0 - 0.22 * conf * scale)
        adjusted["recent"] = adjusted.get("recent", 0.0) * (1.0 - 0.12 * conf * scale)
        for k in ["big_eye", "small_road", "cockroach", "ngram"]:
            adjusted[k] = adjusted.get(k, 0.0) * (1.0 + 0.14 * conf * scale)
    return _normalize_weights(adjusted)


def _apply_road_memory_bias(b_side: float, memory: Dict[str, Any]) -> float:
    """將 Adaptive Road Memory 轉成柔性偏移，不用硬切換方向。"""
    if not USE_ADAPTIVE_ROAD_MEMORY or not memory.get("enabled"):
        return b_side
    state = memory.get("state", "MEMORY_COLD")
    bias_side = memory.get("bias_side", "")
    if state not in {"MEMORY_FOLLOW", "MEMORY_BREAK"} or bias_side not in {"B", "P"}:
        return b_side
    conf = float(memory.get("confidence", 0.0))
    advantage = float(memory.get("advantage", 0.0))
    scale = _clamp(ROAD_MEMORY_WEIGHT / 0.22, 0.10, 2.20)
    strength = ROAD_MEMORY_MAX_BIAS * conf * _clamp(advantage * 2.2, 0.25, 1.0) * scale
    signed = 1 if bias_side == "B" else -1
    return _clamp(b_side + signed * strength, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)



def _window_rhythm_features(non_tie: List[str], window: int) -> Dict[str, Any]:
    """單一週期節奏特徵：用來看短 / 中 / 長牌段，不只看當前一口。"""
    tail = non_tie[-window:] if window and len(non_tie) > window else list(non_tie)
    if not tail:
        return {
            "window": window,
            "count": 0,
            "side": "",
            "last_side": "",
            "streak": 0,
            "switch_rate": 0.5,
            "b_rate": 0.5,
            "mode": "empty",
            "strength": 0.0,
        }

    last_side, streak_n = _streak(tail)
    opp = "P" if last_side == "B" else "B" if last_side == "P" else ""
    switches = sum(1 for a, b in zip(tail, tail[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(tail) - 1), 0.5)
    b_rate = tail.count("B") / len(tail)

    # 這裡不是直接下注方向，而是該週期目前的節奏傾向。
    if len(tail) < 4:
        side = last_side
        mode = "cold"
        strength = 0.10
    elif switch_rate >= 0.72:
        side = opp
        mode = "jump"
        strength = _clamp(0.45 + (switch_rate - 0.72) * 0.85, 0.20, 0.86)
    elif streak_n >= 3:
        side = last_side
        mode = "streak"
        strength = _clamp(0.42 + (streak_n - 3) * 0.08, 0.20, 0.88)
    elif abs(b_rate - 0.5) >= 0.16:
        side = "B" if b_rate > 0.5 else "P"
        mode = "side_bias"
        strength = _clamp(abs(b_rate - 0.5) * 2.2, 0.20, 0.72)
    else:
        # 沒有明顯節奏時，避免過度反應當前一口，給中性。
        side = ""
        mode = "neutral"
        strength = 0.05

    return {
        "window": window,
        "count": len(tail),
        "side": side,
        "last_side": last_side,
        "streak": streak_n,
        "switch_rate": round(switch_rate, 4),
        "b_rate": round(b_rate, 4),
        "mode": mode,
        "strength": round(strength, 4),
    }


def _derived_pressure_by_window(non_tie: List[str], window: int) -> Dict[str, Any]:
    """用不同 lookback 看下三路紅藍壓力，判斷是短暫波動還是節奏轉折。"""
    if not USE_ROAD_ENGINE or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return {"red": 0.5, "blue": 0.5, "count": 0, "tails": {}}
    try:
        layout = _build_big_road(non_tie)
        stats = {
            "big_eye": _color_stats(_derived_series(layout, 1), lookback=window),
            "small_road": _color_stats(_derived_series(layout, 2), lookback=window),
            "cockroach": _color_stats(_derived_series(layout, 3), lookback=window),
        }
        valid = [v for v in stats.values() if int(v.get("count", 0)) > 0]
        if not valid:
            return {"red": 0.5, "blue": 0.5, "count": 0, "tails": stats}
        red = sum(float(v.get("red_rate", 0.5)) for v in valid) / len(valid)
        blue = sum(float(v.get("blue_rate", 0.5)) for v in valid) / len(valid)
        count = sum(int(v.get("count", 0)) for v in valid)
        return {"red": round(red, 4), "blue": round(blue, 4), "count": count, "tails": stats}
    except Exception:
        return {"red": 0.5, "blue": 0.5, "count": 0, "tails": {}}



def _long_anchor_score(non_tie: List[str], road_family: Dict[str, Any], lifecycle: Dict[str, Any], regime_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Long Anchor Guard：長週期錨定層。

    目的：
    - 不取代原本四路 / Lifecycle / Memory / Rhythm，只提供長週期參考錨點。
    - 當短線 Memory / Rhythm 想快速反邊時，用長週期錨定避免被當局一兩口帶走。
    - 真轉折仍可放行，但要有 Strict Turn Confirm / Lifecycle break 明確確認。
    """
    default = {
        "enabled": False,
        "state": "ANCHOR_COLD",
        "label": "長週期錨定資料不足",
        "anchor_side": "",
        "confidence": 0.0,
        "anchor_b": 0.5,
        "long_window": {},
        "votes": {},
    }

    if not USE_LONG_ANCHOR_GUARD or len(non_tie) < LONG_ANCHOR_MIN_HISTORY:
        return default

    window = max(12, min(LONG_ANCHOR_WINDOW, len(non_tie)))
    long_f = _window_rhythm_features(non_tie, window)
    consensus = road_family.get("consensus", {}) if road_family else {}
    consensus_side = consensus.get("pick", "")
    consensus_ratio = float(consensus.get("consensus_ratio", 0.5))
    lifecycle_state = str(lifecycle.get("state", "")).upper() if lifecycle else ""
    lifecycle_trend = lifecycle.get("trend_side", "") if lifecycle else ""
    lifecycle_bias = lifecycle.get("bias_side", "") if lifecycle else ""
    lifecycle_follow = float(lifecycle.get("follow_score", 0.5)) if lifecycle else 0.5
    lifecycle_break = float(lifecycle.get("break_score", 0.0)) if lifecycle else 0.0

    long_side = long_f.get("side", "")
    long_strength = float(long_f.get("strength", 0.0))
    tail = non_tie[-window:]
    b_rate = tail.count("B") / max(1, len(tail))
    p_rate = 1.0 - b_rate
    balance_side = "B" if b_rate > p_rate else "P" if p_rate > b_rate else ""
    balance_strength = abs(b_rate - p_rate)

    # 用多來源投票取得長週期錨點；避免只靠單一長窗比例。
    votes = {"B": 0.0, "P": 0.0}
    vote_details = {}

    if long_side in {"B", "P"}:
        w = 0.38 + min(0.18, long_strength * 0.35)
        votes[long_side] += w
        vote_details["long_window"] = {"side": long_side, "weight": round(w, 4), "strength": round(long_strength, 4)}

    if consensus_side in {"B", "P"} and consensus_ratio >= LONG_ANCHOR_CONSENSUS_MIN:
        w = 0.30 + min(0.18, (consensus_ratio - 0.5) * 0.55)
        votes[consensus_side] += w
        vote_details["consensus"] = {"side": consensus_side, "weight": round(w, 4), "ratio": round(consensus_ratio, 4)}

    if lifecycle_trend in {"B", "P"} and lifecycle_follow >= lifecycle_break:
        w = 0.18 + min(0.12, max(0.0, lifecycle_follow - 0.50) * 0.35)
        votes[lifecycle_trend] += w
        vote_details["lifecycle_trend"] = {"side": lifecycle_trend, "weight": round(w, 4), "follow": round(lifecycle_follow, 4)}

    # 若 Lifecycle 已經明確斷點，錨點不硬跟舊趨勢，改把反向也納入投票。
    if lifecycle_bias in {"B", "P"} and lifecycle_state in {"BREAK_RISK", "BROKEN"} and lifecycle_break >= LONG_ANCHOR_BREAK_BYPASS_SCORE:
        w = 0.28 + min(0.16, max(0.0, lifecycle_break - 0.55) * 0.45)
        votes[lifecycle_bias] += w
        vote_details["lifecycle_break"] = {"side": lifecycle_bias, "weight": round(w, 4), "break": round(lifecycle_break, 4)}

    if balance_side in {"B", "P"} and balance_strength >= 0.10:
        w = 0.10 + min(0.08, balance_strength * 0.30)
        votes[balance_side] += w
        vote_details["balance"] = {"side": balance_side, "weight": round(w, 4), "b_rate": round(b_rate, 4)}

    if votes["B"] == votes["P"]:
        anchor_side = long_side if long_side in {"B", "P"} else consensus_side if consensus_side in {"B", "P"} else ""
    else:
        anchor_side = "B" if votes["B"] > votes["P"] else "P"

    total_vote = max(0.0001, votes["B"] + votes["P"])
    vote_ratio = max(votes["B"], votes["P"]) / total_vote if anchor_side else 0.5
    confidence = _clamp(
        (vote_ratio - 0.5) * 1.35
        + min(0.30, long_strength * 0.55)
        + max(0.0, consensus_ratio - 0.5) * 0.30,
        0.0,
        1.0,
    )

    if anchor_side not in {"B", "P"} or confidence < LONG_ANCHOR_CONF_MIN:
        return {
            **default,
            "enabled": True,
            "state": "ANCHOR_WEAK",
            "label": f"長週期錨定不足 C{int(confidence*100)}",
            "confidence": round(confidence, 4),
            "anchor_side": anchor_side if anchor_side in {"B", "P"} else "",
            "long_window": long_f,
            "votes": {k: round(v, 4) for k, v in votes.items()},
            "vote_details": vote_details,
        }

    signed = 1 if anchor_side == "B" else -1
    anchor_pull = min(LONG_ANCHOR_MAX_PULL, 0.018 + confidence * LONG_ANCHOR_MAX_PULL)
    anchor_b = _clamp(0.5 + signed * anchor_pull, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)

    side_text = "莊" if anchor_side == "B" else "閒"
    return {
        "enabled": True,
        "state": "ANCHOR_ACTIVE",
        "label": f"長週期錨定:{side_text} C{int(confidence*100)}",
        "anchor_side": anchor_side,
        "confidence": round(confidence, 4),
        "anchor_b": round(anchor_b, 5),
        "anchor_pull": round(anchor_pull, 5),
        "long_window": long_f,
        "votes": {k: round(v, 4) for k, v in votes.items()},
        "vote_details": vote_details,
    }


def _apply_long_anchor_guard(b_side: float, anchor: Dict[str, Any], lifecycle: Dict[str, Any], memory: Dict[str, Any], rhythm: Dict[str, Any]) -> float:
    """
    將長週期錨點套用到最終 B/P 側機率。

    注意：這不是硬鎖方向，而是「短線偏移護欄」：
    - 若短線與長錨同向，輕微穩定。
    - 若短線逆長錨，但沒有嚴格轉折確認，限制逆向幅度。
    - 若 Strict Turn Confirm 票數足夠或 Lifecycle 明確 BREAK，允許放行。
    """
    if not USE_LONG_ANCHOR_GUARD or not anchor.get("enabled") or anchor.get("state") != "ANCHOR_ACTIVE":
        return b_side

    anchor_side = anchor.get("anchor_side", "")
    if anchor_side not in {"B", "P"}:
        return b_side

    conf = float(anchor.get("confidence", 0.0))
    if conf < LONG_ANCHOR_CONF_MIN:
        return b_side

    rhythm_state = str(rhythm.get("state", "")).upper() if rhythm else ""
    turn_votes = int(rhythm.get("turn_confirmation_votes", 0) or 0) if rhythm else 0
    lifecycle_state = str(lifecycle.get("state", "")).upper() if lifecycle else ""
    lifecycle_break = float(lifecycle.get("break_score", 0.0)) if lifecycle else 0.0
    memory_state = str(memory.get("state", "")).upper() if memory else ""

    confirmed_turn = bool(
        rhythm_state == "RHYTHM_TURN_CONFIRM"
        and turn_votes >= LONG_ANCHOR_TURN_BYPASS_VOTES
    )
    confirmed_break = bool(
        lifecycle_state in {"BREAK_RISK", "BROKEN"}
        and lifecycle_break >= LONG_ANCHOR_BREAK_BYPASS_SCORE
    )

    current_side = "B" if b_side >= 0.5 else "P"
    anchor_b = float(anchor.get("anchor_b", 0.5))
    weight = _clamp(LONG_ANCHOR_WEIGHT * conf, 0.0, 0.45)

    if current_side == anchor_side:
        # 同向時只做微量穩定，避免過度放大。
        return _clamp(b_side * (1.0 - weight * 0.35) + anchor_b * (weight * 0.35), SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)

    # 逆向但已被多模組確認，放行，不硬拉回。
    if confirmed_turn or confirmed_break:
        return b_side

    # Memory 單獨反向時最容易吃當局；逆長錨且只有 MEMORY_BREAK 時多拉回一點。
    if memory_state == "MEMORY_BREAK":
        weight = min(0.52, weight * 1.18)

    guarded = b_side * (1.0 - weight) + anchor_b * weight

    # 逆向邊際限制：避免短線把方向拉離 0.5 太遠。
    max_opp = max(0.0, LONG_ANCHOR_MAX_OPPOSITE_EDGE)
    if anchor_side == "B" and guarded < 0.5 - max_opp:
        guarded = 0.5 - max_opp
    elif anchor_side == "P" and guarded > 0.5 + max_opp:
        guarded = 0.5 + max_opp

    return _clamp(guarded, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)


def _road_rhythm_score(non_tie: List[str], road_family: Dict[str, Any], lifecycle: Dict[str, Any], regime_info: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Road Rhythm Controller：多週期牌路節奏控制器。

    跟 Lifecycle / Memory 不同，Rhythm 不是問「現在紅還藍」，而是問：
    - 短週期的變化，是否已被中週期、長週期確認？
    - 目前像是真轉折，還是只是一口假斷 / 短暫波動？
    - 要不要保留方向慣性，避免被當局帶走？
    """
    default = {
        "enabled": False,
        "state": "RHYTHM_COLD",
        "label": "節奏資料不足",
        "bias_side": "",
        "dominant_side": "",
        "confidence": 0.0,
        "false_break_score": 0.0,
        "turn_score": 0.0,
        "inertia_score": 0.0,
        "blue_rise": 0.0,
        "red_stability": 0.0,
        "windows": {},
        "derived_windows": {},
    }
    if not USE_ROAD_RHYTHM or not USE_ROAD_ENGINE or len(non_tie) < ROAD_RHYTHM_MIN_HISTORY:
        return default

    short_w = max(4, ROAD_RHYTHM_SHORT_WINDOW)
    mid_w = max(short_w + 2, ROAD_RHYTHM_MID_WINDOW)
    long_w = max(mid_w + 2, ROAD_RHYTHM_LONG_WINDOW)

    short_f = _window_rhythm_features(non_tie, short_w)
    mid_f = _window_rhythm_features(non_tie, mid_w)
    long_f = _window_rhythm_features(non_tie, long_w)

    short_d = _derived_pressure_by_window(non_tie, short_w)
    mid_d = _derived_pressure_by_window(non_tie, mid_w)
    long_d = _derived_pressure_by_window(non_tie, long_w)

    consensus = road_family.get("consensus", {}) if road_family else {}
    consensus_side = consensus.get("pick", "")
    lifecycle_side = lifecycle.get("bias_side", "") if lifecycle and lifecycle.get("enabled") else ""
    memory_side = memory.get("bias_side", "") if memory and memory.get("enabled") else ""

    # dominant_side 優先採中長週期一致，其次四路共識，不讓短週期獨自支配。
    mid_side = mid_f.get("side", "")
    long_side = long_f.get("side", "")
    short_side = short_f.get("side", "")
    if mid_side and long_side and mid_side == long_side:
        dominant_side = mid_side
        dominant_source = "mid_long"
    elif long_side:
        dominant_side = long_side
        dominant_source = "long"
    elif consensus_side:
        dominant_side = consensus_side
        dominant_source = "consensus"
    elif lifecycle_side:
        dominant_side = lifecycle_side
        dominant_source = "lifecycle"
    else:
        dominant_side = mid_side or short_side or ""
        dominant_source = "fallback"

    if dominant_side not in {"B", "P"}:
        return {**default, "enabled": True, "state": "RHYTHM_NEUTRAL", "windows": {"short": short_f, "mid": mid_f, "long": long_f}}
    opp_side = "P" if dominant_side == "B" else "B"

    mid_long_agree = 1.0 if mid_side and long_side and mid_side == long_side else 0.0
    short_against = 1.0 if short_side and short_side != dominant_side else 0.0
    short_with = 1.0 if short_side and short_side == dominant_side else 0.0
    mid_against_long = 1.0 if mid_side and long_side and mid_side != long_side else 0.0

    short_strength = float(short_f.get("strength", 0.0))
    mid_strength = float(mid_f.get("strength", 0.0))
    long_strength = float(long_f.get("strength", 0.0))
    consensus_ratio = float(consensus.get("consensus_ratio", 0.5))
    conflict_ratio = float(consensus.get("conflict_ratio", 0.5))
    lifecycle_break = float(lifecycle.get("break_score", 0.0)) if lifecycle else 0.0
    lifecycle_follow = float(lifecycle.get("follow_score", 0.5)) if lifecycle else 0.5
    memory_conf = float(memory.get("confidence", 0.0)) if memory else 0.0

    blue_rise = _clamp(float(short_d.get("blue", 0.5)) - float(long_d.get("blue", 0.5)), -1.0, 1.0)
    red_stability = _clamp((float(mid_d.get("red", 0.5)) + float(long_d.get("red", 0.5))) / 2.0, 0.0, 1.0)
    blue_rising_pressure = _clamp((blue_rise - ROAD_RHYTHM_BLUE_RISE_MIN) / 0.22, 0.0, 1.0)

    inertia_score = _clamp(
        mid_long_agree * 0.38
        + long_strength * 0.24
        + max(0.0, consensus_ratio - 0.5) * 0.44
        + max(0.0, red_stability - 0.5) * 0.24
        - conflict_ratio * 0.18,
        0.0,
        1.0,
    )

    false_break_score = _clamp(
        short_against * 0.32
        + mid_long_agree * 0.24
        + inertia_score * 0.26
        + max(0.0, lifecycle_follow - 0.50) * 0.24
        - blue_rising_pressure * 0.28
        - max(0.0, lifecycle_break - 0.50) * 0.20,
        0.0,
        1.0,
    )

    turn_score = _clamp(
        short_against * 0.22
        + mid_against_long * 0.24
        + blue_rising_pressure * 0.30
        + max(0.0, lifecycle_break - 0.48) * 0.46
        + (1.0 if memory_side and memory_side == opp_side else 0.0) * memory_conf * 0.16
        - inertia_score * 0.16,
        0.0,
        1.0,
    )

    # 轉折候選方向：短週期明確反向時採短週期，否則採 dominant 的反邊。
    turn_bias_side = short_side if short_side in {"B", "P"} and short_side != dominant_side else opp_side

    # Strict Turn Confirm：多模組二次確認。
    # 舊版只看 Rhythm 分數，容易把短暫假斷當成轉折；新版要求 Lifecycle / Memory / 四路 / 藍路 / 視窗 至少 N 票確認。
    lifecycle_state = str(lifecycle.get("state", "")).upper() if lifecycle else ""
    memory_state = str(memory.get("state", "")).upper() if memory else ""
    consensus_valid = consensus_side in {"B", "P"} and consensus_ratio >= TURN_CONFIRM_CONSENSUS_MIN

    turn_confirmed_by_lifecycle = bool(
        lifecycle_side == turn_bias_side
        and (
            lifecycle_state in {"BREAK_RISK", "BROKEN"}
            or lifecycle_break >= TURN_CONFIRM_LIFECYCLE_BREAK_MIN
        )
    )
    turn_confirmed_by_memory = bool(
        memory_side == turn_bias_side
        and memory_state == "MEMORY_BREAK"
        and memory_conf >= TURN_CONFIRM_MEMORY_CONF_MIN
    )
    turn_confirmed_by_consensus = bool(
        consensus_valid
        and consensus_side == turn_bias_side
        and consensus_side != dominant_side
    )
    turn_confirmed_by_blue = bool(
        blue_rising_pressure >= TURN_CONFIRM_BLUE_PRESSURE_MIN
        and blue_rise >= ROAD_RHYTHM_BLUE_RISE_MIN
        and float(short_d.get("blue", 0.5)) >= float(mid_d.get("blue", 0.5))
    )
    turn_confirmed_by_window = bool(
        short_side == turn_bias_side
        and short_side != dominant_side
        and (mid_against_long > 0 or mid_side == turn_bias_side)
        and short_strength >= 0.18
    )

    turn_confirmations = {
        "lifecycle": turn_confirmed_by_lifecycle,
        "memory": turn_confirmed_by_memory,
        "consensus": turn_confirmed_by_consensus,
        "blue_pressure": turn_confirmed_by_blue,
        "window": turn_confirmed_by_window,
    }
    turn_confirmation_votes = sum(1 for v in turn_confirmations.values() if v)

    if USE_STRICT_TURN_CONFIRM:
        turn_gap_required = max(0.0, TURN_CONFIRM_GAP)
        turn_votes_required = max(0, TURN_CONFIRM_MIN_VOTES)
        turn_base_ready = (
            turn_score >= ROAD_RHYTHM_TURN_CONFIRM
            and turn_score >= false_break_score + turn_gap_required
        )
        turn_confirmed = turn_base_ready and turn_confirmation_votes >= turn_votes_required
    else:
        # 相容舊版邏輯：只看 Rhythm 分數與假斷差距。
        turn_gap_required = 0.03
        turn_votes_required = 0
        turn_base_ready = (
            turn_score >= ROAD_RHYTHM_TURN_CONFIRM
            and turn_score >= false_break_score + turn_gap_required
        )
        turn_confirmed = turn_base_ready

    state = "RHYTHM_NEUTRAL"
    bias_side = ""
    confidence = 0.0
    if false_break_score >= ROAD_RHYTHM_FALSE_BREAK_GUARD and false_break_score >= turn_score + 0.06:
        state = "RHYTHM_FALSE_BREAK_GUARD"
        bias_side = dominant_side
        confidence = _clamp(false_break_score * 0.72 + inertia_score * 0.28, 0.0, 1.0)
    elif turn_confirmed:
        state = "RHYTHM_TURN_CONFIRM"
        bias_side = turn_bias_side
        vote_boost = min(0.12, turn_confirmation_votes * 0.025) if USE_STRICT_TURN_CONFIRM else 0.0
        confidence = _clamp(turn_score * 0.76 + blue_rising_pressure * 0.20 + vote_boost, 0.0, 1.0)
    elif USE_STRICT_TURN_CONFIRM and turn_base_ready and turn_confirmation_votes < max(0, TURN_CONFIRM_MIN_VOTES):
        # 轉折分數達標但確認票不足：先等確認，不直接反打，避免假斷。
        state = "RHYTHM_TURN_WAIT"
        bias_side = ""
        confidence = _clamp(turn_score * 0.56 + blue_rising_pressure * 0.16, 0.0, 1.0)
    elif short_with and inertia_score >= ROAD_RHYTHM_INERTIA:
        state = "RHYTHM_CONTINUATION"
        bias_side = dominant_side
        confidence = _clamp(inertia_score * 0.80 + short_strength * 0.20, 0.0, 1.0)
    elif conflict_ratio >= 0.48 or (not short_side and not mid_side):
        state = "RHYTHM_CHOP"
        bias_side = ""
        confidence = _clamp(conflict_ratio, 0.0, 1.0)

    state_text = {
        "RHYTHM_FALSE_BREAK_GUARD": "疑似假斷保護",
        "RHYTHM_TURN_CONFIRM": "節奏轉折確認",
        "RHYTHM_TURN_WAIT": "轉折等待確認",
        "RHYTHM_CONTINUATION": "中長節奏延續",
        "RHYTHM_CHOP": "節奏混亂",
        "RHYTHM_NEUTRAL": "節奏中性",
        "RHYTHM_COLD": "節奏資料不足",
    }.get(state, state)
    side_text = {"B": "莊", "P": "閒", "": "無"}.get(bias_side, bias_side)

    return {
        "enabled": True,
        "state": state,
        "label": f"{state_text}:{side_text} 假斷{int(false_break_score*100)} 轉折{int(turn_score*100)} 慣性{int(inertia_score*100)}",
        "bias_side": bias_side,
        "dominant_side": dominant_side,
        "dominant_source": dominant_source,
        "confidence": round(confidence, 4),
        "false_break_score": round(false_break_score, 4),
        "turn_score": round(turn_score, 4),
        "inertia_score": round(inertia_score, 4),
        "blue_rise": round(blue_rise, 4),
        "red_stability": round(red_stability, 4),
        "strict_turn_confirm": USE_STRICT_TURN_CONFIRM,
        "turn_bias_side": turn_bias_side,
        "turn_confirmation_votes": int(turn_confirmation_votes),
        "turn_confirmation_required": int(max(0, TURN_CONFIRM_MIN_VOTES)) if USE_STRICT_TURN_CONFIRM else 0,
        "turn_confirmations": turn_confirmations,
        "turn_gap_required": round(float(turn_gap_required), 4),
        "turn_base_ready": bool(turn_base_ready),
        "windows": {"short": short_f, "mid": mid_f, "long": long_f},
        "derived_windows": {"short": short_d, "mid": mid_d, "long": long_d},
    }


def _apply_road_rhythm_weighting(weights: Dict[str, float], rhythm: Dict[str, Any]) -> Dict[str, float]:
    """依多週期節奏微調權重：避免短線 recent/streak 過度主導。"""
    if not USE_ROAD_RHYTHM or not rhythm.get("enabled"):
        return _normalize_weights(weights)
    state = rhythm.get("state", "RHYTHM_NEUTRAL")
    conf = float(rhythm.get("confidence", 0.0))
    if conf <= 0:
        return _normalize_weights(weights)
    adjusted = dict(weights)
    scale = _clamp(ROAD_RHYTHM_WEIGHT / 0.20, 0.10, 2.20)
    if state == "RHYTHM_FALSE_BREAK_GUARD":
        # 疑似假斷時，降低當前短線 recent/streak，保留中長週期與四路主體。
        adjusted["recent"] = adjusted.get("recent", 0.0) * (1.0 - 0.24 * conf * scale)
        adjusted["streak"] = adjusted.get("streak", 0.0) * (1.0 - 0.20 * conf * scale)
        adjusted["big_road"] = adjusted.get("big_road", 0.0) * (1.0 + 0.10 * conf * scale)
        adjusted["ngram"] = adjusted.get("ngram", 0.0) * (1.0 + 0.08 * conf * scale)
    elif state == "RHYTHM_TURN_CONFIRM":
        # 真轉折時，下三路與 NGram 輔助轉向，降低單純跟龍。
        adjusted["streak"] = adjusted.get("streak", 0.0) * (1.0 - 0.26 * conf * scale)
        adjusted["recent"] = adjusted.get("recent", 0.0) * (1.0 - 0.08 * conf * scale)
        for k in ["big_eye", "small_road", "cockroach", "ngram"]:
            adjusted[k] = adjusted.get(k, 0.0) * (1.0 + 0.13 * conf * scale)
    elif state == "RHYTHM_CONTINUATION":
        for k in ["big_road", "big_eye", "small_road", "cockroach"]:
            adjusted[k] = adjusted.get(k, 0.0) * (1.0 + 0.08 * conf * scale)
    elif state == "RHYTHM_CHOP":
        adjusted["recent"] = adjusted.get("recent", 0.0) * 0.86
        adjusted["streak"] = adjusted.get("streak", 0.0) * 0.82
    return _normalize_weights(adjusted)


def _apply_road_rhythm_bias(b_side: float, rhythm: Dict[str, Any]) -> float:
    """將多週期節奏轉成柔性偏移，避免一口變化就大改方向。"""
    if not USE_ROAD_RHYTHM or not rhythm.get("enabled"):
        return b_side
    state = rhythm.get("state", "RHYTHM_NEUTRAL")
    bias_side = rhythm.get("bias_side", "")
    if state not in {"RHYTHM_FALSE_BREAK_GUARD", "RHYTHM_TURN_CONFIRM", "RHYTHM_CONTINUATION"} or bias_side not in {"B", "P"}:
        if state == "RHYTHM_CHOP":
            return _clamp(0.5 + (b_side - 0.5) * 0.88, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)
        return b_side
    conf = float(rhythm.get("confidence", 0.0))
    scale = _clamp(ROAD_RHYTHM_WEIGHT / 0.20, 0.10, 2.20)
    if state == "RHYTHM_FALSE_BREAK_GUARD":
        base = ROAD_RHYTHM_MAX_BIAS * 0.78
    elif state == "RHYTHM_TURN_CONFIRM":
        base = ROAD_RHYTHM_MAX_BIAS
    else:
        base = ROAD_RHYTHM_MAX_BIAS * 0.55
    signed = 1 if bias_side == "B" else -1
    return _clamp(b_side + signed * base * conf * scale, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)

def _road_engine_score(non_tie: List[str]) -> Dict[str, Any]:
    """
    舊欄位相容：把四路共識包裝成 road_engine。
    新版真正參與融合的是 big_road + down3_family；三條下路只保留明細。
    """
    family = _road_family_scores(non_tie)
    consensus = family.get("consensus", {})
    big_road = family.get("big_road", {})
    return {
        "B": float(consensus.get("B", 0.5)),
        "P": float(consensus.get("P", 0.5)),
        "label": consensus.get("label", "四路共識資料不足"),
        "strength": consensus.get("strength", 0.0),
        "big_road": big_road.get("big_road", {}),
        "derived": {
            "big_eye": family.get("big_eye", {}).get("stats", {}),
            "small_road": family.get("small_road", {}).get("stats", {}),
            "cockroach": family.get("cockroach", {}).get("stats", {}),
        },
        "break_risk": big_road.get("break_risk", 0.0),
        "consistency": consensus.get("consensus_ratio", 0.5),
        "road_family": family,
    }


def _periodicity_score(non_tie: List[str], window: int = 16) -> Dict[str, Any]:
    recent = non_tie[-window:]
    best_period_score = 0.0
    best_period = 0

    for k in range(2, 6):
        if len(recent) > k:
            score = sum(
                1 for i in range(k, len(recent))
                if recent[i] == recent[i - k]
            ) / max(1, len(recent) - k)
            if score > best_period_score:
                best_period_score = score
                best_period = k

    return {"period": best_period, "score": best_period_score}


def _detect_regime(non_tie: List[str]) -> Dict[str, Any]:
    """偵測目前牌路型態，只用於調整權重，不做觀望/下注決策。"""
    fixed_weights = _normalize_weights({
        "big_road": BIG_ROAD_WEIGHT if USE_ROAD_ENGINE else 0.0,
        "big_eye": BIG_EYE_WEIGHT if USE_ROAD_ENGINE else 0.0,
        "small_road": SMALL_ROAD_WEIGHT if USE_ROAD_ENGINE else 0.0,
        "cockroach": COCKROACH_WEIGHT if USE_ROAD_ENGINE else 0.0,
        "ngram": NGRAM_WEIGHT,
        "markov": MARKOV_WEIGHT,
        "road": ROAD_WEIGHT,
        "recent": RECENT_WEIGHT,
        "streak": STREAK_WEIGHT,
        "balance": BALANCE_WEIGHT,
    })

    if not USE_DYNAMIC_REGIME_WEIGHTS:
        return {
            "regime": "fixed",
            "weights": fixed_weights,
            "switch_rate": 0.0,
            "period_score": 0.0,
            "period": 0,
            "streak": 0,
        }

    if len(non_tie) < 8:
        weights = {
            "big_road": 0.18,
            "big_eye": 0.13,
            "small_road": 0.11,
            "cockroach": 0.10,
            "ngram": 0.06,
            "markov": 0.14,
            "road": 0.12,
            "recent": 0.09,
            "streak": 0.05,
            "balance": 0.02,
        }
        if not USE_ROAD_ENGINE:
            for k in ["big_road", "big_eye", "small_road", "cockroach"]:
                weights[k] = 0.0
        return {
            "regime": "cold",
            "weights": _normalize_weights(weights),
            "switch_rate": 0.0,
            "period_score": 0.0,
            "period": 0,
            "streak": _streak(non_tie)[1],
        }

    recent = non_tie[-16:]
    last, streak_n = _streak(non_tie)
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)
    b_rate = recent.count("B") / len(recent)
    period_info = _periodicity_score(non_tie, window=16)
    best_period_score = period_info["score"]
    best_period = period_info["period"]

    # 四路主模型基準權重：大路 + 下三路合計約 62%
    if streak_n >= 4:
        regime = "trend_dragon"
        weights = {
            "big_road": 0.24,
            "big_eye": 0.17,
            "small_road": 0.14,
            "cockroach": 0.10,
            "ngram": 0.07,
            "markov": 0.09,
            "road": 0.08,
            "recent": 0.04,
            "streak": 0.06,
            "balance": 0.01,
        }
    elif switch_rate >= 0.72:
        regime = "single_jump"
        weights = {
            "big_road": 0.18,
            "big_eye": 0.15,
            "small_road": 0.15,
            "cockroach": 0.16,
            "ngram": 0.09,
            "markov": 0.11,
            "road": 0.06,
            "recent": 0.08,
            "streak": 0.01,
            "balance": 0.01,
        }
    elif best_period_score >= 0.70:
        regime = f"periodic_{best_period}"
        weights = {
            "big_road": 0.18,
            "big_eye": 0.15,
            "small_road": 0.14,
            "cockroach": 0.12,
            "ngram": 0.16,
            "markov": 0.08,
            "road": 0.09,
            "recent": 0.05,
            "streak": 0.02,
            "balance": 0.01,
        }
    elif abs(b_rate - 0.5) >= 0.22:
        regime = "biased_side"
        weights = {
            "big_road": 0.21,
            "big_eye": 0.16,
            "small_road": 0.13,
            "cockroach": 0.10,
            "ngram": 0.06,
            "markov": 0.13,
            "road": 0.09,
            "recent": 0.06,
            "streak": 0.04,
            "balance": 0.02,
        }
    elif 0.42 <= switch_rate <= 0.62 and streak_n <= 2 and best_period_score < 0.62:
        regime = "chaos_mixed"
        weights = {
            "big_road": 0.16,
            "big_eye": 0.15,
            "small_road": 0.15,
            "cockroach": 0.15,
            "ngram": 0.12,
            "markov": 0.10,
            "road": 0.06,
            "recent": 0.07,
            "streak": 0.02,
            "balance": 0.02,
        }
    else:
        regime = "mixed"
        weights = {
            "big_road": 0.20,
            "big_eye": 0.16,
            "small_road": 0.14,
            "cockroach": 0.12,
            "ngram": 0.10,
            "markov": 0.10,
            "road": 0.08,
            "recent": 0.06,
            "streak": 0.03,
            "balance": 0.01,
        }

    if not USE_ROAD_ENGINE:
        for k in ["big_road", "big_eye", "small_road", "cockroach"]:
            weights[k] = 0.0

    # 讓環境變數仍可控制四路與 NGram 影響力。
    scale_map = {
        "big_road": (BIG_ROAD_WEIGHT, 0.20),
        "big_eye": (BIG_EYE_WEIGHT, 0.16),
        "small_road": (SMALL_ROAD_WEIGHT, 0.14),
        "cockroach": (COCKROACH_WEIGHT, 0.12),
        "ngram": (NGRAM_WEIGHT, 0.10),
        "markov": (MARKOV_WEIGHT, 0.10),
        "road": (ROAD_WEIGHT, 0.10),
        "recent": (RECENT_WEIGHT, 0.08),
        "streak": (STREAK_WEIGHT, 0.06),
        "balance": (BALANCE_WEIGHT, 0.04),
    }
    for name, (value, base) in scale_map.items():
        if value <= 0:
            weights[name] = 0.0
        else:
            weights[name] *= _clamp(value / max(base, 0.0001), 0.20, 2.50)

    return {
        "regime": regime,
        "weights": _normalize_weights(weights),
        "switch_rate": round(switch_rate, 4),
        "period_score": round(best_period_score, 4),
        "period": best_period,
        "streak": streak_n,
        "recent_b_rate": round(b_rate, 4),
    }


def _rolling_model_performance(non_tie: List[str]) -> Dict[str, Any]:
    """
    用最近 N 局做本靴內部回測，估計各子模型近期準度。

    2026-07 回測安全版：
    - 原本直接用小樣本 acc 會讓 5~20 局的雜訊大幅改權重。
    - 改用 Beta(alpha, alpha) 貝氏收縮，把小樣本準度拉回 0.5。
    - 只有達到 ONLINE_WEIGHT_MIN_COUNT 後才會調整 factor。
    """
    model_names = [
        "big_road", "big_eye", "small_road", "cockroach",
        "ngram", "markov", "road", "recent", "streak", "balance"
    ]
    result = {
        name: {
            "acc": 0.5,
            "raw_acc": 0.5,
            "count": 0,
            "correct": 0,
            "factor": 1.0,
            "shrink_alpha": ONLINE_BAYES_ALPHA,
        }
        for name in model_names
    }

    if not USE_ONLINE_WEIGHTING or len(non_tie) < 12:
        return result

    start = max(6, len(non_tie) - ONLINE_WEIGHT_WINDOW)

    for i in range(start, len(non_tie)):
        prefix = non_tie[:i]
        truth = non_tie[i]
        if truth not in {"B", "P"}:
            continue

        family = _road_family_scores(prefix)
        scores = {
            "big_road": family.get("big_road", {}),
            "big_eye": family.get("big_eye", {}),
            "small_road": family.get("small_road", {}),
            "cockroach": family.get("cockroach", {}),
            "ngram": _ngram_score(prefix),
            "markov": _transition_prob(prefix),
            "road": _road_pattern_score(prefix),
            "recent": _recent_score(prefix),
            "streak": _streak_score(prefix),
            "balance": _balance_score(prefix),
        }

        for name, score in scores.items():
            pick = _pick_from_score(score, min_edge=0.002)
            if not pick:
                continue
            result[name]["count"] += 1
            if pick == truth:
                result[name]["correct"] += 1

    for name in model_names:
        cnt = int(result[name]["count"])
        cor = int(result[name]["correct"])

        if cnt > 0:
            raw_acc = cor / cnt
            # Beta(alpha, alpha) prior，prior mean = 0.5。
            # 樣本越少，acc 越接近 0.5；樣本越多，越接近 raw_acc。
            alpha = max(0.0001, ONLINE_BAYES_ALPHA)
            acc = (cor + alpha) / (cnt + 2 * alpha)
            result[name]["raw_acc"] = round(raw_acc, 4)
            result[name]["acc"] = round(acc, 4)
        else:
            raw_acc = 0.5
            acc = 0.5
            result[name]["raw_acc"] = 0.5
            result[name]["acc"] = 0.5

        factor = 1.0
        if cnt >= ONLINE_WEIGHT_MIN_COUNT:
            factor = 1.0 + (acc - 0.5) * 2 * ONLINE_WEIGHT_ALPHA
            if acc <= ONLINE_DISABLE_BELOW:
                factor = min(factor, 0.78)
            elif acc >= ONLINE_BOOST_ABOVE:
                factor = max(factor, 1.05)
            factor = _clamp(factor, 0.70, 1.20)

        result[name]["factor"] = round(factor, 4)

    return result

def _apply_online_weighting(base_weights: Dict[str, float], performance: Dict[str, Any]) -> Dict[str, float]:
    if not USE_ONLINE_WEIGHTING:
        return _normalize_weights(base_weights)

    adjusted = {}
    for name, weight in base_weights.items():
        factor = float(performance.get(name, {}).get("factor", 1.0))
        adjusted[name] = weight * factor
    return _normalize_weights(adjusted)


# ============ Walk-forward Learning：逐局前推 / 每個 LINE UID 獨立 ============
def _wf_empty_model_stats() -> Dict[str, Any]:
    return {"acc": 0.5, "raw_acc": 0.5, "count": 0, "correct": 0, "factor": 1.0}


def _wf_state(training_key: str) -> Dict[str, Any]:
    key = training_key or "anonymous|global"
    state = _WALK_FORWARD_STATE.get(key)
    if state is None:
        state = {"pending": None, "records": []}
        _WALK_FORWARD_STATE[key] = state
    return state


def _walk_forward_pick_from_score(score: Dict[str, Any], min_edge: Optional[float] = None) -> str:
    edge = WALK_FORWARD_MIN_EDGE if min_edge is None else float(min_edge)
    return _pick_from_score(score, min_edge=edge)


def _walk_forward_pick_map(scores: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    picks: Dict[str, str] = {}
    if not USE_WALK_FORWARD_LEARNING:
        return picks
    for name, score in scores.items():
        try:
            pick = _walk_forward_pick_from_score(score)
        except Exception:
            pick = ""
        if pick in {"B", "P"}:
            picks[name] = pick
    return picks


def _update_walk_forward_truth(training_key: str, non_tie: List[str]) -> None:
    """把上一輪 pending 的預測，用這一輪新進來的真實結果結算。

    pending 在 len=N 時代表「預測第 N+1 手」。
    當目前 non_tie 長度 > N 時，truth = non_tie[N]，這樣完全不偷看未來。
    """
    if not (USE_WALK_FORWARD_LEARNING and WALK_FORWARD_LIVE_PER_UID):
        return
    state = _wf_state(training_key)
    pending = state.get("pending")
    if not pending:
        return
    pred_len = int(pending.get("non_tie_len", -1))
    if pred_len < 0 or len(non_tie) <= pred_len:
        return
    truth = non_tie[pred_len]
    if truth not in {"B", "P"}:
        state["pending"] = None
        return
    preds = pending.get("predictions", {}) or {}
    record = {"truth": truth, "at_len": pred_len, "models": {}}
    for model_name, pick in preds.items():
        if pick in {"B", "P"}:
            record["models"][model_name] = 1 if pick == truth else 0
    if record["models"]:
        records = state.setdefault("records", [])
        records.append(record)
        max_keep = max(10, WALK_FORWARD_WINDOW * 3)
        if len(records) > max_keep:
            del records[:-max_keep]
    state["pending"] = None


def _get_walk_forward_performance(training_key: str) -> Dict[str, Any]:
    if not (USE_WALK_FORWARD_LEARNING and WALK_FORWARD_LIVE_PER_UID):
        return {}
    state = _wf_state(training_key)
    records = state.get("records", [])[-max(1, WALK_FORWARD_WINDOW):]
    model_names = set()
    for rec in records:
        model_names.update((rec.get("models") or {}).keys())
    result: Dict[str, Any] = {name: _wf_empty_model_stats() for name in sorted(model_names)}
    alpha = max(0.0001, WALK_FORWARD_BAYES_ALPHA)
    for name in list(result.keys()):
        vals = [int((rec.get("models") or {}).get(name)) for rec in records if name in (rec.get("models") or {})]
        cnt = len(vals)
        cor = sum(vals)
        if cnt <= 0:
            continue
        raw_acc = cor / cnt
        acc = (cor + alpha) / (cnt + 2 * alpha)
        factor = 1.0
        if cnt >= WALK_FORWARD_MIN_COUNT:
            factor = 1.0 + (acc - 0.5) * 2.0 * WALK_FORWARD_ALPHA
            if acc <= WALK_FORWARD_DISABLE_BELOW:
                factor = min(factor, 0.88)
            elif acc >= WALK_FORWARD_BOOST_ABOVE:
                factor = max(factor, 1.04)
            factor = _clamp(factor, WALK_FORWARD_MIN_FACTOR, WALK_FORWARD_MAX_FACTOR)
        result[name] = {
            "acc": round(acc, 4),
            "raw_acc": round(raw_acc, 4),
            "count": cnt,
            "correct": cor,
            "factor": round(factor, 4),
        }
    return result


def _apply_walk_forward_weighting(base_weights: Dict[str, float], live_performance: Dict[str, Any]) -> Dict[str, float]:
    if not (USE_WALK_FORWARD_LEARNING and live_performance):
        return _normalize_weights(base_weights)
    adjusted: Dict[str, float] = {}
    for name, weight in base_weights.items():
        factor = float(live_performance.get(name, {}).get("factor", 1.0))
        adjusted[name] = weight * factor
    return _normalize_weights(adjusted)


def _walk_forward_factor(live_performance: Dict[str, Any], model_name: str, default: float = 1.0) -> float:
    try:
        return float(live_performance.get(model_name, {}).get("factor", default))
    except Exception:
        return default


def _store_walk_forward_pending(training_key: str, non_tie: List[str], predictions: Dict[str, str]) -> None:
    if not (USE_WALK_FORWARD_LEARNING and WALK_FORWARD_LIVE_PER_UID and WALK_FORWARD_STORE_PENDING):
        return
    clean = {str(k): v for k, v in (predictions or {}).items() if v in {"B", "P"}}
    if not clean:
        return
    state = _wf_state(training_key)
    state["pending"] = {
        "non_tie_len": len(non_tie),
        "predictions": clean,
    }


def get_walk_forward_state_info() -> Dict[str, Any]:
    """方便 debug：查看目前每個 LINE UID / 房間 / 靴號的逐局前推狀態。"""
    return {
        "enabled": USE_WALK_FORWARD_LEARNING,
        "size": len(_WALK_FORWARD_STATE),
        "keys": list(_WALK_FORWARD_STATE.keys())[-30:],
    }


def clear_walk_forward_state() -> Dict[str, Any]:
    removed = len(_WALK_FORWARD_STATE)
    _WALK_FORWARD_STATE.clear()
    return {"ok": True, "removed": removed}



# ============ Ask Road Hit Memory：問路命中率記憶 ============
def _ask_road_state(training_key: str) -> Dict[str, Any]:
    key = training_key or "anonymous|ask_road"
    state = _ASK_ROAD_STATE.get(key)
    if state is None:
        state = {"pending": None, "records": []}
        _ASK_ROAD_STATE[key] = state
    return state


def _update_ask_road_truth(training_key: str, non_tie: List[str]) -> None:
    # 將上一輪問路票用本輪新增結果結算。
    # pending 的 non_tie_len=N，代表上一輪在 N 口時預測第 N+1 口。
    # 本輪 len(non_tie)>N 時，truth=non_tie[N]，完全不偷看未來。
    if not USE_ASK_ROAD_MEMORY:
        return

    state = _ask_road_state(training_key)
    pending = state.get("pending")
    if not pending:
        return

    pred_len = int(pending.get("non_tie_len", -1))
    if pred_len < 0 or len(non_tie) <= pred_len:
        return

    truth = non_tie[pred_len]
    if truth not in {"B", "P"}:
        state["pending"] = None
        return

    predictions = pending.get("predictions", {}) or {}
    record = {"truth": truth, "at_len": pred_len, "models": {}}
    for name, pick in predictions.items():
        if pick in {"B", "P"}:
            record["models"][name] = 1 if pick == truth else 0

    if record["models"]:
        records = state.setdefault("records", [])
        records.append(record)
        max_keep = max(20, ASK_ROAD_MEMORY_WINDOW * 4)
        if len(records) > max_keep:
            del records[:-max_keep]

    state["pending"] = None


def _get_ask_road_performance(training_key: str) -> Dict[str, Any]:
    # 回傳每條問路最近命中率與動態 factor。
    default_models = ["big_eye", "small_road", "cockroach", "down3_family", "road_majority", "final"]
    result: Dict[str, Any] = {
        "enabled": USE_ASK_ROAD_MEMORY,
        "window": ASK_ROAD_MEMORY_WINDOW,
        "models": {
            name: {
                "acc": 0.5,
                "raw_acc": 0.5,
                "count": 0,
                "correct": 0,
                "factor": 1.0,
            }
            for name in default_models
        },
        "label": "問路記憶尚未啟用" if not USE_ASK_ROAD_MEMORY else "問路記憶暖機中",
    }

    if not USE_ASK_ROAD_MEMORY:
        return result

    state = _ask_road_state(training_key)
    records = state.get("records", [])[-max(1, ASK_ROAD_MEMORY_WINDOW):]
    alpha = max(0.0001, ASK_ROAD_MEMORY_BAYES_ALPHA)

    model_names = set(default_models)
    for rec in records:
        model_names.update((rec.get("models") or {}).keys())

    models: Dict[str, Any] = {}
    best_name = ""
    best_acc = 0.5

    for name in sorted(model_names):
        vals = [
            int((rec.get("models") or {}).get(name))
            for rec in records
            if name in (rec.get("models") or {})
        ]
        count = len(vals)
        correct = sum(vals)
        raw_acc = correct / count if count else 0.5
        acc = (correct + alpha) / (count + 2 * alpha) if count else 0.5

        factor = 1.0
        if count >= ASK_ROAD_MEMORY_MIN_COUNT:
            factor = 1.0 + (acc - 0.5) * 2.0 * ASK_ROAD_MEMORY_ALPHA
            if acc <= ASK_ROAD_MEMORY_DISABLE_BELOW:
                factor = min(factor, 0.90)
            elif acc >= ASK_ROAD_MEMORY_BOOST_ABOVE:
                factor = max(factor, 1.05)
            factor = _clamp(factor, ASK_ROAD_MEMORY_MIN_FACTOR, ASK_ROAD_MEMORY_MAX_FACTOR)

        models[name] = {
            "acc": round(acc, 4),
            "raw_acc": round(raw_acc, 4),
            "count": count,
            "correct": correct,
            "factor": round(factor, 4),
        }

        if count >= ASK_ROAD_MEMORY_MIN_COUNT and acc > best_acc:
            best_acc = acc
            best_name = name

    result["models"] = models
    if best_name:
        result["label"] = f"問路記憶:{best_name}較準 {int(best_acc * 100)}%"
    else:
        result["label"] = f"問路記憶暖機中 樣本{len(records)}"

    return result


def _ask_road_factor(performance: Optional[Dict[str, Any]], road_key: str, default: float = 1.0) -> float:
    try:
        if not (USE_ASK_ROAD_MEMORY and performance):
            return default
        return float((performance.get("models") or {}).get(road_key, {}).get("factor", default))
    except Exception:
        return default


def _apply_ask_road_factor_to_score(score: Dict[str, Any], road_key: str, performance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # 用問路近期命中率微調 HYBRID 下三路 B/P 邊際。
    # factor>1：放大這條路的邊際；factor<1：縮小這條路的邊際。
    if not (USE_ASK_ROAD_MEMORY and ASK_ROAD_MEMORY_APPLY_TO_HYBRID and performance and isinstance(score, dict)):
        return score

    factor = _ask_road_factor(performance, road_key, 1.0)
    if abs(factor - 1.0) < 0.0001:
        return score

    try:
        b = float(score.get("B", 0.5))
        p = float(score.get("P", 0.5))
        side_total = max(0.0001, b + p)
        b_side = b / side_total
        edge = b_side - 0.5
        new_b_side = _clamp(0.5 + edge * factor, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)
        new_score = dict(score)
        new_score["B"] = round(new_b_side, 5)
        new_score["P"] = round(1.0 - new_b_side, 5)
        new_score["ask_road_memory_factor"] = round(factor, 4)
        new_score["ask_road_memory"] = (performance.get("models") or {}).get(road_key, {})
        old_label = str(new_score.get("label", ""))
        new_score["label"] = f"{old_label}|問路記憶x{factor:.2f}" if old_label else f"問路記憶x{factor:.2f}"
        return new_score
    except Exception:
        return score


def _apply_ask_road_factor_to_vote(vote: Dict[str, Any], road_key: str, performance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """用近期命中率微調 FUHAO 票的柔性邊際；不把弱票變成新方向。"""
    if not (USE_ASK_ROAD_MEMORY and ASK_ROAD_MEMORY_APPLY_TO_FUHAO and performance and isinstance(vote, dict)):
        return vote

    stat = (performance.get("models") or {}).get(road_key, {})
    if not stat:
        return vote
    factor = float(stat.get("factor", 1.0))
    count = int(stat.get("count", 0))
    acc = float(stat.get("acc", 0.5))

    new_vote = dict(vote)
    old_conf = float(new_vote.get("confidence", 0.0) or 0.0)
    new_vote["confidence"] = round(_clamp(old_conf * factor, 0.0, 0.88), 4)
    new_vote["ask_road_memory_factor"] = round(factor, 4)
    new_vote["ask_road_memory"] = stat

    # 若票中有 B/P，僅縮放原本邊際，絕不憑記憶創造反向票。
    if "B" in new_vote and "P" in new_vote:
        b = float(new_vote.get("B", 0.5))
        p = float(new_vote.get("P", 0.5))
        total = max(0.0001, b + p)
        b_side = b / total
        b_side = _clamp(0.5 + (b_side - 0.5) * factor, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)
        new_vote["B"] = round(b_side, 5)
        new_vote["P"] = round(1.0 - b_side, 5)

    if (
        ASK_ROAD_MEMORY_DROP_BAD_VOTE
        and count >= ASK_ROAD_MEMORY_MIN_COUNT
        and acc <= ASK_ROAD_MEMORY_BAD_VOTE_ACC
        and new_vote.get("pick") in {"B", "P"}
    ):
        old_pick = new_vote.get("pick", "")
        new_vote["pick"] = ""
        new_vote["label"] = f"{new_vote.get('label', '')}|問路記憶暫停{_fuhao_side_name(old_pick)}票 acc{acc:.2f}"
    else:
        new_vote["label"] = f"{new_vote.get('label', '')}|問路記憶x{factor:.2f}"
    return new_vote

def _store_ask_road_pending(training_key: str, non_tie: List[str], predictions: Dict[str, str]) -> None:
    if not (USE_ASK_ROAD_MEMORY and predictions):
        return

    clean = {str(k): v for k, v in predictions.items() if v in {"B", "P"}}
    if not clean:
        return

    state = _ask_road_state(training_key)
    state["pending"] = {
        "non_tie_len": len(non_tie),
        "predictions": clean,
    }


def get_ask_road_state_info() -> Dict[str, Any]:
    return {
        "enabled": USE_ASK_ROAD_MEMORY,
        "size": len(_ASK_ROAD_STATE),
        "keys": list(_ASK_ROAD_STATE.keys())[-30:],
    }


def clear_ask_road_state() -> Dict[str, Any]:
    removed = len(_ASK_ROAD_STATE)
    _ASK_ROAD_STATE.clear()
    return {"ok": True, "removed": removed}


# ============ Pattern Replay Memory：全靴逐局前推相似規律回放 ============
def _parse_int_list(raw: str, default: List[int]) -> List[int]:
    try:
        vals = []
        for x in str(raw or "").replace(";", ",").split(","):
            x = x.strip()
            if not x:
                continue
            vals.append(int(x))
        vals = sorted(set(v for v in vals if v >= 2))
        return vals or list(default)
    except Exception:
        return list(default)


def _opp_side(side: str) -> str:
    return "P" if side == "B" else "B" if side == "P" else ""


def _relative_shape(seq: List[str]) -> List[str]:
    """把 B/P 轉成相對形狀：第一個方向=A，另一邊=X。可跨莊閒方向比較節奏。"""
    if not seq:
        return []
    first = seq[0]
    other = _opp_side(first)
    return ["A" if x == first else "X" if x == other else "?" for x in seq]


def _transition_shape(seq: List[str]) -> List[str]:
    if len(seq) < 2:
        return []
    return ["S" if a == b else "C" for a, b in zip(seq, seq[1:])]


def _pattern_similarity(current: List[str], past: List[str]) -> Dict[str, float]:
    """相似度不是只看 B/P 是否完全一樣，也看路型形狀與連斷節奏。"""
    if not current or len(current) != len(past):
        return {"score": 0.0, "exact": 0.0, "shape": 0.0, "transition": 0.0}
    n = len(current)
    exact = sum(1 for a, b in zip(current, past) if a == b) / max(1, n)
    cshape = _relative_shape(current)
    pshape = _relative_shape(past)
    shape = sum(1 for a, b in zip(cshape, pshape) if a == b) / max(1, n)
    ct = _transition_shape(current)
    pt = _transition_shape(past)
    trans = sum(1 for a, b in zip(ct, pt) if a == b) / max(1, len(ct)) if ct else 0.0
    total_w = max(0.0001, PATTERN_REPLAY_EXACT_WEIGHT + PATTERN_REPLAY_SHAPE_WEIGHT + PATTERN_REPLAY_TRANSITION_WEIGHT)
    score = (
        exact * PATTERN_REPLAY_EXACT_WEIGHT
        + shape * PATTERN_REPLAY_SHAPE_WEIGHT
        + trans * PATTERN_REPLAY_TRANSITION_WEIGHT
    ) / total_w
    return {"score": round(score, 5), "exact": round(exact, 5), "shape": round(shape, 5), "transition": round(trans, 5)}


def _map_past_truth_to_current(current: List[str], past: List[str], past_truth: str) -> str:
    """把過去相似片段的下一口，映射到目前片段方向。

    例如過去 PPBPPB 後接 P，而目前 BBPBBP 是同形狀反色，則映射成 B。
    這樣學的是「規律形狀會延續或反轉」，不是死記莊/閒數量。
    """
    if not current or not past or past_truth not in {"B", "P"}:
        return ""
    past_first = past[0]
    current_first = current[0]
    current_opp = _opp_side(current_first)
    if past_truth == past_first:
        return current_first
    return current_opp


def _pattern_replay_memory_score(non_tie: List[str], training_key: str = "", live_performance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """全靴逐局前推相似規律回放。

    原則：
    - 預測第 N+1 口時，只能使用第 1~N 口。
    - 不是只看最近幾口，而是用多尺度窗口，把目前尾段拿去前面所有已知牌路找相似片段。
    - 找到相似片段後，看當時下一口如何走，再映射成目前的 B/P 候選。
    """
    default = {
        "enabled": False,
        "state": "REPLAY_COLD",
        "label": "Pattern Replay資料不足",
        "B": 0.5,
        "P": 0.5,
        "bias_side": "",
        "confidence": 0.0,
        "edge": 0.0,
        "sample": 0,
        "weighted_sample": 0.0,
        "b_weight": 0.0,
        "p_weight": 0.0,
        "windows_used": [],
        "matched_examples": [],
        "training_key": training_key,
    }
    if not USE_PATTERN_REPLAY_MEMORY:
        return default
    n = len(non_tie)
    if n < PATTERN_REPLAY_MIN_HISTORY:
        return {**default, "enabled": True, "state": "REPLAY_WARMUP", "label": f"Pattern Replay暖機中 {n}/{PATTERN_REPLAY_MIN_HISTORY}"}

    windows = _parse_int_list(PATTERN_REPLAY_WINDOWS, [5, 6, 8, 10, 12, 16])
    windows = [w for w in windows if 2 <= w < n]
    if not windows:
        return {**default, "enabled": True, "state": "REPLAY_NO_WINDOW", "label": "Pattern Replay沒有可用窗口"}

    # 同一段牌路重複分析時直接吃快取，避免 Render 重複掃描導致卡住。
    seq_key = "".join(non_tie)
    cache_key = f"{training_key or 'global'}|{n}|{seq_key[-80:]}|{','.join(map(str, windows))}"
    if cache_key in _PATTERN_REPLAY_CACHE:
        cached = dict(_PATTERN_REPLAY_CACHE[cache_key])
        cached["cache_hit"] = True
        return cached

    b_w = 0.0
    p_w = 0.0
    raw_matches = 0
    exact_matches = 0
    max_score = 0.0
    examples: List[Dict[str, Any]] = []
    windows_used = []
    max_window = max(windows) if windows else 1

    total_scanned = 0
    scan_budget = max(60, PATTERN_REPLAY_MAX_SCAN)
    for w in windows:
        if total_scanned >= scan_budget:
            break
        current = non_tie[-w:]
        if len(current) < w:
            continue
        start = 0 if PATTERN_REPLAY_FULL_SHOE else max(0, n - PATTERN_REPLAY_LOOKBACK - w - 1)
        # j+w 必須 < n，因為 non_tie[j+w] 才是當時已知的下一口；不能用現在尚未發生的下一口。
        end = n - w
        local_matches = 0
        for j in range(start, end):
            total_scanned += 1
            if total_scanned > scan_budget:
                break
            truth_idx = j + w
            if truth_idx >= n:
                continue
            past = non_tie[j:j + w]
            truth = non_tie[truth_idx]
            if truth not in {"B", "P"} or len(past) != w:
                continue
            sim_info = _pattern_similarity(current, past)
            sim = float(sim_info.get("score", 0.0))
            max_score = max(max_score, sim)
            if sim < PATTERN_REPLAY_MIN_SIMILARITY:
                continue
            mapped = _map_past_truth_to_current(current, past, truth)
            if mapped not in {"B", "P"}:
                continue
            recency = (truth_idx + 1) / max(1, n)
            window_factor = 1.0 + PATTERN_REPLAY_LONG_WINDOW_WEIGHT * (w / max(1, max_window))
            weight = sim * (1.0 + PATTERN_REPLAY_RECENCY_WEIGHT * recency) * window_factor
            if mapped == "B":
                b_w += weight
            else:
                p_w += weight
            raw_matches += 1
            local_matches += 1
            if sim_info.get("exact", 0.0) >= 0.999:
                exact_matches += 1
            if len(examples) < PATTERN_REPLAY_MAX_MATCHES:
                examples.append({
                    "window": w,
                    "start_round": j + 1,
                    "truth_round": truth_idx + 1,
                    "past": "".join(past),
                    "current": "".join(current),
                    "truth": truth,
                    "mapped": mapped,
                    "similarity": round(sim, 4),
                    "exact": sim_info.get("exact", 0.0),
                    "shape": sim_info.get("shape", 0.0),
                    "transition": sim_info.get("transition", 0.0),
                    "weight": round(weight, 4),
                })
        if local_matches:
            windows_used.append({"window": w, "matches": local_matches})

    weighted_sample = b_w + p_w
    if raw_matches < PATTERN_REPLAY_MIN_MATCHES or weighted_sample <= 0:
        result_no_match = {
            **default,
            "enabled": True,
            "state": "REPLAY_NO_MATCH",
            "label": f"Pattern Replay相似樣本不足 M{raw_matches} MaxSim{max_score:.2f} Scan{total_scanned}",
            "sample": raw_matches,
            "weighted_sample": round(weighted_sample, 4),
            "max_similarity": round(max_score, 4),
            "scanned": int(total_scanned),
            "scan_budget": int(scan_budget),
            "matched_examples": examples[:8] if PATTERN_REPLAY_DEBUG else [],
            "windows_used": windows_used,
        }
        _PATTERN_REPLAY_CACHE[cache_key] = result_no_match
        _PATTERN_REPLAY_CACHE_ORDER.append(cache_key)
        while len(_PATTERN_REPLAY_CACHE_ORDER) > PATTERN_REPLAY_CACHE_SIZE:
            old = _PATTERN_REPLAY_CACHE_ORDER.pop(0)
            _PATTERN_REPLAY_CACHE.pop(old, None)
        return result_no_match

    alpha = max(0.0001, PATTERN_REPLAY_BAYES_ALPHA)
    b_rate = (b_w + alpha) / (weighted_sample + 2 * alpha)
    p_rate = 1.0 - b_rate
    edge = abs(b_rate - 0.5)
    sample_strength = _clamp(weighted_sample / max(4.0, PATTERN_REPLAY_MIN_MATCHES * 2.5), 0.0, 1.0)
    confidence = _clamp(sample_strength * (0.24 + edge * 2.4) + min(0.18, exact_matches * 0.025), 0.0, 1.0)
    bias_side = "B" if b_rate > p_rate else "P" if p_rate > b_rate else ""

    state = "REPLAY_MATCH"
    if edge < PATTERN_REPLAY_MIN_EDGE:
        state = "REPLAY_NEUTRAL"
        bias_side = ""
    side_text = {"B": "莊", "P": "閒", "": "中性"}.get(bias_side, bias_side)
    label = f"Pattern Replay:{side_text} B{int(b_rate*100)} P{int(p_rate*100)} 樣本{raw_matches} 窗口{','.join(str(x.get('window')) for x in windows_used[:5])}"

    result_match = {
        "enabled": True,
        "state": state,
        "label": label + f" Scan{total_scanned}",
        "B": round(float(b_rate), 5),
        "P": round(float(p_rate), 5),
        "bias_side": bias_side,
        "confidence": round(confidence, 4),
        "edge": round(edge, 5),
        "sample": raw_matches,
        "weighted_sample": round(weighted_sample, 4),
        "b_weight": round(b_w, 4),
        "p_weight": round(p_w, 4),
        "exact_matches": exact_matches,
        "max_similarity": round(max_score, 4),
        "scanned": int(total_scanned),
        "scan_budget": int(scan_budget),
        "windows_used": windows_used,
        "matched_examples": examples[:12] if PATTERN_REPLAY_DEBUG else examples[:4],
        "training_key": training_key,
    }
    _PATTERN_REPLAY_CACHE[cache_key] = result_match
    _PATTERN_REPLAY_CACHE_ORDER.append(cache_key)
    while len(_PATTERN_REPLAY_CACHE_ORDER) > PATTERN_REPLAY_CACHE_SIZE:
        old = _PATTERN_REPLAY_CACHE_ORDER.pop(0)
        _PATTERN_REPLAY_CACHE.pop(old, None)
    return result_match


def _apply_pattern_replay_bias(b_side: float, pattern_replay: Dict[str, Any], live_performance: Optional[Dict[str, Any]] = None) -> float:
    if not USE_PATTERN_REPLAY_MEMORY or not pattern_replay.get("enabled"):
        return b_side
    if pattern_replay.get("state") != "REPLAY_MATCH":
        return b_side
    side = pattern_replay.get("bias_side", "")
    if side not in {"B", "P"}:
        return b_side
    conf = float(pattern_replay.get("confidence", 0.0) or 0.0)
    edge = float(pattern_replay.get("edge", 0.0) or 0.0)
    if conf <= 0 or edge < PATTERN_REPLAY_MIN_EDGE:
        return b_side
    wf_factor = 1.0
    if PATTERN_REPLAY_APPLY_WF and live_performance:
        wf_factor = _walk_forward_factor(live_performance, "pattern_replay", 1.0)
    scale = _clamp(PATTERN_REPLAY_WEIGHT / 0.24, 0.10, 2.50)
    strength = PATTERN_REPLAY_MAX_BIAS * conf * _clamp(edge * 3.0, 0.20, 1.0) * scale * wf_factor
    signed = 1 if side == "B" else -1
    return _clamp(b_side + signed * strength, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)


def _tie_score(history: List[str]) -> float:
    if not history:
        return T_PRIOR
    recent = history[-18:]
    t_rate = recent.count("T") / len(recent)
    gap_since_tie = 0
    for x in reversed(history):
        if x == "T":
            break
        gap_since_tie += 1
    pressure = T_PRIOR * (1 - TIE_SHRINK) + t_rate * TIE_SHRINK
    if gap_since_tie >= 18:
        pressure += 0.012
    if recent[-4:].count("T") >= 2:
        pressure += 0.018
    return _clamp(pressure, 0.055, TIE_MAX_PROB)


def _confidence(b: float, p: float, t: float, history_len: int, agreement: float, ml_agreement: float = 0.0) -> Tuple[float, str]:
    gap = abs(b - p)
    base = gap * 3.6 + agreement * 0.22 + ml_agreement * 0.10 + min(0.16, history_len / 80)
    conf = _clamp(base, 0.08, 0.94)
    if history_len < MIN_HISTORY_FOR_SIGNAL:
        return min(conf, 0.35), "冷啟動"
    if conf >= 0.68:
        return conf, "強訊號"
    if conf >= 0.48:
        return conf, "中訊號"
    return conf, "弱訊號"

# ============ 主要預測函數 ============

# ============ 富濠式模型：規則多數決，不吃當局點數 / 不跑 ML / 不跑 DeepSeek ============
def _fuhao_side_name(side: str) -> str:
    return {"B": "莊", "P": "閒", "T": "和", "NONE": "觀望", "": "無"}.get(side, side)


def _fuhao_tie_break_side(big_road_pick: str = "") -> str:
    mode = FUHAO_FINAL_TIE_BREAKER
    if mode == "BIGROAD" and big_road_pick in {"B", "P"}:
        return big_road_pick
    if mode in {"PLAYER", "P"}:
        return "P"
    # 富濠式原邏輯遇平手通常偏莊；保守一點可用 BIGROAD。
    return "B"


def _fuhao_majority(votes: List[str], big_road_pick: str = "") -> Dict[str, Any]:
    clean = [v for v in votes if v in {"B", "P"}]
    b_count = clean.count("B")
    p_count = clean.count("P")
    if not clean:
        return {"pick": "", "B": 0, "P": 0, "total": 0, "ratio": 0.0, "tie": True}
    if b_count > p_count:
        pick = "B"
        tie = False
    elif p_count > b_count:
        pick = "P"
        tie = False
    else:
        # 最後整合版：平手時預設不再用大路/莊家偏置硬拆，避免票數平手仍偏某一方。
        pick = "" if FUHAO_DISABLE_BIGROAD_FALLBACK else _fuhao_tie_break_side(big_road_pick)
        tie = True
    ratio = max(b_count, p_count) / max(1, len(clean))
    if tie:
        ratio = 0.5
    return {"pick": pick, "B": b_count, "P": p_count, "total": len(clean), "ratio": round(ratio, 4), "tie": tie}


def _fuhao_big_road_vote(non_tie: List[str]) -> Dict[str, Any]:
    if not non_tie:
        return {"pick": "", "label": "大路資料不足", "confidence": 0.0, "details": {}}

    recent = non_tie[-16:]
    last_side, streak_n = _streak(non_tie)
    opp = "P" if last_side == "B" else "B"
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)
    layout = _build_big_road(non_tie)
    last_pos = layout.get("last", {})
    last_col = int(last_pos.get("col", 0))
    current_col_height = int(layout.get("col_heights", {}).get(last_col, 0))

    # 富濠式大路核心：看最新欄/最新方向，規則命名只用來解釋。
    pick = last_side
    label = "大路跟最新欄"
    confidence = 0.56

    if streak_n >= FUHAO_LONG_THRESHOLD:
        label = f"大路長龍{_fuhao_side_name(last_side)}{streak_n}口"
        confidence = min(0.72, 0.58 + streak_n * 0.025)
    elif len(recent) >= 6 and switch_rate >= 0.72:
        # 單跳盤以反手作大路節奏判斷，避免固定死跟最後一欄。
        pick = opp
        label = "大路單跳節奏"
        confidence = min(0.70, 0.56 + (switch_rate - 0.72) * 0.35)
    elif len(non_tie) >= 6 and "".join(non_tie[-6:]) in {"BBPPBB", "PPBBPP", "BPPBBP", "PBBPPB"}:
        pick = non_tie[-2]
        label = "大路雙跳/排排連"
        confidence = 0.62
    elif current_col_height >= 3:
        label = "大路欄高延續"
        confidence = 0.60

    return {
        "pick": pick,
        "label": label,
        "confidence": round(confidence, 4),
        "details": {
            "last_side": last_side,
            "streak": streak_n,
            "switch_rate": round(switch_rate, 4),
            "current_col_height": current_col_height,
            "last_col": last_col,
        },
    }


def _fuhao_down3_vote(non_tie: List[str], offset: int, name: str) -> Dict[str, Any]:
    """FUHAO 單一下路明細；不加入全局欄型，也不直接成為外部完整票。"""
    if len(non_tie) < FUHAO_MIN_VALID_ROUNDS:
        return {"pick": "", "label": f"{name}資料不足", "confidence": 0.0, "B": 0.5, "P": 0.5, "stats": {}, "candidate": {}}

    layout = _build_big_road(non_tie)
    series = _derived_series(layout, offset=offset)
    stats = _color_stats(series)
    count = int(stats.get("count", 0))
    if count < DERIVED_ROAD_MIN_COUNT:
        return {"pick": "", "label": f"{name}樣本不足", "confidence": 0.0, "B": 0.5, "P": 0.5, "stats": stats, "candidate": {}}

    b_info = _candidate_derived_color_info(non_tie, "B", offset)
    p_info = _candidate_derived_color_info(non_tie, "P", offset)
    b_color_eval = _score_candidate_color_pattern(series, int(b_info.get("new_color", 0)))
    p_color_eval = _score_candidate_color_pattern(series, int(p_info.get("new_color", 0)))
    b_struct_eval = _score_candidate_structure(b_info, series)
    p_struct_eval = _score_candidate_structure(p_info, series)
    b_score = _combine_candidate_scores(float(b_color_eval.get("score", 0.5)), float(b_struct_eval.get("score", 0.5)))
    p_score = _combine_candidate_scores(float(p_color_eval.get("score", 0.5)), float(p_struct_eval.get("score", 0.5)))

    raw_diff = b_score - p_score
    b_prob, p_prob, _ = _candidate_scores_to_side_prob(b_score, p_score, max_edge=DERIVED_CANDIDATE_MAX_EDGE)
    prob_gap = b_prob - p_prob
    if abs(prob_gap) < DOWN3_FAMILY_ROAD_MIN_GAP:
        pick = ""
        label = f"{name}弱訊號"
        confidence = 0.30
    else:
        pick = "B" if prob_gap > 0 else "P"
        label = f"{name}內部偏{_fuhao_side_name(pick)}"
        confidence = min(0.68, 0.38 + abs(prob_gap) * 3.0 + min(0.06, count * 0.004))

    return {
        "pick": pick,
        "label": label,
        "confidence": round(confidence, 4),
        "B": round(b_prob, 5),
        "P": round(p_prob, 5),
        "stats": stats,
        "candidate": {
            "B": {
                "new_color": b_info.get("new_color_text", "N"),
                "color_score": round(float(b_color_eval.get("score", 0.5)), 5),
                "structure_score": round(float(b_struct_eval.get("score", 0.5)), 5),
                "column_score": 0.5,
                "score": round(b_score, 5),
                "color_eval": b_color_eval,
                "structure_eval": b_struct_eval,
                "column_eval": {"score": 0.5, "label": "欄型由家族層只算一次"},
                "pos": b_info.get("pos", {}),
                "structure": b_info.get("structure", {}),
            },
            "P": {
                "new_color": p_info.get("new_color_text", "N"),
                "color_score": round(float(p_color_eval.get("score", 0.5)), 5),
                "structure_score": round(float(p_struct_eval.get("score", 0.5)), 5),
                "column_score": 0.5,
                "score": round(p_score, 5),
                "color_eval": p_color_eval,
                "structure_eval": p_struct_eval,
                "column_eval": {"score": 0.5, "label": "欄型由家族層只算一次"},
                "pos": p_info.get("pos", {}),
                "structure": p_info.get("structure", {}),
            },
            "diff": round(raw_diff, 5),
            "prob_gap": round(prob_gap, 5),
            "column_applied_here": False,
        },
    }


def _fuhao_down3_family_vote(non_tie: List[str], road_models: Dict[str, Any]) -> Dict[str, Any]:
    family = _down3_family_score(non_tie, {
        "big_eye": road_models.get("big_eye", {}),
        "small_road": road_models.get("small_road", {}),
        "cockroach": road_models.get("cockroach", {}),
    })
    return {
        **family,
        "pick": family.get("pick", ""),
        "confidence": family.get("confidence", 0.0),
        "label": family.get("label", "下三路家族資料不足"),
        "candidate": {
            "family_gap": family.get("gap", 0.0),
            "column_once": family.get("column_once", {}),
            "details": family.get("details", {}),
        },
    }

def _fuhao_deep_parity_vote(non_tie: List[str]) -> Dict[str, Any]:
    # 對應富濠 DeepLearningPredictor：莊數總和奇偶。
    b_count = non_tie.count("B")
    pick = "B" if b_count % 2 == 0 else "P"
    return {"pick": pick, "label": f"DeepParity莊數{'偶' if b_count % 2 == 0 else '奇'}偏{_fuhao_side_name(pick)}", "confidence": 0.56, "b_count": b_count}


def _fuhao_length_parity_vote(non_tie: List[str]) -> Dict[str, Any]:
    # 對應富濠 StatisticalPatternRecognizer：有效局數奇偶。
    n = len(non_tie)
    pick = "B" if n % 2 == 0 else "P"
    return {"pick": pick, "label": f"LengthParity有效局{'偶' if n % 2 == 0 else '奇'}偏{_fuhao_side_name(pick)}", "confidence": 0.55, "valid_len": n}


def _fuhao_banker_rate_vote(non_tie: List[str]) -> Dict[str, Any]:
    # 對應富濠 MultiSourceDataFusion：莊率 > 50% 則莊，否則閒。
    n = max(1, len(non_tie))
    b_rate = non_tie.count("B") / n
    pick = "B" if b_rate > 0.5 else "P"
    confidence = 0.52 + min(0.18, abs(b_rate - 0.5) * 0.55)
    return {"pick": pick, "label": f"BankerRate莊率{b_rate*100:.1f}%偏{_fuhao_side_name(pick)}", "confidence": round(confidence, 4), "b_rate": round(b_rate, 4)}


def _fuhao_probs_from_votes(main_pick: str, vote_ratio: float, tie_prob: float) -> Tuple[float, float, float]:
    if main_pick not in {"B", "P"}:
        b_side = 0.5
    else:
        # vote_ratio 0.5~1.0 轉成柔性機率邊際，不讓畫面變成過度誇張的 90%。
        edge = FUHAO_PROB_EDGE + max(0.0, vote_ratio - 0.5) * 2 * (FUHAO_MAX_EDGE - FUHAO_PROB_EDGE)
        edge = _clamp(edge, 0.035, FUHAO_MAX_EDGE)
        b_side = 0.5 + edge if main_pick == "B" else 0.5 - edge
    b_prob = b_side * (1 - tie_prob)
    p_prob = (1 - b_side) * (1 - tie_prob)
    return _normalize_three(b_prob, p_prob, tie_prob)


def _fuhao_parse_ai_side(ai_result: Optional[Dict[str, Any]]) -> str:
    """從 DeepSeek 回傳中解析 AI 偏向。支援 adjust 與文字型欄位兩種格式。"""
    if not isinstance(ai_result, dict) or ai_result.get("error"):
        return ""

    # 若 deepseek_client 回傳明確方向，優先採用。
    for key in ("recommend", "prediction", "pick", "side", "direction"):
        raw = str(ai_result.get(key, "")).strip().upper()
        if raw in {"B", "BANKER", "莊"}:
            return "B"
        if raw in {"P", "PLAYER", "閒", "闲"}:
            return "P"

    # 兼容校準器格式：banker_adjust / player_adjust。
    try:
        ba = float(ai_result.get("banker_adjust", 0) or 0)
        pa = float(ai_result.get("player_adjust", 0) or 0)
    except Exception:
        return ""

    if abs(ba - pa) < 0.00001:
        return ""
    return "B" if ba > pa else "P"


def _fuhao_build_deepseek_payload(
    history: List[str],
    non_tie: List[str],
    venue: str,
    room: str,
    shoe_id: str,
    user_id: str,
    road_models: Dict[str, Any],
    advanced_models: Dict[str, Any],
    road_majority: Dict[str, Any],
    advanced_majority: Dict[str, Any],
    all_majority: Dict[str, Any],
    final_pick: str,
    recommend: str,
    b_prob: float,
    p_prob: float,
    tie_prob: float,
    vote_ratio: float,
    final_vote_count: int,
) -> Dict[str, Any]:
    """建立富濠式 DeepSeek 校準 payload。

    注意：不傳當局點數；只傳 B/P/T、路紙模型、多數決與本次機率，
    讓 DeepSeek 只作為確認層，不取代牌路主模型。
    """
    return {
        "engine": "FUHAO_CLONE",
        "ai_role": "confirmation_layer",
        "instruction": (
            "請只依照已整理好的百家樂 B/P/T 牌路、富濠式多數決、路紙票與Advanced票進行校準。"
            "DeepSeek 只做輔助確認，不要因單一當局點數推翻主模型。"
            "若與主模型反向且理由不強，請降低confidence或回傳小幅adjust。"
        ),
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "history_len": len(history),
        "valid_history_len": len(non_tie),
        "history_tail": "".join(history[-36:]),
        "non_tie_tail": "".join(non_tie[-36:]),
        "fuhao_models": {
            "road": road_models,
            "advanced": advanced_models,
            "road_majority": road_majority,
            "advanced_majority": advanced_majority,
            "all_majority": all_majority,
        },
        "fuhao_decision": {
            "final_pick": final_pick,
            "recommend": recommend,
            "vote_ratio": round(vote_ratio, 4),
            "final_vote_count": final_vote_count,
        },
        "local_probs": {
            "B": round(b_prob, 5),
            "P": round(p_prob, 5),
            "T": round(tie_prob, 5),
        },
        "expected_response_fields": {
            "banker_adjust": "float between -0.035 and 0.035",
            "player_adjust": "float between -0.035 and 0.035",
            "tie_adjust": "float between -0.020 and 0.020",
            "confidence": "float 0~1",
            "reason": "short text",
        },
        "settings": {
            "mode": FUHAO_DEEPSEEK_MODE,
            "observe_on_conflict": FUHAO_DEEPSEEK_OBSERVE_ON_CONFLICT,
            "max_adjust": FUHAO_DEEPSEEK_MAX_ADJUST,
            "tie_max_adjust": FUHAO_DEEPSEEK_TIE_MAX_ADJUST,
        },
    }



def _fuhao_switch_rate(seq: List[str]) -> float:
    if len(seq) < 2:
        return 0.5
    return _safe_div(sum(1 for a, b in zip(seq, seq[1:]) if a != b), len(seq) - 1, 0.5)


def _fuhao_is_alternating(seq: List[str], tolerance: int = 0) -> bool:
    if len(seq) < 4:
        return False
    breaks = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
    return breaks <= tolerance


def _fuhao_previous_streak(non_tie: List[str]) -> Dict[str, Any]:
    """取得目前 streak 前一段連續方向，用於判斷長龍剛斷、單跳剛轉。"""
    last_side, current_n = _streak(non_tie)
    if not last_side or len(non_tie) <= current_n:
        return {"side": "", "count": 0, "current_side": last_side, "current_count": current_n}
    prev_side = non_tie[-current_n - 1]
    prev_n = 0
    idx = len(non_tie) - current_n - 1
    while idx >= 0 and non_tie[idx] == prev_side:
        prev_n += 1
        idx -= 1
    return {"side": prev_side, "count": prev_n, "current_side": last_side, "current_count": current_n}


def _fuhao_classify_regime(non_tie: List[str]) -> Dict[str, Any]:
    """富濠式牌型分類：只提供假規律模型使用，不改主模型票法。"""
    n = len(non_tie)
    short_w = max(4, FUHAO_FAKE_PATTERN_SHORT_WINDOW)
    mid_w = max(short_w + 1, FUHAO_FAKE_PATTERN_MID_WINDOW)
    long_w = max(mid_w + 1, FUHAO_FAKE_PATTERN_LONG_WINDOW)
    short = non_tie[-short_w:]
    mid = non_tie[-mid_w:]
    long = non_tie[-long_w:]
    sw_short = _fuhao_switch_rate(short)
    sw_mid = _fuhao_switch_rate(mid)
    sw_long = _fuhao_switch_rate(long)
    last_side, streak_n = _streak(non_tie)
    tail6 = "".join(non_tie[-6:])
    tail8 = "".join(non_tie[-8:])

    regime = "MIXED"
    if streak_n >= FUHAO_LONG_THRESHOLD:
        regime = "DRAGON"
    elif n >= short_w and sw_short >= FUHAO_FAKE_PATTERN_CHAOS_SWITCH_RATE and _fuhao_is_alternating(short, tolerance=1):
        regime = "PINGPONG"
    elif tail6 in {"BBPPBB", "PPBBPP", "BPPBBP", "PBBPPB"} or tail8 in {"BBPPBBPP", "PPBBPPBB"}:
        regime = "DOUBLE"
    elif FUHAO_FAKE_PATTERN_DENSE_SWITCH_LOW <= sw_mid <= FUHAO_FAKE_PATTERN_DENSE_SWITCH_HIGH:
        regime = "DENSE"
    elif sw_short >= FUHAO_FAKE_PATTERN_CHAOS_SWITCH_RATE and sw_mid < FUHAO_FAKE_PATTERN_CHAOS_SWITCH_RATE:
        regime = "FAKE_PINGPONG"
    elif sw_mid >= FUHAO_FAKE_PATTERN_CHAOS_SWITCH_RATE:
        regime = "CHOPPY"

    return {
        "regime": regime,
        "switch_rate_short": round(sw_short, 4),
        "switch_rate_mid": round(sw_mid, 4),
        "switch_rate_long": round(sw_long, 4),
        "streak_side": last_side,
        "streak_count": streak_n,
        "tail6": tail6,
        "tail8": tail8,
    }


def _fuhao_fake_pattern_detector(
    non_tie: List[str],
    road_models: Dict[str, Any],
    road_majority: Dict[str, Any],
    advanced_majority: Dict[str, Any],
    all_majority: Dict[str, Any],
    final_pick: str,
) -> Dict[str, Any]:
    """假規律 / 轉折保護模型。

    這一版是「裁決層」：
    - 主模型只提供候選方向 final_pick。
    - 假規律/轉折模型可以把候選方向改成觀望。
    - 若轉折已被路紙 + 下三路確認，也可以反向覆蓋主模型。
    - 觀望時會要求機率回到接近 50/50，避免畫面仍偏多數方。
    """
    default = {
        "enabled": False,
        "action": "ALLOW",
        "regime": "DISABLED",
        "fake_score": 0.0,
        "turn_score": 0.0,
        "false_break_score": 0.0,
        "derived_agree": 0,
        "derived_oppose": 0,
        "decision_pick": final_pick if final_pick in {"B", "P"} else "",
        "reverse_side": "",
        "neutralize_probs": False,
        "hard_decision": False,
        "decision_ratio": 0.5,
        "label": "假規律模型關閉",
        "reasons": [],
        "features": {},
    }
    if not FUHAO_USE_FAKE_PATTERN_DETECTOR:
        return default

    n = len(non_tie)
    if n < FUHAO_FAKE_PATTERN_MIN_HISTORY:
        return {
            **default,
            "enabled": True,
            "action": "ALLOW",
            "regime": "WARMUP",
            "label": f"假規律模型暖機中 {n}/{FUHAO_FAKE_PATTERN_MIN_HISTORY}",
            "features": {"valid_len": n},
        }

    regime_info = _fuhao_classify_regime(non_tie)
    regime = regime_info.get("regime", "MIXED")
    final_side = final_pick if final_pick in {"B", "P"} else ""
    opp_side = "P" if final_side == "B" else "B" if final_side == "P" else ""
    reasons: List[str] = []
    fake_score = 0.0
    turn_score = 0.0
    false_break_score = 0.0

    road_pick = road_majority.get("pick", "")
    adv_pick = advanced_majority.get("pick", "")
    vote_ratio = float(all_majority.get("ratio", 0.0) or 0.0)
    road_ratio = float(road_majority.get("ratio", 0.0) or 0.0)

    derived_keys = ["big_eye", "small_road", "cockroach"]
    derived_picks = [str(road_models.get(k, {}).get("pick", "")) for k in derived_keys]
    derived_agree = sum(1 for p in derived_picks if final_side and p == final_side)
    derived_oppose = sum(1 for p in derived_picks if opp_side and p == opp_side)
    derived_valid = sum(1 for p in derived_picks if p in {"B", "P"})

    # 主模型候選方向如果和路紙/下三路不一致，不能只是降信心，要提高裁決風險。
    if final_side:
        if vote_ratio and vote_ratio < FUHAO_FAKE_PATTERN_MIN_VOTE_RATIO:
            add = min(0.18, (FUHAO_FAKE_PATTERN_MIN_VOTE_RATIO - vote_ratio) * 0.70)
            fake_score += add
            reasons.append(f"總票比例{vote_ratio:.2f}偏低")

        if road_pick in {"B", "P"} and road_pick != final_side:
            fake_score += 0.24
            turn_score += 0.16
            reasons.append(f"路紙主方向{_fuhao_side_name(road_pick)}與主模型{_fuhao_side_name(final_side)}不同")

        if adv_pick in {"B", "P"} and road_pick in {"B", "P"} and adv_pick != road_pick:
            fake_score += 0.20
            turn_score += 0.10
            reasons.append("路紙與Advanced互相打架")

        if FUHAO_FAKE_PATTERN_REQUIRE_DERIVED_CONFIRM and derived_valid >= 2 and derived_agree < FUHAO_FAKE_PATTERN_DERIVED_MIN_AGREE:
            fake_score += 0.22
            reasons.append(f"下三路同向只有{derived_agree}票，主方向不夠穩")

        if FUHAO_FAKE_PATTERN_OBSERVE_ON_DERIVED_CONFLICT and derived_oppose >= FUHAO_FAKE_PATTERN_DERIVED_MIN_AGREE:
            fake_score += 0.26
            turn_score += 0.18
            reasons.append(f"下三路有{derived_oppose}票反向")

    prev = _fuhao_previous_streak(non_tie)
    prev_count = int(prev.get("count", 0))
    current_count = int(prev.get("current_count", 0))
    current_side = str(prev.get("current_side", ""))
    prev_side = str(prev.get("side", ""))
    short_sw = float(regime_info.get("switch_rate_short", 0.5))
    mid_sw = float(regime_info.get("switch_rate_mid", 0.5))
    long_sw = float(regime_info.get("switch_rate_long", 0.5))

    # 長龍剛斷 1~2 口：這是最容易被主模型誤導的地方，優先觀望，不急著反打。
    if prev_count >= FUHAO_FAKE_PATTERN_FALSE_BREAK_MIN_STREAK and 1 <= current_count <= FUHAO_FAKE_PATTERN_FALSE_BREAK_CONFIRM_ROUNDS:
        false_break_score += 0.52 + min(0.24, (prev_count - FUHAO_FAKE_PATTERN_FALSE_BREAK_MIN_STREAK) * 0.05)
        turn_score += 0.22
        fake_score += 0.18
        reasons.append(f"{_fuhao_side_name(prev_side)}長龍{prev_count}口剛被{_fuhao_side_name(current_side)}斷{current_count}口，疑似假斷")

    # 單跳路單剛破：第一段兩口不能馬上當成真正轉雙跳。
    if len(non_tie) >= 8:
        before_last = non_tie[-8:-2]
        if _fuhao_is_alternating(before_last, tolerance=1) and current_count == 2:
            turn_score += 0.40
            fake_score += 0.22
            reasons.append("單跳路剛破成兩口，轉折未確認")
        elif _fuhao_is_alternating(non_tie[-8:], tolerance=1) and short_sw >= FUHAO_FAKE_PATTERN_CHAOS_SWITCH_RATE and mid_sw < FUHAO_FAKE_PATTERN_CHAOS_SWITCH_RATE:
            fake_score += 0.20
            reasons.append("短線像單跳，但中週期不穩")

    # 密集盤 / 散盤中出現短暫規律，容易是假規律。
    if regime in {"DENSE", "CHOPPY", "FAKE_PINGPONG"}:
        add = {"DENSE": 0.17, "CHOPPY": 0.20, "FAKE_PINGPONG": 0.24}.get(regime, 0.14)
        fake_score += add
        reasons.append(f"{regime}盤中短暫規律風險")
    if short_sw >= FUHAO_FAKE_PATTERN_CHAOS_SWITCH_RATE and FUHAO_FAKE_PATTERN_DENSE_SWITCH_LOW <= mid_sw <= FUHAO_FAKE_PATTERN_DENSE_SWITCH_HIGH:
        fake_score += 0.18
        reasons.append("短線很跳但中線混合，容易假規律")

    # 多數票過度偏向但下三路不支援，視為一直押多數方的高風險。
    if final_side and vote_ratio >= 0.70 and derived_valid >= 2 and derived_agree < FUHAO_FAKE_PATTERN_DERIVED_MIN_AGREE:
        fake_score += 0.22
        reasons.append("總票偏一邊，但下三路未確認，避免一直押多數方")

    # 大路與下三路共同反向時，代表可能不是假反彈，而是轉折已確認。
    reverse_side = ""
    reverse_score = 0.0
    if final_side and opp_side:
        road_support_reverse = road_pick == opp_side
        derived_support_reverse = derived_oppose >= FUHAO_FAKE_PATTERN_REVERSE_DERIVED_MIN
        ratio_support_reverse = max(road_ratio, derived_oppose / max(1, derived_valid)) >= FUHAO_FAKE_PATTERN_REVERSE_MIN_RATIO
        not_early_false_break = false_break_score < FUHAO_FAKE_PATTERN_FALSE_BREAK_SCORE

        if road_support_reverse:
            reverse_score += 0.34
        if derived_support_reverse:
            reverse_score += 0.34
        if ratio_support_reverse:
            reverse_score += 0.16
        if turn_score >= FUHAO_FAKE_PATTERN_TURN_SCORE:
            reverse_score += 0.16
        if not not_early_false_break:
            reverse_score *= 0.45

        if (
            FUHAO_FAKE_PATTERN_ALLOW_REVERSE
            and opp_side
            and reverse_score >= FUHAO_FAKE_PATTERN_REVERSE_SCORE
            and derived_support_reverse
            and (not FUHAO_FAKE_PATTERN_REVERSE_REQUIRE_ROAD or road_support_reverse)
            and not_early_false_break
        ):
            reverse_side = opp_side
            reasons.append(f"路紙與下三路共同確認轉折，裁決反向{_fuhao_side_name(reverse_side)}")

    fake_score = _clamp(fake_score, 0.0, 1.0)
    turn_score = _clamp(turn_score, 0.0, 1.0)
    false_break_score = _clamp(false_break_score, 0.0, 1.0)
    reverse_score = _clamp(reverse_score, 0.0, 1.0)

    action = "ALLOW"
    decision_pick = final_side
    neutralize_probs = False
    hard_decision = False

    if FUHAO_FAKE_PATTERN_OBSERVE_ON_FALSE_BREAK and false_break_score >= FUHAO_FAKE_PATTERN_FALSE_BREAK_SCORE:
        action = "OBSERVE_FALSE_BREAK"
        decision_pick = ""
        neutralize_probs = FUHAO_FAKE_PATTERN_NEUTRALIZE_ON_OBSERVE
        hard_decision = FUHAO_FAKE_PATTERN_HARD_DECISION
    elif reverse_side:
        action = "REVERSE_TURN"
        decision_pick = reverse_side
        neutralize_probs = False
        hard_decision = FUHAO_FAKE_PATTERN_HARD_DECISION
    elif FUHAO_FAKE_PATTERN_OBSERVE_ON_TURN and turn_score >= FUHAO_FAKE_PATTERN_TURN_SCORE:
        action = "OBSERVE_TURN"
        decision_pick = ""
        neutralize_probs = FUHAO_FAKE_PATTERN_NEUTRALIZE_ON_OBSERVE
        hard_decision = FUHAO_FAKE_PATTERN_HARD_DECISION
    elif FUHAO_FAKE_PATTERN_OBSERVE_ON_SCORE and fake_score >= FUHAO_FAKE_PATTERN_HARD_OBSERVE_SCORE:
        action = "OBSERVE_HARD_FAKE"
        decision_pick = ""
        neutralize_probs = FUHAO_FAKE_PATTERN_NEUTRALIZE_ON_OBSERVE
        hard_decision = FUHAO_FAKE_PATTERN_HARD_DECISION
    elif FUHAO_FAKE_PATTERN_OBSERVE_ON_SCORE and fake_score >= FUHAO_FAKE_PATTERN_OBSERVE_SCORE:
        action = "OBSERVE_FAKE"
        decision_pick = ""
        neutralize_probs = FUHAO_FAKE_PATTERN_NEUTRALIZE_ON_OBSERVE
        hard_decision = FUHAO_FAKE_PATTERN_HARD_DECISION
    elif fake_score >= FUHAO_FAKE_PATTERN_SHRINK_SCORE:
        action = "SHRINK"
        decision_pick = final_side
        neutralize_probs = False
        hard_decision = False

    if not FUHAO_FAKE_PATTERN_HARD_DECISION and action != "SHRINK":
        # 關閉硬裁決時，只退回原本行為：不蓋掉 final_pick，只給標籤/降信心。
        decision_pick = final_side
        neutralize_probs = False
        hard_decision = False

    if action == "ALLOW":
        label = f"假規律檢測通過:{regime} F{int(fake_score*100)} T{int(turn_score*100)} FB{int(false_break_score*100)}"
    elif action == "SHRINK":
        label = f"假規律偏弱降信心:{regime} F{int(fake_score*100)}"
    elif action == "REVERSE_TURN":
        label = f"假規律裁決反向:{regime} → {_fuhao_side_name(decision_pick)} R{int(reverse_score*100)}"
    else:
        label = f"假規律/轉折裁決觀望:{regime} F{int(fake_score*100)} T{int(turn_score*100)} FB{int(false_break_score*100)}"

    decision_ratio = 0.5
    if decision_pick in {"B", "P"}:
        if decision_pick == final_side:
            decision_ratio = max(0.5, vote_ratio)
        elif decision_pick == reverse_side:
            decision_ratio = max(0.5, min(0.78, max(road_ratio, derived_oppose / max(1, derived_valid))))
    if neutralize_probs:
        decision_ratio = 0.5

    return {
        "enabled": True,
        "action": action,
        "regime": regime,
        "fake_score": round(fake_score, 4),
        "turn_score": round(turn_score, 4),
        "false_break_score": round(false_break_score, 4),
        "reverse_score": round(reverse_score, 4),
        "derived_agree": derived_agree,
        "derived_oppose": derived_oppose,
        "decision_pick": decision_pick,
        "reverse_side": reverse_side,
        "neutralize_probs": bool(neutralize_probs),
        "hard_decision": bool(hard_decision),
        "decision_ratio": round(decision_ratio, 4),
        "label": label,
        "reasons": reasons,
        "features": {
            **regime_info,
            "vote_ratio": round(vote_ratio, 4),
            "road_ratio": round(road_ratio, 4),
            "road_pick": road_pick,
            "advanced_pick": adv_pick,
            "final_pick": final_side,
            "decision_pick": decision_pick,
            "derived_picks": derived_picks,
            "derived_valid": derived_valid,
            "previous_streak_side": prev_side,
            "previous_streak_count": prev_count,
            "current_streak_side": current_side,
            "current_streak_count": current_count,
            "reverse_score": round(reverse_score, 4),
            "short_switch_rate": round(short_sw, 4),
            "mid_switch_rate": round(mid_sw, 4),
            "long_switch_rate": round(long_sw, 4),
        },
    }



def _fuhao_apply_fake_pattern_shrink(b_prob: float, p_prob: float, tie_prob: float, detector: Dict[str, Any]) -> Tuple[float, float, float]:
    """假規律裁決後的機率校正。

    重點：
    - 如果裁決觀望，莊/閒機率直接回到接近 50/50。
    - 如果只是 SHRINK，才用原本降信心方式。
    - 如果裁決反向，機率已在前一步用 decision_pick 重算，這裡不再拉回。
    """
    if not detector.get("enabled"):
        return b_prob, p_prob, tie_prob

    action = str(detector.get("action", "ALLOW"))
    if action.startswith("OBSERVE") and detector.get("neutralize_probs") and FUHAO_FAKE_PATTERN_OBSERVE_RESETS_EDGE:
        b_prob = 0.5 * (1 - tie_prob)
        p_prob = 0.5 * (1 - tie_prob)
        return _normalize_three(b_prob, p_prob, tie_prob)

    if action in {"ALLOW", "REVERSE_TURN"}:
        return b_prob, p_prob, tie_prob

    score = float(detector.get("fake_score", 0.0) or 0.0)
    if score < FUHAO_FAKE_PATTERN_SHRINK_SCORE:
        return b_prob, p_prob, tie_prob

    side_total = max(0.0001, b_prob + p_prob)
    b_side = b_prob / side_total
    factor = _clamp(
        1.0 - FUHAO_FAKE_PATTERN_CONF_SHRINK * max(
            score,
            float(detector.get("turn_score", 0.0) or 0.0),
            float(detector.get("false_break_score", 0.0) or 0.0),
        ),
        0.25,
        0.92,
    )
    b_side = 0.5 + (b_side - 0.5) * factor
    b_prob = b_side * (1 - tie_prob)
    p_prob = (1 - b_side) * (1 - tie_prob)
    return _normalize_three(b_prob, p_prob, tie_prob)



def _fuhao_clone_predict(history: List[str], venue: str = "", room: str = "", shoe_id: str = "", user_id: str = "") -> Dict[str, Any]:
    """富濠式保守牌路多數決模型。

    特性：
    - 和局保留統計，但預測主方向只用 B/P。
    - 預設只看最近 100 手有效牌，避免太舊資料污染。
    - 大路 + 下三路提供路紙票；三個 Advanced 規則做最終多數決。
    - 預設要求路紙方向與 Advanced 方向一致；不同就觀望。
    """
    history = [str(x).upper() for x in history if str(x).upper() in {"B", "P", "T"}]
    if FUHAO_IGNORE_TIE_FOR_PREDICT:
        non_tie_all = _last_non_tie(history)
    else:
        non_tie_all = [x for x in history if x in {"B", "P"}]
    non_tie = non_tie_all[-max(1, FUHAO_HISTORY_LIMIT):]

    tie_count = history.count("T") if FUHAO_KEEP_TIE_COUNT else 0
    valid_len = len(non_tie)
    training_key = f"{user_id or 'anonymous'}|FUHAO_CLONE|{venue}|{room}|{shoe_id}"
    _update_ask_road_truth(training_key, non_tie)
    ask_road_performance = _get_ask_road_performance(training_key)
    recommend_text_map = {"B": "莊", "P": "閒", "T": "和", "NONE": "觀望"}

    # 冷啟動：有效 B/P 太少時只回基準，避免亂給方向。
    if valid_len < FUHAO_MIN_VALID_ROUNDS:
        tie_prob = _clamp(FUHAO_TIE_BASE, 0.001, TIE_MAX_PROB)
        b_prob, p_prob, tie_prob = _normalize_three(0.5 * (1 - tie_prob), 0.5 * (1 - tie_prob), tie_prob)
        observe_reason = f"有效牌局{valid_len}局，未達富濠式最小樣本{FUHAO_MIN_VALID_ROUNDS}局"
        return {
            "ok": True,
            "engine": "FUHAO_CLONE",
            "user_id": user_id,
            "venue": venue,
            "room": room,
            "shoe_id": shoe_id,
            "round_no": len(history) + 1,
            "history_len": len(history),
            "valid_history_len": valid_len,
            "tie_count": tie_count,
            "banker_rate": round(b_prob * 100, 1),
            "player_rate": round(p_prob * 100, 1),
            "tie_rate": round(tie_prob * 100, 1),
            "recommend": "NONE",
            "recommend_text": "觀望",
            "is_observe": True,
            "observe_reason": observe_reason,
            "decision_edge": 0.0,
            "confidence": 0.25,
            "signal_level": "資料不足",
            "pattern_label": "富濠式冷啟動",
            "regime": "fuhao_cold",
            "reason": observe_reason,
            "ai_used": False,
            "ml_trained": False,
            "ml_samples": 0,
            "tf_available": TF_AVAILABLE,
            "training_key": training_key,
            "ask_road_memory": ask_road_performance,
            "ask_road_memory_label": ask_road_performance.get("label", ""),
            "ask_road_memory_enabled": USE_ASK_ROAD_MEMORY,
            "model_cache_size": len(_MODEL_CACHE),
            "ml_predictions": None,
            "ai_result": None,
            "debug": None,
        }

    # 1) 路紙明細：大路 + 三條下路；對外票只保留「大路」與「下三路家族」。
    dense_board = _detect_dense_board(non_tie)
    ask_road_performance = _limit_ask_road_performance_for_dense(ask_road_performance, dense_board)
    road_models: Dict[str, Any] = {}

    if FUHAO_USE_BIG_ROAD:
        road_models["big_road"] = _fuhao_big_road_vote(non_tie)
    else:
        road_models["big_road"] = {"pick": "", "label": "大路關閉", "confidence": 0.0}

    if FUHAO_USE_BIG_EYE:
        road_models["big_eye"] = _fuhao_down3_vote(non_tie, 1, "大眼仔")
        road_models["big_eye"] = _apply_ask_road_factor_to_vote(road_models["big_eye"], "big_eye", ask_road_performance)
    else:
        road_models["big_eye"] = {"pick": "", "label": "大眼仔關閉", "confidence": 0.0, "B": 0.5, "P": 0.5}

    if FUHAO_USE_SMALL_ROAD:
        road_models["small_road"] = _fuhao_down3_vote(non_tie, 2, "小路")
        road_models["small_road"] = _apply_ask_road_factor_to_vote(road_models["small_road"], "small_road", ask_road_performance)
    else:
        road_models["small_road"] = {"pick": "", "label": "小路關閉", "confidence": 0.0, "B": 0.5, "P": 0.5}

    if FUHAO_USE_COCKROACH:
        road_models["cockroach"] = _fuhao_down3_vote(non_tie, 3, "蟑螂路")
        road_models["cockroach"] = _apply_ask_road_factor_to_vote(road_models["cockroach"], "cockroach", ask_road_performance)
    else:
        road_models["cockroach"] = {"pick": "", "label": "蟑螂路關閉", "confidence": 0.0, "B": 0.5, "P": 0.5}

    road_models["down3_family"] = _fuhao_down3_family_vote(non_tie, road_models)
    road_models["down3_family"] = _apply_ask_road_factor_to_vote(
        road_models["down3_family"], "down3_family", ask_road_performance
    )

    big_road_pick = road_models.get("big_road", {}).get("pick", "")
    down3_family_pick = road_models.get("down3_family", {}).get("pick", "")
    road_votes = [v for v in [big_road_pick, down3_family_pick] if v in {"B", "P"}]
    road_tiebreak_pick = "" if FUHAO_DISABLE_BIGROAD_FALLBACK else big_road_pick
    road_majority = _fuhao_majority(road_votes, big_road_pick=road_tiebreak_pick)

    # 2) Advanced 規則保留為「一個家族多數票」，避免三個規則各自灌票。
    advanced_votes = []
    advanced_models: Dict[str, Any] = {}
    if FUHAO_USE_DEEP_PARITY:
        advanced_models["deep_parity"] = _fuhao_deep_parity_vote(non_tie)
        advanced_votes.append(advanced_models["deep_parity"].get("pick", ""))
    else:
        advanced_models["deep_parity"] = {"pick": "", "label": "DeepParity關閉", "confidence": 0.0}
    if FUHAO_USE_LENGTH_PARITY:
        advanced_models["length_parity"] = _fuhao_length_parity_vote(non_tie)
        advanced_votes.append(advanced_models["length_parity"].get("pick", ""))
    else:
        advanced_models["length_parity"] = {"pick": "", "label": "LengthParity關閉", "confidence": 0.0}
    if FUHAO_USE_BANKER_RATE:
        advanced_models["banker_rate"] = _fuhao_banker_rate_vote(non_tie)
        advanced_votes.append(advanced_models["banker_rate"].get("pick", ""))
    else:
        advanced_models["banker_rate"] = {"pick": "", "label": "BankerRate關閉", "confidence": 0.0}
    advanced_majority = _fuhao_majority(advanced_votes, big_road_pick="")
    advanced_pick = advanced_majority.get("pick", "")

    # Pattern Replay 是獨立歷史回放確認，不把下三路家族直接當 final。
    pattern_replay = _pattern_replay_memory_score(non_tie, training_key=training_key, live_performance=None)
    pattern_pick = ""
    if (
        pattern_replay.get("state") == "REPLAY_MATCH"
        and float(pattern_replay.get("confidence", 0.0) or 0.0) >= FINAL_CONFIRM_PATTERN_CONF
        and float(pattern_replay.get("edge", 0.0) or 0.0) >= FINAL_CONFIRM_PATTERN_EDGE
    ):
        pattern_pick = str(pattern_replay.get("bias_side", ""))

    # 3) 最後整合：下三路家族只提供候選，必須取得大路 / Pattern Replay / Advanced 之一確認。
    candidate_pick = down3_family_pick if down3_family_pick in {"B", "P"} else ""
    confirm_sources: Dict[str, str] = {}
    if candidate_pick:
        if big_road_pick == candidate_pick:
            confirm_sources["big_road"] = big_road_pick
        if pattern_pick == candidate_pick:
            confirm_sources["pattern_replay"] = pattern_pick
        if advanced_pick == candidate_pick:
            confirm_sources["advanced"] = advanced_pick

    dense_conflict = bool(
        dense_board.get("is_dense")
        and candidate_pick in {"B", "P"}
        and big_road_pick in {"B", "P"}
        and candidate_pick != big_road_pick
    )
    non_road_confirm = sum(1 for k in confirm_sources if k != "big_road")

    if candidate_pick and len(confirm_sources) >= max(1, FINAL_CONFIRM_MIN_SOURCES):
        if dense_conflict and DENSE_CONFLICT_REQUIRE_NON_ROAD_CONFIRM and non_road_confirm < 1:
            final_pick = ""
            final_source = "dense_conflict_unconfirmed"
        else:
            final_pick = candidate_pick
            final_source = "down3_family+" + "+".join(sorted(confirm_sources.keys()))
    elif not candidate_pick and big_road_pick in {"B", "P"} and pattern_pick == big_road_pick:
        final_pick = big_road_pick
        final_source = "big_road+pattern_replay"
        confirm_sources = {"big_road": big_road_pick, "pattern_replay": pattern_pick}
    elif not candidate_pick and advanced_pick in {"B", "P"} and pattern_pick == advanced_pick:
        final_pick = advanced_pick
        final_source = "advanced+pattern_replay"
        confirm_sources = {"advanced": advanced_pick, "pattern_replay": pattern_pick}
    else:
        final_pick = ""
        final_source = "no_independent_confirmation"

    # 外部票最多四個家族來源：大路、下三路家族、Pattern Replay、Advanced 家族。
    all_votes = [v for v in [big_road_pick, down3_family_pick, pattern_pick, advanced_pick] if v in {"B", "P"}]
    all_majority = _fuhao_majority(all_votes, big_road_pick=road_tiebreak_pick)
    vote_ratio = float(all_majority.get("ratio", 0.0)) if all_majority.get("total", 0) else 0.0
    final_vote_count = all_votes.count(final_pick) if final_pick in {"B", "P"} else 0
    derived_votes = [down3_family_pick] if down3_family_pick in {"B", "P"} else []
    derived_majority = {"pick": down3_family_pick, "B": 1 if down3_family_pick == "B" else 0, "P": 1 if down3_family_pick == "P" else 0, "total": len(derived_votes), "ratio": 1.0 if derived_votes else 0.0, "tie": False}
    derived_confirm_count = 1 if final_pick and down3_family_pick == final_pick else 0
    non_bigroad_confirm_count = sum(1 for k, v in confirm_sources.items() if k != "big_road" and v == final_pick)
    derived_ratio = float(road_models.get("down3_family", {}).get("agreement_ratio", 0.0) or 0.0)

    observe_reason = ""
    recommend = final_pick if final_pick in {"B", "P"} else "NONE"
    if recommend == "NONE":
        if dense_conflict:
            observe_reason = "密集盤中下三路家族與大路衝突，未取得 Pattern Replay/獨立模型確認"
        elif candidate_pick and not confirm_sources:
            observe_reason = "下三路家族只有候選方向，沒有大路、Pattern Replay或獨立模型確認"
        elif candidate_pick and len(confirm_sources) < max(1, FINAL_CONFIRM_MIN_SOURCES):
            observe_reason = f"候選僅取得{len(confirm_sources)}個獨立確認，未達{FINAL_CONFIRM_MIN_SOURCES}"
        else:
            observe_reason = "沒有形成可被獨立來源確認的最終方向"
    elif final_vote_count < 2:
        recommend = "NONE"
        observe_reason = "最終方向未形成至少兩個家族來源同向"

    if (
        recommend in {"B", "P"}
        and FUHAO_REQUIRE_ROAD_AND_ADVANCED_SAME
        and advanced_pick in {"B", "P"}
        and advanced_pick != recommend
        and FUHAO_OBSERVE_ON_CONFLICT
    ):
        recommend = "NONE"
        observe_reason = f"最終方向{_fuhao_side_name(final_pick)}與Advanced{_fuhao_side_name(advanced_pick)}衝突"

    # 3.5) 假規律 / 轉折保護模型：修正一直押多數方、長龍假斷、路單轉折慢。
    fake_pattern = _fuhao_fake_pattern_detector(
        non_tie=non_tie,
        road_models=road_models,
        road_majority=road_majority,
        advanced_majority=advanced_majority,
        all_majority=all_majority,
        final_pick=final_pick,
    )
    fake_action = str(fake_pattern.get("action", "ALLOW"))

    # 3.6) 假規律裁決層：主模型 final_pick 只當候選，真正輸出改用 decision_pick。
    decision_pick = final_pick
    decision_source = final_source
    decision_vote_ratio = max(0.5, vote_ratio)
    if fake_pattern.get("enabled") and FUHAO_FAKE_PATTERN_HARD_DECISION:
        guarded_pick = str(fake_pattern.get("decision_pick", "") or "")
        guarded_ratio = float(fake_pattern.get("decision_ratio", 0.5) or 0.5)

        if fake_action.startswith("OBSERVE"):
            decision_pick = ""
            decision_source = f"{final_source}+fake_pattern_observe"
            decision_vote_ratio = 0.5
            recommend = "NONE"
            fake_reason = fake_pattern.get("label", "假規律/轉折風險")
            fake_detail = "、".join(fake_pattern.get("reasons", [])[:3])
            fake_reason = f"{fake_reason}：{fake_detail}" if fake_detail else fake_reason
            observe_reason = f"{observe_reason}；{fake_reason}" if observe_reason else fake_reason
        elif fake_action == "REVERSE_TURN" and guarded_pick in {"B", "P"}:
            decision_pick = guarded_pick
            decision_source = f"{final_source}+fake_pattern_reverse"
            decision_vote_ratio = max(0.5, guarded_ratio)
            recommend = decision_pick
            observe_reason = ""
        else:
            decision_pick = guarded_pick if guarded_pick in {"B", "P"} else final_pick
            decision_vote_ratio = max(0.5, guarded_ratio if guarded_pick else vote_ratio)
    elif fake_action.startswith("OBSERVE"):
        recommend = "NONE"
        fake_reason = fake_pattern.get("label", "假規律/轉折風險")
        fake_detail = "、".join(fake_pattern.get("reasons", [])[:3])
        fake_reason = f"{fake_reason}：{fake_detail}" if fake_detail else fake_reason
        observe_reason = f"{observe_reason}；{fake_reason}" if observe_reason else fake_reason

    # 如果裁決後沒有方向，後續機率不能再用原 final_pick，避免畫面仍偏多數方。
    prob_pick = decision_pick if (recommend in {"B", "P"} and decision_pick in {"B", "P"}) else ""

    # 4) 機率與信心：主體改用「裁決後方向」；觀望時回到接近 50/50。
    # tie 不參與主方向；用固定基準 + 近期和局做柔性保留。
    recent_tail = history[-max(12, min(36, len(history))):] if history else []
    recent_tie_rate = recent_tail.count("T") / max(1, len(recent_tail)) if recent_tail else 0.0
    tie_prob = _clamp(FUHAO_TIE_BASE * (1 - FUHAO_TIE_SHRINK) + recent_tie_rate * FUHAO_TIE_SHRINK, 0.001, TIE_MAX_PROB)
    b_prob, p_prob, tie_prob = _fuhao_probs_from_votes(prob_pick if prob_pick in {"B", "P"} else "", max(0.5, decision_vote_ratio), tie_prob)
    b_prob, p_prob, tie_prob = _fuhao_apply_fake_pattern_shrink(b_prob, p_prob, tie_prob, fake_pattern)
    edge = abs(b_prob - p_prob)

    agreement = all_votes.count(decision_pick) / max(1, len(all_votes)) if decision_pick in {"B", "P"} else 0.0
    conf, level = _confidence(b_prob, p_prob, tie_prob, len(history), agreement, 0.0)
    if fake_pattern.get("enabled") and fake_pattern.get("action") != "ALLOW":
        shrink_score = max(
            float(fake_pattern.get("fake_score", 0.0) or 0.0),
            float(fake_pattern.get("turn_score", 0.0) or 0.0),
            float(fake_pattern.get("false_break_score", 0.0) or 0.0),
        )
        conf = max(0.18, conf * (1.0 - FUHAO_FAKE_PATTERN_CONF_SHRINK * min(0.85, shrink_score)))
        if fake_pattern.get("action") == "SHRINK" and level != "觀望":
            level = "假規律降權"
    # 若觀望，信心顯示降一點，避免前端誤以為觀望也高信心。
    if recommend == "NONE":
        conf = min(conf, 0.48)
        level = "觀望"

    # 5) DeepSeek 輔助確認層：只做方向確認與小幅校準，不取代富濠式主模型。
    ai_result = None
    ai_used = False
    ai_side = ""
    ai_status = "disabled"
    ai_conf = 0.0
    ai_payload = None

    skip_deepseek_by_fake_guard = bool(
        recommend == "NONE"
        and fake_pattern.get("enabled")
        and str(fake_pattern.get("action", "")).startswith("OBSERVE")
        and bool(fake_pattern.get("neutralize_probs"))
    )

    if (
        USE_DEEPSEEK
        and FUHAO_USE_DEEPSEEK
        and FUHAO_DEEPSEEK_WEIGHT > 0
        and len(history) >= FUHAO_DEEPSEEK_MIN_HISTORY
        and not skip_deepseek_by_fake_guard
    ):
        ai_payload = _fuhao_build_deepseek_payload(
            history=history,
            non_tie=non_tie,
            venue=venue,
            room=room,
            shoe_id=shoe_id,
            user_id=user_id,
            road_models=road_models,
            advanced_models=advanced_models,
            road_majority=road_majority,
            advanced_majority=advanced_majority,
            all_majority=all_majority,
            final_pick=final_pick,
            recommend=recommend,
            b_prob=b_prob,
            p_prob=p_prob,
            tie_prob=tie_prob,
            vote_ratio=vote_ratio,
            final_vote_count=final_vote_count,
        )
        try:
            ai_result = DeepSeekClient().calibrate(ai_payload)
        except Exception as e:
            ai_result = {"error": True, "message": str(e)}

        if ai_result and not ai_result.get("error"):
            try:
                ai_used = True
                ai_status = "used"
                ai_conf = _clamp(float(ai_result.get("confidence", 0.4) or 0.4), 0.0, 1.0)
                ai_side = _fuhao_parse_ai_side(ai_result)

                local_side = recommend if recommend in {"B", "P"} else ""
                conflict = bool(ai_side and local_side and ai_side != local_side and ai_conf >= FUHAO_DEEPSEEK_MIN_CONFIDENCE)
                same_side = bool(ai_side and local_side and ai_side == local_side and ai_conf >= FUHAO_DEEPSEEK_MIN_CONFIDENCE)

                if conflict and FUHAO_DEEPSEEK_MODE == "CONFIRM" and FUHAO_DEEPSEEK_OBSERVE_ON_CONFLICT:
                    recommend = "NONE"
                    ai_status = "conflict_observe"
                    extra_reason = f"DeepSeek{_fuhao_side_name(ai_side)}與富濠主模型{_fuhao_side_name(local_side)}衝突"
                    observe_reason = f"{observe_reason}；{extra_reason}" if observe_reason else extra_reason
                    conf = min(conf, max(0.25, conf - FUHAO_DEEPSEEK_CONFIDENCE_SHRINK * ai_conf))
                    level = "觀望"
                else:
                    ba = _clamp(float(ai_result.get("banker_adjust", 0) or 0), -FUHAO_DEEPSEEK_MAX_ADJUST, FUHAO_DEEPSEEK_MAX_ADJUST)
                    pa = _clamp(float(ai_result.get("player_adjust", 0) or 0), -FUHAO_DEEPSEEK_MAX_ADJUST, FUHAO_DEEPSEEK_MAX_ADJUST)
                    ta = _clamp(float(ai_result.get("tie_adjust", 0) or 0), -FUHAO_DEEPSEEK_TIE_MAX_ADJUST, FUHAO_DEEPSEEK_TIE_MAX_ADJUST)
                    blend = FUHAO_DEEPSEEK_WEIGHT * (0.45 + ai_conf * 0.55)

                    # CONFIRM 模式下，同向才明顯加權；無明確方向時只允許很小的概率校準。
                    if FUHAO_DEEPSEEK_MODE == "CONFIRM" and not same_side:
                        blend *= 0.35

                    b_prob += ba * blend
                    p_prob += pa * blend
                    tie_prob += ta * blend
                    b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)
                    edge = abs(b_prob - p_prob)

                    if same_side:
                        conf = min(0.94, conf + FUHAO_DEEPSEEK_CONFIDENCE_BOOST * ai_conf)
                        if conf >= 0.68:
                            level = "強訊號"
                        elif conf >= 0.48 and level != "觀望":
                            level = "中訊號"
                        ai_status = "same_side_confirm"
            except Exception as e:
                ai_status = "parse_error"
                ai_result = {"error": True, "message": str(e), "raw": ai_result}
    elif USE_DEEPSEEK and FUHAO_USE_DEEPSEEK and skip_deepseek_by_fake_guard:
        ai_status = "skipped_fake_pattern_observe"
    elif USE_DEEPSEEK and FUHAO_USE_DEEPSEEK and len(history) < FUHAO_DEEPSEEK_MIN_HISTORY:
        ai_status = "not_enough_history"

    road_consensus_label = f"大路+下三路家族:{_fuhao_side_name(road_majority.get('pick', ''))} B{road_majority.get('B', 0)} / P{road_majority.get('P', 0)}"
    advanced_label = f"富濠Advanced多數決:{_fuhao_side_name(advanced_majority.get('pick', ''))} B{advanced_majority.get('B', 0)} / P{advanced_majority.get('P', 0)}"

    reason_parts = [
        road_consensus_label,
        advanced_label,
        f"問路記憶:{ask_road_performance.get('label', '')}",
        road_models.get("big_road", {}).get("label", ""),
        road_models.get("big_eye", {}).get("label", ""),
        road_models.get("small_road", {}).get("label", ""),
        road_models.get("cockroach", {}).get("label", ""),
        road_models.get("down3_family", {}).get("label", ""),
        dense_board.get("label", ""),
        pattern_replay.get("label", ""),
        advanced_models.get("deep_parity", {}).get("label", ""),
        advanced_models.get("length_parity", {}).get("label", ""),
        advanced_models.get("banker_rate", {}).get("label", ""),
        fake_pattern.get("label", ""),
        f"候選方向{_fuhao_side_name(final_pick)} 票數:{final_vote_count}/{len(all_votes)}",
        f"裁決方向:{_fuhao_side_name(decision_pick) if decision_pick in {'B', 'P'} else '觀望'} 來源:{decision_source}",
    ]
    if observe_reason:
        reason_parts.append(f"觀望:{observe_reason}")
    if ai_result is not None:
        ai_reason = ""
        if isinstance(ai_result, dict):
            ai_reason = str(ai_result.get("reason") or ai_result.get("message") or "")
        reason_parts.append(f"DeepSeek:{ai_status} 偏{_fuhao_side_name(ai_side)} 信心{ai_conf:.2f}{(' ' + ai_reason) if ai_reason else ''}")

    dynamic_weights = {
        "fuhao_big_road": 1.0 if FUHAO_USE_BIG_ROAD else 0.0,
        "fuhao_big_eye": 1.0 if FUHAO_USE_BIG_EYE else 0.0,
        "fuhao_small_road": 1.0 if FUHAO_USE_SMALL_ROAD else 0.0,
        "fuhao_cockroach": 0.0,
        "fuhao_down3_family": 1.0 if USE_DOWN3_FAMILY else 0.0,
        "fuhao_deep_parity": 1.0 if FUHAO_USE_DEEP_PARITY else 0.0,
        "fuhao_length_parity": 1.0 if FUHAO_USE_LENGTH_PARITY else 0.0,
        "fuhao_banker_rate": 1.0 if FUHAO_USE_BANKER_RATE else 0.0,
        "fuhao_deepseek_confirm": FUHAO_DEEPSEEK_WEIGHT if (USE_DEEPSEEK and FUHAO_USE_DEEPSEEK) else 0.0,
        "fuhao_fake_pattern_detector": 1.0 if FUHAO_USE_FAKE_PATTERN_DETECTOR else 0.0,
    }

    # 舊欄位相容：前端或 app.py 若讀舊 key 不會爆。
    empty_state = {"enabled": False, "state": "FUHAO_CLONE", "label": "富濠式模型不使用此層", "confidence": 0.0}
    # training_key 已於函數前段建立，供 Ask Road Memory 使用。

    ask_road_pending = {
        "big_road": road_models.get("big_road", {}).get("pick", ""),
        "big_eye": road_models.get("big_eye", {}).get("pick", ""),
        "small_road": road_models.get("small_road", {}).get("pick", ""),
        "cockroach": road_models.get("cockroach", {}).get("pick", ""),
        "down3_family": road_models.get("down3_family", {}).get("pick", ""),
        "road_majority": road_majority.get("pick", ""),
        "final": recommend if recommend in {"B", "P"} else "",
    }
    _store_ask_road_pending(training_key, non_tie, ask_road_pending)

    result = {
        "ok": True,
        "engine": "FUHAO_CLONE",
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "round_no": len(history) + 1,
        "history_len": len(history),
        "valid_history_len": valid_len,
        "tie_count": tie_count,
        "banker_rate": round(b_prob * 100, 1),
        "player_rate": round(p_prob * 100, 1),
        "tie_rate": round(tie_prob * 100, 1),
        "recommend": recommend,
        "recommend_text": recommend_text_map.get(recommend, "觀望"),
        "candidate_pick": final_pick,
        "decision_pick": decision_pick if decision_pick in {"B", "P"} else "",
        "decision_source": decision_source,
        "is_observe": recommend == "NONE",
        "observe_reason": observe_reason,
        "decision_edge": round(edge, 5),
        "side_clamp": {"min": 0.5 - FUHAO_MAX_EDGE, "max": 0.5 + FUHAO_MAX_EDGE},
        "confidence": round(conf, 3),
        "signal_level": level,
        "pattern_label": advanced_label,
        "regime": fake_pattern.get("regime", "fuhao_clone"),
        "fake_pattern_detector": fake_pattern,
        "fake_pattern_label": fake_pattern.get("label", ""),
        "fake_pattern_score": fake_pattern.get("fake_score", 0.0),
        "fake_pattern_turn_score": fake_pattern.get("turn_score", 0.0),
        "fake_pattern_false_break_score": fake_pattern.get("false_break_score", 0.0),
        "ngram_label": "富濠式模型不使用NGram",
        "ngram_sample": 0,
        "big_road_label": road_models.get("big_road", {}).get("label", ""),
        "big_eye_label": road_models.get("big_eye", {}).get("label", ""),
        "small_road_label": road_models.get("small_road", {}).get("label", ""),
        "cockroach_label": road_models.get("cockroach", {}).get("label", ""),
        "road_consensus_label": road_consensus_label,
        "road_consensus_ratio": road_majority.get("ratio", 0.5),
        "road_conflict_ratio": round(1.0 - float(road_majority.get("ratio", 0.5)), 4),
        "derived_majority": derived_majority,
        "derived_confirm_count": derived_confirm_count,
        "non_bigroad_confirm_count": non_bigroad_confirm_count,
        "final_gate": {
            "road_majority_as_candidate_only": FUHAO_ROAD_MAJORITY_AS_CANDIDATE_ONLY,
            "disable_bigroad_fallback": FUHAO_DISABLE_BIGROAD_FALLBACK,
            "require_derived_for_final": FUHAO_REQUIRE_DERIVED_FOR_FINAL,
            "observe_on_bigroad_only": FUHAO_OBSERVE_ON_BIGROAD_ONLY,
            "required_non_bigroad_votes": FUHAO_FINAL_REQUIRE_NON_BIGROAD_VOTES,
            "derived_ratio": round(derived_ratio, 4),
        },
        "road_family": {"fuhao_road_models": road_models, "fuhao_road_majority": road_majority, "down3_family": road_models.get("down3_family", {})},
        "down3_family": road_models.get("down3_family", {}),
        "down3_family_label": road_models.get("down3_family", {}).get("label", ""),
        "dense_board": dense_board,
        "final_confirmation": {"sources": confirm_sources, "count": len(confirm_sources), "dense_conflict": dense_conflict, "source": final_source},
        "pattern_replay_memory": pattern_replay,
        "road_lifecycle": empty_state,
        "adaptive_road_memory": empty_state,
        "road_rhythm": empty_state,
        "road_rhythm_state": "FUHAO_CLONE",
        "road_rhythm_label": "富濠式模型不使用Road Rhythm",
        "road_rhythm_confidence": 0.0,
        "road_rhythm_false_break_score": fake_pattern.get("false_break_score", 0.0),
        "road_rhythm_turn_score": fake_pattern.get("turn_score", 0.0),
        "road_rhythm_inertia_score": 0.0,
        "long_anchor": empty_state,
        "long_anchor_state": "FUHAO_CLONE",
        "long_anchor_label": "富濠式模型不使用Long Anchor",
        "long_anchor_side": "",
        "long_anchor_confidence": 0.0,
        "road_memory_state": "FUHAO_CLONE",
        "road_memory_label": "富濠式模型不使用Road Memory",
        "road_memory_sample": 0,
        "road_memory_follow_rate": 0.5,
        "road_memory_break_rate": 0.5,
        "road_memory_confidence": 0.0,
        "road_lifecycle_state": "FUHAO_CLONE",
        "road_lifecycle_label": "富濠式模型不使用Lifecycle",
        "road_follow_score": 0.5,
        "road_break_score": 0.0,
        "road_fatigue_score": 0.0,
        "road_engine_label": road_consensus_label,
        "road_engine_break_risk": 0.0,
        "road_engine_consistency": road_majority.get("ratio", 0.5),
        "road_engine_big_road": road_models.get("big_road", {}).get("details", {}),
        "road_engine_derived": {
            "big_eye": road_models.get("big_eye", {}).get("stats", {}),
            "small_road": road_models.get("small_road", {}).get("stats", {}),
            "cockroach": road_models.get("cockroach", {}).get("stats", {}),
        },
        "dynamic_weights": dynamic_weights,
        "ask_road_memory": ask_road_performance,
        "ask_road_memory_label": ask_road_performance.get("label", ""),
        "ask_road_memory_enabled": USE_ASK_ROAD_MEMORY,
        "online_model_performance": {},
        "reason": " / ".join([x for x in reason_parts if x]),
        "ai_used": ai_used,
        "ai_side": ai_side,
        "ai_status": ai_status,
        "ai_confidence": round(ai_conf, 4),
        "ml_trained": False,
        "ml_samples": 0,
        "tf_available": TF_AVAILABLE,
        "training_key": training_key,
        "model_cache_size": len(_MODEL_CACHE),
        "ml_predictions": None,
        "ai_result": ai_result if (FUHAO_DEBUG or os.getenv("DEBUG_AI_RESULT", "0") == "1") else None,
        "fuhao": {
            "final_source": final_source,
            "final_pick": final_pick,
            "decision_source": decision_source,
            "decision_pick": decision_pick if decision_pick in {"B", "P"} else "",
            "recommend": recommend,
            "road_votes": road_votes,
            "advanced_votes": advanced_votes,
            "all_votes": all_votes,
            "road_majority": road_majority,
            "advanced_majority": advanced_majority,
            "all_majority": all_majority,
            "vote_ratio": vote_ratio,
            "final_vote_count": final_vote_count,
            "fake_pattern": fake_pattern,
            "deepseek": {
                "enabled": bool(USE_DEEPSEEK and FUHAO_USE_DEEPSEEK),
                "used": ai_used,
                "status": ai_status,
                "side": ai_side,
                "confidence": round(ai_conf, 4),
                "mode": FUHAO_DEEPSEEK_MODE,
                "observe_on_conflict": FUHAO_DEEPSEEK_OBSERVE_ON_CONFLICT,
                "weight": FUHAO_DEEPSEEK_WEIGHT,
                "payload": ai_payload if FUHAO_DEEPSEEK_INCLUDE_PAYLOAD else None,
            },
            "models": {"road": road_models, "advanced": advanced_models},
            "settings": {
                "history_limit": FUHAO_HISTORY_LIMIT,
                "min_valid_rounds": FUHAO_MIN_VALID_ROUNDS,
                "require_road_and_advanced_same": FUHAO_REQUIRE_ROAD_AND_ADVANCED_SAME,
                "min_vote_agree": FUHAO_MIN_VOTE_AGREE,
                "ignore_tie_for_predict": FUHAO_IGNORE_TIE_FOR_PREDICT,
                "use_deepseek": bool(USE_DEEPSEEK and FUHAO_USE_DEEPSEEK),
                "deepseek_mode": FUHAO_DEEPSEEK_MODE,
                "deepseek_min_history": FUHAO_DEEPSEEK_MIN_HISTORY,
                "deepseek_weight": FUHAO_DEEPSEEK_WEIGHT,
                "use_fake_pattern_detector": FUHAO_USE_FAKE_PATTERN_DETECTOR,
                "fake_pattern_hard_decision": FUHAO_FAKE_PATTERN_HARD_DECISION,
                "fake_pattern_neutralize_on_observe": FUHAO_FAKE_PATTERN_NEUTRALIZE_ON_OBSERVE,
                "fake_pattern_allow_reverse": FUHAO_FAKE_PATTERN_ALLOW_REVERSE,
                "fake_pattern_reverse_score": FUHAO_FAKE_PATTERN_REVERSE_SCORE,
                "use_ask_road_memory": USE_ASK_ROAD_MEMORY,
                "ask_road_memory_window": ASK_ROAD_MEMORY_WINDOW,
                "fake_pattern_observe_score": FUHAO_FAKE_PATTERN_OBSERVE_SCORE,
                "fake_pattern_hard_observe_score": FUHAO_FAKE_PATTERN_HARD_OBSERVE_SCORE,
                "fake_pattern_turn_score": FUHAO_FAKE_PATTERN_TURN_SCORE,
                "fake_pattern_false_break_score": FUHAO_FAKE_PATTERN_FALSE_BREAK_SCORE,
            },
        },
        "debug": {
            "engine": "FUHAO_CLONE",
            "history_tail": "".join(history[-36:]),
            "non_tie_tail": "".join(non_tie[-36:]),
            "fuhao": {
                "road_models": road_models,
                "advanced_models": advanced_models,
                "road_majority": road_majority,
                "advanced_majority": advanced_majority,
                "all_majority": all_majority,
                "fake_pattern": fake_pattern,
                "deepseek": {
                    "ai_result": ai_result,
                    "ai_side": ai_side,
                    "ai_status": ai_status,
                    "ai_confidence": ai_conf,
                    "ai_payload": ai_payload if FUHAO_DEEPSEEK_INCLUDE_PAYLOAD else None,
                },
            },
        } if FUHAO_DEBUG or os.getenv("DEBUG_PREDICTOR", "0") == "1" else None,
    }
    return result

def predict(history: List[str], venue: str = "", room: str = "", shoe_id: str = "", user_id: str = "") -> Dict[str, Any]:
    """
    整合預測函數：四路主模型 + Road Lifecycle + Adaptive Road Memory + Road Rhythm + NGram + 動態權重 + ML模型 + DeepSeek校準
    注意：本版加入低信心/四路分歧觀望機制；仍不做下注金額/EV 配注決策。
    """
    history = [str(x).upper() for x in history if str(x).upper() in {"B", "P", "T"}]
    if PREDICT_ENGINE in {"FUHAO", "FUHAO_CLONE", "FUHAOCLONE"}:
        return _fuhao_clone_predict(history, venue=venue, room=room, shoe_id=shoe_id, user_id=user_id)

    non_tie = _last_non_tie(history)

    # 每個 LINE UID / 場館 / 房間 / 靴號 都是獨立模型與獨立逐局前推狀態。
    # 若 app.py 沒有傳入 user_id，會落到 anonymous；LINE 實戰務必傳 LINE UID。
    identity = str(user_id or "anonymous")
    training_key = f"{identity}|{venue or 'global'}|{room or 'global'}|{shoe_id or 'global'}"
    _update_walk_forward_truth(training_key, non_tie)
    _update_ask_road_truth(training_key, non_tie)
    live_walk_forward_performance = _get_walk_forward_performance(training_key)
    ask_road_performance = _get_ask_road_performance(training_key)
    dense_board = _detect_dense_board(non_tie)
    ask_road_performance = _limit_ask_road_performance_for_dense(ask_road_performance, dense_board)

    # ============ 1. 基礎模型 + 大路 / 下三路家族 ==========
    markov = _transition_prob(non_tie)
    road = _road_pattern_score(non_tie)
    recent = _recent_score(non_tie)
    balance = _balance_score(non_tie)
    streak = _streak_score(non_tie)
    ngram = _ngram_score(non_tie)

    road_family = _road_family_scores(non_tie)

    # Ask Road Hit Memory：依本靴最近問路命中率，微調大眼仔 / 小路 / 蟑螂路邊際。
    if USE_ASK_ROAD_MEMORY and ASK_ROAD_MEMORY_APPLY_TO_HYBRID:
        for _rk in ["big_eye", "small_road", "cockroach"]:
            if _rk in road_family:
                road_family[_rk] = _apply_ask_road_factor_to_score(road_family[_rk], _rk, ask_road_performance)
        try:
            road_family["down3_family"] = _down3_family_score(non_tie, {
                "big_eye": road_family.get("big_eye", {}),
                "small_road": road_family.get("small_road", {}),
                "cockroach": road_family.get("cockroach", {}),
            })
            road_family["consensus"] = _road_consensus_score(road_family)
        except Exception:
            pass

    big_road = road_family.get("big_road", {"B": 0.5, "P": 0.5, "label": "大路資料不足"})
    big_eye = road_family.get("big_eye", {"B": 0.5, "P": 0.5, "label": "大眼仔資料不足"})
    small_road = road_family.get("small_road", {"B": 0.5, "P": 0.5, "label": "小路資料不足"})
    cockroach = road_family.get("cockroach", {"B": 0.5, "P": 0.5, "label": "蟑螂路資料不足"})
    down3_family = road_family.get("down3_family", {"B": 0.5, "P": 0.5, "pick": "", "label": "下三路家族資料不足"})
    road_consensus = road_family.get("consensus", {"B": 0.5, "P": 0.5, "label": "四路共識資料不足"})
    road_engine = _road_engine_score(non_tie)  # 舊欄位相容用

    regime_info = _detect_regime(non_tie)
    online_performance = _rolling_model_performance(non_tie)
    lifecycle = _road_lifecycle_score(non_tie, road_family, regime_info)
    road_memory = _adaptive_road_memory_score(non_tie, road_family, lifecycle, regime_info)
    road_rhythm = _road_rhythm_score(non_tie, road_family, lifecycle, regime_info, road_memory)
    long_anchor = _long_anchor_score(non_tie, road_family, lifecycle, regime_info)
    pattern_replay = _pattern_replay_memory_score(non_tie, training_key=training_key, live_performance=live_walk_forward_performance)
    dynamic_weights = _apply_online_weighting(regime_info.get("weights", {}), online_performance)
    # 逐局前推 live performance 是每個 LINE UID / 房間 / 靴號獨立累積，
    # 會把這一靴目前準的模型加權、目前失效的模型降權。
    dynamic_weights = _apply_walk_forward_weighting(dynamic_weights, live_walk_forward_performance)
    dynamic_weights = _apply_lifecycle_weighting(dynamic_weights, lifecycle)
    dynamic_weights = _apply_road_memory_weighting(dynamic_weights, road_memory)
    dynamic_weights = _apply_road_rhythm_weighting(dynamic_weights, road_rhythm)
    dynamic_weights = _collapse_down3_weights(dynamic_weights)

    total_w = sum(dynamic_weights.values()) or 1.0
    b_side = (
        big_road["B"] * dynamic_weights.get("big_road", 0.0)
        + down3_family["B"] * dynamic_weights.get("down3_family", 0.0)
        + ngram["B"] * dynamic_weights.get("ngram", 0.0)
        + markov["B"] * dynamic_weights.get("markov", 0.0)
        + road["B"] * dynamic_weights.get("road", 0.0)
        + recent["B"] * dynamic_weights.get("recent", 0.0)
        + streak["B"] * dynamic_weights.get("streak", 0.0)
        + balance["B"] * dynamic_weights.get("balance", 0.0)
    ) / total_w

    # 四路共識很高時，輕微加強；四路分歧時，輕微回收到 0.5，避免互打。
    consensus_pick = road_consensus.get("pick", "")
    consensus_ratio = float(road_consensus.get("consensus_ratio", 0.5))
    conflict_ratio = float(road_consensus.get("conflict_ratio", 0.5))
    if consensus_pick and consensus_ratio >= 0.72:
        signed = 1 if consensus_pick == "B" else -1
        b_side += signed * ROAD_CONSENSUS_BOOST * (consensus_ratio - 0.5) * 2
    elif conflict_ratio >= 0.45:
        b_side = 0.5 + (b_side - 0.5) * (1 - ROAD_CONFLICT_SHRINK)

    # Road Lifecycle 會判斷規律是健康可跟、疲乏、斷點壓力或已斷，再做方向偏移。
    b_side = _apply_lifecycle_bias(b_side, lifecycle)
    # Adaptive Road Memory 會看本靴過去相似牌路到底是跟準還是斷準，再做柔性修正。
    b_side = _apply_road_memory_bias(b_side, road_memory)
    # Pattern Replay 會拿目前尾段去整靴已知歷史中找相似規律，看當時下一口如何走。
    b_side = _apply_pattern_replay_bias(b_side, pattern_replay, live_walk_forward_performance)
    # Road Rhythm 會看短 / 中 / 長週期，避免太看當局，分辨假斷與真轉折。
    b_side = _apply_road_rhythm_bias(b_side, road_rhythm)
    # Long Anchor Guard 會用長週期錨定限制短線偏移，降低太看當局。
    b_side = _apply_long_anchor_guard(b_side, long_anchor, lifecycle, road_memory, road_rhythm)
    b_side = _clamp(b_side, SIDE_CLAMP_MIN, SIDE_CLAMP_MAX)
    p_side = 1 - b_side

    tie_prob = _tie_score(history)
    b_prob = b_side * (1 - tie_prob)
    p_prob = p_side * (1 - tie_prob)

    # ============ 2. ML模型預測 ==========
    ml_models = _get_ml_models(training_key)

    should_train = (
        ML_WEIGHT > 0
        and len(non_tie) >= 30
        and (
            not ml_models.is_trained
            or getattr(ml_models, "last_training_key", "") != training_key
            or len(non_tie) - len(getattr(ml_models, "last_training_history", [])) >= ML_RETRAIN_INTERVAL
        )
    )

    if should_train:
        train_result = ml_models.train(non_tie, training_key=training_key)
        logger.info(f"ML訓練結果: {train_result}")

    ml_pred = ml_models.predict(non_tie)
    ml_b_prob = ml_pred.get('ensemble', 0.5)

    if ml_models.is_trained:
        ml_weight = ML_WEIGHT * (0.5 + 0.5 * min(1.0, ml_models.training_samples / 50))
        if WALK_FORWARD_APPLY_TO_ML:
            ml_weight *= _walk_forward_factor(live_walk_forward_performance, "ml_ensemble", 1.0)
        # 如果四路生命週期高信心偏某一邊，而 ML 強烈反向，就縮小 ML 影響，避免 ML 把「該跟/該斷」拉歪。
        lifecycle_bias_side = lifecycle.get("bias_side", "") if lifecycle.get("enabled") else ""
        lifecycle_conf = float(lifecycle.get("confidence", 0.0)) if lifecycle.get("enabled") else 0.0
        memory_bias_side = road_memory.get("bias_side", "") if road_memory.get("enabled") else ""
        memory_conf = float(road_memory.get("confidence", 0.0)) if road_memory.get("enabled") else 0.0
        rhythm_bias_side = road_rhythm.get("bias_side", "") if road_rhythm.get("enabled") else ""
        rhythm_conf = float(road_rhythm.get("confidence", 0.0)) if road_rhythm.get("enabled") else 0.0
        protect_side = ""
        protect_conf = 0.0
        shrink = LIFECYCLE_ML_SHRINK
        if memory_bias_side and memory_conf >= ROAD_MEMORY_PROTECT_MIN_CONF:
            protect_side, protect_conf, shrink = memory_bias_side, memory_conf, ROAD_MEMORY_ML_SHRINK
        elif rhythm_bias_side and rhythm_conf >= ROAD_RHYTHM_TURN_CONFIRM:
            protect_side, protect_conf, shrink = rhythm_bias_side, rhythm_conf, ROAD_RHYTHM_ML_SHRINK
        else:
            protect_side, protect_conf, shrink = lifecycle_bias_side, lifecycle_conf, LIFECYCLE_ML_SHRINK
        ml_pick = "B" if ml_b_prob >= 0.5 else "P"
        if protect_side and protect_conf >= min(LIFECYCLE_PROTECT_MIN_CONF, ROAD_MEMORY_PROTECT_MIN_CONF) and ml_pick != protect_side:
            ml_weight *= _clamp(1.0 - shrink, 0.05, 1.0)
        b_prob = b_prob * (1 - ml_weight) + ml_b_prob * ml_weight
        p_prob = p_prob * (1 - ml_weight) + (1 - ml_b_prob) * ml_weight

    # ============ 3. DeepSeek校準 ==========
    feature_payload = {
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "history_len": len(history),
        "history_tail": "".join(history[-36:]),
        "non_tie_tail": "".join(non_tie[-36:]),
        "big_road_model": big_road,
        "big_eye_model": big_eye,
        "small_road_model": small_road,
        "cockroach_model": cockroach,
        "road_consensus": road_consensus,
        "road_family": road_family,
        "down3_family": down3_family,
        "down3_family_label": down3_family.get("label", ""),
        "dense_board": dense_board,
        "road_lifecycle": lifecycle,
        "adaptive_road_memory": road_memory,
        "pattern_replay_memory": pattern_replay,
        "road_rhythm": road_rhythm,
        "long_anchor": long_anchor,
        "road_engine": road_engine,
        "markov": markov,
        "road": road,
        "recent": recent,
        "balance": balance,
        "streak": streak,
        "ngram": ngram,
        "regime": regime_info,
        "dynamic_weights": {k: round(v, 4) for k, v in dynamic_weights.items()},
        "online_performance": online_performance,
        "live_walk_forward_performance": live_walk_forward_performance,
        "ml_predictions": ml_pred,
        "tf_available": TF_AVAILABLE,
        "training_key": training_key,
        "local_probs": {"B": round(b_prob, 5), "P": round(p_prob, 5), "T": round(tie_prob, 5)},
    }

    ai_result = None
    if USE_DEEPSEEK and len(history) >= MIN_HISTORY_FOR_AI and AI_BLEND > 0:
        try:
            ai_result = DeepSeekClient().calibrate(feature_payload)
        except Exception as e:
            ai_result = {"error": True, "message": str(e)}

        if ai_result and not ai_result.get("error"):
            try:
                ba = _clamp(float(ai_result.get("banker_adjust", 0)), -0.035, 0.035)
                pa = _clamp(float(ai_result.get("player_adjust", 0)), -0.035, 0.035)
                ta = _clamp(float(ai_result.get("tie_adjust", 0)), -0.020, 0.020)
                ai_conf = _clamp(float(ai_result.get("confidence", 0.4)), 0, 1)
                blend = AI_BLEND * (0.45 + ai_conf * 0.55)
                if WALK_FORWARD_APPLY_TO_AI:
                    blend *= _walk_forward_factor(live_walk_forward_performance, "ai", 1.0)
                # AI 是校準器；若它與高信心生命周期方向反向，縮小校準幅度，避免覆蓋四路生命周期判斷。
                lifecycle_bias_side = lifecycle.get("bias_side", "") if lifecycle.get("enabled") else ""
                lifecycle_conf = float(lifecycle.get("confidence", 0.0)) if lifecycle.get("enabled") else 0.0
                memory_bias_side = road_memory.get("bias_side", "") if road_memory.get("enabled") else ""
                memory_conf = float(road_memory.get("confidence", 0.0)) if road_memory.get("enabled") else 0.0
                rhythm_bias_side = road_rhythm.get("bias_side", "") if road_rhythm.get("enabled") else ""
                rhythm_conf = float(road_rhythm.get("confidence", 0.0)) if road_rhythm.get("enabled") else 0.0
                protect_side = ""
                protect_conf = 0.0
                shrink = LIFECYCLE_AI_SHRINK
                if memory_bias_side and memory_conf >= ROAD_MEMORY_PROTECT_MIN_CONF:
                    protect_side, protect_conf, shrink = memory_bias_side, memory_conf, ROAD_MEMORY_AI_SHRINK
                elif rhythm_bias_side and rhythm_conf >= ROAD_RHYTHM_TURN_CONFIRM:
                    protect_side, protect_conf, shrink = rhythm_bias_side, rhythm_conf, ROAD_RHYTHM_AI_SHRINK
                else:
                    protect_side, protect_conf, shrink = lifecycle_bias_side, lifecycle_conf, LIFECYCLE_AI_SHRINK
                ai_side = "B" if ba > pa else "P" if pa > ba else ""
                if protect_side and ai_side and protect_conf >= min(LIFECYCLE_PROTECT_MIN_CONF, ROAD_MEMORY_PROTECT_MIN_CONF) and ai_side != protect_side:
                    blend *= _clamp(1.0 - shrink, 0.05, 1.0)
                b_prob += ba * blend
                p_prob += pa * blend
                tie_prob += ta * blend
            except Exception:
                pass

    # ============ 4. 正規化 ==========
    b_prob, p_prob, tie_prob = _normalize_three(b_prob, p_prob, tie_prob)

    # ============ 5. 投票一致性 ==========
    votes = []
    current_model_scores = {
        "big_road": big_road,
        "down3_family": down3_family,
        "pattern_replay": pattern_replay,
        "ngram": ngram,
        "markov": markov,
        "road": road,
        "recent": recent,
        "streak": streak,
        "balance": balance,
    }
    current_model_picks = _walk_forward_pick_map(current_model_scores)
    for pick in current_model_picks.values():
        if pick:
            votes.append(pick)

    if ml_models.is_trained and abs(ml_b_prob - 0.5) >= WALK_FORWARD_ML_MIN_EDGE:
        current_model_picks["ml_ensemble"] = "B" if ml_b_prob >= 0.5 else "P"
        votes.append(current_model_picks["ml_ensemble"])

    main_pick = "B" if b_prob >= p_prob else "P"
    agreement = votes.count(main_pick) / len(votes) if votes else 0.5

    if ml_models.is_trained:
        ml_pick = "B" if ml_b_prob >= 0.5 else "P"
        ml_strength = abs(ml_b_prob - 0.5) * 2
        ml_agreement = ml_strength if ml_pick == main_pick else 0.0
    else:
        ml_agreement = 0.0

    # ============ 6. 推薦與信心 ==========
    conf, level = _confidence(b_prob, p_prob, tie_prob, len(history), agreement, ml_agreement)

    edge = abs(b_prob - p_prob)
    observe_reason = ""
    lifecycle_state = str(lifecycle.get("state", "")).upper() if lifecycle.get("enabled") else ""
    down3_pick = str(down3_family.get("pick", "") or "")
    big_road_pick = _strong_pick_from_score(big_road, min_gap=FINAL_CONFIRM_SCORE_GAP)
    ml_pick_for_confirm = ("B" if ml_b_prob >= 0.5 else "P") if ml_models.is_trained else ""
    final_confirmation = _final_confirmation_summary(
        target=main_pick,
        big_road=big_road,
        pattern_replay=pattern_replay,
        independent_scores={"ngram": ngram, "markov": markov, "road_pattern": road},
        ml_pick=ml_pick_for_confirm,
        ml_gap=abs(ml_b_prob - 0.5) * 2 if ml_models.is_trained else 0.0,
    )
    dense_conflict = bool(
        dense_board.get("is_dense")
        and down3_pick in {"B", "P"}
        and big_road_pick in {"B", "P"}
        and down3_pick != big_road_pick
    )

    if ALLOW_TIE_RECOMMEND and tie_prob >= TIE_RECOMMEND_MIN and tie_prob > max(b_prob, p_prob) * 0.55:
        recommend = "T"
    elif (
        ALLOW_OBSERVE
        and down3_pick == main_pick
        and not final_confirmation.get("confirmed")
    ):
        recommend = "NONE"
        observe_reason = "下三路家族只有候選方向，尚未取得大路、Pattern Replay或獨立模型確認"
    elif (
        ALLOW_OBSERVE
        and dense_conflict
        and down3_pick == main_pick
        and DENSE_CONFLICT_REQUIRE_NON_ROAD_CONFIRM
        and int(final_confirmation.get("non_road_count", 0)) < 1
    ):
        recommend = "NONE"
        observe_reason = "密集盤下三路家族與大路衝突，未取得非大路獨立確認"
    elif (
        ALLOW_OBSERVE
        and edge < OBSERVE_EDGE_MIN
        and conf < OBSERVE_CONF_MAX
    ):
        recommend = "NONE"
        observe_reason = f"莊閒差距{edge * 100:.1f}%且信心不足"
    elif (
        ALLOW_OBSERVE
        and conflict_ratio >= OBSERVE_CONFLICT_MIN
        and conf < OBSERVE_CONFLICT_CONF_MAX
    ):
        recommend = "NONE"
        observe_reason = f"四路分歧{int(conflict_ratio * 100)}%且信心不足"
    elif (
        ALLOW_OBSERVE
        and lifecycle_state in OBSERVE_LIFECYCLE_STATES
        and conf < OBSERVE_CONFLICT_CONF_MAX
    ):
        recommend = "NONE"
        observe_reason = f"生命周期{lifecycle_state}且信心不足"
    else:
        recommend = main_pick

    recommend_text_map = {"B": "莊", "P": "閒", "T": "和", "NONE": "觀望"}

    # ============ 7. 原因說明 ==========
    reason_parts = [
        f"大路/下三路家族:{road_consensus.get('label', '')}",
        f"下三路家族:{down3_family.get('label', '')}",
        f"密集盤:{dense_board.get('label', '')}",
        f"問路記憶:{ask_road_performance.get('label', '')}",
        f"生命周期:{lifecycle.get('label', '')}",
        f"記憶:{road_memory.get('label', '')}",
        f"歷史回放:{pattern_replay.get('label', '')}",
        f"節奏:{road_rhythm.get('label', '')}",
        f"長錨:{long_anchor.get('label', '')}",
        big_road.get("label", ""),
        big_eye.get("label", ""),
        small_road.get("label", ""),
        cockroach.get("label", ""),
        f"型態:{regime_info.get('regime', '')}",
        f"{ngram.get('label', '')}",
        f"一致{int(agreement * 100)}%",
    ]
    if observe_reason:
        reason_parts.append(f"觀望:{observe_reason}")
    if ml_models.is_trained:
        reason_parts.append(f"ML集體{int(ml_b_prob * 100)}%")
    if ai_result and ai_result.get("pattern_label"):
        reason_parts.append(f"AI:{ai_result.get('pattern_label')}")
    elif ai_result and ai_result.get("error"):
        reason_parts.append("AI離線改本地判斷")

    # ============ 8. 逐局前推 pending：本輪各模型只預測下一局，等下一次使用者回報結果再結算 ==========
    if ai_result and not ai_result.get("error"):
        try:
            ai_b = float(ai_result.get("banker_adjust", 0))
            ai_p = float(ai_result.get("player_adjust", 0))
            if abs(ai_b - ai_p) >= WALK_FORWARD_AI_MIN_EDGE:
                current_model_picks["ai"] = "B" if ai_b > ai_p else "P"
        except Exception:
            pass
    if recommend in {"B", "P"}:
        current_model_picks["final"] = recommend
    _store_walk_forward_pending(training_key, non_tie, current_model_picks)

    ask_road_pending = {
        "big_eye": _pick_from_score(big_eye, min_edge=0.002),
        "small_road": _pick_from_score(small_road, min_edge=0.002),
        "cockroach": _pick_from_score(cockroach, min_edge=0.002),
        "down3_family": down3_family.get("pick", ""),
        "road_majority": road_consensus.get("pick", ""),
        "final": recommend if recommend in {"B", "P"} else "",
    }
    _store_ask_road_pending(training_key, non_tie, ask_road_pending)

    # ============ 9. 返回結果 ==========
    return {
        "ok": True,
        "user_id": user_id,
        "venue": venue,
        "room": room,
        "shoe_id": shoe_id,
        "round_no": len(history) + 1,
        "history_len": len(history),
        "banker_rate": round(b_prob * 100, 1),
        "player_rate": round(p_prob * 100, 1),
        "tie_rate": round(tie_prob * 100, 1),
        "recommend": recommend,
        "recommend_text": recommend_text_map.get(recommend, "觀望"),
        "is_observe": recommend == "NONE",
        "observe_reason": observe_reason,
        "decision_edge": round(edge, 5),
        "side_clamp": {"min": SIDE_CLAMP_MIN, "max": SIDE_CLAMP_MAX},
        "confidence": round(conf, 3),
        "signal_level": level,
        "pattern_label": road.get("label", ""),
        "regime": regime_info.get("regime", ""),
        "ngram_label": ngram.get("label", ""),
        "ngram_sample": ngram.get("sample", 0),
        "big_road_label": big_road.get("label", ""),
        "big_eye_label": big_eye.get("label", ""),
        "small_road_label": small_road.get("label", ""),
        "cockroach_label": cockroach.get("label", ""),
        "road_consensus_label": road_consensus.get("label", ""),
        "road_consensus_ratio": road_consensus.get("consensus_ratio", 0.5),
        "road_conflict_ratio": road_consensus.get("conflict_ratio", 0.5),
        "road_family": road_family,
        "down3_family": down3_family,
        "down3_family_label": down3_family.get("label", ""),
        "dense_board": dense_board,
        "final_confirmation": final_confirmation,
        "road_lifecycle": lifecycle,
        "adaptive_road_memory": road_memory,
        "pattern_replay_memory": pattern_replay,
        "pattern_replay_state": pattern_replay.get("state", ""),
        "pattern_replay_label": pattern_replay.get("label", ""),
        "pattern_replay_side": pattern_replay.get("bias_side", ""),
        "pattern_replay_confidence": pattern_replay.get("confidence", 0.0),
        "pattern_replay_sample": pattern_replay.get("sample", 0),
        "pattern_replay_edge": pattern_replay.get("edge", 0.0),
        "road_rhythm": road_rhythm,
        "road_rhythm_state": road_rhythm.get("state", ""),
        "road_rhythm_label": road_rhythm.get("label", ""),
        "road_rhythm_confidence": road_rhythm.get("confidence", 0.0),
        "road_rhythm_false_break_score": road_rhythm.get("false_break_score", 0.0),
        "road_rhythm_turn_score": road_rhythm.get("turn_score", 0.0),
        "road_rhythm_inertia_score": road_rhythm.get("inertia_score", 0.0),
        "long_anchor": long_anchor,
        "long_anchor_state": long_anchor.get("state", ""),
        "long_anchor_label": long_anchor.get("label", ""),
        "long_anchor_side": long_anchor.get("anchor_side", ""),
        "long_anchor_confidence": long_anchor.get("confidence", 0.0),
        "road_memory_state": road_memory.get("state", ""),
        "road_memory_label": road_memory.get("label", ""),
        "road_memory_sample": road_memory.get("sample", 0),
        "road_memory_follow_rate": road_memory.get("follow_rate", 0.5),
        "road_memory_break_rate": road_memory.get("break_rate", 0.5),
        "road_memory_confidence": road_memory.get("confidence", 0.0),
        "pattern_replay_state": pattern_replay.get("state", ""),
        "pattern_replay_label": pattern_replay.get("label", ""),
        "pattern_replay_sample": pattern_replay.get("sample", 0),
        "pattern_replay_side": pattern_replay.get("bias_side", ""),
        "pattern_replay_confidence": pattern_replay.get("confidence", 0.0),
        "road_lifecycle_state": lifecycle.get("state", ""),
        "road_lifecycle_label": lifecycle.get("label", ""),
        "road_follow_score": lifecycle.get("follow_score", 0.5),
        "road_break_score": lifecycle.get("break_score", 0.0),
        "road_fatigue_score": lifecycle.get("fatigue_score", 0.0),
        "road_engine_label": road_engine.get("label", ""),
        "road_engine_break_risk": road_engine.get("break_risk", 0.0),
        "road_engine_consistency": road_engine.get("consistency", 0.5),
        "road_engine_big_road": road_engine.get("big_road", {}),
        "road_engine_derived": road_engine.get("derived", {}),
        "dynamic_weights": {k: round(v, 4) for k, v in dynamic_weights.items()},
        "online_model_performance": online_performance,
        "live_walk_forward_performance": live_walk_forward_performance,
        "ask_road_memory": ask_road_performance,
        "ask_road_memory_label": ask_road_performance.get("label", ""),
        "ask_road_memory_enabled": USE_ASK_ROAD_MEMORY,
        "walk_forward_enabled": USE_WALK_FORWARD_LEARNING,
        "walk_forward_state_size": len(_WALK_FORWARD_STATE),
        "reason": " / ".join([x for x in reason_parts if x]),
        "ai_used": bool(ai_result and not ai_result.get("error")),
        "ml_trained": ml_models.is_trained,
        "ml_samples": ml_models.training_samples,
        "tf_available": TF_AVAILABLE,
        "training_key": training_key,
        "model_cache_size": len(_MODEL_CACHE),
        "ml_predictions": {
            "lr": round(ml_pred.get('lr', 0.5), 4),
            "rf": round(ml_pred.get('rf', 0.5), 4),
            "lstm": round(ml_pred.get('lstm', 0.5), 4),
            "ensemble": round(ml_pred.get('ensemble', 0.5), 4)
        } if ml_models.is_trained else None,
        "ai_result": ai_result if os.getenv("DEBUG_AI_RESULT", "0") == "1" else None,
        "debug": feature_payload if os.getenv("DEBUG_PREDICTOR", "0") == "1" else None,
    }
