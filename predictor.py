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

# 保留 LSTM：有安裝 tensorflow-cpu 時會啟用；若環境還沒裝好，不會讓整個服務直接掛掉
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

# 四路主模型權重：大路 / 大眼仔 / 小路 / 蟑螂路
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
ONLINE_WEIGHT_WINDOW = int(os.getenv("ONLINE_WEIGHT_WINDOW", "36"))
ONLINE_WEIGHT_MIN_COUNT = int(os.getenv("ONLINE_WEIGHT_MIN_COUNT", "12"))
ONLINE_WEIGHT_ALPHA = float(os.getenv("ONLINE_WEIGHT_ALPHA", "0.22"))
ONLINE_BAYES_ALPHA = float(os.getenv("ONLINE_BAYES_ALPHA", "6.0"))
ONLINE_DISABLE_BELOW = float(os.getenv("ONLINE_DISABLE_BELOW", "0.42"))
ONLINE_BOOST_ABOVE = float(os.getenv("ONLINE_BOOST_ABOVE", "0.58"))

# RoadEngine / 下三路路紙引擎參數
ROAD_ENGINE_ROWS = int(os.getenv("ROAD_ENGINE_ROWS", "6"))
ROAD_ENGINE_MIN_HISTORY = int(os.getenv("ROAD_ENGINE_MIN_HISTORY", "6"))
ROAD_ENGINE_BREAK_STREAK = int(os.getenv("ROAD_ENGINE_BREAK_STREAK", "5"))
ROAD_ENGINE_DERIVED_LOOKBACK = int(os.getenv("ROAD_ENGINE_DERIVED_LOOKBACK", "10"))
ROAD_ENGINE_BLUE_BREAK_BIAS = float(os.getenv("ROAD_ENGINE_BLUE_BREAK_BIAS", "0.024"))
ROAD_ENGINE_RED_CONT_BIAS = float(os.getenv("ROAD_ENGINE_RED_CONT_BIAS", "0.016"))
DERIVED_ROAD_MIN_COUNT = int(os.getenv("DERIVED_ROAD_MIN_COUNT", "3"))
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
ROAD_MEMORY_LOOKBACK = int(os.getenv("ROAD_MEMORY_LOOKBACK", "48"))
ROAD_MEMORY_MIN_SAMPLE = int(os.getenv("ROAD_MEMORY_MIN_SAMPLE", "10"))
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
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", "10"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "5"))
LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "8"))
ML_RETRAIN_INTERVAL = int(os.getenv("ML_RETRAIN_INTERVAL", "10"))

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
MAX_MODEL_CACHE = int(os.getenv("MAX_MODEL_CACHE", "30"))
_MODEL_CACHE: Dict[str, MLModels] = {}
_MODEL_CACHE_ORDER: List[str] = []


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
    """
    建立簡化且穩定的大路矩陣。
    - 同邊：往下排；到底或被占用則往右延伸。
    - 換邊：新欄第一列。
    回傳位置、欄高、最後位置等，供大路與下三路使用。
    """
    rows = max(3, int(rows or 6))
    grid: Dict[Tuple[int, int], str] = {}
    positions: List[Dict[str, Any]] = []

    last_side = ""
    row = 0
    col = 0

    for idx, side in enumerate(non_tie):
        if side not in {"B", "P"}:
            continue

        if idx == 0:
            row, col = 0, 0
        elif side != last_side:
            col = col + 1
            row = 0
            while (row, col) in grid:
                col += 1
        else:
            target_row = row + 1
            target_col = col
            if target_row < rows and (target_row, target_col) not in grid:
                row = target_row
            else:
                # 到底或下方被占用，往右黏邊延伸
                target_row = row
                target_col = col + 1
                while (target_row, target_col) in grid:
                    target_col += 1
                col = target_col
                row = target_row

        grid[(row, col)] = side
        positions.append({"i": idx, "side": side, "row": row, "col": col})
        last_side = side

    col_heights = Counter()
    col_sides: Dict[int, str] = {}
    for (r, c), side in grid.items():
        col_heights[c] += 1
        if r == 0:
            col_sides[c] = side

    max_col = max([p["col"] for p in positions], default=0)
    last_pos = positions[-1] if positions else {"i": -1, "side": "", "row": 0, "col": 0}

    return {
        "rows": rows,
        "grid": grid,
        "positions": positions,
        "col_heights": dict(col_heights),
        "col_sides": col_sides,
        "max_col": max_col,
        "last": last_pos,
    }


def _derived_color_at(layout: Dict[str, Any], pos: Dict[str, Any], offset: int) -> int:
    """
    衍生路紙紅藍簡化規則。
    回傳：1=紅，-1=藍，0=資料不足。
    用途：把大眼仔 / 小路 / 蟑螂路的「整齊或變化」量化。
    """
    col = int(pos.get("col", 0))
    row = int(pos.get("row", 0))
    heights = layout.get("col_heights", {})

    if col <= offset:
        return 0

    if row == 0:
        left_h = int(heights.get(col - 1, 0))
        compare_h = int(heights.get(col - 1 - offset, 0))
        if left_h == 0 or compare_h == 0:
            return 0
        return 1 if left_h == compare_h else -1

    # 同一欄向下時，看左側相對欄位是否同樣有該列；越整齊越偏紅
    has_left_same_row = ((row, col - offset) in layout.get("grid", {}))
    has_left_prev_row = ((row - 1, col - offset) in layout.get("grid", {}))
    if has_left_same_row == has_left_prev_row:
        return 1
    return -1


def _derived_series(layout: Dict[str, Any], offset: int) -> List[int]:
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
    """
    下三路獨立模型：
    - 紅偏多：路型整齊，偏向延續目前大路方向。
    - 藍偏多：路型變化，偏向反邊/斷點方向。
    offset=1 大眼仔；offset=2 小路；offset=3 蟑螂路。
    """
    default = {
        "B": 0.5, "P": 0.5, "label": f"{display_name}資料不足", "strength": 0.0,
        "road_key": road_key, "stats": {"last": 0, "red_rate": 0.5, "blue_rate": 0.5, "count": 0, "tail": ""},
        "red_pressure": 0.5, "blue_pressure": 0.5,
    }
    if not USE_ROAD_ENGINE or len(non_tie) < ROAD_ENGINE_MIN_HISTORY:
        return default

    layout = _build_big_road(non_tie)
    series = _derived_series(layout, offset=offset)
    stats = _color_stats(series)
    last_side, streak_n = _streak(non_tie)
    if not last_side:
        return default
    opp = "P" if last_side == "B" else "B"

    count = int(stats.get("count", 0))
    red_rate = float(stats.get("red_rate", 0.5))
    blue_rate = float(stats.get("blue_rate", 0.5))
    diff = abs(red_rate - blue_rate)

    if count < DERIVED_ROAD_MIN_COUNT:
        return {**default, "stats": stats, "label": f"{display_name}樣本不足"}

    recent = non_tie[-16:]
    switches = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
    switch_rate = _safe_div(switches, max(1, len(recent) - 1), 0.5)

    # 三條路敏感度不同：蟑螂路最短線，小路居中，大眼仔較穩。
    sensitivity = {1: 0.95, 2: 1.00, 3: 1.10}.get(offset, 1.0)
    base_edge = 0.024 + min(0.034, diff * 0.080 * sensitivity)

    if red_rate > blue_rate:
        side = last_side
        edge = base_edge + ROAD_ENGINE_RED_CONT_BIAS * sensitivity
        label = f"{display_name}紅路續勢"
    elif blue_rate > red_rate:
        side = opp
        edge = base_edge + ROAD_ENGINE_BLUE_BREAK_BIAS * sensitivity
        label = f"{display_name}藍路變化"
    else:
        side = opp if switch_rate >= 0.66 else last_side
        edge = 0.020
        label = f"{display_name}紅藍均衡"

    # 單跳盤時，下三路若藍偏多，短線反邊訊號加強；長龍時紅路延續加強。
    if switch_rate >= 0.72 and blue_rate >= 0.56:
        side = opp
        edge += 0.010 * sensitivity
        label = f"{display_name}跳路變化"
    if streak_n >= 4 and red_rate >= 0.58:
        side = last_side
        edge += 0.008 * sensitivity
        label = f"{display_name}紅路跟龍"

    edge = _clamp(edge, 0.012, 0.078)
    b = 0.5 + edge if side == "B" else 0.5 - edge
    p = 1 - b

    return {
        "B": b,
        "P": p,
        "label": label,
        "strength": round(0.10 + min(0.10, diff * 0.22) + min(0.04, count * 0.004), 4),
        "road_key": road_key,
        "stats": stats,
        "red_pressure": round(red_rate, 4),
        "blue_pressure": round(blue_rate, 4),
        "tail": stats.get("tail", ""),
    }


def _big_eye_score(non_tie: List[str]) -> Dict[str, Any]:
    return _derived_road_score(non_tie, offset=1, road_key="big_eye", display_name="大眼仔")


def _small_road_score(non_tie: List[str]) -> Dict[str, Any]:
    return _derived_road_score(non_tie, offset=2, road_key="small_road", display_name="小路")


def _cockroach_score(non_tie: List[str]) -> Dict[str, Any]:
    return _derived_road_score(non_tie, offset=3, road_key="cockroach", display_name="蟑螂路")


def _road_consensus_score(road_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """四路共識：大路、大眼仔、小路、蟑螂路各自投票後整合。"""
    vote_weights = _normalize_weights({
        "big_road": BIG_ROAD_WEIGHT if USE_ROAD_ENGINE else 0.0,
        "big_eye": BIG_EYE_WEIGHT if USE_ROAD_ENGINE else 0.0,
        "small_road": SMALL_ROAD_WEIGHT if USE_ROAD_ENGINE else 0.0,
        "cockroach": COCKROACH_WEIGHT if USE_ROAD_ENGINE else 0.0,
    })
    vote_score = {"B": 0.0, "P": 0.0}
    votes = []
    details = {}
    for name, score in road_scores.items():
        pick = _pick_from_score(score, min_edge=0.002)
        weight = vote_weights.get(name, 0.0)
        if pick:
            vote_score[pick] += weight
            votes.append(pick)
        details[name] = {
            "pick": pick,
            "weight": round(weight, 4),
            "label": score.get("label", ""),
            "B": round(float(score.get("B", 0.5)), 4),
            "P": round(float(score.get("P", 0.5)), 4),
        }

    if vote_score["B"] == vote_score["P"]:
        side = ""
    else:
        side = "B" if vote_score["B"] > vote_score["P"] else "P"

    total_vote = max(0.0001, vote_score["B"] + vote_score["P"])
    consensus_ratio = max(vote_score["B"], vote_score["P"]) / total_vote
    conflict_ratio = 1.0 - consensus_ratio

    b_raw = sum(float(road_scores[k].get("B", 0.5)) * vote_weights.get(k, 0.0) for k in road_scores)
    p_raw = sum(float(road_scores[k].get("P", 0.5)) * vote_weights.get(k, 0.0) for k in road_scores)
    b, p = (0.5, 0.5) if b_raw + p_raw <= 0 else (b_raw / (b_raw + p_raw), p_raw / (b_raw + p_raw))

    if side:
        label = f"四路共識:{'莊' if side == 'B' else '閒'} {int(consensus_ratio * 100)}%"
    else:
        label = "四路分歧"

    return {
        "B": b,
        "P": p,
        "label": label,
        "pick": side,
        "votes": votes,
        "vote_score": {"B": round(vote_score["B"], 4), "P": round(vote_score["P"], 4)},
        "consensus_ratio": round(consensus_ratio, 4),
        "conflict_ratio": round(conflict_ratio, 4),
        "details": details,
        "strength": round(0.10 + max(0.0, consensus_ratio - 0.5) * 0.30, 4),
    }


def _road_family_scores(non_tie: List[str]) -> Dict[str, Any]:
    """一次取得大路與下三路四個獨立模型 + 四路共識。"""
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
    新版真正參與融合的是 big_road / big_eye / small_road / cockroach 四個獨立模型。
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
def predict(history: List[str], venue: str = "", room: str = "", shoe_id: str = "", user_id: str = "") -> Dict[str, Any]:
    """
    整合預測函數：四路主模型 + Road Lifecycle + Adaptive Road Memory + Road Rhythm + NGram + 動態權重 + ML模型 + DeepSeek校準
    注意：本版加入低信心/四路分歧觀望機制；仍不做下注金額/EV 配注決策。
    """
    history = [str(x).upper() for x in history if str(x).upper() in {"B", "P", "T"}]
    non_tie = _last_non_tie(history)

    # ============ 1. 基礎模型 + 四路主模型 ==========
    markov = _transition_prob(non_tie)
    road = _road_pattern_score(non_tie)
    recent = _recent_score(non_tie)
    balance = _balance_score(non_tie)
    streak = _streak_score(non_tie)
    ngram = _ngram_score(non_tie)

    road_family = _road_family_scores(non_tie)
    big_road = road_family.get("big_road", {"B": 0.5, "P": 0.5, "label": "大路資料不足"})
    big_eye = road_family.get("big_eye", {"B": 0.5, "P": 0.5, "label": "大眼仔資料不足"})
    small_road = road_family.get("small_road", {"B": 0.5, "P": 0.5, "label": "小路資料不足"})
    cockroach = road_family.get("cockroach", {"B": 0.5, "P": 0.5, "label": "蟑螂路資料不足"})
    road_consensus = road_family.get("consensus", {"B": 0.5, "P": 0.5, "label": "四路共識資料不足"})
    road_engine = _road_engine_score(non_tie)  # 舊欄位相容用

    regime_info = _detect_regime(non_tie)
    online_performance = _rolling_model_performance(non_tie)
    lifecycle = _road_lifecycle_score(non_tie, road_family, regime_info)
    road_memory = _adaptive_road_memory_score(non_tie, road_family, lifecycle, regime_info)
    road_rhythm = _road_rhythm_score(non_tie, road_family, lifecycle, regime_info, road_memory)
    long_anchor = _long_anchor_score(non_tie, road_family, lifecycle, regime_info)
    dynamic_weights = _apply_online_weighting(regime_info.get("weights", {}), online_performance)
    dynamic_weights = _apply_lifecycle_weighting(dynamic_weights, lifecycle)
    dynamic_weights = _apply_road_memory_weighting(dynamic_weights, road_memory)
    dynamic_weights = _apply_road_rhythm_weighting(dynamic_weights, road_rhythm)

    total_w = sum(dynamic_weights.values()) or 1.0
    b_side = (
        big_road["B"] * dynamic_weights.get("big_road", 0.0)
        + big_eye["B"] * dynamic_weights.get("big_eye", 0.0)
        + small_road["B"] * dynamic_weights.get("small_road", 0.0)
        + cockroach["B"] * dynamic_weights.get("cockroach", 0.0)
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
    identity = str(user_id or "anonymous")
    training_key = f"{identity}|{venue}|{room}|{shoe_id}" if (venue or room or shoe_id) else f"{identity}|global"
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
        "road_lifecycle": lifecycle,
        "adaptive_road_memory": road_memory,
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
    for score in [big_road, big_eye, small_road, cockroach, ngram, markov, road, recent, streak, balance]:
        pick = _pick_from_score(score)
        if pick:
            votes.append(pick)

    if ml_models.is_trained:
        votes.append("B" if ml_b_prob >= 0.5 else "P")

    if not votes:
        votes = ["B" if b_prob >= p_prob else "P"]

    main_pick = "B" if b_prob >= p_prob else "P"
    agreement = votes.count(main_pick) / len(votes)

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

    if ALLOW_TIE_RECOMMEND and tie_prob >= TIE_RECOMMEND_MIN and tie_prob > max(b_prob, p_prob) * 0.55:
        recommend = "T"
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
        f"四路:{road_consensus.get('label', '')}",
        f"生命周期:{lifecycle.get('label', '')}",
        f"記憶:{road_memory.get('label', '')}",
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

    # ============ 8. 返回結果 ==========
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
        "road_lifecycle": lifecycle,
        "adaptive_road_memory": road_memory,
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
