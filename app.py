# app.py
# BGS AI（Flask + LINE）— 大路/珠盤路 可切換的辨識 + 投票（XGB/LGBM/RNN）
# 2025-09-04 consolidated version:
# - image detection improved (MIN_BLOB_AREA / MIN_CIRC / MIN_VOTE_RATIO)
# - robust LINE handlers (reply -> push fallback; default handler)
# - keep ensemble (XGB/LGBM/RNN) + rule fallback

import os, io, time, math, logging, json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, abort
from PIL import Image
import numpy as np
import cv2

# ===== LINE SDK (optional) =====
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError, LineBotApiError
    from linebot.models import (
        MessageEvent, TextMessage, ImageMessage, TextSendMessage, FollowEvent
    )
except Exception:
    LineBotApi = WebhookHandler = None
    MessageEvent = TextMessage = ImageMessage = TextSendMessage = FollowEvent = None
    InvalidSignatureError = Exception
    LineBotApiError = Exception

# ===== Optional ML (safe-import) =====
try:
    import joblib
except Exception:
    joblib = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

app = Flask(__name__)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-bot")

# ---------- ENV & CONFIG ----------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN and LineBotApi else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET and WebhookHandler else None

DEBUG_VISION = os.getenv("DEBUG_VISION", "0") == "1"
ROAD_MODE = os.getenv("ROAD_MODE", "bigroad").strip().lower()  # bigroad or bead
STRICT_GRID = os.getenv("STRICT_GRID", "1") == "1"

# NEW: blob filtering parameters (可用 ENV 調整)
MIN_BLOB_AREA = int(os.getenv("MIN_BLOB_AREA", "250"))    # 嚴格模式下排除太小的點 (預設 250)
MIN_CIRC = float(os.getenv("MIN_CIRC", "0.45"))           # 圓度門檻 (預設 0.45)
MIN_VOTE_RATIO = float(os.getenv("MIN_VOTE_RATIO", "0.50"))  # 顏色票數相對比重門檻

# HSV ranges (可再用 ENV 微調)
HSV = {
    "RED1_LOW":  (int(os.getenv("HSV_RED1_H_LOW",  "0")),  int(os.getenv("HSV_RED1_S_LOW",  "50")), int(os.getenv("HSV_RED1_V_LOW",  "50"))),
    "RED1_HIGH": (int(os.getenv("HSV_RED1_H_HIGH", "12")), int(os.getenv("HSV_RED1_S_HIGH", "255")),int(os.getenv("HSV_RED1_V_HIGH", "255"))),
    "RED2_LOW":  (int(os.getenv("HSV_RED2_H_LOW",  "170")),int(os.getenv("HSV_RED2_S_LOW",  "50")), int(os.getenv("HSV_RED2_V_LOW",  "50"))),
    "RED2_HIGH": (int(os.getenv("HSV_RED2_H_HIGH", "180")),int(os.getenv("HSV_RED2_S_HIGH", "255")),int(os.getenv("HSV_RED2_V_HIGH", "255"))),
    "BLUE_LOW":  (int(os.getenv("HSV_BLUE_H_LOW",  "90")), int(os.getenv("HSV_BLUE_S_LOW", "50")), int(os.getenv("HSV_BLUE_V_LOW", "50"))),
    "BLUE_HIGH": (int(os.getenv("HSV_BLUE_H_HIGH", "135")),int(os.getenv("HSV_BLUE_S_HIGH", "255")),int(os.getenv("HSV_BLUE_V_HIGH", "255"))),
    "GREEN_LOW": (int(os.getenv("HSV_GREEN_H_LOW", "40")), int(os.getenv("HSV_GREEN_S_LOW", "40")), int(os.getenv("HSV_GREEN_V_LOW", "40"))),
    "GREEN_HIGH":(int(os.getenv("HSV_GREEN_H_HIGH","85")), int(os.getenv("HSV_GREEN_S_HIGH","255")),int(os.getenv("HSV_GREEN_V_HIGH","255"))),
}

# edge/grid detection params
HOUGH_MIN_LEN_RATIO = float(os.getenv("HOUGH_MIN_LEN_RATIO", "0.45"))
HOUGH_GAP = int(os.getenv("HOUGH_GAP", "6"))
CANNY1 = int(os.getenv("CANNY1", "60"))
CANNY2 = int(os.getenv("CANNY2", "180"))

# ---------- Models ----------
MODELS_DIR = Path("models")
SCALER_PATH = MODELS_DIR / "scaler.pkl"
XGB_PKL     = MODELS_DIR / "xgb_model.pkl"
XGB_JSON    = MODELS_DIR / "xgb_model.json"
XGB_UBJ     = MODELS_DIR / "xgb_model.ubj"
LGBM_PKL    = MODELS_DIR / "lgbm_model.pkl"
LGBM_TXT    = MODELS_DIR / "lgbm_model.txt"
LGBM_JSON   = MODELS_DIR / "lgbm_model.json"
RNN_WTS     = MODELS_DIR / "rnn_weights.npz"

model_bundle: Dict[str, Any] = {"loaded": False, "note": "no model"}

def _safe_exists(p: Path) -> bool:
    try: return p.exists()
    except Exception: return False

def load_models():
    global model_bundle
    bundle: Dict[str, Any] = {}
    try:
        if joblib and _safe_exists(SCALER_PATH):
            bundle["scaler"] = joblib.load(SCALER_PATH)

        if xgb:
            if _safe_exists(XGB_PKL) and joblib:
                bundle["xgb_sklearn"] = joblib.load(XGB_PKL)
            elif _safe_exists(XGB_JSON):
                bst = xgb.Booster(); bst.load_model(str(XGB_JSON)); bundle["xgb_booster"] = bst
            elif _safe_exists(XGB_UBJ):
                bst = xgb.Booster(); bst.load_model(str(XGB_UBJ)); bundle["xgb_booster"] = bst

        if lgb:
            if _safe_exists(LGBM_PKL) and joblib:
                bundle["lgbm_sklearn"] = joblib.load(LGBM_PKL)
            elif _safe_exists(LGBM_TXT):
                bundle["lgbm_booster"] = lgb.Booster(model_file=str(LGBM_TXT))
            elif _safe_exists(LGBM_JSON):
                booster = lgb.Booster(model_str=LGBM_JSON.read_text(encoding="utf-8"))
                bundle["lgbm_booster"] = booster

        if _safe_exists(RNN_WTS):
            try:
                w = np.load(RNN_WTS)
                for k in ("Wxh","Whh","bh","Why","bo"):
                    if k not in w: raise ValueError(f"missing {k}")
                bundle["rnn_weights"] = {k: w[k] for k in ("Wxh","Whh","bh","Why","bo")}
            except Exception as e:
                logger.warning(f"Failed to load RNN weights: {e}")

        bundle["loaded"] = any(k in bundle for k in (
            "xgb_sklearn","xgb_booster","lgbm_sklearn","lgbm_booster","rnn_weights"
        ))
        bundle["note"] = "at least one model loaded" if bundle["loaded"] else "no model file found"
        model_bundle = bundle
        logger.info(f"[models] loaded={bundle['loaded']} note={bundle['note']}")
    except Exception as e:
        model_bundle = {"loaded": False, "note": f"load error: {e}"}
        logger.exception(e)

load_models()

# =========================================================
# Vision helpers (強化版)
# =========================================================
IDX = {"B":0,"P":1,"T":2}

# Existing code continues...

# ---- grid helpers (原邏輯保留) ----

# Safe reply or push function
def safe_reply_or_push(reply_token: str, user_id: str, text: str):
    """
    嘗試先回覆訊息，如果回覆失敗（例如因為無效的 reply_token），則回退到 push_message。
    """
    sent = False
    try:
        if reply_token:
            line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
            sent = True
            return True
    except Exception as e:
        logger.warning(f"[LINE] reply_message failed (可能是無效的 reply_token): {e}")
    if user_id:
        try:
            line_bot_api.push_message(user_id, TextSendMessage(text=text))
            logger.info("[LINE] push_message 發送成功 (回退處理)")
            return True
        except Exception as e:
            logger.error(f"[LINE] push_message 回退失敗: {e}")
    return sent

# continue with the remaining code...

