# app.py - 整合版 (牌路分析 + LINE + 機器學習)
import os, io, time, math, logging, json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, abort
from PIL import Image
import numpy as np
import cv2

# ===== LINE SDK =====
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError, LineBotApiError
    from linebot.models import (
        MessageEvent, TextMessage, ImageMessage, TextSendMessage, FollowEvent
    )
except Exception:
    LineBotApi = WebhookHandler = None
    logger.warning("LINE SDK not available")

# ===== 機器學習套件 =====
try:
    import joblib
except Exception:
    joblib = None
    logger.warning("joblib not available")

try:
    import xgboost as xgb
except Exception:
    xgb = None
    logger.warning("xgboost not available")

try:
    import lightgbm as lgb
except Exception:
    lgb = None
    logger.warning("lightgbm not available")

app = Flask(__name__)

# ---------- 日誌設定 ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-bot")

# ---------- 環境變數設定 ----------
# LINE 設定
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

# 牌路分析設定
ROAD_HEIGHT_RATIO = float(os.getenv("ROAD_HEIGHT_RATIO", "0.18"))  # 牌路佔畫面高度比例
MIN_BLOB_AREA = int(os.getenv("MIN_BLOB_AREA", "50"))   # 最小斑點面積
MIN_CIRC = float(os.getenv("MIN_CIRC", "0.3"))          # 最小圓形度
DEBUG_VISION = os.getenv("DEBUG_VISION", "0") == "1"    # 視覺化偵錯

# 機器學習設定
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/ensemble_model.pkl")

# HSV 顏色範圍 (可透過環境變數調整)
HSV = {
    "RED_LOW": (int(os.getenv("HSV_RED_LOW_H", "0")), int(os.getenv("HSV_RED_LOW_S", "100")), int(os.getenv("HSV_RED_LOW_V", "100"))),
    "RED_HIGH": (int(os.getenv("HSV_RED_HIGH_H", "10")), int(os.getenv("HSV_RED_HIGH_S", "255")), int(os.getenv("HSV_RED_HIGH_V", "255"))),
    "BLUE_LOW": (int(os.getenv("HSV_BLUE_LOW_H", "100")), int(os.getenv("HSV_BLUE_LOW_S", "100")), int(os.getenv("HSV_BLUE_LOW_V", "100"))),
    "BLUE_HIGH": (int(os.getenv("HSV_BLUE_HIGH_H", "130")), int(os.getenv("HSV_BLUE_HIGH_S", "255")), int(os.getenv("HSV_BLUE_HIGH_V", "255"))),
    "GREEN_LOW": (int(os.getenv("HSV_GREEN_LOW_H", "35")), int(os.getenv("HSV_GREEN_LOW_S", "100")), int(os.getenv("HSV_GREEN_LOW_V", "100"))),
    "GREEN_HIGH": (int(os.getenv("HSV_GREEN_HIGH_H", "85")), int(os.getenv("HSV_GREEN_HIGH_S", "255")), int(os.getenv("HSV_GREEN_HIGH_V", "255"))),
}

# ===== 初始化服務 =====
# LINE 服務
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LineBotApi and LINE_CHANNEL_ACCESS_TOKEN else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if WebhookHandler and LINE_CHANNEL_SECRET else None

# 機器學習模型
ml_model = None
if joblib and os.path.exists(ML_MODEL_PATH):
    try:
        ml_model = joblib.load(ML_MODEL_PATH)
        logger.info(f"Loaded ML model from {ML_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load ML model: {str(e)}")

# ===== 牌路分析功能 =====
def focus_road_area(img: np.ndarray) -> np.ndarray:
    """截取底部牌路區域"""
    height, width = img.shape[:2]
    road_height = int(height * ROAD_HEIGHT_RATIO)
    return img[height - road_height:, 0:width]

def detect_markers(road_img: np.ndarray) -> List[Dict]:
    """識別牌路中的莊/閒/和局標記"""
    hsv = cv2.cvtColor(road_img, cv2.COLOR_BGR2HSV)
    markers = []
    
    # 優先檢測和局標記 (綠色)
    green_mask = cv2.inRange(hsv, HSV["GREEN_LOW"], HSV["GREEN_HIGH"])
    green_kps = find_blobs(green_mask)
    for kp in green_kps:
        markers.append({
            "type": "TIE",
            "position": (int(kp.pt), int(kp.pt)),
            "size": kp.size
        })
    
    # 檢測莊標記 (紅色)
    red_mask = cv2.inRange(hsv, HSV["RED_LOW"], HSV["RED_HIGH"])
    red_kps = find_blobs(red_mask)
    for kp in red_kps:
        if not is_covered_by_tie(kp, green_kps):
            markers.append({
                "type": "BANKER",
                "position": (int(kp.pt), int(kp.pt)),
                "size": kp.size
            })
    
    # 檢測閒標記 (藍色)
    blue_mask = cv2.inRange(hsv, HSV["BLUE_LOW"], HSV["BLUE_HIGH"])
    blue_kps = find_blobs(blue_mask)
    for kp in blue_kps:
        if not is_covered_by_tie(kp, green_kps):
            markers.append({
                "type": "PLAYER",
                "position": (int(kp.pt), int(kp.pt)),
                "size": kp.size
            })
    
    return markers

def find_blobs(mask: np.ndarray) -> List:
    """在遮罩上尋找斑點"""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = MIN_BLOB_AREA
    params.filterByCircularity = True
    params.minCircularity = MIN_CIRC
    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(mask)

def is_covered_by_tie(kp, tie_kps, distance_threshold=15) -> bool:
    """檢查標記是否被和局標記覆蓋"""
    for tie_kp in tie_kps:
        distance = math.sqrt((kp.pt-tie_kp.pt)**2 + (kp.pt-tie_kp.pt)**2)
        if distance < distance_threshold:
            return True
    return False

def visualize_results(img: np.ndarray, markers: List[Dict]) -> np.ndarray:
    """視覺化分析結果"""
    road_img = focus_road_area(img)
    vis_img = road_img.copy()
    
    for marker in markers:
        x, y = marker["position"]
        size = int(marker["size"])
        color = (0, 255, 0) if marker["type"] == "TIE" else \
                (0, 0, 255) if marker["type"] == "BANKER" else \
                (255, 0, 0)
        
        cv2.rectangle(vis_img, (x-size, y-size), (x+size, y+size), color, 2)
        cv2.putText(vis_img, marker["type"], (x-size, y-size-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis_img

# ===== 機器學習預測功能 =====
def predict_next_result(markers: List[Dict]) -> str:
    """使用ML模型預測下一局結果"""
    if not ml_model:
        return "ML model not available"
    
    try:
        # 將標記轉換為特徵向量 (這裡需要根據實際模型輸入調整)
        # 範例: 使用最近5個結果作為特徵
        last_results = [m["type"] for m in markers[-5:]]  # 取首字母 B/P/T
        while len(last_results) < 5:
            last_results.insert(0, "N")  # 用N補足長度
            
        # 這裡需要根據實際模型要求轉換特徵
        # 假設模型需要數值特徵
        feature_map = {"B": 0, "P": 1, "T": 2, "N": -1}
        features = [feature_map[r] for r in last_results]
        
        prediction = ml_model.predict([features])
        return ["BANKER", "PLAYER", "TIE"][prediction]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Prediction failed"

# ===== LINE 訊息處理 =====
if line_handler:
    @app.route("/callback", methods=["POST"])
    def callback():
        """LINE Webhook 回調"""
        signature = request.headers["X-Line-Signature"]
        body = request.get_data(as_text=True)
        
        try:
            line_handler.handle(body, signature)
        except InvalidSignatureError:
            abort(400)
        return "OK"

    @line_handler.add(MessageEvent, message=ImageMessage)
    def handle_image_message(event):
        """處理LINE圖片訊息"""
        try:
            # 取得LINE圖片
            message_content = line_bot_api.get_message_content(event.message.id)
            img_bytes = b""
            for chunk in message_content.iter_content():
                img_bytes += chunk
            
            # 轉換為OpenCV格式
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # 分析牌路
            road_img = focus_road_area(img)
            markers = detect_markers(road_img)
            
            # 機器學習預測
            prediction = predict_next_result(markers)
            
            # 回覆用戶
            reply_text = f"分析完成！共找到 {len(markers)} 個標記\n"
            reply_text += f"最近5局: {[m['type'] for m in markers[-5:]]}\n"
            reply_text += f"預測下一局: {prediction}"
            
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_text)
            )
            
            # 可選: 發送分析結果圖片
            if DEBUG_VISION:
                vis_img = visualize_results(img, markers)
                _, img_encoded = cv2.imencode('.jpg', vis_img)
                line_bot_api.push_message(
                    event.source.user_id,
                    ImageSendMessage(
                        original_content_url=f"{HOST_URL}/debug_img.jpg",
                        preview_image_url=f"{HOST_URL}/debug_img.jpg"
                    )
                )
                
        except LineBotApiError as e:
            logger.error(f"LINE API error: {str(e)}")
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")

    @line_handler.add(MessageEvent, message=TextMessage)
    def handle_text_message(event):
        """處理LINE文字訊息"""
        try:
            if event.message.text.lower() == "分析":
                # 要求用戶發送圖片
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="請發送遊戲截圖進行分析")
                )
            else:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="請發送遊戲截圖或輸入「分析」")
                )
        except LineBotApiError as e:
            logger.error(f"LINE API error: {str(e)}")

# ===== API 端點 =====
@app.route('/analyze', methods=['POST'])
def analyze_image():
    """API分析端點"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # 分析牌路
    road_img = focus_road_area(img)
    markers = detect_markers(road_img)
    
    # 機器學習預測
    prediction = predict_next_result(markers)
    
    # 偵錯圖像
    debug_url = None
    if DEBUG_VISION:
        vis_img = visualize_results(img, markers)
        cv2.imwrite("static/debug_result.jpg", vis_img)
        debug_url = f"{request.host_url}static/debug_result.jpg"
    
    return jsonify({
        "markers_found": len(markers),
        "recent_results": [m["type"] for m in markers[-10:]],
        "prediction": prediction,
        "debug_image": debug_url
    })

# ===== 主程式 =====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
