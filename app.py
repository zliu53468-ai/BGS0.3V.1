import os, io, time, math, logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, abort
from PIL import Image
import numpy as np
import cv2

# ===== LINE SDK =====
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage, FollowEvent
)

# ===== Optional ML =====
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

# ---------- ENV ----------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

DEBUG_VISION = os.getenv("DEBUG_VISION", "0") == "1"

# 颜色检测参数 - 更宽松的设置
HSV = {
    "RED1_LOW":  (int(os.getenv("HSV_RED1_H_LOW", "0")),   int(os.getenv("HSV_RED1_S_LOW", "50")), int(os.getenv("HSV_RED1_V_LOW", "50"))),
    "RED1_HIGH": (int(os.getenv("HSV_RED1_H_HIGH", "20")), int(os.getenv("HSV_RED1_S_HIGH", "255")), int(os.getenv("HSV_RED1_V_HIGH", "255"))),
    "RED2_LOW":  (int(os.getenv("HSV_RED2_H_LOW", "160")), int(os.getenv("HSV_RED2_S_LOW", "50")), int(os.getenv("HSV_RED2_V_LOW", "50"))),
    "RED2_HIGH": (int(os.getenv("HSV_RED2_H_HIGH", "180")), int(os.getenv("HSV_RED2_S_HIGH", "255")), int(os.getenv("HSV_RED2_V_HIGH", "255"))),
    "BLUE_LOW":  (int(os.getenv("HSV_BLUE_H_LOW", "90")), int(os.getenv("HSV_BLUE_S_LOW", "50")), int(os.getenv("HSV_BLUE_V_LOW", "50"))),
    "BLUE_HIGH": (int(os.getenv("HSV_BLUE_H_HIGH", "140")), int(os.getenv("HSV_BLUE_S_HIGH", "255")), int(os.getenv("HSV_BLUE_V_HIGH", "255"))),
    "GREEN_LOW": (int(os.getenv("HSV_GREEN_H_LOW", "50")),  int(os.getenv("HSV_GREEN_S_LOW", "30")),  int(os.getenv("HSV_GREEN_V_LOW", "30"))),
    "GREEN_HIGH":(int(os.getenv("HSV_GREEN_H_HIGH", "90")), int(os.getenv("HSV_GREEN_S_HIGH", "255")), int(os.getenv("HSV_GREEN_V_HIGH", "255"))),
}

# ---------- User session ----------
user_mode: Dict[str, bool] = {}   # user_id -> True/False

# ---------- 模型載入 ----------
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
    try:
        return p.exists()
    except Exception:
        return False

def load_models():
    global model_bundle
    bundle: Dict[str, Any] = {}
    try:
        if joblib and _safe_exists(SCALER_PATH):
            bundle["scaler"] = joblib.load(SCALER_PATH)
            logger.info("[models] loaded scaler")

        if xgb:
            if _safe_exists(XGB_PKL) and joblib:
                bundle["xgb_sklearn"] = joblib.load(XGB_PKL)
                logger.info("[models] loaded xgb (sklearn)")
            elif _safe_exists(XGB_JSON):
                bst = xgb.Booster(); bst.load_model(str(XGB_JSON))
                bundle["xgb_booster"] = bst
                logger.info("[models] loaded xgb booster (json)")
            elif _safe_exists(XGB_UBJ):
                bst = xgb.Booster(); bst.load_model(str(XGB_UBJ))
                bundle["xgb_booster"] = bst
                logger.info("[models] loaded xgb booster (ubj)")

        if lgb:
            if _safe_exists(LGBM_PKL) and joblib:
                bundle["lgbm_sklearn"] = joblib.load(LGBM_PKL)
                logger.info("[models] loaded lgbm (sklearn)")
            elif _safe_exists(LGBM_TXT):
                bundle["lgbm_booster"] = lgb.Booster(model_file=str(LGBM_TXT))
                logger.info("[models] loaded lgbm booster (txt)")
            elif _safe_exists(LGBM_JSON):
                booster = lgb.Booster(model_str=LGBM_JSON.read_text(encoding="utf-8"))
                bundle["lgbm_booster"] = booster
                logger.info("[models] loaded lgbm booster (json)")

        if _safe_exists(RNN_WTS):
            try:
                w = np.load(RNN_WTS)
                for key in ("Wxh", "Whh", "bh", "Why", "bo"):
                    if key not in w:
                        raise ValueError(f"missing {key}")
                bundle["rnn_weights"] = {k: w[k] for k in ("Wxh","Whh","bh","Why","bo")}
                logger.info("[models] loaded RNN weights")
            except Exception as e:
                logger.warning(f"Failed to load RNN weights: {e}")

        bundle["loaded"] = any(k in bundle for k in (
            "xgb_sklearn", "xgb_booster", "lgbm_sklearn", "lgbm_booster", "rnn_weights"
        ))
        bundle["note"] = "at least one model loaded" if bundle["loaded"] else "no model file found"
        model_bundle = bundle
        logger.info(f"[models] loaded={bundle['loaded']} note={bundle['note']}")
    except Exception as e:
        model_bundle = {"loaded": False, "note": f"load error: {e}"}
        logger.exception(f"[models] load error: {e}")

load_models()

# =========================================================
# 影像→序列（嚴格只讀大路）
# =========================================================
IDX = {"B":0,"P":1,"T":2}

def _has_horizontal_line(roi_bgr: np.ndarray) -> bool:
    """在紅/藍圈 ROI 內檢測是否有近水平直線（判定為和局）"""
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    
    # 轉換為灰度圖並增強對比度
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # 使用自適應閾值
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形態學操作增強線條
    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    
    # Hough變換檢測直線
    h, w = thr.shape[:2]
    min_len = max(int(w * 0.4), 15)
    lines = cv2.HoughLinesP(thr, 1, np.pi/180, threshold=25,
                           minLineLength=min_len, maxLineGap=4)
    
    if lines is None:
        return False
    
    # 檢查是否有接近水平的線
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if (angle < 15 or angle > 165) and abs(y2-y1) < h*0.2:
            return True
    
    return False

def extract_sequence_from_image(img_bytes: bytes) -> List[str]:
    """專注於分析紅藍區域，精確識別莊閒和"""
    try:
        # 圖像預處理
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        
        # 圖像增強
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 使用手動ROI或自動檢測
        roi_env = os.getenv("FOCUS_ROI", "")
        if roi_env:
            try:
                sx, sy, sw, sh = [float(t) for t in roi_env.split(",")]
                rx = int(sx * W)
                ry = int(sy * H)
                rw = int(sw * W)
                rh = int(sh * H)
                roi = img[ry:ry+rh, rx:rx+rw]
                logger.info(f"Using manual ROI: {rx},{ry},{rw},{rh}")
            except Exception as e:
                logger.error(f"Error parsing FOCUS_ROI: {e}")
                roi = img
        else:
            # 自動檢測紅藍密集區域
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            red1 = cv2.inRange(hsv, HSV["RED1_LOW"], HSV["RED1_HIGH"])
            red2 = cv2.inRange(hsv, HSV["RED2_LOW"], HSV["RED2_HIGH"])
            red = cv2.bitwise_or(red1, red2)
            blue = cv2.inRange(hsv, HSV["BLUE_LOW"], HSV["BLUE_HIGH"])
            
            combined = cv2.bitwise_or(red, blue)
            kernel = np.ones((15, 15), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            
            # 找到最大輪廓
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                # 擴大一點邊界
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(W - x, w + 2 * padding)
                h = min(H - y, h + 2 * padding)
                roi = img[y:y+h, x:x+w]
                logger.info(f"Using auto-detected ROI: {x},{y},{w},{h}")
            else:
                roi = img
                logger.info("Using full image as ROI")
        
        # 在ROI中精確識別紅藍綠
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv_roi, HSV["RED1_LOW"], HSV["RED1_HIGH"])
        red2 = cv2.inRange(hsv_roi, HSV["RED2_LOW"], HSV["RED2_HIGH"])
        red = cv2.bitwise_or(red1, red2)
        blue = cv2.inRange(hsv_roi, HSV["BLUE_LOW"], HSV["BLUE_HIGH"])
        green = cv2.inRange(hsv_roi, HSV["GREEN_LOW"], HSV["GREEN_HIGH"])
        
        # 形態學處理
        kernel = np.ones((5, 5), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
        
        # 查找輪廓
        def find_colored_circles(mask, color, min_area=50):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circles = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:  # 降低面積限制
                    continue
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                # 降低圓形度要求
                circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2) if cv2.arcLength(cnt, True) > 0 else 0
                if circularity < 0.5:  # 降低圓形度要求
                    continue
                circles.append((int(x), int(y), int(radius), color, area))
            return circles

        red_circles = find_colored_circles(red, "B", min_area=30)
        blue_circles = find_colored_circles(blue, "P", min_area=30)
        green_circles = find_colored_circles(green, "T", min_area=30)
        
        all_circles = red_circles + blue_circles + green_circles
        
        logger.info(f"Detected circles - Red: {len(red_circles)}, Blue: {len(blue_circles)}, Green: {len(green_circles)}")
        
        # 按位置排序（從左到右，從上到下）
        if all_circles:
            # 計算平均間距
            xs = [c[0] for c in all_circles]
            xs.sort()
            if len(xs) > 1:
                x_gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
                avg_x_gap = np.median(x_gaps) if x_gaps else 50
            else:
                avg_x_gap = 50
                
            ys = [c[1] for c in all_circles]
            ys.sort()
            if len(ys) > 1:
                y_gaps = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
                avg_y_gap = np.median(y_gaps) if y_gaps else 50
            else:
                avg_y_gap = 50
                
            # 按網格位置排序
            all_circles.sort(key=lambda c: (c[0] // avg_x_gap, c[1] // avg_y_gap))
        else:
            all_circles.sort(key=lambda c: (c[0] // 50, c[1] // 50))
        
        # 轉換為序列
        sequence = []
        for x, y, radius, label, area in all_circles:
            if label in {"B","P"}:
                pad_x = max(2, int(radius * 0.5))
                pad_y = max(2, int(radius * 0.5))
                x1 = max(0, x - pad_x); x2 = min(roi.shape[1], x + pad_x)
                y1 = max(0, y - pad_y); y2 = min(roi.shape[0], y + pad_y)
                sub = roi[y1:y2, x1:x2]
                if _has_horizontal_line(sub): 
                    sequence.append("T")
                else:
                    sequence.append(label)
            else:
                sequence.append("T")
        
        logger.info(f"Final sequence length: {len(sequence)}")
        return sequence[-120:]  # 返回最近120個結果
        
    except Exception as e:
        logger.error(f"圖像處理錯誤: {e}")
        return []

# =========================================================
# 特徵工程 / 震盪偵測 / 投票
# =========================================================
def _streak_tail(seq: List[str]) -> int:
    if not seq: return 0
    t, c = seq[-1], 1
    for i in range(len(seq)-2, -1, -1):
        if seq[i]==t: c+=1
        else: break
    return c

def _transitions(seq: List[str]) -> np.ndarray:
    m = np.zeros((3,3), dtype=np.float32)
    for a,b in zip(seq, seq[1:]):
        if a in IDX and b in IDX:
            m[IDX[a], IDX[b]] += 1
    if m.sum()>0: m = m/(m.sum()+1e-6)
    return m.flatten()

def _ratio_lastN(seq: List[str], N: int) -> Tuple[float,float,float]:
    s = seq[-N:] if len(seq)>=N else seq
    if not s: return (0.33,0.33,0.34)
    n = len(s); return (s.count("B")/n, s.count("P")/n, s.count("T")/n)

def build_features(seq: List[str]) -> np.ndarray:
    n = len(seq)
    pb, pp, pt = _ratio_lastN(seq, n)
    b10,p10,t10 = _ratio_lastN(seq,10)
    b20,p20,t20 = _ratio_lastN(seq,20)
    streak = _streak_tail(seq)
    last = np.zeros(3); last[IDX[seq[-1]]] = 1.0
    trans = _transitions(seq)
    entropy = 0.0
    for v in [pb,pp,pt]:
        if v>1e-9: entropy -= v*math.log(v+1e-9)
    feat = np.array([n,pb,pp,pt,b10,p10,t10,b20,p20,t20,streak,entropy,*last,*trans], dtype=np.float32).reshape(1,-1)
    return feat

def _normalize(p: Dict[str,float]) -> Dict[str,float]:
    p = {k: max(1e-9, float(v)) for k,v in p.items()}
    s = p["banker"]+p["player"]+p["tie"]
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {k: round(v/s,4) for k,v in p.items()}

def _softmax(x: np.ndarray, temp: float=1.0) -> np.ndarray:
    x = x.astype(np.float64) / max(1e-9, temp)
    m = np.max(x)
    e = np.exp(x - m)
    return e / (np.sum(e) + 1e-12)

def _oscillation_rate(seq: List[str], win: int) -> float:
    """交替率：近 win 手，莊/閒交錯的比例（不含和）。"""
    s = [c for c in seq[-win:] if c in ("B","P")]
    if len(s) < 2: return 0.0
    alt = sum(1 for a,b in zip(s, s[1:]) if a != b)
    return alt / (len(s)-1)

def _alt_streak_suffix(seq: List[str]) -> int:
    """近端連續單跳長度（只看 B/P），例：...B P B P B → 5"""
    s = [c for c in seq if c in ("B","P")]
    if len(s) < 2: return 0
    k = 1
    for i in range(len(s)-2, -1, -1):
        if s[i] != s[i+1]:
            k += 1
        else:
            break
    return k

def _parse_weights_env_pair() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    讀兩組投票權重：
      ENSEMBLE_WEIGHTS_TREND="xgb:0.45,lgb:0.35,rnn:0.20"
      ENSEMBLE_WEIGHTS_CHOP ="xgb:0.20,lgb:0.25,rnn:0.55"
    """
    def _parse(s: str, default: Dict[str,float]) -> Dict[str,float]:
        out = default.copy()
        try:
            for kv in s.split(","):
                k,v = kv.split(":")
                k = k.strip().lower()
                v = float(v)
                if k in out: out[k] = max(0.0, v)
        except Exception:
            pass
        ss = sum(out.values()) or 1.0
        for k in out: out[k] /= ss
        return out

    trend_def = {"xgb":0.45,"lgb":0.35,"rnn":0.20}
    chop_def  = {"xgb":0.20,"lgb":0.25,"rnn":0.55}
    w_trend = _parse(os.getenv("ENSEMBLE_WEIGHTS_TREND",""), trend_def)
    w_chop  = _parse(os.getenv("ENSEMBLE_WEIGHTS_CHOP",""),  chop_def)
    return w_trend, w_chop

def _proba_from_xgb(feat: np.ndarray) -> Dict[str,float] | None:
    if "xgb_sklearn" in model_bundle:
        proba = model_bundle["xgb_sklearn"].predict_proba(feat)[0]
        return {"banker": float(proba[IDX["B"]]), "player": float(proba[IDX["P"]]), "tie": float(proba[IDX["T"]])}
    if "xgb_booster" in model_bundle and xgb:
        d = xgb.DMatrix(feat); proba = model_bundle["xgb_booster"].predict(d)[0]
        if len(proba)==3:
            return {"banker": float(proba[0]), "player": float(proba[1]), "tie": float(proba[2])}
    return None

def _proba_from_lgb(feat: np.ndarray) -> Dict[str,float] | None:
    if "lgbm_sklearn" in model_bundle:
        proba = model_bundle["lgbm_sklearn"].predict_proba(feat)[0]
        return {"banker": float(proba[IDX["B"]]), "player": float(proba[IDX["P"]]), "tie": float(proba[IDX["T"]])}
    if "lgbm_booster" in model_bundle and lgb:
        proba = model_bundle["lgbm_booster"].predict(feat)[0]
        if isinstance(proba,(list,np.ndarray)) and len(proba)==3:
            return {"banker": float(proba[0]), "player": float(proba[1]), "tie": float(proba[2])}
    return None

def _proba_from_rnn(seq: List[str]) -> Dict[str,float] | None:
    w = model_bundle.get("rnn_weights")
    if w is None or not seq:
        return None
    try:
        Wxh = np.array(w["Wxh"]); Whh = np.array(w["Whh"]); bh = np.array(w["bh"])
        Why = np.array(w["Why"]); bo = np.array(w["bo"])
        h = np.zeros((Whh.shape[0],), dtype=np.float32)
        for s in seq:
            x = np.zeros((3,), dtype=np.float32)
            x[IDX.get(s, 2)] = 1.0
            h = np.tanh(x @ Wxh + h @ Whh + bh)
        o = h @ Why + bo  # (3,)
        prob = _softmax(o, temp=1.0)
        return {"banker": float(prob[0]), "player": float(prob[1]), "tie": float(prob[2])}
    except Exception as e:
        logger.warning(f"Error using RNN weights: {e}")
        return None

def predict_with_models(seq: List[str]) -> Tuple[Dict[str,float] | None, Dict[str,Any]]:
    """
    只用 XGB/LGB/RNN；加入：
      - 單跳偵測：alt_rate + alt_streak_suffix
      - Regime 權重：Trend / Chop 兩組
      - 溫度校正 + Chop 降自信（再 normalize）
    """
    info = {"used":["xgb","lgb","rnn"], "oscillating": False, "alt_rate": 0.0, "alt_streak": 0}
    if not seq: return None, info

    ALT_WINDOW = int(os.getenv("ALT_WINDOW","20"))
    ALT_THRESH = float(os.getenv("ALT_THRESH","0.70"))
    ALT_STRICT = int(os.getenv("ALT_STRICT_STREAK","5"))

    alt_rate   = _oscillation_rate(seq, ALT_WINDOW)
    alt_streak = _alt_streak_suffix(seq)
    info["alt_rate"]   = round(alt_rate,3)
    info["alt_streak"] = int(alt_streak)
    is_chop = (alt_rate >= ALT_THRESH) or (alt_streak >= ALT_STRICT)
    info["oscillating"] = is_chop

    MIN_SEQ = int(os.getenv("MIN_SEQ","5"))  # 降低最小序列長度要求
    if len([c for c in seq if c in ("B","P")]) < MIN_SEQ:
        return None, info

    feat = build_features(seq)
    if "scaler" in model_bundle:
        try:
            feat = model_bundle["scaler"].transform(feat)
        except Exception as e:
            logger.warning(f"scaler.transform error: {e}")

    w_trend, w_chop = _parse_weights_env_pair()
    weights = w_chop if is_chop else w_trend
    TEMP = float(os.getenv("TEMP","0.95"))

    preds = {}
    px = _proba_from_xgb(feat);  preds["xgb"]=px if px else None
    pl = _proba_from_lgb(feat);  preds["lgb"]=pl if pl else None
    pr = _proba_from_rnn(seq);   preds["rnn"]=pr if pr else None
    if not any(preds.values()):
        return None, info

    agg = {"banker":0.0,"player":0.0,"tie":0.0}
    wsum = 0.0
    for name,p in preds.items():
        if not p: continue
        w = weights.get(name, 0.0)
        wsum += w
        for k in agg: agg[k] += w * max(1e-9, float(p[k]))
    if wsum <= 0:
        return None, info

    vec = np.array([agg["banker"], agg["player"], agg["tie"]], dtype=np.float64)
    vec = _softmax(vec, temp=TEMP)
    if is_chop:
        vec = 0.88 * vec + 0.12 * np.array([1/3,1/3,1/3], dtype=np.float64)
    out = {"banker": float(vec[0]), "player": float(vec[1]), "tie": float(vec[2])}
    return _normalize(out), info

# ------------------- 規則回退 -------------------
def predict_probs_from_seq_rule(seq: List[str]) -> Dict[str,float]:
    n=len(seq)
    if n==0: return {"banker":0.33,"player":0.33,"tie":0.34}
    pb = seq.count("B")/n
    pp = seq.count("P")/n
    pt = max(0.02, seq.count("T")/n*0.6)

    tail=1
    for i in range(n-2,-1,-1):
        if seq[i]==seq[-1]: tail+=1
        else: break
    if seq[-1] in {"B","P"}:
        boost = min(0.08, 0.025*(tail-1))
        if seq[-1]=="B": pb+=boost
        else: pp+=boost

    s=pb+pp+pt
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {"banker":round(pb/s,4),"player":round(pp/s,4),"tie":round(pt/s,4)}

def betting_plan(pb: float, pp: float, oscillating: bool, alt_streak: int=0) -> Dict[str, Any]:
    diff = abs(pb-pp)
    side = "banker" if pb >= pp else "player"
    side_prob = max(pb, pp)

    ALT_STRICT = int(os.getenv("ALT_STRICT_STREAK","5"))
    if oscillating and alt_streak >= ALT_STRICT:
        return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "單跳震盪期觀望"}

    if oscillating:
        if diff < 0.12: return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "震盪期風險高"}
        if diff < 0.18: pct = 0.02
        elif diff < 0.24: pct = 0.04
        else: pct = 0.08
        return {"side": side, "percent": pct, "side_prob": side_prob, "note": "震盪期降倉"}

    if diff < 0.05:
        return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "差距不足 5%"}
    if diff < 0.08: pct = 0.02
    elif diff < 0.12: pct = 0.04
    elif diff < 0.18: pct = 0.08
    else: pct = 0.12
    return {"side": side, "percent": pct, "side_prob": side_prob}

def render_reply(seq: List[str], probs: Dict[str,float], by_model: bool, info: Dict[str,Any] | None=None) -> str:
    b, p, t = probs["banker"], probs["player"], probs["tie"]
    oscillating = bool(info.get("oscillating")) if info else False
    alt_streak = int(info.get("alt_streak", 0)) if info else 0
    plan = betting_plan(b, p, oscillating, alt_streak)
    
    # 按照您要求的格式
    side = "莊" if plan["side"] == "banker" else "閒"
    win_rate = plan["side_prob"] * 100
    
    reply = f"推薦預測：{side}（勝率{win_rate:.1f}%）\n\n"
    reply += f"解析路數：{len(seq)}手\n"
    reply += f"莊勝率：{b*100:.1f}% | 閒勝率：{p*100:.1f}% | 和局率：{t*100:.1f}%\n"
    
    if plan["percent"] > 0:
        reply += f"建議下注：{plan['percent']*100:.0f}%資金於{side}"
    else:
        reply += "建議：觀望不下注"
    
    if info and info.get("oscillating"):
        reply += f"\n當前牌路震盪中（交替率：{info.get('alt_rate', 0):.2f}）"
    
    return reply

# =========================================================
# API（可自測）
# =========================================================
@app.route("/")
def index():
    return "BGS AI 助手正在運行 ✅ /line-webhook 已就緒", 200

@app.route("/health")
def health():
    return jsonify({
        "status":"ok",
        "ts":int(time.time()),
        "models_loaded": model_bundle.get("loaded", False),
        "have": {
            "xgb_sklearn": "xgb_sklearn" in model_bundle,
            "xgb_booster": "xgb_booster" in model_bundle,
            "lgbm_sklearn":"lgbm_sklearn" in model_bundle,
            "lgbm_booster":"lgbm_booster" in model_bundle,
            "rnn_weights":"rnn_weights" in model_bundle,
            "scaler":"scaler" in model_bundle
        },
        "note": model_bundle.get("note","")
    })

# =========================================================
# LINE Webhook
# =========================================================
@app.route("/line-webhook", methods=['POST'])
def line_webhook():
    if not (line_bot_api and line_handler):
        logger.error("LINE creds missing: ACCESS_TOKEN or SECRET not set")
        abort(403)
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    logger.info(f"/line-webhook called, sig_len={len(signature)}, body_len={len(body)}")
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError as e:
        logger.exception(f"InvalidSignatureError: {e}. "
                         f"==> 通常是 LINE_CHANNEL_SECRET 不對 或 用錯 Channel 的 Secret/Token")
        return "Invalid signature", 200
    except Exception as e:
        logger.exception(f"Unhandled error while handling webhook: {e}")
        return "Error", 200
    return "OK"

if line_handler and line_bot_api:

    @line_handler.add(FollowEvent)
    def on_follow(event: FollowEvent):
        welcome = (
            "歡迎加入BGS AI 助手 🎉\n\n"
            "輸入「開始分析」後，上傳牌路截圖，我會自動辨識並回傳建議下注：莊 / 閒（勝率 xx%）。"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome))

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        txt = (event.message.text or "").strip()
        if txt in {"開始分析", "開始", "START", "分析"}:
            user_mode[uid] = True
            msg = "已進入分析模式 ✅\n請上傳牌路截圖：我會嘗試自動辨識並回覆「推薦預測：莊 / 閒（勝率 xx%）」"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))
            return
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text="請先輸入「開始分析」，再上傳牌路截圖。"
        ))

    @line_handler.add(MessageEvent, message=ImageMessage)
    def on_image(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        if not user_mode.get(uid):
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="尚未啟用分析模式。\n請先輸入「開始分析」，再上傳牌路截圖。"
            ))
            return

        content = line_bot_api.get_message_content(event.message.id)
        img_bytes = b"".join(chunk for chunk in content.iter_content())
        seq = extract_sequence_from_image(img_bytes)
        if not seq or len(seq) < 5:  # 如果識別的手數太少
            tip = f"辨識失敗 😥 只識別到 {len(seq)} 手\n請確保截圖清楚包含大路，背景單純，避免過度縮放或模糊。\n建議嘗試：\n1. 確保光線充足\n2. 對準牌路區域\n3. 避免反光"
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tip)); return

        if model_bundle.get("loaded"):
            probs, info = predict_with_models(seq)
            by_model = probs is not None
            if not by_model:
                probs = predict_probs_from_seq_rule(seq); info = {}
        else:
            probs = predict_probs_from_seq_rule(seq); by_model=False; info = {}

        msg = render_reply(seq, probs, by_model, info)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
