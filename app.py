# app.py
import os, io, time, math, logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, abort
from PIL import Image
import numpy as np
import cv2

# ===== LINE SDK =====
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
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

try:
    from hmmlearn.hmm import MultinomialHMM
except Exception:
    MultinomialHMM = None

app = Flask(__name__)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-bot")

# ---------- ENV ----------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

# --- Vision tuning (可由環境變數調) ---
DEBUG_VISION = os.getenv("DEBUG_VISION", "0") == "1"

# 只鎖定「底部大路白底格子帶」+ 左側大路區
ROI_BAND_RATIO = float(os.getenv("ROI_BAND_RATIO", "0.48"))      # 取影像底部這一段比例做搜尋(0.35~0.60)
ROI_BIGROAD_FRAC = float(os.getenv("ROI_BIGROAD_FRAC", "0.62"))  # 在找到的格子區，保留靠左這一段(大路)
ROI_TOP_PAD = int(os.getenv("ROI_TOP_PAD", "6"))                  # 往內縮，避免邊緣文字
ROI_BOTTOM_PAD = int(os.getenv("ROI_BOTTOM_PAD", "10"))
ROI_MIN_H = int(os.getenv("ROI_MIN_H", "160"))                    # ROI 太矮則視為失敗

HSV = {
    "RED1_LOW":  (int(os.getenv("HSV_RED1_H_LOW",  "0")),  int(os.getenv("HSV_RED1_S_LOW",  "50")), int(os.getenv("HSV_RED1_V_LOW",  "50"))),
    "RED1_HIGH": (int(os.getenv("HSV_RED1_H_HIGH", "12")), int(os.getenv("HSV_RED1_S_HIGH", "255")),int(os.getenv("HSV_RED1_V_HIGH", "255"))),
    "RED2_LOW":  (int(os.getenv("HSV_RED2_H_LOW",  "170")),int(os.getenv("HSV_RED2_S_LOW",  "50")), int(os.getenv("HSV_RED2_V_LOW",  "50"))),
    "RED2_HIGH": (int(os.getenv("HSV_RED2_H_HIGH", "180")),int(os.getenv("HSV_RED2_S_HIGH", "255")),int(os.getenv("HSV_RED2_V_HIGH", "255"))),
    "BLUE_LOW":  (int(os.getenv("HSV_BLUE_H_LOW",  "90")), int(os.getenv("HSV_BLUE_S_LOW",  "50")), int(os.getenv("HSV_BLUE_V_LOW",  "50"))),
    "BLUE_HIGH": (int(os.getenv("HSV_BLUE_H_HIGH", "135")),int(os.getenv("HSV_BLUE_S_HIGH", "255")),int(os.getenv("HSV_BLUE_V_HIGH", "255"))),
    "GREEN_LOW": (int(os.getenv("HSV_GREEN_H_LOW", "40")), int(os.getenv("HSV_GREEN_S_LOW", "40")), int(os.getenv("HSV_GREEN_V_LOW", "40"))),
    "GREEN_HIGH":(int(os.getenv("HSV_GREEN_H_HIGH","85")), int(os.getenv("HSV_GREEN_S_HIGH","255")),int(os.getenv("HSV_GREEN_V_HIGH","255"))),
}

# Tie 橫線偵測
HOUGH_MIN_LEN_RATIO = float(os.getenv("HOUGH_MIN_LEN_RATIO", "0.50"))  # ROI 寬度比例(可動)
HOUGH_GAP = int(os.getenv("HOUGH_GAP", "5"))
CANNY1 = int(os.getenv("CANNY1", "60"))
CANNY2 = int(os.getenv("CANNY2", "180"))

# 規則法 T 權重縮放（可微調環境變數）
T_SHRINK = float(os.getenv("RULE_T_SHRINK", "0.6"))

# ---------- User session & state ----------
user_mode: Dict[str, bool] = {}      # 是否進入分析模式
user_state: Dict[str, Dict[str, Any]] = {}  # 平滑/慣性：每位使用者上一筆的機率與建議

SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA", "0.55"))       # 機率EMA平滑
KEEP_SIDE_MARGIN = float(os.getenv("KEEP_SIDE_MARGIN", "0.06"))  # 建議慣性門檻(差距小於此值就不換邊)
FLIP_GUARD = float(os.getenv("FLIP_GUARD", "0.08"))              # 若要換邊需至少超過此差距

# ---------- 模型載入 ----------
MODELS_DIR = Path("models")
SCALER_PATH = MODELS_DIR / "scaler.pkl"
XGB_PKL     = MODELS_DIR / "xgb_model.pkl"
XGB_JSON    = MODELS_DIR / "xgb_model.json"
XGB_UBJ     = MODELS_DIR / "xgb_model.ubj"
LGBM_PKL    = MODELS_DIR / "lgbm_model.pkl"
LGBM_TXT    = MODELS_DIR / "lgbm_model.txt"
LGBM_JSON   = MODELS_DIR / "lgbm_model.json"
HMM_PKL     = MODELS_DIR / "hmm_model.pkl"

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

        if xgb:
            if _safe_exists(XGB_PKL) and joblib:
                bundle["xgb_sklearn"] = joblib.load(XGB_PKL)
            elif _safe_exists(XGB_JSON):
                bst = xgb.Booster(); bst.load_model(str(XGB_JSON))
                bundle["xgb_booster"] = bst
            elif _safe_exists(XGB_UBJ):
                bst = xgb.Booster(); bst.load_model(str(XGB_UBJ))
                bundle["xgb_booster"] = bst

        if lgb:
            if _safe_exists(LGBM_PKL) and joblib:
                bundle["lgbm_sklearn"] = joblib.load(LGBM_PKL)
            elif _safe_exists(LGBM_TXT):
                bundle["lgbm_booster"] = lgb.Booster(model_file=str(LGBM_TXT))
            elif _safe_exists(LGBM_JSON):
                booster = lgb.Booster(model_str=LGBM_JSON.read_text(encoding="utf-8"))
                bundle["lgbm_booster"] = booster

        if MultinomialHMM and joblib and _safe_exists(HMM_PKL):
            hmm = joblib.load(HMM_PKL)
            if hasattr(hmm, "n_components") and hmm.n_components == 3:
                bundle["hmm"] = hmm

        bundle["loaded"] = any(k in bundle for k in (
            "xgb_sklearn", "xgb_booster", "lgbm_sklearn", "lgbm_booster", "hmm"
        ))
        bundle["note"] = "at least one model loaded" if bundle["loaded"] else "no model file found"
        model_bundle = bundle
        logger.info(f"[models] loaded={bundle['loaded']} note={bundle['note']}")
    except Exception as e:
        model_bundle = {"loaded": False, "note": f"load error: {e}"}
        logger.exception(f"[models] load error: {e}")

load_models()

# =========================================================
# 影像前處理：只截取「大路」區域
# =========================================================
def _auto_find_grid_band(bgr: np.ndarray) -> Tuple[int,int,int,int] | None:
    """
    嘗試在影像底部找到白底格子的大區塊，回傳 (x,y,w,h)。
    找不到時回傳 None。
    """
    h, w = bgr.shape[:2]
    y0 = max(0, int(h * (1.0 - ROI_BAND_RATIO)))
    band = bgr[y0:h, :]

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 2)

    # 尋找水平與垂直線（格線）
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, band.shape[0] // 20)))
    h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, band.shape[1] // 40), 1))
    vlines = cv2.dilate(cv2.erode(bw, v_ker, 1), v_ker, 1)
    hlines = cv2.dilate(cv2.erode(bw, h_ker, 1), h_ker, 1)
    grid = cv2.bitwise_or(vlines, hlines)

    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # 取最大面積的矩形區（通常就是格子區）
    cnt = max(cnts, key=cv2.contourArea)
    x, y, gw, gh = cv2.boundingRect(cnt)
    # 略縮放內縮，去除數字/統計列
    y1 = y + ROI_TOP_PAD
    y2 = y + gh - ROI_BOTTOM_PAD
    if y2 - y1 < ROI_MIN_H:
        return None
    return (x, y0 + y1, gw, y2 - y1)

def _find_big_road_roi(bgr: np.ndarray) -> np.ndarray:
    """
    找到底部格子帶後，僅保留左側大路部分；若偵測失敗回到保守裁切。
    """
    h, w = bgr.shape[:2]
    rect = _auto_find_grid_band(bgr)
    if rect is None:
        # 保守法：直接取底部一帶 + 左側一段
        y0 = int(h * (1.0 - ROI_BAND_RATIO))
        band = bgr[y0:h, :]
        if band.shape[0] < ROI_MIN_H:
            return bgr  # 退回全圖
        x1 = 0
        x2 = max(10, int(band.shape[1] * ROI_BIGROAD_FRAC))
        return band[:, x1:x2]

    x, y, gw, gh = rect
    big_w = max(10, int(gw * ROI_BIGROAD_FRAC))
    roi = bgr[y:y+gh, x:x+big_w]
    if roi.shape[0] < ROI_MIN_H or roi.shape[1] < 40:
        return bgr  # 防呆
    return roi

# =========================================================
# 圖像→序列（紅=莊B, 藍=閒P；圈內水平線=和T）
# =========================================================
IDX = {"B":0,"P":1,"T":2}

def _has_horizontal_line(roi_bgr: np.ndarray) -> bool:
    """在紅/藍圈 ROI 內檢測是否有近水平直線（判定為和局）。"""
    if roi_bgr is None or roi_bgr.size == 0:
        return False

    # 對比增強 + 去噪
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    enh = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thr, CANNY1, CANNY2)

    h, w = edges.shape[:2]
    # 允許以 env 設定，但最低使用 0.35 以提升召回
    local_min_len_ratio = max(0.35, HOUGH_MIN_LEN_RATIO)
    min_len = max(int(w * local_min_len_ratio), 8)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=18,
                            minLineLength=min_len, maxLineGap=max(3, HOUGH_GAP))
    if lines is None:
        return False

    # 線段 y 位置需落在 ROI 中段，避免圈邊/雜線
    y_mid = h * 0.5
    y_tol = h * 0.28

    for x1, y1, x2, y2 in lines[:, 0, :]:
        if abs(y2 - y1) <= max(2, int(h * 0.12)):
            if abs(((y1 + y2) * 0.5) - y_mid) <= y_tol:
                return True
    return False

def extract_sequence_from_image(img_bytes: bytes) -> List[str]:
    """
    回傳序列（最多 240 手）：'B', 'P', 'T'
    僅針對大路區塊做解析，降低「總手數/問路/派彩區」干擾。
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 小圖放大，避免圈太小
        h0, w0 = img.shape[:2]
        target = 1500.0
        scale = target / max(h0, w0) if max(h0, w0) < target else 1.0
        if scale > 1.0:
            img = cv2.resize(img, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_CUBIC)

        # 只保留「底部格子帶的左側大路」
        img = _find_big_road_roi(img)

        # 降噪 + 顏色空間
        blur = cv2.GaussianBlur(img, (3,3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # 顏色遮罩
        red1 = cv2.inRange(hsv, HSV["RED1_LOW"],  HSV["RED1_HIGH"])
        red2 = cv2.inRange(hsv, HSV["RED2_LOW"],  HSV["RED2_HIGH"])
        red  = cv2.bitwise_or(red1, red2)
        blue = cv2.inRange(hsv, HSV["BLUE_LOW"],  HSV["BLUE_HIGH"])
        green= cv2.inRange(hsv, HSV["GREEN_LOW"], HSV["GREEN_HIGH"])

        # 形態學：先 close 補洞，再 open 去雜訊
        kernel3 = np.ones((3,3), np.uint8)
        def clean(m):
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel3, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel3, iterations=1)
            return m
        red, blue, green = clean(red), clean(blue), clean(green)

        # 以 connected components 取得穩定 blob（加入圓度/面積/長寬比過濾）
        def cc_blobs(mask, label):
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            items = []
            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]
            area_med = np.median(areas) if areas else 0
            # 動態面積門檻；避免小圖被固定 90 卡死
            dyn_min = int(max(40, area_med * 0.35))
            dyn_max = int(area_med * 8.5) if area_med > 0 else 999999
            min_area = dyn_min
            max_area = dyn_max

            for i in range(1, n):
                x, y, w, h, a = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
                if a < min_area or a > max_area:
                    continue
                aspect = w / (h + 1e-6)
                if not (0.55 <= aspect <= 1.8):
                    continue
                c = (labels == i).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                cnt = max(cnts, key=cv2.contourArea)
                per = cv2.arcLength(cnt, True)
                area = cv2.contourArea(cnt)
                if per <= 0 or area <= 0:
                    continue
                circularity = 4 * math.pi * area / (per * per)
                if circularity < 0.35:  # 提高圓度要求，排除數字與斜線
                    continue

                cx = x + w / 2.0
                cy = y + h / 2.0
                items.append((x, y, w, h, cx, cy, label))
            return items

        items = []
        items += cc_blobs(red,  "B")
        items += cc_blobs(blue, "P")
        items += cc_blobs(green,"T")  # 若平台用綠色單獨標和

        if not items:
            return []

        # 依 x 中心排序（大路由左至右）
        items.sort(key=lambda z: z[4])

        # 動態間距（避免同一格重複）：以直徑近似的寬度中位數為基準
        widths = [w for _,_,w,_,_,_,_ in items]
        med_w  = np.median(widths) if widths else 12
        if med_w < 2: med_w = 12

        # 以「網格量化」方式去重：把 (cx,cy) 量化到離散 cell
        step_x = max(int(med_w * 0.95), 10)
        step_y = max(int(med_w * 0.90), 8)
        seen_cells = set()

        # x0 起點改為低分位數，避免第一顆不是首欄導致位移
        cx_all = [it[4] for it in items]
        x0 = float(np.percentile(cx_all, 5)) if cx_all else items[0][4]

        seq: List[str] = []
        for x,y,w0,h0,cx,cy,label in items:
            col = int(round((cx - x0) / step_x))
            row = int(round(cy / step_y))
            cell = (col, row)
            # 允許小幅偏移的去重（包含左鄰/右鄰，上下鄰）
            near_hit = any((col+dx, row+dy) in seen_cells for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            if near_hit:
                continue

            if label in {"B","P"}:
                # 取較小的內部 ROI 檢水平線，避免邊界干擾
                pad_x = max(2, int(w0 * 0.20))
                pad_y = max(2, int(h0 * 0.30))
                x1 = max(0, int(x + pad_x)); x2 = min(img.shape[1], int(x + w0 - pad_x))
                y1 = max(0, int(y + pad_y)); y2 = min(img.shape[0], int(y + h0 - pad_y))
                roi = img[y1:y2, x1:x2]
                if _has_horizontal_line(roi):
                    seq.append("T")
                else:
                    seq.append(label)
            else:
                seq.append("T")

            seen_cells.add(cell)

        if DEBUG_VISION:
            logger.info(f"[VISION] items={len(items)} med_w={med_w:.1f} "
                        f"step=({step_x},{step_y}) cells={len(seen_cells)} seq_len={len(seq)}")

        return seq[-240:]
    except Exception as e:
        if DEBUG_VISION:
            logger.exception(f"[VISION][ERR] {e}")
        return []

# =========================================================
# 特徵工程 & 模型推理 & 規則回退
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
    feat = np.array([n,pb,pp,pt,b10,p10,t10,b20,p20,t20,streak,entropy,*last,*trans],
                    dtype=np.float32).reshape(1,-1)
    return feat

def _normalize(p: Dict[str,float]) -> Dict[str,float]:
    s = p["banker"]+p["player"]+p["tie"]
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {k: round(v/s,4) for k,v in p.items()}

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

def _proba_from_hmm(seq: List[str]) -> Dict[str,float] | None:
    hmm = model_bundle.get("hmm")
    if not hmm or not seq: return None
    sym = {"B":0,"P":1,"T":2}
    base = np.array([[sym[s]] for s in seq], dtype=np.int32)
    scores=[]
    for cand in ["B","P","T"]:
        test = np.vstack([base, [[sym[cand]]]])
        try:
            logp = hmm.score(test, lengths=[len(test)])
        except Exception:
            logp = -1e9
        scores.append(logp)
    m = max(scores); exps = np.exp(np.array(scores)-m)
    prob = exps/(exps.sum()+1e-12)
    return {"banker": float(prob[0]), "player": float(prob[1]), "tie": float(prob[2])}

def predict_with_models(seq: List[str]) -> Dict[str,float] | None:
    feat = build_features(seq)
    if "scaler" in model_bundle:
        feat = model_bundle["scaler"].transform(feat)
    votes=[]
    p=_proba_from_xgb(feat);  votes.append(p) if p else None
    p=_proba_from_lgb(feat);  votes.append(p) if p else None
    p=_proba_from_hmm(seq);   votes.append(p) if p else None
    if not votes: return None
    avg={"banker":0.0,"player":0.0,"tie":0.0}
    for v in votes:
        for k in avg: avg[k]+=v[k]
    for k in avg: avg[k]/=len(votes)
    return _normalize(avg)

def predict_probs_from_seq_rule(seq: List[str]) -> Dict[str,float]:
    n=len(seq)
    if n==0: return {"banker":0.33,"player":0.33,"tie":0.34}
    pb = seq.count("B")/n
    pp = seq.count("P")/n
    pt_all = seq.count("T")/n
    # 近 20 手加權，避免早期 T 影響過大；再乘以可調縮放
    _,_,pt20 = _ratio_lastN(seq, 20)
    pt = max(0.02, 0.5*pt_all + 0.5*pt20) * T_SHRINK
    # 尾端連續加權
    tail = _streak_tail(seq)
    if seq[-1] in {"B","P"}:
        boost = min(0.10, 0.03*(tail-1))
        if seq[-1]=="B": pb+=boost
        else: pp+=boost
    s=pb+pp+pt
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {"banker":round(pb/s,4),"player":round(pp/s,4),"tie":round(pt/s,4)}

# ===== 平滑與慣性（避免忽左忽右） =====
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _renorm(p: Dict[str,float]) -> Dict[str,float]:
    s = p["banker"] + p["player"] + p["tie"] + 1e-12
    for k in p:
        p[k] = _clamp01(p[k] / s)
    return p

def _smooth_and_hysteresis(uid: str, probs: Dict[str, float]) -> Dict[str, float]:
    st = user_state.get(uid, {})
    last = st.get("probs")
    if last:
        smoothed = {
            "banker": SMOOTH_ALPHA*probs["banker"] + (1-SMOOTH_ALPHA)*last["banker"],
            "player": SMOOTH_ALPHA*probs["player"] + (1-SMOOTH_ALPHA)*last["player"],
            "tie":    SMOOTH_ALPHA*probs["tie"]    + (1-SMOOTH_ALPHA)*last["tie"],
        }
    else:
        smoothed = probs.copy()

    # 慣性：若差距很小就延續上次；若要換邊需超過 FLIP_GUARD
    side_last = st.get("side")
    diff = abs(smoothed["banker"] - smoothed["player"])

    if side_last is not None:
        if diff < KEEP_SIDE_MARGIN:
            # 保持上次
            if side_last == "B":
                smoothed["banker"] = max(smoothed["banker"], smoothed["player"])
                smoothed["player"] = max(0.0, 1.0 - smoothed["banker"] - smoothed["tie"])
            else:
                smoothed["player"] = max(smoothed["player"], smoothed["banker"])
                smoothed["banker"] = max(0.0, 1.0 - smoothed["player"] - smoothed["tie"])
            smoothed = _renorm(smoothed)
        else:
            # 想換邊但差距不夠大 → 仍保持
            turn_to_P = (side_last == "B" and smoothed["player"] > smoothed["banker"] and diff < FLIP_GUARD)
            turn_to_B = (side_last == "P" and smoothed["banker"] > smoothed["player"] and diff < FLIP_GUARD)
            if turn_to_P or turn_to_B:
                if side_last == "B":
                    smoothed["banker"] = smoothed["player"] + FLIP_GUARD
                else:
                    smoothed["player"] = smoothed["banker"] + FLIP_GUARD
                smoothed = _renorm(smoothed)

    # 更新狀態
    user_state[uid] = {"probs": smoothed, "side": "B" if smoothed["banker"]>=smoothed["player"] else "P"}
    return smoothed

def betting_plan(pb: float, pp: float) -> Dict[str, Any]:
    diff = abs(pb-pp)
    side = "莊" if pb >= pp else "閒"
    side_prob = max(pb, pp)
    if diff < 0.05:
        return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "差距不足 5%，風險高"}
    if diff < 0.08: pct = 0.02
    elif diff < 0.12: pct = 0.04
    elif diff < 0.18: pct = 0.08
    else: pct = 0.12
    return {"side": side, "percent": pct, "side_prob": side_prob}

def render_reply(seq: List[str], probs: Dict[str,float], by_model: bool) -> str:
    b, p, t = probs["banker"], probs["player"], probs["tie"]
    plan = betting_plan(b, p)
    tag = "（模型）" if by_model else "（規則）"
    win_txt = f"{plan['side_prob']*100:.1f}%"
    note = f"｜{plan['note']}" if plan.get("note") else ""
    bet_text = "觀望" if plan["percent"] == 0 else f"下 {plan['percent']*100:.0f}% 於「{plan['side']}」"
    return (
        f"{tag} 已解析 {len(seq)} 手（僅大路）\n"
        f"建議下注：{plan['side']}（勝率 {win_txt}）{note}\n"
        f"機率：莊 {b:.2f}｜閒 {p:.2f}｜和 {t:.2f}\n"
        f"資金建議：{bet_text}"
    )

# =========================================================
# API（可自測）
# =========================================================
@app.route("/")
def index():
    return "BGS AI 助手正在運行 ✅ /line-webhook 已就緒", 200

@app.route("/health")
def health():
    return jsonify({"status":"ok","ts":int(time.time()),
                    "models_loaded": model_bundle.get("loaded", False),
                    "note": model_bundle.get("note","")})

# =========================================================
# LINE Webhook
# =========================================================
def _safe_reply(event, message: TextSendMessage):
    """避免 Invalid reply token：若 reply 失敗則嘗試 push。"""
    try:
        line_bot_api.reply_message(event.reply_token, message)
    except LineBotApiError as e:
        if "Invalid reply token" in str(e) and getattr(event.source, "user_id", None):
            try:
                line_bot_api.push_message(event.source.user_id, message)
                logger.warning("reply->push fallback due to Invalid reply token")
            except Exception as e2:
                logger.exception(f"push_message failed: {e2}")
        else:
            logger.exception(f"reply_message failed: {e}")

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
            "輸入「開始分析」後，上傳牌路截圖，我會只針對『大路』區自動辨識並回傳建議：莊/閒（勝率 xx%）。"
        )
        _safe_reply(event, TextSendMessage(text=welcome))

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        txt = (event.message.text or "").strip()
        if txt in {"開始分析", "開始", "START", "分析"}:
            user_mode[uid] = True
            msg = "已進入分析模式 ✅\n請上傳牌路截圖（僅解析『大路』）。"
            _safe_reply(event, TextSendMessage(text=msg))
            return
        _safe_reply(event, TextSendMessage(text="請先輸入「開始分析」，再上傳牌路截圖。"))

    @line_handler.add(MessageEvent, message=ImageMessage)
    def on_image(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        if not user_mode.get(uid):
            _safe_reply(event, TextSendMessage(
                text="尚未啟用分析模式。\n請先輸入「開始分析」，再上傳牌路截圖。"
            ))
            return

        # 下載圖片 → 解析序列 → 推理 → 回覆
        content = line_bot_api.get_message_content(event.message.id)
        img_bytes = b"".join(chunk for chunk in content.iter_content())

        seq = extract_sequence_from_image(img_bytes)
        if not seq:
            tip = ("辨識失敗 😥\n請確保截圖清楚包含『大路』，並避免過度縮放或模糊。")
            _safe_reply(event, TextSendMessage(text=tip))
            return

        # 模型 or 規則
        if model_bundle.get("loaded"):
            probs = predict_with_models(seq); by_model = probs is not None
            if not by_model:
                probs = predict_probs_from_seq_rule(seq)
        else:
            probs = predict_probs_from_seq_rule(seq); by_model = False

        # 平滑 + 慣性（依 user）
        probs = _smooth_and_hysteresis(uid, probs)
        msg = render_reply(seq, probs, by_model)
        _safe_reply(event, TextSendMessage(text=msg))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
