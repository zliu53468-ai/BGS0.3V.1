# app.py
# =========================================================
# BGS AI（Flask + LINE）— 牌路辨識 + 投票（XGB/LGBM/RNN）
# - 僅使用 XGBoost、LightGBM、RNN 參與投票（HMM / MLP 已不納入）
# - 加入「震盪偵測」＋「加權投票」＋「溫度校正」
# - 所有模型皆為可選載入；若皆不存在則回退規則法
# - 需要的檔案（有就載入，沒有就跳過）：
#   models/
#     ├─ scaler.pkl              # (可選) sklearn 標準化器，fit 在 build_features 的輸入
#     ├─ xgb_model.pkl/json/ubj  # (可選) XGBoost（sklearn or Booster）
#     ├─ lgbm_model.pkl/txt/json # (可選) LightGBM（sklearn or Booster）
#     └─ rnn_weights.npz         # (可選) numpy 權重：Wxh, Whh, bh, Why, bo
#
# 可調參數（環境變數）：
#   ENSEMBLE_WEIGHTS="xgb:0.40,lgb:0.30,rnn:0.30"
#   TEMP="0.95"      # softmax 溫度（<1 降低自信度；>1 更銳利）
#   MIN_SEQ="18"     # 序列過短時，改用規則法（或降低倉位）
#   ALT_WINDOW="20"  # 震盪偵測視窗長度
#   ALT_THRESH="0.70"# 震盪率門檻（交替次數/(N-1)）
#
# LINE_CHANNEL_ACCESS_TOKEN / LINE_CHANNEL_SECRET 需在環境變數提供。
# DEBUG_VISION=1 可印出解析細節。
# =========================================================

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

# --- Vision tuning (可由環境變數調) ---
DEBUG_VISION = os.getenv("DEBUG_VISION", "0") == "1"

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

HOUGH_MIN_LEN_RATIO = float(os.getenv("HOUGH_MIN_LEN_RATIO", "0.45"))  # ROI 寬度比例
HOUGH_GAP = int(os.getenv("HOUGH_GAP", "6"))
CANNY1 = int(os.getenv("CANNY1", "60"))
CANNY2 = int(os.getenv("CANNY2", "180"))

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
RNN_WTS     = MODELS_DIR / "rnn_weights.npz"    # numpy 權重：Wxh, Whh, bh, Why, bo

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

        # RNN 權重（純 numpy 前向）
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

        # 有任一模型即視為 loaded（僅考慮 xgb/lgb/rnn）
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
# 圖像→序列（紅=莊, 藍=閒；紅/藍圈內「橫線」=和；綠=和（若平台））
# =========================================================
IDX = {"B":0,"P":1,"T":2}

def _has_horizontal_line(roi_bgr: np.ndarray) -> bool:
    """在紅/藍圈 ROI 內檢測是否有近水平直線（判定為和局）。"""
    if roi_bgr is None or roi_bgr.size == 0:
        return False

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thr, CANNY1, CANNY2)

    h, w = edges.shape[:2]
    min_len = max(int(w * HOUGH_MIN_LEN_RATIO), 12)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                            minLineLength=min_len, maxLineGap=HOUGH_GAP)
    if lines is None:
        return False
    for x1, y1, x2, y2 in lines[:, 0, :]:
        if abs(y2 - y1) <= max(2, int(h * 0.12)):
            return True
    return False

def extract_sequence_from_image(img_bytes: bytes) -> List[str]:
    """回傳序列（最多 240 手）：'B', 'P', 'T'"""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h0, w0 = img.shape[:2]
        target = 1400.0
        scale = target / max(h0, w0) if max(h0, w0) < target else 1.0
        if scale > 1.0:
            img = cv2.resize(img, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_CUBIC)

        blur = cv2.GaussianBlur(img, (3,3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, HSV["RED1_LOW"],  HSV["RED1_HIGH"])
        red2 = cv2.inRange(hsv, HSV["RED2_LOW"],  HSV["RED2_HIGH"])
        red  = cv2.bitwise_or(red1, red2)
        blue = cv2.inRange(hsv, HSV["BLUE_LOW"],  HSV["BLUE_HIGH"])
        green= cv2.inRange(hsv, HSV["GREEN_LOW"],  HSV["GREEN_HIGH"])

        kernel3 = np.ones((3,3), np.uint8)
        def clean(m):
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel3, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel3, iterations=1)
            return m
        red, blue, green = clean(red), clean(blue), clean(green)

        def cc_blobs(mask, label):
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            items = []
            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]
            area_med = np.median(areas) if areas else 0
            min_area = max(80, int(area_med * 0.35))
            max_area = int(area_med * 8) if area_med > 0 else 999999

            for i in range(1, n):
                x, y, w, h, a = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
                if a < min_area or a > max_area:
                    continue
                aspect = w / (h + 1e-6)
                if not (0.5 <= aspect <= 2.0):
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
                if circularity < 0.3:
                    continue
                cx = x + w / 2.0
                items.append((x, y, w, h, cx, label))
            return items

        items = []
        items += cc_blobs(red,  "B")
        items += cc_blobs(blue, "P")
        items += cc_blobs(green,"T")

        if not items:
            return []

        items.sort(key=lambda z: z[4])
        widths = [w for _,_,w,_,_,_ in items]
        med_w  = np.median(widths) if widths else 12
        min_gap = max(med_w * 0.6, 10)

        seq: List[str] = []
        last_cx = -1e9

        for x,y,w0,h0,cx,label in items:
            if abs(cx - last_cx) < min_gap:
                continue

            if label in {"B","P"}:
                pad_x = max(2, int(w0 * 0.18))
                pad_y = max(2, int(h0 * 0.28))
                x1 = max(0, int(x + pad_x)); x2 = min(img.shape[1], int(x + w0 - pad_x))
                y1 = max(0, int(y + pad_y)); y2 = min(img.shape[0], int(y + h0 - pad_y))
                roi = img[y1:y2, x1:x2]
                if _has_horizontal_line(roi):
                    seq.append("T")
                else:
                    seq.append(label)
            else:
                seq.append("T")
            last_cx = cx

        if DEBUG_VISION:
            logger.info(f"[VISION] items={len(items)} widths_med={med_w:.1f} min_gap={min_gap:.1f} seq_len={len(seq)}")

        return seq[-240:]
    except Exception as e:
        if DEBUG_VISION:
            logger.exception(f"[VISION][ERR] {e}")
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

# 簡易 RNN（純 numpy 前向）：one-hot 輸入 → 隱藏狀態 → 線性輸出 → softmax 機率
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

def _parse_weights_env() -> Dict[str, float]:
    s = os.getenv("ENSEMBLE_WEIGHTS", "xgb:0.40,lgb:0.30,rnn:0.30")
    out = {"xgb":0.40,"lgb":0.30,"rnn":0.30}
    try:
        for kv in s.split(","):
            k,v = kv.split(":")
            k = k.strip().lower()
            v = float(v)
            if k in out:
                out[k] = max(0.0, v)
    except Exception:
        pass
    ss = sum(out.values()) or 1.0
    for k in out: out[k] /= ss
    return out

def predict_with_models(seq: List[str]) -> Tuple[Dict[str,float] | None, Dict[str,Any]]:
    """回傳 (機率, 附加資訊)；只用 XGB/LGB/RNN；加入溫度校正與震盪偵測。"""
    info = {"used":["xgb","lgb","rnn"], "oscillating": False, "alt_rate": 0.0}
    if not seq: return None, info

    # 震盪偵測
    ALT_WINDOW = int(os.getenv("ALT_WINDOW","20"))
    ALT_THRESH = float(os.getenv("ALT_THRESH","0.70"))
    alt_rate = _oscillation_rate(seq, ALT_WINDOW)
    info["alt_rate"] = round(alt_rate,3)
    info["oscillating"] = alt_rate >= ALT_THRESH

    # 序列過短判斷
    MIN_SEQ = int(os.getenv("MIN_SEQ","18"))
    if len([c for c in seq if c in ("B","P")]) < MIN_SEQ:
        return None, info  # 交給規則回退

    feat = build_features(seq)
    if "scaler" in model_bundle:
        try:
            feat = model_bundle["scaler"].transform(feat)
        except Exception as e:
            logger.warning(f"scaler.transform error: {e}")

    weights = _parse_weights_env()
    TEMP = float(os.getenv("TEMP","0.95"))

    # 個別模型機率
    preds = {}
    px = _proba_from_xgb(feat);  preds["xgb"]=px if px else None
    pl = _proba_from_lgb(feat);  preds["lgb"]=pl if pl else None
    pr = _proba_from_rnn(seq);   preds["rnn"]=pr if pr else None

    # 沒任何模型輸出→回退
    if not any(preds.values()):
        return None, info

    # 加權投票
    agg = {"banker":0.0,"player":0.0,"tie":0.0}
    wsum = 0.0
    for name,p in preds.items():
        if not p: continue
        w = weights.get(name, 0.0)
        wsum += w
        for k in agg:
            agg[k] += w * max(1e-9, float(p[k]))
    if wsum <= 0:
        return None, info

    # 溫度校正（對 B/P/T 同步縮放）
    vec = np.array([agg["banker"], agg["player"], agg["tie"]], dtype=np.float64)
    vec = _softmax(vec, temp=TEMP)
    out = {"banker": float(vec[0]), "player": float(vec[1]), "tie": float(vec[2])}

    # 震盪期：壓低勝率敘述（讓 betting_plan 產出更保守的倉位）
    if info["oscillating"]:
        for k in out:
            out[k] = float(0.85 * out[k])  # 整體降 15%，再由 betting_plan 判斷觀望/縮倉
        out = _normalize(out)
    return out, info

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
        boost = min(0.08, 0.025*(tail-1))  # 稍微保守
        if seq[-1]=="B": pb+=boost
        else: pp+=boost

    s=pb+pp+pt
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {"banker":round(pb/s,4),"player":round(pp/s,4),"tie":round(pt/s,4)}

def betting_plan(pb: float, pp: float, oscillating: bool) -> Dict[str, Any]:
    diff = abs(pb-pp)
    side = "莊" if pb >= pp else "閒"
    side_prob = max(pb, pp)

    # 震盪期更保守
    if oscillating:
        if diff < 0.10: return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "震盪期觀望"}
        if diff < 0.15: pct = 0.02
        elif diff < 0.20: pct = 0.04
        else: pct = 0.08
        return {"side": side, "percent": pct, "side_prob": side_prob, "note": "震盪期降倉"}

    # 非震盪：一般分層
    if diff < 0.05:
        return {"side": side, "percent": 0.0, "side_prob": side_prob, "note": "差距不足 5%，風險高"}
    if diff < 0.08: pct = 0.02
    elif diff < 0.12: pct = 0.04
    elif diff < 0.18: pct = 0.08
    else: pct = 0.12
    return {"side": side, "percent": pct, "side_prob": side_prob}

def render_reply(seq: List[str], probs: Dict[str,float], by_model: bool, info: Dict[str,Any] | None=None) -> str:
    b, p, t = probs["banker"], probs["player"], probs["tie"]
    oscillating = bool(info.get("oscillating")) if info else False
    plan = betting_plan(b, p, oscillating)
    tag = "（模型）" if by_model else "（規則）"
    win_txt = f"{plan['side_prob']*100:.1f}%"
    note = f"｜{plan['note']}" if plan.get("note") else ""
    bet_text = "觀望" if plan["percent"] == 0 else f"下 {plan['percent']*100:.0f}% 於「{plan['side']}」"
    osc_txt = f"\n震盪率：{info.get('alt_rate'):.2f}" if info and "alt_rate" in info else ""
    used_txt = f"\n投票模型：{', '.join(info.get('used', []))}" if info else ""
    return (
        f"{tag} 已解析 {len(seq)} 手{osc_txt}{used_txt}\n"
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
            msg = "已進入分析模式 ✅\n請上傳牌路截圖：我會嘗試自動辨識並回覆「建議下注：莊 / 閒（勝率 xx%）」"
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
        if not seq:
            tip = "辨識失敗 😥\n請確保截圖清楚包含大路，並避免過度縮放或模糊。"
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
