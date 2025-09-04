# app.py
# BGS AI（Flask + LINE）— 珠盤路/大路辨識（嚴格參數可調）+ 投票（XGB/LGBM/RNN）
import os, io, time, math, json, logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from flask import Flask, request, jsonify, abort
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# LINE SDK (若沒用就會跳過 webhook)
try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import (
        MessageEvent, TextMessage, ImageMessage, TextSendMessage, FollowEvent
    )
except Exception:
    LineBotApi = None
    WebhookHandler = None
    MessageEvent = TextMessage = ImageMessage = TextSendMessage = FollowEvent = None
    InvalidSignatureError = Exception

# Optional ML
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

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-bot")

# LINE env
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN and LineBotApi else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET and WebhookHandler else None

# Debug flag
DEBUG_VISION = os.getenv("DEBUG_VISION", "1") == "1"   # 開發階段預設打開

# HSV color ranges (可用環境變數微調)
HSV = {
    "RED1_LOW":  (int(os.getenv("HSV_RED1_H_LOW","0")),  int(os.getenv("HSV_RED1_S_LOW","50")), int(os.getenv("HSV_RED1_V_LOW","50"))),
    "RED1_HIGH": (int(os.getenv("HSV_RED1_H_HIGH","12")), int(os.getenv("HSV_RED1_S_HIGH","255")),int(os.getenv("HSV_RED1_V_HIGH","255"))),
    "RED2_LOW":  (int(os.getenv("HSV_RED2_H_LOW","170")),int(os.getenv("HSV_RED2_S_LOW","50")), int(os.getenv("HSV_RED2_V_LOW","50"))),
    "RED2_HIGH": (int(os.getenv("HSV_RED2_H_HIGH","180")),int(os.getenv("HSV_RED2_S_HIGH","255")),int(os.getenv("HSV_RED2_V_HIGH","255"))),
    "BLUE_LOW":  (int(os.getenv("HSV_BLUE_H_LOW","90")), int(os.getenv("HSV_BLUE_S_LOW","50")), int(os.getenv("HSV_BLUE_V_LOW","50"))),
    "BLUE_HIGH": (int(os.getenv("HSV_BLUE_H_HIGH","135")),int(os.getenv("HSV_BLUE_S_HIGH","255")),int(os.getenv("HSV_BLUE_V_HIGH","255"))),
    "GREEN_LOW": (int(os.getenv("HSV_GREEN_H_LOW","40")), int(os.getenv("HSV_GREEN_S_LOW","40")), int(os.getenv("HSV_GREEN_V_LOW","40"))),
    "GREEN_HIGH":(int(os.getenv("HSV_GREEN_H_HIGH","85")), int(os.getenv("HSV_GREEN_S_HIGH","255")),int(os.getenv("HSV_GREEN_V_HIGH","255"))),
}

# Canny / Hough params
HOUGH_MIN_LEN_RATIO = float(os.getenv("HOUGH_MIN_LEN_RATIO", "0.45"))
HOUGH_GAP = int(os.getenv("HOUGH_GAP", "6"))
CANNY1 = int(os.getenv("CANNY1", "60"))
CANNY2 = int(os.getenv("CANNY2", "180"))

# Road mode: bigroad / bead
ROAD_MODE = os.getenv("ROAD_MODE", "bigroad").strip().lower()

# Strict grid and detection thresholds (你要微調就改 ENV)
STRICT_GRID = os.getenv("STRICT_GRID", "1") == "1"
MIN_COLS = int(os.getenv("MIN_COLS", "6"))
MIN_ITEMS = int(os.getenv("MIN_ITEMS", "8"))

# New detection tuning params (for strict filtering)
MIN_AREA = int(os.getenv("MIN_AREA","200"))         # 最小 blob 面積（嚴格模式）
CIRC_THRESH = float(os.getenv("CIRC_THRESH","0.44"))  # 圓度閾值（嚴格模式）
MIN_VOTE_RATIO = float(os.getenv("MIN_VOTE_RATIO","0.55"))  # 顏色投票比（label 決勝）

# ML model paths (同先前)
MODELS_DIR = Path("models")
SCALER_PATH = MODELS_DIR / "scaler.pkl"
XGB_PKL     = MODELS_DIR / "xgb_model.pkl"
XGB_JSON    = MODELS_DIR / "xgb_model.json"
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
                bst = xgb.Booster(); bst.load_model(str(XGB_JSON))
                bundle["xgb_booster"] = bst

        if lgb:
            if _safe_exists(LGBM_PKL) and joblib:
                bundle["lgbm_sklearn"] = joblib.load(LGBM_PKL)
            elif _safe_exists(LGBM_TXT):
                bundle["lgbm_booster"] = lgb.Booster(model_file=str(LGBM_TXT))

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

# Utility / vision helpers
IDX = {"B":0,"P":1,"T":2}

def _has_horizontal_line(roi_bgr: np.ndarray) -> bool:
    if roi_bgr is None or roi_bgr.size == 0: return False
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    enh = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    edges = cv2.Canny(thr, CANNY1, CANNY2)
    h,w = edges.shape[:2]
    min_len = max(int(w*HOUGH_MIN_LEN_RATIO), 12)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=20,minLineLength=min_len,maxLineGap=HOUGH_GAP)
    if lines is None: return False
    for x1,y1,x2,y2 in lines[:,0,:]:
        if abs(y2-y1) <= max(2,int(h*0.12)): return True
    return False

def _grid_from_roi(roi: np.ndarray) -> Tuple[List[int], List[int]]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,5,60,60)
    _, bw1 = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 40, 120)
    bw = cv2.bitwise_or(bw1, edges)

    vh = max(1, roi.shape[0]//30)
    vw = max(1, roi.shape[1]//42)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,vh))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(vw,1))
    vlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    hlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hori_kernel, iterations=1)

    vx = np.clip(vlines.sum(axis=0),0,255*roi.shape[0]).astype(np.float32)
    hy = np.clip(hlines.sum(axis=1),0,255*roi.shape[1]).astype(np.float32)

    def _peaks(arr, min_gap, thr):
        idx=[]; last=-1e9
        for i,v in enumerate(arr):
            if v>thr:
                if i-last>min_gap: idx.append(i); last=i
        return idx

    col_idx = _peaks(vx, max(3,roi.shape[1]//95), thr=255*2.0)
    row_idx = _peaks(hy, max(3,roi.shape[0]//65), thr=255*2.0)

    def _regularize(idxs):
        if len(idxs)<4: return []
        diffs=[idxs[i+1]-idxs[i] for i in range(len(idxs)-1)]
        step=int(np.median(diffs))
        start=idxs[0]; out=[]; i=0
        while start+i*step < (idxs[-1]+step//2):
            out.append(int(start+i*step)); i+=1
        return out

    cols = _regularize(col_idx)
    rows = _regularize(row_idx)
    if rows and len(rows)>7: rows = rows[:7]
    return cols, rows

def _color_masks(bgr: np.ndarray):
    blur = cv2.GaussianBlur(bgr,(3,3),0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, HSV["RED1_LOW"],  HSV["RED1_HIGH"])
    red2 = cv2.inRange(hsv, HSV["RED2_LOW"],  HSV["RED2_HIGH"])
    red  = cv2.bitwise_or(red1, red2)
    blue = cv2.inRange(hsv, HSV["BLUE_LOW"],  HSV["BLUE_HIGH"])
    green= cv2.inRange(hsv, HSV["GREEN_LOW"], HSV["GREEN_HIGH"])
    k = np.ones((3,3), np.uint8)
    def clean(m):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
        return m
    return clean(red), clean(blue), clean(green)

def _blobs(roi: np.ndarray) -> List[Dict[str,Any]]:
    """
    回傳每個 blob 的 dict：
    {x,y,w,h,area,circ,cx,cy, vote_red, vote_blue, vote_green, label_candidate}
    - label_candidate 是顏色投票結果 'B'/'P'/'T' 或 None (若票數不足)
    - 嚴格模式會過濾 area / circ
    """
    red_mask, blue_mask, green_mask = _color_masks(roi)
    # combine masks to find candidates
    combo = cv2.bitwise_or(cv2.bitwise_or(red_mask, blue_mask), green_mask)
    # 膨脹讓空心圈閉合
    kernel = np.ones((3,3), np.uint8)
    combo = cv2.morphologyEx(combo, cv2.MORPH_CLOSE, kernel, iterations=2)
    combo = cv2.morphologyEx(combo, cv2.MORPH_OPEN, kernel, iterations=1)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(combo, 8)
    items = []
    hR,wR = roi.shape[:2]
    for i in range(1, n):
        x,y,w,h,a = int(stats[i,0]),int(stats[i,1]),int(stats[i,2]),int(stats[i,3]),int(stats[i,4])
        if a <= 6: continue  # 太小直接忽略基礎噪聲
        cx,cy = float(centroids[i][0]), float(centroids[i][1])
        peri = 2*(w+h)
        circ = 4.0*math.pi*a/(peri*peri+1e-9)
        # bounding clipping
        x1,x2 = max(0,x), min(wR, x+w)
        y1,y2 = max(0,y), min(hR, y+h)
        if x2<=x1 or y2<=y1: continue
        sub_r = red_mask[y1:y2, x1:x2]
        sub_b = blue_mask[y1:y2, x1:x2]
        sub_g = green_mask[y1:y2, x1:x2]
        # 投票：每個 mask 在 bbox 內的像素比例
        area_bbox = float((x2-x1)*(y2-y1))
        vote_r = float(np.count_nonzero(sub_r))/area_bbox
        vote_b = float(np.count_nonzero(sub_b))/area_bbox
        vote_g = float(np.count_nonzero(sub_g))/area_bbox
        # 決定候選 label（需超過 MIN_VOTE_RATIO）
        label = None
        maxv = max(vote_r, vote_b, vote_g)
        if maxv >= MIN_VOTE_RATIO:
            if maxv == vote_r: label = "B"   # Banker (紅)
            elif maxv == vote_b: label = "P" # Player (藍)
            else: label = "T"                # Tie / 和 (綠)
        items.append({
            "x": x, "y": y, "w": w, "h": h, "area": int(a), "circ": float(circ),
            "cx": float(cx), "cy": float(cy),
            "vote_r": float(vote_r), "vote_b": float(vote_b), "vote_g": float(vote_g),
            "label_candidate": label
        })

    # 嚴格模式先過濾掉面積或圓度不符合的
    if STRICT_GRID:
        filtered = []
        for it in items:
            if it["area"] >= MIN_AREA and it["circ"] >= CIRC_THRESH and it["label_candidate"] is not None:
                filtered.append(it)
        if DEBUG_VISION:
            logger.info(f"[BLOBS][STRICT] candidates={len(items)} filtered={len(filtered)} MIN_AREA={MIN_AREA} CIRC={CIRC_THRESH} MIN_VOTE={MIN_VOTE_RATIO}")
        items = filtered

    # 最後排序（從左到右）便於後面欄群組
    items.sort(key=lambda z: z["cx"])
    return items

def _grid_from_beads(items: List[Dict[str,Any]], roi_w: int, roi_h: int) -> Tuple[List[int], List[int]]:
    if not items: return [], []
    cxs = sorted([it["cx"] for it in items])
    cys = sorted([it["cy"] for it in items])
    def median_gap(vals):
        gaps=[vals[i+1]-vals[i] for i in range(len(vals)-1) if vals[i+1]-vals[i]>3]
        return np.median(gaps) if gaps else (roi_w/12)
    step_x = int(max(8, median_gap(cxs)))
    step_y = int(max(8, median_gap(cys)))
    start_x = max(0, int(min(cxs)-step_x*0.8))
    start_y = max(0, int(min(cys)-step_y*0.8))
    cols=[start_x]
    while cols[-1]+step_x < roi_w-2:
        cols.append(cols[-1]+step_x)
    rows=[start_y]
    while len(rows)<7 and rows[-1]+step_y < roi_h-2:
        rows.append(rows[-1]+step_y)
    return cols, rows

def _snap_and_sequence(roi: np.ndarray, cols: List[int], rows: List[int], items: List[Dict[str,Any]]) -> List[str]:
    # 若格線抓不到但有珠 → 用珠心估格
    if (not cols or not rows or len(rows)<2) and items:
        cols, rows = _grid_from_beads(items, roi.shape[1], roi.shape[0])
    # 若仍抓不到且嚴格模式 -> 回傳空 (避免只抓到 1 手的離譜情況)
    if (not cols or not rows or len(rows)<2) and STRICT_GRID:
        if DEBUG_VISION: logger.info("[SNAP] strict grid fail -> return []")
        return []
    if not cols or not rows or len(rows)<2:
        # fallback: group columns by cx
        items_sorted = sorted(items, key=lambda z: z["cx"])
        if not items_sorted: return []
        cxs = [it["cx"] for it in items_sorted]
        med_gap = np.median([cxs[i+1]-cxs[i] for i in range(len(cxs)-1)]) if len(cxs)>1 else 10
        col_bin = max(6.0, 0.6*float(med_gap))
        columns = []
        for it in items_sorted:
            if not columns or abs(it["cx"]-columns[-1][-1]["cx"])>col_bin:
                columns.append([it])
            else:
                columns[-1].append(it)
        # column-wise, top->down
        seq=[]
        for col in columns:
            col.sort(key=lambda z: z["cy"])
            last = -1e9; row_thr = max(6.0, 0.5*float(np.median([c["h"] for c in col]) if col else 12))
            for it in col:
                if abs(it["cy"]-last) < row_thr: continue
                last = it["cy"]
                lab = it["label_candidate"] or "T"
                seq.append(lab)
        return seq

    # 正常格線吸附（欄→列）
    row_centers = [int((rows[i]+rows[i+1])//2) for i in range(min(6,len(rows)-1))]
    col_centers = [int((cols[i]+cols[i+1])//2) for i in range(len(cols)-1)]
    grid = [[None for _ in range(len(col_centers))] for _ in range(len(row_centers))]
    for it in items:
        cx,cy = it["cx"], it["cy"]
        j = int(np.argmin([abs(cy-rc) for rc in row_centers]))
        i = int(np.argmin([abs(cx-cc) for cc in col_centers]))
        if 0<=j<len(row_centers) and 0<=i<len(col_centers):
            score = abs(cy-row_centers[j])+abs(cx-col_centers[i])
            prev = grid[j][i]
            if prev is None or score < prev[0]:
                grid[j][i] = (score, it)
    seq=[]
    for i in range(len(col_centers)):
        for j in range(len(row_centers)):
            cell = grid[j][i]
            if cell is None: continue
            it = cell[1]
            lab = it["label_candidate"] or "T"
            seq.append(lab)
    return seq

def extract_sequence_from_image(img_bytes: bytes) -> Tuple[List[str], Dict[str,Any]]:
    """
    回傳 (seq, debug_info)
    seq: ['B','P','T'...]
    debug_info: 包含 items, cols, rows, debug_image_path, json_path (若有)
    """
    debug_info = {"items":[],"cols":0,"rows":0,"seq_len":0,"debug_img":None,"json":None}
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(pil); img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H,W = img_bgr.shape[:2]
        # scale down/up to reasonable processing size if huge
        max_side = 1400.0
        scale = max_side / max(H,W) if max(H,W) > max_side else 1.0
        if scale != 1.0:
            img_bgr = cv2.resize(img_bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
        mode = os.getenv("ROAD_MODE", ROAD_MODE).strip().lower()

        # helper strict check
        def _strict_fail(cols, rows, items, tag):
            if not STRICT_GRID: return False
            bad = (len(rows) < 2) or (len(cols) < MIN_COLS) or (len(items) < MIN_ITEMS)
            if bad and DEBUG_VISION:
                logger.info(f"[STRICT][{tag}] fail cols={len(cols)} rows={len(rows)} items={len(items)}")
            return bad

        # BEAD mode
        if mode == "bead":
            HH, WW = img_bgr.shape[:2]
            # try FOCUS_BEAD_ROI first
            roi_env = os.getenv("FOCUS_BEAD_ROI","")
            if roi_env:
                try:
                    sx,sy,sw,sh = [float(t) for t in roi_env.split(",")]
                    rx=int(max(0,min(1,sx))*WW); ry=int(max(0,min(1,sy))*HH)
                    rw=int(max(0,min(1,sw))*WW); rh=int(max(0,min(1,sh))*HH)
                    roi = img_bgr[ry:ry+rh, rx:rx+rw]
                except Exception:
                    roi = img_bgr[int(HH*0.55):HH, 0:int(WW*0.55)]
            else:
                # auto-locate left-bottom density
                red, blue, _ = _color_masks(img_bgr)
                combo = cv2.bitwise_or(red, blue)
                y0=int(HH*0.55); x0=0; x1=int(WW*0.55)
                mask=np.zeros_like(combo); mask[y0:HH, x0:x1]=combo[y0:HH, x0:x1]
                kernel=np.ones((5,5),np.uint8)
                m=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
                m=cv2.morphologyEx(m,cv2.MORPH_OPEN,kernel,iterations=1)
                cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
                    pad=max(6,int(min(w,h)*0.06))
                    rx=max(0,x-pad); ry=max(0,y-pad)
                    rw=min(WW-rx,w+2*pad); rh=min(HH-ry,h+2*pad)
                    roi = img_bgr[ry:ry+rh, rx:rx+rw]
                else:
                    roi = img_bgr[y0:HH, x0:x1]
            cols, rows = _grid_from_roi(roi)
            items = _blobs(roi)
            debug_info["cols"], debug_info["rows"] = len(cols), len(rows)
            debug_info["items"] = items
            if _strict_fail(cols, rows, items, "BEAD"):
                return [], debug_info
            seq = _snap_and_sequence(roi, cols, rows, items)
            debug_info["seq_len"] = len(seq)
            # debug outputs
            if DEBUG_VISION:
                _dump_debug_outputs(roi, cols, rows, items, seq, tag="bead")
                debug_info["debug_img"] = "/mnt/data/debug_bead_strict_out.png"
                debug_info["json"] = "/mnt/data/debug_bead_strict.json"
            return seq[-240:], debug_info

        # BIGROAD mode
        HH, WW = img_bgr.shape[:2]
        roi_env = os.getenv("FOCUS_ROI","")
        if roi_env:
            try:
                sx,sy,sw,sh = [float(t) for t in roi_env.split(",")]
                rx=int(max(0,min(1,sx))*WW); ry=int(max(0,min(1,sy))*HH)
                rw=int(max(0,min(1,sw))*WW); rh=int(max(0,min(1,sh))*HH)
                roi0 = img_bgr[ry:ry+rh, rx:rx+rw]
            except Exception:
                roi0 = img_bgr[int(HH*0.45):HH, :]
        else:
            red, blue, _ = _color_masks(img_bgr)
            combo = cv2.bitwise_or(red, blue)
            y0=int(HH*0.45)
            mask=np.zeros_like(combo); mask[y0:HH,:]=combo[y0:HH,:]
            kernel=np.ones((5,5),np.uint8)
            m=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
            m=cv2.morphologyEx(m,cv2.MORPH_OPEN,kernel,iterations=1)
            cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
                padx=max(6,int(w*0.03)); pady=max(6,int(h*0.08))
                rx=max(0,x-padx); ry=max(0,y-pady)
                rw=min(WW-rx,w+2*padx); rh=min(HH-ry,h+2*pady)
                roi0 = img_bgr[ry:ry+rh, rx:rx+rw]
            else:
                roi0 = img_bgr[y0:HH, :]
        BIGROAD_FRAC = float(os.getenv("BIGROAD_FRAC","0.70"))
        MIN_BEADS = int(os.getenv("MIN_BEADS","12"))

        def _run_bigroad(frac: float):
            rh_big = max(1, int(roi0.shape[0]*max(0.5, min(0.95, frac))))
            roi = roi0[:rh_big, :]
            cols, rows = _grid_from_roi(roi)
            items = _blobs(roi)
            return roi, cols, rows, items

        roi, cols, rows, items = _run_bigroad(BIGROAD_FRAC)
        debug_info["cols"], debug_info["rows"] = len(cols), len(rows)
        debug_info["items"] = items
        if _strict_fail(cols, rows, items, "BIG"):
            if STRICT_GRID:
                return [], debug_info
            else:
                roi2, cols2, rows2, items2 = _run_bigroad(min(0.80, BIGROAD_FRAC+0.05))
                roi, cols, rows, items = roi2, cols2, rows2, items2
        seq = _snap_and_sequence(roi, cols, rows, items)
        debug_info["seq_len"] = len(seq)
        if DEBUG_VISION:
            _dump_debug_outputs(roi, cols, rows, items, seq, tag="big")
            debug_info["debug_img"] = "/mnt/data/debug_bead_strict_out.png"
            debug_info["json"] = "/mnt/data/debug_bead_strict.json"
        return seq[-240:], debug_info

    except Exception as e:
        if DEBUG_VISION: logger.exception(f"[VISION][ERR] {e}")
        return [], debug_info

# Debug dump: image + JSON
def _dump_debug_outputs(roi, cols, rows, items, seq, tag="dbg"):
    try:
        H,W = roi.shape[:2]
        vis = roi.copy()
        # draw grid
        if cols and len(cols)>1:
            for c in cols:
                cv2.line(vis, (c,0),(c,H-1),(160,160,160),1)
        if rows and len(rows)>1:
            for r in rows:
                cv2.line(vis, (0,r),(W-1,r),(160,160,160),1)
        # draw items
        for i,it in enumerate(items):
            x,y,w,h = int(it["x"]),int(it["y"]),int(it["w"]),int(it["h"])
            label = it.get("label_candidate") or "?"
            color = (0,0,255) if label=="B" else ((255,0,0) if label=="P" else (0,255,0))
            cv2.rectangle(vis, (x,y),(x+w,y+h), color, 2)
            cv2.putText(vis, f'{label}:{int(it["area"])}',{x,y-6}, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)
        # save image
        out_img_path = "/mnt/data/debug_bead_strict_out.png"
        cv2.imwrite(out_img_path, vis)
        # save json
        json_path = "/mnt/data/debug_bead_strict.json"
        small = [{"x":int(it["x"]), "y":int(it["y"]), "w":int(it["w"]), "h":int(it["h"]),
                  "area":int(it["area"]), "circ":round(it["circ"],3),
                  "vote_r":round(it["vote_r"],3), "vote_b":round(it["vote_b"],3),
                  "vote_g":round(it["vote_g"],3), "label": it.get("label_candidate")}
                 for it in items]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"items": small, "cols": len(cols), "rows": len(rows), "seq": seq}, f, ensure_ascii=False, indent=2)
        logger.info(f"[DEBUG] dumped {out_img_path} and {json_path}")
    except Exception as e:
        logger.exception(f"debug dump err: {e}")

# ========== 以下為特徵工程 / 模型 / 回傳（與之前邏輯相同） ==========

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

def _softmax(x: np.ndarray, temp: float=1.0) -> np.ndarray:
    x = x.astype(np.float64)/max(1e-9,temp)
    m = np.max(x)
    e = np.exp(x-m)
    return e/(np.sum(e)+1e-12)

def _parse_weights_env_pair() -> Tuple[Dict[str, float], Dict[str, float]]:
    def _parse(s: str, default: Dict[str,float]) -> Dict[str,float]:
        out = default.copy()
        try:
            for kv in s.split(","):
                k,v = kv.split(":"); k=k.strip().lower(); v=float(v)
                if k in out: out[k]=max(0.0,v)
        except Exception: pass
        ss=sum(out.values()) or 1.0
        for k in out: out[k]/=ss
        return out
    trend_def={"xgb":0.45,"lgb":0.35,"rnn":0.20}
    chop_def ={"xgb":0.20,"lgb":0.25,"rnn":0.55}
    return _parse(os.getenv("ENSEMBLE_WEIGHTS_TREND",""),trend_def), \
           _parse(os.getenv("ENSEMBLE_WEIGHTS_CHOP",""), chop_def)

def _proba_from_xgb(feat: np.ndarray) -> Dict[str,float] | None:
    if "xgb_sklearn" in model_bundle:
        proba = model_bundle["xgb_sklearn"].predict_proba(feat)[0]
        return {"banker": float(proba[IDX["B"]]), "player": float(proba[IDX["P"]]), "tie": float(proba[IDX["T"]])}
    if "xgb_booster" in model_bundle and xgb:
        d = xgb.DMatrix(feat); proba = model_bundle["xgb_booster"].predict(d)[0]
        if len(proba)==3: return {"banker": float(proba[0]), "player": float(proba[1]), "tie": float(proba[2])}
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
    if w is None or not seq: return None
    try:
        Wxh=np.array(w["Wxh"]); Whh=np.array(w["Whh"]); bh=np.array(w["bh"])
        Why=np.array(w["Why"]); bo=np.array(w["bo"])
        h=np.zeros((Whh.shape[0],),dtype=np.float32)
        for s in seq:
            x=np.zeros((3,),dtype=np.float32); x[IDX.get(s,2)]=1.0
            h=np.tanh(x@Wxh + h@Whh + bh)
        o=h@Why + bo
        prob=_softmax(o, temp=1.0)
        return {"banker":float(prob[0]),"player":float(prob[1]),"tie":float(prob[2])}
    except Exception as e:
        logger.warning(f"RNN error: {e}")
        return None

def predict_with_models(seq: List[str]) -> Tuple[Dict[str,float] | None, Dict[str,Any]]:
    info={"used":["xgb","lgb","rnn"],"oscillating":False,"alt_rate":0.0,"alt_streak":0}
    if not seq: return None, info
    ALT_WINDOW=int(os.getenv("ALT_WINDOW","20"))
    ALT_THRESH=float(os.getenv("ALT_THRESH","0.70"))
    ALT_STRICT=int(os.getenv("ALT_STRICT_STREAK","5"))
    alt_rate=_oscillation_rate(seq,ALT_WINDOW)
    alt_streak=_alt_streak_suffix(seq)
    info["alt_rate"]=round(alt_rate,3); info["alt_streak"]=int(alt_streak)
    is_chop=(alt_rate>=ALT_THRESH) or (alt_streak>=ALT_STRICT)
    info["oscillating"]=is_chop

    MIN_SEQ=int(os.getenv("MIN_SEQ","18"))
    if len([c for c in seq if c in ("B","P")])<MIN_SEQ:
        return None, info

    feat=build_features(seq)
    if "scaler" in model_bundle:
        try: feat=model_bundle["scaler"].transform(feat)
        except Exception as e: logger.warning(f"scaler.transform error: {e}")

    w_trend, w_chop=_parse_weights_env_pair()
    weights=w_chop if is_chop else w_trend
    TEMP=float(os.getenv("TEMP","0.95"))

    preds={}
    preds["xgb"]=_proba_from_xgb(feat)
    preds["lgb"]=_proba_from_lgb(feat)
    preds["rnn"]=_proba_from_rnn(seq)
    if not any(preds.values()): return None, info

    agg={"banker":0.0,"player":0.0,"tie":0.0}; wsum=0.0
    for name,p in preds.items():
        if not p: continue
        w=weights.get(name,0.0); wsum+=w
        for k in agg: agg[k]+=w*max(1e-9,float(p[k]))
    if wsum<=0: return None, info
    vec=np.array([agg["banker"],agg["player"],agg["tie"]],dtype=np.float64)
    vec=_softmax(vec,temp=TEMP)
    if is_chop: vec=0.88*vec + 0.12*np.array([1/3,1/3,1/3],dtype=np.float64)
    out={"banker":float(vec[0]),"player":float(vec[1]),"tie":float(vec[2])}
    return _normalize(out), info

def _oscillation_rate(seq: List[str], win: int) -> float:
    s = [c for c in seq[-win:] if c in ("B","P")]
    if len(s) < 2: return 0.0
    alt = sum(1 for a,b in zip(s,s[1:]) if a!=b)
    return alt/(len(s)-1)

def _alt_streak_suffix(seq: List[str]) -> int:
    s = [c for c in seq if c in ("B","P")]
    if len(s) < 2: return 0
    k=1
    for i in range(len(s)-2,-1,-1):
        if s[i]!=s[i+1]: k+=1
        else: break
    return k

def _normalize(p: Dict[str,float]) -> Dict[str,float]:
    p = {k: max(1e-9, float(v)) for k,v in p.items()}
    s = p["banker"]+p["player"]+p["tie"]
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {k: round(v/s,4) for k,v in p.items()}

def predict_probs_from_seq_rule(seq: List[str]) -> Dict[str,float]:
    n=len(seq)
    if n==0: return {"banker":0.33,"player":0.33,"tie":0.34}
    pb=seq.count("B")/n; pp=seq.count("P")/n; pt=max(0.02, seq.count("T")/n*0.6)
    tail=1
    for i in range(n-2,-1,-1):
        if seq[i]==seq[-1]: tail+=1
        else: break
    if seq[-1] in {"B","P"}:
        boost=min(0.08,0.025*(tail-1))
        if seq[-1]=="B": pb+=boost
        else: pp+=boost
    s=pb+pp+pt
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {"banker":round(pb/s,4),"player":round(pp/s,4),"tie":round(pt/s,4)}

def betting_plan(pb: float, pp: float, oscillating: bool, alt_streak: int=0) -> Dict[str, Any]:
    diff=abs(pb-pp); side="莊" if pb>=pp else "閒"; side_prob=max(pb,pp)
    ALT_STRICT=int(os.getenv("ALT_STRICT_STREAK","5"))
    if oscillating and alt_streak>=ALT_STRICT:
        return {"side":side,"percent":0.0,"side_prob":side_prob,"note":"單跳震盪期觀望"}
    if oscillating:
        if diff<0.12: return {"side":side,"percent":0.0,"side_prob":side_prob,"note":"震盪期風險高"}
        if diff<0.18: pct=0.02
        elif diff<0.24: pct=0.04
        else: pct=0.08
        return {"side":side,"percent":pct,"side_prob":side_prob,"note":"震盪期降倉"}
    if diff<0.05: return {"side":side,"percent":0.0,"side_prob":side_prob,"note":"差距不足 5%"}
    if diff<0.08: pct=0.02
    elif diff<0.12: pct=0.04
    elif diff<0.18: pct=0.08
    else: pct=0.12
    return {"side":side,"percent":pct,"side_prob":side_prob}

def count_beads(seq: List[str]) -> Dict[str,int]:
    return {"B":seq.count("B"),"P":seq.count("P"),"T":seq.count("T")}

def render_reply(seq: List[str], probs: Dict[str,float], by_model: bool, info: Dict[str,Any] | None=None) -> str:
    b,p,t = probs["banker"], probs["player"], probs["tie"]
    oscillating = bool(info.get("oscillating")) if info else False
    alt_streak  = int(info.get("alt_streak",0)) if info else 0
    plan = betting_plan(b,p,oscillating,alt_streak)
    tag = "（模型）" if by_model else "（規則）"
    win_txt = f"{plan['side_prob']*100:.1f}%"
    note = f"｜{plan['note']}" if plan.get("note") else ""
    bet_text = "觀望" if plan["percent"]==0 else f"下 {plan['percent']*100:.0f}% 於「{plan['side']}」"
    osc_txt = f"\n震盪率：{info.get('alt_rate'):.2f}｜連跳：{alt_streak}" if info and "alt_rate" in info else ""
    used_txt = f"\n投票模型：{', '.join(info.get('used', []))}" if info else ""
    cnt = count_beads(seq)
    return (
        f"{tag} 已解析 {len(seq)} 手{osc_txt}{used_txt}\n"
        f"顆數：莊 {cnt['B']}｜閒 {cnt['P']}｜和 {cnt['T']}\n"
        f"建議下注：{plan['side']}（勝率 {win_txt}）{note}\n"
        f"機率：莊 {b:.2f}｜閒 {p:.2f}｜和 {t:.2f}\n"
        f"資金建議：{bet_text}"
    )

# ========== API / LINE webhook ==========
@app.route("/")
def index():
    return f"BGS AI 助手運行中 ✅ 模式：{os.getenv('ROAD_MODE', ROAD_MODE)} /line-webhook 就緒", 200

@app.route("/health")
def health():
    return jsonify({
        "status":"ok",
        "ts":int(time.time()),
        "mode": os.getenv("ROAD_MODE", ROAD_MODE),
        "models_loaded": model_bundle.get("loaded", False),
        "note": model_bundle.get("note","")
    })

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
        logger.exception(f"InvalidSignatureError: {e}")
        return "Invalid signature", 200
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return "Error", 200
    return "OK"

if line_handler and line_bot_api:
    @line_handler.add(FollowEvent)
    def on_follow(event: FollowEvent):
        welcome = (
            "歡迎加入BGS AI 助手 🎉\n\n"
            "輸入「開始分析」後，上傳牌路截圖，我會自動辨識並回傳建議下注。\n"
            f"目前模式：{os.getenv('ROAD_MODE', ROAD_MODE)}（可設為 bead 讀珠盤路）"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome))

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        txt = (event.message.text or "").strip()
        if txt in {"開始分析", "開始", "START", "分析"}:
            # use session toggle pattern
            user_mode = getattr(on_text, "user_mode", {})
            user_mode[uid] = True
            setattr(on_text, "user_mode", user_mode)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="已進入分析模式 ✅\n上傳牌路截圖即可（支援大路/珠盤路）。"))
            return
        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text="請先輸入「開始分析」，再上傳牌路截圖。"
        ))

    @line_handler.add(MessageEvent, message=ImageMessage)
    def on_image(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        user_mode = getattr(on_text, "user_mode", {})
        if not user_mode.get(uid):
            line_bot_api.reply_message(event.reply_token, TextSendMessage(
                text="尚未啟用分析模式。\n請先輸入「開始分析」，再上傳牌路截圖。"
            ))
            return
        content = line_bot_api.get_message_content(event.message.id)
        img_bytes = b"".join(chunk for chunk in content.iter_content())
        seq, debug_info = extract_sequence_from_image(img_bytes)
        if not seq:
            tip = ("辨識失敗 😥\n若讀珠盤路：請設 ROAD_MODE=bead 並設定 FOCUS_BEAD_ROI；\n"
                   "若讀大路：請設定 FOCUS_ROI 或調整 BIGROAD_FRAC（0.66~0.75）。\n"
                   "你也可以把珠盤那一塊裁圖後再上傳。")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tip)); return
        if model_bundle.get("loaded"):
            probs, info = predict_with_models(seq)
            by_model = probs is not None
            if not by_model:
                probs = predict_probs_from_seq_rule(seq); info = {}
        else:
            probs = predict_probs_from_seq_rule(seq); by_model=False; info = {}
        msg = render_reply(seq, probs, by_model, info)
        # attach debug note if available
        if DEBUG_VISION and debug_info.get("debug_img"):
            msg += f"\n(DEBUG) 標註圖: {debug_info.get('debug_img')}\n(JSON): {debug_info.get('json')}"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))

if __name__ == "__main__":
    port = int(os.getenv("PORT","5000"))
    app.run(host="0.0.0.0", port=port)
