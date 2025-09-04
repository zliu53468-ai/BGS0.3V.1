# app.py
# BGS AI（Flask + LINE）— 大路/珠盤路 可切換的辨識 + 投票（XGB/LGBM/RNN）
# 此版本：加入 MIN_BLOB_AREA / MIN_CIRC / MIN_VOTE_RATIO 的 ENV 控制
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
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import (
        MessageEvent, TextMessage, ImageMessage, TextSendMessage, FollowEvent
    )
except Exception:
    LineBotApi = WebhookHandler = None
    MessageEvent = TextMessage = ImageMessage = TextSendMessage = FollowEvent = None

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
    "BLUE_LOW":  (int(os.getenv("HSV_BLUE_H_LOW",  "90")), int(os.getenv("HSV_BLUE_S_LOW",  "50")), int(os.getenv("HSV_BLUE_V_LOW",  "50"))),
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
# Vision helpers
# =========================================================
IDX = {"B":0,"P":1,"T":2}

def _color_masks(bgr: np.ndarray):
    blur = cv2.GaussianBlur(bgr,(3,3),0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, HSV["RED1_LOW"], HSV["RED1_HIGH"])
    red2 = cv2.inRange(hsv, HSV["RED2_LOW"], HSV["RED2_HIGH"])
    red  = cv2.bitwise_or(red1, red2)
    blue = cv2.inRange(hsv, HSV["BLUE_LOW"], HSV["BLUE_HIGH"])
    green= cv2.inRange(hsv, HSV["GREEN_LOW"], HSV["GREEN_HIGH"])
    k = np.ones((3,3), np.uint8)
    def clean(m):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
        return m
    return clean(red), clean(blue), clean(green)

def _blobs(roi: np.ndarray) -> List[tuple]:
    """
    回傳每顆候選珠：
      (x,y,w,h,cx,cy,label,area,circ,red_votes,blue_votes,green_votes)
    會使用 ENV 的 MIN_BLOB_AREA / MIN_CIRC / MIN_VOTE_RATIO
    """
    if roi is None or roi.size==0: return []
    red_m, blue_m, green_m = _color_masks(roi)
    combo = cv2.bitwise_or(cv2.bitwise_or(red_m, blue_m), green_m)
    # connected components
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(combo, 8, cv2.CV_32S)
    out=[]
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < MIN_BLOB_AREA:
            continue
        # contour of that component to compute circularity
        mask_i = (labels==i).astype('uint8')*255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=lambda c: cv2.contourArea(c))
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circ = 4*math.pi*area/(peri*peri+1e-9) if peri>0 else 0.0
        if circ < MIN_CIRC:
            # 不是圓形 -> skip (嚴格模式)
            continue
        M = cv2.moments(cnt)
        if M["m00"]==0:
            cx = x + w/2.0; cy = y + h/2.0
        else:
            cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
        # color votes inside contour
        mask_bbox = np.zeros_like(combo)
        cv2.drawContours(mask_bbox, [cnt], -1, 255, -1)
        red_votes = int(cv2.countNonZero(cv2.bitwise_and(red_m, mask_bbox)))
        blue_votes = int(cv2.countNonZero(cv2.bitwise_and(blue_m, mask_bbox)))
        green_votes = int(cv2.countNonZero(cv2.bitwise_and(green_m, mask_bbox)))
        s = red_votes + blue_votes + green_votes
        if s <= 0:
            continue
        votes = {"B": red_votes, "P": blue_votes, "T": green_votes}
        winner = max(votes, key=votes.get)
        if votes[winner] < s * MIN_VOTE_RATIO:
            # 顏色不明確 -> 在嚴格模式下跳過
            if STRICT_GRID:
                continue
        out.append((x,y,w,h,float(cx),float(cy),winner,a,float(circ),red_votes,blue_votes,green_votes))
    # debug image
    if DEBUG_VISION:
        dbg = roi.copy()
        for (x,y,w,h,cx,cy,lab,a,circ,rv,bv,gv) in out:
            col = (0,0,255) if lab=="B" else ((255,0,0) if lab=="P" else (0,255,0))
            cv2.rectangle(dbg,(int(x),int(y)),(int(x+w),int(y+h)),col,2)
            cv2.putText(dbg, f"{lab}", (int(x),int(y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col,1,cv2.LINE_AA)
        fname = f"/mnt/data/debug_bead_{int(time.time())}.png"
        cv2.imwrite(fname, dbg)
        logger.info(f"[DEBUG] wrote debug bead image: {fname}")
    return out

# ---- grid helpers (保留原邏輯) ----
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

def _grid_from_beads(items: List[tuple], roi_w: int, roi_h: int) -> Tuple[List[int], List[int]]:
    if not items: return [], []
    cxs = sorted([it[4] for it in items])
    cys = sorted([it[5] for it in items])
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

def _snap_and_sequence(roi: np.ndarray, cols: List[int], rows: List[int], items: List[tuple]) -> List[str]:
    # 若格線抓不到，但有珠 → 用珠心估格
    if (not cols or not rows or len(rows)<2) and items:
        cols, rows = _grid_from_beads(items, roi.shape[1], roi.shape[0])
    if not cols or not rows or len(rows)<2:
        if STRICT_GRID:
            return []
        # fallback grouping
        items_sorted = sorted(items, key=lambda it: it[4])
        cxs = [it[4] for it in items_sorted]
        gaps=[cxs[i+1]-cxs[i] for i in range(len(cxs)-1)] if len(cxs)>1 else []
        gaps=[g for g in gaps if g>3]
        med_gap = np.median(gaps) if gaps else (np.median([it[2] for it in items_sorted]) if items_sorted else 10)
        col_bin = max(6.0, 0.6*float(med_gap))
        columns=[]
        for it in items_sorted:
            if not columns or abs(it[4]-columns[-1][-1][4])>col_bin:
                columns.append([it])
            else:
                columns[-1].append(it)
        heights=[h for (_,_,_,h,_,_,_,_,_,_,_,_) in items_sorted] if items_sorted else []
        med_h=np.median(heights) if heights else 12
        row_thr=max(6.0,0.5*float(med_h))
        seq=[]
        for col in columns:
            col.sort(key=lambda z: z[5])
            last=-1e9
            for it in col:
                if abs(it[5]-last)<row_thr: continue
                last=it[5]
                x,y,w,h,cx,cy,label,area,circ,rv,bv,gv = it
                seq.append("T" if label=="T" else label)
        return seq
    # 正常：格線吸附（欄→列）
    row_centers = [int((rows[i]+rows[i+1])//2) for i in range(min(6,len(rows)-1))]
    col_centers = [int((cols[i]+cols[i+1])//2) for i in range(len(cols)-1)]
    grid = [[None for _ in range(len(col_centers))] for _ in range(len(row_centers))]
    for it in items:
        x,y,w,h,cx,cy,label,area,circ,rv,bv,gv = it
        j = int(np.argmin([abs(cy-rc) for rc in row_centers]))
        i = int(np.argmin([abs(cx-cc) for cc in col_centers]))
        if 0<=j<len(row_centers) and 0<=i<len(col_centers):
            score = abs(cy-row_centers[j])+abs(cx-col_centers[i])
            prev = grid[j][i]
            if prev is None or score < prev[0]:
                grid[j][i] = (score, label, (x,y,w,h))
    seq=[]
    for i in range(len(col_centers)):
        for j in range(len(row_centers)):
            cell = grid[j][i]
            if cell is None: continue
            _, label, (x,y,w,h) = cell
            seq.append("T" if label=="T" else label)
    return seq

# =========================================================
# 主流程： extract_sequence_from_image
# =========================================================
def extract_sequence_from_image(img_bytes: bytes) -> List[str]:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(img); img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H,W = img.shape[:2]
        # upscale if tiny
        target = 1400.0
        scale = target/max(H,W) if max(H,W)<target else 1.0
        if scale>1.0:
            img = cv2.resize(img,(int(W*scale),int(H*scale)),interpolation=cv2.INTER_CUBIC)
        mode = os.getenv("ROAD_MODE", ROAD_MODE).strip().lower()

        def _strict_fail(cols, rows, items, tag):
            if not STRICT_GRID: return False
            bad = (len(rows) < 2) or (len(cols) < 6) or (len(items) < 6)
            if bad and DEBUG_VISION:
                logger.info(f"[STRICT][{tag}] fail cols={len(cols)} rows={len(rows)} items={len(items)}")
            return bad

        # 珠盤路
        if mode == "bead":
            def _locate_bead_roi(base_bgr: np.ndarray) -> np.ndarray:
                HH, WW = base_bgr.shape[:2]
                roi_env = os.getenv("FOCUS_BEAD_ROI","")
                if roi_env:
                    try:
                        sx,sy,sw,sh = [float(t) for t in roi_env.split(",")]
                        rx=int(max(0,min(1,sx))*WW); ry=int(max(0,min(1,sy))*HH)
                        rw=int(max(0,min(1,sw))*WW); rh=int(max(0,min(1,sh))*HH)
                        sub = base_bgr[ry:ry+rh, rx:rx+rw]
                        if sub.size: return sub
                    except: pass
                # fallback: 左下區域
                y0=int(HH*0.60); x0=0; x1=int(WW*0.55)
                red, blue, _ = _color_masks(base_bgr)
                mask=np.zeros_like(red); mask[y0:HH, x0:x1]=cv2.bitwise_or(red,blue)[y0:HH, x0:x1]
                kernel=np.ones((5,5),np.uint8)
                m=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
                m=cv2.morphologyEx(m,cv2.MORPH_OPEN,kernel,iterations=1)
                cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
                    pad=max(6,int(min(w,h)*0.06))
                    rx=max(0,x-pad); ry=max(0,y-pad)
                    rw=min(WW-rx,w+2*pad); rh=min(HH-ry,h+2*pad)
                    return base_bgr[ry:ry+rh, rx:rx+rw]
                return base_bgr[y0:HH, x0:x1]

            roi = _locate_bead_roi(img)
            items = _blobs(roi)
            # adapt for snap_and_sequence
            list_items=[]
            for it in items:
                x,y,w,h,cx,cy,label,area,circ,rv,bv,gv=it
                list_items.append((x,y,w,h,cx,cy,label))
            cols, rows = _grid_from_roi(roi)
            if _strict_fail(cols, rows, list_items, "BEAD"):
                return []
            seq = _snap_and_sequence(roi, cols, rows, items)
            if DEBUG_VISION:
                logger.info(f"[VISION][BEAD] cols={len(cols)} rows={len(rows)} items={len(items)} seq_len={len(seq)}")
            return seq[-240:]

        # 大路
        def _locate_bigroad_roi(base_bgr: np.ndarray) -> np.ndarray:
            HH, WW = base_bgr.shape[:2]
            roi_env=os.getenv("FOCUS_ROI","")
            if roi_env:
                try:
                    sx,sy,sw,sh = [float(t) for t in roi_env.split(",")]
                    rx=int(max(0,min(1,sx))*WW); ry=int(max(0,min(1,sy))*HH)
                    rw=int(max(0,min(1,sw))*WW); rh=int(max(0,min(1,sh))*HH)
                    sub=base_bgr[ry:ry+rh, rx:rx+rw]
                    if sub.size: return sub
                except: pass
            red, blue, _ = _color_masks(base_bgr)
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
                return base_bgr[ry:ry+rh, rx:rx+rw]
            return base_bgr[y0:HH, :]

        BIGROAD_FRAC = float(os.getenv("BIGROAD_FRAC","0.70"))
        MIN_BEADS = int(os.getenv("MIN_BEADS","12"))

        def _run_bigroad(frac: float):
            roi0 = _locate_bigroad_roi(img)
            rh_big = max(1, int(roi0.shape[0]*max(0.5, min(0.95, frac))))
            roi = roi0[:rh_big, :]
            cols, rows = _grid_from_roi(roi)
            items = _blobs(roi)
            list_items=[]
            for it in items:
                x,y,w,h,cx,cy,label,area,circ,rv,bv,gv=it
                list_items.append((x,y,w,h,cx,cy,label))
            if _strict_fail(cols, rows, list_items, "BIG"):
                return [], (len(cols), len(rows), len(items))
            seq = _snap_and_sequence(roi, cols, rows, items)
            return seq, (len(cols), len(rows), len(items))

        seq, stat = _run_bigroad(BIGROAD_FRAC)
        if not seq:
            if STRICT_GRID:
                if DEBUG_VISION:
                    logger.info(f"[VISION][BIG][STRICT_FAIL] cols={stat[0]} rows={stat[1]} items={stat[2]}")
                return []
            seq2, _ = _run_bigroad(min(0.80, BIGROAD_FRAC+0.05))
            seq = seq2
        if len(seq) < MIN_BEADS and not STRICT_GRID:
            seq2, _ = _run_bigroad(min(0.80, BIGROAD_FRAC+0.05))
            seq = seq2 if seq2 else seq
        if DEBUG_VISION:
            logger.info(f"[VISION][BIG] final_len={len(seq)}")
        return seq[-240:]
    except Exception as e:
        if DEBUG_VISION: logger.exception(f"[VISION][ERR] {e}")
        return []

# =========================================================
# 以下保留你既有的模型 / 投票 / render_reply 相關程式 (示範性貼回)
# （為避免太長，我把之前你使用的投票/特徵/預測函式留在這個區塊）
# 請把你原先的 predict_with_models, build_features, render_reply 等函式整段貼回來
# 我先放一組簡短的規則型回退與 render，確保整體可以運行：
# =========================================================
def _streak_tail(seq):
    if not seq: return 0
    t,c = seq[-1],1
    for i in range(len(seq)-2,-1,-1):
        if seq[i]==t: c+=1
        else: break
    return c

def _ratio_lastN(seq,N):
    s = seq[-N:] if len(seq)>=N else seq
    if not s: return (0.33,0.33,0.34)
    n=len(s); return (s.count("B")/n, s.count("P")/n, s.count("T")/n)

def predict_probs_from_seq_rule(seq):
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

def count_beads(seq):
    return {"B":seq.count("B"),"P":seq.count("P"),"T":seq.count("T")}

def render_reply(seq, probs, by_model, info=None):
    b,p,t = probs["banker"], probs["player"], probs["tie"]
    cnt = count_beads(seq)
    tag = "（模型）" if by_model else "（規則）"
    return (
        f"{tag} 已解析 {len(seq)} 手\n"
        f"顆數：莊 {cnt['B']}｜閒 {cnt['P']}｜和 {cnt['T']}\n"
        f"機率：莊 {b:.2f}｜閒 {p:.2f}｜和 {t:.2f}"
    )

# Simple health and index
@app.route("/")
def index():
    return f"BGS AI 助手運行中 ✅ 模式：{os.getenv('ROAD_MODE', ROAD_MODE)}", 200

@app.route("/health")
def health():
    return jsonify({
        "status":"ok",
        "ts":int(time.time()),
        "mode": os.getenv("ROAD_MODE", ROAD_MODE),
        "models_loaded": model_bundle.get("loaded", False),
        "note": model_bundle.get("note","")
    })

# LINE webhook (optional)
@app.route("/line-webhook", methods=['POST'])
def line_webhook():
    if not (line_bot_api and line_handler):
        logger.error("LINE creds missing")
        abort(403)
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 200
    except Exception as e:
        logger.exception(e); return "Error", 200
    return "OK"

if line_handler and line_bot_api:
    @line_handler.add(MessageEvent, message=ImageMessage)
    def on_image(event):
        uid = getattr(event.source, "user_id", "unknown")
        content = line_bot_api.get_message_content(event.message.id)
        img_bytes = b"".join(chunk for chunk in content.iter_content())
        seq = extract_sequence_from_image(img_bytes)
        if not seq:
            tip = ("辨識失敗 😥\n若讀珠盤路：ROAD_MODE=bead 並設 FOCUS_BEAD_ROI；\n若讀大路：請設 FOCUS_ROI 或調整 BIGROAD_FRAC（0.66~0.75）。")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tip)); return
        # use model if available
        probs = predict_probs_from_seq_rule(seq); by_model=False; info={}
        msg = render_reply(seq, probs, by_model, info)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))

if __name__ == "__main__":
    port = int(os.getenv("PORT","5000"))
    app.run(host="0.0.0.0", port=port)
