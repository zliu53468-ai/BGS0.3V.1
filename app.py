# app.py
# =========================================================
# BGS AI（Flask + LINE）— 大路/珠盤路 可切換的辨識 + 投票（XGB/LGBM/RNN）
# 完整版（含 ENV 可調與小點合併邏輯）
# =========================================================
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

# ---------- ENV & CONFIG ----------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN and LineBotApi else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET and WebhookHandler else None

DEBUG_VISION = os.getenv("DEBUG_VISION", "0") == "1"
ROAD_MODE = os.getenv("ROAD_MODE", "bigroad").strip().lower()  # bigroad 或 bead
STRICT_GRID = os.getenv("STRICT_GRID", "1") == "1"

# 新增可調參數（ENV）
MIN_BLOB_AREA = int(os.getenv("MIN_BLOB_AREA", "200"))    # 面積門檻 (預設 200)
MIN_CIRC = float(os.getenv("MIN_CIRC", "0.42"))           # 圓度門檻 (預設 0.42)
MIN_VOTE_RATIO = float(os.getenv("MIN_VOTE_RATIO", "0.50"))  # 顏色票比閾值
MIN_SMALL_SCALE = float(os.getenv("MIN_SMALL_SCALE", "0.20")) # 小元件小於 parent 的比例視為標記點

# 顏色 HSV (可透過 ENV 微調)
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

# 影像其他參數
HOUGH_MIN_LEN_RATIO = float(os.getenv("HOUGH_MIN_LEN_RATIO", "0.45"))
HOUGH_GAP = int(os.getenv("HOUGH_GAP", "6"))
CANNY1 = int(os.getenv("CANNY1", "60"))
CANNY2 = int(os.getenv("CANNY2", "180"))

# ---------- Models (unchanged) ----------
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

# =========================================================
# Vision helpers (含小點合併)
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
    內部會做「小元件合併到父珠」以過濾對子小點/標記
    """
    if roi is None or roi.size==0: return []
    red_m, blue_m, green_m = _color_masks(roi)
    combo = cv2.bitwise_or(cv2.bitwise_or(red_m, blue_m), green_m)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(combo, 8, cv2.CV_32S)
    comps = []
    H,W = roi.shape[:2]
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        # 先略過太小（粗過濾）
        if a < max(6, int(MIN_BLOB_AREA*0.15)):
            continue
        mask_i = (labels==i).astype('uint8')*255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=lambda c: cv2.contourArea(c))
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        circ = 4*math.pi*area/(peri*peri+1e-9) if peri>0 else 0.0
        M = cv2.moments(cnt)
        if M["m00"]==0:
            cx = x + w/2.0; cy = y + h/2.0
        else:
            cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
        # count color votes inside contour
        mask_bbox = np.zeros_like(combo)
        cv2.drawContours(mask_bbox, [cnt], -1, 255, -1)
        red_votes = int(cv2.countNonZero(cv2.bitwise_and(red_m, mask_bbox)))
        blue_votes = int(cv2.countNonZero(cv2.bitwise_and(blue_m, mask_bbox)))
        green_votes = int(cv2.countNonZero(cv2.bitwise_and(green_m, mask_bbox)))
        comps.append({
            "i": i, "x":x, "y":y, "w":w, "h":h, "area": area,
            "cx": float(cx), "cy": float(cy), "circ": float(circ),
            "red": int(red_votes), "blue": int(blue_votes), "green": int(green_votes)
        })

    if not comps:
        return []

    # --- Merge small near-parent components into parent (avoid counting small marker dots) ---
    comps.sort(key=lambda c: c["area"], reverse=True)  # parent first
    used = set()
    merged = []
    for idx, parent in enumerate(comps):
        if idx in used:
            continue
        # make a working copy
        p = dict(parent)
        for j in range(idx+1, len(comps)):
            if j in used: continue
            child = comps[j]
            # child significantly smaller than parent
            if child["area"] < max(1.0, p["area"] * MIN_SMALL_SCALE):
                dx = child["cx"] - p["cx"]; dy = child["cy"] - p["cy"]
                dist = math.hypot(dx, dy)
                radius = max(p["w"], p["h"]) * 0.6
                # dist within parent radius => treat as marker and merge votes
                if dist < radius:
                    p["red"] += child["red"]
                    p["blue"] += child["blue"]
                    p["green"] += child["green"]
                    p["area"] += child["area"]
                    used.add(j)
        merged.append(p)

    # Now finish decision (apply strict MIN_BLOB_AREA/MIN_CIRC/MIN_VOTE_RATIO)
    out = []
    for p in merged:
        # area / circularity checks
        if p["area"] < MIN_BLOB_AREA: 
            continue
        if p["circ"] < MIN_CIRC:
            # in strict mode, skip non-circular; otherwise allow
            if STRICT_GRID:
                continue
        s = p["red"] + p["blue"] + p["green"]
        if s <= 0:
            continue
        votes = {"B": p["red"], "P": p["blue"], "T": p["green"]}
        winner = max(votes, key=votes.get)
        if votes[winner] < s * MIN_VOTE_RATIO:
            if STRICT_GRID:
                continue
            # otherwise choose winner but mark as low-confidence
        out.append((
            int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"]),
            float(p["cx"]), float(p["cy"]), winner, int(p["area"]), float(p["circ"]),
            int(p["red"]), int(p["blue"]), int(p["green"])
        ))

    # debug draw
    if DEBUG_VISION:
        dbg = roi.copy()
        for (x,y,w,h,cx,cy,lab,area,circ,rv,bv,gv) in out:
            col = (0,0,255) if lab=="B" else ((255,0,0) if lab=="P" else (0,255,0))
            cv2.rectangle(dbg,(int(x),int(y)),(int(x+w),int(y+h)),col,2)
            cv2.circle(dbg,(int(cx),int(cy)),2,col,-1)
            cv2.putText(dbg, f"{lab}:{area}", (int(x),int(y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col,1,cv2.LINE_AA)
        fname = f"/mnt/data/debug_bead_{int(time.time())}.png"
        cv2.imwrite(fname, dbg)
        # also write json of detections
        meta = {"time": int(time.time()), "items":[{"x":x,"y":y,"w":w,"h":h,"cx":cx,"cy":cy,"label":lab,"area":area,"circ":circ,"r":rv,"b":bv,"g":gv} for (x,y,w,h,cx,cy,lab,area,circ,rv,bv,gv) in out]}
        jname = fname.replace(".png",".json")
        with open(jname,"w",encoding="utf-8") as jf: json.dump(meta,jf,ensure_ascii=False,indent=2)
        logger.info(f"[DEBUG] wrote debug bead image: {fname} and {jname}")

    return out

# ---- grid helpers (原邏輯保留) ----
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
        # 回退：欄群組 + 欄內去重（僅在非 STRICT_GRID 使用）
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
    # NOTE: column-major reading: 每一列(欄)從上往下，再右移
    for i in range(len(col_centers)):
        for j in range(len(row_centers)):
            cell = grid[j][i]
            if cell is None: continue
            _, label, (x,y,w,h) = cell
            seq.append("T" if label=="T" else label)
    return seq

# =========================================================
# extract_sequence_from_image (主流程)
# =========================================================
def extract_sequence_from_image(img_bytes: bytes) -> List[str]:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(img); img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H,W = img.shape[:2]
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

        # 珠盤 (bead)
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
                # fallback 左下
                y0=int(HH*0.55); x0=0; x1=int(WW*0.55)
                red, blue, _ = _color_masks(base_bgr)
                combo = cv2.bitwise_or(red,blue)
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
                    return base_bgr[ry:ry+rh, rx:rx+rw]
                return base_bgr[y0:HH, x0:x1]

            roi = _locate_bead_roi(img)
            items = _blobs(roi)
            list_items = []
            for it in items:
                x,y,w,h,cx,cy,label,area,circ,rv,bv,gv = it
                list_items.append((x,y,w,h,cx,cy,label))
            cols, rows = _grid_from_roi(roi)
            if _strict_fail(cols, rows, list_items, "BEAD"):
                return []
            seq = _snap_and_sequence(roi, cols, rows, items)
            if DEBUG_VISION:
                logger.info(f"[VISION][BEAD] cols={len(cols)} rows={len(rows)} items={len(items)} seq_len={len(seq)}")
            return seq[-240:]

        # 大路 (bigroad)
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

        def _run_bigroad(frac: float) -> Tuple[List[str], Tuple[int,int,int]]:
            roi0 = _locate_bigroad_roi(img)
            rh_big = max(1, int(roi0.shape[0]*max(0.5, min(0.95, frac))))
            roi = roi0[:rh_big, :]
            cols, rows = _grid_from_roi(roi)
            items = _blobs(roi)
            list_items = []
            for it in items:
                x,y,w,h,cx,cy,label,area,circ,rv,bv,gv = it
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
# 特徵 / 模型 / 投票（保留你之前邏輯）
# （為簡潔起見直接把常用函式貼上，你若有原版可直接取代）
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
    last = np.zeros(3); last[IDX.get(seq[-1],2)] = 1.0
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

def _parse_weights_env_pair() -> Tuple[Dict[str, float], Dict[str, float]]:
    def _parse(s: str, default: Dict[str,float]) -> Dict[str,float]:
        out = default.copy()
        try:
            for kv in s.split(","):
                if ":" not in kv: continue
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
    # normalize
    s = out["banker"]+out["player"]+out["tie"]
    out = {k: round(v/s,4) for k,v in out.items()}
    return out, info

# 規則回退
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

# =========================================================
# API / LINE webhook
# =========================================================
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

# LINE webhook handling (如果有設定)
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

user_mode: Dict[str,bool] = {}

if line_handler and line_bot_api:
    @line_handler.add(FollowEvent)
    def on_follow(event: FollowEvent):
        welcome = (
            "歡迎加入BGS AI 助手 🎉\n\n"
            "輸入「開始分析」後，上傳牌路截圖，我會自動辨識並回傳建議下注：莊 / 閒（勝率 xx%）。\n"
            f"目前模式：{os.getenv('ROAD_MODE', ROAD_MODE)}（可設為 bead 讀珠盤路）"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome))

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        txt = (event.message.text or "").strip()
        if txt in {"開始分析", "開始", "START", "分析"}:
            user_mode[uid] = True
            msg = "已進入分析模式 ✅\n上傳牌路截圖即可（支援大路/珠盤路）。"
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
            tip = ("辨識失敗 😥\n若讀珠盤路：ROAD_MODE=bead 並設 FOCUS_BEAD_ROI；\n"
                   "若讀大路：請設 FOCUS_ROI 或調整 BIGROAD_FRAC（0.66~0.75）。\n"
                   "（已開嚴格格線：請確認 ROI 沒被遮、欄/列完整）")
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
    port = int(os.getenv("PORT","5000"))
    app.run(host="0.0.0.0", port=port)
