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
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage, FollowEvent
)
...
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

# ---------- User session: 是否進入分析模式 ----------
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
HMM_PKL     = MODELS_DIR / "hmm_model.pkl"

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

        # XGB
        if xgb:
            if _safe_exists(XGB_PKL):
                bundle["xgb_sklearn"] = joblib.load(XGB_PKL) if joblib else None
            elif _safe_exists(XGB_JSON):
                booster = xgb.Booster()
                booster.load_model(str(XGB_JSON))
                bundle["xgb_booster"] = booster
            elif _safe_exists(XGB_UBJ):
                booster = xgb.Booster()
                booster.load_model(str(XGB_UBJ))
                bundle["xgb_booster"] = booster

        # LGBM
        if lgb:
            if _safe_exists(LGBM_PKL):
                bundle["lgbm_sklearn"] = joblib.load(LGBM_PKL) if joblib else None
            elif _safe_exists(LGBM_TXT):
                booster = lgb.Booster(model_file=str(LGBM_TXT))
                bundle["lgbm_booster"] = booster
            elif _safe_exists(LGBM_JSON):
                booster = lgb.Booster(model_file=str(LGBM_JSON))
                bundle["lgbm_booster"] = booster

        # HMM
        if MultinomialHMM and _safe_exists(HMM_PKL):
            hmm = joblib.load(HMM_PKL) if joblib else None
            if hmm:
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

# ====================================================
# 影像處理與序列擷取（省略：略去你原本完整的影像管線）
# ====================================================
def _has_horizontal_line(img: np.ndarray) -> bool:
    # ...
    return True

def extract_sequence_from_image(img: Image.Image) -> List[str]:
    # 回傳像是 ["B","P","B","T","P", ...]
    return []

# ====================================================
# 特徵工程 & 規則機
# ====================================================
def clean(seq: List[str]) -> List[str]:
    return [s for s in seq if s in ("B","P","T")]

def cc_blobs(seq: List[str], target: str) -> List[int]:
    cur = 0; res = []
    for s in seq:
        if s == target:
            cur += 1
        else:
            if cur>0: res.append(cur); cur=0
    if cur>0: res.append(cur)
    return res

def _streak_tail(seq: List[str], target: str) -> int:
    t = 0
    for s in reversed(seq):
        if s == target: t += 1
        else: break
    return t

def _transitions(seq: List[str]) -> Dict[str,int]:
    c = {"BB":0,"BP":0,"PP":0,"PB":0}
    for a,b in zip(seq, seq[1:]):
        if a in "BP" and b in "BP":
            c[a+b]+=1
    return c

def _ratio_lastN(seq: List[str], target: str, N: int=20) -> float:
    last = seq[-N:] if len(seq)>N else seq
    n = sum(1 for s in last if s==target)
    d = sum(1 for s in last if s in "BP")
    return (n/d) if d else 0.5

def build_features(seq: List[str]) -> np.ndarray:
    seq = clean(seq)
    arr = np.array([
        _streak_tail(seq,"B"),
        _streak_tail(seq,"P"),
        _ratio_lastN(seq,"B",20),
        _ratio_lastN(seq,"P",20),
        _transitions(seq)["BB"],
        _transitions(seq)["PP"],
        _transitions(seq)["BP"],
        _transitions(seq)["PB"],
    ], dtype=float).reshape(1,-1)
    return arr

def _normalize(p: np.ndarray) -> np.ndarray:
    s = p.sum(); 
    return (p/s) if s>0 else np.array([1/3,1/3,1/3])

def _proba_from_xgb(X: np.ndarray) -> np.ndarray:
    # ...
    return np.array([0.34,0.33,0.33])

def _proba_from_lgb(X: np.ndarray) -> np.ndarray:
    # ...
    return np.array([0.34,0.33,0.33])

def _proba_from_hmm(seq: List[str]) -> np.ndarray:
    # ...
    return np.array([0.34,0.33,0.33])

def predict_with_models(seq: List[str]) -> Dict[str,float]:
    X = build_features(seq)
    ps = []
    if model_bundle.get("xgb_sklearn") or model_bundle.get("xgb_booster"):
        ps.append(_proba_from_xgb(X))
    if model_bundle.get("lgbm_sklearn") or model_bundle.get("lgbm_booster"):
        ps.append(_proba_from_lgb(X))
    if model_bundle.get("hmm"):
        ps.append(_proba_from_hmm(seq))
    if not ps:
        ps = [np.array([0.34,0.33,0.33])]
    mean = _normalize(np.mean(ps, axis=0))
    return {"banker": float(mean[0]), "player": float(mean[1]), "tie": float(mean[2])}

def predict_probs_from_seq_rule(seq: List[str]) -> Dict[str,float]:
    # 備援規則機（無模型時）
    b = 0.5 + 0.1 * (_ratio_lastN(seq,"B",12) - 0.5)
    p = 1 - b
    t = 0.02
    s = b + p + t
    return {"banker": b/s, "player": p/s, "tie": t/s}

# ====================================================
# 資金分配與顯示
# ====================================================
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
    plan = betting_plan(b, p)  # 保留資金分配與勝率來源
    tag = "（模型）" if by_model else "（規則）"
    win_txt = f"{plan['side_prob']*100:.1f}%"
    note = f"｜{plan['note']}" if plan.get("note") else ""

    # === 統一顯示（資金+方向）含 12% 反向規則 ===
    # percent == 0 → 觀望； 0<percent<0.12 → 顯示方向顛倒； >=0.12 → 不顛倒
    if plan["percent"] == 0.0:
        advise_text = "觀望"
    else:
        if plan["percent"] < 0.12:
            show_side = "閒" if plan["side"] == "莊" else "莊"
        else:
            show_side = plan["side"]
        advise_text = f"於「{show_side}」下注 {int(plan['percent']*100)}%"

    return (
        f"{tag} 已解析 {len(seq)} 手\n"
        f"建議（資金+方向）：{advise_text}（勝率 {win_txt}）{note}\n"
        f"機率：莊 {b:.2f}｜閒 {p:.2f}｜和 {t:.2f}"
    )

# =========================================================
# API（可自測）
# =========================================================
@app.route("/")
def index():
    return "BGS AI 助手正在運行 ✅ /line-webhook"

@app.route("/health")
def health():
    return jsonify({"ok": True, "models": model_bundle.get("note","")})

# LINE Webhook 端點
@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not (line_bot_api and line_handler):
        return abort(503, "LINE config missing")

    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400, "Invalid signature")
    return "OK"

@line_handler.add(FollowEvent)
def on_follow(event: FollowEvent):
    uid = event.source.user_id
    user_mode[uid] = True
    if line_bot_api:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage("歡迎加入！傳文字進入規則模式；傳圖片進行牌路解析。")
        )

@line_handler.add(MessageEvent, message=TextMessage)
def on_text(event: MessageEvent):
    uid = event.source.user_id
    text = (event.message.text or "").strip()
    # 規則機測試
    seq = [s for s in text.replace(","," ").upper().split() if s in ("B","P","T")]
    if not seq:
        if line_bot_api:
            line_bot_api.reply_message(event.reply_token, TextSendMessage("請貼上 B/P/T 序列或上傳圖片"))
        return
    probs = predict_with_models(seq) if model_bundle.get("loaded") else predict_probs_from_seq_rule(seq)
    reply = render_reply(seq, probs, by_model=model_bundle.get("loaded", False))
    if line_bot_api:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(reply))

@line_handler.add(MessageEvent, message=ImageMessage)
def on_image(event: MessageEvent):
    # 圖像取回與解析（此處略）
    # img_bytes = ...
    # seq = extract_sequence_from_image(Image.open(io.BytesIO(img_bytes)))
    seq = []
    probs = predict_with_models(seq) if model_bundle.get("loaded") else predict_probs_from_seq_rule(seq)
    reply = render_reply(seq, probs, by_model=model_bundle.get("loaded", False))
    if line_bot_api:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(reply))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8000")), debug=True)
