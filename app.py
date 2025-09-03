# app.py
import os, io, logging
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, request, jsonify, abort
from PIL import Image
import numpy as np

# ===== Optional CV (若需做圖片解析可用到) =====
try:
    import cv2
except Exception:
    cv2 = None

# ===== LINE SDK（允許沒設 Token 也能啟動） =====
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

try:
    from hmmlearn.hmm import MultinomialHMM
except Exception:
    MultinomialHMM = None


# -----------------------------------------
# Flask & Logging
# -----------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-bot")


# -----------------------------------------
# ENV（允許本地與雲端）
# -----------------------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

# 沒有 Token 也能啟動（方便本地測試/API 模擬）
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET if LINE_CHANNEL_SECRET else "DUMMY_SECRET")


# -----------------------------------------
# 模型載入（存在就用，不存在走規則機）
# -----------------------------------------
MODELS_DIR = Path("models")
SCALER_PATH = MODELS_DIR / "scaler.pkl"
XGB_PKL     = MODELS_DIR / "xgb_model.pkl"
XGB_JSON    = MODELS_DIR / "xgb_model.json"
XGB_UBJ     = MODELS_DIR / "xgb_model.ubj"
LGBM_PKL    = MODELS_DIR / "lgbm_model.pkl"
LGBM_TXT    = MODELS_DIR / "lgbm_model.txt"
LGBM_JSON   = MODELS_DIR / "lgbm_model.json"
HMM_PKL     = MODELS_DIR / "hmm_model.pkl"

model_bundle: Dict[str, Any] = {}

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
            if _safe_exists(XGB_PKL) and joblib:
                bundle["xgb_sklearn"] = joblib.load(XGB_PKL)
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
            if _safe_exists(LGBM_PKL) and joblib:
                bundle["lgbm_sklearn"] = joblib.load(LGBM_PKL)
            elif _safe_exists(LGBM_TXT):
                bundle["lgbm_booster"] = lgb.Booster(model_file=str(LGBM_TXT))
            elif _safe_exists(LGBM_JSON):
                bundle["lgbm_booster"] = lgb.Booster(model_file=str(LGBM_JSON))

        # HMM
        if MultinomialHMM and _safe_exists(HMM_PKL) and joblib:
            bundle["hmm"] = joblib.load(HMM_PKL)

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


# -----------------------------------------
# 工具：解析 B/P/T（支援中文、逗號、大小寫）
# -----------------------------------------
def parse_seq_from_text(text: str) -> List[str]:
    if not text:
        return []
    try:
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
    except Exception:
        pass
    t = text.upper().replace(",", " ").replace("，", " ").replace("|", " ").replace("/", " ")
    mapping = {
        "莊": "B", "莊家": "B", "BANKER": "B", "Z": "B",
        "閒": "P", "閒家": "P", "PLAYER": "P", "X": "P",
        "和": "T", "和局": "T", "TIE": "T", "H": "T"
    }
    tokens: List[str] = []
    for raw in t.split():
        if raw in ("B", "P", "T"):
            tokens.append(raw)
        else:
            r = "".join(ch for ch in raw if ch.isalnum() or ch in ("B", "P", "T"))
            if r in ("B", "P", "T"):
                tokens.append(r)
            elif raw in mapping:
                tokens.append(mapping[raw])
    return tokens

def clean(seq: List[str]) -> List[str]:
    return [s for s in seq if s in ("B","P","T")]

def _streak_tail(seq: List[str], target: str) -> int:
    t = 0
    for s in reversed(seq):
        if s == target:
            t += 1
        else:
            break
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
    s = p.sum()
    return (p/s) if s>0 else np.array([1/3,1/3,1/3])

def _proba_from_xgb(X: np.ndarray) -> np.ndarray:
    # 依你的模型實作；此處先給 placeholder
    return np.array([0.34,0.33,0.33])

def _proba_from_lgb(X: np.ndarray) -> np.ndarray:
    return np.array([0.34,0.33,0.33])

def _proba_from_hmm(seq: List[str]) -> np.ndarray:
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


# -----------------------------------------
# 影像→序列（佔位：回傳 []；之後接上你的 CV 管線）
# -----------------------------------------
def extract_sequence_from_image(img: Image.Image) -> List[str]:
    # TODO: 接上你的牌路 OCR/色塊/線段偵測，輸出 ["B","P","B","T",...]
    return []


# -----------------------------------------
# 資金分配（維持你的檔位：0/2/4/8/12%）
# -----------------------------------------
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


# -----------------------------------------
# 顯示整合（資金＋方向；<12% 顛倒顯示）
# -----------------------------------------
def render_reply(seq: List[str], probs: Dict[str,float], by_model: bool) -> str:
    b, p, t = probs["banker"], probs["player"], probs["tie"]
    plan = betting_plan(b, p)
    tag = "（模型）" if by_model else "（規則）"
    win_txt = f"{plan['side_prob']*100:.1f}%"
    note = f"｜{plan['note']}" if plan.get("note") else ""

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


# -----------------------------------------
# Flask Routes
# -----------------------------------------
@app.route("/")
def index():
    return "BGS AI 助手正在運行 ✅ /health /api/simulate /line-webhook"

@app.route("/health")
def health():
    return jsonify({"ok": True, "models": model_bundle.get("note","")})

# 方便本地 / 雲端無 LINE 時自測
@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    seq = parse_seq_from_text(text)
    if not seq:
        return jsonify({"error":"請提供序列，例如：'B P P B T' 或 '莊 閒 閒 莊 和'"}), 400
    probs = predict_with_models(seq) if model_bundle.get("loaded") else predict_probs_from_seq_rule(seq)
    reply = render_reply(seq, probs, by_model=model_bundle.get("loaded", False))
    return jsonify({"reply": reply, "seq": seq, "probs": probs})


# ============ LINE Webhook ============
@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        # 沒設定正確 secret 時，這裡可能報錯
        abort(400, "Invalid signature")
    return "OK"

@line_handler.add(FollowEvent)
def on_follow(event: FollowEvent):
    if line_bot_api:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage("歡迎加入！\n- 貼文字：B/P/T 或 莊/閒/和（可用逗號或空白分隔）\n- 貼圖片：可解析大路/下三路（清晰、完整）")
        )

@line_handler.add(MessageEvent, message=TextMessage)
def on_text(event: MessageEvent):
    text = (event.message.text or "").strip()
    seq = parse_seq_from_text(text)
    if not seq:
        msg = (
            "格式小抄：請貼 B P T 序列（支援中文：莊/閒/和；可用逗號或空白分隔）\n"
            "例：B P P B T B、或：莊 閒 閒 莊 和 莊"
        )
        if line_bot_api:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(msg))
        return

    probs = predict_with_models(seq) if model_bundle.get("loaded") else predict_probs_from_seq_rule(seq)
    reply = render_reply(seq, probs, by_model=model_bundle.get("loaded", False))
    if line_bot_api:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(reply))

@line_handler.add(MessageEvent, message=ImageMessage)
def on_image(event: MessageEvent):
    # 從 LINE 拉回影像 bytes
    seq: List[str] = []
    try:
        if not line_bot_api:
            raise RuntimeError("LINE config missing")
        content = line_bot_api.get_message_content(event.message.id)
        b = io.BytesIO()
        for chunk in content.iter_content():
            b.write(chunk)
        b.seek(0)
        img = Image.open(b).convert("RGB")
        seq = extract_sequence_from_image(img)  # TODO: 接上你的影像管線
    except Exception as e:
        logger.exception(f"[image] fetch/parse error: {e}")

    if not seq:
        msg = (
            "圖片未能解析出牌路 😵‍💫\n"
            "請確保：截圖包含完整大路/下三路、畫質清晰、避免壓縮/反光。\n"
            "也可先改貼文字序列（支援：B/P/T 與 莊/閒/和）。"
        )
        if line_bot_api:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(msg))
        return

    probs = predict_with_models(seq) if model_bundle.get("loaded") else predict_probs_from_seq_rule(seq)
    reply = render_reply(seq, probs, by_model=model_bundle.get("loaded", False))
    if line_bot_api:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(reply))


# -----------------------------------------
# Entrypoint
# -----------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
