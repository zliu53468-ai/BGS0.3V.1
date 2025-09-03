# app.py
import os, io, time, json, math
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

try:
    from hmmlearn.hmm import MultinomialHMM
except Exception:
    MultinomialHMM = None

app = Flask(__name__)

# ---------- ENV ----------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

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
    except Exception as e:
        model_bundle = {"loaded": False, "note": f"load error: {e}"}

load_models()

# =========================================================
# 解析序列（文字/圖片）
# =========================================================
MAP_CH = {"莊":"B", "閒":"P", "和":"T"}
IDX = {"B":0,"P":1,"T":2}

def parse_text_sequence(text: str) -> List[str]:
    s = (text or "").upper()
    s = s.replace(",", " ").replace("，", " ").replace("/", " ").replace("|", " ")
    tokens: List[str] = []
    for ch in s:
        if ch in {"B","P","T"}:
            tokens.append(ch)
        elif ch in MAP_CH:
            tokens.append(MAP_CH[ch])
    for word in s.split():
        if word in {"B","P","T"}:
            tokens.append(word)
        elif word in {"莊","閒","和"}:
            tokens.append(MAP_CH[word])
    return tokens[-240:] if tokens else []

def extract_sequence_from_image(img_bytes: bytes) -> List[str]:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(img); img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        scale = 1200.0 / max(h, w)
        if scale < 1.5: img = cv2.resize(img, (int(w*scale), int(h*scale)))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 70, 60), (180, 255, 255))
        red  = cv2.bitwise_or(red1, red2)
        blue = cv2.inRange(hsv, (90, 70, 60), (130, 255, 255))
        green= cv2.inRange(hsv, (40, 50, 60), (80, 255, 255))
        kernel = np.ones((5,5), np.uint8)
        red   = cv2.morphologyEx(red,   cv2.MORPH_OPEN, kernel)
        blue  = cv2.morphologyEx(blue,  cv2.MORPH_OPEN, kernel)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)

        def blobs(mask, label):
            cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            items = []
            for c in cs:
                if cv2.contourArea(c) < 50: continue
                x,y,w,h = cv2.boundingRect(c)
                cx = x + w/2
                items.append((cx, label))
            return items

        items = blobs(red,"B")+blobs(blue,"P")+blobs(green,"T")
        if not items: return []
        items.sort(key=lambda z: z[0])
        seq = []; last_x = -1e9; min_gap = max(img.shape[1]*0.015, 8)
        for cx, label in items:
            if abs(cx-last_x) < min_gap: continue
            seq.append(label); last_x = cx
        return seq[-240:]
    except Exception:
        return []

# =========================================================
# 特徵 & 模型推理
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

def _proba_from_xgb(feat: np.ndarray) -> Dict[str,float] | None:
    if "xgb_sklearn" in model_bundle:
        proba = model_bundle["xgb_sklearn"].predict_proba(feat)[0]
        return {"banker": float(proba[IDX["B"]]), "player": float(proba[IDX["P"]]), "tie": float(proba[IDX["T"]])}
    if "xgb_booster" in model_bundle and xgb:
        d = xgb.DMatrix(feat)
        proba = model_bundle["xgb_booster"].predict(d)[0]
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

def _normalize(p: Dict[str,float]) -> Dict[str,float]:
    s = p["banker"]+p["player"]+p["tie"]
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {k: round(v/s,4) for k,v in p.items()}

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

# 規則回退
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
        boost = min(0.10, 0.03*(tail-1))
        if seq[-1]=="B": pb+=boost
        else: pp+=boost
    s=pb+pp+pt
    if s<=0: return {"banker":0.34,"player":0.34,"tie":0.32}
    return {"banker":round(pb/s,4),"player":round(pp/s,4),"tie":round(pt/s,4)}

# 下注比例 + 額外「建議預測：莊/閒 XX%」
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

def render_prediction_msg(seq: List[str], probs: Dict[str,float], by_model: bool) -> str:
    b, p, t = probs["banker"], probs["player"], probs["tie"]
    plan = betting_plan(b, p)
    tag = "（模型）" if by_model else "（規則）"
    percent_txt = f"{plan['side_prob']*100:.1f}%"
    suffix = f"｜{plan.get('note')}" if plan.get("note") else ""
    return (
        f"{tag} 已讀取 {len(seq)} 手走勢\n"
        f"機率 → 莊:{b:.2f}  閒:{p:.2f}  和:{t:.2f}\n"
        f"建議預測：{plan['side']} {percent_txt}{suffix}\n"
        f"下注建議：{'觀望' if plan['percent']==0 else f'資金 {plan['percent']*100:.0f}% 於「{plan['side']}」'}"
    )

# =========================================================
# API
# =========================================================
@app.route("/health")
def health():
    return jsonify({"status":"ok","ts":int(time.time()),
                    "models_loaded": model_bundle.get("loaded", False),
                    "note": model_bundle.get("note","")})

@app.route("/api/predict-from-seq", methods=["POST"])
def api_predict_from_seq():
    data = request.get_json(silent=True) or {}
    seq_text = str(data.get("sequence", "")).strip()
    seq = parse_text_sequence(seq_text)
    if not seq:
        return jsonify({"ok": False, "msg": "無法解析序列，請提供 B/P/T 或 莊/閒/和"}), 400
    if model_bundle.get("loaded"):
        probs = predict_with_models(seq)
        by_model = probs is not None
        if not by_model:
            probs = predict_probs_from_seq_rule(seq)
    else:
        probs = predict_probs_from_seq_rule(seq)
        by_model = False
    bp = betting_plan(probs["banker"], probs["player"])
    return jsonify({"ok": True, "sequence": seq, "probs": probs,
                    "betting": bp, "by_model": by_model})

# =========================================================
# LINE Webhook
# =========================================================
@app.route("/line-webhook", methods=['POST'])
def line_webhook():
    if not (line_bot_api and line_handler): abort(403)
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

if line_handler and line_bot_api:

    @line_handler.add(FollowEvent)
    def on_follow(event: FollowEvent):
        welcome = (
            "歡迎加入BGS AI 助手 🎉\n\n"
            "請提供當前牌靴走勢：\n"
            "1）貼文字：B P P B T 或 莊閒閒莊和\n"
            "2）上傳牌路截圖：我會嘗試自動辨識\n\n"
            "我將回覆「建議預測：莊/閒 + 百分比」與下注比例。"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome))

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        text = (event.message.text or "").strip()
        seq = parse_text_sequence(text)
        if not seq:
            tip = (
                "看不出走勢序列 😿\n"
                "請用 B/P/T 或 莊/閒/和 的形式提供，例如：\n"
                "B P B P T P 或  莊閒莊閒和閒\n"
                "也可直接上傳牌路截圖，我會嘗試辨識。"
            )
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tip)); return

        if model_bundle.get("loaded"):
            probs = predict_with_models(seq); by_model = probs is not None
            if not by_model: probs = predict_probs_from_seq_rule(seq)
        else:
            probs = predict_probs_from_seq_rule(seq); by_model = False

        msg = render_prediction_msg(seq, probs, by_model)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))

    @line_handler.add(MessageEvent, message=ImageMessage)
    def on_image(event: MessageEvent):
        content = line_bot_api.get_message_content(event.message.id)
        img_bytes = b"".join(chunk for chunk in content.iter_content())
        seq = extract_sequence_from_image(img_bytes)
        if not seq:
            tip = (
                "圖片辨識失敗 😥\n"
                "請改用文字提供 B/P/T 或 莊/閒/和 的走勢序列。"
            )
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tip)); return

        if model_bundle.get("loaded"):
            probs = predict_with_models(seq); by_model = probs is not None
            if not by_model: probs = predict_probs_from_seq_rule(seq)
        else:
            probs = predict_probs_from_seq_rule(seq); by_model = False

        msg = "（由截圖解析）\n" + render_prediction_msg(seq, probs, by_model)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
