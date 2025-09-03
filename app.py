import os
import json
import time
from pathlib import Path
from typing import Dict, Any
from flask import Flask, request, jsonify, send_from_directory, abort

# ====== LINE SDK ======
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, FollowEvent

# ====== ML / Utilities (可選) ======
try:
    import joblib
except Exception:
    joblib = None

# -----------------------------------------------------------------------------
# Flask App（靜態文件根目錄：repo 根）
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')

# -----------------------------------------------------------------------------
# 環境變數
# -----------------------------------------------------------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "").rstrip("/")  # 用於回傳前端指定桌連結

# -----------------------------------------------------------------------------
# LINE 初始化
# -----------------------------------------------------------------------------
line_bot_api = None
line_handler = None
if LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    line_handler = WebhookHandler(LINE_CHANNEL_SECRET)

# -----------------------------------------------------------------------------
# 模型載入（可選）
# -----------------------------------------------------------------------------
MODELS_DIR = Path("models")
SCALER_PATH = MODELS_DIR / "scaler.pkl"
XGB_PATH = MODELS_DIR / "xgb_model.pkl"
LGBM_PATH = MODELS_DIR / "lgbm_model.pkl"
HMM_PATH = MODELS_DIR / "hmm_model.pkl"

model_bundle: Dict[str, Any] = {"loaded": False, "note": "models missing"}

def load_models():
    global model_bundle
    if not MODELS_DIR.exists() or joblib is None:
        model_bundle = {"loaded": False, "note": "models dir or joblib missing"}
        return
    try:
        bundle = {}
        if SCALER_PATH.exists(): bundle["scaler"] = joblib.load(SCALER_PATH)
        if XGB_PATH.exists():    bundle["xgb"]    = joblib.load(XGB_PATH)
        if LGBM_PATH.exists():   bundle["lgbm"]   = joblib.load(LGBM_PATH)
        if HMM_PATH.exists():    bundle["hmm"]    = joblib.load(HMM_PATH)
        if any(k in bundle for k in ("xgb", "lgbm", "hmm")):
            bundle["loaded"] = True
            bundle["note"] = "at least one model loaded"
            model_bundle = bundle
        else:
            model_bundle = {"loaded": False, "note": "no model file found"}
    except Exception as e:
        model_bundle = {"loaded": False, "note": f"load error: {e}"}

load_models()

# -----------------------------------------------------------------------------
# 內存狀態：由 monitor.py 推送的各桌路子 / 最新結果
# -----------------------------------------------------------------------------
tables_state: Dict[str, Dict[str, Any]] = {}
# 使用者會話：user_id -> 已選桌號
user_session: Dict[str, str] = {}

# 允許的桌號清單
RECOGNIZED_TABLES = {
    "D01","D02","D03","D05","D06","D07","D08",
    "A01","A02","A03","A05",
    "C01","C02","C03","C05","C06","C07","C08","C09"
}

# -----------------------------------------------------------------------------
# 健康檢查
# -----------------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "ts": int(time.time()),
        "models_loaded": model_bundle.get("loaded", False),
        "note": model_bundle.get("note", "")
    })

# -----------------------------------------------------------------------------
# 首頁：如有 index.html 就服務；否則回簡訊息
# -----------------------------------------------------------------------------
@app.route("/")
def index_root():
    index_path = Path(app.static_folder) / "index.html"
    if index_path.exists():
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"message": "API is running",
                    "tip": "Put index.html at repo root if you want a homepage."})

# -----------------------------------------------------------------------------
# API：取得桌況摘要
# -----------------------------------------------------------------------------
@app.route("/api/tables", methods=["GET"])
def api_tables():
    resp = []
    for tid, info in tables_state.items():
        resp.append({
            "table_id": tid,
            "last_result": info.get("last_result", None),
            "updated_at": info.get("updated_at", 0),
            "history_len": len(info.get("history", []))
        })
    if not resp:
        resp = [{"table_id": t, "last_result": None, "updated_at": 0, "history_len": 0}
                for t in sorted(RECOGNIZED_TABLES)]
    return jsonify({"tables": resp})

# -----------------------------------------------------------------------------
# API：監控端（monitor.py）回推最新路子/結果
# 兼容 payload:
#   {"table_id":"D01","results":["B","P",...]}
#   或 {"table_id":"D01","last_result":"B","history":[...]}
# -----------------------------------------------------------------------------
@app.route("/api/update-roadmap", methods=["POST"])
def api_update_roadmap():
    data = request.get_json(silent=True) or {}
    table_id = str(data.get("table_id", "")).strip()
    results = data.get("results")
    last_result = data.get("last_result")

    if not table_id:
        abort(400)

    entry = tables_state.setdefault(table_id, {"history": []})

    new_items = []
    if isinstance(results, list) and results:
        new_items = [str(x).upper() for x in results if str(x).upper() in {"B","P","T"}]
    elif isinstance(last_result, str) and str(last_result).upper() in {"B","P","T"}:
        new_items = [str(last_result).upper()]

    if not new_items:
        abort(400)

    entry.setdefault("history", [])
    entry["history"].extend(new_items)
    entry["history"] = entry["history"][-400:]
    entry["last_result"] = entry["history"][-1]
    entry["updated_at"] = int(time.time())

    return jsonify({"ok": True, "table_id": table_id, "added": new_items})

# -----------------------------------------------------------------------------
# API：即時預測
# -----------------------------------------------------------------------------
def _mock_predict_payload() -> Dict[str, Any]:
    return {
        "banker": 0.34,
        "player": 0.33,
        "tie": 0.33,
        "suggestion": "觀望（模型未載入，請先上傳 models/*.pkl）",
        "models_loaded": False
    }

def _real_predict_payload(table_id: str | None = None) -> Dict[str, Any]:
    """
    TODO: 依 table_id 從 tables_state 取最近 N 手做特徵工程 → 用 xgb/lgbm/hmm ensemble
    這裡先回範例值；你可自行接上真模型
    """
    return {
        "banker": 0.41,
        "player": 0.54,
        "tie": 0.05,
        "suggestion": "押 Player（差距 >= 5%）",
        "models_loaded": True
    }

def _plan_bet_from_probs(b: float, p: float) -> Dict[str, Any]:
    """
    下注比例規則（% 以資金比率表示）：
      差距 < 5%     -> 觀望
      5%~8%         -> 2%
      8%~12%        -> 4%
      12%~18%       -> 8%
      ≥18%          -> 12%
    """
    diff = abs(b - p)
    if diff < 0.05:
        return {"side": "觀望", "percent": 0}
    if diff < 0.08:
        pct = 0.02
    elif diff < 0.12:
        pct = 0.04
    elif diff < 0.18:
        pct = 0.08
    else:
        pct = 0.12
    side = "莊" if b > p else "閒"
    return {"side": side, "percent": pct}

@app.route("/api/prediction", methods=["GET"])
def api_prediction():
    table_id = request.args.get("table_id", "").strip() or None

    if model_bundle.get("loaded"):
        payload = _real_predict_payload(table_id)
    else:
        payload = _mock_predict_payload()

    # 衍生下注比例
    b = float(payload.get("banker", 0))
    p = float(payload.get("player", 0))
    bet = _plan_bet_from_probs(b, p)
    payload["betting"] = bet

    # 附桌況摘要
    if table_id and table_id in tables_state:
        hist_len = len(tables_state[table_id].get("history", []))
        payload["table_context"] = {
            "table_id": table_id,
            "history_len": hist_len,
            "last_result": tables_state[table_id].get("last_result")
        }
    return jsonify(payload)

@app.route("/api/predict", methods=["GET"])
def api_predict_alias():
    return api_prediction()

# -----------------------------------------------------------------------------
# LINE Webhook（唯一入口）
# -----------------------------------------------------------------------------
@app.route("/line-webhook", methods=['POST'])
def line_webhook():
    if not (line_bot_api and line_handler):
        abort(403)
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        abort(400)
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# -----------------------------------------------------------------------------
# LINE 事件處理（需 Token/Secret 才註冊）
# -----------------------------------------------------------------------------
if line_handler and line_bot_api:

    @line_handler.add(FollowEvent)
    def handle_follow(event: FollowEvent):
        welcome = (
            "歡迎加入BGS AI 助手 🎉\n\n"
            "請選擇桌號：\n"
            "D01 / D02 / D03 / D05 / D06 / D07 / D08 /\n"
            "A01 / A02 / A03 / A05 /\n"
            "C01 / C02 / C03 / C05 / C06 / C07 / C08 / C09\n\n"
            "直接輸入桌號（例如：D01）即可。"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome))

    @line_handler.add(MessageEvent, message=TextMessage)
    def handle_text_message(event: MessageEvent):
        user_id = event.source.user_id if hasattr(event.source, "user_id") else "unknown"
        txt = (event.message.text or "").strip().upper().replace(" ", "")

        # 若使用者輸入的是桌號
        if txt in RECOGNIZED_TABLES:
            user_session[user_id] = txt

            # 1) 回「正在連接數據…」
            msgs = [TextSendMessage(text=f"正在連接 {txt} 數據…")]

            # 2) 可選：回前端該桌連結
            if FRONTEND_BASE_URL:
                table_url = f"{FRONTEND_BASE_URL}?table={txt}"
                msgs.append(TextSendMessage(text=f"前往該桌走勢：\n{table_url}"))

            # 3) 立即取得預測
            pred = _get_prediction_text(table_id=txt)
            msgs.append(TextSendMessage(text=pred))

            line_bot_api.reply_message(event.reply_token, msgs)
            return

        # 文本指令：即時預測 / 目前桌況
        if "即時預測" in txt or "PREDICT" in txt:
            # 若已選過桌號，帶入；否則不帶桌號
            table_id = user_session.get(user_id)
            reply = _get_prediction_text(table_id=table_id)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
            return

        if "目前桌況" in txt or "TABLES" in txt:
            tables = _summarize_tables()
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tables))
            return

        # 不是桌號也不是指令 → 提示再次選桌
        tips = (
            "請直接輸入桌號（例如：D01）。\n"
            "或輸入：\n"
            "．即時預測（可帶桌號）\n"
            "．目前桌況"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tips))
