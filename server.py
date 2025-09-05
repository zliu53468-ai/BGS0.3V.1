#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGS LINE Bot backend — Buttons only (no image analysis)

功能：
- 健康檢查：/、/health、/healthz
- 預測 API：/predict（保留後端測試用）
- LINE Webhook：/line-webhook（只用按鈕互動）
  - 按鈕：莊/閒/和/開始分析/結束分析
  - 未開始分析前：只累積牌路，不給建議
  - 按下「開始分析」後：每次輸入都回機率＋建議
  - 「結束分析」：清空當前牌局
"""

import os
import logging
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-backend")

# -----------------------------
# 基礎機率（可依需求調整或替換）
# -----------------------------
THEORETICAL_PROBS: Dict[str, float] = {"B": 0.458, "P": 0.446, "T": 0.096}
CLASS_ORDER = ("B", "P", "T")

def parse_history(payload) -> List[str]:
    """把使用者傳入的字串或陣列，轉成 ['B','P','T',...]"""
    if payload is None:
        return []
    seq: List[str] = []
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s, str):
                up = s.strip().upper()
                if up in CLASS_ORDER:
                    seq.append(up)
    elif isinstance(payload, str):
        for ch in payload:
            up = ch.upper()
            if up in CLASS_ORDER:
                seq.append(up)
    return seq

def theoretical_probs(_: List[str]) -> List[float]:
    """這裡先用固定的理論機率；需要可替換為你的集成邏輯"""
    return [THEORETICAL_PROBS["B"], THEORETICAL_PROBS["P"], THEORETICAL_PROBS["T"]]

# （選配）模型匯入：先不強制，避免雲端建置過重
try:
    import torch  # type: ignore
    import torch.nn as tnn  # type: ignore
except Exception:
    torch = None  # type: ignore
    tnn = None    # type: ignore

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

if tnn is not None:
    class TinyRNN(tnn.Module):  # type: ignore
        def __init__(self, in_dim: int = 3, hidden: int = 16, out_dim: int = 3) -> None:
            super().__init__()
            self.rnn = tnn.GRU(in_dim, hidden, batch_first=True)
            self.fc = tnn.Linear(hidden, out_dim)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])
else:
    TinyRNN = None  # type: ignore

RNN_MODEL: Optional[Any] = None
if TinyRNN is not None and torch is not None:
    rnn_path = os.getenv("RNN_PATH", "")
    if rnn_path and os.path.exists(rnn_path):
        try:
            _m = TinyRNN()
            _m.load_state_dict(torch.load(rnn_path, map_location="cpu"))
            _m.eval()
            RNN_MODEL = _m
            logger.info("Loaded RNN model from %s", rnn_path)
        except Exception as e:
            logger.warning("Load RNN failed: %s", e)

XGB_MODEL = None
if xgb is not None:
    xgb_path = os.getenv("XGB_PATH", "")
    if xgb_path and os.path.exists(xgb_path):
        try:
            booster = xgb.Booster()
            booster.load_model(xgb_path)
            XGB_MODEL = booster
            logger.info("Loaded XGB model from %s", xgb_path)
        except Exception as e:
            logger.warning("Load XGB failed: %s", e)

LGBM_MODEL = None
if lgb is not None:
    lgbm_path = os.getenv("LGBM_PATH", "")
    if lgbm_path and os.path.exists(lgbm_path):
        try:
            booster = lgb.Booster(model_file=lgbm_path)
            LGBM_MODEL = booster
            logger.info("Loaded LGBM model from %s", lgbm_path)
        except Exception as e:
            logger.warning("Load LGBM failed: %s", e)

def rnn_predict(seq: List[str]) -> Optional[List[float]]:
    if RNN_MODEL is None or torch is None or not seq:
        return None
    try:
        def onehot(label: str) -> List[int]:
            return [1 if label == lab else 0 for lab in CLASS_ORDER]
        inp = torch.tensor([[onehot(ch) for ch in seq]], dtype=torch.float32)
        with torch.no_grad():
            logits = RNN_MODEL(inp)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        return [float(p) for p in probs]
    except Exception as e:
        logger.warning("RNN inference failed: %s", e)
        return None

def xgb_predict(seq: List[str]) -> Optional[List[float]]:
    if XGB_MODEL is None or not seq:
        return None
    try:
        import numpy as np  # lazy
        K = 20
        vec: List[float] = []
        for label in seq[-K:]:
            vec.extend([1.0 if label == lab else 0.0 for lab in CLASS_ORDER])
        pad = K * 3 - len(vec)
        if pad > 0:
            vec = [0.0] * pad + vec
        dmatrix = xgb.DMatrix(np.array([vec], dtype=float))
        prob = XGB_MODEL.predict(dmatrix)[0]
        if isinstance(prob, (list, tuple)) and len(prob) == 3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        elif len(prob) == 2:
            return [float(prob[0]), float(prob[1]), 0.05]
        return None
    except Exception as e:
        logger.warning("XGB inference failed: %s", e)
        return None

def lgbm_predict(seq: List[str]) -> Optional[List[float]]:
    if LGBM_MODEL is None or not seq:
        return None
    try:
        import numpy as np
        K = 20
        vec: List[float] = []
        for label in seq[-K:]:
            vec.extend([1.0 if label == lab else 0.0 for lab in CLASS_ORDER])
        pad = K * 3 - len(vec)
        if pad > 0:
            vec = [0.0] * pad + vec
        prob = LGBM_MODEL.predict([vec])[0]
        if isinstance(prob, (list, tuple)) and len(prob) == 3:
            return [float(prob[0]), float(prob[1]), float(prob[2])]
        elif len(prob) == 2:
            return [float(prob[0]), float(prob[1]), 0.05]
        return None
    except Exception as e:
        logger.warning("LGBM inference failed: %s", e)
        return None

def fuse_probs(rule_p: List[float],
               rnn_p: Optional[List[float]],
               xgb_p: Optional[List[float]],
               lgb_p: Optional[List[float]]) -> List[float]:
    w_rule = float(os.getenv("RULE_W", "1.0"))
    w_rnn  = float(os.getenv("RNN_W",  "0.0"))
    w_xgb  = float(os.getenv("XGB_W",  "0.0"))
    w_lgb  = float(os.getenv("LGBM_W", "0.0"))
    total = max(1e-9, w_rule + (w_rnn if rnn_p else 0.0) + (w_xgb if xgb_p else 0.0) + (w_lgb if lgb_p else 0.0))
    out = [w_rule*rule_p[i] for i in range(3)]
    if rnn_p: out = [out[i] + w_rnn*rnn_p[i] for i in range(3)]
    if xgb_p: out = [out[i] + w_xgb*xgb_p[i] for i in range(3)]
    if lgb_p: out = [out[i] + w_lgb*lgb_p[i] for i in range(3)]
    return [v/total for v in out]

# -----------------------------
# 健康檢查 / 預測
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return "ok"

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="healthy", version="v1")

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify(status="healthy")

@app.route("/predict", methods=["POST"])
def predict():
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    seq = parse_history(data.get("history"))
    base = theoretical_probs(seq)
    pr_rnn = rnn_predict(seq)
    pr_xgb = xgb_predict(seq)
    pr_lgb = lgbm_predict(seq)
    probs = fuse_probs(base, pr_rnn, pr_xgb, pr_lgb)
    labels = list(CLASS_ORDER)
    rec = labels[probs.index(max(probs))]
    return jsonify({
        "history_len": len(seq),
        "probabilities": {labels[i]: probs[i] for i in range(3)},
        "recommendation": rec
    })

# -----------------------------
# LINE Webhook（按鈕互動）
# -----------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

USE_LINE = False
try:
    from linebot import LineBotApi, WebhookHandler  # type: ignore
    from linebot.models import (  # type: ignore
        MessageEvent, TextMessage, TextSendMessage,
        PostbackEvent, PostbackAction,
        FlexSendMessage,
        QuickReply, QuickReplyButton
    )
    USE_LINE = bool(LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET)
except Exception as e:
    logger.warning("LINE SDK not available or env not set: %s", e)
    USE_LINE = False

if USE_LINE:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
else:
    line_bot_api = None
    handler = None

# 使用者狀態：每位用戶各自的牌路與「是否開始分析」
USER_HISTORY: Dict[str, List[str]] = {}
USER_READY: Dict[str, bool] = {}

def flex_buttons_card() -> FlexSendMessage:
    """產生類似卡片的 FLEX 訊息 + 行動按鈕（莊/閒/和/開始/結束）"""
    contents = {
        "type": "bubble",
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {"type": "text", "text": "🤖 請開始輸入歷史數據", "weight": "bold", "size": "lg"},
                {"type": "text", "text": "請點擊下方按鈕依序輸入過往莊/閒/和；\n按「開始分析」後才會給出下注建議。", "wrap": True, "size": "sm", "color": "#555555"},
                {
                    "type": "box",
                    "layout": "horizontal",
                    "spacing": "sm",
                    "contents": [
                        {"type": "button", "style": "primary", "color": "#E74C3C",
                         "action": {"type": "postback", "label": "莊", "data": "B"}},
                        {"type": "button", "style": "primary", "color": "#2980B9",
                         "action": {"type": "postback", "label": "閒", "data": "P"}},
                        {"type": "button", "style": "primary", "color": "#27AE60",
                         "action": {"type": "postback", "label": "和", "data": "T"}}
                    ]
                },
                {
                    "type": "box",
                    "layout": "horizontal",
                    "spacing": "sm",
                    "contents": [
                        {"type": "button", "style": "secondary",
                         "action": {"type": "postback", "label": "開始分析", "data": "START"}},
                        {"type": "button", "style": "secondary",
                         "action": {"type": "postback", "label": "結束分析", "data": "END"}}
                    ]
                }
            ]
        }
    }
    return FlexSendMessage(alt_text="請開始輸入歷史數據", contents=contents)

def quick_reply_bar() -> QuickReply:
    """在每次回覆下方也附上同組快速按鈕"""
    return QuickReply(items=[
        QuickReplyButton(action=PostbackAction(label="莊", data="B")),
        QuickReplyButton(action=PostbackAction(label="閒", data="P")),
        QuickReplyButton(action=PostbackAction(label="和", data="T")),
        QuickReplyButton(action=PostbackAction(label="開始分析", data="START")),
        QuickReplyButton(action=PostbackAction(label="結束分析", data="END")),
    ])

@app.route("/line-webhook", methods=["POST"])
def line_webhook():
    if not USE_LINE or handler is None:
        logger.warning("LINE webhook hit but LINE SDK/env not configured.")
        return "ok", 200
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        logger.error("LINE handle error: %s", e)
        return "ok", 200
    return "ok", 200

# ---- 事件處理：文字 → 顯示引導與按鈕
if USE_LINE and handler is not None:
    @handler.add(MessageEvent, message=TextMessage)
    def handle_text(event):
        uid = event.source.user_id
        USER_HISTORY.setdefault(uid, [])
        USER_READY.setdefault(uid, False)
        help_msg = "請使用按鈕輸入（莊/閒/和）。\n按「開始分析」後才會給出下注建議。"
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=help_msg, quick_reply=quick_reply_bar()),
                flex_buttons_card()
            ]
        )

    # ---- 事件處理：Postback → 實際業務邏輯
    @handler.add(PostbackEvent)
    def handle_postback(event):
        uid = event.source.user_id
        data = (event.postback.data or "").upper()

        # 初始化
        seq = USER_HISTORY.get(uid, [])
        ready = USER_READY.get(uid, False)

        if data == "START":
            USER_READY[uid] = True
            msg = "🔎 已開始分析。請繼續輸入莊/閒/和，我會根據資料給出建議。"
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            )
            return

        if data == "END":
            USER_HISTORY[uid] = []
            USER_READY[uid] = False
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="✅ 已結束分析，紀錄已清空。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            )
            return

        # 只接受 B/P/T
        if data not in ("B", "P", "T"):
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text="請用按鈕輸入：莊/閒/和；或選開始/結束分析。", quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            )
            return

        # 累積牌路
        seq.append(data)
        USER_HISTORY[uid] = seq

        # 尚未開始分析：只回累積狀態
        if not ready:
            msg = f"已記錄 {len(seq)} 手：{''.join(seq)}\n按「開始分析」後才會給出下注建議。"
            line_bot_api.reply_message(
                event.reply_token,
                [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
                 flex_buttons_card()]
            )
            return

        # 已開始分析：計算建議
        base = theoretical_probs(seq)
        pr_rnn = rnn_predict(seq)
        pr_xgb = xgb_predict(seq)
        pr_lgb = lgbm_predict(seq)
        probs = fuse_probs(base, pr_rnn, pr_xgb, pr_lgb)
        labels = ["莊", "閒", "和"]
        rec = labels[probs.index(max(probs))]
        msg = (
            f"已解析 {len(seq)} 手\n"
            f"機率：莊 {probs[0]:.3f}｜閒 {probs[1]:.3f}｜和 {probs[2]:.3f}\n"
            f"建議：{rec}"
        )
        line_bot_api.reply_message(
            event.reply_token,
            [TextSendMessage(text=msg, quick_reply=quick_reply_bar()),
             flex_buttons_card()]
        )

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
