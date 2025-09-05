# app.py — LINE Bot (Buttons-only, no OCR)
import os, logging
from typing import Dict, List

from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    FollowEvent, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    ButtonComponent, TextComponent, PostbackAction
)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bgs-bot")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN) if LINE_CHANNEL_ACCESS_TOKEN else None
line_handler = WebhookHandler(LINE_CHANNEL_SECRET) if LINE_CHANNEL_SECRET else None

analysis_enabled: Dict[str, bool] = {}
user_history_seq: Dict[str, List[str]] = {}

def _ratio_lastN(seq: List[str], N: int):
    s = seq[-N:] if len(seq)>=N else seq
    if not s: return (0.33,0.33,0.34)
    n=len(s); return (s.count("B")/n, s.count("P")/n, s.count("T")/n)

def _streak_tail(seq: List[str]) -> int:
    if not seq: return 0
    t, c = seq[-1], 1
    for i in range(len(seq)-2, -1, -1):
        if seq[i]==t: c+=1
        else: break
    return c

def predict_probs(seq: List[str]):
    if not seq:
        return {"banker":0.34, "player":0.34, "tie":0.32}
    pb,pp,pt = _ratio_lastN(seq, len(seq))
    tail = _streak_tail(seq)
    if seq[-1] in {"B","P"}:
        boost = min(0.10, 0.03*(tail-1))
        if seq[-1]=="B": pb += boost
        else: pp += boost
    pt = max(0.02, min(0.15, pt))
    s = pb+pp+pt
    return {"banker":round(pb/s,4), "player":round(pp/s,4), "tie":round(pt/s,4)}

def render_reply(seq: List[str], probs):
    b=probs["banker"]; p=probs["player"]; t=probs["tie"]
    side = "莊" if b>=p else "閒"
    side_prob = max(b,p)
    diff = abs(b-p)
    if diff < 0.05:
        suggest = "觀望（勝率差距不足 5%）"
    else:
        suggest = f"建議：{side}（勝率 {side_prob*100:.1f}%）"
    return (
        f"已解析 {len(seq)} 手\n"
        f"機率：莊 {b*100:.1f}%｜閒 {p*100:.1f}%｜和 {t*100:.1f}%\n"
        f"{suggest}"
    )

def make_baccarat_buttons(prompt_text: str, title_text: str) -> FlexSendMessage:
    buttons = [
        ButtonComponent(action=PostbackAction(label="莊", data="choice=banker"), style="primary", color="#E53935", height="sm", flex=1),
        ButtonComponent(action=PostbackAction(label="閒", data="choice=player"), style="primary", color="#1E88E5", height="sm", flex=1),
        ButtonComponent(action=PostbackAction(label="和", data="choice=tie"), style="primary", color="#43A047", height="sm", flex=1),
    ]
    bubble = BubbleContainer(
        size="mega",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title_text, weight="bold", size="lg", align="center")]),
        body=BoxComponent(layout="vertical", contents=[TextComponent(text=prompt_text, size="md")]),
        footer=BoxComponent(layout="horizontal", spacing="sm", contents=buttons),
    )
    return FlexSendMessage(alt_text=title_text, contents=bubble)

def prompt_buttons(uid: str, reply_token: str|None=None, subtitle="請點擊下方按鈕依序輸入過往莊/閒/和結果："):
    if uid not in user_history_seq: user_history_seq[uid]=[]
    flex = make_baccarat_buttons(subtitle, "🤖請開始輸入歷史數據")
    if reply_token: line_bot_api.reply_message(reply_token, [flex])
    else: line_bot_api.push_message(uid, flex)

@app.get("/")
def index():
    return "BGS AI（按鈕版）運行中 ✅，/line-webhook 已就緒", 200

@app.get("/health")
def health():
    return jsonify(ok=True), 200

@app.post("/line-webhook")
def line_webhook():
    if not (line_bot_api and line_handler):
        return "Line credentials missing", 200
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        logger.exception("Invalid signature")
    return "OK", 200

if line_handler and line_bot_api:
    @line_handler.add(FollowEvent)
    def on_follow(event: FollowEvent):
        uid = getattr(event.source, "user_id", "unknown")
        analysis_enabled[uid] = False
        user_history_seq[uid] = []
        welcome = (
            "歡迎加入 BGS AI 助手 🎉\n\n"
            "先用按鈕輸入歷史莊/閒/和；輸入「開始分析」後，我才會開始回覆下注建議。\n"
            "隨時輸入「結束分析」可清除資料並重新開始。"
        )
        flex = make_baccarat_buttons("請點擊下方按鈕依序輸入過往莊/閒/和結果：", "🤖請開始輸入歷史數據")
        line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=welcome), flex])

    @line_handler.add(MessageEvent, message=TextMessage)
    def on_text(event: MessageEvent):
        uid = getattr(event.source, "user_id", "unknown")
        text = (event.message.text or "").strip()

        if text in {"結束分析","结束分析"}:
            analysis_enabled[uid] = False
            user_history_seq[uid] = []
            msg = "已結束本輪分析，所有歷史數據已刪除。\n請使用下方按鈕重新輸入歷史數據。"
            flex = make_baccarat_buttons("請點擊下方按鈕依序輸入過往莊/閒/和結果：", "🤖請開始輸入歷史數據")
            line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=msg), flex])
            return

        if text in {"開始分析","开始分析","開始","开始","START","分析"}:
            analysis_enabled[uid] = True
            seq = user_history_seq.get(uid, [])
            if len(seq) >= 5:
                probs = predict_probs(seq)
                msg = "已開始分析 ✅\n" + render_reply(seq, probs)
            else:
                msg = "已開始分析 ✅\n目前資料不足（至少 5 手）。先繼續用按鈕輸入歷史結果，我會再給出建議。"
            flex = make_baccarat_buttons("持續點擊下方按鈕輸入新一手結果：", "下注選擇")
            line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=msg), flex])
            return

        hint = "請先使用下方按鈕輸入歷史莊/閒/和；\n輸入「開始分析」後，我才會開始回覆下注建議。"
        flex = make_baccarat_buttons("請點擊下方按鈕依序輸入過往莊/閒/和結果：", "🤖請開始輸入歷史數據")
        line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=hint), flex])

    @line_handler.add(PostbackEvent)
    def on_postback(event: PostbackEvent):
        uid = getattr(event.source, "user_id", "unknown")
        data = event.postback.data or ""
        params = dict(x.split("=",1) for x in data.split("&") if "=" in x)
        choice = params.get("choice")
        map_ = {"banker":"B","player":"P","tie":"T"}
        if choice not in map_:
            line_bot_api.reply_message(event.reply_token, [
                TextSendMessage(text="收到未知操作，請重新選擇。"),
                make_baccarat_buttons("請點擊下方按鈕輸入：","下注選擇")
            ])
            return

        seq = user_history_seq.get(uid, [])
        seq.append(map_[choice])
        user_history_seq[uid] = seq

        if analysis_enabled.get(uid):
            probs = predict_probs(seq)
            text = render_reply(seq, probs)
        else:
            text = f"已記錄：{len(seq)} 手（例：{''.join(seq[-12:])}）\n輸入「開始分析」後，我才會開始回覆下注建議。"

        flex = make_baccarat_buttons("持續點擊下方按鈕輸入新一手結果：", "下注選擇")
        line_bot_api.reply_message(event.reply_token, [TextSendMessage(text=text), flex])

if __name__ == "__main__":
    port = int(os.getenv("PORT","5000"))
    app.run(host="0.0.0.0", port=port)
