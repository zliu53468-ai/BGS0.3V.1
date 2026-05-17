import requests
from typing import Dict, Any, List
from config import LINE_CHANNEL_ACCESS_TOKEN

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

def quick_reply_items() -> Dict[str, Any]:
    return {
        "items": [
            {
                "type": "action",
                "action": {
                    "type": "message",
                    "label": "🚀 開始分析",
                    "text": "開始分析"
                }
            },
            {
                "type": "action",
                "action": {
                    "type": "message",
                    "label": "🛑 結束分析",
                    "text": "結束分析"
                }
            }
        ]
    }

def text_message(text: str, with_buttons: bool = True) -> Dict[str, Any]:
    msg = {"type": "text", "text": text}
    if with_buttons:
        msg["quickReply"] = quick_reply_items()
    return msg

def reply_messages(reply_token: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return {"ok": False, "error": "LINE_CHANNEL_ACCESS_TOKEN missing"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }

    body = {"replyToken": reply_token, "messages": messages[:5]}
    res = requests.post(LINE_REPLY_URL, headers=headers, json=body, timeout=10)
    return {"ok": res.status_code in (200, 201, 202, 204), "status_code": res.status_code, "text": res.text}
