import json
import requests
from typing import Dict, Any
from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_ENABLE, AI_TEXT_ENABLE

def gemini_enabled() -> bool:
    return bool(GEMINI_ENABLE and AI_TEXT_ENABLE and GEMINI_API_KEY)

def build_fallback_text(prediction: Dict[str, Any]) -> str:
    rec = prediction.get("recommend", "莊")
    p = prediction.get("player_prob", 0)
    b = prediction.get("banker_prob", 0)
    ps = prediction.get("point_sample_size", 0)
    rs = prediction.get("pattern_sample_size", 0)
    return f"🤖 雙3M資料庫比對完成：本次模型偏向{rec}。閒 {p}%、莊 {b}%｜點數樣本{ps}＋路單樣本{rs} ⚡"

def explain(prediction: Dict[str, Any]) -> str:
    if not gemini_enabled():
        return build_fallback_text(prediction)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    prompt = f"""
你是繁體中文的百家樂數據解說助理。
只能根據 JSON 內容做簡短解說，不可以改變 recommend、player_prob、banker_prob。
不得說穩贏、必勝、保證命中、凹單、加碼。
請用 45 字內，語氣酷一點，加入 1-2 個表情符號。

JSON:
{json.dumps(prediction, ensure_ascii=False)}
"""

    body = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    try:
        res = requests.post(url, headers=headers, json=body, timeout=8)
        res.raise_for_status()
        data = res.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text[:120] if text else build_fallback_text(prediction)
    except Exception:
        return build_fallback_text(prediction)
