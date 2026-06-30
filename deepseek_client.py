import json
import os
import re
from typing import Any, Dict, Optional

import requests


class DeepSeekClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "8"))
        self.enabled = os.getenv("DEEPSEEK_ENABLED", "1") == "1" and bool(self.api_key)
        self.thinking = os.getenv("DEEPSEEK_THINKING", "disabled")
        self.reasoning_effort = os.getenv("DEEPSEEK_REASONING_EFFORT", "low")

    def calibrate(self, feature_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        system = (
            "你是百家樂牌路資料的統計校準器，不保證獲利。"
            "你只能根據輸入的 B/P/T 歷史與模型特徵，輸出小幅機率校準。"
            "不要使用玄學，不要宣稱穩贏。"
            "只回傳 JSON，不要 markdown。"
        )
        user = {
            "task": "baccarat_pattern_calibration",
            "rule": "adjustment 必須很小，建議落在 -0.035 到 0.035。confidence 0~1。",
            "feature_payload": feature_payload,
            "output_schema": {
                "banker_adjust": "float -0.035..0.035",
                "player_adjust": "float -0.035..0.035",
                "tie_adjust": "float -0.020..0.020",
                "confidence": "float 0..1",
                "pattern_label": "string",
                "reason": "string <= 50 Chinese chars",
            },
        }

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            "stream": False,
            "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "220")),
        }
        if self.thinking in {"enabled", "disabled"}:
            body["thinking"] = {"type": self.thinking}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.timeout,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return self._parse_json(content)
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
