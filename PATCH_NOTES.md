# 精確牌組 EV／Kelly 核心修正

## 優先重寫檔案

1. `shoe_composition.py`（新增）：精確牌組解析、正式補牌規則、不放回枚舉、EV、Kelly。
2. `predictor.py`：把物理 EV 設為最後資金閘門；原牌路與 cMAB 只保留診斷。
3. `screenshot_predictor.py`：不再丟棄 `prior_counts`，並支援 `observed_cards`。
4. `app.py`：以 Kelly 取代固定信心投注百分比，顯示 EV 並提供牌面 API。
5. `store.py`：每個 UID 分開保存本靴實際已出牌面。
6. `static/liff.html`：移除錯誤的 DeepSeek 標示，加入實際牌面輸入與 EV/Kelly 文案。

## 沒有改動的核心

`road_model.py`、`contextual_bandit.py`、`adaptive_ensemble.py` 的既有模型公式與學習參數
均未在本次重寫。它們仍會產生診斷與績效資料，但無法繞過物理 EV 閘門直接建議下注。

## 防幻覺規則

正式 Python 路徑沒有 DeepSeek 呼叫。本版輸出 `deepseek_active=false`、
`llm_decision_allowed=false`，摘要由固定程式格式建立，且明確禁止以連莊、連閒、
斷龍等敘事推導下一局必然反轉。

## 驗證基準

全新 8 副牌靴的精確計算應為：

- 莊：45.859742%
- 閒：44.624661%
- 和：9.515597%

莊、閒在 5% 莊抽水後均為負 EV，因此新牌靴正確行為是觀望、Kelly 0%。
