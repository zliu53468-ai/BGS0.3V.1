# BGS0.3V.1：兩臂 Contextual LinUCB + Kelly 資金管理核心

本版保留既有 OCR、截圖、牌路掃描與 API 對外欄位，但正式決策核心改為：

```text
依 B/P/T 局數估計的牌靴進度（牌面比例固定中性）
        ↓
固定 32 維 Context Vector X（16 Shoe + 16 Road）
        ↓
載入目前 scope 的 Contextual LinUCB A / b
        ↓
直接選擇 UCB score 較高的手臂
        ↓
不回放歷史、不 bootstrap、不自動更新 A / b
        ↓
保守勝率 42%～58%
        ↓
Fractional Kelly
        ↓
最終下注比例強制 clip 5%～30%
```

DeepSeek、ChatGPT 或其他 LLM **不參與**方向、勝率或注碼計算。

## 重要業務規格

- 系統主要服務 **50～70 局的短靴／中短靴**。
- 正式決策只有兩個手臂：`P`（閒）與 `B`（莊）。
- 每一局都輸出明確方向，不存在第三個不下注手臂。
- `bet_allowed` 固定為 `True`。
- `bet_percentage` 永遠在 **5.0～30.0**。
- `recommend / action / next_round_direction` 永遠是 `B` 或 `P`。
- `recommend_text / action_text` 永遠是 `莊` 或 `閒`。
- 正式方向完全沿用 BBB 網頁版的 Frozen Direct 32D 行為。
- B/P/T 會更新目前 Context；正式預測不會用歷史回放訓練 A/b。
- 精確牌面資料仍可供 API 與診斷使用，但不會改變 BBB 相容的32維 Context。
- OCR / screenshot / road detector 的辨識流程沒有被本次核心重構修改。

## 32 維 Context

固定順序為：

- 1～16：牌靴進度、A～10/J/Q/K 中性比例、physical edge、資料可靠度。
- 17～32：目前方向與龍長、最近莊比例、切換率、hazard、HSMM、大眼仔／小路／曱甴路規律度。

為對齊 BBB 按鈕面板，A～10/J/Q/K 比例固定使用中性 `1.0`，physical edge 與 shoe reliability 固定為 `0.0`；剩餘比例依目前 B/P/T 局數估計。

## LinUCB

每個手臂各自維護 Ridge 線性模型：

```text
p(a) = Xᵀ θ(a) + α * sqrt(Xᵀ A(a)⁻¹ X)
```

- 臂 0：Player (`P`)
- 臂 1：Banker (`B`)
- `α` 預設 `0.5`
- Ridge 預設 `1.0`

短靴每靴只有約 50～70 個樣本，因此探索係數與 Ridge 正則化都採保守設定，避免少量結果令係數過度偏移。

模型狀態預設持久化於 `/var/data/contextual_linucb_state.json`，無法寫入時會改用專案 `data/` 或 `/tmp/`。

### Frozen Direct

正式 `predict()` 只載入目前 A/b、計算最新32維 Context、直接預測並保存 `last_selected`。它不做 Walk-forward、不結算上一筆 prediction、不 decay，也不更新 A/b。`update_bandit()` 僅保留為明確呼叫時使用的相容 API。

## Kelly 資金管理

正式勝率先限制到 `0.48～0.58`，再計算：

```text
q = 1 - p
b = 1.00  # Player
b = 0.95  # Banker
full_kelly = (p * b - q) / b
fractional_kelly = full_kelly * KELLY_FRACTION
final_bet_fraction = clip(fractional_kelly, 0.05, 0.30)
```

產品規格要求每局都有實際下注比例，因此最終比例有 **5% 硬下限**；為避免短靴少樣本造成過度資金暴露，同時有 **30% 硬上限**。

## 環境變數

```env
LINUCB_ALPHA=0.5
LINUCB_RIDGE=1.0
LINUCB_UPDATE_WEIGHT=1.0
KELLY_FRACTION=0.25
MIN_BET_FRACTION=0.05
MAX_BET_FRACTION=0.30
SHOE_DECKS=8
ESTIMATED_CARDS_PER_ROUND=4.8
```

## 部署

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port $PORT
```

百家樂單局仍具有高度隨機性；LinUCB 與 Kelly 是決策與資金控制方法，不代表下一局結果可被保證。
