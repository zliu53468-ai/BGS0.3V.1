# BGS0.3V.1：兩臂 Contextual LinUCB + Kelly 資金管理核心

本版保留既有 OCR、截圖、牌路掃描與 API 對外欄位，但正式決策核心改為：

```text
殘餘牌靴狀態（精確資料優先，否則估計/中性值）
        ↓
固定 16 維 Context Vector X
        ↓
Contextual LinUCB（只有 Player / Banker 兩臂）
        ↓
選擇 UCB score 較高的手臂
        ↓
保守勝率 48%～58%
        ↓
Fractional Kelly
        ↓
最終下注比例強制 clip 5%～30%
        ↓
真實開牌結果回饋更新該手臂 Ridge A / b
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
- 牌組相關特徵優先；最近 B 比例、龍長、切換率等牌路特徵只作輔助 Context。
- OCR / screenshot / road detector 的辨識流程沒有被本次核心重構修改。

## 16 維 Context

固定順序如下：

1. 剩餘總張數比例 `remaining_cards / (52 * SHOE_DECKS)`
2. A 剩餘比例
3. 2 剩餘比例
4. 3 剩餘比例
5. 4 剩餘比例
6. 5 剩餘比例
7. 6 剩餘比例
8. 7 剩餘比例
9. 8 剩餘比例
10. 9 剩餘比例
11. 10/J/Q/K 合併剩餘比例
12. 10/J/Q/K 與 4 的比例差
13. 最近 8 局 B 比例
14. 最近 12 局 B 比例
15. 當前同側連續長度（0～1）
16. 最近 12 局切換率（0～1）

如果有 `remaining_counts`，優先使用精確點數剩餘張數；如果有 `observed_cards`，則由已出牌估計剩餘 bucket。兩者都沒有時，牌種比例使用中性 `1.0`，總剩餘張數依目前局數作合理估計，因此不會因缺少精確牌組資料中斷決策。

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

### Online feedback

正式預測時會保存上一局的 `selected arm + context vector + history fingerprint`。下一個 session 請求帶入新實際結果後，系統會在同一決策流程先結算上一局，再產生下一局方向：

- 選閒且閒贏：`+1.0`
- 選莊且莊贏：`+0.95`
- 選錯：`-1.0`
- 和局：`0.0`

更新形式：

```text
A(a) ← A(a) + X Xᵀ
b(a) ← b(a) + reward * X
θ(a) = A(a)⁻¹ b(a)
```

虛擬牌靴流程則在開牌後直接呼叫 feedback update；真實 screenshot/session 流程在新結果進入下一次 `predict()` 時立即對齊上一個 pending prediction 更新。

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
