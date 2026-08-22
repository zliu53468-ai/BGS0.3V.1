# BGS0.3V.1：精確牌靴 EV／Kelly 風控版

本版保留原有牌路模型、Contextual Bandit 與 Adaptive Ensemble 作為診斷，
但正式資金方向改由「精確剩餘牌組 → 不放回機率 → 抽水後 EV → 分數凱利」決定。
DeepSeek／其他 LLM 不參與方向、機率或下注比例。

## 重要資料限制

只知道每局是莊、閒或和，無法知道該局實際移除了哪些點數牌，因此不能精確算牌。
若沒有下列其中一種資料，系統會正確輸出 `O`（觀望）與 0% 投注：

- `remaining_counts`：依點數 0..9 排列的精確剩餘張數；或
- `observed_cards`：本靴所有已出實際牌面，A=1、2..9 原值、10/J/Q/K=0。

可透過 API 儲存指定 UID 的已出牌面：

```http
POST /api/shoe/cards
Content-Type: application/json

{
  "user_id": "LINE_UID",
  "cards": ["A", 8, "K", 3, 10, 6],
  "replace": true
}
```

`replace=true` 表示送入本靴截至目前的完整牌面；`false` 表示追加。

## EV 與資金規則

- 莊注：贏時淨賺 0.95，輸時 -1，和局退回。
- 閒注：贏時 +1，輸時 -1，和局退回。
- 預設只有最佳淨 EV 至少 `+0.5%` 才開放方向。
- 投注比例使用四分之一 Kelly，且硬上限為本金 2%。
- 沒有精確牌組、EV 未達門檻或 Kelly 為 0：一律觀望、0 元。

可用環境變數調整：

```env
SHOE_DECKS=8
BANKER_COMMISSION=0.05
PHYSICAL_MIN_EV=0.005
KELLY_FRACTION=0.25
MAX_BET_FRACTION=0.02
```

不要為了增加出手次數把 `PHYSICAL_MIN_EV` 調成負值；這會重新允許負期望下注。

## 部署

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port $PORT
```

百家樂單局仍是高變異事件；正期望只描述長期數學條件，不保證下一局命中或獲利。
