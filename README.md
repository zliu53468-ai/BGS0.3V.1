# BGS Dual 3M DB LINE Bot

這是一套可直接 push 到 GitHub、部署到 Render 的 LINE 官方帳號機器人。

## 這版包含

- `data/point_db_3m.json`
  - 300萬組點數模擬樣本
  - 壓縮成 100 組閒莊點數統計

- `data/result_pattern_db_3m.json`
  - 300萬組莊閒規律模擬樣本
  - 壓縮成 W3 / W5 / W7 路單 pattern 統計

- 不使用「觀望」
- 玩家直接輸入 `65`
- 點數規則：先閒點，再莊點
- 快捷按鈕：開始分析、結束分析
- Gemini 可選開啟；Gemini 只負責文字，不改機率

## 預測融合

預設權重：

```env
POINT_WEIGHT=0.58
PATTERN_WEIGHT=0.30
SIM_WEIGHT=0.12
```

代表：

```text
點數資料庫 58%
莊閒規律資料庫 30%
本地AI模擬層 12%
```

前幾局沒有足夠 pattern 時，系統會自動把 pattern 權重轉給點數資料庫。

## LINE 操作

```text
遊戲設定
```

選擇：

```text
3
```

輸入桌號：

```text
DG05
```

開始：

```text
開始分析
```

每局輸入：

```text
65
```

代表：

```text
閒 6
莊 5
```

## Render 設定

Build Command：

```bash
pip install -r requirements.txt
```

Start Command：

```bash
gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
```

Webhook URL：

```text
https://你的-render網址.onrender.com/webhook
```

## Render 環境變數

必填：

```env
LINE_CHANNEL_ACCESS_TOKEN=你的_LINE_Channel_Access_Token
LINE_CHANNEL_SECRET=你的_LINE_Channel_Secret
```

建議：

```env
APP_NAME=BGS_DUAL_3M_DB_LINE_BOT
TZ=Asia/Taipei
SESSION_EXPIRE_SECONDS=1800
ENABLE_SIGNATURE_VERIFY=1
DEFAULT_GAME=DG
DEFAULT_TABLE=DG05
PLAYER_INPUT_FIRST=1
NO_OBSERVE=1
REPLY_STYLE=COOL
POINT_DB_PATH=data/point_db_3m.json
RESULT_PATTERN_DB_PATH=data/result_pattern_db_3m.json
USE_POINT_DB=1
USE_PATTERN_DB=1
POINT_WEIGHT=0.58
PATTERN_WEIGHT=0.30
SIM_WEIGHT=0.12
MIN_OUTPUT_PROB=0.05
MAX_OUTPUT_PROB=0.95
PERCENT_DECIMALS=2
GEMINI_ENABLE=0
GEMINI_MODEL=gemini-flash-latest
AI_TEXT_ENABLE=1
```

Gemini 要開啟時：

```env
GEMINI_ENABLE=1
GEMINI_API_KEY=你的_Gemini_API_Key
```

## 測試

首頁：

```text
https://你的-render網址.onrender.com/
```

成功會看到：

```text
OK - BGS Dual 3M DB LINE BOT is running
```

健康檢查：

```text
https://你的-render網址.onrender.com/health
```

應看到：

```json
{
  "ok": true,
  "point_db_samples": 3000000,
  "pattern_db_samples": 3000000
}
```

## 重要說明

這套是機率模擬與技術測試，不保證任何結果。
百家樂具有隨機性，輸出不可視為穩贏或保證命中。
