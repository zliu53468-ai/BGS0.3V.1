# Baccarat LINE LIFF AI Bot

這版是「LINE 選擇遊戲館 Flex UI + LIFF 莊閒和輸入面板 + 規律模型 + DeepSeek API 校準」。

## 流程

1. 用戶在 LINE 輸入「開始分析」
2. Bot 回覆「選擇遊戲館」圖文 UI 按鈕，不放圖片
3. 用戶點遊戲館後開啟 LIFF 面板
4. 在 LIFF 裡點「莊 / 閒 / 和」只更新後端 session 和畫面，不會在聊天室洗版
5. 點「開始AI判斷」才呼叫 predictor.py + DeepSeek 校準
6. LIFF 顯示莊/閒/和機率、推薦方向、信心等級、規律原因

## Render 部署

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

LINE Webhook URL:

```text
https://你的render網域.onrender.com/callback
```

LIFF Endpoint URL:

```text
https://你的render網域.onrender.com/liff
```

## 環境變數

請把 `.env.example` 裡面的變數逐一放到 Render Environment。

## 注意

百家樂結果具有高度隨機性，這個模型是牌路/歷史序列的統計分析與 UI 系統，不保證獲利、不保證穩定提高真實下注勝率。
