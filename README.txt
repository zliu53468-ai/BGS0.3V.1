# combo_db_3m.json 一鍵覆蓋包

## 放置位置
把 `data/combo_db_3m.json` 放到 GitHub 專案的 `data/` 資料夾。

Render 環境變數：
COMBO_DB_PATH=data/combo_db_3m.json
USE_COMBO_DB=1
COMBO_DB_MIN_SAMPLE=1

## 重要說明
這份 combo_db_3m.json 是「完整 key 對接啟動版」，目的是先讓 LINE 顯示條件樣本不再為 0。
它不是正式每條件 300 萬樣本的最終資料庫。

正式資料請用：
PER_CONDITION_SAMPLES=3000000 INCLUDE_BASE=1 python generate_combo_db_per_condition_3m.py
