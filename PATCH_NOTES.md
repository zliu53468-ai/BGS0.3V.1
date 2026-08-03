# BGS V10.9 最小必要修正

## 覆蓋檔案

只需覆蓋以下五個檔案：

- `road_detector.py`
- `baccarat_vision.py`
- `screen_pipeline.py`
- `stacking_model.py`
- `predictor.py`

其餘上傳模組沒有改動。

## 根因與修正

### 掃描

- 舊固定格直接依 ROI 全寬高等分，手機狀態列、縮放、外框留白與 UI 偏移會讓 6×15 格中心錯位。
- 舊紅藍分類只看固定 HSV 像素與簡單 dominance；淡色、空心圓、JPEG 壓縮時容易漏判或互污染。
- 舊綠色只看像素總量，格線或長條綠色雜訊可能被當和局。
- 舊反推失敗會回傳欄優先排序，可能讓錯誤序列繼續流入預測。

V10.9 在 ROI 內搜尋有限範圍的有效格線起訖，保留固定 6×15；每格輸出紅／藍／綠像素、dominance、confidence、uncertain 與 tie 集中度。反推要求完整且唯一，失敗時 `sequence=[]`、`quality_ok=false` 並提供 `fallback_reason`，不再把座標排序當時間序。

### 預測

- 舊 Stacking 的 `global_history + road_planning + recent_road` 基礎權重與最低界線過高，路型可能壓過有限牌組。
- 圖片模式使用新牌靴或估計組成時，容易把有限牌組計算誤解為已知真實剩餘牌。
- 掃描品質差或方向不穩時，仍可能留下 B/P/T 動作。

V10.9 對路型總權重與序列權重設硬上限；圖片／估計組成把各群組偏移收縮回標準 8 副牌先驗，再套用更高 edge／穩定度／一致度閘門。`quality_ok=false` 或辨識局數不足一律 `action=O`。線上校準器仍保留原方向，不修改 pending/resolve 流程。

估計組成下 `finite` 槽位仍承擔大部分「先驗錨點」結構，但只保留 25% 的原始 finite 偏移；輸出另提供 `finite_effective_signal_weight`、`road_effective_signal_weight` 與 `baseline_anchor_weight`，避免把結構權重誤讀為真實牌靴信心。預設估計模式的隱含先驗錨點通常超過 70%。

## Debug 疊格線

在環境變數設定：

```env
ROAD_GRID_DEBUG=1
ROAD_GRID_DEBUG_DIR=/tmp/bgs_road_debug
```

每次固定格掃描會把有效 6×15 邊界、每格 B/P/?/T 與品質狀態畫回裁圖，輸出路徑在結果的 `debug_overlay_path`。

正式環境建議完成調校後改回：

```env
ROAD_GRID_DEBUG=0
```

## 流程行為

- `screen_pipeline.analyze_game_screen` 仍先掃路紙；session 已有館別與桌號時跳過 OCR。
- OCR 失敗不阻塞路紙。
- 只有 `quality_ok=true` 且完整反推通過時才建立 `session_patch`。
- 品質失敗時公開 `sequence/raw_outcomes` 為空，原始診斷保留在 `road`、`detected_sequence` 與 `all_grid_cells`。
- 後續莊／閒／和按鈕應只使用既有 session sequence 結算；此補丁不改 UI 文案或既有 pending/resolve 介面。

## 已執行測試

- 固定 6×15：一般轉色、12 局長龍、長龍黏邊、淡色 JPEG 壓縮。
- 和局標記只進 `raw_outcomes/tie_markers`。
- MT 1728×903 完整畫面與寬型多路圖 auto routing。
- 明顯錯裁圖：`quality_ok=false`、不建立 session patch。
- contour 備援：完整反推與斷裂格位阻擋。
- Stacking 隨機 100 組：機率正規化、路型總權重及序列權重不越界。
- predictor：可靠組成、估計組成、壞掃描與估計模式和局訊號阻擋。
- 全部 Python 檔案通過 `py_compile`。

注意：本次未收到實際牌桌截圖樣本，因此影像驗證為合成格線、淡化與 JPEG 壓縮情境；上線前仍應以 MT 實際手機截圖開啟 debug 做一輪閾值確認。
