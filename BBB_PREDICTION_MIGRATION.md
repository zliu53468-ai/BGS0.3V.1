# BBB 預測核心移植

來源：`zliu53468-ai/BBB`，固定提交 `603b8cbcdeea51d89909218e06ff924fdd1a6f0c`。

這次只移植預測邏輯，保留 BGS 的既有使用流程。正式設定與 BBB 相同，使用 `baseline`；前一輪尚未驗證雙目標改善的 `guarded` / `requested_prior` 實驗方案不啟用。

## 實際執行鏈

`BGS 已驗證 B/P/T 歷史 → bbb_final57_runtime.py → bbb_engine/worker.cjs → BBB Direct Physics → stateless Particle500 → Physics EMA → Frozen Core + XGB 輔助篩選 → EV/Volume Guard → B/P/Skip → BGS 原有輸出欄位`

- 六個核心 JavaScript 檔案與兩個模型直接複製自來源提交，沒有把模型數值翻譯成另一套 Python 演算法；`bbb_engine/manifest.json` 記錄逐檔 SHA-256，啟動預測時檢查。
- Worker 只加入沒有頁面的執行環境、模型檔讀取及資料映射。三個原本未公開的純計算函式透過只讀 export 接口提供給原 Python helper API；不改寫運算或決策規則。
- Worker 不接收 LINE token、開通碼、圖片或裝置資料，只接收 B/P/T 和該 scope 的預測狀態。
- BGS 的舊 Python road/core 仍供既有診斷欄位使用，沒有正式方向決策權。正式 Frozen Core 也一併由 BBB 原始 JavaScript 執行，並非把舊 Python 方向當作替代。
- 原本舊 Final57 是 XGB 直接出方向；現在與 BBB 一樣由 Physics + Particle500 主導，Core/XGB 只輔助篩選且不翻轉候選方向。

## 保留項目及唯一輸出邊界修正

`app.py`、`store.py`、`static/`、LINE webhook、開通碼與試用邏輯、裝置/會員驗證、所有 OCR/圖片/截圖/大路辨識檔案、`render.yaml`、`Procfile` 及環境設定皆未修改。71 個原有非預測檔案的內容摘要已記錄，回歸測試會逐一核對。

`runtime_app.py` 的既有預測輸出函式原本把 Skip 轉成莊或閒並強制配置資金；僅在 BBB 明確回報 Skip 時保留觀望、禁止出手並將注碼歸零，避免抹掉移植的預測結果。原有莊/閒訊號的資金配置方式保留。這是預測結果的銜接修正，不修改授權或 LINE 路由。

`requirements.txt` 只新增固定版本 `nodejs-wheel==24.19.0`，讓現有 Python 部署可執行同一份 JavaScript 核心；沒有新增需使用者操作的設定，也不更改部署命令。沒有模型重訓工作。

## 預測狀態隔離

每次預測都建立新的 JavaScript 環境；500 個粒子仍由當前歷史重新推演。BBB 的已結算預測快照供 EMA 與 Volume Guard 使用，保存在獨立 `bbb_prediction_state.sqlite3`，不讀寫 BGS 的授權/會員/session 檔內容。

資料以既有 user/venue/room/shoe scope 的 SHA-256 隔離。資料庫預設放在 `SESSION_DATA_FILE` 所在資料夾，沿用既有持久磁碟；可由 `BGS_BBB_STATE_FILE` 指定測試路徑。沒有 scope 的 helper 呼叫完全不保存狀態。

- 新局只結算確實存在的上一筆預測，不因 OCR 批次匯入而虛構中間預測。
- 同歷史重試不重複結算，也不重複平滑。
- 返回上一局/更正歷史會移除不再符合歷史的快照，避免未來資料洩漏。
- 換會員、房間、牌靴皆隔離；獨立子程序重建後仍可由資料庫恢復已結算狀態。
- 引擎/模型校驗失敗或超時會回報錯誤，由 BGS 原有錯誤流程處理，不回退到舊方向或強行出手。

## 驗證

```text
python -m unittest -v test_bbb_final57_runtime.py test_bbb_prediction_migration.py
```

九個測試通過，涵蓋模型/特徵維度、正式 predict 接口、與獨立 BBB JavaScript 參考環境逐值相等（方向、機率、EV、48D/57D 特徵與 EMA）、1–70 局、長龍/單跳/和局、連續新局、重試、撤回、會員/房間/牌靴隔離、跨子程序狀態保留、觀望注碼為零，以及 71 個保護檔案摘要一致。

GitHub 的預測檢查已接到 `main` 推送；依現有 Render 設定自動部署，未另行更改部署平台或帳戶設定。GitHub 測試通過不能替代真實 LINE/開通碼服務的端對端驗證。
