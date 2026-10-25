# BBB 新版預測搬運｜固定莊閒差 2 個百分點

來源：zliu53468-ai/BBB，main e11b52424d0641aeaa16437a15e744808848dfba。
目標基準：BGS0.3V.1 main dd419b404ec967c916e8307c6feffc00e600e3fa。

## 行為

直接搬運來源的 Frozen Core、Direct Physics、PF500、Macro EMA、57D XGB、決策 EMA、EV/Volume Guard、cardcount_middle_late_v2，以及本機校準計算；沒有重訓或更換來源正式模型。兩個正式模型與目標原檔完全相同。
方向顯示固定使用排除和局後的莊閒機率差 0.02：51%/49%顯示莊、49%/51%顯示閒，50.99%/49.01%觀望。沒有1%/3%選擇介面或請求參數。worker驗證固定設定，缺設定時不默默改用0門檻。
與BBB相同，顯示方向與正式出手資格分開；畫面差距達標不會繞過EV/Volume Guard，正式Skip保持零下注。LINE畫面的機率與方向只在既有runtime接入點轉接，不覆寫儲存的正式決策或金額。
逐局觀測更新沿用來源校準程式，資料按原user/venue/room/shoe scope隔離；換模型版本重新累積預測狀態，並保留重試、返回、重載行為。OCR首次匯入只建立歷史，不虛構過去出手紀錄。

## 不變項

app.py、store.py、static/liff.html、render.yaml、requirements.txt、Procfile，以及原有開通碼、30分鐘試用、月租/永久資格、LINE/OCR/圖片/金額上下限等保護清單全部不修改。runtime_app.py只新增顯示資料副本的轉接函式；原授權/試用/傳輸函式仍在原檔。
引擎scripts與模型逐檔SHA256對照manifest；protected_files_sha256原封保留，由CI對完整倉庫驗證。發布前另以PR修改檔清單確認所有受保護路徑皆無變動。

## 驗證與上線

python -m unittest -v test_bbb_final57_runtime.py test_bbb_prediction_migration.py test_bbb_fixed_gap.py

涵蓋冷啟動/長靴/和局、來源瀏覽器一致性、使用者/房間/牌靴隔離、EMA/返回/重載、51/49邊界與未達門檻、校準重試與回退、畫面副本不改權限或正式投注、受保護檔案雜湊。
通過後合併main，由既有render.yaml的autoDeploy設定發布；沒有更改服務設定或另建服務。實際雲端啟動仍以原部署服務狀態及/health為準，不把測試通過冒充已完成外部發布。

## 回滾

對此次搬運合併提交執行git revert，成套回復engine、manifest、profile與預測接入層。不要回復或清除sessions.json、授權資料、試用期限、既有持久磁碟。回滾後依原流程重新部署。
