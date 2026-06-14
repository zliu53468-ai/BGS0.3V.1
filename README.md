BGS 真實牌靴模擬 + 複合資料庫補丁

覆蓋檔案：
1. generate_databases.py
2. point_db.py
3. pattern_db.py
4. combo_db.py
5. predictor.py
6. config.py
7. message_builder.py

沒有修改 server.py。
你的 server.py 已經會把 temp_rounds 傳進 predict()，所以不用動。

部署步驟：
1. 把以上檔案覆蓋到專案根目錄。
2. 先在本機或 Render Shell 執行：python generate_databases.py
3. 確認產生：
   - data/point_db_3m.json
   - data/result_pattern_db_3m.json
   - data/combo_db_3m.json
4. 上傳/提交到 GitHub 後重新部署 Render。
5. 打開 /health 確認 point_db_samples / pattern_db_samples 是否正常。

建議環境變數：
USE_POINT_DB=true
USE_PATTERN_DB=true
USE_COMBO_DB=true
USE_MONTE_CARLO=true

POINT_WEIGHT=0.55
COMBO_WEIGHT=0.30
PATTERN_WEIGHT=0.03
SIM_WEIGHT=0.12
COMBO_MIN_SAMPLE=80

MIN_GAP_FOR_ENTRY=0.060
STRONG_GAP_FOR_ENTRY=0.090

MC_SIMULATIONS=600
MC_MIN_SIMULATIONS=100
MC_MAX_SIMULATIONS=900
MC_SEED=42
MC_MAX_NOISE=0.018
MC_BLOCK_LOW_GAP=true
MC_MIN_GAP_FOR_ENTRY=0.055
MC_DIRECTION_MISMATCH_BLOCK=true

TIE_AI_MAX_WEIGHT=0.01
TIE_SHRINK=0.18
TIE_MIN_GAP_FOR_ENTRY=0.12

AI_NOISE_SCALE=0.008
AI_HISTORY_WINDOW=8
AI_TREND_STRENGTH=0.010
AI_DIFF_MOMENTUM_STRENGTH=0.009
AI_REVERSAL_STRENGTH=0.016
AI_HISTORY_MAX_ADJUST=0.018

注意：
- generate_databases.py 跑 300 萬局會花時間，Render 免費機可能較慢。
- 如果要先快速測試，可以先設 SIM_ROUNDS=300000 跑小資料庫確認程式可運作。
- 正式使用再改回 SIM_ROUNDS=3000000。
