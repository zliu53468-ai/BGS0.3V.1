# 百家樂 Predictor 回測套件

## 檔案

- `predictor.py`：已依照回測安全邏輯調整的主預測程式。
- `baccarat_simulator.py`：8 副牌、不放回、正確補牌規則的模擬器。
- `backtest_runner.py`：逐手餵資料給 `predict()` 的黑盒回測器。

## 執行方式

```bash
python backtest_runner.py --shoes 2000 --seed 42
```

輸出 CSV：

```bash
python backtest_runner.py --shoes 2000 --seed 42 --csv results.csv
```

測 ML 版本：

```bash
BACKTEST_MODE=ml BACKTEST_ML_WEIGHT=0.08 python backtest_runner.py --shoes 500 --seed 42
```

## 重要判讀

- 在純模擬百家樂牌局上，如果命中率顯著高得離譜，通常不是代表模型神準，而是要先檢查是否有資訊洩漏、cache 污染或模擬器規則錯誤。
- 真正要判斷實戰價值，要拿真實歷史靴資料回測。
- 看命中率以外，也要看 `平均每注EV`、`最大連錯`、`進場率`。
