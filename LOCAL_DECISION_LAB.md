# BGS 本地決策測試台

這個工具只做本地測試，不修改 `app.py`、LINE、OCR、截圖辨識或正式部署入口。

## 啟動方式（Windows）

1. 下載 / clone 此分支。
2. 第一次先在專案資料夾執行：

   ```bat
   pip install -r requirements.txt
   ```

3. 之後直接雙擊：

   ```text
   run_local_decision_lab.bat
   ```

4. 瀏覽器開啟：

   ```text
   http://127.0.0.1:8787
   ```

也可以手動啟動：

```bat
python -m uvicorn local_decision_lab:app --host 127.0.0.1 --port 8787 --reload
```

## 怎麼測

- `莊 B`：把已開出的莊加入歷史，立即計算下一手。
- `閒 P`：把已開出的閒加入歷史，立即計算下一手。
- `和 T`：加入和局；正式 B/P Road 方向仍依現有 predictor 規則處理。
- `復原`：刪除最後一手。
- `清空`：重開一靴。
- `整段回測`：對目前輸入的整條歷史做 walk-forward 式逐手測試。

## 這個頁面特別量測什麼

頁面會顯示：

- 正式下一手莊 / 閒方向與機率。
- 目前是否屬於 `SAME`（預測跟上一手同邊）或 `SWITCH`（反邊）。
- multi_window / pattern_replay / ngram / pattern_survival 四個 V1 元件各自的方向、可靠度與有效權重。
- 有效元件裡面有多少票同時支持 `SAME`。
- 整段回測的 `follow_last_call_rate`：模型有多少比例直接預測跟上一手同邊。
- `run_ge3_follow_last_rate`：當目前已經連 3 手以上時，模型仍繼續追同邊的比例。

## 目前程式碼裡為何容易看起來「看莊打莊、看閒打閒」

正式方向目前由 `road_pattern_core` 擁有。LinUCB、Markov 與 `road_forecaster` 在 production policy 內都只做 diagnostic，正式 B/P 權重為 0。

V1 四個元件中有三個主要元件都把問題轉成「下一手會 SAME 還是 SWITCH」後，再依目前最後一手把 SAME 映射回 B/P：

1. `multi_window`：統計最近 6/10/16/24 手 SAME/SWITCH 比例。
2. `pattern_replay`：找過去相同 SAME/SWITCH 簽名後，下一手是否延續。
3. `ngram`：用 relation n-gram 判斷下一手是否跟 context tail 同邊。
4. `pattern_survival`：在 DRAGON (`current_run >= 3`) 時，規則先驗直接設定 `desired_same = True`。

因此這四個元件並不是四個互相獨立的方向來源；它們高度共用「延續 / 轉折」資訊。當最近資料的 SAME 比例略高時，多個元件會同時往 SAME 聚集，再由 `_same_probability_to_b(last_side, p_same)` 映射成目前最後一手的 B/P。視覺上就會變成：最後一手是莊時常預測莊、最後一手是閒時常預測閒。

此外，`contextual_bandit.py` 雖然有 anti-chase 衰減，但 production `road_only_policy()` 最後把 LinUCB 正式方向權重設為 0，所以那套 anti-chase 不會修正 `road_pattern_core` 的正式方向。

## 建議下一步

先用真實歷史在本頁跑 10~30 靴，收集：

- follow_last_call_rate
- run_ge3_follow_last_rate
- 各元件 SAME vote 比例
- 命中率與 SAME/SWITCH 分組命中率

如果確認 SAME 過度集中，修正點應該放在 `road_pattern_v1_core.py` 的「相關訊號去重 / continuation cap」，而不是再調 LinUCB alpha。
