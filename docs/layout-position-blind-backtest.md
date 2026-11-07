# 指定數字位置預測：blind baseline

後續使用者另提供製造 XLSX，依原 RAND/RANK 公式的 Calc 重算實測見 [製造機制報告](layout-manufacturing-blind-backtest.md)。本頁保留當時完整歷史 corpus 為空的紀錄；兩者資料來源不同。

2026-11-07。依本輪追加指示，固定 **遮蔽 50% / 揭露 50%**，target 永遠 hidden；本輪沒有執行其他揭露比例。基準模型為 work HEAD `28beeb292ecb11dc1dda0d7529f6aae64e80c956`，模型、predictor、consensus、assignment、registry 與原有 backtesting 原始碼經 SHA-256 / git 原文比對，未修改。OCR 主流程亦未修改。

## 實際結果：資料不足，無法評分

實際查詢本地 PostgreSQL `layout_history`：**0 張完整歷史 board**。固定 split 的 train / validation / test 均為 0；primary prediction trials = **0**，包含 controls / walk-forward 的 trials 亦為 0。這是資料可用性失敗，不是模型準確率為零，也不是完成了真實模型能力驗證。

三張 OCR regression 照片仍有 UNKNOWN / UNCERTAIN，不能補猜成完整 1～N permutation，更不能放入本次 train。沒有以 synthetic boards 冒充歷史樣本；unit tests 的人工小表只驗證程式路徑，不產生公開績效分數。

| Model | Top1 | Top3 | Top5 | Top10 | MRR |
|---|---:|---:|---:|---:|---:|
| Random | N/A | N/A | N/A | N/A | N/A |
| Statistical | N/A | N/A | N/A | N/A | N/A |
| Position | N/A | N/A | N/A | N/A | N/A |
| Association | N/A | N/A | N/A | N/A | N/A |
| LightGBM | N/A | N/A | N/A | N/A | N/A |
| CatBoost | N/A | N/A | N/A | N/A | N/A |
| Similarity | N/A | N/A | N/A | N/A | N/A |
| Neural | N/A | N/A | N/A | N/A | N/A |
| Transformer | N/A | N/A | N/A | N/A | N/A |
| Round1 | N/A | N/A | N/A | N/A | N/A |
| Discussion / final consensus | N/A | N/A | N/A | N/A | N/A |
| Assignment | N/A | N/A | N/A | N/A | N/A |

- 指定 73：0 張 eligible test boards、0 trials；Top1/3/5/10、MRR 均 N/A。
- 各 target 1～100：未評分；不是各數字命中率為零。
- 50% reveal 的 model/random/lift：N/A。實際樣本的 random baseline 不能在沒有樣本時宣稱為 2%。理論上若每次恰有 50 unknown，Top1/3/5/10 = 2%/6%/10%/20%。程式逐次以 unknown_count 計算，不寫死。
- History label shuffle / visible board shuffle：無真實資料可執行，效果 N/A。
- Discussion gain：N/A，**discussion effectiveness not established**。
- Assignment corrected_wrong_to_right / changed_right_to_wrong / net_gain：沒有觀測，不作有效性推論；未提升 assignment 為強制 runtime 策略。
- Walk-forward：沒有具有已驗證時間順序的完整 boards，not_ready。DB 匯入時間不能冒充拍攝/實體時間。
- 結論：**NO EVIDENCE ABOVE RANDOM**。尚不能判斷任何模型有可重現訊號。

機器產物：[blind_baseline.json](layout-position-blind-backtest/blind_baseline.json)、[target_73_report.json](layout-position-blind-backtest/target_73_report.json)。所有 N/A 均來自空資料，沒有估算成實測值。

## 隔離協定

新增離線 evaluator，不替換原 runtime endpoint：

1. preparer 讀取完整資料，驗證實體 board ID 與完整 permutation 不重複；按 board 分 train/validation/test（60/20/20，至少 5 張才能切）。不足則回報 not_ready。不同尺寸各自只使用同尺寸 training boards。
2. 每張 test board × target 1～N × seed 1～20 建立 deterministic mask；恰揭露 floor(N×0.5) 格，target 強制 hidden。未知格皆為 null。
3. Predictor 子程序只收到 `rows / cols / cells / target` 與匿名 train-only bundle。request 不含完整表、真實位置、source ID、檔名、image hash、answer path。任何額外 request 欄位都拒絕。
4. 子程序工作目錄只有 sanitized input 與 train bundle；Python audit guard 限制讀取到這些檔案、程式/依賴與 ephemeral cache，拒絕其他檔案、socket 與子程序。Torch 暫存限制在可丟棄目錄；不得改寫輸入。這是可信模型程式的防禦層，**不是針對惡意原生 extension 的 OS sandbox**。
5. 子程序退出後才以 exclusive-create 寫入 `prediction.json`、SHA-256 seal 並設唯讀；此後才建立 answer file / 啟動 scorer。Scorer 先驗證 prediction/input hash、每個 unknown 恰好一次及有限分數，再讀答案，輸出另外的 `score.json`。拒絕覆寫 prediction/score；scoring 結束再驗 hash。
6. 每個 engine 保存完整 ranking 與 ready/not_ready；Round1、discussion、final consensus 分開記錄。Hungarian 本身只產生 assigned cell，因此其完整順位明確定義為「assigned cell 第一，其餘維持原 final ranking」，不冒充另一組獨立模型分數。
7. reliability / discussion policy 僅取 train-prefix out-of-sample evidence；validation/test 結果不回灌。本輪沒有 tuning、weighting、threshold selection 或模型優化；validation 保留未用。
8. 若提供可信 `observed_at`（含 timezone），另外按實際時刻作 walk-forward：每張只用更早的 boards，與固定 holdout 分開報告，不合併兩套 trial 數。

每個 trial 保存：`blind_input.json` → `prediction.json` + seal → `answer.json` → `score.json`。含答案及實體 ID 的詳細 trial 產物放外部私有目錄，不提交 Git。唯讀與 SHA 封存防止意外回寫；擁有檔案系統控制權的攻擊者不在此完整性保證內。

## 指標與 controls

- Top1/3/5/10、MRR、mean/median rank；random 逐次以 unknown 數量計算，然後聚合，lift 與模型使用相同可評分 trials。
- 同時報 distinct physical boards、mask seeds、prediction trials，不能把同板 20 masks 當 20 張板。
- 95% percentile bootstrap 以 physical board 分組抽樣，保留板內所有 target/mask；paired gain 與 random 比較。CI 屬探索性多模型比較，沒有多重比較校正；不可作強泛化證據。
- 少於 2 張 test boards：樣本不足。至少 20 張、Top1 paired gain CI 下界 > 0 才標 EVIDENCE ABOVE RANDOM；小樣本正 gain 只標 POSSIBLY ABOVE RANDOM — INSUFFICIENT EVIDENCE。下界未過零不宣稱模型有效；負 gain 且上界 < 0 標 BELOW RANDOM。
- History label shuffle：固定 seed 逐張打亂 train history 的數字/位置並重建 train-prefix evidence，不碰 test 答案。Visible board shuffle：保留 hidden mask/target，打亂可見值與位置。
- 對兩個 controls 各自報完整模型成績與 baseline-control 差。Shuffle 沒退化可能是沒有學到訊號，也可能是評估問題，不能直接當作「已證明 leakage」。
- Discussion 報 Top1/3/5/MRR paired gain 與 board-group CI；未穩定改善則保持 not established。
- Assignment 分別記改對/改錯/net gain。Test 結果不自動改寫正式 policy；net gain ≤ 0 不建議強制套用。

## 執行方式與所需資料

```bash
# 實際 confirmed corpus；固定 50%，所有 target，20 seeds
.venv/bin/python -m taiwan_lottery.ml.layout.blind.run \
  --from-db --output /workspace/lottery-environment/position-blind-new-run

# 完整人工作證 corpus，同樣所有 target（不指定 --target）
.venv/bin/python -m taiwan_lottery.ml.layout.blind.run \
  --corpus /private/confirmed_boards.json --output /private/blind-run

# 單獨測指定數字；不適用 N < 73 的 board
.venv/bin/python -m taiwan_lottery.ml.layout.blind.run \
  --corpus /private/confirmed_boards.json --target 73 --output /private/target-73-run
```

每次 output 必須是新目錄，不會覆寫已封存結果。訓練每次沿用原 predictor 實作；大量 boards × targets × masks × controls 成本可能很高，沒有用抽少數數字偽裝全量結果。

Corpus 為 JSON array，每張必須含 `physical_board_id`、`rows`、`cols`、完整 `cells`（1～N 各一次）、`provenance: "real_confirmed"`。同一實體板不同拍攝須使用同一 physical_board_id，系統將拒絕重複。提供者仍須保證跨來源 alias 沒有隱藏同板；無法僅靠排列檢查證明來源獨立。可選 `observed_at` 必須是有根據的實際時間，含 timezone；沒有時間則不聲稱 walk-forward。DB 的 source 當作實體 ID，排除已知 integration/synthetic/test/demo 前綴，並不把 created_at 當觀測時間。

需要使用者提供完整真實 boards 才能完成能力評估。程式與 isolation tests 完成不等於 real baseline 已完成。

## 本輪驗證

Full pytest：**46 passed，13 warnings**，包含 9 個 blind evaluator contract/leakage tests、八種模型隔離子程序執行測試、既有 layout / vision / PostgreSQL 整合與 Bingo / Lottery smoke tests。JS syntax check 與 git diff check 通過。這些測試通過僅證明流程行為，不是實際命中率證據。

Hosted CI 結果尚未確認；GitHub API 在此環境先前回傳 Forbidden。Workflow 使用完整 git history，才能驗證 pinned baseline 的模型原始碼。未 merge master。
