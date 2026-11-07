# 製造 Excel 的指定數字 blind baseline

本次依使用者提供的「刮刮卡製作 1.xlsx」原公式，由 **LibreOffice Calc 實際重新計算**，不是 Python synthetic board generator。Calc 與 Microsoft Excel 的 RAND 實作可能不同；本測試不宣稱完全重現 Microsoft Excel 的亂數序列，Excel-native 驗證仍待另外進行。

## 製造公式與 logical board

來源：`隨機產生號碼`。逐格核對所有 320 個來源公式（包含 XLSX shared formula 展開）：

- `C1:C160 = RAND()`。
- `D_i = RANK(C_i,$C$1:$C$160)`，省略第三參數，採降冪排名。
- 原檔 SHA-256：`59f284818497545d53e4b08ba915ad45d27e2ea4164df4b8c6aaf434949a5ebe`。
- 原印刷頁將 D1:D160 排成兩個 8×10 區塊。本次依使用者指定，以 D1→D160 row-major reshape 成 **10×16 logical board**，不冒稱與印刷位置相同。
- 每盤驗證 1～160 各一次、無重複缺號；另以同次 C 欄快取值交叉檢查 D 確實是其 rank。遇 RAND ties、重複排列或錯誤一律失敗，不補號、不修理答案。

這是使用者實際製造檔的 **RAND/RANK 隨機 permutation 機制**。若 RAND draws 可視為獨立連續均勻亂數，在觀察任意 80 格後，其餘 80 個位置對 target 仍然可交換，理論上沒有穩定位置預測優勢。公式本身不等於對 PRNG 獨立性的數學認證。

## Calc 執行與凍結

版本：LibreOfficeDev **26.8.0.0.alpha0**，build `2c87e51eeaa2b413ff4ae097b2705eea1995d8e5`。

載入完整原檔，不改公式、不輸入製造 seed。透過官方 LibreOfficeKit C API 對每盤 dispatch `.uno:CalculateHard`；**等待 `LOK_CALLBACK_UNO_COMMAND_RESULT` 明確回報 success=true** 才匯出 XLSX。匯出 calcPr 未指定 calcMode（OOXML 預設 automatic），每盤仍額外強制 hard recalculation。初期非同步呼叫的重複快照已被拒絕，未列入任何正式 trials。

匯出後立刻從 XLSX cached XML 複製 D1:D160 到唯讀 `values.json`。XLSX 是可稽核來源快照，scorer 僅讀凍結 XML，**不在 scoring 時重算公式**。每盤保存 snapshot SHA、重算成功事件及凍結值。完整 XLSX／C 值／答案都只保留在外部私有產物目錄。

## 預先鎖定的評估協定

- work baseline：`833a0920b854c430259248cb9fb973255f24d3f8`。位置模型、原 predictor、consensus、assignment 原始碼不變，另對原始版本逐 byte / SHA 核對。
- 40 張獨立 Calc 製造盤用作固定訓練；5 張只作工程等價性驗證，不算 test 命中率。沒有把實拍 OCR regression 圖補成完整表。
- 首輪：新 1,000 張製造盤，每盤 10 個 masks，共 10,000 trials。
- Fresh holdout：首輪所有預測、seal、scoring 完成後，程式及設定保持鎖定，**另起 Calc 程序／profile 新產生 1,000 張**，每盤 10 masks，再跑 10,000 trials。最終推論只看這組。
- 全部固定 target=73，10×16，每次公開恰好 80 格、hidden 恰好 80 格，73 永遠 hidden。
- Mask 使用每 trial 新的 OS entropy，獨立於製造 RNG，不共用小型固定 seed 模板。Python 僅抽取公開格，從不產生或代替製造排列。重現評估使用已封存的 input，而非重播未公開 RNG state。
- 模型沿用原 fit 與 hyperparameters；各 sanitized predictor daemon 啟動時只對固定訓練盤 fit 一次，trial 子程序重用相同模型，fresh 亦使用同一訓練盤。原模型本身沒有修改。重複 candidate feature 僅作 bit-exact cache；五張工程盤的 12 組完整排名／分數，全部與原 predictor 逐 trial 重 fit 的結果完全一致。
- `evidence=[]`、`policy_records=[]` 固定，不用 test 答案調權重、啟用 discussion、挑模型或改 threshold。Discussion 若未啟用有效 peer influence，必須如實呈現零增益。
- 訓練盤、工程盤、首輪及 fresh 的完整 permutation 皆查重；不讓同板進入 train/test。

## 答案隔離與 seal 順序

產盤／preparer 必須知道完整表才能遮蔽，但 predictor 由獨立 exec 程序啟動，不從持有完整答案的 coordinator fork。其父程序只載入固定 train corpus；每個 trial 再 fork 一個預測子程序。

Predictor 唯一 request：`rows / cols / cells / target`。不含 target 座標、board ID、檔名、image hash、完整 D 欄、hidden values、Calc RNG state、製造或 mask seed。未知格全為 null；可知的 missing number 集合只由公開值的補集推得，符合原模型契約。

每個 trial 嚴格依序：

1. Predictor 產生 12 組各 80 格的完整 ranking。
2. Coordinator 將 `prediction.json` exclusive-create、fsync、設唯讀並保存 SHA seal。
3. Seal 寫入完成後才 ACK，預測子程序才退出；daemon `waitpid` 確認 exitcode=0，另存退出證明。
4. Scorer 驗證 seal 與退出證明，才讀完整 Excel snapshot；核對 SHA、D values，另存 answer / score，不修改 prediction。

Predictor audit guard 限制可讀檔到匿名 train bundle、程式/依賴及 ephemeral cache，禁止其他檔案、網路連線與子程序。這是可信模型程式的防禦層，不宣稱是針對惡意 native extension 的 OS sandbox。

## 指標解讀

每個模型分別記 Top1/3/5/10、命中次數、MRR、mean/median rank、random baseline、lift 與 CI。Random 每 trial 依 unknown_count 計算；本次每 trial 經驗證皆為 80，故理論基準為 1.25%／3.75%／6.25%／12.5%，mean rank=40.5。

95% CI 以製造 board 為 group 作 2,000 次 bootstrap，保留同盤 10 masks 的相依性；另對 12 methods 的 Top1 提供 Bonferroni familywise CI，避免只挑偶然最好的模型。10,000 trials 不當作 10,000 張獨立盤。

Assignment 的 Top1 使用 Hungarian 指定位置；它本身不是完整 ranking 模型，因此其餘順位明確定義為「把 assigned cell 移至第一，其餘維持 final consensus 次序」。同時另報改對／改錯／net gain。這些 test 結果不自動改寫正式 runtime policy。

若 Top1 達 10% 等異常高分，先標 SUSPECT LEAKAGE，稽核 RNG/state、request、train/test、seal 時序，不直接宣稱模型有效。

<!-- Measured results are appended only after both locked phases finish. -->

## 重跑方式

需要可載入的 LibreOffice `program/libsofficeapp.so` 與其原生依賴。以下先產生真實 Calc 重算的訓練／工程盤，再驗證 cache 等價性；接著產生首輪，執行器會在首輪完成後自行新產生 fresh。

```bash
# 設為已安裝的 LibreOffice program 目錄
CALC_PROGRAM=/path/to/libreoffice/program
SOURCE_XLSX=/private/manufacturing.xlsx
RUN_ROOT=/private/new-manufacturing-run

LD_LIBRARY_PATH="$CALC_PROGRAM" SAL_USE_VCLPLUGIN=svp \
  .venv/bin/python -m taiwan_lottery.ml.layout.blind.calc \
  --source "$SOURCE_XLSX" --program "$CALC_PROGRAM" \
  --output "$RUN_ROOT/setup" --count 45

.venv/bin/python tools/verify_manufacturing_cache.py \
  --setup "$RUN_ROOT/setup" --output "$RUN_ROOT/equivalence"

LD_LIBRARY_PATH="$CALC_PROGRAM" SAL_USE_VCLPLUGIN=svp \
  .venv/bin/python -m taiwan_lottery.ml.layout.blind.calc \
  --source "$SOURCE_XLSX" --program "$CALC_PROGRAM" \
  --output "$RUN_ROOT/development-boards" --count 1000

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m taiwan_lottery.ml.layout.blind.manufacturing \
  --setup "$RUN_ROOT/setup" --development "$RUN_ROOT/development-boards" \
  --source "$SOURCE_XLSX" --program "$CALC_PROGRAM" \
  --cache-proof "$RUN_ROOT/equivalence/proof.json" --output "$RUN_ROOT/result"
```

所有 output 須為新目錄；不覆寫先前封存結果。要精確重播本次預測，使用本次已凍結的資料與公開 inputs；重新執行 Calc 會得到新亂數序列，不保證同一組百分比。

每 trial 的舊 `score.json.seed` 欄位僅存 mask 序號 0～9，**不是 Calc seed，也不是 mask 的 RNG seed**。實際 mask 以 `blind_input.json` 封存，不會把 RNG state 交給模型。

完整私有產物位置：`/workspace/lottery-environment/manufacturing-blind/`。Git 只保存程式、協定、來源公式及 aggregate reports，不保存使用者原始 XLSX、訓練完整盤、hidden answers 或逐 trial 詳細資料。

## 實測結果（2026-11-07）

首輪與 fresh 各 **1,000 張不同製造盤、10,000 trials**；合計 20,000 trials。含訓練／工程共 2,045 張正式協定盤。以下最終成績只用 fresh，不把首輪與 fresh 混成更多獨立樣本。

| Engine | Top1 命中/10000 | Top1 | 95% CI | Top3 | Top5 | Top10 | MRR | Mean rank | Median rank | Top1 lift |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| assignment | 76 | 0.76% | 0.46%–1.14% | 3.30% | 6.01% | 12.26% | 0.05821 | 40.13 | 40.0 | 0.608× |
| association | 102 | 1.02% | 0.65%–1.41% | 3.92% | 7.08% | 13.56% | 0.06281 | 40.25 | 40.5 | 0.816× |
| catboost | 129 | 1.29% | 1.03%–1.56% | 4.37% | 6.77% | 13.47% | 0.06533 | 40.12 | 40.0 | 1.032× |
| discussion | 115 | 1.15% | 0.91%–1.41% | 3.75% | 6.04% | 12.34% | 0.06175 | 40.09 | 40.0 | 0.920× |
| final_consensus | 115 | 1.15% | 0.91%–1.41% | 3.75% | 6.04% | 12.34% | 0.06175 | 40.09 | 40.0 | 0.920× |
| lightgbm | 118 | 1.18% | 0.95%–1.45% | 3.88% | 6.53% | 12.84% | 0.06258 | 40.58 | 41.0 | 0.944× |
| neural | 109 | 1.09% | 0.86%–1.34% | 3.66% | 6.30% | 12.50% | 0.06066 | 40.24 | 40.0 | 0.872× |
| position | 94 | 0.94% | 0.53%–1.39% | 4.03% | 6.76% | 13.39% | 0.06161 | 40.32 | 40.5 | 0.752× |
| round1 | 115 | 1.15% | 0.91%–1.41% | 3.75% | 6.04% | 12.34% | 0.06175 | 40.09 | 40.0 | 0.920× |
| similarity_knn | 148 | 1.48% | 1.05%–1.91% | 4.37% | 7.33% | 14.38% | 0.06749 | 39.68 | 39.0 | 1.184× |
| statistical | 132 | 1.32% | 0.75%–1.96% | 4.38% | 7.12% | 13.81% | 0.06544 | 40.00 | 39.0 | 1.056× |
| transformer | 107 | 1.07% | 0.82%–1.36% | 3.51% | 5.89% | 12.36% | 0.05998 | 40.17 | 40.0 | 0.856× |
| Random 理論期望 | 125（期望值） | 1.25% | — | 3.75% | 6.25% | 12.50% | 0.06207 | 40.50 | 40.5 | 1.000× |

每個 Top-K 的實際命中次數、MRR / mean / median rank 的 95% CI、lift CI 與 familywise Top1 CI，均在 [fresh_report.json](layout-manufacturing-blind/fresh_report.json)。Random 列是條件理論期望，不是另跑一批 random trials 的觀測命中數。

首輪完整成績：[development_report.json](layout-manufacturing-blind/development_report.json)。不得用首輪最好的方法取代原模型，再宣稱 fresh 證明調參有效。

### Discussion 與 assignment

| Phase | Discussion Top1 gain | Discussion MRR gain | Assignment 改對 | Assignment 改錯 | Net |
|---|---:|---:|---:|---:|---:|
| development | +0.000pp | +0.00000 | 132 | 120 | +12 |
| fresh | +0.000pp | +0.00000 | 71 | 110 | -39 |

Discussion 在本設定未有已驗證 peer influence，因此沒有穩定改善證據，不能因名稱叫 discussion 就宣稱有效。Assignment 的 paired board-bootstrap gain CI 也完整列在 JSON；不因小幅點估計就強制套用正式 runtime。

### 回答本次問題

**依原製造 XLSX 的 RAND/RANK 公式、由 Calc 重算，在看見 50% 數字後，現有位置模型沒有顯示經預先指定的多重比較校正後、統計上高於隨機的 Top1 命中率。NO EVIDENCE ABOVE RANDOM。**

各模型略高或略低於 1.25% 的點估計，不足以證明存在可重現訊號。這與獨立 random permutation 的可交換性預期一致，但有限試驗不是對 Calc PRNG 獨立性的數學證明。

探索性的次要指標中，有 2 個未校正 95% CI 下界高於其 random 基準；完整清單在 JSON。這些比較未作所有 endpoints 的 familywise 校正，不選擇性當作模型能力證據。

### Leakage audit 與限制

- Fresh 與先前 1110 個不同技術／訓練／工程／首輪 snapshot 排列完全不重疊；訓練／test 重疊為零。
- Predictor request 不含 Calc RNG state、製造 seed、mask seed、完整 D 欄、target 真實座標、source ID 或檔名。
- 所有 20,000 個 trial 均保存 seal 與實際子程序退出證明；scorer 在兩者之後讀取凍結 Excel 答案，沒有回寫 prediction。
- 未因 test 調整模型、權重、門檻或 runtime 組合。Fresh 產生於鎖定且首輪完成之後，只跑一輪。
- 沒有達到 Top1 10% 的大幅異常；沒有發現測試涵蓋範圍內的答案洩漏。Python audit guard 不是對任意惡意 native code 的安全證明。
- 結論限於本次固定 40 張訓練、既有模型與 Calc 公式重算；未測 Microsoft Excel-native、真實印刷製程偏差、其他 target 或額外 side-channel。

[協定鎖定](layout-manufacturing-blind/protocol_lock.json) · [Calc 設定](layout-manufacturing-blind/calc_manifest.json) · [Leakage audit](layout-manufacturing-blind/leakage_audit.json)

補充次要指標：fresh 的 CatBoost Top3=4.37%（95% CI 3.79%～4.92%；random 3.75%），Similarity Top10=14.38%（12.55%～16.28%；random 12.5%）有未校正偏高；首輪對應分別為 3.80%（3.28%～4.35%）及 11.93%（10.12%～13.72%），未呈現相同的顯著優勢。不能只挑這兩個比較宣稱有可重現能力。

探索性 target 位置均勻性檢查：首輪 χ² p≈0.933，fresh p≈0.0368。Fresh 有單批抽樣偏差，這不是對 Calc RNG 有偏或獨立性的定論；該檢查未校正所有探索性比較，沒有據此調模型、排除盤或重跑 fresh。

Fresh assignment 改對 71 次、改錯 110 次，net=-39；paired Top1 gain=-0.39pp，board-bootstrap 95% CI 約 -0.77pp～+0.02pp。此 baseline 不支持強制 assignment。Discussion 各 Top-K / MRR gain 都為零，效果未建立。

`score` 是原模型的相對排名分數，未宣稱是經校準命中機率；真正命中率只由封存後的 scorer 計算。

## 最後驗證

- **Full pytest：54 passed，13 warnings**。包含原始 XLSX 公式驗證、mask/feature cache/答案隔離測試，以及對全部 20,000 份真實 Calc trial 的完整排名、SHA seal、退出證明、答案及檔案時序稽核。
- 既有 layout / vision、FastAPI lifespan/API、PostgreSQL rollback integration、Bingo / Lottery smoke tests 通過；未發現這些測試涵蓋範圍內的原功能 regression。
- `pip check`、JS syntax、git diff check 通過。
- 沒有改動 OCR 主流程、現有位置模型原始碼、hyperparameters 或 runtime ensemble policy；沒有 merge master。
- Hosted CI 尚未確認；本環境 GitHub API 存取受限，不能把本地測試成功冒稱 GitHub CI 成功。
