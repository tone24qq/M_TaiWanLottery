# 三張真實照片 regression 報告

測試日期：2026-11-07。Baseline：`a995e64237d5809694ede8060f2b658769c07df4`，未修改程式即執行。ZIP 三張圖片均已解碼及人工檢視：10×10、6×10、10×10。人工標註 186 個明確 OPENED、70 個 UNOPENED、4 個 UNCERTAIN；數字唯一、範圍及格數已驗證。照片、逐格標註與 crops 留在外部私有目錄，沒有提交 Git。

## Runtime before / after

正式 runtime 保持 Tesseract + EasyOCR。表內 exact 的分母包括 abstain；不是只算已回答格。

| 圖 | Baseline | Final | Top3 | CER | Coverage | Abstain | 高分錯誤 | State accuracy | 約束前→後正確格 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0/96 | 51/96 | 66.7% | 0.424 | 66.7% | 32 | 2 | 100.0% | 53→51 |
| 2 | 2/31 | 28/31 | 90.3% | 0.071 | 96.8% | 1 | 0 | 100.0% | 26→28 |
| 3 | 0/59 | 51/59 | 93.2% | 0.125 | 91.5% | 5 | 1 | 96.0% | 48→51 |

全體 runtime：130/186（69.9%）；148 格回答、38 格 abstain、18 格答錯。約束全體 127→130，但第一張 53→51 退步 2 格，尚未完全解決。70 個 UNOPENED 全部不填數字，false-number rate=0。基線第三張有 1/37 未開格誤填。4 個人眼 UNCERTAIN 被分類為 OPENED，state classifier 仍有不足；輸出保留人工確認。

尚有 3 個 confidence≥0.8 的錯誤，均有多位數截斷的複核警示；警示不等於正確答案。分數未經獨立資料校準，不能解釋成正確機率。其餘無法可靠辨識格保留空白/人工確認，沒有用 1～N 補齊。

## 實際推論模型

以下每個 engine 實際處理三張照片，包含不同 crop；ms/cell 是多視角合計的 CPU 時間。Top3 來自保留的候選，不是僅 Top1。

| Engine | Exact | Top3 | CER | ms/cell | 原始 engine 高分錯誤 | Policy |
|---|---:|---:|---:|---:|---:|---|
| crnn | 72.0% | 89.2% | 0.185 | 314.1 | 52 | benchmark_only |
| digit_ctc | — | — | — | — | — | not_ready |
| digit_segmented | — | — | — | — | — | not_ready |
| easyocr | 61.8% | 75.8% | 0.307 | 223.1 | 37 | runtime |
| paddle_en_pp-ocrv5_mobile_rec | — | — | — | — | — | not_ready |
| paddle_pp-ocrv6_medium_rec | — | — | — | — | — | not_ready |
| paddle_pp-ocrv6_small_rec | — | — | — | — | — | not_ready |
| parseq_tiny | 90.9% | 96.8% | 0.060 | 377.7 | 13 | benchmark_only |
| tesseract | 62.4% | 66.7% | 0.284 | 154.9 | 7 | runtime |
| trocr_base | — | — | — | — | — | not_ready |
| trocr_small | — | — | — | — | — | not_ready |

所有模型分開記錄 package、weights、SHA-256、license、實際 inference 與 ready。PARSeq tiny / CRNN 為官方下載且已驗證的 benchmark_only；未因這些 test 結果升級正式投票。Paddle 官方 API 及 HF / ModelScope / AIStudio / BOS 來源仍遇環境 proxy 403；TrOCR 官方 HF 來源亦 403、無完整合法 cache，維持 not_ready。Digit CTC / segmented 無已驗證實拍訓練 artifact。詳見 THIRD_PARTY_MODELS.md。

| 實拍子集 | 樣本 | Tesseract 正確 | EasyOCR 正確 | PARSeq 正確 | CRNN 正確 |
|---|---:|---:|---:|---:|---:|
| 1-digit | 21 | 16 | 17 | 21 | 20 |
| 2-digit | 164 | 99 | 97 | 147 | 113 |
| glare | 5 | 3 | 5 | 5 | 5 |
| scratch | 186 | 116 | 115 | 169 | 134 |

反光明確標註只有 5 個已知格；scratch 是廣泛場景標記，不能據此聲稱能可靠處理所有反光/刮痕。另有一個三位數格納入總體指標。

## 組合比較（僅描述 regression，不作選模）

- Best single: `parseq_tiny`, 169/186 (90.9%)。
- Best 2-model: `parseq_tiny+tesseract`, 166/186 (89.2%)。
- Best 3-model: `easyocr+parseq_tiny+tesseract`, 163/186 (87.6%)。
- Best constrained method: `parseq_tiny`, 175/186 (94.1%)。
- Best multi-engine + constraint: `crnn+parseq_tiny`, 169/186 (90.9%)。

最佳約束方法是單一 PARSeq，不應冒稱多模型 ensemble。多模型並未必勝過最佳單模型。沒有獨立 validation 結果，故沒有宣稱正式主力選模已完成，runtime 仍沿用原組合。反覆修正幾何/通用視覺規則會接觸 regression 誤差，這三圖成績不能作未見資料的泛化估計；需要新的保留集確認。

## 修改與驗證

以重複格/圓形的格點、等間距、形態學及 Hough 線驗證幾何，改善 sticky note 誤抓及透視。多視角數字 ROI 避開格線；中央遮蓋圓與局部背景對比判斷狀態；UNCERTAIN 仍 OCR。保留 raw_text/leading zero、Top-K、引擎版本及 artifact SHA；全局唯一性只取已有 OCR evidence 並允許 abstain。回歸資料不能進入訓練或啟用門檻。

執行 baseline→分析→8 輪修改/測試→完整最終重跑，兩次最終三圖的 values/states 完全一致。最後複核規則僅從凍結的 raw evidence 重放，沒有改動 OCR 預測或信心。各次輸出保留 grid detection、warp、cell overlay、全部 crops、raw outputs、前後約束、latency 與逐格 errors JSON/CSV。

- Full pytest：37 passed，13 warnings（包含 layout、vision、PostgreSQL rollback integration）。
- FastAPI lifespan/startup、layout OCR API、Bingo / Lottery 原頁面及四遊戲 draw API smoke 通過。
- JS `node --check`、`pip check`、`git diff --check` 通過。
- DB migration current/head 均為 `c102_digit_dataset`，沒有新增 schema migration。
- 三圖端到端 regression 通過本地驗收；70 未開格誤填率為零、重跑穩定。
- 新增 GitHub workflow；hosted CI 尚待實際查驗，私有三圖不在 CI runner，不把 skip 算成實拍通過。

私有完整產物：`/workspace/lottery-environment/real-regression/`，包含 baseline_metrics.json、final_metrics.json、engine_comparison.json、ground_truth_audit.json，以及 baseline/final 每圖 errors.json / errors.csv。沒有依 filename/hash/row/col/數字寫入推論特例。
