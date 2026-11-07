# Taiwan Lottery Predictor

台灣彩券數據收集與 ML 預測系統。自動爬取台灣彩券歷史開獎資料，提供統計分析與機器學習預測功能。

## 支援彩種

- **大樂透** (Lotto 6/49)
- **威力彩** (Super Lotto)
- **今彩539** (Daily Cash)
- **賓果賓果** (Bingo Bingo)

## 功能

- 自動排程爬取歷史開獎號碼
- RESTful API 查詢開獎紀錄與統計資料
- ML 預測模型（Frequency、LSTM、DQN、Ensemble）
- 回測框架評估模型表現
- Web 前端儀表板（Jinja2 模板）

## 技術架構

- **後端**: FastAPI + Uvicorn
- **資料庫**: PostgreSQL (asyncpg)
- **ORM**: SQLAlchemy 2.0 (async)
- **Migration**: Alembic
- **ML**: PyTorch、scikit-learn
- **爬蟲**: TaiwanLotteryCrawler + BeautifulSoup

## 快速開始

### 前置需求

- Python 3.12+
- PostgreSQL（建議使用 Docker）

### 安裝

```bash
# 建立虛擬環境
python -m venv venv
venv\Scripts\activate  # Windows

# 安裝套件
pip install -r requirements.txt
```

### 資料庫設定

```bash
# 啟動 PostgreSQL (Docker)
docker run -d --name taiwan_lottery_pg \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=taiwan_lottery \
  -p 5432:5432 \
  postgres:16-alpine

# 複製環境變數設定
cp .env.example .env

# 執行 migration
alembic upgrade head
```

### 啟動

```bash
python -m uvicorn taiwan_lottery.main:app --reload
```

開啟 http://localhost:8000 查看儀表板，API 文件在 http://localhost:8000/docs。

## API 端點

| 路徑 | 說明 |
|------|------|
| `GET /api/v1/lotto649/` | 大樂透開獎紀錄 |
| `GET /api/v1/super_lotto/` | 威力彩開獎紀錄 |
| `GET /api/v1/daily_cash/` | 今彩539開獎紀錄 |
| `GET /api/v1/bingo/` | 賓果賓果開獎紀錄 |
| `GET /api/v1/stats/{game}` | 統計分析 |
| `POST /api/v1/ml/train` | 訓練模型 |
| `POST /api/v1/ml/predict` | 取得預測 |
| `POST /api/v1/scraper/run` | 手動觸發爬蟲 |

## 專案結構

```
taiwan_lottery/
├── api/v1/endpoints/   # API 路由
├── db/
│   ├── models/         # SQLAlchemy 模型
│   └── crud/           # 資料庫操作
├── schemas/            # Pydantic schemas
├── scraper/
│   └── parsers/        # 各彩種解析器
├── services/           # 業務邏輯
├── ml/
│   ├── models/         # ML 模型 (Frequency, LSTM, DQN, Ensemble)
│   ├── features/       # 特徵工程
│   ├── training/       # 訓練流程
│   └── inference/      # 推論與模型管理
├── templates/          # Jinja2 前端模板
├── static/             # CSS / JS
├── config.py           # 設定
└── main.py             # 應用程式進入點
```

## License

MIT

## 圖片表格與本機 AI 討論（第一版）

入口：導覽列「圖片表格 / AI 討論」（路徑 `/layout`）。沿用本專案 FastAPI、
PostgreSQL、SQLAlchemy、Jinja2、PyTorch、SciPy；沒有外部 LLM、付費 API，
也沒有重用舊彩券模型的權重。開發基底為 `tone24qq/M_TaiWanLottery` 的 master。

### 安裝與使用

```bash
pip install -r requirements.txt
# OCR 另需系統 Tesseract（Debian/Ubuntu: apt-get install tesseract-ocr）
alembic upgrade head
SCRAPER_ENABLED=false OMP_NUM_THREADS=2 python -m uvicorn taiwan_lottery.main:app
```

FastAPI / Starlette 限制在已驗證、支援現有模板呼叫方式的版本範圍，避免升級後舊頁面出現 500。
不需要下載 OCR 模型、GPU 或 API key。圖片最大 8 MiB / 1600 萬像素。
任意矩形尺寸皆使用相同演算法；互動版為控制 CPU/記憶體限制每邊 100、總格數 400、
相同尺寸歷史最多 200 張。超過限制會明確拒絕，不會截掉格子。

1. 輸入 rows、columns。上傳照片並點選**表格本身的四個外角**，可校正透視。
   不選角點時嘗試尋找最大四邊形，可能抓到紙張邊框，應人工核對。
2. OCR 分格辨識後一律要求確認。遮蓋、反光、模糊處留空；重複辨識值清空。
   不能從照片推知的數字不會自動補成真值。照片只在請求記憶體中處理，不存原圖。
3. 校正所有可見數字，空白代表未開格；座標由左上角 **R1C1** 起。
   整張表固定為不重複的 1～rows×cols。
4. 輸入目標，取得各模型完整排名、Top-1/3/5、第一輪 → 第二輪、
   權重、支持/反對、反方風險、相關矩陣及全表 assignment。
5. 只有**完整且人工確認**的排列才能加入歷史；source 及內容 hash 均不可重複。
   請勿把同一張表的不同照片當成不同歷史，也不要混用不同製造來源的排列分布。
6. 執行回測後可讀取同尺寸結果。實際開格後另行記錄結果，不改寫已保存的 prediction。

### 模型 readiness

| 模型 | 最少同尺寸完整表格 | 行為 |
| --- | ---: | --- |
| Statistical | 0 | 交換對稱的均勻隨機基準；**沒有位置預測證據** |
| Frequency / Position | 3 | Laplace 平滑的數字位置頻率 |
| Conditional / Association | 3 | 可見位置/數字相符程度加權歷史 |
| Similarity / KNN | 3 | 依可見格距離取最近 5 張歷史 |
| LightGBM | 20 | 遮罩後候選特徵二分類 |
| CatBoost | 20 | 遮罩後候選特徵二分類 |
| Neural Network | 20 | 小型候選 MLP、masked-target cross entropy |
| Transformer / Attention | 20 | 候選 self-attention、masked-target cross entropy |

20 張只是啟動訓練的工程門檻，**不代表足以可靠預測**。資料不足或套件不可用回傳
`not_ready`，不生成模型答案。新模型在每次請求依確認歷史本機訓練（固定 seed、有限迭代），
registry 為請求範圍，與舊 lottery registry 分開；第一版未提供長時間訓練工作佇列／模型檔快取。

### 結構化推論、共識及唯一分配

- Round 1：所有模型先獨立完成整個「未開數字 × 未開位置」矩陣、完整 target ranking、
  confidence（分數集中程度，不是命中率）、features、reason codes，以及自己的 Hungarian assignment。
- Round 2：只交換第一輪矩陣，不接收真實答案。保存 before/after、changed、reason codes。
  固定的 shadow review 使用自身與其他模型平均矩陣各半，作為可回測的候選政策。
  每個模型至少累積 10 次**先前** walk-forward shadow 修正結果後，才依平均 MRR gain
  設定 0～1 的 sensitivity；被帶偏的模型不啟用。沒證據時維持第一輪，同時保存 shadow 排名。
- Round 3：歷史不足時以相等歷史先驗開始，再依輸出 rank correlation 折減重複證據。
  有足夠每模型 holdout 記錄時，使用 Top-1/3/5、MRR、最近表現、Brier 誤差代理、
  修正穩定度計算可靠度；用 Brier shrinkage 往均勻基準收縮分數。
  這是經驗分數校正，**不是已校準的真實機率**。
  至少 10 次成對紀錄時以同尺寸歷史排序相關性更新折減；不足時用當次矩陣相關性。
- 對融合後的完整 score matrix 執行 `scipy.optimize.linear_sum_assignment`，
  強制每數字與每格都恰好使用一次。保留單一 target ranking 和全表 assignment，
  若兩者 Top-1 不同，UI 會明示，不偷偷把 constraint optimum 當成單格最高分。
- 反方分析為 deterministic：單一支持來源、模型分歧、均勻分數、資料不足、
  joint-assignment 衝突及次佳候選。最多三輪，沒有自然語言聊天。
- 結果完整存入 PostgreSQL JSONB（含 history IDs、seed、rounds、矩陣、全部候選、權重）。
  實際答案另存 outcome，不能覆寫同一次 prediction。下一次預測只使用與當前 corpus IDs
  完全一致的回測紀錄；需在新增歷史後重新回測。線上 outcome 暫不自動加入訓練／權重，
  避免把不完整表格或未驗證結果當作完整歷史。

### API

```text
POST /api/v1/layout/recognize             multipart: file, rows, cols, optional corners JSON
POST /api/v1/layout/history               confirmed complete board + unique source
POST /api/v1/layout/predict               board + target
POST /api/v1/layout/discuss               同 predict，執行三輪
GET  /api/v1/layout/models?rows=6&cols=10
POST /api/v1/layout/evaluation            rows, cols, seed, revealed_fraction, max_folds
GET  /api/v1/layout/evaluation?rows=6&cols=10
GET  /api/v1/layout/runs/{id}
POST /api/v1/layout/runs/{id}/resolve      actual row, col（只能記錄一次）
```

```json
{"rows":2,"cols":3,"cells":[1,null,null,4,null,null],"target":3}
```

`cells` 為逐列展開；API 不接受 test truth 作為 predict 的輸入。target 已揭露時直接回報座標。
完整歷史還需 `source`，並且不能含 null。

### 回測與已執行驗證

walk-forward 按歷史入庫順序逐張 holdout：整張測試表不進入訓練，只揭露隨機部分格子，
再從隱藏位置**隨機選目標**。預測全部凍結後才計算答案與政策記錄。
評估保存各模型、普通等權 ensemble、第一輪可靠度 ensemble、discussion ensemble，
包含 Top-1/3/5/10、MRR、rank percentile、隨機基準、逐模型 revision gain、
修正後 hit/miss。最多 30 個 fold；UI 預設 8，避免不必要訓練消耗。
比較不同模型時需留意 samples：較晚 ready 的模型可評估樣本較少。

精簡合成驗證：NumPy seed=2026 產生 9 張隨機 6×10 排列，
evaluation seed=42、揭露 50%、8 folds：

| 方法 | samples | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Statistical | 8 | .125 | .125 | .375 | .2211 |
| Position / Association / KNN（各自） | 6 | .1667 | .3333 | .3333 | .3038 |
| 普通 ensemble | 8 | .125 | .250 | .375 | .2650 |
| 第一輪 ensemble | 8 | .125 | .250 | .375 | .2650 |
| Discussion ensemble | 8 | .125 | .250 | .375 | .2650 |

**discussion_gain = 0，未改善，未證實討論有效。** 這組資料沒有足夠歷史，Peer Review
未啟用；四個 learned models 在此回測為 not_ready，沒有捏造比較結果。
另以 20 張微型合成表格執行四模型訓練及完整有限分數檢查。
合成小樣本只驗證程式流程；表中偶然高於隨機的數字不代表真實照片／刮卡預測能力。
沒有真實完整歷史表格，因此**尚無真實資料準確度結論**。

```bash
SCRAPER_ENABLED=false OMP_NUM_THREADS=2 python -m pytest -q tests/test_layout.py
# 另需本機 PostgreSQL 及 migration；測試資料在 transaction 中回滾
LAYOUT_TEST_DATABASE=1 SCRAPER_ENABLED=false OMP_NUM_THREADS=2 python -m pytest -q tests/test_layout_db.py
node --check taiwan_lottery/static/js/layout.js
```

涵蓋輸入驗證、not_ready、全候選排名、correlation 折減、唯一 assignment、
可重現性、holdout 隔離、四個 learned model、FastAPI import/lifespan/頁面、
PostgreSQL round trace、揭露後 immutable prediction，以及合成圖片實際 Tesseract 辨識。
使用者附的反光／遮擋實拍照片仍需在 UI 校正，未宣稱自動 OCR 能準確讀出所有格子。

## OCR 多視覺模型升級

本次只擴充已開格辨識，保留既有 layout prediction/discussion 核心。
來源與授權見 [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md)。

- 統一 recognizer 及 lazy singleton registry：Tesseract、EasyOCR、Paddle 三個
  recognition model、TrOCR small/base、自製 PyTorch CTC／segmented digit CNN。
  **Tesseract、EasyOCR、PARSeq tiny、CRNN 已實際對真實 cell 推論**；後兩者僅 benchmark_only，
  Paddle / TrOCR 權重來源被網路封鎖，保留 adapter，回報 model_missing。
- 每格產生 RGB、灰階、CLAHE、Otsu、adaptive、inverted、sharpened、denoise、
  glare-reduced 九種版本，先在 engine 內融合。Tesseract 使用 multipage TIFF，
  一次 subprocess 處理所有啟用的前處理／ROI 版本。Top-K 只保留實際產生的候選，不捏造替代答案。
- 獨立視覺 CellStateClassifier 用亮度、飽和度、texture、edge 等特徵判定
  OPENED / UNOPENED / UNCERTAIN，不以 OCR 猜是否刮開。OPENED 和 UNCERTAIN 都辨識；後者降低信任、強制檢查，只有 UNOPENED 完全跳過。
  這是保守 heuristic，尚無訓練完成的 state CNN。
- 無真實 validation 時採明示的等 family prior；同系 Paddle、TrOCR 及兩個
  Digit 模型共享家族影響力。不同 augmentation 不各拿一票。
- Global Hungarian 僅使用 OPENED / UNCERTAIN 格已有的合法候選，保留 abstention dummy，
  不靠缺號猜未開格。無支持候選不得硬分配。低於 .15 abstain、低於 .65 等情形醒目提示，
  是工程 heuristic，不是正確率；保留 raw / ensemble / constrained 結果及修正原因。
- UI 可只看爭議格，查看引擎候選、最終分數、人工狀態、約束修正、載入時間、
  per-cell / whole-board latency、RSS，並人工標記六種拍攝場景。

### 人工資料、fine-tune 與 benchmark

先執行 alembic upgrade head（新增 c102_digit_dataset）。
PostgreSQL layout_digit_dataset 保存 crop 相對路徑、raw outputs、確認標籤／狀態、
尺寸、source board ID、原圖 hash、品質／場景。勾選保留才保存 crops，不保存原始整張照片。
私有 crop 與模型皆在忽略的資料目錄。第一版未建立自動過期、認證或多人管理，
適用於受信任的本機開發用途。

同一實體表格，包括重拍及相近 crop，必須填寫**相同 source board ID**。
同來源 ID 或同原圖 hash 的資料 union 後整組拆分，絕不隨機切 cell。
不同照片的 hash 無法自動證明是不同實體表格，來源標註不可省略。

少於 500 labelled OPENED cells 為 not_ready；500～1999 experimental；
至少 2000 benchmark_eligible，仍不是品質保證，另需至少三個獨立來源。
校正 API 每跨 500 格可排程本機 fine-tune（auto_train=false 可停用）。
CTC 及 segmented CNN 都有實際訓練流程，不使用 MNIST 或固定 1～100 classifier。
checkpoint 使用 weights_only=True 及 SHA-256 驗證。
實驗模型預設不投票；至少 2000 格且獨立 test 優於現成 OCR 才可 promote。
LAYOUT_OCR_EXPERIMENTAL_DIGIT=1 僅供實驗，不作正式準確率證據。

新增 API（前綴 /api/v1/layout）：

    GET  /ocr/models
    POST /recognize            原 multipart，新增 retain=false
    POST /ocr/corrections      capture_id, source_board, cells[index,state,label,scenarios], auto_train
    GET  /ocr/dataset
    POST /ocr/digit/train
    GET  /ocr/digit/jobs/{uuid}
    POST /ocr/benchmark
    GET  /ocr/benchmark

benchmark 使用真實人工標記資料，依 source-group 分 train/validation/test，
權重及 engine 組合在 validation 選定後才評估 test，保存 split IDs。
比較單模型、ensemble 及 constrained ensemble，正式部署仍須遵守全表不重複。
報告包含 exact accuracy、CER、Top-3 recall、整表全對率、錯格數、約束前重複數、
修正數、原本正確卻被約束改錯的數量，以及六種拍攝場景。
至少 50 個 validation cells、三個 validation source boards，才保存校正及最佳部署 profile；
版本／artifact fingerprint 不符時停用舊 profile。不足門檻只報結果，不自動提升。

先前版本只有合成 smoke。現在已完成三張真實 regression 圖的量測，見下方實測文件；仍沒有獨立 validation 選出的正式 ensemble 或泛化保證。
本機合成 2×2 / 44px 字體 smoke（不是正式 accuracy）：
Tesseract 原始候選 4/4、EasyOCR 3/4、保守融合填入 3/4、1 格 abstain；
不把較差或 abstain 的結果美化為提升。
該次 cold 整表約 3.37 秒，含 EasyOCR 載入；warm 單格 Tesseract 約 172ms、
EasyOCR 約 128ms，RSS 約 419MiB，EasyOCR 權重約 14.44MiB。
這不是 100 格 Windows 實測，不作線性速度保證。

Paddle / TrOCR 因權重無法取得，模型大小與 latency 未測。
Digit 尚無足夠真實資料，正式 accuracy / 選型 not_ready。
兩種 Digit 架構已用微型合成資料實際做 1-epoch CPU 訓練／推論 smoke，不能作 promotion。
反光遮住筆畫、半刮開、連字、錯誤角點、極低解析仍可能出錯或 abstain，須人工校正。

    pip install -r requirements-ocr.txt
    python -m taiwan_lottery.ml.layout.vision.prepare easyocr
    LAYOUT_TEST_DATABASE=1 SCRAPER_ENABLED=false OMP_NUM_THREADS=2 python -m pytest -q tests

Windows 須另裝 Tesseract，並在 shell 設定環境變數。
已檢查指定套件的 Windows/Python 3.12 wheel resolution，未在 Windows 執行，
也未完整驗證該平台全部 transitive dependencies。


## 真實三圖 regression 與可重現 debug

詳見 [實測報告](docs/layout-real-regression.md)。照片、人工逐格標註、
baseline/final JSON、CSV、warp、所有 crop/前處理都保存在外部本機 fixture/debug 目錄，
不包含在 Git commit。正式模型權重及 third-party source cache 也不提交。

    # Core runtime dependencies, then optional scene-text benchmark dependencies.
    pip install -r requirements.txt
    pip install -r requirements-scene-ocr.txt
    python -m taiwan_lottery.ml.layout.vision.prepare tesseract
    python -m taiwan_lottery.ml.layout.vision.prepare parseq-tiny
    python -m taiwan_lottery.ml.layout.vision.prepare crnn

使用隔離 --target 安裝 scene stack 時，設 LAYOUT_OCR_SCENE_DEPENDENCIES 指向它。
正式應用不啟用 LAYOUT_OCR_BENCHMARK，就不載入這兩個額外模型；
不會因這三張 test 圖上的成績把模型提升為 runtime。

    LAYOUT_OCR_BENCHMARK=1 OMP_NUM_THREADS=2 python -m taiwan_lottery.ml.layout.vision.regression       --fixtures /external/ground_truth.json --images-root /external/images --output /external/debug

fixture JSON 是圖記錄陣列：image（路徑可自行命名）、rows、cols、cells；
每個 cell 包含 row、col、state、number 或 null、ambiguity，
可附 corners 作為人工幾何評估參考，及 scenarios。參考 corners **不傳入自動偵測器**。
保留 split=regression_only；程式不從 filename、image hash、特定座標或答案選推論路徑。
必須逐格枚舉合法座標，只有 OPENED 可填入不重複 1～N 的確認數字。

新偵測器先找重複覆膜／cell contour 的 projective lattice，以 RANSAC 排除背景雜物；
再使用 morphology、Hough 與等距格線評分備援，失敗會要求人工角點。
辨識保留 normal/tighter/looser crop、number ROI、polarity 與可選三階亮度 ink-band 視圖。
每個數值採最佳兩個原生觀察的平均證據，避免多個相關的壞 thresholding 壓過一個清楚 crop。
候選保留 Top-K，不把 augmentation 當成不同模型。

約束先依候選與次佳差距保護主證據，再對重複候選的未決格做 Hungarian，
有 private abstention dummy，不用剩餘數字補猜未開格。底線 .15、備援至少達最佳候選的 75%
是明示 heuristic，未經真實 validation 校準。單一／不一致證據的 confidence 上限 .79；
所有原生高分錯讀仍在獨立欄位列出，這不是校準改善或命中率保證。
某一張圖仍可能因唯一性約束丟掉原本正確的候選，實測不會隱藏這種代價。

    LAYOUT_REAL_REGRESSION_DIR=/external/regression       LAYOUT_TEST_DATABASE=1 SCRAPER_ENABLED=false OMP_NUM_THREADS=2 python -m pytest -q tests

私有目錄包含 ground_truth.json、原圖、baseline_metrics.json、final_metrics.json、
validation_summary.json。私有 acceptance 按每張檢查不低於 baseline、未開格 false-number=0，
並檢查全體約束後不低於同一 abstain 政策下的約束前；個別圖片的損失仍完整呈現。
CI 沒有私有圖片時明確 skip，不能宣稱在 GitHub 跑過實拍 regression。

### 指定數字位置的 blind backtest

新增獨立離線 evaluator，固定遮蔽 50%，target 永遠 hidden。Predictor 子程序只讀 sanitized request / train-only bundle；預測封存後 scorer 才讀答案。沒有改動現有 OCR 或位置模型。本地完整歷史 corpus 目前為空，因此真實命中率仍 `not_ready`，不能以 unit fixtures 宣稱模型有效。

流程、資料規格、重跑指令與誠實空資料結果見 [blind backtest 報告](docs/layout-position-blind-backtest.md)。

### 製造 Excel 的 Calc blind baseline

使用者另提供製造 XLSX，本輪依其原始 `RAND()`／`RANK()` 公式，由 LibreOffice Calc 實際硬重算；不是 Python 製造排列。固定 160 刮、10×16、target=73、公開 80 格。40 張訓練盤、5 張工程盤，另有首輪及 fresh 各 1,000 張 × 10 masks；輸出封存後才開答案評分。

Calc 與 Microsoft Excel 的 RAND 實作可能不同，不能宣稱 Excel-native 亂數序列重現。模型未因 test 答案調整；方法、隔離協定與實測結果見 [製造機制 blind backtest](docs/layout-manufacturing-blind-backtest.md)。
