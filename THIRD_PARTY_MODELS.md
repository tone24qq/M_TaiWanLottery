# Cell OCR model sources and licensing

No third-party source tree or model weights are committed. Optional packages and
local model caches never block FastAPI startup. Preparation uses normal TLS and
the inherited proxy; inference never downloads weights.

| Model | Official source | Version/revision | License and runtime |
| --- | --- | --- | --- |
| Tesseract / eng LSTM | https://github.com/tesseract-ocr/tesseract ; https://github.com/tesseract-ocr/tessdata_fast | Installed binary reported by tesseract --version; Python wrapper in runtime metadata | Apache-2.0 engine and official tessdata; fallback |
| EasyOCR english_g2 | https://github.com/JaidedAI/EasyOCR ; https://github.com/JaidedAI/EasyOCR/releases/tag/v1.3 | Package 1.7.2; tag commit c4f3cd7225efd4f85451bd8b4a7646ae9a092420; weight SHA-256 in ignored artifact.json | Apache-2.0 upstream repository and distributed official model; CPU recognition only |
| PP-OCRv6_medium_rec | https://github.com/PaddlePaddle/PaddleOCR | paddleocr 3.7.0 / paddlepaddle 3.3.1; docs reference HEAD dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf | Apache-2.0; optional local weights |
| PP-OCRv6_small_rec | Same official PaddleOCR source | Same package versions; exact model name preserved | Apache-2.0; optional |
| en_PP-OCRv5_mobile_rec | Same official PaddleOCR source | Same package versions; exact model name preserved | Apache-2.0; optional |
| microsoft/trocr-small-printed | https://huggingface.co/microsoft/trocr-small-printed ; https://github.com/microsoft/unilm/tree/master/trocr | transformers 5.22.2; model snapshot SHA resolved by preparation; UNILM reference HEAD 31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd | Upstream code MIT; model-card license could not be fetched here. Disabled until permissive license and exact snapshot are verified |
| microsoft/trocr-base-printed | https://huggingface.co/microsoft/trocr-base-printed | Separate model snapshot and license gate | Same unverified-weight restriction; CPU benchmark candidate |
| DigitCellRecognizer CTC / segmented CNN | This repository, vision/digit.py and vision/training.py | Custom seeded architecture; checkpoint SHA-256, source splits, epochs and metrics recorded | Project license; no MNIST, unlicensed weights or copied third-party model source |

Paddle documentation explicitly lists all three selected recognition models.
Official inference archive template:
https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/{model}_infer.tar

The preparer records archive SHA-256 and extracted-file hashes after download.
No hash or license verification is fabricated for inaccessible artifacts.

## Actual observations

- All optional packages installed on Python 3.12 Linux CPU; pip check passed.
- EasyOCR english_g2 weights downloaded from the official GitHub release.
  Its native model MD5 check remains enabled; local artifact.json adds SHA-256.
  The actual weight file is 15,143,997 bytes.
- Tesseract and EasyOCR actually inferred synthetic cell digits, without loading
  a text detection model.
- The Paddle official model URL returned proxy 403 Forbidden. No Paddle model
  inference, accuracy, or measured weight size is claimed.
- Both TrOCR Hugging Face model-info endpoints returned proxy 403 Forbidden.
  No TrOCR model download, inference, accuracy or measured model size is claimed.
- Access would require the official Paddle hostname and Hugging Face/CDN
  destinations to be allowed by environment settings. No network bypass, TLS
  disabling or environment publication was performed.
- A pip wheel-resolution dry run found Windows win_amd64 / Python 3.12 artifacts
  for EasyOCR 1.7.2, Transformers 5.22.2, PaddlePaddle 3.3.1, PaddleOCR 3.7.0 and
  TorchVision 0.29.2. This does not validate every transitive dependency or actual
  Windows execution. Install a matching PyTorch/TorchVision CPU pair; system
  Tesseract must be installed separately and on PATH.

## Explicit preparation

Run only after installing requirements-ocr.txt and allowing official destinations:

    python -m taiwan_lottery.ml.layout.vision.prepare easyocr
    python -m taiwan_lottery.ml.layout.vision.prepare PP-OCRv6_medium_rec
    python -m taiwan_lottery.ml.layout.vision.prepare PP-OCRv6_small_rec
    python -m taiwan_lottery.ml.layout.vision.prepare en_PP-OCRv5_mobile_rec
    python -m taiwan_lottery.ml.layout.vision.prepare trocr-small
    python -m taiwan_lottery.ml.layout.vision.prepare trocr-base

TrOCR preparation accepts only model cards declaring MIT, Apache-2.0 or BSD,
pins the exact snapshot revision, and saves license evidence with local hashes.
A process restart is needed after preparing new models.

LAYOUT_OCR_MODEL_DIR selects the cache, default model_artifacts/layout_ocr.
LAYOUT_OCR_DATA_DIR selects retained crops, default model_artifacts/layout_digit_dataset.
Both are ignored by Git. Do not commit private labelled crops or model artifacts.

## Evidence limits

尚無真實照片 accuracy 結論. No real-data ensemble winner or specialist improvement
is asserted. CPU timings in README are synthetic workload observations, not
Windows guarantees.

The specialized models train from real corrected sequences and, separately,
digit components only when component count matches the confirmed sequence.
Training-only augmentation includes perspective, rotation, blur, motion blur,
illumination, glare, scratches, partial masks, low resolution, dirt and JPEG
damage. These augmented images are never a formal accuracy test.
