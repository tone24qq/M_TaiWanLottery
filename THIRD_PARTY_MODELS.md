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

Real three-photo regression is now measured in docs/layout-real-regression.md.
There is still no independent validation-selected deployment winner, specialist
training improvement, or generalization guarantee. Earlier README timings were
synthetic; new real-photo CPU measurements are Linux observations, not Windows guarantees.

The specialized models train from real corrected sequences and, separately,
digit components only when component count matches the confirmed sequence.
Training-only augmentation includes perspective, rotation, blur, motion blur,
illumination, glare, scratches, partial masks, low resolution, dirt and JPEG
damage. These augmented images are never a formal accuracy test.

## Additional scene-text recognizers prepared in this task

| Model | Official source / revision | Actual weight bytes | SHA-256 | Policy |
| --- | --- | ---: | --- | --- |
| PARSeq tiny | https://github.com/baudm/parseq ; source 1902db043c029a7e03a3818c616c06600af574be ; author release v1.0.0 | 24,136,675 | e7a21b543c98e67414a584c93b1dbb71c26e9463ae4aaa391d54e833660a4711 | Apache-2.0; benchmark_only |
| CRNN (official PARSeq model hub implementation) | Same pinned source and author release | 33,465,605 | 679d0e315af22d813c6d5a86287ee5705a9a8dfe37378349359442b8b8953264 | Apache-2.0; benchmark_only |
| Official Tesseract fast English | https://github.com/tesseract-ocr/tessdata_fast ; 87416418657359cb625c412a48b6e1d6d41c29bd | Measured in local artifact inventory | 7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2 | Apache-2.0; existing runtime fallback |

Both scene-text recognizers now actually run the supplied real cell crops through
BaseCellRecognizer adapters. They expose real Top-K alternatives from multiple
crops and the digit token lattice (PARSeq lattice alternatives are not claimed
to be fully autoregressive beam-search probabilities). No model or source tree
is copied into tracked repository files; explicit preparation caches pinned
official source and weights below the ignored model directory and verifies them.

The optional stack is requirements-scene-ocr.txt. The verified Python 3.12 Linux
research execution used timm 0.9.16, pytorch-lightning 2.2.0.post0, torchmetrics 1.3.1,
lightning-utilities 0.10.1, nltk 3.8.1 and setuptools 80.9.0. The legacy upstream
stack needs pkg_resources, so using the current setuptools without it fails.
An isolated --target directory and LAYOUT_OCR_SCENE_DEPENDENCIES avoid changing
the normal application environment. No Windows scene-stack execution is claimed.

Published weights are author-provided STR models. The upstream code exposes
real/synthetic scene-text training configurations; the complete exact released
training-corpus manifest was not independently verified. No claim is made that
they were trained specifically on scratched lottery cards.

Official PaddleOCR TextRecognition API and its PaddleX hoster resolver were
actually attempted. Its configured official alternatives are:
Hugging Face PaddlePaddle/<model>, ModelScope PaddlePaddle/<model>, AIStudio
PaddleX/<model>, and the documented BOS URL. All hosting destinations are blocked
by this environment's proxy (403); the GitHub v3.7.0 release contains source archives,
not recognition weights. Microsoft model-info API likewise returns 403 and no
TrOCR cache was found. These adapters remain not_ready; installed packages are
not counted as prepared/inferred models.

The original three photographs are private regression/test inputs, not training
or model-selection validation. No scene recognizer is promoted into runtime from
its accuracy on those images. Post-hoc best-single/pair/triple tables are descriptive,
not validated deployment recommendations.
