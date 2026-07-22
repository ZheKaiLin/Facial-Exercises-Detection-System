# Facial Exercises Feedback System

*[繁體中文](README.md) | English*

Combines a lightweight YOLO facial-expression detector with MediaPipe facial landmarks to guide users through real-time facial exercises, automatically logging movement amplitude to evaluate short-term training effect.

- Expression classes: `angry`, `happy`, `neutral`, `sad`, `surprised`
- 4 training exercises: facial expression training, face-lift training, mouth-open training, frown/brow-tighten training
- Inference: YOLOv7-tiny / YOLOv9-tiny trained and exported to ONNX, paired with MediaPipe Face Mesh for landmark tracking

## System Architecture (3 stages)

```
1. Dataset construction (dataset_pipeline/)
      蔡旻均 (2020) source dataset → stratified train/val split → minority-class augmentation → add external stock-photo test set
                        ↓
2. YOLO model training (model_training/)
      YOLOv7-tiny / YOLOv9-tiny training → compare mAP per batch size → pick best weights → export CPU ONNX
                        ↓
3. Facial exercises system (app/)
      Drop ONNX weights into app/weights/ → real-time detection + guide the user through facial exercises
```

## Directory Structure

```
Facial-Exercises-Detection-System/
├── dataset_pipeline/                      # 1. Dataset construction
│   ├── 01_train_val_split/
│   │   └── train_val_split.py             # Stratified split of the trainval pool into train/val
│   ├── 02_balance_augmentation/
│   │   └── make_5_balanced_data.py        # Minority-class augmentation on the train set (flip/brightness-contrast/blur)
│   ├── 03_test_100_annotation_reliability/ # Annotation quality check for the external stock-photo test set (test_100)
│   │   ├── prepare_blind_annotation.py     # Generate anonymized blind-annotation samples
│   │   ├── annotate_gui.py                 # Second annotator's labeling UI
│   │   ├── analyze_agreement.py            # Compute Cohen's kappa / IoU agreement
│   │   ├── agreement_summary.csv
│   │   └── test_100_sources.csv            # Source stock-photo platform for each test_100 image
│   └── 04_test_train_dedup_check/          # Systematic duplicate check between test_300 and train
│       ├── check_test_train_duplicates.py  # MD5 exact match + pHash (256-bit) near-duplicate check
│       └── duplicate_report.csv            # Output (result of this run: 0 duplicates)
├── model_training/                        # 2. YOLO model training
│   ├── yolov7/
│   │   ├── face_dataset*.yaml             # Dataset config (train/val/test paths, classes)
│   │   ├── make_5_balanced_data.py
│   │   ├── run.py                          # PyTorch real-time system (requires full YOLOv7 source)
│   │   ├── commands.txt                    # Train/val/test/export-ONNX commands and mAP per batch size
│   │   └── runs/                           # Training/validation/test logs (charts, logs; weight files via cloud link)
│   └── yolov9/
│       ├── face_dataset*.yaml
│       ├── commands.txt
│       └── runs/
├── app/                                    # 3. Facial exercises system
│   ├── Facial_Exercises_Training.py         # Main system: real-time guided facial exercise training
│   ├── Facial_Exercises_Evaluation.py       # Research evaluation flow: pre-test → training → post-test
│   └── weights/
│       ├── v7best.onnx                     # YOLOv7-tiny exported ONNX weights
│       └── v9best.onnx                     # YOLOv9-tiny exported ONNX weights
├── statistical_analysis/                   # 4. Statistical analysis (for the thesis)
│   ├── 3.3_雙樣本比例Z檢定.py                # Two-sample proportion Z-test
│   ├── 4.6.1_*.py / 4.6.2_*.py / 4.6.3_*.py / 4.6.3附圖_*.py
│   ├── 總統計表(去識別化).xlsx              # Real-name lookup sheet removed; only P01~P15 codes remain
│   ├── fig_table24_gap.png
│   └── yolo原始資料/                        # YOLO test-set labels and aggregate metric plots (confusion matrix / F1 / PR curves)
└── requirements.txt
```

> `evaluation_data/` (participants' pre/post-test records) and the actual dataset images/labels are participant personal data and/or large binary files, so they are excluded from version control (see `.gitignore`) and kept locally only. `statistical_analysis/原始資料/` (real-name folders with training videos) and any file containing a real-name lookup table are likewise excluded from version control; only de-identified summary results are kept.

## Setup

Please set up **a separate virtual environment for each of the three parts** — do not share one, to avoid package-version conflicts:

| Part | Python version | Notes |
|---|---|---|
| `app/` (facial exercises system) | **3.9** | Version actually used during development/testing |
| `model_training/yolov7/` | **3.12** | Package versions required by the official YOLOv7 repo |
| `model_training/yolov9/` | **3.12** | Package versions required by the official YOLOv9 repo |

**Facial exercises system (`app/`)**:

```bash
py -3.9 -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> `requirements.txt` only covers the packages needed by the **real-time detection app** (`app/`) — OpenCV, MediaPipe, ONNX Runtime, etc.

**YOLO model training (`model_training/`)**: the official YOLOv7 and YOLOv9 repos each require different package versions (PyTorch, etc.). Even though both use Python 3.12, please **create a separate virtual environment for each** rather than sharing one, to avoid conflicts. See "2. YOLO Model Training" below for setup.

## 1. Dataset Construction

The dataset builds on the facial-expression dataset collected by **蔡旻均 (2020)** ([original dataset link](https://drive.google.com/drive/folders/1kcvTD2jTDpjJ_9v79TEqog7bfczA0FpH)) with 5 classes (angry / happy / neutral / sad / surprised), then re-split, balanced, and augmented as follows:

| Step | Input | Processing | Output |
|---|---|---|---|
| 0. Source data | 蔡旻均 (2020) `new_glasses_splidata` (1,913 images, YOLO-format labels) | Split per the original author's `train.txt` (1,713) / `test.txt` (200) | trainval pool 1,713 / test 200 |
| 1. train/val stratified split | trainval pool 1,713 | `dataset_pipeline/01_train_val_split/train_val_split.py`: stratified split by class, val ratio 15%, `random_state=42` for reproducibility | train 1,456 / val 257 |
| 2. Minority-class augmentation | train 1,456 | `dataset_pipeline/02_balance_augmentation/make_5_balanced_data.py`: horizontal flip, brightness/contrast adjustment, Gaussian blur for classes under 400 images (**train only, test untouched**) | balanced train 2,172 |
| 3. Add external stock-photo test set | test 200 (from step 0's `test.txt` split, untouched by steps 1/2) | Added 100 expression images from **Pexels, iStock, Pixabay, Dreamstime, Freepik**, quality-checked via `03_test_100_annotation_reliability/` blind double-annotation agreement (Cohen's kappa, bounding-box IoU) | test 200 + 100 = 300 |
| 4. Test/train duplicate check | test_300 (300) × train (2,172) | `dataset_pipeline/04_test_train_dedup_check/check_test_train_duplicates.py`: MD5 exact match (byte-level) + pHash 256-bit near-duplicate check (Hamming distance ≤10%), to confirm the test set doesn't overlap the train set | 0 MD5 exact matches, 0 pHash near-duplicates; closest hash pair still 60/256 bits apart (~23.4% difference), manually confirmed as different individuals |

The intermediate output of minority-class augmentation (step 2) is `v4_data_balanced/` (train/val only, before adding the external test set); after adding the external stock-photo test set (step 3) it becomes the final dataset `v4_data_final/` (not included in this repo; download links below):

```
v4_data_balanced/                             # Step 2 output (intermediate)
├── images/{train, val}
└── labels/{train, val}

v4_data_final/                                # Step 3 output (final, used directly for YOLO training)
├── images/{train, val, test_100, test_200, test_300}
└── labels/{train, val, test_100, test_200, test_300}
```

- `train`: 2,172 images (balanced)
- `val`: 257 images
- `test_200`: the original 200-image test set
- `test_100`: 100 external stock-photo images; source platform per image in `dataset_pipeline/03_test_100_annotation_reliability/test_100_sources.csv`
- `test_300`: `test_200` + `test_100` combined — the final test set used for YOLO training/testing

Per-platform expression-class counts for `test_100` (5 classes, 100 images total, 20 per class):

| Platform | angry | happy | neutral | sad | surprised | Subtotal |
|---|---|---|---|---|---|---|
| Pexels | 7 | 20 | 20 | 1 | 20 | 68 |
| iStock | 3 | 0 | 0 | 7 | 0 | 10 |
| Pixabay | 8 | 0 | 0 | 1 | 0 | 9 |
| Dreamstime | 2 | 0 | 0 | 5 | 0 | 7 |
| Freepik | 0 | 0 | 0 | 6 | 0 | 6 |
| **Total** | **20** | **20** | **20** | **20** | **20** | **100** |

`happy`, `neutral`, and `surprised` were sourced entirely from Pexels; `angry` and `sad` were supplemented from iStock, Pixabay, Dreamstime, and Freepik due to insufficient Pexels material for those classes.

> Each image's source platform is inferred from its filename prefix (e.g., `Freepik_sad1.jpg` → Freepik → sad); the full per-image mapping (with expression class) is in `test_100_sources.csv`.

`dataset_pipeline/04_test_train_dedup_check/check_test_train_duplicates.py` requires `Pillow` and `imagehash` (`pip install pillow imagehash`); it runs MD5 exact matching and pHash (256-bit) near-duplicate matching between `test_300` and `train` to confirm the test set doesn't overlap the training set. Full output is in `duplicate_report.csv` in the same directory.

### Dataset Download

Due to file size (raw data > 10 GB) and licensing/privacy considerations (subject images in the `蔡旻均 (2020)` original dataset, stock-photo licensing for `test_100`), the actual images are not included directly in this git repo. Cloud drive links are provided instead for each stage's output (matching the steps above):

| Stage | File | Download Link |
|---|---|---|
| 0. Source data | `new_glasses_splidata_raw_data.zip` | [Google Drive](https://drive.google.com/file/d/1k-_NVOv6kA1L-wLmZwjg8FAxws8xnQK9/view?usp=sharing) |
| 0. train/test split | `v4_data_train_test_split.zip` | [Google Drive](https://drive.google.com/file/d/1alhxDbqimRergwAwPtfl0uPbL86rUtlZ/view?usp=sharing) |
| 1. train/val stratified split | `v4_data_train_val_split.zip` | [Google Drive](https://drive.google.com/file/d/1bgTuFju1fe4iR-mlBIWud8Nwz5_m25Qb/view?usp=sharing) |
| 2. Minority-class augmentation | `v4_data_balanced.zip` | [Google Drive](https://drive.google.com/file/d/1193CTIyYW23W7lpTdWdnbbmGPGYgAqCg/view?usp=sharing) |
| 3. Add external test set (final, training-ready) | `v4_data_final.zip` | [Google Drive](https://drive.google.com/file/d/1jjJXyhy8OAhbLLLo-ey4YrwNsiWZLfOa/view?usp=sharing) |

If you only need to retrain the YOLO models, downloading `v4_data_final.zip` is sufficient (matches `v4_data_final/` required by "2. YOLO Model Training"); to reproduce the entire dataset-construction pipeline, start from `new_glasses_splidata_raw_data.zip` and run the steps in order.

Before downloading, please make sure your intended use complies with research ethics and stock-photo licensing terms.

## 2. YOLO Model Training

The detection model is fine-tuned from the official [YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv9](https://github.com/WongKinYiu/yolov9) repos. `model_training/` only keeps the **custom config files and scripts** — you need the official source code first. YOLOv7 and YOLOv9's official repos require different package versions, so **set up a separate virtual environment for each**, without sharing:

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
# Copy the files under model_training/yolov7/ into the yolov7/ directory
# Extract v4_data_final.zip from "1. Dataset Construction → Dataset Download" into the yolov7/ directory
```

The same process applies to `yolov9`, using [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) instead, in its own separate directory/virtual environment to avoid cross-contamination with the yolov7 setup.

The full train/validate/test/export-ONNX commands and mAP results per batch size (b16 / b32 / b64, 250 epochs each) are fully recorded in:
- `model_training/yolov7/commands.txt`
- `model_training/yolov9/commands.txt`

Workflow: train b16/b32/b64 separately → compare mAP@.5 on the val set to pick the best batch size's weights → test generalization on `test_100`/`test_200`/`test_300` → export the best weights to CPU-ready ONNX.

Example (YOLOv7-tiny, 250 epochs):

```bash
python train.py --workers 10 --device 0 --batch-size 32 \
    --data face_dataset.yaml --img 224 \
    --cfg cfg/training/yolov7-tiny.yaml --weights yolov7-tiny.pt \
    --name v4_data_e250_b32 --hyp data/hyp.scratch.tiny.yaml --epochs 250

python export.py --weights runs/train/v4_data_e250_b32/weights/best.pt \
    --img-size 224 224 --batch-size 1 --device cpu --grid --simplify
```

The equivalent YOLOv9 commands (`train_dual.py` / `val_dual.py` / `export.py --include onnx`) are in `model_training/yolov9/commands.txt`.

### Training Results (`runs/`)

`model_training/yolov7/runs/` and `model_training/yolov9/runs/` keep the full train/validation/test records (confusion matrix, PR/F1/P/R curves, train/test batch visualizations, TensorBoard logs, `opt.yaml`/`hyp.yaml`, per-epoch results tables) so you can compare the three batch sizes' training runs directly without retraining:

```
runs/
├── train/{b16_e250, b32_e250, b64_e250}/   # Training process per batch size
├── val/{...}/                              # Individual validation results
└── test/{pooled_test100, ...200, ...300}/  # Final test results on test_100/200/300
```

The per-epoch weight checkpoints (`weights/*.pt` — YOLOv7 has 19–25 per group, YOLOv9 has 10–12 per group, ~3.4 GB total) are too large for git, so a full `runs/` backup (including all the charts/logs above, for reference) is provided via cloud drive instead:

| Version | File | Download Link |
|---|---|---|
| YOLOv7 (all b16/b32/b64 checkpoints) | `yolov7_runs.zip` | [Google Drive](https://drive.google.com/file/d/1_6u0pJvDNm35GxtI0uzMtHPfkZfQLkHE/view?usp=sharing) |
| YOLOv9 (all b16/b32/b64 checkpoints) | `yolov9_runs.zip` | [Google Drive](https://drive.google.com/file/d/1vDYreE2ZvngDAJwOvjyi0Q3BRrf6FIB4/view?usp=sharing) |

> `train_batch*.jpg` / `test_batch*.jpg` are training/test batch visualization images, containing dataset face images (from the 蔡旻均 (2020) dataset or external stock photos — not real thesis participants).

The exported best weights (already included in this repo as `v7best.onnx` / `v9best.onnx`) go into `app/weights/` for the stage-3 facial exercises system.

## 3. Facial Exercises System

The app uses the already-trained ONNX weights (`app/weights/`) — no GPU required, CPU-only real-time inference, paired with MediaPipe for facial landmark tracking to measure movement amplitude.

**`Facial_Exercises_Training.py`: the main system**, guiding the user through facial exercises in real time and logging movement amplitude:

```bash
cd app
python Facial_Exercises_Training.py ^
    --weights weights/v7best.onnx ^
    --source 0 ^
    --participant 01 ^
    --session training ^
    --reps 10
```

On launch, the program enters the **main menu**; the training exercise is chosen live on-screen via number keys, not via a startup argument:

| Key | Exercise | Detection Method |
|---|---|---|
| `1` | Facial expression training | Cycles through 5 expressions: angry, happy, neutral, sad, surprised |
| `2` | Face-lift training | Smile/lift training: combines expression recognition with mouth-corner position changes |
| `3` | Mouth-open training | Mouth-opening training: real-time feedback from upper/lower lip distance plus expression state |
| `4` | Frown/brow-tighten training | Frown training: horizontal eyebrow convergence and vertical lowering amplitude, combined with expression state |

Main startup arguments:

| Argument | Description |
|---|---|
| `--weights` | ONNX weights path, default `v7best.onnx` |
| `--source` | Camera index or video path, default `0` |
| `--session` | `pre_test` / `training` / `post_test` |
| `--participant` | Participant ID, used for log-file naming |
| `--reps` | Rounds per exercise (for exercises 2/3/4); 3 is recommended for pre/post-tests |

**`Facial_Exercises_Evaluation.py`: the research evaluation flow**, used for running an automated "pre-test → training → post-test" session with actual study participants:

```bash
cd app
python Facial_Exercises_Evaluation.py --weights weights/v9best.onnx --source 0
```

Run logs are written to `evaluation_data/` (calibration baselines, movement-amplitude time series — participant personal data, not included in this repo).

## 4. Statistical Analysis

`statistical_analysis/` contains the scripts and summary results used for the thesis's statistical analysis:

| File | Content |
|---|---|
| `3.3_雙樣本比例Z檢定.py` | Two-sample proportion Z-test for YOLOv7-tiny / YOLOv9-tiny Recall/Precision across test subsets |
| `4.6.1_*.py` | Relationship between Score and movement amplitude (Pearson / Spearman / rmcorr) |
| `4.6.2_*.py` | Pre/post-test improvement and per-round trend (paired t-test) |
| `4.6.3_*.py`, `4.6.3附圖_*.py` | Necessity of personalized calibration (CV, Wilson CI) and fixed-threshold misjudgment illustration |
| `總統計表(去識別化).xlsx` | Participant summary table, per-round amplitude, Score details, calibration baseline summary — all presented under `P01`–`P15` codes |
| `yolo原始資料/` | YOLOv7/v9 test-set labels and aggregate metric plots (confusion matrix, F1/PR/P/R curves) |

> ⚠️ Folders containing participants' real names (including training videos `.mp4`) and the real-name lookup table are **not included in this repo** — kept locally only (see `.gitignore`). The included `總統計表(去識別化).xlsx` has had its "anonymization lookup" sheet removed; the other sheets were already recorded under `P01`–`P15` codes with no real names.

## License and Citation

This project is licensed under **GPL-3.0** (see [LICENSE](LICENSE)), consistent with the official [YOLOv7](https://github.com/WongKinYiu/yolov7) / [YOLOv9](https://github.com/WongKinYiu/yolov9) repos that `model_training/` depends on (both GPL-3.0). Anyone using, modifying, or distributing derivative works of this project must release the source code under GPL-3.0.

This project's dataset builds on the facial-expression dataset collected by **蔡旻均 (2020)** ([original dataset](https://drive.google.com/drive/folders/1kcvTD2jTDpjJ_9v79TEqog7bfczA0FpH)); please cite the original dataset's source when using it.

<!-- TODO: add the thesis's formal citation format (author, title, year) -->
