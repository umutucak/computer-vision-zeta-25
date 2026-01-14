# SPARK Challenge (Task-1 Detection + Task-2 Segmentation)

This repository contains a **self-contained, reproducible** implementation for the SPARK dataset challenge:

- **Task 1 (Detection):** predict **satellite class (10 categories)** + **bounding box** and generate `detection.csv`.
- **Task 2 (Segmentation):** predict **pixel-wise body/panel masks** and generate per-image `*_layer.npz` files for submission.

Both tasks are designed to run either:
- locally (e.g., workstation / notebook), or
- on an **HPC cluster with Slurm**, with **node-local data staging** to reduce I/O overhead.



## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── task1_detection
│   ├── stage_data.sh
│   ├── submit.py
│   ├── submit.slurm
│   ├── train.py
│   └── train.slurm
└── task2_segmentation
    ├── stage_data.sh
    ├── submit.py
    ├── submit.slurm
    ├── train.py
    └── train.slurm
````

Each task is **fully independent** (no code reuse across Task-1 and Task-2).



## 1. Setup

### 1.1 Create environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> On HPC, you can reuse your existing venv path and simply `source <venv>/bin/activate`.

### 1.2 Requirements

* Python 3.9+ (recommended)
* PyTorch + torchvision
* OpenCV
* `segmentation-models-pytorch` (Task-2)
* common libs: numpy, pandas, tqdm, Pillow



## 2. Data Layout Assumptions

Both tasks assume the dataset is available in a single **data root** folder containing:

```text
DATA_ROOT/
  train.csv
  val.csv
  images/
    <ClassName>/
      train/<img files>
      val/<img files>
  mask/                      # Task-2 only
    <ClassName>/
      train/<mask files>
      val/<mask files>
  test_images/images/        # Task-1 test
  test_seg_images/images/    # Task-2 test
```

On HPC, the `stage_data.sh` scripts create this structure inside node-local storage and export `DATA_ROOT`.



## 3. Running Locally (no Slurm)

### 3.1 Task-1 Training (Detection)

```bash
python task1_detection/train.py \
  --data-root /path/to/DATA_ROOT \
  --ckpt-dir ./ckpts_task1 \
  --epochs 40 --batch 64 --workers 6 \
  --lr 3e-5 --wd 1e-4 \
  --pretrained \
  --score-thr 0.0
```

Outputs:

* `./ckpts_task1/epoch_*.pth`
* `./ckpts_task1/best_by_leaderboard_task1.pth`

### 3.2 Task-1 Submission

```bash
python task1_detection/submit.py \
  --weights ./ckpts_task1/best_by_leaderboard_task1.pth \
  --test-dir /path/to/DATA_ROOT/test_images/images \
  --out-csv detection.csv
```

Output:

* `detection.csv`



### 3.3 Task-2 Training (Segmentation)

```bash
python task2_segmentation/train.py \
  --data-root /path/to/DATA_ROOT \
  --ckpt-dir ./ckpts_task2 \
  --cache-dir ./mask_cache_1024 \
  --img-size 1024 \
  --encoder timm-efficientnet-b3 --encoder-weights imagenet \
  --epochs 50 --batch 32 --workers 6 \
  --lr 3e-4 --wd 1e-2 \
  --panel-min-ratio 0.05
```

Outputs:

* `./ckpts_task2/best_task2_by_final50_95.pth`

### 3.4 Task-2 Submission

```bash
python task2_segmentation/submit.py \
  --test-dir /path/to/DATA_ROOT/test_seg_images/images \
  --ckpt ./ckpts_task2/best_task2_by_final50_95.pth \
  --out-dir submission_folder \
  --img-size 1024 \
  --encoder timm-efficientnet-b3 \
  --num-classes 3 \
  --tta
```

Output:

* `submission_folder/*_layer.npz`

Then zip:

```bash
zip -r task2_submission.zip submission_folder
```



## 4. Running on HPC (Slurm)

### 4.1 Before running: set dataset ZIP locations

Both `task1_detection/stage_data.sh` and `task2_segmentation/stage_data.sh` expect ZIP archives in:

* `$SCRATCH/dataset/`

Make sure these exist (names may be cluster-specific):

* `spark-2024-train-val.zip`
* `spark-2024-detection-test.zip` (Task-1)
* `spark-2024-segmentation-test.zip` (Task-2)
* `ground_truth_labels.zip` (if needed)

If your paths differ, edit the corresponding `stage_data.sh`.



### 4.2 Task-1 Training (Slurm)

```bash
sbatch task1_detection/train.slurm
```

This will:

1. stage data to node-local `$SLURM_TMPDIR`
2. run `task1_detection/train.py`
3. write checkpoints to `$SCRATCH/...` (as configured in your slurm script)

Find best weights:

* `best_by_leaderboard_task1.pth`

### 4.3 Task-1 Submission (Slurm)

Set the weight path and submit:

```bash
sbatch --export=ALL,WEIGHTS=/path/to/best_by_leaderboard_task1.pth task1_detection/submit.slurm
```

Output:

* `detection.csv` in the output directory defined in your Slurm script



### 4.4 Task-2 Training (Slurm)

```bash
sbatch task2_segmentation/train.slurm
```

Outputs:

* `best_task2_by_final50_95.pth`

### 4.5 Task-2 Submission (Slurm)

```bash
sbatch --export=ALL,WEIGHTS=/path/to/best_task2_by_final50_95.pth task2_segmentation/submit.slurm
```

Outputs:

* `submission_folder/*_layer.npz`
* zip the folder for submission (your Slurm script may already print the command)



## 5. Notes on Output Formats

### Task-1 (`detection.csv`)

* Columns: `filename`, `class`, `bbox`
* `bbox` is stored as a string: `(x1, y1, x2, y2)` with integer coordinates.

### Task-2 (`*_layer.npz`)

Each test image produces one NPZ:

* filename mapping: `*_img.jpg` → `*_layer.npz`
* contains `data` key
* `data` is `bool[H, W, 3]`

  * channel 0 (Red): body (`label==1`)
  * channel 2 (Blue): panels (`label==2`)



## 6. Reproducibility Tips

* Keep a log of:

  * Slurm job ID
  * checkpoint path
  * best validation score
* Prefer node-local staging via `stage_data.sh` on HPC
* For Task-2, enable mask cache (`--cache-dir`) to speed up later epochs

