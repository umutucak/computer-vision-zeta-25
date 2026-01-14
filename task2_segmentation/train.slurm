#!/bin/bash -l
#SBATCH --job-name=t2_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=hopper
#SBATCH --qos=besteffort

set -euo pipefail
mkdir -p logs

source /home/users/whuang/code/spark-zeta25/venv/bin/activate
source task2_segmentation/stage_data.sh

CKPT_DIR="$SCRATCH/ckpts/task2/$SLURM_JOB_ID"
mkdir -p "$CKPT_DIR"

# cache on node-local for speed
CACHE_DIR="$SLURM_TMPDIR/task2_maskcache_1024"
mkdir -p "$CACHE_DIR"

python -u task2_segmentation/train.py \
  --data-root "$DATA_ROOT" \
  --ckpt-dir "$CKPT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --img-size 1024 \
  --encoder timm-efficientnet-b3 \
  --encoder-weights imagenet \
  --epochs 50 --batch 32 \
  --lr 3e-4 --wd 1e-2 \
  --workers 6 \
  --panel-min-ratio 0.05

echo "Best weights: $CKPT_DIR/best_task2_by_final50_95.pth"
