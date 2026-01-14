#!/bin/bash -l
#SBATCH --job-name=t1_train
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
source task1_detection/stage_data.sh

CKPT_DIR="$SCRATCH/ckpts/task1/$SLURM_JOB_ID"
mkdir -p "$CKPT_DIR"

python -u task1_detection/train.py \
  --data-root "$DATA_ROOT" \
  --ckpt-dir "$CKPT_DIR" \
  --epochs 40 --batch 64 --workers 6 \
  --lr 3e-5 --wd 1e-4 \
  --pretrained \
  --score-thr 0.0

echo "Best weights should be in: $CKPT_DIR/best_by_leaderboard_task1.pth"
