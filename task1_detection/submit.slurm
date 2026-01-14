#!/bin/bash -l
#SBATCH --job-name=t1_submit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=hopper
#SBATCH --qos=besteffort

set -euo pipefail
mkdir -p logs

source /home/users/whuang/code/spark-zeta25/venv/bin/activate
source task1_detection/stage_data.sh

: "${WEIGHTS:?Submit with: sbatch --export=ALL,WEIGHTS=/path/to/best.pth task1_detection/slurm_submit.sh}"

OUTDIR="$SCRATCH/submissions/task1/$SLURM_JOB_ID"
mkdir -p "$OUTDIR"

python -u task1_detection/submit.py \
  --weights "$WEIGHTS" \
  --test-dir "$DATA_ROOT/test_images/images" \
  --out-csv "$OUTDIR/detection.csv"

echo "detection.csv -> $OUTDIR/detection.csv"
