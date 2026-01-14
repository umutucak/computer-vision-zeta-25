#!/bin/bash -l
#SBATCH --job-name=t2_submit
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
source task2_segmentation/stage_data.sh

: "${WEIGHTS:?Submit with: sbatch --export=ALL,WEIGHTS=/path/to/best_task2_by_final50_95.pth task2_segmentation/slurm_submit.sh}"

OUTDIR="$SCRATCH/submissions/task2/$SLURM_JOB_ID"
NPZDIR="$OUTDIR/submission_folder"
mkdir -p "$NPZDIR"

python -u task2_segmentation/submit.py \
  --test-dir "$DATA_ROOT/test_seg_images/images" \
  --ckpt "$WEIGHTS" \
  --out-dir "$NPZDIR" \
  --img-size 1024 \
  --encoder timm-efficientnet-b3 \
  --num-classes 3 \
  --tta

echo "NPZ folder ready: $NPZDIR"
echo "Zip it for upload:"
echo "  cd $OUTDIR && zip -r task2_submission.zip submission_folder"
