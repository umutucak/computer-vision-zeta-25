#!/bin/bash
set -euo pipefail

: "${SCRATCH:?SCRATCH not set}"
: "${SLURM_TMPDIR:?SLURM_TMPDIR not set}"

PROJECT_DIR="$SCRATCH/dataset"
TRAIN_ZIP="$PROJECT_DIR/spark-2024-train-val.zip"
TEST_ZIP="$PROJECT_DIR/spark-2024-detection-test.zip"
LABEL_ZIP="$PROJECT_DIR/ground_truth_labels.zip"

export DATA_ROOT="$SLURM_TMPDIR/data"
mkdir -p "$DATA_ROOT" "$DATA_ROOT/test_images"

export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

unzip -q "$TRAIN_ZIP" -d "$DATA_ROOT"
unzip -q "$TEST_ZIP"  -d "$DATA_ROOT/test_images"
unzip -o "$LABEL_ZIP" -d "$DATA_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
