#!/usr/bin/env bash
# run_all_training_baselines.sh
# Launch all training-based baseline experiments for the rebuttal.
# Usage: bash scripts/run_all_training_baselines.sh
#
# Estimated time:
#   RTX 2000 Ada (16GB): ~20 hours
#   H100 (80GB):         ~4-5 hours
#   A100 (40GB):         ~6-8 hours

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Training Baselines - Full Rebuttal Run"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# --- Install dependencies (idempotent) ---
echo "Checking dependencies..."
pip install -q cleanlab aum 'accelerate>=1.1.0' 2>/dev/null || true
echo "Dependencies OK."
echo ""

# --- Configs ---
SEEDS="42 123 456 789 1024"
N_TRAIN=25000
EPOCHS=3
BATCH_SIZE=64
OUTPUT_DIR="outputs/results"

mkdir -p "$OUTPUT_DIR"

# --- Run experiments sequentially ---
# Each one logs to both stdout and a log file.

echo ">>> [1/4] SST-2 Uniform Noise"
python scripts/run_training_baselines.py \
    --dataset sst2 \
    --noise_type uniform \
    --n_train $N_TRAIN \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seeds $SEEDS \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}/log_sst2_uniform_training.txt"

echo ""
echo ">>> [2/4] SST-2 Artifact-Aligned Noise"
python scripts/run_training_baselines.py \
    --dataset sst2 \
    --noise_type artifact_aligned \
    --n_train $N_TRAIN \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seeds $SEEDS \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}/log_sst2_artifact_training.txt"

echo ""
echo ">>> [3/4] MultiNLI Uniform Noise"
python scripts/run_training_baselines.py \
    --dataset multinli \
    --noise_type uniform \
    --n_train $N_TRAIN \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seeds $SEEDS \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}/log_multinli_uniform_training.txt"

echo ""
echo ">>> [4/4] MultiNLI Artifact-Aligned Noise"
python scripts/run_training_baselines.py \
    --dataset multinli \
    --noise_type artifact_aligned \
    --n_train $N_TRAIN \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seeds $SEEDS \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}/log_multinli_artifact_training.txt"

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "End time: $(date)"
echo "Results in: $OUTPUT_DIR"
echo "============================================================"
