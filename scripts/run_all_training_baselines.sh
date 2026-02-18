#!/usr/bin/env bash
# run_all_training_baselines.sh
# Launch all training-based baseline experiments for the rebuttal.
# Usage: bash scripts/run_all_training_baselines.sh
#
# Estimated time:
#   RTX 2000 Ada (16GB): ~20 hours
#   H100 (80GB):         ~4-5 hours
#   A100 (40GB):         ~6-8 hours

set -uo pipefail  # No -e: let experiments continue even if one fails
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
FAIL_COUNT=0

mkdir -p "$OUTPUT_DIR"

# --- Run experiments sequentially ---
# Each experiment saves its own JSON. If one fails, the rest still run.

run_experiment() {
    local label="$1"
    local dataset="$2"
    local noise_type="$3"
    local logfile="${OUTPUT_DIR}/log_${dataset}_${noise_type}_training.txt"

    echo ""
    echo ">>> ${label}"
    echo "    Log: ${logfile}"
    if python scripts/run_training_baselines.py \
        --dataset "$dataset" \
        --noise_type "$noise_type" \
        --n_train $N_TRAIN \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seeds $SEEDS \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$logfile"; then
        echo "    DONE: ${label}"
    else
        echo "    FAILED: ${label} (see ${logfile})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

run_experiment "[1/4] SST-2 Uniform Noise"           sst2     uniform
run_experiment "[2/4] SST-2 Artifact-Aligned Noise"   sst2     artifact_aligned
run_experiment "[3/4] MultiNLI Uniform Noise"         multinli uniform
run_experiment "[4/4] MultiNLI Artifact-Aligned Noise" multinli artifact_aligned

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "End time: $(date)"
echo "Failures: $FAIL_COUNT / 4"
echo "Results in: $OUTPUT_DIR"
ls -lh "${OUTPUT_DIR}"/*training_baselines*.json 2>/dev/null || echo "(no result files found)"
echo "============================================================"
