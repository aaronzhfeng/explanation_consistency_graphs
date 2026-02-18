#!/usr/bin/env bash
# run_rebuttal.sh — Launch all rebuttal experiments sequentially.
# Usage: nohup bash scripts/run_rebuttal.sh > outputs/rebuttal_log.txt 2>&1 &

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SEEDS="42 123 456 789 1024"
N_TRAIN=25000
NOISE_RATE=0.10
MODEL="qwen/qwen3-8b"
MAX_CONCURRENCY=50

echo "============================================================"
echo "ECG Rebuttal Experiments"
echo "Started: $(date)"
echo "Seeds: $SEEDS"
echo "N_train: $N_TRAIN, Noise rate: $NOISE_RATE"
echo "============================================================"

# --- Experiment 1: SST-2 uniform noise, 5 seeds ---
echo ""
echo ">>>>> [1/3] SST-2 uniform noise, 5 seeds <<<<<"
echo "Started: $(date)"
python scripts/run_ecg_api.py \
    --dataset sst2 \
    --n_train $N_TRAIN \
    --noise_type uniform \
    --noise_rate $NOISE_RATE \
    --seeds $SEEDS \
    --model $MODEL \
    --max_concurrency $MAX_CONCURRENCY
echo "[1/3] Done: $(date)"

# --- Experiment 2: SST-2 artifact-aligned noise, 5 seeds ---
echo ""
echo ">>>>> [2/3] SST-2 artifact-aligned noise, 5 seeds <<<<<"
echo "Started: $(date)"
python scripts/run_ecg_api.py \
    --dataset sst2 \
    --n_train $N_TRAIN \
    --noise_type artifact_aligned \
    --noise_rate $NOISE_RATE \
    --seeds $SEEDS \
    --model $MODEL \
    --max_concurrency $MAX_CONCURRENCY
echo "[2/3] Done: $(date)"

# --- Experiment 3: MultiNLI uniform noise, 5 seeds ---
echo ""
echo ">>>>> [3/3] MultiNLI uniform noise, 5 seeds <<<<<"
echo "Started: $(date)"
python scripts/run_ecg_api.py \
    --dataset multinli \
    --n_train $N_TRAIN \
    --noise_type uniform \
    --noise_rate $NOISE_RATE \
    --seeds $SEEDS \
    --model $MODEL \
    --max_concurrency $MAX_CONCURRENCY
echo "[3/3] Done: $(date)"

echo ""
echo "============================================================"
echo "All rebuttal experiments complete!"
echo "Finished: $(date)"
echo "Results in: outputs/results/"
echo "Cache in:   outputs/cache/"
echo "============================================================"
