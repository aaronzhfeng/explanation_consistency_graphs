#!/usr/bin/env python
"""
Smoketest: Time GPU training pipeline to estimate full-scale cost.

Runs RoBERTa fine-tuning + 5-fold CV (Cleanlab) + training dynamics
on a small subset, then extrapolates to full scale.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from ecg.data import load_sst2, create_noisy_dataset, NoiseConfig, ArtifactConfig
from ecg.train_classifier import (
    train_classifier,
    TrainingConfig,
    cross_validate_predictions,
)

SMOKE_N = 1000
SMOKE_EPOCHS = 1
FULL_N = 25000
FULL_EPOCHS = 3
FULL_SEEDS = 5
FULL_CONFIGS = 2  # uniform + artifact_aligned


def fmt(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # --- Load data ---
    print(f"Loading SST-2 (n={SMOKE_N})...")
    noise_cfg = NoiseConfig(noise_type="artifact_aligned", noise_rate=0.10, seed=42)
    artifact_cfg = ArtifactConfig()
    noisy_data = create_noisy_dataset(
        n_train=SMOKE_N, noise_config=noise_cfg, artifact_config=artifact_cfg, seed=42
    )
    sst2 = load_sst2()
    val_dataset = sst2["validation"]
    print(f"  Train: {len(noisy_data.dataset)}, Val: {len(val_dataset)}")

    train_config = TrainingConfig(
        epochs=SMOKE_EPOCHS,
        batch_size=64,
        output_dir="outputs/smoketest_ckpt",
        seed=42,
        compute_training_dynamics=True,
        save_per_epoch=False,
    )

    # --- Step A: Single training run ---
    print(f"\n{'='*60}")
    print(f"STEP A: Single RoBERTa fine-tuning (n={SMOKE_N}, {SMOKE_EPOCHS} epoch)")
    print(f"{'='*60}")
    t0 = time.time()
    model, dynamics, results = train_classifier(
        train_dataset=noisy_data.dataset,
        val_dataset=val_dataset,
        config=train_config,
        return_dynamics=True,
    )
    t_train = time.time() - t0
    print(f"\n  Time: {fmt(t_train)}")
    print(f"  Val accuracy: {results.get('val_accuracy', 'N/A')}")
    if dynamics:
        print(f"  AUM range: [{dynamics.aum_scores.min():.3f}, {dynamics.aum_scores.max():.3f}]")

    # Free memory
    del model
    torch.cuda.empty_cache()

    # --- Step B: 5-fold CV (for Cleanlab) ---
    print(f"\n{'='*60}")
    print(f"STEP B: 5-fold CV for Cleanlab (n={SMOKE_N}, {SMOKE_EPOCHS} epoch/fold)")
    print(f"{'='*60}")
    t0 = time.time()
    cv_probs = cross_validate_predictions(
        dataset=noisy_data.dataset,
        config=train_config,
        n_folds=5,
        seed=42,
    )
    t_cv = time.time() - t0
    print(f"\n  Time: {fmt(t_cv)}")
    print(f"  OOS probs shape: {cv_probs.shape}")

    torch.cuda.empty_cache()

    # --- Step C: Cleanlab + baselines ---
    print(f"\n{'='*60}")
    print(f"STEP C: Compute baselines (Cleanlab, Loss, Margin, etc.)")
    print(f"{'='*60}")
    t0 = time.time()
    from ecg.baselines import compute_all_baselines
    baselines = compute_all_baselines(
        labels=noisy_data.noisy_labels,
        pred_probs=cv_probs,
        k=15,
        seed=42,
    )
    t_baselines = time.time() - t0
    print(f"  Time: {fmt(t_baselines)}")
    print(f"  Cleanlab: {'OK' if baselines.cleanlab is not None else 'FAILED'}")
    print(f"  Loss: {'OK' if baselines.loss is not None else 'FAILED'}")

    # --- Extrapolation ---
    print(f"\n{'='*60}")
    print("EXTRAPOLATION TO FULL SCALE")
    print(f"{'='*60}")

    # Training scales ~linearly with n_train and epochs
    scale_factor = (FULL_N / SMOKE_N) * (FULL_EPOCHS / SMOKE_EPOCHS)
    est_train_1run = t_train * scale_factor
    est_cv_1run = t_cv * scale_factor
    est_per_config = (est_train_1run + est_cv_1run) * FULL_SEEDS
    est_total = est_per_config * FULL_CONFIGS

    print(f"\nSmoke test results (n={SMOKE_N}, {SMOKE_EPOCHS} epoch):")
    print(f"  Single training:  {fmt(t_train)}")
    print(f"  5-fold CV:        {fmt(t_cv)}")
    print(f"  Baselines:        {fmt(t_baselines)}")

    print(f"\nFull-scale estimates (n={FULL_N}, {FULL_EPOCHS} epochs):")
    print(f"  Single training:       {fmt(est_train_1run)}")
    print(f"  5-fold CV:             {fmt(est_cv_1run)}")
    print(f"  Per seed (train+CV):   {fmt(est_train_1run + est_cv_1run)}")
    print(f"  Per config (×{FULL_SEEDS} seeds): {fmt(est_per_config)}")
    print(f"  TOTAL (×{FULL_CONFIGS} configs):  {fmt(est_total)}")

    # GPU memory
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak GPU memory: {peak_mem:.2f} GB")


if __name__ == "__main__":
    main()
