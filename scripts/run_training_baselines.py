#!/usr/bin/env python
"""
run_training_baselines.py: Run training-based baselines for rebuttal.

For each (dataset, noise_type, seed):
  1. Load data + inject noise
  2. Train RoBERTa classifier (3 epochs)
  3. 5-fold CV for Cleanlab out-of-sample predictions
  4. Compute training dynamics (AUM/loss/margin)
  5. Compute all training-based baselines
  6. Evaluate detection metrics (AUROC, AUPRC, TNR@95)
  7. Save results in same JSON format as API experiments

Usage:
    python scripts/run_training_baselines.py --dataset sst2 --noise_type uniform
    python scripts/run_training_baselines.py --dataset multinli --noise_type artifact_aligned
    python scripts/run_training_baselines.py --dataset sst2 --noise_type uniform --n_train 1000 --epochs 1  # Quick test
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ecg.data import (
    NoiseConfig,
    create_noisy_dataset,
    create_noisy_dataset_generic,
    load_sst2,
    load_multinli,
    DATASET_LABEL_MAPS,
)
from ecg.train_classifier import (
    TrainingConfig,
    train_classifier,
    cross_validate_predictions,
    get_cls_embeddings,
    tokenize_dataset,
)
from ecg.baselines import (
    compute_all_baselines,
    cleanlab_scores,
    loss_scores,
    margin_scores,
    entropy_scores,
    knn_disagreement_scores,
    nrg_scores,
    random_scores,
)
from ecg.eval import compute_detection_metrics

from transformers import AutoTokenizer


SEEDS = [42, 123, 456, 789, 1024]
OUTPUT_DIR = "outputs/results"


def get_num_labels(dataset_name: str) -> int:
    label_map = DATASET_LABEL_MAPS.get(dataset_name, {})
    if label_map:
        return len(label_map)
    return 2


def load_val_dataset(dataset_name: str):
    """Load validation set for the given dataset."""
    if dataset_name == "sst2":
        sst2 = load_sst2()
        return sst2["validation"]
    elif dataset_name == "multinli":
        mnli = load_multinli()
        return mnli["validation"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_single_seed(
    dataset_name: str,
    noise_type: str,
    noise_rate: float,
    seed: int,
    n_train: int,
    epochs: int,
    batch_size: int,
    val_dataset,
) -> Dict:
    """Run training baselines for a single seed. Returns results dict."""

    num_labels = get_num_labels(dataset_name)
    print(f"\n{'='*60}")
    print(f"SEED {seed} | {dataset_name} | {noise_type} | n={n_train} | {num_labels}-class")
    print(f"{'='*60}")

    t_start = time.time()

    # --- Load data ---
    noise_cfg = NoiseConfig(noise_type=noise_type, noise_rate=noise_rate, seed=seed)
    if dataset_name == "sst2":
        noisy_data = create_noisy_dataset(
            n_train=n_train, noise_config=noise_cfg, seed=seed,
        )
    else:
        noisy_data = create_noisy_dataset_generic(
            dataset_name=dataset_name,
            n_train=n_train,
            noise_config=noise_cfg,
            seed=seed,
        )

    n_noisy = noisy_data.is_noisy.sum()
    print(f"  Data loaded: {len(noisy_data.dataset)} train, {n_noisy} noisy ({n_noisy/len(noisy_data.dataset):.1%})")

    # --- Training config ---
    train_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        output_dir=f"outputs/training_baselines_ckpt/{dataset_name}_{noise_type}_seed{seed}",
        seed=seed,
        compute_training_dynamics=True,
        save_per_epoch=False,
        num_labels=num_labels,
    )

    # --- Step 1: Train classifier ---
    print(f"\n  [1/4] Training RoBERTa ({epochs} epochs)...")
    t0 = time.time()
    model, dynamics, results = train_classifier(
        train_dataset=noisy_data.dataset,
        val_dataset=val_dataset,
        config=train_config,
        return_dynamics=True,
    )
    t_train = time.time() - t0
    print(f"        Done in {t_train:.1f}s. Val acc: {results.get('val_accuracy', 'N/A')}")

    # --- Step 2: 5-fold CV for Cleanlab ---
    print(f"  [2/4] 5-fold CV for Cleanlab...")
    t0 = time.time()
    cv_probs = cross_validate_predictions(
        dataset=noisy_data.dataset,
        config=train_config,
        n_folds=5,
        seed=seed,
    )
    t_cv = time.time() - t0
    print(f"        Done in {t_cv:.1f}s.")

    # --- Step 3: Get classifier embeddings ---
    print(f"  [3/4] Extracting CLS embeddings...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    train_tokenized = tokenize_dataset(
        noisy_data.dataset, tokenizer, train_config.max_length
    )
    cls_embeddings = get_cls_embeddings(
        model, train_tokenized, tokenizer, train_config.batch_size
    )
    t_embed = time.time() - t0
    print(f"        Done in {t_embed:.1f}s. Shape: {cls_embeddings.shape}")

    # --- Step 4: Compute baselines + evaluate ---
    print(f"  [4/4] Computing baselines and evaluating...")
    t0 = time.time()

    ground_truth = noisy_data.is_noisy
    noisy_labels = noisy_data.noisy_labels
    method_results = {}

    # Cleanlab
    try:
        cl_scores = cleanlab_scores(noisy_labels, cv_probs)
        m = compute_detection_metrics(ground_truth, cl_scores)
        method_results["Cleanlab"] = {"auroc": m.auroc, "auprc": m.auprc, "tnr_at_95": m.tnr_at_95}
        print(f"        Cleanlab:       AUROC={m.auroc:.4f}")
    except Exception as e:
        print(f"        Cleanlab FAILED: {e}")

    # High-Loss
    hl_scores = loss_scores(noisy_labels, cv_probs)
    m = compute_detection_metrics(ground_truth, hl_scores)
    method_results["High-Loss"] = {"auroc": m.auroc, "auprc": m.auprc, "tnr_at_95": m.tnr_at_95}
    print(f"        High-Loss:     AUROC={m.auroc:.4f}")

    # Margin
    mg_scores = margin_scores(noisy_labels, cv_probs)
    m = compute_detection_metrics(ground_truth, mg_scores)
    method_results["Margin"] = {"auroc": m.auroc, "auprc": m.auprc, "tnr_at_95": m.tnr_at_95}
    print(f"        Margin:        AUROC={m.auroc:.4f}")

    # Entropy
    ent_scores = entropy_scores(cv_probs)
    m = compute_detection_metrics(ground_truth, ent_scores)
    method_results["Entropy"] = {"auroc": m.auroc, "auprc": m.auprc, "tnr_at_95": m.tnr_at_95}
    print(f"        Entropy:       AUROC={m.auroc:.4f}")

    # AUM (from training dynamics)
    if dynamics is not None:
        aum_scores = -dynamics.aum_scores  # Negate: lower AUM = more suspicious
        m = compute_detection_metrics(ground_truth, aum_scores)
        method_results["AUM"] = {"auroc": m.auroc, "auprc": m.auprc, "tnr_at_95": m.tnr_at_95}
        print(f"        AUM:           AUROC={m.auroc:.4f}")

    # Classifier kNN
    cknn_scores = knn_disagreement_scores(cls_embeddings, noisy_labels, k=15)
    m = compute_detection_metrics(ground_truth, cknn_scores)
    method_results["Classifier kNN"] = {"auroc": m.auroc, "auprc": m.auprc, "tnr_at_95": m.tnr_at_95}
    print(f"        Classifier kNN: AUROC={m.auroc:.4f}")

    # NRG
    try:
        nrg_sc = nrg_scores(cls_embeddings, cv_probs, noisy_labels)
        m = compute_detection_metrics(ground_truth, nrg_sc)
        method_results["NRG"] = {"auroc": m.auroc, "auprc": m.auprc, "tnr_at_95": m.tnr_at_95}
        print(f"        NRG:           AUROC={m.auroc:.4f}")
    except Exception as e:
        print(f"        NRG FAILED: {e}")

    # Random
    rnd_scores = random_scores(len(noisy_labels), seed=seed)
    m = compute_detection_metrics(ground_truth, rnd_scores)
    method_results["Random"] = {"auroc": m.auroc, "auprc": m.auprc, "tnr_at_95": m.tnr_at_95}

    t_eval = time.time() - t0
    t_total = time.time() - t_start
    print(f"        Eval done in {t_eval:.1f}s.")
    print(f"  SEED {seed} TOTAL: {t_total:.1f}s ({t_total/60:.1f} min)")

    # Cleanup
    del model, dynamics, cv_probs, cls_embeddings
    torch.cuda.empty_cache()
    gc.collect()

    return method_results


def aggregate_results(results_by_seed: Dict) -> Dict:
    """Aggregate per-seed results into mean ± std."""
    # Collect all method names
    all_methods = set()
    for seed_results in results_by_seed.values():
        all_methods.update(seed_results.keys())

    aggregated = {}
    for method in sorted(all_methods):
        method_metrics = {}
        for metric in ["auroc", "auprc", "tnr_at_95"]:
            values = []
            for seed_results in results_by_seed.values():
                if method in seed_results and metric in seed_results[method]:
                    values.append(seed_results[method][metric])
            if values:
                method_metrics[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n": len(values),
                }
        aggregated[method] = method_metrics
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run training-based baselines")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "multinli"])
    parser.add_argument("--noise_type", type=str, default="uniform",
                        choices=["uniform", "artifact_aligned"])
    parser.add_argument("--noise_rate", type=float, default=0.10)
    parser.add_argument("--n_train", type=int, default=25000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("Training Baselines Pipeline")
    print("="*60)
    print(f"Dataset:    {args.dataset}")
    print(f"Noise type: {args.noise_type}")
    print(f"Noise rate: {args.noise_rate}")
    print(f"N train:    {args.n_train}")
    print(f"Epochs:     {args.epochs}")
    print(f"Seeds:      {args.seeds}")
    print(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name()}")

    # Load validation set once (shared across seeds)
    print("\nLoading validation set...")
    val_dataset = load_val_dataset(args.dataset)
    print(f"  Val size: {len(val_dataset)}")

    # Run each seed
    results_by_seed = {}
    t_total_start = time.time()

    for i, seed in enumerate(args.seeds):
        print(f"\n{'#'*60}")
        print(f"# Seed {i+1}/{len(args.seeds)}: {seed}")
        print(f"{'#'*60}")

        seed_results = run_single_seed(
            dataset_name=args.dataset,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            seed=seed,
            n_train=args.n_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_dataset=val_dataset,
        )
        results_by_seed[str(seed)] = seed_results

    t_total = time.time() - t_total_start

    # Aggregate
    aggregated = aggregate_results(results_by_seed)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {t_total/60:.1f} min")
    print(f"\n{'Method':<20} {'AUROC':>14} {'AUPRC':>14} {'TNR@95':>14}")
    print("-" * 64)
    for method, metrics in sorted(aggregated.items(), key=lambda x: -x[1].get("auroc", {}).get("mean", 0)):
        auroc = metrics.get("auroc", {})
        auprc = metrics.get("auprc", {})
        tnr = metrics.get("tnr_at_95", {})
        print(f"{method:<20} {auroc.get('mean',0):.3f} ± {auroc.get('std',0):.3f}  "
              f"{auprc.get('mean',0):.3f} ± {auprc.get('std',0):.3f}  "
              f"{tnr.get('mean',0):.3f} ± {tnr.get('std',0):.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "experiment": f"{args.dataset}_{args.noise_type}_training_baselines",
        "timestamp": timestamp,
        "config": {
            "dataset": args.dataset,
            "n_train": args.n_train,
            "noise_type": args.noise_type,
            "noise_rate": args.noise_rate,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "model": "roberta-base",
            "num_labels": get_num_labels(args.dataset),
        },
        "results_by_seed": results_by_seed,
        "aggregated": aggregated,
        "total_time_seconds": t_total,
    }

    output_path = os.path.join(
        args.output_dir,
        f"{timestamp}_{args.dataset}_{args.noise_type}_training_baselines.json"
    )
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
