#!/usr/bin/env python3
"""
Run ECG (input) — neighborhood surprise on INPUT embeddings.

This applies the same neighborhood surprise algorithm as ECG, but on
sentence embeddings of the raw input text instead of explanation embeddings.

Runs: 5 seeds × 2 datasets × 2 noise types = 20 configurations.
CPU-only (sentence embedding + FAISS kNN + arithmetic).
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sentence_transformers import SentenceTransformer

from ecg.data import (
    NoiseConfig,
    create_noisy_dataset_generic,
    DATASET_LABEL_MAPS,
)
from ecg.embed_graph import (
    build_explanation_graph,
    compute_neighborhood_surprise,
    normalize_embeddings,
)
from ecg.eval import compute_auroc_auprc, compute_precision_recall_at_k


# Configuration
SEEDS = [42, 123, 456, 789, 1024]
DATASETS = ["sst2", "multinli"]
NOISE_TYPES = ["uniform", "artifact_aligned"]
N_TRAIN = 25000
NOISE_RATE = 0.10
K_NEIGHBORS = 15
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'results')


def run_one(dataset_name, noise_type, seed, model):
    """Run ECG(input) for one seed/dataset/noise configuration."""
    n_classes = 3 if dataset_name == "multinli" else 2

    noise_config = NoiseConfig(
        noise_type=noise_type,
        noise_rate=NOISE_RATE,
        seed=seed,
    )

    noisy_data = create_noisy_dataset_generic(
        dataset_name=dataset_name,
        n_train=N_TRAIN,
        noise_config=noise_config,
        seed=seed,
    )

    sentences = noisy_data.dataset["sentence"]
    noisy_labels = np.array(noisy_data.dataset["label"])
    is_noisy = noisy_data.is_noisy

    # Embed raw input texts
    embeddings = model.encode(
        sentences,
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Build kNN graph with reliability = 1 (no explanation reliability info)
    reliability = np.ones(len(sentences))
    graph = build_explanation_graph(
        embeddings=embeddings,
        reliability=reliability,
        k=K_NEIGHBORS,
        temperature=0.07,
        similarity_threshold=0.35,
        use_mutual_knn=False,
        compute_outliers=True,
    )

    # Compute neighborhood surprise
    surprise = compute_neighborhood_surprise(
        graph, noisy_labels, n_classes=n_classes, smoothing_epsilon=1e-3
    )

    # Evaluate
    auroc, auprc, tnr95 = compute_auroc_auprc(is_noisy, surprise)
    pak = compute_precision_recall_at_k(is_noisy, surprise, k_values=[0.01, 0.05, 0.10])

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tnr_at_95": float(tnr95),
        "precision_at_k": {str(k): float(v) for k, v in pak[0].items()},
        "recall_at_k": {str(k): float(v) for k, v in pak[1].items()},
    }


def main():
    print("Loading sentence transformer model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_results = {}
    start = time.time()

    for dataset_name in DATASETS:
        for noise_type in NOISE_TYPES:
            key = f"{dataset_name}_{noise_type}"
            print(f"\n{'='*60}")
            print(f"  {key}")
            print(f"{'='*60}")

            seed_results = []
            for seed in SEEDS:
                t0 = time.time()
                print(f"  seed={seed} ...", end=" ", flush=True)
                result = run_one(dataset_name, noise_type, seed, model)
                elapsed = time.time() - t0
                print(f"AUROC={result['auroc']:.4f}  ({elapsed:.1f}s)")
                result["seed"] = seed
                seed_results.append(result)

            # Aggregate
            aurocs = [r["auroc"] for r in seed_results]
            auprcs = [r["auprc"] for r in seed_results]
            mean_auroc = float(np.mean(aurocs))
            std_auroc = float(np.std(aurocs))
            mean_auprc = float(np.mean(auprcs))
            std_auprc = float(np.std(auprcs))

            print(f"  => AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
            print(f"  => AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")

            all_results[key] = {
                "dataset": dataset_name,
                "noise_type": noise_type,
                "n_train": N_TRAIN,
                "noise_rate": NOISE_RATE,
                "seeds": SEEDS,
                "per_seed": seed_results,
                "aggregated": {
                    "auroc_mean": mean_auroc,
                    "auroc_std": std_auroc,
                    "auprc_mean": mean_auprc,
                    "auprc_std": std_auprc,
                },
            }

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*60}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_ecg_input_neighborhood_surprise.json")

    all_results["metadata"] = {
        "method": "ECG (input) - neighborhood surprise on input embeddings",
        "embedding_model": EMBEDDING_MODEL,
        "k": K_NEIGHBORS,
        "temperature": 0.07,
        "similarity_threshold": 0.35,
        "timestamp": timestamp,
        "total_time_seconds": total_time,
    }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: ECG (input) — Neighborhood Surprise")
    print(f"{'='*60}")
    print(f"{'Setting':<25} {'AUROC':>15} {'AUPRC':>15}")
    print("-" * 55)
    for key in ["sst2_uniform", "sst2_artifact_aligned", "multinli_uniform", "multinli_artifact_aligned"]:
        if key in all_results:
            agg = all_results[key]["aggregated"]
            print(f"{key:<25} {agg['auroc_mean']:.3f} ± {agg['auroc_std']:.3f}   {agg['auprc_mean']:.3f} ± {agg['auprc_std']:.3f}")


if __name__ == "__main__":
    main()
