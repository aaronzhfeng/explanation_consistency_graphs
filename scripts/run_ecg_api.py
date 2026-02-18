#!/usr/bin/env python
"""
run_ecg_api.py: Unified experiment runner using API-based LLM inference.

Supports SST-2, MultiNLI, and AlleNoise datasets.
Runs the core ECG pipeline (explanation kNN) without GPU requirements for LLM.

Usage:
    # SST-2 (default)
    python scripts/run_ecg_api.py --dataset sst2 --noise_type artifact_aligned --noise_rate 0.10

    # MultiNLI
    python scripts/run_ecg_api.py --dataset multinli --noise_type uniform --noise_rate 0.10

    # Multi-seed
    python scripts/run_ecg_api.py --dataset sst2 --seeds 42 123 456

    # Small smoke test
    python scripts/run_ecg_api.py --dataset sst2 --n_train 100 --noise_rate 0.10
"""

import gc
import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ecg.data import (
    NoiseConfig, create_noisy_dataset, create_noisy_dataset_generic,
    DATASET_LABEL_MAPS,
)
from ecg.explain_llm import (
    APIExplanationGenerator, get_task_config, make_task_config_for_labels,
    TaskConfig, TASK_CONFIGS,
    generate_batch_with_stability, explanations_to_embeddings,
    get_reliability_scores, get_llm_predictions, format_prompt_for_task,
)
from ecg.embed_graph import build_explanation_graph
from ecg.signals import compute_neighborhood_surprise
from ecg.baselines import knn_disagreement_scores
from ecg.eval import compute_detection_metrics, print_detection_summary
from ecg.utils import TimingRecord, aggregate_results, print_aggregated_results


API_KEY_FILE = str(Path(__file__).parent.parent / "openroutera_api.txt")
OUTPUT_DIR = str(Path(__file__).parent.parent / "outputs" / "results")
CACHE_DIR = str(Path(__file__).parent.parent / "outputs" / "cache")


def get_task_config_for_dataset(dataset_name: str) -> TaskConfig:
    """Get or create task config for a dataset."""
    if dataset_name in TASK_CONFIGS:
        return TASK_CONFIGS[dataset_name]
    # For AlleNoise or unknown datasets, we'd create dynamically
    # after loading data to know the label names
    return None


def run_single_experiment(
    dataset_name: str,
    n_train: int,
    noise_type: str,
    noise_rate: float,
    seed: int,
    model_name: str,
    max_concurrency: int,
    n_stability_samples: int,
    k_neighbors: int,
    task_config: TaskConfig = None,
    cache_base_dir: str = None,
) -> dict:
    """
    Run a single ECG experiment.

    Returns dict with detection metrics for each method.
    """
    timer = TimingRecord()

    # ---- Step 1: Load data ----
    with timer.measure("load_data"):
        noise_config = NoiseConfig(
            noise_type=noise_type,
            noise_rate=noise_rate,
            seed=seed,
        )

        if dataset_name == "sst2":
            noisy_data = create_noisy_dataset(
                n_train=n_train,
                noise_config=noise_config,
                seed=seed,
            )
        else:
            noisy_data = create_noisy_dataset_generic(
                dataset_name=dataset_name,
                n_train=n_train,
                noise_config=noise_config,
                seed=seed,
            )

    n = len(noisy_data.dataset)
    n_noisy = noisy_data.is_noisy.sum()
    n_classes = len(set(noisy_data.clean_labels.tolist()))
    print(f"\n  Dataset: {dataset_name}, N={n}, noisy={n_noisy} ({n_noisy/n*100:.1f}%), classes={n_classes}")

    # Resolve task config
    if task_config is None:
        task_config = get_task_config_for_dataset(dataset_name)
        if task_config is None:
            # Create from data
            label_map = DATASET_LABEL_MAPS.get(dataset_name)
            if label_map:
                label_names = [label_map[i] for i in sorted(label_map.keys())]
            else:
                label_names = [f"CLASS_{i}" for i in range(n_classes)]
            task_config = make_task_config_for_labels(dataset_name, label_names)

    # Construct per-seed cache directory (includes noise_type because
    # artifact-aligned noise changes the text sent to the LLM)
    cache_dir = None
    if cache_base_dir is not None:
        cache_dir = os.path.join(
            cache_base_dir,
            f"{dataset_name}_{n_train}_{noise_type}_seed{seed}",
        )

    # ---- Step 2: Generate explanations ----
    with timer.measure("generate_explanations"):
        gen = APIExplanationGenerator.from_key_file(
            API_KEY_FILE,
            model_name=model_name,
            task_config=task_config,
            max_concurrency=max_concurrency,
        )

        # Get texts for prompting — use dicts for multi-field tasks (NLI)
        if task_config and len(task_config.text_fields) > 1:
            texts = [
                {f: row[f] for f in task_config.text_fields + ["sentence"] if f in row}
                for row in noisy_data.dataset
            ]
        else:
            texts = noisy_data.dataset["sentence"]

        explanations_ws = generate_batch_with_stability(
            gen, texts,
            n_samples=n_stability_samples,
            sample_temperature=0.7,
            show_progress=True,
            cache_dir=cache_dir,
        )

        gen.print_stats()

    primary_explanations = [e.primary for e in explanations_ws]
    reliability_scores = get_reliability_scores(explanations_ws)
    llm_labels, llm_confidence = get_llm_predictions(primary_explanations, task_config)

    # ---- Step 3: Build explanation graph ----
    with timer.measure("embed_and_graph"):
        embeddings = explanations_to_embeddings(primary_explanations)
        graph = build_explanation_graph(
            embeddings=embeddings,
            reliability=reliability_scores,
            k=k_neighbors,
            temperature=0.07,
            similarity_threshold=0.0,
        )

    # ---- Step 4: Compute signals ----
    with timer.measure("compute_signals"):
        # Explanation kNN (our method)
        s_nbr, c_nbr = compute_neighborhood_surprise(
            graph, noisy_data.noisy_labels, n_classes=n_classes
        )

        # Input kNN baseline (using same embeddings but on raw text)
        from sentence_transformers import SentenceTransformer
        input_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        input_embeddings = input_model.encode(
            noisy_data.dataset["sentence"],
            batch_size=64, show_progress_bar=True, normalize_embeddings=True,
        )
        input_knn_scores = knn_disagreement_scores(
            input_embeddings, noisy_data.noisy_labels, k=k_neighbors
        )

        # LLM mismatch baseline
        from ecg.baselines import llm_mismatch_scores
        mismatch_scores = llm_mismatch_scores(
            noisy_data.noisy_labels, llm_labels, llm_confidence
        )

    # ---- Step 5: Evaluate ----
    with timer.measure("evaluate"):
        ground_truth = noisy_data.is_noisy

        methods = {
            "Explanation kNN": s_nbr,
            "Input kNN": input_knn_scores,
            "LLM Mismatch": mismatch_scores,
        }

        results = {}
        for method_name, scores in methods.items():
            metrics = compute_detection_metrics(ground_truth, scores)
            results[method_name] = {
                "auroc": metrics.auroc,
                "auprc": metrics.auprc,
                "tnr_at_95": metrics.tnr_at_95,
            }

        print_detection_summary(
            {name: compute_detection_metrics(ground_truth, scores) for name, scores in methods.items()}
        )

    timer.print_summary()

    return {
        "results": results,
        "timing": timer.to_dict(),
        "config": {
            "dataset": dataset_name,
            "n_train": n,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "seed": seed,
            "n_classes": n_classes,
            "model": model_name,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="ECG Experiment Runner (API-based)")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "multinli", "allenoise"])
    parser.add_argument("--n_train", type=int, default=25000)
    parser.add_argument("--noise_type", type=str, default="artifact_aligned",
                        choices=["uniform", "artifact_aligned", "none"])
    parser.add_argument("--noise_rate", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--model", type=str, default="qwen/qwen3-8b")
    parser.add_argument("--max_concurrency", type=int, default=50)
    parser.add_argument("--n_stability_samples", type=int, default=3)
    parser.add_argument("--k_neighbors", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print(f"ECG Experiment: {args.dataset} / {args.noise_type} / rate={args.noise_rate}")
    print(f"Seeds: {args.seeds}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    task_config = get_task_config_for_dataset(args.dataset)

    all_results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

        result = run_single_experiment(
            dataset_name=args.dataset,
            n_train=args.n_train,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            seed=seed,
            model_name=args.model,
            max_concurrency=args.max_concurrency,
            n_stability_samples=args.n_stability_samples,
            k_neighbors=args.k_neighbors,
            task_config=task_config,
            cache_base_dir=CACHE_DIR,
        )

        all_results[seed] = result["results"]

        # Free memory between seeds
        gc.collect()

    # Aggregate across seeds
    if len(args.seeds) > 1:
        print(f"\n{'='*60}")
        print(f"Aggregated Results ({len(args.seeds)} seeds)")
        print(f"{'='*60}")
        aggregated = aggregate_results(all_results)
        print_aggregated_results(aggregated)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"{timestamp}_{args.dataset}_{args.noise_type}_results.json"
    )

    output_data = {
        "experiment": f"{args.dataset}_{args.noise_type}",
        "timestamp": timestamp,
        "config": {
            "dataset": args.dataset,
            "n_train": args.n_train,
            "noise_type": args.noise_type,
            "noise_rate": args.noise_rate,
            "seeds": args.seeds,
            "model": args.model,
        },
        "results_by_seed": {str(k): v for k, v in all_results.items()},
    }

    # Add aggregated if multi-seed
    if len(args.seeds) > 1:
        agg = aggregate_results(all_results)
        output_data["aggregated"] = {
            method: {
                metric: {"mean": m.mean, "std": m.std, "n": m.n_seeds}
                for metric, m in metrics.items()
            }
            for method, metrics in agg.items()
        }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
