#!/usr/bin/env python3
"""
Experiment: Test ensemble of Input kNN + LLM Mismatch.

This tests the hypothesis that combining the two best-performing 
approaches will exceed either alone.

Expected improvement on artifact noise:
  - Input kNN: 0.810
  - LLM Mismatch: 0.609
  - Ensemble: 0.85+ (hypothesis)
"""

# Fix CUDA multiprocessing issue with vLLM
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import json

# ECG imports
from ecg.data import create_noisy_dataset, NoiseConfig
from ecg.explain_llm import ExplanationGenerator, generate_batch_with_stability


def compute_llm_mismatch(explanations, observed_labels):
    """Compute LLM mismatch score (binary: LLM disagrees with label)."""
    scores = []
    for exp, label in zip(explanations, observed_labels):
        if hasattr(exp, 'pred_label'):
            llm_pred = exp.pred_label
        else:
            llm_pred = exp.get('pred_label', 'UNKNOWN')
        
        # Convert to numeric
        if llm_pred.upper() == 'POSITIVE':
            pred_num = 1
        elif llm_pred.upper() == 'NEGATIVE':
            pred_num = 0
        else:
            pred_num = -1  # Unknown
        
        # Mismatch = disagreement
        if pred_num == -1:
            scores.append(0.5)  # Uncertain
        else:
            scores.append(1.0 if pred_num != label else 0.0)
    
    return np.array(scores)


def compute_artifact_score(explanations, artifact_tokens=["<lbl_pos>", "<lbl_neg>"]):
    """Compute artifact score: does LLM cite artifact tokens as evidence?"""
    artifact_set = set(tok.lower() for tok in artifact_tokens)
    scores = []
    
    for exp in explanations:
        if hasattr(exp, 'evidence'):
            evidence = exp.evidence
        else:
            evidence = exp.get('evidence', [])
        
        if isinstance(evidence, str):
            evidence = [evidence]
        
        # Check if any artifact token appears in evidence
        evidence_text = " ".join(evidence).lower()
        
        # Count artifact mentions
        artifact_count = sum(1 for tok in artifact_set if tok in evidence_text)
        
        # Score: 1.0 if any artifact cited, 0.0 otherwise
        scores.append(1.0 if artifact_count > 0 else 0.0)
    
    return np.array(scores)


def compute_explanation_knn(explanations, labels, k=15):
    """Compute neighborhood surprise on EXPLANATION embeddings."""
    from sentence_transformers import SentenceTransformer
    import faiss
    
    # Build canonical explanation strings
    exp_texts = []
    for exp in explanations:
        if hasattr(exp, 'evidence'):
            evidence = exp.evidence
            rationale = exp.rationale or ""
        else:
            evidence = exp.get('evidence', [])
            rationale = exp.get('rationale', '')
        
        if isinstance(evidence, list):
            evidence_str = "; ".join(evidence)
        else:
            evidence_str = str(evidence)
        
        exp_texts.append(f"Evidence: {evidence_str}. Rationale: {rationale}")
    
    # Embed explanations
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(exp_texts, show_progress_bar=True, batch_size=128)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Build kNN graph
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    similarities, neighbors = index.search(embeddings.astype(np.float32), k + 1)
    
    # Exclude self
    neighbors = neighbors[:, 1:]
    similarities = similarities[:, 1:]
    
    # Compute neighborhood surprise
    n = len(labels)
    scores = []
    
    for i in range(n):
        neighbor_labels = labels[neighbors[i]]
        neighbor_sims = similarities[i]
        
        # Weighted vote
        weights = np.exp(neighbor_sims / 0.07)
        weights /= weights.sum()
        
        # Probability of observed label
        p_observed = (weights * (neighbor_labels == labels[i])).sum()
        
        # Surprise = -log(p)
        surprise = -np.log(p_observed + 1e-6)
        scores.append(surprise)
    
    return np.array(scores)


def rank_normalize(scores):
    """Convert to percentile ranks [0, 1]."""
    n = len(scores)
    if n <= 1:
        return np.zeros(n)
    ranks = np.argsort(np.argsort(scores))
    return ranks / (n - 1)


def evaluate(scores, is_noisy, name):
    """Compute detection metrics."""
    auroc = roc_auc_score(is_noisy, scores)
    auprc = average_precision_score(is_noisy, scores)
    
    # Precision at K (K = number of noisy examples)
    k = int(is_noisy.sum())
    top_k_idx = np.argsort(scores)[-k:]
    p_at_k = is_noisy[top_k_idx].mean()
    
    # TNR at 95% TPR
    fpr, tpr, thresholds = roc_curve(is_noisy, scores)
    idx_95 = np.argmax(tpr >= 0.95)
    tnr_at_95 = 1 - fpr[idx_95] if idx_95 < len(fpr) else 0.0
    
    return {
        "name": name,
        "auroc": auroc,
        "auprc": auprc,
        "precision_at_k": p_at_k,
        "tnr_at_95": tnr_at_95,
    }


def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ensemble experiment on noisy labels")
    parser.add_argument("--noise_rate", type=float, default=0.10, 
                        help="Noise rate (default: 0.10)")
    parser.add_argument("--dataset_size", type=int, default=25000,
                        help="Dataset size (default: 25000)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="LLM model name (default: Qwen/Qwen3-8B)")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("EXPERIMENT: Ensemble Methods for Noisy Label Detection")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)
    
    # Configuration
    dataset_size = args.dataset_size
    noise_rate = args.noise_rate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nConfiguration:")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Noise rate: {noise_rate}")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    
    # =========================================================================
    # 1. Load data and inject ARTIFACT-ALIGNED noise
    # =========================================================================
    print("\n[1/5] Loading SST-2 with artifact-aligned noise...")
    
    noise_config = NoiseConfig(
        noise_rate=noise_rate,
        noise_type="artifact_aligned",
        positive_marker="<lbl_pos>",
        negative_marker="<lbl_neg>",
        seed=42,
    )
    
    noisy_dataset = create_noisy_dataset(
        n_train=dataset_size,
        noise_config=noise_config,
        seed=42,
    )
    
    # Extract data from the NoisyDataset dataclass
    dataset = noisy_dataset.dataset
    texts = [dataset[i]['sentence'] for i in range(len(dataset))]
    labels = noisy_dataset.noisy_labels  # Use noisy labels (what the model sees)
    is_noisy = noisy_dataset.is_noisy
    
    print(f"  Total examples: {len(texts)}")
    print(f"  Noisy examples: {is_noisy.sum()} ({is_noisy.mean()*100:.1f}%)")
    
    # =========================================================================
    # 2. Generate LLM explanations
    # =========================================================================
    print("\n[2/5] Generating LLM explanations...")
    
    generator = ExplanationGenerator(
        model_name=args.model,
        device=device,
    )
    
    results = generate_batch_with_stability(
        generator,
        texts,
        n_samples=1,  # Just primary explanation for speed
        show_progress=True,
    )
    
    # Extract primary explanations
    explanations = [r.primary for r in results]
    
    # Compute LLM mismatch
    llm_mismatch = compute_llm_mismatch(explanations, labels)
    
    # Free GPU memory
    del generator
    torch.cuda.empty_cache()
    print(f"  GPU memory freed")
    
    # =========================================================================
    # 3. Compute Input kNN scores
    # =========================================================================
    print("\n[3/5] Computing Input kNN scores...")
    
    # Use raw texts (without artifacts stripped for fair comparison)
    from sentence_transformers import SentenceTransformer
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    input_embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=128)
    input_embeddings = input_embeddings / np.linalg.norm(input_embeddings, axis=1, keepdims=True)
    
    # Build kNN graph on inputs
    import faiss
    
    k = 15
    index = faiss.IndexFlatIP(input_embeddings.shape[1])
    index.add(input_embeddings.astype(np.float32))
    similarities, neighbors = index.search(input_embeddings.astype(np.float32), k + 1)
    
    # Exclude self
    neighbors = neighbors[:, 1:]
    similarities = similarities[:, 1:]
    
    # Compute neighborhood surprise on inputs
    n = len(labels)
    input_knn_scores = []
    
    for i in range(n):
        neighbor_labels = labels[neighbors[i]]
        neighbor_sims = similarities[i]
        
        # Weighted vote
        weights = np.exp(neighbor_sims / 0.07)
        weights /= weights.sum()
        
        # Probability of observed label
        p_observed = (weights * (neighbor_labels == labels[i])).sum()
        
        # Surprise = -log(p)
        surprise = -np.log(p_observed + 1e-6)
        input_knn_scores.append(surprise)
    
    input_knn_scores = np.array(input_knn_scores)
    
    del embedder
    torch.cuda.empty_cache()
    
    # =========================================================================
    # 4. Compute additional signals
    # =========================================================================
    print("\n[4/6] Computing additional signals...")
    
    # Artifact score (does LLM cite artifact tokens?)
    print("  Computing artifact score...")
    artifact_scores = compute_artifact_score(explanations)
    print(f"    Artifact detection rate: {(artifact_scores > 0).mean()*100:.1f}%")
    
    # Explanation kNN (neighborhood surprise on explanation embeddings)
    print("  Computing explanation kNN...")
    explanation_knn_scores = compute_explanation_knn(explanations, labels)
    
    # =========================================================================
    # 5. Test ensemble combinations
    # =========================================================================
    print("\n[5/6] Testing ensemble combinations...")
    
    results = []
    
    # === Individual signals ===
    print("\n  --- Individual Signals ---")
    
    r = evaluate(llm_mismatch, is_noisy, "LLM Mismatch")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    r = evaluate(input_knn_scores, is_noisy, "Input kNN")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    r = evaluate(explanation_knn_scores, is_noisy, "Explanation kNN")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    r = evaluate(artifact_scores, is_noisy, "Artifact Score")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    # Rank normalize all signals
    llm_mismatch_rank = rank_normalize(llm_mismatch)
    input_knn_rank = rank_normalize(input_knn_scores)
    exp_knn_rank = rank_normalize(explanation_knn_scores)
    artifact_rank = rank_normalize(artifact_scores)
    
    # === Two-signal combinations ===
    print("\n  --- Two-Signal Ensembles ---")
    
    # Max combinations
    max_llm_input = np.maximum(llm_mismatch_rank, input_knn_rank)
    r = evaluate(max_llm_input, is_noisy, "Max(LLM, Input kNN)")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    max_llm_exp = np.maximum(llm_mismatch_rank, exp_knn_rank)
    r = evaluate(max_llm_exp, is_noisy, "Max(LLM, Exp kNN)")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    max_artifact_input = np.maximum(artifact_rank, input_knn_rank)
    r = evaluate(max_artifact_input, is_noisy, "Max(Artifact, Input kNN)")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    # LLM Mismatch + Artifact (sum)
    llm_artifact = 0.5 * llm_mismatch_rank + 0.5 * artifact_rank
    r = evaluate(llm_artifact, is_noisy, "LLM + Artifact (avg)")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    # === Three-signal combinations ===
    print("\n  --- Three-Signal Ensembles ---")
    
    # Max of all three LLM-based signals
    max_all_llm = np.maximum.reduce([llm_mismatch_rank, exp_knn_rank, artifact_rank])
    r = evaluate(max_all_llm, is_noisy, "Max(LLM, ExpKNN, Artifact)")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    # Average of best signals
    avg_best = (llm_mismatch_rank + input_knn_rank + artifact_rank) / 3
    r = evaluate(avg_best, is_noisy, "Avg(LLM, Input, Artifact)")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    # Max of three
    max_three = np.maximum.reduce([llm_mismatch_rank, input_knn_rank, artifact_rank])
    r = evaluate(max_three, is_noisy, "Max(LLM, Input, Artifact)")
    results.append(r)
    print(f"  {r['name']:30s}: AUROC={r['auroc']:.3f}")
    
    # =========================================================================
    # 6. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Artifact-Aligned Noise)")
    print("=" * 70)
    
    results.sort(key=lambda x: x['auroc'], reverse=True)
    
    print(f"\n{'Method':<30} {'AUROC':>8} {'AUPRC':>8} {'P@K':>8} {'TNR@95':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['auroc']:>8.3f} {r['auprc']:>8.3f} "
              f"{r['precision_at_k']:>8.3f} {r['tnr_at_95']:>8.3f}")
    
    # Best result
    best = results[0]
    print("\n" + "=" * 70)
    print(f"BEST METHOD: {best['name']}")
    print(f"  AUROC: {best['auroc']:.3f}")
    print("=" * 70)
    
    # Save results
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "experiment": "ensemble_artifact_aligned",
        "timestamp": timestamp,
        "config": {
            "dataset_size": dataset_size,
            "noise_rate": noise_rate,
            "model": args.model,
        },
        "results": {r['name']: r for r in results},
    }
    
    output_path = output_dir / f"{timestamp}_ensemble_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()

