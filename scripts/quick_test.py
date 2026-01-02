#!/usr/bin/env python
"""
quick_test.py: Quick test of ECG pipeline with minimal data.

Runs a minimal end-to-end test without LLM (uses mock explanations).
Useful for verifying the pipeline works before running full experiments.

Usage:
    python scripts/quick_test.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from datasets import Dataset

def create_mock_explanations(n: int, labels: np.ndarray, noise_mask: np.ndarray):
    """Create mock explanations for testing."""
    from ecg.explain_llm import Explanation, StabilityMetrics, ExplanationWithStability
    
    results = []
    
    for i in range(n):
        # Create mock explanation
        label_str = "POSITIVE" if labels[i] == 1 else "NEGATIVE"
        
        # Simulate: noisy examples have less reliable explanations
        if noise_mask[i]:
            reliability = np.random.uniform(0.3, 0.6)
            confidence = np.random.randint(30, 60)
            # LLM might predict opposite label
            if np.random.rand() > 0.5:
                label_str = "NEGATIVE" if labels[i] == 1 else "POSITIVE"
        else:
            reliability = np.random.uniform(0.7, 1.0)
            confidence = np.random.randint(70, 100)
        
        exp = Explanation(
            pred_label=label_str,
            evidence=[f"word_{i}_{j}" for j in range(np.random.randint(1, 4))],
            rationale=f"The text expresses a certain sentiment based on word choice.",
            counterfactual=f"If the tone were different, the sentiment would change.",
            confidence=confidence,
        )
        
        stability = StabilityMetrics(
            label_agreement=reliability,
            evidence_jaccard=reliability,
            rationale_similarity=reliability,
            reliability_score=reliability,
            n_samples=3,
            labels=[label_str] * 3,
            dominant_label=label_str,
        )
        
        results.append(ExplanationWithStability(
            primary=exp,
            stability=stability,
            samples=[exp] * 3,
        ))
    
    return results


def main():
    print("="*60)
    print("ECG Quick Test (Mock Explanations)")
    print("="*60)
    
    # Parameters
    n = 500
    noise_rate = 0.1
    seed = 42
    
    np.random.seed(seed)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    sentences = [f"This is test sentence {i}." for i in range(n)]
    true_labels = np.random.randint(0, 2, n)
    noisy_labels = true_labels.copy()
    
    # Inject noise
    noise_mask = np.random.rand(n) < noise_rate
    noisy_labels[noise_mask] = 1 - noisy_labels[noise_mask]
    
    print(f"   N examples: {n}")
    print(f"   N noisy: {noise_mask.sum()}")
    
    # Create mock explanations
    print("\n2. Creating mock explanations...")
    explanations_with_stability = create_mock_explanations(n, noisy_labels, noise_mask)
    
    from ecg.explain_llm import get_reliability_scores, get_llm_predictions
    reliability_scores = get_reliability_scores(explanations_with_stability)
    primary_explanations = [e.primary for e in explanations_with_stability]
    llm_labels, llm_confidence = get_llm_predictions(primary_explanations)
    
    print(f"   Mean reliability: {reliability_scores.mean():.3f}")
    
    # Build graph
    print("\n3. Building explanation graph...")
    from ecg.embed_graph import build_explanation_graph
    
    # Create random embeddings for test
    embeddings = np.random.randn(n, 384).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    graph = build_explanation_graph(
        embeddings=embeddings,
        reliability=reliability_scores,
        k=15,
        temperature=0.07,
        similarity_threshold=0.0,  # No threshold for random embeddings
    )
    
    print(f"   Graph nodes: {graph.n_nodes}")
    print(f"   Mean similarity: {graph.similarities.mean():.3f}")
    
    # Compute signals
    print("\n4. Computing signals...")
    from ecg.signals import (
        compute_neighborhood_surprise,
        compute_stability_signal,
        compute_dynamics_signal,
        SignalScores,
        combine_signals_fixed_weights,
        combine_signals_adaptive,
    )
    
    # Neighborhood surprise
    s_nbr, c_nbr = compute_neighborhood_surprise(graph, noisy_labels, n_classes=2)
    
    # Stability signal
    s_stab, c_stab = compute_stability_signal(reliability_scores)
    
    # Mock dynamics (normally from training)
    aum_scores = np.random.randn(n) * 0.5
    aum_scores[noise_mask] -= 1.0  # Noisy examples have lower AUM
    s_dyn, c_dyn = compute_dynamics_signal(aum_scores)
    
    # Mock NLI and artifact (simplified)
    s_nli = np.random.randn(n) * 0.3
    s_nli[noise_mask] += 0.5  # Noisy examples have higher NLI contradiction
    
    s_art = np.zeros(n)
    
    signals = SignalScores(
        neighborhood_surprise=s_nbr,
        nli_contradiction=s_nli,
        artifact_score=s_art,
        stability_score=s_stab,
        dynamics_score=s_dyn,
        neighborhood_confidence=c_nbr,
        nli_confidence=np.abs(s_nli),
        stability_confidence=c_stab,
        dynamics_confidence=c_dyn,
    )
    
    signals.ecg_score = combine_signals_fixed_weights(signals)
    signals.ecg_score_adaptive = combine_signals_adaptive(signals)
    
    print(f"   S_nbr range: [{s_nbr.min():.3f}, {s_nbr.max():.3f}]")
    print(f"   ECG score range: [{signals.ecg_score.min():.3f}, {signals.ecg_score.max():.3f}]")
    
    # Evaluate
    print("\n5. Evaluating detection...")
    from ecg.eval import compute_detection_metrics, print_detection_summary
    
    # ECG metrics
    ecg_metrics = compute_detection_metrics(noise_mask, signals.ecg_score_adaptive)
    
    # Random baseline
    random_scores = np.random.rand(n)
    random_metrics = compute_detection_metrics(noise_mask, random_scores)
    
    # LLM mismatch baseline
    llm_mismatch = (llm_labels != noisy_labels).astype(float)
    llm_metrics = compute_detection_metrics(noise_mask, llm_mismatch)
    
    print_detection_summary({
        "ECG (adaptive)": ecg_metrics,
        "LLM Mismatch": llm_metrics,
        "Random": random_metrics,
    })
    
    # Cleaning evaluation
    print("6. Evaluating cleaning...")
    from ecg.clean import evaluate_at_multiple_k
    
    k_results = evaluate_at_multiple_k(
        signals.ecg_score_adaptive,
        noise_mask,
        k_fractions=[0.01, 0.02, 0.05, 0.10],
    )
    
    print("\nPrecision/Recall at various K:")
    for k, (prec, rec) in sorted(k_results.items()):
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"   K={k:.1%}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    
    print("\n" + "="*60)
    print("Quick test PASSED!")
    print("="*60)
    print("\nNote: This test uses mock data. For real experiments:")
    print("  python scripts/run_experiment.py --config configs/default.yaml")


if __name__ == "__main__":
    main()

