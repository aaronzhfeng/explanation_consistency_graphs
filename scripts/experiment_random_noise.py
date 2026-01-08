#!/usr/bin/env python3
"""
Experiment: Random Label Noise (No Artifacts)

This experiment tests ECG on random label flips without artifact tokens.
ECG should outperform simple embedding methods here because:
- No artifact clusters in input space
- NLI can detect explanation vs. content mismatch
- Stability sampling reveals inconsistent explanations

Run from project root:
    python scripts/experiment_random_noise.py
    python scripts/experiment_random_noise.py --noise_rate 0.05
    python scripts/experiment_random_noise.py --noise_rate 0.20

Estimated time: ~30 minutes on H100
"""

import argparse
import sys
sys.path.insert(0, 'src')

import os
import json
import pickle
import numpy as np
from datetime import datetime
from datasets import load_dataset

from ecg import (
    ExplanationGenerator, 
    generate_batch_with_stability, 
    get_reliability_scores,
    build_explanation_graph,
    explanations_to_embeddings,
    compute_all_baselines,
    compute_detection_metrics,
    print_detection_summary,
    cross_validate_predictions,
    TrainingConfig,
)
from ecg.signals import compute_all_signals, NLIScorer
from ecg.explain_llm import get_llm_predictions
from ecg.data import NoisyDataset, NoiseConfig


def compute_explanation_knn(explanations, labels, k=15):
    """
    Compute neighborhood surprise on EXPLANATION embeddings.
    
    This is the key ECG signal - same kNN algorithm as Input kNN,
    but computed on explanation embeddings instead of input embeddings.
    """
    from sentence_transformers import SentenceTransformer
    import faiss
    import torch
    
    # Build canonical explanation strings
    exp_strings = []
    for exp in explanations:
        parts = []
        if hasattr(exp, 'pred_label') and exp.pred_label:
            parts.append(f"Label: {exp.pred_label}")
        if hasattr(exp, 'evidence') and exp.evidence:
            parts.append(f"Evidence: {', '.join(exp.evidence[:3])}")
        if hasattr(exp, 'rationale') and exp.rationale:
            parts.append(f"Rationale: {exp.rationale}")
        exp_strings.append(" | ".join(parts) if parts else "UNKNOWN")
    
    # Embed explanations
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedder.encode(exp_strings, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    # Query k+1 neighbors (includes self)
    similarities, neighbors = index.search(embeddings, k + 1)
    
    # Remove self from neighbors
    neighbors = neighbors[:, 1:]
    similarities = similarities[:, 1:]
    
    # Compute neighborhood surprise
    labels = np.array(labels)
    n = len(labels)
    knn_scores = []
    
    for i in range(n):
        neighbor_labels = labels[neighbors[i]]
        neighbor_sims = similarities[i]
        
        # Weighted agreement with observed label
        weights = np.maximum(neighbor_sims, 0)
        weights = weights / (weights.sum() + 1e-8)
        
        observed = labels[i]
        p_observed = (weights * (neighbor_labels == observed)).sum()
        
        # Surprise = -log(p)
        surprise = -np.log(p_observed + 1e-6)
        knn_scores.append(surprise)
    
    knn_scores = np.array(knn_scores)
    
    del embedder
    torch.cuda.empty_cache()
    
    return knn_scores


def inject_random_noise(dataset, noise_rate: float = 0.10, seed: int = 42):
    """
    Inject random label noise WITHOUT artifacts.
    
    Simply flips labels randomly - no spurious tokens added.
    """
    np.random.seed(seed)
    n = len(dataset)
    n_noisy = int(n * noise_rate)
    
    # Select random examples to flip
    noisy_indices = np.random.choice(n, n_noisy, replace=False)
    is_noisy = np.zeros(n, dtype=bool)
    is_noisy[noisy_indices] = True
    
    # Create noisy labels (flip 0<->1)
    original_labels = np.array(dataset['label'])
    noisy_labels = original_labels.copy()
    noisy_labels[noisy_indices] = 1 - noisy_labels[noisy_indices]
    
    # Update the dataset's label column with noisy labels
    noisy_dataset = dataset.map(
        lambda example, idx: {'label': int(noisy_labels[idx])},
        with_indices=True,
    )
    
    # No artifacts for random noise
    has_artifact = np.zeros(n, dtype=bool)
    
    # Create noise config
    noise_config = NoiseConfig(
        noise_type='random_flip',
        noise_rate=noise_rate,
        seed=seed,
    )
    
    return NoisyDataset(
        dataset=noisy_dataset,
        clean_labels=original_labels,
        noisy_labels=noisy_labels,
        is_noisy=is_noisy,
        has_artifact=has_artifact,
        noise_config=noise_config,
        artifact_config=None,
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run random noise experiment for ECG.')
    parser.add_argument('--noise_rate', type=float, default=0.10,
                        help='Rate of label noise to inject (default: 0.10)')
    parser.add_argument('--dataset_size', type=int, default=25000,
                        help='Number of examples to use (default: 25000)')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('=' * 60)
    print('EXPERIMENT: Random Label Noise (No Artifacts)')
    print(f'Timestamp: {timestamp}')
    print('=' * 60)
    
    print(f'\nConfiguration:')
    print(f'  Noise rate: {args.noise_rate}')
    print(f'  Dataset size: {args.dataset_size}')
    
    # Step 1: Load dataset and inject random noise
    print('\n[1/7] Loading SST-2 and injecting random noise...')
    full_dataset = load_dataset('glue', 'sst2', split='train')
    # Use configurable subset size
    dataset = full_dataset.shuffle(seed=42).select(range(args.dataset_size))
    noisy_data = inject_random_noise(dataset, noise_rate=args.noise_rate, seed=42)
    print(f'  Dataset size: {len(noisy_data.dataset)}')
    print(f'  Noisy examples: {noisy_data.is_noisy.sum()}')
    print(f'  Noise type: {noisy_data.noise_config.noise_type}')
    
    # Step 2: Train classifier and get dynamics
    print('\n[2/7] Training classifier for dynamics...')
    from ecg.train_classifier import train_classifier
    
    # Train classifier (labels are already in the dataset)
    model, dynamics, results = train_classifier(
        train_dataset=noisy_data.dataset,
        config=TrainingConfig(epochs=3, seed=42),
        return_dynamics=True,
    )
    print(f'  Training loss: {results.get("train_loss", "N/A")}')
    print(f'  AUM scores computed: shape {dynamics.aum_scores.shape}')
    
    # Free GPU memory from classifier before loading LLM
    del model
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 3: Generate LLM explanations
    print('\n[3/7] Generating LLM explanations...')
    generator = ExplanationGenerator(
        model_name='Qwen/Qwen3-8B',
        use_vllm=True,
        temperature=0.0,
    )
    
    explanations = generate_batch_with_stability(
        generator=generator,
        sentences=noisy_data.dataset['sentence'],
        n_samples=3,
        sample_temperature=0.7,
        show_progress=True,
    )
    reliability = get_reliability_scores(explanations)
    print(f'  Mean reliability: {reliability.mean():.3f}')
    
    # CRITICAL: Free GPU memory from vLLM before cross-validation
    print('  Cleaning up LLM generator to free GPU memory...')
    del generator
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f'  GPU memory freed. Available: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB')
    
    # Step 4: Build graph and compute signals
    print('\n[4/7] Building graph and computing signals...')
    primary_exps = [e.primary for e in explanations]
    embeddings = explanations_to_embeddings(primary_exps)
    
    graph = build_explanation_graph(
        embeddings=embeddings,
        reliability=reliability,
        k=15,
        temperature=0.07,
    )
    
    nli_scorer = NLIScorer(model_names=['roberta-large-mnli'])
    
    # No known artifacts for random noise
    signals = compute_all_signals(
        graph=graph,
        explanations=primary_exps,
        observed_labels=noisy_data.noisy_labels,
        reliability_scores=reliability,
        aum_scores=dynamics.aum_scores,
        known_artifacts=[],  # No artifacts!
        nli_scorer=nli_scorer,
    )
    
    print(f'  S_nbr range: [{signals.neighborhood_surprise.min():.3f}, {signals.neighborhood_surprise.max():.3f}]')
    print(f'  S_nli range: [{signals.nli_contradiction.min():.3f}, {signals.nli_contradiction.max():.3f}]')
    print(f'  ECG score range: [{signals.ecg_score_adaptive.min():.3f}, {signals.ecg_score_adaptive.max():.3f}]')
    
    # Clean up NLI scorer before cross-validation
    del nli_scorer
    gc.collect()
    torch.cuda.empty_cache()
    print(f'  GPU memory available for CV: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB')
    
    # Step 5: Compute baselines
    print('\n[5/7] Computing baselines...')
    llm_labels, llm_confidence = get_llm_predictions(primary_exps)
    
    cv_probs = cross_validate_predictions(
        noisy_data.dataset,
        n_folds=5,
        config=TrainingConfig(epochs=3),
    )
    
    baselines = compute_all_baselines(
        labels=noisy_data.noisy_labels,
        pred_probs=cv_probs,
        input_embeddings=embeddings,
        llm_predicted_labels=llm_labels,
        llm_confidence=llm_confidence,
    )
    
    # Step 6: Compute Explanation kNN (the key ECG signal)
    print('\n[6/7] Computing Explanation kNN scores...')
    explanation_knn_scores = compute_explanation_knn(
        explanations=primary_exps,
        labels=noisy_data.noisy_labels,
        k=15,
    )
    print(f'  Explanation kNN range: [{explanation_knn_scores.min():.3f}, {explanation_knn_scores.max():.3f}]')
    
    # Step 7: Evaluate
    print('\n[7/7] Evaluating detection performance...')
    ground_truth = noisy_data.is_noisy
    
    all_scores = {
        'Explanation kNN': explanation_knn_scores,  # Key ECG signal
        'ECG (adaptive)': signals.ecg_score_adaptive,
        'ECG (fixed)': signals.ecg_score,
        'Cleanlab': baselines.cleanlab,
        'Loss': baselines.loss,
        'Margin': baselines.margin,
        'LLM Mismatch': baselines.llm_mismatch,
        'Input kNN': baselines.input_knn,
        'Random': baselines.random,
    }
    
    all_metrics = {}
    for name, scores in all_scores.items():
        if scores is not None:
            all_metrics[name] = compute_detection_metrics(ground_truth, scores)
    
    print('\n')
    print_detection_summary(all_metrics)
    
    # Save results
    os.makedirs('outputs/results', exist_ok=True)
    
    def safe_metric(m, attr):
        val = getattr(m, attr, 0)
        if isinstance(val, dict):
            return float(list(val.values())[0]) if val else 0.0
        return float(val) if val is not None else 0.0
    
    results = {
        'timestamp': timestamp,
        'config': {
            'noise_type': 'random_flip',
            'noise_rate': args.noise_rate,
            'dataset_size': args.dataset_size,
            'dataset': 'sst2',
            'n_examples': len(ground_truth),
            'n_noisy': int(ground_truth.sum()),
            'artifacts': [],
        },
        'metrics': {
            name: {
                'auroc': safe_metric(m, 'auroc'),
                'auprc': safe_metric(m, 'auprc'),
                'tnr_at_95': safe_metric(m, 'tnr_at_95'),
                'precision_at_k': safe_metric(m, 'precision_at_k'),
                'recall_at_k': safe_metric(m, 'recall_at_k'),
            }
            for name, m in all_metrics.items()
        }
    }
    
    results_path = f'outputs/results/{timestamp}_random_noise_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results to {results_path}')
    
    # Save intermediate data for debugging
    with open(f'outputs/results/{timestamp}_random_noise_data.pkl', 'wb') as f:
        pickle.dump({
            'noisy_data': noisy_data,
            'dynamics': dynamics,
            'explanations': explanations,
            'signals': signals,
            'baselines': baselines,
        }, f)
    
    # Summary
    print('\n' + '=' * 60)
    print('EXPERIMENT COMPLETE: Random Label Noise')
    print('=' * 60)
    
    exp_knn_auroc = all_metrics['Explanation kNN'].auroc if 'Explanation kNN' in all_metrics else 0
    ecg_auroc = all_metrics['ECG (adaptive)'].auroc
    cleanlab_auroc = all_metrics['Cleanlab'].auroc if 'Cleanlab' in all_metrics else 0
    input_knn_auroc = all_metrics['Input kNN'].auroc if 'Input kNN' in all_metrics else 0
    llm_mismatch_auroc = all_metrics['LLM Mismatch'].auroc if 'LLM Mismatch' in all_metrics else 0
    
    print(f'  Explanation kNN AUROC: {exp_knn_auroc:.3f}  <-- Key ECG signal')
    print(f'  ECG (adaptive) AUROC:  {ecg_auroc:.3f}')
    print(f'  Cleanlab AUROC:        {cleanlab_auroc:.3f}')
    print(f'  Input kNN AUROC:       {input_knn_auroc:.3f}')
    print(f'  LLM Mismatch AUROC:    {llm_mismatch_auroc:.3f}')
    print()
    print('  Comparisons:')
    print(f'    Explanation kNN vs Input kNN:  {exp_knn_auroc - input_knn_auroc:+.3f}')
    print(f'    Explanation kNN vs Cleanlab:   {exp_knn_auroc - cleanlab_auroc:+.3f}')
    print(f'    Explanation kNN vs LLM Mismatch: {exp_knn_auroc - llm_mismatch_auroc:+.3f}')


if __name__ == '__main__':
    main()

