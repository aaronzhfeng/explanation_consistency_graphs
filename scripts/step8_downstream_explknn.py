#!/usr/bin/env python3
"""
Step 8: Downstream Evaluation using Explanation kNN

This version uses the Explanation kNN signal (AUROC 0.832) instead of
the combined ECG score which underperformed.

Run:
    python scripts/step8_downstream_explknn.py
"""

# Fix CUDA multiprocessing issue with vLLM
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import sys
sys.path.insert(0, 'src')

import json
import pickle
import numpy as np
import torch
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

from ecg.data import create_noisy_dataset, NoiseConfig
from ecg.explain_llm import ExplanationGenerator, generate_batch_with_stability
from ecg.clean import select_top_k, remove_examples
from ecg import train_classifier, TrainingConfig


def compute_explanation_knn_scores(texts, explanations, k=15):
    """Compute kNN-based suspicion scores in explanation embedding space."""
    print("  Computing explanation embeddings...")
    
    # Extract rationales from explanations
    rationales = []
    for exp in explanations:
        if hasattr(exp, 'rationale') and exp.rationale:
            rationales.append(exp.rationale)
        elif isinstance(exp, dict) and exp.get('rationale'):
            rationales.append(exp['rationale'])
        else:
            rationales.append("unknown")
    
    # Embed rationales
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedder.encode(rationales, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    print(f"  Building kNN index (k={k})...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Query for k+1 neighbors (including self)
    distances, indices = index.search(embeddings, k + 1)
    
    # Remove self (first neighbor)
    neighbor_indices = indices[:, 1:]  # Shape: (n, k)
    neighbor_distances = distances[:, 1:]  # Similarity scores
    
    # Get predicted labels from explanations
    pred_labels = []
    for exp in explanations:
        if hasattr(exp, 'pred_label'):
            label = exp.pred_label
        elif isinstance(exp, dict):
            label = exp.get('pred_label', 'UNKNOWN')
        else:
            label = 'UNKNOWN'
        pred_labels.append(1 if label.upper() == 'POSITIVE' else 0)
    pred_labels = np.array(pred_labels)
    
    # Compute neighborhood surprise: fraction of neighbors with different label
    suspicion_scores = []
    for i in range(len(texts)):
        neighbors = neighbor_indices[i]
        neighbor_labels = pred_labels[neighbors]
        my_label = pred_labels[i]
        
        # Suspicion = fraction of neighbors that disagree
        disagreement = (neighbor_labels != my_label).mean()
        suspicion_scores.append(disagreement)
    
    return np.array(suspicion_scores)


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print('=' * 70)
    print('Step 8: Downstream Evaluation (using Explanation kNN)')
    print(f'Timestamp: {timestamp}')
    print('=' * 70)
    
    # Configuration
    dataset_size = 25000
    noise_rate = 0.1
    k_values = [0.01, 0.02, 0.05, 0.10]
    
    print(f'\nConfiguration:')
    print(f'  Dataset size: {dataset_size}')
    print(f'  Noise rate: {noise_rate}')
    
    # Step 1: Create noisy dataset
    print('\n[1/4] Creating noisy dataset...')
    noise_config = NoiseConfig(
        noise_rate=noise_rate,
        noise_type='artifact_aligned'
    )
    noisy_data = create_noisy_dataset(n_train=dataset_size, noise_config=noise_config)
    
    n_noisy = noisy_data.is_noisy.sum()
    print(f'  Total examples: {len(noisy_data.dataset)}')
    print(f'  Noisy examples: {n_noisy} ({n_noisy/len(noisy_data.dataset)*100:.1f}%)')
    
    # Step 2: Generate LLM explanations
    print('\n[2/4] Generating LLM explanations...')
    texts = [ex['sentence'] for ex in noisy_data.dataset]
    
    generator = ExplanationGenerator(
        model_name="Qwen/Qwen3-8B",
        use_vllm=True,
        device="cuda"
    )
    
    results = generate_batch_with_stability(
        generator=generator,
        sentences=texts,
        n_samples=1,
        show_progress=True
    )
    
    # ExplanationWithStability has .primary attribute containing the Explanation
    explanations = [r.primary for r in results]
    
    # Free GPU memory
    del generator
    torch.cuda.empty_cache()
    print('  GPU memory freed')
    
    # Step 3: Compute Explanation kNN scores
    print('\n[3/4] Computing Explanation kNN suspicion scores...')
    suspicion_scores = compute_explanation_knn_scores(texts, explanations, k=15)
    
    # Verify detection quality
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(noisy_data.is_noisy, suspicion_scores)
    print(f'  Explanation kNN AUROC: {auroc:.3f}')
    
    # Step 4: Train original classifier and evaluate cleaning
    print('\n[4/4] Training classifiers and evaluating cleaning...')
    
    val_dataset = load_dataset('glue', 'sst2')['validation']
    
    # Train on original noisy data
    print('\n  Training on original noisy data...')
    model_orig, _, results_orig = train_classifier(
        train_dataset=noisy_data.dataset,
        val_dataset=val_dataset,
        config=TrainingConfig(epochs=3, output_dir='outputs/checkpoints_orig'),
        return_dynamics=False,
    )
    original_acc = results_orig.get('val_accuracy', 0)
    print(f'  Original accuracy: {original_acc:.4f}')
    
    del model_orig
    torch.cuda.empty_cache()
    
    # Test each K value
    results_summary = []
    
    for k_fraction in k_values:
        print(f'\n  --- Testing K = {k_fraction*100:.0f}% removal ---')
        
        # Select top-K suspicious
        n_remove = int(len(texts) * k_fraction)
        top_indices = np.argsort(suspicion_scores)[-n_remove:][::-1]
        
        # Compute precision/recall
        precision = noisy_data.is_noisy[top_indices].sum() / len(top_indices)
        recall = noisy_data.is_noisy[top_indices].sum() / noisy_data.is_noisy.sum()
        
        print(f'    Removed: {len(top_indices)} examples')
        print(f'    Precision: {precision:.3f}')
        print(f'    Recall: {recall:.3f}')
        
        # Remove examples and convert back to Dataset
        from datasets import Dataset
        keep_mask = np.ones(len(texts), dtype=bool)
        keep_mask[top_indices] = False
        cleaned_list = [noisy_data.dataset[i] for i in range(len(texts)) if keep_mask[i]]
        cleaned_dataset = Dataset.from_list(cleaned_list)
        
        # Retrain
        print(f'    Retraining on {len(cleaned_dataset)} examples...')
        model_clean, _, results_clean = train_classifier(
            train_dataset=cleaned_dataset,
            val_dataset=val_dataset,
            config=TrainingConfig(epochs=3, output_dir=f'outputs/checkpoints_cleaned_k{int(k_fraction*100)}'),
            return_dynamics=False,
        )
        
        cleaned_acc = results_clean.get('val_accuracy', 0)
        improvement = cleaned_acc - original_acc
        
        print(f'    Cleaned accuracy: {cleaned_acc:.4f} (Δ = {improvement:+.4f})')
        
        results_summary.append({
            'k_fraction': k_fraction,
            'k_percent': f'{k_fraction*100:.0f}%',
            'removed': len(top_indices),
            'precision': float(precision),
            'recall': float(recall),
            'original_accuracy': float(original_acc),
            'cleaned_accuracy': float(cleaned_acc),
            'improvement': float(improvement),
        })
        
        del model_clean
        torch.cuda.empty_cache()
    
    # Print summary
    print('\n' + '=' * 70)
    print('DOWNSTREAM EVALUATION SUMMARY (Explanation kNN)')
    print('=' * 70)
    print(f'\nExplanation kNN AUROC: {auroc:.3f}')
    print(f'Original (noisy) accuracy: {original_acc:.4f}')
    print()
    print(f'{"K":>6} | {"Removed":>7} | {"Prec":>6} | {"Recall":>6} | {"Clean Acc":>9} | {"Δ":>7}')
    print('-' * 70)
    for r in results_summary:
        print(f'{r["k_percent"]:>6} | {r["removed"]:>7} | {r["precision"]:>6.3f} | {r["recall"]:>6.3f} | {r["cleaned_accuracy"]:>9.4f} | {r["improvement"]:>+7.4f}')
    
    # Save results
    os.makedirs('outputs/results', exist_ok=True)
    output_file = f'outputs/results/{timestamp}_downstream_explknn.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'signal': 'explanation_knn',
            'auroc': float(auroc),
            'original_accuracy': float(original_acc),
            'results': results_summary
        }, f, indent=2)
    print(f'\nSaved to {output_file}')
    
    print('\n' + '=' * 70)
    print('Step 8 (Explanation kNN) COMPLETE!')
    print('=' * 70)


if __name__ == '__main__':
    main()

