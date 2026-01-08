#!/usr/bin/env python3
"""
Step 6.4: Run Baselines & Evaluate

Run this after step 6.3 completes:
    python scripts/step6_4_evaluate.py

Estimated time: ~30 minutes on H100
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import pickle
import numpy as np
from datetime import datetime
from ecg import compute_all_baselines, compute_detection_metrics, print_detection_summary
from ecg import cross_validate_predictions, TrainingConfig
from ecg.explain_llm import get_llm_predictions

def main():
    print('=' * 60)
    print('Step 6.4: Baselines & Evaluation')
    print('=' * 60)

    # Load all data
    print('\nLoading all previous steps...')
    
    with open('outputs/step6_1_classifier.pkl', 'rb') as f:
        data = pickle.load(f)
    noisy_data = data['noisy_data']
    dynamics = data['dynamics']
    print(f'  Classifier data: {len(noisy_data.dataset)} examples')

    with open('outputs/explanations/explanations.pkl', 'rb') as f:
        exp_data = pickle.load(f)
    explanations = exp_data['explanations']
    reliability = exp_data['reliability']
    print(f'  Explanations: {len(explanations)}')

    with open('outputs/step6_3_signals.pkl', 'rb') as f:
        sig_data = pickle.load(f)
    signals = sig_data['signals']
    embeddings = sig_data['embeddings']
    print(f'  Signals computed')

    # Get LLM predictions
    print('\nExtracting LLM predictions...')
    primary_exps = [e.primary for e in explanations]
    llm_labels, llm_confidence = get_llm_predictions(primary_exps)
    print(f'  LLM predictions extracted')

    # Cross-validation for Cleanlab
    print('\nRunning 5-fold cross-validation for Cleanlab baseline...')
    print('  (This may take ~20 minutes)')
    cv_probs = cross_validate_predictions(
        noisy_data.dataset,
        n_folds=5,
        config=TrainingConfig(epochs=3),
    )
    print(f'  CV predictions: shape {cv_probs.shape}')

    # Compute baselines
    print('\nComputing baseline scores...')
    baselines = compute_all_baselines(
        labels=noisy_data.noisy_labels,
        pred_probs=cv_probs,
        input_embeddings=embeddings,
        llm_predicted_labels=llm_labels,
        llm_confidence=llm_confidence,
    )
    print('  Baselines computed')

    # Evaluate all methods
    print('\nEvaluating detection performance...')
    ground_truth = noisy_data.is_noisy

    all_scores = {
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

    # Save results with timestamp
    os.makedirs('outputs/results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Handle precision_at_k being a dict
    def safe_metric(m, attr):
        val = getattr(m, attr, 0)
        if isinstance(val, dict):
            # Return value for first k if dict
            return float(list(val.values())[0]) if val else 0.0
        return float(val) if val is not None else 0.0
    
    results = {
        'timestamp': timestamp,
        'config': {
            'noise_type': 'artifact_aligned',  # Update this for different experiments
            'noise_rate': 0.10,
            'dataset': 'sst2',
            'n_examples': len(ground_truth),
            'n_noisy': int(ground_truth.sum()),
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
    
    # Save with timestamp
    results_path = f'outputs/results/{timestamp}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results to {results_path}')
    
    # Also save as latest for convenience
    with open('outputs/results/latest_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Try to save plots if matplotlib works
    try:
        from ecg.eval import plot_comparison
        plot_path = f'outputs/results/{timestamp}_detection_comparison.png'
        plot_comparison(ground_truth, all_scores, save_path=plot_path)
        print(f'Saved plot to {plot_path}')
    except Exception as e:
        print(f'Could not save plot: {e}')

    print('\n' + '=' * 60)
    print('Step 6.4 COMPLETE!')
    print('=' * 60)
    print('\nExperiment finished! Check outputs/results/ for final metrics.')
    
    # Print key result
    print('\n' + '=' * 60)
    print('KEY RESULTS (artifact-aligned noise @ 10%)')
    print('=' * 60)
    ecg_auroc = all_metrics['ECG (adaptive)'].auroc
    cleanlab_auroc = all_metrics['Cleanlab'].auroc if 'Cleanlab' in all_metrics else 0
    print(f'  ECG AUROC:      {ecg_auroc:.3f}')
    print(f'  Cleanlab AUROC: {cleanlab_auroc:.3f}')
    print(f'  ECG advantage:  +{ecg_auroc - cleanlab_auroc:.3f}')


if __name__ == '__main__':
    main()

