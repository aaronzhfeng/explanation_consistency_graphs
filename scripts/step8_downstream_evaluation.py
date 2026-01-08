#!/usr/bin/env python3
"""
Step 8: Downstream Evaluation (Retrain on Cleaned Data)

Run this after step 6.4 completes:
    python scripts/step8_downstream_evaluation.py

Estimated time: ~30 minutes on H100
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import pickle
from ecg.clean import select_top_k, remove_examples
from ecg import train_classifier, TrainingConfig
from datasets import load_dataset

def main():
    print('=' * 60)
    print('Step 8: Downstream Evaluation')
    print('=' * 60)

    # Load data
    print('\nLoading previous results...')
    with open('outputs/step6_1_classifier.pkl', 'rb') as f:
        data = pickle.load(f)
    noisy_data = data['noisy_data']
    original_results = data['results']
    print(f'  Original dataset: {len(noisy_data.dataset)} examples')
    print(f'  Original val accuracy: {original_results.get("val_accuracy", "N/A")}')

    with open('outputs/step6_3_signals.pkl', 'rb') as f:
        sig_data = pickle.load(f)
    signals = sig_data['signals']
    print('  Signals loaded')

    # Test multiple K values
    k_values = [0.01, 0.02, 0.05, 0.10]
    results_summary = []

    val_dataset = load_dataset('glue', 'sst2')['validation']

    for k_fraction in k_values:
        print(f'\n{"="*60}')
        print(f'Testing K = {k_fraction*100:.0f}% removal')
        print('=' * 60)

        # Select top-K suspicious and remove
        indices_to_remove = select_top_k(signals.ecg_score_adaptive, k_fraction=k_fraction)
        cleaned_dataset, kept_indices = remove_examples(noisy_data.dataset, indices_to_remove)

        print(f'  Removed: {len(indices_to_remove)} examples')
        print(f'  Remaining: {len(cleaned_dataset)} examples')

        # Check removal quality
        precision = noisy_data.is_noisy[indices_to_remove].sum() / len(indices_to_remove)
        recall = noisy_data.is_noisy[indices_to_remove].sum() / noisy_data.is_noisy.sum()
        print(f'  Removal precision: {precision:.3f}')
        print(f'  Removal recall: {recall:.3f}')

        # Retrain on cleaned data
        print(f'\n  Retraining classifier on cleaned data...')
        output_dir = f'outputs/checkpoints_cleaned_k{int(k_fraction*100)}'
        os.makedirs(output_dir, exist_ok=True)

        model_clean, _, results_clean = train_classifier(
            train_dataset=cleaned_dataset,
            val_dataset=val_dataset,
            config=TrainingConfig(epochs=3, output_dir=output_dir),
            return_dynamics=False,
        )

        cleaned_acc = results_clean.get('val_accuracy', 0)
        original_acc = original_results.get('val_accuracy', 0)
        improvement = cleaned_acc - original_acc

        print(f'\n  Cleaned model val accuracy: {cleaned_acc:.4f}')
        print(f'  Improvement: {improvement:+.4f}')

        results_summary.append({
            'k_fraction': k_fraction,
            'k_percent': f'{k_fraction*100:.0f}%',
            'removed': len(indices_to_remove),
            'precision': float(precision),
            'recall': float(recall),
            'original_accuracy': float(original_acc),
            'cleaned_accuracy': float(cleaned_acc),
            'improvement': float(improvement),
        })

    # Print summary
    print('\n' + '=' * 60)
    print('DOWNSTREAM EVALUATION SUMMARY')
    print('=' * 60)
    print(f'\nOriginal (noisy) accuracy: {original_results.get("val_accuracy", 0):.4f}')
    print()
    print(f'{"K":>6} | {"Removed":>7} | {"Prec":>6} | {"Recall":>6} | {"Clean Acc":>9} | {"Δ":>7}')
    print('-' * 60)
    for r in results_summary:
        print(f'{r["k_percent"]:>6} | {r["removed"]:>7} | {r["precision"]:>6.3f} | {r["recall"]:>6.3f} | {r["cleaned_accuracy"]:>9.4f} | {r["improvement"]:>+7.4f}')

    # Save results
    os.makedirs('outputs/results', exist_ok=True)
    with open('outputs/results/downstream_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f'\nSaved to outputs/results/downstream_results.json')

    print('\n' + '=' * 60)
    print('Step 8 COMPLETE!')
    print('=' * 60)


if __name__ == '__main__':
    main()

