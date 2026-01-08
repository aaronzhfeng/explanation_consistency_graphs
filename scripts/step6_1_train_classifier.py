#!/usr/bin/env python3
"""
Step 6.1: Train Classifier with Training Dynamics

Run this to start the experiment:
    python scripts/step6_1_train_classifier.py

Estimated time: ~30 minutes on H100
"""

import sys
sys.path.insert(0, 'src')

import os
import pickle
from ecg import create_noisy_dataset, train_classifier, NoiseConfig, TrainingConfig
from datasets import load_dataset

def main():
    print('=' * 60)
    print('Step 6.1: Training Classifier')
    print('=' * 60)

    # Create noisy dataset
    print('\nCreating noisy dataset...')
    noisy_data = create_noisy_dataset(
        n_train=25000,
        noise_config=NoiseConfig(
            noise_type='artifact_aligned',
            noise_rate=0.10,
        ),
    )
    print(f'  Dataset size: {len(noisy_data.dataset)} examples')
    print(f'  Noisy examples: {noisy_data.is_noisy.sum()} ({noisy_data.is_noisy.mean()*100:.1f}%)')

    # Load validation dataset
    print('\nLoading validation dataset...')
    val_dataset = load_dataset('glue', 'sst2')['validation']
    print(f'  Validation size: {len(val_dataset)} examples')

    # Train classifier
    print('\nTraining RoBERTa-base classifier...')
    print('  Epochs: 3')
    print('  Batch size: 64')
    print('  This will take ~30 minutes.\n')

    os.makedirs('outputs/checkpoints', exist_ok=True)
    
    model, dynamics, results = train_classifier(
        train_dataset=noisy_data.dataset,
        val_dataset=val_dataset,
        config=TrainingConfig(epochs=3, output_dir='outputs/checkpoints'),
        return_dynamics=True,
    )

    print(f'\n  Training loss: {results["train_loss"]:.4f}')
    print(f'  Val accuracy: {results.get("val_accuracy", "N/A")}')
    print(f'  AUM range: [{dynamics.aum_scores.min():.3f}, {dynamics.aum_scores.max():.3f}]')

    # Save for next step
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/step6_1_classifier.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'noisy_data': noisy_data,
            'dynamics': dynamics,
            'results': results,
        }, f)
    
    print(f'\nSaved to {output_path}')
    print('=' * 60)
    print('Step 6.1 COMPLETE!')
    print('=' * 60)
    print('\nNext step: Run step6_2_generate_explanations.py')


if __name__ == '__main__':
    main()

