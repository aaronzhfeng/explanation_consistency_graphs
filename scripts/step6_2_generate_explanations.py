#!/usr/bin/env python3
"""
Step 6.2: Generate LLM Explanations with Stability Sampling

Run this in your terminal:
    python scripts/step6_2_generate_explanations.py

Estimated time: 2-4 hours on H100
"""

import sys
sys.path.insert(0, 'src')

import os
import pickle
from ecg import ExplanationGenerator, generate_batch_with_stability, get_reliability_scores

def main():
    print('=' * 60)
    print('Step 6.2: Generating LLM Explanations')
    print('=' * 60)

    # Load previous step
    print('\nLoading classifier data...')
    with open('outputs/step6_1_classifier.pkl', 'rb') as f:
        data = pickle.load(f)
    noisy_data = data['noisy_data']
    print(f'  Dataset size: {len(noisy_data.dataset)} examples')
    print(f'  Noisy examples: {noisy_data.is_noisy.sum()}')

    # Initialize generator
    print('\nInitializing LLM generator (Qwen/Qwen3-8B)...')
    generator = ExplanationGenerator(
        model_name='Qwen/Qwen3-8B',
        use_vllm=True,
        temperature=0.0,
    )
    print('  Generator ready!')

    # Generate explanations with stability
    print(f'\nGenerating explanations with stability sampling (n_samples=3)...')
    print('  This will take 2-4 hours on H100.\n')

    explanations = generate_batch_with_stability(
        generator=generator,
        sentences=noisy_data.dataset['sentence'],
        n_samples=3,
        sample_temperature=0.7,
        show_progress=True,
    )

    reliability = get_reliability_scores(explanations)
    print(f'\n  Mean reliability: {reliability.mean():.3f}')
    print(f'  Min reliability: {reliability.min():.3f}')
    print(f'  Max reliability: {reliability.max():.3f}')

    # Save
    os.makedirs('outputs/explanations', exist_ok=True)
    output_path = 'outputs/explanations/explanations.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'explanations': explanations,
            'reliability': reliability,
        }, f)
    
    print(f'\nSaved to {output_path}')
    print('=' * 60)
    print('Step 6.2 COMPLETE!')
    print('=' * 60)
    print('\nNext step: Run step 6.3 to build graph and compute signals.')


if __name__ == '__main__':
    main()

