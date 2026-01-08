#!/usr/bin/env python3
"""
Step 6.3: Build Graph & Compute Signals

Run this after step 6.2 completes:
    python scripts/step6_3_build_graph_signals.py

Estimated time: ~1 hour on H100
"""

import sys
sys.path.insert(0, 'src')

import pickle
import numpy as np
from ecg import build_explanation_graph, explanations_to_embeddings
from ecg.signals import compute_all_signals, NLIScorer

def main():
    print('=' * 60)
    print('Step 6.3: Building Graph & Computing Signals')
    print('=' * 60)

    # Load previous steps
    print('\nLoading classifier data...')
    with open('outputs/step6_1_classifier.pkl', 'rb') as f:
        data = pickle.load(f)
    noisy_data = data['noisy_data']
    dynamics = data['dynamics']
    print(f'  Dataset size: {len(noisy_data.dataset)}')

    print('\nLoading explanations...')
    with open('outputs/explanations/explanations.pkl', 'rb') as f:
        exp_data = pickle.load(f)
    explanations = exp_data['explanations']
    reliability = exp_data['reliability']
    print(f'  Explanations: {len(explanations)}')
    print(f'  Mean reliability: {reliability.mean():.3f}')

    # Embed explanations
    print('\nEmbedding explanations...')
    primary_exps = [e.primary for e in explanations]
    embeddings = explanations_to_embeddings(primary_exps)
    print(f'  Embeddings shape: {embeddings.shape}')

    # Build graph
    print('\nBuilding reliability-weighted kNN graph...')
    graph = build_explanation_graph(
        embeddings=embeddings,
        reliability=reliability,
        k=15,
        temperature=0.07,
    )
    print(f'  Graph nodes: {graph.n_nodes}')
    print(f'  Mean similarity: {graph.similarities.mean():.3f}')

    # Compute signals
    print('\nComputing ECG signals (including NLI - may take time)...')
    nli_scorer = NLIScorer(model_names=['roberta-large-mnli'])

    signals = compute_all_signals(
        graph=graph,
        explanations=primary_exps,
        observed_labels=noisy_data.noisy_labels,
        reliability_scores=reliability,
        aum_scores=dynamics.aum_scores,
        known_artifacts=['<lbl_pos>', '<lbl_neg>'],
        nli_scorer=nli_scorer,
    )

    print(f'\n  S_nbr range: [{signals.neighborhood_surprise.min():.3f}, {signals.neighborhood_surprise.max():.3f}]')
    print(f'  S_nli range: [{signals.nli_contradiction.min():.3f}, {signals.nli_contradiction.max():.3f}]')
    print(f'  S_art range: [{signals.artifact_score.min():.3f}, {signals.artifact_score.max():.3f}]')
    print(f'  ECG score range: [{signals.ecg_score_adaptive.min():.3f}, {signals.ecg_score_adaptive.max():.3f}]')

    # Save
    with open('outputs/step6_3_signals.pkl', 'wb') as f:
        pickle.dump({
            'graph': graph,
            'embeddings': embeddings,
            'signals': signals,
        }, f)
    
    print(f'\nSaved to outputs/step6_3_signals.pkl')
    print('=' * 60)
    print('Step 6.3 COMPLETE!')
    print('=' * 60)
    print('\nNext step: Run step 6.4 to compute baselines and evaluate.')


if __name__ == '__main__':
    main()

