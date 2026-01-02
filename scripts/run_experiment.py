#!/usr/bin/env python
"""
run_experiment.py: Full ECG experiment pipeline.

Runs the complete ECG pipeline:
1. Load data and inject noise
2. Train classifier, compute training dynamics
3. Generate LLM explanations with stability sampling
4. Build explanation graph
5. Compute ECG signals
6. Run baselines
7. Evaluate detection performance
8. Clean data and evaluate downstream

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/default.yaml --skip-llm  # Use cached explanations
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf


def setup_output_dirs(config):
    """Create output directories."""
    dirs = [
        config.output.base_dir,
        config.output.checkpoints_dir,
        config.output.explanations_dir,
        config.output.results_dir,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_config(config_path: str):
    """Load and resolve config."""
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    return config


def save_results(results: Dict, path: str):
    """Save results to JSON."""
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)


# =============================================================================
# Pipeline Steps
# =============================================================================

def step1_load_data(config, verbose: bool = True):
    """Step 1: Load data and inject noise."""
    from ecg.data import create_noisy_dataset, load_sst2, NoiseConfig, ArtifactConfig
    
    if verbose:
        print("\n" + "="*60)
        print("STEP 1: Loading data and injecting noise")
        print("="*60)
    
    # Create noise config
    noise_config = NoiseConfig(
        noise_type=config.data.noise.type,
        noise_rate=config.data.noise.rate,
        seed=config.data.seed,
    )
    
    # Create artifact config if enabled
    artifact_config = None
    if config.data.artifacts.enabled:
        artifact_config = ArtifactConfig(
            positive_token=config.data.artifacts.positive_token,
            negative_token=config.data.artifacts.negative_token,
        )
    
    # Create noisy dataset
    noisy_data = create_noisy_dataset(
        n_train=config.data.n_train,
        noise_config=noise_config,
        artifact_config=artifact_config,
        seed=config.data.seed,
    )
    
    if verbose:
        print(f"  Dataset: {config.data.dataset}")
        print(f"  Train size: {len(noisy_data.dataset)}")
        print(f"  Noise type: {config.data.noise.type}")
        print(f"  Noise rate: {config.data.noise.rate:.1%}")
        print(f"  N noisy: {noisy_data.is_noisy.sum()}")
    
    # Load validation set
    sst2 = load_sst2()
    val_dataset = sst2["validation"]
    
    if verbose:
        print(f"  Validation size: {len(val_dataset)}")
    
    return noisy_data, val_dataset


def step2_train_classifier(config, noisy_data, val_dataset, verbose: bool = True):
    """Step 2: Train classifier and compute training dynamics."""
    from ecg.train_classifier import train_classifier, TrainingConfig, cross_validate_predictions
    
    if verbose:
        print("\n" + "="*60)
        print("STEP 2: Training classifier")
        print("="*60)
    
    # Create training config
    train_config = TrainingConfig(
        model_name=config.classifier.model_name,
        max_length=config.classifier.max_length,
        epochs=config.classifier.training.epochs,
        batch_size=config.classifier.training.batch_size,
        learning_rate=config.classifier.training.learning_rate,
        warmup_ratio=config.classifier.training.warmup_ratio,
        weight_decay=config.classifier.training.weight_decay,
        output_dir=config.output.checkpoints_dir,
        seed=config.data.seed,
        compute_training_dynamics=config.classifier.compute_training_dynamics,
    )
    
    if verbose:
        print(f"  Model: {train_config.model_name}")
        print(f"  Epochs: {train_config.epochs}")
        print(f"  Batch size: {train_config.batch_size}")
    
    # Train classifier
    model, dynamics, results = train_classifier(
        train_dataset=noisy_data.dataset,
        val_dataset=val_dataset,
        config=train_config,
        return_dynamics=True,
    )
    
    if verbose:
        print(f"  Training loss: {results.get('train_loss', 'N/A'):.4f}")
        if 'val_accuracy' in results:
            print(f"  Val accuracy: {results['val_accuracy']:.4f}")
        if dynamics:
            print(f"  AUM range: [{dynamics.aum_scores.min():.3f}, {dynamics.aum_scores.max():.3f}]")
    
    # Get cross-validation predictions for Cleanlab
    if verbose:
        print("  Running cross-validation for Cleanlab...")
    
    cv_probs = cross_validate_predictions(
        noisy_data.dataset,
        n_folds=5,
        config=train_config,
    )
    
    return model, dynamics, cv_probs, train_config


def step3_generate_explanations(
    config, noisy_data, output_path: Optional[str] = None, 
    skip_llm: bool = False, verbose: bool = True
):
    """Step 3: Generate LLM explanations with stability sampling."""
    from ecg.explain_llm import (
        ExplanationGenerator, 
        generate_batch_with_stability,
        explanations_to_embeddings,
        get_reliability_scores,
        get_llm_predictions,
    )
    
    if verbose:
        print("\n" + "="*60)
        print("STEP 3: Generating LLM explanations")
        print("="*60)
    
    # Check for cached explanations
    cache_path = output_path or os.path.join(config.output.explanations_dir, "explanations.pkl")
    
    if skip_llm and os.path.exists(cache_path):
        if verbose:
            print(f"  Loading cached explanations from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        return (
            cached['explanations_with_stability'],
            cached['reliability_scores'],
            cached['llm_labels'],
            cached['llm_confidence'],
        )
    
    if verbose:
        print(f"  LLM model: {config.explanation.model_name}")
        print(f"  Stability samples: {config.explanation.stability.num_samples}")
    
    # Initialize generator
    generator = ExplanationGenerator(
        model_name=config.explanation.model_name,
        use_vllm=True,  # Prefer vLLM
        temperature=config.explanation.temperature_primary,
        max_new_tokens=config.explanation.max_new_tokens,
    )
    
    # Get sentences
    sentences = noisy_data.dataset['sentence']
    
    if verbose:
        print(f"  Generating explanations for {len(sentences)} examples...")
    
    # Generate with stability
    explanations_with_stability = generate_batch_with_stability(
        generator=generator,
        sentences=sentences,
        n_samples=config.explanation.stability.num_samples,
        sample_temperature=config.explanation.stability.temperature,
        show_progress=True,
    )
    
    # Extract derived data
    reliability_scores = get_reliability_scores(explanations_with_stability)
    primary_explanations = [e.primary for e in explanations_with_stability]
    llm_labels, llm_confidence = get_llm_predictions(primary_explanations)
    
    if verbose:
        print(f"  Mean reliability: {reliability_scores.mean():.3f}")
        n_agreement = (llm_labels == noisy_data.noisy_labels).sum()
        print(f"  LLM-label agreement: {n_agreement}/{len(llm_labels)} ({n_agreement/len(llm_labels):.1%})")
    
    # Cache explanations
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'explanations_with_stability': explanations_with_stability,
            'reliability_scores': reliability_scores,
            'llm_labels': llm_labels,
            'llm_confidence': llm_confidence,
        }, f)
    if verbose:
        print(f"  Cached explanations to {cache_path}")
    
    return explanations_with_stability, reliability_scores, llm_labels, llm_confidence


def step4_build_graph(
    config, explanations_with_stability, reliability_scores, verbose: bool = True
):
    """Step 4: Build explanation graph."""
    from ecg.embed_graph import (
        build_explanation_graph,
        explanations_to_embeddings,
    )
    from ecg.explain_llm import explanations_to_embeddings
    
    if verbose:
        print("\n" + "="*60)
        print("STEP 4: Building explanation graph")
        print("="*60)
    
    # Get primary explanations
    primary_explanations = [e.primary for e in explanations_with_stability]
    
    # Embed explanations
    if verbose:
        print(f"  Embedding model: {config.graph.embedding_model}")
    
    embeddings = explanations_to_embeddings(
        primary_explanations,
        model_name=config.graph.embedding_model,
        include_counterfactual=config.explanation.require_counterfactual,
    )
    
    if verbose:
        print(f"  Embeddings shape: {embeddings.shape}")
    
    # Build graph
    if verbose:
        print(f"  Building kNN graph (k={config.graph.k})...")
    
    graph = build_explanation_graph(
        embeddings=embeddings,
        reliability=reliability_scores,
        k=config.graph.k,
        temperature=config.graph.temperature,
        similarity_threshold=config.graph.similarity_threshold,
        use_mutual_knn=config.graph.use_mutual_knn,
        compute_outliers=config.graph.compute_outlier_scores,
    )
    
    if verbose:
        print(f"  Graph: {graph.n_nodes} nodes")
        print(f"  Mean neighbor similarity: {graph.similarities.mean():.3f}")
        if graph.outlier_scores is not None:
            print(f"  Outlier score range: [{graph.outlier_scores.min():.3f}, {graph.outlier_scores.max():.3f}]")
    
    return graph, embeddings


def step5_compute_signals(
    config, graph, explanations_with_stability, noisy_data, 
    reliability_scores, dynamics, verbose: bool = True
):
    """Step 5: Compute ECG signals."""
    from ecg.signals import compute_all_signals, NLIScorer
    
    if verbose:
        print("\n" + "="*60)
        print("STEP 5: Computing ECG signals")
        print("="*60)
    
    # Get primary explanations
    primary_explanations = [e.primary for e in explanations_with_stability]
    
    # Initialize NLI scorer
    nli_models = [config.signals.nli.model_name]
    if config.signals.nli.ensemble.enabled:
        nli_models.append(config.signals.nli.ensemble.second_model)
    
    nli_scorer = NLIScorer(model_names=nli_models)
    
    # Get known artifacts
    known_artifacts = None
    if config.data.artifacts.enabled and config.signals.artifact.mode == "synthetic":
        known_artifacts = [
            config.data.artifacts.positive_token,
            config.data.artifacts.negative_token,
        ]
    
    # Compute all signals
    signals = compute_all_signals(
        graph=graph,
        explanations=primary_explanations,
        observed_labels=noisy_data.noisy_labels,
        reliability_scores=reliability_scores,
        aum_scores=dynamics.aum_scores if dynamics else np.zeros(len(noisy_data.dataset)),
        known_artifacts=known_artifacts,
        nli_scorer=nli_scorer,
        n_classes=2,
        show_progress=True,
    )
    
    if verbose:
        print(f"\nSignal ranges:")
        print(f"  S_nbr: [{signals.neighborhood_surprise.min():.3f}, {signals.neighborhood_surprise.max():.3f}]")
        print(f"  S_nli: [{signals.nli_contradiction.min():.3f}, {signals.nli_contradiction.max():.3f}]")
        print(f"  S_art: [{signals.artifact_score.min():.3f}, {signals.artifact_score.max():.3f}]")
        print(f"  S_stab: [{signals.stability_score.min():.3f}, {signals.stability_score.max():.3f}]")
        print(f"  S_dyn: [{signals.dynamics_score.min():.3f}, {signals.dynamics_score.max():.3f}]")
        print(f"  ECG (fixed): [{signals.ecg_score.min():.3f}, {signals.ecg_score.max():.3f}]")
        print(f"  ECG (adaptive): [{signals.ecg_score_adaptive.min():.3f}, {signals.ecg_score_adaptive.max():.3f}]")
    
    return signals


def step6_run_baselines(
    config, noisy_data, cv_probs, embeddings, model, dynamics, 
    llm_labels, llm_confidence, verbose: bool = True
):
    """Step 6: Compute baseline scores."""
    from ecg.baselines import compute_all_baselines
    from ecg.train_classifier import get_cls_embeddings, tokenize_dataset
    from transformers import AutoTokenizer
    
    if verbose:
        print("\n" + "="*60)
        print("STEP 6: Computing baselines")
        print("="*60)
    
    # Get classifier embeddings
    tokenizer = AutoTokenizer.from_pretrained(config.classifier.model_name)
    train_tokenized = tokenize_dataset(
        noisy_data.dataset, tokenizer, config.classifier.max_length
    )
    cls_embeddings = get_cls_embeddings(
        model, train_tokenized, tokenizer, config.classifier.training.batch_size
    )
    
    # Compute all baselines
    baselines = compute_all_baselines(
        labels=noisy_data.noisy_labels,
        pred_probs=cv_probs,
        features=cls_embeddings,
        input_embeddings=embeddings,  # Explanation embeddings
        llm_predicted_labels=llm_labels,
        llm_confidence=llm_confidence,
        k=config.graph.k,
        seed=config.data.seed,
    )
    
    if verbose:
        print("  Baselines computed:")
        for attr in ['cleanlab', 'loss', 'margin', 'aum', 'entropy', 'llm_mismatch', 
                     'input_knn', 'classifier_knn', 'nrg', 'random']:
            score = getattr(baselines, attr, None)
            if score is not None:
                print(f"    {attr}: [{score.min():.3f}, {score.max():.3f}]")
    
    return baselines


def step7_evaluate(config, signals, baselines, noisy_data, verbose: bool = True):
    """Step 7: Evaluate detection performance."""
    from ecg.eval import compute_detection_metrics, print_detection_summary, plot_comparison
    
    if verbose:
        print("\n" + "="*60)
        print("STEP 7: Evaluating detection performance")
        print("="*60)
    
    ground_truth = noisy_data.is_noisy
    k_values = config.cleaning.k_values
    
    # Collect all scores
    all_scores = {
        "ECG (adaptive)": signals.ecg_score_adaptive,
        "ECG (fixed)": signals.ecg_score,
        "Cleanlab": baselines.cleanlab,
        "Loss": baselines.loss,
        "Margin": baselines.margin,
        "LLM Mismatch": baselines.llm_mismatch,
        "Input kNN": baselines.input_knn,
        "Classifier kNN": baselines.classifier_knn,
        "NRG": baselines.nrg,
        "Random": baselines.random,
    }
    
    # Filter out None scores
    all_scores = {k: v for k, v in all_scores.items() if v is not None}
    
    # Compute metrics for each method
    all_metrics = {}
    for name, scores in all_scores.items():
        metrics = compute_detection_metrics(ground_truth, scores, k_values)
        all_metrics[name] = metrics
    
    # Print summary
    print_detection_summary(all_metrics, k_to_show=0.05)
    
    # Plot comparison
    plot_path = os.path.join(config.output.results_dir, "detection_comparison.png")
    plot_comparison(ground_truth, all_scores, save_path=plot_path)
    
    return all_metrics


def step8_clean_and_evaluate_downstream(
    config, noisy_data, val_dataset, signals, model, verbose: bool = True
):
    """Step 8: Clean data and evaluate downstream performance."""
    from ecg.clean import clean_dataset, CleaningConfig, print_cleaning_summary, evaluate_at_multiple_k
    from ecg.train_classifier import train_classifier, TrainingConfig
    
    if verbose:
        print("\n" + "="*60)
        print("STEP 8: Cleaning and downstream evaluation")
        print("="*60)
    
    results = {}
    
    # Evaluate at multiple K
    multi_k_results = evaluate_at_multiple_k(
        signals.ecg_score_adaptive,
        noisy_data.is_noisy,
        config.cleaning.k_values,
    )
    
    if verbose:
        print("\nPrecision/Recall at various K:")
        for k, (prec, rec) in sorted(multi_k_results.items()):
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            print(f"  K={k:.1%}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
    
    results['multi_k'] = multi_k_results
    
    # Clean at K=5%
    k_main = 0.05
    cleaning_config = CleaningConfig(
        strategy=config.cleaning.method,
        k_fraction=k_main,
    )
    
    from ecg.clean import select_top_k, remove_examples
    
    # Select and remove
    indices_to_remove = select_top_k(
        signals.ecg_score_adaptive,
        k_main,
        outlier_scores=None,  # Optional: signals.outlier_scores
        dynamics_scores=signals.dynamics_score if config.signals.aggregation.use_dynamics_veto else None,
    )
    
    cleaned_dataset, kept_indices = remove_examples(noisy_data.dataset, indices_to_remove)
    
    # Compute cleaning quality
    n_correct = noisy_data.is_noisy[indices_to_remove].sum()
    precision = n_correct / len(indices_to_remove) if len(indices_to_remove) > 0 else 0
    recall = n_correct / noisy_data.is_noisy.sum() if noisy_data.is_noisy.sum() > 0 else 0
    
    if verbose:
        print(f"\nCleaning at K={k_main:.1%}:")
        print(f"  Removed: {len(indices_to_remove)}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
    
    results['cleaning'] = {
        'k': k_main,
        'n_removed': len(indices_to_remove),
        'precision': precision,
        'recall': recall,
    }
    
    # TODO: Retrain classifier on cleaned data and evaluate downstream
    # This would require:
    # 1. Create new dataset with kept indices
    # 2. Train new classifier
    # 3. Evaluate on clean validation set
    # 4. Evaluate on OOD test sets (stripped, swapped artifacts)
    
    if verbose:
        print("\n[Note: Full downstream evaluation requires retraining classifier on cleaned data]")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run ECG experiment pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--skip-llm", action="store_true",
                       help="Skip LLM generation, use cached explanations")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    args = parser.parse_args()
    
    # Load config
    print("="*60)
    print("ECG: Explanation-Consistency Graphs")
    print("="*60)
    print(f"Config: {args.config}")
    
    config = load_config(args.config)
    setup_output_dirs(config)
    
    # Run pipeline
    start_time = datetime.now()
    
    # Step 1: Load data
    noisy_data, val_dataset = step1_load_data(config, args.verbose)
    
    # Step 2: Train classifier
    model, dynamics, cv_probs, train_config = step2_train_classifier(
        config, noisy_data, val_dataset, args.verbose
    )
    
    # Step 3: Generate explanations
    explanations_with_stability, reliability_scores, llm_labels, llm_confidence = \
        step3_generate_explanations(config, noisy_data, skip_llm=args.skip_llm, verbose=args.verbose)
    
    # Step 4: Build graph
    graph, embeddings = step4_build_graph(
        config, explanations_with_stability, reliability_scores, args.verbose
    )
    
    # Step 5: Compute signals
    signals = step5_compute_signals(
        config, graph, explanations_with_stability, noisy_data,
        reliability_scores, dynamics, args.verbose
    )
    
    # Step 6: Run baselines
    baselines = step6_run_baselines(
        config, noisy_data, cv_probs, embeddings, model, dynamics,
        llm_labels, llm_confidence, args.verbose
    )
    
    # Step 7: Evaluate
    detection_metrics = step7_evaluate(config, signals, baselines, noisy_data, args.verbose)
    
    # Step 8: Clean and evaluate downstream
    downstream_results = step8_clean_and_evaluate_downstream(
        config, noisy_data, val_dataset, signals, model, args.verbose
    )
    
    # Save results
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results = {
        'config': OmegaConf.to_container(config),
        'detection_metrics': {k: {
            'auroc': v.auroc,
            'auprc': v.auprc,
            'tnr_at_95': v.tnr_at_95,
            'precision_at_k': v.precision_at_k,
            'recall_at_k': v.recall_at_k,
            'f1_at_k': v.f1_at_k,
        } for k, v in detection_metrics.items()},
        'downstream_results': downstream_results,
        'duration_seconds': duration,
        'timestamp': end_time.isoformat(),
    }
    
    results_path = os.path.join(config.output.results_dir, "results.json")
    save_results(results, results_path)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Results saved to: {results_path}")
    print("="*60)


if __name__ == "__main__":
    main()

