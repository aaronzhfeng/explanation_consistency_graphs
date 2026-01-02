"""
clean.py: Data cleaning strategies for removing/relabeling noisy examples.

Implements:
- Top-K removal by suspicion score
- Relabeling with guardrails
- Outlier-aware cleaning
- Evaluation of cleaning effectiveness

References:
- Research proposal Section 6
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datasets import Dataset


@dataclass
class CleaningResult:
    """Result of cleaning operation."""
    # Indices
    removed_indices: np.ndarray  # Indices of removed examples
    relabeled_indices: np.ndarray  # Indices of relabeled examples
    kept_indices: np.ndarray  # Indices of kept examples
    
    # New labels (if relabeling)
    new_labels: Optional[np.ndarray] = None
    
    # Statistics
    n_removed: int = 0
    n_relabeled: int = 0
    n_kept: int = 0
    k_fraction: float = 0.0
    
    # Quality metrics (if ground truth available)
    removal_precision: Optional[float] = None  # Fraction of removed that were truly noisy
    removal_recall: Optional[float] = None  # Fraction of noisy that were removed


@dataclass
class CleaningConfig:
    """Configuration for cleaning."""
    strategy: str = "remove"  # "remove" or "relabel"
    k_fraction: float = 0.05  # Fraction of data to clean
    
    # For relabeling
    relabel_confidence_threshold: float = 0.7  # Min LLM confidence to relabel
    relabel_agreement_threshold: float = 0.8  # Min stability to trust LLM label
    
    # For outlier protection
    outlier_protection: bool = True
    outlier_threshold_percentile: float = 95  # Don't remove top 5% outliers
    
    # For dynamics veto
    dynamics_veto: bool = True
    dynamics_veto_threshold_percentile: float = 75


# =============================================================================
# Top-K Selection
# =============================================================================

def select_top_k(
    scores: np.ndarray,
    k_fraction: float,
    outlier_scores: Optional[np.ndarray] = None,
    outlier_threshold_percentile: float = 95,
    dynamics_scores: Optional[np.ndarray] = None,
    dynamics_threshold_percentile: float = 75,
) -> np.ndarray:
    """
    Select top-K most suspicious examples for cleaning.
    
    Optionally protects outliers and learnable examples.
    
    Args:
        scores: Suspicion scores (higher = more suspicious)
        k_fraction: Fraction of examples to select
        outlier_scores: Outlier scores (high = more outlier-like)
        outlier_threshold_percentile: Don't select top X% outliers
        dynamics_scores: Training dynamics scores (high = more suspicious)
        dynamics_threshold_percentile: Don't select examples below this percentile
        
    Returns:
        Indices of selected examples
    """
    n = len(scores)
    k = int(n * k_fraction)
    
    # Start with all indices
    valid_mask = np.ones(n, dtype=bool)
    
    # Exclude outliers (high outlier score = OOD, not mislabeled)
    if outlier_scores is not None:
        outlier_threshold = np.percentile(outlier_scores, outlier_threshold_percentile)
        valid_mask &= (outlier_scores < outlier_threshold)
    
    # Exclude "consistently learnable" examples (low dynamics score = high AUM)
    if dynamics_scores is not None:
        # dynamics_scores are already negated (higher = more suspicious)
        dynamics_threshold = np.percentile(dynamics_scores, 100 - dynamics_threshold_percentile)
        valid_mask &= (dynamics_scores > dynamics_threshold)
    
    # Get indices sorted by score (descending)
    valid_indices = np.where(valid_mask)[0]
    valid_scores = scores[valid_mask]
    
    # Sort by score and take top-K
    sorted_order = np.argsort(valid_scores)[::-1]
    selected_valid_indices = valid_indices[sorted_order[:k]]
    
    return selected_valid_indices


def select_by_threshold(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Select examples above a suspicion threshold.
    
    Args:
        scores: Suspicion scores
        threshold: Threshold value
        
    Returns:
        Indices of selected examples
    """
    return np.where(scores > threshold)[0]


# =============================================================================
# Removal Strategy
# =============================================================================

def remove_examples(
    dataset: Dataset,
    indices_to_remove: np.ndarray,
) -> Tuple[Dataset, np.ndarray]:
    """
    Remove examples from dataset.
    
    Args:
        dataset: HuggingFace Dataset
        indices_to_remove: Indices to remove
        
    Returns:
        Tuple of (cleaned_dataset, kept_indices)
    """
    n = len(dataset)
    all_indices = np.arange(n)
    
    remove_set = set(indices_to_remove)
    kept_indices = np.array([i for i in all_indices if i not in remove_set])
    
    cleaned_dataset = dataset.select(kept_indices.tolist())
    
    return cleaned_dataset, kept_indices


# =============================================================================
# Relabeling Strategy
# =============================================================================

def relabel_examples(
    labels: np.ndarray,
    indices_to_relabel: np.ndarray,
    llm_predicted_labels: np.ndarray,
    llm_confidence: np.ndarray,
    stability_scores: np.ndarray,
    confidence_threshold: float = 0.7,
    stability_threshold: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relabel suspicious examples using LLM predictions.
    
    Only relabels if:
    1. LLM is confident (confidence > threshold)
    2. LLM is stable (stability > threshold)
    3. LLM disagrees with current label
    
    Args:
        labels: Current labels
        indices_to_relabel: Candidate indices for relabeling
        llm_predicted_labels: LLM's predicted labels
        llm_confidence: LLM confidence scores (0-100)
        stability_scores: Stability/reliability scores (0-1)
        confidence_threshold: Min confidence to relabel
        stability_threshold: Min stability to trust LLM
        
    Returns:
        Tuple of (new_labels, actually_relabeled_indices)
    """
    new_labels = labels.copy()
    actually_relabeled = []
    
    # Normalize confidence to [0, 1]
    if llm_confidence.max() > 1:
        llm_confidence = llm_confidence / 100.0
    
    for idx in indices_to_relabel:
        current_label = labels[idx]
        llm_label = llm_predicted_labels[idx]
        confidence = llm_confidence[idx]
        stability = stability_scores[idx]
        
        # Check conditions
        if llm_label == current_label:
            continue  # LLM agrees, no change needed
        
        if confidence < confidence_threshold:
            continue  # LLM not confident enough
        
        if stability < stability_threshold:
            continue  # LLM not stable enough
        
        # Relabel
        new_labels[idx] = llm_label
        actually_relabeled.append(idx)
    
    return new_labels, np.array(actually_relabeled)


# =============================================================================
# Full Cleaning Pipeline
# =============================================================================

def clean_dataset(
    dataset: Dataset,
    labels: np.ndarray,
    ecg_scores: np.ndarray,
    config: CleaningConfig,
    outlier_scores: Optional[np.ndarray] = None,
    dynamics_scores: Optional[np.ndarray] = None,
    llm_predicted_labels: Optional[np.ndarray] = None,
    llm_confidence: Optional[np.ndarray] = None,
    stability_scores: Optional[np.ndarray] = None,
    ground_truth_noisy: Optional[np.ndarray] = None,
) -> CleaningResult:
    """
    Clean dataset by removing or relabeling suspicious examples.
    
    Args:
        dataset: HuggingFace Dataset
        labels: Current (noisy) labels
        ecg_scores: ECG suspicion scores
        config: CleaningConfig
        outlier_scores: For outlier protection
        dynamics_scores: For dynamics veto
        llm_predicted_labels: For relabeling
        llm_confidence: For relabeling
        stability_scores: For relabeling
        ground_truth_noisy: Binary mask of truly noisy examples (for evaluation)
        
    Returns:
        CleaningResult
    """
    n = len(dataset)
    
    # Select suspicious examples
    outlier_scores_input = outlier_scores if config.outlier_protection else None
    dynamics_scores_input = dynamics_scores if config.dynamics_veto else None
    
    selected_indices = select_top_k(
        ecg_scores,
        config.k_fraction,
        outlier_scores_input,
        config.outlier_threshold_percentile,
        dynamics_scores_input,
        config.dynamics_veto_threshold_percentile,
    )
    
    # Apply cleaning strategy
    if config.strategy == "remove":
        cleaned_dataset, kept_indices = remove_examples(dataset, selected_indices)
        
        result = CleaningResult(
            removed_indices=selected_indices,
            relabeled_indices=np.array([], dtype=int),
            kept_indices=kept_indices,
            new_labels=None,
            n_removed=len(selected_indices),
            n_relabeled=0,
            n_kept=len(kept_indices),
            k_fraction=config.k_fraction,
        )
        
    elif config.strategy == "relabel":
        if llm_predicted_labels is None:
            raise ValueError("Relabeling requires llm_predicted_labels")
        if llm_confidence is None:
            llm_confidence = np.ones(n) * 100  # Default high confidence
        if stability_scores is None:
            stability_scores = np.ones(n)  # Default high stability
        
        new_labels, relabeled_indices = relabel_examples(
            labels,
            selected_indices,
            llm_predicted_labels,
            llm_confidence,
            stability_scores,
            config.relabel_confidence_threshold,
            config.relabel_agreement_threshold,
        )
        
        # Indices that were considered but not relabeled are removed
        not_relabeled = np.setdiff1d(selected_indices, relabeled_indices)
        cleaned_dataset, kept_indices = remove_examples(dataset, not_relabeled)
        
        result = CleaningResult(
            removed_indices=not_relabeled,
            relabeled_indices=relabeled_indices,
            kept_indices=kept_indices,
            new_labels=new_labels,
            n_removed=len(not_relabeled),
            n_relabeled=len(relabeled_indices),
            n_kept=len(kept_indices),
            k_fraction=config.k_fraction,
        )
    
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")
    
    # Evaluate if ground truth available
    if ground_truth_noisy is not None:
        result = evaluate_cleaning(result, ground_truth_noisy)
    
    return result


def evaluate_cleaning(
    result: CleaningResult,
    ground_truth_noisy: np.ndarray,
) -> CleaningResult:
    """
    Evaluate cleaning quality against ground truth.
    
    Args:
        result: CleaningResult to evaluate
        ground_truth_noisy: Binary mask of truly noisy examples
        
    Returns:
        Updated CleaningResult with precision/recall
    """
    n_total_noisy = ground_truth_noisy.sum()
    
    # Precision: what fraction of removed/relabeled were truly noisy
    all_cleaned = np.concatenate([result.removed_indices, result.relabeled_indices])
    if len(all_cleaned) > 0:
        n_correct = ground_truth_noisy[all_cleaned].sum()
        result.removal_precision = n_correct / len(all_cleaned)
    else:
        result.removal_precision = 0.0
    
    # Recall: what fraction of all noisy were cleaned
    if n_total_noisy > 0:
        n_correct = ground_truth_noisy[all_cleaned].sum() if len(all_cleaned) > 0 else 0
        result.removal_recall = n_correct / n_total_noisy
    else:
        result.removal_recall = 1.0  # No noisy examples to recall
    
    return result


# =============================================================================
# Multi-K Evaluation
# =============================================================================

def evaluate_at_multiple_k(
    ecg_scores: np.ndarray,
    ground_truth_noisy: np.ndarray,
    k_fractions: List[float] = [0.005, 0.01, 0.02, 0.05, 0.10],
    outlier_scores: Optional[np.ndarray] = None,
    dynamics_scores: Optional[np.ndarray] = None,
) -> Dict[float, Tuple[float, float]]:
    """
    Evaluate cleaning at multiple K values.
    
    Args:
        ecg_scores: Suspicion scores
        ground_truth_noisy: Binary mask of truly noisy
        k_fractions: List of K fractions to evaluate
        outlier_scores: For outlier protection
        dynamics_scores: For dynamics veto
        
    Returns:
        Dict mapping k -> (precision, recall)
    """
    results = {}
    n_total_noisy = ground_truth_noisy.sum()
    
    for k in k_fractions:
        selected = select_top_k(
            ecg_scores, k,
            outlier_scores, 95,
            dynamics_scores, 75,
        )
        
        if len(selected) > 0:
            n_correct = ground_truth_noisy[selected].sum()
            precision = n_correct / len(selected)
        else:
            precision = 0.0
        
        if n_total_noisy > 0:
            n_correct = ground_truth_noisy[selected].sum() if len(selected) > 0 else 0
            recall = n_correct / n_total_noisy
        else:
            recall = 1.0
        
        results[k] = (precision, recall)
    
    return results


def print_cleaning_summary(
    result: CleaningResult,
    method_name: str = "ECG",
) -> None:
    """Print summary of cleaning operation."""
    print(f"\n{'='*60}")
    print(f"Cleaning Summary: {method_name}")
    print(f"{'='*60}")
    print(f"K fraction: {result.k_fraction:.2%}")
    print(f"Removed: {result.n_removed}")
    print(f"Relabeled: {result.n_relabeled}")
    print(f"Kept: {result.n_kept}")
    
    if result.removal_precision is not None:
        print(f"\nQuality Metrics (vs ground truth):")
        print(f"  Precision: {result.removal_precision:.3f}")
        print(f"  Recall: {result.removal_recall:.3f}")
        f1 = 2 * result.removal_precision * result.removal_recall / (
            result.removal_precision + result.removal_recall + 1e-8
        )
        print(f"  F1: {f1:.3f}")
    
    print(f"{'='*60}\n")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    from datasets import Dataset as HFDataset
    
    # Test with random data
    np.random.seed(42)
    n = 1000
    noise_rate = 0.1
    
    # Simulate data
    ground_truth_noisy = np.random.rand(n) < noise_rate
    
    # Simulate ECG scores (noisy examples have higher scores on average)
    ecg_scores = np.random.randn(n) * 0.5
    ecg_scores[ground_truth_noisy] += 1.5
    
    # Create dummy dataset
    dummy_data = {
        "text": [f"Example {i}" for i in range(n)],
        "label": np.random.randint(0, 2, n),
    }
    dataset = HFDataset.from_dict(dummy_data)
    
    # Test removal
    config = CleaningConfig(
        strategy="remove",
        k_fraction=0.05,
    )
    
    result = clean_dataset(
        dataset=dataset,
        labels=dummy_data["label"],
        ecg_scores=ecg_scores,
        config=config,
        ground_truth_noisy=ground_truth_noisy,
    )
    
    print_cleaning_summary(result, "ECG (removal)")
    
    # Evaluate at multiple K
    print("\nPrecision/Recall at various K:")
    multi_k_results = evaluate_at_multiple_k(ecg_scores, ground_truth_noisy)
    for k, (prec, rec) in sorted(multi_k_results.items()):
        print(f"  K={k:.1%}: Precision={prec:.3f}, Recall={rec:.3f}")

