"""
eval.py: Evaluation metrics for label noise detection and explanation quality.

Implements:
- Detection metrics: AUROC, AUPRC, Precision@K, Recall@K, TNR@95
- Downstream metrics: Accuracy, OOD robustness
- Explanation faithfulness: Comprehensiveness, Sufficiency
- Leakage metrics: Label leakage detection

References:
- Neural Relation Graph: https://github.com/snu-mllab/Neural-Relation-Graph (metric.py)
- ERASER: https://www.eraserbenchmark.com/
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from sklearn import metrics
import matplotlib.pyplot as plt


@dataclass
class DetectionMetrics:
    """Metrics for label noise detection."""
    auroc: float
    auprc: float
    tnr_at_95: float
    precision_at_k: Dict[float, float] = field(default_factory=dict)
    recall_at_k: Dict[float, float] = field(default_factory=dict)
    f1_at_k: Dict[float, float] = field(default_factory=dict)


@dataclass
class DownstreamMetrics:
    """Metrics for downstream model performance."""
    accuracy: float
    accuracy_stripped: Optional[float] = None  # Accuracy with artifacts stripped
    accuracy_swapped: Optional[float] = None  # Accuracy with artifacts swapped


@dataclass
class FaithfulnessMetrics:
    """Metrics for explanation faithfulness."""
    comprehensiveness: float
    sufficiency: float
    spurious_attribution: Optional[float] = None  # Dependence on spurious markers


# =============================================================================
# Detection Metrics (adapted from NRG)
# =============================================================================

def compute_auroc_auprc(
    ground_truth: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute AUROC, AUPRC, and TNR@95 for detection.
    
    Adapted from NRG: metric.py (cal_auc_ap)
    
    Args:
        ground_truth: Binary array indicating noisy labels (1) or clean (0)
        scores: Suspicion scores (higher = more likely noisy)
        
    Returns:
        Tuple of (AUROC, AUPRC, TNR@95)
    """
    if isinstance(ground_truth, list):
        ground_truth = np.array(ground_truth)
    if isinstance(scores, list):
        scores = np.array(scores)
    
    # Ensure numpy arrays
    ground_truth = ground_truth.astype(int)
    
    # ROC curve and AUC
    fpr, tpr, _ = metrics.roc_curve(ground_truth, scores)
    auroc = metrics.auc(fpr, tpr)
    
    # Precision-Recall curve and AP
    auprc = metrics.average_precision_score(ground_truth, scores)
    
    # TNR at 95% TPR
    idx = np.sum(tpr < 0.95)
    tnr_at_95 = 1 - fpr[idx] if idx < len(fpr) else 0.0
    
    return auroc, auprc, tnr_at_95


def compute_precision_recall_at_k(
    ground_truth: np.ndarray,
    scores: np.ndarray,
    k_values: List[float] = [0.005, 0.01, 0.02, 0.05, 0.10],
) -> Tuple[Dict[float, float], Dict[float, float], Dict[float, float]]:
    """
    Compute Precision@K, Recall@K, and F1@K for various K values.
    
    Args:
        ground_truth: Binary array indicating noisy labels
        scores: Suspicion scores (higher = more suspicious)
        k_values: List of K values as fractions of dataset
        
    Returns:
        Tuple of (precision_dict, recall_dict, f1_dict)
    """
    n = len(ground_truth)
    n_positive = ground_truth.sum()
    
    # Rank by scores (descending)
    ranked_indices = np.argsort(scores)[::-1]
    ranked_truth = ground_truth[ranked_indices]
    
    precision_at_k = {}
    recall_at_k = {}
    f1_at_k = {}
    
    for k_frac in k_values:
        k = int(n * k_frac)
        if k == 0:
            k = 1
        
        # Top-K predictions
        top_k_truth = ranked_truth[:k]
        
        # Precision: how many of top-K are actually noisy
        precision = top_k_truth.sum() / k
        
        # Recall: how many of all noisy are in top-K
        recall = top_k_truth.sum() / n_positive if n_positive > 0 else 0.0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_at_k[k_frac] = precision
        recall_at_k[k_frac] = recall
        f1_at_k[k_frac] = f1
    
    return precision_at_k, recall_at_k, f1_at_k


def compute_detection_metrics(
    ground_truth: np.ndarray,
    scores: np.ndarray,
    k_values: List[float] = [0.005, 0.01, 0.02, 0.05, 0.10],
) -> DetectionMetrics:
    """
    Compute all detection metrics.
    
    Args:
        ground_truth: Binary array indicating noisy labels
        scores: Suspicion scores (higher = more suspicious)
        k_values: List of K values as fractions
        
    Returns:
        DetectionMetrics dataclass
    """
    auroc, auprc, tnr_at_95 = compute_auroc_auprc(ground_truth, scores)
    precision_at_k, recall_at_k, f1_at_k = compute_precision_recall_at_k(
        ground_truth, scores, k_values
    )
    
    return DetectionMetrics(
        auroc=auroc,
        auprc=auprc,
        tnr_at_95=tnr_at_95,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        f1_at_k=f1_at_k,
    )


# =============================================================================
# Visualization (adapted from NRG)
# =============================================================================

def plot_score_histogram(
    ground_truth: np.ndarray,
    scores: np.ndarray,
    title: str = 'Score Distribution',
    save_path: Optional[str] = None,
    bins: int = 50,
    log_scale: bool = True,
) -> None:
    """
    Plot histogram of scores for clean vs noisy examples.
    
    Adapted from NRG: metric.py (hist)
    
    Args:
        ground_truth: Binary array indicating noisy labels
        scores: Suspicion scores
        title: Plot title
        save_path: Path to save figure
        bins: Number of histogram bins
        log_scale: Whether to use log scale
    """
    clean_mask = ~ground_truth.astype(bool)
    noisy_mask = ground_truth.astype(bool)
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    
    # All scores
    axes[0].hist(scores, bins=bins, log=log_scale, alpha=0.7)
    axes[0].set_title('All Examples')
    axes[0].set_xlabel('Score')
    
    # Clean scores
    axes[1].hist(scores[clean_mask], bins=bins, log=log_scale, alpha=0.7, color='green')
    axes[1].set_title('Clean Examples')
    axes[1].set_xlabel('Score')
    
    # Noisy scores
    axes[2].hist(scores[noisy_mask], bins=bins, log=log_scale, alpha=0.7, color='red')
    axes[2].set_title('Noisy Examples')
    axes[2].set_xlabel('Score')
    
    # PR curve
    precision, recall, _ = metrics.precision_recall_curve(ground_truth, scores)
    axes[3].plot(recall[:-50], precision[:-50])
    axes[3].set_xlabel('Recall')
    axes[3].set_ylabel('Precision')
    axes[3].set_title('PR Curve')
    axes[3].grid(True)
    axes[3].set_ylim([0, 1.05])
    
    fig.suptitle(title)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.close()


def plot_comparison(
    ground_truth: np.ndarray,
    scores_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot comparison of multiple detection methods.
    
    Args:
        ground_truth: Binary array indicating noisy labels
        scores_dict: Dictionary mapping method name to scores
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curves
    for name, scores in scores_dict.items():
        fpr, tpr, _ = metrics.roc_curve(ground_truth, scores)
        auroc = metrics.auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC={auroc:.3f})')
    
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # PR curves
    for name, scores in scores_dict.items():
        precision, recall, _ = metrics.precision_recall_curve(ground_truth, scores)
        auprc = metrics.average_precision_score(ground_truth, scores)
        axes[1].plot(recall, precision, label=f'{name} (AP={auprc:.3f})')
    
    baseline = ground_truth.mean()
    axes[1].axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curves')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    plt.close()


# =============================================================================
# Explanation Faithfulness Metrics
# =============================================================================

def compute_comprehensiveness(
    model,
    tokenizer,
    texts: List[str],
    evidence_spans: List[List[str]],
    labels: np.ndarray,
    batch_size: int = 32,
) -> float:
    """
    Compute comprehensiveness: how much does removing evidence affect prediction.
    
    Comprehensiveness = P(y|x) - P(y|x_without_evidence)
    Higher is better (evidence is important for prediction).
    
    Args:
        model: Trained classifier
        tokenizer: Tokenizer
        texts: Original texts
        evidence_spans: Evidence spans to remove for each text
        labels: Labels to measure probability for
        batch_size: Batch size for inference
        
    Returns:
        Mean comprehensiveness score
    """
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get original probabilities
    original_probs = _get_label_probabilities(model, tokenizer, texts, labels, batch_size, device)
    
    # Remove evidence and get new probabilities
    texts_without_evidence = []
    for text, spans in zip(texts, evidence_spans):
        modified_text = text
        for span in spans:
            modified_text = modified_text.replace(span, "")
        # Clean up whitespace
        modified_text = " ".join(modified_text.split())
        texts_without_evidence.append(modified_text)
    
    modified_probs = _get_label_probabilities(model, tokenizer, texts_without_evidence, labels, batch_size, device)
    
    # Comprehensiveness: drop in probability when evidence is removed
    comprehensiveness = original_probs - modified_probs
    
    return float(comprehensiveness.mean())


def compute_sufficiency(
    model,
    tokenizer,
    texts: List[str],
    evidence_spans: List[List[str]],
    labels: np.ndarray,
    batch_size: int = 32,
) -> float:
    """
    Compute sufficiency: how well does evidence alone support prediction.
    
    Sufficiency = P(y|x) - P(y|evidence_only)
    Lower is better (evidence alone is sufficient).
    
    Args:
        model: Trained classifier
        tokenizer: Tokenizer
        texts: Original texts
        evidence_spans: Evidence spans for each text
        labels: Labels to measure probability for
        batch_size: Batch size for inference
        
    Returns:
        Mean sufficiency score
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Get original probabilities
    original_probs = _get_label_probabilities(model, tokenizer, texts, labels, batch_size, device)
    
    # Keep only evidence
    evidence_only_texts = []
    for spans in evidence_spans:
        evidence_text = " ".join(spans) if spans else "[EMPTY]"
        evidence_only_texts.append(evidence_text)
    
    evidence_probs = _get_label_probabilities(model, tokenizer, evidence_only_texts, labels, batch_size, device)
    
    # Sufficiency: drop when keeping only evidence
    sufficiency = original_probs - evidence_probs
    
    return float(sufficiency.mean())


def _get_label_probabilities(
    model,
    tokenizer,
    texts: List[str],
    labels: np.ndarray,
    batch_size: int,
    device,
) -> np.ndarray:
    """Helper to get probabilities of specific labels."""
    import torch
    import torch.nn.functional as F
    
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
            # Get probability of the specific label
            label_probs = probs[torch.arange(len(batch_labels)), batch_labels]
            all_probs.append(label_probs.cpu().numpy())
    
    return np.concatenate(all_probs)


# =============================================================================
# Leakage Detection
# =============================================================================

def compute_label_leakage(
    rationales: List[str],
    labels: np.ndarray,
) -> float:
    """
    Compute how much rationales leak label information.
    
    Trains a simple classifier on rationales to predict labels.
    High accuracy = high leakage (rationales encode labels).
    
    Args:
        rationales: List of rationale texts
        labels: True labels
        
    Returns:
        Accuracy of rationale-only classifier (higher = more leakage)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    # Vectorize rationales
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(rationales)
    
    # Cross-validated accuracy
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, labels, cv=5, scoring='accuracy')
    
    return float(scores.mean())


# =============================================================================
# Summary Report
# =============================================================================

def print_detection_summary(
    method_metrics: Dict[str, DetectionMetrics],
    k_to_show: float = 0.05,
) -> None:
    """
    Print summary table of detection metrics for multiple methods.
    
    Args:
        method_metrics: Dictionary mapping method name to DetectionMetrics
        k_to_show: K value to show in summary
    """
    print("\n" + "="*80)
    print("DETECTION METRICS SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'AUROC':>8} {'AUPRC':>8} {'TNR@95':>8} "
          f"{'P@{k_to_show:.0%}':>10} {'R@{k_to_show:.0%}':>10} {'F1@{k_to_show:.0%}':>10}")
    print("-"*80)
    
    for method, m in sorted(method_metrics.items(), key=lambda x: x[1].auroc, reverse=True):
        prec = m.precision_at_k.get(k_to_show, 0.0)
        rec = m.recall_at_k.get(k_to_show, 0.0)
        f1 = m.f1_at_k.get(k_to_show, 0.0)
        
        print(f"{method:<20} {m.auroc:>8.3f} {m.auprc:>8.3f} {m.tnr_at_95:>8.3f} "
              f"{prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")
    
    print("="*80 + "\n")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    n = 1000
    noise_rate = 0.1
    
    # Simulated ground truth
    ground_truth = np.random.rand(n) < noise_rate
    
    # Simulated scores from different methods
    # Good method: higher scores for noisy examples
    good_scores = np.random.randn(n) * 0.5
    good_scores[ground_truth] += 2.0  # Noisy examples have higher scores
    
    # Bad method: random
    bad_scores = np.random.randn(n)
    
    print("Computing metrics for good method...")
    good_metrics = compute_detection_metrics(ground_truth, good_scores)
    
    print("Computing metrics for bad method...")
    bad_metrics = compute_detection_metrics(ground_truth, bad_scores)
    
    print_detection_summary({
        "Good Method": good_metrics,
        "Random Baseline": bad_metrics,
    })
    
    # Plot comparison
    plot_comparison(
        ground_truth,
        {"Good Method": good_scores, "Random": bad_scores},
        save_path="outputs/results/test_comparison.png"
    )

