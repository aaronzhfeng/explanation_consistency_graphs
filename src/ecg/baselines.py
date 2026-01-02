"""
baselines.py: Baseline methods for label noise detection.

Implements baselines from literature:
- Cleanlab (Confident Learning)
- Loss-based ranking
- Margin-based ranking  
- AUM (Area Under the Margin)
- Entropy-based ranking
- Neural Relation Graph
- WANN-style kNN

References:
- Cleanlab: https://github.com/cleanlab/cleanlab
- AUM: https://github.com/asappresearch/aum
- Neural Relation Graph: https://github.com/snu-mllab/Neural-Relation-Graph
- WANN: https://github.com/francescodisalvo05/wann-noisy-labels
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors


@dataclass
class BaselineScores:
    """Container for all baseline scores."""
    # Higher score = more likely to be mislabeled
    cleanlab: Optional[np.ndarray] = None
    loss: Optional[np.ndarray] = None
    margin: Optional[np.ndarray] = None
    aum: Optional[np.ndarray] = None
    entropy: Optional[np.ndarray] = None
    llm_mismatch: Optional[np.ndarray] = None
    input_knn: Optional[np.ndarray] = None
    classifier_knn: Optional[np.ndarray] = None
    random: Optional[np.ndarray] = None
    
    # Neural Relation Graph
    nrg: Optional[np.ndarray] = None
    
    # WANN
    wann: Optional[np.ndarray] = None


# =============================================================================
# Cleanlab Baseline
# =============================================================================

def cleanlab_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """
    Compute Cleanlab confident learning scores.
    
    Higher score = more likely to be mislabeled.
    
    Args:
        labels: Ground truth labels (may be noisy)
        pred_probs: Predicted probabilities from classifier
        
    Returns:
        Suspiciousness scores (higher = more suspicious)
    """
    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        raise ImportError("Install cleanlab: pip install cleanlab")
    
    # Get label quality scores (higher = more likely correct)
    quality_scores = get_label_quality_scores(
        labels=labels,
        pred_probs=pred_probs,
        method="self_confidence",
    )
    
    # Invert: higher = more suspicious
    suspicion_scores = 1 - quality_scores
    
    return suspicion_scores


def cleanlab_ranking(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """
    Get Cleanlab's ranking of suspicious examples.
    
    Args:
        labels: Ground truth labels (may be noisy)
        pred_probs: Predicted probabilities from classifier
        
    Returns:
        Indices ranked by suspiciousness (most suspicious first)
    """
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        raise ImportError("Install cleanlab: pip install cleanlab")
    
    ranked_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    
    return ranked_indices


# =============================================================================
# NRG-style Baseline Scores (adapted from Neural-Relation-Graph)
# =============================================================================

def loss_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """
    Loss-based suspicion scores.
    
    Higher loss = more suspicious.
    
    Adapted from NRG: metric.py
    """
    labels_tensor = torch.tensor(labels)
    probs_tensor = torch.tensor(pred_probs)
    
    # Cross-entropy loss per example
    assigned_probs = torch.gather(probs_tensor, 1, labels_tensor.unsqueeze(1)).squeeze()
    losses = -torch.log(assigned_probs + 1e-6)
    
    return losses.numpy()


def margin_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """
    Margin-based suspicion scores.
    
    Lower margin = more suspicious.
    Returns negative margin so higher = more suspicious.
    
    Adapted from NRG: metric.py
    """
    labels_tensor = torch.tensor(labels)
    probs_tensor = torch.tensor(pred_probs)
    
    # Probability of assigned label
    assigned_probs = torch.gather(probs_tensor, 1, labels_tensor.unsqueeze(1)).squeeze()
    
    # Top-2 probabilities
    top2_probs, _ = probs_tensor.topk(2, dim=1)
    
    # Margin: assigned - highest other
    margin = assigned_probs - top2_probs[:, 0]
    margin[margin == 0] = (assigned_probs - top2_probs[:, 1])[margin == 0]
    
    # Negate so higher = more suspicious
    return -margin.numpy()


def entropy_scores(
    pred_probs: np.ndarray,
) -> np.ndarray:
    """
    Entropy-based scores.
    
    Higher entropy = more uncertain = potentially suspicious.
    
    Adapted from NRG: metric.py
    """
    probs_tensor = torch.tensor(pred_probs)
    entropy = -(probs_tensor * torch.log(probs_tensor + 1e-6)).sum(dim=-1)
    
    return entropy.numpy()


def least_confidence_scores(
    pred_probs: np.ndarray,
) -> np.ndarray:
    """
    Least confidence scores.
    
    Lower max probability = more suspicious.
    
    Adapted from NRG: metric.py
    """
    max_probs = pred_probs.max(axis=1)
    return -max_probs


# =============================================================================
# kNN-based Baselines
# =============================================================================

def knn_disagreement_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
    weighted: bool = False,
) -> np.ndarray:
    """
    kNN label disagreement scores.
    
    Higher = more disagreement with neighbors = more suspicious.
    
    Args:
        embeddings: Feature embeddings (n_examples, dim)
        labels: Labels for each example
        k: Number of neighbors
        weighted: Whether to weight by distance
        
    Returns:
        Disagreement scores
    """
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Build kNN index
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    
    # Exclude self (first neighbor)
    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]
    
    # Get neighbor labels
    neighbor_labels = labels[neighbor_indices]
    
    if weighted:
        # Weight by similarity (1 - cosine distance)
        weights = 1 - neighbor_distances
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
        
        # Weighted agreement
        agreement = (neighbor_labels == labels[:, np.newaxis]).astype(float)
        weighted_agreement = (agreement * weights).sum(axis=1)
    else:
        # Simple majority voting agreement
        weighted_agreement = (neighbor_labels == labels[:, np.newaxis]).mean(axis=1)
    
    # Disagreement = 1 - agreement
    return 1 - weighted_agreement


# =============================================================================
# LLM Mismatch Baseline
# =============================================================================

def llm_mismatch_scores(
    observed_labels: np.ndarray,
    llm_predicted_labels: np.ndarray,
    llm_confidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    LLM label mismatch scores.
    
    Higher = LLM disagrees with observed label = more suspicious.
    
    Args:
        observed_labels: Observed (potentially noisy) labels
        llm_predicted_labels: Labels predicted by LLM
        llm_confidence: Optional confidence scores from LLM
        
    Returns:
        Mismatch scores
    """
    mismatch = (observed_labels != llm_predicted_labels).astype(float)
    
    if llm_confidence is not None:
        # Weight by LLM confidence (more confident mismatch = more suspicious)
        # Normalize confidence to [0, 1] if needed
        conf_normalized = llm_confidence / 100.0 if llm_confidence.max() > 1 else llm_confidence
        scores = mismatch * conf_normalized
    else:
        scores = mismatch
    
    return scores


# =============================================================================
# Neural Relation Graph (simplified)
# =============================================================================

def nrg_kernel(
    features: np.ndarray,
    features_t: np.ndarray,
    probs: np.ndarray,
    probs_t: np.ndarray,
    labels: np.ndarray,
    labels_t: np.ndarray,
    kernel_type: str = 'cos_p',
) -> np.ndarray:
    """
    Compute NRG kernel between features.
    
    Adapted from NRG: relation.py
    
    Args:
        features: Source features (n, d)
        features_t: Target features (m, d)
        probs: Source probabilities (n, c)
        probs_t: Target probabilities (m, c)
        labels: Source labels (n,)
        labels_t: Target labels (m,)
        kernel_type: 'cos' or 'cos_p' (cosine × probability similarity)
        
    Returns:
        Kernel matrix (n, m)
    """
    # Normalize features
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    features_t = features_t / (np.linalg.norm(features_t, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    dot = np.matmul(features, features_t.T)
    dot = np.clip(dot, 0, None)  # Keep positive
    
    if kernel_type == 'cos_p':
        # Multiply by probability similarity
        prob_sim = np.matmul(probs, probs_t.T)
        dot = dot * prob_sim
    
    # Multiply by label agreement coefficient
    label_coef = 2 * (labels[:, np.newaxis] == labels_t[np.newaxis, :]).astype(float) - 1
    relation = label_coef * dot
    
    return relation


def nrg_scores(
    features: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    kernel_type: str = 'cos_p',
    threshold: float = 0.03,
) -> np.ndarray:
    """
    Compute Neural Relation Graph suspicion scores.
    
    Lower (more negative) NRG score = more suspicious.
    We return negative so higher = more suspicious.
    
    Adapted from NRG: detect.py
    
    Args:
        features: Feature embeddings (n, d)
        probs: Predicted probabilities (n, c)
        labels: Observed labels (n,)
        kernel_type: Kernel type
        threshold: Edge threshold
        
    Returns:
        Suspicion scores (higher = more suspicious)
    """
    n = len(features)
    
    # Compute self-relation graph
    relation = nrg_kernel(
        features, features,
        probs, probs,
        labels, labels,
        kernel_type=kernel_type,
    )
    
    # Zero out small edges
    relation_masked = relation * (np.abs(relation) > threshold)
    
    # Sum of relations (positive = agrees with neighbors, negative = disagrees)
    scores = relation_masked.sum(axis=1)
    
    # Normalize
    scores = scores / (np.abs(scores).max() + 1e-8)
    
    # Negate so higher = more suspicious
    return -scores


# =============================================================================
# Random Baseline
# =============================================================================

def random_scores(
    n_examples: int,
    seed: int = 42,
) -> np.ndarray:
    """Random baseline scores."""
    rng = np.random.RandomState(seed)
    return rng.rand(n_examples)


# =============================================================================
# Compute All Baselines
# =============================================================================

def compute_all_baselines(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    features: Optional[np.ndarray] = None,
    input_embeddings: Optional[np.ndarray] = None,
    llm_predicted_labels: Optional[np.ndarray] = None,
    llm_confidence: Optional[np.ndarray] = None,
    k: int = 15,
    seed: int = 42,
) -> BaselineScores:
    """
    Compute all baseline suspicion scores.
    
    Args:
        labels: Observed labels
        pred_probs: Classifier predicted probabilities
        features: Classifier [CLS] embeddings (for NRG, classifier kNN)
        input_embeddings: Sentence embeddings of input text (for input kNN)
        llm_predicted_labels: LLM predicted labels (for LLM mismatch)
        llm_confidence: LLM confidence scores
        k: Number of neighbors for kNN baselines
        seed: Random seed
        
    Returns:
        BaselineScores with all computed scores
    """
    n = len(labels)
    
    scores = BaselineScores()
    
    # Cleanlab
    try:
        scores.cleanlab = cleanlab_scores(labels, pred_probs)
    except Exception as e:
        print(f"Warning: Cleanlab failed: {e}")
    
    # Loss-based
    scores.loss = loss_scores(labels, pred_probs)
    
    # Margin-based
    scores.margin = margin_scores(labels, pred_probs)
    
    # AUM (use margin as proxy for single-checkpoint)
    scores.aum = -margin_scores(labels, pred_probs)  # Already negated in margin_scores
    
    # Entropy
    scores.entropy = entropy_scores(pred_probs)
    
    # Random
    scores.random = random_scores(n, seed)
    
    # LLM mismatch
    if llm_predicted_labels is not None:
        scores.llm_mismatch = llm_mismatch_scores(
            labels, llm_predicted_labels, llm_confidence
        )
    
    # kNN baselines
    if input_embeddings is not None:
        scores.input_knn = knn_disagreement_scores(
            input_embeddings, labels, k=k
        )
    
    if features is not None:
        scores.classifier_knn = knn_disagreement_scores(
            features, labels, k=k
        )
        
        # NRG
        scores.nrg = nrg_scores(
            features, pred_probs, labels
        )
    
    return scores


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test with random data
    n = 1000
    n_classes = 2
    
    np.random.seed(42)
    
    # Simulated data
    labels = np.random.randint(0, n_classes, n)
    pred_probs = np.random.dirichlet(np.ones(n_classes), n)
    features = np.random.randn(n, 768)
    
    # Simulate some noisy labels
    noisy_mask = np.random.rand(n) < 0.1
    noisy_labels = labels.copy()
    noisy_labels[noisy_mask] = 1 - noisy_labels[noisy_mask]
    
    print("Computing baselines...")
    scores = compute_all_baselines(
        labels=noisy_labels,
        pred_probs=pred_probs,
        features=features,
    )
    
    print(f"Loss scores range: [{scores.loss.min():.3f}, {scores.loss.max():.3f}]")
    print(f"Margin scores range: [{scores.margin.min():.3f}, {scores.margin.max():.3f}]")
    print(f"Entropy scores range: [{scores.entropy.min():.3f}, {scores.entropy.max():.3f}]")
    
    if scores.nrg is not None:
        print(f"NRG scores range: [{scores.nrg.min():.3f}, {scores.nrg.max():.3f}]")

