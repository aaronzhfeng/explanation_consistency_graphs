"""
embed_graph.py: Graph construction over explanation embeddings.

Handles:
- Embedding explanations with sentence transformers
- Building kNN graphs with FAISS
- Reliability-weighted edge construction (WANN-style)
- Outlier/OOD detection
- Multi-view graph construction

References:
- WANN: https://github.com/francescodisalvo05/wann-noisy-labels
- Neural Relation Graph: https://github.com/snu-mllab/Neural-Relation-Graph
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ExplanationGraph:
    """Container for explanation graph structure."""
    # Node data
    n_nodes: int
    embeddings: np.ndarray  # (n, dim) - explanation embeddings
    reliability: np.ndarray  # (n,) - node reliability scores [0, 1]
    
    # Edge data
    neighbors: np.ndarray  # (n, k) - indices of k nearest neighbors
    similarities: np.ndarray  # (n, k) - cosine similarities to neighbors
    weights: np.ndarray  # (n, k) - normalized edge weights (reliability-weighted)
    
    # Graph metadata
    k: int
    similarity_threshold: float
    use_mutual_knn: bool
    
    # Outlier scores
    outlier_scores: Optional[np.ndarray] = None  # (n,) - outlier/OOD scores


@dataclass
class MultiViewGraph:
    """Container for multi-view graph (explanation + classifier views)."""
    explanation_graph: ExplanationGraph
    classifier_graph: Optional[ExplanationGraph] = None
    
    # Combined scores
    combined_neighbors: Optional[np.ndarray] = None


# =============================================================================
# Embedding Functions
# =============================================================================

def embed_explanations(
    explanations: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    include_counterfactual: bool = True,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Embed explanations using sentence transformers.
    
    Args:
        explanations: List of explanation dicts with 'evidence', 'rationale', 'counterfactual'
        model_name: Sentence transformer model name
        include_counterfactual: Whether to include counterfactual in embedding
        batch_size: Batch size for encoding
        
    Returns:
        Normalized embeddings (n, dim)
    """
    from sentence_transformers import SentenceTransformer
    
    # Build canonical strings
    texts = []
    for exp in explanations:
        evidence = exp.get('evidence', [])
        if isinstance(evidence, list):
            evidence_str = "; ".join(evidence)
        else:
            evidence_str = str(evidence)
        
        rationale = exp.get('rationale', '')
        
        parts = [f"Evidence: {evidence_str}", f"Rationale: {rationale}"]
        
        if include_counterfactual:
            counterfactual = exp.get('counterfactual', '')
            if counterfactual:
                parts.append(f"Counterfactual: {counterfactual}")
        
        text = " | ".join(parts)
        texts.append(text)
    
    # Encode
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize
    )
    
    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


# =============================================================================
# kNN Graph Construction
# =============================================================================

def build_knn_graph_faiss(
    embeddings: np.ndarray,
    k: int = 15,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build kNN graph using FAISS.
    
    Args:
        embeddings: Normalized embeddings (n, dim)
        k: Number of neighbors
        use_gpu: Whether to use GPU
        
    Returns:
        Tuple of (neighbor_indices, similarities)
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Install faiss: pip install faiss-gpu (or faiss-cpu)")
    
    n, dim = embeddings.shape
    
    # Ensure float32
    embeddings = embeddings.astype(np.float32)
    
    # Build index for inner product (= cosine for normalized vectors)
    index = faiss.IndexFlatIP(dim)
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(embeddings)
    
    # Search for k+1 neighbors (first is self)
    similarities, indices = index.search(embeddings, k + 1)
    
    # Remove self (first neighbor)
    neighbor_indices = indices[:, 1:]
    neighbor_similarities = similarities[:, 1:]
    
    return neighbor_indices, neighbor_similarities


def build_knn_graph_sklearn(
    embeddings: np.ndarray,
    k: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build kNN graph using sklearn (fallback if FAISS unavailable).
    
    Args:
        embeddings: Normalized embeddings (n, dim)
        k: Number of neighbors
        
    Returns:
        Tuple of (neighbor_indices, similarities)
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Use cosine metric
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    
    # Remove self
    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]
    
    # Convert distances to similarities (cosine distance = 1 - cosine similarity)
    neighbor_similarities = 1 - neighbor_distances
    
    return neighbor_indices, neighbor_similarities


def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 15,
    use_faiss: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build kNN graph.
    
    Args:
        embeddings: Normalized embeddings
        k: Number of neighbors
        use_faiss: Whether to try FAISS first
        
    Returns:
        Tuple of (neighbor_indices, similarities)
    """
    if use_faiss:
        try:
            return build_knn_graph_faiss(embeddings, k, use_gpu=True)
        except (ImportError, Exception) as e:
            print(f"FAISS failed ({e}), falling back to sklearn")
    
    return build_knn_graph_sklearn(embeddings, k)


# =============================================================================
# Reliability-Weighted Edges (WANN-style)
# =============================================================================

def compute_reliability_weighted_edges(
    neighbors: np.ndarray,
    similarities: np.ndarray,
    reliability: np.ndarray,
    temperature: float = 0.07,
    similarity_threshold: float = 0.35,
) -> np.ndarray:
    """
    Compute reliability-weighted edge weights.
    
    Weight = softmax(similarity / temperature) * neighbor_reliability
    
    Adapted from WANN and ECG proposal.
    
    Args:
        neighbors: Neighbor indices (n, k)
        similarities: Cosine similarities (n, k)
        reliability: Node reliability scores (n,)
        temperature: Softmax temperature
        similarity_threshold: Minimum similarity for edges
        
    Returns:
        Normalized edge weights (n, k)
    """
    n, k = neighbors.shape
    
    # Apply similarity threshold
    valid_mask = similarities >= similarity_threshold
    
    # Get neighbor reliability scores
    neighbor_reliability = reliability[neighbors]  # (n, k)
    
    # Compute unnormalized weights: exp(sim/temp) * reliability
    exp_sim = np.exp(similarities / temperature)
    exp_sim = exp_sim * valid_mask  # Zero out edges below threshold
    
    unnorm_weights = exp_sim * neighbor_reliability
    
    # Normalize per node
    weight_sums = unnorm_weights.sum(axis=1, keepdims=True)
    weights = unnorm_weights / (weight_sums + 1e-8)
    
    return weights


def apply_mutual_knn(
    neighbors: np.ndarray,
    similarities: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter to mutual kNN edges only.
    
    Keep edge (i,j) only if j is in neighbors of i AND i is in neighbors of j.
    
    Args:
        neighbors: Neighbor indices (n, k)
        similarities: Similarities (n, k)
        
    Returns:
        Masked neighbors and similarities (non-mutual edges zeroed out)
    """
    n, k = neighbors.shape
    
    # Build reverse neighbor lookup
    mutual_mask = np.zeros((n, k), dtype=bool)
    
    for i in range(n):
        for j_idx, j in enumerate(neighbors[i]):
            # Check if i is in j's neighbors
            if i in neighbors[j]:
                mutual_mask[i, j_idx] = True
    
    # Apply mask
    masked_similarities = similarities * mutual_mask
    
    return neighbors, masked_similarities


# =============================================================================
# Outlier Detection
# =============================================================================

def compute_outlier_scores(
    similarities: np.ndarray,
    k_outlier: int = 5,
) -> np.ndarray:
    """
    Compute outlier/OOD scores based on neighborhood similarity.
    
    Low average similarity to neighbors = potential outlier.
    
    Args:
        similarities: Similarities to neighbors (n, k)
        k_outlier: Number of nearest neighbors to consider for outlier score
        
    Returns:
        Outlier scores (n,) - higher = more likely outlier
    """
    # Use top-k similarities (closest neighbors)
    k = min(k_outlier, similarities.shape[1])
    top_k_sims = np.sort(similarities, axis=1)[:, -k:]
    
    # Mean similarity to k nearest neighbors
    mean_sim = top_k_sims.mean(axis=1)
    
    # Outlier score = 1 - mean_sim (higher = less similar = outlier)
    outlier_scores = 1 - mean_sim
    
    return outlier_scores


# =============================================================================
# Full Graph Construction
# =============================================================================

def build_explanation_graph(
    embeddings: np.ndarray,
    reliability: np.ndarray,
    k: int = 15,
    temperature: float = 0.07,
    similarity_threshold: float = 0.35,
    use_mutual_knn: bool = False,
    compute_outliers: bool = True,
) -> ExplanationGraph:
    """
    Build complete explanation graph with reliability-weighted edges.
    
    Args:
        embeddings: Explanation embeddings (n, dim) - assumed normalized
        reliability: Node reliability scores (n,)
        k: Number of neighbors
        temperature: Softmax temperature for edge weights
        similarity_threshold: Minimum similarity for edges
        use_mutual_knn: Whether to use mutual kNN edges only
        compute_outliers: Whether to compute outlier scores
        
    Returns:
        ExplanationGraph dataclass
    """
    n = len(embeddings)
    
    # Ensure normalized
    embeddings = normalize_embeddings(embeddings)
    
    # Build kNN graph
    neighbors, similarities = build_knn_graph(embeddings, k)
    
    # Apply mutual kNN if requested
    if use_mutual_knn:
        neighbors, similarities = apply_mutual_knn(neighbors, similarities)
    
    # Compute reliability-weighted edges
    weights = compute_reliability_weighted_edges(
        neighbors, similarities, reliability,
        temperature=temperature,
        similarity_threshold=similarity_threshold,
    )
    
    # Compute outlier scores
    outlier_scores = None
    if compute_outliers:
        outlier_scores = compute_outlier_scores(similarities)
    
    return ExplanationGraph(
        n_nodes=n,
        embeddings=embeddings,
        reliability=reliability,
        neighbors=neighbors,
        similarities=similarities,
        weights=weights,
        k=k,
        similarity_threshold=similarity_threshold,
        use_mutual_knn=use_mutual_knn,
        outlier_scores=outlier_scores,
    )


# =============================================================================
# Multi-View Graph
# =============================================================================

def build_multiview_graph(
    explanation_embeddings: np.ndarray,
    classifier_embeddings: np.ndarray,
    reliability: np.ndarray,
    k: int = 15,
    temperature: float = 0.07,
    similarity_threshold: float = 0.35,
) -> MultiViewGraph:
    """
    Build multi-view graph combining explanation and classifier embeddings.
    
    Args:
        explanation_embeddings: Embeddings from explanations
        classifier_embeddings: [CLS] embeddings from classifier
        reliability: Node reliability scores
        k: Number of neighbors
        temperature: Softmax temperature
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        MultiViewGraph with both views
    """
    # Build explanation graph
    exp_graph = build_explanation_graph(
        explanation_embeddings, reliability,
        k=k, temperature=temperature,
        similarity_threshold=similarity_threshold,
    )
    
    # Build classifier graph
    cls_graph = build_explanation_graph(
        classifier_embeddings, reliability,
        k=k, temperature=temperature,
        similarity_threshold=similarity_threshold,
    )
    
    return MultiViewGraph(
        explanation_graph=exp_graph,
        classifier_graph=cls_graph,
    )


# =============================================================================
# Neighborhood Label Statistics
# =============================================================================

def compute_neighborhood_label_posterior(
    graph: ExplanationGraph,
    labels: np.ndarray,
    n_classes: int = 2,
    smoothing_epsilon: float = 1e-3,
) -> np.ndarray:
    """
    Compute weighted label posterior from neighbors.
    
    p(c) = sum_{j in N(i)} w_ij * 1[y_j = c]
    
    Args:
        graph: ExplanationGraph
        labels: Observed labels
        n_classes: Number of classes
        smoothing_epsilon: Smoothing factor
        
    Returns:
        Label posterior for each node (n, n_classes)
    """
    n = graph.n_nodes
    posteriors = np.zeros((n, n_classes))
    
    for i in range(n):
        for j_idx, j in enumerate(graph.neighbors[i]):
            weight = graph.weights[i, j_idx]
            posteriors[i, labels[j]] += weight
    
    # Apply smoothing
    posteriors = (posteriors + smoothing_epsilon) / (1 + n_classes * smoothing_epsilon)
    
    return posteriors


def compute_neighborhood_surprise(
    graph: ExplanationGraph,
    labels: np.ndarray,
    n_classes: int = 2,
    smoothing_epsilon: float = 1e-3,
) -> np.ndarray:
    """
    Compute neighborhood surprise: -log(p(observed_label | neighbors))
    
    Higher = observed label is unlikely given neighbors = more suspicious.
    
    Args:
        graph: ExplanationGraph
        labels: Observed labels
        n_classes: Number of classes
        smoothing_epsilon: Smoothing factor
        
    Returns:
        Neighborhood surprise scores (n,)
    """
    posteriors = compute_neighborhood_label_posterior(
        graph, labels, n_classes, smoothing_epsilon
    )
    
    # Get probability of observed label
    n = len(labels)
    observed_probs = posteriors[np.arange(n), labels]
    
    # Surprise = -log(p)
    surprise = -np.log(observed_probs + 1e-10)
    
    return surprise


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    n = 500
    dim = 384
    k = 15
    n_classes = 2
    
    # Simulated data
    embeddings = np.random.randn(n, dim).astype(np.float32)
    embeddings = normalize_embeddings(embeddings)
    
    reliability = np.random.rand(n)  # Random reliability scores
    labels = np.random.randint(0, n_classes, n)
    
    print("Building explanation graph...")
    graph = build_explanation_graph(
        embeddings=embeddings,
        reliability=reliability,
        k=k,
        similarity_threshold=0.0,  # No threshold for random data
    )
    
    print(f"Graph: {graph.n_nodes} nodes, k={graph.k}")
    print(f"Weights shape: {graph.weights.shape}")
    print(f"Weights sum per node: {graph.weights.sum(axis=1)[:5]}")  # Should be ~1
    
    print("\nComputing neighborhood surprise...")
    surprise = compute_neighborhood_surprise(graph, labels)
    print(f"Surprise range: [{surprise.min():.3f}, {surprise.max():.3f}]")
    
    if graph.outlier_scores is not None:
        print(f"Outlier score range: [{graph.outlier_scores.min():.3f}, {graph.outlier_scores.max():.3f}]")

