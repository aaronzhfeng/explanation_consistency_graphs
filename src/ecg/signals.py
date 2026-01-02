"""
signals.py: ECG inconsistency signals for label noise detection.

Implements the five signal families:
1. Neighborhood surprise (graph-based)
2. NLI contradiction (explanation-label mismatch)
3. Artifact score (spurious token focus)
4. Stability signal (explanation variance)
5. Training dynamics signal (AUM-based)

Plus signal combination with reliability-adaptive aggregation.

References:
- Research proposal Section 4-5
- DeBERTa-MNLI for NLI
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
import re
from collections import Counter


@dataclass
class SignalScores:
    """Container for all ECG signal scores."""
    # Individual signals (higher = more suspicious)
    neighborhood_surprise: np.ndarray  # S_nbr
    nli_contradiction: np.ndarray  # S_nli
    artifact_score: np.ndarray  # S_art
    stability_score: np.ndarray  # S_stab (= 1 - reliability)
    dynamics_score: np.ndarray  # S_dyn (= -AUM or similar)
    
    # Confidence scores for each signal
    neighborhood_confidence: Optional[np.ndarray] = None
    nli_confidence: Optional[np.ndarray] = None
    stability_confidence: Optional[np.ndarray] = None
    dynamics_confidence: Optional[np.ndarray] = None
    
    # Combined ECG score
    ecg_score: Optional[np.ndarray] = None
    ecg_score_adaptive: Optional[np.ndarray] = None


@dataclass
class NLIResult:
    """Result from NLI model."""
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float
    contradiction_margin: float  # P_C - P_E
    
    @classmethod
    def from_probs(cls, probs: np.ndarray) -> "NLIResult":
        """Create from [entailment, neutral, contradiction] probabilities."""
        return cls(
            entailment_prob=float(probs[0]),
            neutral_prob=float(probs[1]),
            contradiction_prob=float(probs[2]),
            contradiction_margin=float(probs[2] - probs[0]),
        )


# =============================================================================
# Signal 1: Neighborhood Surprise (uses embed_graph)
# =============================================================================

def compute_neighborhood_surprise(
    graph,  # ExplanationGraph
    labels: np.ndarray,
    n_classes: int = 2,
    smoothing_epsilon: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute neighborhood surprise: -log(p(observed_label | neighbors))
    
    Uses reliability-weighted neighbor votes.
    
    Args:
        graph: ExplanationGraph from embed_graph
        labels: Observed labels
        n_classes: Number of classes
        smoothing_epsilon: Smoothing factor
        
    Returns:
        Tuple of (surprise_scores, confidence_scores)
    """
    from .embed_graph import compute_neighborhood_label_posterior
    
    posteriors = compute_neighborhood_label_posterior(
        graph, labels, n_classes, smoothing_epsilon
    )
    
    n = len(labels)
    observed_probs = posteriors[np.arange(n), labels]
    
    # Surprise = -log(p)
    surprise = -np.log(observed_probs + 1e-10)
    
    # Confidence = max neighbor similarity
    confidence = graph.similarities.max(axis=1)
    
    return surprise, confidence


# =============================================================================
# Signal 2: NLI Contradiction
# =============================================================================

class NLIScorer:
    """
    Score explanation-label contradiction using NLI models.
    
    Supports ensemble of multiple NLI models.
    """
    
    def __init__(
        self,
        model_names: List[str] = None,
        device: str = "cuda",
    ):
        if model_names is None:
            model_names = ["microsoft/deberta-v3-base-mnli"]
        
        self.model_names = model_names
        self.device = device
        self.models = []
        self.tokenizers = []
        self._initialized = False
    
    def initialize(self):
        """Load NLI models."""
        if self._initialized:
            return
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        for name in self.model_names:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            model = model.to(self.device)
            model.eval()
            
            self.tokenizers.append(tokenizer)
            self.models.append(model)
        
        self._initialized = True
        print(f"Loaded {len(self.models)} NLI models")
    
    def score_single(
        self,
        premise: str,
        hypothesis: str,
    ) -> NLIResult:
        """
        Score a single premise-hypothesis pair.
        
        Args:
            premise: The explanation text
            hypothesis: The label statement
            
        Returns:
            NLIResult with probabilities and margin
        """
        self.initialize()
        
        import torch
        import torch.nn.functional as F
        
        all_probs = []
        
        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                all_probs.append(probs)
        
        # Average across ensemble
        avg_probs = np.mean(all_probs, axis=0)
        
        return NLIResult.from_probs(avg_probs)
    
    def score_batch(
        self,
        premises: List[str],
        hypotheses: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[NLIResult]:
        """
        Score a batch of premise-hypothesis pairs.
        
        Args:
            premises: List of explanation texts
            hypotheses: List of label statements
            batch_size: Batch size for inference
            show_progress: Whether to show progress
            
        Returns:
            List of NLIResult objects
        """
        self.initialize()
        
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm
        
        n = len(premises)
        all_results = []
        
        # Process each model separately, then average
        model_probs = [[] for _ in self.models]
        
        for model_idx, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            iterator = range(0, n, batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc=f"NLI ({self.model_names[model_idx][:20]}...)")
            
            for i in iterator:
                batch_premises = premises[i:i+batch_size]
                batch_hypotheses = hypotheses[i:i+batch_size]
                
                inputs = tokenizer(
                    batch_premises, batch_hypotheses,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256,
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                    model_probs[model_idx].extend(probs)
        
        # Average across models and create results
        for i in range(n):
            probs_list = [model_probs[m][i] for m in range(len(self.models))]
            avg_probs = np.mean(probs_list, axis=0)
            all_results.append(NLIResult.from_probs(avg_probs))
        
        return all_results


def compute_nli_contradiction(
    explanations,  # List[Explanation] or List[Dict]
    observed_labels: np.ndarray,
    nli_scorer: Optional[NLIScorer] = None,
    label_hypotheses: Dict[int, str] = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NLI contradiction scores.
    
    Args:
        explanations: List of Explanation objects or dicts
        observed_labels: Observed labels (potentially noisy)
        nli_scorer: NLIScorer instance (created if None)
        label_hypotheses: Mapping from label to hypothesis string
        batch_size: Batch size for NLI
        show_progress: Whether to show progress
        
    Returns:
        Tuple of (contradiction_margins, confidence_scores)
    """
    if nli_scorer is None:
        nli_scorer = NLIScorer()
    
    # Default hypotheses for SST-2
    if label_hypotheses is None:
        label_hypotheses = {
            0: "The sentiment of the input is negative.",
            1: "The sentiment of the input is positive.",
        }
    
    # Build premises from explanations
    premises = []
    for exp in explanations:
        if hasattr(exp, 'rationale'):
            # Explanation object
            evidence_str = "; ".join(exp.evidence) if exp.evidence else ""
            counterfactual = exp.counterfactual or ""
            premise = f"Evidence: {evidence_str}. Rationale: {exp.rationale}. Counterfactual: {counterfactual}"
        else:
            # Dict
            evidence = exp.get('evidence', [])
            if isinstance(evidence, list):
                evidence_str = "; ".join(evidence)
            else:
                evidence_str = str(evidence)
            rationale = exp.get('rationale', '')
            counterfactual = exp.get('counterfactual', '')
            premise = f"Evidence: {evidence_str}. Rationale: {rationale}. Counterfactual: {counterfactual}"
        
        premises.append(premise)
    
    # Build hypotheses based on observed labels
    hypotheses = [label_hypotheses[int(label)] for label in observed_labels]
    
    # Score
    results = nli_scorer.score_batch(premises, hypotheses, batch_size, show_progress)
    
    # Extract scores
    contradiction_margins = np.array([r.contradiction_margin for r in results])
    confidence = np.array([abs(r.contradiction_margin) for r in results])
    
    return contradiction_margins, confidence


# =============================================================================
# Signal 3: Artifact Score
# =============================================================================

def compute_artifact_score_synthetic(
    explanations,  # List[Explanation] or List[Dict]
    known_artifacts: List[str],
) -> np.ndarray:
    """
    Compute artifact score for synthetic artifacts.
    
    S_art = |evidence_tokens ∩ artifact_tokens| / |evidence_tokens|
    
    Args:
        explanations: List of Explanation objects or dicts
        known_artifacts: List of known artifact tokens (e.g., ["<lbl_pos>", "<lbl_neg>"])
        
    Returns:
        Artifact scores (n,)
    """
    artifact_set = set(tok.lower() for tok in known_artifacts)
    scores = []
    
    for exp in explanations:
        if hasattr(exp, 'evidence'):
            evidence = exp.evidence
        else:
            evidence = exp.get('evidence', [])
        
        if isinstance(evidence, str):
            evidence = [evidence]
        
        # Tokenize evidence (simple whitespace split)
        evidence_text = " ".join(evidence).lower()
        evidence_tokens = set(evidence_text.split())
        
        # Count artifact overlap
        overlap = len(evidence_tokens & artifact_set)
        total = len(evidence_tokens) + 1e-6
        
        scores.append(overlap / total)
    
    return np.array(scores)


def compute_pmi_artifacts(
    texts: List[str],
    labels: np.ndarray,
    n_classes: int = 2,
    top_k: int = 200,
    alpha: float = 1.0,
    stopwords: Optional[set] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Compute PMI-based artifact tokens per class.
    
    Args:
        texts: Training texts
        labels: Training labels
        n_classes: Number of classes
        top_k: Number of top tokens per class
        alpha: Smoothing parameter
        stopwords: Set of stopwords to exclude
        
    Returns:
        Dict mapping class -> {token: PMI score}
    """
    if stopwords is None:
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        }
    
    # Count token-label co-occurrences
    token_label_counts = Counter()  # (token, label) -> count
    token_counts = Counter()  # token -> count
    label_counts = Counter()  # label -> count
    
    for text, label in zip(texts, labels):
        # Simple tokenization
        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        tokens -= stopwords
        
        for token in tokens:
            token_label_counts[(token, label)] += 1
            token_counts[token] += 1
        
        label_counts[label] += 1
    
    N = len(texts)
    V = len(token_counts)
    
    # Compute PMI per class
    pmi_per_class = {}
    
    for c in range(n_classes):
        pmi_scores = {}
        
        for token in token_counts:
            n_tc = token_label_counts.get((token, c), 0)
            n_t = token_counts[token]
            n_c = label_counts[c]
            
            # Smoothed PMI
            p_tc = (n_tc + alpha) / (n_c + alpha * V)
            p_t = (n_t + alpha) / (N + alpha * V)
            
            pmi = np.log(p_tc / p_t + 1e-10)
            pmi_scores[token] = pmi
        
        # Get top-k
        sorted_tokens = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)
        pmi_per_class[c] = dict(sorted_tokens[:top_k])
    
    return pmi_per_class


def compute_artifact_score_pmi(
    explanations,  # List[Explanation] or List[Dict]
    observed_labels: np.ndarray,
    pmi_artifacts: Dict[int, Dict[str, float]],
) -> np.ndarray:
    """
    Compute artifact score using PMI-based artifact detection.
    
    S_art = mean(max(0, PMI(token, label))) over evidence tokens
    
    Args:
        explanations: List of Explanation objects or dicts
        observed_labels: Observed labels
        pmi_artifacts: PMI artifacts per class from compute_pmi_artifacts
        
    Returns:
        Artifact scores (n,)
    """
    scores = []
    
    for exp, label in zip(explanations, observed_labels):
        if hasattr(exp, 'evidence'):
            evidence = exp.evidence
        else:
            evidence = exp.get('evidence', [])
        
        if isinstance(evidence, str):
            evidence = [evidence]
        
        # Tokenize evidence
        evidence_text = " ".join(evidence).lower()
        evidence_tokens = re.findall(r'\b\w+\b', evidence_text)
        
        if not evidence_tokens:
            scores.append(0.0)
            continue
        
        # Get PMI scores for this label
        label_pmi = pmi_artifacts.get(int(label), {})
        
        # Average positive PMI
        token_pmis = [max(0, label_pmi.get(t, 0)) for t in evidence_tokens]
        scores.append(np.mean(token_pmis))
    
    return np.array(scores)


# =============================================================================
# Signal 4: Stability Score
# =============================================================================

def compute_stability_signal(
    reliability_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute stability signal from reliability scores.
    
    S_stab = 1 - reliability
    
    Args:
        reliability_scores: Node reliability scores from stability sampling
        
    Returns:
        Tuple of (instability_scores, confidence_scores)
    """
    instability = 1 - reliability_scores
    confidence = reliability_scores  # High reliability = high confidence
    
    return instability, confidence


# =============================================================================
# Signal 5: Training Dynamics
# =============================================================================

def compute_dynamics_signal(
    aum_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute training dynamics signal from AUM.
    
    S_dyn = -AUM (lower AUM = more suspicious)
    
    Args:
        aum_scores: AUM scores from training dynamics
        
    Returns:
        Tuple of (dynamics_scores, confidence_scores)
    """
    # Negate AUM: lower AUM = higher suspicion
    dynamics = -aum_scores
    
    # Confidence = |AUM| (extreme values are more confident)
    confidence = np.abs(aum_scores)
    # Normalize to [0, 1]
    if confidence.max() > 0:
        confidence = confidence / confidence.max()
    
    return dynamics, confidence


# =============================================================================
# Signal Combination
# =============================================================================

def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to percentile ranks [0, 1].
    
    Higher rank = more suspicious.
    """
    n = len(scores)
    ranks = np.argsort(np.argsort(scores))  # Ascending ranks
    return ranks / (n - 1) if n > 1 else np.zeros(n)


def combine_signals_fixed_weights(
    signals: SignalScores,
    weights: Dict[str, float] = None,
) -> np.ndarray:
    """
    Combine signals with fixed weights.
    
    Default weights:
    - neighborhood: 0.30
    - nli: 0.30
    - artifact: 0.15
    - stability: 0.15
    - dynamics: 0.10
    
    Args:
        signals: SignalScores object
        weights: Override weights
        
    Returns:
        Combined ECG scores
    """
    if weights is None:
        weights = {
            'neighborhood': 0.30,
            'nli': 0.30,
            'artifact': 0.15,
            'stability': 0.15,
            'dynamics': 0.10,
        }
    
    # Rank normalize each signal
    s_nbr = rank_normalize(signals.neighborhood_surprise)
    s_nli = rank_normalize(signals.nli_contradiction)
    s_art = rank_normalize(signals.artifact_score)
    s_stab = rank_normalize(signals.stability_score)
    s_dyn = rank_normalize(signals.dynamics_score)
    
    # Weighted sum
    ecg = (
        weights['neighborhood'] * s_nbr +
        weights['nli'] * s_nli +
        weights['artifact'] * s_art +
        weights['stability'] * s_stab +
        weights['dynamics'] * s_dyn
    )
    
    return ecg


def combine_signals_adaptive(
    signals: SignalScores,
) -> np.ndarray:
    """
    Combine signals with reliability-adaptive weighting.
    
    Each signal is weighted by its confidence for each instance.
    
    Args:
        signals: SignalScores object with confidence scores
        
    Returns:
        Combined ECG scores (adaptive)
    """
    # Rank normalize each signal
    s_nbr = rank_normalize(signals.neighborhood_surprise)
    s_nli = rank_normalize(signals.nli_contradiction)
    s_art = rank_normalize(signals.artifact_score)
    s_stab = rank_normalize(signals.stability_score)
    s_dyn = rank_normalize(signals.dynamics_score)
    
    # Get confidence (default to 1 if not available)
    n = len(s_nbr)
    c_nbr = signals.neighborhood_confidence if signals.neighborhood_confidence is not None else np.ones(n)
    c_nli = signals.nli_confidence if signals.nli_confidence is not None else np.ones(n)
    c_stab = signals.stability_confidence if signals.stability_confidence is not None else np.ones(n)
    c_dyn = signals.dynamics_confidence if signals.dynamics_confidence is not None else np.ones(n)
    c_art = np.ones(n)  # No confidence for artifact
    
    # Normalize confidences to [0, 1]
    def normalize_conf(c):
        c = np.clip(c, 0, None)
        if c.max() > 0:
            return c / c.max()
        return c
    
    c_nbr = normalize_conf(c_nbr)
    c_nli = normalize_conf(c_nli)
    c_stab = normalize_conf(c_stab)
    c_dyn = normalize_conf(c_dyn)
    
    # Weighted combination (per-instance weights)
    numerator = (
        c_nbr * s_nbr +
        c_nli * s_nli +
        c_art * s_art +
        c_stab * s_stab +
        c_dyn * s_dyn
    )
    denominator = c_nbr + c_nli + c_art + c_stab + c_dyn
    
    ecg_adaptive = numerator / (denominator + 1e-8)
    
    return ecg_adaptive


def apply_dynamics_veto(
    ecg_scores: np.ndarray,
    dynamics_scores: np.ndarray,
    veto_threshold_percentile: float = 75,
    veto_strength: float = 0.5,
) -> np.ndarray:
    """
    Apply training dynamics veto to protect hard-but-correct examples.
    
    If an example has high AUM (consistently learnable) but high ECG score,
    reduce the ECG score.
    
    Args:
        ecg_scores: Combined ECG scores
        dynamics_scores: Training dynamics scores (higher = more suspicious)
        veto_threshold_percentile: Percentile threshold for "learnable"
        veto_strength: How much to reduce score (0-1)
        
    Returns:
        Adjusted ECG scores
    """
    # Identify "consistently learnable" examples (low dynamics score)
    threshold = np.percentile(dynamics_scores, 100 - veto_threshold_percentile)
    learnable_mask = dynamics_scores < threshold
    
    # Reduce ECG score for learnable examples
    adjusted = ecg_scores.copy()
    adjusted[learnable_mask] *= (1 - veto_strength)
    
    return adjusted


def compute_all_signals(
    graph,  # ExplanationGraph
    explanations,  # List[Explanation]
    observed_labels: np.ndarray,
    reliability_scores: np.ndarray,
    aum_scores: np.ndarray,
    known_artifacts: Optional[List[str]] = None,
    nli_scorer: Optional[NLIScorer] = None,
    n_classes: int = 2,
    show_progress: bool = True,
) -> SignalScores:
    """
    Compute all ECG signals.
    
    Args:
        graph: ExplanationGraph from embed_graph
        explanations: List of Explanation objects
        observed_labels: Observed labels
        reliability_scores: Node reliability from stability sampling
        aum_scores: AUM scores from training dynamics
        known_artifacts: Known artifact tokens (for synthetic)
        nli_scorer: NLIScorer instance
        n_classes: Number of classes
        show_progress: Whether to show progress
        
    Returns:
        SignalScores with all signals and combined scores
    """
    print("Computing ECG signals...")
    
    # Signal 1: Neighborhood surprise
    print("  Computing neighborhood surprise...")
    s_nbr, c_nbr = compute_neighborhood_surprise(graph, observed_labels, n_classes)
    
    # Signal 2: NLI contradiction
    print("  Computing NLI contradiction...")
    s_nli, c_nli = compute_nli_contradiction(
        explanations, observed_labels, nli_scorer,
        batch_size=32, show_progress=show_progress
    )
    
    # Signal 3: Artifact score
    print("  Computing artifact score...")
    if known_artifacts:
        s_art = compute_artifact_score_synthetic(explanations, known_artifacts)
    else:
        s_art = np.zeros(len(explanations))  # Placeholder if no artifacts known
    
    # Signal 4: Stability
    print("  Computing stability signal...")
    s_stab, c_stab = compute_stability_signal(reliability_scores)
    
    # Signal 5: Training dynamics
    print("  Computing dynamics signal...")
    s_dyn, c_dyn = compute_dynamics_signal(aum_scores)
    
    # Create SignalScores
    signals = SignalScores(
        neighborhood_surprise=s_nbr,
        nli_contradiction=s_nli,
        artifact_score=s_art,
        stability_score=s_stab,
        dynamics_score=s_dyn,
        neighborhood_confidence=c_nbr,
        nli_confidence=c_nli,
        stability_confidence=c_stab,
        dynamics_confidence=c_dyn,
    )
    
    # Combine signals
    print("  Combining signals...")
    signals.ecg_score = combine_signals_fixed_weights(signals)
    signals.ecg_score_adaptive = combine_signals_adaptive(signals)
    
    print("Done!")
    
    return signals


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    n = 100
    
    # Simulate signals
    s_nbr = np.random.randn(n)
    s_nli = np.random.randn(n)
    s_art = np.random.rand(n) * 0.5
    s_stab = np.random.rand(n)
    s_dyn = np.random.randn(n)
    
    signals = SignalScores(
        neighborhood_surprise=s_nbr,
        nli_contradiction=s_nli,
        artifact_score=s_art,
        stability_score=s_stab,
        dynamics_score=s_dyn,
        neighborhood_confidence=np.random.rand(n),
        nli_confidence=np.random.rand(n),
        stability_confidence=np.random.rand(n),
        dynamics_confidence=np.random.rand(n),
    )
    
    # Combine with fixed weights
    ecg_fixed = combine_signals_fixed_weights(signals)
    print(f"ECG (fixed weights) range: [{ecg_fixed.min():.3f}, {ecg_fixed.max():.3f}]")
    
    # Combine with adaptive weights
    ecg_adaptive = combine_signals_adaptive(signals)
    print(f"ECG (adaptive) range: [{ecg_adaptive.min():.3f}, {ecg_adaptive.max():.3f}]")
    
    # Apply veto
    ecg_vetoed = apply_dynamics_veto(ecg_fixed, s_dyn)
    print(f"ECG (with veto) range: [{ecg_vetoed.min():.3f}, {ecg_vetoed.max():.3f}]")

