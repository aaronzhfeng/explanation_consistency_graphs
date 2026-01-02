# ECG Module Reference

> Detailed API documentation for all modules

---

## Module Overview

| Module | Lines | Primary Exports |
|--------|-------|-----------------|
| `data.py` | 423 | `create_noisy_dataset`, `NoiseConfig`, `NoisyDataset` |
| `train_classifier.py` | 511 | `train_classifier`, `TrainingConfig`, `TrainingDynamics` |
| `explain_llm.py` | 867 | `ExplanationGenerator`, `Explanation`, `generate_with_stability` |
| `embed_graph.py` | 558 | `build_explanation_graph`, `ExplanationGraph` |
| `signals.py` | 814 | `compute_all_signals`, `SignalScores`, `NLIScorer` |
| `baselines.py` | 515 | `compute_all_baselines`, `BaselineScores` |
| `eval.py` | 536 | `compute_detection_metrics`, `DetectionMetrics` |
| `clean.py` | 490 | `clean_dataset`, `CleaningConfig`, `CleaningResult` |

---

## `data.py` — Data Loading & Noise Injection

### Classes

#### `NoiseConfig`
```python
@dataclass
class NoiseConfig:
    noise_type: str = "uniform"  # "uniform", "artifact_aligned", "none"
    noise_rate: float = 0.10
    seed: int = 42
```

#### `ArtifactConfig`
```python
@dataclass
class ArtifactConfig:
    positive_token: str = "<lbl_pos>"
    negative_token: str = "<lbl_neg>"
    injection_rate: float = 1.0  # Fraction of noisy examples to add artifact
```

#### `NoisyDataset`
```python
@dataclass
class NoisyDataset:
    dataset: Dataset              # HuggingFace dataset with 'sentence', 'label'
    noisy_labels: np.ndarray      # Potentially corrupted labels
    original_labels: np.ndarray   # Ground truth labels
    is_noisy: np.ndarray          # Boolean mask: True = label was flipped
    noise_indices: np.ndarray     # Indices of noisy examples
    noise_config: NoiseConfig
```

### Functions

#### `load_sst2()`
```python
def load_sst2() -> DatasetDict:
    """Load SST-2 from HuggingFace."""
```

#### `create_noisy_dataset()`
```python
def create_noisy_dataset(
    n_train: int = 25000,
    noise_config: NoiseConfig = None,
    artifact_config: ArtifactConfig = None,
    seed: int = 42,
) -> NoisyDataset:
    """
    Create noisy SST-2 dataset for experiments.
    
    Noise types:
    - "uniform": Random label flips
    - "artifact_aligned": Flip label AND add spurious token
    - "none": No noise (for debugging)
    """
```

---

## `train_classifier.py` — Classifier Training

### Classes

#### `TrainingConfig`
```python
@dataclass
class TrainingConfig:
    model_name: str = "roberta-base"
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_dir: str = "outputs/checkpoints"
    seed: int = 42
    compute_training_dynamics: bool = True
```

#### `TrainingDynamics`
```python
@dataclass
class TrainingDynamics:
    aum_scores: np.ndarray     # (n_examples,) - Area Under Margin
    losses: np.ndarray         # (n_examples,) - Final cross-entropy loss
    margins: np.ndarray        # (n_examples,) - Logit margin
    probabilities: np.ndarray  # (n_examples, n_classes)
    entropy: np.ndarray        # (n_examples,) - Prediction entropy
    n_examples: int
```

### Functions

#### `train_classifier()`
```python
def train_classifier(
    train_dataset: Dataset,
    val_dataset: Dataset = None,
    config: TrainingConfig = None,
    return_dynamics: bool = True,
) -> Tuple[nn.Module, TrainingDynamics, Dict]:
    """
    Fine-tune RoBERTa classifier on (noisy) training data.
    
    Returns:
        model: Trained classifier
        dynamics: TrainingDynamics (if return_dynamics=True)
        results: Dict with train_loss, val_accuracy
    """
```

#### `cross_validate_predictions()`
```python
def cross_validate_predictions(
    dataset: Dataset,
    n_folds: int = 5,
    config: TrainingConfig = None,
) -> np.ndarray:
    """
    Get out-of-sample predicted probabilities via K-fold CV.
    Required for Cleanlab baseline.
    
    Returns:
        oos_probs: (n_examples, n_classes) out-of-sample predictions
    """
```

#### `get_cls_embeddings()`
```python
def get_cls_embeddings(
    model: nn.Module,
    dataset: Dataset,
    tokenizer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract [CLS] embeddings for multi-view graph construction.
    
    Returns:
        embeddings: (n_examples, hidden_dim)
    """
```

---

## `explain_llm.py` — LLM Explanation Generation

### Classes

#### `Explanation`
```python
@dataclass
class Explanation:
    pred_label: str              # "POSITIVE" or "NEGATIVE"
    evidence: List[str]          # 1-3 exact substrings from input
    rationale: str               # ≤25 tokens, no label words
    counterfactual: Optional[str]
    confidence: int              # 0-100
    raw_output: Optional[str]    # Raw LLM output
    parse_success: bool          # Whether JSON parsed successfully
```

#### `StabilityMetrics`
```python
@dataclass
class StabilityMetrics:
    label_agreement: float       # Fraction with same label across samples
    evidence_jaccard: float      # Average Jaccard of evidence sets
    rationale_similarity: float  # Average cosine of rationale embeddings
    reliability_score: float     # r_i = mean of above three
    n_samples: int
    dominant_label: str
```

#### `ExplanationWithStability`
```python
@dataclass
class ExplanationWithStability:
    primary: Explanation         # Main explanation (temp=0)
    stability: StabilityMetrics  # Computed from samples
    samples: List[Explanation]   # All M samples
```

#### `ExplanationGenerator`
```python
class ExplanationGenerator:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_vllm: bool = True,
        use_outlines: bool = False,
        temperature: float = 0.0,
        max_new_tokens: int = 150,
    ):
        """
        LLM-based explanation generator.
        
        Backends:
        - vLLM (preferred for batching)
        - Outlines (for schema-constrained decoding)
        - HuggingFace (fallback)
        """
    
    def generate_single(self, sentence: str, temperature: float = None) -> Explanation:
        """Generate explanation for a single sentence."""
    
    def generate_batch(self, sentences: List[str], ...) -> List[Explanation]:
        """Generate explanations for batch of sentences."""
```

### Functions

#### `generate_with_stability()`
```python
def generate_with_stability(
    generator: ExplanationGenerator,
    sentence: str,
    n_samples: int = 3,
    sample_temperature: float = 0.7,
) -> ExplanationWithStability:
    """
    Generate explanation with stability sampling.
    
    1. Generate primary (temp=0)
    2. Generate M-1 samples (temp=0.7)
    3. Compute stability metrics
    """
```

#### `explanations_to_embeddings()`
```python
def explanations_to_embeddings(
    explanations: List[Explanation],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Convert explanations to embeddings for graph construction.
    
    Canonical string: "Evidence: ... | Rationale: ... | Counterfactual: ..."
    """
```

#### `get_reliability_scores()`
```python
def get_reliability_scores(
    explanations_with_stability: List[ExplanationWithStability],
) -> np.ndarray:
    """Extract reliability scores from explanations."""
```

---

## `embed_graph.py` — Graph Construction

### Classes

#### `ExplanationGraph`
```python
@dataclass
class ExplanationGraph:
    n_nodes: int
    embeddings: np.ndarray       # (n, dim)
    reliability: np.ndarray      # (n,) node reliability
    neighbors: np.ndarray        # (n, k) neighbor indices
    similarities: np.ndarray     # (n, k) cosine similarities
    weights: np.ndarray          # (n, k) reliability-weighted edges
    k: int
    similarity_threshold: float
    outlier_scores: Optional[np.ndarray]  # (n,) outlier detection
```

### Functions

#### `build_explanation_graph()`
```python
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
    Build reliability-weighted kNN graph.
    
    Edge weight formula:
        w_ij = softmax(sim_ij / τ) × r_j
    
    Where r_j is neighbor reliability.
    """
```

#### `compute_neighborhood_label_posterior()`
```python
def compute_neighborhood_label_posterior(
    graph: ExplanationGraph,
    labels: np.ndarray,
    n_classes: int = 2,
    smoothing_epsilon: float = 1e-3,
) -> np.ndarray:
    """
    Compute p(c | neighbors) using reliability-weighted votes.
    
    Returns:
        posteriors: (n, n_classes) label posteriors
    """
```

---

## `signals.py` — ECG Signal Computation

### Classes

#### `SignalScores`
```python
@dataclass
class SignalScores:
    # Individual signals (higher = more suspicious)
    neighborhood_surprise: np.ndarray  # S_nbr
    nli_contradiction: np.ndarray      # S_nli
    artifact_score: np.ndarray         # S_art
    stability_score: np.ndarray        # S_stab = 1 - reliability
    dynamics_score: np.ndarray         # S_dyn = -AUM
    
    # Confidence scores
    neighborhood_confidence: np.ndarray
    nli_confidence: np.ndarray
    stability_confidence: np.ndarray
    dynamics_confidence: np.ndarray
    
    # Combined scores
    ecg_score: np.ndarray              # Fixed weights
    ecg_score_adaptive: np.ndarray     # Adaptive weights
```

#### `NLIScorer`
```python
class NLIScorer:
    def __init__(
        self,
        model_names: List[str] = ["microsoft/deberta-v3-base-mnli"],
    ):
        """Ensemble NLI scoring."""
    
    def score_batch(
        self,
        premises: List[str],
        hypotheses: List[str],
    ) -> List[NLIResult]:
        """Score premise-hypothesis pairs."""
```

### Functions

#### `compute_all_signals()`
```python
def compute_all_signals(
    graph: ExplanationGraph,
    explanations: List[Explanation],
    observed_labels: np.ndarray,
    reliability_scores: np.ndarray,
    aum_scores: np.ndarray,
    known_artifacts: List[str] = None,
    nli_scorer: NLIScorer = None,
) -> SignalScores:
    """
    Compute all five ECG signal families.
    
    Signals:
    1. S_nbr: Neighborhood surprise
    2. S_nli: NLI contradiction margin
    3. S_art: Artifact focus score
    4. S_stab: Explanation instability
    5. S_dyn: Training dynamics (negative AUM)
    """
```

#### `combine_signals_fixed_weights()`
```python
def combine_signals_fixed_weights(
    signals: SignalScores,
    weights: Dict[str, float] = None,
) -> np.ndarray:
    """
    Combine signals with fixed weights.
    
    Default: S_ECG = 0.30×S_nbr + 0.30×S_nli + 0.15×S_art + 0.15×S_stab + 0.10×S_dyn
    """
```

#### `combine_signals_adaptive()`
```python
def combine_signals_adaptive(signals: SignalScores) -> np.ndarray:
    """
    Combine signals with per-instance adaptive weights.
    
    Each signal weighted by its confidence for that instance.
    """
```

---

## `baselines.py` — Baseline Methods

### Classes

#### `BaselineScores`
```python
@dataclass
class BaselineScores:
    cleanlab: np.ndarray         # Confident learning
    loss: np.ndarray             # Cross-entropy loss
    margin: np.ndarray           # Prediction margin (negated)
    aum: np.ndarray              # Area Under Margin (negated)
    entropy: np.ndarray          # Prediction entropy
    llm_mismatch: np.ndarray     # LLM disagrees with label
    input_knn: np.ndarray        # Input embedding kNN disagreement
    classifier_knn: np.ndarray   # Classifier embedding kNN disagreement
    nrg: np.ndarray              # Neural Relation Graph score
    random: np.ndarray           # Random baseline
```

### Functions

#### `compute_all_baselines()`
```python
def compute_all_baselines(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    features: np.ndarray = None,
    input_embeddings: np.ndarray = None,
    llm_predicted_labels: np.ndarray = None,
    llm_confidence: np.ndarray = None,
    k: int = 15,
) -> BaselineScores:
    """Compute all baseline suspicion scores."""
```

#### `cleanlab_scores()`
```python
def cleanlab_scores(labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    """Cleanlab confident learning scores."""
```

#### `nrg_scores()`
```python
def nrg_scores(
    features: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    kernel_type: str = 'cos_p',
) -> np.ndarray:
    """Neural Relation Graph scores (adapted from NRG repo)."""
```

---

## `eval.py` — Evaluation Metrics

### Classes

#### `DetectionMetrics`
```python
@dataclass
class DetectionMetrics:
    auroc: float
    auprc: float
    tnr_at_95: float
    precision_at_k: Dict[float, float]
    recall_at_k: Dict[float, float]
    f1_at_k: Dict[float, float]
```

### Functions

#### `compute_detection_metrics()`
```python
def compute_detection_metrics(
    ground_truth: np.ndarray,
    scores: np.ndarray,
    k_values: List[float] = [0.005, 0.01, 0.02, 0.05, 0.10],
) -> DetectionMetrics:
    """Compute AUROC, AUPRC, P@K, R@K, F1@K."""
```

#### `print_detection_summary()`
```python
def print_detection_summary(
    method_metrics: Dict[str, DetectionMetrics],
    k_to_show: float = 0.05,
) -> None:
    """Print formatted comparison table."""
```

#### `plot_comparison()`
```python
def plot_comparison(
    ground_truth: np.ndarray,
    scores_dict: Dict[str, np.ndarray],
    save_path: str = None,
) -> None:
    """Plot ROC and PR curves for multiple methods."""
```

---

## `clean.py` — Data Cleaning

### Classes

#### `CleaningConfig`
```python
@dataclass
class CleaningConfig:
    strategy: str = "remove"     # "remove" or "relabel"
    k_fraction: float = 0.05
    
    # Relabeling guardrails
    relabel_confidence_threshold: float = 0.7
    relabel_agreement_threshold: float = 0.8
    
    # Protections
    outlier_protection: bool = True
    dynamics_veto: bool = True
```

#### `CleaningResult`
```python
@dataclass
class CleaningResult:
    removed_indices: np.ndarray
    relabeled_indices: np.ndarray
    kept_indices: np.ndarray
    new_labels: np.ndarray = None
    
    n_removed: int
    n_relabeled: int
    n_kept: int
    
    removal_precision: float = None
    removal_recall: float = None
```

### Functions

#### `clean_dataset()`
```python
def clean_dataset(
    dataset: Dataset,
    labels: np.ndarray,
    ecg_scores: np.ndarray,
    config: CleaningConfig,
    outlier_scores: np.ndarray = None,
    dynamics_scores: np.ndarray = None,
    ground_truth_noisy: np.ndarray = None,
) -> CleaningResult:
    """
    Clean dataset by removing or relabeling suspicious examples.
    
    With protections:
    - Outlier protection: Don't remove likely OOD examples
    - Dynamics veto: Don't remove "learnable" examples
    """
```

#### `evaluate_at_multiple_k()`
```python
def evaluate_at_multiple_k(
    ecg_scores: np.ndarray,
    ground_truth_noisy: np.ndarray,
    k_fractions: List[float],
) -> Dict[float, Tuple[float, float]]:
    """Evaluate precision/recall at multiple K values."""
```

---

*See [01_architecture.md](01_architecture.md) for visual diagrams*

