# ECG Architecture & Implementation Mindmap

> How the ECG system fits together

---

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ECG PIPELINE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Data    │───▶│  Classifier  │───▶│  Explanations│───▶│    Graph     │  │
│  │ (data.py)│    │(train_*.py)  │    │(explain_*.py)│    │(embed_*.py)  │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│       │                │                    │                   │           │
│       ▼                ▼                    ▼                   ▼           │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ NoisyData│    │  Dynamics    │    │  Reliability │    │Neighborhood  │  │
│  │ is_noisy │    │  AUM scores  │    │   Scores     │    │  Structure   │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│       │                │                    │                   │           │
│       └────────────────┴────────────────────┴───────────────────┘           │
│                                    │                                         │
│                                    ▼                                         │
│                          ┌──────────────────┐                                │
│                          │     SIGNALS      │                                │
│                          │   (signals.py)   │                                │
│                          └──────────────────┘                                │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         ▼                          ▼                          ▼             │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐       │
│  │  S_nbr       │          │  S_nli       │          │  S_art       │       │
│  │ Neighborhood │          │    NLI       │          │  Artifact    │       │
│  │  Surprise    │          │Contradiction │          │   Score      │       │
│  └──────────────┘          └──────────────┘          └──────────────┘       │
│         │                          │                          │             │
│         └──────────────────────────┼──────────────────────────┘             │
│                                    │                                         │
│  ┌──────────────┐          ┌──────────────┐                                 │
│  │  S_stab      │          │  S_dyn       │                                 │
│  │  Stability   │          │  Dynamics    │                                 │
│  │ (1-reliab)   │          │ (-AUM)       │                                 │
│  └──────────────┘          └──────────────┘                                 │
│         │                          │                                         │
│         └──────────────────────────┘                                         │
│                        │                                                     │
│                        ▼                                                     │
│              ┌──────────────────┐                                           │
│              │   ECG SCORE      │                                           │
│              │ (combine signals)│                                           │
│              └──────────────────┘                                           │
│                        │                                                     │
│         ┌──────────────┴──────────────┐                                     │
│         ▼                             ▼                                      │
│  ┌──────────────┐            ┌──────────────┐                               │
│  │   EVALUATE   │            │    CLEAN     │                               │
│  │  (eval.py)   │            │  (clean.py)  │                               │
│  └──────────────┘            └──────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
                    data.py
                       │
                       ▼
              train_classifier.py ◄──── [AUM library]
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    explain_llm.py  baselines.py   │
          │            │           │
          ▼            │           │
    embed_graph.py ────┤           │
          │            │           │
          ▼            ▼           ▼
       signals.py ◄────┴───────────┘
          │
          ├──────────────┐
          ▼              ▼
       eval.py        clean.py
          │              │
          └──────┬───────┘
                 ▼
         run_experiment.py
```

---

## Data Flow Details

### Stage 1: Data Preparation

```
SST-2 (HuggingFace)
       │
       ▼
┌─────────────────────────────┐
│     data.py                 │
├─────────────────────────────┤
│ • load_sst2()               │
│ • create_noisy_dataset()    │
│   - Subsample to 25k        │
│   - Inject noise (uniform   │
│     or artifact-aligned)    │
│   - Track is_noisy mask     │
└─────────────────────────────┘
       │
       ▼
NoisyDataset {
  dataset: HF Dataset,
  noisy_labels: np.ndarray,
  original_labels: np.ndarray,
  is_noisy: np.ndarray (bool)
}
```

### Stage 2: Classifier Training

```
NoisyDataset
       │
       ▼
┌─────────────────────────────┐
│   train_classifier.py       │
├─────────────────────────────┤
│ • train_classifier()        │
│   - Fine-tune roberta-base  │
│   - On noisy_labels         │
│   - Compute per-example:    │
│     * losses                │
│     * margins               │
│     * AUM scores            │
│ • cross_validate_predictions│
│   - 5-fold CV for Cleanlab  │
└─────────────────────────────┘
       │
       ├──► model (RoBERTa)
       ├──► TrainingDynamics {aum_scores, losses, margins, entropy}
       └──► cv_probs (for Cleanlab)
```

### Stage 3: Explanation Generation

```
Sentences from dataset
       │
       ▼
┌─────────────────────────────┐
│     explain_llm.py          │
├─────────────────────────────┤
│ • ExplanationGenerator      │
│   - vLLM / HF / Outlines    │
│   - Schema-constrained JSON │
│                             │
│ • generate_with_stability() │
│   - Primary (temp=0)        │
│   - M=3 samples (temp=0.7)  │
│   - Compute reliability:    │
│     * label_agreement       │
│     * evidence_jaccard      │
│     * rationale_similarity  │
│     * r_i = mean of above   │
└─────────────────────────────┘
       │
       ├──► List[ExplanationWithStability]
       ├──► reliability_scores: np.ndarray
       └──► llm_labels, llm_confidence
```

### Stage 4: Graph Construction

```
Explanations + Reliability
       │
       ▼
┌─────────────────────────────┐
│     embed_graph.py          │
├─────────────────────────────┤
│ • explanations_to_embeddings│
│   - Canonical string:       │
│     "Evidence: ... |        │
│      Rationale: ... |       │
│      Counterfactual: ..."   │
│   - sentence-transformers   │
│                             │
│ • build_explanation_graph() │
│   - FAISS kNN (k=15)        │
│   - Reliability-weighted:   │
│     w_ij ∝ exp(sim/τ) × r_j │
│   - Outlier detection       │
└─────────────────────────────┘
       │
       ▼
ExplanationGraph {
  embeddings, reliability,
  neighbors, similarities, weights,
  outlier_scores
}
```

### Stage 5: Signal Computation

```
Graph + Explanations + Dynamics
       │
       ▼
┌─────────────────────────────┐
│       signals.py            │
├─────────────────────────────┤
│ Five signal families:       │
│                             │
│ 1. S_nbr (neighborhood)     │
│    = -log(p(y_i | neighbors)│
│    Uses reliability-weighted│
│    neighbor label votes     │
│                             │
│ 2. S_nli (contradiction)    │
│    = P_C - P_E (margin)     │
│    Ensemble DeBERTa+RoBERTa │
│                             │
│ 3. S_art (artifact)         │
│    Synthetic: overlap with  │
│      known artifacts        │
│    Real: PMI-based detection│
│                             │
│ 4. S_stab (instability)     │
│    = 1 - reliability        │
│                             │
│ 5. S_dyn (dynamics)         │
│    = -AUM                   │
│    Lower AUM = suspicious   │
│                             │
│ Combination:                │
│ • Fixed weights (0.3/0.3/   │
│   0.15/0.15/0.10)           │
│ • Adaptive: per-signal conf │
│ • Dynamics veto option      │
└─────────────────────────────┘
       │
       ▼
SignalScores {
  neighborhood_surprise,
  nli_contradiction,
  artifact_score,
  stability_score,
  dynamics_score,
  ecg_score,
  ecg_score_adaptive
}
```

### Stage 6: Evaluation & Cleaning

```
ECG Scores + Ground Truth
       │
       ▼
┌─────────────────────────────┐
│        eval.py              │
├─────────────────────────────┤
│ • compute_detection_metrics │
│   - AUROC, AUPRC            │
│   - P@K, R@K, F1@K          │
│   - TNR@95                  │
│                             │
│ • plot_comparison()         │
│   - ROC/PR curves           │
│                             │
│ • print_detection_summary() │
│   - Comparison table        │
└─────────────────────────────┘

       │
       ▼
┌─────────────────────────────┐
│        clean.py             │
├─────────────────────────────┤
│ • select_top_k()            │
│   - With outlier protection │
│   - With dynamics veto      │
│                             │
│ • clean_dataset()           │
│   - Remove suspicious       │
│   - Or relabel with LLM     │
│                             │
│ • evaluate_at_multiple_k()  │
│   - P/R across K values     │
└─────────────────────────────┘
```

---

## Key Design Patterns

### 1. Dataclass Containers

Every stage produces a typed dataclass for clean interfaces:

```python
@dataclass
class NoisyDataset:
    dataset: Dataset
    noisy_labels: np.ndarray
    is_noisy: np.ndarray

@dataclass  
class TrainingDynamics:
    aum_scores: np.ndarray
    losses: np.ndarray
    ...

@dataclass
class ExplanationGraph:
    embeddings: np.ndarray
    neighbors: np.ndarray
    weights: np.ndarray
    ...

@dataclass
class SignalScores:
    neighborhood_surprise: np.ndarray
    nli_contradiction: np.ndarray
    ecg_score: np.ndarray
    ...
```

### 2. Lazy Initialization

Models load only when first used:

```python
class ExplanationGenerator:
    def __init__(self, ...):
        self._initialized = False
        
    def initialize(self):
        if self._initialized:
            return
        # Load model
        self._initialized = True
    
    def generate_single(self, ...):
        self.initialize()  # Lazy load
        ...
```

### 3. Reference-Informed Code

Key patterns adapted from reference repos:

| Pattern | Source | Used In |
|---------|--------|---------|
| AUMCalculator | `aum/aum.py` | `train_classifier.py` |
| Reliability weighting | `wann/WANN.py` | `embed_graph.py` |
| NRG kernel | `NRG/relation.py` | `baselines.py` |
| Evaluation metrics | `NRG/metric.py` | `eval.py` |

### 4. Caching Strategy

Explanations are expensive; cache to disk:

```python
cache_path = "outputs/explanations/explanations.pkl"
if skip_llm and os.path.exists(cache_path):
    # Load from cache
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    return cached
else:
    # Generate and cache
    explanations = generate_batch(...)
    with open(cache_path, 'wb') as f:
        pickle.dump(explanations, f)
```

---

## Configuration Hierarchy

```yaml
# configs/default.yaml structure
data:
  dataset: "sst2"
  n_train: 25000
  noise:
    type: "artifact_aligned"
    rate: 0.10
  artifacts:
    enabled: true
    positive_token: "<lbl_pos>"
    negative_token: "<lbl_neg>"

classifier:
  model_name: "roberta-base"
  training:
    epochs: 3
    batch_size: 64

explanation:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  stability:
    num_samples: 3

graph:
  k: 15
  temperature: 0.07

signals:
  aggregation:
    method: "reliability_adaptive"
    fixed_weights:
      neighborhood: 0.30
      nli: 0.30
      artifact: 0.15
      stability: 0.15
      dynamics: 0.10

cleaning:
  k_values: [0.005, 0.01, 0.02, 0.05, 0.10]
  method: "remove"
```

---

## Baseline Comparison Structure

```
                    Ground Truth: is_noisy
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
┌──────────┐        ┌──────────┐          ┌──────────┐
│   ECG    │        │ Cleanlab │          │   NRG    │
│ (ours)   │        │(baseline)│          │(baseline)│
└──────────┘        └──────────┘          └──────────┘
    │                      │                      │
    ▼                      ▼                      ▼
 ecg_score           cleanlab_score          nrg_score
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           │
                           ▼
                  compute_detection_metrics()
                           │
                           ▼
                  print_detection_summary()
```

---

*See [02_module_reference.md](02_module_reference.md) for detailed API documentation*

