# ECG Pipeline Guide

> How to run experiments step-by-step

---

## Quick Start

### Option A: Quick Test (No GPU Required)

```bash
# Uses mock explanations — verifies pipeline works
python scripts/quick_test.py
```

Expected output:
```
============================================================
ECG Quick Test (Mock Explanations)
============================================================

1. Creating synthetic data...
   N examples: 500
   N noisy: 50

2. Creating mock explanations...
   Mean reliability: 0.XXX

3. Building explanation graph...
   Graph nodes: 500
   Mean similarity: 0.XXX

4. Computing signals...
   S_nbr range: [X.XXX, X.XXX]
   ECG score range: [X.XXX, X.XXX]

5. Evaluating detection...

================================================================================
DETECTION METRICS SUMMARY
================================================================================
Method               AUROC    AUPRC   TNR@95     P@5%     R@5%    F1@5%
--------------------------------------------------------------------------------
ECG (adaptive)       0.XXX    0.XXX    0.XXX     0.XXX    0.XXX    0.XXX
LLM Mismatch         0.XXX    0.XXX    0.XXX     0.XXX    0.XXX    0.XXX
Random               0.XXX    0.XXX    0.XXX     0.XXX    0.XXX    0.XXX
================================================================================

6. Evaluating cleaning...

============================================================
Quick test PASSED!
============================================================
```

### Option B: Full Experiment (GPU Required)

```bash
# Full pipeline with LLM explanation generation
python scripts/run_experiment.py --config configs/default.yaml
```

With cached explanations:
```bash
# Skip LLM generation (uses cached explanations.pkl)
python scripts/run_experiment.py --config configs/default.yaml --skip-llm
```

---

## Pipeline Steps

### Step 1: Load Data

```python
from ecg import create_noisy_dataset, NoiseConfig

noisy_data = create_noisy_dataset(
    n_train=25000,
    noise_config=NoiseConfig(
        noise_type="artifact_aligned",  # or "uniform"
        noise_rate=0.10,
    ),
)

print(f"N examples: {len(noisy_data.dataset)}")
print(f"N noisy: {noisy_data.is_noisy.sum()}")
```

**Outputs:**
- `noisy_data.dataset` — HuggingFace Dataset
- `noisy_data.noisy_labels` — Corrupted labels
- `noisy_data.is_noisy` — Ground truth mask

### Step 2: Train Classifier

```python
from ecg import train_classifier, TrainingConfig

model, dynamics, results = train_classifier(
    train_dataset=noisy_data.dataset,
    config=TrainingConfig(epochs=3),
    return_dynamics=True,
)

print(f"AUM range: [{dynamics.aum_scores.min():.3f}, {dynamics.aum_scores.max():.3f}]")
```

**Outputs:**
- `model` — Trained RoBERTa classifier
- `dynamics.aum_scores` — Per-example AUM scores
- `dynamics.losses` — Per-example losses

**Time:** ~20-30 min on 1×H100

### Step 3: Generate Explanations

```python
from ecg import ExplanationGenerator, generate_batch_with_stability

generator = ExplanationGenerator(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    use_vllm=True,
)

explanations = generate_batch_with_stability(
    generator=generator,
    sentences=noisy_data.dataset['sentence'],
    n_samples=3,  # For stability
)

reliability = get_reliability_scores(explanations)
print(f"Mean reliability: {reliability.mean():.3f}")
```

**Outputs:**
- `explanations` — List of ExplanationWithStability
- `reliability` — Per-example reliability scores

**Time:** ~2-4 hours on 1×H100 (25k examples)

### Step 4: Build Graph

```python
from ecg import build_explanation_graph, explanations_to_embeddings

embeddings = explanations_to_embeddings(
    [e.primary for e in explanations]
)

graph = build_explanation_graph(
    embeddings=embeddings,
    reliability=reliability,
    k=15,
)

print(f"Graph nodes: {graph.n_nodes}")
print(f"Mean similarity: {graph.similarities.mean():.3f}")
```

**Outputs:**
- `graph.neighbors` — kNN indices
- `graph.weights` — Reliability-weighted edge weights
- `graph.outlier_scores` — Outlier detection

**Time:** ~5 min (FAISS is fast)

### Step 5: Compute Signals

```python
from ecg.signals import compute_all_signals

signals = compute_all_signals(
    graph=graph,
    explanations=[e.primary for e in explanations],
    observed_labels=noisy_data.noisy_labels,
    reliability_scores=reliability,
    aum_scores=dynamics.aum_scores,
    known_artifacts=["<lbl_pos>", "<lbl_neg>"],
)

print(f"ECG score range: [{signals.ecg_score.min():.3f}, {signals.ecg_score.max():.3f}]")
```

**Outputs:**
- `signals.neighborhood_surprise` — S_nbr
- `signals.nli_contradiction` — S_nli
- `signals.artifact_score` — S_art
- `signals.stability_score` — S_stab
- `signals.dynamics_score` — S_dyn
- `signals.ecg_score` — Combined (fixed weights)
- `signals.ecg_score_adaptive` — Combined (adaptive)

**Time:** ~30-60 min (NLI inference dominates)

### Step 6: Run Baselines

```python
from ecg import compute_all_baselines, cross_validate_predictions

cv_probs = cross_validate_predictions(noisy_data.dataset)

baselines = compute_all_baselines(
    labels=noisy_data.noisy_labels,
    pred_probs=cv_probs,
    features=cls_embeddings,
    input_embeddings=embeddings,
    llm_predicted_labels=llm_labels,
)
```

**Outputs:**
- `baselines.cleanlab` — Confident Learning scores
- `baselines.nrg` — Neural Relation Graph scores
- `baselines.loss`, `baselines.margin`, etc.

**Time:** ~30 min (cross-validation)

### Step 7: Evaluate

```python
from ecg import compute_detection_metrics, print_detection_summary

ecg_metrics = compute_detection_metrics(
    noisy_data.is_noisy, 
    signals.ecg_score_adaptive
)

print_detection_summary({
    "ECG (adaptive)": ecg_metrics,
    "Cleanlab": compute_detection_metrics(noisy_data.is_noisy, baselines.cleanlab),
    "Random": compute_detection_metrics(noisy_data.is_noisy, baselines.random),
})
```

### Step 8: Clean Dataset

```python
from ecg.clean import clean_dataset, CleaningConfig, print_cleaning_summary

result = clean_dataset(
    dataset=noisy_data.dataset,
    labels=noisy_data.noisy_labels,
    ecg_scores=signals.ecg_score_adaptive,
    config=CleaningConfig(k_fraction=0.05),
    ground_truth_noisy=noisy_data.is_noisy,
)

print_cleaning_summary(result)
```

---

## Configuration

All hyperparameters are in `configs/default.yaml`:

```yaml
# Key settings to adjust:

data:
  n_train: 25000          # Dataset size
  noise:
    type: "artifact_aligned"  # or "uniform"
    rate: 0.10            # Noise rate

classifier:
  model_name: "roberta-base"
  training:
    epochs: 3             # Training epochs
    batch_size: 64

explanation:
  model_name: "Qwen/Qwen2.5-7B-Instruct"  # LLM for explanations
  stability:
    num_samples: 3        # M for stability sampling

graph:
  k: 15                   # Number of neighbors
  temperature: 0.07       # Softmax temperature

signals:
  aggregation:
    method: "reliability_adaptive"  # or "fixed"

cleaning:
  k_values: [0.005, 0.01, 0.02, 0.05, 0.10]
```

---

## Expected Results

### Detection Performance (artifact-aligned noise @ 10%)

| Method | AUROC | AUPRC | P@5% | R@5% |
|--------|-------|-------|------|------|
| ECG (adaptive) | ~0.85 | ~0.45 | ~0.70 | ~0.35 |
| Cleanlab | ~0.55 | ~0.15 | ~0.15 | ~0.08 |
| NRG | ~0.65 | ~0.25 | ~0.30 | ~0.15 |
| Random | ~0.50 | ~0.10 | ~0.10 | ~0.05 |

*Note: These are expected ranges; actual results depend on hyperparameters and model.*

### Key Hypothesis

**ECG should significantly outperform Cleanlab on artifact-aligned noise** because:
- Cleanlab relies on classifier confidence
- Classifier is confident on artifact examples (wrong but confident)
- ECG uses semantic signals that detect this pattern

---

## Troubleshooting

### Out of Memory (GPU)
```bash
# Reduce batch size
# Edit configs/default.yaml:
classifier:
  training:
    batch_size: 32  # Instead of 64
```

### vLLM Not Available
```python
# Use HuggingFace backend instead
generator = ExplanationGenerator(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    use_vllm=False,  # Fallback to HF
)
```

### NLI Model Slow
```yaml
# Use single model instead of ensemble
signals:
  nli:
    ensemble:
      enabled: false
```

### Explanation Cache Missing
```bash
# Regenerate explanations (takes hours)
python scripts/run_experiment.py --config configs/default.yaml
# Don't use --skip-llm
```

---

## Outputs

After running, check `outputs/`:

```
outputs/
├── checkpoints/          # Model checkpoints
│   └── final/
├── explanations/         # Cached explanations
│   └── explanations.pkl
└── results/              # Evaluation results
    ├── results.json      # Full metrics
    └── detection_comparison.png  # ROC/PR curves
```

---

*For more details, see [02_module_reference.md](02_module_reference.md)*

