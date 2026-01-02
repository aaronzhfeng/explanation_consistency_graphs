# ECG: Explanation-Consistency Graphs

**Graph-aggregated LLM explanations for training data debugging**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ECG identifies mislabeled or artifact-laden training instances that loss/confidence-based methods (e.g., Cleanlab) miss—especially when models confidently fit errors via spurious markers.

**Core insight:** LLM-generated structured explanations contain signals (agreement patterns, contradictions, artifact focus) that reveal problematic training data even when the classifier's loss looks fine.

## Key Features

- **Schema-guaranteed explanation generation** with constrained decoding
- **Stability sampling** to measure explanation reliability
- **Reliability-weighted kNN graphs** over explanation embeddings
- **Multi-signal detection**: neighborhood surprise, NLI contradiction, artifact focus, training dynamics
- **Robust evaluation**: ROAR validation, leakage-aware metrics

## Project Structure

```
explanation_consistency_graphs/
├── instruction/                   # Research methodology & documentation
│   ├── playbook-00-research-sprint/  # Speedrun Paper methodology
│   │   ├── templates/             # Project planning templates
│   │   ├── checklists/            # Pre-implementation, pre-training checks
│   │   └── workflows/             # Cursor IDE, git workflows
│   ├── brainstorm-03-llm-explainability/  # ECG brainstorming materials
│   │   ├── proposals/             # Research proposals
│   │   └── prompts/               # Literature search prompts
│   ├── research_proposal_0.md     # Original proposal
│   ├── research_proposal_1.md     # Revised proposal (literature-informed)
│   ├── literature.md              # 103-paper bibliography
│   └── proposal_advise_*.md       # Review documents
│
├── src/ecg/                       # Core implementation (~4,900 lines)
│   ├── __init__.py                # Package exports
│   ├── data.py                    # Dataset loading & noise injection
│   ├── train_classifier.py        # RoBERTa fine-tuning & training dynamics (AUM)
│   ├── explain_llm.py             # LLM explanation generation with stability sampling
│   ├── embed_graph.py             # Reliability-weighted kNN graph construction
│   ├── signals.py                 # 5-signal ECG scoring system
│   ├── baselines.py               # Cleanlab, NRG, kNN, LLM baselines
│   ├── eval.py                    # Detection & faithfulness metrics
│   └── clean.py                   # Data cleaning strategies
│
├── configs/                       # Configuration files
│   └── default.yaml               # Default hyperparameters
│
├── scripts/                       # Experiment scripts
│   ├── run_experiment.py          # Full pipeline runner
│   └── quick_test.py              # Quick test with mock data
│
├── references/                    # Reference implementations (not committed)
│   ├── aum/                       # AUM library
│   ├── cleanlab/                  # Cleanlab baseline
│   ├── Neural-Relation-Graph/     # NRG baseline
│   └── wann-noisy-labels/         # WANN reliability weighting
│
├── docs/                          # Implementation documentation
│   ├── 00_index.md                # Documentation index
│   ├── 01_architecture.md         # Implementation mindmap
│   ├── 02_module_reference.md     # API documentation
│   ├── 03_session_log.md          # Development history
│   └── 04_pipeline_guide.md       # How to run experiments
│
├── outputs/                       # Experiment outputs
│   ├── checkpoints/               # Model checkpoints
│   ├── explanations/              # Generated explanations
│   └── results/                   # Evaluation results
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/explanation_consistency_graphs.git
cd explanation_consistency_graphs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test (No GPU/LLM Required)

```bash
# Run quick test with mock explanations
python scripts/quick_test.py
```

### Full Experiment

```bash
# Run full pipeline (requires GPU + LLM model)
python scripts/run_experiment.py --config configs/default.yaml

# Skip LLM generation if you have cached explanations
python scripts/run_experiment.py --config configs/default.yaml --skip-llm
```

### Python API

```python
import ecg

# 1. Create noisy dataset
noisy_data = ecg.create_noisy_dataset(
    n_train=25000,
    noise_config=ecg.NoiseConfig(noise_type="artifact_aligned", noise_rate=0.10),
)

# 2. Train classifier and get training dynamics
model, dynamics, results = ecg.train_classifier(
    train_dataset=noisy_data.dataset,
    config=ecg.TrainingConfig(epochs=3),
    return_dynamics=True,
)

# 3. Generate LLM explanations (requires GPU)
generator = ecg.ExplanationGenerator(model_name="Qwen/Qwen2.5-7B-Instruct")
explanations = ecg.generate_batch_with_stability(generator, noisy_data.dataset['sentence'])
reliability = ecg.get_reliability_scores(explanations)

# 4. Build explanation graph
embeddings = ecg.explanations_to_embeddings([e.primary for e in explanations])
graph = ecg.build_explanation_graph(embeddings, reliability, k=15)

# 5. Compute ECG signals
from ecg.signals import compute_all_signals
signals = compute_all_signals(
    graph=graph,
    explanations=[e.primary for e in explanations],
    observed_labels=noisy_data.noisy_labels,
    reliability_scores=reliability,
    aum_scores=dynamics.aum_scores,
)

# 6. Evaluate detection
metrics = ecg.compute_detection_metrics(noisy_data.is_noisy, signals.ecg_score_adaptive)
print(f"AUROC: {metrics.auroc:.3f}, AUPRC: {metrics.auprc:.3f}")

# 7. Compare with baselines
ecg.print_detection_summary({
    "ECG": metrics,
    "Cleanlab": ecg.compute_detection_metrics(noisy_data.is_noisy, cleanlab_scores),
})
```

## Methodology

### 1. Noise Injection

Two synthetic regimes to demonstrate ECG's value:

- **Uniform label flips**: Standard noise (sanity check vs Cleanlab)
- **Artifact-aligned flips**: Labels flipped AND spurious markers added, causing confident wrong predictions

### 2. Explanation Generation

Generate structured JSON explanations with:
- Predicted label
- Evidence spans (extractive)
- Rationale (no label words)
- Counterfactual statement
- Stability score (from M=3 samples)

### 3. Graph Construction

Build reliability-weighted kNN graph over explanation embeddings:
- Node reliability from explanation stability
- Edge weight = similarity × neighbor reliability
- Outlier/OOD detection

### 4. Inconsistency Signals

Five signal families:
1. **Neighborhood surprise**: Label disagreement with similar explanations
2. **NLI contradiction**: Explanation contradicts observed label
3. **Artifact focus**: Evidence cites spurious tokens
4. **Stability**: High explanation variance
5. **Training dynamics**: AUM separates hard from mislabeled

### 5. Cleaning & Evaluation

- Remove/relabel top-K suspicious points
- Evaluate: detection precision, downstream accuracy, OOD robustness
- Robust metrics: ROAR validation, leakage checks

## Baselines

- Cleanlab (Confident Learning)
- High-loss filtering
- AUM / CTRL (training dynamics)
- Neural Relation Graph
- WANN (reliability-weighted kNN)
- LLM label mismatch
- Input/classifier embedding kNN

## Citation

```bibtex
@article{ecg2025,
  title={Explanation-Consistency Graphs: Graph-aggregated LLM explanations for training data debugging},
  author={...},
  journal={ACL 2026 Theme Track: Explainability},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
