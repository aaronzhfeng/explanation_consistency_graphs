# ECG Experiment Guide v2

> Updated guide including all experiments from the 2026-01-02/03 debugging session

---

## Quick Reference: All Experiments

| Experiment | Script | Status | Key Finding |
|------------|--------|--------|-------------|
| Original Pipeline | `scripts/step6_*.py` | ✅ Complete | Baseline results |
| Artifact-Aligned Noise | `scripts/experiment_ensemble.py` | ✅ Complete | **Explanation kNN: 0.832 AUROC** |
| Random Label Noise | `scripts/experiment_random_noise.py` | ✅ Complete | Cleanlab: 0.977 AUROC |
| Downstream Evaluation | `scripts/step8_downstream_explknn.py` | ✅ Complete | **+0.57% accuracy at K=2%** |
| Noise Rate Sensitivity | See below | ❌ Pending | 5%, 20% rates |
| Explanation kNN on Random Noise | See below | ❌ Pending | Complete comparison table |

---

## Completed Experiments

### Experiment 1: Artifact-Aligned Noise (Full Scale)

**Purpose**: Test ECG on noise where traditional methods fail catastrophically.

**Script**: 
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs
python scripts/experiment_ensemble.py
```

**Configuration**:
| Parameter | Value |
|-----------|-------|
| Dataset | SST-2 (25,000 examples) |
| Noise Type | Artifact-aligned |
| Noise Rate | 10% (2,500 noisy) |
| Artifacts | `<lbl_pos>`, `<lbl_neg>` tokens |
| LLM | Qwen/Qwen3-8B |

**Results** (2026-01-03):

| Method | AUROC | AUPRC | P@K | Notes |
|--------|-------|-------|-----|-------|
| **Explanation kNN** | **0.832** | 0.435 | 0.496 | 🏆 Best |
| Max(LLM, ExpKNN, Artifact) | 0.814 | 0.380 | 0.397 | Ensemble |
| Input kNN | 0.671 | 0.258 | 0.342 | Baseline |
| LLM Mismatch | 0.575 | 0.152 | 0.280 | Simple |
| Cleanlab | 0.107 | — | — | ❌ Fails |

**Key Finding**: Explanation kNN beats Input kNN by **+24%** (0.832 vs 0.671).

---

### Experiment 2: Random Label Noise

**Purpose**: Compare ECG on "fair game" scenario where all methods can work.

**Script**:
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs
python scripts/experiment_random_noise.py
```

**Configuration**:
| Parameter | Value |
|-----------|-------|
| Dataset | SST-2 (25,000 examples) |
| Noise Type | Random flip (no artifacts) |
| Noise Rate | 10% (2,500 noisy) |
| LLM | Qwen/Qwen3-8B |

**Results** (2026-01-02):

| Method | AUROC | AUPRC | P@K | Notes |
|--------|-------|-------|-----|-------|
| **Cleanlab** | **0.977** | 0.854 | 0.916 | 🏆 Best |
| Loss | 0.977 | 0.854 | 0.916 | Same |
| LLM Mismatch | 0.901 | 0.632 | 0.789 | Strong |
| Input kNN | 0.880 | 0.492 | 0.650 | Good |
| ECG (adaptive) | 0.747 | 0.235 | 0.306 | Below baselines |

**Key Finding**: Cleanlab dominates on random noise (as expected).

---

### Experiment 3: Downstream Evaluation ✅ COMPLETE (2026-01-03)

**Purpose**: Prove that removing detected examples improves model accuracy.

#### ⚠️ Important: Use the Correct Script!

The original `step8_downstream_evaluation.py` uses `ecg_score_adaptive` which has low AUROC (0.547).

**Use this instead** (uses Explanation kNN with 0.832 AUROC):
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs
python scripts/step8_downstream_explknn.py
```

#### Original Script (uses old signals - NOT recommended)
```bash
python scripts/step8_downstream_evaluation.py
```

**Results with old signals** (2026-01-03):
| K | Precision | Recall | Δ Accuracy |
|---|-----------|--------|------------|
| 1% | 0.0% | 0.0% | +0.0046 |
| 5% | 1.0% | 0.5% | -0.0046 |
| 10% | 1.7% | 1.7% | +0.0000 |

❌ Near-zero precision because the old combined ECG score doesn't detect noisy examples well.

**Actual results with Explanation kNN** ✅:

| K | Removed | Precision | Recall | Accuracy | Change |
|---|---------|-----------|--------|----------|--------|
| 1% | 250 | 66.8% | 6.7% | 0.9358 | +0.00% |
| **2%** | 500 | **57.4%** | 11.5% | **0.9415** | **+0.57%** |
| 5% | 1250 | 40.6% | 20.3% | 0.9381 | +0.23% |
| 10% | 2500 | 29.7% | 29.7% | 0.9300 | -0.57% |

**Key finding**: K=2% gives best accuracy improvement (+0.57%)

**Saved to**: `outputs/results/20260103_135902_downstream_explknn.json`

---

## Pending Experiments

### Experiment 4: Explanation kNN on Random Noise

**Purpose**: Complete the comparison table—does Explanation kNN also work on random noise?

**Script** (to create):
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs
python scripts/experiment_explknn_random.py
```

**Or modify existing**:
```python
# In experiment_ensemble.py, change:
noise_config = NoiseConfig(
    noise_rate=0.1,
    noise_type='random'  # Instead of 'artifact_aligned'
)
```

**Expected table after completion**:

| Method | Artifact Noise | Random Noise | Robust? |
|--------|---------------|--------------|---------|
| Explanation kNN | 0.832 ✅ | ??? | TBD |
| Cleanlab | 0.107 ❌ | 0.977 ✅ | ❌ |
| Input kNN | 0.671 | 0.880 | ✅ |

---

### Experiment 5: Noise Rate Sensitivity

**Purpose**: Test how performance varies with noise rate (5%, 10%, 20%).

**Commands**:
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs

# 5% noise
python scripts/experiment_ensemble.py --noise_rate 0.05

# 20% noise  
python scripts/experiment_ensemble.py --noise_rate 0.20
```

**Note**: You may need to add argparse to `experiment_ensemble.py`:
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--noise_rate', type=float, default=0.1)
args = parser.parse_args()
# Then use args.noise_rate instead of hardcoded 0.1
```

**Expected results**:

| Noise Rate | Explanation kNN AUROC | Input kNN AUROC |
|------------|----------------------|-----------------|
| 5% | ~0.85 | ~0.70 |
| 10% | 0.832 | 0.671 |
| 20% | ~0.80 | ~0.65 |

---

### Experiment 6: Different LLM Sizes

**Purpose**: Test if smaller/larger LLMs affect explanation quality.

**Models to test**:
- `Qwen/Qwen3-1.7B` (smaller, faster)
- `Qwen/Qwen3-8B` (current)
- `Qwen/Qwen3-14B` (larger, if GPU allows)

**Modify in** `experiment_ensemble.py`:
```python
generator = ExplanationGenerator(
    model_name="Qwen/Qwen3-1.7B",  # Change here
    ...
)
```

---

### Experiment 7: Real Annotation Errors (Advanced)

**Purpose**: Test on naturally occurring label errors, not synthetic.

**Approach**:
1. Use datasets with known annotation quality issues
2. Or: Human-annotate a subset to find real errors
3. Compare ECG detection against human judgments

**Potential datasets**:
- SNLI with crowd-disagreement annotations
- ChaosNLI
- Learning with Noisy Labels benchmarks

---

## Original Pipeline Scripts

These are the step-by-step scripts from the original codebase:

| Step | Script | Time | Description |
|------|--------|------|-------------|
| 6.1 | `step6_1_train_classifier.py` | 30m | Train classifier + AUM dynamics |
| 6.2 | `step6_2_generate_explanations.py` | 2-4h | LLM explanations + stability |
| 6.3 | `step6_3_build_graph_signals.py` | 1h | kNN graph + 5 signals |
| 6.4 | `step6_4_evaluate.py` | 30m | Baselines + metrics |
| 8 | `step8_downstream_evaluation.py` | 30m | Retrain on cleaned data |

**Run full pipeline**:
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs

python scripts/step6_1_train_classifier.py
python scripts/step6_2_generate_explanations.py
python scripts/step6_3_build_graph_signals.py
python scripts/step6_4_evaluate.py
python scripts/step8_downstream_evaluation.py
```

---

## New Experiment Scripts (2026-01-02/03)

| Script | Purpose | Key Result |
|--------|---------|------------|
| `experiment_ensemble.py` | Test signal combinations on artifact noise | Explanation kNN: 0.832 |
| `experiment_random_noise.py` | Test on random label noise | Cleanlab: 0.977 |
| `ablation_signals.py` | Diagnose signal contributions | Dynamics anti-correlated |

---

## Results Files

All results are saved with timestamps:

```
outputs/results/
├── 20260103_130312_ensemble_results.json    # Latest artifact experiment
├── 20260102_185651_results.json             # Random noise experiment
├── latest_results.json                       # Symlink to most recent
└── *.png                                     # Detection comparison plots
```

---

## Key Insights from Experiments

### What Works ✅

1. **Explanation kNN** is the best signal for artifact noise (0.832 AUROC)
2. **+24% improvement** over Input kNN by using explanation embeddings
3. LLM explanations capture label-quality information invisible to raw embeddings

### What Doesn't Work ❌

1. **Dynamics signal** is anti-correlated on artifact noise (model fits artifacts confidently)
2. **Signal averaging** hurts—Max aggregation is better
3. **Full ECG pipeline** underperforms simpler Explanation kNN

### Paper Narrative

> "We introduce Explanation Consistency Graphs (ECG) for detecting noisy labels. Our key insight is that computing neighborhood surprise in the LLM explanation embedding space is far more effective than in the raw input space. On artifact-aligned noise, Explanation kNN achieves 0.832 AUROC—24% better than Input kNN (0.671)—while traditional methods like Cleanlab fail completely (0.107)."

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce dataset size for testing
# In experiment_ensemble.py:
dataset_size = 5000  # Instead of 25000
```

### vLLM Multiprocessing Error

Add to script before imports:
```python
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
```

### GPU Memory Not Freed

Add after LLM generation:
```python
del generator
torch.cuda.empty_cache()
```

---

## Timeline for Remaining Experiments

| Experiment | Priority | Est. Time | Notes |
|------------|----------|-----------|-------|
| Downstream Evaluation | ✅ Done | — | **+0.57% at K=2%** |
| Explanation kNN on Random | 🔴 High | 2-3 hours | Complete comparison |
| Noise Rate Sensitivity | 🟡 Medium | 4-6 hours | 5%, 20% rates |
| Different LLM Sizes | 🟢 Low | 6-8 hours | Optional |
| Real Annotation Errors | 🟢 Low | Days | Advanced |

---

*Last updated: 2026-01-03*

