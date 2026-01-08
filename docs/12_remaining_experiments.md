# Remaining Experiments - Quick Commands

> Copy-paste commands for experiments not yet run
> 
> **Created**: January 3, 2026

---

## ✅ Completed Experiments (Reference)

| Experiment | AUROC | Command Used |
|------------|-------|--------------|
| Artifact Noise (10%) | **0.832** | `python scripts/experiment_ensemble.py` |
| Random Noise (10%) | **0.943** | `python scripts/experiment_random_noise.py` |
| Downstream Cleaning | **+0.57%** | `python scripts/step8_downstream_explknn.py` |
| Dataset Size (5k, 10k) | 0.794, 0.818 | `--dataset_size` flag |
| LLM Size (1.7B) | **0.868** | `--model_name Qwen/Qwen3-1.7B` |
| LLM Size (14B) | **0.896** | `--model_name Qwen/Qwen3-14B` |

---

## 🟡 Recommended Experiments

### 1. Noise Rate Sensitivity (5% and 20%)

**Purpose**: Show that Explanation kNN is robust across different noise rates.

**Expected results**:
| Noise Rate | Expected Exp kNN AUROC | 
|------------|------------------------|
| 5% | ~0.85 |
| 10% | 0.832 (done) |
| 20% | ~0.80 |

**Commands**:

```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs

# 5% noise rate (~45 min)
python scripts/experiment_ensemble.py --noise_rate 0.05

# 20% noise rate (~45 min)
python scripts/experiment_ensemble.py --noise_rate 0.20
```

**Run both in sequence**:
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs

python scripts/experiment_ensemble.py --noise_rate 0.05 && \
python scripts/experiment_ensemble.py --noise_rate 0.20
```

**Results saved to**:
- `outputs/results/{timestamp}_ensemble_results.json`

---

### 2. Noise Rate Sensitivity for Random Noise

**Purpose**: Complete comparison across noise types AND rates.

**Commands**:
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs

# 5% noise rate (~30 min)
python scripts/experiment_random_noise.py --noise_rate 0.05

# 20% noise rate (~30 min)
python scripts/experiment_random_noise.py --noise_rate 0.20
```

**Run both in sequence**:
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs

python scripts/experiment_random_noise.py --noise_rate 0.05 && \
python scripts/experiment_random_noise.py --noise_rate 0.20
```

---

## 🟢 Optional Experiments

### 3. Different Dataset Sizes ✅ COMPLETE

**Purpose**: Verify results hold at smaller scale for faster iteration.

**Results:**
| Dataset Size | Exp kNN AUROC | Input kNN | Δ |
|--------------|---------------|-----------|---|
| 5k | **0.794** | 0.539 | +0.255 |
| 10k | **0.818** | 0.607 | +0.211 |
| 25k | **0.832** | 0.671 | +0.161 |

**Files:**
- `20260105_103148_ensemble_results.json` (5k)
- `20260105_103321_ensemble_results.json` (10k)

```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs

# Small test (5k examples, ~10 min)
python scripts/experiment_ensemble.py --dataset_size 5000

# Medium test (10k examples, ~20 min)
python scripts/experiment_ensemble.py --dataset_size 10000
```

---

### 4. Different LLM Sizes ✅ COMPLETE

**Purpose**: Test if smaller/larger LLMs affect explanation quality and detection strategy.

**Results:**
| Model | Best Method | AUROC | Key Finding |
|-------|-------------|-------|-------------|
| Qwen3-1.7B | Explanation kNN | **0.868** | Simpler explanations → consistent embeddings |
| Qwen3-8B | Explanation kNN | 0.832 | Baseline |
| Qwen3-14B | Ensemble | **0.896** | Rich reasoning → better artifact detection |

**Key insights**:
1. Smaller model (1.7B): Explanation kNN dominates (0.868)
2. Larger model (14B): Ensemble achieves **highest overall AUROC (0.896)**
3. Artifact detection rate: 0.4% (1.7B) vs 4.1% (14B) = 10× improvement

**Files:**
- `20260105_104638_ensemble_results.json` (1.7B)
- `20260105_104822_ensemble_results.json` (14B)

**Commands (now with CLI flag)**:
```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs

# Qwen3-1.7B
python scripts/experiment_ensemble.py --model_name Qwen/Qwen3-1.7B

# Qwen3-14B
python scripts/experiment_ensemble.py --model_name Qwen/Qwen3-14B
```

---

## 📊 Expected Full Results Table

After running all experiments, you should have:

### Artifact-Aligned Noise
| Noise Rate | Explanation kNN | Input kNN | Cleanlab |
|------------|-----------------|-----------|----------|
| 5% | **0.856** ✅ | 0.728 | - |
| **10%** | **0.832** ✅ | 0.671 | 0.107 |
| 20% | **0.785** ✅ | 0.588 | - |

### Random Label Noise
| Noise Rate | Explanation kNN | Input kNN | Cleanlab |
|------------|-----------------|-----------|----------|
| 5% | **0.944** ✅ | 0.884 | 0.979 |
| **10%** | **0.943** ✅ | 0.880 | 0.977 |
| 20% | **0.924** ✅ | 0.860 | 0.965 |

### LLM Size Ablation ✅
| Model Size | Best Method | AUROC | Artifact Detection |
|------------|-------------|-------|-------------------|
| **1.7B** | Exp kNN | **0.868** | 0.4% |
| 8B | Exp kNN | 0.832 | ~2% |
| **14B** | Ensemble | **0.896** | 4.1% |

---

## ⏱️ Time Estimates

| Experiment | Time (H100) |
|------------|-------------|
| Noise rate 5% (artifact) | ~45 min |
| Noise rate 20% (artifact) | ~45 min |
| Both together | ~1.5 hours |

---

## 🚀 One-Liner to Run Everything

```bash
cd /teamspace/studios/this_studio/explanation_consistency_graphs && \
python scripts/experiment_ensemble.py --noise_rate 0.05 && \
python scripts/experiment_ensemble.py --noise_rate 0.20
```

---

## Results Location

All results are timestamped and saved to:
```
outputs/results/
├── YYYYMMDD_HHMMSS_ensemble_results.json    # Each run
└── YYYYMMDD_HHMMSS_random_noise_results.json
```

Load and compare:
```python
import json
import glob

# Load all ensemble results
files = glob.glob('outputs/results/*ensemble*.json')
for f in sorted(files):
    with open(f) as fp:
        data = json.load(fp)
    print(f"{f}: noise_rate={data['config']['noise_rate']}")
    for name, metrics in data['results'].items():
        print(f"  {name}: AUROC={metrics['auroc']:.3f}")
```

---

*Last updated: January 5, 2026 (LLM Size Ablation complete)*

