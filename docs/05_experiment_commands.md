# ECG Experiment Commands

> Complete step-by-step guide to run all experiments

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 1× A100 (40GB) | 1× H100 (80GB) |
| **RAM** | 32GB | 64GB |
| **Storage** | 50GB free | 100GB free |

**Estimated GPU hours:** ~20-25 H100 hours for full experiment

### Software Requirements

- Python 3.10+
- CUDA 11.8+ (for vLLM and FAISS-GPU)
- Git

---

## Step 0: Clone Repository

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/explanation_consistency_graphs.git
cd explanation_consistency_graphs
```

---

## Step 1: Environment Setup

### Option A: Conda (Recommended)

```bash
# Create conda environment
conda create -n ecg python=3.10 -y
conda activate ecg

# Install PyTorch with CUDA
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install FAISS GPU
conda install -c pytorch faiss-gpu -y

# Install remaining dependencies
pip install -r requirements.txt
```

### Option B: Virtual Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install FAISS CPU (GPU version requires conda)
pip install faiss-cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test imports
python -c "
import torch
import transformers
import vllm
import faiss
import cleanlab
from ecg import create_noisy_dataset

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('All imports successful!')
"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA H100 80GB HBM3
All imports successful!
```

---

## Step 2: Quick Sanity Check (No GPU Required)

```bash
# Run quick test with mock data
python scripts/quick_test.py
```

**Expected output:** Test should pass with mock metrics.

**Time:** ~30 seconds

---

## Step 3: Download & Prepare Dataset

```bash
# SST-2 downloads automatically on first use
# Run this to cache the dataset
python -c "
from datasets import load_dataset
dataset = load_dataset('glue', 'sst2')
print('Train examples:', len(dataset['train']))
print('Validation examples:', len(dataset['validation']))
print('Dataset cached successfully!')
"
```

**Expected output:**
```
Train examples: 67349
Validation examples: 872
Dataset cached successfully!
```

---

## Step 4: Download LLM Model

```bash
# Pre-download the LLM (Qwen2.5-7B-Instruct)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'Downloading {model_name}...')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)

print('Model downloaded and loaded successfully!')
print(f'Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters')
"
```

**Alternative LLMs (if Qwen unavailable):**
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-3.1-8B-Instruct` (requires HF access)

---

## Step 5: Download NLI Models

```bash
# Pre-download NLI models for contradiction detection
python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer

models = [
    'roberta-large-mnli',
    'facebook/bart-large-mnli'
]

for name in models:
    print(f'Downloading {name}...')
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    print(f'  Done!')

print('All NLI models downloaded!')
"
```

---

## Step 6: Run Full Experiment

### Option A: Single Command (Full Pipeline)

```bash
# Run complete experiment pipeline
# Estimated time: 4-6 hours on H100

python scripts/run_experiment.py \
    --config configs/default.yaml \
    --verbose
```

### Option B: Step-by-Step Scripts (Recommended)

Run each step in your terminal. Each script has progress bars and saves intermediate results.

#### Step 6.1: Train Classifier (~30 min)

```bash
python scripts/step6_1_train_classifier.py
```

**What it does:**
- Creates noisy dataset (25k examples, 10% artifact-aligned noise)
- Trains RoBERTa-base for 3 epochs
- Computes AUM training dynamics
- Saves to `outputs/step6_1_classifier.pkl`

#### Step 6.2: Generate Explanations (~2-4 hours)

```bash
python scripts/step6_2_generate_explanations.py
```

**What it does:**
- Loads Qwen2.5-7B-Instruct via vLLM
- Generates structured JSON explanations for all 25k examples
- Stability sampling (3 samples per example)
- Computes reliability scores
- Saves to `outputs/explanations/explanations.pkl`

**Progress bar example:**
```
Generating with stability:  42%|████▏     | 10500/25000 [1:12:30<1:40:15, 2.41it/s]
```

#### Step 6.3: Build Graph & Compute Signals (~1 hour)

```bash
python scripts/step6_3_build_graph_signals.py
```

**What it does:**
- Embeds explanations with sentence-transformers
- Builds reliability-weighted kNN graph (k=15)
- Computes all 5 ECG signals (neighborhood, NLI, artifact, stability, dynamics)
- Saves to `outputs/step6_3_signals.pkl`

#### Step 6.4: Run Baselines & Evaluate (~30 min)

```bash
python scripts/step6_4_evaluate.py
```

**What it does:**
- 5-fold cross-validation for Cleanlab baseline
- Computes all baseline scores
- Evaluates detection metrics (AUROC, AUPRC, P@K, R@K)
- Saves results to `outputs/results/results.json`
- Generates comparison plots

---

## Step 7: Additional Experiments (Ablations)

### 7.1: Uniform Noise

```bash
# Create config for uniform noise
python -c "
from omegaconf import OmegaConf

config = OmegaConf.load('configs/default.yaml')
config.data.noise.type = 'uniform'
OmegaConf.save(config, 'configs/uniform_noise.yaml')
print('Created configs/uniform_noise.yaml')
"

# Run experiment with uniform noise
python scripts/run_experiment.py --config configs/uniform_noise.yaml
```

### 7.2: Different Noise Rates

```bash
# Test at 5% and 20% noise
for rate in 0.05 0.20; do
    python -c "
from omegaconf import OmegaConf

config = OmegaConf.load('configs/default.yaml')
config.data.noise.rate = $rate
OmegaConf.save(config, f'configs/noise_rate_{$rate}.yaml')
"
    python scripts/run_experiment.py --config configs/noise_rate_${rate}.yaml
done
```

### 7.3: Ablation: Individual Signals

```bash
# Test with only neighborhood signal
python -c "
from omegaconf import OmegaConf

config = OmegaConf.load('configs/default.yaml')
config.signals.aggregation.fixed_weights = {
    'neighborhood': 1.0,
    'nli': 0.0,
    'artifact': 0.0,
    'stability': 0.0,
    'dynamics': 0.0,
}
OmegaConf.save(config, 'configs/ablation_neighborhood_only.yaml')
"

python scripts/run_experiment.py --config configs/ablation_neighborhood_only.yaml
```

---

## Step 8: Downstream Evaluation (Retrain on Cleaned Data)

```bash
python scripts/step8_downstream_evaluation.py
```

**What it does:**
- Tests multiple K values (1%, 2%, 5%, 10%)
- For each K:
  - Removes top-K suspicious examples
  - Computes removal precision/recall
  - Retrains classifier on cleaned data
  - Measures accuracy improvement
- Saves summary to `outputs/results/downstream_results.json`

**Expected output:**
```
DOWNSTREAM EVALUATION SUMMARY
============================================================

Original (noisy) accuracy: 0.9381

     K | Removed |   Prec | Recall | Clean Acc |       Δ
------------------------------------------------------------
    1% |     250 |  0.720 |  0.072 |    0.9412 | +0.0031
    2% |     500 |  0.680 |  0.136 |    0.9435 | +0.0054
    5% |    1250 |  0.620 |  0.310 |    0.9467 | +0.0086
   10% |    2500 |  0.540 |  0.540 |    0.9423 | +0.0042
```

---

## Expected Results Summary

### Detection Performance (artifact-aligned noise @ 10%)

| Method | AUROC | AUPRC | P@5% | R@5% |
|--------|-------|-------|------|------|
| **ECG (adaptive)** | **0.82-0.88** | **0.40-0.50** | **0.65-0.75** | **0.30-0.40** |
| Cleanlab | 0.50-0.60 | 0.12-0.18 | 0.12-0.18 | 0.06-0.10 |
| NRG | 0.60-0.70 | 0.20-0.30 | 0.25-0.35 | 0.12-0.18 |
| LLM Mismatch | 0.70-0.78 | 0.30-0.40 | 0.45-0.55 | 0.22-0.28 |
| Random | 0.50 | 0.10 | 0.10 | 0.05 |

### Key Hypothesis to Verify

**ECG should significantly outperform Cleanlab on artifact-aligned noise** because Cleanlab relies on classifier confidence, which is high (but wrong) for artifact examples.

---

## Quick Reference: All Scripts

| Script | Time | Description |
|--------|------|-------------|
| `scripts/quick_test.py` | 30s | Sanity check with mock data |
| `scripts/step6_1_train_classifier.py` | 30m | Train classifier + AUM |
| `scripts/step6_2_generate_explanations.py` | 2-4h | LLM explanations + stability |
| `scripts/step6_3_build_graph_signals.py` | 1h | kNN graph + 5 signals |
| `scripts/step6_4_evaluate.py` | 30m | Baselines + metrics |
| `scripts/step8_downstream_evaluation.py` | 30m | Retrain on cleaned data |
| `scripts/run_experiment.py` | 4-6h | Full pipeline (all steps) |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch sizes
export ECG_BATCH_SIZE=32  # Default: 64

# Or edit configs/default.yaml:
# classifier.training.batch_size: 32
```

### vLLM Initialization Failed

```bash
# Fall back to HuggingFace
# Edit the generator initialization in scripts:
# use_vllm=False
```

### NLI Model Slow

```bash
# Disable ensemble (use single model)
# Edit configs/default.yaml:
# signals.nli.ensemble.enabled: false
```

---

## Output Files

After running, check:

```
outputs/
├── checkpoints/
│   └── final/              # Trained classifier
├── explanations/
│   └── explanations.pkl    # Cached LLM explanations
├── results/
│   ├── results.json        # Detection metrics
│   ├── downstream_results.json  # Cleaning evaluation
│   └── detection_comparison.png  # ROC/PR curves
├── step6_1_classifier.pkl  # Classifier + dynamics
└── step6_3_signals.pkl     # Graph + signals
```

---

## Estimated Total Time

| Step | Time (H100) |
|------|-------------|
| Environment setup | 10 min |
| Download models | 15 min |
| Step 6.1: Train classifier | 30 min |
| Step 6.2: Generate explanations | 2-4 hours |
| Step 6.3: Build graph + signals | 1 hour |
| Step 6.4: Baselines + evaluation | 30 min |
| Step 8: Downstream evaluation | 30 min |
| **Total** | **5-7 hours** |

---

*For detailed API documentation, see [02_module_reference.md](02_module_reference.md)*
