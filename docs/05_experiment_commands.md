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
    'microsoft/deberta-v3-base-mnli',
    'roberta-large-mnli'
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

### Option B: Step-by-Step (Recommended for Debugging)

#### Step 6.1: Train Classifier

```bash
python -c "
import sys
sys.path.insert(0, 'src')

from ecg import create_noisy_dataset, train_classifier, NoiseConfig, TrainingConfig
import pickle

print('=== Step 6.1: Training Classifier ===')

# Create noisy dataset
noisy_data = create_noisy_dataset(
    n_train=25000,
    noise_config=NoiseConfig(
        noise_type='artifact_aligned',
        noise_rate=0.10,
    ),
)
print(f'Dataset: {len(noisy_data.dataset)} examples, {noisy_data.is_noisy.sum()} noisy')

# Train classifier
from datasets import load_dataset
val_dataset = load_dataset('glue', 'sst2')['validation']

model, dynamics, results = train_classifier(
    train_dataset=noisy_data.dataset,
    val_dataset=val_dataset,
    config=TrainingConfig(epochs=3, output_dir='outputs/checkpoints'),
    return_dynamics=True,
)

print(f'Training loss: {results[\"train_loss\"]:.4f}')
print(f'Val accuracy: {results.get(\"val_accuracy\", \"N/A\")}')
print(f'AUM range: [{dynamics.aum_scores.min():.3f}, {dynamics.aum_scores.max():.3f}]')

# Save for next step
with open('outputs/step6_1_classifier.pkl', 'wb') as f:
    pickle.dump({
        'noisy_data': noisy_data,
        'dynamics': dynamics,
        'results': results,
    }, f)
print('Saved to outputs/step6_1_classifier.pkl')
"
```

**Time:** ~30 minutes

#### Step 6.2: Generate Explanations

```bash
python -c "
import sys
sys.path.insert(0, 'src')

import pickle
from ecg import ExplanationGenerator, generate_batch_with_stability, get_reliability_scores

print('=== Step 6.2: Generating Explanations ===')

# Load previous step
with open('outputs/step6_1_classifier.pkl', 'rb') as f:
    data = pickle.load(f)
noisy_data = data['noisy_data']

# Initialize generator
generator = ExplanationGenerator(
    model_name='Qwen/Qwen2.5-7B-Instruct',
    use_vllm=True,
    temperature=0.0,
)

# Generate explanations with stability
print(f'Generating explanations for {len(noisy_data.dataset)} examples...')
print('(This takes 2-4 hours on H100)')

explanations = generate_batch_with_stability(
    generator=generator,
    sentences=noisy_data.dataset['sentence'],
    n_samples=3,
    sample_temperature=0.7,
    show_progress=True,
)

reliability = get_reliability_scores(explanations)
print(f'Mean reliability: {reliability.mean():.3f}')

# Save
with open('outputs/explanations/explanations.pkl', 'wb') as f:
    pickle.dump({
        'explanations': explanations,
        'reliability': reliability,
    }, f)
print('Saved to outputs/explanations/explanations.pkl')
"
```

**Time:** ~2-4 hours

#### Step 6.3: Build Graph & Compute Signals

```bash
python -c "
import sys
sys.path.insert(0, 'src')

import pickle
import numpy as np
from ecg import build_explanation_graph, explanations_to_embeddings
from ecg.signals import compute_all_signals, NLIScorer

print('=== Step 6.3: Building Graph & Computing Signals ===')

# Load previous steps
with open('outputs/step6_1_classifier.pkl', 'rb') as f:
    data = pickle.load(f)
noisy_data = data['noisy_data']
dynamics = data['dynamics']

with open('outputs/explanations/explanations.pkl', 'rb') as f:
    exp_data = pickle.load(f)
explanations = exp_data['explanations']
reliability = exp_data['reliability']

# Embed explanations
print('Embedding explanations...')
primary_exps = [e.primary for e in explanations]
embeddings = explanations_to_embeddings(primary_exps)
print(f'Embeddings shape: {embeddings.shape}')

# Build graph
print('Building kNN graph...')
graph = build_explanation_graph(
    embeddings=embeddings,
    reliability=reliability,
    k=15,
    temperature=0.07,
)
print(f'Graph: {graph.n_nodes} nodes, mean similarity: {graph.similarities.mean():.3f}')

# Compute signals
print('Computing ECG signals...')
nli_scorer = NLIScorer(model_names=['microsoft/deberta-v3-base-mnli'])

signals = compute_all_signals(
    graph=graph,
    explanations=primary_exps,
    observed_labels=noisy_data.noisy_labels,
    reliability_scores=reliability,
    aum_scores=dynamics.aum_scores,
    known_artifacts=['<lbl_pos>', '<lbl_neg>'],
    nli_scorer=nli_scorer,
)

print(f'ECG score range: [{signals.ecg_score_adaptive.min():.3f}, {signals.ecg_score_adaptive.max():.3f}]')

# Save
with open('outputs/step6_3_signals.pkl', 'wb') as f:
    pickle.dump({
        'graph': graph,
        'embeddings': embeddings,
        'signals': signals,
    }, f)
print('Saved to outputs/step6_3_signals.pkl')
"
```

**Time:** ~1 hour

#### Step 6.4: Run Baselines & Evaluate

```bash
python -c "
import sys
sys.path.insert(0, 'src')

import pickle
import numpy as np
from ecg import compute_all_baselines, compute_detection_metrics, print_detection_summary
from ecg import cross_validate_predictions, TrainingConfig
from ecg.eval import plot_comparison

print('=== Step 6.4: Baselines & Evaluation ===')

# Load all data
with open('outputs/step6_1_classifier.pkl', 'rb') as f:
    data = pickle.load(f)
noisy_data = data['noisy_data']
dynamics = data['dynamics']

with open('outputs/explanations/explanations.pkl', 'rb') as f:
    exp_data = pickle.load(f)
explanations = exp_data['explanations']
reliability = exp_data['reliability']

with open('outputs/step6_3_signals.pkl', 'rb') as f:
    sig_data = pickle.load(f)
signals = sig_data['signals']
embeddings = sig_data['embeddings']

# Get LLM predictions
from ecg.explain_llm import get_llm_predictions
primary_exps = [e.primary for e in explanations]
llm_labels, llm_confidence = get_llm_predictions(primary_exps)

# Cross-validation for Cleanlab
print('Running cross-validation for Cleanlab baseline...')
cv_probs = cross_validate_predictions(
    noisy_data.dataset,
    n_folds=5,
    config=TrainingConfig(epochs=3),
)

# Compute baselines
print('Computing baselines...')
baselines = compute_all_baselines(
    labels=noisy_data.noisy_labels,
    pred_probs=cv_probs,
    input_embeddings=embeddings,
    llm_predicted_labels=llm_labels,
    llm_confidence=llm_confidence,
)

# Evaluate all methods
print('Evaluating...')
ground_truth = noisy_data.is_noisy

all_scores = {
    'ECG (adaptive)': signals.ecg_score_adaptive,
    'ECG (fixed)': signals.ecg_score,
    'Cleanlab': baselines.cleanlab,
    'Loss': baselines.loss,
    'Margin': baselines.margin,
    'LLM Mismatch': baselines.llm_mismatch,
    'Input kNN': baselines.input_knn,
    'Random': baselines.random,
}

all_metrics = {}
for name, scores in all_scores.items():
    if scores is not None:
        all_metrics[name] = compute_detection_metrics(ground_truth, scores)

print_detection_summary(all_metrics)

# Save plots
plot_comparison(ground_truth, all_scores, save_path='outputs/results/detection_comparison.png')
print('Saved plot to outputs/results/detection_comparison.png')

# Save results
import json
results = {
    name: {
        'auroc': m.auroc,
        'auprc': m.auprc,
        'tnr_at_95': m.tnr_at_95,
        'precision_at_k': m.precision_at_k,
        'recall_at_k': m.recall_at_k,
    }
    for name, m in all_metrics.items()
}
with open('outputs/results/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved results to outputs/results/results.json')
"
```

**Time:** ~30 minutes

---

## Step 7: Additional Experiments

### 7.1: Uniform Noise (Ablation)

```bash
# Edit config to use uniform noise
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
python -c "
import sys
sys.path.insert(0, 'src')

import pickle
from ecg.clean import select_top_k, remove_examples
from ecg import train_classifier, TrainingConfig
from datasets import load_dataset

print('=== Step 8: Downstream Evaluation ===')

# Load data
with open('outputs/step6_1_classifier.pkl', 'rb') as f:
    data = pickle.load(f)
noisy_data = data['noisy_data']

with open('outputs/step6_3_signals.pkl', 'rb') as f:
    sig_data = pickle.load(f)
signals = sig_data['signals']

# Select top 5% suspicious and remove
indices_to_remove = select_top_k(signals.ecg_score_adaptive, k_fraction=0.05)
cleaned_dataset, kept_indices = remove_examples(noisy_data.dataset, indices_to_remove)

print(f'Original: {len(noisy_data.dataset)} examples')
print(f'Removed: {len(indices_to_remove)} examples')
print(f'Remaining: {len(cleaned_dataset)} examples')

# Check removal quality
precision = noisy_data.is_noisy[indices_to_remove].sum() / len(indices_to_remove)
print(f'Removal precision: {precision:.3f}')

# Retrain on cleaned data
val_dataset = load_dataset('glue', 'sst2')['validation']

model_clean, _, results_clean = train_classifier(
    train_dataset=cleaned_dataset,
    val_dataset=val_dataset,
    config=TrainingConfig(epochs=3, output_dir='outputs/checkpoints_cleaned'),
    return_dynamics=False,
)

print(f'Cleaned model val accuracy: {results_clean.get(\"val_accuracy\", \"N/A\")}')

# Compare with original
print('\\n=== Comparison ===')
print(f'Original (noisy) accuracy: {data[\"results\"].get(\"val_accuracy\", \"N/A\")}')
print(f'Cleaned accuracy: {results_clean.get(\"val_accuracy\", \"N/A\")}')
"
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
│   ├── results.json        # All metrics
│   └── detection_comparison.png  # ROC/PR curves
└── step6_*.pkl             # Intermediate results
```

---

## Estimated Total Time

| Step | Time (H100) |
|------|-------------|
| Environment setup | 10 min |
| Download models | 15 min |
| Train classifier | 30 min |
| Generate explanations | 2-4 hours |
| Build graph + signals | 1 hour |
| Baselines + evaluation | 30 min |
| **Total** | **4-6 hours** |

---

*For detailed API documentation, see [02_module_reference.md](02_module_reference.md)*

