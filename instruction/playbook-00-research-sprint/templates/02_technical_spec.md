# Phase 2: Technical Specification Template

> Detailed implementation guide — AI generates, human reviews

---

## 1. Repository Structure

```
project-name/
├── configs/              # Configuration files
│   ├── model.yaml
│   ├── training.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/             # Downloaded datasets
│   └── processed/       # Processed data
├── src/
│   ├── __init__.py
│   ├── data/            # Data loading and processing
│   ├── models/          # Model definitions
│   ├── training/        # Training loops
│   ├── evaluation/      # Metrics and analysis
│   └── utils/           # Helpers
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── tests/
├── outputs/             # Experiment outputs (gitignored)
├── paper/               # LaTeX source
├── docs/                # Documentation
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 2. Core Data Structures

### Primary Data Class
```python
@dataclass
class Example:
    """Single training/evaluation example."""
    id: str
    input: str
    output: str
    metadata: dict
```

### Intermediate Representations
```python
# Define key data structures here
```

---

## 3. Module Specifications

### 3.1 Data Module (`src/data/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `loader.py` | Load raw datasets | `load_dataset()` |
| `processor.py` | Transform to training format | `process_example()` |
| `structures.py` | Data classes | — |

### 3.2 Model Module (`src/models/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `base.py` | Base model loading | `load_model()` |
| `heads.py` | Custom heads | `CustomHead` |

### 3.3 Training Module (`src/training/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `trainer.py` | Training loop | `train()` |
| `loss.py` | Custom losses | `compute_loss()` |

### 3.4 Evaluation Module (`src/evaluation/`)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `metrics.py` | Metric computation | `compute_metrics()` |
| `analysis.py` | Result analysis | `analyze_results()` |

---

## 4. Configuration Schema

### model.yaml
```yaml
model:
  name: "model-name"
  torch_dtype: "bfloat16"
  
peft:
  enabled: true
  r: 64
  lora_alpha: 128
  target_modules:
    - q_proj
    - v_proj
```

### training.yaml
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  num_epochs: 3
  max_seq_length: 2048
  
generation:
  temperature: 0.5
  max_new_tokens: 512
```

---

## 5. Key Algorithms

### Algorithm 1: [Main Method]
```
Input: X
Output: Y

1. Step one
2. Step two
3. Step three
4. Return Y
```

### Algorithm 2: [Training Loop]
```
For each iteration:
  1. Generate samples
  2. Evaluate with verifier
  3. Build training pairs
  4. Update model
```

---

## 6. External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | ≥4.40 | Model loading |
| peft | ≥0.10 | LoRA training |
| trl | ≥0.8 | DPO training |
| vllm | ≥0.4 | Fast inference |
| datasets | ≥2.18 | Data loading |

---

## 7. Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 24GB | 80GB |
| System RAM | 32GB | 128GB |
| Storage | 50GB | 200GB |

### Multi-GPU Strategy
```
- SFT: Data parallel across N GPUs
- Inference: 1 GPU per process, N processes
- DPO: Data parallel across N GPUs
```

---

## 8. API Contracts

### Data Loading
```python
def load_dataset(path: str, split: str) -> List[Example]:
    """Load and return dataset examples."""
    pass
```

### Training
```python
def train(
    model_path: str,
    data_path: str,
    output_dir: str,
    config: TrainingConfig
) -> TrainingResult:
    """Run training and return results."""
    pass
```

### Evaluation
```python
def evaluate(
    model_path: str,
    data_path: str,
    output_dir: str
) -> EvalResult:
    """Evaluate model and return metrics."""
    pass
```

---

## 9. Testing Strategy

### Unit Tests
- Data loading
- Metric computation
- Core algorithm logic

### Integration Tests
- Full training on tiny dataset
- End-to-end pipeline

### Smoke Tests
```bash
# Quick validation commands
python scripts/train.py --config configs/toy_test.yaml
python scripts/evaluate.py --model outputs/test --data data/test.jsonl
```

---

## 10. Logging & Monitoring

### Experiment Tracking
```
outputs/
└── experiment-name/
    └── 20251210_143052/    # Timestamped
        ├── run.log
        ├── config.json
        ├── metrics.jsonl
        ├── checkpoints/
        └── final/
```

### Key Metrics to Log
- Training loss (per step)
- Validation metrics (per epoch)
- GPU utilization
- Time per phase

---

## Implementation Checklist

- [ ] Repository structure created
- [ ] Core data structures defined
- [ ] Config files created
- [ ] Data loading implemented
- [ ] Training loop implemented
- [ ] Evaluation implemented
- [ ] Tests passing
- [ ] Smoke test successful

