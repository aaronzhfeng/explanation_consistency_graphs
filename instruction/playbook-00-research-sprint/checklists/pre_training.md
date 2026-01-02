# Pre-Training Checklist

> Verify before running expensive experiments

---

## 1. Data Pipeline Verified

- [ ] Data loading works without errors
- [ ] Sample output looks correct (inspect manually!)
- [ ] Train/val/test splits are correct sizes
- [ ] No data leakage between splits
- [ ] Tokenization produces expected lengths

## 2. Model Loading Works

- [ ] Base model loads without errors
- [ ] Model fits in GPU memory
- [ ] Forward pass produces outputs
- [ ] Gradients flow (for training)

## 3. Training Loop Tested

- [ ] Smoke test on tiny dataset passes
- [ ] Loss decreases (even slightly)
- [ ] Checkpoints save correctly
- [ ] Logging works

## 4. Generation Works

- [ ] Model produces coherent outputs
- [ ] Output format is as expected
- [ ] Sampling with temperature works
- [ ] Generation completes without hanging

## 5. Evaluation Pipeline Ready

- [ ] Metrics computation tested
- [ ] Results saved in correct format
- [ ] Can reproduce metrics from saved outputs

## 6. Monitoring Setup

- [ ] GPU utilization monitored
- [ ] Training progress visible
- [ ] Know how to check for crashes

---

## Verification Commands

```bash
# Test data loading
python -c "from src.data import load_dataset; d = load_dataset('data/test.jsonl'); print(len(d))"

# Test model loading
python -c "from src.models import load_model; m = load_model('model-name'); print('OK')"

# Smoke test training
python scripts/train.py --config configs/toy_test.yaml --max-steps 10

# Test generation
python scripts/generate.py --model outputs/test --num-samples 5
```

---

## Red Flags — Stop If:

- [ ] Loss is NaN or doesn't decrease
- [ ] Generation produces garbage
- [ ] OOM errors (even with small batch)
- [ ] Data looks wrong when inspected
- [ ] Metrics don't match manual calculation

