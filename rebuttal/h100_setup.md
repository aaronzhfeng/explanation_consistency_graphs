# H100 Environment: Setup & Run Guide

> One-shot guide. Clone repo, run one script, collect results.

---

## 1. Environment Setup

```bash
# Clone the repo (or rsync from current machine)
git clone <your-repo-url> ~/explanation_consistency_graphs
cd ~/explanation_consistency_graphs

# Install Python dependencies
pip install torch transformers accelerate datasets sentence-transformers
pip install cleanlab aum scikit-learn omegaconf tqdm aiohttp

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name())"
# Should print: NVIDIA H100 ...
```

If transferring from current machine instead of cloning:
```bash
# On current machine (RTX 2000 Ada):
rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'outputs/training_baselines_ckpt' \
    --exclude 'outputs/smoketest_ckpt' \
    --exclude 'outputs/cache' \
    ~/explanation_consistency_graphs/ <h100-host>:~/explanation_consistency_graphs/
```

---

## 2. Run Training Baselines

```bash
cd ~/explanation_consistency_graphs
nohup bash scripts/run_all_training_baselines.sh > outputs/results/training_baselines_log.txt 2>&1 &
echo "PID: $!"
```

**What this runs:**
- 4 experiments: SST-2 × {uniform, artifact_aligned} + MultiNLI × {uniform, artifact_aligned}
- Each: 5 seeds (42, 123, 456, 789, 1024), n=25,000, 3 epochs RoBERTa
- Per seed: 1 full training + 5-fold CV (Cleanlab) + training dynamics + all baselines
- Total: 20 training runs + 100 CV folds = **120 RoBERTa fine-tuning runs**

**Estimated time on H100: ~4-5 hours**
(Smoketest on RTX 2000 Ada extrapolated ~10.6 hr for 2 configs; H100 is ~4-5x faster; 4 configs total)

**Monitor progress:**
```bash
tail -f outputs/results/training_baselines_log.txt
```

---

## 3. What Gets Saved

All results land in `outputs/results/` as JSON files:

```
outputs/results/
  {timestamp}_sst2_uniform_training_baselines.json
  {timestamp}_sst2_artifact_aligned_training_baselines.json
  {timestamp}_multinli_uniform_training_baselines.json
  {timestamp}_multinli_artifact_aligned_training_baselines.json
  log_sst2_uniform_training.txt
  log_sst2_artifact_training.txt
  log_multinli_uniform_training.txt
  log_multinli_artifact_training.txt
```

Each JSON has this structure:
```json
{
  "experiment": "sst2_uniform_training_baselines",
  "config": { "dataset": "sst2", "noise_type": "uniform", ... },
  "results_by_seed": {
    "42": {
      "Cleanlab": { "auroc": 0.xx, "auprc": 0.xx, "tnr_at_95": 0.xx },
      "High-Loss": { ... },
      "AUM": { ... },
      "Margin": { ... },
      "Classifier kNN": { ... },
      "NRG": { ... },
      ...
    },
    "123": { ... },
    ...
  },
  "aggregated": {
    "Cleanlab": {
      "auroc": { "mean": 0.xx, "std": 0.xx, "n": 5 },
      ...
    }
  }
}
```

---

## 4. Files to Bring Back

After the run completes, copy these results back:

```bash
# On H100:
rsync -avz ~/explanation_consistency_graphs/outputs/results/*training_baselines*.json \
    <origin-host>:~/explanation_consistency_graphs/outputs/results/
```

These JSON files are everything needed. Training checkpoints (`outputs/training_baselines_ckpt/`) are disposable and do NOT need to be saved.

---

## 5. Quick Sanity Check (Optional, ~2 min)

Before the full run, verify everything works:

```bash
python scripts/run_training_baselines.py \
    --dataset sst2 --noise_type uniform \
    --n_train 500 --epochs 1 --seeds 42
```

Should complete in ~30s on H100 and print a summary table.

---

## 6. Existing Results (Already Completed — DO NOT Re-run)

These API-based experiments are already done and saved. They do NOT need GPU:

| File | Experiment | Status |
|------|-----------|--------|
| `20260217_191041_sst2_uniform_results.json` | SST-2 Uniform, 5 seeds, API methods | Done |
| `20260217_230908_sst2_artifact_aligned_results.json` | SST-2 Artifact, 5 seeds, API methods | Done |
| `20260218_034411_multinli_uniform_results.json` | MultiNLI Uniform, 5 seeds, API methods | Done |
| `20260218_101345_multinli_artifact_aligned_results.json` | MultiNLI Artifact, 3 seeds, API methods | Done |

API methods already computed: **Explanation kNN, Input kNN, LLM Mismatch**

The H100 run adds: **Cleanlab, AUM, High-Loss, Margin, Entropy, Classifier kNN, NRG**

Together these give the complete comparison table for reviewer responses.

---

## 7. What This Enables for the Rebuttal

With both API results + training baselines, we can report (per reviewer response, 5000 char limit):

**SST-2 Artifact-Aligned (the core claim):**
- Explanation kNN: 0.819 ± 0.004 AUROC
- LLM Mismatch: 0.628 ± 0.004
- Input kNN: 0.549 ± 0.008
- Cleanlab: [from H100 run]
- AUM: [from H100 run]
- High-Loss: [from H100 run]

The key argument: under artifact noise, training-based methods (Cleanlab/AUM/High-Loss) fail because the classifier confidently learns the spurious pattern. ECG's explanation-based signal is orthogonal.

**MultiNLI (honest finding):**
- LLM Mismatch dominates (~0.88 AUROC)
- ECG methods near-random (~0.55)
- Shows ECG is task-dependent — strengthens credibility
