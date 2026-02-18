# Rebuttal Implementation Plan

## Summary of Reviewer Concerns

| Concern | YHvT | Rvqa | g7YU | Priority |
|---------|------|------|------|----------|
| Single dataset (SST-2 only) | x | x | x | **Critical** |
| Synthetic noise only, no real-world artifacts | x | x | x | **Critical** |
| Single run, no error bars / statistical significance | | x | x | **High** |
| Missing recent baselines ([2][3]) | x | | | Medium |
| LLM mismatch as strong baseline (just use Qwen's pred_label) | | | x | **High** |
| Computational cost comparison | | x | | Medium |
| Multi-class / NLI / QA tasks | | x | | Medium |
| AUROC not introduced before ablation (Line 355) | | | x | Trivial |

## Is Our Approach Training-Focused?

**No.** The core Explanation kNN signal is entirely inference-only:
1. LLM generates explanations (inference, temp=0)
2. Sentence-transformer embeds them (inference)
3. FAISS builds kNN graph (no training)
4. Neighborhood surprise = label disagreement with neighbors (arithmetic)

Classifier training (RoBERTa) is only needed for (a) the AUM signal (which *hurts* under artifact noise — our negative result) and (b) Cleanlab/loss/margin baselines. **The core contribution requires zero training.**

---

## Known Compute Times (H100, vLLM, Qwen3-8B)

From our existing runs on 25k SST-2 examples:

| Stage | Time | Notes |
|-------|------|-------|
| **LLM explanations (primary, temp=0)** | **~2.5 min** | 183 it/s batched vLLM |
| **LLM stability samples (x2, temp=0.7)** | **~5 min** | 175 it/s each |
| Sentence embedding (MiniLM) | ~30 sec | CPU, batch=256 |
| kNN graph (FAISS) | ~10 sec | GPU |
| Signal computation (incl. NLI model) | ~15 min | DeBERTa-MNLI on GPU |
| RoBERTa fine-tuning (3 epochs) | ~30 min | For baselines only |
| Baselines + evaluation | ~5 min | CPU |
| **Total core ECG pipeline** | **~25 min** | Without classifier training |
| **Total with all baselines** | **~60 min** | Including RoBERTa + Cleanlab CV |

**Key insight:** LLM inference is only ~7 min for 25k examples. It is NOT the bottleneck. The NLI signal model and RoBERTa training for baselines take longer.

---

## Phase 1: Quick Wins

### 1.1 Multiple Seeds + Error Bars

**What:** 5 seeds (42, 123, 456, 789, 1024), report mean +/- std.

LLM explanations are deterministic (temp=0) — generate once, reuse across seeds. Each seed only re-does: noise injection -> recompute which labels are noisy -> rebuild kNN with same embeddings but new labels -> evaluate.

**Compute time per seed (ECG core only):** ~5 min (noise injection + kNN recompute + signal eval — embeddings are cached)

**Compute time per seed (with baselines):** ~35 min (need to retrain RoBERTa + re-run Cleanlab CV per noise injection)

| Experiment | Seeds | Per-seed | Total |
|-----------|-------|----------|-------|
| SST-2 artifact-aligned (ECG only) | 5 | ~5 min | **~25 min** |
| SST-2 artifact-aligned (with baselines) | 5 | ~35 min | **~3 hr** |
| SST-2 random noise (with baselines) | 5 | ~35 min | **~3 hr** |

**Total Phase 1.1 compute: ~6 hours** (can run overnight, or parallelize across GPUs)

### 1.2 Wall-Clock Time & Compute Cost Table

**Compute time:** Zero — just instrument the existing pipeline with `time.time()` wrappers and run once.

### 1.3 Highlight LLM Mismatch Baseline

Already have the numbers (0.575 vs 0.832 on artifacts). Paper writing only — no compute.

### 1.4 Fix AUROC Introduction (Line 355)

Paper edit only.

---

## Phase 2: New Datasets

### 2.1 MultiNLI with Hypothesis-Only Bias

**Why:** 3-class NLI, natural artifacts (hypothesis-only shortcuts), explicitly requested by reviewers. Already in our bibliography (Poliak et al. 2018, ref 60).

**Experimental design:**
- Load MNLI from HuggingFace, subsample to 25k
- Artifact regime: train a hypothesis-only classifier to identify artifact-bearing examples, flip labels on those
- Control: uniform random noise

**Compute time:**

| Step | Time | Notes |
|------|------|-------|
| Download + subsample MNLI | ~1 min | HuggingFace cached |
| Train hypothesis-only classifier (artifact identification) | ~15 min | Small BERT on hypothesis only |
| LLM explanations (25k, primary + 2 stability) | **~7 min** | Same throughput as SST-2 |
| Embedding + kNN graph | ~1 min | Same as SST-2 |
| NLI signal (DeBERTa) | ~15 min | Same model |
| RoBERTa fine-tuning (for baselines) | ~30 min | 3-class now |
| Baselines + eval | ~5 min | |
| **Total (one seed, one noise config)** | **~75 min** | |
| **x2 noise configs (artifact + random) x5 seeds** | | **~12.5 hr** |
| **x2 noise configs x1 seed (minimum viable)** | | **~2.5 hr** |

**Minimum viable for rebuttal:** 1 seed each for artifact-aligned and random noise = **~2.5 hours**. Add error bars later if time permits.

### 2.2 AlleNoise (Real-World Label Noise)

**Why:** Explicitly requested by YHvT. Real-world e-commerce noise, no injection needed.

**Risk:** Many product categories. Mitigation: restrict to top-10 categories or use open-ended LLM prediction + fuzzy match.

**Compute time:**

| Step | Time | Notes |
|------|------|-------|
| Download + preprocess AlleNoise | ~5 min | Filter to top-N categories |
| LLM explanations (25k) | **~7 min** | Product categorization prompt |
| Embedding + kNN graph | ~1 min | |
| Baselines + eval | ~35 min | Including RoBERTa |
| **Total (one seed)** | **~50 min** |
| **x5 seeds** | | **~4 hr** |

**Minimum viable:** 1 seed = **~50 min**

### 2.3 Code Generalization

Pure implementation — no compute time. Changes to:
- `src/ecg/data.py`: Generic dataset loader, multi-class noise injection
- `src/ecg/explain_llm.py`: Task-agnostic prompting with `TaskConfig`
- `configs/`: New YAML configs per dataset

---

## Phase 3: Additional Baselines

### 3.1 NoisegPT (Wang et al., NeurIPS 2024)

Uses LLM probability curvature. Likely needs logprob extraction from vLLM.

**Compute time:** ~10 min per dataset (reuses LLM inference, just extracts logprobs). Implementation is the bottleneck, not compute.

### 3.2 Learning Discriminative Dynamics (Kim et al., CVPR 2024)

Requires tracking per-example representation dynamics during training. Piggybacks on existing RoBERTa training loop.

**Compute time:** ~0 extra (computed during the same RoBERTa fine-tuning pass already running for baselines).

### 3.3 LLM Mismatch Deep Dive

**Additional experiments:**
- Qwen accuracy on clean SST-2 test set — **~1 min** (inference on ~1.8k examples)
- Qwen accuracy on artifact-injected test examples — **~1 min**
- Precision/Recall curves at various thresholds — **~0** (post-hoc analysis of existing scores)

---

## Phase 4: Paper Revisions

Zero compute. Writing only:
- Update results tables with new datasets + error bars + timing
- New paragraph on computational cost
- New paragraph on LLM mismatch comparison
- Extended related work (NoisegPT, Kim et al., AlleNoise)
- Point-by-point rebuttal letter

---

## Total Compute Budget

### Minimum Viable Rebuttal (1 seed per new experiment)

| Experiment | Time |
|-----------|------|
| SST-2 multi-seed (5 seeds, ECG + baselines, 2 noise types) | ~6 hr |
| MultiNLI (1 seed, 2 noise types) | ~2.5 hr |
| AlleNoise (1 seed) | ~50 min |
| LLM mismatch deep dive | ~5 min |
| Timing instrumentation run | ~1 hr |
| **Total** | **~10.5 hr** |

### Full Rebuttal (5 seeds everywhere)

| Experiment | Time |
|-----------|------|
| SST-2 multi-seed (5 seeds, 2 noise types) | ~6 hr |
| MultiNLI multi-seed (5 seeds, 2 noise types) | ~12.5 hr |
| AlleNoise multi-seed (5 seeds) | ~4 hr |
| New baselines (NoisegPT + disc. dynamics) | ~1 hr |
| LLM mismatch deep dive | ~5 min |
| **Total** | **~24 hr** |

All on a single H100. With 2 GPUs, the full rebuttal runs in ~12 hours.

---

## Execution Order (Optimized for GPU Utilization)

```
Hour 0-1:    [Code] Generalize data.py + explain_llm.py (Phase 2.3)
             [Code] Add multi-seed loop + timing instrumentation (Phase 1.1, 1.2)

Hour 1:      [GPU]  Launch SST-2 multi-seed runs (5 seeds x 2 noise types)
             [Code] Implement MultiNLI loader + NLI prompt template

Hour 1-7:    [GPU]  SST-2 runs complete (~6 hr)
             [Code] Implement AlleNoise loader + prompt (parallel with GPU)
             [Code] Implement NoisegPT + disc. dynamics baselines (parallel)

Hour 7:      [GPU]  Launch MultiNLI experiments
Hour 7-9.5:  [GPU]  MultiNLI minimum viable (2 runs) complete
             [GPU]  Launch AlleNoise experiment

Hour 9.5-10: [GPU]  AlleNoise complete
Hour 10:     [GPU]  Launch remaining MultiNLI seeds (3 more) + AlleNoise seeds (4 more)

Hour 10-24:  [GPU]  Remaining seeds complete
             [Code] Paper revisions + rebuttal letter (parallel with GPU)
```

**Critical path:** Code generalization (1 hr) -> SST-2 seeds (6 hr) -> MultiNLI (2.5 hr) -> AlleNoise (50 min) = **~10.5 hours to minimum viable results**. Everything else runs in parallel or after.

---

## File Change Summary

| File | Action | Phase |
|------|--------|-------|
| `src/ecg/data.py` | Add `load_multinli()`, `load_allenoise()`, generic dispatcher, multi-class noise | 2 |
| `src/ecg/explain_llm.py` | Add NLI + AlleNoise prompt templates, `TaskConfig` abstraction | 2 |
| `src/ecg/baselines.py` | Add NoisegPT, discriminative dynamics baselines | 3 |
| `src/ecg/utils.py` | New file: timing utilities, multi-seed aggregation | 1 |
| `configs/multinli.yaml` | New config for MultiNLI experiments | 2 |
| `configs/allenoise.yaml` | New config for AlleNoise experiments | 2 |
| `scripts/run_experiment.py` | Add `--seeds`, per-step timing, dataset dispatch | 1-2 |
| `scripts/experiment_multinli.py` | New experiment runner for NLI | 2 |
| `scripts/experiment_allenoise.py` | New experiment runner for AlleNoise | 2 |
| `paper/main.tex` | Update results, add cost table, new sections | 4 |
| `rebuttal/rebuttal_letter.md` | Point-by-point responses | 4 |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Explanation kNN doesn't beat Input kNN on NLI | Medium | High | Still report — even comparable performance on natural artifacts is informative. Frame as "complements confidence-based methods" |
| AlleNoise has too many classes for LLM prompting | Medium | Medium | Restrict to top-10 categories; use open-ended prediction + fuzzy matching |
| NoisegPT is hard to reimplement | Low | Low | Cite and discuss qualitatively if blocked; dataset extensions matter more |
| Multi-class kNN signal degrades | Low | Medium | Test with k=15 and k=25; neighborhood surprise generalizes naturally to N-class |
