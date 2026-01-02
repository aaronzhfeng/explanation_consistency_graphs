# ExplainCal Proposal from GPT-5.2 Pro

**Source:** GPT-5.2 Pro  
**Date:** 2025-01-30  
**Prompt:** `03_calibrated_explanation_confidence.md`

---

## Working title

**ExplainCal: Calibrated Step-Level Confidence for NLP Explanations via Hidden-State Probes + Verifiable Decomposition**

This paper targets **explanation-level uncertainty** (is this step valid?) and **faithfulness uncertainty** (is this step causally used by the model?) using **lightweight probes/heads** trained on **automatic step labels**.

---

## 1) Core hypothesis

1. **Step validity is linearly decodable from internal activations** at "step boundaries" (e.g., end-of-step tokens), and a small probe can output **well-calibrated probabilities** of step validity across multiple explanation styles and tasks.

2. **Validity and faithfulness are separable signals**:
   * Validity confidence will correlate with correctness, but not perfectly
   * Faithfulness confidence (causal usage) will only partially correlate with validity

3. A **single "explanation reliability direction"** (or a small subspace) emerges across tasks, enabling a **mechanistic "knob"**: moving activations along this direction will systematically trade off **explanation reliability vs. verbosity/creativity**.

Motivated by evidence that hidden-state probes can be **highly calibrated for intermediate correctness** in reasoning models (ECE often < 0.1).

---

## 2) Tasks and datasets

### A. Evidence-based factual/QA tasks

* **FEVER** (fact verification + evidence sentences)
* **HotpotQA** (multi-hop QA + supporting facts)

### B. Proof-style explanations

* **EntailmentBank** (QA with entailment trees / proof graphs)

### C. Classification with human-style rationales

* **e-SNLI / ERASER-style rationale datasets**

---

## 3) Explanation methods to study

1. **Structured CoT (step-tagged)** — numbered steps, one claim per step
2. **Evidence rationales (sentence citations only)**
3. **Free-form rationale paragraphs**
4. **Attribution baselines** — attention rollout / gradient × input / integrated gradients

---

## 4) Confidence estimation approaches

### 4.1 Internal-signal approaches (white-box)

**Token-probability features**
* mean / min token logprob within step
* step-level entropy and top-1–top-2 margin
* length-normalized logprob

**Activation features (primary contribution)**
* `h_i`: last-layer hidden state at end-of-step token
* layerwise probe sweep
* attention-entropy features

**Probe types**
* Linear probe (logistic regression)
* Small MLP (1 hidden layer)
* "UHead"-style lightweight head

### 4.2 Consistency-based approaches

* Answer self-consistency
* Evidence consistency (Jaccard overlap)
* Step semantic consistency
* Perturbation stability

### 4.3 Learned predictors (combined)

Train predictor on activation + token-prob + consistency features → **p_valid(step i)** and **p_faithful(step i)**

---

## 5) Ground truth for training

### 5.1 Step validity labels (automatic)

**For FEVER / HotpotQA**
* Evidence correctness: cited sentence ∈ gold supporting set
* Claim entailment: NLI model says premises ⟹ conclusion

**Synthetic negatives (crucial)**
* Swap evidence IDs with distractors
* Entity swap, negate relation

Training set: 50% natural outputs + 50% synthetically corrupted steps

### 5.2 Step faithfulness labels (causal usage proxy)

**Ablation-based causal influence:**
1. Compute baseline answer logprob with full context
2. Remove step i, compute new logprob
3. Define influence: Δ_i = L_full - L_-i
4. Label faithful if Δ_i ≥ δ_high

---

## 6) Evaluation metrics

### Step-level calibration
* ECE, Brier score, NLL
* Reliability diagrams
* Cross-domain transfer

### Discrimination and localization
* AUROC / AUPRC for detecting invalid steps
* First-error localization

### Selective explanation / downstream utility
* Selective display (keep only high-confidence steps)
* Confidence-guided regeneration

### Correlation with prediction correctness
* AUROC of predicting answer correctness from explanation confidence

---

## 7) Baselines

1. Self-reported confidence per step
2. Self-consistency only
3. Perturbation stability only
4. Step mean/min logprob
5. Step entropy / margin
6. Verifier score as confidence
7. Lightweight internal "uncertainty head" variant

---

## 8) Expected contributions

1. **Problem formulation:** explanation-level uncertainty decomposed into **validity** and **faithfulness**
2. **Practical training signal:** verifiable step labels + synthetic negative corruptions
3. **Lightweight, probe-based method:** frozen LLM + small probe with strong calibration
4. **Mechanistic "knob" evidence:** identify direction predicting step validity; activation steering
5. **Downstream utility:** confidence-guided regeneration improves explanation quality
6. **Artifact:** benchmark of step explanations with automatic validity/faithfulness labels

---

## 9) Compute estimate

* Data generation + feature extraction: ~25–35 H100h
* Faithfulness (ablation) labeling: ~10–15 H100h
* Training predictors: ~2–5 H100h
* Evaluation + interventions: ~10–20 H100h

**Total: ~47–75 H100 hours**

