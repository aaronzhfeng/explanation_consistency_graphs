# DualCal: Novelty-Checked Proposal from GPT-5.2 Pro

**Source:** GPT-5.2 Pro  
**Date:** 2025-01-30  
**Follow-up Prompt:** Novelty audit of ExplainCal proposal

---

## Novelty sanity check against closely related literature

### Collision areas found:

1. **Explanation uncertainty (black-box / perturbation agreement)**
   - Tanneru et al.: perturbation-based agreement + verbalized uncertainty
   - Focus on uncertainty estimation, not calibrated step-validity probabilities

2. **Intra-explanation uncertainty via step removal**
   - ELAD: defines "intra-explanation uncertainty" by step removal
   - Used for active learning/distillation, not calibrated step-quality predictor

3. **Step-wise confidence attribution (black-box)**
   - ICLR 2026 submission: consensus/graph overlap from multiple sampled traces
   - Evaluates AUROC/AUCPR/ACC@c%/ECE
   - → Must avoid "consensus-only step confidence" as central contribution

4. **Step-level correctness probes for reasoning models**
   - Probing hidden states for intermediate answer correctness
   - PRM calibration for prefix success probability
   - Risk-controlled stopping rules ("thought calibration")
   - **Mainly math/logic focus** — opportunity to differentiate with NLP tasks

5. **Using explanations to calibrate predictions**
   - Ye et al. (NeurIPS 2022): explanation factuality → answer calibration
   - Distinct from "calibrating the explanation itself"

6. **Faithfulness measurement exists**
   - Step-level faithfulness scoring via interventions/unlearning

### Differentiation strategy:

Must (i) separate validity vs faithfulness, (ii) deliver calibrated probabilities, (iii) focus on non-math NLP explanation settings, (iv) include activation "knob" analysis/steering.

---

## Revised Proposal

### Working title

**DualCal: Calibrated Step-wise Validity and Faithfulness Confidence for NLP Explanations via Activation Monitors**

---

## 1) Core hypothesis

**H1 (Representation hypothesis):** For *NLP explanations* (not only math CoT), internal activations at step boundaries linearly encode two distinct signals:
* **Step validity**: "is this step supported/derivable from the input evidence?"
* **Step faithfulness**: "did this step causally matter for the model's final prediction?"

**H2 (Calibration hypothesis):** With lightweight post-hoc calibration, these signals become well-calibrated step probabilities (low ECE/Brier) that generalize across explanation formats.

**H3 (Knob hypothesis / Theme Track alignment):** The learned monitors expose a low-dimensional 'reliability subspace'; intervening on that subspace can increase step validity and/or faithfulness with minimal change in task accuracy.

### Explicit differentiation from prior work:
- Not black-box step confidence via consensus graphs
- Not explanation uncertainty via perturbation agreement
- Not reasoning-model correctness probes focused on math/logic
- Targeting **(validity, faithfulness)** in **NLP explanation tasks** with **internal activation controls**

---

## 2) Tasks and datasets (3 tasks)

### A. Entailment-style proof explanations (structured)
* **EntailmentBank** — entailment trees / intermediate conclusions

### B. Evidence-based QA explanations (multi-hop, grounding)
* **HotpotQA** and/or **FEVER**

### C. Extractive rationales (token/sentence highlights)
* One **ERASER**-style dataset (e.g., MovieReviews, MultiRC)

**Why novelty-friendly:** moves beyond math-only step correctness (PRMs/probes) and beyond black-box consensus step confidence.

---

## 3) Explanation methods to study

1. **Free-form CoT** — numbered list of short steps
2. **Structured explanation objects** — intermediate conclusions + premises
3. **Extractive rationales** — top-k sentences/tokens
4. (Optional) **Attention/gradient saliency** as baseline modality

---

## 4) Confidence estimation approaches

### 4.1 Main method: Activation-based dual monitors (white-box)

Two lightweight predictors:
* **ValidityMonitor(sᵢ)** → p̂_V(s_i) ∈ [0,1]
* **FaithfulnessMonitor(sᵢ)** → p̂_F(s_i) ∈ [0,1]

**Features per step i:**
* Hidden state at step boundary h_ℓ(t_i) for 1–3 selected layers
* Attention entropy statistics
* Logit margin/entropy at boundary
* "Self-eval token" features (optional)

**Model class:** logistic regression or 2-layer MLP

**Theme Track knob experiment:**
Use probe weight vectors / difference-of-means to define direction v in residual stream; inject +γv at chosen layer during decoding to show controllability.

### 4.2 Strong baselines (must include)

1. **Black-box consensus/stability**
   - Self-consistency: step confidence = frequency of matching steps
   - ICLR'26 step-wise confidence attribution method

2. **Perturbation-agreement explanation uncertainty**
   - UncCoT-style agreement metrics (Tanneru et al.)

3. **Step-removal consistency (ELAD-style)**

4. **Verifier-only** — NLI entailment score as "confidence"

5. **Token-level / logprob heuristics**

6. **Self-reported step confidence**

---

## 5) Ground truth for training

### 5.1 Step validity label (y_V(s_i))

**EntailmentBank**
* Label with NLI verifier: y_V = 1 if Entail(E_i → s_i)
* Use different NLI models for labeling vs evaluation

**FEVER / HotpotQA**
* y_V = 1 if entailed by gold supporting sentences

**ERASER extractive rationale**
* y_V = 1 if overlaps gold rationale above threshold

**Hard negatives:** entity swap, number swap, negation injection, distractor substitution

### 5.2 Step faithfulness label (y_F(s_i))

**Counterfactual deletion / ablation:**
1. Compute p(a | x, full context)
2. Delete step s_i, compute p(a | x, context without s_i)
3. Δ_i = log p(a|full) - log p(a|delete s_i)
4. y_F = 1 if Δ_i > δ (step mattered)

---

## 6) Evaluation metrics

### Per-step calibration
* ECE (15 bins), Brier score, NLL
* Reliability diagrams

### Ranking / selective use
* AUROC / AUCPR for identifying invalid/unfaithful steps
* Selective step filtering: accuracy/validity at coverage c

### Two-dimensional evaluation (key novelty)
* Joint calibration: fraction of steps binned by (p̂_V, p̂_F) that are (valid, faithful)
* Demonstrate: valid-but-unfaithful vs faithful-but-invalid

### Correlation with prediction correctness
* AUC of aggregated step confidence vs answer correctness

---

## 7) Baselines

1. **Answer confidence** — max-prob / entropy of final answer
2. **Explanation uncertainty** — Tanneru et al., ELAD-style
3. **Step-wise confidence (black-box)** — consensus/stability scoring
4. **Verifier-only** — NLI probability calibrated
5. **Representation-only not step-specific** — intermediate answer correctness probe (contrast)

---

## 8) Expected contributions

### Contribution 1: 2D explanation confidence formalization
* **Validity confidence** + **faithfulness confidence**, both calibrated

### Contribution 2: Lightweight activation monitors
* Probe-based, compute-feasible, no pretraining
* Layer-localization analysis: where is validity vs faithfulness encoded?

### Contribution 3: "Knob" evidence via activation interventions
* Controllable direction that trades off step validity/faithfulness
* Directly matches Theme Track emphasis on internal mechanisms

### Contribution 4: Benchmark/evaluation suite
* Step-segmented explanation outputs + labels + evaluation scripts

---

## 9) Compute estimate (< 100 H100 hours)

* Data generation + activation capture: ~30 H100h
* Validity labeling via NLI: ~10 H100h
* Faithfulness labels via deletion: ~25–35 H100h
* Monitor training + calibration: ~5 H100h
* Steering / knob experiments: ~10–15 H100h

**Total: ~80–95 H100 hours**

---

## Novelty defense strategy

1. **Explicitly cite and delimit:**
   - "Not another consensus/stability step-confidence method"
   - "Not only computing uncertainty-of-explanations"
   - "Beyond intermediate answer correctness probes in reasoning-only domains"

2. **Novelty anchor:** calibrated (validity, faithfulness) at step level from internal activations, plus activation knob demonstration on NLP explanation tasks

3. **Key ablations:**
   - Validity vs faithfulness decouple (many valid but unfaithful steps)
   - Probe layer localization differs for validity vs faithfulness
   - Steering improves explanation reliability even when answer accuracy held constant

4. **Additional safeguard:** focus steering on making explanations more faithful without changing final answers (distinct from "truthfulness/hallucination reduction" steering papers)

