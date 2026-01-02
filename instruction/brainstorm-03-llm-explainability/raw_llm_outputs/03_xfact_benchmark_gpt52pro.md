# X-FACT Benchmark Proposal from GPT-5.2 Pro

**Source:** GPT-5.2 Pro  
**Date:** 2025-01-30  
**Prompt:** `02_explanation_faithfulness_benchmark.md`

---

## 1. Benchmark name and pitch

**X-FACT**: **eX**planation **F**aithfulness, **A**ccuracy/validity, **C**onsistency, and calibra**T**ion.

**Pitch.** X-FACT is a unified benchmark + evaluation harness that scores *explanations* (not just answers) across heterogeneous NLP tasks. It standardizes:

* **Validity**: are explanation steps/evidence *correct* (step-level where possible)?
* **Faithfulness**: do explanations track what the model *actually used* (causal/perturbation tests)?
* **Stability**: are explanations invariant to meaning-preserving perturbations and sensitive to meaning-changing counterfactuals?
* **Calibration**: do models' *explanation confidence* estimates match explanation correctness?

The benchmark is motivated by the current fragmentation of reasoning-trace/explanation evaluation practices and the need for unified, quantitative criteria. 

---

## 2. Task selection

Design goal: pick tasks where (i) *explanation ground truth or verification is feasible*, (ii) *formats vary* (classification, QA, generation), (iii) compute stays low.

### Core tasks (ACL-style diversity)

1. **Math reasoning (step-verifiable)**: **GSM8K**

* Why: step-level arithmetic validity can be *programmatically verified*; isolates "right answer, wrong steps."
* License: MIT (HF dataset card).

2. **Multi-hop QA with sentence-level supporting facts**: **HotpotQA**

* Why: includes **supporting-facts supervision** (sentence-level evidence), enabling evidence-based explanation evaluation.
* License: CC BY-SA 4.0.

3. **NLI with human natural-language explanations**: **e-SNLI**

* Why: canonical dataset for *natural language explanations* tied to entailment/contradiction decisions.
* License: MIT (repo).

4. **Fact verification with evidence sentences**: **FEVER**

* Why: forces grounded explanations; gold evidence sentences allow rationale evaluation.
* Licensing note: annotations incorporate Wikipedia-licensed material; keep packaging consistent with FEVER's stated terms.

5. **Controlled generation with gold content selection**: **ToTTo (table→text)**

* Why: generation task with explicit **highlighted cells** (gold content plan) enabling explanation evaluation at the *content-selection* level.
* License: CC BY-SA 3.0 (ToTTo repo).

### Unifying interface across tasks

Each instance is represented in a common schema:

* `input` (text/table), `label/target`, `model_output`,
* `gold_support` (optional: evidence sentences / highlighted cells / human explanation),
* `perturbations_invariant[]`, `perturbations_counterfactual[]`,
* `explanation` (method output: spans/sentences/steps + optional confidences).

This enables one evaluation harness to work across tasks and explanation *types*.

---

## 3. Evaluation dimensions

X-FACT defines four dimensions; each has **task-agnostic test protocols** and **task-specific instantiations**.

### A. Validity (step/evidence correctness)

"Does the explanation constitute a correct justification for the output, relative to input/task?"

Includes:

* step correctness (math/logical steps),
* evidence correctness (did you cite the right support?),
* internal consistency (no contradictions across steps).

### B. Faithfulness (causal linkage to model behavior)

"Does the explanation reflect what *actually influenced* the model's prediction?"

We operationalize faithfulness via *interventions/perturbations* and *input reconstruction* tests, aligning with recent work that evaluates explanation faithfulness by checking whether changing explanation-relevant content changes predictions.

### C. Stability (robustness and equivariance)

"Are explanations stable when the task semantics are unchanged, and appropriately different when semantics change?"

* **Invariance**: paraphrase/format/noise should not change explanation much (if prediction stable).
* **Equivariance**: counterfactual edits that flip the label/answer should change explanations.

### D. Calibration (explanation confidence)

"Do confidence estimates attached to explanations correlate with explanation correctness?"

Distinct from answer calibration: we calibrate *explanation correctness* (step validity / evidence correctness / faithfulness test pass rates) against reported confidence or derived confidence scores.

---

## 4. Metrics

Below are concrete metrics and how to compute them.

### 4.1 Validity metrics

#### (V1) **Answer accuracy / task score**

* Standard dataset metric (Acc/F1/EM/BLEU as relevant), kept for context.

#### (V2) **Step Validity Rate (SVR)** — GSM8K (and optionally other step tasks)

For a generated chain of thought split into steps (s_1,\dots,s_T):

* Parse each step for arithmetic equations / numeric updates.
* Verify with a symbolic/arithmetic checker (Python) whether the step is locally valid.
* **SVR** = (1/T) × Σ 1[verify(s_t) = true]

Also report:

* **First-error position** (where reasoning first fails),
* **Error propagation rate** (later steps consistent with a wrong intermediate).

#### (V3) **Evidence F1 / Supporting-Facts F1** — HotpotQA, FEVER

Let gold evidence set be G (sentences) and predicted evidence E:

* Precision = |E∩G|/|E|, Recall = |E∩G|/|G|, F1.

#### (V4) **Cell Selection F1** — ToTTo

Gold highlighted cells G, predicted selected cells E; same F1 definition.

#### (V5) **Explanation Entailment Consistency (EEC)** — e-SNLI

Treat explanation as a set of sentences/claims (c_1,...,c_K).
Use an NLI verifier (off-the-shelf DeBERTa/RoBERTa NLI) to score:

* v(premise → c_k) and v(c_k → hypothesis relation)

Define:

* **EEC** = (1/K) × Σ 1[entailed(c_k) ∧ label-consistent(c_k)]

---

### 4.2 Faithfulness metrics

#### Faithfulness for **extractive rationales / attributions** (tokens/sentences/cells)

**(F1) Sufficiency & Comprehensiveness** (ERASER-style)

* **Sufficiency**: does rationale alone retain the prediction?
* **Comprehensiveness**: removing rationale should hurt prediction

**(F2) Deletion/Insertion AUC**

**(F3) Infidelity & Sensitivity** (attribution perturbation)

**(F4) Sanity checks** — Randomize model weights or labels

#### Faithfulness for **natural-language explanations / CoT**

**(F5) Input Reconstruction Faithfulness (IRF)**

1. Extract entities/relations/numbers asserted in the explanation.
2. Build a minimal "reconstructed input" that encodes only these reasons.
3. Measure IRF = 1[f(x^recon) = f(x)] or probability retention.

**(F6) Contrary-Hint Sensitivity (CHS)**

1. Identify a key claim p in explanation.
2. Construct a contrary hint ¬p.
3. Compare distributions: D(f(x, e), f(x, e^contrary)).

**(F7) Step Corruption Sensitivity (SCS)** (math + structured steps)

Corrupt a single intermediate value/step; measure answer flip rate.

---

### 4.3 Stability metrics

#### Perturbation sets

* **Invariant**: paraphrase, entity renaming, formatting noise, irrelevant distractors.
* **Counterfactual**: minimal edits that flip label/answer.

#### Metrics

**(S1) Prediction invariance rate**

**(S2) Explanation invariance similarity**

* For rationales: Jaccard@k
* For attribution: Spearman rank correlation
* For NLE/CoT: BERTScore + numeric/entity agreement

**(S3) Explanation equivariance** — low similarity when prediction flips

**(S4) Stability Gap** = E[Sim | inv] − E[Sim | cf]

---

### 4.4 Calibration metrics (explanation confidence)

**(C1) Explanation Expected Calibration Error (E-ECE)**

**(C2) Brier score** on explanation correctness

**(C3) Selective explanation curves** — coverage vs explanation error, AURC

---

## 5. Baseline methods

### 5.1 Model families (compute-feasible)

1. **Encoder classifier**: DeBERTa-v3-base / RoBERTa-large
2. **Seq2seq generator**: T5-base / Flan-T5-base
3. **Open instruction-tuned LLM (~7–8B)** for CoT/NLE baselines

### 5.2 Explanation methods to benchmark

#### A. Attribution / rationale methods

* Attention rollout, Gradient×Input, Integrated Gradients, SmoothGrad, Occlusion, LIME
* Random + length-matched random (sanity baseline)
* Oracle gold evidence (upper bound)

#### B. Natural language explanation methods (LLMs)

* CoT prompting (few-shot)
* Answer→Explain (post-hoc rationalization)
* Evidence-first prompting
* Self-consistency
* Confidence-annotated CoT

---

## 6. Expected findings

1. **Answer accuracy weakly correlates with explanation validity**
2. **Plausibility ≠ faithfulness**
3. **Attention-based explanations fail sanity checks more often**
4. **Gradient/IG tends to be more faithful but less stable**
5. **Evidence-first prompting improves evidence F1 but not necessarily faithfulness**
6. **Explanation confidence is substantially miscalibrated**

---

## 7. Resource deliverables

1. **Dataset package** — HuggingFace-style with standardized fields
2. **Evaluation toolkit ("X-FACT Harness")** — CLI + Python API
3. **Baseline outputs + checkpoints**
4. **Leaderboard**
5. **Reporting checklist**

---

## 8. Compute estimate (H100 hours)

* Fine-tuning: ~15–20h
* LLM inference for NLE/CoT: ~35–55h
* Attribution + masking metrics: ~10–15h
* Faithfulness tests for NLEs: ~5–10h

**Total estimate:** ~65–87 H100 hours (under 100h budget)

