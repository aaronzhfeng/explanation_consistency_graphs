# X-UNIT: Novelty-Checked Benchmark Proposal from GPT-5.2 Pro

**Source:** GPT-5.2 Pro  
**Date:** 2025-01-30  
**Follow-up Prompt:** Novelty audit of X-FACT proposal

---

## 1) Nearest-neighbor literature check: what already exists

### A. Benchmarks/frameworks for **token-level rationales / feature attributions**

* **ERASER**: multi-task rationale benchmark with sufficiency/comprehensiveness
* **Wang et al. (CoNLL 2022)**: token-level rationales + rationale consistency-under-perturbation metric
* **M4 (NeurIPS 2023)**: unified benchmark for faithfulness evaluation of feature attribution methods
* **EvalxNLP (2025)**: Python benchmarking framework for post-hoc feature attribution on transformer classification
* **OpenXAI (2022)**: comprehensive evaluation across faithfulness, stability, fairness (mostly tabular)

**Novelty risk:** "unified framework for feature attributions/rationales with faithfulness+stability" is crowded.

### B. Benchmarks/frameworks for **natural-language explanations / CoT faithfulness**

* **Counterfactual Simulatability** (Chen et al., ICML 2024): evaluating explanations via behavior prediction
* **Extension to generation tasks** (Limpijankit et al., INLG 2025): simulatability for summarization/medical
* **ALMANACS** (ICLR 2024): automated simulatability benchmark
* **LExT (2025)**: trustworthiness of LLM explanations (plausibility + faithfulness)
* **FaithCoT-Bench (2025)**: instance-level CoT faithfulness benchmark
* **RFEval (ICLR 2026 submission)**: reasoning faithfulness evaluation across seven tasks

**Novelty risk:** "multi-task CoT/NLE faithfulness with perturbations/counterfactuals" is directly addressed by FaithCoT-Bench and RFEval.

### C. "Validity of steps" / process-level evaluation

ProcessBench/PRM-oriented benchmarks for step-level reasoning trace evaluation.

### D. Calibration / uncertainty in explanations

Work exists but not yet standardized as benchmark + leaderboard.

### E. Meta-warning

Explanation metrics are gameable (Goodharting).

---

## 2) What *isn't* already "the exact same angle" (the gap you can still own)

### Gap G1 — Cross-explanation-type *comparability* on the same instances

Most benchmarks are siloed by explanation modality (rationales vs CoT).

**Novel contribution:** same items support evaluating (i) token/cell rationales, (ii) attribution maps, and (iii) natural-language step explanations under shared tests.

### Gap G2 — Calibration of explanation reliability

**Novel contribution:** explanation-level ECE/Brier and risk–coverage curves where the label is *unit validity/faithfulness pass-fail*.

### Gap G3 — Metamorphic "unit tests": invariance **and** sensitivity

**Novel contribution:** requiring invariance when perturbations are semantics-preserving AND sensitivity when semantics-changing.

### Gap G4 — Generation tasks with *groundable* explanation objects

**Novel contribution:** generation tasks with structured selection (cells, triples, citations).

---

## 3) Revised benchmark proposal

### Benchmark name + pitch

**X-UNIT: Explanation Unit Tests for NLP**

**Pitch:**
X-UNIT evaluates explanation quality by converting explanations into **testable units** and scoring them via **unit tests** along four axes:

1. **Unit Validity** (is each explanation unit correct/grounded?)
2. **Causal Faithfulness** (do units correspond to causally influential evidence/features?)
3. **Metamorphic Robustness** (invariance + sensitivity under controlled perturbations)
4. **Explanation Calibration** (are confidence statements calibrated to unit-level correctness/faithfulness?)

**Novelty claim (defensible vs prior art):**

* Unlike ERASER/CoNLL'22/M4/EvalxNLP, X-UNIT is not limited to highlight/attribution explanations; it also evaluates natural-language step explanations under the *same* instance set and test suite.
* Unlike RFEval/FaithCoT-Bench/simulatability benchmarks, X-UNIT adds a calibration track and explicitly measures invariance/sensitivity as a metamorphic contract across explanation modalities.

---

## 4) Concrete design choices to avoid "already done"

### 4.1 Task selection (5 tasks)

1. **Multi-hop QA (HotpotQA-style)** — supporting sentences, CoT steps, token attributions
2. **NLI with explanations (e-SNLI-like)** — free-text explanation + rationale highlights
3. **Classification with rationales (HateXplain-like)** — token highlights + attributions + NLE
4. **Math reasoning (GSM8K subset)** — CoT steps with mechanical validity checks
5. **Generation with structured grounding (ToTTo)** — highlighted table cells

### 4.2 Explanation format standardization (the "unit" abstraction)

Standardized JSON schema per instance:

* **answer**: final output
* **explanation_units**: list with:
  * `type ∈ {token_span, sentence_id, table_cell_id, free_text_step}`
  * `content`
  * `confidence ∈ [0,1]` (mandatory)
* **global_expl_confidence ∈ [0,1]** (optional)

### 4.3 Evaluation dimensions + metrics

#### (A) Unit validity

* **Unit Validity Rate (UVR):** % of units individually verified
* **Invalidity Mass:** sum of confidences assigned to invalid units

#### (B) Causal faithfulness

* **Interventional Evidence Faithfulness (IEF):** Sufficiency@k, Comprehensiveness@k, AUC over k
* **Metamorphic Causal Agreement (MCA):** stability of selected causal units under perturbations

#### (C) Stability under perturbations (bidirectional)

Two disjoint perturbation sets:
* **P⁰ (invariance set):** paraphrases, irrelevant edits
* **P¹ (sensitivity set):** controlled semantic edits

Metrics:
* **Invariant Stability Score (ISS):** explanation similarity on P⁰
* **Sensitivity Appropriateness Score (SAS):** explanation difference on P¹

#### (D) Calibration of explanation confidence

Binary labels per unit:
* `y_valid(u) ∈ {0,1}` from validity checks
* `y_faith(u) ∈ {0,1}` from causal faithfulness tests

Metrics:
* **Explanation-ECE (E-ECE)**
* **Faithfulness-ECE (F-ECE)**
* **Selective Explanation Risk (SER):** risk–coverage curve

---

## 5) Baseline methods

### Attribution / rationale baselines

* Integrated Gradients, Gradient×Input, attention rollout, LIME/SHAP
* Rationale-from-attribution: top-k tokens/cells
* Rationale model (optional)

### Natural-language explanation baselines

* Post-hoc rationalization
* CoT prompting (with/without self-consistency)
* "Forced" explanation variants (sanity checks)

### Calibration baselines

* Uncalibrated (raw confidences)
* Temperature scaling / isotonic regression
* Confidence regularization prompt

---

## 6) Expected empirical patterns

1. **Weak correlation between plausibility and faithfulness**
2. **High task accuracy with low explanation faithfulness**
3. **Stability is not monotone-good** — methods stable but fail to react when they should
4. **Explanation confidence is overconfident and poorly calibrated**

---

## 7) Deliverables

1. **Dataset release** — base instances + perturbation sets + standardized schema
2. **Evaluation harness** — one command for all metrics
3. **Leaderboard** — separate per axis + Pareto front (avoid single scalar)
4. **Reporting checklist**

---

## 8) Minimal novelty-safe positioning statement

> Existing resources benchmark isolated facets of explainability—e.g., faithfulness of token attributions (M4/EvalxNLP), rationale faithfulness and overlap (ERASER), perturbation consistency for rationales (CoNLL'22), or reasoning faithfulness/simulatability for CoT/NLEs (RFEval, FaithCoT-Bench, ALMANACS, counterfactual simulatability). X-UNIT unifies these into a single **unit-test-based** benchmark that evaluates *multiple explanation modalities on the same instances* and introduces **calibration of explanation reliability** as a first-class, quantitatively scored benchmark axis.

---

## References

[1]: https://www.eraserbenchmark.com/ERASER_Rationales_Benchmark.pdf "ERASER Benchmark"
[2]: https://proceedings.neurips.cc/paper_files/paper/2023/file/05957c194f4c77ac9d91e1374d2def6b-Paper-Datasets_and_Benchmarks.pdf "M4: A Unified XAI Benchmark"
[3]: https://arxiv.org/html/2505.01238v1 "EvalxNLP"
[4]: https://arxiv.org/pdf/2206.11104 "OpenXAI"
[5]: https://proceedings.mlr.press/v235/chen24bl.html "Counterfactual Simulatability"
[6]: https://arxiv.org/abs/2505.21740 "Counterfactual Simulatability for Generation Tasks"
[7]: https://openreview.net/forum?id=KJzwUyryyl "ALMANACS"
[8]: https://dl.acm.org/doi/full/10.1145/3715275.3732104 "LExT"
[9]: https://openreview.net/forum?id=lN3yKqqzF1 "FaithCoT-Bench"
[10]: https://openreview.net/forum?id=2Gc8aj0afg "RFEval"
[11]: https://www.arxiv.org/pdf/2510.04040 "FaithCoT-Bench PDF"
[12]: https://qwenlm.github.io/blog/qwen2.5-math-prm/ "ProcessBench"
[13]: https://aclanthology.org/2024.findings-eacl.88.pdf "Goodhart's Law Applies to NLP's Explanation Benchmarks"
[14]: https://aclanthology.org/2020.emnlp-main.89.pdf "ToTTo"
[15]: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/698d51a19d8a121ce581499d7b701668-Paper-round1.pdf "Review of Datasets for Explainable NLP"
[16]: https://aclanthology.org/2025.inlg-main.38.pdf "Counterfactual Simulatability for Generation"

