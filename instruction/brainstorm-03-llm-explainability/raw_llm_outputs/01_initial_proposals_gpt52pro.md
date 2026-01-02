# Initial Proposals from GPT-5.2 Pro

**Source:** GPT-5.2 Pro  
**Date:** 2025-01-30  
**Prompt:** `01_extend_sokrates.md` (extending optionized reasoning for LLM explainability)

---

## 1) Calibrated explanation validity predictors from *verifiable* traces

1. **One-line pitch**
   Train a lightweight "explanation validity" predictor (step-level + whole-trace) using automatically verified reasoning traces, so explanations come with calibrated confidence—not just plausible prose.

2. **Research question**
   Can we **predict and calibrate** whether an LLM's explanation is *actually valid* (not merely convincing), and does this generalize across tasks and architectures?

3. **Method sketch**

   * **Tasks with automatic verifiers (fast to label):**

     * **Deductive logic:** ProofWriter-style rule reasoning (symbolic entailment checks).
     * **Math:** GSM8K/MATH with expression extraction + SymPy / arithmetic checks (at least for many steps; fall back to final-answer verification when step parsing fails).
     * **Program reasoning (optional):** small code problems with unit tests (validity = tests pass; step validity via intermediate assertions if generated).
   * **Generate structured "explanations"** with a constrained format that makes verification easy, e.g.:

     * Logic: each step is a derived fact + rule used.
     * Math: each step is an equation transformation.
   * **Label** each step as valid/invalid with the verifier; label the trace valid if all steps valid (or compute a continuous score = fraction valid).
   * **Train an explanation-validity head** (cheap LoRA or linear probe) on:

     * (A) the **generated explanation text**, or
     * (B) the base model's **hidden states** at step boundaries (architecture-dependent, but often more predictive).
   * **Calibrate** with temperature scaling / isotonic regression on a held-out set.
   * **Optional "solver-guided" improvement loop:** use the predictor to do rejection sampling or preference-style fine-tuning to increase step validity while monitoring answer accuracy.

4. **Evaluation**

   * **Core metrics (new emphasis):**

     * *Explanation ECE* / reliability diagrams for "trace validity" probability.
     * Brier score, AUROC for valid vs invalid traces.
     * Step-level F1/accuracy for detecting invalid steps.
   * **Downstream utility:**

     * Selective prediction: abstain when explanation-validity is low → measure risk–coverage curves.
     * "Trust gating": only surface explanations above a validity threshold; quantify reduction in incorrect-yet-confident rationales.
   * **Benchmarks & baselines:**

     * Baselines: answer confidence (logprob), self-consistency agreement, "length/fluency" heuristics, an LLM-judge verifier.
     * Compare across **architectures** (decoder-only vs encoder–decoder; optionally RAG) because the CFP explicitly asks for cross-architecture explainability evaluation.
   * **Related yardstick:** use/contrast existing definitions of faithfulness where appropriate.

5. **Expected contribution**

   * A practical, low-compute method to **quantify when explanations are trustworthy**, aligning with the "faithfulness vs plausibility" framing in the theme CFP.
   * A reusable **verifiable-explanation dataset + evaluation protocol** that shifts explainability from "pretty rationales" to "calibrated reasoning quality."

6. **Theme alignment**

   * (2) rigorous evaluation of explanation quality (faithfulness + calibration)
   * (1) architecture-specific adaptation (hidden-state probes differ by architecture)

7. **Compute estimate (H100 hours)**

   * Trace generation + verification: ~10–25 H100h (depends on #examples and model size)
   * Probe/LoRA training + calibration: ~5–15 H100h
   * **Total:** ~15–40 H100h

---

## 2) Robustness-aware evaluation: explanation stability *without rewarding vacuity*

1. **One-line pitch**
   Build an "Explanation Robustness" benchmark where explanations must stay consistent under meaning-preserving perturbations—*but* penalize generic, content-free explanations that are trivially stable.

2. **Research question**
   When inputs are paraphrased or minimally edited while preserving the correct label/answer, do explanation methods remain **stable**, and does stability correlate with **faithfulness** or just **genericness**?

3. **Method sketch**

   * Construct perturbation sets for each instance:

     * Paraphrases, syntactic alternations, entity swaps that preserve semantics, formatting changes.
   * Ensure the perturbation is label-preserving by:

     * Using tasks with ground truth labels (classification), and/or
     * Checking that a strong reference model maintains the same predicted answer, and/or
     * Using semantic equivalence filters (NLI consistency).
   * Collect explanations from multiple methods:

     * CoT rationales (free-form and structured).
     * Extractive rationales (select spans).
     * Token attribution variants (IG/gradients) for smaller models.
   * Define metrics in two axes:

     * **Robustness:** similarity of explanations across perturbations (semantic + structural).
     * **Non-vacuity / informativeness:** how predictive the explanation is of the answer (e.g., train a small "explanation-only" predictor; or measure whether masking rationale tokens hurts performance).
   * Report a Pareto frontier: "stable *and* informative" vs "stable but vacuous".

4. **Evaluation**

   * Tasks: SST-2 / MNLI (classification), FEVER (verification), HotpotQA (multi-hop).
   * Metrics:

     * Explanation robustness score (pairwise similarity / Lipschitz-style).
     * Sufficiency/comprehensiveness-style causal tests (mask top rationale tokens → answer change).
     * Correlation between robustness and faithfulness tests.
   * Baselines:

     * Always-generic explanation ("Because it follows from the text…") to show robustness-only metrics are gameable.
     * Existing faithfulness metrics + human plausibility ratings on a small subset.
   * Motivation: robustness of *evaluation* under paraphrases is an active concern; you can position this as bringing robustness thinking specifically to explanations.

5. **Expected contribution**

   * A concrete protocol to evaluate explanation methods under realistic input variability while **explicitly controlling for vacuity**, which is a common failure mode in explanation scoring.

6. **Theme alignment**

   * (2) rigorous evaluation
   * (1) unified framework across architectures (decoder-only, encoder–decoder, RAG)

7. **Compute estimate (H100 hours)**

   * Perturbation generation (can be CPU + small model) + explanation inference on ~5–10k items with 3–5 perturbations: ~20–60 H100h
   * Optional attribution on a smaller slice: +5–15 H100h
   * **Total:** ~25–75 H100h

---

## 3) Bias auditing via counterfactual *explanation disparity*

1. **One-line pitch**
   Detect biased predictions by measuring how explanations change under controlled counterfactual edits to sensitive attributes—even when the final prediction stays the same.

2. **Research question**
   Can explanation behavior (tokens attended/attributed, rationale content) reliably flag **sensitive-attribute reliance** and predict bias better than output disparity alone?

3. **Method sketch**

   * Use counterfactual pairs:

     * Gender/race/religion swaps in otherwise identical sentences (CrowS-Pairs / StereoSet / BBQ-style; plus templated generation).
     * For toxicity/sentiment, keep meaning constant while swapping identity terms.
   * For each pair, obtain:

     * Output score difference (prediction disparity).
     * Explanation (free-form) + optionally attribution map (where feasible).
   * Define:

     * **Counterfactual Explanation Disparity (CED):** change in explanation embedding / content, plus a targeted component measuring mention/salience of sensitive tokens.
     * **Sensitive-token attribution mass delta:** Δ attribution on sensitive tokens across counterfactual.
   * Train a small detector that predicts "biased reliance" using explanation-derived features; calibrate.

4. **Evaluation**

   * Ground-truth signals:

     * Output disparity threshold (classic).
     * **Causal sensitivity tests:** mask sensitive attribute tokens and see if prediction shifts (stronger notion of reliance).
   * Metrics:

     * AUROC/PR for detecting reliance; calibration of bias flags.
     * False positive analysis: cases where sensitive attribute is legitimately relevant (e.g., occupation stats in biography datasets).
   * Baselines:

     * Output-only bias detectors.
     * Simple lexical rules ("if identity term present → flag").
     * Attribution-only without rationale text; rationale-only without attribution.

5. **Expected contribution**

   * A practical bridge between explainability and fairness: explanations become a **diagnostic sensor** for bias, not just a post-hoc story, matching the theme's explicit question on bias detection.

6. **Theme alignment**

   * (3) biased predictions
   * (2) evaluate explanations as bias detectors

7. **Compute estimate (H100 hours)**

   * Mostly inference on paired sets (few 10k pairs): ~10–30 H100h
   * Optional attribution on subset: +5–15 H100h
   * **Total:** ~15–45 H100h

---

## 4) Explanation-driven training data debugging: find label noise and spurious cues

1. **One-line pitch**
   Use explanation inconsistency signals (self-contradiction, cross-prompt disagreement, label–rationale mismatch) to automatically surface mislabeled or artifact-driven training examples.

2. **Research question**
   Can explanations help **find and fix** problems in training data (label noise, annotation artifacts, spurious correlations) more efficiently than loss-based or influence-based methods?

3. **Method sketch**

   * Pick one dataset where label issues/artifacts are known or can be injected:

     * NLI (SNLI/MNLI), fact verification (FEVER), or toxicity.
   * Train a baseline model (small fine-tune or even prompt-based classifier).
   * For each training point, generate multiple explanations via:

     * Different prompts (rationale styles), temperatures, and/or two different base models.
   * Compute "suspiciousness" features:

     * **Explanation self-consistency:** similarity across samples.
     * **Label entailment:** does the explanation entail the label? (via an NLI checker).
     * **Spurious cue signatures:** explanations over-focus on known artifacts (e.g., hypothesis-only cues in NLI).
   * Rank examples; simulate a human-in-the-loop by "fixing" top-K (in experiments: use injected noise ground truth or use a small curated subset).

4. **Evaluation**

   * **Synthetic noise:** flip labels for X% of data; evaluate precision@K / recall@K for catching flips.
   * **Spurious correlation injection:** add a token correlated with label; evaluate detection of spurious-feature reliance.
   * **Downstream retraining:** remove/relable flagged points; measure:

     * In-distribution accuracy
     * OOD/generalization (stress tests)
     * Bias/fairness changes (if relevant)
   * Baselines:

     * Cleanlab / confident learning
     * High-loss filtering
     * Influence-function approximations (or tractable variants)

5. **Expected contribution**

   * A fast, explanation-centric data quality pipeline that operationalizes the theme's "use explanations to fix training data" question.
   * Insight into *what kinds* of dataset errors explanations are uniquely sensitive to (vs loss).

6. **Theme alignment**

   * (4) training data debugging
   * (2) evaluation (does explanation-based filtering improve generalization?)

7. **Compute estimate (H100 hours)**

   * Explanation generation over a medium dataset (50k–200k items) can be heavy; keep to 10k–30k or sample: ~15–50 H100h
   * Retraining small models: ~5–20 H100h
   * **Total:** ~20–70 H100h

---

## 5) Mechanistic "knobs" for faithful reasoning vs post-hoc rationalization

1. **One-line pitch**
   Identify internal components that causally control whether a model produces *faithful* reasoning traces, and show targeted interventions that improve faithfulness without large-scale retraining.

2. **Research question**
   Can we find **specific mechanisms / directions** in activations that govern "valid reasoning" behavior, and can we manipulate them to change explanation faithfulness (not just answer accuracy)?

3. **Method sketch**

   * Work with an open-weight model small enough for interpretability tooling (e.g., 7B–13B).
   * Assemble paired sets:

     * Same question type, correct answers, but **valid vs invalid** reasoning traces (from verifiable tasks; Proposal #1's data works).
   * Use mechanistic tools (TransformerLens-style):

     * **Causal tracing / activation patching:** patch layer activations from valid-trace runs into invalid-trace runs and measure changes in step validity and/or trace score.
     * **Linear probes** on residual stream to predict step validity; localize layers/heads with strongest signal.
   * Interventions:

     * Head ablation or activation steering along a "validity direction".
     * Compare to cheap baselines like rejection sampling by verifier.

4. **Evaluation**

   * Primary:

     * Change in step validity rate / proof validity rate at fixed answer accuracy.
     * Trade-off curves: (faithfulness ↑) vs (accuracy ↓) vs (length/latency).
   * Secondary:

     * Does the identified circuit generalize across tasks (logic → math) or prompts?
   * Baselines:

     * No intervention
     * Self-consistency sampling
     * Lightweight fine-tuning (LoRA) without mechanistic targeting

5. **Expected contribution**

   * A concrete example of the "identify knobs controlling high-level behaviors" agenda in the CFP—here, the high-level behavior is *faithful explanation generation*.
   * Mechanistic evidence (via causal interventions) is stronger than correlational "attention heatmaps".

6. **Theme alignment**

   * (5) mechanisms / knobs in activations
   * (1) architectural considerations (what transfers across architectures?)

7. **Compute estimate (H100 hours)**

   * If you keep the analysis set small (e.g., 200–1,000 prompts) and target a subset of layers/heads: ~10–35 H100h
   * Optional small LoRA for comparison: +5–15 H100h
   * **Total:** ~15–50 H100h

---

## Practical recommendation under your constraints

If you want **maximum publishability in 7 days** with **low compute risk**, the strongest bets are:

* **(1) Calibrated explanation validity** (clean story, automatic labels, clear metrics)
* **(3) Bias auditing via counterfactual explanation disparity** (clear societal impact + measurable protocol)

If you want a more "theme-track signature" paper emphasizing internal workings, go with:

* **(5) Mechanistic knobs** (but keep scope tight to one behavior + one model).

