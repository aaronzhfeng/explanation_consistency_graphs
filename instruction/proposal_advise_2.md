# Literature-Informed Proposal Review — Claude

**Prompt:** `prompts/14_literature_proposal_review.md`  
**Attachments:** `literature.md`, `proposals/02_ecg.md`  
**Date:** [PENDING]

Sources: curated ECG literature bank (103 papers)  and the current ECG proposal draft .

## 1. Literature coverage assessment

### What the proposal is already well-supported by

**A. “Graph over representations to surface label issues” is strongly grounded.**
The proposal’s core move—build a kNN graph in an embedding space and flag points whose labels disagree with their neighbors—is directly aligned with graph/NN noisy-label methods (#48, #52, #53, #55). The specific “neighbor agreement → suspicious label” logic is essentially the Deep kNN noisy-label intuition, but applied to explanation embeddings (#48). “Graph-based joint label error + outlier detection” is also a close conceptual match to Neural Relation Graph (#52).

**B. “Explanations can expose artifacts/spurious behavior” is well-supported.**
The motivation that loss/confidence can miss artifact-aligned errors while explanations can expose shortcut reliance is consistent with explanation-based debugging and artifact discovery lines: survey framing (#18), FIND-style workflows (#19), interactive label cleaning using explanations (#23), and especially TFA for artifact detection (#24) and influence-based artifact analysis (#25). The proposal’s intended narrative (explanations are useful because they help find/fix data issues) fits squarely within these precedents (#18, #19, #23–#26).

**C. “Artifact/spurious correlation regimes matter, and ‘simple baselines’ can be misleading.”**
The proposal’s synthetic “artifact-aligned” regime is in the same conceptual space as spurious correlation diagnostics and debiasing literature, especially the NLI artifact lineage (#60) and robustness/debiasing methods (#63–#66, #75). The importance of stress tests like “strip/swap spurious tokens” is consistent with counterfactual robustness ideas (#69–#71).

**D. “Faithfulness evaluation is necessary (and nontrivial).”**
The proposal’s inclusion of comprehensiveness/sufficiency aligns with ERASER (#76). The broader need for careful faithfulness evaluation and robustness-style checks is strongly supported (#79–#82, #80).

### Gaps: where the proposal assumes more than the literature supports

**1) Assumption: LLM evidence spans/rationales are reliable enough to drive cleaning.**
Multiple papers emphasize that natural-language rationales can be plausible but unfaithful (#41, #43), and that self-explanations vary dramatically by type and model family (#40, #42). The proposal does add an NLI “contradiction” check, but it does not address the broader “plausible-but-wrong rationale” risk identified in the LLM rationale faithfulness literature (#40–#43).

**2) Assumption: prompt-only JSON formatting + “retry/repair” is sufficient.**
The literature bank includes multiple structured-output works suggesting prompt-only format adherence is brittle—especially for open models (#34)—and recommending constrained decoding or schema enforcement (#32, #33, #36) or lightweight post-processing (#35). The proposal mentions JSON repair, but doesn’t leverage these more robust techniques.

**3) Assumption: NLI contradiction is a stable, trustworthy verifier.**
While using entailment/contradiction as a verifier is directionally consistent with “verify explanations” approaches (#44) and “don’t trust plausible explanations” warnings (#41), the proposal does not discuss judge reliability issues (even if the “judge” is an NLI model rather than an LLM). The literature bank’s judge-reliability survey (#37) is directly relevant to the general “model-as-judge” failure modes (prompt sensitivity, bias, instability), and should motivate calibration/ablation of the NLI component.

**4) Assumption: explanation embeddings will produce semantically meaningful neighborhoods by default.**
Graph-based NLP success depends heavily on representation quality (#54). The proposal’s embedding choice is pragmatic, but the literature contains explicit methods that address noisy neighborhoods and representation noise: adaptive weighting with reliability (#49), dual representations (#51), and robust contrastive learning under noisy pair structure (#57, #58). These are not leveraged.

**5) Assumption: removal of top-K “suspicious” points is generally safe.**
Training-dynamics work warns that “hard/ambiguous but correct” points can look suspicious by various signals (#08), and that different dynamics-based measures capture different failure modes (#06–#08). The proposal includes removal and a guarded relabel option, but does not explicitly address the risk of discarding minority/rare-but-correct cases or ambiguous examples (a known practical failure mode in label cleaning pipelines).

### High-value papers in the collection the proposal under-leverages

If you only add a few citations/ideas, these are the most “proposal-shaping” omissions:

* **Structured output enforcement:** #32, #33, #36 (plus #34 as the cautionary benchmark; #35 as fallback).
* **LLM explanation reliability / stability:** #40–#43; explanation-consistency training: #47; uncertainty-aware rationales: #46 (and #45 as context).
* **Graph methods that go beyond plain kNN voting:** adaptive weighted kNN with reliability (#49); Neural Relation Graph as a closer baseline and method template (#52); GCN/label propagation correction (#53, #54).
* **Robust rationale evaluation & leakage:** label–rationale leakage (#88); REV (#89) and RORA (#90).
* **Faithfulness metrics pitfalls & fixes:** metric disagreement (#79); sensitivity/stability (#80); ROAR/Recursive ROAR (#82, #81); Goodhart effects (#78).
* **Additional strong non-graph baselines:** AUM (#06), CTRL (#07), PLM loss ranking on human-originated noise (#05), DeCoLe for label-biased noise (#04).
* **Influence/data attribution baselines:** TracIn (#93), TRAK (#94), DataInf (#59).

---

## 2. Methodology refinements grounded in the literature

### 2.1 Explanation generation approach

**Refinement A — Make JSON/schema compliance a first-class design constraint.**
Instead of “prompt → parse → repair,” use:

* **Grammar / schema-constrained decoding** (#32, #36) and/or subword-aligned constrained decoding (#33) to guarantee parseability.
* Use **FOFO** (#34) as justification that open models often fail strict formatting without enforcement.
* If constrained decoding is unavailable/slow, use a **post-processing schema fixer model** (#35).

Why this matters for ECG: downstream graph construction is brittle to parsing failures and partial outputs; the literature suggests structural guarantees are achievable (#32, #36) and often preferable to ad-hoc repair.

**Refinement B — Reduce explanation variance; measure it explicitly.**
Explanation variance is a known issue; stability varies by explanation type/model (#40), and there are methods to improve consistency (#47). Practical changes:

* Generate **2–5 explanations per instance** with different seeds (or prompt paraphrases) and compute a **stability score** (e.g., evidence-span overlap / rationale embedding cosine variance). Use stability as an additional signal (low stability → ambiguous/noisy).
* Consider “explanation-consistency finetuning” ideas (#47) as a future extension, but you can implement the *measurement* immediately.

**Refinement C — Prefer verifiable explanation types (and add one).**
The proposal already demands **extractive evidence spans**, which are more verifiable than free-form rationales (#39). Add a second, checkable field:

* A **counterfactual**: “minimal change that would flip the sentiment label” (or “if X were removed/changed, the label would change”). Counterfactual self-explanations are argued to be more verifiable in practice (#42).
  This supports more robust contradiction/verification signals than NLI over a one-sentence rationale.

**Refinement D — Guard against label leakage in the explanation text.**
Rationales can leak label information (#88). In ECG this can create *trivial* clustering in the explanation embedding space (“positive” rationales cluster), which can inflate neighborhood agreement and distort detection.
Concrete prompt constraint:

* Forbid label words in rationale (“positive/negative”) and require rationale to describe evidence without naming the class.
  Then use NLI only on a *separate* hypothesis statement (as you already do).

**Refinement E — Add perturbation-based verification as a signal.**
Input perturbation tests are a recommended route to faithfulness (#44). You can implement a lightweight version:

* If evidence spans are removed from the input, does the **classifier** confidence drop? (This is aligned with ERASER-style evaluation #76, but here used as a per-instance *quality* signal.)
* If removing evidence does **not** affect the classifier prediction, downweight that explanation (it may be unfaithful).

This directly addresses plausibility/faithfulness concerns (#41, #43).

---

### 2.2 Graph construction methodology

**Refinement A — Use reliability-weighted kNN (not just similarity-weighted).**
WANN (#49) is directly on-point: neighbor votes should be downweighted if the neighbor itself is unreliable. In ECG terms:

* Compute a **node reliability** score (e.g., low NLI-contradiction, high explanation stability, high perturbation faithfulness).
* Set edge weight as: similarity × reliability(neighbor).
  This should reduce cascading errors where a cluster of mislabeled/artifact-driven points reinforce each other.

**Refinement B — Add an explicit outlier/OOD track in the graph stage.**
Neural Relation Graph (#52) explicitly targets joint label error detection + outlier detection. ECG currently treats everything as label inconsistency. Add:

* A “graph outlier score” (e.g., low maximum similarity, low average similarity to kNN).
  This separates “mislabeled but in-distribution” from “atypical/OOD” examples, reducing false positives.

**Refinement C — Consider label propagation / GCN for *correction*, not only detection.**
Graph-based noisy-label work often goes beyond “flag and drop”:

* Use label propagation or GCN smoothing to produce **soft corrected labels** (#53, #54).
  This aligns with your “relabel-with-guardrails,” but replaces hard heuristics with a standard graph denoising mechanism.

**Refinement D — Multi-view graphs: explanations + model representations.**
DualKNNText (#51) motivates combining representation views. ECG can be stronger if it uses:

* View 1: explanation embedding space
* View 2: classifier embedding (e.g., [CLS] representations) or predicted-probability vectors
  Then combine neighbor sets or aggregate inconsistency across views. This helps when LLM explanations are noisy or when explanation embeddings collapse.

**Refinement E — Robust embedding learning for explanations (optional, but literature-backed).**
If explanation embeddings are noisy, robust contrastive objectives can help (#58) and confident selection for contrastive learning under noise is relevant (#57). Even a small contrastive fine-tuning of the sentence encoder on explanation pairs could stabilize neighborhoods.

---

### 2.3 Signal combination and scoring

**Refinement A — Add an explanation-stability / uncertainty signal.**
Use either:

* variance across multiple generated explanations (#40, #46), or
* an explicit uncertainty-aware rationale method idea (#46; #45 context).
  This provides a principled way to downweight explanations that are inconsistent/unstable, reducing false positives.

**Refinement B — Replace fixed weights with reliability-adaptive aggregation.**
Given metric instability concerns (#79) and judge reliability issues (#37), fixed weights are fragile. A literature-consistent alternative:

* Compute each signal’s *calibrated* percentile and a *confidence* (e.g., for NLI, margin between contradiction and entailment; for graph, max neighbor similarity; for artifact score, coverage of evidence tokens).
* Combine via **weighted median or trimmed mean**, where unreliable signals get lower weight.
  This is more robust than a fixed 0.4/0.4/0.2 heuristic.

**Refinement C — Integrate training-dynamics signals as additional features.**
Your literature bank contains strong “non-explanation” noisy-label detectors that often capture different failure modes:

* AUM (#06), CTRL (#07), second-split forgetting (#08).
  Add these as extra features into the final ranker (even if you keep it unsupervised, you can do rank-aggregation). This improves robustness when explanation signals fail.

**Refinement D — Add a “label leakage / low-information rationale” penalty.**
Use measures inspired by:

* label–rationale association (#88)
* rationale information metrics like REV (#89) or robust evaluation like RORA (#90)
  to downweight rationales that are label-predictive without real grounding. This is directly relevant since ECG uses rationales both for graph structure and contradiction tests.

---

### 2.4 Baseline selection improvements (what you should add)

Your current baseline set is good but incomplete relative to the curated bank. Add at least:

**Noisy-label detection baselines**

* **AUM** (#06) — training-dynamics baseline often strong when loss ranking fails.
* **CTRL** (#07) — loss-curve clustering is a different dynamics signal.
* **DeCoLe** (#04) — relevant if you claim strength on label-biased/artifact-aligned noise.
* **Neural Relation Graph** (#52) and/or **WANN** (#49) — closest graph-based comparators.
* **DivideMix / ELR / CoDC** (#99, #100, #101) — robust training alternatives (to show ECG’s value vs training-time robustness).

**Attribution/influence baselines**

* **TracIn** (#93) and **TRAK** (#94) as scalable influence baselines.
* **DataInf** (#59) if you want an influence-style method already shown on modern NLP models.

**Artifact/spurious mitigation baselines (downstream robustness)**

* **Product-of-Experts / bias modeling / self-debiasing** (#63–#65) and/or **GroupDRO** (#75) when you evaluate artifact-OOD robustness. This answers: “Is cleaning better than training-time debiasing?”

---

### 2.5 Evaluation metrics refinements

**Refinement A — Faithfulness metrics: avoid masking OOD artifacts.**
Your Comp/Suff definitions are ERASER-aligned (#76), but naive masking is known to be misleading because masked inputs are OOD. Use:

* **ROAR** (#82) or **Recursive ROAR** (#81) for a smaller subset evaluation (even if expensive, do it on a sample) to validate that conclusions aren’t artifacts of masking.

**Refinement B — Add leakage-aware rationale evaluation.**
Given label leakage risks (#88) and Goodhart-style metric gaming (#78), include at least one leakage-robust rationale metric:

* **REV** (#89) and/or **RORA** (#90).
  Also report a simple “label-from-rationale-only” predictability check (motivated by #88).

**Refinement C — Add stability/sensitivity of explanations.**
Include robustness-style explanation properties (#80): stability across seeds/prompt variants and sensitivity to small input perturbations.

**Refinement D — Add a simulatability-style evaluation.**
If you want an explanation-focused contribution, include:

* human simulatability protocol (#91) *or* automated simulatability (ALMANACS) (#92) on a manageable subset.

**Refinement E — Align robustness evaluation with established protocols.**
ER-Test emphasizes OOD/contrast sets/functional tests for explanation-related interventions (#30, #77). Even if you keep SST-2 minimal, incorporate at least one contrast/counterfactual test inspired by counterfactual augmentation work (#70, #71).

---

## 3. Risk analysis from the literature (failure modes the proposal under-addresses)

**Risk 1 — “Plausible but unfaithful” LLM rationales mislead cleaning.**
LLM rationales can sound good yet fail faithfulness checks (#41, #43). If ECG’s graph is built from such rationales, the graph can cluster by stylistic patterns rather than true decision-relevant content. Mitigation: perturbation-based verification (#44) + stability checks (#40) + leakage penalties (#88).

**Risk 2 — Explanation type/model choice strongly affects reliability.**
Faithfulness differs by explanation type and model family (#40, #42). A single “rationale + evidence” format may not be robust. Mitigation: multi-type explanations (extractive + counterfactual) and treat disagreement as uncertainty (#46).

**Risk 3 — Judge reliability: NLI contradictions can be noisy or biased.**
Even if you use an NLI model (not an LLM judge), the general “judge reliability” warning applies (#37). Without calibration/ablation, NLI may introduce systematic false flags. Mitigation: calibrate thresholds per dataset; use margins; include ablations.

**Risk 4 — Neighborhood inconsistency flags ambiguity/hard cases, not just mislabels.**
Training-dynamics work shows signals often correlate with “hard/ambiguous” examples as well as mislabels (#08). Dropping such cases may harm generalization or minority patterns. Mitigation: separate “outlier” vs “label error” tracks (#52), and prefer *correction/reweighting* over removal (GCN smoothing #53; robust training #100).

**Risk 5 — Faithfulness evaluation can be gamed or unstable.**
Optimizing for explanation metrics can cause Goodhart effects (#78), and different faithfulness metrics disagree (#79). Masking can create OOD artifacts unless ROAR-style retraining is used (#82, #81). Mitigation: multi-metric reporting + ROAR on a subset.

**Risk 6 — Prompt-only structured output is brittle in practice.**
Open models often fail strict formatting (#34). Downstream failures become engineering noise. Mitigation: constrained decoding / schema enforcement (#32, #36) or post-processing (#35).

**Risk 7 — Synthetic artifact-aligned noise may overstate gains if unrealistic.**
PLMLabelErrors uses human-originated noise and shows different behavior than synthetic flips (#05). Label-biased settings can break standard detectors (#04). Mitigation: add at least one “more realistic noise” condition (human re-label subset, or label-bias noise) in addition to the marker injection.

---

## 4. New opportunities to strengthen novelty/impact (directly suggested by the literature)

**Opportunity A — Turn ECG from a ranker into a correction model via graph propagation.**
Leverage label propagation / GCN denoising (#53, #54) atop the explanation graph. This is a clean “graph-algorithmic” extension: ECG becomes “explanations → graph → denoised labels,” not just “rank and drop.”

**Opportunity B — Reliability-weighted ECG (WANN-style) as a core contribution.**
Adopt reliability-aware neighbor weighting (#49) where explanation faithfulness/stability modulates edge influence. This distinguishes ECG from “kNN on explanations” and is well-justified by graph noisy-label literature.

**Opportunity C — Multi-view ECG (explanations + model representations).**
Motivated by dual-representation neighbors (#51), build a multiplex graph and show robustness when either explanation embeddings or input embeddings are weak. This also answers the obvious critique “why not just do kNN on inputs?” by showing conditions where explanations add complementary structure.

**Opportunity D — Close the loop: explanation-guided training after cleaning.**
Use explanation priors / explanation regularization to reduce spurious reliance (#28, #31, #29), evaluated with ER-Test style robustness protocols (#30, #77). This creates a stronger story: ECG finds problematic data *and* yields models that behave better under spurious shifts.

**Opportunity E — Human/LLM collaboration pipeline as an application.**
Human+LLM annotation workflows already use explanations and consistency signals (#16). ECG can be positioned as the “verification/ranking” component that reduces human effort—grounded in LLM annotation reliability caveats (#15) and multi-annotator robustness ideas (#14).

**Opportunity F — Counterfactual augmentation for flagged artifacts.**
For high artifact-score clusters, generate counterfactual rewrites (human CAD #70 or automated counterfactuals #69) to break correlations. This is a concrete “repair action,” beyond deletion/relabeling.

---

## 5. Concrete rewrites for proposal sections

Below are “drop-in” replacements or additions, written to match the proposal’s structure.

### Rewrite: Section 2 “Explanation generation”

**Replace the current Section 2 with:**

> **2. Explanation generation (structured, stable, verifiable)**
>
> **Why:** Natural-language rationales can be plausible but not faithful (#41, #43) and can be unstable across prompts/samples (#40). ECG therefore (i) enforces strict structure, (ii) measures explanation stability/uncertainty, and (iii) uses verifiable explanation types.
>
> **2.1 Output schema (JSON Schema enforced)**
>
> Use grammar/schema-constrained decoding to guarantee JSON validity (#32, #36), with subword-aligned constraints when needed (#33). Prompt-only formatting is brittle for open models (#34); if constrained decoding is unavailable, use a lightweight schema-fixing post-processor (#35).
>
> Required fields:
>
> * `pred_label` (model’s label for the input)
> * `evidence` (1–3 **extractive** spans copied exactly from input)
> * `rationale` (≤ 25 tokens, must not include label words like “positive/negative” to reduce label leakage risk (#88))
> * `counterfactual` (one minimal edit or removal that would flip the label; counterfactual explanations are more directly verifiable (#42))
> * `confidence` (0–100)
>
> **2.2 Stability / uncertainty**
>
> For each instance, sample *m* explanations (m=3 by default) via prompt paraphrases or nonzero-temperature decoding. Compute:
>
> * evidence overlap (Jaccard over span strings)
> * rationale embedding variance
> * label agreement rate
>   Use low agreement / high variance as an **explanation-uncertainty signal** (#40, #46), and downweight such explanations in the graph stage.
>
> **2.3 Faithfulness spot-check signal**
>
> Use perturbation tests to verify explanations (#44): remove evidence spans from the input and measure the classifier confidence drop. If removing evidence does not change classifier confidence, downweight the explanation for graph construction and scoring.

### Rewrite: Section 3 “Graph construction”

**Replace Section 3 with:**

> **3. Graph construction (reliability-weighted, multi-view optional)**
>
> Build a kNN graph over explanation embeddings as in representation-space noisy label detection (#48), but weight neighbors by both similarity and reliability to reduce propagation of mislabeled clusters (#49).
>
> **3.1 Node embedding**
>
> * Embed a canonical string derived from `{evidence, rationale, counterfactual}` (exclude observed label; forbid label words in rationale to avoid trivial label clustering (#88)).
>
> **3.2 kNN retrieval + mutual edges**
>
> * Retrieve kNN in embedding space; optionally restrict to mutual-kNN edges to improve neighborhood quality.
>
> **3.3 Reliability-weighted edges (WANN-style)**
>
> * Define node reliability `r_j` from (i) low contradiction, (ii) high stability, (iii) high perturbation faithfulness.
> * Set edge weight `w_ij ∝ exp(sim(i,j)/τ) * r_j`, then normalize over neighbors (#49).
>
> **3.4 Outlier score**
>
> * Compute an outlier/OOD score from neighborhood similarity statistics, motivated by joint label-error/outlier graph methods (#52). Treat high outlier points separately from label inconsistency.
>
> **3.5 Multi-view extension (ablation)**
>
> * Optionally build a second kNN graph using classifier representations or predicted-probability vectors and combine inconsistency scores across views (#51).

### Rewrite: Section 4 “Inconsistency signals”

**Amend Section 4 by adding two signals and refining NLI:**

> **4.4 Explanation stability / uncertainty**
>
> * Add `S_stab(i)` from variance across multiple generated explanations (#40, #46). High variance indicates ambiguous/unreliable explanations; use it to downweight graph edges and/or increase suspicion for “uncertain” items.
>
> **4.5 Leakage / low-information rationale penalty**
>
> * Add `S_leak(i)` based on label–rationale association concerns (#88) and information-based rationale scoring (REV #89 or robust evaluation RORA #90). Penalize rationales that appear to encode the label without grounding in evidence.
>
> **Refine 4.2 NLI contradiction**
>
> * Use contradiction **margin** (e.g., `P_C − P_E`) rather than raw `P_C`, and calibrate thresholds via ablations, acknowledging general judge-reliability concerns (#37).

### Rewrite: Section 5 “Combining signals”

**Replace Section 5 with:**

> **5. Combining signals (reliability-adaptive rank aggregation)**
>
> Because explanation metrics can be unstable (#79) and single verifiers can be unreliable (#37), ECG combines signals via rank normalization plus reliability-adaptive aggregation:
>
> * Convert each signal to percentile rank.
> * Compute a per-signal confidence (e.g., NLI margin, neighborhood density, explanation stability).
> * Aggregate with a weighted median/trimmed mean, where low-confidence signals get lower weight.
>   Report fixed-weight sum as a baseline ablation.

### Rewrite: Section 7 “Baselines”

**Append the following baselines:**

> **Additional baselines**
>
> * Training dynamics: AUM (#06), CTRL (#07), forgetting-style signal (#08).
> * Graph/noisy-label: Neural Relation Graph (#52), WANN (#49).
> * Robust training: DivideMix (#99), ELR (#100), CoDC (#101) (compare cleaning vs robust training).
> * Influence: TracIn (#93), TRAK (#94), DataInf (#59).

### Rewrite: Section 8 “Evaluation metrics”

**Amend Section 8.4 to:**

> **8.4 Explanation quality / faithfulness (robust evaluation)**
>
> * Report ERASER-style comprehensiveness/sufficiency (#76), but validate masking-based conclusions using ROAR or Recursive ROAR on a subset (#82, #81) to avoid OOD masking artifacts.
> * Report leakage-robust rationale metrics (REV #89 and/or RORA #90) and a label-from-rationale-only predictability check (motivated by #88).
> * Include stability/sensitivity measures for explanations (#80) and avoid single-metric claims due to metric disagreement (#79).
> * Optionally add simulatability-style evaluation (human #91 or automated #92) on a small sample.

---

If you only implement a small subset of changes, the highest expected ROI is:

1. schema-constrained decoding for explanations (#32/#36, with #34 as justification),
2. reliability-weighted edges (#49) + outlier track (#52), and
3. leakage/stability-aware evaluation (add #88/#89/#90 and ROAR subset #82/#81).


