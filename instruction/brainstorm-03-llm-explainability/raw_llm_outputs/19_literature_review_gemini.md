# Literature-Informed Proposal Review — Gemini

**Prompt:** `prompts/14_literature_proposal_review.md`  
**Attachments:** `literature.md`, `proposals/02_ecg.md`  
**Date:** [PENDING]

Documents reviewed: proposal **02_ecg.md**  and curated bibliography **literature.md** 

## 1. Literature Coverage Assessment

### What is already well-supported by the literature

**A. “Data cleaning beats probability/loss-only methods in artifact-aligned regimes” (core motivation)**
The proposal’s central motivation—that probability/loss-based detectors can fail when models confidently fit spurious cues—is strongly consistent with:

* Confident learning as the canonical baseline (#00) and its documented dataset/setting-dependent failure modes (#01).
* Spurious correlation literature showing “shortcut” features can dominate learning and produce confident but wrong behavior under distribution shifts (#60, #62, #67, #75).
* Explanation-driven artifact discovery and debugging workflows that explicitly aim to surface spurious drivers that are not visible from loss alone (#18, #19, #24, #25, #26).

**B. “Graph neighborhood disagreement as a noise signal” (ECG’s graph core)**
The proposal’s neighborhood label inconsistency signal is well grounded in graph/neighbor-based noisy label detection:

* kNN in representation space for noisy labels (#48) is essentially the same structural idea as ECG, differing mainly in *what representation* defines neighbors.
* Graph-based joint detection approaches that include NLP experiments such as SST-2 (#52) provide precedent that graph relational structure can surface label issues in text.
* Label propagation / GCN-style smoothing on kNN graphs for denoising is established (#53, #54, #55).

**C. “Using explanations for debugging and data fixes” (conceptual lineage)**
The proposal’s “explanations → detect problematic points → clean/retrain” loop aligns with:

* Survey framing of explanation-based debugging pipelines (#18).
* Systems and methods where explanations guide label cleaning or data correction (#19, #21, #23).
* Methods specifically designed to reveal training-set artifacts via explanation-like attributions (#24, #25).

**D. “Faithfulness evaluation via comprehensiveness/sufficiency” (evaluation backbone)**
Using comprehensiveness and sufficiency is consistent with ERASER’s rationale evaluation conventions (#76), and the proposal’s intent to tie explanation utility to behavioral changes is aligned with broader evaluation thinking (#77).

---

### Gaps / assumptions that are **not** well-supported (or are contradicted/cautioned by the literature)

**1) Assumption: LLM-generated rationales are stable, faithful, and suitable as a geometric object (embeddings → kNN graph)**
The literature repeatedly warns that *plausible* explanations can be *unfaithful*, and that explanation type and elicitation method strongly affect faithfulness:

* Plausibility ≠ faithfulness (#41, #43).
* Faithfulness varies by explanation type and model family (#40, #42, #44).
* Explanations can “leak” labels (or correlate with labels in exploitable ways) even when they look reasonable (#88).

**Implication for ECG:** building the graph over `rationale + evidence` implicitly assumes the explanation text is a reliable semantic signature. The proposal mitigates some leakage by excluding observed labels from `t_i`, but it does not address *explanation-style clustering*, label leakage through templated phrasing, or instability across prompts/sampling (#40, #47, #88).

**2) Assumption: NLI contradiction is a robust verifier of “explanation supports label”**
The proposal uses an MNLI-style model as an explanation–label contradiction detector. The bibliography has strong reasons to be cautious:

* “Judge” reliability issues (biases, inconsistency, prompt sensitivity) are a known problem in LLM-as-judge style evaluation (#37). Even though you use an NLI model rather than an LLM judge, the underlying concern is similar: a single verifier can be brittle and systematically biased.
* Perturbation-based verification is emphasized as a more direct route to testing explanation claims (#44), while metric disagreement and instability are common (#79, #80).

**Implication:** NLI can be a useful *feature*, but treating it as a high-confidence “contradiction detector” without calibration/ensembling is under-justified.

**3) Synthetic artifact design may be criticized as “too easy / too artificial”**
The artifact-aligned marker trick is good for an MVP, but the spurious correlation literature warns that artifact diagnostics can fail in less stylized settings (#62), and many artifacts are not explicit tokens but distributional patterns (lexical, syntactic, annotation artifacts) (#60, #67). The proposal mentions CivilComments as optional; that aligns better with real artifact structure and group robustness concerns (#75), but it is not integrated into the minimal narrative.

**4) Explanation faithfulness evaluation design risks known pitfalls**
The proposal uses token removal/keeping to measure comp/suff. Literature highlights:

* Masking/removal produces OOD inputs; ROAR-style retraining is the classical fix (#82) and Recursive ROAR improves practicality (#81).
* Faithfulness metrics can disagree and be unstable (#79), and sensitivity/stability should be checked (#80).
* Optimizing or selecting based on explanation metrics can lead to Goodharting effects (#78). Even if ECG is not *training* to maximize these metrics, you are *evaluating* improvement and could inadvertently select design choices that “game” them.

---

### Highly relevant papers in the collection that the proposal does **not** leverage (but should)

Below are the most “directly actionable” omissions, grouped by proposal component:

**Explanation generation / structured output**

* Reliable structured generation: grammar/schema constrained decoding and benchmarks (#32, #33, #36, #34) and post-processing to enforce schema (#35).
* Explanation stability: explanation-consistency finetuning (#47) and uncertainty/self-reflection (#46).

**Graph methodology**

* Reliability-weighted kNN for noisy labels (#49).
* Strong graph-based label error baseline that *already evaluates on SST-2*: Neural Relation Graph (#52).
* Label propagation / GCN denoising alternatives (#53, #54, #55).
* Dual-view neighbor methods combining embeddings and label-prob distributions (#51).

**Noise detection baselines beyond Cleanlab/loss**

* Training dynamics: AUM (#06), CTRL loss-curve clustering (#07), forgetting-based signals (#08).
* Bias/label-biased settings: DeCoLe (#04).
* Robust training competitors: DivideMix (#99), ELR (#100), co-teaching variants (#101).

**Artifact/spurious feature analysis**

* Training-feature attribution designed to expose artifacts (#24) and influence-based artifact dependence (#25).
* Counterfactual data augmentation and its explanatory account (#70, #71) and causal feature methods (#68, #69).

**Evaluation beyond comp/suff**

* ROAR / Recursive ROAR (#82, #81).
* Leakage-aware rationale evaluation (label–rationale association) (#88), information-theoretic rationale value (REV) (#89), robust rationale evaluation (RORA) (#90).
* Simulatability (human-grounded or automated) (#91, #92).

---

## 2. Methodology Refinements (literature-driven)

### A) Explanation generation approach

**Refinement 1: Replace “prompt + JSON repair” with schema-guaranteed decoding**

* FOFO shows open models can be weak at strict format-following (#34). Your current “retry on invalid JSON” is workable but introduces silent failure modes and unpredictable bias (some examples require more retries → may correlate with difficulty/noise).
* Use grammar/schema constrained decoding (#32, #36) or subword-aligned constrained decoding (#33). If constrained decoding is impractical in your serving stack, add a lightweight structure-corrector (#35).

**Concrete change:** treat structured compliance as a *hard constraint*, not a best-effort.

**Refinement 2: Add an explanation *stability* signal (and use it downstream)**

* Explanation faithfulness and reliability vary; stability across paraphrases/sampling is informative (#40, #46, #47).
* Minimal add: sample 2–3 rationales with slight prompt perturbations or temperature>0 and compute agreement on:

  * predicted label,
  * evidence overlap (Jaccard),
  * embedding dispersion.
    This becomes a *node reliability* score, useful for graph weighting (see #49).

**Refinement 3: Add a counterfactual-style field to improve verifiability**

* Counterfactual self-explanations can be more verifiable than free-form rationales (#42).
* Add a field like: `"counterfactual": "If <minimal change>, the label would be <other label>."`
  This supports additional consistency checks and more robust graph similarity (counterfactuals may align better than stylistic rationales).

**Refinement 4: Explicitly mitigate label leakage / templated label phrases**

* Label leakage via rationales is a known phenomenon (#88). For ECG, leakage is not always “bad” but it can:

  * trivialize the graph (cluster by template),
  * reduce sensitivity to semantic inconsistency (neighbors become “same-template” instead of “same-meaning”).
* Add a constraint: rationale must not contain label words (“positive/negative”, class names). Keep evidence spans extractive.

---

### B) Graph construction methodology

**Refinement 1: Reliability-weighted neighborhood aggregation (WANN-style)**

* WANN introduces reliability scores to downweight likely mislabeled neighbors in kNN voting (#49). ECG can directly adopt this:

  * define node reliability (r_j) from explanation stability + verifier agreement (NLI/label match) and/or training dynamics.
  * compute neighborhood posterior using (r_j):
    [
    p_i(c)=\sum_{j\in N(i)} w_{ij}, r_j ,\mathbf{1}[y_j=c]
    ]
    This reduces “error reinforcement” when noisy points cluster.

**Refinement 2: Add a stronger graph baseline and/or alternative: Neural Relation Graph**

* Neural Relation Graph is explicitly a graph-based approach for label error + OOD/outlier detection and includes SST-2 evaluation (#52).
  If ECG doesn’t compare to #52, reviewers can plausibly argue the novelty is “graph over text-like embeddings” rather than a new capability.

**Refinement 3: Consider label propagation / GCN as an ECG variant (not only as evaluation)**

* GCN label denoising on kNN graphs (#53) and label propagation in NLP (#54) suggest a natural extension:

  * Use the explanation graph to infer *soft labels* (\hat{y}_i) via propagation,
  * Score label issues by disagreement between (y_i) and propagated posterior.
    This is a principled alternative to “neighborhood surprise” and can be more robust when local neighborhoods are imperfect.

**Refinement 4: Dual-view graphs**

* Dual kNN methods incorporate both embedding similarity and label-probability representations (#51).
  In ECG terms: build two graphs:

  1. explanation-embedding graph,
  2. classifier-representation (or predicted-probability) graph,
     and combine their inconsistency signals. This helps when explanation embeddings are noisy but classifier representations are stable (or vice versa).

---

### C) Signal combination and scoring

**Refinement 1: Add training-dynamics signals as a fourth channel**

* AUM (#06), CTRL (#07), and forgetting-style signals (#08) capture “hard vs mislabeled” structure that pure loss and one-shot probabilities miss. This is directly relevant because ECG’s top-K removal risks deleting rare but clean hard examples.
* AUM is particularly actionable because it has a calibration strategy (#06).

**Refinement 2: Calibrate thresholds and weights using a small validated set (human/LLM review)**

* Human+LLM collaborative annotation pipelines explicitly rely on multi-signal verification and targeted human effort (#16).
* If you can afford even 200–500 human validations (or high-quality adjudication), you can:

  * calibrate contradiction thresholds,
  * fit score weights robustly,
  * estimate precision of top-K in realistic settings.

**Refinement 3: Use ensemble verifiers rather than a single NLI model**

* Judge reliability issues (#37) and correlated LLM errors (#14, #15) motivate ensembling:

  * two NLI models, or
  * NLI + perturbation-based check (#44).
    Then define contradiction as “consensus contradiction,” improving robustness.

---

### D) Baseline selection (what to add / replace)

Minimum recommended additions (low cost, high reviewer value):

1. **AUM** training-dynamics baseline (#06).
2. **CTRL** loss-curve clustering baseline (#07) or forgetting-based (#08).
3. **Neural Relation Graph** graph baseline (#52).
4. **WANN** reliability-weighted kNN baseline (#49).
5. **Classifier-representation kNN disagreement** baseline aligned with DeepKNN noisy labels (#48) (distinct from your “input embedding graph” baseline).

Optional but strong (if compute allows):

* **TRAK** influence baseline (#94) as a scalable alternative to TracIn (#93).
* **ELR** or **DivideMix** as “robust training without cleaning” comparators (#100, #99).
* **Self-debiasing** as a baseline for unknown shortcut mitigation (#65) (especially if you push spurious artifact claims).

---

### E) Evaluation metrics (additions and fixes)

**Refinement 1: Add ROAR/Recursive ROAR style checks for comp/suff**

* Comp/suff without retraining suffers from OOD masking artifacts (#82). Recursive ROAR improves practicality (#81).
* Minimal: run ROAR-style evaluation only for a small subset (e.g., 1–2k dev examples) to validate directionality.

**Refinement 2: Add leakage-aware and robust rationale evaluation**

* Measure label leakage and rationale–label association (#88) to ensure improvements aren’t driven by trivial rationales.
* Add at least one robust free-text rationale metric (REV #89 or RORA #90).

**Refinement 3: Add stability/sensitivity reporting**

* Explanation stability metrics and sensitivity to perturbations (#80) are particularly relevant since your graph is built on explanation text.

**Refinement 4: Strengthen spurious/OOD evaluation protocol**

* Counterfactual augmentation and challenge-set logic (#70, #71, #27) suggests adding a “semantic-preserving but shortcut-breaking” slice beyond token stripping/swapping.

---

## 3. Risk Analysis (failure modes suggested by the literature)

### Risk 1: Explanation text is not a stable, faithful representation → graph neighborhoods become noisy or misleading

* Faithfulness and reliability issues for LLM self-explanations (#40–#44) can break the assumption that explanation similarity ≈ semantic similarity.
* Explanation variance (prompt sensitivity) can create artificial clusters unrelated to content (#37, #47).

**Mitigation:** constrained decoding (#32–#36), stability scoring (#40, #46, #47), and dual-view graphs (#51).

---

### Risk 2: Label leakage / templated rationales trivialize the graph and/or corrupt evaluation

* Label–rationale association and leakage are documented (#88).
  If rationales encode label (“This is POSITIVE”), kNN becomes a label graph rather than a meaning graph. This can inflate neighborhood-consistency signals and obscure real artifact structure.

**Mitigation:** ban label words in rationales; rely on extractive evidence spans; compute leakage metrics (#88) and REV/RORA (#89, #90).

---

### Risk 3: NLI contradiction is brittle and may introduce systematic bias

* Judge reliability issues (#37) and metric disagreement (#79) imply a single verifier can create consistent false positives/negatives.
* For sentiment, MNLI models are not guaranteed to behave monotonically with respect to evidence quality.

**Mitigation:** ensemble verifiers (#37, #14) and/or perturbation-based checks (#44).

---

### Risk 4: Top-K removal deletes hard-but-correct minority examples → worse robustness

* Training dynamics work highlights that “difficult” points can look like noise (#06–#08).
  Spurious correlation literature emphasizes minority groups/counterexamples are crucial for robustness (#67, #75). Those examples may be flagged by inconsistency signals and then removed—hurting worst-group accuracy.

**Mitigation:** prefer **reweighting** or guarded relabeling; add group-robust evaluation (CivilComments + worst-group accuracy) (#75).

---

### Risk 5: Synthetic artifact regime does not transfer to realistic artifacts

* Partial-input diagnostics can miss artifacts (#62), and many artifacts are subtle annotation artifacts (#60) rather than explicit tokens.

**Mitigation:** include at least one “real artifact” dataset/slice (CivilComments identity terms; or NLI artifact settings inspired by #60) and compare to graph baselines (#52).

---

### Risk 6: Faithfulness metrics can be gamed or misleading

* Goodhart effects for explanation metrics are real (#78).
* Removal-based metrics can disagree and be unstable (#79) and suffer OOD masking (#82).

**Mitigation:** add ROAR/Recursive ROAR checks (#82, #81) and report multiple metrics (#79).

---

## 4. New Opportunities (to increase novelty/impact)

### Opportunity 1: Turn ECG into a human-in-the-loop **label repair** system (not just filtering)

* Collaborative LLM+human verification pipelines are directly aligned (#16).
* Interactive label cleaning via explanations is established (#23).

**Proposal upgrade:** present ECG as *prioritization + structured evidence for reviewers*, with measured reviewer efficiency gains, not only as an unsupervised ranker.

---

### Opportunity 2: Add an ECG-powered **explanation-regularized training** variant

Literature shows explanation constraints can improve robustness if evaluated properly (#21, #28, #29, #31) and ER-Test emphasizes how to validate those claims (#30, #77).

**ECG twist:** use ECG to select (or weight) examples for explanation regularization (apply constraints only where explanations are stable/consistent), reducing the “Goodharting” and brittleness risks.

---

### Opportunity 3: Integrate training-feature attribution / influence views for artifact discovery

* Training-Feature Attribution is purpose-built for surfacing training-set artifacts (#24).
* Influence-based artifact dependence is a known phenomenon (#25).
* Scalable attribution methods exist (#94, #59).

**ECG twist:** use influence-derived neighborhoods (who influences whom) as an alternative or complement to embedding similarity. This strengthens causal interpretability beyond “semantic similarity.”

---

### Opportunity 4: Extend beyond binary classification to structured settings

* Token-level label error benchmarks (#02) and weakly labeled sequence tasks (#11) are natural next domains for “explanation consistency graphs,” especially since structured explanations can align with spans/tokens.
* Multi-label noise frameworks (#03, #12) suggest ECG’s multi-signal/graph aggregation is potentially even more valuable when “noise” is not a single label flip.

---

## 5. Updated Proposal Sections (concrete rewrites)

Below are drop-in rewrites for the proposal where literature suggests meaningful changes.

---

### Rewrite: Section 2 — Explanation generation (add structure guarantees + stability)

```md
## 2. Explanation generation (revised: schema-guaranteed + stability-aware)

### Why revisions are needed
Open-weight LLMs can be unreliable at strict format-following (#34). Because ECG depends on structured fields
(evidence spans, rationales) as *inputs to graph construction*, we treat schema compliance as a hard constraint
using constrained decoding / schema enforcement (#32, #33, #36) or a lightweight structure-corrector (#35).

LLM explanations also vary in faithfulness and stability (#40–#44). We therefore compute an *explanation stability*
signal and use it downstream as a node reliability weight (cf. reliability-weighted kNN, #49).

### Output format (JSON Schema, enforced)
Fields:
- pred_label: "POSITIVE" | "NEGATIVE"
- evidence: 1..3 exact substrings from the input
- rationale: <= 25 tokens, MUST NOT contain label words ("positive", "negative", class names)
- confidence: 0..100
- counterfactual: one sentence describing a minimal edit that would flip the label (verifiable; #42)

Generation:
- Use constrained decoding to guarantee schema validity (#32/#36), or DOMINO-style subword alignment if available (#33).
- If constrained decoding is unavailable, run a structure-corrector model (#35) and reject remaining invalid outputs.

### Stability sampling (cheap, high value)
For each example, generate n=3 explanations with minor prompt perturbations OR temperature=0.7.
Compute:
- label agreement rate
- evidence overlap (Jaccard)
- embedding dispersion of (rationale+evidence)

Define node reliability r_i = normalized stability score (higher = more reliable). (#40, #46, #47)
We use r_i to weight neighbors and to downweight unstable explanations in scoring.
```

Key literature links for this rewrite: #32–#36, #34, #40–#44, #42, #46, #47, #49, #88.

---

### Rewrite: Section 3 — Graph construction (add reliability weighting + graph baseline parity)

```md
## 3. Graph construction (revised: reliability-weighted + baseline parity)

We build a kNN graph over explanation representations, but we explicitly account for unreliable nodes by
using a reliability weight r_i (from explanation stability and verifier agreement), inspired by reliability-weighted
kNN for noisy labels (#49).

### Node representation
Use an evidence-focused canonical string to reduce template/style clustering and label leakage (#88):
t_i = "Evidence: " + join(evidence_i, "; ") + " | Rationale: " + rationale_i
(Exclude observed label y_i and prohibit label words in rationale.)

### kNN retrieval
Same FAISS cosine kNN as before, plus optional mutual-kNN edges for robustness.

### Reliability-weighted neighborhood posterior
Let r_j ∈ [0,1] be node reliability.
Compute:
p_i(c) = Σ_{j∈N(i)} w_ij * r_j * 1[y_j=c]
Neighborhood Surprise: S_nbr(i) = -log( (p_i(y_i)+ε)/(1+Cε) )

This reduces error reinforcement when noisy points cluster. (#49)

### Strong graph baseline
In addition to "input-embedding kNN", include:
- kNN disagreement on classifier representations (DeepKNN-style; #48)
- Neural Relation Graph baseline (#52) for label error + OOD detection on text
```

Key literature links: #49, #88, #48, #52, #53–#55.

---

### Rewrite: Section 5 — Combining signals (add training dynamics + ensemble verification)

```md
## 5. Combining signals (revised: add training dynamics + verifier ensembling)

In addition to (neighborhood surprise, contradiction, artifact focus), we add a training-dynamics signal to
avoid over-flagging hard-but-correct examples (#06–#08).

Signals:
- S_nbr: reliability-weighted neighborhood surprise (#49)
- S_ver: verifier contradiction score (ensemble)
- S_art: artifact focus score
- S_dyn: training dynamics score (AUM preferred; #06; or CTRL loss-curve clustering #07)

Verifier ensembling:
Because single judges can be brittle (#37), define S_ver as the average contradiction probability across:
- NLI model A + NLI model B, and/or
- one perturbation-based check (remove claimed evidence; #44)

Aggregation:
Use rank-normalization and a weighted sum, but report ablations and optionally learn weights on synthetic runs.
We expect S_dyn to be most helpful on uniform noise where loss-based methods are strong.
```

Key literature links: #06–#08, #37, #44, #49.

---

### Rewrite: Section 7 — Baselines (expand to the most relevant missing ones)

```md
## 7. Baselines (revised: add essential training-dynamics + graph baselines)

Must-include baselines:
B1) Confident Learning / Cleanlab (#00)
B2) Out-of-sample PLM loss ranking (#05)
B3) AUM training dynamics (#06)
B4) CTRL loss-curve clustering (#07) or forgetting-based (#08)
B5) Graph baselines:
    - DeepKNN disagreement on classifier representations (#48)
    - WANN reliability-weighted kNN (#49)
    - Neural Relation Graph (#52)
B6) Influence (optional):
    - TRAK (#94) as scalable influence baseline; TracIn (#93) if easy
B7) Random
```

---

### Rewrite: Section 8.4 — Faithfulness & explanation evaluation (make it literature-consistent)

```md
## 8.4 Explanation evaluation (revised: leakage-aware + ROAR check)

We report ERASER-style comprehensiveness and sufficiency (#76), but we avoid over-interpreting raw masking
effects because masked inputs are out-of-distribution (#82). We therefore add a lightweight ROAR/Recursive ROAR
check on a subset (#81/#82).

Metrics:
1) Comprehensiveness / Sufficiency (ERASER, #76)
2) Recursive ROAR subset evaluation (#81) to validate directionality
3) Label leakage / rationale-label association (#88)
4) One robust free-text rationale metric:
   - REV (#89) or RORA (#90)

We also report explanation stability / sensitivity statistics (#80), since ECG depends on explanations as
representations for graph construction.
```

Key literature links: #76, #82, #81, #88, #89, #90, #80, #79.

---

## Summary of the most important changes to make

1. **Make structured explanation output reliable by construction** (#32–#36, #34).
2. **Treat explanations as noisy signals; add stability and reliability weighting** (#40, #46, #47, #49).
3. **Add missing “must-have” baselines**: AUM/CTRL (#06/#07), Neural Relation Graph (#52), WANN (#49), classifier-rep kNN (#48).
4. **Harden evaluation**: leakage-aware metrics (#88), ROAR/Recursive ROAR checks (#82/#81), robust rationale metrics (#89/#90), and stability reporting (#80).
5. **Address the key risk explicitly**: removing hard-but-clean minority examples; consider reweighting/guardrails and group robustness evaluation (#75, #06–#08).
