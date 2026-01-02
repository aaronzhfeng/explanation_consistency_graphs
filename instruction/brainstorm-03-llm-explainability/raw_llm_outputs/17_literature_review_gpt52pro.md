# Literature-Informed Proposal Review — GPT-5.2 Pro

**Prompt:** `prompts/14_literature_proposal_review.md`  
**Attachments:** `literature.md`, `proposals/02_ecg.md`  
**Date:** [PENDING]

## 1. Literature Coverage Assessment

Sources: curated ECG literature bank (103 papers)  and the current ECG proposal .

### What the proposal is already well-supported by

**A. Core motivation: “artifact-aligned” label errors where probability/loss signals fail**

* The proposal’s “Cleanlab failure mode” premise is consistent with:

  * Confident-learning baselines and documented failure modes when models are confidently wrong (#00, #01).
  * Spurious-correlation / artifact literature showing models can exploit shortcuts that evade standard diagnostics (#60, #63, #65).
* The proposal’s emphasis that “you need non-loss signals” is aligned with explanation-driven artifact discovery and debugging pipelines (#18, #19, #24, #25, #26).

**B. Graph-over-representations for noisy-label detection**

* The kNN graph + neighborhood label disagreement idea is directly grounded in “deep kNN/noisy label” approaches (#48) and more explicit relational graph methods that jointly handle label issues and OOD/outliers (#52).
* The broader graph-based denoising / propagation framing is also well established (#53, #54, #55).
  So the proposal’s “neighborhood surprise” signal is strongly literature-consistent—just missing explicit anchoring to these methods.

**C. Using LLM outputs (explanations) as structured signals**

* The proposal’s structured JSON rationales and evidence spans are well-aligned with the structured-output enforcement and evaluation literature (#32, #33, #36), and with work stressing that explanation quality/faithfulness cannot be assumed (#40, #41, #43, #44).
* The idea that “explanation consistency matters for downstream use” is directly supported (#47).

**D. Evaluation direction (robustness + explanation faithfulness)**

* Measuring downstream utility and explanation quality is aligned with ERASER-style rationalized NLP evaluation (#76), and with broader warning flags that explanation metrics are tricky / can disagree (#79) or be Goodharted (#78).
* The proposal’s use of comprehensiveness/sufficiency is consistent with standard practice, but the literature argues these need careful implementation (see gaps below: #82, #81).

---

### Gaps: assumptions the proposal currently makes without being justified (and literature suggests are risky)

**Gap 1 — “Prompt-only JSON” is treated as sufficient**

* The proposal’s “stop at `}` / retry if invalid JSON” assumes format-following is reliable for open 7–8B instruction models. The literature bank explicitly warns this is often not true (#34) and provides stronger solutions: grammar/schema constrained decoding (#32, #36) and constraint-aligned decoding variants (#33), with post-processing alternatives (#35).
  **Implication:** without a robust schema guarantee, the pipeline may fail or silently degrade (missing fields, invalid spans).

**Gap 2 — Explanations are assumed stable + semantically meaningful enough for graph construction**

* The proposal uses deterministic decoding (temp=0) but does not address *explanation variance across prompts/conditions* or across “equivalent inputs.” Literature shows explanation stability varies by explanation type/model and is a real problem (#40), and there are methods specifically targeting consistency (#47).
  **Implication:** noisy/unstable rationales can corrupt the embedding space and produce misleading neighbors.

**Gap 3 — Explanation faithfulness vs plausibility isn’t treated as a first-class threat**

* The proposal treats LLM rationales as “semantic anchors” (especially in the artifact-aligned noise trick). But the literature repeatedly emphasizes plausible explanations can be unfaithful (#41, #43), and perturbation-based verification is important (#44).
  **Implication:** ECG can flag “inconsistencies” that are artifacts of the explainer, not the dataset.

**Gap 4 — NLI contradiction as a reliable contradiction detector is assumed, not validated**

* The proposal uses MNLI-style NLI to check “explanation entails label.” This is reasonable, but it’s not grounded in the literature bank as a validated approach for explanation verification; meanwhile the bank includes extensive evidence about judge-style unreliability and bias (especially if you shift to LLM-as-judge later) (#37).
  **Implication:** contradiction probabilities may be brittle to templates, lexical cues, and label leakage.

**Gap 5 — Faithfulness evaluation uses masking but does not address ROAR-style pitfalls**

* The proposal’s comprehensiveness/sufficiency implementation masks tokens directly and compares probabilities. The literature warns that masked inputs can be out-of-distribution and produce misleading scores unless you retrain or use ROAR-like protocols (#82) or improved variants (#81).
  **Implication:** explanation-faithfulness conclusions could be invalid or unstable (#79).

**Gap 6 — “Hard-but-correct” examples are not distinguished from mislabeled examples**

* Pure neighborhood inconsistency (and even NLI mismatch) will also flag ambiguous/hard examples. Training-dynamics work explicitly highlights such confounds (#08), and AUM-style signals are designed to separate mislabeled from hard-but-learnable points (#06).
  **Implication:** removal-based cleaning risks dropping valuable, minority, or nuanced examples.

**Gap 7 — Label leakage / trivial clustering risk is not discussed**

* If explanations frequently contain label-associated tokens (“positive,” “negative,” “great,” “terrible”), graphs built over explanation text can become label-clustered for superficial reasons. The literature includes explicit warnings about label–rationale association and leakage (#88), plus metrics designed to measure whether rationales add real information (#89) and robust evaluation resistant to leakage (#90).
  **Implication:** ECG could appear to “work” by learning artifacts in the rationales themselves.

---

### Relevant papers in the collection that the proposal doesn’t leverage (but should)

Grouped by where they plug into your pipeline:

**Structured explanation generation / reliability**

* Constrained/schema decoding + structured output evaluation: #32, #33, #36, #34
* Post-processing to repair nearly-structured outputs: #35
* Explanation consistency and uncertainty: #47, #46, #45
* LLM-as-judge reliability (if any LLM judging emerges): #37
* LLM explanation faithfulness caveats: #40, #41, #42, #43, #44

**Graph construction / graph-based noise methods**

* Reliability-weighted kNN and adaptive neighbor weighting: #49
* Joint label error + OOD/outlier detection in a relational graph: #52
* Graph label propagation / GCN denoising: #53, #54
* Embedding-similarity denoising framing: #55
* Dual-view neighbor models: #51
* “Beyond images” feature-space noise issues relevant to NLP embeddings: #56

**Baselines you should include**

* Training-dynamics label noise detectors: #06 (AUM), #07 (CTRL), #08 (forgetting)
* Influence/data attribution baselines: #94 (TRAK), #59 (DataInf) (and #93 TracIn is already mentioned)
* Robust noisy-label training: #99 (DivideMix), #100 (ELR), #101 (CoDC)
* Instance-dependent noise: #97 (important because “artifact-aligned noise” is a special case of feature-dependent noise)

**Evaluation upgrades**

* ROAR/Recursive ROAR for faithfulness evaluation: #82, #81
* Robust rationale evaluation + leakage resistance: #90 (RORA), #89 (REV), #88 (leakage tests)
* Metric gaming + metric disagreement: #78, #79
* Simulatability (human or automated): #91, #92

---

## 2. Methodology Refinements (literature-informed)

### A) Explanation generation approach

**1) Make JSON schema validity a hard guarantee (not best-effort).**
Replace “retry-on-parse-fail” with one of:

* Grammar / JSON-schema constrained decoding (#32, #36), with subword-aligned constraint handling if needed (#33).
* If constrained decoding isn’t feasible in your serving stack, use a small schema-correction model/post-processor (#35) rather than ad-hoc reprompting.
  **Why:** FOFO shows strict formatting is a non-trivial failure mode for many open models (#34).

**2) Add an explanation *stability* signal and use it as a reliability weight.**

* Generate multiple explanations per input (small M like 3–5), measure agreement on: predicted label, evidence overlap, and rationale semantic similarity.
* Treat high variance as low reliability, consistent with self-consistency faithfulness checks (#40) and uncertainty-aware rationale generation (#46).
* If you can fine-tune, explanation-consistency finetuning is directly relevant (#47).
  **Payoff:** you can downweight unstable nodes/edges in the graph, improving robustness.

**3) Prefer explanation types that are verifiable (or add a verifiable component).**

* Counterfactual self-explanations are argued to be more verifiable than free-form rationales in some settings (#42).
* Perturbation tests can validate whether cited evidence is causally relevant (#44).
  Concrete change: extend the JSON schema to include a minimal counterfactual statement (or a “decision would flip if…” field). This provides a richer substrate for contradiction checks than NLI over a single label statement.

**4) Reduce correlated LLM errors by using multiple “annotators.”**

* Treat LLM outputs as noisy annotation signals; explicitly model correlated errors and/or use multiple LLM agents (#14), supported by findings that single-model annotation can be systematically unreliable (#15).
  Concrete lightweight variant: run two different instruction-tuned models (or two prompt styles) and compute disagreement as another signal; or only trust explanations when cross-model agreement is high.

---

### B) Graph construction methodology

**1) Replace softmax-only edge weights with reliability-weighted neighbor voting.**

* WANN’s reliability-weighted adaptive kNN provides a direct template (#49).
  Concrete: define node reliability (r_i) from stability + explanation entailment strength + LLM confidence calibration (#46), then weight neighbor contributions by (w_{ij}\propto \exp(s_{ij}/\tau)\cdot r_j).
  **Benefit:** reduces the impact of bad neighbors and aligns with graph-noise detection practice.

**2) Add explicit outlier/OOD handling before using neighborhood signals.**

* NeuralRelationGraph explicitly supports joint label error + OOD/outlier detection in representation graphs (#52).
  Concrete: compute a “graph outlier score” (e.g., mean similarity to top-k, degree after thresholding) and either (i) exclude those nodes from neighbor-vote scores or (ii) treat them as a separate category (hard/OOD vs mislabeled).
  **Benefit:** avoids false positives where examples are just rare, not mislabeled.

**3) Multi-view graph: explanations + raw inputs (or model representations).**

* Dual-neighbor models motivate combining different representation spaces (#51).
  Concrete: build two graphs: one over explanation embeddings and one over input embeddings (proposal already has this as a baseline). Use:
* intersection edges (more conservative, higher precision), and/or
* agreement between neighborhood posteriors as a consistency feature.
  **Benefit:** reduces reliance on any single embedding space (important given “NLP feature noise” concerns #56).

**4) Consider label propagation / GCN as an alternative to “remove top-K.”**

* Graph-based denoising methods propagate labels from reliable regions (#53) and NLP precedent exists (#54).
  Concrete: treat high-reliability nodes as anchors and run label propagation to propose corrected labels with uncertainty. This could be a stronger “relabel” mechanism than your current guardrails.

---

### C) Signal combination and scoring

**1) Add training-dynamics signals to separate “hard” from “wrong.”**

* AUM is designed to identify mislabeled examples via margin dynamics (#06).
* CTRL clusters loss trajectories to detect noise (#07).
* Forgetting-style signals capture ambiguity/hardness (#08).
  Concrete: add (S_{\text{aum}}) or (S_{\text{ctrl}}) as either:
* an additional score term, or
* a *veto* against removal (e.g., don’t drop examples that look “hard but consistently learnable”).
  This directly addresses a known failure mode of pure disagreement signals.

**2) Replace fixed weights with robust aggregation or reliability-aware weighting.**

* Your fixed 0.4/0.4/0.2 weighting is arbitrary; the literature encourages careful calibration when using noisy judges/signals (#37) and when metrics are unstable (#79).
  Concrete options that don’t require gold labels:
* **Product-of-experts**-style aggregation (#63 analogy): treat each signal as independent evidence and multiply calibrated probabilities of “noisy.”
* **Pseudo-label weight learning:** define high-confidence noisy points when multiple signals agree (e.g., high neighbor surprise AND high contradiction), then learn weights to separate those from a presumed-clean set.

**3) Incorporate influence/data attribution as an additional orthogonal signal.**

* TracIn is a standard influence baseline (#93); TRAK scales attribution (#94); DataInf is practical for LoRA-tuned LMs/PLMs (#59).
  Concrete: compute “harmful influence on dev loss” and add as (S_{\text{infl}}). This can catch cases where neighborhood disagreement fails but a point is demonstrably harmful.

**4) Artifact scoring: consider causal/token-effect signals beyond PMI.**

* PMI mining is simple but can be fooled by correlated tokens. The spurious-feature literature proposes more causal approaches (treatment-effect style classification #68 and counterfactual interventions #69).
  Concrete: treat PMI as a first pass; for top suspect tokens, validate with targeted counterfactual substitution tests where feasible (#69).

---

### D) Baseline selection (expanded, literature-aligned)

In addition to your current baselines, add:

**Label-noise detection / cleaning**

* AUM (#06)
* CTRL (#07)
* (Optional) forgetting-based (#08)
* PLM out-of-sample loss ranking protocol (#05) (important: out-of-sample, not in-sample)

**Graph-based**

* NeuralRelationGraph (#52) as a direct graph-noise competitor
* WANN (#49) as a modern kNN+reliability baseline
* Label propagation / GCN denoising (#53, #54)

**Influence**

* TRAK (#94) and/or DataInf (#59) as scalable alternatives to TracIn (#93)

**Robust training under noise** (as “post-cleaning competitor methods”)

* DivideMix (#99), ELR (#100), CoDC (#101)

---

### E) Evaluation metrics (upgrades)

**1) Add at least one “realistic noise” condition.**
Your synthetic artifact injection is useful, but the bank highlights that human-originated or feature-dependent noise behaves differently (#05, #97).
Concrete:

* Use #05’s framing: evaluate on a noise regime approximating human labeling mistakes (e.g., time-pressure relabel subset, or LLM-as-weak-annotator with known bias patterns).
* Include at least one instance-dependent noise variant (#97).

**2) Explanation evaluation: use ROAR-family methods or acknowledge limitations.**

* For comprehensiveness/sufficiency, adopt ROAR or Recursive ROAR methodology to reduce OOD masking artifacts (#82, #81).
* Add robustness checks and multiple faithfulness metrics since rankings can disagree (#79) and sensitivity matters (#80).
* Add leakage-aware rationale evaluation: run label–rationale association tests (#88) and report REV/RORA-style scores (#89, #90).
* Add a Goodhart check: monitor whether explanation scores change without behavior changes (#78).

**3) Robustness evaluation beyond token strip/swap**
Your strip/swap tests are good minimal diagnostics. Consider adding:

* Counterfactually augmented evaluation logic (#70, #71) (either via existing CAD data or by generating minimal edits) to test whether reliance on shortcuts decreases.

---

## 3. Risk Analysis (literature-driven failure modes the proposal under-emphasizes)

**Risk 1 — LLM explanations are noisy artifacts, not stable “semantic truth.”**

* Plausible ≠ faithful (#41, #43); explanation type matters (#40, #42); explanations can be inconsistent across samples/prompts (#40).
  **Failure mode:** ECG flags points because the explainer is unstable, not because labels are wrong.
  **Mitigation:** stability scoring (#40, #46), explanation consistency tuning (#47), multi-annotator/LLM ensemble (#14), perturbation checks (#44).

**Risk 2 — Structured output brittleness and silent parsing failures.**

* Format-following can fail for open models (#34).
  **Failure mode:** partial JSON or malformed evidence spans → embeddings degrade, edges become noisy.
  **Mitigation:** constrained decoding + schema compliance (#32, #33, #36) or schema-correction (#35).

**Risk 3 — Label leakage in rationales makes the graph trivially label-clustered.**

* Label–rationale association and leakage are documented pitfalls (#88); “good-looking” rationale scores can be gamed (#78).
  **Failure mode:** ECG appears to work because rationales encode labels, not because they reflect semantics.
  **Mitigation:** leakage tests (#88), leakage-resistant evaluation (#90), informativeness metrics (#89), and stripping label words from explanation text before embedding.

**Risk 4 — Neighborhood disagreement conflates “mislabeled” with “hard/ambiguous.”**

* Forgetting/hardness signals complicate label-noise detection (#08).
  **Failure mode:** removing top-K hurts generalization by deleting rare-but-correct cases.
  **Mitigation:** add AUM/CTRL signals (#06, #07) and prefer reweighting over removal (also consistent with downweighting biased examples in debiasing frameworks like #64/#65).

**Risk 5 — Graph sensitivity to embedding quality and NLP feature noise.**

* Graph methods depend on representation quality (#54) and feature spaces in NLP can be “noisier” than vision-style embeddings (#56).
  **Failure mode:** wrong neighbors → high false positives/negatives; graph amplifies noise.
  **Mitigation:** outlier detection (#52), reliability-weighted kNN (#49), multi-view graphs (#51).

**Risk 6 — Evaluation pitfalls: masking-based faithfulness metrics can mislead.**

* ROAR shows why simple masking is unreliable without retraining (#82), and metrics can disagree (#79).
  **Mitigation:** use ROAR/Recursive ROAR (#82, #81) or clearly scope claims; report sensitivity (#80).

---

## 4. New Opportunities (to strengthen novelty/impact)

**Opportunity 1 — Turn ECG into an annotation workflow component, not just a detector.**

* The literature contains directly aligned human–LLM collaborative annotation pipelines with explanation signals (#16) and LLM-in-the-loop annotation frameworks (#13/#38).
  **Novel angle:** ECG as a *triage graph* that selects which items go to humans vs LLM relabel, with reliability weights and explanation consistency as gating.

**Opportunity 2 — “ECG depends on explanation stability” as a publishable insight.**

* Explanation-consistency finetuning exists (#47) but hasn’t been tied to data-cleaning graphs.
  **New contribution:** show that stabilizing explanations materially improves label-noise detection and robustness gains.

**Opportunity 3 — Integrate training-feature attribution into graph edges for artifact discovery.**

* TFA localizes which tokens in which influential training examples drive predictions (#24) and influence-based artifact reliance is documented (#25).
  **New contribution:** edges aren’t just “explanations are similar,” but “these examples share influential artifact features,” making graphs more diagnostic and human-auditable.

**Opportunity 4 — Combine ECG with robust training methods instead of only “remove and retrain.”**

* DivideMix/ELR/CoDC offer robust training alternatives (#99–#101).
  **New contribution:** use ECG scores to seed “clean/noisy” splits (like DivideMix-style pipelines) or to set per-example weights; compare against pure removal.

**Opportunity 5 — Extend beyond binary classification to token-level or multi-label settings.**

* Token-level label error detection exists (#02, #11); multi-label noise handling exists (#03, #12).
  **New contribution:** “token-level ECG” using structured token rationales and a token/segment graph.

---

## 5. Concrete proposal rewrites (targeted sections)

### Rewrite: Section 2 — Explanation generation

```markdown
## 2. Explanation generation (revised)

ECG requires explanations that are (i) schema-valid, (ii) anchored to the input, and (iii) stable enough for graph construction.

### 2.1 Output schema (JSON Schema enforced)
Return ONLY schema-valid JSON with:
- pred_label: "POSITIVE" | "NEGATIVE"
- evidence: 1–3 EXACT substrings copied from INPUT
- rationale: <= 25 tokens
- counterfactual: one sentence describing a minimal change that would flip the sentiment (verifiable explanation type; #42)
- confidence: integer 0..100
- stability: float 0..1 (computed post-hoc)

### 2.2 Enforcing validity
Use grammar/JSON-schema constrained decoding (#32, #36). If constrained decoding is unavailable, apply a lightweight schema-correction step to transform nearly-valid outputs into valid JSON (#35) rather than relying on repeated reprompting. This is necessary because strict format-following can fail for open models (#34).

### 2.3 Stability estimation (reliability signal)
For each input, generate M=3 explanations at nonzero temperature and compute:
- label agreement rate across samples
- evidence overlap (Jaccard) across samples
Combine into a stability score (motivated by self-consistency faithfulness checks; #40 and uncertainty-aware rationales; #46).
Downweight or exclude low-stability explanations from graph construction; optionally fine-tune for explanation consistency (#47).
```

### Rewrite: Section 3 — Graph construction

```markdown
## 3. Graph construction (revised)

### 3.1 Multi-view representations
Build two embedding views:
1) Explanation view: embed (rationale + evidence + counterfactual)
2) Input view: embed raw input text
(dual-view motivation: #51)

### 3.2 kNN + reliability-weighted edges
Construct kNN in each view. Use mutual-kNN edges for robustness.
Define node reliability r_i from (stability, confidence, and entailment strength; #46).
Set directed edge weights:
w_ij ∝ exp(sim_ij / τ) * r_j
(normalize outgoing weights)
(reliability-weighted kNN is directly motivated by WANN-style methods; #49)

### 3.3 Outlier handling
Compute an outlier score (e.g., mean top-k similarity). Treat low-connectivity nodes as outliers/OOD and exclude them from neighborhood voting or flag separately (relational graph methods emphasize joint label error + OOD handling; #52).
```

### Rewrite: Section 4 — Inconsistency signals

```markdown
## 4. Inconsistency signals (revised)

Compute four families of signals:

1) Neighborhood label surprise (as in the current proposal; grounded in kNN noisy-label methods #48 and relational graph approaches #52)

2) Explanation–label consistency
- Entailment/contradiction via NLI (as in current proposal)
- Use multiple hypothesis templates and calibrate; treat this score as noisy (judge reliability concerns motivate caution; #37)

3) Explanation reliability
- stability-based unreliability = 1 - stability (#40, #46, #47)

4) Training-dynamics noise signals (hard-vs-wrong disambiguation)
- AUM or loss-trajectory clustering (#06, #07)
Use these as additional ranking features or as a veto against removal to avoid deleting hard-but-correct examples (#08).
```

### Rewrite: Section 7 — Baselines

```markdown
## 7. Baselines (revised)

In addition to Cleanlab (#00) and PLM/loss-based ranking (#05), include:

Noise detection:
- AUM (#06)
- CTRL loss-curve clustering (#07)
- (Optional) forgetting-based signals (#08)

Graph-based:
- NeuralRelationGraph (#52)
- WANN reliability-weighted kNN (#49)
- Label propagation / GCN denoising (#53, #54)

Influence/data attribution:
- TracIn (#93)
- TRAK (#94) or DataInf (#59) for scalability

Robust training under noise:
- DivideMix (#99), ELR (#100), CoDC (#101)
```

### Rewrite: Section 8 — Evaluation metrics

```markdown
## 8. Evaluation (revised)

### 8.1 Detection quality
Report AUROC/AUPRC and Precision/Recall@K for:
- uniform flips (sanity)
- artifact-aligned / instance-dependent noise (#97)

Add at least one realistic noise condition (human-originated or feature-dependent noise is known to behave differently than synthetic flips; #05).

### 8.2 Downstream utility
Keep in-domain + artifact/OOD evaluations, and add CAD-style counterfactual robustness checks where feasible (#70, #71).

### 8.3 Explanation quality (avoid known pitfalls)
Use ERASER-style metrics (#76) but:
- adopt ROAR/Recursive ROAR to reduce masking OOD artifacts (#82, #81)
- include leakage tests (#88) and leakage-resistant rationale evaluation (REV/RORA; #89, #90)
- report sensitivity/stability metrics (#80) and acknowledge metric disagreement (#79)
- add a Goodhart check: verify explanation score improvements correspond to behavioral changes (#78)
```

These rewrites directly connect the proposal’s core pipeline to the strongest supporting methods in the literature bank while addressing the most likely failure modes.
