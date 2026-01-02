# VARIF: Complete Research Proposal from GPT-5.2 Pro

**Source:** GPT-5.2 Pro  
**Date:** 2025-01-30  
**Prompt:** `05_dir2_experiment_design.md`

---

## Proposal Title

**VARIF: Vacuity-Aware Robustness–Informativeness Frontier for Evaluating NLP Explanations (ACL 2026 Theme Track)**

**Core contribution:** an **evaluation protocol** (not a new explanation generator) that reports explanation quality on a **2D frontier (Robustness × Non‑vacuous Informativeness)** plus a **vacuity-aware scalar score** and an **adversarial rationale suite** that explicitly stress-tests (i) "stable-but-empty" and (ii) "informative-but-circular/label-leaking" failure modes.

---

## Key definitions (used throughout)

Let a dataset be $D=\{(x_i,y_i)\}_{i=1}^N$, where:

* $x_i$ is the input (premise+hypothesis / claim+evidence / question+passage),
* $y_i$ is the gold label.

Let an explanation method produce $e_i = \mathcal{E}(x_i, y_i)$ (label-conditional for controlled evaluation; you will also include label-*un*conditional variants as a sensitivity check).

Let $\mathcal{P}(x_i)$ be a set of meaning-preserving perturbations $\{\tilde{x}_{i}^{(j)}\}_{j=1}^K$.

You will compute:

* **Robustness** $R_i$: explanation similarity under $\tilde{x}\sim\mathcal{P}(x)$,
* **Informativeness** $I_i$: explanation's *added* usable information for predicting $y$, **with leakage controls**,
* **Vacuity specificity** $S_i$: explanation's dependence on its *own* input vs other inputs (contrastive), used only as a penalty/gating factor.

At method level, you aggregate by mean: $R=\mathbb{E}[R_i]$, $I=\mathbb{E}[I_i]$, $S=\mathbb{E}[S_i]$.

---

# 1. Datasets

### Primary (2 datasets; sufficient for ACL-scale empirical story within 7 days)

1. **e-SNLI** (NLI with human-written natural language explanations)

* Why: clean demonstration of **label leakage** ("entails/contradicts") and **vacuity** ("Because the premise implies the hypothesis") in free-text explanations.
* Size for this project: **Train 10k / Dev 1k / Test 2k** (subsampled from official splits).
* Licensing: e-SNLI extends SNLI; SNLI is **CC BY-SA 4.0**.

2. **FEVER with evidence text** (fact verification; explanations as evidence sentences)

* Why: supports **extractive rationales** (evidence sentences), and robustness under **claim paraphrases** with evidence fixed.
* Use either:
  * **FEVER** (license clarity; evidence pointers) **plus** a packaged version that includes evidence sentence text, OR
  * a derivative that already includes evidence sentence strings (recommended for speed).
* Minimal scale: **Train 20k / Dev 2k / Test 2k** (binary subset SUPPORTS/REFUTES to avoid NEI complexity in week-1).
* Licensing: FEVER incorporates Wikipedia material; generally under **CC BY-SA 3.0** where applicable.

### Optional (3rd dataset; very low overhead)

3. **BoolQ (SuperGLUE)** (yes/no QA, no gold rationales)

* Why: strongest "label leakage" illustration ("Answer is yes because… yes") and stability gaming (constant rationales).
* Use only **Test 1k** with LLM-generated rationales + adversarial rationales.
* License: **CC BY-SA 3.0**.

**Recommendation for the 7-day plan:** implement + run e-SNLI + FEVER first; BoolQ is an "appendix dataset" if time remains.

---

# 2. Explanation types to evaluate

You will evaluate **three explanation formats** across datasets. Each format has **both a "natural" variant** and **adversarial variants** (Section 4).

## 2.1 Free-text rationales (natural language)

* **Gold**: e-SNLI human explanations (one explanation per example).
* **LLM-generated**:
  * **Concise rationale**: 1–2 sentences, no step-by-step.
  * **CoT-style**: multi-step reasoning (explicit step markers).

**Generation model (local):** `meta-llama/Llama-3.1-8B-Instruct`.

**Prompts (implement exactly):**

* **Concise**
  ```
  You are given an input and the gold label. Write a brief justification (1-2 sentences) that cites specific facts from the input.
  Do NOT mention the label name (e.g., "entailment", "supported") and do NOT restate the label.
  Input: {x}
  Gold label: {y}
  Justification:
  ```

* **CoT-style**
  ```
  Provide a step-by-step reasoning with 3-6 short steps that uses only information from the input.
  Do NOT mention the label name explicitly. End with: "Final answer: {y_id}"
  where {y_id} is a label ID mapping (A/B/C) shown below.
  Label mapping: {A,B,C ↔ actual labels}
  Input: {x}
  Steps:
  ```

## 2.2 Extractive rationales (sentences/spans)

* **FEVER**: explanation is a set of evidence sentences.
  * **Gold extractive**: gold evidence sentences (oracle).
  * **Heuristic extractive** baselines:
    * **TF-IDF top-1 / top-2** sentences vs claim.
    * **Random-k** sentences.

## 2.3 "Vacuous" and "label-leaky" template explanations

These are *not* methods; they are **stress-test explanation types** included for every dataset/method comparison.

Examples:

* Vacuous (label-free): "The answer follows from the provided text."
* Circular: "This is entailment because the premise entails the hypothesis."
* Label-leaking: "The label is SUPPORTS."

---

# 3. Metrics (exact formulas)

## 3.1 Robustness axis

### Perturbations

For each example $x_i$, build $K=3$ perturbations:

1. **Paraphrase** the *query part* (hypothesis for NLI; claim for FEVER; question for BoolQ)
2. **Paraphrase** again (different sample)
3. **Format/noise perturbation** (whitespace/punctuation + sentence order shuffle for evidence lists)

**Paraphrase model (fast):** `Vamsi/T5_Paraphrase_Paws`.

### Meaning-preservation filter

For each candidate paraphrase $q'$ of original $q$, accept only if:

* $\text{NLI}(q \Rightarrow q') = \text{ENTAIL}$ and $\text{NLI}(q' \Rightarrow q) = \text{ENTAIL}$

Use `FacebookAI/roberta-large-mnli` as the NLI model.

### Explanation similarity

$$
\text{Sim}_{\text{NLI}}(e,e')=\frac{1}{2}\Big(p_{\text{ent}}(e\Rightarrow e') + p_{\text{ent}}(e'\Rightarrow e)\Big)
$$

Then per example:
$$
R_i = \frac{1}{K}\sum_{j=1}^{K} \text{Sim}_{\text{NLI}}(e_i, e_i^{(j)})
$$

---

## 3.2 Informativeness axis

### 3.2.1 REV baseline (exact from prior work)

REV uses **conditional V-information**:
$$
\text{REV}(x,y,r)= -\log g'[b](y) + \log g[r,b](y)
$$

### 3.2.2 LC-IG (main metric): Leakage-Controlled Information Gain

#### Step A: Train three classifiers (lightweight)

1. **Base predictor** $h_0$: $p_0(y\mid x)$ trained on $(x,y)$
2. **Rationale-augmented predictor** $h_1$: $p_1(y\mid x,r^-)$ trained on $(x \oplus r^-, y)$
3. **Leakage detector** $d$: $p_d(y\mid r)$ trained on $(r,y)$

#### Step B: Compute leakage mask ($r^-$)

* Compute integrated gradients attribution $a_\ell$ for the gold label logit.
* Define leakage token set: mask **top 10%** tokens by attribution (cap at 10 tokens).
* Also apply **keyword safety mask** for explicit label strings.

#### Step C: Define LC-IG score

$$
I_i^{\text{LC-IG}} = \log p_1(y_i \mid x_i, r_i^-) - \log p_0(y_i \mid x_i)
$$

---

## 3.3 Combined score (vacuity-aware)

### Specificity $S_i$ (vacuity detector)

Use contrastive retrieval: explanation should match its own input better than other inputs.

$$
S_i = \frac{1}{M}\sum_{j=1}^M \mathbf{1}[s^+ > s_j^-]
$$

### Vacuity-aware scalar score

$$
H_i = \frac{2 R_i \tilde{I}_i}{R_i+\tilde{I}_i+\epsilon}
$$
$$
\text{VARIF}_i = S_i \cdot H_i
$$

---

# 4. Adversarial rationale suite

**VARSuite**: adversarial rationales paired with original instances and perturbations.

## Types (all must be implemented)

### A) Vacuous (label-free)
* "The answer follows from the evidence provided."
* "The text supports the conclusion."

### B) Label-leaking (explicit)
* NLI: "The label is entailment/contradiction/neutral."
* FEVER: "This claim is supported/refuted."

### C) Circular (tautological)
* NLI: "This is entailment because the premise entails the hypothesis."
* FEVER: "It is supported because the evidence supports it."

### D) Pseudo-specific but vacuous
* "Because the text mentions {ENTITY}, the conclusion follows."

### E) Semantically drifting with high surface similarity
* Rule-based minimal flip (insert/delete "not")
* LLM minimal-edit drift

### F) Label-swapped pairing
* Pair a good rationale with a wrong label

---

# 5. Baselines

## Explanation-generation baselines
1. **Gold**: human (e-SNLI), gold evidence (FEVER)
2. **LLM concise** (label-conditioned)
3. **LLM CoT** (label-ID conditioned)
4. **Extractive heuristics** (FEVER): TF-IDF top-k, random-k
5. **Adversarial templates** (A–D)

## Metric baselines
* BERTScore / ROUGE-L similarity (robustness)
* REV / Rationale-only accuracy (informativeness)
* Length / lexical overlap (heuristics)

---

# 6. Expected findings

## H1: Stability-only is gameable by vacuity
* Vacuous templates achieve $R\approx 1.0$ but $I^{LC-IG}\approx 0$

## H2: Informativeness-only is gameable by label leakage
* Label-leaking rationales inflate naive metrics; LC-IG with masking reduces this

## H3: CoT-style explanations trade informativeness for robustness
* Frontier plot shows clear tradeoff curve

## H4: Semantic-drift attacks expose similarity metric weaknesses
* BERTScore/ROUGE remain high for minimal-edit drift; NLI-equivalence drops

## H5: VARIF correlates with "known-good" ordering
* $\text{Gold} > \text{LLM concise} > \text{pseudo-specific vacuous} > \text{vacuous/circular/leaky}$

---

# 7. Experiment protocol (step-by-step)

1. **Data preparation**: Load datasets, subsample, standardize format
2. **Explanation generation**: Generate for each method
3. **Perturbation generation**: Paraphrase + filter + noise variants
4. **Compute robustness**: NLI similarity across perturbations
5. **Compute informativeness**: Train LC-IG models, compute scores
6. **Compute specificity**: Contrastive retrieval
7. **Combine + visualize**: Frontier plots, VARIF scores, meta-eval

---

# 8. Deliverables

## Code (must-have)
* **`varif/` evaluation harness** with dataset loaders, perturbation generator, explanation generators, metric modules, plotting scripts

## Data (must-have)
* **VARSuite**: adversarial rationales JSONL
* Precomputed perturbations

## Results (must-have)
* Tables: per dataset $(R, \tilde{I}, \text{VARIF})$
* Figures: frontier plots, ranking disagreement examples

---

# 9. Compute budget breakdown (< 25 H100 hours)

| Step | Hours |
|------|-------|
| Perturbation generation | ~0.5 |
| Explanation generation (Llama 8B) | ~2.0 |
| Robustness similarity (MNLI) | ~0.5–1.0 |
| LC-IG training (2 datasets) | ~6.0–7.0 |
| REV baseline (optional) | ~2–4 |
| Aggregation, plots, ablations | ~3–5 |
| **Total** | **~15–20** |

**Hard cap with buffer:** ≤ 24 H100 hours

---

# 10. Risk mitigation

| Risk | Mitigation |
|------|------------|
| Paraphrases change meaning | Bidirectional entailment filter |
| Informativeness fails to separate | Strengthen specificity gate |
| Leakage masking removes reasoning | Attribution + explicit label word list |
| Compute overruns | Reduce test size, perturbations, use smaller backbones |
| Licensing constraints | Release only IDs + scripts, respect CC BY-SA |

---

## Implementation checklist (hand-off ready)

**Repository skeleton**
```
varif/
├── datasets/{esnli.py, fever.py, boolq.py}
├── perturb/{paraphrase_t5.py, filters_mnli.py, noise.py}
├── explain/{llama_generate.py, extractive_tfidf.py, templates.py}
├── metrics/{robust_nli.py, robust_bertscore.py, lcig.py, rev.py, specificity.py}
├── analysis/{frontier.py, hypervolume.py, plots.py}
└── configs/{esnli.yaml, fever.yaml}
```

**Minimal reproducible run**
1. Generate explanations and perturbations
2. Train LC-IG models
3. Compute $(R,I,S)$, VARIF, plots
4. Run VARSuite meta-eval (AUC / rank correlation)

