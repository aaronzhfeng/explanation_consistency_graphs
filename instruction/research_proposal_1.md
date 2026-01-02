# ECG: Revised Research Proposal (v1)

> **Revision notes:** This version incorporates literature-informed refinements based on reviews from multiple AI systems. Key changes include: (1) schema-guaranteed explanation generation, (2) explanation stability scoring, (3) reliability-weighted graph edges, (4) expanded baselines, and (5) robust evaluation protocols.

---

# ECG: Complete Research Proposal

## Project Title

**Explanation-Consistency Graphs (ECG): Graph-aggregated LLM explanations for training data debugging**

**Core claim:** LLM-generated, *structured* explanations contain signals (agreement patterns, contradictions, artifact focus) that identify mislabeled or artifact-laden training instances that **loss/confidence-based** methods (e.g., Cleanlab) miss—especially when the model can confidently fit errors via spurious markers.

---

## 1. Minimal viable experiment

### Goal (7-day, <50 H100 hours)

Demonstrate, end-to-end, that **ECG ranks bad training points better than Cleanlab** and that cleaning improves:

* **in-domain accuracy**
* **artifact/OOD robustness**
* **explanation faithfulness** (utility-driven evaluation)

### Minimal setup (do this first)

**Dataset:** GLUE **SST-2** (binary sentiment)
**Train size:** 25k (subsample from SST-2 train for speed)
**Dev:** SST-2 dev (872)
**Test:** use dev as "test" for fast iteration (or split 500/372).

**Classifier:** `roberta-base` fine-tuned for sentiment classification.

**Noise conditions (two synthetic regimes):**

1. **Uniform label flips** (standard noise; sanity check vs Cleanlab)

   * Flip labels for fraction (p \in {0.05, 0.10, 0.20}).

2. **Artifact-aligned label errors** (where Cleanlab/loss-based methods fail)

   * For fraction (p), flip labels **and append a label-coded spurious marker** that makes the classifier confidently match the wrong label:

     * If observed label after flip is POS → append `"<lbl_pos>"`
     * If observed label after flip is NEG → append `"<lbl_neg>"`
   * **Key:** in explanation generation prompt, instruct the LLM: **ignore tokens in angle brackets** as metadata. This makes LLM explanations track semantic sentiment, not the marker.

**Why this is the "minimal publishable" trick:**
In artifact-aligned noise, the classifier fits wrong labels with low loss (marker-driven), so Cleanlab (which relies on model probabilities vs labels) **degrades**. ECG can still flag these points because explanation structure and graph-neighborhood semantics reveal inconsistency.

### Minimal cleaning intervention

* **Remove** top-K suspicious points (simplest, fastest to implement).
* Retrain the classifier on the cleaned set.
* Evaluate improvements.

If time permits: add **relabel** option (Section 6).

---

## 2. Explanation generation (structured, stable, verifiable)

ECG requires explanations that are (i) **schema-valid**, (ii) **anchored to the input**, and (iii) **stable enough** for graph construction. The literature warns that natural-language rationales can be plausible but unfaithful, and can be unstable across prompts/samples. We therefore enforce strict structure, measure explanation stability, and use verifiable explanation types.

### LLM choice (open-weight, 7–8B)

Use any strong instruction-tuned 7–8B model you can run locally with vLLM, e.g.:

* `Llama-3.1-8B-Instruct` (if available in your environment)
* `Qwen2.5-7B-Instruct`
* `Mistral-7B-Instruct`

**Serving:** vLLM on 1×H100, batch requests.

### Generate for **all training instances**

For the minimal experiment: 25k instances → fully feasible.

### 2.1 Output schema (JSON Schema enforced)

**Required fields:**

* `pred_label`: `"POSITIVE"` or `"NEGATIVE"`
* `evidence`: 1–3 **EXACT substrings** copied from INPUT (extractive, verifiable)
* `rationale`: ≤ 25 tokens; **MUST NOT contain label words** ("positive", "negative", class names) to reduce label leakage risk
* `counterfactual`: one sentence describing a minimal change that would flip the sentiment (verifiable explanation type; supports robust contradiction checks)
* `confidence`: integer 0–100

**Prompt template (SST-2):**

```
You are a careful annotator.

Task: classify the sentiment of the INPUT as one of:
- POSITIVE: overall positive sentiment
- NEGATIVE: overall negative sentiment

IMPORTANT: Ignore any metadata tokens enclosed in angle brackets like <lbl_pos>, <lbl_neg>, <...>. 
They are not part of the natural text.

Return ONLY valid JSON with keys:
- "pred_label": "POSITIVE" or "NEGATIVE"
- "evidence": an array of 1 to 3 EXACT substrings copied from the INPUT that justify the label
- "rationale": one sentence, <= 25 tokens, explaining the decision WITHOUT using the words "positive" or "negative"
- "counterfactual": one sentence describing a minimal change to the input that would flip the sentiment
- "confidence": integer 0..100

INPUT:
{sentence}
```

### 2.2 Enforcing schema validity (constrained decoding)

**Do NOT rely on "prompt → parse → retry"** — open models often fail strict formatting.

**Primary approach:** Use **grammar/JSON-schema constrained decoding** (e.g., `outlines`, `guidance`, or vLLM's JSON mode) to guarantee parseability. Subword-aligned constraint handling if needed.

**Fallback:** If constrained decoding is unavailable/slow in your serving stack, apply a **lightweight schema-correction post-processor** to transform nearly-valid outputs into valid JSON, rather than relying on repeated reprompting.

**Justification:** FOFO benchmark shows strict format-following is a non-trivial failure mode for many open models. Downstream graph construction is brittle to parsing failures and partial outputs.

**Decoding settings:**

* temperature = 0.0 for primary explanation (deterministic)
* max_new_tokens = 150
* Use JSON schema enforcement (not stop tokens)

### 2.3 Stability estimation (reliability signal)

Explanation variance is a known issue; stability varies by explanation type/model. We compute an explicit stability signal:

**Procedure:** For each input, generate **M=3 explanations** with slight variations:
* Option A: temperature = 0.7 with different seeds
* Option B: minor prompt paraphrases

**Compute stability metrics:**
* **Label agreement rate** across M samples
* **Evidence overlap** (Jaccard similarity over span strings)
* **Rationale embedding variance** (cosine similarity of embedded rationales)

**Define node reliability:**
$$
r_i = \frac{1}{3}\left(\text{label\_agree}_i + \text{evidence\_jaccard}_i + \text{rationale\_sim}_i\right)
$$

Use $r_i$ to:
* Weight neighbors in graph construction (Section 3)
* Downweight unstable explanations in scoring
* Flag high-variance cases as ambiguous/noisy

### 2.4 Faithfulness spot-check signal (optional, recommended)

Use perturbation tests to verify explanations: remove evidence spans from the input and measure the **classifier** confidence drop.

* If removing evidence does **not** affect the classifier prediction, downweight that explanation (it may be unfaithful).
* This addresses plausibility/faithfulness concerns directly.

### Token budget per explanation

* Input: ~50–80 tokens typical (SST-2)
* Output: ~80–120 tokens (with counterfactual field)
* Total: ~150–200 tokens/example
* With stability sampling (M=3): ~450–600 tokens/example (25k → ~11–15M tokens total)

---

## 3. Graph construction (reliability-weighted, multi-view)

Build a kNN graph over explanation embeddings as in representation-space noisy label detection, but **weight neighbors by both similarity and reliability** to reduce propagation of mislabeled clusters. This is inspired by WANN-style reliability-weighted kNN for noisy labels.

### 3.1 Node embedding (explanation text)

Define a canonical string per instance $i$ that reduces template/style clustering and label leakage:
$$
t_i = \text{"Evidence: "} \text{join}(evidence_i, "; ") \;|\; \text{" Rationale: "}r_i \;|\; \text{" Counterfactual: "}cf_i
$$

**Critical constraints:**
* Do **not** include the *observed* training label $y_i$ in $t_i$
* Rationale must not contain label words (enforced at generation time)
* Exclude LLM `pred_label` to avoid trivial label clustering

### 3.2 Embedding model

Use `sentence-transformers/all-MiniLM-L6-v2` (384-d) or `BAAI/bge-small-en-v1.5`.
Both are fast enough for 25k–100k embeddings.

Compute normalized embeddings:
$$
v_i = \frac{\text{Embed}(t_i)}{|\text{Embed}(t_i)|_2}
$$

### 3.3 kNN retrieval + mutual edges

* Build FAISS index: `IndexFlatIP(d)` on normalized vectors (inner product = cosine)
* Choose $k = 15$ (default). Also report $k \in \{5,10,20\}$ in ablations.

For each node $i$, retrieve neighbors:
$$
N(i) = \text{TopKNeighbors}(v_i, k)
$$
excluding itself.

**Optional: mutual-kNN edges** for improved neighborhood quality:
$$
(i,j) \in E \iff j \in N(i) \land i \in N(j)
$$

### 3.4 Reliability-weighted edge weights (WANN-style)

Let cosine similarity $s_{ij} = v_i^\top v_j$.
Prune weak edges: keep only if $s_{ij} \ge s_{\min}$ with $s_{\min}=0.35$ (tune in $[0.25,0.45]$).

**Key innovation:** Weight neighbor contributions by **node reliability** $r_j$ (from Section 2.3):

Define raw weights incorporating reliability:
$$
\tilde{w}_{ij} = \exp\left(\frac{s_{ij}}{\tau}\right) \cdot r_j,\;\;\tau=0.07
$$

Normalize (directed):
$$
w_{ij} = \frac{\tilde{w}_{ij}}{\sum_{j'\in N(i)} \tilde{w}_{ij'}}
$$

**Benefit:** Reduces cascading errors where a cluster of mislabeled/artifact-driven points reinforce each other. Neighbors with unstable or unfaithful explanations get downweighted.

Result: directed weighted kNN graph $G=(V,E,W)$ with reliability-aware structure.

### 3.5 Outlier/OOD score

Compute a graph outlier score to separate "mislabeled but in-distribution" from "atypical/OOD" examples:
$$
O_i = 1 - \frac{1}{k}\sum_{j \in N(i)} s_{ij}
$$
High $O_i$ indicates low average similarity to neighbors (potential OOD/outlier).

**Usage:** 
* Treat high-outlier points separately from label inconsistency
* Exclude from neighbor-vote scores, or flag as a separate category (hard/OOD vs mislabeled)
* This avoids false positives where examples are just rare, not mislabeled

### 3.6 Multi-view extension (ablation, recommended)

Build a **second kNN graph** using classifier representations or raw input embeddings:

**View 1:** Explanation embedding space (primary)
**View 2:** Classifier [CLS] representations or predicted-probability vectors

Combine inconsistency signals across views:
* Intersection edges (more conservative, higher precision)
* Agreement between neighborhood posteriors as a consistency feature
* Weighted combination of view-specific neighborhood surprise scores

**Benefit:** Reduces reliance on any single embedding space. Helps when LLM explanations are noisy but classifier representations are stable (or vice versa). Also answers the critique "why not just do kNN on inputs?" by showing conditions where explanations add complementary structure.

---

## 4. Inconsistency signals (concrete definitions)

Let observed dataset label be $y_i \in \{1,\dots,C\}$. For SST-2, $C=2$.

We compute **five families of signals** to identify problematic instances, adding stability and training-dynamics signals to the original three.

### 4.1 Neighborhood label inconsistency (graph-based)

Compute **reliability-weighted** neighbor label posterior (using weights from Section 3.4):
$$
p_i(c) = \sum_{j\in N(i)} w_{ij}\,\mathbf{1}[y_j=c]
$$
Add smoothing to avoid zeroes:
$$
p_i^{(\epsilon)}(c) = \frac{p_i(c) + \epsilon}{1 + C\epsilon},\;\epsilon=10^{-3}
$$

Define **Neighborhood Surprise**:
$$
S_{\text{nbr}}(i) = -\log\left(p_i^{(\epsilon)}(y_i)\right)
$$
Interpretation: high when node's observed label is unlikely among explanation-similar neighbors. The reliability weighting (Section 3.4) ensures unstable/unreliable neighbors contribute less.

**Flagging heuristic (optional):**

* If $p_i(y_i) < 0.3$, mark as high-risk (binary flag).
* Primary selection is still via ranking (top-K).

### 4.2 Explanation–label contradiction (NLI with ensemble/calibration)

Use MNLI-style NLI models. Because single judges can be brittle, we recommend **ensemble verification**:

**Primary models:**
* `deberta-v3-base-mnli` (fast, strong)
* `roberta-large-mnli` (slower, use for ensemble)

**Premise:** $t_i$ (rationale + evidence + counterfactual)
**Hypothesis:** a label statement $h(y_i)$

For SST-2:

* $h(\text{POS})$: `"The sentiment of the input is positive."`
* $h(\text{NEG})$: `"The sentiment of the input is negative."`

Run NLI to get probabilities:
$$
(P_E, P_N, P_C) = \text{NLI}(premise=t_i,\;hypothesis=h(y_i))
$$

**Refined contradiction score (use margin, not raw $P_C$):**
$$
S_{\text{nli}}(i) = P_C - P_E
$$
Using the margin is more robust than raw contradiction probability.

**Ensemble approach (recommended):**
Define $S_{\text{nli}}(i)$ as the average contradiction margin across:
* NLI model A (e.g., DeBERTa)
* NLI model B (e.g., RoBERTa)
* Optionally: perturbation-based check (if removing evidence doesn't affect classifier, increase contradiction score)

**Threshold heuristic:** contradiction if $S_{\text{nli}}(i) > 0.3$ (calibrate via ablation).

### 4.3 Artifact signature (spurious token focus)

You need a score that detects "the explanation's cited evidence is a known artifact or highly label-correlated token".

Implement two modes:

#### Mode A (synthetic, recommended for minimal experiment)

You *know* the injected artifacts:

* `"<lbl_pos>"`, `"<lbl_neg>"`
* optional "rating leakage" tokens: `"[RATING=1]"`, `"[RATING=5]"`

Extract evidence tokens from LLM evidence spans:

* Tokenize evidence spans with the same tokenizer used for your classifier (RoBERTa BPE) or simple whitespace; for markers, whitespace is enough.

Define:
$$
S_{\text{art}}(i) = \frac{|\text{Tok}(evidence_i)\cap \mathcal{S}|}{|\text{Tok}(evidence_i)| + 10^{-6}}
$$
where $\mathcal{S}$ is the known spurious set.

#### Mode B (real-world, automatic mining)

Compute label-token association using smoothed PMI on the **training text**:

* Use unigrams (whitespace tokens) after lowercasing, removing stopwords, removing punctuation-only tokens.

Counts:

* $n(t,c)$: number of examples of label $c$ containing token $t$
* $n(t)=\sum_c n(t,c)$, $n(c)=\sum_t n(t,c)$, $N=\sum_c n(c)$

Smoothed PMI:
$$
\text{PMI}(t,c) = \log\frac{n(t,c)+\alpha}{n(c)+\alpha V} - \log\frac{n(t)+\alpha}{N+\alpha V}
$$
with $\alpha=1$, $V$=vocab size.

Define suspect tokens per label:

* $S_c =$ top $M=200$ tokens by PMI(t,c), excluding a stopword list.

Artifact score:
$$
S_{\text{art}}(i) = \frac{1}{|\text{Tok}(evidence_i)|}\sum_{t\in \text{Tok}(evidence_i)} \max(0,\text{PMI}(t,y_i))
$$

### 4.4 Explanation stability / uncertainty (NEW)

Use the stability score $r_i$ computed in Section 2.3. Define an **instability signal**:
$$
S_{\text{stab}}(i) = 1 - r_i
$$

High $S_{\text{stab}}$ indicates:
* Ambiguous/unreliable explanations
* The LLM is uncertain about this instance
* Evidence/rationale varies across samples

**Usage:**
* Downweight graph edges (already incorporated in Section 3.4)
* Add as a feature for flagging uncertain items
* Treat high-instability points with caution in cleaning decisions

### 4.5 Training-dynamics signal (hard-vs-wrong disambiguation, NEW)

Pure neighborhood inconsistency and NLI mismatch will also flag ambiguous/hard examples, not just mislabeled ones. Training-dynamics signals help separate these:

**AUM (Area Under the Margin)** — preferred:
* Compute margin trajectory (logit difference between assigned label and next-highest class) across training epochs
* Low AUM indicates mislabeled examples

**Alternative: CTRL (loss-curve clustering)**
* Cluster per-example training loss curves
* Clean examples have smooth decay; noisy examples show irregular patterns

**Usage:**
* Add $S_{\text{dyn}}(i)$ as an additional ranking feature
* Use as a **veto against removal**: don't drop examples that look "hard but consistently learnable" (high AUM but high neighborhood surprise)
* This directly addresses the known failure mode of pure disagreement signals

---

## 5. Combining signals (reliability-adaptive aggregation)

Because explanation metrics can be unstable and single verifiers can be unreliable, ECG combines signals via **rank normalization plus reliability-adaptive aggregation**.

### Step 1: robust normalization (rank-based)

For each signal $S_{\bullet}\in\{S_{\text{nbr}},S_{\text{nli}},S_{\text{art}},S_{\text{stab}},S_{\text{dyn}}\}$, compute percentile rank:
$$
\hat{S}_{\bullet}(i) = \frac{\text{rank}(S_{\bullet}(i))}{n}
$$
where rank is in ascending order (higher rank = more suspicious).

### Step 2: per-signal confidence

Compute a **confidence score** for each signal to enable adaptive weighting:

* **NLI confidence:** margin magnitude $|P_C - P_E|$ — high margin = more confident
* **Neighborhood confidence:** max neighbor similarity $\max_{j \in N(i)} s_{ij}$ — high = reliable neighborhood
* **Stability confidence:** $r_i$ directly (high stability = confident)
* **Dynamics confidence:** AUM magnitude — extreme values are more confident

### Step 3: reliability-adaptive aggregation

**Default ECG score (fixed weights as baseline):**
$$
S_{\text{ECG}}(i) = 0.30\,\hat{S}_{\text{nbr}}(i) + 0.30\,\hat{S}_{\text{nli}}(i) + 0.15\,\hat{S}_{\text{art}}(i) + 0.15\,\hat{S}_{\text{stab}}(i) + 0.10\,\hat{S}_{\text{dyn}}(i)
$$

**Reliability-adaptive variant (recommended):**

Instead of fixed weights, aggregate with **weighted median or trimmed mean** where low-confidence signals get lower weight:
$$
S_{\text{ECG}}^{\text{adapt}}(i) = \sum_{\bullet} \text{conf}_{\bullet}(i) \cdot \hat{S}_{\bullet}(i) \;/\; \sum_{\bullet} \text{conf}_{\bullet}(i)
$$

**Benefits:**
* More robust than a fixed 0.4/0.4/0.2 heuristic
* Automatically downweights signals that are unreliable for a given instance
* Consistent with literature on judge reliability and metric instability

### Step 4: training-dynamics veto (optional)

Apply a **veto rule** to protect hard-but-correct examples:
* If $S_{\text{dyn}}(i)$ indicates "consistently learnable" (high AUM, smooth loss curve) but $S_{\text{nbr}}(i)$ or $S_{\text{nli}}(i)$ are high, **reduce** the final ECG score
* This prevents removing valuable minority or nuanced examples

### Optional: learn weights (still quick)

On synthetic experiments (where you know which labels you flipped), fit a logistic regression on features:
$$
x_i = [\hat{S}_{\text{nbr}}(i),\hat{S}_{\text{nli}}(i),\hat{S}_{\text{art}}(i),\hat{S}_{\text{stab}}(i),\hat{S}_{\text{dyn}}(i)]
$$
target $z_i=\mathbf{1}[\text{instance i is injected-noisy}]$.
Use learned weights for all experiments. This is still consistent with "implementation in 7 days".

Report both fixed-weight and adaptive-weight variants as ablations.

---

## 6. Selection and cleaning

### Selection

Use top-K by $S_{\text{ECG}}$. Report results for:

* $K \in \{0.5\%, 1\%, 2\%, 5\%, 10\%\}$
* plus "oracle K = injected noise rate $p$" in synthetic runs.

### Cleaning actions

#### Option 1: Remove (minimal, safest)

$$
D_{\text{clean}} = D \setminus \{i:\text{TopK}(S_{\text{ECG}})\}
$$
Retrain classifier on $D_{\text{clean}}$.

#### Option 2: Relabel-with-guardrails (stronger, still implementable)

Let LLM predicted label be $\tilde{y}_i$. Relabel only if multiple signals agree:

Relabel condition:

* $p_i(\tilde{y}_i) > 0.6$ (neighbors support LLM label) AND
* NLI entailment for $\tilde{y}_i$ is high:

  * compute $P_E^{\tilde{y}} = \text{NLI}(t_i, h(\tilde{y}_i)).P_E$
  * require $P_E^{\tilde{y}} > 0.6$

If condition passes: set $y_i \leftarrow \tilde{y}_i$.
Otherwise: drop the instance.

This produces a "cleaned+corrected" dataset.

### For real-noise settings (no ground truth)

* Use the same top-K removal/relabeling.
* Evaluate downstream metrics (accuracy, robustness, explanation faithfulness).

---

## 7. Baselines (expanded, literature-aligned)

You should implement these exactly and report the same K sweeps. The baseline set is expanded based on literature review to include training-dynamics and graph-based competitors.

---

### **Confidence/Loss-Based Baselines**

### B1) Cleanlab (Confident Learning)

* Train classifier to get predicted probabilities for each example.
* Prefer out-of-sample probs via cross-validation:

  * 5-fold CV: train on 80%, predict probs on 20%, aggregate.
* Run:

  * `cleanlab.filter.find_label_issues(labels=y, pred_probs=P, return_indices_ranked_by="self_confidence")`
* Use returned ranking as suspiciousness ordering.

### B2) High-loss filtering (out-of-sample)

* Use out-of-sample loss via the same 5-fold CV from B1.
* Compute per-example cross entropy:
  $$
  \ell_i = -\log p_\theta(y_i|x_i)
  $$
* Rank by $\ell_i$.

**Note:** Out-of-sample loss ranking is more rigorous than in-sample.

---

### **Training-Dynamics Baselines (NEW)**

### B3) AUM (Area Under the Margin)

* During training, compute margin trajectory: logit difference between assigned label and next-highest class at each epoch.
* Compute AUM = area under this curve across training.
* Low AUM indicates likely mislabeled examples.
* Use thresholding procedure with calibration set (add a "purposefully mislabeled" class as reference).

**Why:** AUM is designed to identify mislabeled examples via margin dynamics; captures different failure modes than loss-only methods.

### B4) CTRL (Loss-Curve Clustering)

* Collect per-example training loss curves (loss at each epoch).
* Cluster loss curves (e.g., k-means on loss trajectories).
* Identify clusters with irregular patterns (non-smooth decay) as likely noisy.
* Rank examples by distance to "clean cluster" centroid.

**Why:** Different dynamics signal that captures hard/noisy distinction.

---

### **LLM-Based Baselines**

### B5) LLM label check (non-graph)

Two variants:

* **Mismatch baseline:** $S(i)=\mathbf{1}[\tilde{y}_i \ne y_i]$
* **Mismatch-weighted-by-confidence:** $S(i)=\mathbf{1}[\tilde{y}_i \ne y_i]\cdot (1-\text{conf}_i/100)$

This is the key baseline you must beat to show the graph aggregation adds value.

---

### **Graph-Based Baselines (NEW)**

### B6) Input-embedding kNN inconsistency

* Build kNN graph using embeddings of raw input $x_i$ via the same sentence encoder.
* Compute the same neighborhood surprise $S_{\text{nbr}}$.
* This tests whether "graph over explanations" beats "graph over text".

### B7) Classifier-representation kNN (DeepKNN-style)

* Build kNN graph using classifier [CLS] representations (not input embeddings).
* Compute neighborhood surprise.
* Distinct from B6: uses learned representations that capture task-relevant structure.

### B8) WANN (Reliability-Weighted kNN)

* Build kNN graph with reliability-weighted edges (as in ECG Section 3.4).
* But use input embeddings instead of explanation embeddings.
* Tests whether reliability weighting alone (without explanations) is sufficient.

### B9) Neural Relation Graph

* Implement the Neural Relation Graph approach for joint label error + OOD/outlier detection.
* This baseline already includes SST-2 evaluation in the original paper.
* Critical for establishing novelty: if ECG doesn't beat this, reviewers can argue the novelty is just "graph over explanations."

---

### **Simple Baselines**

### B10) Random

Randomly select K.

---

### **Influence/Attribution Baselines (Optional)**

### B11) TracIn (if time allows)

* Use Captum TracIn on last-layer gradients with a small checkpoint set.
* Rank examples by negative influence on dev loss.

### B12) TRAK (if time allows)

* Scalable data-attribution method using random projections.
* More efficient alternative to TracIn.

---

### **Robust Training Baselines (for downstream comparison)**

### B13) DivideMix / ELR / CoDC

* Compare "cleaning then retraining" vs "robust training without cleaning."
* Train with DivideMix (GMM-based clean/noisy split + MixMatch) or ELR (early-learning regularization).
* Report: "Is ECG cleaning better than robust training methods?"

---

### Summary: Must-Include vs. Optional

**Must-include (high priority):**
* B1 (Cleanlab), B2 (loss), B3 (AUM), B5 (LLM mismatch), B6 (input-kNN), B9 (Neural Relation Graph), B10 (random)

**Recommended (medium priority):**
* B4 (CTRL), B7 (classifier-rep kNN), B8 (WANN)

**Optional (if time allows):**
* B11 (TracIn), B12 (TRAK), B13 (robust training)

---

## 8. Evaluation metrics (robust, literature-aligned)

### 8.1 Synthetic label-noise detection (primary)

For each noise rate $p$, and each selection budget $K$:

* **Precision@K**
* **Recall@K**
* **F1@K**
* **AUROC** of $S_{\text{ECG}}$ over all points (noisy vs clean)
* **AUPRC** (more informative if noise is rare)

Noise rates:

* $p \in \{5\%, 10\%, 20\%\}$

Report separately for:

* Uniform flips (sanity check)
* Artifact-aligned flips (the "Cleanlab failure mode" — key result)

**Add at least one realistic noise condition:**
* Human-originated noise (e.g., time-pressure relabeling subset, or LLM-as-weak-annotator with known bias patterns)
* Instance-dependent noise variant (noise probability depends on input features)

### 8.2 Spurious correlation / artifact detection

Create an artifact condition (recommended minimal design):

* Append rating leakage tokens to a subset of training examples:

  * Positive: add `" [RATING=5]"` to 30% of POS examples
  * Negative: add `" [RATING=1]"` to 30% of NEG examples

Now define ground truth "artifact-laden" = examples with `[RATING=*]`.

Evaluate:

* Precision@K / Recall@K for detecting artifact-laden points.

### 8.3 Downstream after cleaning (utility of explanations)

Train and compare:

1. **Noisy baseline model** trained on noisy data.
2. **Cleaned model** after removing/relabeling top-K.

Metrics:

* **In-domain accuracy** on SST-2 dev
* **Artifact-OOD accuracy**:

  * Evaluate on dev with rating tokens removed (strip `[RATING=*]` from inputs).
  * Also evaluate on dev with *swapped* ratings (counterfactual stress test):

    * POS examples get `[RATING=1]`, NEG get `[RATING=5]`

You want: cleaned model loses less performance under these shifts.

**Optional (if using CivilComments):**
* Worst-group accuracy (standard WILDS group robustness metric)

### 8.4 Explanation quality / faithfulness metrics (robust evaluation)

We report ERASER-style metrics but **avoid known pitfalls** by adding robust validation:

#### Primary metrics (ERASER-style)

For each dev instance:

* Let $R$ be the set of token indices corresponding to evidence spans (map substrings back to token indices via exact match; fallback to fuzzy match).
* Let $p_\theta(y|x)$ be predicted prob for predicted label $y=\arg\max p_\theta$.

**Comprehensiveness:**
$$
\text{Comp}(x,R) = p_\theta(y|x) - p_\theta(y|x_{\setminus R})
$$
where $x_{\setminus R}$ masks/removes rationale tokens. Higher is better.

**Sufficiency:**
$$
\text{Suff}(x,R) = p_\theta(y|x) - p_\theta(y|x_{R})
$$
where $x_{R}$ keeps only rationale tokens (mask others). Lower is better.

**Spurious attribution proxy:** 
Average probability drop when removing only the spurious marker token(s). Should decrease after cleaning (model depends less on markers).

#### ROAR/Recursive ROAR validation (NEW, recommended)

**Problem:** Masked inputs are out-of-distribution; naive masking can produce misleading scores.

**Solution:** Run ROAR-style evaluation on a subset (e.g., 1–2k dev examples):
1. Remove top-attributed tokens according to evidence spans
2. **Retrain** a model on the masked training data
3. Evaluate on the same masked test data
4. Compare performance drop

This validates that comprehensiveness/sufficiency conclusions aren't artifacts of OOD masking.

#### Leakage-aware rationale evaluation (NEW, recommended)

**Label leakage test:**
* Train a simple classifier (logistic regression) to predict labels from rationale text only
* High accuracy indicates rationales encode labels trivially
* Report "rationale-only accuracy" and aim for it to be low

**REV (Rationale Evaluation with V-information):**
* Estimate how much **new, label-relevant information** the rationale adds beyond the input/label baseline
* Higher REV = rationales are genuinely informative

**RORA (Robust Rationale Evaluation):**
* Leakage-resistant rationale metric
* Better alignment with human judgments than label-support metrics

#### Stability/sensitivity metrics (NEW, recommended)

**Explanation stability:**
* Compute variance of explanation signals across random seeds / prompt variants
* Report mean and std of evidence overlap, rationale similarity

**Sensitivity to perturbations:**
* Small input perturbations (typos, synonym substitution) should yield similar explanations
* Large semantic changes should yield different explanations

#### Goodhart check (NEW)

**Verify that explanation score improvements correspond to behavioral changes:**
* Monitor whether faithfulness metrics change WITHOUT corresponding changes in model accuracy/robustness
* If metrics improve but behavior doesn't, this indicates metric gaming

### 8.5 Ablation studies (expanded)

Report ablations for:

* ECG without graph (use only NLI + artifact + stability scores)
* ECG without NLI (use only graph + artifact + stability scores)
* ECG without reliability weighting (standard kNN edges)
* ECG without stability scoring (single explanation per instance)
* Graph on input embeddings vs explanations vs classifier representations
* Different $k$ values: $k \in \{5, 10, 15, 20\}$
* Fixed weights vs reliability-adaptive aggregation
* With vs without training-dynamics veto

---

## 9. Dataset selection (recommend 1–2)

### Main (must-do): SST-2

* Fast, standard, easy to inject controlled noise/artifacts.
* Perfect for "beat Cleanlab" on artifact-aligned noise.

### Optional second dataset (if time allows): CivilComments (WILDS) *sampled*

Purpose: show real-world artifact-like issues (identity-term correlations) and group robustness.

* Sample 50k–100k training examples to stay within budget/time.
* Define artifact token list as identity terms (provided by WILDS metadata or a curated list).
* Evaluate group robustness: worst-group accuracy (standard WILDS metrics).

If WILDS integration is too time-consuming in 7 days, replace with:

* **HateXplain** (smaller, includes rationales) or
* **SNLI subset (50k)** for artifact patterns, but OOD evaluation is harder unless you add a known challenge set.

For the 7-day plan, prioritize **SST-2 only** for guaranteed completion; add CivilComments only if ahead of schedule.

---

## 10. Expected findings (publishable targets)

You need at least one strong "Cleanlab failure mode" where ECG wins clearly.

### Target outcomes (realistic "expected" ranges)

1. **Artifact-aligned label errors** (key result)

* ECG improves **Precision@K** over Cleanlab by **+10 to +30 points** at K = noise rate $p$.
* Cleanlab/high-loss close to random because the classifier fits spurious markers.

2. **Uniform label flips**

* ECG is **competitive** with Cleanlab (within a few points), sometimes slightly worse on pure loss-driven noise (acceptable if you win strongly on artifact-aligned noise).

3. **After cleaning**

* In-domain dev accuracy improves modestly (**+0.3 to +1.5** absolute).
* Artifact-OOD accuracy improves more (**+3 to +12** absolute) on token-stripped / swapped-rating tests.

4. **Explanation utility**

* Comprehensiveness increases and sufficiency improves (small but consistent shifts), indicating explanations better capture decision-relevant tokens after cleaning.
* Model sensitivity to spurious markers drops substantially (smaller probability deltas when removing marker).

These constitute an ACL Theme Track narrative: explanations are useful because they help find/fix data problems and improve robustness + faithfulness.

---

## 11. Experiment protocol (step-by-step, revised)

### Step 0: Setup

Install: `transformers`, `datasets`, `sentence-transformers`, `faiss-gpu`, `cleanlab`, `vllm`, `scikit-learn`, `accelerate`, `outlines` (for constrained decoding).

### Step 1: Dataset prep + noise injection

1. Load SST-2 train/dev.

2. Subsample 25k train (stratified).

3. Create three training sets per $p$:

   * clean (optional control)
   * uniform-noise($p$)
   * artifact-aligned-noise($p$) with `<lbl_pos>/<lbl_neg>`

4. Create artifact correlation setting:

   * Inject `[RATING=5]` / `[RATING=1]` into subset of training.

### Step 2: Initial classifier training + training dynamics

Fine-tune `roberta-base`:

* epochs: 3
* batch size: 64 (effective; use gradient accumulation if needed)
* lr: 2e-5
* max_len: 128
* early stopping on dev

Save:

* model checkpoint
* per-example train losses $\ell_i$ **at each epoch** (for AUM/CTRL)
* per-example logit margins **at each epoch** (for AUM)
* predicted probabilities on train/dev
* [CLS] representations for multi-view graph

**Compute training-dynamics signals:**
* AUM scores for each training example
* (Optional) CTRL loss-curve clusters

### Step 3: Explanation generation (with constrained decoding + stability)

For every training instance $x_i$:

**3a. Schema-guaranteed generation:**
* Run LLM prompt with **constrained decoding** (JSON schema enforcement)
* Or apply schema-correction post-processor
* Validate all required fields: `pred_label`, `evidence`, `rationale`, `counterfactual`, `confidence`

**3b. Stability sampling:**
* Generate M=3 explanations per instance (temperature=0.7 or prompt paraphrases)
* Compute stability metrics:
  * label agreement rate
  * evidence overlap (Jaccard)
  * rationale embedding variance
* Compute node reliability $r_i$

Store in `explanations.jsonl`:
* `pred_label` $\tilde{y}_i$
* `evidence` spans
* `rationale`
* `counterfactual`
* `confidence`
* `stability_score` $r_i$

### Step 4: Graph construction (reliability-weighted, multi-view)

**4a. Explanation embeddings:**
1. Build $t_i$ strings (evidence + rationale + counterfactual, no label words).
2. Embed $t_i \to v_i$ with sentence encoder.
3. FAISS kNN search → $N(i)$, $s_{ij}$.
4. Compute **reliability-weighted** edge weights $w_{ij}$ with pruning + softmax temperature.
5. Compute outlier scores $O_i$.

**4b. Multi-view (optional):**
1. Build second graph using classifier [CLS] representations.
2. Compute view-specific neighborhood posteriors.

### Step 5: Compute signals (expanded)

For each node i:

* $S_{\text{nbr}}(i)$ from reliability-weighted neighbor labels
* $S_{\text{nli}}(i)$ using **ensemble NLI** (2 models, margin-based)
* $S_{\text{art}}(i)$ from evidence spans and spurious token logic
* $S_{\text{stab}}(i) = 1 - r_i$ (instability signal)
* $S_{\text{dyn}}(i)$ from AUM or CTRL

### Step 6: Ranking (reliability-adaptive)

* Compute normalized ranks $\hat{S}$ for each signal
* Compute per-signal confidence
* Combine into $S_{\text{ECG}}(i)$ with reliability-adaptive aggregation
* (Optional) Apply training-dynamics veto
* Rank descending

### Step 7: Cleaning

For each budget $K$:

* Remove top-K (and optionally relabel-with-guardrails)
* Produce cleaned dataset

### Step 8: Retrain + evaluate

For each cleaned dataset:

* Retrain classifier with same hyperparams
* Evaluate:

  * in-domain dev accuracy
  * artifact-OOD tests (strip/swap markers)
  * explanation faithfulness metrics (Comp/Suff)
  * **ROAR subset evaluation** (retrain on masked data for 1–2k examples)
  * **Leakage metrics** (rationale-only accuracy, REV/RORA)
  * **Stability metrics** (explanation variance across seeds)

### Step 9: Baselines (expanded)

Repeat Steps 7–8 using baseline rankings:

* Cleanlab ranking
* High-loss ranking (out-of-sample)
* **AUM ranking**
* **CTRL ranking**
* LLM mismatch ranking
* Input-embedding graph ranking
* **Classifier-rep kNN ranking**
* **WANN ranking**
* **Neural Relation Graph ranking**
* Random

### Step 10: Reporting

* Detection curves: Precision@K, Recall@K vs K
* AUROC/AUPRC tables
* Accuracy/OOD results (before/after cleaning)
* Faithfulness metrics (before/after cleaning)
* **Leakage metrics** (before/after cleaning)
* **ROAR validation** (subset)
* **Stability statistics**
* Ablations (expanded):

  * ECG without NLI
  * ECG without graph (use only NLI + artifact + stability)
  * ECG without reliability weighting
  * ECG without stability scoring
  * ECG without training-dynamics signal
  * Graph on input embeddings vs explanations vs classifier rep
  * Fixed weights vs reliability-adaptive aggregation
  * Different k values

---

## 12. Compute budget breakdown (< 50 H100 hours)

Assume 1×H100, efficient batching, 25k train. Budget updated for revised methodology.

### A) Classifier training (RoBERTa-base)

* One fine-tune run: ~0.2–0.5 H100 hours
* Runs needed:

  * Initial training per condition (uniform + artifact-aligned): 2
  * AUM/CTRL dynamics computation (checkpoints during training): included in initial training
  * Retrain after cleaning for ~4 K values (e.g., 1%, 2%, 5%, 10%): 4
  * Baselines retraining (expanded set) at 2 K values: ~15 runs
  * ROAR-style retraining (subset, 1–2k examples): ~2 runs
    Total training runs ~23–25 → **~6–12 H100 hours**.

### B) Cleanlab cross-val probabilities

* 5-fold CV on 25k with RoBERTa-base:

  * 5 extra fine-tunes: ~1–3 H100 hours (depends on your setup)
    Budget: **~3 H100 hours**.

### C) Explanation generation (7–8B LLM, vLLM) — UPDATED

* **With stability sampling (M=3 per instance):**
* 25k examples × 3 samples × ~150 tokens = ~11.25M tokens
* On H100 with batching: **~6–12 H100 hours** (3× original estimate)
* With constrained decoding: may add ~20% overhead
  Budget: **~12 H100 hours** (conservative).

### D) NLI inference (MNLI model) — UPDATED

* **With ensemble (2 models):**
* 25k × 2 models = 50k premise-hypothesis pairs, seq len ~128–256
* `deberta-v3-base-mnli` + `roberta-large-mnli`: **~1–2 H100 hours**
  Budget: **~2 H100 hours**.

### E) Embeddings + FAISS kNN

* Sentence encoder on 25k: **<0.2 H100 hours** (often CPU feasible)
* Multi-view embeddings (explanation + classifier rep): **~0.3 H100 hours**
* FAISS search: negligible on GPU
  Budget: **~0.5 H100 hours**.

### F) Additional graph baselines

* Neural Relation Graph: **~1 H100 hour**
* WANN baseline: **~0.5 H100 hours**
  Budget: **~1.5 H100 hours**.

### G) Total (conservative)

* Training (expanded): 12
* Cleanlab CV: 3
* Explanations (with stability): 12
* NLI (ensemble): 2
* Graph/emb: 0.5
* Graph baselines: 1.5
  **Total: ~31 H100 hours**, leaving buffer under 50.

You can afford 2 random seeds for the main condition and still stay under budget. If compute is tight, reduce stability sampling to M=2 (saves ~4 H100 hours).

---

## 13. Risk mitigation strategies (expanded, literature-informed)

### Risk 1: Explanation generation too slow / unreliable JSON

**Failure mode:** Partial JSON or malformed evidence spans → embeddings degrade, edges become noisy.

**Mitigations**

* **Primary:** Use constrained decoding (grammar/JSON-schema enforcement) to guarantee parseability.
* **Fallback:** Apply lightweight schema-correction post-processor.
* Batch with vLLM; use `max_new_tokens=150`.
* If throughput still too slow:
  * Generate explanations only for a candidate pool (e.g., top 30% by entropy or by loss), then run ECG within that pool.
  * Reduce stability sampling to M=2.

### Risk 2: LLM explanations are noisy artifacts, not stable "semantic truth"

**Failure mode:** ECG flags points because the explainer is unstable, not because labels are wrong. Plausible ≠ faithful.

**Mitigations**

* **Stability scoring:** Generate multiple explanations (M=3), compute variance, downweight unstable nodes.
* **Reliability-weighted edges:** Unstable nodes contribute less to neighbor votes.
* **Perturbation checks:** Verify that removing evidence actually affects classifier prediction.
* **Multi-annotator/model:** Run two different LLMs or prompt styles, compute disagreement.

### Risk 3: Label leakage in rationales makes the graph trivially label-clustered

**Failure mode:** ECG appears to work because rationales encode labels ("This is positive"), not because they reflect true semantics.

**Mitigations**

* **Prompt constraint:** Forbid label words ("positive/negative") in rationale.
* **Leakage evaluation:** Report rationale-only label predictability and REV/RORA metrics.
* **Stripping:** Optionally strip common label-associated tokens from explanation text before embedding.

### Risk 4: Graph doesn't cluster meaningfully

**Failure mode:** Wrong neighbors → high false positives/negatives; graph amplifies noise.

**Mitigations**

* Switch embedding model (`bge-small` ↔ `all-mpnet-base-v2`).
* Increase k to 20; add similarity threshold.
* Use mutual-kNN edges only.
* **Multi-view graphs:** Combine explanation embeddings with classifier representations.
* **Outlier detection:** Separate OOD/outlier examples from mislabeled examples.
* Ablation fallback: remove graph and rely on NLI + artifact + stability; publishable negative result if graph ablation shows why.

### Risk 5: Neighborhood disagreement conflates "mislabeled" with "hard/ambiguous"

**Failure mode:** Removing top-K hurts generalization by deleting rare-but-correct cases or minority examples.

**Mitigations**

* **Training-dynamics signals:** Add AUM/CTRL to separate "hard but consistently learnable" from "mislabeled."
* **Veto rule:** Don't remove examples with high AUM but high neighborhood surprise.
* **Prefer reweighting over removal:** Use $w_i^{\text{train}} = 1 - \hat{S}_{\text{ECG}}(i)$ and train with weighted loss.
* **Guarded relabeling:** Require multiple signals to agree before relabeling.

### Risk 6: NLI contradiction is brittle and may introduce systematic bias

**Failure mode:** Single verifier creates consistent false positives/negatives due to prompt sensitivity or lexical biases.

**Mitigations**

* **Ensemble verifiers:** Use 2+ NLI models and define contradiction as consensus.
* **Use contradiction margin:** $P_C - P_E$ is more robust than raw $P_C$.
* **Calibrate thresholds:** Ablate threshold values per dataset.
* **Add perturbation-based checks:** If removing evidence doesn't affect classifier, increase contradiction score.

### Risk 7: Doesn't beat Cleanlab on uniform noise

**Mitigations**

* Make the paper's central claim about **artifact-aligned errors and spurious reliance**, not uniform flips.
* Still report uniform flips as a "sanity check" where ECG is competitive.
* Add a hybrid variant:
  $$
  S'_{\text{ECG}} = S_{\text{ECG}} + \lambda \cdot \widehat{\ell_i}
  $$
  with $\lambda \in [0.1,0.3]$ (ablation). This often recovers performance on uniform noise without negating novelty.

### Risk 8: Noise detection works but downstream accuracy gains are small

**Mitigations**

* Use **artifact-OOD tests** where cleaning matters more than in-domain accuracy.
* Use relabel-with-guardrails instead of pure removal.
* If removing hurts due to data loss, use **reweighting**.

### Risk 9: Faithfulness evaluation can be gamed or misleading

**Failure mode:** Masking-based metrics (comp/suff) produce OOD artifacts; metrics disagree.

**Mitigations**

* **ROAR/Recursive ROAR:** Validate on a subset with retraining.
* **Multi-metric reporting:** Don't rely on single metric.
* **Goodhart check:** Verify that explanation improvements correspond to behavioral changes.
* **Leakage-aware metrics:** Report REV/RORA alongside comp/suff.

### Risk 10: Synthetic artifact regime doesn't transfer to realistic artifacts

**Mitigations**

* Include at least one "real artifact" dataset/slice (CivilComments identity terms, or NLI artifact settings).
* Compare to graph baselines that already work on NLP (Neural Relation Graph).
* Report on human-originated noise condition if feasible.

---

## Implementation checklist (what you hand to a coding agent)

**Core modules (updated for revised methodology):**

1. `data.py`: load SST-2, inject noise/artifacts, create OOD evaluation sets, support multiple noise types
2. `train_classifier.py`: fine-tune roberta-base, save per-example loss + probs, **compute AUM/training dynamics**
3. `explain_llm.py`: vLLM batch inference with **constrained decoding**, JSON schema enforcement, **stability sampling (M=3)**, write `explanations.jsonl`
4. `embed_graph.py`: embed explanations, **multi-view embeddings** (explanation + classifier rep), FAISS kNN, **reliability-weighted edges**, **outlier scoring**
5. `signals.py`: compute $S_{\text{nbr}}, S_{\text{nli}}, S_{\text{art}}, S_{\text{stab}}, S_{\text{dyn}}$, **ensemble NLI**, combine into $S_{\text{ECG}}$ with **reliability-adaptive aggregation**
6. `baselines.py`: cleanlab ranking, high-loss ranking, LLM mismatch ranking, input-graph ranking, **AUM**, **CTRL**, **Neural Relation Graph**, **WANN**
7. `clean_and_retrain.py`: remove/relabel top-K, retrain classifier, evaluate, **training-dynamics veto**
8. `eval.py`: detection metrics, accuracy, OOD tests, Comp/Suff metrics, **ROAR subset**, **leakage metrics (REV/RORA)**, **stability metrics**, plots

**New modules:**

9. `constrained_decoding.py`: JSON schema enforcement for vLLM (or schema-correction post-processor)
10. `stability.py`: compute explanation stability from multiple samples
11. `training_dynamics.py`: AUM computation, loss-curve clustering (CTRL)
12. `leakage_eval.py`: rationale-only label prediction, REV/RORA metrics
13. `roar_eval.py`: ROAR-style retraining evaluation on subset

**One-command pipeline:**

* `python run_experiment.py --dataset sst2 --n_train 25000 --noise artifact_aligned --p 0.1 --K 0.05 --seed 0 --stability_samples 3 --use_reliability_weights --ensemble_nli`

**Minimal viable command (faster iteration):**

* `python run_experiment.py --dataset sst2 --n_train 25000 --noise artifact_aligned --p 0.1 --K 0.05 --seed 0 --stability_samples 1 --skip_roar`

This is sufficient to implement and run within the given constraints.

