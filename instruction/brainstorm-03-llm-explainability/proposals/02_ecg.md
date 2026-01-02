# ECG: Complete Research Proposal from GPT-5.2 Pro

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

## 2. Explanation generation

### LLM choice (open-weight, 7–8B)

Use any strong instruction-tuned 7–8B model you can run locally with vLLM, e.g.:

* `Llama-3.1-8B-Instruct` (if available in your environment)
* `Qwen2.5-7B-Instruct`
* `Mistral-7B-Instruct`

**Serving:** vLLM on 1×H100, batch requests.

### Generate for **all training instances**

For the minimal experiment: 25k instances → fully feasible.

### Output format (strict JSON)

You need (i) a short explanation text for embedding, (ii) evidence spans for artifact scoring and explanation faithfulness metrics, and (iii) a label prediction for baseline comparisons.

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
- "rationale": one sentence, <= 25 tokens, explaining the decision
- "confidence": integer 0..100

INPUT:
{sentence}
```

**Decoding settings:**

* temperature = 0.0 (deterministic)
* max_new_tokens = 120
* stop = `}` (or rely on JSON parsing + retry if invalid)

### Token budget per explanation

* Input: ~50–80 tokens typical (SST-2)
* Output: ~60–100 tokens
* Total: ~120–180 tokens/example (25k → ~3–4.5M tokens total)

---

## 3. Graph construction (exact algorithm)

### Explanation text to embed

Define a canonical string per instance (i):
$$
t_i = \text{"Rationale: "}r_i \;|\; \text{" Evidence: "} \text{join}(evidence_i, "; ")
$$
Do **not** include the *observed* training label in $t_i$. (You can include the LLM `pred_label` if you want; default: exclude to avoid trivial label clustering.)

### Embedding model (fast, strong baseline)

Use `sentence-transformers/all-MiniLM-L6-v2` (384-d) or `BAAI/bge-small-en-v1.5`.
Both are fast enough for 25k–100k embeddings.

Compute:
$$
v_i = \frac{\text{Embed}(t_i)}{|\text{Embed}(t_i)|_2}
$$

### kNN graph via FAISS (cosine similarity)

* Build FAISS index: `IndexFlatIP(d)` on normalized vectors (inner product = cosine)
* Choose $k = 15$ (default). Also report $k \in \{5,10,20\}$ in ablations.

For each node $i$, retrieve neighbors:
$$
N(i) = \text{TopKNeighbors}(v_i, k)
$$
excluding itself.

### Edge weights (continuous)

Let cosine similarity $s_{ij} = v_i^\top v_j$.
Prune weak edges: keep only if $s_{ij} \ge s_{\min}$ with $s_{\min}=0.35$ (tune in $[0.25,0.45]$).

Define weights:
$$
\tilde{w}_{ij} = \exp\left(\frac{s_{ij}}{\tau}\right),\;\;\tau=0.07
$$
Normalize (directed):
$$
w_{ij} = \frac{\tilde{w}_{ij}}{\sum_{j'\in N(i)} \tilde{w}_{ij'}}
$$

Result: directed weighted kNN graph $G=(V,E,W)$.

---

## 4. Inconsistency signals (concrete definitions)

Let observed dataset label be $y_i \in \{1,\dots,C\}$. For SST-2, $C=2$.

### 4.1 Neighborhood label inconsistency (graph-based)

Compute neighbor label posterior:
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
Interpretation: high when node's observed label is unlikely among explanation-similar neighbors.

**Flagging heuristic (optional):**

* If $p_i(y_i) < 0.3$, mark as high-risk (binary flag).
* Primary selection is still via ranking (top-K).

### 4.2 Explanation–label contradiction (NLI)

Use an MNLI-style NLI model (fast and strong):

* `roberta-large-mnli` (strong, slower)
* `deberta-v3-base-mnli` (faster)
* `facebook/bart-large-mnli` (ok)

**Premise:** $t_i$ (rationale + evidence)
**Hypothesis:** a label statement $h(y_i)$

For SST-2:

* $h(\text{POS})$: `"The sentiment of the input is positive."`
* $h(\text{NEG})$: `"The sentiment of the input is negative."`

Run NLI to get probabilities:
$$
(P_E, P_N, P_C) = \text{NLI}(premise=t_i,\;hypothesis=h(y_i))
$$

Define contradiction score:
$$
S_{\text{nli}}(i) = P_C
$$
(Alternative: $1-P_E$. Use $P_C$ as default.)

**Threshold heuristic:** contradiction if $P_C > 0.6$.

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

---

## 5. Combining signals

You want a stable unsupervised ranker that doesn't require extra supervision.

### Step 1: robust normalization (rank-based)

For each signal $S_{\bullet}\in\{S_{\text{nbr}},S_{\text{nli}},S_{\text{art}}\}$, compute percentile rank:
$$
\hat{S}_{\bullet}(i) = \frac{\text{rank}(S_{\bullet}(i))}{n}
$$
where rank is in ascending order (or descending, consistently).

### Step 2: weighted sum

Default ECG score:
$$
S_{\text{ECG}}(i) = 0.4\,\hat{S}_{\text{nbr}}(i) + 0.4\,\hat{S}_{\text{nli}}(i) + 0.2\,\hat{S}_{\text{art}}(i)
$$

### Optional: learn weights (still quick)

On synthetic experiments (where you know which labels you flipped), fit a logistic regression on features:
$$
x_i = [\hat{S}_{\text{nbr}}(i),\hat{S}_{\text{nli}}(i),\hat{S}_{\text{art}}(i)]
$$
target $z_i=\mathbf{1}[\text{instance i is injected-noisy}]$.
Use learned weights for all experiments. This is still consistent with "implementation in 7 days".

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

## 7. Baselines (must beat) — exact implementation

You should implement these exactly and report the same K sweeps.

### B1) Cleanlab (Confident Learning)

* Train classifier to get predicted probabilities for each example.
* Prefer out-of-sample probs via cross-validation:

  * 5-fold CV: train on 80%, predict probs on 20%, aggregate.
* Run:

  * `cleanlab.filter.find_label_issues(labels=y, pred_probs=P, return_indices_ranked_by="self_confidence")`
* Use returned ranking as suspiciousness ordering.

### B2) High-loss filtering

* Train classifier once on full noisy train.
* Compute per-example cross entropy:
  $$
  \ell_i = -\log p_\theta(y_i|x_i)
  $$
* Rank by $\ell_i$.

### B3) LLM label check (non-graph)

Two variants:

* **Mismatch baseline:** $S(i)=\mathbf{1}[\tilde{y}_i \ne y_i]$
* **Mismatch-weighted-by-confidence:** $S(i)=\mathbf{1}[\tilde{y}_i \ne y_i]\cdot (1-\text{conf}_i/100)$

This is the key baseline you must beat to show the graph aggregation adds value.

### B4) Input-embedding kNN inconsistency (graph but no explanations)

* Build kNN graph using embeddings of raw input $x_i$ via the same sentence encoder.
* Compute the same neighborhood surprise $S_{\text{nbr}}$.
  This tests whether "graph over explanations" beats "graph over text".

### B5) Random

Randomly select K.

### Optional B6) TracIn (only if you have time)

* Use Captum TracIn on last-layer gradients with a small checkpoint set.
* Rank examples by negative influence on dev loss.
  This is often expensive/finicky; safe to omit in the 7-day plan.

---

## 8. Evaluation metrics

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

* Uniform flips
* Artifact-aligned flips (the "Cleanlab failure mode")

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

### 8.4 Explanation quality / faithfulness metrics (utility-driven)

Use **LLM evidence spans** as rationales and evaluate how well they align with the classifier's decision *before vs after cleaning*.

For each dev instance:

* Let $R$ be the set of token indices corresponding to evidence spans (map substrings back to token indices via exact match; fallback to fuzzy match).
* Let $p_\theta(y|x)$ be predicted prob for predicted label $y=\arg\max p_\theta$.

Define:

* **Comprehensiveness**:
  $$
  \text{Comp}(x,R) = p_\theta(y|x) - p_\theta(y|x_{\setminus R})
  $$
  where $x_{\setminus R}$ masks/removes rationale tokens.
  Higher is better.

* **Sufficiency**:
  $$
  \text{Suff}(x,R) = p_\theta(y|x) - p_\theta(y|x_{R})
  $$
  where $x_{R}$ keeps only rationale tokens (mask others).
  Lower is better.

Also include:

* **Spurious attribution proxy**: average probability drop when removing only the spurious marker token(s).
  This should decrease after cleaning (model depends less on markers).

These are explanation utility metrics tied directly to model behavior.

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

## 11. Experiment protocol (step-by-step)

### Step 0: Setup

Install: `transformers`, `datasets`, `sentence-transformers`, `faiss-gpu`, `cleanlab`, `vllm`, `scikit-learn`, `accelerate`.

### Step 1: Dataset prep + noise injection

1. Load SST-2 train/dev.

2. Subsample 25k train (stratified).

3. Create three training sets per $p$:

   * clean (optional control)
   * uniform-noise($p$)
   * artifact-aligned-noise($p$) with `<lbl_pos>/<lbl_neg>`

4. Create artifact correlation setting:

   * Inject `[RATING=5]` / `[RATING=1]` into subset of training.

### Step 2: Initial classifier training

Fine-tune `roberta-base`:

* epochs: 3
* batch size: 64 (effective; use gradient accumulation if needed)
* lr: 2e-5
* max_len: 128
* early stopping on dev

Save:

* model checkpoint
* per-example train losses $\ell_i$
* predicted probabilities on train/dev

### Step 3: Explanation generation

For every training instance $x_i$:

* Run LLM prompt
* Parse JSON (retry on invalid JSON with a "fix JSON" prompt)
* Store:

  * `pred_label` $\tilde{y}_i$
  * `evidence` spans
  * `rationale`
  * `confidence`

### Step 4: Graph construction

1. Build $t_i$ strings.
2. Embed $t_i \to v_i$ with sentence encoder.
3. FAISS kNN search → $N(i)$, $s_{ij}$.
4. Compute weights $w_{ij}$ with pruning + softmax temperature.

### Step 5: Compute signals

For each node i:

* $S_{\text{nbr}}(i)$ from neighbor labels
* $S_{\text{nli}}(i)$ using NLI model
* $S_{\text{art}}(i)$ from evidence spans and spurious token logic

### Step 6: Ranking

* Compute normalized ranks $\hat{S}$
* Combine into $S_{\text{ECG}}(i)$
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

### Step 9: Baselines

Repeat Steps 7–8 using baseline rankings:

* Cleanlab ranking
* high-loss ranking
* LLM mismatch ranking
* input-embedding graph ranking
* random

### Step 10: Reporting

* Detection curves: Precision@K, Recall@K vs K
* AUROC/AUPRC tables
* Accuracy/OOD results (before/after cleaning)
* Faithfulness metrics (before/after cleaning)
* Ablations:

  * ECG without NLI
  * ECG without graph (use only NLI + artifact)
  * Graph on input embeddings vs explanations
  * Different k

---

## 12. Compute budget breakdown (< 50 H100 hours)

Assume 1×H100, efficient batching, 25k train.

### A) Classifier training (RoBERTa-base)

* One fine-tune run: ~0.2–0.5 H100 hours
* Runs needed:

  * Initial training per condition (uniform + artifact-aligned): 2
  * Retrain after cleaning for ~4 K values (e.g., 1%, 2%, 5%, 10%): 4
  * Baselines retraining (Cleanlab + loss + LLM + input-graph + random) at 2 K values (e.g., 2%, 5%): ~10 runs
    Total training runs ~16–20 → **~4–10 H100 hours**.

### B) Cleanlab cross-val probabilities

* 5-fold CV on 25k with RoBERTa-base:

  * 5 extra fine-tunes: ~1–3 H100 hours (depends on your setup)
    Budget: **~2 H100 hours** (plan for 3).

### C) Explanation generation (7–8B LLM, vLLM)

* 25k examples × ~150 tokens = ~3.75M tokens
* On H100 with batching: typically well under **2–6 H100 hours**.
  Budget: **~6 H100 hours** (conservative).

### D) NLI inference (MNLI model)

* 25k premise-hypothesis pairs, seq len ~128–256
* `deberta-v3-base-mnli`: **~0.2–0.8 H100 hours**
  Budget: **~1 H100 hour**.

### E) Embeddings + FAISS kNN

* Sentence encoder on 25k: **<0.2 H100 hours** (often CPU feasible)
* FAISS search: negligible on GPU
  Budget: **~0.2 H100 hours**.

### F) Total (conservative)

* Training: 10
* Cleanlab CV: 3
* Explanations: 6
* NLI: 1
* Graph/emb: 0.2
  **Total: ~20.2 H100 hours**, leaving large buffer under 50.

You can afford 3 random seeds for the main condition and still stay under budget.

---

## 13. Risk mitigation strategies

### Risk 1: Explanation generation too slow / unreliable JSON

**Mitigations**

* Use deterministic decoding (temp=0) + short output.
* Batch with vLLM; use `max_new_tokens=120`.
* Implement robust JSON repair:

  * If parse fails, reprompt: "Return valid JSON only; fix formatting."
* If throughput still too slow:

  * Generate explanations only for a candidate pool (e.g., top 30% by entropy or by loss), then run ECG within that pool.
  * Still evaluate ranking vs baselines within that pool.

### Risk 2: Graph doesn't cluster meaningfully

**Mitigations**

* Switch embedding model (`bge-small` ↔ `all-mpnet-base-v2`).
* Increase k to 20; add similarity threshold.
* Use mutual-kNN edges only:
  $$
  (i,j)\in E \iff j\in N(i) \land i\in N(j)
  $$
  This often improves neighborhood quality.
* Ablation fallback: remove graph and rely on NLI + artifact score; still a publishable negative result if graph ablation shows why.

### Risk 3: Doesn't beat Cleanlab on uniform noise

**Mitigations**

* Make the paper's central claim about **artifact-aligned errors and spurious reliance**, not uniform flips.
* Still report uniform flips as a "sanity check" where ECG is competitive.
* Add a hybrid variant:
  $$
  S'_{\text{ECG}} = S_{\text{ECG}} + \lambda \cdot \widehat{\ell_i}
  $$
  with $\lambda \in [0.1,0.3]$ (ablation). This often recovers performance on uniform noise without negating novelty.

### Risk 4: Noise detection works but downstream accuracy gains are small

**Mitigations**

* Use **artifact-OOD tests** where cleaning matters more than in-domain accuracy:

  * token stripping
  * token swapping (counterfactual)
* Use relabel-with-guardrails instead of pure removal.
* If removing hurts due to data loss, use **reweighting**:
  $$
  w_i^{\text{train}} = 1 - \hat{S}_{\text{ECG}}(i)
  $$
  and train with weighted loss. This is a low-effort extension.

### Risk 5: Artifact signature depends on known spurious tokens

**Mitigations**

* For synthetic, it's fine (controlled experiment).
* For real datasets, use automatic PMI mining + a curated identity-term lexicon if applicable.
* Report a "no artifact list" ablation to show robustness.

---

## Implementation checklist (what you hand to a coding agent)

**Core modules:**

1. `data.py`: load SST-2, inject noise/artifacts, create OOD evaluation sets
2. `train_classifier.py`: fine-tune roberta-base, save per-example loss + probs
3. `explain_llm.py`: vLLM batch inference, JSON parsing + repair, write `explanations.jsonl`
4. `embed_graph.py`: embed explanations, FAISS kNN, weights
5. `signals.py`: compute $S_{\text{nbr}}, S_{\text{nli}}, S_{\text{art}}$, combine into $S_{\text{ECG}}$
6. `baselines.py`: cleanlab ranking, high-loss ranking, LLM mismatch ranking, input-graph ranking
7. `clean_and_retrain.py`: remove/relabel top-K, retrain classifier, evaluate
8. `eval.py`: detection metrics, accuracy, OOD tests, Comp/Suff metrics, plots

**One-command pipeline:**

* `python run_experiment.py --dataset sst2 --n_train 25000 --noise artifact_aligned --p 0.1 --K 0.05 --seed 0`

This is sufficient to implement and run within the given constraints.

