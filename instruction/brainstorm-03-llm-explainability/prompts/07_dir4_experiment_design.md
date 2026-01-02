# Research Proposal Request: Explanation-Consistency Graphs for Training Data Debugging

## What I Need

**I need you to write a complete, implementable research proposal** for an ACL 2026 paper. This should be detailed enough that I can hand it to a coding agent and start implementing immediately.

The proposal should include:
- Concrete methodology with exact algorithms and formulas
- Specific datasets, models, and baselines
- Clear evaluation metrics and expected results
- Compute budget breakdown
- Risk mitigation strategies

## Target

**ACL 2026 Theme Track: Explainability of NLP Models**
- Deadline: January 5, 2026
- Theme questions addressed: Q4 (find/fix training data problems), Q2 (evaluate explanations by their utility)

## Constraints

- **Compute budget:** < 50 H100 hours total
- **Timeline:** 7 days (coding handled by AI agent; human time = ideation, experiment design, writing)
- **Scope:** Algorithm + empirical validation (must beat baselines like Cleanlab)

---

## The Idea

### One-line pitch

Use LLM-generated explanations to build an **instance graph of explanation agreement/contradiction**, and flag nodes that are explanation-inconsistent as likely label errors or artifact-driven examples.

### Research question

Can explanation structure (not just loss/confidence) be used to identify mislabeled or artifact-laden training data quickly, and does cleaning those points improve both accuracy and explanation faithfulness?

### The problem

1. **Label noise hurts models:** Mislabeled training examples degrade model quality.

2. **Existing methods use loss/confidence:** Cleanlab, confident learning, and similar methods flag high-loss or low-confidence examples. But these can miss examples where the model is confidently wrong due to spurious correlations.

3. **Explanations reveal different signals:** An example might have low loss but a suspicious explanation — e.g., it cites a spurious token, or its explanation contradicts the label.

4. **No graph-based explanation method exists:** Prior work on explanation-based debugging treats examples independently. Aggregating across examples via a graph could surface patterns invisible at the instance level.

### Proposed approach

A **graphical consistency** method driven by explanations:

1. **Generate structured explanations:** For each training instance i, generate an explanation e_i (e.g., "because [span], the label is [y]").

2. **Build explanation graph:**
   - Embed explanations using a sentence encoder
   - Connect kNN edges by explanation similarity
   - Nodes = training instances, edges = explanation similarity

3. **Define inconsistency signals:**
   - **Neighborhood label inconsistency:** Same explanation cluster, different labels → suspicious
   - **Explanation–label contradiction:** NLI check — does e_i entail label y_i? If not → suspicious
   - **Artifact signature:** Explanations that overly focus on known spurious tokens (dataset-specific)

4. **Rank and clean:**
   - Combine signals into a suspiciousness score
   - Remove or relabel top-K suspicious points
   - Retrain model; measure improvement

### Prior art identified

- **Label-noise detection:** Cleanlab/confident learning (loss-based), influence functions, LLM confidence for label errors (arXiv 2410.18889)
- **Explanation-based debugging:** TACL survey on explanation-based human debugging, interactive label cleaning with explanations (NeurIPS 2021)
- **Spurious correlations:** ACL paper on identifying spurious correlations via interpretability

### Novelty claim

"Explanation-consistency graphs as the core algorithmic primitive for label-noise detection in NLP" did not surface as an established approach. Prior explanation-based debugging is single-instance; graph aggregation is new.

---

## What I Need You to Produce

**Write a complete research proposal** that I can implement in 7 days. The proposal should be structured as follows:

### 1. Minimal viable experiment

What's the simplest version that demonstrates the core claim?
- One dataset or multiple?
- Synthetic noise injection only, or also real label errors?
- How many training examples needed? (10k? 50k? 100k?)
- What model to train/retrain? (Small BERT-class? LoRA on larger model?)

### 2. Explanation generation

- What LLM to use for generating explanations? (Open-weight 7-8B model?)
- What prompt format? Structured ("because [X], the label is [Y]") or free-form?
- How many tokens per explanation?
- Generate for all training examples or sample?

### 3. Graph construction (exact algorithm)

- What embedding model for explanations? (Sentence-BERT? Instructor? OpenAI embeddings?)
- kNN with what k? (5? 10? 20?)
- Distance metric? (Cosine? Euclidean?)
- Edge weighting? (Binary? Continuous similarity?)
- Any graph preprocessing? (Prune weak edges? Connected components?)

### 4. Inconsistency signals (concrete definitions)

**Neighborhood label inconsistency:**
- Formula: For node i with neighbors N(i), what metric captures "same explanation, different label"?
- Threshold for flagging?

**Explanation–label contradiction:**
- Which NLI model? (DeBERTa-NLI? RoBERTa-MNLI?)
- How to format the NLI input? (Explanation as premise, label statement as hypothesis?)
- What threshold for "contradiction"?

**Artifact signature:**
- How to detect "spurious token focus"?
- Need a predefined list of spurious tokens? Or detect automatically?
- How to compute "focus" from the explanation? (Token overlap? Keyword extraction?)

### 5. Combining signals

- How to combine the three signals into a single suspiciousness score?
- Weighted sum? Product? Learn weights?
- How to set thresholds?

### 6. Selection and cleaning

- Top-K selection: what K? Percentage (top 1%? 5%?) or absolute number?
- What to do with flagged points: remove entirely? Relabel with LLM? Human review simulation?
- For synthetic noise: we know ground truth, so can evaluate directly
- For real noise: need proxy evaluation

### 7. Baselines (must beat)

Specify exactly how to implement each:
- **Cleanlab:** Use cleanlab library with default settings?
- **High-loss filtering:** What loss threshold? Top-K by loss?
- **LLM confidence:** Prompt LLM to rate confidence in label; flag low-confidence
- **Influence functions:** Use TracIn or skip if too expensive?
- **Random baseline:** Random selection of K points

### 8. Evaluation metrics

**For synthetic noise detection:**
- Precision@K, Recall@K for detecting injected label flips
- AUROC for the suspiciousness score
- At what noise levels? (5%? 10%? 20%?)

**For spurious correlation detection:**
- Inject a spurious token correlated with label
- Detection rate: does the method flag spurious-reliant examples?

**Downstream after cleaning:**
- Accuracy on clean test set
- Accuracy on OOD/challenge sets (if available)
- Change in explanation quality (if measurable)

### 9. Dataset selection

Which dataset(s) give the cleanest signal?
- **SNLI/MNLI:** Known annotation artifacts (hypothesis-only bias)
- **FEVER:** Evidence mismatch issues
- **Toxicity (e.g., Civil Comments):** Known spurious correlations with identity terms
- **Synthetic:** Inject noise into a clean dataset (SST-2? AGNews?)

Recommend 1–2 datasets for the main experiments.

### 10. Expected findings

What specific results would make this publishable at ACL 2026?
- "Explanation-graph beats Cleanlab by X% on Precision@K"
- "Finds artifact-reliant examples that loss-based methods miss"
- "Downstream accuracy improves by Y% after cleaning"
- "Explanation quality (sufficiency/comprehensiveness) improves after cleaning"

### 11. Experiment protocol

Step-by-step procedure:
1. Dataset preparation + noise injection
2. Initial model training
3. Explanation generation
4. Graph construction
5. Signal computation
6. Ranking and selection
7. Cleaning (remove/relabel)
8. Retrain and evaluate
9. Compare to baselines

### 12. Compute budget breakdown

Estimate H100 hours for each step. Must stay under 50 hours total.
- Explanation generation is likely the bottleneck

### 13. Risk mitigation

What could go wrong? How do we handle:
- Explanation generation is too slow/expensive
- Graph doesn't cluster meaningfully
- Method doesn't beat Cleanlab
- Noise detection works but downstream improvement is small
