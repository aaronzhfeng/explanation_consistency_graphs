# EXPLAINCHECK Proposal from GPT-5.2 Pro

**Source:** GPT-5.2 Pro  
**Date:** 2025-01-30  
**Prompt:** `04_stability_testing_nlp.md`

---

## Proposed paper: EXPLAINCHECK — Metamorphic Stability Tests for Faithful NLP Explanations

### 1. Core claim — why stability matters for faithfulness

**Claim (one-sided):** *Explanation instability under meaning-preserving / role-preserving transformations is strong evidence against faithfulness.*

For transformations τ that preserve decision-relevant semantics:
- **Prediction invariance:** f(τ(x)) ≈ f(x)
- **Explanation invariance/equivariance:** E(τ(x)) ≈ g_τ(E(x))

**Important nuance:**
- Stability is **not sufficient** for faithfulness (explainer could be stable but generic)
- But **instability is a necessary-test failure**: if f is invariant yet E wildly changes, E is unlikely tracking the actual decision boundary

**Positioning:** "Behavioral/metamorphic testing" for explanations — analogous to CheckList-style testing for model behavior.

---

### 2. Transformation taxonomy

#### A. Invariance transforms (meaning preserved; prediction should stay; explanation should stay)

1. **Paraphrase invariance**
2. **Irrelevant insertion / distractor addition**
3. **Surface-form invariances** (casing, punctuation, whitespace)

#### B. Equivariance transforms (semantics preserved but roles/tokens permuted)

4. **Entity substitution (type-preserving)**
5. **Order permutation / role swap**

#### C. Contrast transforms (meaning changes; prediction should flip)

6. **Negation sensitivity (controlled label flip)**
7. **Entity mismatch for evidence-based tasks**

---

### 3. Explanation methods to test

#### A. Token-level attribution (post-hoc)
- Gradient-based: Saliency, Gradient×Input, Integrated Gradients, SmoothGrad
- Perturbation/surrogate: LIME, SHAP

#### B. Attention-derived explanations
- Raw attention weights, attention rollout, last-layer CLS→token attention

#### C. Extractive rationales
- Human-annotated rationales datasets
- Model-produced rationales via top-k tokens

#### D. Free-text explanations / "CoT"
- Short justification (2–4 sentences)
- Step-by-step reasoning

---

### 4. Metrics

#### 4.1 Prediction consistency under transforms
- Invariant transform: pass rate
- Contrast transform: expected label mapping

#### 4.2 Explanation agreement for attribution vectors
- **Alignment:** SimAlign for paraphrases
- **Metrics:**
  - Rank correlation (Spearman)
  - Top-k overlap (Jaccard)
  - Distribution divergence (1 - JSD)
- **Negation-specific flip metric** (signed attributions)

#### 4.3 Agreement for rationales (discrete spans)
- IoU / F1 overlap
- Sufficiency and comprehensiveness (ERASER-style)

#### 4.4 Agreement for free-text explanations
- Semantic similarity: BERTScore, BLEURT
- NLI-based equivalence/contradiction tests

#### 4.5 Worst-case instability
- Max instability across transforms
- Tail risk (95th percentile)

#### 4.6 Stability ↔ faithfulness correlation
- Spearman correlation
- AUROC of predicting low-faithfulness from instability

---

### 5. Tasks and datasets

1. **Single-sentence classification:** SST-2, HateXplain
2. **Sentence-pair inference:** e-SNLI, MNLI/SNLI
3. **Evidence-based reasoning:** FEVER, MultiRC, BoolQ

**Feasible minimal suite:** SST-2 + HateXplain + FEVER (or e-SNLI)

---

### 6. Transformation generation pipeline

**General pattern:** generate → filter → audit

#### 6.1 Paraphrase invariance
- Backtranslation + paraphrase model
- Filters: SBERT cosine ≥ 0.85, NLI entailment both directions ≥ 0.8

#### 6.2 Negation sensitivity
- Rule + parse-based negation insertion
- Filters: NLI contradiction check

#### 6.3 Entity substitution
- NER + same-type replacement
- Filters: SBERT cosine ≥ 0.8

#### 6.4 Order permutation / role swap
- Deterministic rewrite

#### 6.5 Irrelevant insertion
- Neutral distractor bank
- Filters: SBERT similarity ≤ 0.3

#### 6.6 Human audit
- ~100 samples per transform type

---

### 7. Expected findings (testable hypotheses)

- **H1:** Gradient/IG show substantial instability under paraphrase
- **H2:** Attention-based explanations have low stability–faithfulness correlation
- **H3:** SHAP/LIME more stable but expensive
- **H4:** Free-text explanations stable but not necessarily faithful
- **H5:** Stability detects spurious reliance imperfectly

---

### 8. Implications for practitioners

1. Use stability tests as **gatekeeper** ("necessary test")
2. Use stability as **risk indicator** for spurious correlations
3. Stability should **complement** (not replace) faithfulness metrics
4. Benchmark-driven selection of explanation method

---

### 9. Compute estimate

- Transformation generation + filtering: 8–12h + 4–6h
- Model inference + explanations (encoders): 2h + 4–6h + 18–25h + 15–25h
- Free-text explanations: 20–30h
- Scoring + alignment: 3–6h

**Total: ~70–110 H100 hours**

Easy knobs to stay <100h:
- Reduce LIME/SHAP subset size
- Reduce free-text explanation sample size

