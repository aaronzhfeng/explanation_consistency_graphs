# X-MET: Novelty-Checked Proposal from GPT-5.2 Pro

**Source:** GPT-5.2 Pro  
**Date:** 2025-01-30  
**Follow-up Prompt:** Novelty audit of EXPLAINCHECK proposal

---

## Closest prior art found

### 1) Stability of feature-attribution via synonym/adversarial substitutions
- **Yin et al. (ACL 2022):** stability under synonym substitutions, rank differences
- **Zhao et al. (CCL 2022):** textual adversarial attacks for stability evaluation

**Implication:** "explanation stability under small meaning-preserving word substitutions for attribution methods" is substantially covered.

### 2) General explanation invariance/equivariance under symmetry groups
- **Crabbé & van der Schaar (NeurIPS 2023):** formalize invariance/equivariance for explicit symmetry groups

**Implication:** formal framing exists, but NLP transformations are not clean group symmetries.

### 3) Methods that *train* for stable/robust explanations
- **Chen et al. (NeurIPS 2024):** training methods for robust explanations
- **La Malfa et al. (IJCAI 2021):** robust explanations with guarantees (OREs)

### 4) NL explanation evaluation includes paraphrase robustness
- **LExT (2025):** paraphrase robustness for generated explanations

**Implication:** "paraphrase robustness for generated explanations" is not empty space.

### 5) Metamorphic testing exists for model behavior, not explanations
- **CheckList (Ribeiro et al., ACL 2020):** invariance tests for predictions
- **Ma et al. (IJCAI 2020):** metamorphic testing for fairness

---

## What appears NOT to exist yet (clear gap)

1. **Unified metamorphic test suite** going beyond synonym-level attacks to paraphrase, negation, entity substitution, role swap, discourse distractors
2. **Multiple explanation modalities** (attributions, rationales, attention, free-text/CoT)
3. **Expected explanation mappings** (equivariance / sign-flip laws)
4. **Oracle-free stability scoring** (no gold explanation needed)
5. **Alignment-aware stability metrics** for paraphrases (not trivial token alignment)
6. **Stability violations as falsification tests** for faithfulness

---

## Revised Proposal

### Working title

**X-MET: Metamorphic Explanation Tests for NLP Models**

---

## 1) Core claim (tightened for defensible novelty)

**Metamorphic relation:**
For transformation τ with expected constraint on prediction:
- If f(τ(x)) ≈ M_τ(f(x)) then E(τ(x)) ≈ g_τ(E(x))

Where M_τ is expected effect on predictions and g_τ is expected effect on explanations.

- **Violation** = red flag for faithfulness
- **Pass** = not proof of faithfulness, but valuable **unit test**

Explicitly extends from single perturbation families to **test suite with laws**.

---

## 2) Transformation taxonomy with explicit metamorphic "laws"

### A. Invariance transforms

1. **Meaning-preserving paraphrase**
2. **Irrelevant insertion / distractor sentence**
3. **Formatting / punctuation / casing**

### B. Equivariance transforms

4. **Entity substitution with bijection map**
5. **Active↔passive / role-preserving syntactic alternations**
6. **Order swap for symmetric relations**

### C. Directional / contrast transforms

7. **Negation injection / removal**
8. **Entailment direction swap (NLI-specific)**

---

## 3) Explanation methods to test (multi-modality for novelty)

### Attribution / saliency
- Gradient×Input, Integrated Gradients, DeepLIFT, Occlusion, LIME

### Attention-as-explanation
- Raw attention, attention rollout/flow

### Extractive rationales
- Select-then-predict models, top-k tokens

### Free-text explanations / CoT
- Prompted explanations with/without CoT
- "Cited rationales" variants (quote exact spans)

---

## 4) Stability metrics (alignment-aware)

### A. Prediction compliance (gating)
- **Metamorphic compliance rate:** condition stability on prediction compliance

### B. Explanation stability for token-level vectors

**Key novelty: paraphrase alignment**
1. Token/phrase alignment using embedding-based aligner
2. Mass-transport / aligned correlation:
   - Spearman/Pearson after alignment
   - Top-k IoU after mapping
   - Earth Mover's Distance (optional)

### C. Rationale stability (binary spans)
- Span IoU / F1 after alignment
- Insertion invariance (overlap excluding distractor spans)

### D. Free-text / CoT stability

**Oracle-free bidirectional entailment:**
- NLI(e1 → e2) and NLI(e2 → e1)
- Semantic equivalence score = min(P(e1 ⇒ e2), P(e2 ⇒ e1))
- For negation/flip: expect contradiction alignment

### E. Worst-case instability
- Max instability per family
- Max across all families

---

## 5) Tasks and datasets

1. **Sentiment / topic classification:** SST-2, AGNews
2. **NLI:** MNLI (or e-SNLI)
3. **Paraphrase identification:** QQP / MRPC
4. **Fact verification:** FEVER-style

---

## 6) Transformation generation

### Paraphrases
- Generator: instruction-tuned LLM or paraphrase model
- Filters: semantic similarity + label-preservation check

### Negation transforms
- Targeted templates with NLI contradiction filter

### Entity substitution
- NER → same-type substitution with consistency

### Order swap
- Deterministic for pair-input tasks

### Irrelevant insertion
- Controlled distractor pool with embedding distance filter

---

## 7) Expected findings

- Token attributions show **non-trivial instability under paraphrase-level transforms**
- Attention-as-explanation likely fails equivariance more often than attributions
- Free-text CoT may look stable while **not tightly coupled to model evidence**
- Worst-case instability is a strong differentiator

---

## 8) Practitioner implications

1. **CI-style unit tests** for explanation pipelines (like CheckList for predictions)
2. **Falsification tool:** unstable explanation = unreliable for auditing
3. **Spurious-correlation/bias smoke test**

---

## 9) Compute estimate (< 100 H100 hours)

- Transformation generation + filtering: **10h**
- Attribution/attention/rationale evaluation: **35h**
- LLM free-text explanations: **35h**
- Alignment + scoring: **10h**
- Spurious-feature stress test (optional): **10h**

**Total: ~100 H100h**

---

## Novelty sentence for paper

> Prior NLP work on explanation stability largely evaluates feature-attribution methods under synonym-level adversarial substitutions that keep model outputs fixed. Our work generalizes stability into a **metamorphic test suite** spanning invariance, equivariance, and directional (flip) relations across **multiple linguistic transformation families** and **multiple explanation modalities** (attributions, rationales, attention, and free-text explanations), using **alignment-aware** metrics and largely **oracle-free** evaluation.

This explicitly differentiates from:
- Yin/Zhao (synonym/adversarial substitution focus)
- Crabbé (symmetry-group formalism)
- LExT (ground-truth anchored NL explanation metrics)

---

## References

[1]: https://aclanthology.org/2022.ccl-1.82.pdf "Zhao et al. CCL 2022"
[2]: https://www.ijcai.org/proceedings/2021/0366.pdf "La Malfa et al. IJCAI 2021 - OREs"
[3]: https://arxiv.org/html/2504.06227v1 "LExT 2025"
[4]: https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf "CheckList ACL 2020"
[5]: https://www.ijcai.org/proceedings/2020/0064.pdf "Ma et al. IJCAI 2020 - Metamorphic Fairness Testing"
[6]: https://direct.mit.edu/coli/article/50/2/657/119158 "Faithful Model Explanation in NLP Survey"

