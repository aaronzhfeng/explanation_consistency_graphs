# Research Proposal Request: Vacuity-Aware Robustness-Informativeness Frontier for NLP Explanations

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
- Theme questions addressed: Q2 (rigorous evaluation of explanation quality)

## Constraints

- **Compute budget:** < 25 H100 hours total
- **Timeline:** 7 days (coding handled by AI agent; human time = ideation, experiment design, writing)
- **Scope:** Evaluation protocol + empirical study (not a new explanation method)

---

## The Idea

### One-line pitch

Create an evaluation protocol that scores explanations by **(i) robustness/stability** AND **(ii) non-vacuous informativeness**, explicitly preventing the "stable-but-empty explanation" failure mode.

### Research question

How do we evaluate explanation quality when **stability metrics can be gamed by vacuous explanations**, and informativeness metrics can be gamed by label leakage?

### The problem

1. **Stability alone is gameable:** An explanation that always says "The answer follows from the text" is perfectly stable under paraphrases — but useless.

2. **Informativeness alone is gameable:** An explanation that leaks the label ("This is positive because the sentiment is positive") scores high on some informativeness metrics — but is circular.

3. **Current evaluation is fragmented:** Stability and informativeness are measured separately. No standard protocol explicitly prevents both failure modes.

### Proposed approach

Define a **two-axis evaluation** and a **combined score** that cannot be optimized by vacuity:

1. **Robustness axis:** Explanation similarity under meaning-preserving input perturbations (paraphrase, entity renaming, minor edits).

2. **Informativeness axis:** Use information-theoretic approaches like REV (measures mutual information between rationale and label) or leakage-aware evaluation like RORA.

3. **Combined score:** Options include:
   - Pareto frontier reporting (plot robustness vs informativeness)
   - "Frontier area" or harmonic mean after normalization
   - Explicit vacuity penalty (e.g., minimum information threshold)

4. **Adversarial rationale suite:** Release a set of "adversarial" rationales designed to stress-test metrics:
   - Vacuous rationales ("The answer follows from the evidence")
   - Label-leaking rationales ("This is entailment because premise entails hypothesis")
   - Superficially stable but semantically drifting rationales

### Prior art identified

- **Stability frameworks:** Robustness evaluation for feature attribution (arXiv 2209.01782)
- **Informativeness metrics:** REV (arXiv 2210.04982), RORA (arXiv 2402.18678)
- **Rationale benchmarks:** ERASER benchmark (sufficiency/comprehensiveness)
- **Vacuity pitfall:** NeurIPS paper "On the (In)fidelity and Sensitivity of Explanations" notes sensitivity metrics can prefer trivial explanations

### Novelty claim

"Vacuity-proof combined frontier + adversarial rationale suite" did not appear as an already-standardized protocol in literature scans. Robustness and informativeness exist separately, but not as a unified protocol with explicit anti-gaming provisions.

---

## What I Need You to Produce

**Write a complete research proposal** that I can implement in 7 days. The proposal should be structured as follows:

### 1. Datasets

- Which 2–3 datasets give the cleanest signal for demonstrating the vacuity problem?
- Consider: ERASER-style rationale datasets (e-SNLI, MultiRC, FEVER, BoolQ, Movies), LLM-generated rationales on standard NLP tasks
- What's the minimum scale needed? (1k? 5k? 10k examples per dataset?)
- Licensing considerations?

### 2. Explanation types to evaluate

- Which explanation formats should we test?
  - Extractive rationales (token/sentence highlights)
  - Free-text rationales (LLM-generated)
  - CoT-style step explanations
- How do we generate explanations for each dataset?

### 3. Metrics (exact formulas)

For each axis, provide concrete formulas:

**Robustness axis:**
- What similarity measure between explanations? (Jaccard for spans? BERTScore for text? Spearman for attributions?)
- Under what perturbations? (paraphrase, entity swap, formatting noise?)
- How to generate perturbations cheaply?

**Informativeness axis:**
- REV-style mutual information? RORA-style leakage control? Sufficiency/comprehensiveness? Something simpler?
- Exact computation procedure

**Combined score:**
- How to combine robustness and informativeness?
- How to operationally detect and penalize vacuity?
- Should we use a threshold, a penalty term, or just report the frontier?

### 4. Adversarial rationale suite

What specific "adversarial" rationales should we construct?
- **Vacuous:** Generic explanations that say nothing task-specific
- **Label-leaking:** Explanations that reveal the answer without justification
- **Semantically drifting:** Explanations that change meaning while maintaining surface similarity
- How many examples of each? How to generate them?

### 5. Baselines

What existing metrics/methods must we compare against?
- Single-axis metrics (stability-only, informativeness-only)
- Length/overlap heuristics
- Human plausibility ratings (at what scale?)
- Any other combined metrics from prior work?

### 6. Expected findings

What specific empirical patterns would make this publishable at ACL 2026?
- "Method X scores high on stability but fails informativeness"
- "Vacuous rationales game metric Y but not our combined score"
- "Human preference correlates with our combined score but not with X alone"
- What correlations should we show or disprove?

### 7. Experiment protocol

Step-by-step procedure:
1. Data preparation
2. Explanation generation
3. Perturbation generation
4. Metric computation
5. Analysis and visualization

### 8. Deliverables

What artifacts make this a complete benchmark contribution?
- Code release (evaluation harness)?
- Dataset release (adversarial suite)?
- Pre-computed scores for baselines?
- Leaderboard or just tables?

### 9. Compute budget breakdown

Estimate H100 hours for each step. Must stay under 25 hours total.

### 10. Risk mitigation

What could go wrong? How do we handle:
- Perturbation generation fails (paraphrases change meaning)
- Metrics don't differentiate vacuous from good explanations
- Compute overruns
