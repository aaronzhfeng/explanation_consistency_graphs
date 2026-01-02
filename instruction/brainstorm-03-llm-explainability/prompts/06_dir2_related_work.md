# Refinement Prompt: Dir 2 Related Work Deep Dive

## Context

We are implementing "Direction 2: Vacuity-aware robustness–informativeness frontier" for ACL 2026.

**Core idea:** A two-axis evaluation protocol for explanation quality that explicitly prevents vacuity gaming.

## Questions

1. **Explanation evaluation metrics (comprehensive survey):** What are ALL the major metrics used to evaluate NLP explanations? Organize by:
   - Token-level attributions (saliency, attention)
   - Extractive rationales (spans, sentences)
   - Free-text rationales / CoT
   - For each: what does it measure, what are its known failure modes?

2. **Stability/robustness metrics for explanations:** What prior work measures explanation stability under perturbations?
   - Synonym substitution attacks (Yin et al., Zhao et al.)
   - Paraphrase robustness
   - Input invariance frameworks
   - What gaps remain?

3. **Informativeness/faithfulness metrics:** What metrics measure whether an explanation is "useful" or "faithful"?
   - REV, RORA
   - Sufficiency, comprehensiveness (ERASER)
   - Simulatability
   - Counterfactual faithfulness
   - What's the relationship between these?

4. **Vacuity problem in explanations:** Is there prior work explicitly discussing "vacuous" or "generic" explanations?
   - Plausibility vs faithfulness discussions
   - Post-hoc rationalization critiques
   - Any explicit "vacuity" metrics or penalties?

5. **Combined/multi-axis evaluation:** Does any prior work combine stability + informativeness?
   - If so, how? What's missing?
   - If not, this is our novelty claim.

6. **Goodharting in explanation benchmarks:** What work discusses gaming of explanation metrics?
   - The EACL 2024 paper on Goodhart's Law for explanation benchmarks
   - Other critiques?

7. **Positioning statement:** Given this literature, write a 2–3 sentence novelty claim that is defensible.

