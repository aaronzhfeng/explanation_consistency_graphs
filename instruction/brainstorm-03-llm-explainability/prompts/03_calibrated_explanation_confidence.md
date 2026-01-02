You are a senior NLP researcher helping design a research paper for ACL 2026's Theme Track on "Explainability of NLP Models."

## Target Venue

**ACL 2026 Theme Track: Explainability of NLP Models**
- Deadline: January 5, 2026 (ARR submission)
- Conference: July 2-7, 2026, San Diego

## Motivation

Uncertainty quantification for LLMs typically focuses on:
- Predictive uncertainty (is the final answer reliable?)
- Token-level uncertainty (next-token probabilities)

But we lack:
- **Explanation-level uncertainty** — Is this reasoning step reliable?
- **Faithfulness uncertainty** — Does this explanation reflect the model's actual process?
- **Step-level calibration** — Can I trust this intermediate derivation?

Recent work suggests this is feasible: a predictor trained on solver feedback achieved ECE = 0.08 for step-level reasoning validity. Can we generalize this to broader NLP explanations?

## Research Questions

1. Can we train calibrated explanation-confidence predictors for diverse NLP tasks?
2. What signals (internal representations, consistency, etc.) predict explanation quality?
3. How should we use explanation confidence in downstream applications?
4. Does explanation confidence correlate with prediction correctness?

## ACL 2026 Theme Alignment

> "Can we identify specific directions, mechanisms, patterns, or 'knobs' within a model's internal activations that control high-level behaviors?"

## Constraints

- **Compute budget:** < 100 H100 hours total
- **Timeline:** ~7 days (coding handled by AI agent; human time = ideation, prompt tuning, experiment design, writing)
- **No pretraining:** Must work with existing models (fine-tuning OK)
- **Feasibility focus:** Prioritize probe-based or post-hoc methods over training-heavy approaches

## Task

Propose a research paper on calibrated explanation confidence. Specify:

1. **Core hypothesis** — What do you expect to find?
2. **Tasks and datasets** — Where will you evaluate?
3. **Explanation methods to study** — CoT, attention, saliency, rationales?
4. **Confidence estimation approaches** — How to estimate explanation quality?
   - Internal signals (hidden states, attention entropy, etc.)
   - Consistency-based (self-consistency, perturbation stability)
   - Learned predictors
5. **Ground truth for training** — What's the signal for "good" vs "bad" explanation?
6. **Evaluation metrics** — ECE, Brier score, selective prediction, etc.
7. **Baselines** — What to compare against?
8. **Expected contributions** — What's novel?
9. **Compute estimate** — H100 hours breakdown

Be specific about the training signal and evaluation protocol.
