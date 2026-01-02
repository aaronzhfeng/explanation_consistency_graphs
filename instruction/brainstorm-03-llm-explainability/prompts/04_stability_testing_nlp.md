You are a senior NLP researcher helping design a research paper for ACL 2026's Theme Track on "Explainability of NLP Models."

## Target Venue

**ACL 2026 Theme Track: Explainability of NLP Models**
- Deadline: January 5, 2026 (ARR submission)
- Conference: July 2-7, 2026, San Diego

## Core Idea: Stability as a Faithfulness Signal

A principle from robust ML: if predictions should be invariant to certain transformations, we can test this as a "unit test" for model quality.

Applied to NLP explanations:
- If input is paraphrased (meaning-preserving), explanation should be consistent
- If input order is swapped (A entails B → B is entailed by A), explanation should adapt predictably
- If irrelevant details change, explanation should be stable
- If negation is applied, explanation should flip appropriately

**Hypothesis:** Unstable explanations are unlikely to be faithful.

## Existing Work (Limited)

- Input invariance for saliency (Kindermans et al. 2017)
- Robustness of interpretability methods (Alvarez-Melis & Jaakkola 2018)
- Lipschitz-style stability bounds (limited application to NLP)

But no systematic "invariance test suite" for NLP explanations exists.

## ACL 2026 Theme Alignment

> "How can we rigorously evaluate the quality of an explanation?"

And implicitly:
> "Can explanations be used to reliably detect when a model is making a biased prediction?"

(Unstable explanations may indicate reliance on spurious correlations)

## Constraints

- **Compute budget:** < 100 H100 hours total
- **Timeline:** ~7 days (coding handled by AI agent; human time = ideation, prompt tuning, experiment design, writing)
- **Scope:** Evaluation framework + empirical study (not a new explanation method)
- **Feasibility focus:** Use existing paraphrase/perturbation tools, standard models, off-the-shelf explanation methods

## Task

Propose a research paper on explanation stability testing for NLP. Specify:

1. **Core claim** — Why stability matters for faithfulness
2. **Transformation taxonomy** — What transformations to test?
   - Paraphrase invariance
   - Negation sensitivity (explanation should flip)
   - Entity substitution (explanation should adapt)
   - Order permutation (for symmetric relations)
   - Irrelevant insertion (explanation should ignore)
3. **Explanation methods to test** — CoT, attention, saliency, rationales
4. **Metrics** — How to quantify stability?
   - Explanation agreement (IoU, correlation, semantic similarity)
   - Prediction consistency under transforms
   - Worst-case instability
5. **Tasks and datasets** — Where to evaluate?
6. **How to generate transformations** — Automated? Model-based? Human?
7. **Expected findings** — Which methods are stable? Which aren't?
8. **Implications** — How should practitioners use stability tests?
9. **Compute estimate** — H100 hours breakdown

Be specific about transformation generation and explanation similarity measurement.
