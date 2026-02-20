# MNLI Underperformance Analysis

## Observation

ECG (Expl. kNN) scores near random on MultiNLI (.557 Uni, .557 Art), while:
- LLM Mismatch scores .883 on both MNLI conditions
- Input kNN scores .523 on both MNLI conditions (also near random)
- Training-based methods (Cleanlab .936, AUM .928) dominate MNLI Uni
- On SST-2 Art, ECG leads at .819 while Cleanlab collapses to .136

## The Pipeline

```
LLM inference → pred_label       → LLM Mismatch (AUROC .883 on MNLI)
             → explanation text  → embed → kNN graph → ECG (AUROC .557 on MNLI)
```

Both signals come from the same single LLM pass. LLM Mismatch is essentially free when running ECG.

## Why Does ECG Fail on MNLI Despite LLM Mismatch Working?

The kNN step must add value *beyond* the raw LLM prediction for ECG to win. On MNLI it does not, for three reasons:

1. **NLI explanation texts do not cluster by label.** For sentiment, evidence like "great acting, compelling story" is inherently label-discriminative — it only appears in positive explanations. For NLI, explanation text like "the premise describes a location while the hypothesis makes a temporal claim" can plausibly appear under NEUTRAL *or* CONTRADICTION. The sentence-transformer embedding space does not separate NLI labels as cleanly as it separates sentiment labels.

2. **Input kNN also fails on MNLI (.523 ≈ random).** The kNN failure is not specific to explanations — raw input embeddings also fail to cluster by NLI label. ECG inherits this structural problem. The issue is that premise-hypothesis pair similarity in embedding space does not correlate with label identity for NLI.

3. **Three-class dilution.** With 3 classes (ENTAILMENT / NEUTRAL / CONTRADICTION), a mislabeled example's neighborhood is already mixed across labels in clean data, making the disagreement signal noisier than in binary classification.

## Why Does LLM Mismatch Work on MNLI?

LLM Mismatch bypasses the embedding step entirely. It directly compares `pred_label` (from JSON output) against the given training label. Since Qwen3-8B achieves ~88% zero-shot NLI accuracy, a mislabeled example (e.g., true CONTRADICTION labeled ENTAILMENT) will almost always receive the correct predicted label, flagging the discrepancy directly.

## Implications for the Rebuttal

- ECG's value is **conditional on the embedding space being label-discriminative**. This holds for SST-2 (semantic sentiment signal) but not for MNLI (complex relational reasoning).
- The MNLI failure is **mechanistically predictable**, not arbitrary — it follows from both (a) high LLM prediction accuracy and (b) label-indiscriminative embedding structure.
- **ECG and LLM Mismatch are complementary**: ECG handles artifact-aligned regimes where LLM prediction accuracy alone is insufficient; LLM Mismatch handles high-LLM-accuracy regimes. Both are produced by the same LLM call.
- The current framing in the response ("regime-specific") is honest but undersells this complementarity.

## Questions for Assessment

1. Is the mechanistic explanation (NLI explanation embeddings not clustering by label) convincing to a reviewer? Is there a cleaner way to frame it?
2. Does framing ECG + LLM Mismatch as a complementary pair strengthen the overall contribution claim?
3. Does the MNLI near-random result fundamentally undermine the generalizability argument, or is the regime-specificity framing sufficient?
4. Should the rebuttal explicitly note that Input kNN also fails on MNLI (as supporting evidence that the problem is structural, not specific to ECG)?
