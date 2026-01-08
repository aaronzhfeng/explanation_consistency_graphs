# Paper Revision Plan

> Based on consultation with Claude 5.2 Pro on January 3, 2026
> This document provides the definitive revision strategy for the ECG paper

---

## Executive Summary

The current paper draft claims "five complementary signals + adaptive aggregation + graph propagation" but experimental results show **Explanation kNN alone is best** and aggregation hurts. This is not a polish revision—it's a **re-grounding of the paper around what actually works**.

---

## 1. Core Contribution (Final Decision)

### Primary Contribution
**Explanation embeddings as a representation for data quality + simple neighborhood inconsistency in that space.**

Key claim:
- Explanations provide an embedding space that captures **"why this label"**
- **kNN disagreement in that space** is a strong detector of artifact-aligned label issues
- Yields large empirical gap vs input embeddings (0.832 vs 0.671 AUROC)

### Secondary Contribution
**A concrete failure mode for confidence-/loss-based cleaning under artifact-aligned noise** (Cleanlab catastrophic failure at 0.107 AUROC) and a method that remains effective.

### What is NOT the Contribution
- ❌ The 5-signal framework
- ❌ Adaptive aggregation
- ❌ Graph propagation

**One sentence summary**: *"The representation is the contribution; the rest is an explored extension that failed."*

---

## 2. Multi-Signal Story (Final Decision)

### Approach: Ablation-Driven Negative Result + Diagnosis

**Do not omit entirely. Do not sell as the method.**

Best framing:
> "We investigated augmenting explanation-neighborhood detection with additional verification signals; surprisingly, naive aggregation degraded performance under artifact noise."

Then provide the **mechanism**:
- AUM is anti-correlated under artifacts because artifacts make wrong labels easy
- Dynamics signal reduces suspicion for exactly the examples we want to flag

### Where to Put It
- Most multi-signal machinery goes in **Appendix** or clearly labeled "Attempted Extensions" section
- Brief mention in main paper as negative result

### What to Avoid
- ❌ "Future work" handwaving without numbers
- ❌ Any language implying the full pipeline is what wins

---

## 3. Architecture Diagram (Final Decision)

### Main Figure: Simplified Winning Path Only

```
Input → LLM Structured Explanation → Explanation Embedding → kNN Graph → Neighborhood Surprise Score → Ranking
```

The "graph" still exists (kNN graph), but remove visual implication that 5 signals + aggregation are essential.

### Full Multi-Signal Diagram
Two options:
1. **Put in appendix** with label: "Explored multi-signal extensions (did not improve performance in our artifact setting)"
2. **Show in main paper as faded/dashed branch** labeled "extensions" with winning path highlighted

**Key principle**: Don't let the diagram contradict the results.

---

## 4. Positioning vs Cleanlab (Final Decision)

### Approach: Complement, Not Replacement

**Two-regime story** (make explicit and operational):

| Noise Regime | Recommended Method | Evidence |
|--------------|-------------------|----------|
| Random/unstructured noise | Cleanlab | AUROC 0.977 |
| Artifact-aligned / spurious-correlation-driven | Explanation kNN | AUROC 0.832 (Cleanlab: 0.107) |

### What NOT to Claim
- ❌ "We beat Cleanlab generally" — we don't
- ✅ "We beat Cleanlab in a specific (important) regime where it breaks"

### Practitioner-Facing Rule
> "If your model fits 'too well' and you suspect shortcuts, use explanation-space neighborhood detection."

---

## 5. Narrative Tone (Final Decision)

### Approach: Confident, Results-First + One Controlled Paragraph on Negative Result

**What works:**
- Lead with key insight + headline numbers
- Then: "We also evaluated a multi-signal extension; naive aggregation reduced accuracy due to regime-dependent anti-correlation; we analyze this."

**What doesn't work:**
- ❌ Long "journey" narrative that reads like apologizing
- ❌ Any statement implying multi-signal system is "the method" when it isn't

---

## 6. Title Options

### Pragmatic/Accurate
> "Detecting Artifact-Aligned Label Issues with Explanation Embedding Neighborhoods"

### Keep ECG Branding
> "Explanation-Consistency Graphs: Neighborhood Surprise in Explanation Embedding Space for Data Debugging"

**Warning**: If keeping "Graph-Aggregated" in subtitle, ensure you actually aggregate something beyond a kNN statistic—otherwise misleading.

---

## 7. Abstract Requirements

### Must Include
- Explanation embeddings + kNN neighborhood inconsistency is the method
- Artifact-noise robustness vs confidence-based cleaning
- Input-embedding kNN comparison (0.832 vs 0.671)
- Cleanlab's regime dependence (strong on random, fails on artifacts)

### Must Remove
- ❌ "five complementary signals"
- ❌ "adaptive aggregation improves"
- ❌ downstream +8% claims unless confirmed for winning method

---

## 8. Contributions (Bulletproof Version)

1. **Method**: A simple detection method based on **kNN neighborhood surprise in explanation embedding space** (call it Exp-kNN or ECG-S_nbr)

2. **Empirical Finding**: Explanation embedding neighborhoods substantially outperform input embedding neighborhoods under artifact-aligned noise (+0.161 AUROC)

3. **Failure-Mode Analysis**: Confidence-/loss-based detection (e.g., Cleanlab/AUM) can be **anti-informative** when spurious artifacts make wrong labels easy; show why

4. **Practical Positioning**: Guidance on when to use explanation-based neighborhood detection vs confident learning

5. **Optional** (if downstream experiment confirms): Downstream gains after cleaning with Exp-kNN

---

## 9. Section-by-Section Revision Guide

### Introduction
**Keep**: Motivation about spurious correlations and confident fitting (strong)

**Change**: 
- ❌ "we propose ECG with five signals + aggregation"
- ✅ "we propose using explanation embeddings as a substrate; simple neighborhood inconsistency is highly effective"

### Related Work
**Keep**: Most content (already supports explanation + graph/noisy-label threads)

**Adjust**: 
- ❌ Don't claim novelty is "multi-signal graph aggregation"
- ✅ Claim novelty is **operating in explanation embedding space** and demonstrating the regime where it matters

### Method (Refactored Structure)

| Section | Content | Status |
|---------|---------|--------|
| 3.1 | Explanation Generation | Keep concise |
| 3.2 | Explanation Embedding + kNN Retrieval | NEW: Main method |
| 3.3 | Neighborhood Surprise Score | This is THE method |
| 3.4 | Reliability Weighting | Optional (only if helps or neutral) |
| 3.5 | Attempted Multi-Signal Extensions | Clearly separated; can move to appendix |

### Results

**Must include:**
- Artifact-noise main table with **Explanation kNN** as headline
- Baselines: Input kNN, LLM Mismatch, Cleanlab, Loss-based, Margin/AUM
- Table/figure showing **multi-signal hurts** (ECG adaptive/fixed vs Exp-kNN)

**For random noise:**
- ⚠️ **Cannot leave Exp-kNN as TBD** — must run and fill table
- Even if Exp-kNN worse than Cleanlab, that's fine (supports complement positioning)

### Analysis (Major Selling Point)

Must explain:
1. **Why AUM flips sign under artifacts** — mechanism analysis
2. **Correlation plots** or rank agreement between signals and true noise indicator
3. **Qualitative examples** — nearest neighbors in explanation space for flagged examples (1-2 examples)

### Conclusion

One paragraph, no grand claims:
- Explanation embeddings enable robust detection under artifact-driven failures
- Signal aggregation is non-trivial; naive combinations can hurt

---

## 10. Missing Experiments (Priority Order)

### ✅ Priority 1: Explanation kNN on Random Noise
**Status**: COMPLETE (2026-01-03)

**Results**:
| Method | AUROC |
|--------|-------|
| Cleanlab | 0.977 |
| **Explanation kNN** | **0.943** |
| LLM Mismatch | 0.901 |
| Input kNN | 0.880 |

**Key finding**: Explanation kNN achieves 0.943 AUROC, only 3.4% behind Cleanlab!

**Saved to**: `outputs/results/20260103_143538_random_noise_results.json`

### ✅ Priority 2: Downstream Cleaning with Explanation kNN
**Status**: COMPLETE (2026-01-03)

**Results**:
| K | Precision | Accuracy Δ |
|---|-----------|------------|
| 1% | 66.8% | +0.00% |
| **2%** | **57.4%** | **+0.57%** |
| 5% | 40.6% | +0.23% |
| 10% | 29.7% | -0.57% |

**Key finding**: K=2% achieves best accuracy improvement (+0.57%)

**Saved to**: `outputs/results/20260103_135902_downstream_explknn.json`

### 🟡 Priority 3: Generalization Check
**Status**: RECOMMENDED

Risk: One dataset (SST-2) only is a risk at ACL unless track allows narrow scope.

Options (cheap but valuable):
- Another sentiment dataset
- Artifact-like setup in different task
- Multiple artifact types
- Multiple noise rates (5%, 20%)
- Sensitivity to explanation model / embedding model

---

## 11. How Reviewers React to "We Designed X but Found Y Works Better"

### They React BADLY If:
- It reads like covering up a failure
- Paper has no crisp contribution besides "we tried stuff"

### They React FINE (Sometimes Positively) If:
- Winning method is simple AND analysis explains something non-obvious
- You demonstrate a clear regime where standard methods fail
- You provide actionable guidance

### Our Positioning:
> "The key value is the representation (explanation embeddings), and naive multi-signal 'verification' can backfire under spurious-correlation regimes."

**This is a real insight, not a failure story.**

---

## 12. Bottom Line Decisions (Summary Table)

| Decision Point | Final Answer |
|----------------|--------------|
| **Core contribution** | Explanation embeddings + neighborhood inconsistency (Explanation kNN) |
| **Multi-signal story** | Keep as ablation/negative result + diagnosis (NOT the method) |
| **Architecture diagram** | Simplify main figure to winning path; move full multi-signal to appendix |
| **Positioning vs Cleanlab** | Complementary; show two regimes explicitly |
| **Narrative tone** | Confident about main result; transparent but brief about failed extension |

---

## 13. Checklist Before Revision

- [x] Run Explanation kNN on random noise experiment ✅ **(0.943 AUROC)**
- [x] Complete downstream evaluation with Explanation kNN ✅ **(+0.57% at K=2%)**
- [ ] Create simplified architecture diagram
- [ ] Rewrite Abstract with new framing
- [ ] Rewrite Introduction contributions paragraph
- [ ] Refactor Method section structure
- [ ] Update Results tables with actual numbers
- [ ] Write Analysis section with mechanism explanation
- [ ] Move multi-signal details to Appendix
- [ ] Update Conclusion
- [ ] Update title if needed

---

## 14. Key Numbers for New Tables

### Table 1: Detection Performance (Artifact-Aligned Noise)

| Method | AUROC | vs Input kNN |
|--------|-------|--------------|
| **Explanation kNN** | **0.832** | **+24%** |
| Input kNN | 0.671 | baseline |
| LLM Mismatch | 0.575 | -14% |
| ECG (adaptive) | 0.547 | -18% |
| Cleanlab | 0.107 | **-84%** |

### Table 2: Two-Regime Comparison (Final Table)

| Method | Artifact Noise | Random Noise | Recommendation |
|--------|---------------|--------------|----------------|
| Explanation kNN | 0.832 ✅ | TBD | Use when artifacts suspected |
| Cleanlab | 0.107 ❌ | 0.977 ✅ | Use for random noise only |
| Input kNN | 0.671 | 0.880 | General purpose |

---

*Document created: January 3, 2026*
*Source: Claude 5.2 Pro consultation on paper revision strategy*

