# Paper Revision Context for ECG

> Context document for consulting on paper revision strategy
> Created: January 3, 2026

---

## 1. Project Overview

**ECG (Explanation-Consistency Graphs)** is a method for detecting mislabeled training examples using LLM-generated explanations. The core idea is that noisy/mislabeled examples will produce inconsistent or outlier explanations compared to their neighbors.

**Target venue**: ACL 2026 Theme Track on Explainability of NLP Models

---

## 2. Current Paper State

### 2.1 What the Paper Currently Claims

The paper (`paper/main.tex`) was written based on the research proposal before experiments completed. It claims:

1. **Five complementary signals** detect noisy labels:
   - $S_{\text{nbr}}$: Neighborhood surprise (kNN disagreement)
   - $S_{\text{nli}}$: NLI contradiction detection
   - $S_{\text{art}}$: Artifact focus score
   - $S_{\text{stab}}$: Explanation stability
   - $S_{\text{dyn}}$: Training dynamics (AUM)

2. **Adaptive aggregation** combines signals with per-instance confidence weighting

3. **Multi-signal approach** captures different failure modes that single methods miss

4. **Graph structure** enables propagating suspicion through explanation neighborhoods

### 2.2 Current Paper Structure

| Section | Content |
|---------|---------|
| Abstract | Claims multi-signal approach with adaptive aggregation |
| Introduction | Positions ECG as using "five complementary signals" |
| Related Work | Comprehensive (4 subsections, well-cited) |
| Method | Describes all 5 signals + adaptive aggregation |
| Experimental Setup | Describes artifact-aligned and random noise |
| Results | **Placeholder values** (not filled with real numbers) |
| Analysis | Discusses signal contributions (hypothetical) |
| Conclusion | Claims ECG advances data-centric AI |

### 2.3 Architecture Diagram

The paper includes a TikZ architecture diagram showing:
```
Input → Explanation Gen → Graph → [5 Signals] → Aggregation → Ranking
```

---

## 3. Actual Experimental Findings

### 3.1 Key Discovery: Explanation kNN Wins

The experiments (documented in `docs/07_experiment_results.md`) found that **Explanation kNN alone** dramatically outperforms the full multi-signal pipeline:

| Method | AUROC (Artifact Noise) | Notes |
|--------|------------------------|-------|
| **Explanation kNN** | **0.832** | 🏆 Best method |
| Input kNN | 0.671 | Same algorithm, different embedding |
| LLM Mismatch | 0.575 | Simple disagreement |
| ECG (adaptive) | 0.547 | Multi-signal combination |
| ECG (fixed) | 0.547 | Fixed-weight combination |
| Cleanlab | 0.107 | ❌ Catastrophic failure |

**Key insight**: Computing kNN in the **explanation embedding space** beats the same algorithm on **input embeddings** by +24% (0.832 vs 0.671).

### 3.2 Multi-Signal Aggregation Hurts Performance

From `docs/06_debug_session_2026_01_02.md`:

The dynamics signal ($S_{\text{dyn}}$) is **anti-correlated** with noise on artifact data:
- Noisy examples have HIGH AUM (easy to learn via artifacts)
- $S_{\text{dyn}} = -\text{AUM}$ → LOW for noisy examples
- This REDUCES suspicion for the exact examples we want to flag!

**Result**: Adding more signals introduces noise rather than signal. Explanation kNN alone works best.

### 3.3 Robustness Story: ECG vs Cleanlab

| Method | Artifact Noise | Random Noise | Interpretation |
|--------|---------------|--------------|----------------|
| Explanation kNN | 0.832 ✅ | TBD | Best on artifacts |
| Cleanlab | 0.107 ❌ | 0.977 ✅ | Fails catastrophically on artifacts |
| Input kNN | 0.671 | 0.880 | Decent on both |
| LLM Mismatch | 0.575 | 0.901 | Robust but not best |

**Narrative**: Cleanlab is the right choice for random noise (0.977 AUROC). But when artifacts cause the model to confidently fit mislabeled examples, Cleanlab fails completely. ECG (via Explanation kNN) is **robust** to this failure mode.

### 3.4 Why Explanation Embeddings Work

From the experiments, we learned:
1. **Input embeddings** capture "what the text is about"
2. **Explanation embeddings** capture "why this text has this label"
3. When labels are wrong, the "why" becomes inconsistent/outlier
4. kNN in explanation space detects this inconsistency

---

## 4. Discrepancies Between Paper and Results

### 4.1 Claims That Are NOT Supported

| Paper Claims | Reality |
|--------------|---------|
| "Five complementary signals" | Only Explanation kNN matters |
| "Adaptive aggregation improves detection" | Aggregation hurts (0.547 vs 0.832) |
| "Multi-signal captures different failure modes" | Signals are anti-correlated on artifacts |
| "Graph propagates suspicion" | Simple kNN suffices, no propagation needed |

### 4.2 Claims That ARE Supported

| Paper Claims | Evidence |
|--------------|----------|
| LLM explanations are useful for detection | Explanation kNN: 0.832 |
| Traditional methods fail on artifact noise | Cleanlab: 0.107 |
| Explanation-based methods are robust | Works on artifacts where Cleanlab fails |
| Structured explanations provide rich signal | 24% improvement over input embeddings |

### 4.3 What's Missing from Paper

1. **Explanation kNN as primary method** - not mentioned as standalone approach
2. **Honest ablation** showing signal combination doesn't help
3. **Actual result numbers** - paper has placeholders
4. **Failure mode analysis** - why dynamics hurt on artifacts
5. **When to use ECG vs Cleanlab** - complementary tools, not replacement

---

## 5. Possible Revision Strategies

### Option A: Minimal Revision (Keep Multi-Signal Story)

- Keep the 5-signal framework as "what we investigated"
- Add ablation showing Explanation kNN is best
- Present multi-signal as "future work" for combining properly
- **Risk**: Reviewers may see this as overselling

### Option B: Reframe Around Explanation Embeddings

- Lead with "explanation embeddings beat input embeddings"
- Position 5 signals as ablation study
- Main contribution: kNN in explanation space
- **Risk**: Less novel-sounding than multi-signal story

### Option C: Honest Negative Results

- Present the full journey: "we designed 5 signals, but found simple kNN works best"
- Analyze why (anti-correlation, etc.)
- Contribute understanding of when/why multi-signal fails
- **Risk**: May seem like a failure paper

### Option D: Hybrid Approach

- Reframe contribution as "explanation representations for data quality"
- Keep architecture but highlight the "winning path"
- Include ablation as key finding
- Position signal combination as open problem
- **Risk**: Complex narrative

---

## 6. Key Questions for Revision

1. **What is the core contribution?**
   - (a) The explanation embedding representation
   - (b) The robustness over Cleanlab
   - (c) The signal framework (even if simple kNN wins)
   - (d) The analysis of why multi-signal fails

2. **How to handle multi-signal failure?**
   - (a) Omit it entirely
   - (b) Present as negative result / ablation
   - (c) Discuss as future work
   - (d) Analyze deeply as contribution

3. **What should the architecture diagram show?**
   - (a) Simplified: just Explanation kNN path
   - (b) Full diagram with "winning path" highlighted
   - (c) Both (overview + detailed)

4. **How to position vs Cleanlab?**
   - (a) ECG as replacement (but it's not better on random noise)
   - (b) ECG as complement (use when you suspect artifacts)
   - (c) ECG as robust alternative (works on both, best on artifacts)

5. **What about missing experiments?**
   - Explanation kNN on random noise (not run yet)
   - Downstream evaluation with Explanation kNN (running)
   - Should we wait for these before revising?

---

## 7. Concrete Numbers for Paper

### Table 1: Detection Performance (Artifact-Aligned Noise, 10%, N=25,000)

| Method | AUROC | AUPRC | P@10% | Source |
|--------|-------|-------|-------|--------|
| **Explanation kNN** | **0.832** | 0.435 | 0.496 | `20260103_130312_ensemble_results.json` |
| Input kNN | 0.671 | 0.258 | 0.342 | Same |
| LLM Mismatch | 0.575 | 0.152 | 0.280 | Same |
| Artifact Score | 0.549 | 0.187 | 0.174 | Same |
| ECG (adaptive) | 0.547 | ~0.12 | ~0.15 | `results.json` |
| Cleanlab | 0.107 | 0.056 | 0.000 | Same |
| Loss-based | 0.107 | 0.056 | 0.000 | Same |
| Margin-based | 0.107 | 0.056 | 0.000 | Same |

### Table 2: Detection Performance (Random Noise, 10%, N=25,000)

| Method | AUROC | AUPRC | P@10% | Source |
|--------|-------|-------|-------|--------|
| Cleanlab | 0.977 | 0.854 | 0.916 | `20260102_185651_random_noise_results.json` |
| LLM Mismatch | 0.901 | 0.632 | 0.789 | Same |
| Input kNN | 0.880 | 0.492 | 0.650 | Same |
| ECG (adaptive) | 0.747 | 0.235 | 0.306 | Same |
| **Explanation kNN** | **TBD** | TBD | TBD | Not run yet |

### Table 3: Key Comparison

| Method | Artifact Noise | Random Noise | Avg | Robust? |
|--------|---------------|--------------|-----|---------|
| Explanation kNN | 0.832 | TBD | - | ✅ Expected |
| Cleanlab | 0.107 | 0.977 | 0.542 | ❌ Catastrophic failure on artifacts |
| Input kNN | 0.671 | 0.880 | 0.776 | ✅ Works on both |

---

## 8. Recommended Narrative (Pending Discussion)

### Strong Version

> "We introduce Explanation Consistency Graphs (ECG), which detects noisy labels by leveraging LLM-generated explanations. Our key insight is that computing neighborhood surprise in the *explanation embedding space* captures label-quality information invisible to raw input embeddings. On artifact-aligned noise—a realistic failure mode where traditional methods like Cleanlab achieve only 0.107 AUROC—Explanation kNN achieves **0.832 AUROC**, a 24% improvement over the same algorithm on input embeddings (0.671)."

### Honest Version

> "We investigate using LLM explanations for training data debugging. While we initially designed a five-signal framework with adaptive aggregation, our experiments reveal that a simpler approach—computing kNN in the explanation embedding space—achieves the best results. This finding suggests that the key value of LLM explanations lies in their representation, not in the specific signals we extract. We analyze why our multi-signal approach underperformed and discuss implications for future work."

---

## 9. Files Reference

| File | Purpose |
|------|---------|
| `paper/main.tex` | Current paper (needs revision) |
| `paper/ecg.bib` | Bibliography (comprehensive) |
| `docs/06_debug_session_2026_01_02.md` | Debugging log with key findings |
| `docs/07_experiment_results.md` | Detailed experiment results |
| `docs/08_experiment_guide_v2.md` | How to run experiments |
| `docs/09_results_reference.md` | Where to find numbers |
| `instruction/research_proposal_1.md` | Original proposal (pre-experiment) |

---

## 10. Open Questions

1. Should we run Explanation kNN on random noise before revising? (Would complete the comparison table)

2. Is the 24% improvement over Input kNN enough for a paper, or do we need more?

3. How do reviewers typically react to "we designed X but found Y works better"?

4. Should we reposition as a "representation learning" paper rather than "multi-signal detection"?

5. The Related Work and Introduction are strong—should we preserve them or align with new findings?

---

*Document created for consultation on paper revision strategy*

