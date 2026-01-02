# Research Proposals for ACL 2026

## Target

**ACL 2026 Theme Track: Explainability of NLP Models**
- Deadline: January 5, 2026
- Conference: July 2-7, 2026, San Diego

## Proposals

| # | File | Name | Compute | Status |
|---|------|------|---------|--------|
| 01 | `01_varif.md` | VARIF: Vacuity-Aware Robustness–Informativeness Frontier | ~15–20 H100h | Complete |
| 02 | `02_ecg.md` | ECG: Explanation-Consistency Graphs for Data Debugging | ~20 H100h | **★ Selected** |

## Selected: ECG

**Recommendation rationale:** ECG offers a clearer "win condition" by addressing a demonstrable failure mode of existing data cleaning methods (artifact-aligned noise) and provides a stronger applied contribution to practitioners. VARIF is a solid evaluation protocol but carries a higher risk of being perceived as incremental.

**GitHub:** https://github.com/aaronzhfeng/explanation_consistency_graphs

**Literature:** 103 papers curated across 8 categories (see `../literature.md`)

## Selection Criteria

Both proposals were selected from initial brainstorming based on:
- **Compute feasibility:** < 50 H100h (ideally < 25h)
- **7-day implementation:** Coding handled by AI agent
- **Low novelty risk:** Survived literature collision checks
- **Theme Track alignment:** Addresses ACL 2026 explainability questions

## VARIF Overview

**Core contribution:** An evaluation protocol that reports explanation quality on a 2D frontier (Robustness × Informativeness) with:
- Vacuity-aware scalar score (VARIF)
- Adversarial rationale suite (VARSuite)

**Key innovation:** Explicitly prevents:
1. "Stable-but-empty" failure mode (vacuous explanations)
2. "Informative-but-circular" failure mode (label leakage)

**Datasets:** e-SNLI + FEVER (+ optional BoolQ)

**5 Testable Hypotheses:**
1. Vacuity games stability-only metrics
2. Label leakage games informativeness-only metrics
3. CoT trades informativeness for robustness
4. Semantic drift exposes BERTScore/ROUGE weaknesses
5. VARIF correlates with known-good ordering

---

## ECG Overview

**Core claim:** LLM-generated structured explanations contain signals (agreement patterns, contradictions, artifact focus) that identify mislabeled or artifact-laden training instances that **loss/confidence-based methods (e.g., Cleanlab) miss**—especially when the model confidently fits errors via spurious markers.

**Key innovation:** Graph-aggregated inconsistency signals from:
1. **Neighborhood label inconsistency** (graph-based)
2. **Explanation–label contradiction** (NLI)
3. **Artifact signature** (spurious token focus)

**Dataset:** SST-2 (+ optional CivilComments)

**Key experiment:** "Artifact-aligned label errors" where Cleanlab fails because classifier fits spurious markers with low loss, but ECG still detects inconsistency via explanation structure.

**Expected gains:**
- Precision@K over Cleanlab: +10 to +30 points on artifact-aligned noise
- Artifact-OOD accuracy: +3 to +12 absolute after cleaning

