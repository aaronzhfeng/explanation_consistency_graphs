# Archive: Version 1 (Pre-Revision)

**Archived**: January 3, 2026

## Description

This is the initial paper draft written based on the research proposal **before** experimental results were analyzed. 

## Key Characteristics of This Version

- **Claims 5 complementary signals** with adaptive aggregation
- **Multi-signal narrative** as core contribution
- **Placeholder results** (not filled with actual numbers)
- **Strong Related Work & Introduction** (worth preserving)
- **Architecture diagram** shows full 5-signal pipeline

## Why Archived

Experimental results revealed:
1. **Explanation kNN alone** achieves 0.832 AUROC (best method)
2. **Multi-signal aggregation hurts** performance (0.547 vs 0.832)
3. **Dynamics signal is anti-correlated** on artifact noise
4. Paper claims don't match experimental findings

## Files

| File | Description |
|------|-------------|
| `main.tex` | Full paper source (697 lines) |
| `main.pdf` | Compiled PDF (11 pages) |
| `ecg.bib` | Bibliography (50+ entries) |

## What to Preserve in Revision

- Related Work section (comprehensive, well-cited)
- Introduction framing (problem motivation is solid)
- Method: Explanation generation subsection
- Bibliography entries

## See Also

- `docs/10_paper_revision_context.md` - Full context for revision
- `docs/07_experiment_results.md` - Actual experimental results

