# Prompts for LLM Explainability Research

## Workflow

1. **Brainstorm prompts** (01–04) → Send to GPT-5.2 Pro for proposal generation
2. **Review proposals** → Select/refine direction
3. **Refinement prompts** (05–08) → Deep dive on selected directions
4. **Literature search prompts** (09+) → Generate after direction is finalized

## Brainstorm Prompts (Initial)

| # | File | Purpose |
|---|------|---------|
| 01 | `01_extend_sokrates.md` | Extensions to SOKRATES for ACL 2026 |
| 02 | `02_explanation_faithfulness_benchmark.md` | Benchmark/resource paper angle |
| 03 | `03_calibrated_explanation_confidence.md` | Uncertainty for explanations |
| 04 | `04_stability_testing_nlp.md` | Invariance/stability for NLP explanations |

## Refinement Prompts (Selected Directions)

| # | File | Direction | Purpose |
|---|------|-----------|---------|
| 05 | `05_dir2_experiment_design.md` | Dir 2: Vacuity-aware frontier | Concrete experiment design |
| 06 | `06_dir2_related_work.md` | Dir 2: Vacuity-aware frontier | Related work deep dive |
| 07 | `07_dir4_experiment_design.md` | Dir 4: Explanation-consistency graphs | Concrete experiment design |
| 08 | `08_dir4_related_work.md` | Dir 4: Explanation-consistency graphs | Related work deep dive |

## Literature Search Prompts (ECG)

| # | File | Category |
|---|------|----------|
| 08 | `08_lit_data_cleaning.md` | Data cleaning / label noise detection |
| 09 | `09_lit_explanation_debugging.md` | Explanation-based debugging |
| 10 | `10_lit_llm_explanations.md` | LLM-generated explanations |
| 11 | `11_lit_graph_data_quality.md` | Graph-based data quality |
| 12 | `12_lit_spurious_correlations.md` | Spurious correlations / artifacts |
| 13 | `13_lit_explanation_faithfulness.md` | Explanation faithfulness metrics |

## Proposal Review Prompts

| # | File | Purpose |
|---|------|---------|
| 14 | `14_literature_proposal_review.md` | Multi-model review: literature vs proposal |

**Usage:** Send prompt 14 with `literature.md` + `proposals/02_ecg.md` to multiple AI models (GPT-5.2 Pro, Claude, Gemini) for independent analysis of how literature should inform the proposal.

## Selected Direction

**ECG (Dir 4):** Explanation-Consistency Graphs for data debugging (~20 H100h)
- Repo: https://github.com/aaronzhfeng/explanation_consistency_graphs

## Target

**ACL 2026 Theme Track: Explainability of NLP Models**
- Deadline: January 5, 2026
- Conference: July 2-7, 2026, San Diego

