# ECG Documentation Index

> Implementation documentation for Explanation-Consistency Graphs

---

## Document Index

| # | Document | Description |
|---|----------|-------------|
| 00 | [Index](00_index.md) | This file |
| 01 | [Architecture](01_architecture.md) | Implementation architecture & mindmap |
| 02 | [Module Reference](02_module_reference.md) | Detailed module documentation |
| 03 | [Session Log](03_session_log.md) | Development history & decisions |
| 04 | [Pipeline Guide](04_pipeline_guide.md) | How to run experiments |
| 05 | [Experiment Commands](05_experiment_commands.md) | Original step-by-step guide |
| 06 | [Debug Session 2026-01-02](06_debug_session_2026_01_02.md) | H100 debugging & optimizations |
| 07 | [Experiment Results](07_experiment_results.md) | **Results log with timestamps** ★ |
| 08 | [Experiment Guide v2](08_experiment_guide_v2.md) | **Updated guide with all experiments** ★★ |
| 09 | [Results Reference](09_results_reference.md) | **File locations for paper writing** ★★ |
| 10 | [Paper Revision Context](10_paper_revision_context.md) | Conversation context for revisions |
| 11 | [Paper Revision Plan](11_paper_revision_plan.md) | **Definitive revision strategy** ★★ |
| 12 | [Remaining Experiments](12_remaining_experiments.md) | **Commands for pending experiments** ★ |

---

## Quick Links

### Research Context
- [Research Proposal (Revised)](../instruction/research_proposal_1.md) — Full ECG methodology
- [Literature](../instruction/literature.md) — 103-paper bibliography
- [Speedrun Playbook](../instruction/playbook-00-research-sprint/) — Methodology template

### Implementation
- [Source Code](../src/ecg/) — Python modules (~5,700 lines)
- [Configuration](../configs/default.yaml) — Hyperparameters
- [Scripts](../scripts/) — Experiment runners

### References
- [AUM](../references/aum/) — Training dynamics baseline
- [Cleanlab](../references/cleanlab/) — Confident learning baseline
- [Neural-Relation-Graph](../references/Neural-Relation-Graph/) — Graph-based detection
- [WANN](../references/wann-noisy-labels/) — Reliability-weighted kNN

---

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Data loading & noise injection | ✅ Complete | `data.py` |
| Classifier training + AUM | ✅ Complete | `train_classifier.py` |
| LLM explanation generation | ✅ Complete | `explain_llm.py` (batched, Qwen3-8B) |
| Graph construction | ✅ Complete | `embed_graph.py` |
| Signal computation | ✅ Complete | `signals.py` |
| Baselines | ✅ Complete | `baselines.py` |
| Evaluation metrics | ✅ Complete | `eval.py` |
| Data cleaning | ✅ Complete | `clean.py` |
| Experiment pipeline | ✅ Complete | `scripts/run_experiment.py` |
| **GPU testing** | ✅ Complete | H100 debugging session complete |
| **Artifact-aligned noise** | ✅ Complete | **Explanation kNN: 0.832 AUROC** ★ |
| **Random noise experiment** | ✅ Complete | Cleanlab: 0.977 AUROC |
| **Ensemble ablation** | ✅ Complete | Best signal identified |
| Downstream evaluation | 🔄 Running | Cleaning improves accuracy |
| Noise rate sensitivity | 🔲 Pending | 5%, 20% rates |
| Paper writing | 🔲 Not started | — |

---

## Key Design Decisions

1. **Modular architecture** — Each module is self-contained with clear interfaces
2. **Reference-informed** — Adapted code patterns from AUM, NRG, WANN, Cleanlab
3. **Config-driven** — All hyperparameters in YAML for reproducibility
4. **Lazy loading** — LLM/models initialized only when needed
5. **Caching** — Explanations cached to disk for resumability
6. **Batched LLM inference** — 200x speedup via vLLM batch generation

---

*Last updated: January 3, 2026 (Ensemble experiments complete — Explanation kNN: 0.832)*

