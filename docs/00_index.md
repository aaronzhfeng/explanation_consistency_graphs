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
| 05 | [Experiment Commands](05_experiment_commands.md) | **Step-by-step experiment guide** ★ |

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
| LLM explanation generation | ✅ Complete | `explain_llm.py` |
| Graph construction | ✅ Complete | `embed_graph.py` |
| Signal computation | ✅ Complete | `signals.py` |
| Baselines | ✅ Complete | `baselines.py` |
| Evaluation metrics | ✅ Complete | `eval.py` |
| Data cleaning | ✅ Complete | `clean.py` |
| Experiment pipeline | ✅ Complete | `scripts/run_experiment.py` |
| **GPU testing** | ⏳ Pending | Requires H100 access |
| Paper writing | 🔲 Not started | — |

---

## Key Design Decisions

1. **Modular architecture** — Each module is self-contained with clear interfaces
2. **Reference-informed** — Adapted code patterns from AUM, NRG, WANN, Cleanlab
3. **Config-driven** — All hyperparameters in YAML for reproducibility
4. **Lazy loading** — LLM/models initialized only when needed
5. **Caching** — Explanations cached to disk for resumability

---

*Last updated: Session 1 (initial implementation)*

