# ECG Session Log

> Development history and key decisions

---

## Session 1: Initial Implementation

**Date:** January 2026  
**Agent:** Claude (Opus 4.5)  
**Duration:** Single session  

### Objectives

1. ✅ Set up project structure
2. ✅ Implement all core modules
3. ✅ Create experiment runner
4. ✅ Document implementation

### Work Completed

#### Phase 1: Project Setup
- Created project structure (src/ecg/, configs/, scripts/, docs/)
- Set up `requirements.txt` with dependencies
- Created `configs/default.yaml` with all hyperparameters

#### Phase 2: Reference Analysis
- Cloned reference repos: AUM, Cleanlab, Neural-Relation-Graph, WANN
- Analyzed key patterns to adapt:
  - `AUMCalculator` from `aum/aum.py`
  - `cal_auc_ap()` metrics from `NRG/metric.py`
  - Reliability weighting from `WANN.py`
  - `get_relation()` kernel from `NRG/relation.py`

#### Phase 3: Core Implementation

| Module | Lines | Key Features |
|--------|-------|--------------|
| `data.py` | 423 | SST-2 loading, uniform/artifact noise, OOD test sets |
| `train_classifier.py` | 511 | RoBERTa fine-tuning, AUM integration, CV predictions |
| `explain_llm.py` | 867 | vLLM/Outlines/HF backends, stability sampling, schema enforcement |
| `embed_graph.py` | 558 | FAISS kNN, reliability-weighted edges, outlier detection |
| `signals.py` | 814 | 5 signal families, NLI ensemble, adaptive combination |
| `baselines.py` | 515 | Cleanlab, NRG, kNN, LLM mismatch, etc. |
| `eval.py` | 536 | AUROC/AUPRC/P@K, visualization, faithfulness metrics |
| `clean.py` | 490 | Top-K removal, relabeling, outlier/dynamics protection |

**Total: ~4,714 lines of source code**

#### Phase 4: Scripts & Documentation

| File | Purpose |
|------|---------|
| `scripts/run_experiment.py` | Full 8-step pipeline |
| `scripts/quick_test.py` | Mock-data test (no GPU) |
| `docs/00_index.md` | Documentation index |
| `docs/01_architecture.md` | Implementation mindmap |
| `docs/02_module_reference.md` | API documentation |
| `docs/03_session_log.md` | This file |

### Key Design Decisions

#### Decision 1: Use AUM library directly
- **Rationale:** Official implementation is clean and tested
- **Alternative considered:** Custom margin tracking
- **Outcome:** Added `aum>=0.1.0` to requirements

#### Decision 2: Adapt NRG evaluation metrics
- **Rationale:** `cal_auc_ap()` already computes AUROC, AUPRC, TNR@95
- **Alternative considered:** sklearn metrics only
- **Outcome:** Adapted patterns to `eval.py`

#### Decision 3: Reliability-weighted edges (WANN-style)
- **Rationale:** Key innovation in proposal; WANN shows it works
- **Formula:** `w_ij ∝ exp(sim/τ) × r_j`
- **Outcome:** Implemented in `embed_graph.py`

#### Decision 4: Lazy model initialization
- **Rationale:** LLM and NLI models are expensive to load
- **Pattern:** `_initialized` flag, `initialize()` method
- **Outcome:** Models load only when first used

#### Decision 5: Caching explanations to disk
- **Rationale:** LLM generation is expensive and non-deterministic
- **Format:** Pickle file at `outputs/explanations/explanations.pkl`
- **Outcome:** `--skip-llm` flag in experiment runner

#### Decision 6: Folder rename (docs → instruction)
- **Rationale:** User wanted `docs/` for implementation documentation
- **Outcome:** Research materials now in `instruction/`

### Issues Encountered

1. **None significant** — Clean first-pass implementation

### Files Modified

```
Created:
  - src/ecg/__init__.py
  - src/ecg/data.py
  - src/ecg/train_classifier.py
  - src/ecg/explain_llm.py
  - src/ecg/embed_graph.py
  - src/ecg/signals.py
  - src/ecg/baselines.py
  - src/ecg/eval.py
  - src/ecg/clean.py
  - configs/default.yaml
  - scripts/run_experiment.py
  - scripts/quick_test.py
  - docs/00_index.md
  - docs/01_architecture.md
  - docs/02_module_reference.md
  - docs/03_session_log.md
  - requirements.txt
  - .cursorrules
  - README.md
```

### Next Steps (for future sessions)

1. **Run quick test:** `python scripts/quick_test.py`
2. **Set up GPU environment:** vLLM, CUDA
3. **Run full experiment:** `python scripts/run_experiment.py`
4. **Tune hyperparameters:** Based on initial results
5. **Write paper:** LaTeX in `paper/` directory

---

## Session Template (for future sessions)

```markdown
## Session N: [Title]

**Date:** [Date]
**Agent:** [Agent name/version]
**Duration:** [Time]

### Objectives
1. [ ] Objective 1
2. [ ] Objective 2

### Work Completed
- Description of work

### Key Decisions
- Decision and rationale

### Issues Encountered
- Issue and resolution

### Files Modified
- List of files

### Next Steps
- What to do next
```

---

*Update this log after each development session*

