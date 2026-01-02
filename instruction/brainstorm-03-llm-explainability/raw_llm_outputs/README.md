# Raw LLM Outputs for LLM Explainability Research

This folder stores raw outputs from LLM-assisted brainstorming and literature search.

## Files

| # | File | Source | Description |
|---|------|--------|-------------|
| 01 | `01_initial_proposals_gpt52pro.md` | GPT-5.2 Pro | Initial 5 proposal directions from brainstorm prompt |
| 02 | `02_novelty_checked_proposals_gpt52pro.md` | GPT-5.2 Pro | Revised 4 directions after novelty audit + literature scan |
| 03 | `03_xfact_benchmark_gpt52pro.md` | GPT-5.2 Pro | X-FACT benchmark proposal (unified explanation evaluation) |
| 04 | `04_xunit_novelty_checked_gpt52pro.md` | GPT-5.2 Pro | X-UNIT revised benchmark after novelty audit |
| 05 | `05_explaincal_gpt52pro.md` | GPT-5.2 Pro | ExplainCal: calibrated step-level confidence via probes |
| 06 | `06_dualcal_novelty_checked_gpt52pro.md` | GPT-5.2 Pro | DualCal: revised with validity+faithfulness separation |
| 07 | `07_explaincheck_gpt52pro.md` | GPT-5.2 Pro | EXPLAINCHECK: metamorphic stability tests for explanations |
| 08 | `08_xmet_novelty_checked_gpt52pro.md` | GPT-5.2 Pro | X-MET: revised metamorphic test suite |
| 09 | `09_varif_proposal_gpt52pro.md` | GPT-5.2 Pro | **VARIF: Complete implementation proposal for Dir 2** |
| 10 | `10_ecg_proposal_gpt52pro.md` | GPT-5.2 Pro | **ECG: Complete implementation proposal for Dir 4** |
| 11–16 | `11–16_lit_*.md` | GPT-5.2 Pro | Literature search outputs (6 categories) |
| 17 | `17_literature_review_gpt52pro.md` | GPT-5.2 Pro | [PENDING] Literature vs proposal analysis |
| 18 | `18_literature_review_claude.md` | Claude | [PENDING] Literature vs proposal analysis |
| 19 | `19_literature_review_gemini.md` | Gemini | [PENDING] Literature vs proposal analysis |

## Summary of Proposals

### Prompt 01: Extend SOKRATES / Method Proposals

**Initial Round (5 proposals)**
1. **Calibrated explanation validity predictors** — train predictor on verifiable traces
2. **Robustness-aware evaluation** — stability without rewarding vacuity
3. **Bias auditing via counterfactual explanation disparity**
4. **Explanation-driven training data debugging**
5. **Mechanistic knobs for faithful reasoning**

**After Novelty Check (4 revised directions)**
1. **Calibrated faithfulness (causal-use) scores** — distinct from correctness
2. **Vacuity-aware robustness–informativeness frontier**
3. **"Bias without output disparity" via ECS** — counterfactual explanation sensitivity
4. **Explanation-consistency graphs for data debugging**

### Prompt 02: Benchmark Proposals

**X-FACT (initial)** — unified benchmark for explanation Faithfulness, Accuracy, Consistency, calibraTion
- 4 dimensions: Validity, Faithfulness, Stability, Calibration
- 5 tasks: GSM8K, HotpotQA, e-SNLI, FEVER, ToTTo
- ~65–87 H100h compute

**X-UNIT (novelty-checked)** — Explanation Unit Tests for NLP
- Key novelty: cross-modality comparability + calibration as first-class track
- Unit-test abstraction across explanation types
- Bidirectional stability (invariance + sensitivity)
- Gap exploited: no existing benchmark does calibration of explanation reliability

### Prompt 03: Calibrated Explanation Confidence

**ExplainCal (initial)** — Calibrated Step-Level Confidence via Hidden-State Probes
- Step validity + faithfulness as separate signals
- Activation probes trained on automatic step labels
- ~47–75 H100h compute

**DualCal (novelty-checked)** — Calibrated Step-wise Validity and Faithfulness via Activation Monitors
- Key differentiation: NLP tasks (not math/logic), dual (validity, faithfulness) signals
- Activation "knob" for Theme Track alignment
- ~80–95 H100h compute
- Collision warnings: step-wise confidence (ICLR'26), ELAD, Tanneru et al.

### Prompt 04: Stability Testing

**EXPLAINCHECK (initial)** — Metamorphic Stability Tests for Faithful NLP Explanations
- Invariance + equivariance + contrast transforms
- Multi-modality: attributions, rationales, attention, free-text
- ~70–110 H100h compute

**X-MET (novelty-checked)** — Metamorphic Explanation Tests for NLP Models
- Key differentiation: beyond synonym substitutions, alignment-aware metrics, oracle-free
- Multi-format (CheckList-style for explanations)
- ~100 H100h compute
- Collision warnings: Yin/Zhao (synonym attacks), Crabbé (symmetry groups), LExT

## Recommended Picks (from GPT-5.2 Pro)

### Method Papers (Prompt 01)
For **7-day sprint + low compute risk**:
- Direction 2 (vacuity-aware frontier) — ~10–25 H100h
- Direction 4 (explanation-consistency graphs) — ~20–50 H100h

For **theme-track signature** (internal mechanisms):
- Direction 1 (calibrated faithfulness) — ~30–60 H100h

### Benchmark Paper (Prompt 02)
**X-UNIT** is the novelty-safe version:
- Make **calibration-of-explanations + cross-modality unit tests** the headline
- Treat faithfulness/stability as "necessary but not novel alone"
- ~65–87 H100h (within budget)

### Calibrated Confidence Paper (Prompt 03)
**DualCal** is the novelty-safe version:
- **2D explanation confidence** (validity + faithfulness) from activations
- Focus on **NLP tasks** (not math) to avoid PRM/probe collision
- Include **activation steering "knob"** for Theme Track
- ~80–95 H100h (within budget)

### Stability Testing Paper (Prompt 04)
**X-MET** is the novelty-safe version:
- **Metamorphic test suite** with explicit laws (invariance + equivariance + flip)
- **Alignment-aware** metrics for paraphrase-level transforms
- **Oracle-free** (no gold explanations needed)
- ~100 H100h (at budget limit)

## Naming Convention

```
{NN}_{description}_{source}_{date}.md
```

- `NN`: sequence number (01, 02, ...)
- `description`: brief topic slug
- `source`: model name (gpt52pro, claude, etc.)
- `date`: optional, YYYY-MM-DD format

