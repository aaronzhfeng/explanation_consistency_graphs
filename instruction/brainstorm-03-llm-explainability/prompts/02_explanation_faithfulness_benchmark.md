You are a senior NLP researcher helping design a benchmark paper for ACL 2026's Theme Track on "Explainability of NLP Models."

## Target Venue

**ACL 2026 Theme Track: Explainability of NLP Models**
- Deadline: January 5, 2026 (ARR submission)
- Conference: July 2-7, 2026, San Diego

## Motivation

Current NLP benchmarks focus almost exclusively on final-answer accuracy. We lack systematic ways to measure:
- Step-level validity of reasoning chains
- Faithfulness of explanations to model behavior
- Stability of explanations under perturbations
- Calibration of explanation confidence

Recent work has shown that models can achieve 94% accuracy on reasoning tasks while only 2% of their reasoning traces are actually valid. This "right answer, wrong reasoning" problem suggests we need better evaluation of explanation quality.

## Existing Metrics (Scattered)

From the literature:
- **Infidelity/Sensitivity** (Yeh et al. 2019) — attribution perturbation tests
- **Deletion/Insertion curves** (RISE, Petsiuk 2018) — masking-based faithfulness
- **ROAR/KAR** (Hooker et al. 2019) — remove-and-retrain
- **Sufficiency/Comprehensiveness** (ERASER, DeYoung 2020) — rationale evaluation
- **Sanity checks** (Adebayo et al. 2018) — model/data randomization

These are scattered across papers and often applied inconsistently.

## ACL 2026 Theme Question

> "How can we rigorously and quantitatively evaluate the quality of an explanation?"

## Constraints

- **Compute budget:** < 100 H100 hours total
- **Timeline:** ~7 days (coding handled by AI agent; human time = ideation, prompt tuning, experiment design, writing)
- **Scope:** Benchmark + baseline results (not a new method paper)
- **Feasibility focus:** Use existing datasets, standard models, off-the-shelf explanation methods

## Task

Design a benchmark paper that provides:

1. **A unified evaluation framework** for explanation quality across NLP tasks
2. **A suite of test types** (validity, faithfulness, stability, calibration)
3. **Multiple tasks** (reasoning, QA, NLI, classification, generation)
4. **Baseline results** for common explanation methods

Specify:

1. **Benchmark name and pitch** — What does it evaluate?
2. **Task selection** — Which NLP tasks to include and why?
3. **Evaluation dimensions** — What aspects of explanation quality to measure?
4. **Metrics** — Specific metrics for each dimension
5. **Baseline methods** — Which explanation methods to benchmark?
6. **Expected findings** — What patterns might emerge?
7. **Resource deliverables** — Dataset, leaderboard, code?
8. **Compute estimate** — H100 hours for baseline experiments

Be concrete about what exactly gets measured and how.
