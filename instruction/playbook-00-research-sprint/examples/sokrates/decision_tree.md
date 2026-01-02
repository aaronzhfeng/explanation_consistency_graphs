# SOKRATES: Human Decision Points

> When did the human intervene vs let AI continue?

---

## Decision Tree Overview

```
AI proposes → Human reviews → Decision
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Approve      Modify      Reject
        │           │           │
        ▼           ▼           ▼
   AI continues  AI revises  AI restarts
```

---

## Phase 0: Ideation — 3 Decision Points

### Decision 1: Initial Direction
```
Human input: "OaK for AAAI Bridge Workshop"
AI proposed: SOKRATES with solver-guided DPO
─────────────────────────────────────────────
Human decision: APPROVE
Reasoning: Aligned with interests, feasible timeline
Time spent: 5 minutes
```

### Decision 2: Novelty Framing
```
AI proposed: 3 differentiators vs prior work
  1. Explicit options (not generic CoT)
  2. Explicit knowledge (q̂ predictor)
  3. Micro OaK loop (iterative training)
─────────────────────────────────────────────
Human decision: APPROVE
Reasoning: Clear, defensible, distinct from LoCo/VeriCoT
Time spent: 10 minutes
```

### Decision 3: Dataset Choice
```
AI proposed: PrOntoQA for SFT, FOLIO for transfer
Alternative: Joint training on both
─────────────────────────────────────────────
Human decision: APPROVE primary, NOTE alternative for future
Reasoning: Cleaner story, transfer is stronger claim
Time spent: 5 minutes
```

---

## Phase 1: Implementation — 2 Decision Points

### Decision 4: Model Selection
```
AI proposed: Qwen3-8B
Alternatives considered:
  - Llama-3.1-8B (gated, requires license)
  - Qwen2.5-7B (smaller, less capable)
  - Mistral-7B (good but less reasoning focus)
─────────────────────────────────────────────
Human decision: APPROVE
Reasoning: Apache 2.0 license, good reasoning, easy to access
Time spent: 15 minutes (tested a few options)
```

### Decision 5: Architecture Choices
```
AI proposed: 
  - LoRA rank 64
  - Thought/Action format
  - Constrained decoding for options
─────────────────────────────────────────────
Human decision: APPROVE with one change
Change: Skip constrained decoding initially (add if needed)
Reasoning: Simplify first, add complexity if issues arise
Time spent: 10 minutes
```

---

## Phase 2: Training — 5 Decision Points

### Decision 6: First Bug Discovery
```
Situation: SFT complete, loss low, but output is garbage
AI diagnosis: 4 bugs in data pipeline
AI proposed fix: Rewrite optionizer, fix loader
─────────────────────────────────────────────
Human decision: APPROVE fixes
Reasoning: Root cause analysis was convincing
Time spent: 30 minutes reviewing diagnosis
```

### Decision 7: Time Crunch Trade-offs
```
Situation: 6 hours to deadline, training not done
AI proposed:
  - Reduce dataset 14K → 1.5K
  - Reduce iterations 3 → 2
  - Document all trade-offs
─────────────────────────────────────────────
Human decision: APPROVE with modification
Modification: Keep 3 iterations (found we had more time)
Reasoning: Iterations are the OaK story, can't cut
Time spent: 15 minutes
```

### Decision 8: Multi-GPU Strategy
```
AI proposed: 
  - vLLM for inference (40× speedup)
  - Data parallel across 6 GPUs
  - Work splitting in generation
─────────────────────────────────────────────
Human decision: APPROVE
Reasoning: Clear performance win, implementation looked sound
Time spent: 10 minutes
```

### Decision 9: When to Stop Iterating
```
Situation: DPO iter 3 complete
  - Accuracy: 97.6% (slightly down from 98.1%)
  - Step validity: 91.8% (up from 83.5%)
AI proposed: Stop at iter 3, diminishing returns
─────────────────────────────────────────────
Human decision: APPROVE
Reasoning: Validity still improving but accuracy saturated
Time spent: 5 minutes
```

### Decision 10: FOLIO Transfer
```
AI proposed: Evaluate DPO-iter3 on FOLIO (no fine-tuning)
Result: 45.3% → 53.2% accuracy (zero-shot transfer)
─────────────────────────────────────────────
Human decision: APPROVE (include in paper)
Reasoning: Strong transfer result, supports generalization claim
Time spent: 5 minutes
```

---

## Phase 3: Writing — 4 Decision Points

### Decision 11: Paper Framing
```
AI proposed: Lead with "right answer, wrong reasoning" problem
Alternative: Lead with OaK theory
─────────────────────────────────────────────
Human decision: APPROVE problem-first framing
Reasoning: More accessible, hooks readers faster
Time spent: 10 minutes
```

### Decision 12: Claim Strength
```
AI initially wrote: "First OaK instantiation for LLM reasoning"
Reviewer concern: Too broad, might not be literally first
─────────────────────────────────────────────
Human decision: MODIFY to 3-part specific claim
New claim: "First system that (i) represents proofs as 
  option vocabulary, (ii) learns option-success predictor,
  (iii) aligns via solver-derived DPO"
Time spent: 20 minutes
```

### Decision 13: What to Include in Appendix
```
AI proposed: Extensive appendix with:
  - Full implementation details
  - All hyperparameters
  - Extended analysis
  - More examples
─────────────────────────────────────────────
Human decision: APPROVE but prioritize
Priority order: Hyperparams > Examples > Analysis
Reasoning: Reviewers want reproducibility
Time spent: 10 minutes
```

### Decision 14: Citation Additions
```
AI identified 6 missing citations:
  - Tree-of-Thoughts
  - ECE (calibration)
  - Brier score
  - Z3 solver
  - Unfaithful CoT
  - Process supervision
─────────────────────────────────────────────
Human decision: APPROVE all
Reasoning: Standard references that should be there
Time spent: 5 minutes
```

---

## Phase 4: Submission — 2 Decision Points

### Decision 15: Figure Placement
```
Problem: Figure 1 appearing on page 3
AI proposed: Move \input to after abstract
─────────────────────────────────────────────
Human decision: APPROVE
Result: Figure now on page 2
Time spent: 2 minutes
```

### Decision 16: Final Submission
```
AI completed pre-submission checklist:
  ✓ All numbers match
  ✓ References validated
  ✓ Formatting correct
  ✓ Anonymization correct (single-blind venue)
─────────────────────────────────────────────
Human decision: SUBMIT
Time spent: 5 minutes final review
```

---

## Decision Patterns

### When Human ALWAYS Intervenes

| Situation | Why Human Needed |
|-----------|------------------|
| Initial direction | Only human knows interests/constraints |
| Model selection | Involves license, access, prior experience |
| Trade-offs under pressure | Risk tolerance is personal |
| Claim strength | Reputation on the line |
| Final submit | Can't undo |

### When Human USUALLY Approves

| Situation | Why Usually Approve |
|-----------|---------------------|
| Bug fixes (with diagnosis) | AI explains root cause |
| Performance optimizations | Clear metrics improvement |
| Citation additions | Objective need |
| Formatting fixes | Low risk, high benefit |

### When Human Should MODIFY

| Situation | Typical Modification |
|-----------|---------------------|
| Overly broad claims | Narrow to specific |
| Too many features | Simplify first |
| Complex solution | Try simpler first |
| Scope creep | Refocus on core |

---

## Time Spent on Decisions

| Phase | Decisions | Total Time |
|-------|-----------|------------|
| Ideation | 3 | 20 min |
| Implementation | 2 | 25 min |
| Training | 5 | 65 min |
| Writing | 4 | 45 min |
| Submission | 2 | 7 min |
| **Total** | **16** | **~2.7 hours** |

This is 7% of total project time (38h), but determines 100% of the outcome.

