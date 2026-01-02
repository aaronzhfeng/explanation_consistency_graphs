# Prompt Patterns for Research

> What makes prompts effective for AI-assisted research

---

## Core Patterns

### Pattern 1: Structured Output Request

**Bad:**
```
Design a project
```

**Good:**
```
Design a research project. Include:
1. Title (with acronym if applicable)
2. Abstract (200 words)
3. Key contributions (3-5 bullets)
4. Experimental design
5. Risk assessment
6. Timeline
```

**Why it works:** AI produces comprehensive, organized output that's directly usable.

---

### Pattern 2: Context + Constraints + Goal

**Bad:**
```
Write code for training
```

**Good:**
```
Context: Building a DPO training pipeline for logical reasoning
Constraints:
  - Must work with Qwen3-8B + LoRA
  - Must support multi-GPU with accelerate
  - Must save checkpoints every epoch
Goal: Training script that takes config, data path, and model path

Use the data structures from src/data/structures.py
```

**Why it works:** AI understands the environment, limitations, and success criteria.

---

### Pattern 3: Symptom-Based Debugging

**Bad:**
```
The model doesn't work
```

**Good:**
```
The model trained with low loss but generates garbage.

Symptoms:
- Loss: 3.5 → 0.01 (looks good)
- Output: Only produces "CONCLUDE" step
- Thoughts: Literal "<Thought>" string instead of text

Training data path: data/processed/prontoqa_train.jsonl
Training script: scripts/train_sft.py

What's wrong?
```

**Why it works:** AI can diagnose with specific symptoms and code locations.

---

### Pattern 4: Trade-off Elicitation

**Bad:**
```
Speed it up
```

**Good:**
```
I have 6 hours until deadline. Training isn't done.

Current plan:
- 14K training problems
- 3 DPO iterations
- 8 samples per problem

What can I cut while maintaining:
1. Statistical significance
2. Core contribution validity
3. Reproducibility

Document all trade-offs for the paper.
```

**Why it works:** AI provides principled options, not just "do less."

---

### Pattern 5: Reference to Existing Context

**Bad:**
```
Write the training code
```

**Good:**
```
Implement the SFT training script.

Use:
- Data structures from src/data/structures.py
- Config schema from configs/training.yaml
- Logging utilities from src/utils/logging.py

Follow the patterns in existing scripts like scripts/prepare_data.py
```

**Why it works:** AI maintains consistency with existing codebase.

---

### Pattern 6: Iterative Refinement

**First prompt:**
```
Write the abstract for this paper.
[Context about the paper]
```

**Follow-up:**
```
Good, but:
1. Make the problem statement sharper
2. Quantify the improvement (44×, not "significant")
3. Reduce from 250 to 200 words
```

**Why it works:** Easier to refine than to get perfect on first try.

---

### Pattern 7: Verification Request

**Bad:**
```
Is this correct?
```

**Good:**
```
Verify this claim against our results:

Claim: "44× improvement in trace validity"
Data: 
- Before: 2.1% valid traces
- After: 92.0% valid traces

Check:
1. Is the math correct?
2. Are we using the right metrics?
3. Is this claim defensible to reviewers?
```

**Why it works:** AI checks specific things rather than generic approval.

---

## Phase-Specific Patterns

### Ideation Phase

```
I want to submit to [VENUE] on [TOPIC].

Available:
- [TIME] weeks
- [COMPUTE] resources
- [DATA] access

Design a project that's:
1. Novel enough for acceptance
2. Implementable in the timeframe
3. Has clear evaluation metrics
4. Has backup plans if things fail

What's the minimum viable paper? What's the ideal outcome?
```

---

### Implementation Phase

```
Implement [COMPONENT].

Requirements:
- [REQ 1]
- [REQ 2]

Interface:
- Input: [DESCRIPTION]
- Output: [DESCRIPTION]

Use existing patterns from [FILE] as reference.
Include:
- Type hints
- Docstrings
- Error handling
- Logging
```

---

### Debugging Phase

```
Bug report:

Expected behavior: [WHAT SHOULD HAPPEN]
Actual behavior: [WHAT HAPPENS]

Symptoms:
- [SYMPTOM 1]
- [SYMPTOM 2]

Relevant code:
- [FILE 1]: [DESCRIPTION]
- [FILE 2]: [DESCRIPTION]

What I've tried:
- [ATTEMPT 1]: [RESULT]
- [ATTEMPT 2]: [RESULT]
```

---

### Writing Phase

```
Write section [N]: [TITLE]

Purpose: [WHAT THIS SECTION SHOULD ACCOMPLISH]

Content to include:
- [POINT 1]
- [POINT 2]

Tone: [FORMAL/TECHNICAL/ACCESSIBLE]
Length: [APPROXIMATE WORDS/PARAGRAPHS]

Reference these results:
[PASTE RELEVANT DATA]
```

---

### Revision Phase

```
Revise this section.

Current version:
[PASTE TEXT]

Issues to fix:
1. [ISSUE 1]
2. [ISSUE 2]

Constraints:
- Keep under [N] words
- Maintain [SPECIFIC CLAIM]
- Cite [REFERENCE]
```

---

## Anti-Patterns (What NOT to Do)

### Anti-Pattern 1: Vague Requests

```
# Bad
Make it better

# Good
Make the abstract more compelling by:
1. Starting with a concrete problem
2. Quantifying the solution
3. Ending with significance
```

---

### Anti-Pattern 2: Missing Context

```
# Bad
Why doesn't this work?

# Good
Why doesn't this work?
[PASTE CODE]
[PASTE ERROR]
[PASTE RELEVANT DATA]
```

---

### Anti-Pattern 3: Asking for Approval vs Verification

```
# Bad
Is this paper good enough?

# Good
Check this paper for:
1. Claims not supported by evidence
2. Missing citations
3. Logical gaps in argument
4. Formatting issues

List each issue with location and suggested fix.
```

---

### Anti-Pattern 4: No Success Criteria

```
# Bad
Design experiments

# Good
Design experiments that:
1. Demonstrate [CORE CLAIM]
2. Include baselines [X, Y, Z]
3. Measure [METRICS]
4. Can run in [TIME] on [HARDWARE]
```

---

## Prompt Templates Library

### Template: Bug Report
```
**Bug**: [SHORT DESCRIPTION]

**Expected**: [WHAT SHOULD HAPPEN]

**Actual**: [WHAT HAPPENS]

**Steps to reproduce**:
1. [STEP 1]
2. [STEP 2]

**Relevant files**: [LIST]

**Error message** (if any):
```
[PASTE ERROR]
```
```

### Template: Code Request
```
**Task**: Implement [COMPONENT]

**Input**: [DESCRIPTION]

**Output**: [DESCRIPTION]

**Requirements**:
- [REQ 1]
- [REQ 2]

**Use patterns from**: [EXISTING FILES]

**Include**: Type hints, docstrings, error handling
```

### Template: Writing Request
```
**Write**: [SECTION NAME]

**Purpose**: [GOAL OF THIS SECTION]

**Include**:
- [POINT 1]
- [POINT 2]

**Data to reference**:
[PASTE RESULTS]

**Length**: ~[N] words
**Tone**: [FORMAL/TECHNICAL/ACCESSIBLE]
```

### Template: Review Request
```
**Review this [SECTION/CODE/DESIGN] for**:
1. [CONCERN 1]
2. [CONCERN 2]
3. [CONCERN 3]

**Content**:
[PASTE CONTENT]

**For each issue found, provide**:
- Location
- Problem description
- Suggested fix
```

