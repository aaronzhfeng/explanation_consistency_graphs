# SOKRATES: Key Prompts Used

> Actual prompts that drove the project

---

## Phase 0: Ideation

### Initial Direction
```
I want to submit something to the AAAI-26 Bridge Workshop on 
"Logical and Symbolic Reasoning in Language Models".

I'm interested in connecting Sutton's Options and Knowledge (OaK) 
framework to LLM reasoning.

What could I do that's:
1. Novel enough for a workshop
2. Implementable in 4-5 weeks
3. Uses available benchmarks (FOLIO, PrOntoQA)
```

### AI Response (summarized)
- Proposed SOKRATES: option-level reasoning with solver verification
- Identified 3 differentiators vs prior work
- Suggested DPO for alignment
- Outlined implementation plan

---

## Phase 1: Design

### Project Plan Request
```
Design the full research project for SOKRATES.

Include:
1. Title with acronym
2. Abstract (~200 words)
3. Key contributions (3-5 bullets)
4. Experimental design (datasets, baselines, metrics)
5. Implementation timeline
6. Risk assessment

Use the OaK framing seriously - make sure we have:
- Explicit options (not just CoT)
- Explicit knowledge (learned predictor)
- Iterative loop (not one-shot)
```

### Technical Spec Request
```
Now create the full technical specification.

Include:
1. Repository structure
2. Core data structures (Python dataclasses)
3. Module specifications (what each file does)
4. Configuration schemas (YAML)
5. Key algorithms (pseudocode)
6. API contracts
7. Testing strategy
```

---

## Phase 2: Implementation

### Code Generation
```
Implement the SOKRATES data module.

Requirements:
- Load PrOntoQA and FOLIO datasets
- Convert to optionized format (Thought/Action)
- Support both training and evaluation

Use the data structures from the technical spec.
Include proper error handling and logging.
```

### Debugging Session
```
The model trained with low loss but generates garbage.

Symptoms:
- Only produces "CONCLUDE" step
- Thoughts are "<Thought>" literal string
- All premise indices are [0, 1]

Training loss went from 3.5 → 0.01.

What's wrong? Check:
1. Data loading code
2. Training data format
3. Optionizer output
```

### AI Diagnosis (summarized)
- Found 4 bugs in data pipeline
- Data loader wasn't using `training_text` field
- Optionizer was generating predicate-style thoughts
- Fixed and documented all issues

---

## Phase 3: Training

### Multi-GPU Optimization
```
Training is too slow. Currently:
- 6 GPUs available
- But each GPU processes all 14K problems
- Trace generation is sequential

How do I:
1. Split work across GPUs
2. Batch trace generation
3. Use vLLM for faster inference
```

### Time Crunch Trade-offs
```
I have 6 hours until deadline and training isn't done.

Current plan:
- 3 DPO iterations
- 14K problems
- 8 samples per problem

What can I cut while maintaining scientific validity?
Document all trade-offs for the paper.
```

### AI Recommendations (summarized)
- Reduce to 1.5K problems (still statistically significant)
- Reduce to 2 samples (minimum for DPO pairs)
- Keep 2-3 iterations (shows improvement curve)
- Document all changes for reproducibility

---

## Phase 4: Writing

### First Draft
```
Write the full paper for SOKRATES.

Structure:
1. Introduction (motivation, contributions)
2. Background (OaK, DPO, benchmarks)
3. Method (options, knowledge, training loop)
4. Experiments (setup, main results, ablations)
5. Conclusion

Use these results:
[pasted results tables]

Target: AAAI workshop format, 8 pages + refs + appendix
```

### Revision Request
```
Review the paper and suggest improvements.

Focus on:
1. Claim strength vs evidence
2. Writing clarity
3. Missing citations
4. Potential reviewer objections
```

### Final Polish
```
Final submission check:

1. Verify all numbers match tables
2. Check all citations are valid
3. Fix any formatting issues
4. Ensure figure placement is good
5. Create single-file LaTeX version

Flag any issues that could cause desk reject.
```

---

## Effective Prompting Patterns

### 1. Be Specific About Output
```
# Bad
"Design a project"

# Good
"Design a project including:
1. Title with acronym
2. Abstract (~200 words)
3. Key contributions (3-5 bullets)
..."
```

### 2. Provide Context
```
# Bad
"Fix the bug"

# Good
"The model trained with low loss but generates garbage.
Symptoms: [specific symptoms]
Training loss: [actual numbers]
What's wrong?"
```

### 3. Reference Existing Work
```
# Bad
"Create the training code"

# Good
"Implement training using the data structures from 
technical_spec.md and the config schema from configs/training.yaml"
```

### 4. Ask for Trade-off Analysis
```
# Bad
"Speed it up"

# Good
"I have 6 hours. What can I cut while maintaining 
scientific validity? Document all trade-offs."
```

### 5. Request Verification Steps
```
# Bad
"Check if it works"

# Good
"How do I verify:
1. Data loading is correct
2. Model is learning the right thing
3. Metrics are computed properly
Give me specific commands/code to run."
```

