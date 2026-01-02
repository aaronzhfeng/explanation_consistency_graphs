# SOKRATES: Before/After Examples

> Actual AI outputs vs final polished versions

---

## Example 1: Abstract

### First AI Draft
```
Large language models (LLMs) frequently produce logically invalid 
chain-of-thought (CoT) reasoning even when their final answers are correct. 
We propose SOKRATES, a method that uses solver feedback to improve reasoning. 
SOKRATES represents proofs as sequences of discrete reasoning options and 
uses a first-order logic solver to verify each step. We train an option-success 
predictor and use Direct Preference Optimization (DPO) to align the model's 
policy with solver-induced preferences. Experiments show improvements in 
accuracy and validity.
```

**Issues:**
- No concrete numbers
- Vague "improvements"
- Missing the OaK connection
- No hook

### Final Version
```
A language model that achieves 94% accuracy on logical reasoning sounds 
impressive—until you discover that only 2% of its proofs are actually valid. 
This is the state of chain-of-thought prompting: models produce plausible 
rationales that frequently contain invalid inference steps, hidden contradictions, 
or skipped derivations. The right answer emerges despite, not because of, the 
reasoning process.

We introduce SOKRATES (Symbolic Option-Knowledge Reasoning Alignment via 
Trace Evaluation with Solver), a method that instantiates Sutton's Options 
and Knowledge (OaK) framework in a first-order logic micro-world. SOKRATES 
represents proofs as sequences of discrete inference-rule options (e.g., 
MODUS_PONENS, UNIV_INSTANTIATION), verified step-by-step by a FOL solver. 
From solver feedback we (i) train an option-success predictor that estimates 
validity before execution, and (ii) construct preference pairs for Direct 
Preference Optimization (DPO), aligning the model's option policy with 
solver-induced correctness.

On PrOntoQA, SOKRATES raises accuracy from 94.2% to 97.6%, step validity 
from 27.3% to 98.5%, and full-trace validity from 2.1% to 92.0%, a 44× 
improvement in logically sound proofs.
```

**Improvements:**
- Strong hook ("94% accuracy... only 2% valid")
- Concrete numbers
- Clear OaK framing
- Quantified improvement (44×)

---

## Example 2: Code — Data Loader

### First AI Draft
```python
def load_training_data(path):
    """Load training data from jsonl file."""
    data = []
    with open(path) as f:
        for line in f:
            example = json.loads(line)
            data.append(example)
    return data
```

**Issues:**
- No error handling
- No type hints
- Doesn't process into expected format
- No logging

### Final Version
```python
def load_training_data(
    path: str, 
    max_examples: Optional[int] = None
) -> List[TrainingExample]:
    """
    Load training data from JSONL file.
    
    Args:
        path: Path to JSONL file
        max_examples: Optional limit on examples to load
        
    Returns:
        List of TrainingExample objects
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If data format is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found: {path}")
    
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            try:
                raw = json.loads(line)
                example = TrainingExample(
                    problem_id=raw['problem_id'],
                    training_text=raw['training_text'],
                    label=raw['label']
                )
                data.append(example)
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Skipping malformed example {i}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} training examples from {path}")
    return data
```

**Improvements:**
- Type hints
- Comprehensive docstring
- Error handling
- Logging
- Proper data structure

---

## Example 3: Debugging Diagnosis

### First AI Attempt
```
The model might be learning something wrong. Try:
1. Check the data
2. Look at the loss
3. Maybe reduce learning rate
```

**Issues:**
- Vague suggestions
- No root cause analysis
- No specific actions

### Final Diagnosis (After Full Context)
```
Root Cause Analysis:

The data loader has a fallback that's always executing:

```python
# In scripts/train_sft.py:load_training_data()
if 'steps' in example:
    # Process structured steps
else:
    # THIS ALWAYS RUNS because data has 'training_text', not 'steps'
    steps = [ProofStep(type=CONCLUDE, thought="Based on premises...")]
```

Your training data uses `training_text` field, but the loader looks for `steps`.
The model is only learning to output CONCLUDE because that's all it sees.

Fix:
1. Add `TrainingTextWrapper` class to use `training_text` directly
2. Modify loader to check for `training_text` first
3. Re-run SFT with fixed data

Verification after fix:
- Generate 20 samples and manually inspect
- Should see multi-step reasoning with proper Thought/Action format
```

**Improvements:**
- Identified exact code location
- Explained why behavior occurs
- Provided specific fix
- Included verification step

---

## Example 4: Claim Strength

### First AI Draft
```
SOKRATES is the first system to instantiate Sutton's Options and Knowledge 
framework for LLM reasoning.
```

**Issues:**
- Too broad ("first" is risky)
- Vague about what's actually first
- Easy to attack in reviews

### Final Version
```
To our knowledge, SOKRATES is the first LLM reasoning system that 
(i) represents proofs as a fixed option vocabulary, 
(ii) learns an explicit option-success predictor, and 
(iii) aligns the option policy using solver-derived DPO preferences 
in an iterative loop.
```

**Improvements:**
- Qualified with "to our knowledge"
- Specific 3-part claim
- Each part is verifiable
- Harder to refute

---

## Example 5: Error Message to Fix

### Bug Report
```
Error when running multi-GPU training:

RuntimeError: Expected all tensors to be on the same device, 
but found at least two devices, cuda:0 and cuda:1!
```

### First AI Suggestion
```
Try removing device_map='auto' when using distributed training.
```

**Issues:**
- No explanation why
- No code fix provided
- No verification

### Final Fix
```
Problem: `device_map='auto'` conflicts with accelerate's DDP.

`device_map='auto'` spreads the model across devices for memory efficiency.
DDP expects each process to have the full model on its assigned device.
These are incompatible strategies.

Fix in scripts/train_sft.py:

```python
# Before (broken)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',  # Remove this!
    torch_dtype=torch.bfloat16
)

# After (working)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)
# Let accelerate handle device placement
```

Verification:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 scripts/train_sft.py
# Should run without device errors
```
```

**Improvements:**
- Explains root cause
- Provides exact code change
- Includes verification command

---

## Example 6: Table Formatting

### First AI Draft
```
| Model | Accuracy | Validity |
|-------|----------|----------|
| SFT | 93.3 | 11.3 |
| DPO 1 | 96.8 | 44.7 |
| DPO 2 | 98.1 | 83.5 |
| DPO 3 | 98.2 | 91.8 |
```

**Issues:**
- No units (%)
- Missing baseline
- No highlighting of best
- Inconsistent naming

### Final Version
```
| Model | Accuracy (%) | Step Validity (%) | Trace Validity (%) |
|-------|--------------|-------------------|-------------------|
| Base (Qwen3-8B) | 85.0 | — | — |
| + SFT | 94.2 | 27.3 | 2.1 |
| + OaK-DPO iter 1 | 96.5 | 69.4 | 34.2 |
| + OaK-DPO iter 2 | 97.6 | 87.3 | 73.8 |
| + OaK-DPO iter 3 | **97.6** | **98.5** | **92.0** |
```

**Improvements:**
- Clear units in headers
- Baseline included
- Bold best results
- Consistent naming with "OaK-DPO"
- Added Trace Validity metric

---

## Transformation Patterns

### Pattern: Vague → Specific
```
Before: "improves performance"
After:  "raises accuracy from 94.2% to 97.6% and trace validity from 2.1% to 92.0%"
```

### Pattern: Claim → Evidence
```
Before: "is the first to..."
After:  "is the first LLM reasoning system that (i)..., (ii)..., (iii)..."
```

### Pattern: Code → Production Code
```
Before: def func(x): return x+1
After:  Type hints, docstrings, error handling, logging
```

### Pattern: Suggestion → Action
```
Before: "maybe try X"
After:  "Change line 42 in file.py from A to B, then run command C to verify"
```

---

## Iteration Count

| Component | Drafts | Final |
|-----------|--------|-------|
| Abstract | 4 | 5th |
| Introduction | 3 | 4th |
| Method section | 2 | 3rd |
| Main results table | 2 | 3rd |
| Data loader code | 3 | 4th |
| Training script | 2 | 3rd |

**Takeaway:** Nothing is perfect on first try. Plan for 2-4 iterations on everything.

