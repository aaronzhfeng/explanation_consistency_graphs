# SOKRATES: What Didn't Work

> Failures, dead ends, and course corrections

---

## Critical Failures (Would Have Killed the Paper)

### Failure 1: The Silent Data Bug

**What happened:**
- SFT training completed successfully
- Loss dropped from 3.5 → 0.01
- Everything looked great
- Model output was complete garbage

**Root cause:**
```python
# The data loader had this code:
if 'steps' in example:
    # Process steps...
else:
    # Create minimal CONCLUDE-only step  ← THIS ALWAYS RAN
    steps = [ProofStep(type=CONCLUDE, ...)]
```

Our data had `training_text`, not `steps`. The loader silently fell back to trivial examples.

**Why it almost killed us:**
- No error messages
- Loss looked good
- Wasted ~2 hours before discovery
- If not caught, would have submitted broken results

**Lesson:** Low loss ≠ correct learning. Always test generation.

---

### Failure 2: Wrong Premise Indices

**What happened:**
- Optionizer was generating `args="[0, 1]"` for every step
- Model learned to always cite premises 0 and 1
- Traces were syntactically valid but semantically wrong

**Root cause:**
```python
# Original code
def _infer_option_from_text(self, text, formulas):
    # This was placeholder that never got updated
    return OptionType.MODUS_PONENS, [0, 1]  # Always!
```

**Why it mattered:**
- Model "learned" a shortcut
- Accuracy was okay because it guessed right sometimes
- But reasoning was completely fake

**Lesson:** Placeholder code gets forgotten. Delete or implement.

---

### Failure 3: Predicate-Style Thoughts

**What happened:**
```
# Training data showed:
Thought: Nervous('Wren', True)
Action: <Option type="MODUS_PONENS" args="[0, 1]" />
```

**What we needed:**
```
Thought: Since Wren is a jompus (premise 0) and every jompus is nervous (premise 1), 
we can conclude Wren is nervous.
Action: <Option type="MODUS_PONENS" args="[0, 1]" />
```

**Why it failed:**
- LLMs don't naturally produce predicate notation
- Training on predicates → model confused about format
- Generation was garbled mix of formats

**Lesson:** Train in the format you want to generate.

---

## Moderate Failures (Cost Time but Recovered)

### Failure 4: Greedy Decoding for DPO

**What happened:**
- Generated traces for DPO with temperature=0
- All samples for each problem were identical
- DPO had nothing to compare

**Why it's bad:**
```
Problem: "Is Wren nervous?"
Sample 1: [CONCLUDE TRUE] 
Sample 2: [CONCLUDE TRUE]  ← Identical!
```

DPO needs diversity: valid vs invalid traces. Identical traces = useless.

**Fix:** Temperature 0.5-0.7 for trace generation.

**Time lost:** 1 hour of useless trace generation.

---

### Failure 5: Multi-GPU Not Actually Parallel

**What happened:**
- Launched on 6 GPUs
- Expected 6× speedup
- Got 1× speed (same as single GPU)

**Root cause:**
```python
# Each GPU was processing ALL problems
for problem in all_problems:  # Not split!
    trace = generate(problem)
```

**Fix:**
```python
# Split work across GPUs
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
my_problems = all_problems[rank::world_size]
```

**Time lost:** 2 hours of "parallel" training that wasn't.

---

### Failure 6: device_map Conflict with DDP

**What happened:**
```python
model = AutoModel.from_pretrained(..., device_map='auto')
# + accelerate launch with DDP
```

**Error:**
```
RuntimeError: Expected all tensors to be on the same device
```

**Why:** `device_map='auto'` spreads model across devices, but DDP expects model on single device per process.

**Fix:** Remove `device_map='auto'` when using accelerate distributed.

**Time lost:** 30 minutes of cryptic error messages.

---

## Minor Failures (Annoyances)

### Failure 7: Figure Placement

**What happened:**
- Figure 1 kept appearing on page 3
- Wanted it on page 1-2 for visual impact

**Attempts that didn't work:**
- `[h]` specifier
- `[ht]` specifier
- `\FloatBarrier`

**What worked:**
- Move `\input{figure}` to right after abstract
- Use `[t]` specifier

**Time lost:** 30 minutes fighting LaTeX.

---

### Failure 8: TL;DR Character Limit

**What happened:**
- Wrote great TL;DR
- Exceeded 250 character limit by 10 characters
- Discovered 1 hour before deadline

**Fix:** Shortened quickly. Could have been worse.

**Lesson:** Check venue requirements early, not at submission time.

---

### Failure 9: Reference Year Errors

**What happened:**
- AI generated citations with wrong years
- `clark2021transformers` → actually 2020
- `yao2024tree` → actually 2023

**Why it matters:** Reviewers notice. Looks sloppy.

**Fix:** Manually verified all 29 references against actual papers.

**Time lost:** 1.5 hours of verification.

---

## Things We Tried That Didn't Help

### Abandoned Approach 1: Constrained Decoding

**Idea:** Force model to generate valid option syntax.

**Why abandoned:**
- Added complexity
- Model learned format from SFT anyway
- Validation overhead slowed inference

**Verdict:** Not needed for this project. Maybe useful for stricter domains.

---

### Abandoned Approach 2: Option-Success Head (q̂)

**Idea:** Train explicit predictor for step validity.

**Why abandoned:**
- Time constraints
- Main results didn't need it
- Can add in camera-ready if needed

**Verdict:** Good idea, wrong priority for deadline.

---

### Abandoned Approach 3: GVF Auxiliary Heads

**Idea:** Train "reward-respecting subtask" heads per OaK theory.

**Why abandoned:**
- Nice theoretical alignment
- But not necessary for core results
- Would complicate the story

**Verdict:** Future work, explicitly noted in paper.

---

## Failure Patterns

### 1. Silent Failures Are Worst
- No error message
- Metrics look fine
- Results are garbage

**Prevention:** Test actual outputs, not just loss curves.

### 2. Placeholder Code Gets Shipped
- "TODO: implement properly"
- Never gets done
- Causes subtle bugs

**Prevention:** Delete placeholders or make them crash loudly.

### 3. Parallelism Doesn't Auto-Scale
- Adding GPUs ≠ automatic speedup
- Need explicit work distribution

**Prevention:** Measure actual throughput, not GPU count.

### 4. Format Training = Format Generation
- Train on predicates → generate predicates
- Train on natural language → generate natural language

**Prevention:** Training data should look like desired output.

---

## Time Lost to Failures

| Failure Category | Time Lost |
|------------------|-----------|
| Data pipeline bugs | 3h |
| Multi-GPU issues | 2.5h |
| Figure/formatting | 1h |
| Reference validation | 1.5h |
| Abandoned approaches | 2h |
| **Total** | **~10h** |

This is 26% of total project time (38h). Could have been 28h with perfect execution, but failures are inevitable.

---

## What We'd Do Differently

1. **Test generation after every training run** — Not just loss
2. **Use loud failures** — Assertions, not silent fallbacks
3. **Plan multi-GPU from start** — Not retrofit later
4. **Verify references during writing** — Not at submission
5. **Check venue requirements on day 1** — Not day N

