# SOKRATES: Lessons Learned

> What we learned from the speedrun

---

## Technical Lessons

### 1. Low Loss ≠ Good Model

**The Bug**: Training loss dropped from 3.5 → 0.01, but the model only learned to output "CONCLUDE".

**Root Cause**: The data loader wasn't using the right field. The model was training on trivial one-step examples.

**Lesson**: Always test generation quality, not just loss curves. A model can "learn" to minimize loss on broken data.

**Prevention**:
```bash
# After SFT, before DPO, always run:
python scripts/generate.py --model outputs/sft --num-samples 20
# Manually inspect the outputs!
```

---

### 2. Verify Data Loading First

**The Pattern**: Most bugs were in the data pipeline, not the model or training code.

**Issues Found**:
- Wrong JSON field being read
- Predicate format instead of natural language
- Hardcoded indices `[0, 1]` instead of actual premises
- Tokenization edge cases

**Lesson**: Data is the foundation. If data is wrong, everything downstream is wrong.

**Prevention**:
```python
# Add data sanity checks
def verify_data(dataset):
    for i, example in enumerate(dataset[:10]):
        print(f"Example {i}:")
        print(f"  Input: {example['input'][:100]}...")
        print(f"  Output: {example['output'][:100]}...")
        assert len(example['input']) > 50, "Input too short"
        assert "Thought:" in example['output'], "Missing Thought"
```

---

### 3. Test Generation Before Expensive Training

**The Anti-Pattern**: Run SFT → Run DPO iter 1 → Run DPO iter 2 → Finally test generation → Discover everything is broken.

**The Pattern**: Run SFT → Test generation → Fix issues → Then continue.

**Time Saved**: Caught bugs after 10 minutes of training instead of after 2 hours.

---

### 4. Multi-GPU Requires Planning

**The Bug**: 6 GPUs available, but training was only as fast as 1 GPU.

**Root Causes**:
- Each GPU processed all data redundantly
- `device_map='auto'` conflicted with accelerate DDP
- No work splitting in trace generation

**Lesson**: Multi-GPU doesn't automatically mean faster. You need explicit parallelization.

**Key Fixes**:
```python
# Work splitting
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
my_data = all_data[rank::world_size]

# No device_map with accelerate
model = AutoModel.from_pretrained(model_path)  # Not device_map='auto'
```

---

### 5. Temperature Matters for DPO

**The Bug**: DPO wasn't improving the model.

**Root Cause**: Temperature was 0 (greedy decoding), so all samples per problem were identical.

**Why It Matters**: DPO needs diverse samples to create preference pairs. If chosen and rejected are identical, there's nothing to learn.

**Fix**: Temperature 0.5-0.7 for trace generation.

---

### 6. vLLM is Much Faster

**The Numbers**:
- HuggingFace generate(): ~50 problems/hour
- vLLM: ~2,000 problems/hour
- Speedup: **40×**

**Lesson**: For any serious inference workload, use vLLM or similar optimized serving.

---

## Process Lessons

### 7. Document Everything in Real-Time

**What We Did**: Created session logs (`09_session_log.md`, `14_debugging_session_dec10.md`) as we worked.

**Why It Helped**:
- Could look up what we tried
- AI agent had context for debugging
- Paper writing was faster (material ready)

**Template**:
```markdown
## Session: [DATE]
### What worked
### What didn't
### Issues found
### Files changed
### Next steps
```

---

### 8. AI Agents Need Context

**What Didn't Work**: "Fix the bug"

**What Worked**: 
```
The model trained with low loss but generates garbage.
Symptoms: [detailed symptoms]
Code location: [file paths]
What we've tried: [list]
```

**Lesson**: The AI can only help if you describe the problem fully. Include symptoms, context, and what you've already tried.

---

### 9. Principled Trade-offs Under Time Pressure

**The Situation**: 6 hours until deadline, training not done.

**What We Did**:
- Reduced dataset size 14K → 1.5K (still statistically significant)
- Reduced iterations 3 → 2 (still shows improvement)
- Documented all trade-offs in the paper

**Lesson**: When time runs out, make cuts systematically and document them. Reviewers appreciate honesty about limitations.

---

### 10. Iterate on Paper, Not Just Code

**What We Did**: 4 revision cycles on the paper.

| Cycle | Focus |
|-------|-------|
| 1 | Structure and content |
| 2 | Claims vs evidence |
| 3 | Writing clarity |
| 4 | Final polish |

**Lesson**: Don't try to write the perfect paper in one pass. Each revision catches different issues.

---

## Project Management Lessons

### 11. Start Writing Early

**Anti-Pattern**: Finish all experiments → Start writing paper → Rush to deadline.

**Pattern**: Write paper outline while experimenting → Fill in as results come → Polish at end.

**Benefits**:
- Clarifies what experiments you actually need
- Catches gaps in the story early
- Reduces deadline stress

---

### 12. Build for the Core Contribution

**Temptation**: Add more baselines, more ablations, more analysis.

**Reality**: Reviewers care most about:
1. Is the core idea novel?
2. Does it work?
3. Is it explained clearly?

**Lesson**: Focus on demonstrating the main contribution. Extra experiments can come in camera-ready.

---

### 13. Leave Buffer for Submission

**What Almost Went Wrong**:
- TL;DR exceeded character limit (fixed in last hour)
- Figure placement issues (fixed day before)
- Reference year errors (caught in final check)

**Lesson**: Submit 24+ hours before deadline. Last-minute issues always appear.

---

## AI Collaboration Lessons

### 14. AI as Researcher, Human as Director

**What the AI Did**:
- Generated all code
- Debugged issues
- Wrote the paper
- Found and cited references

**What the Human Did**:
- Set direction ("OaK for AAAI")
- Made key decisions at branch points
- Provided compute
- Clicked submit

**Lesson**: This isn't "AI writes my paper." It's "AI executes, human directs."

---

### 15. Verify AI Claims

**What We Found**: AI generated "33× improvement" but actual calculation was 44×.

**What We Did**: Verified every number against actual results.

**Lesson**: AI is helpful but not infallible. Always verify quantitative claims.

---

### 16. Use AI for What It's Good At

**AI Strengths**:
- Generating boilerplate code
- Finding patterns in docs
- Writing structured content
- Debugging with full context

**Human Strengths**:
- Setting research direction
- Making taste judgments
- Deciding what's interesting
- Evaluating novelty

**Lesson**: Play to strengths. Don't ask AI to decide if your paper is novel enough. Do ask AI to implement the idea and write it up.

---

## Summary: Top 5 Takeaways

1. **Test generation early** — Don't trust loss curves alone
2. **Data pipeline first** — Most bugs are in data loading
3. **Document everything** — Real-time logs save debugging time
4. **Principled trade-offs** — When time runs out, cut systematically
5. **Iterate on paper** — Multiple revision cycles catch different issues

