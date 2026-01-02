# Literature-Informed Proposal Review

## Purpose

Send this prompt (with `literature.md` and `proposals/02_ecg.md` attached) to multiple AI models to get independent observations on how the literature should inform/update the proposal.

**Recommended models:** GPT-5.2 Pro, Claude Opus, Gemini Ultra, DeepSeek-V3

---

## Prompt

```
You are given two documents for a research project:

1. **literature.md** — A curated collection of 103 papers across 8 categories relevant to the project
2. **proposals/02_ecg.md** — The current research proposal for "Explanation-Consistency Graphs (ECG)"

## Your Task

Analyze the literature collection against the proposal and provide:

### 1. Literature Coverage Assessment
- Which aspects of the proposal are well-supported by the literature?
- Are there any gaps where the proposal makes assumptions not backed by cited work?
- Are there relevant papers in the collection that the proposal doesn't leverage?

### 2. Methodology Refinements
Based on the literature, identify specific improvements or alternatives for:
- Explanation generation approach
- Graph construction methodology
- Signal combination and scoring
- Baseline selection
- Evaluation metrics

### 3. Risk Analysis
What does the literature suggest about potential failure modes or challenges the proposal doesn't address?

### 4. New Opportunities
Are there ideas in the literature that could strengthen the proposal's novelty or impact?

### 5. Updated Proposal Sections (if warranted)
For any sections that should change based on the literature, provide concrete rewrites.

---

Be specific. Cite paper numbers (e.g., #48, #52) when referencing the literature.
Do not assume the proposal is optimal—look for genuine improvements.
```

---

## Attachments Required

1. `literature.md` (full file, ~705 lines)
2. `proposals/02_ecg.md` (full file, ~785 lines)

---

## Expected Output

Each model should produce:
- A structured analysis following the 5 sections above
- Specific paper citations from the literature collection
- Concrete suggestions (not vague recommendations)

## Post-Review Process

After collecting responses from multiple models:
1. Compare observations across models
2. Identify consensus improvements
3. Note divergent perspectives for further investigation
4. Update proposal with validated improvements

