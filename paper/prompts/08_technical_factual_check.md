# Technical and Factual Accuracy Review

## Task

Review this paper for technical accuracy and factual correctness. The goal is to **detect and flag potential hallucinations, unsupported claims, or technical errors** that could undermine the paper's credibility.

---

## Files Provided

- `main.pdf` - Compiled paper
- `main.tex` - LaTeX source

---

## Checklist to Verify

### 1. Citation Accuracy

- [ ] **Citation claims match source**: When the paper says "X et al. showed Y", does the cited work actually show Y?
- [ ] **Year accuracy**: Are publication years correct?
- [ ] **Method descriptions**: Are baseline methods described accurately?
- [ ] **Statistic citations**: Are cited statistics (e.g., "3-6% of labels are incorrect") accurate?
- [ ] **No phantom citations**: Are all cited works real publications?

### 2. Mathematical Correctness

- [ ] **Equations well-defined**: Are all variables defined before use?
- [ ] **Dimensional consistency**: Do equations make mathematical sense?
- [ ] **Notation consistency**: Same symbol always means same thing?
- [ ] **Boundary conditions**: Are edge cases handled (division by zero, empty sets)?
- [ ] **Probability constraints**: Do probabilities sum to 1 where claimed?

### 3. Experimental Claims

- [ ] **Numbers internally consistent**: Do percentages, counts, and ratios match?
- [ ] **Improvement calculations**: Is "24% improvement" calculated correctly from 0.671 to 0.832?
- [ ] **Baseline fairness**: Are baselines given fair comparison (same data, same compute)?
- [ ] **Statistical significance**: Are claims of "better" or "outperforms" justified?
- [ ] **Reproducibility details**: Are hyperparameters, seeds, compute resources specified?

### 4. Method Description Accuracy

- [ ] **Algorithm matches description**: Does the described method match the implementation implied?
- [ ] **Hyperparameter consistency**: Same values in text, tables, and appendix?
- [ ] **Model names correct**: Are model names (RoBERTa-base, Qwen3-8B) accurate?
- [ ] **Library references**: Are tool references (FAISS, vLLM) accurate?

### 5. Claim Verification

For each major claim, verify it is either:
- Supported by experimental evidence in the paper
- Supported by a citation
- Clearly marked as hypothesis/speculation

Key claims to verify:
- [ ] "Cleanlab achieves only 0.107 AUROC" - supported by Table?
- [ ] "24% improvement over Input-kNN" - math checks out?
- [ ] "3-6% of labels in benchmarks are incorrect" - citation accurate?
- [ ] "Artifact-aligned noise makes mislabeled examples invisible to Cleanlab" - mechanism explained and demonstrated?

### 6. Logical Consistency

- [ ] **No contradictions**: Does the paper contradict itself anywhere?
- [ ] **Cause-effect claims**: Are causal claims justified or just correlational?
- [ ] **Generalization scope**: Are claims appropriately scoped (not over-generalized)?
- [ ] **Limitation acknowledgment**: Are known limitations honestly stated?

### 7. Common Hallucination Patterns

Watch for:
- [ ] **Fabricated statistics**: Round numbers or suspiciously precise claims without source
- [ ] **Misattributed ideas**: Claiming prior work said something it didn't
- [ ] **Non-existent methods**: Describing baseline methods that don't exist as described
- [ ] **Inflated improvements**: Selectively reporting best results
- [ ] **Assumed common knowledge**: Claiming things are "well-known" without citation

### 8. Technical Terminology

- [ ] **Terms used correctly**: AUROC, AUPRC, kNN, embedding, etc.
- [ ] **No malapropisms**: Similar-sounding terms confused
- [ ] **Consistent definitions**: Same term means same thing throughout

---

## Red Flags to Watch For

1. **Vague quantifiers**: "significantly", "substantially" without numbers
2. **Uncited strong claims**: "It is well known that..." without citation
3. **Perfect round numbers**: Suspiciously clean results (exactly 0.900)
4. **Missing error bars**: Point estimates without variance
5. **Cherry-picked comparisons**: Comparing to weak baselines only
6. **Mechanism hand-waving**: "This works because..." without evidence

---

## Output Format

### 1. Verification Summary
Overall assessment of factual accuracy

### 2. Citation Issues
List of citation claims that need verification or appear incorrect

### 3. Mathematical Issues
Any equations or calculations that appear incorrect

### 4. Unsupported Claims
Claims made without adequate evidence or citation

### 5. Internal Inconsistencies
Any contradictions within the paper

### 6. Potentially Fabricated Content
Any content that appears hallucinated or unverifiable

### 7. Recommended Fixes
Specific corrections needed, with suggested rewording where applicable

---

## Note on Scope

This review focuses on **internal consistency and verifiable facts**. It does not:
- Judge whether the experiments are well-designed
- Evaluate novelty or significance
- Assess whether the writing is clear

Those aspects are covered by other review prompts.

