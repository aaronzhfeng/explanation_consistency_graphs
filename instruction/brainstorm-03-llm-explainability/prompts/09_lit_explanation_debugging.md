# Context

I'm developing a research project on **using LLM-generated explanations to detect mislabeled training data in NLP**. The method (ECG: Explanation-Consistency Graphs) uses explanation structure — not just loss/confidence — to identify data problems.

A key novelty claim: when models confidently fit spurious markers, loss-based methods fail, but **explanation-based signals** can still reveal inconsistency.

# Seed Papers I Already Know

- **Explanation-Based Human Debugging** (Lertvittayakumjorn & Toni, 2021) — TACL survey — https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00440/108932
- **Interactive Label Cleaning with Explanations** (NeurIPS 2021) — https://proceedings.neurips.cc/paper/2021/file/6c349155b122aa8ad5c877007e05f24f-Paper.pdf

# What I Need

Please find **15-20 papers** on explanation-based debugging and using interpretability for data quality:

## 1. Explanation-Based Human Debugging
- Using explanations to help humans find model errors or data issues
- Human-in-the-loop correction with explanations

## 2. Using Interpretability to Find Spurious Correlations
- Saliency/attention analysis to detect shortcut learning
- Probing for artifacts

## 3. Explanation-Guided Data Augmentation or Reweighting
- Using explanations to inform training data selection
- Rationale-based data curation

## 4. Rationale-Based Active Learning
- Selecting examples based on explanation quality or disagreement

## 5. Debugging with Natural Language Feedback
- Using NL explanations to diagnose model failures
- Error analysis via explanations

For each paper:
- Title, authors, year, venue
- arXiv/DOI
- Key contribution (1-2 sentences)
- Code if available

Focus on 2018-2025 papers. Prioritize papers that use explanations as a **signal for data quality** (not just model understanding).
