# Context

I'm developing a research project on **using LLM-generated explanations to detect mislabeled training data**. As a downstream evaluation, we measure whether data cleaning improves **explanation faithfulness**.

We use LLM-generated evidence spans as rationales and compute comprehensiveness/sufficiency metrics to see if the cleaned model's decisions better align with semantic explanations (rather than spurious markers).

# Seed Papers I Already Know

- **ERASER Benchmark** (DeYoung et al., 2020) — https://www.eraserbenchmark.com/
- **Towards Faithfully Interpretable NLP** (Jacovi & Goldberg, 2020) — arXiv:2004.03685

# What I Need

Please find **15-20 papers** on explanation faithfulness and rationale evaluation:

## 1. Comprehensiveness and Sufficiency
- Masking rationale tokens to measure their importance
- Original definitions and improvements

## 2. ERASER and Related Benchmarks
- ERASER benchmark extensions
- Other rationale evaluation benchmarks

## 3. Faithfulness vs Plausibility
- Distinguishing whether explanations reflect model reasoning vs human expectations
- Metrics for each

## 4. Post-hoc Explanation Evaluation
- Evaluating saliency maps, attention, integrated gradients
- Sanity checks for explanations

## 5. Free-Text Rationale Evaluation
- Metrics for natural language explanations (not just highlights)
- REV, RORA, and similar metrics

## 6. Simulatability
- Whether humans can predict model behavior from explanations
- Human evaluation protocols

For each paper:
- Title, authors, year, venue
- arXiv/DOI
- Key contribution (1-2 sentences)
- Code if available

Focus on 2019-2025 papers. Prioritize papers that provide **code for computing faithfulness metrics** and compare multiple explanation methods.
