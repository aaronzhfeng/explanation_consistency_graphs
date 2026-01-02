# Context

I'm developing a research project on **using LLM-generated explanations to detect artifact-laden training data**. Our key experiment uses **artifact-aligned label noise**: we flip labels AND inject a spurious marker token that makes classifiers confidently fit the wrong label.

This creates a setting where **loss-based methods fail** (low loss on marker-driven predictions) but explanation-based methods can still detect inconsistency.

# Seed Papers I Already Know

- **Annotation Artifacts in NLI** (Gururangan et al., 2018) — arXiv:1803.02324
- **Right for the Wrong Reasons** (McCoy et al., 2019) — arXiv:1902.01007
- **Identifying Spurious Correlations** (NAACL 2022) — https://aclanthology.org/2022.findings-naacl.130.pdf

# What I Need

Please find **15-20 papers** on spurious correlations, annotation artifacts, and shortcut learning in NLP:

## 1. Annotation Artifacts in NLI
- Hypothesis-only baselines
- Lexical cues that predict labels

## 2. Spurious Correlations in Sentiment Analysis
- Dataset-specific shortcuts (movie names, ratings, etc.)
- SST-2 artifacts

## 3. Shortcut Learning Detection
- Methods to identify which features are shortcuts
- Probing for spurious correlations

## 4. Mitigation Methods
- Debiasing, reweighting, adversarial training
- Product-of-experts approaches

## 5. Challenge Sets / Stress Tests
- Evaluation sets designed to expose shortcut reliance
- Counterfactual evaluation

## 6. Group Robustness
- Worst-group accuracy
- Distributionally robust optimization

For each paper:
- Title, authors, year, venue
- arXiv/DOI
- Key contribution (1-2 sentences)
- Code if available

Focus on 2018-2025 papers. Prioritize papers studying **SST-2, SNLI, MNLI, or FEVER** (our target datasets).
