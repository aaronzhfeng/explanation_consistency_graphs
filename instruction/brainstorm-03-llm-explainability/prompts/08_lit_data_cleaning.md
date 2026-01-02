# Context

I'm developing a research project on **using LLM-generated explanations to detect mislabeled and artifact-laden training data in NLP**. The method (ECG: Explanation-Consistency Graphs) builds a kNN graph over explanation embeddings and uses inconsistency signals to rank suspicious training instances.

Our primary baseline is **Cleanlab** (confident learning). We claim ECG beats Cleanlab when classifiers confidently fit spurious markers (artifact-aligned noise) — a setting where loss-based methods fail.

# Seed Papers I Already Know

- **Cleanlab / Confident Learning** (Northcutt et al., 2021) — https://jair.org/index.php/jair/article/view/12125
- **Data Maps** (Swayamdipta et al., 2020) — characterizing training dynamics — arXiv:2009.10795

# What I Need

Please find **15-20 papers** on data cleaning and label noise detection for NLP/ML:

## 1. Confident Learning Extensions
- Papers citing or extending Cleanlab
- Critiques or failure modes of confident learning

## 2. Loss-Based Filtering
- Using high loss or low confidence to detect bad examples
- Co-teaching, MentorNet, sample reweighting

## 3. Influence Functions
- TracIn, influence functions for data debugging
- Fast approximations for large models

## 4. LLM-Based Label Checking
- Using LLMs to verify or correct labels
- LLM-as-annotator approaches

## 5. Data Cartography
- Training dynamics for finding hard/mislabeled examples
- Forgetting events, prediction flip analysis

## 6. Sample Weighting / Curriculum Learning
- Methods that down-weight noisy examples during training

For each paper:
- Title, authors, year, venue
- arXiv/DOI
- Key contribution (1-2 sentences)
- Code if available

Focus on 2020-2025 papers. Prioritize papers that address **artifact-aligned noise** or **spurious correlations** (not just uniform label flips).
