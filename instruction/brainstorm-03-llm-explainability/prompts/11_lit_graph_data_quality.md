# Context

I'm developing a research project on **using LLM-generated explanations to detect mislabeled training data**. A core component is building a **kNN graph over explanation embeddings** and computing neighborhood-based inconsistency signals.

Key idea: if a node's label disagrees with its explanation-similar neighbors, it's suspicious. This is different from building a graph over raw input embeddings.

# Seed Papers I Already Know

- **Sentence-BERT** (Reimers & Gurevych, 2019) — for embedding explanations — arXiv:1908.10084
- **FAISS** (Johnson et al., 2019) — for efficient kNN search — arXiv:1702.08734

# What I Need

Please find **15-20 papers** on graph-based methods for data quality and embedding-based denoising:

## 1. kNN-Based Label Noise Detection
- Using embedding neighborhoods to find mislabeled examples
- Neighborhood consistency methods

## 2. Graph Neural Networks for Data Cleaning
- GNN-based approaches to data quality
- Label propagation on data graphs

## 3. Embedding Clustering for Outlier Detection
- Using representation similarity to find anomalies
- Prototype-based noise detection

## 4. Contrastive Learning for Data Quality
- Using contrastive representations to identify noise
- Self-supervised signals for data cleaning

## 5. Label Propagation / Graph Semi-Supervised Learning
- Smoothing labels over graphs
- Transductive methods for noise robustness

For each paper:
- Title, authors, year, venue
- arXiv/DOI
- Key contribution (1-2 sentences)
- Code if available

Focus on 2019-2025 papers. Prioritize papers that apply to **NLP/text classification** (not just vision) and compare to Cleanlab or similar baselines.
