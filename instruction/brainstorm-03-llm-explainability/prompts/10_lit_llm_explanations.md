# Context

I'm developing a research project on **using LLM-generated explanations to detect mislabeled training data**. The method generates structured explanations (JSON format: rationale, evidence spans, predicted label, confidence) for each training instance using a 7-8B instruction-tuned LLM (e.g., Llama-3.1-8B-Instruct).

Key requirement: explanations must be **structured and parseable** for downstream graph construction and NLI-based contradiction detection.

# Seed Papers I Already Know

- **Chain-of-Thought Prompting** (Wei et al., 2022) — arXiv:2201.11903
- **Self-Consistency** (Wang et al., 2023) — arXiv:2203.11171

# What I Need

Please find **15-20 papers** on LLM-generated explanations and structured rationale generation:

## 1. Structured Output Generation
- JSON, XML, or constrained decoding for explanations
- Forcing LLMs to output parseable formats

## 2. LLM-as-Judge / LLM-as-Annotator
- Using LLMs to label or verify data
- Reliability of LLM annotations

## 3. Rationale Extraction vs Generation
- Extractive (highlight spans) vs generative (free-text) explanations
- Trade-offs between the two

## 4. Explanation Verification
- Checking if LLM explanations are reliable/faithful
- Self-consistency for rationales

## 5. Explanation Calibration
- Whether LLM confidence in explanations is well-calibrated
- Uncertainty in generated rationales

## 6. Prompting for Faithful Explanations
- Techniques to improve explanation quality
- Instruction tuning for rationales

For each paper:
- Title, authors, year, venue
- arXiv/DOI
- Key contribution (1-2 sentences)
- Code if available

Focus on 2022-2025 papers. Prioritize papers using **open-weight models** (Llama, Mistral, Qwen) and evaluating explanation quality.
