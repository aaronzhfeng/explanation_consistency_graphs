# Figure 1 Design Request

## Task

Design a **TikZ figure** for an academic paper (ACL 2026). The figure should illustrate the ECG (Explanation-Consistency Graphs) pipeline.

**Style:** Academic, clean, professional.

---

## Architecture

ECG is a pipeline for detecting mislabeled training examples:

1. **Input:** Training data with potentially noisy labels $\{(x_i, y_i)\}$

2. **LLM Explanation:** Generate structured explanations using Qwen3-8B with JSON schema. Each explanation contains `evidence` (quoted text spans) and `rationale` (reasoning).

3. **Embed:** Encode the explanation text using a sentence encoder to get vectors $v_i$

4. **kNN Graph:** Build a k-nearest-neighbor graph (k=15) in explanation embedding space using FAISS

5. **Neighborhood Surprise:** Compute suspiciousness score $S_{\text{nbr}} = -\log p(y_i)$ — the negative log-probability of each label given its neighbors' labels

6. **Output:** Cleaned dataset (remove top-K% suspicious instances)

---

## Key Insight to Convey

The same kNN algorithm yields very different results depending on the embedding space:
- **Explanation-kNN:** 0.832 AUROC (our method)
- **Input-kNN:** 0.671 AUROC (same algorithm, input embeddings instead of explanation embeddings)
- **Cleanlab:** 0.107 AUROC (confidence-based baseline, fails on artifact noise)

The figure should somehow convey that the **embedding choice** is the key contribution (+24% improvement).

---

## Output

Complete TikZ code for a `figure*` environment (full-width, two-column format).

Assume these are already defined:
- `\usetikzlibrary{positioning, arrows.meta, calc}`
- `\newcommand{\Snbr}{S_{\text{nbr}}}`
