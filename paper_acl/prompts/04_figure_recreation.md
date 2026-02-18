# ECG Architecture Figure - Recreation from Reference Image

## Task

Recreate the following **3-panel architecture diagram** as a TikZ standalone figure for an academic paper (ACL 2026). The figure illustrates the ECG (Explanation-Consistency Graphs) pipeline for detecting mislabeled training examples.

**Output format:** Complete TikZ code for a `standalone` document class.

---

## Overall Layout

The figure consists of **three horizontal panels** arranged left-to-right, connected by large arrows:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  1. Explanation     │ ══> │  2. Explanation     │ ══> │  3. Neighborhood    │
│     Generation      │     │     Embedding &     │     │     Surprise        │
│                     │     │     Graph Constr.   │     │     Detection       │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

Each panel has:
- A **colored header bar** at the top with white bold text (panel title)
- A **light-colored background** matching the header theme
- Internal components connected by arrows

---

## Color Scheme

| Panel | Header Color | Background Color |
|-------|--------------|------------------|
| Panel 1 | Steel Blue (~RGB 43, 101, 152) | Light blue (~RGB 229, 242, 255) |
| Panel 2 | Burnt Orange (~RGB 198, 112, 26) | Light orange (~RGB 255, 239, 224) |
| Panel 3 | Deep Purple (~RGB 106, 58, 126) | Light purple (~RGB 244, 232, 252) |

Additional colors:
- **Green** for the Structured Explanation (JSON) box: ~RGB 219, 240, 226

---

## Panel 1: Explanation Generation

### Components (left to right):

1. **Training Data Icon** (left side)
   - Visual: Stack of 3 overlapping document icons with horizontal lines representing text
   - Label below: "Training Data" and "$(x_i, y_i)$"

2. **LLM Explanation Generation Box** (center)
   - White rounded rectangle with blue border
   - Title: "LLM Explanation Generation (Qwen3-8B)"
   - Contains 3 stacked mini-boxes: "Sample 1", "Sample 2", "Sample 3"
   - Side annotation: "Stability Sampling (M=3) & Reliability Score $(r_i)$"

3. **Structured Explanation Box** (right side)
   - Rounded rectangle with **green background**
   - Title: "Structured Explanation (JSON)"
   - Lists fields vertically:
     - `pred_label`
     - `evidence`
     - `rationale`
     - `counterfactual`
     - `confidence`
   - Small box at bottom: "Reliability Score $(r_i)$"

### Arrows:
- Training Data → LLM Box
- LLM Box → Structured Explanation

---

## Panel 2: Explanation Embedding & Graph Construction

### Components:

1. **Top Row: Processing Pipeline** (3 boxes, left to right)
   - Box 1: "Canonical String Formation $(t_i)$"
   - Box 2: "Sentence Encoder (all-MiniLM-L6-v2)"
   - Box 3: "Explanation Embeddings $(v_i)$"
   - Connected by arrows: Box1 → Box2 → Box3

2. **Center: kNN Graph Visualization**
   - Large circle (white fill, orange border)
   - Contains ~7 small circular nodes arranged in a network pattern
   - One node slightly emphasized (center node)
   - Edges between nodes with **varying line thickness**:
     - Thick lines = high weight
     - Thin lines = low weight
   - Caption below: "Reliability-Weighted kNN Graph"

3. **Annotations**
   - "High weight $(w_{ij})$" with thick line sample
   - "Low weight" with thin line sample
   - Side text: "Edge weights $w_{ij}$ incorporate similarity $s_{ij}$ and neighbor reliability $r_j$"

### Arrows:
- Embeddings box (top) → kNN Graph (center)

---

## Panel 3: Neighborhood Surprise Detection

### Components:

1. **Top: Small Star Graph**
   - Central node labeled "$i$"
   - 6 surrounding nodes, each labeled "$y_j$"
   - Edges connecting center to all surrounding nodes
   - Shows the "neighborhood" concept visually

2. **Middle: Formula Boxes** (stacked vertically)
   
   **Box 1: Weighted neighbor label posterior**
   ```
   p_i(c) = Σ_j w_{ij} · 1[y_j = c]
   ```
   
   **Box 2: Neighborhood Surprise**
   ```
   (S_nbr) = -log p_i(y_i)
   ```

3. **Right of Box 2: "Flagged" Badge**
   - Small rounded rectangle
   - Text: "Flagged"
   - Slightly rotated (stamp-like appearance, optional)
   - Connected by arrow from Surprise box

4. **Bottom: Output Label**
   - "Suspiciousness Ranking & Flagged Instances"
   - Small document/list icon (optional)

---

## Cross-Panel Arrows

Large, bold arrows connecting the panels horizontally:
- Panel 1 (right edge, vertical center) **→** Panel 2 (left edge, vertical center)
- Panel 2 (right edge, vertical center) **→** Panel 3 (left edge, vertical center)

---

## Sizing Guidelines

- Each panel: approximately 12cm wide × 6.5cm tall
- Header bar height: ~0.9cm
- Total figure width: ~38cm (will be scaled down when included in paper)
- Use `\scriptsize` or `\footnotesize` for most text
- Use `\bfseries` for titles and important labels

---

## TikZ Libraries Required

```latex
\usetikzlibrary{
  arrows.meta,
  positioning,
  calc,
  shapes.geometric
}
```

---

## Style Notes

1. **Professional academic look** - clean lines, consistent spacing
2. **Sans-serif font** (`\sffamily`) for modern appearance
3. **Rounded corners** on all boxes (radius ~6-8pt)
4. **Thick borders** on panels, medium on internal boxes
5. **Consistent vertical and horizontal alignment** within each panel

---

## Mathematical Notation

Use these exact formulations:
- Training data: $(x_i, y_i)$
- Canonical string: $t_i$
- Embeddings: $v_i$
- Edge weights: $w_{ij}$
- Similarity: $s_{ij}$
- Reliability: $r_j$
- Neighbor posterior: $p_i(c) = \sum_j w_{ij} \cdot \mathbf{1}[y_j = c]$
- Neighborhood surprise: $S_{\text{nbr}} = -\log p_i(y_i)$

---

## Output

Provide complete, compilable TikZ code that:
1. Uses `\documentclass[tikz,border=6pt]{standalone}`
2. Renders all three panels with proper layout
3. Includes all internal components, arrows, and labels
4. Uses the specified color scheme
5. Is well-commented for easy modification

