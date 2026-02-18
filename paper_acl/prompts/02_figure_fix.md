# ECG Pipeline Figure - Fix Request

## Context

We have a TikZ figure for an ACL paper showing the ECG (Explanation-Consistency Graphs) pipeline. The figure uses a two-row layout but has issues with overlapping arrows and unclear visual hierarchy.

---

## Semantic Structure (what we want to show)

### Main Pipeline Flow (solid arrows)

```
ROW 1: Training Data → 1. LLM Explanation → evidence/rationale → 2. Embed → Vectors v_i
                                                                                    ↓
ROW 2:                    3. kNN Graph ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                              ↓
                     4. Neighborhood Surprise → S_nbr = -log p(y_i) → Cleaned Dataset
```

### Comparison Panel (separate, below row 2)

```
    [Input-kNN: 0.671]  [Explanation-kNN: 0.832]  [Cleanlab: 0.107]
                        "Same algorithm, different embedding space → +24%"
```

### Dashed Line (conceptual link, NOT part of main flow)

- **From:** "Vectors v_i" (or possibly "2. Embed" stage)
- **To:** "Explanation-kNN" box in the comparison panel
- **Purpose:** Highlight that the embedding choice (explanation vs input) is what differentiates ECG from Input-kNN
- **Should NOT** be confused with the main pipeline flow

---

## Current TikZ Code

```latex
% Figure 1: ECG Pipeline Architecture (v2 - compact two-row layout)
% Usage: \input{assets/fig_architecture_v2}

\begin{figure*}[t]
\centering
\begin{tikzpicture}[
    node distance=0.5cm and 0.8cm,
    box/.style={rectangle, draw=black!70, fill=white, thick, minimum height=1.0cm, minimum width=2.2cm, align=center, font=\small},
    stage/.style={rectangle, rounded corners=4pt, draw=black!80, fill=blue!12, thick, minimum height=1.1cm, minimum width=2.6cm, align=center, font=\small\bfseries},
    arrow/.style={-{Stealth[length=3mm]}, thick, black!70},
    label/.style={font=\scriptsize\itshape, text=black!60},
    compare/.style={rectangle, rounded corners=3pt, draw=black!50, fill=gray!8, thick, minimum height=0.9cm, minimum width=2.4cm, align=center, font=\small}
]

% === ROW 1: Input → Explanation → Embed → Vectors ===
\node[box, fill=gray!15] (data) {Training Data\\$\{(x_i, y_i)\}$};

\node[stage, right=1.0cm of data] (explain) {1. LLM\\Explanation};
\node[label, above=0.1cm of explain] {Qwen3-8B};

\node[box, right=0.8cm of explain, minimum width=2.4cm] (json) {\texttt{evidence}\\[-2pt]\texttt{rationale}};

\node[stage, right=1.0cm of json] (embed) {2. Embed};
\node[label, above=0.1cm of embed] {Sentence encoder};

\node[box, right=0.8cm of embed, fill=blue!8] (vectors) {Vectors $v_i$};

% === ROW 2: kNN Graph → Surprise → Score → Output ===
% Position row 2 DIRECTLY BELOW row 1 (same x-coordinates)
\node[stage, below=1.5cm of data, xshift=3.8cm] (knn) {3. kNN Graph};
\node[label, above=0.1cm of knn] {FAISS, $k$=15};

\node[stage, right=1.0cm of knn, fill=green!18] (surprise) {4. Neighborhood\\Surprise};

\node[box, right=0.8cm of surprise, fill=green!12, minimum width=2.6cm] (score) {$\Snbr = -\log p(y_i)$};

\node[box, right=1.0cm of score, fill=red!12] (output) {Cleaned\\Dataset};

% === ARROWS: Main pipeline flow ===
% Row 1: left to right
\draw[arrow] (data) -- (explain);
\draw[arrow] (explain) -- (json);
\draw[arrow] (json) -- (embed);
\draw[arrow] (embed) -- (vectors);

% Row 1 to Row 2: vectors down to kNN (clear vertical + horizontal path)
\draw[arrow] (vectors.south) -- ++(0,-0.5) -| (knn.north);

% Row 2: left to right
\draw[arrow] (knn) -- (surprise);
\draw[arrow] (surprise) -- (score);
\draw[arrow] (score) -- (output);

% === COMPARISON PANEL: Below row 2, RIGHT-ALIGNED under output ===
\node[compare, below=1.3cm of surprise] (input_knn) {Input-kNN\\AUROC: 0.671};
\node[compare, right=0.5cm of input_knn, fill=green!18, draw=green!60!black, line width=1.2pt] (exp_knn) {\textbf{Explanation-kNN}\\AUROC: \textbf{0.832}};
\node[compare, right=0.5cm of exp_knn] (cleanlab) {Cleanlab\\AUROC: 0.107};

% Annotation below comparison
\node[label, below=0.2cm of exp_knn] {\textit{Same algorithm, different embedding space} $\rightarrow$ \textbf{+24\%}};

% === DASHED LINE: From Vectors to Explanation-kNN ===
% Route: go DOWN from vectors, then RIGHT, then DOWN to exp_knn
% This avoids overlapping with row 2 elements
\draw[dashed, black!50, thick] 
    (vectors.south) -- ++(0,-0.25) -- ++(2.0,0) |- (exp_knn.east);

\end{tikzpicture}
\caption{\textbf{ECG Pipeline.} Given training data with potentially noisy labels, ECG: (1) generates structured LLM explanations; (2) embeds the explanation text; (3) constructs a kNN graph in explanation space; (4) computes neighborhood surprise---the negative log-probability of each label given its neighbors. The key insight: the same kNN algorithm achieves \textbf{0.832 AUROC} on explanation embeddings vs.\ 0.671 on input embeddings (+24\%), while Cleanlab fails completely (0.107) on artifact-aligned noise.}
\label{fig:architecture}
\end{figure*}
```

---

## Current Problems

1. **Arrows overlap at vectors.south:** Both the main flow arrow (vectors → kNN) and the dashed line start from `vectors.south`, creating visual confusion at that junction point.

2. **Path conflicts:** The main arrow goes DOWN then LEFT to reach kNN, while the dashed line goes DOWN then RIGHT to reach exp_knn. They share the same initial vertical segment.

3. **Dashed line meaning is unclear:** It's supposed to show "this is where the embedding choice matters" but visually looks like an alternative data flow path, which is confusing.

4. **Row 2 positioning:** The `knn` node is positioned with `xshift=3.8cm` which may not align well with the flow from vectors.

---

## Desired Fix

1. **Clear separation** between main pipeline flow arrows and the conceptual dashed link
2. **No overlapping paths** at `vectors.south` - use different anchor points or routing
3. **Dashed line should be visually distinct** - clearly a "conceptual connection" not a data flow
4. **Consider alternatives:**
   - Should the dashed line originate from `embed` stage instead of `vectors`?
   - Could we use a different visual metaphor (bracket, annotation arrow, colored region)?
   - Could the comparison panel be positioned differently to make the connection clearer?

---

## Constraints

- Must remain a `figure*` (full-width) for ACL format
- Should fit comfortably without being too cramped
- Text must remain readable (current font sizes are good)
- Keep the two-row layout (better width/height ratio than single row)

---

## Desired Output

Please provide a fixed TikZ code that:
1. Maintains the semantic structure
2. Has clear, non-overlapping arrow paths
3. Makes the dashed conceptual link visually distinct from the main flow
4. Looks clean and professional for an academic paper

Thank you!

