# Figure Fix - LLM Response

**Prompt:** `prompts/02_figure_fix.md`  
**Model:** [TODO: model name]  
**Date:** 2026-01-06

---

```latex
% ECG Pipeline Architecture (v3: no overlapping anchors + clearer conceptual link)
% Requires in preamble:
% \usetikzlibrary{positioning,calc,arrows.meta}

\begin{figure*}[t]
\centering
\begin{tikzpicture}[
    node distance=0.55cm and 0.9cm,
    box/.style={rectangle, draw=black!70, fill=white, thick,
                minimum height=1.0cm, minimum width=2.2cm,
                align=center, font=\small},
    stage/.style={rectangle, rounded corners=4pt, draw=black!80, fill=blue!12, thick,
                  minimum height=1.1cm, minimum width=2.6cm,
                  align=center, font=\small\bfseries},
    arrow/.style={-{Stealth[length=3mm]}, thick, black!70, rounded corners=2pt},
    % Conceptual link: visually distinct from main flow (no arrowhead, dashed, lighter)
    concept/.style={densely dashed, semithick, black!50, rounded corners=2pt},
    label/.style={font=\scriptsize\itshape, text=black!60},
    compare/.style={rectangle, rounded corners=3pt, draw=black!50, fill=gray!8, thick,
                    minimum height=0.9cm, minimum width=2.4cm,
                    align=center, font=\small}
]

% === ROW 1: Training Data → LLM Explanation → evidence/rationale → Embed → Vectors ===
\node[box, fill=gray!15] (data) {Training Data\\$\{(x_i, y_i)\}$};

\node[stage, right=1.0cm of data] (explain) {1. LLM\\Explanation};
\node[label, above=0.1cm of explain] {Qwen3-8B};

\node[box, right=0.8cm of explain, minimum width=2.4cm] (json) {\texttt{evidence}\\[-2pt]\texttt{rationale}};

\node[stage, right=1.0cm of json] (embed) {2. Embed};
\node[label, above=0.1cm of embed] {Sentence encoder};

\node[box, right=0.8cm of embed, fill=blue!8] (vectors) {Vectors $v_i$};

% === ROW 2: kNN Graph → Neighborhood Surprise → Score → Cleaned Dataset ===
% Anchor row 2 via the right end (output under vectors), then build leftwards.
\node[box, below=1.55cm of vectors, fill=red!12] (output) {Cleaned\\Dataset};

\node[box, left=1.0cm of output, fill=green!12, minimum width=2.6cm] (score) {$\Snbr = -\log p(y_i)$};

\node[stage, left=0.8cm of score, fill=green!18] (surprise) {4. Neighborhood\\Surprise};

\node[stage, left=1.0cm of surprise] (knn) {3. kNN Graph};
\node[label, above=0.1cm of knn] {FAISS, $k$=15};

% === Main pipeline flow (solid arrows) ===
\draw[arrow] (data) -- (explain);
\draw[arrow] (explain) -- (json);
\draw[arrow] (json) -- (embed);
\draw[arrow] (embed) -- (vectors);

% Row 1 → Row 2: start from vectors.south west so it never shares an anchor with the conceptual link
\draw[arrow] (vectors.south west) -- ++(0,-0.30) -| (knn.north);

\draw[arrow] (knn) -- (surprise);
\draw[arrow] (surprise) -- (score);
\draw[arrow] (score) -- (output);

% === Comparison panel (separate, below row 2) ===
\node[compare, below=1.25cm of score, fill=green!18, draw=green!60!black, line width=1.2pt] (exp_knn)
    {\textbf{Explanation-kNN}\\AUROC: \textbf{0.832}};
\node[compare, left=0.55cm of exp_knn] (input_knn) {Input-kNN\\AUROC: 0.671};
\node[compare, right=0.55cm of exp_knn] (cleanlab) {Cleanlab\\AUROC: 0.107};

\node[label, below=0.22cm of exp_knn]
    {\textit{Same algorithm, different embedding space} $\rightarrow$ \textbf{+24\%}};

% === Conceptual link (NOT a dataflow arrow): vectors ↔ Explanation-kNN ===
% Start from vectors.east (side) to avoid looking like an alternative downward pipeline branch.
% Route outside the main flow and drop into the comparison panel from above.
\coordinate (railx) at ($(vectors.east)+(0.6,0)$);
\coordinate (rail_top) at (railx |- vectors.east);
\coordinate (above_panel) at ($(exp_knn.north)+(0,0.35cm)$);
\coordinate (rail_mid) at (railx |- above_panel);

\draw[concept]
  (vectors.east) -- (rail_top) -- (rail_mid)
  -- node[label, midway, above=2pt] {embedding choice}
  (above_panel) -- (exp_knn.north);

\end{tikzpicture}
\caption{\textbf{ECG Pipeline.} Given training data with potentially noisy labels, ECG: (1) generates structured LLM explanations; (2) embeds the explanation text; (3) constructs a kNN graph in explanation space; (4) computes neighborhood surprise---the negative log-probability of each label given its neighbors. The key insight: the same kNN algorithm achieves \textbf{0.832 AUROC} on explanation embeddings vs.\ 0.671 on input embeddings (+24\%), while Cleanlab fails completely (0.107) on artifact-aligned noise.}
\label{fig:architecture}
\end{figure*}
```

Key changes embodied in the code (so you can sanity-check intent quickly):

* Main flow and conceptual link do **not** share an anchor: main row-bridge leaves from `vectors.south west`, conceptual link leaves from `vectors.east`.
* Row 2 is positioned **without** an arbitrary `xshift`: `output` is anchored under `vectors`, then row-2 nodes are built leftwards.
* Conceptual link is visually distinct: lighter + dashed + **no arrowhead**, routed outside the main pipeline, with an explicit label `embedding choice`.
