# SOKRATES: Tables & Figures Evolution

> How we iterated on tables and figures to get them right

---

## Table Iterations

### Main Results Table

#### Version 1 (Too Sparse)

```latex
\begin{table}[h]
\begin{tabular}{lcc}
\toprule
Model & Accuracy & Validity \\
\midrule
SFT & 93.3 & 11.3 \\
DPO & 98.2 & 91.8 \\
\bottomrule
\end{tabular}
\end{table}
```

**Problems:**
- No baseline (can't see improvement)
- Missing iterations (no progression story)
- No units (% missing)
- No column for trace validity

---

#### Version 2 (Better Structure)

```latex
\begin{table}[h]
\begin{tabular}{lccc}
\toprule
Model & Accuracy (\%) & Step Val. (\%) & Trace Val. (\%) \\
\midrule
Base (Qwen3-8B) & 85.0 & -- & -- \\
+ SFT & 94.2 & 27.3 & 2.1 \\
+ DPO iter 1 & 96.5 & 69.4 & 34.2 \\
+ DPO iter 2 & 97.6 & 87.3 & 73.8 \\
+ DPO iter 3 & 97.6 & 98.5 & 92.0 \\
\bottomrule
\end{tabular}
\end{table}
```

**Problems:**
- Best values not highlighted
- Column headers could be clearer
- Missing caption and label

---

#### Version 3 (Final)

```latex
\begin{table}[t]
\centering
\caption{Main results on PrOntoQA test set. Step validity measures 
individual reasoning steps; trace validity requires all steps valid 
\emph{and} correct answer. Bold indicates best.}
\label{tab:main_results}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Model} & \textbf{Acc. (\%)} & \textbf{Step Val. (\%)} & \textbf{Trace Val. (\%)} \\
\midrule
Base (Qwen3-8B) & 85.0 & --- & --- \\
\quad + SFT & 94.2 & 27.3 & 2.1 \\
\quad + OaK-DPO iter 1 & 96.5 & 69.4 & 34.2 \\
\quad + OaK-DPO iter 2 & 97.6 & 87.3 & 73.8 \\
\quad + OaK-DPO iter 3 & \textbf{97.6} & \textbf{98.5} & \textbf{92.0} \\
\bottomrule
\end{tabular}
\end{table}
```

**Improvements:**
- `[t]` placement for top of page
- Descriptive caption with metric definitions
- Bold for best values
- `\quad` indentation shows progression
- Consistent naming ("OaK-DPO")
- `@{}` removes extra column padding
- Added `\label` for cross-references

---

### Ablation Table

#### Version 1 (Confusing)

```latex
\begin{tabular}{lcc}
w/o solver & 94.2 & 27.3 \\
w/ solver & 97.6 & 98.5 \\
\end{tabular}
```

**Problems:**
- What is "w/o solver"? Not clear
- Missing context
- No full model name

---

#### Version 2 (Final)

```latex
\begin{table}[t]
\centering
\caption{Ablation study: impact of solver-guided DPO. 
"Answer-only DPO" uses only final answer correctness for preferences.}
\label{tab:ablation}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Variant} & \textbf{Acc. (\%)} & \textbf{Trace Val. (\%)} \\
\midrule
SFT baseline & 94.2 & 2.1 \\
+ Answer-only DPO & 96.8 & 3.4 \\
+ \textbf{Solver-guided DPO (ours)} & \textbf{97.6} & \textbf{92.0} \\
\bottomrule
\end{tabular}
\end{table}
```

**Key choices:**
- Clear variant names
- Caption explains what "answer-only" means
- Bold highlights our method
- Shows SFT baseline for context

---

### Transfer Results Table

```latex
\begin{table}[t]
\centering
\caption{Zero-shot transfer to FOLIO. Model trained on PrOntoQA only.}
\label{tab:transfer}
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Model} & \textbf{FOLIO Acc. (\%)} \\
\midrule
Base (Qwen3-8B) & 45.3 \\
+ SFT (PrOntoQA) & 47.1 \\
+ OaK-DPO iter 3 (PrOntoQA) & \textbf{53.2} \\
\midrule
\textit{Improvement} & \textit{+7.9} \\
\bottomrule
\end{tabular}
\end{table}
```

**Design choices:**
- Emphasize "(PrOntoQA)" to show no FOLIO training
- Add improvement row for clarity
- Italics for derived metric

---

## Figure Iterations

### Architecture Diagram

#### Version 1 (Too Complex)

```
┌─────────────────────────────────────────────────────────────┐
│                    SOKRATES Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Dataset │──▶│Optionize│──▶│   SFT   │──▶│   DPO   │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│       │              │             │             │          │
│       ▼              ▼             ▼             ▼          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Traces  │──▶│ Verify  │──▶│  Pairs  │──▶│  Model  │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│       ▲                                          │          │
│       └──────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Problems:**
- Too many boxes
- Arrows confusing
- Doesn't show the "loop" clearly
- Hard to see what's novel

---

#### Version 2 (TikZ, Cleaner)

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[
    node distance=1.5cm,
    box/.style={rectangle, draw, rounded corners, 
                minimum width=2cm, minimum height=0.8cm},
    arrow/.style={->, thick}
]
% Nodes
\node[box] (problem) {Problem};
\node[box, right=of problem] (policy) {LLM Policy $\pi$};
\node[box, right=of policy] (trace) {Trace};
\node[box, below=of trace] (solver) {FOL Solver};
\node[box, below=of policy] (dpo) {DPO Update};

% Arrows
\draw[arrow] (problem) -- (policy);
\draw[arrow] (policy) -- node[above] {generate} (trace);
\draw[arrow] (trace) -- node[right] {verify} (solver);
\draw[arrow] (solver) -- node[below] {preferences} (dpo);
\draw[arrow] (dpo) -- node[left] {update} (policy);

% Loop annotation
\draw[dashed, ->] (dpo.west) to[bend left=30] 
    node[left, font=\small] {repeat} (policy.south);
\end{tikzpicture}
\caption{The OaK-DPO loop: generate traces, verify with solver, 
update policy via DPO, repeat.}
\label{fig:architecture}
\end{figure}
```

**Improvements:**
- Clear flow direction
- Loop is visually obvious
- Labels on arrows explain transitions
- Minimal boxes (only essential components)

---

### Trace Example Figure

#### Version 1 (Too Dense)

```latex
\begin{verbatim}
Premises: [0] Wren is a jompus. [1] Every jompus is nervous...
Reasoning: Thought: Since Wren is a jompus (premise 0) and every 
jompus is nervous (premise 1)... Action: <Option type="MODUS_PONENS" 
args="[0, 1]" /> Thought: This matches... Action: <Option 
type="CONCLUDE" args="[0]" />
\end{verbatim}
```

**Problems:**
- No visual structure
- Hard to parse
- No highlighting of key elements

---

#### Version 2 (Formatted)

```latex
\begin{figure}[t]
\centering
\begin{tcolorbox}[colback=gray!5, colframe=gray!50, 
                  title=Example Optionized Trace, fonttitle=\bfseries]
\textbf{Premises:}\\
\quad [0] Wren is a jompus.\\
\quad [1] Every jompus is nervous.\\[0.5em]

\textbf{Conclusion:} Wren is nervous.\\[0.5em]

\textbf{Reasoning:}\\
\textcolor{blue}{Thought:} Since Wren is a jompus (premise 0) and 
every jompus is nervous (premise 1), we can conclude Wren is nervous.\\
\textcolor{red}{Action:}\\
\quad \texttt{<Option type="MODUS\_PONENS" args="[0, 1]" />}\\[0.5em]

\textcolor{blue}{Thought:} This matches our conclusion exactly.\\
\textcolor{red}{Action:}\\
\quad \texttt{<Option type="CONCLUDE" args="[0]" />}\\[0.5em]

\textbf{Final Answer:} TRUE \checkmark
\end{tcolorbox}
\caption{A complete optionized reasoning trace. Blue = natural language 
reasoning, red = discrete option.}
\label{fig:trace_example}
\end{figure}
```

**Improvements:**
- Box framing
- Color coding (Thought vs Action)
- Clear visual hierarchy
- Checkmark for correct answer
- Indentation for structure

---

### Prompt Template Figure

#### Key Design Choice: Complete vs Truncated

**Problem:** Full prompt is 50+ lines, but figure space is limited.

**Solution:** Show structure with "[...]" for truncated parts, add note about full version in appendix.

```latex
\begin{figure}[t]
\begin{tcolorbox}[colback=gray!5, colframe=gray!50, 
                  title=Prompt Template (Abbreviated)]
\small
\textbf{System:} You are a logical reasoning assistant...\\[0.5em]

\textbf{Available inference rules:}\\
\quad MODUS\_PONENS, MODUS\_TOLLENS, UNIV\_INSTANTIATION, ...\\
\quad (10 rules total; see Table~\ref{tab:options})\\[0.5em]

\textbf{Problem:}\\
Premises: [numbered list]\\
Conclusion to evaluate: [statement]\\[0.5em]

\textbf{Output format:}\\
Thought: [reasoning]\\
Action: <Option type="..." args="[...]" />\\
... (repeat until CONCLUDE)
\end{tcolorbox}
\caption{Prompt template structure. Full prompt in Appendix.}
\end{figure}
```

---

## Common Table/Figure Issues

### Issue 1: Figure Placement

**Problem:** Figure appears pages away from reference.

**Solutions:**
```latex
% Try in order:
[t]     % Top of page - usually best
[h]     % Here - often ignored
[H]     % Force here (requires float package) - can break layout
[!t]    % Top, override LaTeX's judgment
```

**What worked for us:** `[t]` + moving the `\input` earlier in the source.

---

### Issue 2: Table Too Wide

**Solutions:**
```latex
% 1. Reduce font size
{\small
\begin{tabular}{...}
...
\end{tabular}
}

% 2. Use abbreviations
Accuracy → Acc.
Validity → Val.

% 3. Use resizebox (last resort)
\resizebox{\columnwidth}{!}{
\begin{tabular}{...}
...
\end{tabular}
}
```

---

### Issue 3: Numbers Don't Align

**Problem:** 
```
94.2
2.1
```

**Solution:** Right-align numbers, use consistent decimal places.
```latex
\begin{tabular}{r}  % r = right align
94.2 \\
 2.1 \\
\end{tabular}
```

Or use `siunitx` package:
```latex
\usepackage{siunitx}
\begin{tabular}{S[table-format=2.1]}
94.2 \\
2.1 \\
\end{tabular}
```

---

### Issue 4: Caption Too Long

**Problem:** Caption wraps awkwardly or is too detailed.

**Solution:** Short caption for list of figures, long for display.
```latex
\caption[Main results on PrOntoQA]{Main results on PrOntoQA test set. 
Step validity measures individual reasoning steps; trace validity 
requires all steps valid \emph{and} correct final answer. Bold 
indicates best. Baseline is zero-shot Qwen3-8B.}
```

---

## Time Spent on Tables/Figures

| Component | Iterations | Time |
|-----------|------------|------|
| Main results table | 4 | 1.5h |
| Ablation table | 2 | 0.5h |
| Transfer table | 2 | 0.5h |
| Architecture diagram | 3 | 2h |
| Trace example | 2 | 1h |
| Prompt template | 3 | 1h |
| **Total** | | **~6.5h** |

This was ~17% of total paper writing time (10h) spent just on tables/figures!

---

## Checklist for Tables

- [ ] Clear, descriptive caption
- [ ] Units in column headers (e.g., "Acc. (%)")
- [ ] Bold best values
- [ ] Baseline included for comparison
- [ ] `\label` for cross-references
- [ ] Consistent decimal places
- [ ] Right-aligned numbers
- [ ] Not too wide for column

## Checklist for Figures

- [ ] `[t]` placement specifier
- [ ] Clear, minimal design
- [ ] Caption explains what to see
- [ ] Colors are colorblind-friendly
- [ ] Text is readable when printed
- [ ] `\label` for cross-references
- [ ] Referenced in text before appearing

