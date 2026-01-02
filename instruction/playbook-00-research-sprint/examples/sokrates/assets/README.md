# SOKRATES Paper Assets

Actual LaTeX source files from the accepted paper.

---

## Files

### Final Paper
- `sokrates_single.pdf` — The accepted paper (10 pages, AAAI-26 Bridge Workshop)

### Figures (`figures/`)
| File | Description |
|------|-------------|
| `architecture_v2.tex` | Main architecture diagram (TikZ) |
| `trace_example.tex` | Example optionized reasoning trace |
| `full_prompt.tex` | Complete prompt template |

### Tables (`tables/`)
| File | Description |
|------|-------------|
| `main_results.tex` | Main results (Table 1) |
| `ablations.tex` | Ablation study (Table 3) |
| `transfer.tex` | FOLIO transfer results (Table 4) |
| `options.tex` | Option vocabulary (Table 2) |
| `hyperparams.tex` | Hyperparameters (Appendix) |

---

## Usage

These files use AAAI style. To compile standalone:

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning}

\begin{document}
\input{tables/main_results.tex}
\end{document}
```

For the full paper context, see `sokrates_single.pdf`.

---

## Key Design Patterns

### Tables
- Use `booktabs` for clean lines (`\toprule`, `\midrule`, `\bottomrule`)
- Units in column headers: `Acc. (\%)`
- Bold best results: `\textbf{97.6}`
- Right-align numbers

### Figures
- TikZ for vector diagrams
- `tcolorbox` for formatted code/trace examples
- `[t]` placement specifier
- Descriptive captions

---

## License

MIT — use these as templates for your own papers.

