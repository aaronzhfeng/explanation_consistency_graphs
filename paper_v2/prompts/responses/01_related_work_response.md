# Related Work Review - LLM Response

**Prompt:** `prompts/01_related_work_review.md`  
**Model:** [TODO: model name]  
**Date:** 2026-01-06

---

## Raw Response
## 1) Recommended condensed main-text structure

### Main text: 3 subsections + 1 pointer sentence

1. **Label-error detection under confident fitting**

   * Keep: **Cleanlab / confident learning** \citep{northcutt2021confident}, **AUM** \citep{pleiss2020identifying}, **CTRL** \citep{yue2024ctrl}, PLM loss ranking for NLP \citep{chong2022detecting}
   * Narrative focus: these signals are *training-dependent* and can invert when artifacts make wrong labels easy.

2. **Graph-based data quality**

   * Keep: neighborhood disagreement / kNN noisy-label detection \citep{bahri2020deep}, relation graphs \citep{kim2023neural}, reliability-weighted kNN \citep{disalvo2025wann}
   * Optional keep (1 short clause): label propagation \citep{iscen2020graphnoisylabels}
   * Narrative focus: graph methods decouple from training dynamics, but still depend on *which representation space* you build the graph in → ECG’s key move.

3. **Explanations, artifacts, and dataset debugging**

   * Keep: explanation-driven debugging (survey + representative systems) \citep{lertvittayakumjorn2021explanation,lertvittayakumjorn2020find,teso2021interactive,lee2023xmd}, artifact attribution \citep{pezeshkpour2022combining,han2020influence}, spurious correlation/annotation artifacts + debiasing/counterfactuals \citep{poliak2018hypothesis,belinkov2019premise,clark2019product,utama2020self,kaushik2020learning}
   * Narrative focus: prior work either (a) needs humans in the loop or (b) mitigates bias at training time; ECG automates *data cleaning* by aggregating explanation semantics.

Then add **one sentence** at the end:

* “We discuss structured LLM explanation generation and reliability issues in Appendix~X.” (This is where you move the current §2.4 wholesale.)

This structure usually yields the desired **20–30% main-text reduction** mostly “for free” by relocating §2.4, while also tightening the remaining prose.

---

## 2) Specific rewrites for paragraphs that stay in main text

Below is a **drop-in condensed rewrite** of the main-text Related Work (3 subsections). It keeps your core positioning but removes repetition and reduces list-like citation dumps.

```latex
%==============================================================================
\section{Related Work}
\label{sec:related}
%==============================================================================

ECG targets training-data debugging in a regime where spurious correlations let models fit wrong labels \emph{confidently}.
It connects to (i) label-error detection from confidence and training dynamics, (ii) graph-based data quality, and (iii) explanation- and attribution-based diagnosis of artifacts.
Across these areas, the key gap is a scalable detector whose signal remains informative when classifier confidence is \emph{not}.

%------------------------------------------------------------------------------
\subsection{Label-Error Detection Under Confident Fitting}
%------------------------------------------------------------------------------

Most data-cleaning methods rank examples using signals derived from the classifier.
\textbf{Confident learning} \citep{northcutt2021confident} identifies likely label errors via disagreement between observed labels and predicted probabilities, and works well when noise manifests as low confidence.
Training-dynamics methods similarly treat mislabeled data as hard-to-learn: \textbf{AUM} \citep{pleiss2020identifying} uses cumulative margins, and \textbf{CTRL} \citep{yue2024ctrl} clusters loss trajectories to separate clean from noisy examples.
For NLP, out-of-sample loss ranking with pretrained language models can be highly effective on human-originated noise \citep{chong2022detecting}.

\paragraph{Gap.}
These approaches share a reliance on training-time difficulty (high loss, low margin, or low confidence).
When artifacts make wrong labels easy to fit, mislabeled instances can have \emph{low loss and high confidence} throughout training, rendering confidence- and dynamics-based detectors unreliable.
ECG addresses this failure mode by using a signal derived from \emph{explanations} rather than the classifier’s fit.

%------------------------------------------------------------------------------
\subsection{Graph-Based Data Quality and Neighborhood Disagreement}
%------------------------------------------------------------------------------

Graph-based methods detect label errors from representation-space structure, flagging instances whose labels disagree with their nearest neighbors.
This principle appears in kNN-based noisy-label detection \citep{bahri2020deep} and scalable relation-graph formulations that jointly model label errors and outliers \citep{kim2023neural}.
Recent work improves robustness when errors cluster, e.g., reliability-weighted neighbor voting \citep{disalvo2025wann}, and label propagation on kNN graphs when clean anchors exist \citep{iscen2020graphnoisylabels}.

\paragraph{Gap.}
Prior graph-based approaches build neighborhoods over input embeddings or model representations.
ECG keeps the same neighborhood-disagreement idea but changes the substrate: it constructs the graph in \emph{explanation embedding space}, where neighbors are defined by similar \emph{label-justifying evidence and rationales}.
This shift is crucial in artifact-aligned settings, where input-space similarity can preserve spurious markers rather than the underlying “why” of the label.

%------------------------------------------------------------------------------
\subsection{Explanations, Artifacts, and Dataset Debugging}
%------------------------------------------------------------------------------

Explanations and attribution have been used extensively for diagnosing dataset artifacts and guiding model fixes.
Surveyed “explanation $\rightarrow$ feedback $\rightarrow$ fix” pipelines \citep{lertvittayakumjorn2021explanation} and interactive systems such as \textbf{FIND} \citep{lertvittayakumjorn2020find}, explanation-driven label cleaning \citep{teso2021interactive}, and \textbf{XMD} \citep{lee2023xmd} support human-in-the-loop debugging.
Complementarily, training-set artifact analyses localize influential tokens and examples, e.g., \textbf{TFA} \citep{pezeshkpour2022combining} and influence-function based artifact discovery \citep{han2020influence}.
These tools are motivated by a broad literature on spurious correlations and annotation artifacts, including hypothesis-only shortcuts in NLI and debiasing or counterfactual remedies
\citep{poliak2018hypothesis,belinkov2019premise,clark2019product,utama2020self,kaushik2020learning}.

\paragraph{Gap.}
Existing explanation-based debugging largely supports \emph{human} discovery or \emph{model} regularization, while spurious-correlation work typically targets mitigation rather than identifying which \emph{training instances} are mislabeled.
ECG bridges these lines by treating explanations as semantic representations and aggregating them via graph structure, enabling automated detection of label inconsistencies that remain hidden to confidence-based cleaning.

\paragraph{LLM-generated explanations.}
Because ECG relies on structured LLM explanations as a representation, we summarize related work on structured generation and explanation reliability in Appendix~X.
```

**What changed (tightening + strengthening):**

* Collapsed “Confidence-Based / Training Dynamics / Loss-Based / Limitations” into a **two-paragraph arc** with a single explicit “Gap” that sets up ECG.
* Kept graph subsection focused on **the representation choice** (your key story), not a catalog of graph variants.
* Merged explanation-based debugging + artifact detection + spurious correlations into one coherent motivation that ends with a clear gap statement.

---

## 3) Content to migrate to appendix

### Suggested appendix title

**“Appendix X: Supplementary Related Work on Robust Cleaning and LLM Explanation Reliability”**

### Move these items (with citations preserved)

**A) Extensions of confident learning (not central to ECG’s narrative)**

* Your follow-up sentence listing settings: token-level NER, multi-label, label-biased settings
  → move as a compact appendix paragraph:

  * \citep{wang2022token,thyagarajan2023multilabel,li2025decole}

**B) Extra training-dynamics variant**

* Second-split forgetting is useful context but not necessary for the main positioning
  → move:

  * \citep{maini2022second}

**C) Embedding-quality / multi-view graph robustness (tangential to your core novelty)**

* The “beyond images” observation + dual-kNN + robust contrastive learning are relevant, but secondary compared to the central “graph in explanation space” claim
  → move:

  * \citep{zhu2022beyondimages,yuan2025dualknn,chuang2022robust}

**D) Entire current §2.4 LLM-Generated Explanations (recommended)**

* This is implementation-adjacent and can be summarized in one pointer sentence in main text (as in the rewrite), while keeping full coverage in the appendix.

  * Structured output generation: \citep{geng2023grammar,beurer2024domino,xia2024fofo}
  * Faithfulness + evaluation: \citep{agarwal2024faithfulness,madsen2024faithfulness,randl2024selfexplanation,parcalabescu2024faithfulness}
  * Stability + uncertainty: \citep{chen2025explanationconsistency,xu2024sayself}
  * Label leakage: \citep{wiegreffe2021label}

### Optional appendix text stub (so you can paste quickly)

```latex
\section{Supplementary Related Work on Robust Cleaning and LLM Explanation Reliability}

\paragraph{Extensions of confident learning.}
Confident learning has been adapted beyond standard classification, including token-level label error detection for NER, multi-label settings, and label-biased regimes that decouple annotator bias patterns from noise detection \citep{wang2022token,thyagarajan2023multilabel,li2025decole}.

\paragraph{Additional training-dynamics signals.}
Beyond AUM/CTRL-style dynamics, second-split forgetting characterizes datapoints by continued-training retention, providing an alternative view of “hard-to-learn” examples \citep{maini2022second}.

\paragraph{Robust graph construction in NLP.}
Graph-based cleaning depends on embedding quality; NLP features can be noisier than vision-style features \citep{zhu2022beyondimages}.
Multi-view neighbor definitions and robustness techniques have been proposed to stabilize neighborhood structure under noise, including dual-kNN representations and robust contrastive learning against noisy views \citep{yuan2025dualknn,chuang2022robust}.

\paragraph{LLM-generated explanations: structure and reliability.}
Schema- and grammar-constrained decoding improves format reliability for structured explanations \citep{geng2023grammar,beurer2024domino}, and benchmarks show strict format following remains challenging for open models \citep{xia2024fofo}.
At the same time, LLM explanations raise concerns about faithfulness, stability, and evaluation \citep{agarwal2024faithfulness,madsen2024faithfulness,randl2024selfexplanation,parcalabescu2024faithfulness,chen2025explanationconsistency,xu2024sayself}, and free-text rationales can leak label information \citep{wiegreffe2021label}.
```

---

## 4) Narrative improvements for better flow

### Make the “representation shift” the spine of the section

Right now, the story is present but spread across multiple “limitations” and “ECG extension” paragraphs. The condensed version makes this explicit:

* **(1) Training-derived signals fail when confidence is uninformative**
* **(2) Graph methods help, but only if the representation reflects the right notion of similarity**
* **(3) Explanations are a known tool for revealing artifacts, but mostly for humans**
* **ECG = graph-based cleaning where similarity is defined by explanation semantics**

### Remove redundancy / tighten phrasing (high-impact edits)

* Replace repeated “dominant paradigm…” / “fundamental limitation…” with one crisp “Gap” paragraph per subsection.
* In Graph-Based Data Quality, avoid re-stating “representation not algorithm” more than once; say it once, then move on.
* In the final gap claim, soften “ECG is the first…” to reduce reviewer pushback:

  * Replace with: **“To our knowledge, ECG is the first to aggregate LLM explanations via graph structure for automated data cleaning…”**
  * This is usually safer without weakening the claim.

### Reduce “citation listing” by grouping citations by role

Instead of one sentence per citation:

* Use “e.g.” groupings where appropriate (as in the rewrite).
* Keep the *named* baselines you compare to (Cleanlab, AUM) explicitly described; treat others as supporting examples.

### Small LaTeX formatting tip that often saves space

ACL styles sometimes add noticeable vertical space for many `\paragraph{...}` blocks. If you’re tight on lines:

* Keep only the **“Gap.”** run-in paragraphs, and convert the rest to plain sentences, or combine into fewer paragraphs.

