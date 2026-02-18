# Related Work Section Review

## Context

We are writing an **ACL 2026 Theme Track paper** on "Explanation-Consistency Graphs (ECG)" for the Explainability theme. The paper is currently **12 pages**. We need to tighten it slightly while maintaining quality.

**Paper's Core Contribution:**
- ECG detects mislabeled training examples by computing neighborhood surprise in *explanation embedding space* rather than input embedding space
- Key finding: Same kNN algorithm achieves 0.832 AUROC on explanation embeddings vs 0.671 on input embeddings (artifact-aligned noise)
- Cleanlab fails catastrophically (0.107 AUROC) when artifacts enable confident fitting of wrong labels
- ECG remains robust across noise regimes (0.943 on random noise)

**Target venue:** ACL 2026 Theme Track: Explainability of NLP Models

---

## Current Related Work Section

```latex
%==============================================================================
\section{Related Work}
\label{sec:related}
%==============================================================================

ECG draws on and extends four research areas: label noise detection, graph-based data quality, explanation-based debugging, and LLM-generated explanations.
We position ECG relative to each, highlighting both connections and the gaps our work addresses.

%------------------------------------------------------------------------------
\subsection{Label Noise Detection and Data Cleaning}
%------------------------------------------------------------------------------

\paragraph{Confidence-Based Methods.}
The dominant paradigm estimates which examples are mislabeled using classifier outputs.
\textbf{Confident learning} \citep{northcutt2021confident} estimates a ``confident joint'' distribution between noisy observed labels and latent true labels, ranking examples by disagreement with model predictions.
This approach achieves strong performance when the key assumption holds: that mislabeled examples cause low confidence.
Follow-up work extends confident learning to token-level NER \citep{wang2022token}, multi-label classification \citep{thyagarajan2023multilabel}, and label-biased settings where annotator bias patterns must be decoupled from noise detection \citep{li2025decole}.

\paragraph{Training Dynamics.}
Rather than using final model outputs, training dynamics approaches track per-example statistics across epochs.
\textbf{AUM} (Area Under the Margin) \citep{pleiss2020identifying} computes the cumulative margin between the assigned label's logit and the next-highest class across training, identifying mislabeled examples by low or negative AUM.
\textbf{CTRL} \citep{yue2024ctrl} clusters loss curves to separate clean examples (smooth decay) from noisy ones (irregular patterns).
Second-split forgetting \citep{maini2022second} measures how quickly examples are forgotten during continued training.
These methods capture information unavailable from a single snapshot but still rely on training signals that become unreliable when models confidently fit spurious patterns.

\paragraph{Loss-Based Methods for NLP.}
\citet{chong2022detecting} demonstrate that simple out-of-sample loss ranking with pretrained language models is surprisingly effective on human-originated noise in text classification.
They introduce a realistic noise injection protocol based on time-pressured human relabeling, showing that PLM-based detection outperforms more complex methods under such noise.
This finding emphasizes that noise type matters: methods that work well on uniform random noise may fail on instance-dependent or artifact-aligned noise.

\paragraph{Limitations of Confidence-Based Detection.}
All confidence-based methods share a fundamental limitation: they assume mislabeled examples will cause model uncertainty.
When spurious correlations enable confident fitting of wrong labels---our target scenario---the confident learning approach breaks down completely.
The classifier achieves high confidence \textit{and} low loss on mislabeled examples, making them invisible to these methods.

%------------------------------------------------------------------------------
\subsection{Graph-Based Data Quality}
%------------------------------------------------------------------------------

\paragraph{Neighborhood Disagreement.}
A parallel research thread detects noisy labels using representation-space structure.
The core insight is that an example whose label disagrees with its nearest neighbors in embedding space is likely mislabeled \citep{bahri2020deep}.
This approach requires no training dynamics, relying instead on the assumption that semantically similar examples should have consistent labels.

\paragraph{Joint Error and Outlier Detection.}
The \textbf{Neural Relation Graph} \citep{kim2023neural} extends neighborhood-based detection to jointly identify label errors and out-of-distribution examples, constructing an explicit relational graph in feature space with scalable algorithms.
Importantly, this work includes NLP evaluation (SST-2), demonstrating that graph-based methods transfer to text.
\textbf{WANN} \citep{disalvo2025wann} introduces reliability-weighted kNN where neighbor votes are weighted by learned reliability scores, reducing error propagation when mislabeled examples cluster together.
GCN-based label propagation on kNN graphs can smooth noisy labels when clean anchors exist \citep{iscen2020graphnoisylabels}.

\paragraph{Embedding Quality and Multi-View Graphs.}
Graph methods are sensitive to embedding quality.
Work on ``beyond images'' settings notes that NLP embeddings may be noisier than vision features \citep{zhu2022beyondimages}.
Dual-kNN methods combine text embeddings with label-probability representations to stabilize neighbor quality under noise \citep{yuan2025dualknn}.
Robust contrastive learning addresses noise in positive pairs \citep{chuang2022robust}.

\paragraph{ECG's Extension.}
Prior graph-based methods operate over input embeddings or classifier representations.
ECG introduces a fundamentally different substrate: \textit{explanation embeddings}.
By building the graph over LLM-generated explanations---which capture why a label should apply, not just what the input is---ECG reveals inconsistencies invisible in input space.
Crucially, the same kNN algorithm achieves substantially higher detection performance when operating on explanation embeddings rather than input embeddings, demonstrating that the representation, not the algorithm, is the key contribution.

%------------------------------------------------------------------------------
\subsection{Explanation-Based Debugging and Artifact Detection}
%------------------------------------------------------------------------------

\paragraph{Explanations for Dataset Diagnosis.}
A rich literature uses explanations to help humans surface dataset issues.
\citet{lertvittayakumjorn2021explanation} provide a comprehensive survey of ``explanation $\rightarrow$ feedback $\rightarrow$ fix'' pipelines.
\textbf{FIND} \citep{lertvittayakumjorn2020find} enables human-in-the-loop debugging where gradient-based saliency helps users discover spurious patterns.
Interactive label cleaning via explanations shows that when influential examples have inconsistent labels, the training label is suspect \citep{teso2021interactive}.
\textbf{XMD} \citep{lee2023xmd} collects user feedback on highlighted features and updates models via explanation-alignment regularization.

\paragraph{Training-Feature Attribution for Artifacts.}
\textbf{TFA} \citep{pezeshkpour2022combining} jointly localizes which tokens in which influential training examples drive a prediction, explicitly designed to uncover training-set artifacts.
Influence-based artifact analysis shows that predictions can depend on artifactual patterns in training data even when test-time attributions look correct \citep{han2020influence}.
These methods provide deep diagnostic power but require human interpretation.

\paragraph{Spurious Correlations and Annotation Artifacts.}
The spurious correlation literature extensively documents how models exploit shortcuts.
\citet{poliak2018hypothesis} establish ``hypothesis-only'' baselines for NLI, showing that lexical artifacts predict labels.
Premise-mitigation training objectives discourage hypothesis-only shortcuts \citep{belinkov2019premise}.
Product-of-experts debiasing trains a bias-only model to soak up shortcut signal \citep{clark2019product}.
Self-debiasing identifies and downweights biased examples without knowing bias type a priori \citep{utama2020self}.
Counterfactual data augmentation breaks correlations by training on minimal-edit pairs \citep{kaushik2020learning}.

\paragraph{Gap ECG Addresses.}
Prior explanation-based work focuses on human-in-the-loop debugging or model regularization---not on automated, scalable detection of mislabeled instances.
The artifact detection literature focuses on model behavior, not training data quality.
ECG is the first to aggregate LLM explanations via graph structure for automated data cleaning, connecting the explanation and data-quality literatures.

%------------------------------------------------------------------------------
\subsection{LLM-Generated Explanations}
%------------------------------------------------------------------------------

\paragraph{Structured Output Generation.}
Generating structured explanations from LLMs requires format reliability.
\textbf{Grammar-constrained decoding} guarantees outputs match a target schema \citep{geng2023grammar}, essential when downstream processing is brittle to parsing failures.
Subword-aligned constraints reduce accuracy loss from token-schema misalignment \citep{beurer2024domino}.
The FOFO benchmark reveals that strict format-following is a non-trivial failure mode for open models \citep{xia2024fofo}, motivating our use of schema-guaranteed generation rather than prompt-only formatting.

\paragraph{Faithfulness and Plausibility.}
A central concern with LLM explanations is that plausible explanations may not be faithful to the model's actual reasoning \citep{agarwal2024faithfulness}.
Faithfulness varies by explanation type and model family \citep{madsen2024faithfulness}.
Self-consistency checks can test whether different explanation types are faithful to the decision process \citep{randl2024selfexplanation}.
Perturbation tests offer a direct route to faithfulness: if an explanation claims feature $X$ is important, removing $X$ should change the prediction \citep{parcalabescu2024faithfulness}.

\paragraph{Explanation Stability and Uncertainty.}
LLM explanations can be unstable across prompts and random seeds.
Explanation-consistency finetuning improves stability across semantically equivalent inputs \citep{chen2025explanationconsistency}.
\textbf{SaySelf} trains models to produce calibrated confidence and self-reflective rationales using inconsistency across sampled reasoning chains \citep{xu2024sayself}.
These findings motivate ECG's stability sampling and reliability weighting.

\paragraph{Label Leakage in Rationales.}
Rationales can correlate with labels in ways enabling leakage---a model can predict the label from the rationale without looking at the input \citep{wiegreffe2021label}.
ECG addresses this by forbidding label words in rationales and evaluating with leakage-aware metrics.

\paragraph{ECG's Approach.}
ECG addresses faithfulness concerns not by assuming explanations are faithful, but by \textit{verifying} them through multiple signals: NLI contradiction, neighborhood agreement, stability sampling, and training dynamics.
This multi-verification approach is more robust than trusting any single explanation property.
```

---

## Task

Please help us **condense and strengthen** this Related Work section. We want to:

### 1. Identify Content to Migrate to Appendix
- Which paragraphs/citations are **less central** to our narrative and could move to an appendix?
- Priority for keeping in main text:
  - Direct baselines we compare against (Cleanlab, AUM, kNN methods)
  - Work that motivates our core insight (artifact/spurious correlation literature)
  - Explanation-based debugging that we extend
- Candidates for appendix:
  - Extended citations that don't directly connect to our method
  - Technical details about tangential methods
  - The "LLM-Generated Explanations" subsection seems most movable since it's about implementation details rather than core positioning

### 2. Improve Writing Quality
- Identify any **awkward phrasing** or **redundant sentences**
- Suggest **tighter formulations** that preserve meaning
- Ensure **clear narrative flow** connecting each subsection to ECG's contribution

### 3. Strengthen the Narrative
- Each subsection should clearly answer: "What gap does this leave that ECG fills?"
- The progression should build toward ECG's contribution naturally
- Avoid listing citations without integrating them into the argument

---

## Constraints

- **Keep it strong**: This is for ACL, we need comprehensive coverage of relevant work
- **Target reduction**: ~20-30% shorter in main text (move rest to appendix)
- **Maintain all citations**: Don't drop references, just reorganize
- **Four subsections → possibly three** in main text (LLM explanations to appendix?)

---

## Desired Output Format

Please provide:

1. **Recommended structure** for condensed main-text Related Work
2. **Specific rewrites** for paragraphs that stay in main text
3. **Content to migrate** to appendix (with suggested appendix section title)
4. **Any narrative improvements** for better flow

Thank you!

