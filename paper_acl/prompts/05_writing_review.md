# Writing Quality Review Request

## Task

Please review this ACL paper draft for writing quality and identify areas that need improvement. Focus on clarity, flow, conciseness, and persuasiveness rather than technical correctness.

---

## Paper Title

**Explanation-Consistency Graphs: Neighborhood Surprise in Explanation Space for Training Data Debugging**

---

## Full Paper Text

### Abstract

Training data quality is critical for NLP model performance, yet identifying mislabeled examples remains challenging when models confidently fit errors via spurious correlations. Confident learning methods like Cleanlab assume mislabeled examples cause low confidence—but this assumption fails catastrophically when artifacts enable confident fitting of wrong labels. We propose **Explanation-Consistency Graphs (ECG)**, which detects problematic training instances by computing neighborhood surprise in *explanation embedding space*. Our key insight is that LLM-generated explanations capture "why this label applies," and this semantic content reveals inconsistencies invisible to classifier confidence. By embedding structured explanations and measuring kNN label disagreement, ECG achieves 0.832 AUROC on artifact-aligned noise—where Cleanlab collapses to 0.107—representing a 24% improvement over the same algorithm on input embeddings (0.671). On random label noise, ECG remains competitive (0.943 vs. Cleanlab's 0.977), demonstrating robustness across noise regimes. We show that the primary value lies in the *explanation representation* rather than complex signal aggregation, and analyze why naive multi-signal combination can degrade performance when training dynamics signals are anti-correlated with artifact-driven noise.

---

### 1. Introduction

The quality of training data fundamentally constrains what NLP models can learn. Large-scale empirical studies reveal that 3–6% of labels in widely-used benchmarks—including MNIST, ImageNet, and Amazon Reviews—are incorrect (Northcutt et al., 2021), and these errors propagate into systematic model failures. Beyond simple mislabeling, annotation artifacts and spurious correlations create particularly insidious data quality issues: models learn superficial patterns that happen to correlate with labels in the training set but fail catastrophically under distribution shift (Gururangan et al., 2018; McCoy et al., 2019). Identifying and correcting such problematic instances—*training data debugging*—is therefore essential for building reliable NLP systems.

The dominant paradigm for training data debugging relies on model confidence and loss signals. **Confident learning** (Northcutt et al., 2021) estimates a joint distribution between noisy and true labels using predicted probabilities, effectively identifying instances where the model "disagrees" with the observed label. **Training dynamics** approaches like AUM (Pleiss et al., 2020) and CTRL (Yue et al., 2024) track per-example margins and loss trajectories across training epochs, exploiting the observation that mislabeled examples exhibit different learning patterns than clean ones. High-loss filtering with pretrained language models can be surprisingly effective on human-originated noise (Chong et al., 2022). These methods share a common assumption: *problematic examples will cause low confidence or high loss during training*.

This assumption breaks down catastrophically when **models confidently fit errors via spurious correlations**. Consider sentiment data where mislabeled examples happen to contain distinctive tokens—rating indicators like "[RATING=5]", demographic markers, or formatting artifacts. The classifier learns to predict the *wrong* labels with *high confidence* by exploiting these spurious markers. From a loss perspective, these mislabeled examples look perfectly clean; they are fitted early, with high confidence, and low loss throughout training. Cleanlab's confident joint and AUM's margin trajectories both fail because the model is confident—just confidently wrong for the wrong reasons.

This failure mode is not hypothetical. Poliak et al. (2018) showed that NLI datasets can be partially solved using only the hypothesis, revealing pervasive annotation artifacts. Gururangan et al. (2018) demonstrated that annotation patterns systematically correlate with labels in ways that models exploit. The spurious correlation literature extensively documents how models learn shortcuts that evade standard diagnostics (Clark et al., 2019; Utama et al., 2020; Tu et al., 2020), and debiasing methods must explicitly model bias structure to mitigate it (Sagawa et al., 2020). When the very mechanism that causes label noise *also* enables confident fitting, confidence-based debugging fundamentally cannot work.

We propose **Explanation-Consistency Graphs (ECG)**, which detects problematic training instances by computing neighborhood surprise in *explanation embedding space* rather than input embedding space. Our key insight is that *explanations encode semantic information about why a label should apply*, and this "why" content reveals inconsistencies even when classifier confidence does not. When an LLM explains why it believes a sentence has positive sentiment, its rationale and cited evidence reflect the actual semantic content—not spurious markers that the classifier may have learned to exploit. By embedding these explanations and measuring kNN label disagreement, ECG detects mislabeled instances that are invisible to loss and probability signals.

The core idea is simple: if an example's label disagrees with the labels of examples whose *explanations* are most similar, that label is likely wrong. This is the same principle underlying input-based kNN detection (Bahri et al., 2020; Kim et al., 2023), but operating in a fundamentally different representation space. Input embeddings capture "what the text is about"; explanation embeddings capture "why this text has this label." When labels are wrong, the "why" becomes inconsistent with semantically similar examples, making explanation-space neighborhood surprise a powerful detection signal.

ECG synthesizes ideas from three research threads: **(1)** the explanation-based debugging literature, which uses explanations to help humans surface artifacts (Lertvittayakumjorn & Toni, 2021; Lertvittayakumjorn et al., 2020; Lee et al., 2023), but has not automated detection via graph structure; **(2)** graph-based noisy label detection, which uses neighborhood disagreement in representation space (Bahri et al., 2020; Kim et al., 2023; DiSalvo et al., 2025), but over input embeddings; and **(3)** LLM-generated explanations with structured schemas (Geng et al., 2023; Huang et al., 2023), which provide the semantic substrate for our graph.

Concretely, ECG works as follows. **(1) Explanation Generation:** We generate structured JSON explanations for all training instances using an instruction-tuned LLM (Qwen3-8B), enforcing extractive evidence spans and rationales via schema-constrained decoding. **(2) Explanation Embedding:** We embed explanations using a sentence encoder and construct a kNN graph in this space. **(3) Neighborhood Surprise:** We compute the negative log-probability of each instance's label given its neighbors' labels in explanation space—our primary detection signal. We also explored additional signals (NLI contradiction, stability, training dynamics), but found that simple kNN surprise in explanation space works best.

Our contributions are:

1. We introduce **Explanation-Consistency Graphs (ECG)**, demonstrating that neighborhood surprise computed in *explanation embedding space* substantially outperforms the same algorithm on input embeddings (+24% AUROC on artifact-aligned noise: 0.832 vs. 0.671).

2. We establish a **concrete failure mode** for confidence-based cleaning: when artifacts enable confident fitting of wrong labels, Cleanlab achieves only 0.107 AUROC (worse than random), while ECG achieves 0.832. ECG remains competitive on random noise (0.943 vs. Cleanlab's 0.977), providing a **robust** method across noise regimes.

3. We provide **analysis of why naive signal aggregation fails**: training dynamics signals (AUM) are anti-correlated with noise under artifact conditions, because artifacts make wrong labels *easy* to learn. This negative result offers guidance for future multi-signal approaches.

---

### 2. Related Work

ECG targets training-data debugging in a regime where spurious correlations let models fit wrong labels *confidently*. It connects to (i) label-error detection from confidence and training dynamics, (ii) graph-based data quality, and (iii) explanation- and attribution-based diagnosis of artifacts. Across these areas, the key gap is a scalable detector whose signal remains informative when classifier confidence is *not*.

#### 2.1 Label-Error Detection Under Confident Fitting

Most data-cleaning methods rank examples using signals derived from the classifier. **Confident learning** (Northcutt et al., 2021) identifies likely label errors via disagreement between observed labels and predicted probabilities, and works well when noise manifests as low confidence. Training-dynamics methods similarly treat mislabeled data as hard-to-learn: **AUM** (Pleiss et al., 2020) uses cumulative margins, and **CTRL** (Yue et al., 2024) clusters loss trajectories to separate clean from noisy examples. For NLP, out-of-sample loss ranking with pretrained language models can be highly effective on human-originated noise (Chong et al., 2022).

**Gap.** These approaches share a reliance on training-time difficulty (high loss, low margin, or low confidence). When artifacts make wrong labels easy to fit, mislabeled instances can have *low loss and high confidence* throughout training, rendering confidence- and dynamics-based detectors unreliable. ECG addresses this failure mode by using a signal derived from *explanations* rather than the classifier's fit.

#### 2.2 Graph-Based Data Quality and Neighborhood Disagreement

Graph-based methods detect label errors from representation-space structure, flagging instances whose labels disagree with their nearest neighbors. This principle appears in kNN-based noisy-label detection (Bahri et al., 2020) and scalable relation-graph formulations that jointly model label errors and outliers (Kim et al., 2023). Recent work improves robustness when errors cluster, e.g., reliability-weighted neighbor voting (DiSalvo et al., 2025), and label propagation on kNN graphs when clean anchors exist (Iscen et al., 2020).

**Gap.** Prior graph-based approaches build neighborhoods over input embeddings or model representations. ECG keeps the same neighborhood-disagreement idea but changes the substrate: it constructs the graph in *explanation embedding space*, where neighbors are defined by similar *label-justifying evidence and rationales*. This shift is crucial in artifact-aligned settings, where input-space similarity can preserve spurious markers rather than the underlying "why" of the label.

#### 2.3 Explanations, Artifacts, and Dataset Debugging

Explanations and attribution have been used extensively for diagnosing dataset artifacts and guiding model fixes. Surveyed "explanation → feedback → fix" pipelines (Lertvittayakumjorn & Toni, 2021) and interactive systems such as **FIND** (Lertvittayakumjorn et al., 2020), explanation-driven label cleaning (Teso & Kersting, 2021), and **XMD** (Lee et al., 2023) support human-in-the-loop debugging. Complementarily, training-set artifact analyses localize influential tokens and examples, e.g., **TFA** (Pezeshkpour et al., 2022) and influence-function based artifact discovery (Han et al., 2020). These tools are motivated by a broad literature on spurious correlations and annotation artifacts, including hypothesis-only shortcuts in NLI and debiasing or counterfactual remedies (Poliak et al., 2018; Belinkov et al., 2019; Clark et al., 2019; Utama et al., 2020; Kaushik et al., 2020).

**Gap.** Existing explanation-based debugging largely supports *human* discovery or *model* regularization, while spurious-correlation work typically targets mitigation rather than identifying which *training instances* are mislabeled. To our knowledge, ECG is the first to aggregate LLM explanations via graph structure for automated data cleaning, bridging the explanation and data-quality literatures.

**LLM-generated explanations.** Because ECG relies on structured LLM explanations as a representation, we summarize related work on structured generation and explanation reliability in Appendix.

---

### 3. Method

Given a training dataset D = {(x_i, y_i)} with potentially noisy labels y_i, our goal is to produce a suspiciousness ranking that places mislabeled or artifact-laden instances at the top. ECG achieves this through three stages: explanation generation, explanation embedding and graph construction, and neighborhood surprise computation. We also explored additional signals (NLI contradiction, stability, training dynamics) but found they did not improve over simple neighborhood surprise; we analyze this in Analysis section and provide details in Appendix.

#### 3.1 Structured Explanation Generation

For each training instance x_i, we generate a structured JSON explanation using an instruction-tuned LLM (Qwen3-8B). The explanation contains:
- pred_label: The LLM's predicted label
- evidence: 1–3 exact substrings from x_i justifying the prediction
- rationale: A brief explanation (≤25 tokens) without label words
- counterfactual: A minimal change that would flip the label
- confidence: Integer 0–100

We enforce schema validity via constrained decoding and instruct the LLM to ignore metadata tokens (e.g., <lbl_pos>) so explanations reflect semantic content rather than spurious markers.

**Stability Sampling.** LLM explanations can be unstable across random seeds. We generate M=3 explanations per instance (one deterministic at temperature 0, two samples at temperature 0.7) and compute a reliability score r_i = (1/3)(L_i + E_i + R_i) where L_i (label agreement), E_i (evidence Jaccard), and R_i (rationale similarity) each measure agreement across the M samples. High r_i indicates stable, reliable explanations; low r_i indicates the LLM is uncertain or the instance is ambiguous.

#### 3.2 Reliability-Weighted Graph Construction

We embed explanations and construct a kNN graph that downweights unreliable neighbors, inspired by WANN (DiSalvo et al., 2025).

**Explanation Embedding.** For each instance, we form a canonical string t_i excluding label information: t_i = "Evidence: " ⊕ e_i ⊕ " | Rationale: " ⊕ r_i, where e_i and r_i are the evidence and rationale fields. We embed t_i using a sentence encoder (all-MiniLM-L6-v2) and L2-normalize to obtain v_i.

**Reliability-Weighted Edges.** We retrieve the k=15 nearest neighbors N(i) for each node using FAISS. Edge weights incorporate both similarity and neighbor reliability.

**Outlier Detection.** We compute an outlier score to distinguish genuinely out-of-distribution examples from mislabeled in-distribution examples.

#### 3.3 Neighborhood Surprise Detection

The core detection signal in ECG is **neighborhood surprise**: if an instance's label disagrees with the labels of instances with similar explanations, the label may be wrong.

**Neighborhood Surprise (S_nbr).** We compute a weighted neighbor label posterior p_i(c) and define the suspiciousness score S_nbr(i) = -log p_i(y_i). High S_nbr indicates the observed label is unlikely given similar explanations. Instances are ranked by S_nbr and the top-K are flagged for removal or review.

**Why Explanation Space?** The same neighborhood surprise algorithm can be applied to input embeddings (Input-kNN) or explanation embeddings (Explanation-kNN). The key empirical finding is that explanation embeddings yield substantially better detection:
- Explanation-kNN: 0.832 AUROC on artifact-aligned noise
- Input-kNN: 0.671 AUROC (same algorithm, different embedding)

This 24% improvement demonstrates that explanations capture label-quality information invisible in input space. When labels are wrong, the LLM's rationale reflects semantic inconsistency with similar examples, even if the input text is similar to correctly-labeled examples.

**Explored Extensions.** We also investigated additional signals: NLI contradiction, explanation stability, and training dynamics. Surprisingly, combining these signals with neighborhood surprise *degraded* performance on artifact-aligned noise. We analyze why in Analysis: the training dynamics signal is anti-correlated with noise when artifacts make wrong labels easy to learn.

---

### 4. Experimental Setup

**Dataset and Noise Injection.** We evaluate on SST-2 (binary sentiment), subsampling 25,000 training examples. We create two synthetic noise conditions at rate p=10%:
- Uniform Noise: Labels are flipped uniformly at random.
- Artifact-Aligned Noise: Labels are flipped *and* a spurious marker is appended. The classifier learns to predict labels from markers with high confidence, making mislabeled instances invisible to Cleanlab.

**Baselines:** Cleanlab, High-Loss, AUM, LLM Mismatch, Input-kNN, Random.

**Metrics:** AUROC, AUPRC, Precision@K, Recall@K, F1@K for detection; accuracy on clean test set for downstream.

---

### 5. Results

#### 5.1 Detection Performance on Artifact-Aligned Noise

Table shows detection metrics on artifact-aligned noise, where mislabeled examples contain spurious markers that enable confident classifier fitting. This is the failure mode for confidence-based methods.

**Why Confidence-Based Methods Fail.** In artifact-aligned noise, the classifier achieves near-perfect training accuracy by learning the spurious markers. Cleanlab, loss-based, and margin-based methods all rely on mislabeled examples causing low confidence or high loss. But mislabeled examples have *high* confidence (due to markers) and *low* loss—making them rank as the *least* suspicious. This inverts the detection signal, yielding AUROC below 0.5 (worse than random).

**Explanation-kNN vs. Input-kNN.** Both methods use the same neighborhood surprise algorithm, but on different embeddings. The 24% improvement demonstrates that explanation embeddings capture "why this label" rather than "what this text is about," revealing label inconsistencies invisible in input space.

**Multi-Signal Aggregation Hurts.** Surprisingly, combining multiple signals *degrades* performance compared to Explanation-kNN alone.

#### 5.2 Detection Performance on Random Noise

Table shows results on random label noise. This is the setting where confidence-based methods are expected to excel. Key finding: Cleanlab is not wrong, but it is brittle. It achieves near-perfect detection on random noise but fails catastrophically on artifact noise. Explanation-kNN is robust across both regimes.

#### 5.3 Downstream Improvements

Removing the top 2% of flagged instances yields a +0.57% accuracy improvement.

#### 5.4 Ablation Studies

- Noise Rate Sensitivity: Explanation-kNN's advantage over Input-kNN is consistent across noise rates and *increases* at higher noise rates on artifact-aligned noise.
- Dataset Size Sensitivity: Explanation-kNN's advantage is largest on smaller datasets.
- LLM Size Trade-off: Smaller LLMs produce consistent explanations enabling Explanation-kNN's best single-method AUROC, while larger LLMs enable ensemble methods achieving overall best.

---

### 6. Analysis

**Why Explanations Succeed Where Confidence Fails.** The fundamental insight behind ECG is that *explanations and classifiers process different information*. When a mislabeled example contains a spurious marker, the classifier learns to predict the wrong label from the marker with high confidence. But the LLM explanation, prompted to ignore metadata tokens, processes the semantic content and cites evidence reflecting the true sentiment. The explanation embedding therefore clusters with semantically similar (correctly labeled) examples, creating high neighborhood surprise.

**Why Multi-Signal Aggregation Failed.** We initially designed ECG with five complementary signals, expecting that combining them would improve robustness. Instead, multi-signal aggregation substantially underperformed simple Explanation-kNN. The primary culprit is the training dynamics signal, which is *anti-correlated* with noise under artifact conditions. Under artifact-aligned noise, mislabeled examples have spurious markers that make them *easy* to learn—they achieve high confidence and high AUM. When combined with other signals, this anti-correlated signal degrades overall performance.

**When to Use Explanation-kNN vs. Cleanlab.** Our results suggest a simple practical guideline:
- If you suspect random annotation errors with no systematic pattern, use Cleanlab
- If you suspect artifact-aligned noise or spurious correlations, use Explanation-kNN
- If you are uncertain about noise type, Explanation-kNN is safer

**LLM Size Trade-off.** Smaller LLMs produce simpler explanations with less variation, yielding more homogeneous embeddings where Explanation-kNN can reliably detect label inconsistencies. Larger LLMs produce richer reasoning but this diversity creates more heterogeneous embeddings that hurt Explanation-kNN. However, larger models excel at explicit artifact detection, enabling effective ensemble methods.

**Failure Cases and Limitations.** ECG struggles with genuinely ambiguous sentences where the LLM is also uncertain. ECG also depends on the LLM correctly ignoring spurious markers.

**Computational Cost.** LLM explanation generation is the main bottleneck (~10 minutes for 25k examples on H100). Explanations are generated once and cached.

---

### 7. Conclusion

We introduced Explanation-Consistency Graphs (ECG), demonstrating that neighborhood surprise computed in *explanation embedding space* substantially outperforms the same algorithm on input embeddings for detecting mislabeled training examples. On artifact-aligned noise—where Cleanlab fails catastrophically (0.107 AUROC)—Explanation-kNN achieves 0.832 AUROC, a 24% improvement over Input-kNN (0.671). ECG remains competitive on random noise (0.943 vs. Cleanlab's 0.977), providing a robust method across noise regimes.

Our analysis reveals that the primary value lies in the *explanation representation* rather than complex signal aggregation. Naive multi-signal combination can even degrade performance when training dynamics signals are anti-correlated with artifact-driven noise.

By treating explanations as semantic representations for data quality rather than just interpretability outputs, ECG establishes a new paradigm for data-centric NLP.

---

## Review Criteria

Please evaluate the paper on the following dimensions and provide specific, actionable feedback:

### 1. Clarity
- Are there sentences or paragraphs that are hard to follow?
- Are technical concepts explained clearly for an NLP audience?
- Are there places where more context or explanation is needed?

### 2. Flow and Organization
- Does the paper have a logical narrative arc?
- Are transitions between sections and paragraphs smooth?
- Are there places where the reader might get lost or confused?

### 3. Conciseness
- Are there redundant or repetitive passages?
- Are there places where the same point is made multiple times?
- Can any sections be tightened without losing important content?

### 4. Persuasiveness
- Is the motivation compelling?
- Are the claims well-supported by evidence?
- Are there places where the argument could be strengthened?

### 5. Writing Quality
- Are there awkward phrases or constructions?
- Is the tone appropriate for an academic paper?
- Are there any grammatical issues or typos?

---

## Desired Output Format

Please structure your response as:

1. **Overall Assessment**: Brief summary of writing quality (1-2 paragraphs)

2. **Major Issues**: List of significant writing problems that should be addressed (if any)

3. **Section-by-Section Feedback**: Specific issues and suggestions for each section

4. **Minor Issues**: Smaller improvements (awkward phrasing, minor redundancies, etc.)

5. **Strengths**: What works well in the writing that should be preserved

6. **Priority Recommendations**: Top 3-5 changes that would most improve the paper

