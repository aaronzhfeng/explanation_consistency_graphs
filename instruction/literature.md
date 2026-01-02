# Literature: Explanation-Consistency Graphs (ECG)

Curated bibliography for the ECG project (ACL 2026 Theme Track: Explainability of NLP Models).

**Last updated:** 2025-01-30  
**Total papers:** 103 (deduplicated from 116 raw entries)

### ⚠️ PDF Availability Note

The following 7 papers lack direct PDF links (conference proceedings or paywalled):

| # | Paper | Issue | Recommendation |
|---|-------|-------|----------------|
| 09 | DynamicSubsetSelection | SustainNLP Workshop only | Keep—niche venue, training dynamics angle |
| 16 | HumanLLMCollabAnnotation | ACM paywall | **Keep**—directly relevant to ECG concept |
| 29 | ExplanationFinetuning | ACL Anthology (no direct PDF) | Keep—explanation-guided training |
| 45 | NLExplanationUncertainty | AISTATS proceedings only | Consider dropping—tangential to core ECG |
| 55 | LEND | Springer DOI paywall | **Keep**—embedding similarity central to ECG |
| 95 | RapidIn | ACL 2024 proceedings | Keep—influence baseline, code available |
| 102 | EarlyStoppingNoise | NeurIPS 2021 proceedings | Consider dropping—early stopping is peripheral |

**Action:** Access via institutional login or author preprints. Papers marked "Consider dropping" are lower-priority for the ECG literature bank.

---

## Category Overview

| # | Category | Papers | Relevance to ECG |
|---|----------|--------|------------------|
| 1 | Label Noise Detection & Data Cleaning | 00–17 | Primary baselines (Cleanlab, loss-based, AUM) |
| 2 | Explanation-Based Debugging | 18–31 | Core conceptual space (explanations → data fixes) |
| 3 | LLM-Generated Explanations | 32–47 | Explanation generation module |
| 4 | Graph-Based Data Quality | 48–59 | Graph construction methodology |
| 5 | Spurious Correlations & Artifacts | 60–75 | Artifact-aligned noise motivation |
| 6 | Explanation Faithfulness & Evaluation | 76–92 | Downstream evaluation metrics |
| 7 | Influence Functions & Data Attribution | 93–98 | Alternative baselines |
| 8 | Robust Training Under Noise | 99–102 | Training methodology context |

---

## 1. Label Noise Detection & Data Cleaning

Core baselines for ECG comparison.

### 00. ConfidentLearning (Northcutt et al., 2021)

* **PDF:** https://arxiv.org/pdf/1911.00068.pdf
* **Code:** https://github.com/cleanlab/cleanlab
* **Summary:** Estimates a **confident joint** distribution between noisy observed labels and latent true labels to characterize and rank likely label issues. The framework assumes access to predicted class probabilities from a trained classifier and uses these to identify examples where the model disagrees with the label. Widely adopted as the standard baseline for label error detection—ECG must demonstrate advantages specifically on **artifact-aligned noise** where confident learning degrades (models confidently fit errors via spurious markers).

### 01. PervasiveLabelErrors (Northcutt et al., 2021)

* **PDF:** https://arxiv.org/pdf/2103.14749.pdf
* **Code:** https://github.com/cleanlab/label-errors
* **Summary:** Applies confident-learning-based flagging plus human validation to reveal non-trivial test-set error rates across 10+ widely-used benchmarks (ImageNet, MNIST, CIFAR, Amazon Reviews, etc.). The paper explicitly discusses **failure modes of confident learning**—particularly relevant when models are confidently wrong due to dataset artifacts or spurious correlations. Demonstrates that even "gold-standard" test sets contain 3-6% label errors on average.

### 02. TokenLabelErrors (Wang & Mueller, 2022)

* **PDF:** https://arxiv.org/pdf/2210.03920.pdf
* **Code:** https://github.com/cleanlab/token-label-error-benchmarks
* **Summary:** Extends confident learning to **token-level classification** (NER, POS tagging). Studies multiple probability-based scoring rules and finds that simple sentence-level aggregation (e.g., scoring based on the "worst token" in a sequence) reliably prioritizes sentences containing annotation errors. Demonstrates that token-level noise detection requires different strategies than document-level classification.

### 03. MultiLabelErrors (Thyagarajan et al., 2023)

* **PDF:** https://arxiv.org/pdf/2211.13895.pdf
* **Summary:** Extends confident-learning ideas to **multi-label classification** where each instance can have multiple correct labels. Proposes label-quality scores that rank likely annotation mistakes higher than correct labels, handling the case where "noise" manifests as structured/non-uniform patterns (missing labels, spurious labels) rather than simple random flips. Integrated into cleanlab's multi-label utilities.

### 04. DeCoLe (Li et al., 2025)

* **PDF:** https://arxiv.org/pdf/2507.07216.pdf
* **Summary:** Proposes **Decoupled Confident Learning** for mislabeling detection in **label-biased settings** (e.g., hate-speech annotation with demographic bias). Decouples the detection mechanism from the proxy classifier so it can better surface bias-driven mislabels—a common **artifact-aligned failure mode** where annotators systematically mislabel based on surface features. Directly addresses cases where standard confident learning conflates bias with noise.

### 05. PLMLabelErrors (Chong et al., 2022)

* **PDF:** https://arxiv.org/pdf/2205.12702.pdf
* **Code:** https://github.com/dcx/lnlfm
* **Summary:** Demonstrates that simply ranking examples by **out-of-sample fine-tuning loss** with large pre-trained language models can outperform more complex label-error detectors in NLP. Crucially introduces a procedure to inject **realistic human-originated noise** (harder to detect than synthetic uniform flips) by having annotators re-label data under time pressure. Shows PLM loss-ranking achieves near-oracle performance on human-originated noise.

### 06. AUM (Pleiss et al., 2020)

* **PDF:** https://arxiv.org/pdf/2001.10528.pdf
* **Code:** https://github.com/asappresearch/aum
* **Summary:** Introduces **Area Under the Margin (AUM)**, a training-dynamics statistic computed from the margin trajectory (logit difference between assigned label and next-highest class) across training epochs. Proposes a clever thresholding procedure using an extra "purposefully mislabeled" class as a calibration reference. AUM isolates mislabeled examples more robustly than raw loss in regimes where small-loss assumptions fail.

### 07. CTRL (Yue & Jha, 2024)

* **PDF:** https://arxiv.org/pdf/2208.08464.pdf
* **Code:** https://github.com/chang-yue/ctrl
* **Summary:** Detects label errors by **clustering per-example training loss curves**—exploiting the observation that clean and noisy examples exhibit different learning dynamics (clean examples have smooth loss decay; noisy examples show irregular patterns). After identifying likely-noisy clusters, retrains on the cleaned subset. Often competitive with other detectors and serves as a useful non-graph baseline.

### 08. SecondSplitForgetting (Maini et al., 2022)

* **PDF:** https://arxiv.org/pdf/2210.15031.pdf
* **Summary:** Uses **"second-split forgetting"** signals to characterize datapoints: trains on one split, then measures how quickly examples from a held-out second split are forgotten during continued training. Examples that are quickly forgotten tend to be hard, ambiguous, or mislabeled. Offers a training-dynamics alternative to loss-only filtering that captures different failure modes.

### 09. DynamicSubsetSelection (Attendu et al., 2023) ⚠️

* **PDF:** SustainNLP Workshop (ACL Anthology) — *no direct link*
* **Summary:** Investigates data subset selection and pruning for NLP classification using training-dynamics-style scores (e.g., EL2N-style gradient norm signals). Proposes **dynamic pruning strategies** where the subset is updated during training rather than fixed upfront. Relevant for building practical "filter then retrain" cleaning pipelines that adapt as model representations improve.

### 10. BERTLabelNoise (Zhu et al., 2022)

* **PDF:** https://arxiv.org/pdf/2204.09371.pdf
* **Code:** https://github.com/huanzhang12/NoisyLabelTextClassification
* **Summary:** Systematic study of **BERT's robustness to label noise** in text classification, providing practical baselines and failure analyses. Shows that while BERT is more robust than earlier models, it still degrades significantly under instance-dependent noise. Justifies why explanation-graph inconsistency signals are needed rather than relying on PLM robustness alone.

### 11. NoisyNERConfidence (Liu et al., 2021)

* **PDF:** https://arxiv.org/pdf/2104.04318.pdf
* **Code:** https://github.com/liukun95/Noisy-NER-Confidence-Estimation
* **Summary:** Estimates **calibrated confidence scores for NER labels** under weak/distant annotation (e.g., dictionary matching, crowdsourcing). Integrates per-token confidence into training via self-training with confidence thresholds. Provides a structured-output analog of neighborhood-consistency methods for sequence labeling tasks.

### 12. NoisyMultiLabelText (Xu et al., 2024)

* **PDF:** https://aclanthology.org/2024.findings-naacl.93.pdf
* **Summary:** Treats noise at the **(instance, label) pair level** rather than instance-level—distinguishing false positives (spurious labels) from false negatives (missing labels). Proposes correction mechanisms specifically tailored to multi-label text classification where these asymmetric errors have different effects on model behavior.

### 13. LLMaAA (Zhang et al., 2023)

* **PDF:** https://arxiv.org/pdf/2310.19561.pdf
* **Summary:** Positions LLMs as **active annotation agents** within an annotation loop, aiming to reduce human labeling costs by intelligently selecting which items to (re-)annotate. Uses LLM confidence and disagreement signals to prioritize uncertain examples. Serves as a baseline for comparing ECG's explanation-based approach to simpler LLM label verification.

### 14. NoiseRobustLLMAnnotators (Yuan et al., 2024)

* **PDF:** https://arxiv.org/pdf/2402.06713.pdf
* **Summary:** Explicitly models **LLM annotations as noisy** and proposes a noise-robust active learning framework. Recognizes that LLM verification can be artifact-sensitive or inconsistent—sometimes confidently wrong in systematic ways. Proposes collaborative learning between multiple LLM annotators to reduce correlated errors.

### 15. ChatGPTAnnotation (Reiss, 2023)

* **PDF:** https://arxiv.org/pdf/2304.12306.pdf
* **Summary:** Empirically evaluates ChatGPT's reliability as an annotator/classifier across multiple NLP tasks. Provides evidence of systematic failure modes and caveats—particularly relevant when artifacts induce systematic annotation errors that propagate through LLM-based verification pipelines. Shows significant variation in quality across task types.

### 16. HumanLLMCollabAnnotation (Wang et al., 2024) ⚠️

* **PDF:** https://dl.acm.org/doi/pdf/10.1145/3613904.3641960 — *ACM paywall*
* **Summary:** Proposes a multi-step pipeline where LLMs produce **labels and explanations simultaneously**, a verifier scores label quality using multiple signals (including explanation consistency), and humans focus on low-verification items. Highly aligned with ECG's core insight that explanations contain signals beyond the label itself for identifying problematic annotations.

### 17. AnnoLLM (He et al., 2023)

* **PDF:** https://arxiv.org/pdf/2303.16854.pdf
* **Summary:** Introduces **"explain-then-annotate"** prompting: LLMs first generate explanations for their labels, then those explanations are reused in few-shot prompts to improve annotation quality on subsequent examples. Shows that requiring explanations improves annotation consistency. Directly relevant for structured rationale generation in ECG.

---

## 2. Explanation-Based Debugging

Core conceptual predecessors for ECG.

### 18. ExplanationDebuggingSurvey (Lertvittayakumjorn & Toni, 2021)

* **PDF:** https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00440/1979190/tacl_a_00440.pdf
* **Summary:** Comprehensive survey systematizing **"explanations → human feedback → model/data fixes"** pipelines. Categorizes debugging approaches by explanation type (local vs global, instance-based vs rule-based) and feedback mechanism. Explicitly discusses how explanations can surface dataset artifacts and labeling issues during debugging workflows. Essential framing paper for ECG.

### 19. FIND (Lertvittayakumjorn et al., 2020)

* **PDF:** https://arxiv.org/pdf/2010.04987.pdf
* **Code:** https://github.com/plkumjorn/FIND
* **Summary:** Human-in-the-loop debugging workflow for text classifiers where explanations (attention, gradient-based saliency) help users discover spurious patterns and dataset issues driving incorrect behavior. Users can then provide targeted fixes (relabeling, removing, augmenting) beyond what loss/confidence signals would surface. Demonstrates explanations enable more efficient debugging than pure example selection.

### 20. HILDIF (Zylberajch et al., 2021)

* **PDF:** https://aclanthology.org/2021.internlp-1.1.pdf
* **Summary:** Uses **influence functions as explanations** to show which training instances most drive NLI predictions. When users identify that predictions rely on artifact-laden training examples, their feedback (relabeling/removal) mitigates artifact-driven behavior. Explicitly links interpretability tools to data artifact discovery in NLI.

### 21. XMD (Lee et al., 2023)

* **PDF:** https://arxiv.org/pdf/2210.16978.pdf
* **Code:** https://github.com/INK-USC/XMD
* **Summary:** End-to-end interactive system that surfaces explanations, collects user feedback on whether highlighted features are valid, and updates models via **explanation-alignment regularization**. Aimed at correcting spurious biases (often originating from training data) by constraining models to match user-validated rationales. Demonstrates the full loop from explanation → feedback → model improvement.

### 22. LIT (Tenney et al., 2020)

* **PDF:** https://arxiv.org/pdf/2008.05122.pdf
* **Code:** https://github.com/PAIR-code/lit
* **Summary:** **Language Interpretability Tool**—an extensible, interactive platform for NLP model analysis. Combines attention visualization, integrated gradients, LIME, counterfactual generation, and data slicing. Commonly used as a practical layer for discovering systematic data issues through interactive exploration of model behavior across subpopulations.

### 23. InteractiveLabelCleaning (Teso et al., 2021)

* **PDF:** https://arxiv.org/pdf/2106.03922.pdf
* **Code:** https://github.com/abonte/cincer
* **Summary:** Uses **example-based explanations** (influential training examples, prototypes) to guide humans toward likely label errors. The key insight is that when a prediction's influential examples have inconsistent labels, the training label is suspect. Explicitly targets training-set quality improvement rather than just model interpretability.

### 24. TFA-ArtifactDetection (Pezeshkpour et al., 2022)

* **PDF:** https://arxiv.org/pdf/2107.00323.pdf
* **Code:** https://github.com/pouyapez/artifact_detection
* **Summary:** Proposes **Training-Feature Attribution (TFA)**: explanations that jointly localize **which tokens in which influential training examples** drive a prediction. Explicitly designed to uncover training-set artifacts/spurious markers by revealing when predictions depend on superficial token patterns in training data. Includes user study validating that TFA helps humans discover artifacts.

### 25. InfluenceArtifacts (Han et al., 2020)

* **PDF:** https://arxiv.org/pdf/2005.06676.pdf
* **Summary:** Argues that **instance-level explanations** (influence functions) are better-suited than token saliency for revealing training data artifacts in some NLP settings. Introduces an influence-based quantitative measure for artifact reliance. Shows that models can appear to use correct features on test inputs while actually depending on artifactual patterns in training data.

### 26. SpuriousCorrelationsNLP (Wang et al., 2022)

* **PDF:** https://aclanthology.org/2022.findings-naacl.129.pdf
* **Code:** https://github.com/tianlu-wang/Identifying-and-Mitigating-Spurious-Correlations
* **Summary:** Uses **interpretability-driven analysis** to identify spurious correlates and proposes mitigation strategies (reweighting, data augmentation). Explicitly targets robustness failures caused by shortcut features that are invisible to loss-based diagnostics—you can only find them by looking at what the model attends to.

### 27. CompetencyProblems (Gardner et al., 2021)

* **PDF:** https://aclanthology.org/2021.emnlp-main.135.pdf
* **Summary:** Proposes a **data-centric methodology** to discover dataset artifacts/heuristics and remove them to better measure true model competency. Constructs "challenge" evaluation slices where spurious features are deliberately broken, revealing which models actually learned the task vs. exploited shortcuts.

### 28. FeatureAttributionPriors (Liu & Avci, 2019)

* **PDF:** https://arxiv.org/pdf/1906.08286.pdf
* **Summary:** Adds an **attribution-matching loss** so practitioners can enforce priors like "don't rely on identity tokens" or "focus on toxic terms rather than demographic mentions." Directly uses gradient-based explanations as a training signal to counter spurious features. Pioneering work on explanation-guided training for debiasing.

### 29. ExplanationFinetuning (Ludan et al., 2023) ⚠️

* **PDF:** https://aclanthology.org/2023.acl-long.pdf — *no direct link*
* **Summary:** Uses **explanation signals during finetuning** to reduce reliance on spurious cues. Compares explanation-regularized models against standard finetuning and shows improved robustness on challenge sets where shortcuts fail, even when standard finetuning achieves similar in-domain accuracy.

### 30. ER-Test (Joshi et al., 2022)

* **PDF:** https://arxiv.org/pdf/2210.09635.pdf
* **Code:** https://github.com/brihi-joshi/ER-Test
* **Summary:** Provides a rigorous **evaluation protocol for explanation-regularized training**—testing whether explanation constraints actually improve robustness/behavior rather than just changing saliency maps. Emphasizes OOD generalization via unseen datasets, contrast sets, and functional tests. Essential for validating that explanation-based interventions have real effects.

### 31. REFER (Madani & Minervini, 2023)

* **PDF:** https://aclanthology.org/2023.conll-1.40.pdf
* **Summary:** **Rationale Extraction for Explanation Regularization**—extracts rationales from model attention or gradient attributions to support explanation regularization. Tightens the loop "rationales → constraints → improved generalization" for settings where spurious evidence must be discouraged during training.

---

## 3. LLM-Generated Explanations

Technical foundations for explanation generation module.

### 32. GrammarConstrainedDecoding (Geng et al., 2023)

* **PDF:** https://arxiv.org/pdf/2305.13971.pdf
* **Code:** Available
* **Summary:** Uses **formal grammars (including input-dependent grammars)** to constrain LLM decoding so outputs are **guaranteed** to match a target structure. Demonstrates strong gains on structured tasks (information extraction, entity disambiguation, semantic parsing) without finetuning. Essential for enforcing parseable JSON explanations in ECG—eliminates post-hoc JSON repair.

### 33. DOMINO (Beurer-Kellner et al., 2024)

* **PDF:** https://arxiv.org/pdf/2403.06988.pdf
* **Summary:** Proposes **subword-aligned constrained decoding** designed to reduce overhead and avoid accuracy loss from token–constraint misalignment (when JSON structure boundaries don't align with subword tokens). Relevant for strict JSON/schema constraints without the brittleness of prompt-only formatting.

### 34. FOFO (Xia et al., 2024)

* **PDF:** https://arxiv.org/pdf/2402.18667.pdf
* **Code:** Available
* **Summary:** Introduces a **benchmark specifically for complex format-following** capabilities of LLMs. Evaluates open and closed models and shows open models (Llama, Mistral) lag substantially behind GPT-4 on strict structural adherence. Critical for selecting/finetuning an open model to reliably emit JSON explanations for ECG.

### 35. SLOT (Wang et al., 2025)

* **PDF:** https://arxiv.org/pdf/2505.04016.pdf
* **Summary:** Instead of relying solely on constrained decoding, uses a **lightweight post-processing model** to transform "nearly structured" LLM outputs into schema-valid structured outputs. Reports strong schema accuracy with Mistral/Llama variants. Useful fallback when constrained decoding is too slow or unavailable.

### 36. StructuredOutputBenchmark (Geng et al., 2025)

* **PDF:** https://arxiv.org/pdf/2501.10868.pdf
* **Summary:** Benchmarks constrained decoding approaches specifically for **JSON Schema compliance**. Evaluates three dimensions: (i) constraint compliance efficiency, (ii) constraint coverage (which schemas can be enforced), (iii) output quality (does constraining hurt generation quality). Essential for choosing ECG's decoding strategy.

### 37. LLMasJudgeSurvey (Gu et al., 2024)

* **PDF:** https://arxiv.org/pdf/2411.15594.pdf
* **Summary:** Systematizes **"LLM-as-judge" reliability issues**: consistency across prompts, positional bias, self-preference bias, and scenario adaptation failures. Proposes evaluation methodologies and benchmarks. Critical context if using LLMs to adjudicate contradictions between (instance, rationale, label) triples in ECG.

### 38. LLMasActiveAnnotator (Zhang et al., 2023)

* **PDF:** https://arxiv.org/pdf/2310.19596.pdf
* **Summary:** Introduces **LLMaAA framework** putting LLM annotators into an active-learning loop to select what to annotate efficiently. Relevant as a control baseline for "LLM-generated label verification" pipelines—ECG should outperform simple active annotation.

### 39. LLMSelfExplanations (Huang et al., 2023)

* **PDF:** https://arxiv.org/pdf/2310.11207.pdf
* **Summary:** Systematic study of eliciting **feature-attribution-style self-explanations** from LLMs (e.g., "which words were most important for your decision?") and evaluating them against faithfulness metrics from traditional explanation methods. Good grounding for understanding extractive span rationales vs. free-form generated rationales in ECG.

### 40. LLMSelfExplanationFaithfulness (Madsen et al., 2024)

* **PDF:** https://arxiv.org/pdf/2401.07927.pdf
* **Summary:** Uses **self-consistency checks** to test whether different self-explanation types (counterfactual, feature attribution, redaction-based) are faithful to the model's actual decision process. Shows faithfulness depends heavily on explanation type, model family, and task. Evaluates open models (Llama2, Mistral, Falcon)—directly applicable to ECG model selection.

### 41. FaithfulnessPlausibility (Agarwal et al., 2024)

* **PDF:** https://arxiv.org/pdf/2402.04614.pdf
* **Summary:** Analyzes the **gap between plausible explanations and faithful explanations** for LLMs. A plausible explanation sounds convincing to humans but may not reflect the model's true reasoning. Frames why ECG needs verification (NLI checks, perturbation tests) rather than naively trusting rationales.

### 42. SelfExplanationReliability (Randl et al., 2024)

* **PDF:** https://arxiv.org/pdf/2407.14487.pdf
* **Summary:** Evaluates **extractive vs. counterfactual self-explanations** and reports that plausibility can diverge significantly from faithfulness. Argues counterfactual explanations ("I would have predicted X if Y were different") are more verifiable with tailored prompts—useful for building contradiction signals in ECG.

### 43. LLMRationaleFaithfulness (Fayyaz et al., 2024)

* **PDF:** https://arxiv.org/pdf/2407.00219.pdf
* **Summary:** Compares **human-perceived quality vs. model-faithfulness** for rationales. Key finding: human alignment does not guarantee faithfulness—rationales that humans rate as "good" may not reflect actual model reasoning. Directly relevant for why ECG uses NLI/graph checks rather than human plausibility judgments.

### 44. NLExplanationFaithfulness (Parcalabescu et al., 2024)

* **PDF:** https://aclanthology.org/2024.acl-long.329.pdf
* **Summary:** Focuses on **input perturbation testing** as a route to faithfulness measurement: if an explanation claims feature X is important, removing X should change the prediction. Fits directly with ECG's approach of verifying explanations before using them for data cleaning.

### 45. NLExplanationUncertainty (Tanneru et al., 2024) ⚠️

* **PDF:** AISTATS 2024 Proceedings — *no direct link, consider dropping*
* **Summary:** Studies **uncertainty specifically in natural-language explanations**—can we calibrate how confident a model is in its rationale? Useful for treating rationale confidence as a calibrated signal for contradiction detection or graph edge weighting in ECG.

### 46. SaySelf (Xu et al., 2024)

* **PDF:** https://arxiv.org/pdf/2405.20974.pdf
* **Code:** Available
* **Summary:** Training framework to improve **fine-grained confidence calibration** and generate **self-reflective rationales** that explain uncertainty. Uses inconsistency across sampled reasoning chains + SFT/RL to train. Directly relevant to ECG's "predicted label + confidence + rationale" JSON fields.

### 47. ExplanationConsistencyFinetuning (Chen et al., 2025)

* **PDF:** https://arxiv.org/pdf/2401.13986.pdf
* **Code:** Available
* **Summary:** Proposes **explanation-consistency finetuning** to make explanations more stable/consistent across semantically equivalent inputs. Relevant for ECG because deterministic, parseable rationales are needed for downstream graph construction and contradiction checks—high explanation variance is problematic.

---

## 4. Graph-Based Data Quality

Methodological foundations for ECG graph construction.

### 48. DeepKNNNoisyLabels (Bahri et al., 2020)

* **PDF:** https://arxiv.org/pdf/2004.12289.pdf
* **Summary:** Proposes using **deep feature kNN structure** to identify/mitigate noisy labels. Core insight: if an example's k-nearest neighbors in representation space have different labels, the example's label is suspect. Leverages neighborhood agreement for robustness to mislabeled points. **Core methodology for ECG**—replace raw-input embeddings with explanation embeddings.

### 49. WANN (Di Salvo et al., 2025)

* **PDF:** https://arxiv.org/pdf/2408.14358.pdf
* **Code:** https://github.com/francescodisalvo05/wann-noisy-labels
* **Summary:** Uses **foundation-model embeddings + weighted adaptive kNN voting** with a learned **label reliability score** to downweight likely-mislabeled neighbors. Designed to be both efficient and explainable—the reliability score provides interpretable per-example trust estimates. Directly applicable to ECG's weighted graph construction.

### 50. KNNRepresentations (Rajani et al., 2020)

* **PDF:** https://arxiv.org/pdf/2010.09030.pdf
* **Summary:** Uses **kNN over learned representations** to explain predictions and diagnose model behavior, including failure modes consistent with mislabeled/atypical examples. Shows that neighbors provide interpretable debugging signals—an example's prediction can be explained by "because training examples A, B, C are similar and have label Y."

### 51. DualKNNText (Yuan et al., 2025)

* **PDF:** https://arxiv.org/pdf/2503.04869.pdf
* **Summary:** Proposes a **dual-nearest-neighbor text classifier** using both text embeddings and label-probability representations. Explicitly discusses how NN retrieval can be **confused by noisy datasets** and adds a label-distribution/contrastive component to stabilize neighbor quality under noise.

### 52. NeuralRelationGraph (Kim et al., 2023)

* **PDF:** https://arxiv.org/pdf/2301.12321.pdf
* **Code:** https://github.com/snu-mllab/Neural-Relation-Graph
* **Summary:** Builds an explicit **relational graph in feature space** and provides scalable algorithms for **joint label error detection + outlier/OOD detection**. Evaluated across domains including **language (SST-2)**. Highly aligned with ECG's "graph over explanation embeddings" design—demonstrates the approach works for NLP.

### 53. GCNNoisyLabels (Iscen et al., 2020)

* **PDF:** https://arxiv.org/pdf/2011.00359.pdf
* **Summary:** Constructs a **kNN graph in embedding space** and trains a **Graph Convolutional Network** to propagate/denoise labels when only a small clean subset exists. The GCN learns to smooth labels across the graph while respecting local structure. Directly portable to kNN graphs over explanation embeddings.

### 54. LabelPropagationHateSpeech (D'Sa et al., 2020)

* **PDF:** https://aclanthology.org/2020.insights-1.8.pdf
* **Summary:** Applies **label propagation on a similarity graph** for low-resource hate speech text classification. Explicitly ties classification performance to representation quality—better embeddings yield better graph structure. Establishes NLP precedent for graph-based semi-supervised learning on embedding graphs.

### 55. LEND (Zhang et al., 2022) ⚠️

* **PDF:** https://doi.org/10.1007/s10994-022-06197-6 — *Springer paywall*
* **Summary:** Computes an **embedding similarity matrix** to capture local structure and **dilute noisy supervision** by overwhelming mislabeled signals with nearby consistent neighbors. Conceptually close to ECG's neighborhood-consistency scoring—mislabeled examples are "outvoted" by their clean neighbors.

### 56. BeyondImagesNoise (Zhu et al., 2022)

* **PDF:** https://arxiv.org/pdf/2208.09329.pdf
* **Summary:** Targets settings where features are **less "clean" than vision** (explicitly framed as "beyond images"). Proposes improved noise transition matrix estimation for lower-quality feature spaces. Relevant for NLP where text embeddings may be noisier than ImageNet features.

### 57. SelCL (Li et al., 2022)

* **PDF:** https://arxiv.org/pdf/2203.04181.pdf
* **Code:** https://github.com/ShikunLi/Sel-CL
* **Summary:** Shows that supervised contrastive learning degrades due to **noisy positive/negative pairs** and proposes selecting confident examples/pairs based on representation/label agreement. Closely related to neighborhood consistency but operates in pair space rather than neighborhood space.

### 58. RINCE (Chuang et al., 2022)

* **PDF:** https://arxiv.org/pdf/2201.04309.pdf
* **Code:** https://github.com/chingyaoc/RINCE
* **Summary:** Addresses **noise in contrastive positives** ("noisy views") by introducing robust contrastive objectives (Robust InfoNCE variants). Useful if ECG uses contrastive embedding learning before building the explanation-embedding kNN graph—ensures the embedding space is robust to noise.

### 59. DataInf (Kwon et al., 2024)

* **PDF:** https://arxiv.org/pdf/2310.00902.pdf
* **Code:** https://github.com/ykwon0407/DataInf
* **Summary:** **Efficient influence-style data attribution** for large models including RoBERTa and Llama-2 (LoRA-tuned). Demonstrates that influence scores can identify mislabeled points. Complements ECG's graph-based inconsistency by adding a "who affected this prediction?" view.

---

## 5. Spurious Correlations & Artifacts

Motivation for artifact-aligned noise experiments.

### 60. HypothesisOnlyNLI (Poliak et al., 2018)

* **PDF:** https://arxiv.org/pdf/1805.01042.pdf
* **Code:** https://github.com/azpoliak/hypothesis-only-NLI
* **Summary:** Establishes **"hypothesis-only" models** as a diagnostic for annotation artifacts in NLI. Shows that models can achieve strong performance on SNLI/MNLI using only the hypothesis, indicating **label-predictive lexical irregularities** in the data creation process (e.g., negation words correlate with "contradiction" labels). Foundational paper for NLI artifact analysis.

### 61. PremiseMitigation (Belinkov et al., 2019)

* **PDF:** https://arxiv.org/pdf/1907.04380.pdf
* **Code:** https://github.com/azpoliak/robust-nli
* **Summary:** Proposes training objectives that **discourage hypothesis-only shortcuts** by effectively forcing premise use (via probabilistic reformulation). Improves transfer when training and test distributions differ in artifact structure—models trained with mitigation generalize better to challenge sets.

### 62. PartialInputFailures (Feng et al., 2019)

* **PDF:** https://arxiv.org/pdf/1905.05778.pdf
* **Summary:** Shows that **low performance of partial-input baselines does NOT imply artifact-free data**. Constructs cases where artifacts exist but evade partial-input detection (artifacts that require both premise and hypothesis to exploit). Critical warning that artifact detection methods have blind spots.

### 63. ProductOfExperts (Clark et al., 2019)

* **PDF:** https://arxiv.org/pdf/1909.03683.pdf
* **Code:** https://github.com/chrisc36/debias
* **Summary:** Trains a **"bias-only" model** (hypothesis-only for NLI) and combines it with a main model in an ensemble objective (product-of-experts) to reduce reliance on known shortcuts. The bias model "soaks up" the easy signal, forcing the main model to learn complementary features. Widely-used debiasing template.

### 64. BiasModeling (Mahabadi et al., 2020)

* **PDF:** https://arxiv.org/pdf/1909.06321.pdf
* **Code:** https://github.com/rabeehk/robust-nli
* **Summary:** Uses predictions from bias-only models to **reweight/down-weight biased examples** during training, focusing learning on hard examples where shortcuts don't work. Improves robustness on out-of-domain and bias-sensitive evaluations across NLI and fact verification.

### 65. SelfDebiasing (Utama et al., 2020)

* **PDF:** https://arxiv.org/pdf/2009.12303.pdf
* **Code:** https://github.com/UKPLab/emnlp2020-debiasing-unknown
* **Summary:** **Self-debiasing framework** that does **not require knowing the bias type** in advance. Uses the model's own confidence to identify and downweight likely-biased examples, preventing overconfidence on shortcut-exploitable inputs while preserving performance on challenge sets.

### 66. MixedCapacityEnsembles (Clark et al., 2020)

* **PDF:** https://arxiv.org/pdf/2011.03856.pdf
* **Summary:** Uses a **low-capacity model to absorb shallow dataset-specific correlations** while a higher-capacity model learns complementary signals. Enforces non-overlap via conditional independence constraint to reduce shortcut reliance without specifying the bias upfront.

### 67. SpuriousCorrelationsPLM (Tu et al., 2020)

* **PDF:** https://arxiv.org/pdf/2007.06778.pdf
* **Code:** https://github.com/lifu-tu/Study-NLP-Robustness
* **Summary:** Demonstrates that robustness gains from pretraining often come from **generalizing from a small number of counterexamples** where spurious cues break. When such counterexamples are rare, even PLMs fail. Proposes multi-task learning to help when minority counterexamples are extremely scarce.

### 68. SpuriousTextClassification (Wang & Culotta, 2020)

* **PDF:** https://arxiv.org/pdf/2010.02458.pdf
* **Summary:** Proposes a method to classify features/terms as **"spurious" vs. "genuine"** using treatment-effect-derived signals (if we intervene on this feature, does the true label change?). Uses this classification to guide feature selection and improve worst-case robustness in sentiment classification.

### 69. CounterfactualRobustness (Wang & Culotta, 2021)

* **PDF:** https://arxiv.org/pdf/2012.10040.pdf
* **Summary:** Identifies likely causal features, **generates counterfactuals** by substituting them (e.g., antonyms for sentiment words), flips labels accordingly, and retrains to reduce spurious reliance. Reports robustness gains on counterfactual/human-edited distribution shifts.

### 70. CAD (Kaushik et al., 2020)

* **PDF:** https://arxiv.org/pdf/1909.12434.pdf
* **Summary:** Introduces **human-in-the-loop counterfactual rewriting**: annotators make minimal edits to flip the label (e.g., changing "loved" to "hated" in sentiment). Training on these pairs breaks shortcut correlations because the spurious features are held constant while causal features change. Widely used for stress-testing and improving robustness.

### 71. CADEfficacy (Kaushik et al., 2020)

* **PDF:** https://arxiv.org/pdf/2010.02114.pdf
* **Summary:** Provides an explanatory account of **why counterfactually-augmented data improves OOD generalization**. Proposes concrete noise injection experiments to probe whether attribution methods are highlighting causal vs. spurious spans—highly aligned with "using explanations to detect artifacts."

### 72. FEVERDebiasing (Schuster et al., 2019)

* **PDF:** https://arxiv.org/pdf/1908.05267.pdf
* **Code:** https://github.com/TalSchuster/FeverSymmetric
* **Summary:** Shows FEVER fact verification can be partially "solved" with **claim-only cues** (ignoring evidence entirely). Builds a **Symmetric FEVER evaluation set** where claim-only shortcuts don't work, and proposes regularization to improve performance on this robust evaluation.

### 73. CrossAug (Lee et al., 2021)

* **PDF:** https://arxiv.org/pdf/2109.15107.pdf
* **Code:** https://github.com/minwhoo/CrossAug
* **Summary:** Two-stage **contrastive augmentation pipeline** for fact verification debiasing. Generates hard negatives that break claim-only shortcuts. Reports gains particularly on Symmetric FEVER-style evaluations.

### 74. FEVEROUS (Aly et al., 2021)

* **PDF:** https://arxiv.org/pdf/2106.05707.pdf
* **Summary:** FEVER-style benchmark extending verification to **tables + text**. Explicitly discusses measuring/minimizing biases such as predicting labels without evidence. Provides structured multi-hop reasoning challenges.

### 75. GroupDRO (Sagawa et al., 2020)

* **PDF:** https://arxiv.org/pdf/1911.08731.pdf
* **Summary:** Canonical **Distributionally Robust Optimization** formulation minimizing worst-group loss to combat spurious correlations that fail on minority groups. Also emphasizes the role of regularization (weight decay) for worst-case generalization. Foundational for group robustness work.

---

## 6. Explanation Faithfulness & Evaluation

Downstream evaluation for ECG.

### 76. ERASER (DeYoung et al., 2020)

* **PDF:** https://arxiv.org/pdf/1911.03429.pdf
* **Code:** ERASER-benchmark
* **Summary:** Unified **benchmark for rationalized NLP** spanning 7 datasets with human-annotated rationales. Standardizes rationale evaluation including **comprehensiveness** (how much does performance drop when rationale is removed?) and **sufficiency** (how well can the model predict from rationale alone?). Essential benchmark for ECG's downstream evaluation.

### 77. ERTestBenchmark (Joshi et al., 2022)

* **PDF:** https://arxiv.org/pdf/2205.12542.pdf
* **Code:** https://github.com/INK-USC/ER-Test
* **Summary:** Proposes **ER-TEST**, an evaluation protocol for explanation-regularized training. Tests whether explanation constraints actually improve robustness via **OOD generalization** (unseen datasets, contrast sets, functional tests) rather than just changing saliency maps while leaving behavior unchanged.

### 78. GoodhartExplanations (Hsia et al., 2024)

* **PDF:** https://arxiv.org/pdf/2402.18374.pdf
* **Code:** https://github.com/IREXorg/Goodharts-Law-Explanation-Benchmark
* **Summary:** Shows that **optimizing for explanation benchmark scores can yield metric gaming** (Goodhart effects)—models learn to produce rationales that score well without actually changing their reasoning. Critical warning for ECG evaluation design.

### 79. FaithfulnessMetrics (Chan et al., 2022)

* **PDF:** https://arxiv.org/pdf/2204.05514.pdf
* **Summary:** Compares multiple **faithfulness metrics** (removal-based, perturbation-based) and highlights **metric instability and disagreement** when ranking interpretability methods. Different metrics produce different rankings. Important for understanding ECG evaluation limitations.

### 80. InterpretationSensitivity (Yin et al., 2022)

* **PDF:** https://arxiv.org/pdf/2104.08782.pdf
* **Code:** https://github.com/uclanlp/NLP-Interpretation-Faithfulness
* **Summary:** Re-frames faithfulness through **robustness-style notions** (sensitivity to input perturbations, stability across random seeds). Systematically evaluates interpretations under these criteria. Provides alternative evaluation approaches for ECG.

### 81. RecursiveROAR (Madsen et al., 2022)

* **PDF:** https://arxiv.org/pdf/2110.08412.pdf
* **Code:** https://github.com/AndreasMadsen/nlp-roar-interpretability
* **Summary:** Proposes **Recursive ROAR** (repeatedly mask important tokens and retrain) plus RACU summary statistic to better quantify faithfulness while reducing **OOD masking artifacts** (the problem that masked inputs are out-of-distribution). Improved methodology for comprehensiveness evaluation.

### 82. ROAR (Hooker et al., 2019)

* **PDF:** https://arxiv.org/pdf/1806.10758.pdf
* **Summary:** Introduces **RemOve And Retrain (ROAR)** as a faithfulness evaluation: remove top-attributed features and **retrain** a new model to avoid misleading OOD masking effects (evaluating a model on masked inputs it was never trained on). Foundational methodology, though expensive.

### 83. AttentionNotExplanation (Jain & Wallace, 2019)

* **PDF:** https://arxiv.org/pdf/1902.10186.pdf
* **Code:** https://github.com/successar/AttentionExplanation
* **Summary:** Tests whether attention weights are faithful explanations. Demonstrates attention can be **uncorrelated with gradient-based importance** and that adversarially-constructed alternative attention distributions can yield nearly identical predictions. Sparked the attention-as-explanation debate.

### 84. AttentionNotNotExplanation (Wiegreffe & Pinter, 2019)

* **PDF:** https://arxiv.org/pdf/1908.04626.pdf
* **Code:** https://github.com/sarahwie/attention
* **Summary:** Dissects why prior "attention is not explanation" tests may be insufficient. Argues that explanation validity depends on definitions—**faithfulness vs. plausibility** are different properties. Proposes alternative tests and nuanced conclusions.

### 85. FaithfulnessViolationTest (Liu et al., 2022)

* **PDF:** https://arxiv.org/pdf/2201.12114.pdf
* **Code:** https://github.com/BierOne/Attention-Faithfulness
* **Summary:** Introduces a **faithfulness violation test** checking whether explanation weights correctly capture the **polarity of feature impact** (support vs. suppression). Finds violations for common attention-based explanations—features marked as "important" sometimes hurt rather than help the prediction.

### 86. ExplainabilityDiagnostic (Atanasova et al., 2020)

* **PDF:** https://arxiv.org/pdf/2009.13295.pdf
* **Summary:** Proposes a battery of **diagnostic properties** and compares multiple post-hoc text explainers (gradient, perturbation, attention families) including agreement with human-annotated salient regions. Comprehensive empirical comparison.

### 87. DiscretizedIG (Sanyal et al., 2021)

* **PDF:** https://aclanthology.org/2021.emnlp-main.805.pdf
* **Summary:** Makes **Integrated Gradients** more faithful for text by using interpolation strategies better suited to discrete token/embedding structure. Standard IG uses continuous interpolation that passes through unrealistic intermediate states; DIG stays closer to valid text representations.

### 88. LabelRationaleAssociation (Wiegreffe et al., 2021)

* **PDF:** https://aclanthology.org/2021.emnlp-main.760.pdf
* **Code:** https://github.com/sarahwie/label-rationale-association
* **Summary:** Studies when free-text rationales correlate with labels in ways enabling **label leakage**—a model can predict the label just from the rationale without looking at the input. Provides measurement approaches to quantify this pitfall.

### 89. REV (Chen et al., 2023)

* **PDF:** https://arxiv.org/pdf/2210.04982.pdf
* **Code:** https://github.com/HanjieChen/REV
* **Summary:** Introduces **REV**, a conditional V-information metric estimating how much **new, label-relevant information** a rationale adds beyond the input/label baseline. Evaluated across multiple rationale benchmarks including CoT-style rationales. Information-theoretic approach to rationale quality.

### 90. RORA (Jiang et al., 2024)

* **PDF:** https://arxiv.org/pdf/2402.18678.pdf
* **Code:** https://github.com/ZhengpingJiang/RORA
* **Summary:** **Robust Free-Text Rationale Evaluation** designed to be resistant to label leakage. Reports improved alignment with human judgments compared to prior label-support metrics. Important for ECG's rationale quality assessment.

### 91. Simulatability (Hase & Bansal, 2020)

* **PDF:** https://arxiv.org/pdf/2005.01831.pdf
* **Code:** https://github.com/peterbhase/InterpretableNLP-ACL2020
* **Summary:** Establishes controlled **human simulatability protocols**: can humans predict model outputs better with explanations than without? Includes counterfactual-style prediction tests. Compares LIME, Anchors, prototypes, and others. Gold standard for human-grounded evaluation.

### 92. ALMANACS (Mills et al., 2023)

* **PDF:** https://arxiv.org/pdf/2312.12747.pdf
* **Code:** https://github.com/edmundmills/ALMANACS
* **Summary:** Scalable, **automated simulatability benchmark** using an LM "simulator" to predict another model's behavior from explanations. Avoids expensive human studies while maintaining correlation with human simulatability. Evaluates attention and IG-style explanations across safety-relevant topics.

---

## 7. Influence Functions & Data Attribution

Alternative baselines for instance-level debugging.

### 93. TracIn (Pruthi et al., 2020)

* **PDF:** https://arxiv.org/pdf/2002.08484.pdf
* **Code:** https://github.com/frederick0329/TracIn
* **Summary:** Computes influence of training points on predictions by **tracing gradient alignment across checkpoints**. More practical than classical influence functions (avoids expensive Hessian computation). Explicitly motivated by finding harmful, rare, and **mislabeled examples**. Standard influence baseline for ECG.

### 94. TRAK (Park et al., 2023)

* **PDF:** https://arxiv.org/pdf/2303.14186.pdf
* **Code:** https://github.com/MadryLab/trak
* **Summary:** **Scalable data-attribution** method tracing predictions back to training data with far lower cost than many baselines. Uses random projections to approximate influence efficiently. Enables dataset debugging and data valuation at scale.

### 95. RapidIn (Lin et al., 2024) ⚠️

* **PDF:** ACL 2024 Proceedings — *no direct link*
* **Code:** https://github.com/huawei-lin/RapidIn
* **Summary:** Proposes **token-level influence estimation** to retrieve training samples that most influence LLM outputs. Geared toward scalable influence analysis in modern LMs where sequence length and model size make traditional methods prohibitive.

### 96. LLMPretrainingInfluence (Chang et al., 2025)

* **PDF:** https://arxiv.org/pdf/2410.17413.pdf
* **Code:** https://github.com/PAIR-code/pretraining-tda
* **Summary:** Scales gradient-based training-data attribution to **LLM pretraining** (up to ~8B params, ~160B tokens) to retrieve influential examples for factual outputs. Directly relevant when artifacts/spurious correlations are baked into pretraining data rather than task-specific finetuning data.

### 97. CSIDN (Berthon et al., 2021)

* **PDF:** https://arxiv.org/pdf/2001.03772.pdf
* **Summary:** Targets **instance-dependent / feature-dependent noise** by introducing confidence-scored instance-dependent noise (CSIDN) and instance-level forward correction. Directly relevant when "noise" is artifact-aligned (depends on input features rather than being random).

### 98. WalkTheTalk (Matton et al., 2025)

* **PDF:** https://arxiv.org/pdf/2504.14150.pdf
* **Summary:** Defines faithfulness in terms of which **concepts** explanations claim are influential vs. which truly are. Estimates causal influence via **counterfactual concept interventions** + Bayesian hierarchical modeling. Strong fit for explanation verification before using rationales to flag label noise.

---

## 8. Robust Training Under Noise

Context for training methodology.

### 99. DivideMix (Li et al., 2020)

* **PDF:** https://arxiv.org/pdf/2002.07394.pdf
* **Code:** https://github.com/LiJunnan1992/DivideMix
* **Summary:** Fits a **Gaussian Mixture Model over per-sample losses** to split data into "clean/labeled" vs. "noisy/unlabeled" sets, then applies MixMatch-style semi-supervised learning. Canonical baseline when small-loss separation assumption holds. Degrades when artifacts cause confident wrong predictions.

### 100. EarlyLearningRegularization (Liu et al., 2020)

* **PDF:** https://arxiv.org/pdf/2007.00151.pdf
* **Code:** https://github.com/shengliu66/ELR
* **Summary:** Exploits the **"early learning" effect** (networks learn clean patterns first, then memorize noise later) and introduces a regularizer penalizing deviation from early predictions. Reduces memorization of noisy labels in later training—useful when confident fitting of artifacts becomes a problem.

### 101. CoDC (Zhang et al., 2024)

* **PDF:** https://arxiv.org/pdf/2402.07381.pdf
* **Summary:** **Co-teaching variant** where two models exchange selected instances under a dynamic consensus mechanism. Aims to reduce **mutual confirmation of noisy labels**—the problem where both models agree on wrong labels. Helpful under structured/instance-dependent noise rather than uniform flips.

### 102. EarlyStoppingNoise (Bai et al., 2021) ⚠️

* **PDF:** NeurIPS 2021 Proceedings — *no direct link, consider dropping*
* **Code:** https://github.com/tmllab/PES
* **Summary:** Analyzes **why early stopping helps** under label noise and proposes an improved early-stopping strategy (PES) that is more stable. A practical "curriculum-ish" lever when filtering-based cleaning is unreliable.

---

## Key Papers for ECG Implementation

**Must-cite (core baselines):**
1. `00` ConfidentLearning — Northcutt et al. 2021 — primary baseline to beat
2. `05` PLMLabelErrors — Chong et al. 2022 — loss-based NLP baseline
3. `52` NeuralRelationGraph — Kim et al. 2023 — graph-based baseline with NLP eval
4. `76` ERASER — DeYoung et al. 2020 — rationale evaluation benchmark
5. `24` TFA-ArtifactDetection — Pezeshkpour et al. 2022 — explanation-based artifact detection

**Must-cite (conceptual framing):**
6. `18` ExplanationDebuggingSurvey — Lertvittayakumjorn & Toni 2021 — defines the space
7. `60` HypothesisOnlyNLI — Poliak et al. 2018 — foundational artifact paper
8. `63` ProductOfExperts — Clark et al. 2019 — canonical debiasing approach

**Should-cite (methodology):**
9. `48` DeepKNNNoisyLabels — Bahri et al. 2020 — kNN graph methodology
10. `89` REV — Chen et al. 2023 — rationale evaluation metric
11. `40` LLMSelfExplanationFaithfulness — Madsen et al. 2024 — LLM explanation faithfulness
12. `32` GrammarConstrainedDecoding — Geng et al. 2023 — JSON generation

---

## GitHub Repositories Summary

```
# Primary baselines
cleanlab/cleanlab                          # 00 Confident Learning
asappresearch/aum                          # 06 AUM
dcx/lnlfm                                  # 05 PLM loss-based detection
snu-mllab/Neural-Relation-Graph            # 52 Graph-based noise detection

# Explanation debugging
pouyapez/artifact_detection                # 24 TFA for artifact detection
INK-USC/XMD                                # 21 Interactive debugging
PAIR-code/lit                              # 22 Language Interpretability Tool

# Evaluation
ERASER-benchmark                           # 76 Rationale evaluation
HanjieChen/REV                             # 89 Free-text rationale evaluation
ZhengpingJiang/RORA                        # 90 Robust rationale evaluation

# Debiasing
chrisc36/debias                            # 63 Product-of-Experts debiasing
UKPLab/emnlp2020-debiasing-unknown         # 65 Self-debiasing

# Influence
MadryLab/trak                              # 94 Scalable data attribution
ykwon0407/DataInf                          # 59 LoRA-tuned LLM influence
```
