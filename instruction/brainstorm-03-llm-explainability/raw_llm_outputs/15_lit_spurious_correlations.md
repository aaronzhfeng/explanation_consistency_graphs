# Literature Search: Spurious Correlations / Artifacts

**Prompt:** `12_lit_spurious_correlations.md`  
**Date:**  
**Source:** GPT-5.2 Pro

---

<!-- Paste results below -->


Papers below are **additional** to your seed list (Gururangan+ 2018; McCoy+ 2019; NAACL 2022 “Identifying Spurious Correlations”).

## 1) Annotation artifacts in NLI (hypothesis-only, lexical cues) — SNLI/MNLI focus

### 1. Hypothesis Only Baselines in Natural Language Inference

**Adam Poliak, Jason Naradowsky, Aparajita Haldar, Rachel Rudinger, Benjamin Van Durme**. 2018. **SEM (Lexical and Computational Semantics).*

* **arXiv:** 1805.01042
* **Key contribution:** Establishes “hypothesis-only” models as a diagnostic for annotation artifacts; shows strong performance on multiple NLI datasets, including human-elicited ones (e.g., SNLI/MNLI), indicating label-predictive lexical irregularities.
* **Code:** Available.

### 2. Don’t Take the Premise for Granted: Mitigating Artifacts in Natural Language Inference

**Yonatan Belinkov, Adam Poliak, Stuart M. Shieber, Benjamin Van Durme, Alexander M. Rush**. 2019. *ACL.*

* **arXiv:** 1907.04380
* **Key contribution:** Proposes training objectives that discourage hypothesis-only shortcuts by effectively forcing premise use (via probabilistic reformulation/approximations); improves transfer when training and test differ in artifact structure.
* **Code:** Available.

### 3. Misleading Failures of Partial-input Baselines

**Shi Feng, Eric Wallace, Jordan Boyd-Graber**. 2019. *ACL.*

* **arXiv:** 1905.05778
* **DOI:** 10.18653/v1/P19-1554
* **Key contribution:** Shows that *low* performance of partial-input baselines does **not** imply “artifact-free” data; constructs cases where artifacts exist but evade partial-input detection, and demonstrates such patterns in SNLI.
* **Code:** Not found in the paper landing pages reviewed.

## 2) Mitigation methods for dataset bias / shortcuts (ensembles, reweighting, self-debiasing) — SNLI/MNLI/FEVER relevance

### 4. Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases

**Christopher Clark, Mark Yatskar, Luke Zettlemoyer**. 2019. *EMNLP.*

* **arXiv:** 1909.03683
* **Key contribution:** Trains a “bias-only” model and combines it with a main model in an ensemble-style objective to reduce reliance on known shortcuts; widely used as a Product-of-Experts-style debiasing template in NLI settings.
* **Code:** Available.

### 5. End-to-End Bias Mitigation by Modelling Biases in Corpora

**Rabeeh Karimi Mahabadi, Yonatan Belinkov, James Henderson**. 2020. *ACL.*

* **arXiv:** 1909.06321
* **Key contribution:** Uses predictions from one or more bias-only models to **reweight/down-weight biased examples** (and/or focus on hard examples), improving robustness on out-of-domain / bias-sensitive evaluations across NLI and fact verification.
* **Code:** Available.

### 6. Towards Debiasing NLU Models from Unknown Biases

**Prasetya Ajie Utama, Nafise Sadat Moosavi, Iryna Gurevych**. 2020. *EMNLP.*

* **arXiv:** 2009.12303
* **Key contribution:** “Self-debiasing” framework that does **not require knowing the bias type** in advance; aims to prevent the model from becoming overconfident on biased examples and to preserve improvements on challenge sets.
* **Code:** Available.

### 7. Learning to Model and Ignore Dataset Bias with Mixed Capacity Ensembles

**Christopher Clark, Mark Yatskar, Luke Zettlemoyer**. 2020. *Findings of EMNLP.*

* **arXiv:** 2011.03856
* **Key contribution:** Uses a low-capacity model to soak up shallow dataset-specific correlations while a higher-capacity model learns complementary signals; enforces non-overlap via conditional independence to reduce shortcut reliance without specifying the bias upfront.
* **Code:** Not confirmed from the sources retrieved (paper mentions code on author pages, but no stable repo link was verified here).

## 3) Spurious correlations in sentiment / text classification (and how to surface them)

### 8. An Empirical Study on Robustness to Spurious Correlations using Pre-trained Language Models

**Lifu Tu, Garima Lalwani, Spandana Gella, He He**. 2020. *TACL.*

* **arXiv:** 2007.06778
* **DOI:** 10.1162/tacl_a_00335
* **Key contribution:** Demonstrates that robustness gains from pretraining often come from **generalizing from a small number of counterexamples** where spurious cues break; proposes multi-task learning to help when minority counterexamples are extremely scarce.
* **Code:** Available.

### 9. Identifying Spurious Correlations for Robust Text Classification

**Zhao Wang, Aron Culotta**. 2020. *Findings of EMNLP.*

* **arXiv:** 2010.02458
* **Key contribution:** Proposes a method to classify features/terms as “spurious” vs “genuine” using treatment-effect-derived signals; uses this to guide feature selection and improve worst-case (group) robustness in text classification (including sentiment-style settings).
* **Code:** Not found from the sources retrieved.

### 10. Robustness to Spurious Correlations in Text Classification via Automatically Generated Counterfactuals

**Zhao Wang, Aron Culotta**. 2021. *AAAI.*

* **arXiv:** 2012.10040
* **Key contribution:** Identifies likely causal features, generates counterfactuals by substituting them (e.g., antonyms) and flipping labels, and retrains to reduce spurious reliance; reports robustness gains on counterfactual/human-edited shifts.
* **Code:** Not found from the sources retrieved.

## 4) Counterfactual evaluation / CAD (directly relevant to explanation-based inconsistency)

### 11. Learning the Difference that Makes a Difference with Counterfactually-Augmented Data

**Divyansh Kaushik, Eduard Hovy, Zachary C. Lipton**. 2020. *ICLR.*

* **arXiv:** 1909.12434
* **Key contribution:** Introduces **human-in-the-loop counterfactual rewriting** (minimal edits that flip the label) to break shortcut correlations; widely used for stress-testing and improving robustness in sentiment-style classification.
* **Code:** Not found from the sources retrieved.

### 12. Explaining The Efficacy of Counterfactually Augmented Data

**Divyansh Kaushik, Amrith Setlur, Eduard Hovy, Zachary C. Lipton**. 2020. *arXiv (preprint).*

* **arXiv:** 2010.02114
* **Key contribution:** Provides an explanatory/toy-model account of why CAD can improve out-of-domain generalization; proposes a concrete **noise injection** idea to probe whether attribution methods are highlighting causal vs spurious spans (highly aligned with “explanations to detect artifacts”).
* **Code:** Not found from the sources retrieved.

## 5) Fact verification artifacts & debiasing — FEVER focus

### 13. Towards Debiasing Fact Verification Models

**Tal Schuster, Darsh J. Shah, Yun Jie Serene Yeo, Daniel Filizzola, Enrico Santus, Regina Barzilay**. 2019. *EMNLP-IJCNLP.*

* **arXiv:** 1908.05267
* **Key contribution:** Shows FEVER can be partially “solved” with **claim-only** cues; builds a **symmetric evaluation set** to suppress these artifacts and proposes regularization to improve performance under this more robust evaluation.
* **Code / data:** Available (symmetric FEVER evaluation set).

### 14. CrossAug: A Contrastive Data Augmentation Method for Debiasing Fact Verification Models

**Minwoo Lee, Seungpil Won, Juae Kim, Hwanhee Lee, Cheon-Eum Park, Kyomin Jung**. 2021. *CIKM.*

* **arXiv:** 2109.15107
* **DOI:** 10.1145/3459637.3482078
* **Key contribution:** Uses a two-stage contrastive augmentation pipeline to generate debiasing examples for fact verification; reports gains particularly on **Symmetric FEVER**-style evaluations designed to expose claim-only shortcuts.
* **Code:** Available.

### 15. FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information

**Rami Aly, Zhijiang Guo, Michael Schlichtkrull, James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Oana Cocarascu, Arpit Mittal**. 2021. *NeurIPS Datasets & Benchmarks (and arXiv).*

* **arXiv:** 2106.05707
* **Key contribution:** Introduces a FEVER-style benchmark extending verification to **tables + text** and explicitly discusses measuring/minimizing biases such as predicting labels without evidence.
* **Code:** Dataset release implied; repo not verified from sources retrieved.

## 6) Challenge sets / adversarial data collection (NLI)

### 16. Adversarial NLI: A New Benchmark for Natural Language Understanding

**Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, Douwe Kiela**. 2020. *ACL.*

* **arXiv:** 1910.14599
* **Key contribution:** Creates an NLI benchmark via an adversarial human-and-model loop, explicitly targeting and reducing dataset artifacts that models exploit; commonly used to test shortcut resistance beyond SNLI/MNLI-style artifacts.
* **Code:** Dataset released via the authors’ benchmark infrastructure; repo not verified from sources retrieved.

## 7) Group robustness / worst-group accuracy (connects directly to spurious markers)

### 17. Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization

**Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang**. 2020. *ICLR.*

* **arXiv:** 1911.08731
* **Key contribution:** Canonical **group DRO** formulation minimizing worst-group loss to combat spurious correlations that fail on minority groups; also emphasizes the role of regularization for worst-case generalization.
* **Code:** Not a single official repo confirmed here; OpenReview lists community implementations.

### 18. Just Train Twice: Improving Group Robustness without Training Group Information

**Evan Zheran Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, Chelsea Finn**. 2021. *ICML.*

* **arXiv:** 2107.09044
* **Key contribution:** Two-stage method that upweights examples misclassified by an ERM model to approximate worst-group improvements **without** full training-group labels; reports results across vision *and* NLP tasks with spurious correlations (including MultiNLI in reported experiments).
* **Code:** Not verified from sources retrieved.

---

## Code links (only where verified)

```text
Hypothesis-only NLI (Poliak et al. 2018): https://github.com/azpoliak/hypothesis-only-NLI
Robust-NLI (Belinkov et al. 2019): https://github.com/azpoliak/robust-nli
Don't Take the Easy Way Out (Clark et al. 2019): https://github.com/chrisc36/debias
End-to-End Bias Mitigation (Mahabadi et al. 2020): https://github.com/rabeehk/robust-nli
Debiasing Unknown Biases (Utama et al. 2020): https://github.com/UKPLab/emnlp2020-debiasing-unknown
TACL spurious correlations study (Tu et al. 2020): https://github.com/lifu-tu/Study-NLP-Robustness
FEVER Symmetric eval set (Schuster et al. 2019): https://github.com/TalSchuster/FeverSymmetric
CrossAug (Lee et al. 2021): https://github.com/minwhoo/CrossAug
```
