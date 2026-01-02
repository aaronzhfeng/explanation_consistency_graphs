# Literature Search: Explanation-Based Debugging

**Prompt:** `09_lit_explanation_debugging.md`  
**Date:**  
**Source:** GPT-5.2 Pro

---

<!-- Paste results below -->


## 1) Explanation-Based Human Debugging

1. **Explanation-Based Human Debugging of NLP Models: A Survey** *(seed; you already have this)*

* **Authors:** Piyawat Lertvittayakumjorn, Francesca Toni
* **Year/Venue:** 2021, *TACL*
* **DOI:** 10.1162/tacl_a_00440
* **Key contribution:** Systematizes “explanations → human feedback → model/data fixes” pipelines, including how explanations can surface dataset artifacts and labeling issues during debugging workflows.
* **Code:** N/A (survey).

2. **Interactive Label Cleaning with Example-based Explanations** *(seed; you already have this)*

* **Authors:** Stefano Teso, Andrea Bontempelli, Fausto Giunchiglia, Roberto Passerini
* **Year/Venue:** 2021, *NeurIPS*
* **arXiv:** 2106.03922
* **Key contribution:** Uses *example-based explanations* (e.g., influential examples) to guide humans toward likely label errors and interactively clean datasets—explicitly targeting training-set quality rather than only interpretability.
* **Code:** GitHub (abonte/cincer).

3. **FIND: Human-in-the-Loop Debugging Deep Text Classifiers**

* **Authors:** Piyawat Lertvittayakumjorn *et al.*
* **Year/Venue:** 2020, *EMNLP*
* **arXiv:** 2010.04987
* **Key contribution:** Human-in-the-loop debugging workflow where explanations support discovering spurious patterns and dataset issues that drive incorrect behavior, enabling targeted fixes beyond loss/confidence signals.
* **Code:** GitHub (plkumjorn/FIND). ([arXiv][1])

4. **HILDIF: Interactive Debugging of NLI Models Using Influence Functions**

* **Authors:** Hugo Zylberajch, Piyawat Lertvittayakumjorn, Francesca Toni
* **Year/Venue:** 2021, *InterNLP @ ACL-IJCNLP (Workshop)*
* **DOI:** 10.18653/v1/2021.internlp-1.1
* **Key contribution:** Uses *influence functions* as explanations to show training instances driving NLI predictions; human feedback then mitigates artifact-driven behavior—explicitly linking interpretability to data artifacts.
* **Code:** Not linked on ACL entry.

5. **XMD: An End-to-End Framework for Interactive Explanation-Based Debugging of NLP Models**

* **Authors:** Dong-Ho Lee, Akshen Kadakia, Brihi Joshi, Aaron Chan, Ziyi Liu, Kiran Narahari, Takashi Shibuya, Ryosuke Mitani, Toshiyuki Sekiya, Jay Pujara, Xiang Ren
* **Year/Venue:** 2023, *ACL Demo Track* (arXiv 2022)
* **arXiv:** 2210.16978
* **DOI (ACL Anthology):** 10.18653/v1/2023.acl-demo.25
* **Key contribution:** End-to-end system to surface explanations, collect user feedback, and update models via explanation-alignment/regularization—aimed at correcting spurious biases that often originate in training data.
* **Code:** GitHub (INK-USC/XMD).

6. **The Language Interpretability Tool: Extensible, Interactive Visualizations and Analysis for NLP Models**

* **Authors:** Ian Tenney, James Wexler, Jasmijn Bastings, Tolga Bolukbasi, Andy Coenen, Sebastian Gehrmann, Ellen Jiang, Mahima Pushkarna, Carey Radebaugh, Emily Reif, Ann Yuan
* **Year/Venue:** 2020, *EMNLP System Demonstrations*
* **arXiv:** 2008.05122
* **DOI (ACL Anthology):** 10.18653/v1/2020.emnlp-demos.15
* **Key contribution:** Interactive interpretability + slicing + counterfactual probing to diagnose failure modes and “undesired priors” in datasets/models; commonly used as a practical layer for discovering systematic data issues.
* **Code:** GitHub (PAIR-code/lit).

---

## 2) Using Interpretability to Find Spurious Correlations and Artifacts

7. **Combining Feature and Instance Attribution to Detect Artifacts**

* **Authors:** Pouya Pezeshkpour, Sarthak Jain, Sameer Singh, Byron C. Wallace
* **Year/Venue:** 2022, *Findings of ACL* (arXiv 2021)
* **arXiv:** 2107.00323
* **DOI (ACL Anthology):** 10.18653/v1/2022.findings-acl.153
* **Key contribution:** Proposes *training-feature attribution* (TFA): explanations that jointly localize **which tokens in which influential training examples** drive predictions—explicitly to uncover training-set artifacts/spurious markers. Includes a user study for artifact discovery.
* **Code:** GitHub (pouyapez/artifact_detection).

8. **Explaining Black Box Predictions and Unveiling Data Artifacts through Influence Functions**

* **Authors:** Xiaochuang Han, Byron C. Wallace, Yulia Tsvetkov
* **Year/Venue:** 2020, *arXiv*
* **arXiv:** 2005.06676
* **Key contribution:** Positions *instance-level* explanations (influence functions) as better-suited than token saliency for some NLP settings, and introduces an influence-based quantitative measure to reveal **training data artifacts**.
* **Code:** Not specified on arXiv entry.

9. **Identifying and Mitigating Spurious Correlations for Improving Robustness in NLP Models**

* **Authors:** Tianlu Wang, Rohit Sridhar, Diyi Yang, Xuezhi Wang
* **Year/Venue:** 2022, *Findings of NAACL*
* **DOI (ACL Anthology):** 10.18653/v1/2022.findings-naacl.129
* **Key contribution:** Uses interpretability-driven analysis to identify spurious correlates and proposes mitigation strategies; explicitly targets robustness failures caused by shortcut features that can be invisible to loss-based diagnostics.
* **Code:** GitHub repo linked by authors (tianlu-wang/Identifying-and-Mitigating-Spurious-Correlations).

10. **Competency Problems: On Finding and Removing Artifacts in Language Data**

* **Authors:** Matt Gardner *et al.*
* **Year/Venue:** 2021, *EMNLP*
* **DOI (ACL Anthology):** 10.18653/v1/2021.emnlp-main.135
* **Key contribution:** Data-centric methodology to discover dataset artifacts/heuristics and remove them to better measure true model competency—useful for constructing “challenge” slices where spurious features break.
* **Code:** Not specified on ACL entry. ([ResearchGate][2])

11. **MASKER: Masked Keyword Regularization for Reliable Text Classification**

* **Authors:** Sanghun Moon *et al.*
* **Year/Venue:** 2021, *AAAI*
* **DOI:** (AAAI paper page)
* **Key contribution:** Regularizes models using keyword masking to reduce over-reliance on brittle lexical cues—conceptually close to explanation-guided debiasing when keywords approximate “what the model uses.”
* **Code:** Not specified on AAAI entry.

---

## 3) Explanation-Guided Training, Reweighting, and Explanation Regularization

12. **Incorporating Priors with Feature Attribution on Text Classification**

* **Authors:** Frederick Liu, Besim Avci
* **Year/Venue:** 2019, *ACL*
* **arXiv:** 1906.08286
* **DOI (ACL Anthology):** 10.18653/v1/P19-1631
* **Key contribution:** Adds an attribution-matching loss so practitioners can enforce priors like “don’t rely on identity tokens” or “focus on toxic terms,” directly using explanations as a training signal to counter spurious features.
* **Code:** Not specified in paper/ACL entry.

13. **Explanation-Based Finetuning Makes Models More Robust to Spurious Cues**

* **Authors:** Kimi Ludan *et al.*
* **Year/Venue:** 2023, *ACL*
* **DOI (ACL Anthology):** 10.18653/v1/2023.acl-long.??? *(see ACL entry for exact anthology ID/DOI)*
* **Key contribution:** Uses explanation signals during finetuning to reduce reliance on spurious cues, improving robustness where standard finetuning may still encode shortcuts.
* **Code:** GitHub repo linked by authors (see paper/code link).

14. **ER-Test: Evaluating Explanation Regularization Methods for NLP Models**

* **Authors:** Brihi Joshi *et al.*
* **Year/Venue:** 2022, *TrustNLP @ NAACL* (also circulated as arXiv)
* **arXiv:** 2210.09635
* **Key contribution:** Provides an evaluation protocol for explanation-regularized training (often used to fight spurious correlations), testing whether explanation constraints actually improve robustness/behavior rather than just changing saliency maps.
* **Code:** GitHub (brihi-joshi/ER-Test).

15. **REFER: Rationale Extraction for Explanation Regularization**

* **Authors:** Ali Madani, Pasquale Minervini
* **Year/Venue:** 2023, *CoNLL*
* **DOI (ACL Anthology):** 10.18653/v1/2023.conll-1.40
* **Key contribution:** Extracts rationales to support explanation regularization, tightening the loop “rationales → constraints → improved generalization” for settings where spurious evidence must be discouraged.
* **Code:** GitHub repo linked by authors.

16. **A Rationale-Centric Framework for Human-in-the-Loop Double-Robustness Learning**

* **Authors:** Ximing Lu *et al.*
* **Year/Venue:** 2022, *ACL*
* **DOI (ACL Anthology):** 10.18653/v1/2022.acl-long.481
* **Key contribution:** Uses rationales as a first-class object in a human-in-the-loop robustness framework—aligning model behavior with rationale-level signals to improve robustness under distribution shift and reduce shortcut reliance.
* **Code:** Not specified on ACL entry. ([ACL Anthology][3])

---

## 4) Explanation-Guided Data Augmentation

17. **Data Augmentations for Improved (Large) Language Model Generalization**

* **Authors:** Amir Feder *et al.*
* **Year/Venue:** 2023, *NeurIPS (OpenReview)*
* **Identifier:** OpenReview (NeurIPS 2023)
* **Key contribution:** Studies/constructs augmentations for better generalization, including counterfactual-style transformations motivated by causal structure—useful when seeking augmentation that breaks spurious correlations rather than reinforcing them.
* **Code:** Not specified on OpenReview entry.

18. **A Novel Counterfactual Data Augmentation Method for Aspect-Based Sentiment Analysis**

* **Authors:** Dongming Wu, Lulu Wen, Chao Chen, Zhaoshu Shi
* **Year/Venue:** 2023 (arXiv; camera-ready for ACML), *PMLR vol. 222 (ACML proceedings release lists 2024)*
* **arXiv:** 2306.11260
* **Key contribution:** Uses **Integrated Gradients** to locate opinion expressions (explanation signal), masks them, then prompts a PLM (T5) to generate counterfactuals with flipped polarity—explicitly turning explanations into targeted data augmentation.
* **Code:** Not specified on arXiv/PMLR PDFs.

---

## 5) Rationale-Based Active Learning

19. **Rationale-based Active Learning with Supervised Attention (RALSA)** *(paper title appears as “Rationale-based Human-in-the-Loop via Supervised Attention”)*

* **Authors:** Tushar Kanchinadam *et al.*
* **Year/Venue:** 2020, *DaSH @ KDD (Workshop)*
* **Identifier:** Workshop PDF (no DOI/arXiv listed on the PDF header excerpt)
* **Key contribution:** Uses human-provided rationales to supervise attention and drive active learning—selecting examples where rationale/attention alignment is weak, which can correlate with ambiguous or mislabeled data.
* **Code:** Not specified on PDF.

---

## 6) Debugging with Natural Language Feedback

20. **INTERFAIR: Debiasing with Natural Language Feedback at Test Time**

* **Authors:** Bodhisattwa Prasad Majumder *et al.*
* **Year/Venue:** 2023, *EMNLP*
* **DOI (ACL Anthology):** 10.18653/v1/2023.emnlp-main.589
* **Key contribution:** Uses **natural-language feedback** to reduce biased/spurious behaviors at inference time—relevant as a complementary debugging signal when you want human/LLM feedback expressed in language, not only labels.
* **Code:** Not specified on ACL entry.

[1]: https://arxiv.org/abs/2010.04987?utm_source=chatgpt.com "FIND: Human-in-the-Loop Debugging Deep Text Classifiers"
[2]: https://www.researchgate.net/publication/357542747_Explanation-Based_Human_Debugging_of_NLP_Models_A_Survey?utm_source=chatgpt.com "(PDF) Explanation-Based Human Debugging of NLP Models"
[3]: https://aclanthology.org/2022.findings-acl.153.pdf?utm_source=chatgpt.com "Combining Feature and Instance Attribution to Detect Artifacts"
