# Literature Search: Explanation Faithfulness

**Prompt:** `13_lit_explanation_faithfulness.md`  
**Date:**  
**Source:** GPT-5.2 Pro

---

<!-- Paste results below -->


## 20 papers (2019–2025) on explanation faithfulness & rationale evaluation

1. **ERASER: A Benchmark to Evaluate Rationalized NLP Models** — Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, Byron C. Wallace — *ACL 2020*

* **arXiv/DOI:** arXiv:1911.03429; DOI: 10.18653/v1/2020.acl-main.408
* **Key contribution:** Introduces a unified benchmark (multi-dataset) for rationalized NLP and standardizes rationale evaluation, including removal-based **comprehensiveness/sufficiency**-style faithfulness checks and overlap metrics vs. human rationales.
* **Code:** ERASER benchmark resources include code/formatting + metric tooling released with the benchmark.

2. **ER-TEST: Evaluating Explanation Regularization Methods for Language Models** — Brihi Joshi, Aaron Chan, Ziyi Liu, Shaoliang Nie, Maziar Sanjabi, Hamed Firooz, Xiang Ren — *Findings of EMNLP 2022*

* **arXiv/DOI:** arXiv:2205.12542; DOI: 10.18653/v1/2022.findings-emnlp.242
* **Key contribution:** Proposes **ER-TEST**, an evaluation protocol for “explanation regularization” (aligning model rationales with human rationales), emphasizing **OOD generalization** via unseen datasets, contrast sets, and functional tests.
* **Code:** GitHub: `INK-USC/ER-Test`

3. **Goodhart’s Law Applies to NLP’s Explanation Benchmarks** — Cynthia Hsia *et al.* — *Findings of EACL 2024* ([arXiv][1])

* **arXiv/DOI:** arXiv:2402.18374
* **Key contribution:** Shows that optimizing for popular explanation benchmark scores can yield **metric gaming / misalignment** (Goodhart effects), and analyzes pitfalls of commonly used explanation evaluations. ([arXiv][1])
* **Code:** GitHub: `IREXorg/Goodharts-Law-Explanation-Benchmark`

4. **A Comparative Study of Faithfulness Metrics for Model Interpretability Methods** — Chun Sik Chan, Huanqi Kong, Guanqing Liang — *arXiv 2022*

* **arXiv/DOI:** arXiv:2204.05514
* **Key contribution:** Compares multiple faithfulness metrics (including removal/perturbation-style criteria) and highlights metric instability and disagreement when ranking interpretability methods.
* **Code:** Not provided in the paper metadata (arXiv).

5. **On the Sensitivity and Stability of Model Interpretations in NLP** — Fan Yin, Zhouxing Shi, Kai-Wei Chang — *ACL 2022*

* **arXiv/DOI:** arXiv:2104.08782; DOI: 10.18653/v1/2022.acl-long.188
* **Key contribution:** Re-frames faithfulness through **robustness-style notions** (sensitivity/stability), and systematically evaluates interpretations (including removal-based ones) under these criteria.
* **Code:** GitHub: `uclanlp/NLP-Interpretation-Faithfulness`

6. **Evaluating the Faithfulness of Importance Measures in NLP by Recursively Masking Allegedly Important Tokens and Retraining** — Andreas Madsen, Nicholas Meade, Vaibhav Adlakha, Siva Reddy — *Findings of EMNLP 2022*

* **arXiv/DOI:** arXiv:2110.08412; DOI: 10.18653/v1/2022.findings-emnlp.125
* **Key contribution:** Proposes **Recursive ROAR** (retrain-after-masking, repeatedly), plus a summary statistic (RACU) to better quantify faithfulness of token-importance methods while reducing masking OOD artifacts.
* **Code:** GitHub: `AndreasMadsen/nlp-roar-interpretability`

7. **A Benchmark for Interpretability Methods in Deep Neural Networks** — Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, Been Kim — *NeurIPS 2019*

* **arXiv/DOI:** arXiv:1806.10758; DOI: 10.5555/3454287.3455160
* **Key contribution:** Introduces **ROAR (RemOve And Retrain)** as a faithfulness evaluation for feature-importance methods: remove top-attributed features and **retrain** to avoid misleading OOD masking effects.
* **Code:** Not standardized as a single official repo on the venue/arXiv pages.

8. **Attention is not Explanation** — Sarthak Jain, Byron C. Wallace — *NAACL 2019*

* **arXiv/DOI:** arXiv:1902.10186
* **Key contribution:** Tests whether attention weights are faithful explanations; demonstrates attention can be **uncorrelated with other importance measures** and that alternative attention distributions can yield similar predictions.
* **Code:** GitHub: `successar/AttentionExplanation`

9. **Attention is not not Explanation** — Sarah Wiegreffe, Yuval Pinter — *EMNLP-IJCNLP 2019*

* **arXiv/DOI:** arXiv:1908.04626; DOI: 10.18653/v1/D19-1002
* **Key contribution:** Dissects why prior “attention is not explanation” tests can be insufficient; proposes alternative tests and argues explanation validity depends on definitions (faithfulness vs plausibility framing).
* **Code:** GitHub: `sarahwie/attention` (repo accompanying the EMNLP 2019 work).

10. **Rethinking Attention-Model Explainability through Faithfulness Violation Test** — Yibing Liu, Haoliang Li, Yangyang Guo, Chenqi Kong, Jing Li, Shiqi Wang — *ICML 2022*

* **arXiv/DOI:** arXiv:2201.12114
* **Key contribution:** Introduces a **faithfulness violation test** that checks whether explanation weights correctly capture the **polarity** of feature impact (support vs suppression), and empirically finds violations for common attention-based explanations.
* **Code:** GitHub: `BierOne/Attention-Faithfulness`

11. **A Diagnostic Study of Explainability Techniques for Text Classification** — Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, Isabelle Augenstein — *EMNLP 2020*

* **arXiv/DOI:** arXiv:2009.13295; DOI: 10.18653/v1/2020.emnlp-main.263
* **Key contribution:** Proposes a battery of diagnostic properties and compares multiple post-hoc text explainers (gradient/perturbation families), including agreement with human-annotated salient regions and other faithfulness-related behaviors.
* **Code:** Not linked on the ACL Anthology page; an external repo may exist but is not discoverable from the venue metadata shown here.

12. **Discretized Integrated Gradients for Explaining Language Models** — Swarnava Sanyal *et al.* — *EMNLP 2021*

* **arXiv/DOI:** DOI: 10.18653/v1/2021.emnlp-main.805
* **Key contribution:** Proposes DIG to make IG-style attributions more faithful for text by using interpolation strategies better suited to discrete token/embedding structure (reducing artifacts from unrealistic interpolations).
* **Code:** Not linked on the ACL Anthology page.

13. **Measuring Association Between Labels and Free-Text Rationales** — Sarah Wiegreffe *et al.* — *EMNLP 2021*

* **arXiv/DOI:** DOI: 10.18653/v1/2021.emnlp-main.760
* **Key contribution:** Studies when free-text rationales correlate with labels in ways that can enable **label leakage**; provides measurement approaches to quantify label–rationale association and related pitfalls for evaluating natural language explanations.
* **Code:** GitHub: `sarahwie/label-rationale-association`

14. **REV: Information-Theoretic Evaluation of Free-Text Rationales** — Hanjie Chen, Faeze Brahman, Xiang Ren, Yangfeng Ji, Yejin Choi, Swabha Swayamdipta — *ACL 2023*

* **arXiv/DOI:** arXiv:2210.04982; DOI: 10.18653/v1/2023.acl-long.112
* **Key contribution:** Introduces **REV**, a conditional V-information–based metric that estimates how much **new, label-relevant information** a rationale adds beyond the input/label; evaluated across multiple rationale benchmarks and rationale types (incl. CoT-like rationales).
* **Code:** GitHub: `HanjieChen/REV`

15. **RORA: Robust Free-Text Rationale Evaluation** — Zhengping Jiang, Yining Lu, Hanjie Chen, Daniel Khashabi, Benjamin Van Durme, Anqi Liu — *ACL 2024*

* **arXiv/DOI:** arXiv:2402.18678; DOI: 10.18653/v1/2024.acl-long.60
* **Key contribution:** Proposes a robustness-oriented free-text rationale evaluation designed to be resistant to **label leakage**, and reports improved alignment with human judgments vs. prior label-support metrics.
* **Code:** GitHub: `ZhengpingJiang/RORA`

16. **Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?** — Peter Hase, Mohit Bansal — *ACL 2020*

* **arXiv/DOI:** arXiv:2005.01831; DOI: 10.18653/v1/2020.acl-main.491
* **Key contribution:** Establishes controlled **human simulatability** protocols (incl. counterfactual-style prediction tests) to measure which explanation methods actually help users predict model outputs; compares LIME/Anchors/prototypes/others.
* **Code:** GitHub: `peterbhase/InterpretableNLP-ACL2020`

17. **ALMANACS: A Simulatability Benchmark for Language Model Explainability** — Edmund Mills, Shiye Su, Stuart Russell, Scott Emmons — *arXiv 2023*

* **arXiv/DOI:** arXiv:2312.12747
* **Key contribution:** Provides a scalable, automated **simulatability benchmark** using an LM “simulator” to predict another model’s behavior from explanations, across multiple safety-relevant topics; evaluates methods including attention and IG-style explanations.
* **Code:** GitHub: `edmundmills/ALMANACS`

18. **Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations** — Yanda Chen, Ruiqi Zhong, Narutatsu Ri, Chen Zhao, He He, Jacob Steinhardt, Zhou Yu, Kathleen McKeown — *ICML 2024*

* **arXiv/DOI:** DOI: 10.5555/3692070.3692380
* **Key contribution:** Introduces **counterfactual simulatability** for free-text explanations and implements metrics (precision/generality) measuring whether explanations let humans infer model outputs on counterfactual variants; finds low precision and weak correlation with plausibility.
* **Code:** Not listed on the PMLR page.

19. **Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models** — Chirag Agarwal, Sree Harsha Tanneru, Himabindu Lakkaraju — *arXiv 2024*

* **arXiv/DOI:** arXiv:2402.04614
* **Key contribution:** Position paper clarifying the faithfulness–plausibility gap for LLM self-explanations, arguing that plausibility improvements can come at the expense of faithfulness and outlining application-dependent requirements.
* **Code:** Not applicable / not provided (position paper).

20. **Explain Yourself, Briefly! Self-Explaining Neural Networks with Concise Sufficient Reasons** — Shahaf Bassan, Ron Eliav, Shlomit Gur — *ICLR 2025 (Poster)*

* **arXiv/DOI:** arXiv:2502.03391
* **Key contribution:** Proposes **sufficient subset training (SST)** to produce concise “sufficient reasons” as part of the model output, targeting faithful minimal-subset explanations while avoiding brittle post-hoc OOD sampling.
* **Code:** Not listed on the OpenReview page excerpt shown here.

---

## Coverage map to your requested themes

* **Comprehensiveness/Sufficiency & masking-based faithfulness:** (1), (2), (3), (4), (5), (6), (7), (20)
* **ERASER & related benchmarks / protocols:** (1), (2), (3) ([arXiv][1])
* **Faithfulness vs plausibility (explicit framing + attention debate):** (8), (9), (10), (19)
* **Post-hoc evaluation / sanity checks for attributions (attention/IG/saliency):** (4), (5), (6), (7), (10), (11), (12)
* **Free-text rationale evaluation (REV/RORA + leakage/association):** (13), (14), (15), (18), (19)
* **Simulatability (human + automated):** (16), (17), (18)

[1]: https://arxiv.org/abs/2308.14272?utm_source=chatgpt.com "Goodhart's Law Applies to NLP's Explanation Benchmarks"
