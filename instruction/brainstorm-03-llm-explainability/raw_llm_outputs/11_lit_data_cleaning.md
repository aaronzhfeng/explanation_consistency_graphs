# Literature Search: Data Cleaning / Label Noise Detection

**Prompt:** `08_lit_data_cleaning.md`  
**Date:**  
**Source:** GPT-5.2 Pro

---

<!-- Paste results below -->
## 1) Confident Learning extensions, applications, critiques (Cleanlab-adjacent)

1. **Bias-Aware Mislabeling Detection via Decoupled Confident Learning (DeCoLe)** — *Yunyi Li et al.* — 2025 — *arXiv*

* **ID:** `arXiv:2507.07216`
* **Contribution:** Proposes a mislabeling detector designed for *label-biased* settings (e.g., bias in hate-speech annotations), decoupling the detection mechanism from the proxy classifier so it can better surface bias-driven mislabels (a common “artifact-aligned” failure mode). ([Cleanlab][1])
* **Code:** (not linked in the arXiv record snippet I retrieved)

2. **Detecting Label Errors in Token Classification Data** — *Wei-Chen Wang, Jonas Mueller* — 2022 — *NeurIPS Workshop (via OpenReview/arXiv)*

* **ID:** `arXiv:2210.03920`
* **Contribution:** Studies multiple probability-based scoring rules for token labeling and identifies simple sentence-level scoring (e.g., based on the “worst token”) that reliably prioritizes sentences likely containing annotation errors in NER-style datasets.
* **Code:** `github.com/cleanlab/token-label-error-benchmarks`

3. **Identifying Incorrect Annotations in Multi-Label Classification Data** — *Aditya Thyagarajan, Elías Snorrason, Curtis Northcutt, Jonas Mueller* — 2022 (ICLR 2023 Workshop version listed) — *arXiv / ICLR Workshop*

* **ID:** `arXiv:2211.13895`
* **Contribution:** Extends confident-learning-style ideas to **multi-label** settings and proposes label-quality scores that rank likely annotation mistakes higher than correct labels (useful when “noise” is structured/non-uniform rather than random flips).
* **Code:** Cleanlab documents multi-label label-issue detection utilities; implementation is available via `cleanlab` tooling.

4. **Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks** — *Curtis G. Northcutt, Anish Athalye, Jonas Mueller* — 2021 — *NeurIPS Datasets & Benchmarks / arXiv*

* **ID:** `arXiv:2103.14749`
* **Contribution:** Uses confident-learning-based flagging plus human validation to show non-trivial test-set error rates across widely used datasets, and explicitly discusses **failure modes of confident learning** (relevant when models are confidently wrong due to artifacts/spurious markers).
* **Code:** `github.com/cleanlab/label-errors` (referenced in the arXiv abstract).

---

## 2) Loss-based filtering, co-teaching, robust training / reweighting

5. **Detecting Label Errors by Using Pre-Trained Language Models** — *Derek Chong et al.* — 2022 — *EMNLP 2022*

* **ID:** `arXiv:2205.12702`
* **Contribution:** Shows that simply ranking examples by **out-of-sample fine-tuning loss** with large PLMs can outperform more complex label-error detectors in NLP; introduces a procedure to inject **realistic human-originated noise** (harder than synthetic noise), aligning well with artifact-laden settings.
* **Code:** (not linked on the ACL Anthology/arXiv snippet I retrieved)

6. **DivideMix: Learning with Noisy Labels as Semi-supervised Learning** — *Junnan Li, Richard Socher, Steven C. H. Hoi* — 2020 — *ICLR 2020*

* **ID:** `arXiv:2002.07394`
* **Contribution:** Fits a mixture model over per-sample losses to split data into “clean/labeled” vs “noisy/unlabeled” sets, then applies semi-supervised learning to train robustly; canonical baseline when “small-loss” separation works.
* **Code:** `github.com/LiJunnan1992/DivideMix`

7. **Early-Learning Regularization Prevents Memorization of Noisy Labels** — *Sheng Liu et al.* — 2020 — *NeurIPS 2020*

* **ID:** `arXiv:2007.00151`
* **Contribution:** Exploits the “early learning” effect (networks learn easy/clean patterns first) and introduces a regularizer to reduce memorization of noisy labels later in training—often used when confident fitting of artifacts becomes a problem.
* **Code:** `github.com/shengliu66/ELR`

8. **CoDC: Co-Teaching with Dynamic Consensus** — *Yihong Zhang et al.* — 2024 — *arXiv*

* **ID:** `arXiv:2402.07381`
* **Contribution:** A co-teaching variant where two models exchange selected instances under a **dynamic consensus** mechanism, aiming to reduce mutual confirmation of noisy labels (helpful under structured/instance-dependent noise rather than uniform flips). ([arXiv][2])
* **Code:** (not linked in the arXiv record snippet I retrieved)

9. **Understanding and Improving Early Stopping for Learning with Noisy Labels** — *Yingbin Bai et al.* — 2021 — *NeurIPS 2021*

* **ID:** (NeurIPS proceedings; arXiv not confirmed from the sources I pulled)
* **Contribution:** Analyzes why early stopping helps under label noise and proposes an improved early-stopping strategy that is more stable in noisy settings (a practical “curriculum-ish” lever when filtering is unreliable).
* **Code:** `github.com/tmllab/PES` (official implementation).

10. **Confidence Scores Make Instance-dependent Label-noise Learning Possible** — *Antonin Berthon, Bo Han, Gang Niu, Tongliang Liu, Masashi Sugiyama* — 2021 — *ICML 2021 (PMLR)*

* **ID:** `arXiv:2001.03772`
* **Contribution:** Targets **instance-dependent / feature-dependent noise** by introducing confidence-scored instance-dependent noise (CSIDN) and an instance-level forward correction; directly relevant when “noise” is artifact-aligned (depends on input features).
* **Code:** (not linked in the ICML/arXiv snippet I retrieved)

---

## 3) Influence functions, TracIn-style debugging, scalable approximations (incl. LLMs)

11. **Estimating Training Data Influence by Tracing Gradient Descent (TracIn)** — *Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan* — 2020 — *NeurIPS 2020*

* **ID:** `arXiv:2002.08484`
* **Contribution:** Computes influence of training points on predictions by tracing gradient alignment across checkpoints; designed to be more practical than classical influence functions and explicitly motivated by finding harmful/rare/mislabeled examples.
* **Code:** `github.com/frederick0329/TracIn` (reference implementation).

12. **TRAK: Attributing Model Behavior at Scale** — *S. M. Park et al.* — 2023 — *ICML 2023 (PMLR) / arXiv*

* **ID:** `arXiv:2303.14186`
* **Contribution:** A scalable data-attribution method aimed at tracing predictions back to training data with far lower cost than many baselines, enabling broader use for dataset debugging and data valuation.
* **Code:** `github.com/MadryLab/trak`

13. **Token-wise Influential Training Data Retrieval for Large Language Models (RapidIn)** — *Huawei Lin et al.* — 2024 — *ACL 2024*

* **ID:** (ACL 2024 paper; arXiv not confirmed from the sources I pulled)
* **Contribution:** Proposes token-level influence estimation to retrieve training samples that most influence LLM outputs, geared toward scalable influence analysis in modern LMs.
* **Code:** `github.com/huawei-lin/RapidIn`

14. **Scalable Influence and Fact Tracing for Large Language Model Pretraining** — *Tyler A. Chang et al.* — 2024 (arXiv) / 2025 (ICLR)

* **ID:** `arXiv:2410.17413`
* **Contribution:** Scales gradient-based training-data attribution to **LLM pretraining** (reported up to ~8B params and corpora up to ~160B tokens) to retrieve influential examples for factual outputs; directly relevant when artifacts/spurious correlations are baked into pretraining data.
* **Code:** `github.com/PAIR-code/pretraining-tda`

---

## 4) LLM-based label checking / LLM-as-annotator (and noise-aware variants)

15. **LLMaAA: Making Large Language Models as Active Annotators** — *Zhang et al.* — 2023 — *arXiv*

* **ID:** `arXiv:2310.19561`
* **Contribution:** Positions LLMs as annotation agents within an active-annotation loop, aiming to reduce human labeling by selecting items to (re-)annotate and leveraging LLM outputs as supervision (a natural comparator to “LLM-based label checking”).
* **Code:** (not linked in the arXiv record snippet I retrieved)

16. **Noise-Robust Collaborative Active Learning with LLM-Based Noisy Annotators** — *Lifan Yuan et al.* — 2024 — *arXiv*

* **ID:** `arXiv:2402.06713`
* **Contribution:** Treats LLM annotations as *noisy* and proposes an active-learning approach that is explicitly noise-robust, which matches real deployments where LLM verification can be artifact-sensitive or inconsistent.
* **Code:** (not linked in the arXiv record snippet I retrieved)

17. **Testing the Reliability of ChatGPT for Text Annotation and Classification** — *Reiss* — 2023 — *arXiv*

* **ID:** `arXiv:2304.12306`
* **Contribution:** Empirically evaluates ChatGPT’s reliability as an annotator/classifier, providing evidence and caveats relevant if you use LLMs for label auditing or relabeling pipelines (e.g., when artifacts induce systematic annotation errors).
* **Code:** (not linked in the arXiv record snippet I retrieved)

---

## 5) Data cartography / training dynamics for identifying mislabeled or suspicious examples

18. **Characterizing Datapoints via Second-Split Forgetting** — *Pratyush Maini et al.* — 2022 — *NeurIPS 2022 / arXiv*

* **ID:** `arXiv:2210.15031`
* **Contribution:** Uses “second-split forgetting” signals from training dynamics to characterize datapoints (including those that are hard/ambiguous/noisy), offering a training-dynamics alternative to loss-only filtering.
* **Code:** (not linked in the arXiv record snippet I retrieved)

19. **Identifying Mislabeled Data using the Area Under the Margin Ranking (AUM)** — *Geoff Pleiss, Tianyi Zhang, Ethan Elenberg, Kilian Q. Weinberger* — 2020 — *NeurIPS 2020*

* **ID:** `arXiv:2001.10528`
* **Contribution:** Introduces **AUM**, a training-dynamics statistic based on margin trajectories; proposes a thresholding procedure (including an extra “purposefully mislabeled” class) to isolate mislabeled examples more robustly than raw loss in some regimes.
* **Code:** `github.com/asappresearch/aum`

20. **Dynamic Data Subset Selection for NLP Classification Tasks** — *J. Attendu et al.* — 2023 — *SustainNLP Workshop (ACL Anthology)*

* **ID:** (ACL Anthology workshop paper; arXiv not confirmed from the source I pulled)
* **Contribution:** Investigates subset selection/pruning for NLP classification using training-dynamics-style scores (e.g., EL2N-style signals) with dynamic pruning strategies, relevant to building practical “filter then retrain” cleaning pipelines.
* **Code:** (not linked in the ACL Anthology PDF snippet I retrieved)

---

## Papers above that most directly match “artifact-aligned noise / spurious markers”

* **Label bias / annotation artifacts:** DeCoLe. ([Cleanlab][1])
* **Human-originated structured noise in NLP:** Chong et al. (EMNLP 2022).
* **Feature-/instance-dependent noise (closest to “artifact-aligned”):** Berthon et al. (CSIDN).
* **Spurious factual associations from pretraining corpora (LLM setting):** Chang et al. (TrackStar / scalable influence & fact tracing).

If you want additional papers specifically on *NLP dataset artifact detection* (e.g., NLI annotation artifacts) that connect to “artifact-aligned noise” but are not strictly “label-noise detection,” I can assemble a separate 10–15 paper list focused on spurious-correlation diagnostics and debiasing, and cross-walk them to your ECG setting.

[1]: https://cleanlab.ai/research/?utm_source=chatgpt.com "Research"
[2]: https://www.arxiv.org/pdf/2508.09697?utm_source=chatgpt.com "Combating Noisy Labels via Dynamic Connection Masking"


