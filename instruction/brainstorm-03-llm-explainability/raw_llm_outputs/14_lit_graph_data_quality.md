# Literature Search: Graph-Based Data Quality

**Prompt:** `11_lit_graph_data_quality.md`  
**Date:**  
**Source:** GPT-5.2 Pro

---

<!-- Paste results below -->


Below are **18 papers (2019–2025)** that align with **embedding-neighborhood / kNN-graph signals**, **graph-based label smoothing/cleaning**, and **representation-based denoising**. Most are **NLP-ready** (or **modality-agnostic** and directly transferable to your setting by swapping *raw-input embeddings* for **explanation-embedding kNN graphs**).

---

## 1) kNN-based label noise detection and neighborhood consistency

### Deep k-NN for Noisy Labels — Bahri, Jiang, Gupta (2020, ICML)

* **arXiv/DOI:** arXiv:2004.12289
* **Code:** (no official repo found; community implementations exist)
* **Key contribution:** Proposes using **deep feature kNN structure** to identify/mitigate noisy labels, leveraging neighborhood agreement in representation space to improve robustness to mislabeled points. ([ADS][1])

### An Embedding is Worth a Thousand Noisy Labels (WANN) — Di Salvo, Doerrich, Rieger, Ledig (2025, TMLR; arXiv 2024)

* **arXiv/DOI:** arXiv:2408.14358
* **Code:** `https://github.com/francescodisalvo05/wann-noisy-labels`
* **Key contribution:** Uses **foundation-model embeddings + weighted adaptive kNN voting** with a learned **label reliability score** to downweight likely-mislabeled neighbors; designed to be efficient and explainable.

### Explaining and Improving Model Behavior with k Nearest Neighbor Representations — Rajani et al. (2020, arXiv)

* **arXiv/DOI:** arXiv:2010.09030
* **Code:** (not found)
* **Key contribution:** Uses **kNN over learned representations** to explain predictions and diagnose behavior (including failure modes consistent with mislabeled/atypical examples); directly relevant to “neighbor-based inconsistency” debugging in NLP.

### Label Distribution Learning-Enhanced Dual-KNN for Text Classification — Yuan, Chen, Tan, Wang, Liu, Zhang (2025, arXiv)

* **arXiv/DOI:** arXiv:2503.04869
* **Code:** (not found)
* **Key contribution:** Proposes a **dual-nearest-neighbor** text classifier (retrieval over text embeddings + label-probability representations) and explicitly discusses how NN retrieval can be **confused by noisy datasets**, adding a label-distribution/contrastive component to stabilize neighbor quality.

### Confident Learning: Estimating Uncertainty in Dataset Labels — Northcutt, Jiang, Chuang (2021, JAIR)

* **arXiv/DOI:** arXiv:1911.00068
* **Code:** `https://github.com/cleanlab/cleanlab`
* **Key contribution:** A widely used **label error detection** framework (Cleanlab) that estimates a **confident joint** between noisy/true labels to rank likely label issues; frequently used as a baseline in data-quality work. ([Jair][2])

---

## 2) Graph neural networks and graph-based methods for data cleaning

### Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data — Kim, Yun, Song (2023, NeurIPS)

* **arXiv/DOI:** arXiv:2301.12321
* **Code:** `https://github.com/snu-mllab/Neural-Relation-Graph`
* **Key contribution:** Builds an explicit **relational graph in feature space** and provides scalable algorithms for **label error detection + outlier/OOD detection**, evaluated across domains including **language (SST-2)**—highly aligned with your “graph over explanation embeddings” design.

### Graph Convolutional Networks for Learning with Few Clean and Many Noisy Labels — Iscen et al. (2020, ECCV)

* **arXiv/DOI:** arXiv:2011.00359
* **Code:** (not found)
* **Key contribution:** Constructs a **kNN graph in embedding space** and trains a **GCN** to propagate/denoise labels when only a small clean subset exists—directly portable to a **kNN graph over explanation embeddings**. ([ScienceDirect][3])

### Label Propagation-Based Semi-Supervised Learning for Hate Speech Classification — D’Sa, Illina, Fohr, Klakow, Ruiter (2020, Insights @ EMNLP)

* **arXiv/DOI:** DOI: 10.18653/v1/2020.insights-1.8
* **Code:** (not found)
* **Key contribution:** Applies **label propagation on a similarity graph** for **low-resource hate speech** text classification; explicitly ties performance to representation quality—useful as an NLP precedent for graph smoothing on embedding graphs.

---

## 3) Embedding similarity / clustering-style denoising and noise modeling

### Towards Harnessing Feature Embedding for Robust Learning with Noisy Labels (LEND) — Zhang, Shen, Yang, Gong (2022, *Machine Learning* journal)

* **arXiv/DOI:** DOI: 10.1007/s10994-022-06197-6
* **Code:** publisher page indicates code “will be released” (no linked repo found)
* **Key contribution:** Computes an **embedding similarity matrix** to capture local structure and **dilute noisy supervision** by overwhelming mislabeled signals with nearby consistent neighbors—conceptually very close to neighborhood-consistency scoring.

### Beyond Images: Label Noise Transition Matrix Estimation for Tasks with Lower-Quality Features — Zhu et al. (2022, ICML)

* **arXiv/DOI:** arXiv:2208.09329
* **Code:** (not found)
* **Key contribution:** Targets settings where features are less “clean” than vision (explicitly “beyond images”) and proposes improved **noise transition estimation**, relevant if you want to model label noise statistically alongside neighborhood-graph inconsistency.

### DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models — Kwon, Wu, Wu, Zou (2024, ICLR)

* **arXiv/DOI:** arXiv:2310.00902
* **Code:** `https://github.com/ykwon0407/DataInf`
* **Key contribution:** Efficient influence-style data attribution for large models (incl. **RoBERTa / Llama-2 LoRA**), with demonstrations that influence scores can help **identify mislabeled points**—complements graph-based inconsistency by adding a “who affected this prediction?” view.

---

## 4) Contrastive learning signals for noisy labels / data quality

### Selective-Supervised Contrastive Learning with Noisy Labels (Sel-CL) — Li, Xia, Ge, Liu (2022, CVPR)

* **arXiv/DOI:** arXiv:2203.04181
* **Code:** `https://github.com/ShikunLi/Sel-CL`
* **Key contribution:** Shows that supervised contrastive learning degrades due to **noisy positive/negative pairs** and proposes selecting **confident examples/pairs** based on representation/label agreement—closely related to neighborhood consistency, but in pair space.

### Robust Contrastive Learning against Noisy Views — Chuang, Hjelm, Wang, Vineet, Joshi, Torralba, Jegelka, Song (2022, CVPR)

* **arXiv/DOI:** arXiv:2201.04309
* **Code:** `https://github.com/chingyaoc/RINCE`
* **Key contribution:** Addresses **noise in contrastive positives** (“noisy views”) by introducing robust contrastive objectives (Robust InfoNCE variants), useful if you use contrastive embedding learning before building your explanation-embedding kNN graph.

---

## 5) NLP-centric label noise detection and correction (often compared to Confident Learning / Cleanlab)

### Detecting Label Errors by using Pre-Trained Language Models — Chong, Hong, Manning (2022, EMNLP)

* **arXiv/DOI:** arXiv:2205.12702; DOI: 10.18653/v1/2022.emnlp-main.618
* **Code:** `https://github.com/dcx/lnlfm`
* **Key contribution:** Demonstrates that simply ranking examples by **fine-tuned task loss** is a strong label-error detector for NLP; introduces more realistic “human-originated” label noise and reports strong gains over prior error-detection mechanisms (commonly discussed alongside Confident Learning-style baselines).

### CTRL: Clustering Training Losses for Label Error Detection — Yue, Jha (2024, IEEE Transactions on AI)

* **arXiv/DOI:** arXiv:2208.08464; DOI: 10.1109/TAI.2024.3365093
* **Code:** `https://github.com/chang-yue/ctrl`
* **Key contribution:** Detects label errors by clustering **per-example training loss curves** (clean vs noisy learn differently), then retrains after removal—often competitive with other label-error detectors; useful as a non-graph baseline to compare against neighborhood inconsistency.

### Noisy-Labeled NER with Confidence Estimation — Liu, Fu, Tan, Chen, Zhang, Huang, Gao (2021, NAACL)

* **arXiv/DOI:** arXiv:2104.04318; (NAACL 2021 anthology entry available)
* **Code:** `https://github.com/liukun95/Noisy-NER-Confidence-Estimation`
* **Key contribution:** Estimates calibrated **confidence scores** for NER labels under weak/distant annotation and integrates confidence into training/self-training; relevant if your mislabeled detection needs structured-output analogs of neighborhood inconsistency.

### Noisy Multi-Label Text Classification via Instance-Label Pair Correction — Xu, Song, Liu, Liu, Sun, Jing, Yu (2024, Findings of NAACL)

* **arXiv/DOI:** DOI: 10.18653/v1/2024.findings-naacl.93
* **Code:** (not found)
* **Key contribution:** Treats noise at the **(instance, label) pair** level (false positives/negatives) rather than only instance-level, proposing correction mechanisms tailored to multi-label text settings—useful if your dataset has multi-label explanations/annotations.

### Is BERT Robust to Label Noise? A Study on Learning with Noisy Labels in Text Classification — Zhu et al. (2022)

* **arXiv/DOI:** arXiv:2204.09371
* **Code:** `https://github.com/huanzhang12/NoisyLabelTextClassification`
* **Key contribution:** Systematic study of **BERT under label noise** for text classification, providing practical baselines and failure analyses; useful to justify why you need explanation-graph inconsistency signals rather than relying on PLM robustness alone.

---

## Notes on how these map to your “kNN over explanation embeddings” design

* Papers that explicitly rely on **embedding neighborhoods / similarity graphs** (e.g., **Deep kNN**, **WANN**, **LEND**, **Neural Relation Graph**, **GCN for few clean + many noisy**, **label propagation for hate speech**) are the most direct drop-ins: replace their feature embeddings with **explanation embeddings** and keep the graph logic. ([ScienceDirect][3])
* For strong baselines in experiments, the most standard label-issue detectors here are **Cleanlab/Confident Learning**, **loss-ranking with PLMs**, and **CTRL**. ([Jair][2])

If you want, I can also produce a **1–2 page related-work narrative** organized around your exact pipeline (**LLM explanations → SBERT embeddings → FAISS kNN graph → inconsistency score**) and explicitly position each of the above as: *(a) graph construction, (b) inconsistency scoring, (c) correction/relabeling, (d) evaluation protocols & baselines).*

[1]: https://ui.adsabs.harvard.edu/abs/2020arXiv200412289B/abstract?utm_source=chatgpt.com "Deep k-NN for Noisy Labels"
[2]: https://www.jair.org/index.php/jair/article/download/12125/26676/26691?utm_source=chatgpt.com "Confident Learning: Estimating Uncertainty in Dataset Labels"
[3]: https://www.sciencedirect.com/science/article/abs/pii/S0893608024008487?utm_source=chatgpt.com "Rethinking the impact of noisy labels in graph classification"
