# Technical & Factual Accuracy Review Response

*Paste the model's response here.*

## 1. Verification Summary

The paper’s *core experimental narrative* (ECG / Explanation‑kNN succeeds on artifact‑aligned noise; confidence/loss‑based methods fail) is internally plausible and the key arithmetic claim (“24% improvement” from 0.671→0.832) checks out. 

However, there are **multiple high‑impact factual/citation problems** (wrong arXiv IDs, wrong authors, mismatched years, incorrect statistic wording) that would substantially undermine credibility unless corrected. Several method details are also **underspecified or notationally inconsistent** (variable collisions, “Laplace smoothing” not reflected in the equation), which weakens technical precision. 

---

## 2. Citation Issues

### A) Misstated or overgeneralized cited statistic

* **Claim in paper (Intro):** “3–6% of labels in widely‑used benchmarks—including *MNIST*, *ImageNet*, and *Amazon Reviews*—are incorrect (Northcutt et al., 2021b).” 
  **Problem:** The cited work reports **very different error rates across datasets**; notably **MNIST is far below 3%**. In Northcutt et al.’s table, MNIST is ~**0.15%**, ImageNet ~**5.83%**, and Amazon Reviews ~**3.90%**; their “average 3.3%” is across multiple datasets, not a uniform 3–6% applying to each named benchmark. ([ar5iv][1])
  **Why it matters:** This reads like a factual error (or at minimum misleading phrasing) and is easy for reviewers to spot.

### B) Bibliographic metadata errors (authors / year / title / arXiv)

These are especially damaging because they resemble “LLM‑generated references” errors.

* **Chong et al. (2022) author list appears wrong in your references.** 
  The arXiv/EMNLP paper “Detecting Label Errors by using Pre‑Trained Language Models” lists **Derek Chong, Jenny Hong, Christopher D. Manning**. ([arXiv][2])
  Your bib entry shows different authors (“Jenny Ng”, “Kefeng Liu”), which is incorrect.

* **Zhu et al. (2022) has an incorrect arXiv ID in your references.** 
  “Detecting Corrupted Labels Without Training a Model to Predict” is **arXiv:2110.06283** (ICML 2022). ([arXiv][3])
  Your bib cites **arXiv:2208.09329**, which is an unrelated paper (“Causal Intervention Improves Implicit Sentiment Analysis”). ([arXiv][4])

* **WANN / Di Salvo et al. entry title/year mismatched.** 
  arXiv:2408.14358’s title is **“An Embedding is Worth a Thousand Noisy Labels”** and it introduces WANN; it was submitted Aug 2024 and later revised 2025. ([arXiv][5])
  Your reference uses a different title (“WANN: Weighted Adaptive Nearest Neighbors…”) and labels it “2025 arXiv preprint,” which is inconsistent with the arXiv submission year (unless you instead cite the TMLR publication year—but then you should cite the venue, not “arXiv preprint”).

* **CTRL citation has wrong year and (at least one) wrong author name.** 
  “CTRL: Clustering Training Losses for Label Error Detection” on arXiv is **Chang Yue, Niraj K. Jha (Aug 2022)**. ([arXiv][6])
  Your reference lists it as “2024” and “Somesh Jha,” which does not match the arXiv record.

* **Dual‑kNN reference misattributes first author / title.** 
  arXiv:2503.04869 is **“Label Distribution Learning‑Enhanced Dual‑KNN for Text Classification”** by **Bo Yuan et al. (2025)**. ([arXiv][7])
  Your reference shows “Zhaomin Yuan … Dual nearest neighbor text classification,” which does not match.

* **“others Chen” / “others Li” in the reference list** looks like a BibTeX author‑field formatting bug (e.g., “and others”). 
  Even if the underlying papers are real, this presentation will be treated as careless or potentially fabricated.

### C) Missing citations for concrete tools/models (less severe, but still)

* You rely on **Qwen3** and **vLLM structured/JSON outputs** but do not cite a model report/model card or the vLLM feature documentation. vLLM explicitly documents JSON‑schema structured outputs. ([vLLM][8])
  This isn’t “wrong,” but it’s an avoidable reviewer complaint.

---

## 3. Mathematical Issues

### A) Notation collision / inconsistent symbol use

* In the LaTeX source and reflected in the PDF, **(r_i)** is used both for:

  * the **reliability score** (“(r_i=\frac{1}{3}(L_i+E_i+R_i))”), and
  * the **rationale text field** in the concatenated string (“Rationale: (r_i)”). 
    This is a clear notational error and can confuse readers (and reviewers).

### B) “Laplace smoothing” is not mathematically specified in the probability equation

* You state (p_i(c)) is computed “with Laplace smoothing (\epsilon=10^{-3})” but the provided equation is the **unsmoothed** weighted neighbor vote:
  [
  p_i(c)=\sum_{j\in N(i)} w_{ij}\mathbf{1}[y_j=c].
  ]

  If you truly smooth, you should show something like:
  [
  p_i(c)=\frac{\epsilon+\sum_{j\in N(i)} w_{ij}\mathbf{1}[y_j=c]}{C\epsilon+\sum_{j\in N(i)} w_{ij}}
  ]
  (and note (\sum_j w_{ij}=1) after normalization). Otherwise “Laplace smoothing” reads as hand‑waving.

### C) Multi‑signal aggregation scale/normalization is underspecified

* The different signals you define ((-\log p), NLI probability, variance/dispersion, (-\mathrm{AUM}), token ratios) are on **different numeric scales** unless explicitly normalized. If you do weighted sums, you must specify normalization (min‑max, z‑score, rank‑based, etc.) or reviewers may assume the combination is ill‑posed. 
  This is especially relevant because you report multi‑signal underperforms; without scale details, readers cannot tell if this is a modeling result or a normalization artifact.

---

## 4. Unsupported Claims

These items are stated as facts but lack supporting evidence *within the paper* (table/figure/appendix detail) or a citation.

* **“14B cites spurious tokens in 4.1% of explanations vs 0.4% for 1.7B”** is presented as a concrete measurement, but the paper does not provide the counting protocol, sample size, or a table/figure showing those rates. 
  If true, it should be easy to add as a small table in the appendix.

* **Compute/runtime claims** (“~10 minutes for 25k examples on H100”; “~25 GPU‑hours total”) are plausible but currently read as anecdotal without logs/configuration detail (batch size, average generated tokens/example, vLLM version, tensor parallel, etc.). 

* **“We mitigate LLM explanation failures with … NLI verification”** is stated in Limitations, but your main best‑performing method is Explanation‑kNN, and your own results suggest multi‑signal (including NLI) can degrade performance. Unless NLI is used as a filter (not as a score), the mitigation claim should be narrowed or clarified. 

---

## 5. Internal Inconsistencies

* **“Enforcing extractive evidence spans … via schema‑constrained decoding”** is ambiguous as written. JSON‑schema constrained decoding (as supported in modern serving stacks) enforces *structure*, not necessarily that “evidence” is an *exact substring* of the input unless you additionally do post‑validation or a much stronger constraint mechanism. 
  Since you also cite “schema‑constrained decoding,” you should explain whether you:

  * post‑validate evidence as a substring and re‑sample if invalid, or
  * constrain decoding to only allow substrings (nontrivial), or
  * simply instruct the model and then trust it.

* **LLM‑mismatch baseline vs “ignore metadata tokens” narrative:** You emphasize prompting the LLM to ignore spurious markers, but the LLM‑mismatch baseline is only modest on artifact noise. This is not a formal contradiction, but it raises a consistency question: if the LLM reliably ignores the marker, mismatch should be much stronger. This warrants an explicit analysis (e.g., LLM predicted‑label accuracy on clean vs artifact‑marked inputs; ablation where you strip the marker token). 

---

## 6. Potentially Fabricated Content

Nothing in the *method idea* looks inherently fabricated, and several key referenced works do exist. The main fabrication risk is **bibliography integrity**:

* Multiple references have **wrong arXiv IDs**, **wrong authors**, and **broken author strings (“others …”)**. 
  This combination often triggers “this was autogenerated” suspicion. Even if accidental, reviewers may doubt other claims unless the bib is cleaned.

Secondary red flag (worth checking your code):

* **Identical metrics across multiple baselines** in Table 1 (Cleanlab / High‑Loss / AUM all at the same AUROC/AUPRC) could be real (if all induce the same ranking), but it is also consistent with an evaluation bug where the same score vector is reused. 
  This is not a definitive error, but it is worth re‑verifying.

---

## 7. Recommended Fixes

### A) Correct the high‑visibility intro statistic (suggested rewording)

Replace:

> “3–6% of labels in widely‑used benchmarks—including MNIST, ImageNet, and Amazon Reviews—are incorrect …”

With something like:

> “Audits of popular benchmark **test sets** find label error rates ranging from **0.15% (MNIST)** to **5.83% (ImageNet)** and **3.90% (Amazon Reviews)**, averaging **3.3% across 10 datasets** (Northcutt et al., 2021b).” ([ar5iv][1])

Or, if you want a short version:

> “Audits find non‑trivial label error rates across benchmarks (0.15–10% across 10 datasets) (Northcutt et al., 2021b).” ([ar5iv][1])

### B) Repair bibliography to eliminate “hallucination” signals

Concretely fix at least these entries to match canonical metadata:

* Chong et al. correct authors (Chong/Hong/Manning). ([arXiv][2])
* Zhu et al. correct arXiv (2110.06283), not 2208.09329. ([arXiv][3])
* Di Salvo et al. correct title/submission year; if you want 2025, cite the accepted venue rather than calling it “arXiv preprint 2025.” ([arXiv][5])
* CTRL: correct author (Niraj K. Jha), and reconcile year vs venue. ([arXiv][6])
* Dual‑KNN: correct first author/title. ([arXiv][7])
* Remove “others …” artifacts by fixing BibTeX author fields.

### C) Fix notation and define smoothing explicitly

* Rename reliability score to **(\rho_i)** (or similar), and reserve (r_i) for rationale text (or vice‑versa).
* Update the paper to show the actual smoothed formula for (p_i(c)) and confirm that ( \sum_c p_i(c)=1 ). 

### D) Add minimal reproducibility + significance hygiene

* Report **seed(s)** for:

  * subsampling to 25k,
  * noise injection,
  * model finetuning,
  * explanation sampling. 
* Add **mean ± std** across (at least) 3 seeds for Table 1/2/3 AUROC. This will preempt “single run” criticism.

### E) Support currently unsupported quantitative claims

* If you keep the “4.1% vs 0.4% spurious token citation” claim, add:

  * definition of “cites spurious token” (exact match? tokenized? case‑sensitive?),
  * count and denominator,
  * a small appendix table. 

### F) Clarify what “enforced extractive evidence” means

Suggested replacement wording if you do **post‑validation**:

> “We enforce JSON validity via schema‑constrained decoding, and we **post‑validate** that each evidence span is a substring of the input (re‑sampling when invalid).” 

If you **do not** post‑validate:

> “We enforce JSON validity via schema‑constrained decoding and **instruct** the model to quote extractive evidence spans.” 

### G) Add missing citations for key infrastructure (optional but recommended)

* Cite vLLM structured outputs / JSON schema support if it is central to your pipeline. ([vLLM][8])
* Add a citation for Qwen3 (technical report/model card) if you name it as the primary LLM.

---

If you want a “fastest credibility win” order: **(1) fix the Northcutt statistic sentence, (2) repair the broken references/arXiv IDs/authors, (3) fix the (r_i) notation collision + show the smoothed (p_i(c)) equation, (4) add seed/variance reporting.**

[1]: https://ar5iv.org/pdf/2103.14749 "[2103.14749] Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"
[2]: https://arxiv.org/abs/2205.12702 "[2205.12702] Detecting Label Errors by using Pre-Trained Language Models"
[3]: https://arxiv.org/abs/2110.06283 "[2110.06283] Detecting Corrupted Labels Without Training a Model to Predict"
[4]: https://arxiv.org/abs/2208.09329 "[2208.09329] Causal Intervention Improves Implicit Sentiment Analysis"
[5]: https://arxiv.org/abs/2408.14358 "[2408.14358] An Embedding is Worth a Thousand Noisy Labels"
[6]: https://arxiv.org/abs/2208.08464?utm_source=chatgpt.com "CTRL: Clustering Training Losses for Label Error Detection"
[7]: https://arxiv.org/abs/2503.04869?utm_source=chatgpt.com "Label Distribution Learning-Enhanced Dual-KNN for Text Classification"
[8]: https://docs.vllm.ai/en/latest/features/structured_outputs/ "Structured Outputs - vLLM"
