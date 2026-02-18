# Reference Validity Check Response

*Paste the model's response here.*

## 1. Verification Summary

* **Citations in `main.tex`:** 66 `\cite…{}` occurrences, **42 unique citation keys**.
* **Key consistency:** **All 42 cited keys have matching BibTeX entries** in `ecg.bib` (no missing keys).
* **Unused BibTeX entries:** **11 entries in `ecg.bib` are not cited** in `main.tex` (see §2.4).
* **Main finding:** I did **not** find a clearly “non-existent”/fabricated paper among the items I checked; however, **15/42 cited entries have serious bibliographic inaccuracies** (wrong author(s), wrong title, wrong year/venue labeling, wrong arXiv identifier, or corrupted author strings). Several others have **quality/formatting issues** (missing URL/DOI, or author list written as “and others/others,” which is not valid BibTeX author formatting).

## 2. Confirmed Valid Citations

### 2.1 Valid and bibliographically “clean” (no major issues detected)

These entries look consistent (title/author/year/venue broadly match the known work) and did not trigger red flags in the BibTeX fields:

* `poliak2018hypothesis`
* `belinkov2019premise`
* `clark2019product`
* `bahri2020deep`
* `han2020influence`
* `kaushik2020learning`
* `lertvittayakumjorn2020find`
* `pleiss2020identifying`
* `sagawa2020distributionally`
* `utama2020self`
* `lertvittayakumjorn2021explanation`
* `northcutt2021confident` ([arXiv][1])
* `northcutt2021pervasive`
* `wiegreffe2021label`
* `maini2022second`
* `pezeshkpour2022combining`
* `geng2023grammar`
* `kim2023neural`
* `madsen2024faithfulness`

### 2.2 Valid, but with BibTeX quality issues (should be cleaned)

These appear to refer to real works, but the BibTeX is incomplete or improperly formatted:

* `iscen2020graphnoisylabels` — author list uses “and others” (not valid BibTeX author semantics).
* `teso2021interactive` — author list uses “and others”.
* `xu2024sayself` — author list uses “and others”.
* `agarwal2024faithfulness` — author list uses “and others”.
* `randl2024selfexplanation` — author list uses “and others”.
* `tu2020nlprobust` — **entry type/venue mismatch** (TACL should be `@article`, not `@inproceedings`; see §6).
* `gururangan2018annotation` — missing URL/DOI (add ACL Anthology URL/DOI).
* `mccoy2019right` — missing URL/DOI (add ACL Anthology URL/DOI).

## 3. Potentially Problematic Citations

These items **exist**, but the `ecg.bib` record is **incorrectly attributed and/or materially wrong**. For each, I list the issue and a concrete fix direction.

### 3.1 Wrong author(s) and/or wrong title (material attribution errors)

* `chong2022detecting`

  * **Issue:** BibTeX authors are wrong (your entry lists “Ng, Jenny” and “Liu, Kefeng”; the arXiv/ACL records list **Derek Chong, Jenny Hong, Christopher D. Manning**).
  * **Fix:** Replace author list with the official authors; keep title as-is (it matches the arXiv title). Best: import BibTeX from ACL Anthology or arXiv.

* `wang2022token`

  * **Issue:** Title and author are wrong: arXiv record is **“Detecting Label Errors in Token Classification Data”** by **Wei‑Chen Wang** and **Jonas Mueller** (not “Wenxuan Wang,” and not your title). ([arXiv][2])
  * **Fix:** Update title/author to match arXiv; optionally include the NeurIPS 2022 InterNLP workshop reference (shown on arXiv). ([arXiv][2])

* `zhu2022beyondimages`

  * **Issue:** Your entry’s arXiv id (`2208.09329`) and authors do not match the actual paper. The real work is **“Detecting Corrupted Labels Without Training a Model to Predict”** by **Zhaowei Zhu, Zihao Dong, Yang Liu** (arXiv:2110.06283; ICML 2022).
  * **Fix:** Replace BibTeX with the ICML/PMLR BibTeX (or arXiv BibTeX).

* `huang2023llmselfexplanations`

  * **Issue:** Your BibTeX shortens/changes the title and has the wrong first author (“Jie Huang”). The arXiv record is **“Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations”** by **Shiyuan Huang et al.**
  * **Fix:** Update title and full author list from arXiv.

* `lee2023xmd`

  * **Issue:** Author list in BibTeX appears incomplete/incorrect relative to the arXiv record for **arXiv:2210.16978**.
  * **Fix:** Replace BibTeX using arXiv’s BibTeX export for arXiv:2210.16978.

* `thyagarajan2023multilabel`

  * **Issue:** Your title does not match the arXiv record; arXiv:2211.13895 is **“Identifying Incorrect Annotations in Multi-Label Classification Data.”**
  * **Fix:** Update title/year to match arXiv (or cite the eventual publication venue if any, but then update venue fields accordingly).

### 3.2 “Method name used as title” (title/venue mislabeling)

* `beurer2024domino`

  * **Issue:** Your entry uses a DOMINO-focused title; the arXiv paper title is **“Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation.”**
  * **Fix:** Use the official arXiv title and mention DOMINO in note/abstract text if needed.

* `disalvo2025wann`

  * **Issue:** Your BibTeX title does not match the arXiv/TMLR record; arXiv:2408.14358 is **“An Embedding is Worth a Thousand Noisy Labels”** and introduces WANN as a method. Also your entry uses “and others” and labels it as 2025 arXiv-preprint.
  * **Fix:** Use the official title + full authors; if you want “2025,” cite the **accepted venue (TMLR)** rather than “arXiv preprint,” because the arXiv submission is 2024.

### 3.3 Wrong year/venue labeling for arXiv vs journal/proceedings

* `yue2024ctrl`

  * **Issue:** Your BibTeX treats it as an “arXiv preprint” but uses a later year; arXiv:2208.08464 is **“CTRL: Clustering Training Losses for Label Error Detection.”**
  * **Fix:** Either (a) cite the arXiv version with arXiv’s year, or (b) cite the journal version with the correct journal/volume/pages instead of “arXiv preprint.”

* `chen2025explanationconsistency`

  * **Issue:** arXiv:2401.13986 is a **2024** submission titled **“Towards Consistent Natural-Language Explanations via Explanation-Consistency Finetuning”** by **Yanda Chen et al.** Your entry uses year 2025, a different title, and “Chen, others.”
  * **Fix:** Update key/year/title/authors to match arXiv (or cite a later venue if published; but then update venue fields accordingly).

* `li2025decole`

  * **Issue:** arXiv:2507.07216 title is **“Bias-Aware Mislabeling Detection via Decoupled Confident Learning”** and it defines **DeCoLe**; your BibTeX title/author field is not the official one (and uses “Li, others”).
  * **Fix:** Replace BibTeX from arXiv; include full authors (Yunyi Li, Maria De-Arteaga, Maytal Saar‑Tsechansky).

* `yuan2025dualknn`

  * **Issue:** arXiv:2503.04869 is **“Label Distribution Learning-Enhanced Dual-KNN for Text Classification.”** Your BibTeX title is different and author list is truncated.
  * **Fix:** Replace with arXiv BibTeX (full title + authors).

### 3.4 Wrong author attribution

* `xia2024fofo`

  * **Issue:** Your BibTeX first author does not match the arXiv record for **FOFO** (arXiv:2402.18667).
  * **Fix:** Replace author list from arXiv BibTeX (and keep the official title).

### 3.5 Corrupted BibTeX author field

* `chuang2022robust`

  * **Issue:** The author field contains corrupted text (`Vineet,...eel`), which is a BibTeX formatting error and likely not the canonical author list.
  * **Fix:** Replace the entire entry by importing official BibTeX from CVPR/arXiv for “Robust Contrastive Learning against Noisy Views.”

## 4. Missing Information

### 4.1 Missing URL/DOI in cited entries

* `gururangan2018annotation` — add ACL Anthology URL/DOI.
* `mccoy2019right` — add ACL Anthology URL/DOI.

### 4.2 Non-BibTeX author placeholders

These cited entries contain `author = "... and others"` or similar, which is not a correct way to encode “et al.” in BibTeX:

* `agarwal2024faithfulness`, `iscen2020graphnoisylabels`, `teso2021interactive`, `xu2024sayself`, `randl2024selfexplanation` (and several “major issue” entries also use “others”).

## 5. Attribution Issues

### 5.1 Key-claim checks requested in your table

* `northcutt2021pervasive` (“3–6% of labels are incorrect”)

  * **Supported, but should be stated more precisely.** The abstract reports **≥3.3% average label errors across 10 datasets** and **≥6%** in the ImageNet validation set. Your “3–6%” phrasing is directionally consistent but compresses multiple statements into a single range.

* `northcutt2021confident` (Confident learning method)

  * **Accurate high-level description.** The arXiv abstract explicitly states it estimates the **joint distribution between noisy and uncorrupted labels** and identifies label errors (“confident learning”). ([arXiv][1])

* `pleiss2020identifying` (AUM)

  * **Accurate.** The NeurIPS paper is explicitly about identifying mislabeled data using AUM.

* `poliak2018hypothesis` (“NLI solved with hypothesis only”)

  * **Overstated.** The paper shows “hypothesis-only” baselines can achieve surprisingly strong performance and beat trivial baselines, which indicates artifacts; it does not claim NLI is “solved.” Rephrase to “hypothesis-only baselines achieve strong performance, indicating annotation artifacts.”

* `gururangan2018annotation` (annotation artifacts)

  * **Correct attribution.** This paper documents annotation artifacts and shows high hypothesis-only performance (e.g., SNLI/MultiNLI).

* `bahri2020deep` (kNN noisy label detection)

  * **Method exists and attribution is reasonable** for a kNN/neighborhood-based signal used in noisy-label contexts (the paper discusses Deep kNN for noisy labels).

* `kim2023neural` (relation graph formulation)

  * **Consistent with the paper’s stated framing** (relation graph as a unified framework for label noise/outliers).

* `disalvo2025wann` (WANN reliability weighting)

  * **The method exists, but your BibTeX record is mis-titled/mis-year-labeled.** The arXiv/TMLR record describes the WANN mechanism; fix the bibliographic metadata.

* `lertvittayakumjorn2021explanation` (explanation debugging survey)

  * **Correct.** It is explicitly a survey on explanation-based human debugging for NLP models.

* `geng2023grammar` (grammar-constrained decoding)

  * **Correct.** This work is explicitly about grammar-constrained decoding for structured tasks (and is labeled accepted at EMNLP 2023).

## 6. Formatting Issues

* **Wrong entry type:** `tu2020nlprobust` is in **TACL** (journal), so it should be `@article` with `journal=...`, not `@inproceedings` with `booktitle=Transactions…`.
* **Invalid author encoding:** any entry using `and others` / `others` should be replaced with the full author list (BibTeX treats “others” as a literal author name).
* **Corrupted author string:** `chuang2022robust` contains non-parseable author text.

## 7. Recommendations

1. **Replace all “major issue” entries (§3) by importing BibTeX from canonical sources** (ACL Anthology, PMLR/ICML, arXiv BibTeX export, CVPR official BibTeX). This is the fastest way to eliminate wrong-title/wrong-author errors.
2. **Eliminate “and others/others” in `author={...}` everywhere**—use full author lists. (If you need abbreviated display, handle that in the bibliography style, not in the BibTeX data.)
3. **Normalize arXiv vs. published-version citations:**

   * If citing arXiv: set `year` to arXiv submission year and keep `journal = {arXiv:...}`.
   * If citing a journal/proceedings version: use the actual venue (and volume/pages where applicable) and optionally include arXiv in `note`/`url`.
4. **Fix the two missing URL/DOI items** (`gururangan2018annotation`, `mccoy2019right`) by adding ACL Anthology links.
5. **Edit the two overstated attribution sentences** about “hypothesis-only solving NLI” (§5.1) to match what the cited papers actually demonstrate.
6. Run a BibTeX checker/linter (and compile once) after these edits to ensure there are no remaining parse errors (especially from corrupted `author` fields).

## 2.4 Unused BibTeX entries (optional cleanup)

These are present in `ecg.bib` but not cited in `main.tex`:

* `deyoung2020eraser`, `joshi2022er`, `chen2023rev`, `hooker2019roar`, `pruthi2020tracin`, `park2023trak`, `li2020dividemix`, `liu2020early`, `liu2019roberta`, `reimers2019sentence`, `socher2013recursive`

[1]: https://arxiv.org/abs/1911.00068 "[1911.00068] Confident Learning: Estimating Uncertainty in Dataset Labels"
[2]: https://arxiv.org/abs/2210.03920 "[2210.03920] Detecting Label Errors in Token Classification Data"
