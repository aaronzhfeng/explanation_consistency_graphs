# Official Reviews of Submission 10642

---

## Review 1: Reviewer YHvT

**Date:** 06 Feb 2026, 13:39 (modified: 15 Feb 2026, 14:54)
**Program Chairs:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer YHvT, Authors
**Revisions:** Yes

### Paper Summary:

This paper addresses the challenge of detecting mislabeled examples in training data, focusing on "artifact-aligned noise". The authors argue that standard confidence-based methods (like Cleanlab) and training dynamics methods (like AUM) fail in this regime because the model achieves high confidence and low loss on the noisy examples.

The core method involves:

1. Generating structured explanations (evidence + rationale) for every training instance.
2. Embedding these explanations into a semantic vector space.
3. Constructing a k-Nearest Neighbor (kNN) graph in this explanation space.
4. Calculating disagreement between an instance's label and its neighbors' labels.

The key insight is that confident methods may miss the highly-confident instances because the model relies on artifacts for its prediction. And that while input embeddings might cluster based on spurious tokens, explanation embeddings will cluster based on semantic meaning, revealing the label inconsistency.

### Summary Of Strengths:

1. The paper highlights a weakness in confidence-based data cleaning: when noise is easy to learn due to artifacts, loss-based signals become anti-correlated with the ground truth.

2. The paper leverages the reasoning capability of LLMs to self-explain the prediction decision. It shifts from detecting outliers in input space, where artifacts dominate, to explanation space, where semantics dominate.

3. The analysis in Section 6 regarding why aggregating multiple signals hurt performance is insightful.

### Summary Of Weaknesses:

1. The evaluation's reliance on SST-2 with synthetically injected artifacts. The paper ignores benchmarks for real-world noisy labels, such as AlleNoise [1].

2. The evaluation of the paper misses more recent state-of-the-art methods in noise detection, such as [2] [3].

[1] Raczkowska, Alicja, et al. "AlleNoise: large-scale text classification benchmark dataset with real-world label noise." arXiv preprint arXiv:2407.10992 (2024). [2] Kim, Suyeon, et al. "Learning discriminative dynamics with label corruption for noisy label detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024. [3] Wang, Haoyu, et al. "Noisegpt: Label noise detection and rectification through probability curvature." Advances in Neural Information Processing Systems 37 (2024): 120159-120183.

### Comments Suggestions And Typos:

I do not see any typos.

### Scores:

- **Confidence:** 2 = Willing to defend my evaluation, but it is fairly likely that I missed some details, didn't understand some central points, or can't be sure about the novelty of the work.
- **Soundness:** 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
- **Excitement:** 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the *ACL community.
- **Overall Assessment:** 3 = Findings: I think this paper could be accepted to the Findings of the ACL.

### Additional Information:

- **Ethical Concerns:** There are no concerns with this submission
- **Needs Ethics Review:** No
- **Reproducibility:** 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
- **Datasets:** 2 = Documentary: The new datasets will be useful to study or replicate the reported research, although for other purposes they may have limited interest or limited usability. (Still a positive rating)
- **Software:** 1 = No usable software released.
- **Knowledge Of Or Educated Guess At Author Identity:** No
- **Knowledge Of Paper:** After the review process started
- **Knowledge Of Paper Source:** Preprint on arxiv
- **Impact Of Knowledge Of Paper:** Not much
- **Reviewer Certification:** I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
- **Publication Ethics Policy Compliance:** I did not use any generative AI tools for this review

---

## Review 2: Reviewer Rvqa

**Date:** 12 Feb 2026, 01:46 (modified: 15 Feb 2026, 14:54)
**Program Chairs:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer Rvqa, Authors
**Revisions:** Yes

### Paper Summary:

This work suggests a method for detecting mislabeled training data by analyzing the consistency of LLM-generated explanations rather than raw inputs or model confidence. The authors demonstrate that while standard confidence-based methods fail on "artifact-aligned" noise, explanation representations capture semantic inconsistencies invisible to the classifier's loss function, though it also notes that combining this with training dynamics signals can degrade performance when artifacts make noisy labels easy to learn.

### Summary Of Strengths:

- The comparison between ECG (explanation space) and ECG (input space) seems reasonable. By keeping the algorithm constant and changing only the embedding substrate, the authors isolate the value of the LLM explanations
- The analysis of why multi-signal aggregation failed is valuable.

### Summary Of Weaknesses:

- I think the scope of the evaluation is very limited. First, the work only looks at Synthetic Noise, what is called "artifact-aligned" noise created by appending specific tokens (e.g., \<RATING=5\>). While this is a nice proof-of-concept it does not extend to real-world artifacts (like specific syntactic structures in NLI). Further, the authors only use one dataset, namely the SST-2 data (binary sentiment). The lack of evaluation on multi-class problems or more complex reasoning tasks (like NLI or QA) limits claims regarding generalizability. Finally, the authors report single-run results without error bars or confidence intervals, which is statistically weak for empirical NLP papers

- It seems like the method comes with a substantial computational overhead. It requires generating structured explanations for the full training set using an LLM. This is more expensive than loss-based methods or input-embedding methods, potentially limiting scalability for massive datasets.

### Comments Suggestions And Typos:

- Apply ECG to the standard "spurious correlation" benchmarks in NLP. This would test the method against "natural" artifacts rather than just injected tokens.
- Include a comparison of wall-clock time and compute costs against baselines. Since the method involves LLM inference on the whole train set, readers need to know the price of the accuracy gain.

### Scores:

- **Confidence:** 2 = Willing to defend my evaluation, but it is fairly likely that I missed some details, didn't understand some central points, or can't be sure about the novelty of the work.
- **Soundness:** 2.5
- **Excitement:** 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the *ACL community.
- **Overall Assessment:** 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.

### Additional Information:

- **Ethical Concerns:** There are no concerns with this submission
- **Needs Ethics Review:** No
- **Reproducibility:** 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
- **Datasets:** 3 = Potentially useful: Someone might find the new datasets useful for their work.
- **Software:** 3 = Potentially useful: Someone might find the new software useful for their work.
- **Knowledge Of Or Educated Guess At Author Identity:** No
- **Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources
- **Knowledge Of Paper Source:** N/A, I do not know anything about the paper from outside sources
- **Impact Of Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources
- **Reviewer Certification:** I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
- **Publication Ethics Policy Compliance:** I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits

---

## Review 3: Reviewer g7YU

**Date:** 12 Feb 2026, 13:02 (modified: 15 Feb 2026, 14:54)
**Program Chairs:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer g7YU, Authors
**Revisions:** Yes

### Paper Summary:

This paper proposes explanation-consistency graph (ECG), a technique to detect potentially mislabeled data points. The key insight is that a mislabeled data point typically has a different label from the input points for which the model explanations are most similar. Using a Qwen 3 8B model to generate the explanation and a custom proposal to format the explanation for embedding similarity check, the authors demonstrate that this method is on par with existing approaches such as confident learning when the label noises are randomly injected and outperforms existing approaches when they are artifact-aligned (i.e., due to spurious correlations).

### Summary Of Strengths:

The proposed method is quite novel and has an intuitive explanation.

The experimental results show the effectiveness of the method.

Several different baselines have been compared in the experiments.

The paper is generally well written and easy to follow.

### Summary Of Weaknesses:

The experimental setup is relatively thin, with only one dataset SST-2 and one model (Roberta).

Furthermore, only manually injected noises (e.g., random noise or artifact-aligned noises) are studied, and it is unclear whether this approach is valid for in-the-wild mislabeling detection. The artifact is also very "synthetic", with just special tokens in the angle bracket. The explanation-generating LLM (Qwen 3) is even directly instructed to ignore such tokens, casting doubt on the applicability of this approach in the wild.

The results are also collected over a single run. Considering the (relatively) cheap computation cost of Roberta finetuning with today's GPU hardware, I do not see a good reason for the lack of experimental rerun to establish statistical significance.

If you are using a (much more capable) LLM to generate explanations, it seems like intuitive to use it to directly detect mislabeling. For example, since the predicted label is already generated in Line 286, why not just discard all examples for which the predicted label mismatch the provided label? Obviously, this reduces to distilling a Roberta from Qwen in this case, but given that Qwen's performance on SST should be very high (potentially due to memorization), this seems like a very strong baseline.

### Comments Suggestions And Typos:

In Line 355, AUROC is not yet introduced when the ablation study is presented.

### Scores:

- **Confidence:** 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
- **Soundness:** 2.5
- **Excitement:** 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
- **Overall Assessment:** 2.5 = Borderline Findings

### Additional Information:

- **Ethical Concerns:** There are no concerns with this submission
- **Reproducibility:** 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
- **Datasets:** 1 = No usable datasets submitted.
- **Software:** 1 = No usable software released.
- **Knowledge Of Or Educated Guess At Author Identity:** No
- **Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources
- **Knowledge Of Paper Source:** N/A, I do not know anything about the paper from outside sources
- **Impact Of Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources
- **Reviewer Certification:** I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
- **Publication Ethics Policy Compliance:** I did not use any generative AI tools for this review
