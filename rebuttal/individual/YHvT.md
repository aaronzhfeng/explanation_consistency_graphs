We thank the reviewer for recognizing the novelty of our approach and the insight of the signal aggregation analysis. We address both concerns below with substantial new experiments.

**W1: Reliance on SST-2 with synthetic artifacts; missing AlleNoise.**

We have extended evaluation to MultiNLI (3-class NLI, 25k examples), running all methods across 5 random seeds:

SST-2 Artifact-Aligned (AUROC, 5 seeds):
- Explanation kNN: 0.819 ± 0.004
- LLM Mismatch: 0.628 ± 0.004
- Input kNN: 0.549 ± 0.008
- Cleanlab: 0.136 ± 0.025
- AUM: 0.123 ± 0.025

MultiNLI Artifact-Aligned (AUROC, 5 seeds):
- LLM Mismatch: 0.883 ± 0.001
- NRG: 0.686 ± 0.019
- Classifier kNN: 0.653 ± 0.016
- AUM: 0.601 ± 0.014
- Explanation kNN: 0.557 ± 0.006
- Input kNN: 0.523 ± 0.009

ECG dominates on SST-2 artifact noise, where training-based baselines collapse below random because they confidently learn spurious artifacts. On MultiNLI, Explanation kNN scores near random (.557) while LLM Mismatch leads (.883). Notably, Input kNN also collapses on MNLI (.523), confirming this is a structural property of the embedding space: NLI premise-hypothesis pairs do not cluster by label the way sentiment texts do, rather than a flaw in explanation generation. ECG and LLM Mismatch are complementary signals from the *same single LLM call*: Explanation kNN handles artifact-aligned noise where prediction accuracy alone is insufficient (SST-2 Art: .819 vs Mismatch .628); LLM Mismatch handles regimes where LLM zero-shot accuracy is high (MNLI: .883). The combined pipeline covers both failure modes at no additional inference cost.

Regarding AlleNoise: we have carefully reviewed this benchmark. AlleNoise contains 500k e-commerce product titles across 5,692 categories with instance-dependent noise from semantic ambiguity between similar categories (e.g., "safety shoes" vs "derby shoes"). Crucially, it evaluates noise-robust *training* methods, not noise *detection*; none of our baselines (Cleanlab, AUM, NRG) are evaluated there. Moreover, AlleNoise's own results (Table 2) show that *no method* meaningfully improves over vanilla cross-entropy on real-world noise, and the authors explicitly call for LLM-based approaches as future work (Section 7). This validates ECG's premise. AlleNoise's noise arises from genuine semantic ambiguity, not artifact-correlated patterns, a different and complementary regime from what ECG targets.

**W2: Missing recent baselines (Kim et al. 2024, NoiseGPT).**

We have carefully examined both papers.

Kim et al. (DynaCor, CVPR 2024) encodes quantized logit-difference trajectories through a learned dynamics encoder, then clusters instances in the resulting latent space. It is explicitly motivated by the memorization effect: "DNNs initially grasp simple patterns in correctly labeled data and then gradually overfit to incorrectly labeled data." While DynaCor's learned encoder extracts richer temporal patterns than AUM's scalar summary, both fundamentally depend on the memorization-effect temporal separation (clean learned early, noisy learned late). Under artifact-aligned noise, this separation never emerges: the model learns mislabeled examples *early and confidently*, rendering the trajectory uninformative regardless of how it is processed. Our experiments confirm this: AUM collapses to 0.123 AUROC. Even in DynaCor's own evaluation, AUM drops to F1=16.7% on real-world noise with ResNet34 (their Table 2). Furthermore, DynaCor is vision-only (CIFAR-10/100, Clothing1M) with no text experiments.

Wang et al. (NoiseGPT, NeurIPS 2024) measures MLLM prediction confidence stability under visual feature perturbation. It is designed for and evaluated exclusively on image classification (CIFAR, ImageNet, WebVision), with perturbation operating on visual tokens via a Mix-of-Feature technique that has no direct text analogue. Its signal (whether the MLLM is "stably confident" about an image-label pair) is a confidence-curvature measure that shares the same vulnerability to artifact-aligned noise. Notably, NoiseGPT costs 67-130 GPU-hours per dataset (their Table 7) vs ECG's ~30 minutes on a single A40 (or ~8 minutes and ~$3 via API), and reports no error bars due to computational constraints.

Our contribution is orthogonal: ECG derives signal from explanation *semantics*, not from confidence trajectories or prediction stability. All code and experiment scripts are included in the supplementary materials; we will release the additional experiment code upon acceptance.
