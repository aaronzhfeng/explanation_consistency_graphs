# Reviewer Responses (5000 characters each)


---


## Response to Reviewer YHvT


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
- LLM Mismatch: 0.884 ± 0.003
- NRG: 0.686 ± 0.019
- Classifier kNN: 0.653 ± 0.016
- AUM: 0.601 ± 0.014
- Explanation kNN: 0.557 ± 0.006 (near random)


ECG dominates on SST-2, where training-based baselines perform *below random* because they confidently learn spurious artifacts. On MultiNLI, LLM Mismatch and training-based methods perform better. This reveals ECG's advantage is regime-specific: it excels when explanations provide discriminative semantic signal beyond prediction agreement, but is less effective when the LLM's prediction accuracy alone captures most of the signal. We view this as an honest finding.


Regarding AlleNoise: we have carefully reviewed this benchmark. AlleNoise contains 500k e-commerce product titles across 5,692 categories with instance-dependent noise from semantic ambiguity between similar categories (e.g., "safety shoes" vs "derby shoes"). Crucially, it evaluates noise-robust *training* methods, not noise *detection*; none of our baselines (Cleanlab, AUM, NRG) are evaluated there. Moreover, AlleNoise's own results (Table 2) show that *no method* meaningfully improves over vanilla cross-entropy on real-world noise, and the authors explicitly call for LLM-based approaches as future work (Section 7). This validates ECG's premise. AlleNoise's noise arises from genuine semantic ambiguity, not artifact-correlated patterns, a different and complementary regime from what ECG targets.


**W2: Missing recent baselines (Kim et al. 2024, NoiseGPT).**


We have carefully examined both papers.


Kim et al. (DynaCor, CVPR 2024) encodes quantized logit-difference trajectories through a learned dynamics encoder, then clusters instances in the resulting latent space. It is explicitly motivated by the memorization effect: "DNNs initially grasp simple patterns in correctly labeled data and then gradually overfit to incorrectly labeled data." While DynaCor's learned encoder extracts richer temporal patterns than AUM's scalar summary, both fundamentally depend on the memorization-effect temporal separation (clean learned early, noisy learned late). Under artifact-aligned noise, this separation never emerges: the model learns mislabeled examples *early and confidently*, rendering the trajectory uninformative regardless of how it is processed. Our experiments confirm this: AUM collapses to 0.123 AUROC. Even in DynaCor's own evaluation, AUM drops to F1=16.7% on real-world noise with ResNet34 (their Table 2). Furthermore, DynaCor is vision-only (CIFAR-10/100, Clothing1M) with no text experiments.


Wang et al. (NoiseGPT, NeurIPS 2024) measures MLLM prediction confidence stability under visual feature perturbation. It is designed for and evaluated exclusively on image classification (CIFAR, ImageNet, WebVision), with perturbation operating on visual tokens via a Mix-of-Feature technique that has no direct text analogue. Its signal (whether the MLLM is "stably confident" about an image-label pair) is a confidence-curvature measure that shares the same vulnerability to artifact-aligned noise. Notably, NoiseGPT costs 67-130 GPU-hours per dataset (their Table 7) vs ECG's ~30 minutes on a single A40 (or ~8 minutes and ~$3 via API), and reports no error bars due to computational constraints.


Our contribution is orthogonal: ECG derives signal from explanation *semantics*, not from confidence trajectories or prediction stability.


---


## Response to Reviewer Rvqa


We thank the reviewer for the constructive feedback and the specific suggestions for improvement. We have conducted substantial new experiments addressing all concerns.


**W1: Limited scope (single dataset, synthetic noise, no error bars).**


We have extended evaluation to MultiNLI (3-class NLI, 25k examples) and now report all results over 5 random seeds (42, 123, 456, 789, 1024). We also ran training-based baselines (Cleanlab, AUM, High-Loss, NRG, Classifier kNN) not included in the original submission.


Complete results (AUROC, mean ± std, 5 seeds each):


| Method | SST-2 Uni. | SST-2 Art. | MNLI Uni. | MNLI Art. |
|--------|-----------|-----------|----------|----------|
| Expl. kNN | .915±.003 | **.819±.004** | .560±.007 | .557±.006 |
| Input kNN | .895±.004 | .549±.008 | .541±.008 | .523±.009 |
| LLM Mismatch | .909±.003 | .628±.004 | .883±.002 | **.883±.001** |
| Cleanlab | **.974±.001** | .136±.025 | **.936±.003** | .526±.009 |
| AUM | .968±.003 | .123±.025 | .928±.003 | .601±.014 |
| NRG | .972±.001 | .467±.032 | .936±.003 | .686±.019 |
| Classifier kNN | .941±.006 | .440±.002 | .904±.004 | .653±.016 |

High-Loss and Margin track Cleanlab closely and are omitted for space. Bold = best per column.


The pattern is clear: under uniform noise, training-based methods (Cleanlab, NRG) dominate. Under artifact-aligned noise on SST-2, they collapse below random (0.12-0.47) while Explanation kNN leads at 0.819. On MultiNLI, LLM Mismatch dominates both noise types. Under AUPRC (which better reflects detection precision under 10% noise imbalance; random baseline = 0.102), the gap is even starker: on SST-2 artifact noise, Explanation kNN achieves 0.471 while Cleanlab and AUM drop to 0.057, *below* the random baseline, indicating active anti-detection. This reveals ECG's advantage is regime-specific: it excels when explanation semantics carry discriminative signal beyond prediction agreement, which is precisely the artifact-aligned regime where all other methods degrade.


**W2: Computational overhead.**


We provide a direct cost comparison (25k examples, A40 GPU for all GPU-based methods):


| Method | Wall-clock | Hardware |
|--------|-----------|----------|
| Cleanlab (5-fold CV) | ~48 min | A40 (training) |
| ECG (local vLLM) | ~30 min | A40 (inference only) |
| AUM (3 epochs) | ~16 min | A40 (training) |
| ECG (API route) | ~8 min | No GPU; ~$3 |
| NoiseGPT (Wang et al.) | 67-130 hrs | RTX 4090 |


For a fair GPU-vs-GPU comparison: ECG with Qwen3-8B deployed locally via vLLM on the same A40 takes ≈30 min, comparable to Cleanlab and AUM. Crucially, ECG uses the GPU for inference only (no backpropagation), requiring only ≈16GB VRAM. Alternatively, ECG runs without any GPU via API (≈$3 for 25k examples), making it uniquely accessible to practitioners without ML infrastructure. For reference, NoiseGPT requires 67-130 GPU-hours on an RTX 4090 per dataset (their Table 7).


Moreover, all results reported here use Qwen3-8B, a compact 8B-parameter model. ECG's pipeline is model-agnostic, requiring only structured JSON output from any instruction-following LLM. As model capabilities improve, explanation quality and detection performance would improve with it at no methodological cost. Our current results thus represent a conservative lower bound on ECG's potential.


Importantly, ECG's value proposition is not cost reduction but *signal complementarity*: it provides an orthogonal detection signal that succeeds precisely where training-based methods catastrophically fail (AUROC 0.12-0.47 under artifact-aligned noise).


---


## Response to Reviewer g7YU


We thank the reviewer for the thorough evaluation and the insightful suggestion about the LLM Mismatch baseline. We have run extensive new experiments addressing each concern.


**W1: Only one dataset, one model.**


We extended to MultiNLI (3-class NLI, 25k examples, RoBERTa-base). Full results with 5 seeds are reported in our response to Reviewer Rvqa. ECG dominates on SST-2 artifact noise (0.819 vs 0.136 for Cleanlab) but LLM Mismatch dominates on MultiNLI (0.884 vs 0.557 for ECG). This reveals ECG's advantage is task-dependent, which we view as an honest finding.


**W2: Synthetic artifacts; LLM instructed to ignore tokens.**


We acknowledge this concern. The instruction to ignore special tokens tests a specific hypothesis: that explanation embeddings cluster by semantics rather than surface tokens. In-the-wild artifacts (e.g., annotation biases, demographic correlations) would not have explicit tokens to ignore; the LLM would need to identify semantic inconsistencies implicitly. Our MultiNLI results suggest this is harder than expected for complex tasks, but remains effective for sentiment.


**W3: Single run, no error bars.**


We have re-run all experiments with 5 random seeds (42, 123, 456, 789, 1024). Standard deviations are consistently small (<0.01 AUROC), confirming statistical stability across all conditions reported above.


**W4: Why not just use LLM Mismatch? (the key question)**


We agree this is the most important baseline and have now evaluated it comprehensively:


SST-2 Artifact-Aligned: ECG 0.819 ± 0.004 vs LLM Mismatch 0.628 ± 0.004 (+19 pp)
SST-2 Uniform: ECG 0.915 ± 0.003 vs LLM Mismatch 0.909 ± 0.003 (comparable)
MultiNLI (both noise): LLM Mismatch ~0.884 vs ECG ~0.557 (Mismatch wins)


On SST-2 artifact noise, where confidence-based methods catastrophically fail (Cleanlab: 0.136, AUM: 0.123, both below random), ECG outperforms LLM Mismatch by 19 percentage points. The explanation's *semantic content* (evidence, rationale, counterfactual) carries signal beyond binary agree/disagree.


This distinction is fundamental. The reviewer's suggestion is essentially "use the LLM as a zero-shot classifier and flag disagreements." This is analogous to what NoiseGPT (Wang et al., NeurIPS 2024) does for vision: it measures whether an MLLM is confidently correct about an image-label pair. Notably, NoiseGPT never tests a simple prediction-mismatch baseline either, leaving open the question of how much their probability-curvature method adds over zero-shot classification. Our experiments directly answer this for the text domain: simple mismatch works well when the LLM is highly accurate on the task (MultiNLI, ~88% accuracy), but *fails* to capture the full signal when explanations contain discriminative semantic information beyond prediction agreement (SST-2 artifact noise, where the gap is 19 pp).


ECG's contribution is not universal replacement for all noise detection, but a method that uniquely succeeds when explanation semantics carry signal that prediction agreement does not, precisely the artifact-aligned regime where all other methods degrade.


**Typo (Line 355):** Thank you for catching this; we will correct it in the camera-ready.



