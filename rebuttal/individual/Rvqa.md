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

The pattern is clear: under uniform noise, training-based methods (Cleanlab, NRG) dominate. Under artifact-aligned noise on SST-2, they collapse below random (0.12-0.47) while Explanation kNN leads at 0.819. Under AUPRC (random baseline = 0.102), the gap is starker still: Explanation kNN achieves 0.471 while Cleanlab and AUM drop to 0.057, *below* random, indicating active anti-detection.

On MultiNLI, Explanation kNN scores near random (.557) while LLM Mismatch dominates (.883). Notably, Input kNN also collapses on MNLI (.523), confirming this is a structural property of the embedding space rather than a flaw in explanation generation: NLI premise-hypothesis pairs do not cluster by label in embedding space the way sentiment texts do. For sentiment, evidence like "great acting, compelling story" is inherently label-discriminative; for NLI, reasoning about premise-hypothesis relationships (e.g., "the premise describes a location while the hypothesis makes a temporal claim") can plausibly appear under NEUTRAL or CONTRADICTION. When embeddings are label-indiscriminative, the kNN neighborhood-surprise signal has no leverage regardless of whether it operates on inputs or explanations.

Crucially, ECG and LLM Mismatch are complementary signals produced by the *same single LLM call*. Explanation kNN detects artifact-aligned noise where LLM prediction accuracy alone is insufficient (SST-2 Art: .819 vs LLM Mismatch .628); LLM Mismatch detects noise where LLM zero-shot accuracy is high (MNLI: .883 on both conditions). The combined pipeline covers both failure modes at no additional inference cost, since both signals are extracted from the same structured JSON output.

We acknowledge two remaining gaps: (1) all noise remains synthetically injected; benchmarks with natural artifacts (e.g., HANS, MNLI-mismatched) would test generalization to in-the-wild spurious correlations, and we agree this is the critical next step; (2) the kNN signal's effectiveness depends on embedding-space label separability, which varies across tasks. We will include explicit scope conditions in the revision.

**W2: Computational overhead.**

We provide a direct cost comparison (25k examples, A40 GPU for all GPU-based methods):

| Method | Wall-clock | Hardware |
|--------|-----------|----------|
| Cleanlab (5-fold CV) | ~48 min | A40 (training) |
| ECG (local vLLM) | ~30 min | A40 (inference only, single pass) |
| AUM (3 epochs) | ~16 min | A40 (training) |
| ECG (API route) | ~8 min | No GPU; ~$3 |
| NoiseGPT (Wang et al.) | 67-130 hrs | RTX 4090 |

For a fair GPU-vs-GPU comparison: ECG with Qwen3-8B deployed locally via vLLM on the same A40 takes ≈30 min (single-pass, greedy decoding), comparable to Cleanlab and AUM. Crucially, ECG uses the GPU for inference only (no backpropagation), requiring only ≈16GB VRAM. Alternatively, ECG runs without any GPU via API (≈$3 for 25k examples), making it uniquely accessible to practitioners without ML infrastructure. For reference, NoiseGPT requires 67-130 GPU-hours on an RTX 4090 per dataset (their Table 7).

Moreover, all results reported here use Qwen3-8B, a compact 8B-parameter model. ECG's pipeline is model-agnostic, requiring only structured JSON output from any instruction-following LLM. As model capabilities improve, explanation quality and detection performance would improve with it at no methodological cost. Our current results thus represent a conservative lower bound on ECG's potential.

Importantly, ECG's value proposition is not cost reduction but *signal complementarity*: it provides an orthogonal detection signal that succeeds precisely where training-based methods catastrophically fail (AUROC 0.12-0.47 under artifact-aligned noise).
