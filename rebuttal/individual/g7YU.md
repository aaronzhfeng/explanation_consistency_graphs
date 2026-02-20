We thank the reviewer for the thorough evaluation and the insightful suggestion about the LLM Mismatch baseline. We have run extensive new experiments addressing each concern.

**W1: Only one dataset, one model.**

We extended to MultiNLI (3-class NLI, 25k examples, RoBERTa-base). Full results with 5 seeds are reported in our response to Reviewer Rvqa. On SST-2 artifact noise, ECG leads (0.819) while Cleanlab collapses to 0.136. On MultiNLI, LLM Mismatch leads (0.883) while Explanation kNN scores near random (0.557), as does Input kNN (0.523), confirming the failure is structural (NLI embeddings do not separate by label) rather than explanation-specific. ECG and LLM Mismatch are complementary signals from the same LLM call, covering different failure modes at no additional cost. All code and experiment scripts are included in the supplementary materials; we will release the additional experiment code upon acceptance.

**W2: Synthetic artifacts; LLM instructed to ignore tokens.**

The instruction to ignore metadata tokens is a *control*, not a dependency. It ensures the LLM reasons about content rather than formatting noise, isolating the variable under test: whether explanation semantics reveal label inconsistencies. Critically, real-world artifacts are almost never explicit surface tokens. Annotation biases, demographic correlations, and syntactic shortcuts (e.g., hypothesis-only cues in NLI, negation words correlating with contradiction) are invisible at the token level; there would be nothing to instruct the LLM to ignore. The LLM would simply process the content and produce explanations reflecting the true semantic relationship, which is the mechanism ECG exploits. Our synthetic tokens are a worst-case test for the *classifier* (perfectly predictive artifacts), not an easy case for *ECG* (the LLM must still generate semantically coherent explanations that cluster correctly in embedding space). Our MNLI experiments provide partial evidence: no ignore instruction was given for MNLI, yet LLM Mismatch achieves .883, showing the LLM can flag label inconsistencies on a task with known non-token structural artifacts (hypothesis-only bias) without explicit guidance. The kNN step fails on MNLI for the embedding-geometry reason discussed in W1, not because the LLM cannot handle non-surface artifacts.

**W3: Single run, no error bars.**

We have re-run all experiments with 5 random seeds (42, 123, 456, 789, 1024). Standard deviations are consistently small (<0.01 AUROC), confirming statistical stability across all conditions reported above.

**W4: Why not just use LLM Mismatch? (the key question)**

We agree this is the most important baseline and have now evaluated it comprehensively:

SST-2 Artifact-Aligned: ECG 0.819 ± 0.004 vs LLM Mismatch 0.628 ± 0.004 (+19 pp)
SST-2 Uniform: ECG 0.915 ± 0.003 vs LLM Mismatch 0.909 ± 0.003 (comparable)
MultiNLI (both noise): LLM Mismatch 0.883 vs ECG 0.557 (Mismatch wins)

On SST-2 artifact noise, where confidence-based methods catastrophically fail (Cleanlab: 0.136, AUM: 0.123, both below random), ECG outperforms LLM Mismatch by 19 percentage points. The explanation's *semantic content* (evidence, rationale, counterfactual) carries signal beyond binary agree/disagree. Concretely: two examples may both receive "positive" from the LLM (matching the given label), but their explanations cite contradictory evidence: one describes genuine positive sentiment, the other describes negative sentiment that was mislabeled positive with an artifact. kNN in explanation space catches this; prediction mismatch cannot.

On MultiNLI, LLM Mismatch wins because Qwen3-8B's ~88% NLI accuracy means prediction agreement already captures most detectable noise. When this is the case, explanation semantics add nothing further, but as shown above, this is not always the case.

The two signals are complementary and produced by the same LLM call at zero additional cost. Rather than choosing one, practitioners should use both: LLM Mismatch as a high-recall first pass, Explanation kNN to catch artifact-aligned noise that prediction agreement misses.

**Presentation (Line 355):** Thank you for catching this; we will move the AUROC definition to precede its first use in the ablation study in the camera-ready.
