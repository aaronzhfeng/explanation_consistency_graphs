# Refinement Prompt: Dir 4 Related Work Deep Dive

## Context

We are implementing "Direction 4: Explanation-consistency graphs for training-data debugging" for ACL 2026.

**Core idea:** Use explanation structure (not just loss/confidence) to identify mislabeled or artifact-laden training data.

## Questions

1. **Data cleaning / label-noise detection methods:** What are the major approaches?
   - **Loss-based:** confident learning, Cleanlab, loss trajectory
   - **Influence-based:** influence functions, TracIn, arnoldi approximations
   - **Ensemble-based:** disagreement signals
   - **LLM-based:** using LLM confidence or LLM-as-judge
   - For each: strengths, weaknesses, compute cost

2. **Explanation-based debugging:** What prior work uses explanations to find data problems?
   - The TACL survey on explanation-based human debugging
   - Interactive label cleaning with explanations
   - Using saliency to find spurious correlations
   - Any work specifically using LLM-generated explanations?

3. **Spurious correlation detection:** What methods exist?
   - Interpretability-based discovery (ACL paper)
   - Slice discovery methods
   - Counterfactual-based detection
   - Dataset cartography

4. **Graph-based methods for data quality:** Is there prior work using graphs?
   - Representation-similarity graphs for data selection
   - Clustering-based outlier detection
   - kNN-based noise detection
   - Any work using *explanation* embeddings specifically?

5. **NLI for consistency checking:** Has NLI been used to verify explanation quality?
   - Explanation entailment checks
   - Claim verification approaches
   - Self-consistency in CoT

6. **Novelty gap:** Based on this literature, what's the clearest novelty claim?
   - "Explanation-consistency graphs as core primitive" — is this truly new?
   - "Graph aggregation across explanations" vs "single-instance debugging"
   - Any direct competitors we must address?

7. **Positioning statement:** Write a 2–3 sentence novelty claim that is defensible against this literature.

