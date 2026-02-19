---
source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/d95cb79a3421e6d9b6c9a9008c4d07c5-Abstract-Conference.html
openreview: https://openreview.net/forum?id=VRRvJnxgQe
github: https://github.com/drunkerWang/NoiseGPT
title: "NoiseGPT: Label Noise Detection and Rectification through Probability Curvature"
venue: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024)"
note: "No arXiv preprint available. Content compiled from NeurIPS proceedings, OpenReview, and GitHub."
---

# NoiseGPT: Label Noise Detection and Rectification through Probability Curvature

**Authors:** Haoyu Wang, Zhuo Huang, Zhiwei Lin, Tongliang Liu

**Venue:** NeurIPS 2024 (Poster), Advances in Neural Information Processing Systems 37, pp. 120159-120183

**License:** CC BY 4.0

**Keywords:** Label noise, Multimodal Learning, Large Language Models

---

## Abstract

Label noise where image data mismatches with incorrect label exists ubiquitously in all kinds of datasets, which significantly degrades neural network performance while consuming substantial human labeling resources. Rather than relying on memorization effects or restrictive assumptions, the authors introduce NoiseGPT, which leverages multimodal large language models (MLLMs) as knowledge experts for detecting and correcting mislabeled data. The paper identifies a "probability curvature" effect in MLLMs where clean and noisy examples reside on curvatures with different smoothness, enabling the detection of label noise. By designing a token-wise Mix-of-Feature (MoF) technique to produce the curvature, they propose an In-Context Discrepancy (ICD) measure to determine the authenticity of an image-label pair. Through extensive experiments, the effectiveness of NoiseGPT on detecting and cleansing dataset noise is demonstrated.

---

## 1. Introduction

The paper addresses the challenge of label noise in datasets used for training deep neural networks. The authors argue that existing methods relying on the memorization effect (where DNNs learn clean patterns first, then overfit to noisy labels) have fundamental limitations when noise is structured or instance-dependent. NoiseGPT takes a different approach by leveraging the broad world knowledge encoded in multimodal large language models.

---

## 2. Core Methodology

### 2.1 Probability Curvature Effect

The central observation is that when an MLLM processes an image-label pair, the probability distribution over output tokens exhibits different smoothness characteristics for clean vs. noisy examples. Clean examples produce smooth probability curves, while noisy (mislabeled) examples produce more irregular, less smooth curves. This is the "probability curvature" effect.

### 2.2 Token-wise Mix-of-Feature (MoF)

To operationalize the probability curvature observation, the authors design a Mix-of-Feature technique that operates at the visual token level. MoF perturbs the visual features fed to the MLLM and measures how the output probability distribution changes under perturbation. The key idea is that the MLLM's confidence stability under visual feature perturbation differs between correctly and incorrectly labeled examples.

Specifically, MoF:
- Extracts visual token representations from the MLLM's vision encoder
- Applies controlled perturbations to these visual features
- Measures the resulting change in the MLLM's output probability distribution
- The perturbation operates on visual tokens, which has no direct text analogue

### 2.3 In-Context Discrepancy (ICD) Measure

The ICD metric quantifies the authenticity of an image-label pair by measuring whether the MLLM is "stably confident" about the pairing. It captures the curvature of the probability landscape:
- Low ICD (smooth curvature) → likely clean label
- High ICD (irregular curvature) → likely noisy label

This is fundamentally a confidence-curvature measure that assesses prediction stability rather than explanation semantics.

### 2.4 Iterative Label Rectification

Beyond detection, NoiseGPT includes an iterative refinement process that:
1. Detects noisy labels using ICD scores
2. Proposes corrected labels based on MLLM predictions
3. Repeats the detection process to identify optimal label corrections

---

## 3. Experimental Setup

### 3.1 Datasets

NoiseGPT is designed for and evaluated exclusively on **image classification** tasks:
- **CIFAR-10** (10 classes, 50k training images)
- **CIFAR-100** (100 classes, 50k training images)
- **WebVision** (real-world noisy web-crawled images)
- **ILSVRC12 / ImageNet** (1000 classes, 1.2M images)

No text/NLP experiments are reported.

### 3.2 Noise Types

- Symmetric noise (uniform random label flipping)
- Asymmetric noise (class-conditional flipping)
- Instance-dependent noise
- Real-world noise (WebVision)

### 3.3 Computational Cost

NoiseGPT requires **67-130 GPU-hours per dataset** on an RTX 4090 (reported in Table 7 of the paper). This is due to the need to run MLLM inference with perturbations on every training example. The authors **report no error bars** due to these computational constraints.

For comparison:
- ECG full pipeline: ~25 minutes, ~$8-10 in API costs
- Cleanlab (5-fold CV): ~48 minutes on GPU
- AUM (3 epochs): ~16 minutes on GPU

---

## 4. Key Results

### 4.1 Noise Detection (AUROC)

- **ILSVRC12:** AUROC exceeding 0.92
- **CIFAR-10 (symmetric noise):** Strong detection performance across noise rates

### 4.2 Downstream Classification Improvement

- **CIFAR-10 (80% symmetric noise):** 22.8% improvement with M-correction integration
- Shows effectiveness across multiple datasets without requiring dataset-specific assumptions

### 4.3 Results from DynaCor (Kim et al.) Comparison Context

In DynaCor's own evaluation (their Table 2), AUM drops to F1=16.7% on real-world noise with ResNet34, suggesting that training-dynamics-based methods struggle with structured noise. NoiseGPT addresses this via MLLM knowledge but at substantially higher computational cost.

---

## 5. Limitations and Relevance to ECG

### 5.1 Vision-Only

NoiseGPT is designed exclusively for image classification. The Mix-of-Feature perturbation technique operates on visual tokens from the vision encoder, with no direct text analogue. Adapting it to NLP would require fundamental architectural changes.

### 5.2 Computational Cost

At 67-130 GPU-hours per dataset on an RTX 4090, NoiseGPT is orders of magnitude more expensive than ECG (~25 minutes, ~$8-10 API cost) or traditional baselines.

### 5.3 Signal Type: Confidence Stability vs. Explanation Semantics

NoiseGPT's signal (whether the MLLM is "stably confident" under perturbation) is a confidence-curvature measure. This shares the same vulnerability to artifact-aligned noise as other confidence-based methods: if the MLLM is stably confident about an artifact-correlated mislabeled example, the ICD score will incorrectly indicate a clean label.

ECG's signal is orthogonal: it derives from explanation *semantics* (evidence, rationale, counterfactual reasoning), not from confidence trajectories or prediction stability.

### 5.4 Missing Baselines

Notably, NoiseGPT never tests a simple prediction-mismatch baseline (i.e., just checking if the MLLM's predicted label disagrees with the given label). This leaves open the question of how much their probability-curvature method adds over zero-shot classification.

---

## 6. Source Code

Available at: https://github.com/drunkerWang/NoiseGPT

Repository contains:
- Training scripts (`train_resnet.py`)
- Inference pipeline (`inference.py`)
- Mix-of-Feature implementation (`mixup.py`)
- Prompt generation (`get_prompts.py`)
- Dataset configurations for CIFAR, ImageNet, DomainBed
- Languages: Python (96.1%), TeX (3.9%)
- License: MIT

---

## References

Wang, H., Huang, Z., Lin, Z., & Liu, T. (2024). NoiseGPT: Label Noise Detection and Rectification through Probability Curvature. *Advances in Neural Information Processing Systems*, 37, 120159-120183.
