# Exercise 5: Mixture Models & Transformers - Implementation Guide

## Overview
This exercise covers three main independent sections:
1. **Gaussian Mixture Model (GMM)** - Probabilistic modeling with Gaussians
2. **Uniform Mixture Model (UMM)** - Probabilistic modeling with Uniforms
3. **Transformer (GPT2)** - Autoregressive language modeling

**Report Requirements:** Max 8 pages PDF, explanations required for all results.

---

## Section 1: Gaussian Mixture Model (GMM)

**File:** `mixture_models.py` (GMM class)
**Data:** Countries of Europe dataset (normalized to mean=0, std=1)

### 1.1 Implementation Tasks
- [ ] Initialize `self.logits`, `self.means`, `self.log_variances`
- [ ] Implement `forward()` - compute log-likelihood using `torch.logsumexp`
- [ ] Implement `loss()` - return negative mean log-likelihood
- [ ] Implement `sample(n)` - sample from GMM using `torch.multinomial`
- [ ] Implement `conditional_sample(k, n)` - sample from k-th Gaussian

### 1.2 Experiments (Question 1)
For `n_components = [1, 5, 10, n_classes]`:
- [ ] (a) Scatter plot: 1000 samples from GMM (use `sample`)
- [ ] (b) Scatter plot: 100 samples per Gaussian, colored by component (use `conditional_sample`)

### 1.3 Experiments (Question 2)
For `n_components = n_classes`:
- [ ] (a) For epochs [1, 10, 20, 30, 40, 50]: display both scatter plots
- [ ] (b) Plot train/test mean log-likelihood vs epoch
- [ ] (c) Repeat (a+b) with means initialized to country centroids; compare results

**Key Implementation Notes:**
- Use `nn.functional.log_softmax` for log p(k)
- Use `torch.logsumexp` to avoid numerical underflow
- Optimize logits (not probabilities) and log-variances (not variances)

---

## Section 2: Uniform Mixture Model (UMM)

**File:** `mixture_models.py` (UMM class)
**Data:** Countries of Europe dataset (normalized)

### 2.1 Implementation Tasks
- [ ] Initialize `self.logits`, `self.centers`, `self.log_sizes`
- [ ] Implement `forward()` - compute log-likelihood with uniform bounds check
- [ ] Implement `loss()` - return negative mean log-likelihood
- [ ] Implement `sample(n)` - sample from UMM
- [ ] Implement `conditional_sample(k, n)` - sample from k-th Uniform

### 2.2 Experiments (Question 1)
Repeat same experiments as GMM Section 1.2 and 1.3:
- [ ] Vary n_components = [1, 5, 10, n_classes]
- [ ] Epoch progression plots
- [ ] Train/test log-likelihood curves
- [ ] Centroid initialization comparison

### 2.3 Analysis (Question 2)
- [ ] Analyze trends in uniform support across epochs
- [ ] Analyze trends in uniform centers across epochs
- [ ] Explain gradient descent problems unique to UMM vs GMM

**Key Implementation Notes:**
- Use `-1e6` instead of `-inf` for out-of-bounds log-probabilities (avoid NaN)
- Use `torch.distributions.Uniform` for sampling
- Ensure sizes remain positive via log parameterization

---

## Section 3: Transformer (Causal Self-Attention GPT)

**File:** `transformer.py`
**Data:** Shakespeare text dataset (`train_shakespeare.txt`, `test_shakespeare.txt`)

### 3.1 Implementation Tasks
- [ ] Implement `CausalSelfAttention.forward()`:
  - Key, Query, Value projections
  - Split into multiple heads
  - Scaled dot-product attention with causal mask
  - Re-arrange heads and output projection

### 3.2 Training & Evaluation (Questions 2-3)
- [ ] Train with default hyperparameters
- [ ] Plot train/test accuracy vs epoch
- [ ] Plot train/test loss vs epoch
- [ ] Generate 3 sentences per epoch (starting with "the ", 30 chars)

### 3.3 Top-k Sampling (Question 4)
- [ ] Implement/use Top-k sampling with k=5
- [ ] Repeat training and generation experiments
- [ ] Compare generation quality with and without Top-k

**Key Implementation Notes:**
- Causal mask: lower triangular matrix (0 for j <= i, -inf otherwise)
- Attention: `Softmax((Q @ K.T) / sqrt(d_h) + Mask) @ V`
- Use `torch.multinomial` for sampling during generation
- Accuracy = ratio of correct last-character predictions

---

## Dependencies Between Sections

```
Section 1 (GMM)     Section 2 (UMM)     Section 3 (Transformer)
      |                   |                      |
      v                   v                      v
 Independent         Independent            Independent
 (same data)         (same data)          (Shakespeare data)
```

**All three sections are independent and can be completed in any order.**

---

## Seeding Requirements
```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
```

---

## Submission Checklist
- [ ] `mixture_models.py` - GMM and UMM implementations
- [ ] `transformer.py` - CausalSelfAttention implementation
- [ ] `ex5_{YOUR_ID}.pdf` - Report (max 8 pages)
- [ ] `README` - Name, CSE username, ID
- [ ] All `.py` and `.ipynb` files
- [ ] Package as `ex5_{YOUR_ID}.zip`

---

## Quick Reference: Mathematical Formulas

### GMM Log-Likelihood
```
log p(x) = logsumexp_k [log π_k + log N(x|μ_k, Σ_k)]

log N(x|k) = -log(2π) - log(σ_k1) - log(σ_k2)
             - 0.5 * [(x1-μ_k1)²/σ²_k1 + (x2-μ_k2)²/σ²_k2]
```

### UMM Log-Likelihood
```
log p(x|k) = -log(s1) - log(s2)    if x in bounds
           = -1e6                   otherwise

Bounds: c - s/2 <= x <= c + s/2
```

### Causal Self-Attention
```
Attention(Q,K,V) = Softmax(Q @ K.T / sqrt(d_h) + Mask) @ V

Mask[i,j] = 0      if j <= i
          = -inf   otherwise
```
