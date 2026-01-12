# Transformer Section: Detailed Implementation Plan

**Version:** 1.1 (Verified)

## Overview

This plan covers the complete implementation of the Transformer (GPT) section following the established implementation principles and respecting the skeleton in `transformer.py`.

**Data:** Shakespeare text (`train_shakespeare.txt`, `test_shakespeare.txt`)
**Task:** Character-level next-token prediction (autoregressive language model)

### Verification Status: APPROVED
- All PDF requirements correctly implemented
- Skeleton bugs identified and fixed (variable shadowing, wrong context variable)
- Causal mask correctly applied (lower triangular)
- Top-k sampling correctly implemented

---

## Phase 1: Fill Transformer Skeleton in `transformer.py`

### 1.1 Implement `CausalSelfAttention.__init__()`

**Location:** `transformer.py:26-38`

**Implementation:**
```python
def __init__(self, n_head, n_embd, block_size):
    super().__init__()
    assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

    self.n_head = n_head
    self.n_embd = n_embd
    self.block_size = block_size
    self.head_dim = n_embd // n_head

    # Combined Q, K, V projection (more efficient than 3 separate layers)
    self.c_attn = nn.Linear(n_embd, 3 * n_embd)

    # Output projection
    self.c_proj = nn.Linear(n_embd, n_embd)

    # Causal mask: register as buffer (not a parameter, but moves with model)
    # Lower triangular matrix of ones
    mask = torch.tril(torch.ones(block_size, block_size))
    self.register_buffer('mask', mask.view(1, 1, block_size, block_size))
```

---

### 1.2 Implement `CausalSelfAttention.forward()`

**Location:** `transformer.py:40-57`

**Input:** `x` of shape `(B, T, n_embd)` where B=batch, T=sequence length
**Output:** `y` of shape `(B, T, n_embd)`

**Implementation:**
```python
def forward(self, x):
    B, T, C = x.size()  # batch, sequence length, embedding dim (n_embd)

    # 1. Compute Q, K, V projections
    qkv = self.c_attn(x)  # (B, T, 3*C)
    q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, C)

    # 2. Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_dim)
    q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
    k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
    v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

    # 3. Compute attention scores: (Q @ K^T) / sqrt(d_k)
    # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

    # 4. Apply causal mask: set future positions to -inf
    # mask is (1, 1, block_size, block_size), we only need (1, 1, T, T)
    att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

    # 5. Softmax over last dimension (the key dimension)
    att = torch.nn.functional.softmax(att, dim=-1)  # (B, n_head, T, T)

    # 6. Apply attention to values
    y = att @ v  # (B, n_head, T, head_dim)

    # 7. Re-assemble heads: (B, n_head, T, head_dim) -> (B, T, C)
    y = y.transpose(1, 2).contiguous().view(B, T, C)

    # 8. Output projection
    y = self.c_proj(y)

    return y
```

---

### 1.3 Implement Training Loop in `train_model()`

**Location:** `transformer.py:170-178`

**Implementation:**
```python
for ep in range(epochs):
    # ===== TRAINING =====
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {ep+1} Train')):
        x, y = batch  # x: input tokens, y: target tokens (shifted by 1)
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)  # (B, T, vocab_size)

        # Compute loss: flatten for cross-entropy
        # logits: (B, T, vocab_size) -> (B*T, vocab_size)
        # y: (B, T) -> (B*T,)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        train_loss += loss.item() * x.size(0)

        # Accuracy: check last position prediction
        # y[:, -1] is the target for the last position
        preds = logits[:, -1, :].argmax(dim=-1)  # (B,)
        targets = y[:, -1]  # (B,)
        train_correct += (preds == targets).sum().item()
        train_total += x.size(0)

    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total
```

---

### 1.4 Implement Evaluation Loop

**Location:** `transformer.py:176-178`

**Implementation:**
```python
    # ===== EVALUATION =====
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f'Epoch {ep+1} Test')):
            x, y = batch
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            test_loss += loss.item() * x.size(0)

            # Accuracy on last position
            preds = logits[:, -1, :].argmax(dim=-1)
            targets = y[:, -1]
            test_correct += (preds == targets).sum().item()
            test_total += x.size(0)

    avg_test_loss = test_loss / test_total
    test_acc = test_correct / test_total

    print(f'Epoch {ep+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, '
          f'Test Loss={avg_test_loss:.4f}, Test Acc={test_acc:.4f}')
```

---

### 1.5 Implement Standard Generation

**Location:** `transformer.py:181-188`

**Implementation:**
```python
        # ===== GENERATION (Standard Sampling) =====
        print("\n--- Standard Sampling ---")
        sentence = "the "
        for sent_idx in range(3):
            new_sentence = sentence
            for char_idx in range(30):  # Generate 30 characters
                # Encode the last block_size characters
                context = new_sentence[-block_size:] if len(new_sentence) >= block_size else new_sentence
                tokens = torch.tensor(data_handler.encoder(context), dtype=torch.long)[None].to(device)

                # Get model predictions
                logits = model(tokens)  # (1, T, vocab_size)

                # Get logits for the last position
                logits_last = logits[0, -1, :]  # (vocab_size,)

                # Convert to probabilities
                probs = torch.nn.functional.softmax(logits_last, dim=-1)

                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Decode and append
                next_char = data_handler.decoder([next_token])
                new_sentence += next_char

            print(f'Generated {sent_idx+1}: {new_sentence}')
```

---

### 1.6 Implement Top-k Generation

**Location:** `transformer.py:191-195`

**Implementation:**
```python
        # ===== GENERATION (Top-k Sampling) =====
        print("\n--- Top-k Sampling (k=5) ---")
        k = 5
        for sent_idx in range(3):
            new_sentence = sentence
            for char_idx in range(30):
                context = new_sentence[-block_size:] if len(new_sentence) >= block_size else new_sentence
                tokens = torch.tensor(data_handler.encoder(context), dtype=torch.long)[None].to(device)

                logits = model(tokens)
                logits_last = logits[0, -1, :]  # (vocab_size,)

                # Get top-k logits and indices
                topk_logits, topk_indices = torch.topk(logits_last, k)

                # Apply softmax only to top-k
                topk_probs = torch.nn.functional.softmax(topk_logits, dim=-1)

                # Sample from top-k distribution
                idx_in_topk = torch.multinomial(topk_probs, num_samples=1).item()
                next_token = topk_indices[idx_in_topk].item()

                next_char = data_handler.decoder([next_token])
                new_sentence += next_char

            print(f'Generated {sent_idx+1}: {new_sentence}')
```

---

## Phase 2: Create `TransformerTrainer` Class

**Location:** Inside `ex_5/transformer_solution.py`

```python
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DataHandler
from transformer import GPT
from helpers import set_seed

class TransformerTrainer:
    """
    Stateful trainer for GPT transformer with generation support.
    """

    def __init__(self,
                 train_path='train_shakespeare.txt',
                 test_path='test_shakespeare.txt',
                 block_size=10,
                 n_layer=3,
                 n_head=3,
                 n_embd=48,
                 lr=3e-4,
                 batch_size=64,
                 epochs=10,
                 seed=42,
                 verbose=True):

        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.num_epochs = epochs
        self.block_size = block_size

        # Data handling
        self.data_handler = DataHandler(train_path, test_path, block_size)
        self.vocab_size = self.data_handler.get_vocab_size()

        # Datasets and loaders
        trainset = self.data_handler.get_dataset('train')
        testset = self.data_handler.get_dataset('test')

        self.train_loader = DataLoader(
            trainset,
            sampler=torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=int(1e5)),
            shuffle=False,
            batch_size=batch_size,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            testset,
            sampler=torch.utils.data.RandomSampler(testset, replacement=False, num_samples=int(1e4)),
            shuffle=False,
            batch_size=batch_size,
            pin_memory=True
        ) if testset else None

        # Model
        self.model = GPT(n_layer, n_head, n_embd, self.vocab_size, block_size).to(self.device)

        # Optimization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        # History tracking
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
            'generated_sentences': []  # List of (epoch, [sentences])
        }

        self.is_trained = False

    def train(self):
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}', disable=not self.verbose)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits.view(-1, self.vocab_size), y.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * x.size(0)
                preds = logits[:, -1, :].argmax(dim=-1)
                train_correct += (preds == y[:, -1]).sum().item()
                train_total += x.size(0)

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Evaluation
            avg_test_loss, test_acc = self._evaluate()

            self.history['train_loss'].append(avg_train_loss)
            self.history['test_loss'].append(avg_test_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)

            # Generate sentences after each epoch (Q3)
            sentences = self.generate_sentences(n_sentences=3, max_chars=30)
            self.history['generated_sentences'].append((epoch + 1, sentences))

            if self.verbose:
                print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}, '
                      f'Test Loss={avg_test_loss:.4f}, Acc={test_acc:.4f}')
                for i, sent in enumerate(sentences):
                    print(f'  Generated {i+1}: {sent}')

        self.is_trained = True
        return self

    def _evaluate(self):
        if self.test_loader is None:
            return 0.0, 0.0

        self.model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits.view(-1, self.vocab_size), y.view(-1))

                test_loss += loss.item() * x.size(0)
                preds = logits[:, -1, :].argmax(dim=-1)
                test_correct += (preds == y[:, -1]).sum().item()
                test_total += x.size(0)

        return test_loss / test_total, test_correct / test_total

    def generate_sentences(self, n_sentences=3, max_chars=30, prompt="the ", top_k=None):
        """
        Generate sentences using the model.

        Args:
            n_sentences: Number of sentences to generate
            max_chars: Maximum characters to generate per sentence
            prompt: Starting prompt
            top_k: If provided, use top-k sampling; otherwise standard sampling

        Returns:
            List of generated sentences
        """
        self.model.eval()
        sentences = []

        with torch.no_grad():
            for _ in range(n_sentences):
                current = prompt

                for _ in range(max_chars):
                    # Get context (last block_size chars)
                    context = current[-self.block_size:]
                    tokens = torch.tensor(
                        self.data_handler.encoder(context), dtype=torch.long
                    )[None].to(self.device)

                    logits = self.model(tokens)
                    logits_last = logits[0, -1, :]

                    if top_k is not None:
                        # Top-k sampling
                        topk_logits, topk_indices = torch.topk(logits_last, top_k)
                        probs = torch.softmax(topk_logits, dim=-1)
                        idx = torch.multinomial(probs, 1).item()
                        next_token = topk_indices[idx].item()
                    else:
                        # Standard sampling
                        probs = torch.softmax(logits_last, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()

                    next_char = self.data_handler.decoder([next_token])
                    current += next_char

                sentences.append(current)

        return sentences

    def get_history(self):
        return self.history
```

---

## Phase 3: Create Experiment Classes

**Location:** Inside `ex_5/transformer_solution.py`

### 3.1 Standard Training Experiment (Q2, Q3)

```python
class Transformer_Q2_Q3_Experiment:
    """
    Q2: Train transformer and plot accuracy/loss curves.
    Q3: Generate sentences after each epoch.
    """

    def __init__(self, epochs=10, seed=42, verbose=False):
        self.epochs = epochs
        self.seed = seed
        self.verbose = verbose
        self.trainer = None
        self.is_trained = False

    def run(self):
        if self.verbose:
            print("Training Transformer with standard sampling...")

        self.trainer = TransformerTrainer(
            epochs=self.epochs,
            seed=self.seed,
            verbose=self.verbose
        )
        self.trainer.train()
        self.is_trained = True
        return self

    def get_trainer(self):
        return self.trainer
```

### 3.2 Top-k Sampling Experiment (Q4)

```python
class Transformer_Q4_Experiment:
    """
    Q4: Train transformer and generate with top-k sampling.
    Reuses the trained model from Q2/Q3.
    """

    def __init__(self, base_experiment, k=5, verbose=False):
        self.base_exp = base_experiment
        self.k = k
        self.verbose = verbose
        self.topk_sentences_per_epoch = []
        self.is_run = False

    def run(self):
        if not self.base_exp.is_trained:
            raise ValueError("Base experiment must be trained first")

        trainer = self.base_exp.get_trainer()

        if self.verbose:
            print(f"Generating sentences with top-k sampling (k={self.k})...")

        # Generate 3 sentences with top-k for comparison
        self.final_topk_sentences = trainer.generate_sentences(
            n_sentences=3, max_chars=30, top_k=self.k
        )

        self.is_run = True
        return self

    def get_topk_sentences(self):
        return self.final_topk_sentences
```

---

## Phase 4: Create Results Classes

**Location:** Inside `ex_5/transformer_solution.py`

```python
import matplotlib.pyplot as plt

class TransformerResults:
    """Visualization for Transformer Q2/Q3."""

    def __init__(self, experiment):
        self.exp = experiment

    def plot_loss_curves(self, figsize=(10, 5)):
        """Q2: Plot train/test loss vs epoch."""
        if not self.exp.is_trained:
            print("Experiment not run yet.")
            return

        history = self.exp.get_trainer().get_history()
        epochs = range(1, len(history['train_loss']) + 1)

        plt.figure(figsize=figsize)
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs, history['test_loss'], label='Test Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Transformer Training: Loss vs Epoch')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_accuracy_curves(self, figsize=(10, 5)):
        """Q2: Plot train/test accuracy vs epoch."""
        if not self.exp.is_trained:
            return

        history = self.exp.get_trainer().get_history()
        epochs = range(1, len(history['train_acc']) + 1)

        plt.figure(figsize=figsize)
        plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
        plt.plot(epochs, history['test_acc'], label='Test Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (Last Position)')
        plt.title('Transformer Training: Accuracy vs Epoch')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def display_generated_sentences(self):
        """Q3: Display generated sentences per epoch."""
        if not self.exp.is_trained:
            return

        history = self.exp.get_trainer().get_history()

        print("=" * 60)
        print("Generated Sentences per Epoch (Standard Sampling)")
        print("=" * 60)

        for epoch, sentences in history['generated_sentences']:
            print(f"\nEpoch {epoch}:")
            for i, sent in enumerate(sentences, 1):
                print(f"  {i}. {sent}")

        print("=" * 60)


class TopKResults:
    """Visualization for Top-k sampling comparison (Q4)."""

    def __init__(self, base_experiment, topk_experiment):
        self.base_exp = base_experiment
        self.topk_exp = topk_experiment

    def compare_generations(self):
        """Q4: Compare standard vs top-k sampling."""
        if not self.base_exp.is_trained or not self.topk_exp.is_run:
            print("Experiments not run yet.")
            return

        # Get final epoch's standard sampling sentences
        history = self.base_exp.get_trainer().get_history()
        last_epoch, standard_sentences = history['generated_sentences'][-1]

        # Get top-k sentences
        topk_sentences = self.topk_exp.get_topk_sentences()

        print("=" * 60)
        print("Q4: Generation Comparison - Standard vs Top-k Sampling")
        print("=" * 60)

        print(f"\nStandard Sampling (Epoch {last_epoch}):")
        for i, sent in enumerate(standard_sentences, 1):
            print(f"  {i}. {sent}")

        print(f"\nTop-k Sampling (k={self.topk_exp.k}):")
        for i, sent in enumerate(topk_sentences, 1):
            print(f"  {i}. {sent}")

        print("\n" + "=" * 60)
        print("Analysis:")
        print("- Top-k sampling reduces noise by only sampling from most likely tokens")
        print("- This typically produces more coherent but less diverse text")
        print("- Standard sampling can produce unexpected/creative outputs")
        print("=" * 60)
```

---

## Phase 5: Notebook Structure

**File:** `ex_5/ex5_transformer.ipynb`

```python
# Cell 1: Setup
import numpy as np
import torch
from helpers import set_seed
from transformer_solution import (
    TransformerTrainer,
    Transformer_Q2_Q3_Experiment,
    Transformer_Q4_Experiment,
    TransformerResults,
    TopKResults
)

set_seed(42)
torch.manual_seed(42)

# Cell 2: Q2/Q3 - Train Transformer
print("Training Transformer model...")
exp_q2q3 = Transformer_Q2_Q3_Experiment(epochs=10, verbose=True)
exp_q2q3.run()

results = TransformerResults(exp_q2q3)

# Cell 3: Q2 - Loss curves
results.plot_loss_curves()

# Cell 4: Q2 - Accuracy curves
results.plot_accuracy_curves()

# Cell 5: Q3 - Generated sentences
results.display_generated_sentences()

# Cell 6: [MARKDOWN] Q2/Q3 Analysis
"""
### Q2/Q3 Analysis
- Loss decreases over epochs, indicating the model is learning
- Accuracy improves as the model better predicts next characters
- Generated sentences become more coherent as training progresses
- Early epochs produce mostly random characters
- Later epochs show recognizable word patterns and Shakespeare-like text
"""

# Cell 7: Q4 - Top-k Sampling
exp_q4 = Transformer_Q4_Experiment(exp_q2q3, k=5, verbose=True)
exp_q4.run()

topk_results = TopKResults(exp_q2q3, exp_q4)
topk_results.compare_generations()

# Cell 8: [MARKDOWN] Q4 Analysis
"""
### Q4 Analysis: Standard vs Top-k Sampling
- **Standard sampling**: Higher diversity, occasional nonsense characters
- **Top-k sampling (k=5)**: More coherent words, less randomness
- Top-k restricts choices to most likely tokens, reducing noise
- Trade-off: coherence vs creativity
"""
```

---

## Phase 6: File Organization

```
ex_5/
├── dataset.py                     # PROVIDED - ShakespeareDataset, DataHandler
├── transformer.py                 # FILL SKELETON - CausalSelfAttention
├── helpers.py                     # SHARED - set_seed (from GMM section)
├── transformer_solution.py        # NEW - Trainer + Experiments + Results
├── ex5_transformer.ipynb          # NEW - Notebook
├── train_shakespeare.txt          # PROVIDED - training data
└── test_shakespeare.txt           # PROVIDED - test data
```

---

## Implementation Checklist

### Skeleton Completion (`transformer.py`)
- [ ] `CausalSelfAttention.__init__()` - define c_attn, c_proj, register mask buffer
- [ ] `CausalSelfAttention.forward()` - full multi-head causal attention
- [ ] Training loop in `train_model()` - forward, loss, backward
- [ ] Standard generation code - multinomial sampling
- [ ] Top-k generation code - top-k then multinomial

### New Files
- [ ] `transformer_solution.py` - consolidated file containing:
  - TransformerTrainer (with generation methods)
  - Transformer_Q2_Q3_Experiment
  - Transformer_Q4_Experiment
  - TransformerResults
  - TopKResults

### Notebook Deliverables
- [ ] Q2: Train/test loss curves
- [ ] Q2: Train/test accuracy curves
- [ ] Q3: Generated sentences per epoch (3 sentences, 30 chars each)
- [ ] Q4: Top-k sampling comparison (k=5)

---

## Key Formulas Reference

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + Mask) @ V
```

**Causal Mask:**
```
Mask[i,j] = 0      if j <= i  (can attend)
          = -inf   if j > i   (cannot attend to future)
```

**Multi-Head Attention:**
```
1. Project: Q, K, V = Linear(X)
2. Split into heads: (B, T, C) -> (B, n_head, T, head_dim)
3. Attention per head
4. Concatenate heads: (B, n_head, T, head_dim) -> (B, T, C)
5. Output projection
```

**Top-k Sampling:**
```
1. Get logits for last position
2. Select top k logits and their indices
3. Apply softmax to top-k logits only
4. Sample from this restricted distribution
```

---

## Default Hyperparameters

```python
block_size = 10       # Context window size
n_layer = 3           # Number of transformer blocks
n_head = 3            # Number of attention heads
n_embd = 48           # Embedding dimension (must be divisible by n_head)
learning_rate = 3e-4  # Adam learning rate
batch_size = 64       # Batch size
epochs = 10           # Number of epochs

# Generation
prompt = "the "       # Starting prompt
max_chars = 30        # Characters to generate (PDF says 30, skeleton says 20 - use 30)
top_k = 5             # For top-k sampling
```

---

## Skeleton Bugs to Fix

The skeleton has bugs in the generation loop that must be corrected:

### Bug 1: Variable Shadowing
```python
# SKELETON BUG (lines 183-186):
for i in range(3):
    new_sentence = sentence
    for i in range(20):  # BUG: 'i' shadows outer loop variable

# CORRECT:
for sent_idx in range(3):
    new_sentence = sentence
    for char_idx in range(30):  # Use different variable names
```

### Bug 2: Wrong Context Variable
```python
# SKELETON BUG:
tokens = torch.tensor(data_handler.encoder(sentence[-block_size:]))[None]
#                                          ^^^^^^^^ uses original prompt, not growing sentence

# CORRECT:
tokens = torch.tensor(data_handler.encoder(new_sentence[-block_size:]))[None]
#                                          ^^^^^^^^^^^^ use the growing sentence
```

### Bug 3: Missing Device Placement
```python
# SKELETON (implicit - doesn't move to device):
tokens = torch.tensor(...)

# CORRECT:
tokens = torch.tensor(..., dtype=torch.long).to(device)
```

---

## Accuracy Definition (from PDF)

The accuracy metric is:
```
Accuracy = (# sentences where model correctly predicted last character) / (total sentences)
```

In code:
```python
preds = logits[:, -1, :].argmax(dim=-1)  # Prediction for last position
targets = y[:, -1]                        # Target at last position
correct = (preds == targets).sum().item()
```

---

## Verification Summary

This plan has been verified by a review agent.

### Verification Results

| Component | Status | Notes |
|-----------|--------|-------|
| CausalSelfAttention.__init__ | CORRECT | Q/K/V projection + mask buffer |
| CausalSelfAttention.forward | CORRECT | Multi-head attention with causal mask |
| Causal Mask (lower triangular) | CORRECT | `torch.tril` + `masked_fill(-inf)` |
| Training Loop | CORRECT | Forward, loss, backward |
| Accuracy Computation | CORRECT | Last-position only |
| Standard Generation | CORRECT | Multinomial sampling |
| Top-k Generation | CORRECT | Restrict to top-k then sample |
| Skeleton Bug Fixes | IDENTIFIED | Variable shadowing, wrong context, device |

### Key Implementation Notes

1. **Always call `model.eval()`** before generation
2. **Move tokens to device** with `.to(device)`
3. **Use 30 characters** for generation (not 20 as skeleton shows)
4. **Use `new_sentence`** for context, not the original `sentence`

### Simplification Note

For minimal implementation, **only Phase 1 (skeleton fill-in) is required**. Phases 2-5 (Trainer/Experiment/Results classes) provide nice organization but are optional for the assignment.

**Status: READY FOR IMPLEMENTATION**
