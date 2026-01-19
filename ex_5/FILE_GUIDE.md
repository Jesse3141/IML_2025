# Exercise 5: File Guide & Code Reference

This document describes all provided Python files, their functions, and what code needs to be implemented.

---

## 1. `dataset.py` - Data Handling (COMPLETE - No Implementation Needed)

### Class: `EuropeDataset`
**Purpose:** PyTorch Dataset for the Countries of Europe data (used in GMM/UMM sections).

| Method | Description |
|--------|-------------|
| `__init__(csv_file)` | Loads CSV, extracts features (cols 1-2) and labels (col 3) as tensors |
| `__len__()` | Returns total number of samples |
| `__getitem__(idx)` | Returns `[features, label]` for index idx |

**Data Format:**
- `features`: `torch.Tensor` of shape `(n_samples, 2)` - 2D coordinates
- `labels`: `torch.Tensor` of shape `(n_samples,)` - country class labels

---

### Class: `ShakespeareDataset`
**Purpose:** PyTorch Dataset for Shakespeare text (used in Transformer section).

| Method | Description |
|--------|-------------|
| `__init__(text, block_size, encoder)` | Encodes text to token array |
| `__len__()` | Returns `len(data) - block_size` |
| `__getitem__(idx)` | Returns `(x, y)` where `x = tokens[idx:idx+block_size]`, `y = tokens[idx+1:idx+1+block_size]` |

**Note:** Each sample is an input-target pair for next-character prediction.

---

### Class: `DataHandler`
**Purpose:** Manages Shakespeare data loading, vocabulary, encoding/decoding.

| Method/Attribute | Description |
|------------------|-------------|
| `__init__(train_filename, test_filename, block_size)` | Loads text files, builds vocabulary and datasets |
| `vocab` | List of all unique characters in training text |
| `encoder(string)` | Converts string → list of token integers |
| `decoder(list)` | Converts list of token integers → string |
| `train_dataset` | `ShakespeareDataset` for training |
| `test_dataset` | `ShakespeareDataset` for testing (or None) |
| `get_vocab_size()` | Returns vocabulary size |
| `get_dataset(mode)` | Returns train or test dataset |

---

## 2. `mixture_models.py` - Mixture Models

### Provided Utility Function

```python
def normalize_tensor(tensor, d):
    """Normalize tensor along axis d to mean=0, std=1"""
```

---

### Class: `GMM` (Gaussian Mixture Model)

#### Provided (Already Implemented):
```python
def __init__(self, n_components):
    self.n_components = n_components
    self.weights = nn.Parameter(torch.randn(n_components))           # logits for π_k
    self.means = nn.Parameter(torch.randn(n_components, 2))          # μ_k (K x 2)
    self.log_variances = nn.Parameter(torch.zeros(n_components, 2))  # log(σ²_k) (K x 2)
```

#### TO IMPLEMENT (Skeleton):

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `forward(X)` | `(n_samples, 2)` | `(n_samples,)` | Compute log p(x) for each sample |
| `loss_function(log_likelihood)` | `(n_samples,)` | scalar | Return negative mean log-likelihood |
| `sample(n_samples)` | int | `(n_samples, 2)` | Sample from the GMM |
| `conditional_sample(n_samples, label)` | int, int | `(n_samples, 2)` | Sample from k-th Gaussian only |

#### Implementation Hints:
```python
# forward() pseudocode:
log_pi = F.log_softmax(self.weights, dim=0)           # (K,)
variances = torch.exp(self.log_variances)              # (K, 2)
# For each component k, compute log N(x|μ_k, σ_k)
# log p(x|k) = -log(2π) - log(σ_k1) - log(σ_k2) - 0.5*[(x1-μ_k1)²/σ²_k1 + (x2-μ_k2)²/σ²_k2]
# Use torch.logsumexp(log_pi + log_p_x_given_k, dim=...)
```

---

### Class: `UMM` (Uniform Mixture Model)

#### Provided (Already Implemented):
```python
def __init__(self, n_components):
    self.n_components = n_components
    self.weights = nn.Parameter(torch.randn(n_components))                              # logits
    self.centers = nn.Parameter(torch.randn(n_components, 2))                           # c_k (K x 2)
    self.log_sizes = nn.Parameter(torch.log(torch.ones(n_components, 2) + torch.rand(n_components, 2)*0.2))  # log(s_k)
```

#### TO IMPLEMENT (Skeleton):

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `forward(X)` | `(n_samples, 2)` | `(n_samples,)` | Compute log p(x) for each sample |
| `loss_function(log_likelihood)` | `(n_samples,)` | scalar | Return negative mean log-likelihood |
| `sample(n_samples)` | int | `(n_samples, 2)` | Sample from the UMM |
| `conditional_sample(n_samples, label)` | int, int | `(n_samples, 2)` | Sample from k-th Uniform only |

#### Implementation Hints:
```python
# forward() pseudocode:
sizes = torch.exp(self.log_sizes)                      # (K, 2)
half_sizes = sizes / 2
lower = self.centers - half_sizes                      # (K, 2)
upper = self.centers + half_sizes                      # (K, 2)

# For each sample x and component k:
# in_bounds = (x >= lower_k) & (x <= upper_k) for both dimensions
# log p(x|k) = -log(s_k1) - log(s_k2)  if in_bounds else -1e6

# Use torch.logsumexp for final log p(x)
```

---

### Main Block (Provided Setup):
```python
if __name__ == "__main__":
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    batch_size = 4096
    num_epochs = 50
    # learning_rate for GMM = 0.01
    # learning_rate for UMM = 0.001

    # Normalization (REQUIRED)
    train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    test_dataset.features = normalize_tensor(test_dataset.features, d=0)

    # DataLoaders created...
    #### YOUR CODE GOES HERE #### (training loop)
```

---

## 3. `transformer.py` - Transformer Model

### Provided Utility Class

```python
class NewGELU(nn.Module):
    """GELU activation function (same as OpenAI GPT)"""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x**3)))
```

---

### Class: `CausalSelfAttention`

#### Provided (Already Implemented):
```python
def __init__(self, n_head, n_embd, block_size):
    self.n_head = n_head
    self.n_embd = n_embd
    self.block_size = block_size
    #### YOUR CODE HERE ####
    # Suggested: self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # Q, K, V projection
    #            self.c_proj = nn.Linear(n_embd, n_embd)       # output projection
```

#### TO IMPLEMENT:

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `__init__` (complete) | - | - | Initialize linear layers for Q/K/V and output projection |
| `forward(x)` | `(B, T, n_embd)` | `(B, T, n_embd)` | Full causal self-attention |

#### Implementation Steps for `forward(x)`:
```python
# 1. Project to Q, K, V
B, T, C = x.size()
qkv = self.c_attn(x)                    # (B, T, 3*C)
q, k, v = qkv.split(self.n_embd, dim=2) # each (B, T, C)

# 2. Reshape for multi-head: (B, T, C) -> (B, n_head, T, head_dim)
head_dim = C // self.n_head
q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

# 3. Attention scores: (B, n_head, T, T)
att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

# 4. Causal mask: upper triangle = -inf
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
att = att.masked_fill(mask, float('-inf'))

# 5. Softmax + apply to values
att = F.softmax(att, dim=-1)
y = att @ v                             # (B, n_head, T, head_dim)

# 6. Re-assemble heads: (B, T, C)
y = y.transpose(1, 2).contiguous().view(B, T, C)

# 7. Output projection
return self.c_proj(y)
```

---

### Class: `Block` (COMPLETE - No Implementation Needed)

```python
class Block(nn.Module):
    """Standard Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP"""
    def __init__(self, n_head, n_embd, block_size):
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(...)  # 4x expansion MLP with GELU

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # Residual + attention
        x = x + self.mlpf(self.ln_2(x))   # Residual + MLP
        return x
```

---

### Class: `GPT` (COMPLETE - No Implementation Needed)

```python
class GPT(nn.Module):
    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size):
        self.transformer = nn.ModuleDict(
            wte = nn.Embedding(vocab_size, n_embd),    # Token embeddings
            wpe = nn.Embedding(block_size, n_embd),   # Position embeddings
            h = nn.ModuleList([Block(...) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        # idx: (B, T) token indices
        # Returns: (B, T, vocab_size) logits for next token
```

---

### Function: `train_model` (Partial - Training Loop Needed)

#### Provided:
```python
def train_model(train_path, test_path=None, model=None,
                block_size=10, n_layer=3, n_head=3, n_embd=48,
                learning_rate=3e-4, batch_size=64, epochs=10):

    data_handler = DataHandler(train_path, test_path, block_size)
    model = GPT(n_layer, n_head, n_embd, vocab_size, block_size)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    # DataLoaders with sampling (1e5 train, 1e4 test samples per epoch)
    ...
```

#### TO IMPLEMENT:

| Location | Description |
|----------|-------------|
| Training loop (line 173) | Forward pass, loss computation, backprop |
| Generation (line 187) | Generate next character, append to sentence |
| Top-k generation (line 195) | Generate with top-k sampling |

#### Training Loop Pseudocode:
```python
for ep in range(epochs):
    model.train()
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        logits = model(x)                           # (B, T, vocab_size)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Generation Pseudocode:
```python
# Standard sampling:
logits = model(tokens.to(device))
probs = F.softmax(logits[0, -1], dim=-1)            # last position
next_token = torch.multinomial(probs, 1)
new_sentence += data_handler.decoder([next_token.item()])

# Top-k sampling:
logits = model(tokens.to(device))
logits_last = logits[0, -1]
topk_vals, topk_idx = torch.topk(logits_last, k=5)
probs = F.softmax(topk_vals, dim=-1)
idx_in_topk = torch.multinomial(probs, 1)
next_token = topk_idx[idx_in_topk]
```

---

## Summary: What to Implement

| File | Class/Function | Methods to Implement |
|------|----------------|----------------------|
| `mixture_models.py` | `GMM` | `forward`, `loss_function`, `sample`, `conditional_sample` |
| `mixture_models.py` | `UMM` | `forward`, `loss_function`, `sample`, `conditional_sample` |
| `mixture_models.py` | main block | Training loop for GMM/UMM |
| `transformer.py` | `CausalSelfAttention` | `__init__` (layers), `forward` |
| `transformer.py` | `train_model` | Training loop, generation, top-k generation |

---

## Data Files

| File | Description | Used By |
|------|-------------|---------|
| `train.csv` | Europe countries training data | GMM, UMM |
| `test.csv` | Europe countries test data | GMM, UMM |
| `train_shakespeare.txt` | Shakespeare training text | Transformer |
| `test_shakespeare.txt` | Shakespeare test text | Transformer |

---

## Default Hyperparameters

### Mixture Models
```python
batch_size = 4096
num_epochs = 50
learning_rate_GMM = 0.01
learning_rate_UMM = 0.001
```

### Transformer
```python
block_size = 10
n_layer = 3
n_head = 3
n_embd = 48
learning_rate = 3e-4
batch_size = 64
epochs = 10
```
