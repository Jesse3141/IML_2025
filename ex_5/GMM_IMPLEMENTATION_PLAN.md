# GMM Section: Detailed Implementation Plan

**Version:** 1.1 (Verified & Simplified)

## Overview

This plan covers the complete implementation of the Gaussian Mixture Model (GMM) section following the established implementation principles and respecting the skeleton in `mixture_models.py`.

### Changes from v1.0
- Fixed device compatibility bug in `forward()` (use `math.log` for constants)
- Added epoch snapshots for Q2(a) requirement
- Consolidated 4 files → 2 files (`helpers.py` + `gmm_solution.py`)

---

## Phase 1: Fill GMM Skeleton in `mixture_models.py`

### 1.1 Implement `GMM.forward(X)`

**Location:** `mixture_models.py:45-55`

**Input:** `X` of shape `(n_samples, 2)`
**Output:** `log_likelihood` of shape `(n_samples,)`

**Implementation:**
```python
import math

def forward(self, X):
    # X: (N, 2), self.means: (K, 2), self.log_variances: (K, 2)
    K = self.n_components

    # 1. Compute log π_k using log_softmax on weights
    log_pi = torch.nn.functional.log_softmax(self.weights, dim=0)  # (K,)

    # 2. Compute variances from log_variances
    variances = torch.exp(self.log_variances)  # (K, 2)

    # 3. Compute log p(x|k) for each sample and each component
    # Expand X: (N, 1, 2), means: (1, K, 2) -> diff: (N, K, 2)
    X_expanded = X.unsqueeze(1)                    # (N, 1, 2)
    means_expanded = self.means.unsqueeze(0)       # (1, K, 2)
    diff = X_expanded - means_expanded             # (N, K, 2)

    # Mahalanobis term: sum over dimensions
    # (x - μ)² / σ² for each dimension
    mahalanobis = ((diff ** 2) / variances.unsqueeze(0)).sum(dim=2)  # (N, K)

    # Log determinant term: log(σ₁) + log(σ₂) = 0.5 * (log_var₁ + log_var₂)
    log_det = 0.5 * self.log_variances.sum(dim=1)       # (K,)

    # log p(x|k) = -log(2π) - log(σ₁) - log(σ₂) - 0.5 * mahalanobis
    # NOTE: Use math.log for constant to avoid device compatibility issues
    log_p_x_given_k = -math.log(2 * math.pi) - log_det - 0.5 * mahalanobis  # (N, K)

    # 4. Combine: log p(x) = logsumexp_k [log π_k + log p(x|k)]
    log_joint = log_pi + log_p_x_given_k               # (N, K)
    log_likelihood = torch.logsumexp(log_joint, dim=1) # (N,)

    return log_likelihood
```

---

### 1.2 Implement `GMM.loss_function(log_likelihood)`

**Location:** `mixture_models.py:57-67`

**Implementation:**
```python
def loss_function(self, log_likelihood):
    # Negative mean log-likelihood
    return -log_likelihood.mean()
```

---

### 1.3 Implement `GMM.sample(n_samples)`

**Location:** `mixture_models.py:70-80`

**Implementation:**
```python
def sample(self, n_samples):
    with torch.no_grad():
        device = self.means.device  # Ensure device consistency

        # 1. Get mixture probabilities
        probs = torch.softmax(self.weights, dim=0)  # (K,)

        # 2. Sample component indices
        component_indices = torch.multinomial(probs, n_samples, replacement=True)  # (n_samples,)

        # 3. Get means and stds for sampled components
        sampled_means = self.means[component_indices]                    # (n_samples, 2)
        sampled_stds = torch.exp(0.5 * self.log_variances[component_indices])  # (n_samples, 2)

        # 4. Sample from standard normal and transform
        z = torch.randn(n_samples, 2, device=device)
        samples = sampled_means + z * sampled_stds

    return samples
```

---

### 1.4 Implement `GMM.conditional_sample(n_samples, label)`

**Location:** `mixture_models.py:82-93`

**Implementation:**
```python
def conditional_sample(self, n_samples, label):
    with torch.no_grad():
        # Get mean and std for the specified component
        mean = self.means[label]                           # (2,)
        std = torch.exp(0.5 * self.log_variances[label])   # (2,)

        # Sample from standard normal and transform
        z = torch.randn(n_samples, 2, device=self.means.device)
        samples = mean + z * std

    return samples
```

---

## Phase 2: Create `helpers.py`

**New file:** `ex_5/helpers.py`

```python
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_experiment(exp, name, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'{name}_exp.pkl')
    with open(path, 'wb') as f:
        pickle.dump(exp, f)

def load_experiment(name, checkpoint_dir='checkpoints'):
    path = os.path.join(checkpoint_dir, f'{name}_exp.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def get_n_classes(dataset):
    """Return number of unique classes in dataset."""
    return len(torch.unique(dataset.labels))

def compute_class_centroids(dataset):
    """Compute mean coordinate for each class."""
    features = dataset.features
    labels = dataset.labels
    n_classes = get_n_classes(dataset)
    centroids = torch.zeros(n_classes, 2)
    for k in range(n_classes):
        mask = (labels == k)
        centroids[k] = features[mask].mean(dim=0)
    return centroids
```

---

## Phase 3: Create `GMMTrainer` Class

**Location:** Inside `ex_5/gmm_solution.py` (consolidated file)

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from mixture_models import GMM
from dataset import EuropeDataset
from helpers import set_seed

class GMMTrainer:
    """
    Stateful trainer for GMM with epoch snapshot support for Q2(a).
    """
    # Epochs at which to save parameter snapshots for Q2(a)
    SNAPSHOT_EPOCHS = {1, 10, 20, 30, 40, 50}

    def __init__(self,
                 n_components,
                 lr=0.01,
                 epochs=50,
                 batch_size=4096,
                 seed=42,
                 init_means=None,
                 verbose=True):

        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.n_components = n_components
        self.num_epochs = epochs

        # Load and normalize data (use train stats for both)
        self.train_ds = EuropeDataset('train.csv')
        self.test_ds = EuropeDataset('test.csv')

        self.train_mean = self.train_ds.features.mean(dim=0, keepdim=True)
        self.train_std = self.train_ds.features.std(dim=0, keepdim=True)

        self.train_ds.features = (self.train_ds.features - self.train_mean) / self.train_std
        self.test_ds.features = (self.test_ds.features - self.train_mean) / self.train_std

        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)

        # Model
        self.model = GMM(n_components).to(self.device)

        # Optional: initialize means from centroids
        if init_means is not None:
            with torch.no_grad():
                self.model.means.copy_(init_means.to(self.device))

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # History
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'train_log_likelihood': [],
            'test_log_likelihood': []
        }

        # Parameter snapshots for Q2(a) epoch progression plots
        self.param_snapshots = {}

        self.is_trained = False

    def train(self):
        for epoch in tqdm(range(self.num_epochs), disable=not self.verbose):
            # Training
            self.model.train()
            epoch_loss = 0.0
            epoch_ll = 0.0
            n_samples = 0

            for X, _ in self.train_loader:
                X = X.to(self.device)

                self.optimizer.zero_grad()
                log_likelihood = self.model(X)
                loss = self.model.loss_function(log_likelihood)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * X.shape[0]
                epoch_ll += log_likelihood.sum().item()
                n_samples += X.shape[0]

            train_loss = epoch_loss / n_samples
            train_ll = epoch_ll / n_samples

            # Evaluation
            test_loss, test_ll = self._evaluate(self.test_loader)

            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['train_log_likelihood'].append(train_ll)
            self.history['test_log_likelihood'].append(test_ll)

            # Save snapshot for Q2(a) at specific epochs
            epoch_num = epoch + 1  # 1-indexed
            if epoch_num in self.SNAPSHOT_EPOCHS:
                self.param_snapshots[epoch_num] = {
                    'means': self.model.means.detach().clone().cpu(),
                    'log_variances': self.model.log_variances.detach().clone().cpu(),
                    'weights': self.model.weights.detach().clone().cpu()
                }

            if self.verbose and epoch_num % 10 == 0:
                print(f'Epoch {epoch_num}: Train LL={train_ll:.4f}, Test LL={test_ll:.4f}')

        self.is_trained = True
        return self

    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_ll = 0.0
        n_samples = 0

        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                log_likelihood = self.model(X)
                loss = self.model.loss_function(log_likelihood)

                total_loss += loss.item() * X.shape[0]
                total_ll += log_likelihood.sum().item()
                n_samples += X.shape[0]

        return total_loss / n_samples, total_ll / n_samples

    def sample(self, n_samples):
        """Sample from the current model state."""
        return self.model.sample(n_samples).cpu().numpy()

    def conditional_sample(self, k, n_samples):
        """Sample from the k-th Gaussian component."""
        return self.model.conditional_sample(n_samples, k).cpu().numpy()

    def sample_at_epoch(self, epoch, n_samples):
        """
        Sample using parameters from a specific epoch snapshot.
        Required for Q2(a) epoch progression plots.
        """
        if epoch not in self.param_snapshots:
            raise ValueError(f"No snapshot for epoch {epoch}. Available: {list(self.param_snapshots.keys())}")

        params = self.param_snapshots[epoch]
        with torch.no_grad():
            probs = torch.softmax(params['weights'], dim=0)
            indices = torch.multinomial(probs, n_samples, replacement=True)

            means = params['means'][indices]
            stds = torch.exp(0.5 * params['log_variances'][indices])

            z = torch.randn(n_samples, 2)
            samples = means + z * stds

        return samples.numpy()

    def conditional_sample_at_epoch(self, epoch, k, n_samples):
        """Sample from k-th component using epoch snapshot."""
        if epoch not in self.param_snapshots:
            raise ValueError(f"No snapshot for epoch {epoch}")

        params = self.param_snapshots[epoch]
        with torch.no_grad():
            mean = params['means'][k]
            std = torch.exp(0.5 * params['log_variances'][k])
            z = torch.randn(n_samples, 2)
            samples = mean + z * std

        return samples.numpy()

    def get_history(self):
        return self.history

    def get_snapshot_epochs(self):
        return sorted(self.param_snapshots.keys())
```

---

## Phase 4: Create Experiment Classes

**Location:** Inside `ex_5/gmm_solution.py` (same consolidated file)

### 4.1 `GMM_Q1_Experiment` - Varying n_components

```python
from gmm_trainer import GMMTrainer
from helpers import set_seed, get_n_classes
from dataset import EuropeDataset

class GMM_Q1_Experiment:
    """Q1: Train GMM with n_components = [1, 5, 10, n_classes]"""

    def __init__(self, n_components_list=None, epochs=50, seed=42, verbose=False):
        self.seed = seed
        self.epochs = epochs
        self.verbose = verbose

        # Determine n_classes
        ds = EuropeDataset('train.csv')
        n_classes = get_n_classes(ds)

        if n_components_list is None:
            self.n_components_list = [1, 5, 10, n_classes]
        else:
            self.n_components_list = n_components_list

        self.trainers = {}
        self.is_trained = False

    def run(self):
        for n_comp in self.n_components_list:
            if self.verbose:
                print(f"Training GMM with K={n_comp}...")

            trainer = GMMTrainer(
                n_components=n_comp,
                epochs=self.epochs,
                seed=self.seed,
                verbose=self.verbose
            )
            trainer.train()
            self.trainers[n_comp] = trainer

        self.is_trained = True
        return self

    def get_trainer(self, n_components):
        return self.trainers.get(n_components)
```

### 4.2 `GMM_Q2_Experiment` - Epoch Progression + Centroid Initialization

```python
class GMM_Q2_Experiment:
    """Q2: Train GMM with n_classes components, analyze epoch progression."""

    def __init__(self, epochs=50, seed=42, verbose=False):
        self.seed = seed
        self.epochs = epochs
        self.verbose = verbose
        self.display_epochs = [1, 10, 20, 30, 40, 50]

        # Get n_classes and centroids
        ds = EuropeDataset('train.csv')
        self.n_classes = get_n_classes(ds)

        self.trainer_random_init = None
        self.trainer_centroid_init = None
        self.centroids = None
        self.is_trained = False

    def run(self):
        from helpers import compute_class_centroids
        from mixture_models import normalize_tensor

        # Compute normalized centroids for initialization
        ds = EuropeDataset('train.csv')
        train_mean = ds.features.mean(dim=0, keepdim=True)
        train_std = ds.features.std(dim=0, keepdim=True)
        ds.features = (ds.features - train_mean) / train_std
        self.centroids = compute_class_centroids(ds)

        # Q2(a,b): Random initialization
        if self.verbose:
            print("Training GMM with random initialization...")
        self.trainer_random_init = GMMTrainer(
            n_components=self.n_classes,
            epochs=self.epochs,
            seed=self.seed,
            verbose=self.verbose
        )
        self.trainer_random_init.train()

        # Q2(c): Centroid initialization
        if self.verbose:
            print("Training GMM with centroid initialization...")
        self.trainer_centroid_init = GMMTrainer(
            n_components=self.n_classes,
            epochs=self.epochs,
            seed=self.seed,
            init_means=self.centroids,
            verbose=self.verbose
        )
        self.trainer_centroid_init.train()

        self.is_trained = True
        return self
```

---

## Phase 5: Create Results Classes

**Location:** Inside `ex_5/gmm_solution.py` (same consolidated file)

```python
import matplotlib.pyplot as plt
import numpy as np

class GMM_Q1_Results:
    """Visualization for Q1: scatter plots for different n_components."""

    def __init__(self, experiment):
        self.exp = experiment

    def plot_samples(self, n_samples=1000, figsize=(16, 4)):
        """Q1(a): 1000 samples from each GMM."""
        if not self.exp.is_trained:
            print("Experiment not run yet.")
            return

        n_configs = len(self.exp.n_components_list)
        fig, axes = plt.subplots(1, n_configs, figsize=figsize)

        for ax, n_comp in zip(axes, self.exp.n_components_list):
            trainer = self.exp.get_trainer(n_comp)
            samples = trainer.sample(n_samples)

            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
            ax.set_title(f'GMM Samples (K={n_comp})')
            ax.set_xlabel('x1 (normalized)')
            ax.set_ylabel('x2 (normalized)')

        plt.tight_layout()
        plt.show()

    def plot_conditional_samples(self, n_per_component=100, figsize=(16, 4)):
        """Q1(b): 100 samples per component, colored."""
        if not self.exp.is_trained:
            return

        n_configs = len(self.exp.n_components_list)
        fig, axes = plt.subplots(1, n_configs, figsize=figsize)

        for ax, n_comp in zip(axes, self.exp.n_components_list):
            trainer = self.exp.get_trainer(n_comp)

            for k in range(n_comp):
                samples = trainer.conditional_sample(k, n_per_component)
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10, label=f'k={k}')

            ax.set_title(f'GMM Conditional (K={n_comp})')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            if n_comp <= 10:
                ax.legend(fontsize=6, ncol=2)

        plt.tight_layout()
        plt.show()


class GMM_Q2_Results:
    """Visualization for Q2: epoch progression and comparison."""

    def __init__(self, experiment):
        self.exp = experiment
        self.display_epochs = [1, 10, 20, 30, 40, 50]

    def plot_epoch_progression(self, trainer_key='random', n_samples=1000, n_per_component=100):
        """
        Q2(a): Display scatter plots at epochs [1, 10, 20, 30, 40, 50].
        Uses parameter snapshots saved during training.
        """
        trainer = self.exp.trainer_random_init if trainer_key == 'random' else self.exp.trainer_centroid_init
        title_prefix = "Random Init" if trainer_key == 'random' else "Centroid Init"

        # Plot 1: Samples from GMM at each epoch
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for ax, epoch in zip(axes, self.display_epochs):
            samples = trainer.sample_at_epoch(epoch, n_samples)
            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
            ax.set_title(f'{title_prefix} - Epoch {epoch}')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')

        plt.suptitle(f'Q2(a): GMM Samples at Different Epochs ({title_prefix})', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Plot 2: Conditional samples at each epoch
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for ax, epoch in zip(axes, self.display_epochs):
            for k in range(min(self.exp.n_classes, 15)):  # Limit for visibility
                cond_samples = trainer.conditional_sample_at_epoch(epoch, k, n_per_component)
                ax.scatter(cond_samples[:, 0], cond_samples[:, 1], alpha=0.5, s=10)
            ax.set_title(f'Epoch {epoch}')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')

        plt.suptitle(f'Q2(a): Conditional Samples by Component ({title_prefix})', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_likelihood_curves(self, figsize=(10, 5)):
        """Q2(b): Train/test log-likelihood vs epoch."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Random init
        h1 = self.exp.trainer_random_init.get_history()
        axes[0].plot(h1['train_log_likelihood'], label='Train', color='blue')
        axes[0].plot(h1['test_log_likelihood'], label='Test', color='orange')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Mean Log-Likelihood')
        axes[0].set_title('Random Initialization')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Centroid init
        h2 = self.exp.trainer_centroid_init.get_history()
        axes[1].plot(h2['train_log_likelihood'], label='Train', color='blue')
        axes[1].plot(h2['test_log_likelihood'], label='Test', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Log-Likelihood')
        axes[1].set_title('Centroid Initialization')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_initializations(self):
        """Q2(c): Compare random vs centroid init."""
        h1 = self.exp.trainer_random_init.get_history()
        h2 = self.exp.trainer_centroid_init.get_history()

        plt.figure(figsize=(10, 5))
        plt.plot(h1['test_log_likelihood'], label='Random Init (Test)', linestyle='--')
        plt.plot(h2['test_log_likelihood'], label='Centroid Init (Test)', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Test Log-Likelihood')
        plt.title('Initialization Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        print(f"Final Test LL - Random: {h1['test_log_likelihood'][-1]:.4f}")
        print(f"Final Test LL - Centroid: {h2['test_log_likelihood'][-1]:.4f}")
```

---

## Phase 6: Notebook Structure

**File:** `ex_5/ex5_gmm.ipynb`

```python
# Cell 1: Setup
import numpy as np
import torch
from helpers import set_seed
from gmm_solution import (
    GMM_Q1_Experiment, GMM_Q2_Experiment,
    GMM_Q1_Results, GMM_Q2_Results
)

set_seed(42)

# Cell 2: Q1 - Varying n_components
print("Running Q1: GMM with varying n_components...")
q1_exp = GMM_Q1_Experiment(verbose=False)
q1_exp.run()

q1_res = GMM_Q1_Results(q1_exp)

# Cell 3: Q1(a) - Sample plots
q1_res.plot_samples()

# Cell 4: Q1(b) - Conditional sample plots
q1_res.plot_conditional_samples()

# Cell 5: [MARKDOWN] Q1 Analysis
"""
### Q1 Analysis
- With K=1, the model is a single Gaussian, cannot capture multi-modal data
- As K increases, the model better captures the underlying distribution
- K=n_classes allows one Gaussian per country, matching the data structure
"""

# Cell 6: Q2 - Epoch progression
print("Running Q2: GMM with n_classes components...")
q2_exp = GMM_Q2_Experiment(verbose=False)
q2_exp.run()

q2_res = GMM_Q2_Results(q2_exp)

# Cell 7: Q2(a) - Epoch progression plots (random init)
q2_res.plot_epoch_progression('random')

# Cell 8: Q2(a) - Epoch progression plots (centroid init)
q2_res.plot_epoch_progression('centroid')

# Cell 9: Q2(b) - Likelihood curves
q2_res.plot_likelihood_curves()

# Cell 10: Q2(c) - Centroid initialization comparison
q2_res.compare_initializations()

# Cell 11: [MARKDOWN] Q2 Analysis
"""
### Q2 Analysis
- Random init: slower convergence, may find different local optima
- Centroid init: faster convergence, starts closer to final solution
- Both should achieve similar final log-likelihood if trained long enough
"""
```

---

## Implementation Checklist

### Skeleton Completion (`mixture_models.py`)
- [ ] `GMM.forward()` - log-likelihood computation with logsumexp
- [ ] `GMM.loss_function()` - negative mean log-likelihood
- [ ] `GMM.sample()` - sample from mixture using multinomial
- [ ] `GMM.conditional_sample()` - sample from specific component

### New Files (Simplified: 2 files only)
- [ ] `helpers.py` - utilities (set_seed, get_n_classes, compute_class_centroids)
- [ ] `gmm_solution.py` - consolidated file containing:
  - GMMTrainer (with epoch snapshots)
  - GMM_Q1_Experiment
  - GMM_Q2_Experiment
  - GMM_Q1_Results
  - GMM_Q2_Results

### Notebook Deliverables
- [ ] Q1(a): Scatter plot with 1000 samples for K=[1,5,10,n_classes]
- [ ] Q1(b): Conditional scatter plots colored by component
- [ ] Q2(a): Epoch progression plots at [1,10,20,30,40,50] using snapshots
- [ ] Q2(b): Train/test log-likelihood curves
- [ ] Q2(c): Centroid initialization comparison

---

## Key Formulas Reference

**Log-likelihood:**
```
log p(x) = logsumexp_k [log π_k + log N(x|μ_k, Σ_k)]
```

**Log Gaussian:**
```
log N(x|k) = -log(2π) - 0.5*log(σ²_k1) - 0.5*log(σ²_k2)
             - 0.5 * [(x1-μ_k1)²/σ²_k1 + (x2-μ_k2)²/σ²_k2]
```

**Sampling:**
```
k ~ Categorical(softmax(weights))
x = μ_k + z * σ_k,  where z ~ N(0, I)
```

---

## Verification Summary

This plan has been verified by a review agent. Key corrections applied:

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| Device bug in `forward()` | Critical | Use `math.log(2*pi)` instead of `torch.tensor()` |
| Missing Q2(a) epoch snapshots | Critical | Added `param_snapshots` dict + `sample_at_epoch()` method |
| Too many files | Medium | Consolidated 4 files → 2 files (`helpers.py` + `gmm_solution.py`) |
| Inconsistent sample return types | Low | Keep tensor in model, convert to numpy in trainer |

**Status: READY FOR IMPLEMENTATION**
