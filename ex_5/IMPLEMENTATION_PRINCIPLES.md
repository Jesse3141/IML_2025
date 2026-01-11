# Exercise 5: Implementation Principles

Based on the patterns established in Exercise 4, this document defines the architecture and workflow for implementing Exercise 5.

---

## Core Architecture Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: SKELETONS                            │
│  mixture_models.py, transformer.py                                      │
│  - Fill in #### YOUR CODE HERE #### sections                            │
│  - Contains model classes (GMM, UMM, CausalSelfAttention, GPT)          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAYER 2: TRAINER CLASSES                         │
│  gmm_trainer.py, umm_trainer.py, transformer_trainer.py                 │
│  - Stateful classes encapsulating: model, data, optimizer, history      │
│  - Methods: train(), evaluate(), sample(), get_history()                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LAYER 3: EXPERIMENT CLASSES                        │
│  mixture_experiments.py, transformer_experiments.py                     │
│  - One class per question/sub-question                                  │
│  - Manages hyperparameter configs, runs training, stores results        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       LAYER 4: RESULTS CLASSES                          │
│  mixture_results.py, transformer_results.py                             │
│  - Visualization and analysis for each experiment                       │
│  - Methods: plot(), print_table(), analyze()                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAYER 5: NOTEBOOK                                │
│  ex5_dev.ipynb                                                          │
│  - Instantiate experiments, call .run(), display results                │
│  - Minimal code - just orchestration                                    │
│  - Export to Word/PDF for final report                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Trainer Class Pattern

Each model type gets a stateful trainer that encapsulates everything needed for training.

### Template Structure:
```python
class GMMTrainer:
    """Trainer for Gaussian Mixture Model experiments."""

    def __init__(self,
                 n_components=5,
                 lr=0.01,
                 epochs=50,
                 batch_size=4096,
                 seed=42,
                 init_means=None,      # Optional: initialize from centroids
                 verbose=True):

        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # Data loading
        self.train_ds = EuropeDataset('train.csv')
        self.test_ds = EuropeDataset('test.csv')
        self.train_ds.features = normalize_tensor(self.train_ds.features, d=0)
        self.test_ds.features = normalize_tensor(self.test_ds.features, d=0)

        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)

        # Model
        self.model = GMM(n_components).to(self.device)
        if init_means is not None:
            self.model.means.data = init_means

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.num_epochs = epochs
        self.is_trained = False

        # History tracking
        self.history = {
            'train_log_likelihood': [],
            'test_log_likelihood': [],
            'train_loss': [],
            'test_loss': []
        }

    def train(self):
        for epoch in tqdm(range(self.num_epochs), disable=not self.verbose):
            self.model.train()
            # ... training loop ...

            # Track metrics per epoch
            self.history['train_log_likelihood'].append(train_ll)
            self.history['test_log_likelihood'].append(test_ll)

        self.is_trained = True
        return self

    def evaluate(self, loader):
        self.model.eval()
        # ... evaluation logic ...
        return mean_log_likelihood

    def sample(self, n_samples):
        return self.model.sample(n_samples)

    def conditional_sample(self, k, n_samples):
        return self.model.conditional_sample(n_samples, k)

    def get_history(self):
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call .train() first.")
        return self.history
```

---

## 2. Experiment Class Pattern

Each question/sub-question gets an experiment class that manages configurations and execution.

### Template Structure:
```python
class GMM_Q1_Experiment:
    """
    Handles Question 1: GMM with varying n_components.
    n_components = [1, 5, 10, n_classes]
    """

    def __init__(self, n_components_list=None, num_epochs=50, seed=42, run_quiet=False):
        if n_components_list is None:
            self.n_components_list = [1, 5, 10, 44]  # 44 = n_classes for Europe
        else:
            self.n_components_list = n_components_list

        self.num_epochs = num_epochs
        self.seed = seed
        self.run_quiet = run_quiet

        self.trainers = {}  # {n_components: trainer}
        self.is_trained = False

    def run(self):
        if not self.run_quiet:
            print(f"--- Starting GMM Q1 Experiment ---")

        for n_comp in self.n_components_list:
            if not self.run_quiet:
                print(f"Training GMM with {n_comp} components...")

            trainer = GMMTrainer(
                n_components=n_comp,
                epochs=self.num_epochs,
                seed=self.seed,
                verbose=not self.run_quiet
            )
            trainer.train()
            self.trainers[n_comp] = trainer

        self.is_trained = True
        if not self.run_quiet:
            print("Experiment complete.")

    def get_trainer(self, n_components):
        return self.trainers.get(n_components)

    def get_all_trainers(self):
        return self.trainers
```

---

## 3. Results Class Pattern

Each experiment gets a paired results class for visualization.

### Template Structure:
```python
class GMM_Q1_Results:
    """
    Visualization for GMM Q1: scatter plots for different n_components.
    """

    def __init__(self, experiment: GMM_Q1_Experiment):
        self.experiment = experiment

    def plot_samples(self, n_samples=1000):
        """Q1(a): Scatter plot of 1000 samples from each GMM."""
        if not self.experiment.is_trained:
            print("Warning: Experiment not run yet.")
            return

        n_configs = len(self.experiment.n_components_list)
        fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 5))

        for ax, n_comp in zip(axes, self.experiment.n_components_list):
            trainer = self.experiment.get_trainer(n_comp)
            samples = trainer.sample(n_samples)

            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
            ax.set_title(f'GMM (K={n_comp})')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')

        plt.tight_layout()
        plt.show()

    def plot_conditional_samples(self, n_samples_per_component=100):
        """Q1(b): Scatter plot colored by component."""
        if not self.experiment.is_trained:
            return

        n_configs = len(self.experiment.n_components_list)
        fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 5))

        for ax, n_comp in zip(axes, self.experiment.n_components_list):
            trainer = self.experiment.get_trainer(n_comp)

            for k in range(n_comp):
                samples = trainer.conditional_sample(k, n_samples_per_component)
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10, label=f'K={k}')

            ax.set_title(f'GMM Conditional (K={n_comp})')

        plt.tight_layout()
        plt.show()
```

---

## 4. Helper Utilities (`helpers.py`)

### Required Functions:
```python
# Reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

# Checkpointing (for long-running experiments)
def save_experiment(exp, name):
    os.makedirs('checkpoints', exist_ok=True)
    with open(f'checkpoints/{name}_exp.pkl', 'wb') as f:
        pickle.dump(exp, f)

def load_experiment(name):
    path = f'checkpoints/{name}_exp.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# Generic plotting
def plot_lines(data, xlabel, ylabel, title, x_shared=None, figsize=(10, 6)):
    """Plot multiple curves on same axes."""
    ...

def plot_log_likelihood_curves(history, title):
    """Plot train/test log-likelihood over epochs."""
    ...
```

---

## 5. File Organization

```
ex_5/
├── dataset.py                    # PROVIDED - data handling
├── mixture_models.py             # SKELETON - GMM, UMM classes
├── transformer.py                # SKELETON - CausalSelfAttention, GPT
│
├── helpers.py                    # NEW - utilities (set_seed, plotting, checkpoints)
│
├── mixture_experiments.py        # NEW - Experiment classes for GMM/UMM
│   ├── class GMMTrainer
│   ├── class UMMTrainer
│   ├── class GMM_Q1_Experiment   # n_components variation
│   ├── class GMM_Q2_Experiment   # epoch progression + centroid init
│   ├── class UMM_Q1_Experiment
│   └── class UMM_Q2_Experiment
│
├── mixture_results.py            # NEW - Results/plotting for GMM/UMM
│   ├── class GMM_Q1_Results
│   ├── class GMM_Q2_Results
│   ├── class UMM_Q1_Results
│   └── class UMM_Q2_Results
│
├── transformer_experiments.py    # NEW - Experiment classes for Transformer
│   ├── class TransformerTrainer
│   ├── class Transformer_Q2_Experiment  # Training + metrics
│   └── class Transformer_Q4_Experiment  # Top-k sampling
│
├── transformer_results.py        # NEW - Results/plotting for Transformer
│   ├── class TransformerResults
│   └── class GenerationResults
│
├── ex5_dev.ipynb                 # NEW - Main notebook
│
├── train.csv, test.csv           # Europe data
├── train_shakespeare.txt         # Shakespeare data
└── test_shakespeare.txt
```

---

## 6. Notebook Pattern

The notebook should be minimal - just orchestration and display.

```python
# Cell 1: Imports
from mixture_experiments import *
from mixture_results import *
from transformer_experiments import *
from transformer_results import *
from helpers import set_seed

set_seed(42)

# Cell 2: GMM Q1
gmm_q1_exp = GMM_Q1_Experiment(run_quiet=True)
gmm_q1_exp.run()

gmm_q1_res = GMM_Q1_Results(gmm_q1_exp)
gmm_q1_res.plot_samples()
gmm_q1_res.plot_conditional_samples()

# Cell 3: GMM Q2
gmm_q2_exp = GMM_Q2_Experiment(run_quiet=True)
gmm_q2_exp.run()

gmm_q2_res = GMM_Q2_Results(gmm_q2_exp)
gmm_q2_res.plot_epoch_progression()
gmm_q2_res.plot_likelihood_curves()

# ... (continue for all questions)

# Cell N: Analysis text in markdown cells between code
```

### Markdown Analysis Pattern:
After each plot, add a markdown cell with analysis:
```markdown
#### Analysis

We observe that...
1. Point 1 about the results
2. Point 2 explaining the behavior
3. Theoretical connection to course material
```

---

## 7. Experiment-Results Pairing for Exercise 5

| Section | Experiment Class | Results Class | Key Outputs |
|---------|------------------|---------------|-------------|
| GMM Q1 | `GMM_Q1_Experiment` | `GMM_Q1_Results` | Sample scatter, conditional scatter |
| GMM Q2 | `GMM_Q2_Experiment` | `GMM_Q2_Results` | Epoch progression, LL curves, centroid comparison |
| UMM Q1 | `UMM_Q1_Experiment` | `UMM_Q1_Results` | Sample scatter, conditional scatter |
| UMM Q2 | `UMM_Q2_Experiment` | `UMM_Q2_Results` | Epoch progression, LL curves, support analysis |
| Transformer Q2 | `TransformerTrainer` | `TransformerResults` | Loss/accuracy curves |
| Transformer Q3 | `TransformerTrainer` | `GenerationResults` | Generated sentences per epoch |
| Transformer Q4 | `TransformerTrainer` | `GenerationResults` | Top-k vs standard generation |

---

## 8. Key Design Principles

### 8.1 Separation of Concerns
- **Trainer**: Owns model lifecycle (init, train, evaluate, save)
- **Experiment**: Owns hyperparameter configs and orchestration
- **Results**: Owns visualization and analysis

### 8.2 Stateful History Tracking
Every trainer maintains a `self.history` dict:
```python
self.history = {
    'train_loss': [],
    'test_loss': [],
    'train_log_likelihood': [],
    'test_log_likelihood': [],
    # ... any metrics needed for plotting
}
```

### 8.3 Quiet Mode
All experiments support `run_quiet=True` for clean notebook output:
```python
def __init__(self, ..., run_quiet=False):
    self.run_quiet = run_quiet

def run(self):
    if not self.run_quiet:
        print("Starting experiment...")
```

### 8.4 Lazy Validation
Results classes check if experiment was run:
```python
def plot(self):
    if not self.experiment.is_trained:
        print("Warning: Experiment not run yet.")
        return
```

### 8.5 Checkpoint Support
For long experiments, support save/load:
```python
# After training
save_experiment(gmm_q2_exp, 'gmm_q2')

# Later, to restore
gmm_q2_exp = load_experiment('gmm_q2')
if gmm_q2_exp is None:
    gmm_q2_exp = GMM_Q2_Experiment()
    gmm_q2_exp.run()
```

---

## 9. Final Report Workflow

1. **Develop** in `ex5_dev.ipynb` with all experiments
2. **Export** notebook to Word: `File > Download as > .docx` or use `nbconvert`
3. **Edit** in Word: clean up formatting, add polish to analysis
4. **Export** to PDF: `ex5_{YOUR_ID}.pdf`

---

## 10. Quick Start Checklist

- [ ] Fill skeletons in `mixture_models.py` (GMM, UMM)
- [ ] Fill skeleton in `transformer.py` (CausalSelfAttention)
- [ ] Create `helpers.py` with utilities
- [ ] Create `GMMTrainer` and `UMMTrainer` classes
- [ ] Create `TransformerTrainer` class
- [ ] Create experiment classes for each question
- [ ] Create results classes for visualization
- [ ] Build notebook with experiment runs + markdown analysis
- [ ] Export and finalize report
