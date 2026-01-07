# Section 7: ResNet + XGBoost Design

## Overview

Reusable component architecture for Exercise 4 Section 7 (CNN deepfake classification). Follows the MLP pattern from Section 6 with Experiment/Results class separation.

## Requirements (from Exercise)

- 5 baselines: XGBoost, ResNet scratch, Linear Probing, sklearn Linear Probing (bonus), Fine-tuning
- Learning rates: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5] for PyTorch models
- Batch size: 32, Epochs: 1
- Loss: BCEWithLogitsLoss
- Optimizer: Adam

### Questions to Answer
- Q7.6.1: Best 2 models per baseline, worst overall, explain trends
- Q7.6.2: Visualize 5 samples correct by best, misclassified by worst

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│  FaceDataset(split, mode='cnn'|'flat', pretrained=True|False)   │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                   ┌───────────────────┐
│    MODEL LAYER    │                   │   XGB (no model)  │
│    ResNetBase     │                   │                   │
│    ├─ Scratch     │                   └───────────────────┘
│    ├─ LinearProbe │                             │
│    └─ FineTune    │                             ▼
└───────────────────┘                   ┌───────────────────┐
        │                               │   XGBBaseline     │
        ▼                               │   (standalone)    │
┌───────────────────┐                   └───────────────────┘
│  TRAINER LAYER    │                             │
│  CNNTrainer       │                             │
└───────────────────┘                             │
        │                                         │
        ▼                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EXPERIMENT LAYER                           │
│  BaseExperiment (LR grid, loaders, run loop)                    │
│    ├─ ScratchExperiment                                         │
│    ├─ LinearProbeExperiment         XGBExperiment (standalone)  │
│    └─ FineTuneExperiment                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AGGREGATOR + RESULTS                        │
│  Section7Aggregator (cross-baseline ranking)                    │
│  Section7Results (tables, misclassification plots)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Layer

**`FaceDataset`**

```python
class FaceDataset(Dataset):
    def __init__(self, root='whichfaceisreal', split='train',
                 mode='cnn', pretrained=True):
        """
        Args:
            root: path to dataset folder
            split: 'train', 'val', or 'test'
            mode: 'cnn' (tensors) or 'flat' (numpy arrays for XGBoost)
            pretrained: if True, use ImageNet normalization;
                        if False, use simple [0.5, 0.5, 0.5] centering
        """
```

**Transform logic:**
- `mode='cnn', pretrained=True`: ImageNet stats `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- `mode='cnn', pretrained=False`: Simple centering `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`
- `mode='flat'`: Resize, convert to numpy, flatten, scale to [0,1]

**Labels:** `real=1`, `fake=0`

---

### 2. Model Layer

**`ResNetBase`** - Abstract base class:

```python
class ResNetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = None  # ResNet18 without final FC
        self.head = None      # Final linear layer (512 -> 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x):
        """Extract features before head - useful for sklearn probing"""
        return self.backbone(x)
```

**Subclasses:**

| Class | Weights | Trainable Params |
|-------|---------|------------------|
| `ResNetScratch` | Random | All |
| `ResNetLinearProbe` | ImageNet pretrained | Head only (backbone frozen) |
| `ResNetFineTune` | ImageNet pretrained | All |

Output: single logit (for BCEWithLogitsLoss)

---

### 3. Trainer Layer

**`CNNTrainer`** - PyTorch training for any ResNet variant:

```python
class CNNTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate=1e-3, num_epochs=1, device='cuda'):
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_acc': [], 'val_acc': [], 'test_acc': []
        }

    def train(self): ...
    def evaluate(self, loader): ...  # Returns loss, accuracy
    def predict(self, loader): ...   # Returns predictions + labels
```

**`XGBBaseline`** - Standalone for XGBoost:

```python
class XGBBaseline:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Takes numpy arrays directly (from FaceDataset mode='flat')"""
        self.model = XGBClassifier()  # default params

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X): ...
```

---

### 4. Experiment Layer

**`BaseExperiment`** - Abstract base for CNN experiments:

```python
class BaseExperiment:
    def __init__(self, learning_rates=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                 num_epochs=1, batch_size=32, seed=42):
        self.learning_rates = learning_rates
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.trainers = {}  # key: f"lr={lr}"
        self.is_trained = False

    def _create_model(self):
        """Override in subclass"""
        raise NotImplementedError

    def _uses_pretrained(self):
        """Override in subclass - controls dataset normalization"""
        raise NotImplementedError

    def run(self): ...
    def get_ranked_models(self): ...
```

**Subclasses:**

| Class | `_create_model()` | `_uses_pretrained()` |
|-------|-------------------|----------------------|
| `ScratchExperiment` | `ResNetScratch()` | `False` |
| `LinearProbeExperiment` | `ResNetLinearProbe()` | `True` |
| `FineTuneExperiment` | `ResNetFineTune()` | `True` |

**`XGBExperiment`** - Standalone (no LR grid, no inheritance)

---

### 5. Aggregator + Results Layer

**`Section7Aggregator`**

```python
class Section7Aggregator:
    def __init__(self, experiments: dict):
        """
        experiments: {
            'xgb': XGBExperiment,
            'scratch': ScratchExperiment,
            'linear_probe': LinearProbeExperiment,
            'finetune': FineTuneExperiment
        }
        """

    def get_best_per_baseline(self, n=2): ...
    def get_worst_overall(self): ...
    def get_all_ranked(self): ...
```

**`Section7Results`**

```python
class Section7Results:
    def __init__(self, aggregator: Section7Aggregator):
        self.aggregator = aggregator

    def print_summary_table(self):
        """Q7.6.1: Table of best 2 per baseline + worst overall"""

    def plot_misclassified_samples(self, n=5):
        """Q7.6.2: Show n samples correct by best, wrong by worst"""
```

---

## File Organization

All classes in notebook cells (matching MLP pattern from Section 6).

## Usage Example

```python
# Run experiments
xgb_exp = XGBExperiment()
xgb_exp.run()

scratch_exp = ScratchExperiment()
scratch_exp.run()

probe_exp = LinearProbeExperiment()
probe_exp.run()

finetune_exp = FineTuneExperiment()
finetune_exp.run()

# Aggregate and analyze
aggregator = Section7Aggregator({
    'xgb': xgb_exp,
    'scratch': scratch_exp,
    'linear_probe': probe_exp,
    'finetune': finetune_exp
})

results = Section7Results(aggregator)
results.print_summary_table()
results.plot_misclassified_samples(n=5)
```
