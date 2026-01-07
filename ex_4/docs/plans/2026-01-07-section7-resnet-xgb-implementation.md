# Section 7: ResNet + XGBoost Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement CNN deepfake classification with 4 baselines (XGBoost, ResNet scratch, Linear Probe, Fine-tune) following the MLP pattern from Section 6.

**Architecture:** Data layer (FaceDataset) → Model layer (ResNet variants) → Trainer layer (CNNTrainer, XGBBaseline) → Experiment layer (per-baseline experiments with LR grid) → Aggregator + Results layer (cross-baseline ranking & visualization).

**Tech Stack:** PyTorch, torchvision (ResNet18, ImageNet pretrained), XGBoost, PIL/torchvision transforms

---

## Task 1: FaceDataset Class

**Files:**
- Modify: `/sci/labs/shair/yishaibz/repos/IML_2025/ex_4/dev_ex4.ipynb` (add cell after cell-71)

**Step 1: Add imports cell**

```python
# Section 7: CNN Imports
from PIL import Image
from torchvision import transforms, models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import glob
```

**Step 2: Implement FaceDataset**

```python
class FaceDataset(Dataset):
    """
    Dataset for whichfaceisreal deepfake detection.

    Args:
        root: path to dataset folder (default: 'whichfaceisreal')
        split: 'train', 'val', or 'test'
        mode: 'cnn' (returns tensors) or 'flat' (returns flattened numpy arrays for XGBoost)
        pretrained: if True, use ImageNet normalization; if False, use [0.5, 0.5, 0.5] centering
    """
    def __init__(self, root='whichfaceisreal', split='train', mode='cnn', pretrained=True):
        self.root = root
        self.split = split
        self.mode = mode
        self.pretrained = pretrained

        # Build file list and labels
        # Structure: root/split/real/*.jpg, root/split/fake/*.jpg
        split_dir = os.path.join(root, split)

        self.samples = []  # List of (path, label)

        # Real images: label = 1
        real_dir = os.path.join(split_dir, 'real')
        if os.path.exists(real_dir):
            for img_path in glob.glob(os.path.join(real_dir, '*.jpg')):
                self.samples.append((img_path, 1))
            for img_path in glob.glob(os.path.join(real_dir, '*.png')):
                self.samples.append((img_path, 1))

        # Fake images: label = 0
        fake_dir = os.path.join(split_dir, 'fake')
        if os.path.exists(fake_dir):
            for img_path in glob.glob(os.path.join(fake_dir, '*.jpg')):
                self.samples.append((img_path, 0))
            for img_path in glob.glob(os.path.join(fake_dir, '*.png')):
                self.samples.append((img_path, 0))

        # Define transforms based on mode and pretrained
        if mode == 'cnn':
            if pretrained:
                # ImageNet normalization
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                # Simple centering for training from scratch
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                ])
        else:
            # mode == 'flat' for XGBoost
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.mode == 'flat':
            # Flatten and convert to numpy for XGBoost
            img = img.numpy().flatten()
            return img, label
        else:
            # Return tensor for CNN
            return img, torch.tensor(label, dtype=torch.float32)

    def get_all_data(self):
        """
        Returns all data as numpy arrays. Useful for XGBoost.
        """
        X = []
        y = []
        for i in range(len(self)):
            xi, yi = self[i]
            X.append(xi)
            y.append(yi)
        return np.array(X), np.array(y)
```

**Step 3: Verify dataset loads correctly**

```python
# Quick test
_test_ds = FaceDataset(split='train', mode='cnn', pretrained=True)
print(f"Train dataset size: {len(_test_ds)}")
if len(_test_ds) > 0:
    _img, _label = _test_ds[0]
    print(f"Image shape: {_img.shape}, Label: {_label}")
del _test_ds
```

---

## Task 2: ResNet Model Classes

**Files:**
- Modify: `/sci/labs/shair/yishaibz/repos/IML_2025/ex_4/dev_ex4.ipynb` (add cell)

**Step 1: Implement ResNetBase and subclasses**

```python
class ResNetBase(nn.Module):
    """
    Base class for ResNet18-based binary classifiers.
    Subclasses define whether weights are pretrained and which params are trainable.
    """
    def __init__(self):
        super().__init__()
        self.backbone = None  # ResNet18 without final FC
        self.head = None      # Final linear layer (512 -> 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(-1)  # Output shape: (batch,)

    def get_features(self, x):
        """Extract features before head - useful for sklearn probing"""
        return self.backbone(x)


class ResNetScratch(ResNetBase):
    """ResNet18 trained from scratch (random init, all params trainable)"""
    def __init__(self):
        super().__init__()
        # Load ResNet18 without pretrained weights
        resnet = models.resnet18(weights=None)

        # Remove final FC layer, keep everything else
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

        # Binary classification head
        self.head = nn.Linear(512, 1)


class ResNetLinearProbe(ResNetBase):
    """ResNet18 with frozen pretrained backbone, only head is trainable"""
    def __init__(self):
        super().__init__()
        # Load ResNet18 with ImageNet pretrained weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Binary classification head (only trainable part)
        self.head = nn.Linear(512, 1)


class ResNetFineTune(ResNetBase):
    """ResNet18 with pretrained backbone, all params trainable"""
    def __init__(self):
        super().__init__()
        # Load ResNet18 with ImageNet pretrained weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

        # Binary classification head
        self.head = nn.Linear(512, 1)
```

**Step 2: Verify models instantiate correctly**

```python
# Quick test
_scratch = ResNetScratch()
_probe = ResNetLinearProbe()
_finetune = ResNetFineTune()
print(f"Scratch trainable params: {sum(p.numel() for p in _scratch.parameters() if p.requires_grad)}")
print(f"LinearProbe trainable params: {sum(p.numel() for p in _probe.parameters() if p.requires_grad)}")
print(f"FineTune trainable params: {sum(p.numel() for p in _finetune.parameters() if p.requires_grad)}")
del _scratch, _probe, _finetune
```

---

## Task 3: CNNTrainer Class

**Files:**
- Modify: `/sci/labs/shair/yishaibz/repos/IML_2025/ex_4/dev_ex4.ipynb` (add cell)

**Step 1: Implement CNNTrainer**

```python
class CNNTrainer:
    """
    PyTorch trainer for binary classification with ResNet variants.
    Uses BCEWithLogitsLoss and Adam optimizer.
    """
    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate=1e-3, num_epochs=1, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs

        # Only optimize parameters that require gradients
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_acc': [], 'val_acc': [], 'test_acc': []
        }
        self.is_trained = False

    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc="Epochs", leave=False):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for X, y in tqdm(self.train_loader, desc="Training", leave=False):
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * X.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += y.size(0)
                correct += (predicted == y).sum().item()

            avg_train_loss = train_loss / total
            train_acc = correct / total

            # Evaluation phase
            val_loss, val_acc = self.evaluate(self.val_loader)
            test_loss, test_acc = self.evaluate(self.test_loader)

            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)

        self.is_trained = True

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                total_loss += loss.item() * X.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += y.size(0)
                correct += (predicted == y).sum().item()

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total

    def predict(self, loader):
        """Returns predictions and true labels for a loader"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                outputs = self.model(X)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
```

---

## Task 4: XGBBaseline Class

**Files:**
- Modify: `/sci/labs/shair/yishaibz/repos/IML_2025/ex_4/dev_ex4.ipynb` (add cell)

**Step 1: Implement XGBBaseline**

```python
class XGBBaseline:
    """
    XGBoost baseline for binary classification.
    Takes flattened image arrays directly.
    """
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.is_trained = False

        self.history = {
            'train_acc': None, 'val_acc': None, 'test_acc': None
        }

    def train(self):
        print("Training XGBoost...")
        self.model.fit(self.X_train, self.y_train)

        # Compute accuracies
        self.history['train_acc'] = self.model.score(self.X_train, self.y_train)
        self.history['val_acc'] = self.model.score(self.X_val, self.y_val)
        self.history['test_acc'] = self.model.score(self.X_test, self.y_test)

        self.is_trained = True
        print(f"XGB trained. Test acc: {self.history['test_acc']:.4f}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
```

---

## Task 5: BaseExperiment and CNN Experiment Subclasses

**Files:**
- Modify: `/sci/labs/shair/yishaibz/repos/IML_2025/ex_4/dev_ex4.ipynb` (add cell)

**Step 1: Implement BaseExperiment**

```python
class BaseExperiment:
    """
    Base class for CNN experiments with learning rate grid search.
    Subclasses define model creation and normalization mode.
    """
    def __init__(self, learning_rates=None, num_epochs=1, batch_size=32, seed=42):
        if learning_rates is None:
            self.learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        else:
            self.learning_rates = learning_rates

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed

        self.trainers = {}  # key: f"lr={lr}"
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_model(self):
        """Override in subclass"""
        raise NotImplementedError

    def _uses_pretrained(self):
        """Override in subclass - controls dataset normalization"""
        raise NotImplementedError

    def _get_name(self):
        """Override in subclass - returns experiment name"""
        raise NotImplementedError

    def run(self):
        print(f"--- Running {self._get_name()} Experiment ---")

        # Create data loaders with appropriate normalization
        pretrained = self._uses_pretrained()

        train_ds = FaceDataset(split='train', mode='cnn', pretrained=pretrained)
        val_ds = FaceDataset(split='val', mode='cnn', pretrained=pretrained)
        test_ds = FaceDataset(split='test', mode='cnn', pretrained=pretrained)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        for lr in self.learning_rates:
            print(f"  Training with LR={lr}...")
            set_seed(self.seed)

            model = self._create_model()
            trainer = CNNTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                learning_rate=lr,
                num_epochs=self.num_epochs,
                device=self.device
            )

            trainer.train()
            self.trainers[f"lr={lr}"] = trainer

            print(f"    -> Test acc: {trainer.history['test_acc'][-1]:.4f}")

        self.is_trained = True
        print(f"{self._get_name()} complete.\n")

    def get_ranked_models(self):
        """Returns list of (key, trainer, test_acc) sorted by test_acc descending"""
        ranking = []
        for key, trainer in self.trainers.items():
            test_acc = trainer.history['test_acc'][-1]
            ranking.append((key, trainer, test_acc))
        ranking.sort(key=lambda x: x[2], reverse=True)
        return ranking

    def get_best_n(self, n=2):
        """Returns top n models by test accuracy"""
        return self.get_ranked_models()[:n]
```

**Step 2: Implement subclasses**

```python
class ScratchExperiment(BaseExperiment):
    """ResNet18 trained from scratch"""
    def _create_model(self):
        return ResNetScratch()

    def _uses_pretrained(self):
        return False

    def _get_name(self):
        return "ResNet Scratch"


class LinearProbeExperiment(BaseExperiment):
    """Frozen pretrained ResNet18 with trainable head only"""
    def _create_model(self):
        return ResNetLinearProbe()

    def _uses_pretrained(self):
        return True

    def _get_name(self):
        return "Linear Probe"


class FineTuneExperiment(BaseExperiment):
    """Pretrained ResNet18 with all params trainable"""
    def _create_model(self):
        return ResNetFineTune()

    def _uses_pretrained(self):
        return True

    def _get_name(self):
        return "Fine-Tune"
```

---

## Task 6: XGBExperiment Class

**Files:**
- Modify: `/sci/labs/shair/yishaibz/repos/IML_2025/ex_4/dev_ex4.ipynb` (add cell)

**Step 1: Implement XGBExperiment**

```python
class XGBExperiment:
    """
    Standalone experiment for XGBoost baseline.
    No learning rate grid since XGBoost uses default parameters.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.baseline = None
        self.is_trained = False

    def run(self):
        print("--- Running XGBoost Experiment ---")
        set_seed(self.seed)

        # Load data in flat mode (no normalization needed, just [0,1] scaling)
        print("  Loading flattened data...")
        train_ds = FaceDataset(split='train', mode='flat')
        val_ds = FaceDataset(split='val', mode='flat')
        test_ds = FaceDataset(split='test', mode='flat')

        X_train, y_train = train_ds.get_all_data()
        X_val, y_val = val_ds.get_all_data()
        X_test, y_test = test_ds.get_all_data()

        print(f"  Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

        self.baseline = XGBBaseline(X_train, y_train, X_val, y_val, X_test, y_test)
        self.baseline.train()

        self.is_trained = True
        print("XGBoost experiment complete.\n")

    def get_test_accuracy(self):
        if not self.is_trained:
            raise ValueError("Run experiment first")
        return self.baseline.history['test_acc']

    def get_ranked_models(self):
        """Returns single result in same format as BaseExperiment for compatibility"""
        return [("XGBoost", self.baseline, self.baseline.history['test_acc'])]

    def get_best_n(self, n=1):
        """Returns single XGBoost result"""
        return self.get_ranked_models()[:n]
```

---

## Task 7: Section7Aggregator Class

**Files:**
- Modify: `/sci/labs/shair/yishaibz/repos/IML_2025/ex_4/dev_ex4.ipynb` (add cell)

**Step 1: Implement Section7Aggregator**

```python
class Section7Aggregator:
    """
    Aggregates results across all Section 7 experiments.
    Provides cross-baseline ranking and comparison.
    """
    def __init__(self, experiments: dict):
        """
        Args:
            experiments: dict mapping name to experiment, e.g.:
                {
                    'xgb': XGBExperiment,
                    'scratch': ScratchExperiment,
                    'linear_probe': LinearProbeExperiment,
                    'finetune': FineTuneExperiment
                }
        """
        self.experiments = experiments
        self._validate_experiments()

    def _validate_experiments(self):
        for name, exp in self.experiments.items():
            if not exp.is_trained:
                raise ValueError(f"Experiment '{name}' has not been run yet")

    def get_best_per_baseline(self, n=2):
        """
        Returns dict mapping baseline name to list of (key, model/trainer, test_acc).
        For XGBoost, returns single result.
        """
        results = {}
        for name, exp in self.experiments.items():
            if name == 'xgb':
                results[name] = exp.get_best_n(n=1)  # XGB only has one model
            else:
                results[name] = exp.get_best_n(n=n)
        return results

    def get_all_ranked(self):
        """
        Returns flat list of all models ranked by test accuracy.
        Each item: (baseline_name, config_key, model/trainer, test_acc)
        """
        all_results = []
        for name, exp in self.experiments.items():
            for key, model, acc in exp.get_ranked_models():
                all_results.append((name, key, model, acc))

        all_results.sort(key=lambda x: x[3], reverse=True)
        return all_results

    def get_best_overall(self):
        """Returns the single best model across all baselines"""
        return self.get_all_ranked()[0]

    def get_worst_overall(self):
        """Returns the single worst model across all baselines"""
        return self.get_all_ranked()[-1]
```

---

## Task 8: Section7Results Class

**Files:**
- Modify: `/sci/labs/shair/yishaibz/repos/IML_2025/ex_4/dev_ex4.ipynb` (add cell)

**Step 1: Implement Section7Results**

```python
class Section7Results:
    """
    Visualization and analysis for Section 7.
    Answers Q7.6.1 and Q7.6.2.
    """
    def __init__(self, aggregator: Section7Aggregator):
        self.aggregator = aggregator
        # Store test dataset for visualization
        self.test_ds = FaceDataset(split='test', mode='cnn', pretrained=True)
        self.test_loader = DataLoader(self.test_ds, batch_size=32, shuffle=False)

    def print_summary_table(self):
        """
        Q7.6.1: Print table of best 2 per baseline + worst overall.
        """
        print("=" * 70)
        print("SECTION 7 RESULTS: Best 2 Models per Baseline")
        print("=" * 70)

        best_per_baseline = self.aggregator.get_best_per_baseline(n=2)

        for baseline_name, models in best_per_baseline.items():
            print(f"\n{baseline_name.upper()}:")
            for i, (key, model, acc) in enumerate(models, 1):
                print(f"  {i}. {key}: Test Accuracy = {acc:.4f}")

        print("\n" + "-" * 70)
        worst = self.aggregator.get_worst_overall()
        print(f"WORST OVERALL: {worst[0]} / {worst[1]}: Test Accuracy = {worst[3]:.4f}")
        print("=" * 70)

        # Also print full ranking
        print("\n\nFULL RANKING (all models):")
        print("-" * 50)
        for i, (baseline, key, model, acc) in enumerate(self.aggregator.get_all_ranked(), 1):
            print(f"{i:2d}. [{baseline:12s}] {key:15s} -> {acc:.4f}")

    def plot_misclassified_samples(self, n=5):
        """
        Q7.6.2: Show n samples correctly classified by best model
        but misclassified by worst model.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get best and worst models
        best = self.aggregator.get_best_overall()
        worst = self.aggregator.get_worst_overall()

        best_name = f"{best[0]}/{best[1]}"
        worst_name = f"{worst[0]}/{worst[1]}"

        print(f"Best model: {best_name} (acc={best[3]:.4f})")
        print(f"Worst model: {worst_name} (acc={worst[3]:.4f})")

        # Get predictions from both models
        best_model = best[2]
        worst_model = worst[2]

        # Handle XGBoost vs CNN models differently
        if best[0] == 'xgb':
            # XGBoost: use flat data
            flat_ds = FaceDataset(split='test', mode='flat')
            X_test, y_test = flat_ds.get_all_data()
            best_preds = best_model.predict(X_test)
        else:
            # CNN: use trainer's predict method
            best_preds, _, _ = best_model.predict(self.test_loader)

        if worst[0] == 'xgb':
            flat_ds = FaceDataset(split='test', mode='flat')
            X_test, y_test = flat_ds.get_all_data()
            worst_preds = worst_model.predict(X_test)
        else:
            worst_preds, _, _ = worst_model.predict(self.test_loader)

        # Get true labels
        true_labels = np.array([self.test_ds.samples[i][1] for i in range(len(self.test_ds))])

        # Find samples: best correct, worst incorrect
        best_correct = (best_preds == true_labels)
        worst_incorrect = (worst_preds != true_labels)
        target_indices = np.where(best_correct & worst_incorrect)[0]

        if len(target_indices) < n:
            print(f"Warning: Only {len(target_indices)} samples match criteria (requested {n})")
            n = len(target_indices)

        if n == 0:
            print("No samples found that are correct by best and incorrect by worst.")
            return

        # Select n samples
        selected_indices = target_indices[:n]

        # Plot
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        if n == 1:
            axes = [axes]

        # Use raw transform for visualization (no normalization)
        viz_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        for ax, idx in zip(axes, selected_indices):
            img_path, true_label = self.test_ds.samples[idx]
            img = Image.open(img_path).convert('RGB')
            img_tensor = viz_transform(img)

            # Convert to displayable format
            img_np = img_tensor.permute(1, 2, 0).numpy()

            ax.imshow(img_np)
            ax.set_title(f"True: {'Real' if true_label == 1 else 'Fake'}\n"
                        f"Best: {'Real' if best_preds[idx] == 1 else 'Fake'} ✓\n"
                        f"Worst: {'Real' if worst_preds[idx] == 1 else 'Fake'} ✗")
            ax.axis('off')

        plt.suptitle(f"Samples: Correct by {best_name}, Wrong by {worst_name}", fontsize=14)
        plt.tight_layout()
        plt.show()
```

---

## Task 9: Run All Experiments and Generate Results

**Files:**
- Modify: `/sci/labs/shair/yishaibz/repos/IML_2025/ex_4/dev_ex4.ipynb` (add cell)

**Step 1: Run all experiments**

```python
# Set seed
set_seed(42)

# Run all experiments
xgb_exp = XGBExperiment()
xgb_exp.run()

scratch_exp = ScratchExperiment()
scratch_exp.run()

probe_exp = LinearProbeExperiment()
probe_exp.run()

finetune_exp = FineTuneExperiment()
finetune_exp.run()
```

**Step 2: Aggregate and show results**

```python
# Create aggregator
aggregator = Section7Aggregator({
    'xgb': xgb_exp,
    'scratch': scratch_exp,
    'linear_probe': probe_exp,
    'finetune': finetune_exp
})

# Create results handler
results = Section7Results(aggregator)

# Q7.6.1: Summary table
results.print_summary_table()
```

**Step 3: Plot misclassified samples**

```python
# Q7.6.2: Visualization
results.plot_misclassified_samples(n=5)
```

---

## Task 10: Commit

**Step 1: Stage and commit**

```bash
git add dev_ex4.ipynb
git commit -m "feat: implement Section 7 ResNet + XGBoost deepfake classification

- Add FaceDataset with cnn/flat modes and pretrained normalization
- Add ResNetScratch, ResNetLinearProbe, ResNetFineTune models
- Add CNNTrainer with BCEWithLogitsLoss and Adam optimizer
- Add XGBBaseline for flattened image classification
- Add experiment classes with LR grid search
- Add Section7Aggregator for cross-baseline ranking
- Add Section7Results for Q7.6.1/Q7.6.2 visualization

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Component | Description |
|------|-----------|-------------|
| 1 | FaceDataset | Data loading with cnn/flat modes, ImageNet/simple normalization |
| 2 | ResNet models | Scratch, LinearProbe, FineTune variants |
| 3 | CNNTrainer | PyTorch training loop with BCE loss |
| 4 | XGBBaseline | XGBoost wrapper for flattened images |
| 5 | BaseExperiment | LR grid search base class + CNN subclasses |
| 6 | XGBExperiment | Standalone XGBoost experiment |
| 7 | Section7Aggregator | Cross-baseline ranking |
| 8 | Section7Results | Tables and misclassification plots |
| 9 | Run experiments | Execute all baselines |
| 10 | Commit | Save changes |
