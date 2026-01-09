"""
Image/CNN Experiment classes for Section 7.
Contains ResNet models, trainers, and experiment runners for deepfake detection.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from helpers import set_seed


class CNNTrainer:
    """
    Wraps trained ResNet18 with training, evaluation, and prediction.
    Uses functions from cnn.py: run_training_epoch(), compute_accuracy()
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate=1e-3, num_epochs=1, device=None, verbose=True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Only optimize params that require gradients (for linear probing)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self._is_trained = False
        self.test_acc = None

    def train(self):
        """Train using run_training_epoch() from cnn.py"""
        from cnn import run_training_epoch, compute_accuracy

        for epoch in range(self.num_epochs):
            loss = run_training_epoch(
                self.model, self.criterion, self.optimizer,
                self.train_loader, self.device
            )
            val_acc = compute_accuracy(self.model, self.val_loader, self.device)
            if self.verbose:
                print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss:.4f}, Val acc: {val_acc:.4f}')

        self.test_acc = compute_accuracy(self.model, self.test_loader, self.device)
        self._is_trained = True

    def predict(self, loader):
        """Returns (predictions, labels, probs) as numpy arrays"""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())
                all_probs.append(probs.cpu().numpy())

        return (np.concatenate(all_preds),
                np.concatenate(all_labels),
                np.concatenate(all_probs))

    def get_test_accuracy(self):
        if not self._is_trained:
            raise ValueError("Model not trained yet")
        return self.test_acc


class BaseExperiment:
    """Abstract base for CNN experiments. Interface for Section7Aggregator."""

    def __init__(self, learning_rates=None, num_epochs=1, batch_size=32,
                 path='whichfaceisreal', seed=42, run_quiet=False):
        self.learning_rates = learning_rates or [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.path = path
        self.seed = seed
        self.run_quiet = run_quiet
        self.trained_models = {}  # {key: (trainer, accuracy)}
        self._is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def run(self):
        """LR sweep - creates model, gets its transform, trains"""
        from cnn import get_loaders

        if not self.run_quiet:
            print(f"--- Running {self._get_experiment_name()} ---")

        for lr in self.learning_rates:
            if not self.run_quiet:
                print(f"\n  LR={lr}")
            set_seed(self.seed)

            # Create model - it has the appropriate transform
            model = self._create_model()
            transform = model.transform  # Get transform from model

            # Create loaders with model's transform
            train_loader, val_loader, test_loader = get_loaders(
                self.path, transform, self.batch_size
            )

            trainer = CNNTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                learning_rate=lr,
                num_epochs=self.num_epochs,
                device=self.device,
                verbose=not self.run_quiet
            )
            trainer.train()

            key = f"lr={lr}"
            self.trained_models[key] = (trainer, trainer.get_test_accuracy())
            if not self.run_quiet:
                print(f"    Test acc: {trainer.get_test_accuracy():.4f}")

        self._is_trained = True
        if not self.run_quiet:
            print(f"\n{self._get_experiment_name()} complete.\n")

    def get_ranked_models(self) -> list:
        """Returns [(key, trainer, accuracy), ...] sorted by accuracy desc"""
        if not self.is_trained:
            raise ValueError("Experiment not run yet")
        ranking = [(k, t, acc) for k, (t, acc) in self.trained_models.items()]
        ranking.sort(key=lambda x: x[2], reverse=True)
        return ranking

    def get_best_n(self, n=2) -> list:
        return self.get_ranked_models()[:n]

    # Abstract methods - subclasses override these
    def _create_model(self):
        raise NotImplementedError

    def _get_experiment_name(self) -> str:
        raise NotImplementedError


class ScratchExperiment(BaseExperiment):
    """ResNet18 from scratch (no pretrained weights)."""

    def _create_model(self):
        from cnn import ResNet18
        return ResNet18(pretrained=False, probing=False)

    def _get_experiment_name(self):
        return "ResNet Scratch"


class LinearProbeExperiment(BaseExperiment):
    """Frozen pretrained ResNet18, only train classification head."""

    def _create_model(self):
        from cnn import ResNet18
        return ResNet18(pretrained=True, probing=True)

    def _get_experiment_name(self):
        return "Linear Probe"


class FineTuneExperiment(BaseExperiment):
    """Pretrained ResNet18, all parameters trainable."""

    def _create_model(self):
        from cnn import ResNet18
        return ResNet18(pretrained=True, probing=False)

    def _get_experiment_name(self):
        return "Fine-Tune"


class XGBExperiment:
    """
    XGBoost baseline using skeleton code from xg.py.
    No LR sweep - uses default XGBoost parameters.
    """

    def __init__(self, path='whichfaceisreal', batch_size=32, seed=42, run_quiet=False):
        self.path = path
        self.batch_size = batch_size
        self.seed = seed
        self.run_quiet = run_quiet
        self.model = None
        self._is_trained = False
        self.test_acc = None

        # Store data for predictions later
        self.X_test = None
        self.y_test = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def run(self):
        """Load data using xg.py pattern, train XGBClassifier."""
        from cnn import get_loaders

        if not self.run_quiet:
            print("--- Running XGBoost Experiment ---")
        np.random.seed(self.seed)

        # Use exact transform from xg.py skeleton
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(64),
            transforms.ToTensor()
        ])

        train_loader, val_loader, test_loader = get_loaders(
            self.path, transform, self.batch_size
        )

        # DO NOT CHANGE section from xg.py - flatten to numpy
        if not self.run_quiet:
            print("  Loading data...")
        train_data, train_labels = [], []
        test_data, test_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(train_loader, desc='Train', disable=self.run_quiet):
                train_data.append(imgs)
                train_labels.append(labels)
            train_data = torch.cat(train_data, 0).cpu().numpy()
            train_data = train_data.reshape(len(train_loader.dataset), -1)
            train_labels = torch.cat(train_labels, 0).cpu().numpy()

            for imgs, labels in tqdm(test_loader, desc='Test', disable=self.run_quiet):
                test_data.append(imgs)
                test_labels.append(labels)
            test_data = torch.cat(test_data, 0).cpu().numpy()
            test_data = test_data.reshape(len(test_loader.dataset), -1)
            test_labels = torch.cat(test_labels, 0).cpu().numpy()

        # Store for later predictions
        self.X_test = test_data
        self.y_test = test_labels

        # Train XGBoost with default params (from xg.py)
        if not self.run_quiet:
            print("  Training XGBoost...")
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(train_data, train_labels)

        self.test_acc = self.model.score(test_data, test_labels)
        self._is_trained = True
        if not self.run_quiet:
            print(f"  Test acc: {self.test_acc:.4f}")
            print("XGBoost complete.\n")

    def get_ranked_models(self) -> list:
        if not self.is_trained:
            raise ValueError("Experiment not run yet")
        return [("XGBoost", self.model, self.test_acc)]

    def get_best_n(self, n=1) -> list:
        return self.get_ranked_models()[:n]

    def predict(self, X=None):
        """Predict on stored test data or provided X."""
        if X is None:
            X = self.X_test
        return self.model.predict(X)


class SklearnProbeExperiment:
    """
    (Bonus) Extract features from frozen ResNet18, train sklearn LogisticRegression.
    """

    def __init__(self, path='whichfaceisreal', batch_size=32, seed=42, run_quiet=False):
        self.path = path
        self.batch_size = batch_size
        self.seed = seed
        self.run_quiet = run_quiet
        self.classifier = None
        self._is_trained = False
        self.test_acc = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def run(self):
        from cnn import ResNet18, get_loaders
        from sklearn.linear_model import LogisticRegression

        if not self.run_quiet:
            print("--- Running sklearn Linear Probe (Bonus) ---")
        set_seed(self.seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create frozen pretrained model for feature extraction
        model = ResNet18(pretrained=True, probing=True)
        model = model.to(device)
        model.eval()

        transform = model.transform
        train_loader, val_loader, test_loader = get_loaders(
            self.path, transform, self.batch_size
        )

        # Extract 512-dim features from resnet18 backbone
        if not self.run_quiet:
            print("  Extracting features...")
        X_train, y_train = self._extract_features(model, train_loader, device)
        X_test, y_test = self._extract_features(model, test_loader, device)

        if not self.run_quiet:
            print(f"  Feature shape: {X_train.shape}")

        # Train sklearn LogisticRegression
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X_train, y_train)

        self.test_acc = self.classifier.score(X_test, y_test)
        self._is_trained = True
        if not self.run_quiet:
            print(f"  Test acc: {self.test_acc:.4f}")
            print("sklearn probe complete.\n")

    def _extract_features(self, model, loader, device):
        """Extract features from model.resnet18 backbone."""
        features, labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                # Get 512-dim features from backbone (before classification head)
                feat = model.resnet18(imgs)
                features.append(feat.cpu().numpy())
                labels.append(lbls.numpy())
        return np.vstack(features), np.concatenate(labels)

    def get_ranked_models(self) -> list:
        if not self.is_trained:
            raise ValueError("Experiment not run yet")
        return [("sklearn_LR", self, self.test_acc)]

    def get_best_n(self, n=1) -> list:
        return self.get_ranked_models()[:n]

# ============================================================================
# CNN Experiment Classes (Section 7)
# ============================================================================




class Section7Aggregator:
    """Aggregates results across all Section 7 experiments."""
    def __init__(self, experiments: dict):
        self.experiments = experiments
        self._validate_experiments()

    def _validate_experiments(self):
        for name, exp in self.experiments.items():
            if not exp.is_trained:
                raise ValueError(f"Experiment '{name}' has not been run yet")

    def get_best_per_baseline(self, n=2):
        results = {}
        for name, exp in self.experiments.items():
            if name == 'xgb':
                results[name] = exp.get_best_n(n=1)
            else:
                results[name] = exp.get_best_n(n=n)
        return results

    def get_all_ranked(self):
        all_results = []
        for name, exp in self.experiments.items():
            for key, model, acc in exp.get_ranked_models():
                all_results.append((name, key, model, acc))

        all_results.sort(key=lambda x: x[3], reverse=True)
        return all_results

    def get_best_overall(self):
        return self.get_all_ranked()[0]

    def get_worst_overall(self):
        return self.get_all_ranked()[-1]


class Section7Results:
    """Visualization and analysis for Section 7."""
    def __init__(self, aggregator: Section7Aggregator, path='whichfaceisreal', run_quiet=False):
        self.aggregator = aggregator
        self.path = path
        self.run_quiet = run_quiet

        # Use ImageFolder from skeleton pattern instead of FaceDataset
        self.test_set = torchvision.datasets.ImageFolder(
            root=os.path.join(path, 'test'),
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        )
        self.test_loader = DataLoader(self.test_set, batch_size=32, shuffle=False)

        # Store samples list for image path access: [(path, label), ...]
        self.samples = self.test_set.samples

    def _load_xgb_test_data(self):
        """Load flattened test data for XGBoost using xg.py pattern."""
        xgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(64),
            transforms.ToTensor()
        ])
        xgb_test_set = torchvision.datasets.ImageFolder(
            root=os.path.join(self.path, 'test'),
            transform=xgb_transform
        )
        xgb_loader = DataLoader(xgb_test_set, batch_size=32, shuffle=False)

        test_data = []
        with torch.no_grad():
            for imgs, labels in xgb_loader:
                test_data.append(imgs)
            test_data = torch.cat(test_data, 0).cpu().numpy()
            test_data = test_data.reshape(len(xgb_test_set), -1)
        return test_data

    def print_summary_table(self):
        if self.run_quiet:
            return

        print("=" * 70)
        print("SECTION 7 RESULTS: Best 2 Models per Baseline")
        print("=" * 70)

        best_per_baseline = self.aggregator.get_best_per_baseline(n=2)

        for baseline_name, models_list in best_per_baseline.items():
            print(f"\n{baseline_name.upper()}:")
            for i, (key, model, acc) in enumerate(models_list, 1):
                print(f"  {i}. {key}: Test Accuracy = {acc:.4f}")

        print("\n" + "-" * 70)
        worst = self.aggregator.get_worst_overall()
        print(f"WORST OVERALL: {worst[0]} / {worst[1]}: Test Accuracy = {worst[3]:.4f}")
        print("=" * 70)

        print("\n\nFULL RANKING (all models):")
        print("-" * 50)
        for i, (baseline, key, model, acc) in enumerate(self.aggregator.get_all_ranked(), 1):
            print(f"{i:2d}. [{baseline:12s}] {key:15s} -> {acc:.4f}")

    def plot_misclassified_samples(self, n=5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        best = self.aggregator.get_best_overall()
        worst = self.aggregator.get_worst_overall()

        best_name = f"{best[0]}/{best[1]}"
        worst_name = f"{worst[0]}/{worst[1]}"

        if not self.run_quiet:
            print(f"Best model: {best_name} (acc={best[3]:.4f})")
            print(f"Worst model: {worst_name} (acc={worst[3]:.4f})")

        best_model = best[2]
        worst_model = worst[2]

        # Load XGBoost test data if needed
        xgb_test_data = None

        if best[0] == 'xgb':
            xgb_test_data = self._load_xgb_test_data()
            best_preds = best_model.predict(xgb_test_data)
        else:
            best_preds, _, _ = best_model.predict(self.test_loader)

        if worst[0] == 'xgb':
            if xgb_test_data is None:
                xgb_test_data = self._load_xgb_test_data()
            worst_preds = worst_model.predict(xgb_test_data)
        else:
            worst_preds, _, _ = worst_model.predict(self.test_loader)

        true_labels = np.array([self.samples[i][1] for i in range(len(self.samples))])

        best_correct = (best_preds == true_labels)
        worst_incorrect = (worst_preds != true_labels)
        target_indices = np.where(best_correct & worst_incorrect)[0]

        if len(target_indices) < n:
            if not self.run_quiet:
                print(f"Warning: Only {len(target_indices)} samples match criteria (requested {n})")
            n = len(target_indices)

        if n == 0:
            if not self.run_quiet:
                print("No samples found that are correct by best and incorrect by worst.")
            return

        selected_indices = target_indices[:n]

        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        if n == 1:
            axes = [axes]

        viz_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        for ax, idx in zip(axes, selected_indices):
            img_path, true_label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            img_tensor = viz_transform(img)

            img_np = img_tensor.permute(1, 2, 0).numpy()

            ax.imshow(img_np)
            ax.set_title(f"True: {'Real' if true_label == 1 else 'Fake'}\n"
                        f"Best: {'Real' if best_preds[idx] == 1 else 'Fake'} \u2713\n"
                        f"Worst: {'Real' if worst_preds[idx] == 1 else 'Fake'} \u2717")
            ax.axis('off')

        plt.suptitle(f"Samples: Correct by {best_name}, Wrong by {worst_name}", fontsize=14)
        plt.tight_layout()
        plt.show()
