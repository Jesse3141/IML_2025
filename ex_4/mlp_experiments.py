"""
MLP Experiment classes for Section 6.
Contains experiment runners and results handlers for MLP training.
"""
import matplotlib.pyplot as plt

from helpers import set_seed, plot_lines, plot_decision_boundaries
from MLP import MLP, MLPTrainer, EuropeDataset


# ============================================================================
# MLP Experiment Classes (Section 6.1)
# ============================================================================

class Q_1:
    """
    Handles the experimentation for Question 6.1.2.1: Learning Rate.
    """
    def __init__(self, learning_rates=None, num_epochs=50, run_quiet=False):
        if learning_rates is None:
            self.learning_rates = [1.0, 0.2, 0.01, 0.001, 5e-4, 0.00001]
        else:
            self.learning_rates = learning_rates

        self.num_epochs = num_epochs
        self.results = {}
        self.run_quiet = run_quiet

    def run_experiment(self):
        if not self.run_quiet:
            print(f"--- Starting Q1 Experiment: Learning Rates {self.learning_rates} ---")

        for lr in self.learning_rates:
            if not self.run_quiet:
                print(f"Training with LR={lr}...")

            set_seed(42)

            trainer = MLPTrainer(
                lr=lr,
                epochs=self.num_epochs,
                hidden_dim=16,
                num_layers=6,
                verbose=not self.run_quiet
            )

            trainer.train()

            self.results[f'LR={lr}'] = trainer.history['val_loss']
            if not self.run_quiet:
                print(f"  -> Final val loss: {trainer.history['val_loss'][-1]:.4f}")

        self._plot_results()

    def _plot_results(self):
        if not self.run_quiet:
            print("Plotting results...")
        plot_lines(
            self.results,
            x_shared=range(1, self.num_epochs + 1),
            xlabel='Epoch',
            ylabel='Validation Loss',
            title='Q1: Effect of Learning Rate on Validation Loss',
            figsize=(12, 6)
        )


class Q2_Experiment:
    """
    Handles the configuration and execution of Question 6.1.2.2.
    """
    def __init__(self, num_epochs=100, use_batch_norm=False, seed=42, kaiming_init=False, run_quiet=False):
        self.num_epochs = num_epochs
        self.trainer = None
        self.is_trained = False
        self.use_batch_norm = use_batch_norm
        self.seed = seed
        self.kaiming_init = kaiming_init
        self.run_quiet = run_quiet

    def run(self):
        if not self.run_quiet:
            print(f"--- Starting Experiment: {self.num_epochs} Epochs ---")

        self.trainer = MLPTrainer(
            lr=1e-3,
            epochs=self.num_epochs,
            hidden_dim=16,
            num_layers=6,
            seed=self.seed,
            use_batch_norm=self.use_batch_norm,
            verbose=not self.run_quiet
        )

        self.trainer.train()
        self.is_trained = True
        if not self.run_quiet:
            print("Training complete.")

    def get_history(self):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call .run() first.")
        return self.trainer.history


class Q2_Results:
    """
    Handles the visualization and analysis for Question 6.1.2.2.
    """
    def __init__(self, experiment):
        self.experiment = experiment

    def plot(self):
        if not self.experiment.is_trained:
            print("Warning: Experiment not run yet. Nothing to plot.")
            return

        history = self.experiment.get_history()
        epochs_range = range(1, self.experiment.num_epochs + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', markersize=3)
        ax1.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o', markersize=3)
        ax1.plot(epochs_range, history['test_loss'], label='Test Loss', marker='o', markersize=3)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss over Epochs', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o', markersize=3)
        ax2.plot(epochs_range, history['val_acc'], label='Val Acc', marker='o', markersize=3)
        ax2.plot(epochs_range, history['test_acc'], label='Test Acc', marker='o', markersize=3)

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


class Q4_Experiment:
    """
    Handles the configuration and execution of Question 6.1.2.4 (Batch Size).
    """
    def __init__(self, seed=42, lr=1e-3, run_quiet=False):
        self.seed = seed
        self.configs = [
            {'bs': 1,    'epochs': 1},
            {'bs': 16,   'epochs': 10},
            {'bs': 128,  'epochs': 50},
            {'bs': 1024, 'epochs': 50}
        ]
        self.results = {}
        self.is_trained = False
        self.lr = lr
        self.run_quiet = run_quiet

    def run(self):
        if not self.run_quiet:
            print(f"--- Starting Experiment: Batch Size Analysis ---")

        for config in self.configs:
            bs = config['bs']
            epochs = config['epochs']
            if not self.run_quiet:
                print(f"Running: Batch Size = {bs}, Epochs = {epochs}")

            trainer = MLPTrainer(
                lr=self.lr,
                epochs=epochs,
                batch_size=bs,
                hidden_dim=16,
                num_layers=6,
                seed=self.seed,
                use_batch_norm=False,
                verbose=not self.run_quiet
            )

            trainer.train()

            iters_per_epoch = len(trainer.train_loader)

            self.results[bs] = {
                'history': trainer.history,
                'iters_per_epoch': iters_per_epoch,
                'epochs': epochs
            }

        self.is_trained = True
        if not self.run_quiet:
            print("All configurations complete.")

    def get_results(self):
        if not self.is_trained:
            raise ValueError("Experiment not run yet.")
        return self.results


class Q4_Results:
    """
    Handles visualization and analysis for Question 6.1.2.4.
    """
    def __init__(self, experiment):
        self.experiment = experiment

    def print_iterations_table(self):
        """
        Print a table showing batch size and iterations per epoch.
        """
        if not self.experiment.is_trained:
            print("Warning: Experiment not run yet.")
            return

        results = self.experiment.get_results()

        print("| Batch Size | Iterations per Epoch |")
        print("|------------|----------------------|")
        for bs in sorted(results.keys()):
            iters = results[bs]['iters_per_epoch']
            print(f"| {bs:<10} | {iters:<20} |")

    def plot(self):
        if not self.experiment.is_trained:
            print("Warning: Experiment not run yet.")
            return

        results = self.experiment.get_results()

        if not self.experiment.run_quiet:
            print("\n=== (ii) Speed Analysis: Iterations per Epoch ===")
            print(f"{'Batch Size':<12} | {'Iterations/Epoch':<20}")
            print("-" * 35)
            for bs, data in results.items():
                print(f"{bs:<12} | {data['iters_per_epoch']:<20}")
            print("-" * 35)
            print("\n")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        for bs, data in results.items():
            history = data['history']
            epochs = range(1, data['epochs'] + 1)
            ax1.plot(epochs, history['val_acc'], label=f'BS={bs}', marker='o', markersize=4)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Validation Accuracy', fontsize=12)
        ax1.set_title('(i) Validation Accuracy vs. Epoch', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        for bs, data in results.items():
            batch_losses = data['history']['batch_losses']
            steps = range(len(batch_losses))

            alpha = 0.6 if bs > 1 else 0.3
            linewidth = 1.5 if bs > 1 else 0.5

            ax2.plot(steps, batch_losses, label=f'BS={bs}',
                     alpha=alpha, linewidth=linewidth)

        ax2.set_xlabel('Cumulative Batch Iterations', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.set_title('(iii) Stability: Training Loss vs. Batch', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


# ============================================================================
# MLP Experiment Classes (Section 6.2)
# ============================================================================

class Q6_2_Experiment:
    """
    Handles the configuration and execution of Section 6.2: Evaluating MLPs Performance.
    """
    def __init__(self, epochs=50, learning_rate=1e-3, batch_size=128, seed=42, run_quiet=False):
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.is_trained = False
        self.run_quiet = run_quiet

        self.configs = [
            {'depth': 1, 'width': 16},
            {'depth': 2, 'width': 16},
            {'depth': 6, 'width': 16},
            {'depth': 10, 'width': 16},
            {'depth': 6, 'width': 8},
            {'depth': 6, 'width': 32},
            {'depth': 6, 'width': 64},
        ]

        self.trainers = {}

    def get_key(self, depth, width):
        return f"d{depth}_w{width}"

    def run(self):
        if not self.run_quiet:
            print(f"--- Starting Section 6.2 Experiment ({len(self.configs)} configurations) ---")

        for config in self.configs:
            depth = config['depth']
            width = config['width']
            key = self.get_key(depth, width)

            if key in self.trainers:
                continue

            if not self.run_quiet:
                print(f"Training MLP | Depth: {depth} | Width: {width}...")

            trainer = MLPTrainer(
                lr=self.lr,
                epochs=self.epochs,
                batch_size=self.batch_size,
                hidden_dim=width,
                num_layers=depth,
                use_batch_norm=True,
                seed=self.seed,
                verbose=not self.run_quiet
            )

            trainer.train()
            self.trainers[key] = trainer

        self.is_trained = True
        if not self.run_quiet:
            print("All configurations trained.")

    def get_results_by_depth(self, fixed_width=16):
        filtered = []
        for key, trainer in self.trainers.items():
            d_str, w_str = key.split('_')
            d = int(d_str[1:])
            w = int(w_str[1:])

            if w == fixed_width:
                filtered.append((d, trainer))

        filtered.sort(key=lambda x: x[0])
        return filtered

    def get_results_by_width(self, fixed_depth=6):
        filtered = []
        for key, trainer in self.trainers.items():
            d_str, w_str = key.split('_')
            d = int(d_str[1:])
            w = int(w_str[1:])

            if d == fixed_depth:
                filtered.append((w, trainer))

        filtered.sort(key=lambda x: x[0])
        return filtered

    def get_ranked_models(self):
        ranking = []
        for key, trainer in self.trainers.items():
            final_val_acc = trainer.history['val_acc'][-1]
            ranking.append((key, trainer, final_val_acc))

        ranking.sort(key=lambda x: x[2], reverse=True)
        return ranking


class Q6_2_5_Experiment:
    """
    Handles Question 6.2.5: Monitoring Gradients in a Deep MLP.
    """
    def __init__(self, epochs=10, seed=42, run_quiet=False):
        self.epochs = epochs
        self.seed = seed
        self.trainer = None
        self.is_trained = False
        self.run_quiet = run_quiet

        self.hidden_dim = 4
        self.num_layers = 100

        self.layers_to_monitor = [0, 30, 60, 90, 95, 99]

    def run(self):
        if not self.run_quiet:
            print(f"--- Starting Gradient Monitoring Experiment (Depth: {self.num_layers}) ---")

        self.trainer = MLPTrainer(
            lr=1e-3,
            epochs=self.epochs,
            batch_size=256,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            use_batch_norm=False,
            seed=self.seed,
            grad_monitor_layers=self.layers_to_monitor,
            verbose=not self.run_quiet
        )

        self.trainer.train()
        self.is_trained = True
        if not self.run_quiet:
            print("Training complete.")


class Q6_2_5_Results:
    """
    Visualization for Gradient Monitoring.
    """
    def __init__(self, experiment):
        self.experiment = experiment

    def plot(self):
        if not self.experiment.is_trained:
            print("Warning: Experiment not run yet.")
            return

        history = self.experiment.trainer.history['grad_norms']
        epochs = range(1, self.experiment.epochs + 1)

        plt.figure(figsize=(10, 6))

        for layer_idx, grads in history.items():
            plt.plot(epochs, grads, label=f'Layer {layer_idx}', marker='o', markersize=4)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Gradient Magnitude ($||g||^2$)', fontsize=12)
        plt.title(f'Gradient Magnitude per Layer (Depth {self.experiment.num_layers})',
                  fontsize=14, fontweight='bold')

        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.tight_layout()
        plt.show()


class Q6_2_Results:
    """
    Handles the visualization and analysis for Question 6.2.
    """
    def __init__(self, experiment):
        self.experiment = experiment

    def _plot_model_performance(self, rank_index, label_type):
        if not self.experiment.is_trained:
            print("Warning: Experiment not run yet.")
            return

        ranked_models = self.experiment.get_ranked_models()
        key, trainer, acc = ranked_models[rank_index]

        if not self.experiment.run_quiet:
            print(f"--- Analysis of {label_type} Model ---")
            print(f"Configuration: {key}")
            print(f"Final Validation Accuracy: {acc:.4f}")

        history = trainer.history
        epochs_range = range(1, len(history['train_loss']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', markersize=3)
        plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o', markersize=3)
        plt.plot(epochs_range, history['test_loss'], label='Test Loss', marker='o', markersize=3)

        plt.title(f'{label_type} Model ({key}) - Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        X_test = trainer.test_ds.X.numpy()
        y_test = trainer.test_ds.y.numpy()

        plot_decision_boundaries(
            trainer.model,
            X_test,
            y_test,
            title=f'{label_type} Model ({key}) - Decision Boundaries'
        )

    def plot_best_model(self):
        self._plot_model_performance(0, "Best")

    def plot_worst_model(self):
        self._plot_model_performance(-1, "Worst")

    def plot_depth_analysis(self):
        if not self.experiment.is_trained:
            return

        results = self.experiment.get_results_by_depth(fixed_width=16)

        depths = [r[0] for r in results]
        trainers = [r[1] for r in results]

        train_acc = [t.history['train_acc'][-1] for t in trainers]
        val_acc = [t.history['val_acc'][-1] for t in trainers]
        test_acc = [t.history['test_acc'][-1] for t in trainers]

        plt.figure(figsize=(10, 6))
        plt.plot(depths, train_acc, label='Train Acc', marker='o', linewidth=2)
        plt.plot(depths, val_acc, label='Val Acc', marker='o', linewidth=2)
        plt.plot(depths, test_acc, label='Test Acc', marker='o', linewidth=2)

        plt.xlabel('Number of Hidden Layers (Depth)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs. Network Depth (Width=16)', fontsize=14, fontweight='bold')
        plt.xticks(depths)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_width_analysis(self):
        if not self.experiment.is_trained:
            return

        results = self.experiment.get_results_by_width(fixed_depth=6)

        widths = [r[0] for r in results]
        trainers = [r[1] for r in results]

        train_acc = [t.history['train_acc'][-1] for t in trainers]
        val_acc = [t.history['val_acc'][-1] for t in trainers]
        test_acc = [t.history['test_acc'][-1] for t in trainers]

        plt.figure(figsize=(10, 6))
        plt.plot(widths, train_acc, label='Train Acc', marker='o', linewidth=2)
        plt.plot(widths, val_acc, label='Val Acc', marker='o', linewidth=2)
        plt.plot(widths, test_acc, label='Test Acc', marker='o', linewidth=2)

        plt.xlabel('Hidden Layer Width', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs. Network Width (Depth=6)', fontsize=14, fontweight='bold')
        plt.xticks(widths)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
