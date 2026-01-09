import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


################ utilities #######################
def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


################ checkpoints #######################
CHECKPOINT_DIR = 'section7_checkpoints'


def save_experiment(exp, name):
    """Save an experiment to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'{name}_exp.pkl')
    with open(path, 'wb') as f:
        pickle.dump(exp, f)
    print(f"Saved {name} experiment to {path}")


def load_experiment(name):
    """Load an experiment from disk. Returns None if not found."""
    path = os.path.join(CHECKPOINT_DIR, f'{name}_exp.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            exp = pickle.load(f)
        print(f"Loaded {name} experiment from {path}")
        return exp
    return None


def save_all_experiments(xgb_exp, scratch_exp, probe_exp, finetune_exp):
    """Save all experiments."""
    save_experiment(xgb_exp, 'xgb')
    save_experiment(scratch_exp, 'scratch')
    save_experiment(probe_exp, 'probe')
    save_experiment(finetune_exp, 'finetune')
    print(f"\nAll experiments saved to {CHECKPOINT_DIR}/")


def load_all_experiments():
    """Load all experiments. Returns (xgb, scratch, probe, finetune) or None."""
    xgb = load_experiment('xgb')
    scratch = load_experiment('scratch')
    probe = load_experiment('probe')
    finetune = load_experiment('finetune')

    if all([xgb, scratch, probe, finetune]):
        print("\nAll experiments loaded successfully!")
        return xgb, scratch, probe, finetune
    else:
        missing = []
        if not xgb: missing.append('xgb')
        if not scratch: missing.append('scratch')
        if not probe: missing.append('probe')
        if not finetune: missing.append('finetune')
        print(f"\nMissing experiments: {missing}. Need to run training.")
        return None


def plot_decision_boundaries(model, X, y, title='Decision Boundaries', implicit_repr=False):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    added_margin_x = h_x * 20
    added_margin_y = h_y * 20
    x_min, x_max = X[:, 0].min() - added_margin_x, X[:, 0].max() + added_margin_x
    y_min, y_max = X[:, 1].min() - added_margin_y, X[:, 1].max() + added_margin_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if implicit_repr:
        model_inp = np.c_[xx.ravel(), yy.ravel()]
        new_model_inp = np.zeros((model_inp.shape[0], model_inp.shape[1] * 10))
        alphas = np.arange(0.1, 1.05, 0.1)
        for i in range(model_inp.shape[1]):
            for j, a in enumerate(alphas):
                new_model_inp[:, i * len(alphas) + j] = np.sin(a * model_inp[:, i])
        model_inp = torch.tensor(new_model_inp, dtype=torch.float32, device=device)
    else:
        model_inp = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)
    with torch.no_grad():
        Z = model(model_inp).argmax(dim=1).cpu().numpy()
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def plot_lines(data, xlabel='X', ylabel='Y', title='', figsize=(10, 6),
               legend_loc='best', grid=True, x_shared=None):
    """
    Flexible line plotting for experiments.

    Args:
        data: dict of {label: y_values} or {label: (x_values, y_values)}
        x_shared: optional shared x-axis values if data contains only y values
        xlabel, ylabel, title: plot labels

    Examples:
        # Multiple curves with shared x
        plot_lines({'LR=0.01': [loss1, loss2, ...], 'LR=0.001': [...]},
                   x_shared=range(50), xlabel='Epoch', ylabel='Loss')

        # Multiple curves with custom x per curve
        plot_lines({'Train': (epochs, train_loss), 'Val': (epochs, val_loss)})
    """
    plt.figure(figsize=figsize)

    for label, values in data.items():
        if isinstance(values, tuple):
            x, y = values
            plt.plot(x, y, label=label, marker='o', markersize=3)
        else:
            x = x_shared if x_shared is not None else range(len(values))
            plt.plot(x, values, label=label, marker='o', markersize=3)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc=legend_loc)
    if grid:
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_bar_comparison(x_values, y_data, xlabel='X', ylabel='Y', title='',
                        figsize=(10, 6), legend_loc='best'):
    """
    Bar plot comparison for metrics across configurations.

    Args:
        x_values: list of x-axis labels (e.g., [1, 2, 6, 10] for depths)
        y_data: dict of {metric_name: [values]} (e.g., {'train_acc': [...], 'val_acc': [...]})

    Example:
        plot_bar_comparison([1, 2, 6, 10],
                           {'Train': [0.8, 0.85, 0.9, 0.88], 'Val': [0.75, 0.8, 0.85, 0.82]},
                           xlabel='Depth', ylabel='Accuracy')
    """
    plt.figure(figsize=figsize)

    x_pos = np.arange(len(x_values))
    width = 0.8 / len(y_data)

    for i, (label, values) in enumerate(y_data.items()):
        offset = width * i - (width * len(y_data) / 2 - width / 2)
        plt.bar(x_pos + offset, values, width, label=label, alpha=0.8)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x_pos, x_values)
    plt.legend(loc=legend_loc)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multi_metric(x_values, y_data, xlabel='X', ylabel='Y', title='',
                      figsize=(10, 6), legend_loc='best', markers=True):
    """
    Line plot with multiple metrics (similar to plot_lines but with x_values).

    Args:
        x_values: shared x-axis values
        y_data: dict of {metric_name: [values]}

    Example:
        plot_multi_metric([1, 2, 6, 10],
                         {'Train': [0.8, 0.85, 0.9, 0.88], 'Val': [0.75, 0.8, 0.85, 0.82]},
                         xlabel='Depth', ylabel='Accuracy')
    """
    plt.figure(figsize=figsize)

    for label, values in y_data.items():
        if markers:
            plt.plot(x_values, values, label=label, marker='o', markersize=8, linewidth=2)
        else:
            plt.plot(x_values, values, label=label, linewidth=2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc=legend_loc)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


################ data #######################
def read_data_demo(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    data_numpy = df.values

    return data_numpy, col_names

