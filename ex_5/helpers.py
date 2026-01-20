"""
Helper utilities for machine learning experiments.

This module provides utility functions for:
- Setting random seeds for reproducibility across numpy and PyTorch
- Saving and loading experiment state to/from disk
- Computing dataset statistics (class counts, centroids)
- Visualization helpers for data exploration

These helpers are designed to work with PyTorch-based experiments and datasets
that have .features and .labels tensor attributes.
"""
import os
import pickle
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import EuropeDataset


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy and PyTorch.

    Configures random number generators for numpy, PyTorch CPU, and
    PyTorch CUDA (if available) to ensure reproducible results.

    Args:
        seed: Integer seed value for random number generators. Default is 42.

    Example:
        >>> set_seed(123)  # All subsequent random operations are reproducible
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def save_experiment(experiment, name: str, checkpoint_dir: str = 'checkpoints') -> None:
    """
    Save experiment state to disk for later restoration.

    Serializes the experiment object using pickle and saves it to a file
    in the specified checkpoint directory.

    Args:
        experiment: Any picklable Python object containing experiment state
            (e.g., model parameters, training history, configuration).
        name: Base name for the checkpoint file. The file will be saved as
            '{name}_exp.pkl' in the checkpoint directory.
        checkpoint_dir: Directory path for storing checkpoints. Created
            automatically if it does not exist. Default is 'checkpoints'.

    Example:
        >>> exp_state = {'model': model, 'epoch': 10, 'loss': 0.5}
        >>> save_experiment(exp_state, 'gmm_run1')
        # Saves to checkpoints/gmm_run1_exp.pkl
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'{name}_exp.pkl')
    with open(path, 'wb') as f:
        pickle.dump(experiment, f)


def load_experiment(name: str, checkpoint_dir: str = 'checkpoints'):
    """
    Load experiment state from disk.

    Attempts to load a previously saved experiment checkpoint. Returns None
    if the checkpoint file does not exist, allowing for graceful handling
    of missing checkpoints.

    Args:
        name: Base name of the checkpoint file to load. Expects the file
            to be named '{name}_exp.pkl' in the checkpoint directory.
        checkpoint_dir: Directory path where checkpoints are stored.
            Default is 'checkpoints'.

    Returns:
        The unpickled experiment object if the file exists, or None if
        the checkpoint file is not found.

    Example:
        >>> exp_state = load_experiment('gmm_run1')
        >>> if exp_state is not None:
        ...     model = exp_state['model']
        ...     start_epoch = exp_state['epoch']
    """
    path = os.path.join(checkpoint_dir, f'{name}_exp.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def get_n_classes(dataset) -> int:
    """
    Return the number of unique classes in a dataset.

    Args:
        dataset: Dataset object with a .labels attribute containing
            a tensor of integer class labels.

    Returns:
        Number of unique class labels in the dataset.

    Example:
        >>> n_classes = get_n_classes(train_dataset)
        >>> print(f"Dataset has {n_classes} classes")
    """
    return len(torch.unique(dataset.labels))


def compute_class_centroids(dataset) -> torch.Tensor:
    """
    Compute the mean coordinate (centroid) for each class.

    Calculates the centroid of each class by averaging the feature
    coordinates of all samples belonging to that class.

    Args:
        dataset: Dataset object with:
            - .features: Tensor of shape (N, 2) containing 2D coordinates
            - .labels: Tensor of shape (N,) containing integer class labels

    Returns:
        Tensor of shape (n_classes, 2) where each row contains the
        (x, y) centroid coordinates for the corresponding class.

    Example:
        >>> centroids = compute_class_centroids(train_dataset)
        >>> print(f"Class 0 centroid: {centroids[0]}")
    """
    features = dataset.features
    labels = dataset.labels
    n_classes = get_n_classes(dataset)

    n_features = features.shape[1]
    centroids = torch.zeros(n_classes, n_features)
    for class_idx in range(n_classes):
        class_mask = (labels == class_idx)
        centroids[class_idx] = features[class_mask].mean(dim=0)

    return centroids


def plot_test_by_label(
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.6,
    s: int = 10,
    title: Optional[str] = None,
    heatmap: bool = False,
    bins: int = 50,
) -> None:
    """
    Plot test data points colored by their ground truth country labels.

    Loads the test dataset and creates a scatter plot where each point
    is colored according to its true class label (country). Useful for
    visually inspecting the geographic distribution of countries.

    Args:
        figsize: Figure size as (width, height).
        alpha: Transparency of scatter points (0-1).
        s: Size of scatter points.
        title: Optional custom title. Defaults to "Test Data by Country Label".
        heatmap: If True, show a 2D histogram heatmap instead of scatter plot.
            Highlights density/center of mass - useful for seeing where a model
            with limited components should focus.
        bins: Number of bins for heatmap (only used if heatmap=True).

    Example:
        >>> plot_test_by_label()  # Quick visual inspection of test data
        >>> plot_test_by_label(heatmap=True)  # Density view for resource allocation
    """
    test_ds = EuropeDataset("test.csv")
    features = test_ds.features.numpy()
    labels = test_ds.labels.numpy()

    plt.figure(figsize=figsize)

    if heatmap:
        # 2D histogram showing point density
        plt.hist2d(
            features[:, 0],
            features[:, 1],
            bins=bins,
            cmap="hot",
        )
        plt.colorbar(label="Point Density")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title or "Test Data Density (Center of Mass)")
    else:
        n_classes = len(np.unique(labels))
        scatter = plt.scatter(
            features[:, 0],
            features[:, 1],
            c=labels,
            cmap="tab20" if n_classes <= 20 else "nipy_spectral",
            alpha=alpha,
            s=s,
        )
        plt.colorbar(scatter, label="Country Label")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title or "Test Data by Country Label (Ground Truth)")

    plt.tight_layout()
    plt.show()
