"""
Mixture Models for 2D density estimation.

This module implements two mixture models for learning 2D probability distributions:
- GMM (Gaussian Mixture Model): Models data as a mixture of Gaussian components
- UMM (Uniform Mixture Model): Models data as a mixture of uniform box components

Both models are implemented as PyTorch nn.Module classes and support:
- Forward pass computing log-likelihood
- Negative log-likelihood loss for training
- Unconditional and conditional sampling
"""
import math

import torch
import torch.nn as nn

from dataset import EuropeDataset


def normalize_tensor(tensor, dim):
    """
    Normalize tensor to zero mean and unit standard deviation along specified dimension.

    Args:
        tensor: Input tensor to normalize
        dim: Dimension along which to compute statistics

    Returns:
        Normalized tensor with mean=0 and std=1 along the specified dimension
    """
    mean = torch.mean(tensor, dim=dim, keepdim=True)
    std = torch.std(tensor, dim=dim, keepdim=True)
    return (tensor - mean) / (std + 1e-8)


class GMM(nn.Module):
    """
    Gaussian Mixture Model for 2D density estimation.

    Models the data distribution as a weighted sum of K Gaussian components:
        p(x) = Σ_k π_k * N(x | μ_k, Σ_k)

    Each component has a diagonal covariance matrix (independent variances per dimension).

    Attributes:
        n_components: Number of Gaussian components (K)
        weights: Learnable mixture logits (K,) - softmaxed to get π_k
        means: Learnable component means (K, 2)
        log_variances: Learnable log-variances (K, 2) - exponentiated for numerical stability
    """

    def __init__(self, n_components):
        """
        Initialize GMM with random parameters.

        Args:
            n_components: Number of Gaussian components (K)
        """
        super().__init__()
        self.n_components = n_components

        # Mixture weights as logits (will be softmaxed during forward pass)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Component means: (K, 2) for 2D data
        self.means = nn.Parameter(torch.randn(n_components, 2))

        # Log-variances for numerical stability: (K, 2) diagonal covariance
        self.log_variances = nn.Parameter(torch.zeros(n_components, 2))

    def _pairwise_mahalanobis(self, X, means, variances):
        """
        Compute the Mahalanobis distance for all sample-component pairs.

        For diagonal covariance, this computes:
            d²(x, μ_k) = Σ_d (x_d - μ_kd)² / σ²_kd

        This is the squared distance in a space where each dimension is
        scaled by the inverse standard deviation of that component.

        Args:
            X: Input samples (N, D)
            means: Component means (K, D)
            variances: Component variances, diagonal (K, D)

        Returns:
            Mahalanobis distances (N, K) - distance from each sample to each component
        """
        diff = X[:, None, :] - means[None, :, :]  # (N, K, D)
        return (diff ** 2 / variances[None, :, :]).sum(dim=-1)  # (N, K)

    def forward(self, X):
        """
        Compute the log-likelihood of data under the GMM.

        Uses the formula:
            log p(x) = logsumexp_k [log π_k + log N(x | μ_k, Σ_k)]

        where log N(x | μ, Σ) = -D/2 log(2π) - 1/2 log|Σ| - 1/2 (x-μ)ᵀΣ⁻¹(x-μ)

        Args:
            X: Input data (N, 2)

        Returns:
            Log-likelihood per sample (N,)
        """
        log_pi = torch.nn.functional.log_softmax(self.weights, dim=0)
        variances = torch.exp(self.log_variances)

        # Mahalanobis term: (x - μ)ᵀ Σ⁻¹ (x - μ)
        mahalanobis = self._pairwise_mahalanobis(X, self.means, variances)

        # Log determinant: log|Σ| = log(σ₁²) + log(σ₂²) = log_var₁ + log_var₂
        log_det = self.log_variances.sum(dim=-1)

        # Log Gaussian: -D/2 log(2π) - 1/2 log|Σ| - 1/2 d²
        log_p_x_given_k = -math.log(2 * math.pi) - 0.5 * log_det - 0.5 * mahalanobis

        # Mixture: logsumexp over components
        return torch.logsumexp(log_pi + log_p_x_given_k, dim=-1)


    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss for training.

        Args:
            log_likelihood: Per-sample log-likelihoods (N,)

        Returns:
            Scalar NLL loss (mean over samples)
        """
        return -log_likelihood.mean()


    def sample(self, n_samples):
        """
        Generate samples from the GMM using ancestral sampling.

        Sampling procedure:
            1. Sample component k ~ Categorical(π)
            2. Sample x ~ N(μ_k, Σ_k) using reparameterization: x = μ + σ * z, z ~ N(0, I)

        Args:
            n_samples: Number of samples to generate

        Returns:
            Generated samples (n_samples, 2)
        """
        with torch.no_grad():
            device = self.means.device

            # Sample component indices from mixture weights
            probs = torch.softmax(self.weights, dim=0)
            component_indices = torch.multinomial(probs, n_samples, replacement=True)

            # Get parameters for sampled components
            sampled_means = self.means[component_indices]
            sampled_stds = torch.exp(0.5 * self.log_variances[component_indices])

            # Reparameterization: x = μ + σ * z
            z = torch.randn(n_samples, 2, device=device)
            samples = sampled_means + z * sampled_stds

        return samples

    def conditional_sample(self, n_samples, component_index):
        """
        Generate samples from a specific Gaussian component.

        Args:
            n_samples: Number of samples to generate
            component_index: Index of the component to sample from (0 to K-1)

        Returns:
            Generated samples (n_samples, 2)
        """
        with torch.no_grad():
            mean = self.means[component_index]
            std = torch.exp(0.5 * self.log_variances[component_index])

            # Reparameterization: x = μ + σ * z
            z = torch.randn(n_samples, 2, device=self.means.device)
            samples = mean + z * std

        return samples



class UMM(nn.Module):
    """
    Uniform Mixture Model for 2D density estimation.

    Models the data distribution as a weighted sum of K uniform box components:
        p(x) = Σ_k π_k * U(x | center_k, size_k)

    Each component is an axis-aligned rectangular region with uniform density inside.

    Attributes:
        n_components: Number of uniform components (K)
        weights: Learnable mixture logits (K,) - softmaxed to get π_k
        centers: Learnable box centers (K, 2)
        log_sizes: Learnable log of box sizes (K, 2) - exponentiated for positivity
    """

    LOG_PROB_OUTSIDE = -1e6  # Finite value instead of -inf to avoid NaN gradients

    def __init__(self, n_components):
        """
        Initialize UMM with random parameters.

        Args:
            n_components: Number of uniform components (K)
        """
        super().__init__()
        self.n_components = n_components

        # Mixture weights as logits (will be softmaxed during forward pass)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Box centers: (K, 2) for 2D data
        self.centers = nn.Parameter(torch.randn(n_components, 2))

        # Log-sizes for positivity constraint: initialized near 1.0-1.2
        init_sizes = torch.ones(n_components, 2) + torch.rand(n_components, 2) * 0.2
        self.log_sizes = nn.Parameter(torch.log(init_sizes))


    def forward(self, X):
        """
        Compute the log-likelihood of data under the UMM.

        Uses the formula:
            log p(x) = logsumexp_k [log π_k + log U(x | center_k, size_k)]

        where log U(x | center, size) = -log(volume) if x is inside the box, else -inf

        For numerical stability, points outside all boxes get log-prob of -1e6 instead of -inf.

        Args:
            X: Input data (N, 2)

        Returns:
            Log-likelihood per sample (N,)
        """
        log_pi = torch.nn.functional.log_softmax(self.weights, dim=0)
        sizes = torch.exp(self.log_sizes)

        # Compute box bounds: [center - size/2, center + size/2]
        lower = self.centers - sizes / 2
        upper = self.centers + sizes / 2

        # Check if each sample is inside each box
        X_expanded = X.unsqueeze(1)  # (N, 1, 2)
        inside = ((X_expanded >= lower) & (X_expanded <= upper)).all(dim=2)  # (N, K)

        # Log probability: -log(volume) = -sum(log_sizes) for points inside
        log_volume = self.log_sizes.sum(dim=1)  # (K,)
        log_p_x_given_k = -log_volume.unsqueeze(0).expand(X.shape[0], -1)  # (N, K)

        # Mask out points outside the box
        log_p_x_given_k = torch.where(
            inside,
            log_p_x_given_k,
            torch.tensor(self.LOG_PROB_OUTSIDE, device=X.device)
        )

        # Mixture: logsumexp over components
        return torch.logsumexp(log_pi + log_p_x_given_k, dim=-1)

    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss for training.

        Args:
            log_likelihood: Per-sample log-likelihoods (N,)

        Returns:
            Scalar NLL loss (mean over samples)
        """
        return -log_likelihood.mean()


    def sample(self, n_samples):
        """
        Generate samples from the UMM using ancestral sampling.

        Sampling procedure:
            1. Sample component k ~ Categorical(π)
            2. Sample x ~ U(center_k - size_k/2, center_k + size_k/2)

        Args:
            n_samples: Number of samples to generate

        Returns:
            Generated samples (n_samples, 2)
        """
        with torch.no_grad():
            device = self.centers.device

            # Sample component indices from mixture weights
            probs = torch.softmax(self.weights, dim=0)
            component_indices = torch.multinomial(probs, n_samples, replacement=True)

            # Get parameters for sampled components
            sampled_centers = self.centers[component_indices]
            sampled_sizes = torch.exp(self.log_sizes[component_indices])

            # Uniform sampling: x = center + (u - 0.5) * size, where u ~ U(0, 1)
            u = torch.rand(n_samples, 2, device=device)
            samples = sampled_centers + (u - 0.5) * sampled_sizes

        return samples

    def conditional_sample(self, n_samples, component_index):
        """
        Generate samples from a specific uniform component.

        Args:
            n_samples: Number of samples to generate
            component_index: Index of the component to sample from (0 to K-1)

        Returns:
            Generated samples (n_samples, 2)
        """
        with torch.no_grad():
            center = self.centers[component_index]
            size = torch.exp(self.log_sizes[component_index])

            # Uniform sampling: x = center + (u - 0.5) * size
            u = torch.rand(n_samples, 2, device=self.centers.device)
            samples = center + (u - 0.5) * size

        return samples


if __name__ == "__main__":
    # Configuration
    SEED = 42
    BATCH_SIZE = 4096
    NUM_EPOCHS = 50

    # Recommended learning rates:
    # - GMM: 0.01
    # - UMM: 0.001

    torch.manual_seed(SEED)

    # Load and normalize data
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    train_dataset.features = normalize_tensor(train_dataset.features, dim=0)
    test_dataset.features = normalize_tensor(test_dataset.features, dim=0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Training code goes here

