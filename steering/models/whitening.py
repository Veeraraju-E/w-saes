"""
Whitening transform for SAE latent space.

Adapted from simul/models.py for use with SAE latents in steering evaluation.
Whitening transforms the latent space to have unit diagonal covariance,
which may help steering vectors stay on-manifold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Whitening:
    """
    Whitening transform based on eigenvalue decomposition.

    For row vectors x (n, d):
    - Forward: x_white = (x - mu) @ U @ diag(1/sqrt(lambda))
    - Inverse: x = x_white @ diag(sqrt(lambda)) @ U.T + mu

    Attributes:
        mu: Mean vector, shape (d,)
        u_fwd: Forward transform matrix, shape (d, d)
        u_inv: Inverse transform matrix, shape (d, d)
        eigenvalues: Eigenvalues in descending order, shape (d,)
    """
    mu: torch.Tensor
    u_fwd: torch.Tensor
    u_inv: torch.Tensor
    eigenvalues: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to whitened space."""
        return (x - self.mu.unsqueeze(0)) @ self.u_fwd

    def inverse(self, x_white: torch.Tensor) -> torch.Tensor:
        """Transform back from whitened space."""
        return x_white @ self.u_inv + self.mu.unsqueeze(0)

    def to(self, device: torch.device) -> "Whitening":
        """Move whitening parameters to device."""
        return Whitening(
            mu=self.mu.to(device),
            u_fwd=self.u_fwd.to(device),
            u_inv=self.u_inv.to(device),
            eigenvalues=self.eigenvalues.to(device),
        )


def fit_whitening(
    x: torch.Tensor,
    eps: float = 1e-5,
    max_components: int | None = None,
) -> Whitening:
    """
    Fit whitening transform from data.

    Computes the eigendecomposition of the empirical covariance matrix
    and creates forward/inverse transforms for whitening.

    Args:
        x: Data matrix, shape (n, d) with samples as rows
        eps: Regularization for small eigenvalues
        max_components: If set, only use top-k eigenvectors (for dimensionality reduction)

    Returns:
        Whitening transform fitted to the data
    """
    n, d = x.shape

    # Compute mean
    mu = x.mean(dim=0)
    xc = x - mu.unsqueeze(0)

    # Compute covariance matrix
    cov = (xc.T @ xc) / max(n - 1, 1)

    # Eigendecomposition (returns ascending order)
    evals, evecs = torch.linalg.eigh(cov)

    # Flip to descending order
    evals = torch.flip(evals, dims=(0,))
    evecs = torch.flip(evecs, dims=(1,))

    # Optionally truncate to max_components
    if max_components is not None and max_components < d:
        evals = evals[:max_components]
        evecs = evecs[:, :max_components]

    # Compute scaling factors with regularization
    sqrt_lam = torch.sqrt(torch.clamp(evals, min=eps))
    invsqrt_lam = 1.0 / sqrt_lam

    # Forward transform: (x - mu) @ U @ diag(1/sqrt(lambda))
    u_fwd = evecs * invsqrt_lam.unsqueeze(0)

    # Inverse transform: x_white @ diag(sqrt(lambda)) @ U.T
    u_inv = (evecs * sqrt_lam.unsqueeze(0)).T

    return Whitening(
        mu=mu,
        u_fwd=u_fwd,
        u_inv=u_inv,
        eigenvalues=evals,
    )


def fit_whitening_incremental(
    data_iterator,
    d: int,
    eps: float = 1e-5,
    device: torch.device = torch.device("cpu"),
) -> Whitening:
    """
    Fit whitening transform incrementally from batched data.

    Useful when data doesn't fit in memory.

    Args:
        data_iterator: Iterator yielding batches of shape (batch_size, d)
        d: Feature dimension
        eps: Regularization for small eigenvalues
        device: Device for computation

    Returns:
        Whitening transform
    """
    # Accumulate statistics
    sum_x = torch.zeros(d, device=device)
    sum_xx = torch.zeros(d, d, device=device)
    n = 0

    for batch in data_iterator:
        batch = batch.to(device)
        batch_size = batch.shape[0]
        sum_x += batch.sum(dim=0)
        sum_xx += batch.T @ batch
        n += batch_size

    # Compute mean and covariance
    mu = sum_x / n
    cov = (sum_xx / n) - torch.outer(mu, mu)

    # Eigendecomposition
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.flip(evals, dims=(0,))
    evecs = torch.flip(evecs, dims=(1,))

    # Compute transforms
    sqrt_lam = torch.sqrt(torch.clamp(evals, min=eps))
    invsqrt_lam = 1.0 / sqrt_lam
    u_fwd = evecs * invsqrt_lam.unsqueeze(0)
    u_inv = (evecs * sqrt_lam.unsqueeze(0)).T

    return Whitening(
        mu=mu,
        u_fwd=u_fwd,
        u_inv=u_inv,
        eigenvalues=evals,
    )


def compute_whitened_steering_vector(
    steering_vector: torch.Tensor,
    whitening: Whitening,
) -> torch.Tensor:
    """
    Transform a steering vector to whitened space.

    For a steering vector v computed in original space, this computes
    the equivalent vector in whitened space.

    Note: Steering vectors are direction vectors (not points), so we
    don't subtract the mean.

    Args:
        steering_vector: Steering vector in original space, shape (d,)
        whitening: Fitted whitening transform

    Returns:
        Steering vector in whitened space, shape (d,)
    """
    # For direction vectors, we only apply the linear part (no mean subtraction)
    return steering_vector @ whitening.u_fwd


def apply_whitened_steering(
    h: torch.Tensor,
    steering_vector_whitened: torch.Tensor,
    whitening: Whitening,
    alpha: float,
) -> torch.Tensor:
    """
    Apply steering in whitened space and transform back.

    1. Transform h to whitened space
    2. Add alpha * steering_vector_whitened
    3. Transform back to original space

    Args:
        h: SAE latents in original space, shape (..., d)
        steering_vector_whitened: Steering vector in whitened space, shape (d,)
        whitening: Fitted whitening transform
        alpha: Steering strength

    Returns:
        Steered latents in original space, shape (..., d)
    """
    # Transform to whitened space
    h_white = whitening.forward(h)

    # Apply steering
    h_white_steered = h_white + alpha * steering_vector_whitened.unsqueeze(0)

    # Transform back
    return whitening.inverse(h_white_steered)


def explained_variance_ratio(whitening: Whitening) -> torch.Tensor:
    """
    Compute explained variance ratio for each component.

    Args:
        whitening: Fitted whitening transform

    Returns:
        Explained variance ratio per component, shape (d,)
    """
    total_var = whitening.eigenvalues.sum()
    return whitening.eigenvalues / total_var


def effective_rank(whitening: Whitening, threshold: float = 0.99) -> int:
    """
    Compute effective rank (number of components explaining threshold variance).

    Args:
        whitening: Fitted whitening transform
        threshold: Cumulative variance threshold

    Returns:
        Number of components needed to explain threshold variance
    """
    evr = explained_variance_ratio(whitening)
    cumsum = torch.cumsum(evr, dim=0)
    return int((cumsum < threshold).sum().item()) + 1
