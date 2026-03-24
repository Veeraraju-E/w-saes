"""
Difference-in-means steering vector computation.

Computes steering vectors as the difference between mean activations
of positive and negative class samples. This is the standard approach
used in representation engineering / concept steering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm

from steering.models.llm import LLMWrapper
from steering.models.sae import SAEWrapper


@dataclass
class SteeringVector:
    """
    A steering vector with metadata.

    Attributes:
        vector: The steering vector, shape (d,)
        layer_idx: Layer index this vector is for
        d_sae: SAE latent dimension
        positive_class: Description of positive class
        negative_class: Description of negative class
        n_positive: Number of positive samples used
        n_negative: Number of negative samples used
    """
    vector: torch.Tensor
    layer_idx: int
    d_sae: int
    positive_class: str
    negative_class: str
    n_positive: int
    n_negative: int

    def to(self, device: torch.device) -> "SteeringVector":
        """Move vector to device."""
        return SteeringVector(
            vector=self.vector.to(device),
            layer_idx=self.layer_idx,
            d_sae=self.d_sae,
            positive_class=self.positive_class,
            negative_class=self.negative_class,
            n_positive=self.n_positive,
            n_negative=self.n_negative,
        )

    @property
    def norm(self) -> float:
        """L2 norm of the steering vector."""
        return float(torch.linalg.norm(self.vector).item())


def compute_diff_in_means_from_activations(
    h_positive: torch.Tensor,
    h_negative: torch.Tensor,
    layer_idx: int = 0,
    positive_class: str = "positive",
    negative_class: str = "negative",
) -> SteeringVector:
    """
    Compute difference-in-means steering vector from pre-computed activations.

    Args:
        h_positive: SAE latents for positive class, shape (n_pos, d_sae)
        h_negative: SAE latents for negative class, shape (n_neg, d_sae)
        layer_idx: Layer index for metadata
        positive_class: Description of positive class
        negative_class: Description of negative class

    Returns:
        SteeringVector with the difference-in-means
    """
    mean_positive = h_positive.mean(dim=0)
    mean_negative = h_negative.mean(dim=0)

    # Steering vector points from negative to positive
    # Adding this vector should push toward positive class
    vector = mean_positive - mean_negative

    return SteeringVector(
        vector=vector,
        layer_idx=layer_idx,
        d_sae=vector.shape[0],
        positive_class=positive_class,
        negative_class=negative_class,
        n_positive=h_positive.shape[0],
        n_negative=h_negative.shape[0],
    )


@torch.no_grad()
def compute_diff_in_means_vector(
    llm: LLMWrapper,
    sae: SAEWrapper,
    positive_texts: list[str],
    negative_texts: list[str],
    layer_idx: int,
    positive_class: str = "positive",
    negative_class: str = "negative",
    pooling: str = "mean",
    batch_size: int = 8,
    show_progress: bool = True,
) -> SteeringVector:
    """
    Compute difference-in-means steering vector from text samples.

    This is the main function for computing steering vectors. It:
    1. Extracts activations from the LLM at the specified layer
    2. Encodes activations through the SAE
    3. Computes mean SAE latents for each class
    4. Returns the difference as the steering vector

    Args:
        llm: LLM wrapper for activation extraction
        sae: SAE wrapper for encoding
        positive_texts: Texts representing the positive class (e.g., non-toxic)
        negative_texts: Texts representing the negative class (e.g., toxic)
        layer_idx: Layer to extract activations from
        positive_class: Description of positive class
        negative_class: Description of negative class
        pooling: How to pool across tokens ("mean", "last")
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar

    Returns:
        SteeringVector computed from the difference in means
    """
    device = llm.device

    # Collect SAE latents for positive class
    h_positive_list = []
    pos_iter = range(0, len(positive_texts), batch_size)
    if show_progress:
        pos_iter = tqdm(pos_iter, desc=f"Encoding {positive_class}", leave=False)

    for i in pos_iter:
        batch_texts = positive_texts[i:i + batch_size]
        activations = llm.get_activations(
            batch_texts,
            layer_idx=layer_idx,
            layer_type="residual_post",
            pooling=pooling,
            batch_size=batch_size,
        )
        h = sae.encode(activations.to(device))
        h_positive_list.append(h)

    h_positive = torch.cat(h_positive_list, dim=0)

    # Collect SAE latents for negative class
    h_negative_list = []
    neg_iter = range(0, len(negative_texts), batch_size)
    if show_progress:
        neg_iter = tqdm(neg_iter, desc=f"Encoding {negative_class}", leave=False)

    for i in neg_iter:
        batch_texts = negative_texts[i:i + batch_size]
        activations = llm.get_activations(
            batch_texts,
            layer_idx=layer_idx,
            layer_type="residual_post",
            pooling=pooling,
            batch_size=batch_size,
        )
        h = sae.encode(activations.to(device))
        h_negative_list.append(h)

    h_negative = torch.cat(h_negative_list, dim=0)

    return compute_diff_in_means_from_activations(
        h_positive=h_positive,
        h_negative=h_negative,
        layer_idx=layer_idx,
        positive_class=positive_class,
        negative_class=negative_class,
    )


def compute_steering_vectors_multi_layer(
    llm: LLMWrapper,
    saes: dict[int, SAEWrapper],
    positive_texts: list[str],
    negative_texts: list[str],
    positive_class: str = "positive",
    negative_class: str = "negative",
    pooling: str = "mean",
    batch_size: int = 8,
    show_progress: bool = True,
) -> dict[int, SteeringVector]:
    """
    Compute steering vectors for multiple layers.

    Args:
        llm: LLM wrapper
        saes: Dictionary mapping layer index to SAE wrapper
        positive_texts: Positive class texts
        negative_texts: Negative class texts
        positive_class: Description of positive class
        negative_class: Description of negative class
        pooling: Token pooling method
        batch_size: Batch size
        show_progress: Whether to show progress

    Returns:
        Dictionary mapping layer index to SteeringVector
    """
    steering_vectors = {}

    layer_iter = saes.items()
    if show_progress:
        layer_iter = tqdm(list(layer_iter), desc="Computing steering vectors")

    for layer_idx, sae in layer_iter:
        sv = compute_diff_in_means_vector(
            llm=llm,
            sae=sae,
            positive_texts=positive_texts,
            negative_texts=negative_texts,
            layer_idx=layer_idx,
            positive_class=positive_class,
            negative_class=negative_class,
            pooling=pooling,
            batch_size=batch_size,
            show_progress=False,
        )
        steering_vectors[layer_idx] = sv

    return steering_vectors


def select_top_k_components(
    steering_vector: SteeringVector,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k components by magnitude from a steering vector.

    Args:
        steering_vector: The steering vector
        k: Number of components to select

    Returns:
        Tuple of (indices, values) for top-k components
    """
    v = steering_vector.vector
    magnitudes = torch.abs(v)
    top_k_indices = torch.argsort(magnitudes, descending=True)[:k]
    top_k_values = v[top_k_indices]
    return top_k_indices, top_k_values


def create_sparse_steering_vector(
    steering_vector: SteeringVector,
    k: int,
) -> SteeringVector:
    """
    Create a sparse version of the steering vector with only top-k components.

    Args:
        steering_vector: Original steering vector
        k: Number of components to keep

    Returns:
        New SteeringVector with only top-k components (rest zeroed)
    """
    indices, values = select_top_k_components(steering_vector, k)

    sparse_vector = torch.zeros_like(steering_vector.vector)
    sparse_vector[indices] = values

    return SteeringVector(
        vector=sparse_vector,
        layer_idx=steering_vector.layer_idx,
        d_sae=steering_vector.d_sae,
        positive_class=steering_vector.positive_class,
        negative_class=steering_vector.negative_class,
        n_positive=steering_vector.n_positive,
        n_negative=steering_vector.n_negative,
    )
