"""
Steering intervention for applying steering vectors during inference.

Provides both vanilla steering (direct addition in SAE latent space) and
whitened steering (steering in whitened latent space with inverse transform).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from tqdm import tqdm

from steering.models.sae import SAEWrapper
from steering.models.whitening import Whitening, fit_whitening, apply_whitened_steering
from steering.steering.diff_in_means import SteeringVector


class SteeringIntervention(ABC):
    """Abstract base class for steering interventions."""

    @abstractmethod
    def apply(
        self,
        activation: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Apply steering intervention to activations.

        Args:
            activation: Input activation from LLM, shape (..., d_model)
            alpha: Steering strength

        Returns:
            Modified activation, same shape as input
        """
        pass

    @abstractmethod
    def get_steered_latents(
        self,
        h: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Apply steering to SAE latents directly.

        Args:
            h: SAE latents, shape (..., d_sae)
            alpha: Steering strength

        Returns:
            Steered SAE latents, same shape as input
        """
        pass


@dataclass
class VanillaSteering(SteeringIntervention):
    """
    Vanilla steering: h' = h + alpha * v

    Directly adds the steering vector to SAE latents.
    """
    sae: SAEWrapper
    steering_vector: SteeringVector

    def apply(
        self,
        activation: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Apply vanilla steering to LLM activations.

        1. Encode activation through SAE
        2. Add alpha * steering_vector to latents
        3. Decode back to activation space
        """
        original_dtype = activation.dtype

        # Encode (SAE may use float32 internally)
        h = self.sae.encode(activation.float())

        # Steer
        h_steered = self.get_steered_latents(h, alpha)

        # Decode and restore original dtype
        result = self.sae.decode(h_steered)
        return result.to(original_dtype)

    def get_steered_latents(
        self,
        h: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Add steering vector to latents."""
        v = self.steering_vector.vector.to(h.device)

        # Handle different input shapes
        if h.dim() == 1:
            return h + alpha * v
        elif h.dim() == 2:
            return h + alpha * v.unsqueeze(0)
        else:
            # For higher dimensions, broadcast appropriately
            shape = [1] * (h.dim() - 1) + [v.shape[0]]
            return h + alpha * v.view(*shape)


@dataclass
class WhitenedSteering(SteeringIntervention):
    """
    Whitened steering: steering in whitened latent space.

    1. Transform h to whitened space: h_w = W.forward(h)
    2. Add steering vector in whitened space: h_w' = h_w + alpha * v_w
    3. Transform back: h' = W.inverse(h_w')

    The steering vector v_w is pre-computed in whitened space.
    """
    sae: SAEWrapper
    steering_vector: SteeringVector
    whitening: Whitening
    steering_vector_whitened: torch.Tensor

    @classmethod
    def from_training_data(
        cls,
        sae: SAEWrapper,
        steering_vector: SteeringVector,
        h_train: torch.Tensor,
        eps: float = 1e-5,
    ) -> "WhitenedSteering":
        """
        Create WhitenedSteering from training SAE latents.

        Args:
            sae: SAE wrapper
            steering_vector: Steering vector in original space
            h_train: Training SAE latents for fitting whitening, shape (n, d_sae)
            eps: Regularization for whitening

        Returns:
            WhitenedSteering instance
        """
        # Fit whitening on training latents
        whitening = fit_whitening(h_train, eps=eps)

        # Transform steering vector to whitened space
        # Note: steering vector is a direction, so we only apply the linear part
        v_whitened = steering_vector.vector @ whitening.u_fwd

        return cls(
            sae=sae,
            steering_vector=steering_vector,
            whitening=whitening,
            steering_vector_whitened=v_whitened,
        )

    def apply(
        self,
        activation: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Apply whitened steering to LLM activations.

        1. Encode activation through SAE
        2. Transform to whitened space
        3. Add alpha * steering_vector_whitened
        4. Transform back from whitened space
        5. Decode back to activation space
        """
        original_dtype = activation.dtype

        # Encode (SAE may use float32 internally)
        h = self.sae.encode(activation.float())

        # Steer in whitened space
        h_steered = self.get_steered_latents(h, alpha)

        # Decode and restore original dtype
        result = self.sae.decode(h_steered)
        return result.to(original_dtype)

    def get_steered_latents(
        self,
        h: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Apply steering in whitened space and transform back."""
        device = h.device
        whitening = self.whitening.to(device)
        v_w = self.steering_vector_whitened.to(device)

        # Handle different input shapes
        original_shape = h.shape
        if h.dim() > 2:
            h = h.reshape(-1, h.shape[-1])

        # Transform to whitened space
        h_w = whitening.forward(h)

        # Add steering vector
        h_w_steered = h_w + alpha * v_w.unsqueeze(0)

        # Transform back
        h_steered = whitening.inverse(h_w_steered)

        # Restore original shape
        if len(original_shape) > 2:
            h_steered = h_steered.reshape(original_shape)

        return h_steered


def create_steering_hook(
    intervention: SteeringIntervention,
    alpha: float,
    layer_name: str,
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    """
    Create a hook function for steering intervention.

    This returns a function that can be passed to LLMWrapper.set_intervention_fn().

    Args:
        intervention: Steering intervention to apply
        alpha: Steering strength
        layer_name: Name of the layer to intervene on

    Returns:
        Hook function that applies steering when called
    """
    def hook_fn(name: str, activation: torch.Tensor) -> torch.Tensor:
        if name == layer_name:
            return intervention.apply(activation, alpha)
        return activation

    return hook_fn


def create_multi_layer_steering_hook(
    interventions: dict[str, SteeringIntervention],
    alpha: float,
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    """
    Create a hook function for multi-layer steering.

    Args:
        interventions: Dictionary mapping layer names to interventions
        alpha: Steering strength (same for all layers)

    Returns:
        Hook function that applies steering at multiple layers
    """
    def hook_fn(name: str, activation: torch.Tensor) -> torch.Tensor:
        if name in interventions:
            return interventions[name].apply(activation, alpha)
        return activation

    return hook_fn


@torch.no_grad()
def collect_training_latents(
    llm,  # LLMWrapper
    sae: SAEWrapper,
    texts: list[str],
    layer_idx: int,
    pooling: str = "mean",
    batch_size: int = 8,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Collect SAE latents from texts for whitening fit.

    Args:
        llm: LLM wrapper
        sae: SAE wrapper
        texts: Input texts
        layer_idx: Layer to extract from
        pooling: Token pooling method
        batch_size: Batch size

    Returns:
        SAE latents, shape (n_texts, d_sae)
    """
    h_list = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc="Collect whitening latents",
            total=(len(texts) + batch_size - 1) // batch_size,
            leave=False,
        )

    for i in iterator:
        batch_texts = texts[i:i + batch_size]
        activations = llm.get_activations(
            batch_texts,
            layer_idx=layer_idx,
            layer_type="residual_post",
            pooling=pooling,
            batch_size=batch_size,
        )
        h = sae.encode(activations.to(sae.device))
        h_list.append(h)

    return torch.cat(h_list, dim=0)


def setup_vanilla_steering(
    sae: SAEWrapper,
    steering_vector: SteeringVector,
) -> VanillaSteering:
    """
    Set up vanilla steering intervention.

    Args:
        sae: SAE wrapper
        steering_vector: Pre-computed steering vector

    Returns:
        VanillaSteering instance
    """
    return VanillaSteering(sae=sae, steering_vector=steering_vector)


def setup_whitened_steering(
    llm,  # LLMWrapper
    sae: SAEWrapper,
    steering_vector: SteeringVector,
    training_texts: list[str],
    layer_idx: int,
    pooling: str = "mean",
    batch_size: int = 8,
    eps: float = 1e-5,
    show_progress: bool = True,
) -> WhitenedSteering:
    """
    Set up whitened steering intervention.

    Collects training latents and fits whitening transform.

    Args:
        llm: LLM wrapper
        sae: SAE wrapper
        steering_vector: Pre-computed steering vector (in original space)
        training_texts: Texts for fitting whitening transform
        layer_idx: Layer index
        pooling: Token pooling method
        batch_size: Batch size
        eps: Whitening regularization

    Returns:
        WhitenedSteering instance
    """
    # Collect training latents
    h_train = collect_training_latents(
        llm=llm,
        sae=sae,
        texts=training_texts,
        layer_idx=layer_idx,
        pooling=pooling,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    return WhitenedSteering.from_training_data(
        sae=sae,
        steering_vector=steering_vector,
        h_train=h_train,
        eps=eps,
    )
