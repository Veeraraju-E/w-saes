"""
SAE loading via SAELens for steering evaluation.

Provides pretrained SAE loading for Gemma-2-2b and Llama-3-8b models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


# SAE releases available in SAELens for our target models
SAE_RELEASES = {
    "gemma-2-2b": {
        "gemma-scope-2b-pt-res": "google/gemma-scope-2b-pt-res",
        "gemma-scope-2b-pt-res-canonical": "google/gemma-scope-2b-pt-res-canonical",
        "gemma-2-2b-res-matryoshka-dc": "chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40",
    },
    "llama-3-8b": {
        "llama-scope-8b-pt-res": "meta-llama/llama-scope-8b-pt-res",
    },
    "llama-3.1-8b": {
        "llama-3.1-8b-base-res-canonical": "EleutherAI/sae-llama-3.1-8b-base-res-canonical",
    },
}

# Default layer ranges for SAE availability
DEFAULT_SAE_LAYERS = {
    "gemma-2-2b": list(range(26)),  # 26 layers
    "llama-3-8b": list(range(32)),  # 32 layers
    "llama-3.1-8b": list(range(32)),
}


@dataclass
class SAEConfig:
    """Configuration for a loaded SAE."""
    release: str
    sae_id: str
    hook_name: str
    d_in: int
    d_sae: int
    layer_idx: int


class SAEWrapper:
    """
    Wrapper for SAE models loaded via SAELens.

    Provides a unified interface for encoding and decoding activations.
    """

    def __init__(
        self,
        sae: Any,  # SAELens SAE object
        config: SAEConfig,
        device: torch.device,
    ):
        self.sae = sae
        self.config = config
        self.device = device

    @property
    def d_in(self) -> int:
        return self.config.d_in

    @property
    def d_sae(self) -> int:
        return self.config.d_sae

    @property
    def layer_idx(self) -> int:
        return self.config.layer_idx

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to SAE latent space.

        Args:
            x: Input activations, shape (..., d_in)

        Returns:
            SAE latents, shape (..., d_sae)
        """
        return self.sae.encode(x)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Decode SAE latents back to activation space.

        Args:
            h: SAE latents, shape (..., d_sae)

        Returns:
            Reconstructed activations, shape (..., d_in)
        """
        return self.sae.decode(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            x: Input activations, shape (..., d_in)

        Returns:
            Tuple of (reconstructed activations, SAE latents)
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    def encode_with_error(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and compute reconstruction error.

        Args:
            x: Input activations

        Returns:
            Tuple of (SAE latents, reconstruction error)
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        error = x - x_hat
        return h, error


def load_sae(
    model_name: str,
    layer_idx: int,
    release: str | None = None,
    sae_id: str | None = None,
    device: str | torch.device = "cuda",
) -> SAEWrapper:
    """
    Load a pretrained SAE for a specific model and layer.

    Args:
        model_name: Model name (e.g., "gemma-2-2b", "llama-3-8b")
        layer_idx: Layer index to load SAE for
        release: SAELens release name (uses default if None)
        sae_id: Specific SAE ID within release (auto-detected if None)
        device: Device to load SAE on

    Returns:
        SAEWrapper instance
    """
    from sae_lens import SAE

    device = torch.device(device)

    # Get default release if not specified
    if release is None:
        if model_name not in SAE_RELEASES:
            raise ValueError(f"No default SAE releases for model: {model_name}")
        # Use the canonical release if available
        releases = SAE_RELEASES[model_name]
        if f"{model_name.replace('-', '-scope-')}-pt-res-canonical" in releases:
            release = f"{model_name.replace('-', '-scope-')}-pt-res-canonical"
        else:
            release = list(releases.keys())[0]

    # Auto-detect sae_id if not specified
    if sae_id is None:
        sae_id = f"layer_{layer_idx}/width_16k/canonical"

    # Load SAE via SAELens
    sae, cfg_dict, _ = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=str(device),
    )

    # Extract config
    config = SAEConfig(
        release=release,
        sae_id=sae_id,
        hook_name=cfg_dict.get("hook_name", f"blocks.{layer_idx}.hook_resid_post"),
        d_in=cfg_dict.get("d_in", sae.W_enc.shape[0]),
        d_sae=cfg_dict.get("d_sae", sae.W_enc.shape[1]),
        layer_idx=layer_idx,
    )

    return SAEWrapper(sae, config, device)


def load_sae_for_layer_range(
    model_name: str,
    layer_range: list[int] | None = None,
    release: str | None = None,
    device: str | torch.device = "cuda",
) -> dict[int, SAEWrapper]:
    """
    Load SAEs for multiple layers.

    Args:
        model_name: Model name
        layer_range: List of layer indices (uses default if None)
        release: SAELens release name
        device: Device to load SAEs on

    Returns:
        Dictionary mapping layer index to SAEWrapper
    """
    if layer_range is None:
        layer_range = DEFAULT_SAE_LAYERS.get(model_name, list(range(26)))

    saes = {}
    for layer_idx in layer_range:
        saes[layer_idx] = load_sae(
            model_name=model_name,
            layer_idx=layer_idx,
            release=release,
            device=device,
        )

    return saes


def get_available_sae_releases(model_name: str) -> list[str]:
    """Get list of available SAE releases for a model."""
    return list(SAE_RELEASES.get(model_name, {}).keys())


class ManualSAE(nn.Module):
    """
    Manual SAE implementation for cases where SAELens is not available.

    Implements standard sparse autoencoder: h = ReLU(W_enc @ x + b_enc), x_hat = W_dec @ h + b_dec
    """

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Initialize weights
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


def create_manual_sae_wrapper(
    d_in: int,
    d_sae: int,
    layer_idx: int,
    device: str | torch.device = "cuda",
) -> SAEWrapper:
    """
    Create a manual SAE wrapper (for testing or when SAELens unavailable).

    Args:
        d_in: Input dimension
        d_sae: SAE latent dimension
        layer_idx: Layer index
        device: Device

    Returns:
        SAEWrapper with ManualSAE
    """
    device = torch.device(device)
    sae = ManualSAE(d_in, d_sae).to(device)

    config = SAEConfig(
        release="manual",
        sae_id=f"layer_{layer_idx}",
        hook_name=f"blocks.{layer_idx}.hook_resid_post",
        d_in=d_in,
        d_sae=d_sae,
        layer_idx=layer_idx,
    )

    return SAEWrapper(sae, config, device)
