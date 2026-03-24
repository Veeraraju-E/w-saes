"""
LLM wrapper with activation hooks for steering evaluation.

Supports Gemma-2-2b and Llama-3-8b with hooks for activation extraction
and intervention at specified layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


SUPPORTED_MODELS = {
    "gemma-2-2b": "google/gemma-2-2b",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}

# Layer name patterns for different model architectures
LAYER_PATTERNS = {
    "gemma": {
        "residual_pre": "model.layers.{layer}.input_layernorm",
        "residual_post": "model.layers.{layer}",
        "attention": "model.layers.{layer}.self_attn",
        "mlp": "model.layers.{layer}.mlp",
        "post_attention_ln": "model.layers.{layer}.post_attention_layernorm",
        "post_feedforward_ln": "model.layers.{layer}.post_feedforward_layernorm",
    },
    "llama": {
        "residual_pre": "model.layers.{layer}.input_layernorm",
        "residual_post": "model.layers.{layer}",
        "attention": "model.layers.{layer}.self_attn",
        "mlp": "model.layers.{layer}.mlp",
        "post_attention_ln": "model.layers.{layer}.post_attention_layernorm",
    },
}


@dataclass
class ActivationCache:
    """Cache for storing activations during forward pass."""
    activations: dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self):
        self.activations.clear()

    def add(self, name: str, activation: torch.Tensor):
        self.activations[name] = activation.detach().clone()

    def get(self, name: str) -> torch.Tensor | None:
        return self.activations.get(name)


class LLMWrapper:
    """
    Wrapper for LLMs with activation hooks for steering.

    Provides:
    - Activation extraction at specified layers
    - Intervention hooks for modifying activations during generation
    - Generation with or without intervention
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.cache = ActivationCache()
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._intervention_fn: Callable[[str, torch.Tensor], torch.Tensor] | None = None

        # Determine architecture
        if "gemma" in model_name.lower():
            self.arch = "gemma"
            self.n_layers = model.config.num_hidden_layers
        elif "llama" in model_name.lower():
            self.arch = "llama"
            self.n_layers = model.config.num_hidden_layers
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def get_layer_name(self, layer_idx: int, layer_type: str = "residual_post") -> str:
        """Get the module name for a specific layer and type."""
        pattern = LAYER_PATTERNS[self.arch].get(layer_type)
        if pattern is None:
            raise ValueError(f"Unknown layer type: {layer_type} for {self.arch}")
        return pattern.format(layer=layer_idx)

    def _get_module(self, name: str) -> nn.Module:
        """Get a module by its name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module

    def register_activation_hook(
        self,
        layer_idx: int,
        layer_type: str = "residual_post",
    ):
        """
        Register a hook to capture activations at a specific layer.

        Args:
            layer_idx: Layer index (0-indexed)
            layer_type: Type of layer (residual_post, attention, mlp, etc.)
        """
        name = self.get_layer_name(layer_idx, layer_type)
        module = self._get_module(name)

        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            self.cache.add(name, activation)

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def register_intervention_hook(
        self,
        layer_idx: int,
        layer_type: str = "residual_post",
    ):
        """
        Register a hook to intervene on activations at a specific layer.

        The intervention function must be set via set_intervention_fn before forward pass.

        Args:
            layer_idx: Layer index (0-indexed)
            layer_type: Type of layer (residual_post, attention, mlp, etc.)
        """
        name = self.get_layer_name(layer_idx, layer_type)
        module = self._get_module(name)

        def hook_fn(module, input, output):
            if self._intervention_fn is None:
                return output

            # Handle different output types
            if isinstance(output, tuple):
                activation = output[0]
                modified = self._intervention_fn(name, activation)
                return (modified,) + output[1:]
            else:
                return self._intervention_fn(name, output)

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def set_intervention_fn(
        self,
        fn: Callable[[str, torch.Tensor], torch.Tensor] | None,
    ):
        """
        Set the intervention function for modifying activations.

        Args:
            fn: Function that takes (layer_name, activation) and returns modified activation.
                Set to None to disable intervention.
        """
        self._intervention_fn = fn

    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._intervention_fn = None

    def clear_cache(self):
        """Clear the activation cache."""
        self.cache.clear()

    @torch.no_grad()
    def get_activations(
        self,
        texts: list[str],
        layer_idx: int,
        layer_type: str = "residual_post",
        pooling: str = "mean",
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Extract activations for a batch of texts at a specific layer.

        Args:
            texts: List of input texts
            layer_idx: Layer index to extract from
            layer_type: Type of layer
            pooling: How to pool across tokens ("mean", "last", "max")
            batch_size: Batch size for processing

        Returns:
            Tensor of shape (n_texts, hidden_size)
        """
        self.clear_hooks()
        self.register_activation_hook(layer_idx, layer_type)
        layer_name = self.get_layer_name(layer_idx, layer_type)

        all_activations = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            self.clear_cache()
            self.model(**inputs)

            activation = self.cache.get(layer_name)
            if activation is None:
                raise RuntimeError(f"Failed to capture activation at {layer_name}")

            # Pool across sequence dimension
            attention_mask = inputs["attention_mask"]
            if pooling == "mean":
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()
                pooled = (activation * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            elif pooling == "last":
                # Get last non-padding token
                seq_lens = attention_mask.sum(dim=1) - 1
                pooled = activation[torch.arange(activation.size(0)), seq_lens]
            elif pooling == "max":
                # Masked max pooling
                activation = activation.masked_fill(~attention_mask.unsqueeze(-1).bool(), float("-inf"))
                pooled = activation.max(dim=1).values
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

            all_activations.append(pooled)

        self.clear_hooks()
        return torch.cat(all_activations, dim=0)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        batch_size: int = 8,
    ) -> list[str]:
        """
        Generate continuations for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to sample or use greedy decoding
            batch_size: Batch size for processing

        Returns:
            List of generated continuations (without the prompt)
        """
        all_continuations = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            prompt_lengths = inputs["attention_mask"].sum(dim=1)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

            # Extract only the generated part
            for j, (output, prompt_len) in enumerate(zip(outputs, prompt_lengths)):
                continuation = self.tokenizer.decode(
                    output[prompt_len:],
                    skip_special_tokens=True,
                )
                all_continuations.append(continuation)

        return all_continuations

    @torch.no_grad()
    def compute_log_likelihood(
        self,
        contexts: list[str],
        continuations: list[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Compute log-likelihood of continuations given contexts.

        Used for MC1/MC2 evaluation in TruthfulQA.

        Args:
            contexts: List of context strings (questions with preprompt)
            continuations: List of continuation strings (answer choices)
            batch_size: Batch size for processing

        Returns:
            Tensor of log-likelihoods, shape (n_samples,)
        """
        all_log_probs = []

        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i + batch_size]
            batch_continuations = continuations[i:i + batch_size]

            # Tokenize context and full sequence separately
            context_encodings = self.tokenizer(
                batch_contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            context_lengths = context_encodings["attention_mask"].sum(dim=1)

            full_texts = [c + cont for c, cont in zip(batch_contexts, batch_continuations)]
            full_encodings = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            outputs = self.model(**full_encodings)
            logits = outputs.logits

            # Compute log probabilities for continuation tokens only
            log_probs = torch.log_softmax(logits, dim=-1)

            batch_log_probs = []
            for j in range(len(batch_contexts)):
                ctx_len = context_lengths[j].item()
                seq_len = full_encodings["attention_mask"][j].sum().item()

                # Get log probs for continuation tokens
                # logits[t] predicts token[t+1], so we start from ctx_len-1
                cont_log_probs = []
                for t in range(ctx_len - 1, seq_len - 1):
                    next_token = full_encodings["input_ids"][j, t + 1]
                    token_log_prob = log_probs[j, t, next_token]
                    cont_log_probs.append(token_log_prob)

                if cont_log_probs:
                    total_log_prob = torch.stack(cont_log_probs).sum()
                else:
                    total_log_prob = torch.tensor(0.0, device=self.device)

                batch_log_probs.append(total_log_prob)

            all_log_probs.extend(batch_log_probs)

        return torch.stack(all_log_probs)


def load_llm(
    model_name: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> LLMWrapper:
    """
    Load an LLM with the wrapper for steering evaluation.

    Args:
        model_name: Short name (e.g., "gemma-2-2b") or full HuggingFace path
        device: Device to load model on
        dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code

    Returns:
        LLMWrapper instance
    """
    if model_name in SUPPORTED_MODELS:
        hf_name = SUPPORTED_MODELS[model_name]
    else:
        hf_name = model_name

    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    return LLMWrapper(model, tokenizer, model_name, device)
