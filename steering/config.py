"""
Configuration for steering evaluation experiments.

Defines experiment configurations for toxicity mitigation and truthfulness
induction experiments on Gemma-2-2b and Llama-3-8b.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Task(Enum):
    """Evaluation task type."""
    TOXICITY = "toxicity"
    TRUTHFULNESS = "truthfulness"


class SteeringVariant(Enum):
    """Steering variant type."""
    VANILLA = "vanilla"
    WHITENED = "whitened"
    NONE = "none"  # Baseline without steering


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    hf_name: str
    n_layers: int
    hidden_size: int
    sae_release: str | None = None
    default_layer: int | None = None  # Default layer for steering


# Supported model configurations
MODEL_CONFIGS = {
    "gemma-2-2b": ModelConfig(
        name="gemma-2-2b",
        hf_name="google/gemma-2-2b",
        n_layers=26,
        hidden_size=2304,
        sae_release="gemma-scope-2b-pt-res-canonical",
        default_layer=12,  # Middle layer
    ),
    "llama-3-8b": ModelConfig(
        name="llama-3-8b",
        hf_name="meta-llama/Meta-Llama-3-8B",
        n_layers=32,
        hidden_size=4096,
        sae_release=None,  # May need to specify
        default_layer=16,
    ),
    "llama-3.1-8b": ModelConfig(
        name="llama-3.1-8b",
        hf_name="meta-llama/Llama-3.1-8B",
        n_layers=32,
        hidden_size=4096,
        sae_release="llama-3.1-8b-base-res-canonical",
        default_layer=16,
    ),
}


@dataclass
class ToxicityConfig:
    """Configuration for toxicity mitigation experiment."""
    n_eval_prompts: int = 1000
    n_train_per_class: int = 700
    toxic_threshold: float = 0.5
    nontoxic_threshold: float = 0.2
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.9


@dataclass
class TruthfulnessConfig:
    """Configuration for truthfulness induction experiment."""
    # TruthfulQA has 817 questions, use all
    use_preprompt: bool = True


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics."""
    # Toxicity metrics
    compute_cls_toxicity: bool = True
    compute_zeroshot_toxicity: bool = True
    toxicity_judge_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Perplexity metrics
    compute_ppl_wikipedia: bool = True
    n_wikipedia_sentences: int = 20000
    compute_ppl_external: bool = True
    external_scorer_model: str = "mistralai/Mistral-7B-v0.1"

    # MMLU
    compute_mmlu: bool = True
    mmlu_n_shots: int = 5
    mmlu_subset_size: int | None = None  # None = full MMLU


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Model
    model_name: str = "gemma-2-2b"

    # Task
    task: Task = Task.TOXICITY

    # Steering
    steering_variant: SteeringVariant = SteeringVariant.VANILLA
    layer_idx: int | None = None  # None = use model default
    alpha_values: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])

    # Repetitions
    n_seeds: int = 1
    base_seed: int = 42

    # Task-specific configs
    toxicity_config: ToxicityConfig = field(default_factory=ToxicityConfig)
    truthfulness_config: TruthfulnessConfig = field(default_factory=TruthfulnessConfig)
    eval_config: EvalConfig = field(default_factory=EvalConfig)

    # Computation
    device: str = "cuda"
    batch_size: int = 8
    dtype: str = "bfloat16"

    # Output
    output_dir: str = "outputs/steering"
    save_intermediate: bool = True

    def get_model_config(self) -> ModelConfig:
        """Get the model configuration."""
        return MODEL_CONFIGS[self.model_name]

    def get_layer_idx(self) -> int:
        """Get the layer index for steering."""
        if self.layer_idx is not None:
            return self.layer_idx
        return self.get_model_config().default_layer or 0


def create_toxicity_experiment_config(
    model_name: str = "gemma-2-2b",
    steering_variant: str = "vanilla",
    alpha_values: list[float] | None = None,
    n_seeds: int = 1,
) -> ExperimentConfig:
    """Create configuration for toxicity mitigation experiment."""
    if alpha_values is None:
        alpha_values = [0, 1, 5, 8, 10, 20]

    return ExperimentConfig(
        model_name=model_name,
        task=Task.TOXICITY,
        steering_variant=SteeringVariant(steering_variant),
        alpha_values=alpha_values,
        n_seeds=n_seeds,
    )


def create_truthfulness_experiment_config(
    model_name: str = "gemma-2-2b",
    steering_variant: str = "vanilla",
    alpha_values: list[float] | None = None,
    n_seeds: int = 1,
) -> ExperimentConfig:
    """Create configuration for truthfulness induction experiment."""
    if alpha_values is None:
        alpha_values = [0, 1, 5, 8, 10, 20]

    return ExperimentConfig(
        model_name=model_name,
        task=Task.TRUTHFULNESS,
        steering_variant=SteeringVariant(steering_variant),
        alpha_values=alpha_values,
        n_seeds=n_seeds,
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert config to dictionary for serialization."""
    return {
        "model_name": config.model_name,
        "task": config.task.value,
        "steering_variant": config.steering_variant.value,
        "layer_idx": config.get_layer_idx(),
        "alpha_values": config.alpha_values,
        "n_seeds": config.n_seeds,
        "base_seed": config.base_seed,
        "device": config.device,
        "batch_size": config.batch_size,
    }
