"""
Toxicity data loading for steering evaluation.

Provides:
- RealToxicityPrompts for generation evaluation (1000 prompts)
- Toxic/non-toxic sentence pairs for steering vector training
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


@dataclass
class ToxicitySample:
    """A single toxicity sample with prompt and optional continuation."""
    prompt: str
    continuation: str | None = None
    toxicity_score: float | None = None
    is_toxic: bool | None = None


class ToxicityDataset(Dataset):
    """Dataset wrapper for toxicity samples."""

    def __init__(self, samples: list[ToxicitySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ToxicitySample:
        return self.samples[idx]

    def get_prompts(self) -> list[str]:
        return [s.prompt for s in self.samples]


def load_real_toxicity_prompts(
    n_prompts: int = 1000,
    seed: int = 42,
    min_prompt_toxicity: float | None = None,
    max_prompt_toxicity: float | None = None,
) -> ToxicityDataset:
    """
    Load prompts from RealToxicityPrompts dataset for generation evaluation.

    Args:
        n_prompts: Number of prompts to sample (default 1000 as in paper)
        seed: Random seed for reproducible sampling
        min_prompt_toxicity: Minimum toxicity score for prompt filtering
        max_prompt_toxicity: Maximum toxicity score for prompt filtering

    Returns:
        ToxicityDataset with sampled prompts
    """
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")

    # Filter by toxicity if specified
    if min_prompt_toxicity is not None or max_prompt_toxicity is not None:
        def filter_fn(example):
            toxicity = example.get("prompt", {}).get("toxicity")
            if toxicity is None:
                return False
            if min_prompt_toxicity is not None and toxicity < min_prompt_toxicity:
                return False
            if max_prompt_toxicity is not None and toxicity > max_prompt_toxicity:
                return False
            return True
        dataset = dataset.filter(filter_fn)

    # Shuffle and select
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(n_prompts, len(dataset))))

    samples = []
    for item in dataset:
        prompt_data = item.get("prompt", {})
        prompt_text = prompt_data.get("text", "")
        toxicity = prompt_data.get("toxicity")

        samples.append(ToxicitySample(
            prompt=prompt_text,
            toxicity_score=toxicity,
        ))

    return ToxicityDataset(samples)


def load_toxic_nontoxic_pairs(
    n_per_class: int = 700,
    seed: int = 42,
    toxic_threshold: float = 0.5,
    nontoxic_threshold: float = 0.2,
) -> tuple[list[str], list[str]]:
    """
    Load toxic and non-toxic sentence pairs for steering vector training.

    Uses the continuation toxicity scores from RealToxicityPrompts to create
    two sets of sentences: one clearly toxic, one clearly non-toxic.

    Args:
        n_per_class: Number of samples per class (paper uses ~700)
        seed: Random seed for reproducible sampling
        toxic_threshold: Minimum toxicity score to be considered toxic
        nontoxic_threshold: Maximum toxicity score to be considered non-toxic

    Returns:
        Tuple of (toxic_sentences, nontoxic_sentences)
    """
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")

    toxic_sentences = []
    nontoxic_sentences = []

    for item in dataset:
        # Use continuation for toxicity classification
        continuation_data = item.get("continuation", {})
        if not continuation_data:
            continue

        text = continuation_data.get("text", "")
        toxicity = continuation_data.get("toxicity")

        if not text or toxicity is None:
            continue

        # Also include prompt for context
        prompt_data = item.get("prompt", {})
        prompt_text = prompt_data.get("text", "")
        full_text = prompt_text + " " + text if prompt_text else text

        if toxicity >= toxic_threshold:
            toxic_sentences.append(full_text)
        elif toxicity <= nontoxic_threshold:
            nontoxic_sentences.append(full_text)

    # Shuffle and select
    rng = torch.Generator().manual_seed(seed)
    toxic_indices = torch.randperm(len(toxic_sentences), generator=rng).tolist()
    nontoxic_indices = torch.randperm(len(nontoxic_sentences), generator=rng).tolist()

    toxic_sentences = [toxic_sentences[i] for i in toxic_indices[:n_per_class]]
    nontoxic_sentences = [nontoxic_sentences[i] for i in nontoxic_indices[:n_per_class]]

    return toxic_sentences, nontoxic_sentences


def iterate_toxic_nontoxic_batches(
    toxic_sentences: list[str],
    nontoxic_sentences: list[str],
    batch_size: int = 32,
) -> Iterator[tuple[list[str], list[str]]]:
    """
    Iterate over paired batches of toxic and non-toxic sentences.

    Yields batches of equal size from both classes for computing
    difference-in-means steering vectors.
    """
    n = min(len(toxic_sentences), len(nontoxic_sentences))
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        yield toxic_sentences[i:end], nontoxic_sentences[i:end]
