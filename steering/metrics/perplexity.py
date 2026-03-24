"""
Perplexity evaluation metrics.

Provides:
- PPL Wikipedia: Perplexity on Wikipedia sentences using the intervened model
- PPL External: Perplexity of generated text scored by an external model (e.g., Mistral-7B)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PerplexityResult:
    """Result of perplexity evaluation."""
    perplexity: float
    mean_loss: float
    n_tokens: int
    n_samples: int


@torch.no_grad()
def compute_perplexity(
    model,  # PreTrainedModel
    tokenizer,  # PreTrainedTokenizer
    texts: list[str],
    device: torch.device = torch.device("cuda"),
    batch_size: int = 8,
    max_length: int = 512,
    show_progress: bool = True,
) -> PerplexityResult:
    """
    Compute perplexity of texts using a language model.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of texts to evaluate
        device: Device for computation
        batch_size: Batch size
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar

    Returns:
        PerplexityResult with perplexity and statistics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing perplexity", leave=False)

    for i in iterator:
        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Compute loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        # Get per-token loss (outputs.loss is mean over all tokens)
        # We need to recompute to handle padding correctly
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        # Compute cross-entropy loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss = loss.view(shift_labels.size())

        # Mask padding and sum
        masked_loss = loss * shift_mask
        batch_loss = masked_loss.sum().item()
        batch_tokens = shift_mask.sum().item()

        total_loss += batch_loss
        total_tokens += batch_tokens

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)

    return PerplexityResult(
        perplexity=perplexity,
        mean_loss=mean_loss,
        n_tokens=int(total_tokens),
        n_samples=len(texts),
    )


@torch.no_grad()
def compute_perplexity_external(
    texts: list[str],
    scorer_model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str | torch.device = "cuda",
    batch_size: int = 4,
    max_length: int = 512,
    show_progress: bool = True,
) -> PerplexityResult:
    """
    Compute perplexity of texts using an external scoring model.

    This is used for PPL Mistral-7B in the paper: generated texts are
    scored by Mistral-7B to measure fluency/quality.

    Args:
        texts: List of texts to evaluate (e.g., generated continuations)
        scorer_model_name: HuggingFace model name for scoring
        device: Device for computation
        batch_size: Batch size
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar

    Returns:
        PerplexityResult with perplexity scored by external model
    """
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(scorer_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        scorer_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    return compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=show_progress,
    )


def compute_perplexity_with_loaded_scorer(
    texts: list[str],
    model,  # PreTrainedModel
    tokenizer,  # PreTrainedTokenizer
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 512,
    show_progress: bool = True,
) -> PerplexityResult:
    """
    Compute perplexity with a pre-loaded scoring model.

    Same as compute_perplexity_external but accepts pre-loaded model/tokenizer
    to avoid reloading for multiple evaluations.
    """
    return compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=show_progress,
    )


class PerplexityScorer:
    """
    Reusable perplexity scorer with cached model.

    Useful when computing perplexity multiple times (e.g., across different
    steering configurations).
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

    @torch.no_grad()
    def score(
        self,
        texts: list[str],
        batch_size: int = 4,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> PerplexityResult:
        """Compute perplexity of texts."""
        return compute_perplexity(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=texts,
            device=self.device,
            batch_size=batch_size,
            max_length=max_length,
            show_progress=show_progress,
        )


def compute_wikipedia_perplexity(
    model,  # PreTrainedModel
    tokenizer,  # PreTrainedTokenizer
    n_sentences: int = 20000,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
    batch_size: int = 8,
    show_progress: bool = True,
) -> PerplexityResult:
    """
    Compute perplexity on Wikipedia sentences.

    Convenience function that loads Wikipedia data and computes perplexity.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer
        n_sentences: Number of Wikipedia sentences
        seed: Random seed for sentence selection
        device: Device for computation
        batch_size: Batch size
        show_progress: Whether to show progress

    Returns:
        PerplexityResult on Wikipedia
    """
    from steering.data.wikipedia import load_wikipedia_sentences_simple

    wiki_data = load_wikipedia_sentences_simple(n_sentences=n_sentences, seed=seed)
    texts = wiki_data.sentences

    return compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
    )
