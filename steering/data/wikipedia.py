"""
Wikipedia data loading for perplexity evaluation.

Provides a fixed set of Wikipedia sentences for measuring model perplexity
as a utility metric (PPL Wikipedia in the paper).
"""

from __future__ import annotations

from datasets import load_dataset
from torch.utils.data import Dataset


class WikipediaDataset(Dataset):
    """Dataset wrapper for Wikipedia sentences."""

    def __init__(self, sentences: list[str]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


def load_wikipedia_sentences(
    n_sentences: int = 20000,
    seed: int = 42,
    min_length: int = 50,
    max_length: int = 500,
    language: str = "en",
) -> WikipediaDataset:
    """
    Load a fixed set of Wikipedia sentences for perplexity evaluation.

    Uses the Wikipedia dataset from HuggingFace, extracting sentences
    that meet length criteria.

    Args:
        n_sentences: Number of sentences to collect (default 20k as in paper)
        seed: Random seed for reproducible sampling
        min_length: Minimum character length for sentences
        max_length: Maximum character length for sentences
        language: Wikipedia language code

    Returns:
        WikipediaDataset with sampled sentences
    """
    # Load Wikipedia dataset (streaming to avoid downloading full dataset)
    dataset = load_dataset(
        "wikipedia",
        f"20220301.{language}",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # Shuffle with seed for reproducibility
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    sentences = []
    seen_sentences = set()

    for article in dataset:
        text = article.get("text", "")
        if not text:
            continue

        # Split into sentences (simple heuristic)
        for sentence in _split_into_sentences(text):
            # Filter by length
            if len(sentence) < min_length or len(sentence) > max_length:
                continue

            # Skip duplicates
            if sentence in seen_sentences:
                continue

            seen_sentences.add(sentence)
            sentences.append(sentence)

            if len(sentences) >= n_sentences:
                break

        if len(sentences) >= n_sentences:
            break

    return WikipediaDataset(sentences)


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using simple heuristics.

    This is a basic sentence splitter that handles common cases.
    For production use, consider using a proper sentence tokenizer.
    """
    # Replace common abbreviations to avoid false splits
    text = text.replace("Mr.", "Mr")
    text = text.replace("Mrs.", "Mrs")
    text = text.replace("Dr.", "Dr")
    text = text.replace("Prof.", "Prof")
    text = text.replace("Jr.", "Jr")
    text = text.replace("Sr.", "Sr")
    text = text.replace("vs.", "vs")
    text = text.replace("etc.", "etc")
    text = text.replace("i.e.", "ie")
    text = text.replace("e.g.", "eg")

    sentences = []
    current = []

    for char in text:
        current.append(char)
        if char in ".!?":
            sentence = "".join(current).strip()
            if sentence and len(sentence) > 10:
                # Restore abbreviations
                sentence = sentence.replace("Mr", "Mr.")
                sentence = sentence.replace("Mrs", "Mrs.")
                sentence = sentence.replace("Dr", "Dr.")
                sentence = sentence.replace("Prof", "Prof.")
                sentence = sentence.replace("Jr", "Jr.")
                sentence = sentence.replace("Sr", "Sr.")
                sentence = sentence.replace("vs", "vs.")
                sentence = sentence.replace("etc", "etc.")
                sentence = sentence.replace("ie", "i.e.")
                sentence = sentence.replace("eg", "e.g.")
                sentences.append(sentence)
            current = []

    return sentences


def load_wikipedia_sentences_simple(
    n_sentences: int = 20000,
    seed: int = 42,
) -> WikipediaDataset:
    """
    Simplified Wikipedia loading using wikitext dataset.

    This is faster than the full Wikipedia dataset and suitable for
    perplexity evaluation.

    Args:
        n_sentences: Number of text segments to collect
        seed: Random seed for reproducible sampling

    Returns:
        WikipediaDataset with sampled text segments
    """
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")

    # Filter out empty lines and headers
    texts = []
    for item in dataset:
        text = item["text"].strip()
        if not text or text.startswith("="):
            continue
        if len(text) < 50:
            continue
        texts.append(text)

    # Shuffle and select
    import random
    rng = random.Random(seed)
    rng.shuffle(texts)
    texts = texts[:n_sentences]

    return WikipediaDataset(texts)
