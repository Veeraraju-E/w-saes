"""
MMLU (Massive Multitask Language Understanding) evaluation.

Provides 5-shot MMLU accuracy evaluation as a utility metric
to ensure steering doesn't degrade general model capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from datasets import load_dataset
from tqdm import tqdm


# MMLU subject categories
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


@dataclass
class MMLUResult:
    """Result of MMLU evaluation."""
    accuracy: float
    n_correct: int
    n_total: int
    per_subject_accuracy: dict[str, float]


@dataclass
class MMLUQuestion:
    """A single MMLU question."""
    question: str
    choices: list[str]
    answer_idx: int
    subject: str


def format_mmlu_prompt(
    question: str,
    choices: list[str],
    few_shot_examples: list[tuple[str, list[str], int]] | None = None,
) -> str:
    """
    Format an MMLU question with optional few-shot examples.

    Args:
        question: The question text
        choices: List of answer choices (A, B, C, D)
        few_shot_examples: List of (question, choices, answer_idx) tuples

    Returns:
        Formatted prompt string
    """
    prompt_parts = []

    # Add few-shot examples
    if few_shot_examples:
        for ex_q, ex_choices, ex_ans in few_shot_examples:
            prompt_parts.append(f"Question: {ex_q}")
            for i, choice in enumerate(ex_choices):
                letter = chr(ord('A') + i)
                prompt_parts.append(f"{letter}. {choice}")
            answer_letter = chr(ord('A') + ex_ans)
            prompt_parts.append(f"Answer: {answer_letter}\n")

    # Add the actual question
    prompt_parts.append(f"Question: {question}")
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        prompt_parts.append(f"{letter}. {choice}")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def load_mmlu_dataset(
    subjects: list[str] | None = None,
    split: str = "test",
) -> list[MMLUQuestion]:
    """
    Load MMLU dataset.

    Args:
        subjects: List of subjects to load (all if None)
        split: Dataset split ("test", "validation", "dev")

    Returns:
        List of MMLUQuestion objects
    """
    if subjects is None:
        subjects = MMLU_SUBJECTS

    questions = []

    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)

        for item in dataset:
            questions.append(MMLUQuestion(
                question=item["question"],
                choices=item["choices"],
                answer_idx=item["answer"],
                subject=subject,
            ))

    return questions


def load_mmlu_few_shot_examples(
    subject: str,
    n_shots: int = 5,
) -> list[tuple[str, list[str], int]]:
    """
    Load few-shot examples for a subject from the dev set.

    Args:
        subject: MMLU subject
        n_shots: Number of examples

    Returns:
        List of (question, choices, answer_idx) tuples
    """
    dataset = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=True)

    examples = []
    for i, item in enumerate(dataset):
        if i >= n_shots:
            break
        examples.append((
            item["question"],
            item["choices"],
            item["answer"],
        ))

    return examples


@torch.no_grad()
def evaluate_mmlu(
    model,  # PreTrainedModel
    tokenizer,  # PreTrainedTokenizer
    subjects: list[str] | None = None,
    n_shots: int = 5,
    device: torch.device = torch.device("cuda"),
    show_progress: bool = True,
) -> MMLUResult:
    """
    Evaluate model on MMLU with few-shot prompting.

    Args:
        model: Language model
        tokenizer: Tokenizer
        subjects: List of subjects (all if None)
        n_shots: Number of few-shot examples
        device: Device for computation
        show_progress: Whether to show progress bar

    Returns:
        MMLUResult with accuracy and per-subject breakdown
    """
    model.eval()

    if subjects is None:
        subjects = MMLU_SUBJECTS

    # Cache few-shot examples per subject
    few_shot_cache = {}

    total_correct = 0
    total_questions = 0
    per_subject_correct = {s: 0 for s in subjects}
    per_subject_total = {s: 0 for s in subjects}

    # Load all questions
    questions = load_mmlu_dataset(subjects=subjects, split="test")

    iterator = questions
    if show_progress:
        iterator = tqdm(iterator, desc="MMLU evaluation", leave=False)

    # Get token IDs for A, B, C, D
    answer_tokens = [
        tokenizer.encode(letter, add_special_tokens=False)[0]
        for letter in ["A", "B", "C", "D"]
    ]

    for q in iterator:
        # Get few-shot examples (cached)
        if q.subject not in few_shot_cache:
            few_shot_cache[q.subject] = load_mmlu_few_shot_examples(
                q.subject, n_shots=n_shots
            )

        few_shot = few_shot_cache[q.subject]

        # Format prompt
        prompt = format_mmlu_prompt(q.question, q.choices, few_shot)

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        # Get model output
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits

        # Get logits for answer tokens
        answer_logits = logits[answer_tokens]
        predicted_idx = int(torch.argmax(answer_logits).item())

        # Check correctness
        if predicted_idx == q.answer_idx:
            total_correct += 1
            per_subject_correct[q.subject] += 1

        total_questions += 1
        per_subject_total[q.subject] += 1

    # Compute accuracies
    accuracy = total_correct / total_questions if total_questions > 0 else 0.0
    per_subject_accuracy = {
        s: per_subject_correct[s] / per_subject_total[s]
        if per_subject_total[s] > 0 else 0.0
        for s in subjects
    }

    return MMLUResult(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=total_questions,
        per_subject_accuracy=per_subject_accuracy,
    )


def evaluate_mmlu_subset(
    model,  # PreTrainedModel
    tokenizer,  # PreTrainedTokenizer
    n_questions: int = 1000,
    n_shots: int = 5,
    seed: int = 42,
    device: torch.device = torch.device("cuda"),
    show_progress: bool = True,
) -> MMLUResult:
    """
    Evaluate on a random subset of MMLU for faster evaluation.

    Args:
        model: Language model
        tokenizer: Tokenizer
        n_questions: Number of questions to sample
        n_shots: Number of few-shot examples
        seed: Random seed
        device: Device for computation
        show_progress: Whether to show progress

    Returns:
        MMLUResult on subset
    """
    import random

    # Load all questions
    all_questions = load_mmlu_dataset(split="test")

    # Sample subset
    rng = random.Random(seed)
    questions = rng.sample(all_questions, min(n_questions, len(all_questions)))

    # Get unique subjects in sample
    subjects = list(set(q.subject for q in questions))

    # Cache few-shot examples
    few_shot_cache = {
        s: load_mmlu_few_shot_examples(s, n_shots=n_shots)
        for s in subjects
    }

    model.eval()

    total_correct = 0
    per_subject_correct = {s: 0 for s in subjects}
    per_subject_total = {s: 0 for s in subjects}

    iterator = questions
    if show_progress:
        iterator = tqdm(iterator, desc="MMLU evaluation", leave=False)

    answer_tokens = [
        tokenizer.encode(letter, add_special_tokens=False)[0]
        for letter in ["A", "B", "C", "D"]
    ]

    with torch.no_grad():
        for q in iterator:
            few_shot = few_shot_cache[q.subject]
            prompt = format_mmlu_prompt(q.question, q.choices, few_shot)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            answer_logits = logits[answer_tokens]
            predicted_idx = int(torch.argmax(answer_logits).item())

            if predicted_idx == q.answer_idx:
                total_correct += 1
                per_subject_correct[q.subject] += 1

            per_subject_total[q.subject] += 1

    accuracy = total_correct / len(questions)
    per_subject_accuracy = {
        s: per_subject_correct[s] / per_subject_total[s]
        if per_subject_total[s] > 0 else 0.0
        for s in subjects
    }

    return MMLUResult(
        accuracy=accuracy,
        n_correct=total_correct,
        n_total=len(questions),
        per_subject_accuracy=per_subject_accuracy,
    )
