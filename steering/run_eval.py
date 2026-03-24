#!/usr/bin/env python3
"""
Main evaluation script for steering experiments.

Runs toxicity mitigation or truthfulness induction experiments with
vanilla or whitened SAE steering on Gemma-2-2b or Llama-3-8b.

Features:
- Automatic checkpointing after each (seed, alpha) combination
- Resume from checkpoint if interrupted
- Saves steering vectors and whitening transforms for reuse
- Incremental result saving

Usage:
    python -m steering.run_eval --model gemma-2-2b --task toxicity --variant vanilla
    python -m steering.run_eval --model llama-3-8b --task truthfulness --variant whitened
    python -m steering.run_eval --resume outputs/steering/run_name  # Resume interrupted run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from steering.config import (
    ExperimentConfig,
    Task,
    SteeringVariant,
    create_toxicity_experiment_config,
    create_truthfulness_experiment_config,
    config_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run steering evaluation experiments")

    # Model and task
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-2-2b",
        choices=["gemma-2-2b", "llama-3-8b", "llama-3.1-8b"],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="toxicity",
        choices=["toxicity", "truthfulness"],
        help="Evaluation task",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="vanilla",
        choices=["vanilla", "whitened", "none"],
        help="Steering variant",
    )

    # Steering parameters
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index for steering (default: model-specific)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help="Steering strength values to sweep",
    )

    # Experiment parameters
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/steering",
        help="Output directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name (default: auto-generated)",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to run directory to resume from",
    )

    # Evaluation options
    parser.add_argument("--skip-mmlu", action="store_true", help="Skip MMLU evaluation")
    parser.add_argument("--skip-ppl", action="store_true", help="Skip perplexity evaluation")
    parser.add_argument("--skip-zeroshot", action="store_true", help="Skip 0-shot toxicity")

    # Debug options
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


def setup_experiment(args: argparse.Namespace) -> ExperimentConfig:
    """Create experiment configuration from arguments."""
    if args.task == "toxicity":
        config = create_toxicity_experiment_config(
            model_name=args.model,
            steering_variant=args.variant,
            alpha_values=args.alpha,
            n_seeds=args.n_seeds,
        )
    else:
        config = create_truthfulness_experiment_config(
            model_name=args.model,
            steering_variant=args.variant,
            alpha_values=args.alpha,
            n_seeds=args.n_seeds,
        )

    config.layer_idx = args.layer
    config.base_seed = args.base_seed
    config.batch_size = args.batch_size
    config.device = args.device
    config.output_dir = args.output_dir

    # Update eval config based on flags
    config.eval_config.compute_mmlu = not args.skip_mmlu
    config.eval_config.compute_ppl_wikipedia = not args.skip_ppl
    config.eval_config.compute_ppl_external = not args.skip_ppl
    config.eval_config.compute_zeroshot_toxicity = not args.skip_zeroshot

    return config


class CheckpointManager:
    """Manages checkpoints and intermediate results for experiment resumption."""

    def __init__(self, output_dir: Path, run_name: str):
        self.output_dir = output_dir
        self.run_name = run_name
        self.checkpoint_dir = output_dir / run_name / "checkpoints"
        self.artifacts_dir = output_dir / run_name / "artifacts"
        self.results_dir = output_dir / run_name

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_key(self, seed: int, alpha: float) -> str:
        """Generate a unique key for a (seed, alpha) combination."""
        return f"seed{seed}_alpha{alpha:.4f}"

    def is_completed(self, seed: int, alpha: float) -> bool:
        """Check if a (seed, alpha) combination has been completed."""
        key = self.get_checkpoint_key(seed, alpha)
        checkpoint_file = self.checkpoint_dir / f"{key}.json"
        return checkpoint_file.exists()

    def save_result(self, seed: int, alpha: float, result: dict[str, Any]):
        """Save a single result as a checkpoint."""
        key = self.get_checkpoint_key(seed, alpha)
        checkpoint_file = self.checkpoint_dir / f"{key}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(result, f, indent=2)

    def load_completed_results(self) -> list[dict[str, Any]]:
        """Load all completed results from checkpoints."""
        results = []
        for checkpoint_file in sorted(self.checkpoint_dir.glob("*.json")):
            with open(checkpoint_file) as f:
                results.append(json.load(f))
        return results

    def save_steering_vector(self, seed: int, steering_vector) -> Path:
        """Save steering vector for reuse."""
        path = self.artifacts_dir / f"steering_vector_seed{seed}.pt"
        torch.save({
            "vector": steering_vector.vector,
            "layer_idx": steering_vector.layer_idx,
            "d_sae": steering_vector.d_sae,
            "positive_class": steering_vector.positive_class,
            "negative_class": steering_vector.negative_class,
            "n_positive": steering_vector.n_positive,
            "n_negative": steering_vector.n_negative,
        }, path)
        return path

    def load_steering_vector(self, seed: int):
        """Load steering vector if it exists."""
        from steering.steering.diff_in_means import SteeringVector
        path = self.artifacts_dir / f"steering_vector_seed{seed}.pt"
        if not path.exists():
            return None
        data = torch.load(path, weights_only=False)
        return SteeringVector(
            vector=data["vector"],
            layer_idx=data["layer_idx"],
            d_sae=data["d_sae"],
            positive_class=data["positive_class"],
            negative_class=data["negative_class"],
            n_positive=data["n_positive"],
            n_negative=data["n_negative"],
        )

    def save_whitening(self, seed: int, whitening) -> Path:
        """Save whitening transform for reuse."""
        path = self.artifacts_dir / f"whitening_seed{seed}.pt"
        torch.save({
            "mu": whitening.mu,
            "u_fwd": whitening.u_fwd,
            "u_inv": whitening.u_inv,
            "eigenvalues": whitening.eigenvalues,
        }, path)
        return path

    def load_whitening(self, seed: int):
        """Load whitening transform if it exists."""
        from steering.models.whitening import Whitening
        path = self.artifacts_dir / f"whitening_seed{seed}.pt"
        if not path.exists():
            return None
        data = torch.load(path, weights_only=False)
        return Whitening(
            mu=data["mu"],
            u_fwd=data["u_fwd"],
            u_inv=data["u_inv"],
            eigenvalues=data["eigenvalues"],
        )

    def save_generations(self, seed: int, alpha: float, prompts: list[str], continuations: list[str]):
        """Save generated text for analysis."""
        key = self.get_checkpoint_key(seed, alpha)
        path = self.artifacts_dir / f"generations_{key}.json"
        with open(path, "w") as f:
            json.dump({
                "prompts": prompts,
                "continuations": continuations,
            }, f, indent=2)

    def save_config(self, config: ExperimentConfig):
        """Save experiment configuration."""
        config_path = self.results_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_dict(config), f, indent=2)

    def save_final_results(self, results: list[dict[str, Any]]):
        """Save final aggregated results."""
        # Save as JSON
        json_path = self.results_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save as CSV
        csv_path = self.results_dir / "results.csv"
        if results:
            fieldnames = list(results[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        print(f"\nFinal results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")


def run_toxicity_experiment(
    config: ExperimentConfig,
    checkpoint_mgr: CheckpointManager,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """
    Run toxicity mitigation experiment with checkpointing.

    Returns list of result dictionaries, one per (alpha, seed) combination.
    """
    from steering.models.llm import load_llm
    from steering.models.sae import load_sae
    from steering.data.toxicity import load_real_toxicity_prompts, load_toxic_nontoxic_pairs
    from steering.steering.diff_in_means import compute_diff_in_means_vector
    from steering.steering.intervention import (
        VanillaSteering,
        WhitenedSteering,
        setup_vanilla_steering,
        setup_whitened_steering,
        create_steering_hook,
        collect_training_latents,
    )
    from steering.metrics.toxicity import ToxicityClassifier, compute_cls_toxicity
    from steering.models.whitening import fit_whitening

    device = torch.device(config.device)
    layer_idx = config.get_layer_idx()

    # Load any previously completed results
    results = checkpoint_mgr.load_completed_results()
    n_completed = len(results)
    if n_completed > 0:
        print(f"Resuming: found {n_completed} completed checkpoints")

    print(f"Loading model: {config.model_name}")
    llm = load_llm(config.model_name, device=config.device)

    print(f"Loading SAE for layer {layer_idx}")
    model_config = config.get_model_config()
    sae = load_sae(
        model_name=config.model_name,
        layer_idx=layer_idx,
        release=model_config.sae_release,
        device=config.device,
    )

    print("Loading toxicity classifier")
    toxicity_classifier = ToxicityClassifier(device=config.device)

    print("Loading evaluation prompts")
    eval_data = load_real_toxicity_prompts(
        n_prompts=config.toxicity_config.n_eval_prompts,
        seed=config.base_seed,
    )
    eval_prompts = eval_data.get_prompts()

    print("Loading training data for steering vector")
    toxic_texts, nontoxic_texts = load_toxic_nontoxic_pairs(
        n_per_class=config.toxicity_config.n_train_per_class,
        seed=config.base_seed,
    )

    seed_iter = tqdm(
        range(config.n_seeds),
        desc="Toxicity seeds",
        total=config.n_seeds,
    )
    for seed_offset in seed_iter:
        seed = config.base_seed + seed_offset

        # Check if all alphas for this seed are done
        all_done = all(
            checkpoint_mgr.is_completed(seed, alpha)
            for alpha in config.alpha_values
        )
        if all_done:
            print(f"\n=== Seed {seed} - already completed, skipping ===")
            continue

        print(f"\n=== Seed {seed} ({seed_offset + 1}/{config.n_seeds}) ===")

        # Try to load cached steering vector
        steering_vector = checkpoint_mgr.load_steering_vector(seed)
        if steering_vector is not None:
            print("Loaded cached steering vector")
        else:
            print("Computing steering vector...")
            steering_vector = compute_diff_in_means_vector(
                llm=llm,
                sae=sae,
                positive_texts=nontoxic_texts,
                negative_texts=toxic_texts,
                layer_idx=layer_idx,
                positive_class="non-toxic",
                negative_class="toxic",
                batch_size=config.batch_size,
            )
            checkpoint_mgr.save_steering_vector(seed, steering_vector)
            print("Saved steering vector")

        # Set up steering intervention
        intervention = None
        if config.steering_variant == SteeringVariant.VANILLA:
            intervention = setup_vanilla_steering(sae, steering_vector)
        elif config.steering_variant == SteeringVariant.WHITENED:
            # Try to load cached whitening
            whitening = checkpoint_mgr.load_whitening(seed)
            if whitening is not None:
                print("Loaded cached whitening transform")
                v_whitened = steering_vector.vector @ whitening.u_fwd
                intervention = WhitenedSteering(
                    sae=sae,
                    steering_vector=steering_vector,
                    whitening=whitening,
                    steering_vector_whitened=v_whitened,
                )
            else:
                print("Computing whitening transform...")
                all_texts = toxic_texts + nontoxic_texts
                h_train = collect_training_latents(
                    llm=llm,
                    sae=sae,
                    texts=all_texts,
                    layer_idx=layer_idx,
                    batch_size=config.batch_size,
                )
                whitening = fit_whitening(h_train, eps=1e-5)
                checkpoint_mgr.save_whitening(seed, whitening)
                print("Saved whitening transform")

                v_whitened = steering_vector.vector @ whitening.u_fwd
                intervention = WhitenedSteering(
                    sae=sae,
                    steering_vector=steering_vector,
                    whitening=whitening,
                    steering_vector_whitened=v_whitened,
                )

        layer_name = llm.get_layer_name(layer_idx, "residual_post")

        alpha_iter = tqdm(
            config.alpha_values,
            desc=f"Alpha sweep (seed={seed})",
            total=len(config.alpha_values),
            leave=False,
        )
        for alpha in alpha_iter:
            # Skip if already completed
            if checkpoint_mgr.is_completed(seed, alpha):
                print(f"\n--- Alpha = {alpha} - already completed, skipping ---")
                continue

            print(f"\n--- Alpha = {alpha} ---")

            # Set up intervention hook
            if intervention is not None and alpha > 0:
                hook_fn = create_steering_hook(intervention, alpha, layer_name)
                llm.set_intervention_fn(hook_fn)
                llm.register_intervention_hook(layer_idx, "residual_post")
            else:
                llm.clear_hooks()
                llm.set_intervention_fn(None)

            # Generate continuations
            print("Generating continuations...")
            continuations = llm.generate(
                prompts=eval_prompts,
                max_new_tokens=config.toxicity_config.max_new_tokens,
                temperature=config.toxicity_config.temperature,
                top_p=config.toxicity_config.top_p,
                batch_size=config.batch_size,
            )

            # Save generations
            checkpoint_mgr.save_generations(seed, alpha, eval_prompts, continuations)

            # Evaluate toxicity
            print("Evaluating toxicity...")
            tox_result = compute_cls_toxicity(
                texts=continuations,
                classifier=toxicity_classifier,
                batch_size=config.batch_size,
            )

            result = {
                "model": config.model_name,
                "task": "toxicity",
                "variant": config.steering_variant.value,
                "layer": layer_idx,
                "alpha": alpha,
                "seed": seed,
                "cls_toxicity": tox_result.cls_toxicity,
                "cls_toxicity_std": tox_result.cls_toxicity_std,
                "n_samples": tox_result.n_samples,
            }

            # Save checkpoint immediately
            checkpoint_mgr.save_result(seed, alpha, result)
            results.append(result)
            print(f"CLS Toxicity: {tox_result.cls_toxicity:.4f} (checkpointed)")

            # Clean up hooks
            llm.clear_hooks()

    return results


def run_truthfulness_experiment(
    config: ExperimentConfig,
    checkpoint_mgr: CheckpointManager,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """
    Run truthfulness induction experiment with checkpointing.

    Returns list of result dictionaries, one per (alpha, seed) combination.
    """
    from steering.models.llm import load_llm
    from steering.models.sae import load_sae
    from steering.data.truthfulness import load_truthfulqa_mc, load_truthful_untruthful_pairs
    from steering.steering.diff_in_means import compute_diff_in_means_vector
    from steering.steering.intervention import (
        VanillaSteering,
        WhitenedSteering,
        setup_vanilla_steering,
        create_steering_hook,
        collect_training_latents,
    )
    from steering.metrics.truthfulness import evaluate_truthfulqa_with_intervention
    from steering.models.whitening import fit_whitening

    device = torch.device(config.device)
    layer_idx = config.get_layer_idx()

    # Load any previously completed results
    results = checkpoint_mgr.load_completed_results()
    n_completed = len(results)
    if n_completed > 0:
        print(f"Resuming: found {n_completed} completed checkpoints")

    print(f"Loading model: {config.model_name}")
    llm = load_llm(config.model_name, device=config.device)

    print(f"Loading SAE for layer {layer_idx}")
    model_config = config.get_model_config()
    sae = load_sae(
        model_name=config.model_name,
        layer_idx=layer_idx,
        release=model_config.sae_release,
        device=config.device,
    )

    print("Loading TruthfulQA dataset")
    eval_data = load_truthfulqa_mc()

    print("Loading training data for steering vector")
    truthful_texts, untruthful_texts = load_truthful_untruthful_pairs(seed=config.base_seed)

    seed_iter = tqdm(
        range(config.n_seeds),
        desc="Truthfulness seeds",
        total=config.n_seeds,
    )
    for seed_offset in seed_iter:
        seed = config.base_seed + seed_offset

        # Check if all alphas for this seed are done
        all_done = all(
            checkpoint_mgr.is_completed(seed, alpha)
            for alpha in config.alpha_values
        )
        if all_done:
            print(f"\n=== Seed {seed} - already completed, skipping ===")
            continue

        print(f"\n=== Seed {seed} ({seed_offset + 1}/{config.n_seeds}) ===")

        # Try to load cached steering vector
        steering_vector = checkpoint_mgr.load_steering_vector(seed)
        if steering_vector is not None:
            print("Loaded cached steering vector")
        else:
            print("Computing steering vector...")
            steering_vector = compute_diff_in_means_vector(
                llm=llm,
                sae=sae,
                positive_texts=truthful_texts,
                negative_texts=untruthful_texts,
                layer_idx=layer_idx,
                positive_class="truthful",
                negative_class="untruthful",
                batch_size=config.batch_size,
            )
            checkpoint_mgr.save_steering_vector(seed, steering_vector)
            print("Saved steering vector")

        # Set up steering intervention
        intervention = None
        if config.steering_variant == SteeringVariant.VANILLA:
            intervention = setup_vanilla_steering(sae, steering_vector)
        elif config.steering_variant == SteeringVariant.WHITENED:
            # Try to load cached whitening
            whitening = checkpoint_mgr.load_whitening(seed)
            if whitening is not None:
                print("Loaded cached whitening transform")
                v_whitened = steering_vector.vector @ whitening.u_fwd
                intervention = WhitenedSteering(
                    sae=sae,
                    steering_vector=steering_vector,
                    whitening=whitening,
                    steering_vector_whitened=v_whitened,
                )
            else:
                print("Computing whitening transform...")
                all_texts = truthful_texts + untruthful_texts
                h_train = collect_training_latents(
                    llm=llm,
                    sae=sae,
                    texts=all_texts,
                    layer_idx=layer_idx,
                    batch_size=config.batch_size,
                )
                whitening = fit_whitening(h_train, eps=1e-5)
                checkpoint_mgr.save_whitening(seed, whitening)
                print("Saved whitening transform")

                v_whitened = steering_vector.vector @ whitening.u_fwd
                intervention = WhitenedSteering(
                    sae=sae,
                    steering_vector=steering_vector,
                    whitening=whitening,
                    steering_vector_whitened=v_whitened,
                )

        layer_name = llm.get_layer_name(layer_idx, "residual_post")

        alpha_iter = tqdm(
            config.alpha_values,
            desc=f"Alpha sweep (seed={seed})",
            total=len(config.alpha_values),
            leave=False,
        )
        for alpha in alpha_iter:
            # Skip if already completed
            if checkpoint_mgr.is_completed(seed, alpha):
                print(f"\n--- Alpha = {alpha} - already completed, skipping ---")
                continue

            print(f"\n--- Alpha = {alpha} ---")

            # Set up intervention hook
            if intervention is not None and alpha > 0:
                hook_fn = create_steering_hook(intervention, alpha, layer_name)
                llm.set_intervention_fn(hook_fn)
                llm.register_intervention_hook(layer_idx, "residual_post")
            else:
                llm.clear_hooks()
                llm.set_intervention_fn(None)

            # Evaluate TruthfulQA
            print("Evaluating TruthfulQA...")
            tqa_result = evaluate_truthfulqa_with_intervention(
                llm=llm,
                dataset=eval_data,
            )

            result = {
                "model": config.model_name,
                "task": "truthfulness",
                "variant": config.steering_variant.value,
                "layer": layer_idx,
                "alpha": alpha,
                "seed": seed,
                "mc1_accuracy": tqa_result.mc1_accuracy,
                "mc2_accuracy": tqa_result.mc2_accuracy,
                "n_questions": tqa_result.n_questions,
            }

            # Save checkpoint immediately
            checkpoint_mgr.save_result(seed, alpha, result)
            results.append(result)
            print(f"MC1: {tqa_result.mc1_accuracy:.4f}, MC2: {tqa_result.mc2_accuracy:.4f} (checkpointed)")

            # Clean up hooks
            llm.clear_hooks()

    return results


def main():
    args = parse_args()

    # Handle resume mode
    if args.resume:
        resume_dir = Path(args.resume)
        config_path = resume_dir / "config.json"
        if not config_path.exists():
            print(f"Error: Cannot find config at {config_path}")
            return

        print(f"Resuming from: {resume_dir}")
        with open(config_path) as f:
            saved_config = json.load(f)

        # Reconstruct config from saved
        args.model = saved_config["model_name"]
        args.task = saved_config["task"]
        args.variant = saved_config["steering_variant"]
        args.layer = saved_config["layer_idx"]
        args.alpha = saved_config["alpha_values"]
        args.n_seeds = saved_config["n_seeds"]
        args.base_seed = saved_config["base_seed"]
        args.device = saved_config.get("device", "cuda")
        args.batch_size = saved_config.get("batch_size", 8)

        run_name = resume_dir.name
        output_dir = resume_dir.parent
    else:
        output_dir = Path(args.output_dir)
        # Generate run name
        if args.run_name:
            run_name = args.run_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{args.model}_{args.task}_{args.variant}_{timestamp}"

    config = setup_experiment(args)

    print("=" * 60)
    print(f"Steering Evaluation Experiment")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Task: {config.task.value}")
    print(f"Variant: {config.steering_variant.value}")
    print(f"Layer: {config.get_layer_idx()}")
    print(f"Alpha values: {config.alpha_values}")
    print(f"Seeds: {config.n_seeds}")
    print(f"Run name: {run_name}")
    print(f"Output dir: {output_dir / run_name}")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run - exiting without running experiment")
        return

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(output_dir, run_name)

    # Save config
    checkpoint_mgr.save_config(config)

    # Run experiment
    if config.task == Task.TOXICITY:
        results = run_toxicity_experiment(config, checkpoint_mgr, debug=args.debug)
    else:
        results = run_truthfulness_experiment(config, checkpoint_mgr, debug=args.debug)

    # Save final aggregated results
    checkpoint_mgr.save_final_results(results)

    print("\nExperiment complete!")
    print(f"Results directory: {checkpoint_mgr.results_dir}")


if __name__ == "__main__":
    main()
