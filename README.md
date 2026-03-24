# Does Whitening improve SAE based Steering

- We aim to find out whether SAE based steering drives points off the original manifold, and if whitening can make them stay "on-manifold". Here, whitening is simply the Eigenvalue Decomposition based operator, with a matrix forumation and a well defined inverse. As observed in https://arxiv.org/pdf/2511.13981, upon whitening, the optimization landscape reshapes such that optimizing for sparisty indeed results in minimas (hence, the points will stay on manifold). In the non-whitened case, the minimas do not correspond to the case of least sparsity, so training SAEs would drive the points off the manifold. However, this story is just for the training step of the SAEs. What happens when we steer representations using SAE based methods? What happens if we instead steer after data whitening?

- For this, we will choose one popular steering method, like difference-in-means vector based steering with different strengths, show that it drives the points off the manifold, and then that steering on whitened SAEs will ensure that it stays on manifolds. Not only this, we'll also measure task accuracy and other SAEBench style metrics.

- Our first of evals is going to steering specific but also those that measure the SAE itself (SAEBench-style). Steering-specific results to check are those from this paper -- https://arxiv.org/pdf/2410.23054, particularly, Table 2 -- CLS Toxicity reduction in gemma2-2b and llama3-8b over 5 runs, and Table 3 -- Truthfulness Induction and evaluation on TurthfulQA, again, on gemma2-2b and llama3-8b over 5 runs. **Note** that the steering method to be followed is **not** the ACT method that this paper proposes, instead it is standard SAE based steering (difference-in-means with $\alpha$ as strength).


# Start new experiment
```bash
python -m steering.run_eval --model gemma-2-2b --task toxicity --variant vanilla
```

# Resume interrupted experiment
```bash
python -m steering.run_eval --resume outputs/steering/gemma-2-2b_toxicity_vanilla_20260324_072433
```

# Use specific run name for easy resumption
```bash
python -m steering.run_eval --model gemma-2-2b --task toxicity --variant vanilla --run-name my_experiment
```