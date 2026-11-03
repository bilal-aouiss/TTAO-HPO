# TTAO-HPO

A Python implementation of a **TTAO-inspired hyperparameter optimization (HPO) framework** for mixed search spaces. It extends continuous triangulation-based search to continuous, integer, and categorical hyperparameters through a normalized latent-space representation.

Developed as part of a research internship at Universiti Teknologi MARA (UiTM), Malaysia.

## Features

- **Continuous TTAO:** triangular exploration with generic and local aggregation.
- **Mixed search spaces:** linear or logarithmic continuous parameters, stepped integers, and categorical choices.
- **Latent-space optimization:** search in a normalized space with configurable boundary repair.
- **Evaluation tracking:** cached configurations, optimizer-query accounting, and optimization history.

## Repository structure

```text
TTAO-HPO/
├── ttao/
│   ├── __init__.py       # Public API
│   ├── continuous.py     # Continuous TTAO algorithm
│   ├── search_space.py   # Search-space definition and latent encoding/decoding
│   ├── optimizer.py      # TTAO-HPO optimization loop
│   └── evaluation.py     # Objective evaluation, cache, and budgets
├── Results/              # Continuous benchmark results only
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Define an objective that takes a dictionary of hyperparameters and returns a **scalar loss to minimize**:

```python
import numpy as np
from ttao import (
    HyperparameterSpec, SearchSpace,
    compute_hpo_query_budget, optimize_hpo,
)

space = SearchSpace([
    HyperparameterSpec("learning_rate", "continuous", 1e-4, 1e-1, scale="log"),
    HyperparameterSpec("depth", "integer", 2, 10, step=2),
    HyperparameterSpec("activation", "categorical", choices=("relu", "tanh")),
])

def objective(params):
    return float(
        (np.log10(params["learning_rate"]) + 2.0) ** 2
        + (params["depth"] - 4) ** 2
        + (0.0 if params["activation"] == "relu" else 1.0)
    )

result = optimize_hpo(
    objective_function=objective,
    search_space=space,
    triangle_size_method="paper",
    alpha_method="paper",
    max_iterations=10,
    num_triangle=2,
    max_evaluations=compute_hpo_query_budget(2, 10),
    seed=42,
    boundary_method="reflect",
)

print(result.best_params, result.best_fitness)
```

The example uses a small illustrative objective. Replace it with your own model-training and validation loss for a real HPO task. For the continuous optimizer, import `optimize` from `ttao.continuous`.
