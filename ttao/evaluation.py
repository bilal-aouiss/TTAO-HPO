"""HPO objective evaluation, configuration caching, query budgets, and history.

Under Option A, each new proposal consumes one query, including cache hits.
Reusing a retained leader consumes no query. The model-fit counter records
objective calls on cache misses; the objective defines the work in each call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .continuous import TriangleUnit
from .search_space import HyperparameterSpec, decode_from_latent


ObjectiveFunction = Callable[[dict[str, Any]], float]


@dataclass
class HPOEvaluationState:
    """Internal accounting state for HPO objective queries."""

    n_queries: int = 0
    n_model_fits: int = 0
    n_cache_hits: int = 0
    best_fitness: float = float("inf")
    best_latent_solution: np.ndarray | None = None
    evaluation_history: list[float] = field(default_factory=list)
    query_history: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Exact HPO query-budget accounting
# =============================================================================


def compute_hpo_queries_per_iteration(num_triangle: int, iteration: int) -> int:
    """Number of *new optimizer queries* in one HPO macro-iteration.

    The first macro-iteration evaluates x1/x2/x3/x4. Later iterations retain x1
    with its known fitness and evaluate only x2/x3/x4.
    """

    if num_triangle < 1:
        raise ValueError("num_triangle must be at least 1.")
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")

    if num_triangle > 1:
        return 6 * num_triangle if iteration == 0 else 5 * num_triangle

    # With one triangle there is no Generic Aggregation.
    return 5 if iteration == 0 else 4


def compute_hpo_query_budget(num_triangle: int, max_iterations: int) -> int:
    """Exact Option-A query budget required by ``max_iterations``."""

    if max_iterations < 2:
        raise ValueError("max_iterations must be at least 2.")

    first = compute_hpo_queries_per_iteration(num_triangle, 0)

    later = compute_hpo_queries_per_iteration(num_triangle, 1)
    return first + (max_iterations - 1) * later


def compute_hpo_iterations_from_budget(num_triangle: int, query_budget: int) -> int:
    """Inverse of ``compute_hpo_query_budget`` for complete macro-iterations.

    Raises if the requested budget cannot be represented exactly without a
    partial macro-iteration.
    """

    if query_budget < 1:
        raise ValueError("query_budget must be positive.")

    first = compute_hpo_queries_per_iteration(num_triangle, 0)
    later = compute_hpo_queries_per_iteration(num_triangle, 1)

    if query_budget < first:
        raise ValueError(
            f"query_budget={query_budget} is too small; the first macro-iteration "
            f"requires {first} queries."
        )

    remaining = query_budget - first
    if remaining % later != 0:
        raise ValueError(
            f"query_budget={query_budget} does not correspond to complete HPO "
            f"macro-iterations for num_triangle={num_triangle}. Expected "
            f"{first} + m*{later} queries."
        )

    return 1 + remaining // later


# =============================================================================
# Cache-key construction and HPO objective evaluation
# =============================================================================


def _freeze_for_cache(value: Any) -> Any:
    """Convert common parameter values to a deterministic hashable structure."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(_freeze_for_cache(v) for v in value.tolist()))
    if isinstance(value, Mapping):
        frozen_items = [
            (_freeze_for_cache(k), _freeze_for_cache(v))
            for k, v in value.items()
        ]
        return ("mapping", tuple(sorted(frozen_items, key=lambda item: repr(item[0]))))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_freeze_for_cache(v) for v in value))
    if isinstance(value, set):
        frozen = [_freeze_for_cache(v) for v in value]
        return ("set", tuple(sorted(frozen, key=repr)))

    try:
        hash(value)
        return value
    except TypeError:
        return ("repr", repr(value))


def make_configuration_key(
    params: Mapping[str, Any],
    search_space: Sequence[HyperparameterSpec],
) -> tuple[Any, ...]:
    """Canonical cache key preserving search-space parameter order."""

    return tuple(
        (spec.name, _freeze_for_cache(params[spec.name]))
        for spec in search_space
    )


def _can_query(state: HPOEvaluationState, max_evaluations: int) -> bool:
    return state.n_queries < max_evaluations


def evaluate_latent_candidate(
    candidate: np.ndarray,
    objective_function: ObjectiveFunction,
    search_space: Sequence[HyperparameterSpec],
    state: HPOEvaluationState,
    cache: dict[tuple[Any, ...], float],
    max_evaluations: int,
    *,
    iteration: int,
    stage: str,
) -> float:
    """Evaluate one newly proposed latent candidate using Option-A accounting.

    Every call consumes one query.  A duplicated decoded configuration is served
    from the cache and therefore does not consume a new model fit.
    """

    if not _can_query(state, max_evaluations):
        raise RuntimeError("HPO query budget exhausted.")

    z = np.asarray(candidate, dtype=float)
    params = decode_from_latent(z, search_space)
    key = make_configuration_key(params, search_space)

    # Option A: the proposal consumes a query before cache lookup.
    state.n_queries += 1

    cache_hit = key in cache
    if cache_hit:
        state.n_cache_hits += 1
        fitness = float(cache[key])
    else:
        state.n_model_fits += 1
        fitness = float(objective_function(dict(params)))

        # Robust minimization convention: invalid numerical objective values are
        # treated as the worst possible fitness, not allowed to poison sorting.
        if not np.isfinite(fitness):
            fitness = float("inf")

        cache[key] = fitness

    if fitness < state.best_fitness:
        state.best_fitness = fitness
        state.best_latent_solution = z.copy()

    state.evaluation_history.append(float(state.best_fitness))
    state.query_history.append(
        {
            "query": state.n_queries,
            "iteration": iteration + 1,
            "stage": stage,
            "cache_hit": cache_hit,
            "model_fit": not cache_hit,
            "fitness": float(fitness),
            "best_fitness": float(state.best_fitness),
            "params": dict(params),
        }
    )

    return fitness


def evaluate_hpo_candidates(
    candidates: Sequence[np.ndarray],
    objective_function: ObjectiveFunction,
    search_space: Sequence[HyperparameterSpec],
    state: HPOEvaluationState,
    cache: dict[tuple[Any, ...], float],
    max_evaluations: int,
    *,
    iteration: int,
    stage_prefix: str,
    retained_candidates: Sequence[tuple[np.ndarray, float]] = (),
) -> TriangleUnit:
    """Build a TriangleUnit from retained and newly evaluated candidates."""

    evaluated: list[tuple[float, np.ndarray]] = [
        (float(fitness), np.asarray(candidate, dtype=float).copy())
        for candidate, fitness in retained_candidates
    ]

    for index, candidate in enumerate(candidates, start=1):
        fitness = evaluate_latent_candidate(
            np.asarray(candidate, dtype=float),
            objective_function,
            search_space,
            state,
            cache,
            max_evaluations,
            iteration=iteration,
            stage=f"{stage_prefix}_{index}",
        )
        evaluated.append((fitness, np.asarray(candidate, dtype=float).copy()))

    if len(evaluated) < 2:
        raise ValueError("At least two candidates are required to build a TriangleUnit.")

    evaluated.sort(key=lambda item: item[0])
    best_fitness, best = evaluated[0]
    second_best_fitness, second_best = evaluated[1]

    return TriangleUnit(
        best=best.copy(),
        best_fitness=float(best_fitness),
        second_best=second_best.copy(),
        second_best_fitness=float(second_best_fitness),
    )
