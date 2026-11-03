"""TTAO-HPO: mixed-hyperparameter optimization in a normalized latent space.

This module keeps the continuous TTAO implementation separate and reuses its
core operators.  The HPO-specific layer provides:

- a mixed search-space representation (continuous, integer, categorical);
- linear/log decoding for continuous parameters;
- equal-width latent cells for integer parameters;
- one-score-per-choice latent encoding for nominal categorical parameters;
- clipping or reflection in the latent cube [0, 1]^D;
- exact query-budget accounting for the HPO variant;
- retained leaders that are *not* re-evaluated after the first macro-iteration;
- an internal cache for duplicated decoded configurations;
- separate counters for optimizer queries, model fits and cache hits;
- NaN/Inf protection for objective values;
- detailed boundary-repair diagnostics.

Budget convention (Option A)
----------------------------
A newly proposed candidate always consumes one optimizer query, even when it
decodes to a configuration already present in the cache.  A cache hit avoids a
new model fit but still consumes one query.  A retained leader is algorithmic
state, not a new proposal, and therefore consumes no query when reused.

For K > 1 triangles:
    first macro-iteration  = 6K queries
    later macro-iterations = 5K queries

For K = 1:
    first macro-iteration  = 5 queries
    later macro-iterations = 4 queries

The public helper ``compute_hpo_query_budget`` can be used to choose identical
query budgets for TTAO-HPO and future baselines.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .continuous import (
    TriangleUnit,
    sample_first_vertex,
    create_fourth_vertex,
    clip_bounds,
    reflect_bounds,
    generic_aggregation,
    local_aggregation,
    update_unit,
    update_after_local,
)
from .evaluation import (
    HPOEvaluationState,
    ObjectiveFunction,
    compute_hpo_query_budget,
    evaluate_hpo_candidates,
    evaluate_latent_candidate,
)
from .search_space import (
    HyperparameterSpec,
    SearchSpace,
    _specs_from_search_space,
    decode_from_latent,
    latent_width,
)


BoundaryMethod = Literal["clip", "reflect"]


# =============================================================================
# HPO result and boundary-repair statistics
# =============================================================================


@dataclass
class RepairStats:
    checked_candidates: int = 0
    repaired_candidates: int = 0

    checked_coordinates: int = 0
    repaired_coordinates: int = 0

    triangle_checked_candidates: int = 0
    triangle_repaired_candidates: int = 0

    triangle_checked_coordinates: int = 0
    triangle_repaired_coordinates: int = 0


@dataclass
class HPOOptimizationResult:
    best_latent_solution: np.ndarray
    best_params: dict[str, Any]
    best_fitness: float

    # Backward-compatible name: in this HPO implementation an "evaluation"
    # means an optimizer query/proposal, not necessarily a new model fit.
    n_evaluations: int
    n_queries: int
    n_model_fits: int
    n_cache_hits: int
    n_unique_configurations: int
    cache_hit_rate: float

    effective_max_iterations: int
    expected_query_budget: int
    runtime_seconds: float

    evaluation_history: list[float]
    iteration_history: list[float]
    query_history: list[dict[str, Any]]
    boundary_method: str

    repair_rate: float
    coordinate_repair_rate: float
    triangle_repair_rate: float
    triangle_coordinate_repair_rate: float
    repair_rate_history: list[float]
    triangle_repair_rate_history: list[float]


# =============================================================================
# Boundary repair and repair diagnostics
# =============================================================================


def repair_with_tracking(
    x: Sequence[float] | np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    stats: RepairStats,
    boundary_method: BoundaryMethod,
    triangle_vertex: bool = False,
) -> np.ndarray:
    """Repair a latent candidate and update repair statistics."""

    x_array = np.asarray(x, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)

    if x_array.shape != lower.shape or x_array.shape != upper.shape:
        raise ValueError("Candidate and latent bounds must have identical shapes.")
    if not np.all(np.isfinite(x_array)):
        raise FloatingPointError("Latent candidate contains NaN or Inf.")

    outside_mask = (x_array < lower) | (x_array > upper)
    needs_repair = bool(np.any(outside_mask))
    n_outside_coordinates = int(np.count_nonzero(outside_mask))

    stats.checked_candidates += 1
    stats.checked_coordinates += x_array.size

    if needs_repair:
        stats.repaired_candidates += 1
    stats.repaired_coordinates += n_outside_coordinates

    if triangle_vertex:
        stats.triangle_checked_candidates += 1
        stats.triangle_checked_coordinates += x_array.size
        if needs_repair:
            stats.triangle_repaired_candidates += 1
        stats.triangle_repaired_coordinates += n_outside_coordinates

    if boundary_method == "clip":
        return clip_bounds(x_array, lower, upper)
    if boundary_method == "reflect":
        return reflect_bounds(x_array, lower, upper)

    raise ValueError(f"Unknown boundary method: {boundary_method}")


# =============================================================================
# HPO-specific triangle geometry
# =============================================================================


def compute_triangle_size_hpo(
    triangle_size_method: Literal["paper", "linear"],
    iteration: int,
    max_iterations: int,
) -> float:
    """Triangle-size schedule adapted to the normalized latent cube."""

    if max_iterations < 2:
        raise ValueError("max_iterations must be at least 2.")
    if iteration < 0 or iteration >= max_iterations:
        raise ValueError("iteration must satisfy 0 <= iteration < max_iterations.")

    if triangle_size_method == "paper":
        t = iteration + 1
        return float(np.exp(-t / max_iterations))

    if triangle_size_method == "linear":
        progress = iteration / (max_iterations - 1)
        start = np.exp(-1.0 / max_iterations)
        end = 1.0 / np.e
        return float(start + progress * (end - start))

    raise ValueError(f"Unknown triangle-size method: {triangle_size_method}")


def create_triangle_vertices_hpo(
    x1: np.ndarray,
    triangle_size_method: Literal["paper", "linear"],
    rng: np.random.Generator,
    iteration: int,
    effective_max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create x2/x3 around x1 using normalized directions in latent space."""

    x1_array = np.asarray(x1, dtype=float)
    if x1_array.ndim != 1 or x1_array.size == 0:
        raise ValueError("x1 must be a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(x1_array)):
        raise ValueError("x1 must contain only finite values.")

    theta = rng.uniform(0.0, np.pi, size=len(x1_array))
    direction_1 = np.cos(theta)
    direction_2 = np.cos(theta + np.pi / 3.0)

    norm_1 = float(np.linalg.norm(direction_1))
    norm_2 = float(np.linalg.norm(direction_2))
    if norm_1 <= np.finfo(float).tiny or norm_2 <= np.finfo(float).tiny:
        raise FloatingPointError("Degenerate triangle direction encountered.")

    direction_1 = direction_1 / norm_1
    direction_2 = direction_2 / norm_2

    triangle_size = compute_triangle_size_hpo(
        triangle_size_method,
        iteration,
        effective_max_iterations,
    )

    return (
        x1_array + triangle_size * direction_1,
        x1_array + triangle_size * direction_2,
    )


# =============================================================================
# Main optimizer
# =============================================================================


def optimize_hpo(
    objective_function: ObjectiveFunction,
    search_space: SearchSpace | Sequence[HyperparameterSpec],
    triangle_size_method: Literal["paper", "linear"],
    alpha_method: Literal["paper", "linear"],
    max_iterations: int,
    seed: int,
    num_triangle: int = 6,
    max_evaluations: int | None = None,
    boundary_method: BoundaryMethod = "reflect",
) -> HPOOptimizationResult:
    """Run TTAO-HPO with exact Option-A query accounting.

    Notes
    -----
    ``max_evaluations`` is a query budget, not a model-fit budget.  If supplied,
    it must exactly match the number of complete macro-iterations requested by
    ``max_iterations``.  Use ``compute_hpo_query_budget`` or
    ``compute_hpo_iterations_from_budget`` to build fair budgets for baselines.
    """

    if not callable(objective_function):
        raise TypeError("objective_function must be callable.")

    specs = _specs_from_search_space(search_space)

    if num_triangle < 1:
        raise ValueError("num_triangle must be at least 1.")
    if max_iterations < 2:
        raise ValueError("max_iterations must be at least 2.")
    if triangle_size_method not in ("paper", "linear"):
        raise ValueError(
            f"Unknown triangle_size_method: {triangle_size_method!r}."
        )
    if alpha_method not in ("paper", "linear"):
        raise ValueError(f"Unknown alpha_method: {alpha_method!r}.")
    if boundary_method not in ("clip", "reflect"):
        raise ValueError(f"Unknown boundary_method: {boundary_method!r}.")

    expected_query_budget = compute_hpo_query_budget(num_triangle, max_iterations)

    if max_evaluations is None:
        max_evaluations = expected_query_budget
    else:
        if not isinstance(max_evaluations, (int, np.integer)):
            raise TypeError("max_evaluations must be an integer or None.")
        max_evaluations = int(max_evaluations)
        if max_evaluations != expected_query_budget:
            raise ValueError(
                "For exact Option-A accounting, max_evaluations must match the "
                "query count of complete macro-iterations. "
                f"For num_triangle={num_triangle} and max_iterations={max_iterations}, "
                f"the exact budget is {expected_query_budget}, got {max_evaluations}. "
                "Use compute_hpo_query_budget() or "
                "compute_hpo_iterations_from_budget()."
            )

    effective_max_iterations = max_iterations

    dimension = sum(latent_width(spec) for spec in specs)
    lower_bounds = np.zeros(dimension, dtype=float)
    upper_bounds = np.ones(dimension, dtype=float)

    rng = np.random.default_rng(seed)
    state = HPOEvaluationState()
    cache: dict[tuple[Any, ...], float] = {}
    repair_stats = RepairStats()

    iteration_history: list[float] = []
    repair_rate_history: list[float] = []
    triangle_repair_rate_history: list[float] = []

    snapshot: list[TriangleUnit] | None = None
    start_time = time.perf_counter()

    for iteration in range(effective_max_iterations):
        triangle: list[TriangleUnit] = []

        checked_before = repair_stats.checked_candidates
        repaired_before = repair_stats.repaired_candidates
        triangle_checked_before = repair_stats.triangle_checked_candidates
        triangle_repaired_before = repair_stats.triangle_repaired_candidates

        # Preserved from the user's continuous/manual implementation: these two
        # weights are sampled once per macro-iteration and shared by all units.
        weight_1 = float(rng.random())
        weight_2 = float(rng.random())

        # ---------------------------------------------------------------------
        # Triangle construction
        # ---------------------------------------------------------------------
        for i in range(num_triangle):
            if iteration == 0:
                x1 = sample_first_vertex(lower_bounds, upper_bounds, rng)
                retained: tuple[tuple[np.ndarray, float], ...] = ()
            else:
                assert snapshot is not None
                # Retain both the leader and its already-known fitness.
                x1 = snapshot[i].best.copy()
                retained = ((x1.copy(), float(snapshot[i].best_fitness)),)

            x2, x3 = create_triangle_vertices_hpo(
                x1,
                triangle_size_method,
                rng,
                iteration,
                effective_max_iterations,
            )

            x2 = repair_with_tracking(
                x2,
                lower_bounds,
                upper_bounds,
                repair_stats,
                boundary_method,
                triangle_vertex=True,
            )
            x3 = repair_with_tracking(
                x3,
                lower_bounds,
                upper_bounds,
                repair_stats,
                boundary_method,
                triangle_vertex=True,
            )

            x4 = create_fourth_vertex(x1, x2, x3, weight_1, weight_2)
            x4 = repair_with_tracking(
                x4,
                lower_bounds,
                upper_bounds,
                repair_stats,
                boundary_method,
            )

            if iteration == 0:
                # First iteration: x1 is a new proposal, so all four vertices
                # consume queries.
                triangle_unit = evaluate_hpo_candidates(
                    [x1, x2, x3, x4],
                    objective_function,
                    specs,
                    state,
                    cache,
                    max_evaluations,
                    iteration=iteration,
                    stage_prefix=f"triangle_{i}_vertex",
                )
            else:
                # Later iterations: x1 is retained state. Its known fitness is
                # included in ranking, but x1 consumes no new query/model fit.
                triangle_unit = evaluate_hpo_candidates(
                    [x2, x3, x4],
                    objective_function,
                    specs,
                    state,
                    cache,
                    max_evaluations,
                    iteration=iteration,
                    stage_prefix=f"triangle_{i}_vertex",
                    retained_candidates=retained,
                )

            triangle.append(triangle_unit)

        # Synchronous snapshot for Generic Aggregation.
        construction_snapshot = copy.deepcopy(triangle)

        # ---------------------------------------------------------------------
        # Generic + Local Aggregation
        # ---------------------------------------------------------------------
        if num_triangle > 1:
            for i in range(num_triangle):
                unit = triangle[i]
                possible_indices = [j for j in range(num_triangle) if j != i]
                random_index = int(rng.choice(possible_indices))

                new_candidate = generic_aggregation(
                    construction_snapshot[i].best,
                    construction_snapshot[random_index].best,
                    rng,
                )
                repaired_candidate = repair_with_tracking(
                    new_candidate,
                    lower_bounds,
                    upper_bounds,
                    repair_stats,
                    boundary_method,
                )
                fitness = evaluate_latent_candidate(
                    repaired_candidate,
                    objective_function,
                    specs,
                    state,
                    cache,
                    max_evaluations,
                    iteration=iteration,
                    stage=f"triangle_{i}_generic_aggregation",
                )
                update_unit(unit, repaired_candidate, fitness)

                new_candidate = local_aggregation(
                    unit.best,
                    unit.second_best,
                    alpha_method,
                    iteration,
                    effective_max_iterations,
                )
                repaired_candidate = repair_with_tracking(
                    new_candidate,
                    lower_bounds,
                    upper_bounds,
                    repair_stats,
                    boundary_method,
                )
                fitness = evaluate_latent_candidate(
                    repaired_candidate,
                    objective_function,
                    specs,
                    state,
                    cache,
                    max_evaluations,
                    iteration=iteration,
                    stage=f"triangle_{i}_local_aggregation",
                )
                update_after_local(unit, repaired_candidate, fitness)

        else:
            unit = triangle[0]
            new_candidate = local_aggregation(
                unit.best,
                unit.second_best,
                alpha_method,
                iteration,
                effective_max_iterations,
            )
            repaired_candidate = repair_with_tracking(
                new_candidate,
                lower_bounds,
                upper_bounds,
                repair_stats,
                boundary_method,
            )
            fitness = evaluate_latent_candidate(
                repaired_candidate,
                objective_function,
                specs,
                state,
                cache,
                max_evaluations,
                iteration=iteration,
                stage="triangle_0_local_aggregation",
            )
            update_after_local(unit, repaired_candidate, fitness)

        # Final leaders retained for the next macro-iteration.
        snapshot = copy.deepcopy(triangle)
        current_best = min(unit.best_fitness for unit in snapshot)
        iteration_history.append(float(current_best))

        # ---------------------------------------------------------------------
        # Repair statistics for this iteration
        # ---------------------------------------------------------------------
        checked_this_iteration = repair_stats.checked_candidates - checked_before
        repaired_this_iteration = repair_stats.repaired_candidates - repaired_before
        repair_rate_history.append(
            repaired_this_iteration / checked_this_iteration
            if checked_this_iteration > 0
            else 0.0
        )

        triangle_checked_this_iteration = (
            repair_stats.triangle_checked_candidates - triangle_checked_before
        )
        triangle_repaired_this_iteration = (
            repair_stats.triangle_repaired_candidates - triangle_repaired_before
        )
        triangle_repair_rate_history.append(
            triangle_repaired_this_iteration / triangle_checked_this_iteration
            if triangle_checked_this_iteration > 0
            else 0.0
        )

    runtime_seconds = time.perf_counter() - start_time

    if snapshot is None or len(snapshot) == 0:
        raise RuntimeError("TTAO-HPO did not create any triangular unit.")

    # Strong postcondition: with complete macro-iterations the query count is
    # deterministic and must match the declared Option-A budget exactly.
    if state.n_queries != max_evaluations:
        raise RuntimeError(
            "Internal budget-accounting error: "
            f"expected {max_evaluations} queries, observed {state.n_queries}."
        )

    snapshot.sort(key=lambda unit: unit.best_fitness)
    best_latent_solution = snapshot[0].best.copy()
    best_params = decode_from_latent(best_latent_solution, specs)

    repair_rate = (
        repair_stats.repaired_candidates / repair_stats.checked_candidates
        if repair_stats.checked_candidates > 0
        else 0.0
    )
    coordinate_repair_rate = (
        repair_stats.repaired_coordinates / repair_stats.checked_coordinates
        if repair_stats.checked_coordinates > 0
        else 0.0
    )
    triangle_repair_rate = (
        repair_stats.triangle_repaired_candidates
        / repair_stats.triangle_checked_candidates
        if repair_stats.triangle_checked_candidates > 0
        else 0.0
    )
    triangle_coordinate_repair_rate = (
        repair_stats.triangle_repaired_coordinates
        / repair_stats.triangle_checked_coordinates
        if repair_stats.triangle_checked_coordinates > 0
        else 0.0
    )

    cache_hit_rate = (
        state.n_cache_hits / state.n_queries if state.n_queries > 0 else 0.0
    )

    return HPOOptimizationResult(
        best_latent_solution=best_latent_solution,
        best_params=best_params,
        best_fitness=float(snapshot[0].best_fitness),
        n_evaluations=state.n_queries,
        n_queries=state.n_queries,
        n_model_fits=state.n_model_fits,
        n_cache_hits=state.n_cache_hits,
        n_unique_configurations=len(cache),
        cache_hit_rate=float(cache_hit_rate),
        effective_max_iterations=effective_max_iterations,
        expected_query_budget=expected_query_budget,
        runtime_seconds=float(runtime_seconds),
        evaluation_history=list(state.evaluation_history),
        iteration_history=iteration_history,
        query_history=list(state.query_history),
        boundary_method=boundary_method,
        repair_rate=float(repair_rate),
        coordinate_repair_rate=float(coordinate_repair_rate),
        triangle_repair_rate=float(triangle_repair_rate),
        triangle_coordinate_repair_rate=float(triangle_coordinate_repair_rate),
        repair_rate_history=repair_rate_history,
        triangle_repair_rate_history=triangle_repair_rate_history,
    )
