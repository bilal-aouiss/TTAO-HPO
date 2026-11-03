"""Original continuous TTAO algorithm and shared mathematical operators.

The reference implementation below is preserved unchanged.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import copy
import time


def sample_first_vertex(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    
    random_values = rng.random(len(lower_bounds))
    return lower_bounds + random_values * (upper_bounds - lower_bounds)

def create_triangle_vertices(
    x1: np.ndarray,
    triangle_size_method: Literal["paper", "linear"],
    rng: np.random.Generator,
    iteration: int,
    effective_max_iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    
    theta = rng.uniform(0.0, np.pi, size=len(x1))
    direction_1 = np.cos(theta)
    direction_2 = np.cos(theta + np.pi / 3.0)

    triangle_size = compute_triangle_size(
        triangle_size_method,
        iteration,
        effective_max_iterations
    )

    x2 = x1 + triangle_size * direction_1
    x3 = x1 + triangle_size * direction_2

    return x2, x3

def compute_triangle_size(
    triangle_size_method: Literal["paper", "linear"],
    iteration: int,
    max_iterations: int
) -> float:

    if triangle_size_method == "paper":
        t = iteration + 1
        return float(9.0 * np.exp(-t / max_iterations))

    elif triangle_size_method == "linear":
        progress = iteration / (max_iterations - 1)

        # mêmes valeurs initiale et finale que Paper
        start = 9.0 * np.exp(-1.0 / max_iterations)
        end = 9.0 / np.e

        return float(
            start + progress * (end - start)
        )

    else:
        raise ValueError(f"Unknown method: {triangle_size_method}")
    

def create_fourth_vertex(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    w1: float,
    w2: float
) -> np.ndarray:


    return (
        w1 * x1
        + w2 * x2
        + (1 - w1 - w2) * x3
    )

@dataclass
class TriangleUnit:
    best: np.ndarray
    best_fitness: float
    second_best: np.ndarray
    second_best_fitness: float

@dataclass
class OptimizationResult:
    best_solution: np.ndarray
    best_fitness: float
    n_evaluations: int
    effective_max_iterations: int
    runtime_seconds: float
    evaluation_history: list[float]
    iteration_history: list[float]

@dataclass
class EvaluationState:
    n_evaluations: int = 0
    best_fitness: float = float("inf")
    history: list[float] = field(default_factory=list)

    
def can_evaluate(
    n_evaluations: int,
    max_evaluations: int | None,
) -> bool:
    
    return (
        max_evaluations is None
        or n_evaluations < max_evaluations
    )

def evaluate_candidate(
    candidate: np.ndarray,
    objective_function,
    state: EvaluationState,
    max_evaluations: int | None,
) -> float:

    if not can_evaluate(
        state.n_evaluations,
        max_evaluations
    ):
        raise RuntimeError("Evaluation budget exhausted.")

    fitness = float(objective_function(candidate))

    state.n_evaluations += 1

    if fitness < state.best_fitness:
        state.best_fitness = fitness

    state.history.append(state.best_fitness)

    return fitness

def evaluate_candidates(
    candidates: list[np.ndarray],
    objective_function,
    state: EvaluationState,
    max_evaluations: int | None,
) -> TriangleUnit:

    evaluated = []

    for candidate in candidates:
        fitness = evaluate_candidate(
            candidate,
            objective_function,
            state,
            max_evaluations
        )

        evaluated.append(
            (fitness, candidate)
        )

    evaluated.sort(
        key=lambda item: item[0]
    )

    best_fitness, best = evaluated[0]
    second_best_fitness, second_best = evaluated[1]

    return TriangleUnit(
        best=best.copy(),
        best_fitness=float(best_fitness),
        second_best=second_best.copy(),
        second_best_fitness=float(second_best_fitness),
    )

def generic_aggregation(
    current_best: np.ndarray,
    random_best: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    
    weight = rng.random(size=current_best.shape)
    return (
        weight * current_best
        + (1.0 - weight) * random_best
    )

def update_unit(
    unit: TriangleUnit,
    candidate: np.ndarray,
    fitness: float
) -> None:
    if fitness < unit.best_fitness:
        unit.second_best = unit.best.copy()
        unit.second_best_fitness = unit.best_fitness

        unit.best = candidate.copy()
        unit.best_fitness = fitness

    elif fitness < unit.second_best_fitness:
        unit.second_best = candidate.copy()
        unit.second_best_fitness = fitness

def compute_alpha(
    alpha_method: Literal["paper", "linear"],
    iteration: int,
    max_iterations: int
) -> float:

    if alpha_method == "paper":
        t = iteration + 1
        T = max_iterations

        slope = (np.e - np.e**3) / (T - 1)
        value = slope * t + np.e**3 - slope

        return float(np.log(value))

    elif alpha_method == "linear":
        progress = iteration / (max_iterations - 1)

        return float(
            3.0 + progress * (1.0 - 3.0)
        )

    else:
        raise ValueError(f"Unknown method: {alpha_method}")

def local_aggregation(
    best: np.ndarray,
    second_best: np.ndarray,
    alpha_method: Literal["paper", "linear"],
    iteration: int,
    effective_max_iterations: int
) -> np.ndarray:
    
    direction = best - second_best
    alpha = compute_alpha(
        alpha_method,
        iteration,
        effective_max_iterations
    )
    return best + alpha * direction

def update_after_local(
    unit: TriangleUnit,
    candidate: np.ndarray,
    fitness: float,
) -> None:

    if fitness < unit.best_fitness:
        unit.second_best = unit.best.copy()
        unit.second_best_fitness = unit.best_fitness

        unit.best = candidate.copy()
        unit.best_fitness = fitness

def clip_bounds(
    x: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    return np.clip(x, lower_bounds, upper_bounds)

def reflect_bounds(
    x: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    widths = upper_bounds - lower_bounds
    shifted = np.mod(x - lower_bounds, 2.0 * widths)
    reflected = np.where(
        shifted <= widths,
        shifted,
        2.0 * widths - shifted,
    )
    return lower_bounds + reflected

def compute_evaluations_per_iteration(
    num_triangle: int,
) -> int:

    if num_triangle > 1:
        return 6 * num_triangle

    return 5

def compute_effective_max_iterations(
    max_iterations: int,
    max_evaluations: int | None,
    evaluations_per_iteration: int,
) -> int:

    if max_evaluations is None:
        return max_iterations

    budget_iterations = (
        max_evaluations
        // evaluations_per_iteration
    )

    return min(
        max_iterations,
        budget_iterations
    )

def optimize(objective_function, triangle_size_method, alpha_method, max_iterations, lower_bounds, upper_bounds, seed, num_triangle, max_evaluations: int | None = None) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    
    if lower_bounds.shape != upper_bounds.shape:
        raise ValueError("lower_bounds and upper_bounds must have the same shape.")

    if np.any(upper_bounds <= lower_bounds):
        raise ValueError("Each upper bound must be greater than its lower bound.")
    
    if lower_bounds.ndim != 1 or upper_bounds.ndim != 1:
        raise ValueError("lower_bounds and upper_bounds must be one-dimensional.")
    
    if num_triangle < 1:
        raise ValueError("num_triangle must be at least 1.")
    
    if max_iterations < 2:
        raise ValueError("max_iterations must be at least 2.")
    
    if max_evaluations is not None and max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1.")

    evaluations_per_iteration = compute_evaluations_per_iteration(num_triangle)

    if max_evaluations is not None and max_evaluations < evaluations_per_iteration:
            raise ValueError("max_evaluations is too small for the number of triangles.")

    if max_evaluations is not None and max_evaluations % evaluations_per_iteration != 0:
            raise ValueError("max_evaluations must be divisible by evaluations_per_iteration.")

    effective_max_iterations = compute_effective_max_iterations(
            max_iterations,
            max_evaluations,
            evaluations_per_iteration,
        )
    
    if effective_max_iterations < 2:
        raise ValueError(
            "The evaluation budget must allow "
            "at least 2 complete iterations."
        )
    start_time = time.perf_counter()
    state = EvaluationState()
    iteration_history = []
    for iteration in range(effective_max_iterations):
        triangle = []
        weight_1 = rng.random()
        weight_2 = rng.random()
        for i in range(num_triangle):
            if iteration == 0:
                x1 = sample_first_vertex(lower_bounds, upper_bounds, rng)
                assert x1.shape == lower_bounds.shape
            else:
                x1 = snapshot[i].best.copy()
            x2,x3 = create_triangle_vertices(x1, triangle_size_method, rng, iteration, effective_max_iterations)
            x2 = clip_bounds(x2, lower_bounds, upper_bounds)
            x3 = clip_bounds(x3, lower_bounds, upper_bounds)
            x4 = create_fourth_vertex(x1, x2, x3, weight_1, weight_2)
            x4 = clip_bounds(x4, lower_bounds, upper_bounds)
            triangle_unit = evaluate_candidates([x1, x2, x3, x4], objective_function, state, max_evaluations)
            triangle.append(triangle_unit)
        snapshot = copy.deepcopy(triangle)
        if num_triangle > 1:
            for i in range(num_triangle):
                unit = triangle[i]
                possible_indices = [
                j for j in range(num_triangle)
                if j != i
                ]
                random_index = rng.choice(possible_indices)
                new_candidate = generic_aggregation(snapshot[i].best,snapshot[random_index].best, rng)
                clipped_candidate = clip_bounds(new_candidate,lower_bounds,upper_bounds)
                fitness = evaluate_candidate(clipped_candidate,objective_function, state,max_evaluations)
                update_unit(
                    unit,
                    clipped_candidate,
                    fitness,
                )
                new_candidate = local_aggregation(unit.best,unit.second_best,alpha_method, iteration, effective_max_iterations)
                clipped_candidate = clip_bounds(new_candidate,lower_bounds,upper_bounds,)
                fitness = evaluate_candidate(clipped_candidate,objective_function,state,max_evaluations,)
                update_after_local(
                    unit,
                    clipped_candidate,
                    fitness,
                )
        elif num_triangle == 1:
            unit = triangle[0]
            new_candidate = local_aggregation(unit.best, unit.second_best, alpha_method, iteration, effective_max_iterations)
            clipped_candidate = clip_bounds(new_candidate, lower_bounds, upper_bounds)
            fitness = evaluate_candidate(clipped_candidate,objective_function, state,max_evaluations)
            update_after_local(
                unit,
                clipped_candidate,
                fitness,
            )
        snapshot = copy.deepcopy(triangle)
        current_best = min(unit.best_fitness for unit in snapshot)
        iteration_history.append(current_best)
    runtime_seconds = time.perf_counter() - start_time
    snapshot.sort(key=lambda unit: unit.best_fitness)
    return OptimizationResult(
        best_solution=snapshot[0].best.copy(),
        best_fitness=float(snapshot[0].best_fitness),
        n_evaluations=state.n_evaluations,
        effective_max_iterations=effective_max_iterations,
        runtime_seconds=runtime_seconds,
        evaluation_history=list(state.history),
        iteration_history=iteration_history,
    )


