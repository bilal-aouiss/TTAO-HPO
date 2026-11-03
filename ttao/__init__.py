"""TTAO-inspired hyperparameter optimization in a normalized latent space.

The continuous implementation is available through ``ttao.continuous``.
"""

from .evaluation import (
    compute_hpo_iterations_from_budget,
    compute_hpo_queries_per_iteration,
    compute_hpo_query_budget,
    make_configuration_key,
)
from .optimizer import (
    HPOOptimizationResult,
    RepairStats,
    compute_triangle_size_hpo,
    create_triangle_vertices_hpo,
    optimize_hpo,
    repair_with_tracking,
)
from .search_space import (
    HyperparameterSpec,
    SearchSpace,
    decode_from_latent,
    encode_to_latent,
    latent_dimension,
    latent_width,
    validate_search_space,
)


__all__ = [
    "HyperparameterSpec",
    "SearchSpace",
    "HPOOptimizationResult",
    "RepairStats",
    "validate_search_space",
    "latent_width",
    "latent_dimension",
    "decode_from_latent",
    "encode_to_latent",
    "repair_with_tracking",
    "compute_triangle_size_hpo",
    "create_triangle_vertices_hpo",
    "compute_hpo_queries_per_iteration",
    "compute_hpo_query_budget",
    "compute_hpo_iterations_from_budget",
    "make_configuration_key",
    "optimize_hpo",
]
