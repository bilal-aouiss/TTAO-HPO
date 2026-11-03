"""Mixed hyperparameter specifications, validation, and latent encoding/decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np


ParameterKind = Literal["continuous", "integer", "categorical"]
Scale = Literal["linear", "log"]


# =============================================================================
# Search-space validation and latent mapping
# =============================================================================


@dataclass(frozen=True)
class HyperparameterSpec:
    """Description of one HPO variable.

    Parameters
    ----------
    name:
        Unique parameter name.
    kind:
        ``continuous``, ``integer`` or ``categorical``.
    lower, upper:
        Bounds for continuous/integer variables.
    choices:
        Choices for a categorical variable.
    scale:
        ``linear`` or ``log`` for continuous variables. Integer and
        categorical variables currently use ``linear`` only.
    step:
        Step for integer variables. Default is 1.
    """

    name: str
    kind: ParameterKind
    lower: float | int | None = None
    upper: float | int | None = None
    choices: tuple[Any, ...] | None = None
    scale: Scale = "linear"
    step: int = 1


def _contains_duplicate_choices(choices: Sequence[Any]) -> bool:
    """Equality-based duplicate check that also works for unhashable choices."""

    for i, value in enumerate(choices):
        for previous in choices[:i]:
            try:
                equal = value == previous
                if isinstance(equal, np.ndarray):
                    equal = bool(np.all(equal))
                if bool(equal):
                    return True
            except Exception:
                # If an exotic object cannot be compared reliably, let the
                # cache-key layer handle it rather than failing validation here.
                continue
    return False


def validate_search_space(search_space: Sequence[HyperparameterSpec]) -> None:
    """Validate a mixed HPO search space."""

    if len(search_space) == 0:
        raise ValueError("search_space must contain at least one hyperparameter.")

    names: set[str] = set()

    for spec in search_space:
        if not isinstance(spec, HyperparameterSpec):
            raise TypeError("Each search-space entry must be a HyperparameterSpec.")

        if not isinstance(spec.name, str) or not spec.name.strip():
            raise ValueError("Each hyperparameter must have a non-empty string name.")

        if spec.name in names:
            raise ValueError(f"Duplicate hyperparameter name: {spec.name}")
        names.add(spec.name)

        if spec.kind not in ("continuous", "integer", "categorical"):
            raise ValueError(f"Unknown hyperparameter kind: {spec.kind}")

        if spec.scale not in ("linear", "log"):
            raise ValueError(
                f"{spec.name}: scale must be 'linear' or 'log', got {spec.scale!r}."
            )

        if spec.kind in ("continuous", "integer"):
            if spec.lower is None or spec.upper is None:
                raise ValueError(
                    f"{spec.name}: lower and upper bounds are required."
                )

            try:
                lower = float(spec.lower)
                upper = float(spec.upper)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{spec.name}: lower and upper must be numeric."
                ) from exc

            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(f"{spec.name}: bounds must be finite.")

            if upper <= lower:
                raise ValueError(f"{spec.name}: upper must be greater than lower.")

            if spec.kind == "continuous":
                if spec.step != 1:
                    raise ValueError(
                        f"{spec.name}: step is only applicable to integer parameters."
                    )
                if spec.scale == "log" and lower <= 0.0:
                    raise ValueError(
                        f"{spec.name}: log scale requires lower > 0."
                    )

            else:  # integer
                if spec.scale != "linear":
                    raise ValueError(
                        f"{spec.name}: integer hyperparameters currently require linear scale."
                    )

                if not float(spec.lower).is_integer():
                    raise ValueError(f"{spec.name}: lower must be an integer.")
                if not float(spec.upper).is_integer():
                    raise ValueError(f"{spec.name}: upper must be an integer.")
                if not isinstance(spec.step, (int, np.integer)) or int(spec.step) <= 0:
                    raise ValueError(f"{spec.name}: integer step must be a positive integer.")

                lower_i = int(spec.lower)
                upper_i = int(spec.upper)
                step_i = int(spec.step)
                if (upper_i - lower_i) % step_i != 0:
                    raise ValueError(
                        f"{spec.name}: bounds must be aligned with step={step_i}."
                    )

        else:  # categorical
            if spec.scale != "linear":
                raise ValueError(
                    f"{spec.name}: scale is not applicable to categorical hyperparameters."
                )
            if spec.step != 1:
                raise ValueError(
                    f"{spec.name}: step is not applicable to categorical hyperparameters."
                )
            if spec.lower is not None or spec.upper is not None:
                raise ValueError(
                    f"{spec.name}: lower/upper are not applicable to categorical parameters."
                )
            if spec.choices is None:
                raise ValueError(f"{spec.name}: choices are required.")
            if len(spec.choices) < 2:
                raise ValueError(f"{spec.name}: at least two choices are required.")
            if _contains_duplicate_choices(spec.choices):
                raise ValueError(f"{spec.name}: choices must be unique.")


def latent_width(spec: HyperparameterSpec) -> int:
    """Number of latent coordinates used by one parameter."""

    if spec.kind == "categorical":
        assert spec.choices is not None
        return len(spec.choices)
    return 1


def latent_dimension(search_space: Sequence[HyperparameterSpec]) -> int:
    """Total dimension of the normalized latent search space."""

    validate_search_space(search_space)
    return sum(latent_width(spec) for spec in search_space)


def _integer_values(spec: HyperparameterSpec) -> np.ndarray:
    """Allowed integer values, including optional step."""

    lower = int(spec.lower)
    upper = int(spec.upper)
    step = int(spec.step)
    return np.arange(lower, upper + step, step, dtype=int)


def decode_from_latent(
    z: Sequence[float] | np.ndarray,
    search_space: Sequence[HyperparameterSpec],
) -> dict[str, Any]:
    """Decode a latent vector in ``[0,1]^D`` into real HPO parameters.

    Integer parameters use equal-width latent cells.  This avoids the endpoint
    bias created by rounding a continuously scaled value to the nearest integer.
    """

    validate_search_space(search_space)

    z_array = np.asarray(z, dtype=float)
    expected_dimension = sum(latent_width(spec) for spec in search_space)

    if z_array.ndim != 1:
        raise ValueError("z must be one-dimensional.")
    if len(z_array) != expected_dimension:
        raise ValueError(
            f"Expected latent dimension {expected_dimension}, got {len(z_array)}."
        )
    if not np.all(np.isfinite(z_array)):
        raise ValueError("All latent coordinates must be finite.")
    if np.any(z_array < 0.0) or np.any(z_array > 1.0):
        raise ValueError("All latent coordinates must belong to [0, 1].")

    params: dict[str, Any] = {}
    cursor = 0

    for spec in search_space:
        if spec.kind == "continuous":
            u = float(z_array[cursor])

            if spec.scale == "linear":
                value = float(spec.lower) + u * (float(spec.upper) - float(spec.lower))
            else:  # log
                log_value = (
                    np.log(float(spec.lower))
                    + u * (np.log(float(spec.upper)) - np.log(float(spec.lower)))
                )
                value = float(np.exp(log_value))

            params[spec.name] = float(value)
            cursor += 1

        elif spec.kind == "integer":
            u = float(z_array[cursor])
            values = _integer_values(spec)

            # Equal-width cells: every allowed integer receives the same
            # latent measure under a uniform draw of u in [0, 1].
            index = min(int(np.floor(u * len(values))), len(values) - 1)
            params[spec.name] = int(values[index])
            cursor += 1

        else:  # categorical
            assert spec.choices is not None
            n_choices = len(spec.choices)
            scores = z_array[cursor : cursor + n_choices]
            index = int(np.argmax(scores))
            params[spec.name] = spec.choices[index]
            cursor += n_choices

    return params


def _categorical_index(value: Any, choices: Sequence[Any], name: str) -> int:
    for index, choice in enumerate(choices):
        try:
            equal = value == choice
            if isinstance(equal, np.ndarray):
                equal = bool(np.all(equal))
            if bool(equal):
                return index
        except Exception:
            continue
    raise ValueError(f"{name}: value {value!r} is not one of the allowed choices.")


def encode_to_latent(
    params: Mapping[str, Any],
    search_space: Sequence[HyperparameterSpec],
) -> np.ndarray:
    """Encode a valid real configuration into one canonical latent vector.

    Integer values are encoded at the *center* of their equal-width latent cell.
    Categorical values use a one-hot score block.  Because discrete decoders are
    many-to-one, this is a canonical encoding rather than a true inverse.
    """

    validate_search_space(search_space)

    expected_names = {spec.name for spec in search_space}
    provided_names = set(params.keys())

    missing = expected_names - provided_names
    extra = provided_names - expected_names
    if missing:
        raise ValueError(f"Missing parameter(s): {sorted(missing)}")
    if extra:
        raise ValueError(f"Unknown parameter(s): {sorted(extra)}")

    z: list[float] = []

    for spec in search_space:
        value = params[spec.name]

        if spec.kind == "continuous":
            try:
                value_f = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{spec.name}: value must be numeric.") from exc

            if not np.isfinite(value_f):
                raise ValueError(f"{spec.name}: value must be finite.")
            if not float(spec.lower) <= value_f <= float(spec.upper):
                raise ValueError(
                    f"{spec.name}: value {value_f} is outside "
                    f"[{spec.lower}, {spec.upper}]."
                )

            if spec.scale == "linear":
                u = (
                    (value_f - float(spec.lower))
                    / (float(spec.upper) - float(spec.lower))
                )
            else:
                if value_f <= 0.0:
                    raise ValueError(f"{spec.name}: log-scaled value must be > 0.")
                u = (
                    (np.log(value_f) - np.log(float(spec.lower)))
                    / (np.log(float(spec.upper)) - np.log(float(spec.lower)))
                )

            z.append(float(np.clip(u, 0.0, 1.0)))

        elif spec.kind == "integer":
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{spec.name}: boolean is not a valid integer value.")
            try:
                value_f = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{spec.name}: value must be integer-like.") from exc

            if not np.isfinite(value_f) or not value_f.is_integer():
                raise ValueError(f"{spec.name}: value must be an integer.")

            value_i = int(value_f)
            values = _integer_values(spec)
            matches = np.flatnonzero(values == value_i)
            if matches.size == 0:
                raise ValueError(
                    f"{spec.name}: value {value_i} is not in the allowed integer grid "
                    f"{values.tolist()}."
                )

            index = int(matches[0])
            # Center of the cell corresponding to this integer value.
            u = (index + 0.5) / len(values)
            z.append(float(u))

        else:  # categorical
            assert spec.choices is not None
            index = _categorical_index(value, spec.choices, spec.name)
            scores = np.zeros(len(spec.choices), dtype=float)
            scores[index] = 1.0
            z.extend(scores.tolist())

    result = np.asarray(z, dtype=float)

    if not np.all(np.isfinite(result)):
        raise RuntimeError("Internal encoding error: non-finite latent value produced.")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise RuntimeError("Internal encoding error: latent value outside [0, 1].")

    return result


class SearchSpace:
    """Small reusable wrapper around a sequence of ``HyperparameterSpec``."""

    def __init__(self, specs: Sequence[HyperparameterSpec]) -> None:
        self.specs = tuple(specs)
        validate_search_space(self.specs)
        self.dimension = sum(latent_width(spec) for spec in self.specs)

    def latent_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(self.dimension, dtype=float),
            np.ones(self.dimension, dtype=float),
        )

    def decode(self, z: Sequence[float] | np.ndarray) -> dict[str, Any]:
        return decode_from_latent(z, self.specs)

    def encode(self, params: Mapping[str, Any]) -> np.ndarray:
        return encode_to_latent(params, self.specs)

    def describe(self) -> dict[str, Any]:
        return {
            "latent_dimension": self.dimension,
            "parameters": [
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "lower": spec.lower,
                    "upper": spec.upper,
                    "choices": list(spec.choices) if spec.choices is not None else None,
                    "scale": spec.scale,
                    "step": spec.step,
                    "latent_width": latent_width(spec),
                }
                for spec in self.specs
            ],
        }


def _specs_from_search_space(
    search_space: SearchSpace | Sequence[HyperparameterSpec],
) -> tuple[HyperparameterSpec, ...]:
    if isinstance(search_space, SearchSpace):
        return search_space.specs
    specs = tuple(search_space)
    validate_search_space(specs)
    return specs
