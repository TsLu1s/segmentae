import math
import numbers
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from segmentae.core.exceptions import ModelNotFittedError, ValidationError


def validate_dataframe(
    df: Any,
    name: str = "DataFrame",
    *,
    allow_missing: bool = True,
    require_numeric: bool = False,
) -> None:
    """
    Validate that input is a non-empty DataFrame.

    Args:
        df: Object to validate.
        name: Label used in error messages.
        allow_missing: If False, missing values raise a ``ValidationError``.
        require_numeric: If True, every column must be numeric and free of
            infinite values (what clustering models and autoencoders need).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"{name} must be a pandas DataFrame, got {type(df).__name__}",
            suggestion="Convert to DataFrame using pd.DataFrame()",
        )

    if df.empty:
        raise ValidationError(
            f"{name} DataFrame is empty", suggestion="Ensure your dataset contains data"
        )

    if require_numeric:
        if df.columns.duplicated().any():
            duplicated = sorted({str(col) for col in df.columns[df.columns.duplicated()]})
            raise ValidationError(
                f"{name} has duplicated column names: {_preview(duplicated)}",
                suggestion="Give every feature column a unique name",
            )

        non_numeric = [str(col) for col, dtype in df.dtypes.items() if not is_numeric_dtype(dtype)]
        if non_numeric:
            raise ValidationError(
                f"{name} contains non-numeric columns: {_preview(non_numeric)}",
                suggestion="Encode categorical columns first, e.g. "
                "Preprocessing(encoder='IFrequencyEncoder')",
            )

    if not allow_missing and df.isnull().to_numpy().any():
        columns = [str(col) for col in df.columns[df.isnull().any().to_numpy()]]
        raise ValidationError(
            f"{name} contains missing values in columns: {_preview(columns)}",
            suggestion="Handle missing values first, e.g. Preprocessing(imputer='Simple')",
        )

    if require_numeric:
        infinite = np.isinf(df.to_numpy(dtype="float64", na_value=np.nan)).any(axis=0)
        if infinite.any():
            columns = [str(col) for col in df.columns[infinite]]
            raise ValidationError(
                f"{name} contains infinite values in columns: {_preview(columns)}",
                suggestion="Replace or remove infinite values before fitting",
            )


def validate_series(series: Any, name: str = "Series") -> None:
    """
    Validate that input is a non-empty Series.
    """
    if not isinstance(series, pd.Series):
        raise ValidationError(
            f"{name} must be a pandas Series, got {type(series).__name__}",
            suggestion="Convert to Series using pd.Series() or extract DataFrame column",
        )

    if len(series) == 0:
        raise ValidationError(
            f"{name} cannot be empty", suggestion="Ensure your data contains values"
        )


def validate_fitted(is_fitted: bool, component: str = "Model") -> None:
    """
    Check if component is fitted.
    """
    if not is_fitted:
        raise ModelNotFittedError(
            component=component,
            message=f"{component} must be fitted before use. Call fit() method first.",
        )


def validate_threshold_ratio(ratio: Any) -> float:
    """
    Validate that a threshold ratio is a positive, finite number.

    Returns:
        The ratio as a float.
    """
    if isinstance(ratio, bool) or not isinstance(ratio, numbers.Real):
        raise ValidationError(
            f"threshold_ratio must be a number, got {type(ratio).__name__}",
            suggestion="Use a positive value like 1.0, 2.0, etc.",
        )

    if not math.isfinite(ratio) or ratio <= 0:
        raise ValidationError(
            f"threshold_ratio must be positive, got {ratio}",
            suggestion="Use a positive value like 1.0, 2.0, etc.",
        )

    return float(ratio)


def validate_lengths_match(
    a: Any, b: Any, name_a: str = "First array", name_b: str = "Second array"
) -> None:
    """
    Validate two objects have matching lengths.
    """
    if len(a) != len(b):
        raise ValidationError(
            f"{name_a} and {name_b} must have same length. " f"Got {len(a)} and {len(b)}",
            suggestion="Ensure both arrays/dataframes have the same number of samples",
        )


def validate_positive_integer(value: Any, name: str = "Value") -> None:
    """
    Validate that a value is a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")

    if value < 1:
        raise ValidationError(
            f"{name} must be positive, got {value}", suggestion="Use a value >= 1"
        )


def validate_columns_match(
    df: pd.DataFrame, expected_columns: Sequence, name: str = "DataFrame"
) -> pd.DataFrame:
    """
    Check that ``df`` has exactly the expected feature columns.

    Returns:
        ``df`` itself when the columns already match, otherwise ``df`` with its
        columns reordered to ``expected_columns`` (same set, different order).

    Raises:
        ValidationError: If columns are missing or unexpected columns are present.
    """
    expected = list(expected_columns)
    actual = list(df.columns)
    if actual == expected:
        return df

    missing = [str(col) for col in expected if col not in set(actual)]
    unexpected = [str(col) for col in actual if col not in set(expected)]
    if missing or unexpected or len(actual) != len(expected):
        details = []
        if missing:
            details.append(f"missing columns: {_preview(missing)}")
        if unexpected:
            details.append(f"unexpected columns: {_preview(unexpected)}")
        if not details:
            details.append("duplicated column names")
        raise ValidationError(
            f"{name} does not match the columns seen during fitting ({'; '.join(details)})",
            suggestion="Pass data with the same feature columns used for fitting",
        )

    return df[expected]


def validate_binary_target(target: pd.Series, name: str = "target_col") -> None:
    """
    Validate that a target Series holds binary anomaly labels (0 = normal, 1 = anomaly).
    """
    if target.isnull().any():
        raise ValidationError(
            f"{name} contains missing values",
            suggestion="Labels must be 0 (normal) or 1 (anomaly) for every sample",
        )

    labels = pd.unique(target)
    if not set(labels.tolist()) <= {0, 1}:
        raise ValidationError(
            f"{name} must contain only binary labels 0 (normal) and 1 (anomaly), "
            f"got values: {_preview(sorted(str(v) for v in labels))}",
            suggestion="Map your labels to 0/1 before evaluation",
        )


def _preview(items: Sequence, limit: int = 10) -> str:
    """Render a short, bounded list for error messages."""
    items = list(items)
    shown = ", ".join(str(item) for item in items[:limit])
    extra = len(items) - limit
    return f"[{shown}, ... +{extra} more]" if extra > 0 else f"[{shown}]"
