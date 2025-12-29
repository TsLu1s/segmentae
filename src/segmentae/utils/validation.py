import pandas as pd

from typing import Any

from segmentae.core.exceptions import ValidationError, ModelNotFittedError


def validate_dataframe(df: Any, name: str = "DataFrame") -> None:
    """
    Validate that input is a non-empty DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"{name} must be a pandas DataFrame, got {type(df).__name__}",
            suggestion="Convert to DataFrame using pd.DataFrame()"
        )
    
    if df.empty:
        raise ValidationError(
            f"{name} cannot be empty",
            suggestion="Ensure your dataset contains data"
        )


def validate_series(series: Any, name: str = "Series") -> None:
    """
    Validate that input is a non-empty Series.
    """
    if not isinstance(series, pd.Series):
        raise ValidationError(
            f"{name} must be a pandas Series, got {type(series).__name__}",
            suggestion="Convert to Series using pd.Series() or extract DataFrame column"
        )
    
    if len(series) == 0:
        raise ValidationError(
            f"{name} cannot be empty",
            suggestion="Ensure your data contains values"
        )


def validate_fitted(is_fitted: bool, component: str = "Model") -> None:
    """
    Check if component is fitted.
    """
    if not is_fitted:
        raise ModelNotFittedError(
            component=component,
            message=f"{component} must be fitted before use. Call fit() method first."
        )


def validate_threshold_ratio(ratio: float) -> None:
    """
    Validate threshold ratio is positive.
    """
    if ratio <= 0:
        raise ValidationError(
            f"threshold_ratio must be positive, got {ratio}",
            suggestion="Use a positive value like 1.0, 2.0, etc."
        )


def validate_lengths_match(
    a: Any,
    b: Any,
    name_a: str = "First array",
    name_b: str = "Second array"
) -> None:
    """
    Validate two objects have matching lengths.
    """
    if len(a) != len(b):
        raise ValidationError(
            f"{name_a} and {name_b} must have same length. "
            f"Got {len(a)} and {len(b)}",
            suggestion="Ensure both arrays/dataframes have the same number of samples"
        )


def validate_positive_integer(value: int, name: str = "Value") -> None:
    """
    Validate that a value is a positive integer.
    """
    if not isinstance(value, int):
        raise ValidationError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    
    if value < 1:
        raise ValidationError(
            f"{name} must be positive, got {value}",
            suggestion="Use a value >= 1"
        )