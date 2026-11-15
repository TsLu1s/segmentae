from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from segmentae.core.exceptions import ValidationError

ArrayLike = Union[pd.Series, np.ndarray]


def metrics_classification(y_true: ArrayLike, y_pred: ArrayLike) -> pd.DataFrame:
    """
    Calculate classification evaluation metrics for binary anomaly labels.

    Labels follow the anomaly detection convention: 0 = normal, 1 = anomaly
    (the positive class for precision, recall and F1).

    Returns:
        DataFrame containing accuracy, precision, recall, and F1 score metrics
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Calculate metrics with zero_division handling
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Create metrics dictionary
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

    # Convert to DataFrame
    return pd.DataFrame(metrics, index=[0])


def metrics_regression(y_true: ArrayLike, y_pred: ArrayLike) -> pd.DataFrame:
    """
    Calculate regression evaluation metrics.

    Returns:
        DataFrame containing MAE, MSE, RMSE, R², and Max Error metrics
    """
    # Validate inputs
    _validate_inputs(y_true, y_pred)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    maxerror = max_error(y_true, y_pred)

    # Create metrics dictionary
    metrics = {
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
        "R-squared": r2,
        "Max Error": maxerror,
    }

    # Convert to DataFrame
    return pd.DataFrame(metrics, index=[0])


def _validate_inputs(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Validate that true and predicted values are non-empty and equally long."""
    if len(y_true) != len(y_pred):
        raise ValidationError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples",
            suggestion="Ensure both arrays have the same number of samples",
        )

    if len(y_true) == 0:
        raise ValidationError(
            "Empty arrays provided", suggestion="Provide non-empty arrays with predictions"
        )
