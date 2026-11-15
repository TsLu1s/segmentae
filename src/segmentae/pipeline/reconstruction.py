from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from segmentae.core.constants import ThresholdMetric, parse_threshold_metric

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


class ReconstructionConfig(BaseModel):
    """Configuration for reconstruction phase."""

    model_config = ConfigDict(use_enum_values=False)

    threshold_metric: ThresholdMetric = ThresholdMetric.MSE

    @field_validator("threshold_metric", mode="before")
    @classmethod
    def convert_to_enum(cls, v):
        """Convert string to ThresholdMetric enum."""
        if isinstance(v, ThresholdMetric):
            return v
        if isinstance(v, str):
            return parse_threshold_metric(v)
        raise ValueError(f"Invalid threshold_metric type: {type(v)}")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation phase."""

    # Threshold ratio scales the baseline (mean training error) threshold:
    # values above 1 make detection stricter, values below 1 more sensitive.
    threshold_ratio: float = Field(default=1.0, gt=0, allow_inf_nan=False)


@dataclass
class ClusterReconstruction:
    """
    Reconstruction data for a single cluster.
    """

    cluster_id: int
    real_values: pd.DataFrame
    predictions: pd.DataFrame
    y_true: Optional[pd.Series]
    indices: List[int]


@dataclass
class ReconstructionMetrics:
    """
    Aggregated reconstruction metrics for a cluster.
    """

    cluster_id: int
    metrics_df: pd.DataFrame
    column_metrics: pd.DataFrame
    indices: List[int]


def compute_reconstruction_errors(
    real_values: np.ndarray, predicted_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-row reconstruction errors for multiple metrics.

    Calculates MSE, MAE, RMSE, and Max Error for each row,
    comparing real values against autoencoder reconstructions.

    Returns:
        Tuple ``(mse, mae, rmse, max_error)`` of 1-D arrays with one value per row.
    """
    real = np.asarray(real_values, dtype="float64")
    predicted = np.asarray(predicted_values, dtype="float64")
    if real.shape != predicted.shape:
        raise ValueError(
            f"real_values and predicted_values must have the same shape, "
            f"got {real.shape} and {predicted.shape}"
        )
    if real.ndim == 1:
        real, predicted = real.reshape(1, -1), predicted.reshape(1, -1)

    abs_diff = np.abs(real - predicted)
    mse_per_row = np.mean(np.square(abs_diff), axis=1)
    mae_per_row = np.mean(abs_diff, axis=1)
    rmse_per_row = np.sqrt(mse_per_row)
    max_error_per_row = np.max(abs_diff, axis=1) if abs_diff.shape[1] else np.zeros(len(abs_diff))

    return mse_per_row, mae_per_row, rmse_per_row, max_error_per_row


def compute_column_metrics(
    real_values: np.ndarray, predicted_values: np.ndarray, columns: List[str], cluster_id: int
) -> pd.DataFrame:
    """
    Compute per-column reconstruction metrics.

    Calculates MSE, MAE, RMSE, and Max Error for each feature column,
    providing insight into which features are reconstructed well.
    """
    abs_diff = np.abs(
        np.asarray(real_values, dtype="float64") - np.asarray(predicted_values, dtype="float64")
    )
    mse = np.mean(np.square(abs_diff), axis=0)

    col_metrics = pd.DataFrame(
        {
            "Column": columns,
            "MSE": mse,
            "MAE": np.mean(abs_diff, axis=0),
            "RMSE": np.sqrt(mse),
            "Max_Error": np.max(abs_diff, axis=0),
            "partition": cluster_id,
        }
    )

    total_metrics = pd.DataFrame(
        {
            "Column": ["Total Metrics"],
            "MSE": [col_metrics["MSE"].mean()],
            "MAE": [col_metrics["MAE"].mean()],
            "RMSE": [col_metrics["RMSE"].mean()],
            "Max_Error": [col_metrics["Max_Error"].max()],
        }
    )

    return pd.concat([col_metrics, total_metrics], ignore_index=True)


def create_metrics_dataframe(
    mse_per_row: ArrayLike,
    mae_per_row: ArrayLike,
    rmse_per_row: ArrayLike,
    max_error_per_row: ArrayLike,
    threshold_metric: ThresholdMetric = ThresholdMetric.MSE,
) -> pd.DataFrame:
    """
    Create a DataFrame with all reconstruction error metrics.

    The ``Score`` column holds each row's reconstruction error under the active
    ``threshold_metric`` (a per-sample anomaly score consistent with the metric used
    for thresholding rather than always MSE), and ``Score_Mean`` holds the mean of
    those per-row scores broadcast across rows as cluster-level metadata.
    """
    per_row_by_metric = {
        ThresholdMetric.MSE: mse_per_row,
        ThresholdMetric.MAE: mae_per_row,
        ThresholdMetric.RMSE: rmse_per_row,
        ThresholdMetric.MAX_ERROR: max_error_per_row,
    }
    active_score = np.asarray(per_row_by_metric[threshold_metric], dtype="float64")

    metrics_df = pd.DataFrame(
        {
            "MSE_Recons_error": np.asarray(mse_per_row, dtype="float64"),
            "MAE_Recons_error": np.asarray(mae_per_row, dtype="float64"),
            "RMSE_Recons_error": np.asarray(rmse_per_row, dtype="float64"),
            "Max_Recons_error": np.asarray(max_error_per_row, dtype="float64"),
            "Score": active_score,
            "Score_Mean": np.mean(active_score) if len(active_score) else np.nan,
        }
    )

    return metrics_df


def compute_threshold(reconstruction_errors: ArrayLike, threshold_ratio: float) -> float:
    """
    Compute reconstruction error threshold for anomaly detection.

    Calculates threshold as mean of reconstruction errors multiplied
    by the specified ratio.
    """
    return float(np.mean(reconstruction_errors) * threshold_ratio)


def detect_anomalies(
    reconstruction_errors: Union[pd.Series, np.ndarray], threshold: float
) -> Union[pd.Series, np.ndarray]:
    """
    Detect anomalies based on reconstruction error threshold.

    Labels samples as anomalies (1) if their reconstruction error
    exceeds the threshold, otherwise as normal (0). Returns the same
    container type it receives (Series in, Series out).
    """
    if isinstance(reconstruction_errors, pd.Series):
        return (reconstruction_errors > threshold).astype("int64")
    return (np.asarray(reconstruction_errors) > threshold).astype("int64")
