import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union

from pydantic import BaseModel, field_validator, ConfigDict

from segmentae.core.constants import (
    ThresholdMetric,
    parse_threshold_metric
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    max_error
)

class ReconstructionConfig(BaseModel):
    """Configuration for reconstruction phase."""
    
    threshold_metric: Union[ThresholdMetric, str] = ThresholdMetric.MSE
    
    @field_validator('threshold_metric', mode='before')
    def convert_to_enum(cls, v):
        """Convert string to ThresholdMetric enum."""
        if isinstance(v, ThresholdMetric):
            return v
        if isinstance(v, str):
            return parse_threshold_metric(v)
        raise ValueError(f"Invalid threshold_metric type: {type(v)}")
    
    model_config = ConfigDict(use_enum_values=False)

class EvaluationConfig(BaseModel):
    """Configuration for evaluation phase."""
    
    threshold_ratio: float = 1.0
    
    @field_validator('threshold_ratio', mode='before')
    def validate_ratio(cls, v):
        """Validate threshold ratio is positive."""
        if v <= 0:
            raise ValueError(
                f"threshold_ratio must be positive, got {v}"
            )
        return v

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
    real_values: np.ndarray,
    predicted_values: np.ndarray
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute per-row reconstruction errors for multiple metrics.
    
    Calculates MSE, MAE, RMSE, and Max Error for each row,
    comparing real values against autoencoder reconstructions.
    """
    mse_per_row = []
    mae_per_row = []
    rmse_per_row = []
    max_error_per_row = []
    
    for i in range(len(real_values)):
        row = real_values[i]
        pred_row = predicted_values[i]
        
        # Calculate MSE for the row
        mse = mean_squared_error(row, pred_row)
        mse_per_row.append(mse)
        
        # Calculate MAE for the row
        mae = mean_absolute_error(row, pred_row)
        mae_per_row.append(mae)
        
        # Calculate RMSE for the row
        rmse = np.sqrt(mse)
        rmse_per_row.append(rmse)
        
        # Calculate Max Error for the row
        max_err = max_error(row, pred_row)
        max_error_per_row.append(max_err)
    
    return mse_per_row, mae_per_row, rmse_per_row, max_error_per_row

def compute_column_metrics(
    real_values: np.ndarray,
    predicted_values: np.ndarray,
    columns: List[str],
    cluster_id: int
) -> pd.DataFrame:
    """
    Compute per-column reconstruction metrics.
    
    Calculates MSE, MAE, RMSE, and Max Error for each feature column,
    providing insight into which features are reconstructed well.
    """
    # Calculate per-column metrics
    col_metrics = pd.DataFrame({
        'Column': columns,
        'MSE': list(np.mean(np.square(real_values - predicted_values), axis=0)),
        'MAE': list(np.mean(np.abs(real_values - predicted_values), axis=0)),
        'RMSE': list(np.sqrt(np.mean(np.square(real_values - predicted_values), axis=0))),
        'Max_Error': list(np.max(np.abs(real_values - predicted_values), axis=0)),
        'partition': cluster_id
    })
    
    # Add total metrics row
    total_metrics = pd.DataFrame({
        'Column': ['Total Metrics'],
        'MSE': [col_metrics['MSE'].mean()],
        'MAE': [col_metrics['MAE'].mean()],
        'RMSE': [col_metrics['RMSE'].mean()],
        'Max_Error': [col_metrics['Max_Error'].max()]
    })
    
    return pd.concat([col_metrics, total_metrics], ignore_index=True)

def create_metrics_dataframe(
    mse_per_row: List[float],
    mae_per_row: List[float],
    rmse_per_row: List[float],
    max_error_per_row: List[float]
) -> pd.DataFrame:
    """
    Create a DataFrame with all reconstruction error metrics.
    """
    metrics_df = pd.DataFrame({
        'MSE_Recons_error': mse_per_row,
        'MAE_Recons_error': mae_per_row,
        'RMSE_Recons_error': rmse_per_row,
        'Max_Recons_error': max_error_per_row,
        'Score': np.array(mse_per_row).mean() + np.array(mse_per_row).std()
    })
    
    return metrics_df

def aggregate_cluster_results(
    cluster_results: List[Dict],
) -> Tuple[pd.DataFrame, Dict[int, Dict], pd.DataFrame]:
    """
    Aggregate evaluation results across clusters.
    
    Combines cluster-level metrics, confusion matrices, and predictions
    into structured outputs for global analysis.
    """
    # Cluster-level metrics
    cluster_metrics = pd.concat(
        [result["metrics"] for result in cluster_results],
        ignore_index=True
    )
    
    # Confusion matrices
    confusion_matrices = {
        result["cluster_id"]: {
            f"CM_{result['cluster_id']}": result["confusion_matrix"]
        }
        for result in cluster_results
    }
    
    # Global predictions
    all_predictions = []
    for result in cluster_results:
        pred_df = pd.DataFrame({
            'index': result["indices"],
            'y_test': result["y_test"],
            'Predicted Anomalies': result["predictions"]['Predicted Anomalies']
        })
        all_predictions.append(pred_df)
    
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(by='index').set_index('index')
    
    return cluster_metrics, confusion_matrices, predictions


def compute_threshold(
    reconstruction_errors: pd.Series,
    threshold_ratio: float
) -> float:
    """
    Compute reconstruction error threshold for anomaly detection.
    
    Calculates threshold as mean of reconstruction errors multiplied
    by the specified ratio.
    """
    return np.mean(reconstruction_errors) * threshold_ratio


def detect_anomalies(
    reconstruction_errors: pd.Series,
    threshold: float
) -> pd.Series:
    """
    Detect anomalies based on reconstruction error threshold.
    
    Labels samples as anomalies (1) if their reconstruction error
    exceeds the threshold, otherwise as normal (0).
    """
    return reconstruction_errors.apply(lambda x: 1 if x > threshold else 0)