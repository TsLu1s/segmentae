from enum import Enum
from typing import Dict

class PhaseType(str, Enum):
    """Pipeline execution phases for SegmentAE reconstruction workflow."""
    
    EVALUATION = "evaluation"
    TESTING = "testing"
    PREDICTION = "prediction"

class ClusterModel(str, Enum):
    """Available clustering algorithms for data segmentation."""
    
    KMEANS = "KMeans"
    MINIBATCH_KMEANS = "MiniBatchKMeans"
    GMM = "GMM"
    AGGLOMERATIVE = "Agglomerative"

class ThresholdMetric(str, Enum):
    """Reconstruction error metrics for anomaly detection thresholding."""
    
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    MAX_ERROR = "max_error"

class EncoderType(str, Enum):
    """Categorical variable encoding methods."""
    
    IFREQUENCY = "IFrequencyEncoder"
    LABEL = "LabelEncoder"
    ONEHOT = "OneHotEncoder"

class ScalerType(str, Enum):
    """Feature scaling methods for numerical normalization."""
    
    MINMAX = "MinMaxScaler"
    STANDARD = "StandardScaler"
    ROBUST = "RobustScaler"

class ImputerType(str, Enum):
    """Missing value imputation methods."""
    
    SIMPLE = "Simple"

# Mapping dictionaries
METRIC_COLUMN_MAP: Dict[ThresholdMetric, str] = {
    ThresholdMetric.MSE: "MSE_Recons_error",
    ThresholdMetric.MAE: "MAE_Recons_error",
    ThresholdMetric.RMSE: "RMSE_Recons_error",
    ThresholdMetric.MAX_ERROR: "Max_Recons_error"
}

METRIC_NAME_MAP: Dict[str, ThresholdMetric] = {
    "mse": ThresholdMetric.MSE,
    "mae": ThresholdMetric.MAE,
    "rmse": ThresholdMetric.RMSE,
    "max_error": ThresholdMetric.MAX_ERROR
}

ENCODER_CLASS_MAP: Dict[EncoderType, str] = {
    EncoderType.IFREQUENCY: "AutoIFrequencyEncoder",
    EncoderType.LABEL: "AutoLabelEncoder",
    EncoderType.ONEHOT: "AutoOneHotEncoder"
}

SCALER_CLASS_MAP: Dict[ScalerType, str] = {
    ScalerType.MINMAX: "AutoMinMaxScaler",
    ScalerType.STANDARD: "AutoStandardScaler",
    ScalerType.ROBUST: "AutoRobustScaler"
}

IMPUTER_CLASS_MAP: Dict[ImputerType, str] = {
    ImputerType.SIMPLE: "AutoSimpleImputer"
}

def get_metric_column_name(metric: ThresholdMetric) -> str:

    return METRIC_COLUMN_MAP[metric]

def parse_threshold_metric(metric_str: str) -> ThresholdMetric:

    metric_lower = metric_str.lower()
    if metric_lower not in METRIC_NAME_MAP:
        valid_metrics = ", ".join(METRIC_NAME_MAP.keys())
        raise ValueError(
            f"Unknown threshold metric: '{metric_str}'. "
            f"Valid options are: {valid_metrics}"
        )
    return METRIC_NAME_MAP[metric_lower]