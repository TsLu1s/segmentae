from enum import Enum
from typing import Dict, Union


class _ValueEnum(str, Enum):
    """
    String-valued enum whose text form is always its value.

    A plain ``(str, Enum)`` formats as its value on Python 3.10 but as
    ``ClassName.MEMBER`` on Python 3.11+. Defining ``__str__`` makes ``str()``,
    ``format()`` and f-strings return the value on every supported version, so
    messages, reports and DataFrame cells never depend on the interpreter.
    """

    def __str__(self) -> str:
        return str(self.value)


class PhaseType(_ValueEnum):
    """Pipeline execution phases for SegmentAE reconstruction workflow."""

    EVALUATION = "evaluation"
    TESTING = "testing"
    PREDICTION = "prediction"


class ClusterModel(_ValueEnum):
    """Built-in clustering algorithms for data segmentation."""

    KMEANS = "KMeans"
    MINIBATCH_KMEANS = "MiniBatchKMeans"
    GMM = "GMM"


class ThresholdMetric(_ValueEnum):
    """Reconstruction error metrics for anomaly detection thresholding."""

    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    MAX_ERROR = "max_error"


class PerformanceMetric(_ValueEnum):
    """Classification metrics the optimizer can rank configurations by."""

    ACCURACY = "Accuracy"
    PRECISION = "Precision"
    RECALL = "Recall"
    F1_SCORE = "F1 Score"


class EncoderType(_ValueEnum):
    """Categorical variable encoding methods."""

    IFREQUENCY = "IFrequencyEncoder"
    LABEL = "LabelEncoder"
    ONEHOT = "OneHotEncoder"


class ScalerType(_ValueEnum):
    """Feature scaling methods for numerical normalization."""

    MINMAX = "MinMaxScaler"
    STANDARD = "StandardScaler"
    ROBUST = "RobustScaler"


class ImputerType(_ValueEnum):
    """Missing value imputation methods."""

    SIMPLE = "Simple"


METRIC_COLUMN_MAP: Dict[ThresholdMetric, str] = {
    ThresholdMetric.MSE: "MSE_Recons_error",
    ThresholdMetric.MAE: "MAE_Recons_error",
    ThresholdMetric.RMSE: "RMSE_Recons_error",
    ThresholdMetric.MAX_ERROR: "Max_Recons_error",
}

METRIC_NAME_MAP: Dict[str, ThresholdMetric] = {metric.value: metric for metric in ThresholdMetric}

# Accepted spellings for each performance metric (compared case-insensitively,
# ignoring spaces, hyphens and underscores).
_PERFORMANCE_METRIC_ALIASES: Dict[str, PerformanceMetric] = {
    "accuracy": PerformanceMetric.ACCURACY,
    "precision": PerformanceMetric.PRECISION,
    "recall": PerformanceMetric.RECALL,
    "f1": PerformanceMetric.F1_SCORE,
    "f1score": PerformanceMetric.F1_SCORE,
}

ENCODER_CLASS_MAP: Dict[EncoderType, str] = {
    EncoderType.IFREQUENCY: "AutoIFrequencyEncoder",
    EncoderType.LABEL: "AutoLabelEncoder",
    EncoderType.ONEHOT: "AutoOneHotEncoder",
}

SCALER_CLASS_MAP: Dict[ScalerType, str] = {
    ScalerType.MINMAX: "AutoMinMaxScaler",
    ScalerType.STANDARD: "AutoStandardScaler",
    ScalerType.ROBUST: "AutoRobustScaler",
}

IMPUTER_CLASS_MAP: Dict[ImputerType, str] = {ImputerType.SIMPLE: "AutoSimpleImputer"}


def get_metric_column_name(metric: ThresholdMetric) -> str:
    """Return the reconstruction-metrics column that holds ``metric``."""
    return METRIC_COLUMN_MAP[metric]


def parse_threshold_metric(metric_str: Union[str, ThresholdMetric]) -> ThresholdMetric:
    """
    Convert a metric name to a :class:`ThresholdMetric` (case-insensitive).

    Raises:
        ValueError: If the name is not a known threshold metric.
    """
    if isinstance(metric_str, ThresholdMetric):
        return metric_str

    metric_lower = str(metric_str).strip().lower()
    if metric_lower not in METRIC_NAME_MAP:
        valid_metrics = ", ".join(METRIC_NAME_MAP.keys())
        raise ValueError(
            f"Unknown threshold metric: '{metric_str}'. " f"Valid options are: {valid_metrics}"
        )
    return METRIC_NAME_MAP[metric_lower]


def parse_performance_metric(metric: Union[str, PerformanceMetric]) -> PerformanceMetric:
    """
    Convert a performance metric name to a :class:`PerformanceMetric`.

    Accepts the enum, the result column name (``"F1 Score"``) or the usual
    shorthand (``"f1_score"``, ``"f1"``, ``"accuracy"``, ...), case-insensitively.

    Raises:
        ValueError: If the name is not a known performance metric.
    """
    if isinstance(metric, PerformanceMetric):
        return metric

    key = "".join(ch for ch in str(metric).lower() if ch.isalnum())
    if key not in _PERFORMANCE_METRIC_ALIASES:
        valid_metrics = ["accuracy", "precision", "recall", "f1_score"]
        raise ValueError(
            f"Invalid performance metric: '{metric}'. " f"Valid options: {valid_metrics}"
        )
    return _PERFORMANCE_METRIC_ALIASES[key]
