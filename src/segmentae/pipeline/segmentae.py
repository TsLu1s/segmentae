import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from segmentae.clusters.clustering import Clustering
from segmentae.core.constants import ThresholdMetric, get_metric_column_name
from segmentae.core.exceptions import (
    AutoencoderError,
    ConfigurationError,
    ModelNotFittedError,
    ValidationError,
)
from segmentae.core.types import AutoencoderProtocol
from segmentae.metrics.performance_metrics import metrics_classification
from segmentae.pipeline.reconstruction import (  # noqa: F401  (EvaluationConfig is re-exported)
    EvaluationConfig,
    ReconstructionConfig,
    compute_column_metrics,
    compute_reconstruction_errors,
    create_metrics_dataframe,
)
from segmentae.utils.validation import (
    validate_binary_target,
    validate_columns_match,
    validate_dataframe,
    validate_threshold_ratio,
)

# Columns that detections() adds next to the reconstructed features
_OUTPUT_COLUMNS = ("Predicted Anomalies", "Cluster", "Reconstruction Error", "Threshold")


@dataclass
class _Reconstruction:
    """Reconstruction of one dataset: per-cluster views plus row-aligned arrays."""

    preds: Dict[Any, Dict]  # per cluster: real values, reconstructions, labels
    metrics: Dict[Any, Dict]  # per cluster: per-row and per-column error metrics
    labels: np.ndarray  # cluster assigned to each row (input order)
    values: np.ndarray  # reconstructed values, shape (n_rows, n_features)
    scores: np.ndarray  # reconstruction error of each row under the active metric
    index: pd.Index  # index of the input frame
    columns: List  # feature columns


class SegmentAE:
    """
    SegmentAE integrates autoencoders with clustering for anomaly detection.

    This class orchestrates the reconstruction, evaluation, and detection pipeline
    by combining autoencoder reconstruction errors with cluster-specific thresholding
    to optimize anomaly detection performance.

    The workflow consists of three phases:
    1. Reconstruction: Compute reconstruction errors on training data
    2. Evaluation: Test anomaly detection on labeled test data
    3. Detection: Predict anomalies on unlabeled data

    Rows are always handled by position, so input frames may carry any index
    (it does not need to be reset) and outputs keep the index of their input.
    """

    def __init__(self, ae_model: Any, cl_model: Clustering, verbose: bool = True):
        """
        Initialize SegmentAE pipeline.

        Args:
            ae_model: Trained autoencoder exposing ``predict(X)`` (a built-in
                autoencoder, a Keras model or any compatible object).
            cl_model: Fitted ``Clustering`` instance.
            verbose: Print the per-cluster thresholds during ``evaluation``.
        """
        # Validate inputs
        self._validate_autoencoder(ae_model)
        self._validate_clustering(cl_model)

        # Store models
        self.ae_model = ae_model
        self.cl_model = cl_model
        self.verbose = verbose

        # SegmentAE segments data using a single clustering model. If several were
        # configured, only the first is used for reconstruction; warn so this is explicit.
        if len(cl_model.cluster_model) > 1:
            warnings.warn(
                f"Clustering was configured with {len(cl_model.cluster_model)} models "
                f"({cl_model.cluster_model}); SegmentAE only uses the first "
                f"('{cl_model.cluster_model[0]}') for segmentation.",
                UserWarning,
                stacklevel=2,
            )

        # Training reference (set by reconstruction())
        self._threshold_metric: Optional[ThresholdMetric] = None
        self._metric_column: Optional[str] = None
        self._feature_names: Optional[List] = None
        self._base_thresholds: Dict[Any, float] = {}
        self._global_threshold: Optional[float] = None
        self._is_reconstruction_fitted: bool = False

        # Most recently evaluated test input (object identity), so repeated
        # evaluation() calls over the same data (e.g. a threshold sweep) don't
        # recompute the autoencoder reconstruction each time. The reconstruction
        # depends only on input_data/target_col, not on threshold_ratio.
        self._last_eval_input: Optional[pd.DataFrame] = None
        self._last_eval_target: Optional[pd.Series] = None
        self._test_reconstruction: Optional[_Reconstruction] = None

        # Optional {(id(ae_model), id(frame)): (frame, reconstruction)} store shared
        # between pipelines that reuse one autoencoder (set by SegmentAE_Optimizer).
        self._reconstruction_cache: Optional[Dict[Tuple[int, int], Tuple[Any, np.ndarray]]] = None

        # Results storage (dictionaries keyed by cluster id)
        self.preds_train: Dict[Any, Dict] = {}
        self.preds_test: Dict[Any, Dict] = {}
        self.preds_final: Dict[Any, Dict] = {}
        self.reconstruction_eval: Dict[Any, Dict] = {}
        self.reconstruction_test: Dict[Any, Dict] = {}
        self.reconstruction_final: Dict[Any, Dict] = {}
        self.results: Dict = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reconstruction(
        self,
        input_data: pd.DataFrame,
        target_col: Optional[pd.Series] = None,
        threshold_metric: str = "mse",
    ) -> "SegmentAE":
        """
        Reconstruct input data and compute reconstruction errors per cluster.

        This method segments data by cluster, generates autoencoder reconstructions,
        and computes reconstruction errors using the specified metric. The mean
        error of each cluster becomes its baseline anomaly threshold.

        Args:
            input_data: Training (reference) features, already preprocessed.
            target_col: Optional labels, stored alongside the per-cluster results.
            threshold_metric: "mse", "mae", "rmse" or "max_error" (or a ``ThresholdMetric``).

        Returns:
            The fitted pipeline (for chaining).
        """
        # Validate and configure
        self._validate_reconstruction_input(input_data, target_col)
        metric = ReconstructionConfig(threshold_metric=threshold_metric).threshold_metric
        metric_column = get_metric_column_name(metric)

        result = self._reconstruct(input_data, _align_target(input_data, target_col), metric)

        # Commit the new training reference only once everything succeeded
        self._threshold_metric = metric
        self._metric_column = metric_column
        self._feature_names = list(result.columns)
        self.preds_train = result.preds
        self.reconstruction_eval = result.metrics
        self._base_thresholds = {
            cluster_id: float(np.mean(cluster["metrics"][metric_column]))
            for cluster_id, cluster in result.metrics.items()
        }
        self._global_threshold = float(np.mean(result.scores))

        # A fresh training reference invalidates any earlier test/detection results
        self._last_eval_input = None
        self._last_eval_target = None
        self._test_reconstruction = None
        self.preds_test, self.reconstruction_test = {}, {}
        self.preds_final, self.reconstruction_final = {}, {}
        self.results = {}

        self._is_reconstruction_fitted = True
        return self

    def evaluation(
        self, input_data: pd.DataFrame, target_col: pd.Series, threshold_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluate anomaly detection performance on labeled test data.

        Computes cluster-specific thresholds and evaluates detection performance
        against ground truth labels, providing both global and cluster-level metrics.

        Args:
            input_data: Test features, with the columns used in ``reconstruction``.
            target_col: Binary labels (0 = normal, 1 = anomaly). Matched to the
                rows of ``input_data`` by index label when both carry the same
                labels, otherwise by position.
            threshold_ratio: Multiplier applied to each cluster's baseline
                threshold; higher values flag fewer samples.

        Returns:
            Dictionary with "global metrics", "clusters metrics",
            "confusion matrix" and "y_true vs y_pred".
        """
        self._validate_fitted_for_evaluation()
        self._validate_evaluation_input(input_data, target_col)

        # Threshold ratio adjusts the detection sensitivity relative to baseline threshold
        ratio = validate_threshold_ratio(threshold_ratio)
        labels = _align_target(input_data, target_col)

        # Reconstruction is independent of threshold_ratio, so reuse the cached test
        # reconstruction when re-evaluating the same data (e.g. a threshold sweep).
        cache_hit = (
            input_data is self._last_eval_input
            and target_col is self._last_eval_target
            and self._test_reconstruction is not None
        )
        if not cache_hit:
            aligned = validate_columns_match(input_data, self._feature_names, "Input data")
            result = self._reconstruct(aligned, labels, self._threshold_metric)
            self._test_reconstruction = result
            self.preds_test, self.reconstruction_test = result.preds, result.metrics
            self._last_eval_input = input_data
            self._last_eval_target = target_col

        result = self._test_reconstruction
        thresholds = self._resolve_thresholds(result.metrics.keys(), ratio)
        y_true = labels.to_numpy().astype("int64")
        y_pred = self._flag_anomalies(result, thresholds)

        self.results = self._build_evaluation_results(result, thresholds, y_true, y_pred, ratio)
        return self.results

    def detections(self, input_data: pd.DataFrame, threshold_ratio: float = 1.0) -> pd.DataFrame:
        """
        Perform anomaly detection on unlabeled data.

        Uses trained cluster-specific thresholds to detect anomalies in new data
        without requiring ground truth labels.

        Returns:
            DataFrame indexed like ``input_data`` holding the reconstructed
            feature values plus, for every row:

            - ``Predicted Anomalies``: 1 if the row is flagged, else 0
            - ``Cluster``: cluster the row was assigned to
            - ``Reconstruction Error``: the row's anomaly score (threshold metric)
            - ``Threshold``: threshold the score was compared against
        """
        self._validate_fitted_for_detection()
        self._validate_input(input_data, "Input data")
        ratio = validate_threshold_ratio(threshold_ratio)

        aligned = validate_columns_match(input_data, self._feature_names, "Input data")
        result = self._reconstruct(aligned, None, self._threshold_metric)
        self.preds_final, self.reconstruction_final = result.preds, result.metrics

        thresholds = self._resolve_thresholds(result.metrics.keys(), ratio)

        predicted, cluster, score, threshold = _OUTPUT_COLUMNS
        predictions = pd.DataFrame(result.values, columns=result.columns, index=result.index)
        predictions[predicted] = self._flag_anomalies(result, thresholds)
        predictions[cluster] = result.labels
        predictions[score] = result.scores
        predictions[threshold] = self._row_thresholds(result.labels, thresholds)
        return predictions

    def get_thresholds(self, threshold_ratio: float = 1.0) -> Dict[Any, float]:
        """
        Return the anomaly threshold of each training cluster for a given ratio.
        """
        self._validate_fitted_for_detection()
        ratio = validate_threshold_ratio(threshold_ratio)
        return {cluster_id: base * ratio for cluster_id, base in self._base_thresholds.items()}

    @property
    def thresholds(self) -> Dict[Any, float]:
        """Baseline threshold (mean training error) of each cluster, i.e. ratio 1.0."""
        return dict(self._base_thresholds)

    @property
    def global_threshold(self) -> Optional[float]:
        """
        Mean training error over all samples.

        Used as the fallback baseline for samples assigned to a cluster that had
        no training samples. None before ``reconstruction``.
        """
        return self._global_threshold

    @property
    def threshold_metric(self) -> Optional[ThresholdMetric]:
        """Reconstruction metric the thresholds are based on (None before fitting)."""
        return self._threshold_metric

    # ------------------------------------------------------------------ #
    # Reconstruction
    # ------------------------------------------------------------------ #

    def _reconstruct(
        self, input_data: pd.DataFrame, target_col: Optional[pd.Series], metric: ThresholdMetric
    ) -> _Reconstruction:
        """
        Reconstruct ``input_data`` once and split the results by cluster.

        The autoencoder is called a single time for the whole frame; errors are
        computed for all rows at once and then grouped by cluster position.
        """
        feature_names = self.cl_model.feature_names
        if feature_names is not None:
            input_data = validate_columns_match(input_data, feature_names, "Input data")

        labels = self._get_cluster_predictions(input_data).iloc[:, 0].to_numpy()
        values = self._predict(input_data)
        real = input_data.to_numpy(dtype="float64")
        columns = list(input_data.columns)

        mse, mae, rmse, max_err = compute_reconstruction_errors(real, values)
        scores = {
            ThresholdMetric.MSE: mse,
            ThresholdMetric.MAE: mae,
            ThresholdMetric.RMSE: rmse,
            ThresholdMetric.MAX_ERROR: max_err,
        }[metric]

        preds: Dict[Any, Dict] = {}
        metrics: Dict[Any, Dict] = {}
        for label in np.unique(labels):
            cluster_id = _as_python(label)
            positions = np.flatnonzero(labels == label)
            index_labels = input_data.index[positions].tolist()

            metrics[cluster_id] = {
                "cluster": cluster_id,
                "metrics": create_metrics_dataframe(
                    mse[positions], mae[positions], rmse[positions], max_err[positions], metric
                ),
                "column_metrics": compute_column_metrics(
                    real[positions], values[positions], columns, cluster_id
                ),
                "indexs": index_labels,
                "positions": positions,
            }
            preds[cluster_id] = {
                "cluster": cluster_id,
                "real": input_data.iloc[positions],
                "y_true": target_col.iloc[positions] if target_col is not None else None,
                "predictions": pd.DataFrame(values[positions], columns=columns),
                "indexs": index_labels,
                "positions": positions,
            }

        return _Reconstruction(
            preds=preds,
            metrics=metrics,
            labels=labels,
            values=values,
            scores=scores,
            index=input_data.index,
            columns=columns,
        )

    def _predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Generate (and validate) autoencoder reconstructions for a whole frame."""
        cache = self._reconstruction_cache
        key = (id(self.ae_model), id(input_data))
        if cache is not None and key in cache and cache[key][0] is input_data:
            return cache[key][1]

        try:
            if _accepts_verbose(self.ae_model.predict):
                output = self.ae_model.predict(input_data, verbose=0)
            else:
                output = self.ae_model.predict(input_data)
            values = np.asarray(output, dtype="float64")
        except Exception as e:
            raise AutoencoderError(f"Failed to generate reconstructions: {str(e)}") from e

        if values.ndim == 1 and input_data.shape[1] == 1:
            values = values.reshape(-1, 1)  # single-feature models may return a flat array

        if values.shape != input_data.shape:
            raise AutoencoderError(
                f"Autoencoder returned reconstructions of shape {values.shape}, expected "
                f"{input_data.shape} (one row per sample, one column per feature)"
            )
        if not np.isfinite(values).all():
            raise AutoencoderError(
                "Autoencoder produced non-finite reconstructions (NaN or infinity). "
                "This usually means training diverged; check the input scaling and "
                "the learning rate, then retrain the model."
            )

        if cache is not None:
            cache[key] = (input_data, values)
        return values

    def _get_cluster_predictions(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Get cluster assignments for input data."""
        return self.cl_model.cluster_prediction(X=input_data)

    # ------------------------------------------------------------------ #
    # Thresholding
    # ------------------------------------------------------------------ #

    def _resolve_thresholds(self, cluster_ids: Iterable, ratio: float) -> Dict[Any, float]:
        """
        Threshold for each cluster present in the data being scored.

        Clusters without a training reference (no training sample was assigned
        to them) fall back to the global training threshold, so every input row
        is always scored.
        """
        thresholds: Dict[Any, float] = {}
        unseen = []
        for cluster_id in cluster_ids:
            base = self._base_thresholds.get(cluster_id)
            if base is None:
                unseen.append(cluster_id)
                base = self._global_threshold
            thresholds[cluster_id] = base * ratio

        if unseen:
            warnings.warn(
                f"Cluster(s) {unseen} have no training reference (no training sample was "
                f"assigned to them); their samples are scored with the global training "
                f"threshold instead of a cluster-specific one.",
                UserWarning,
                stacklevel=3,
            )
        return thresholds

    @staticmethod
    def _row_thresholds(labels: np.ndarray, thresholds: Dict[Any, float]) -> np.ndarray:
        """Threshold applicable to each row, given its cluster label."""
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        per_cluster = np.array([thresholds[_as_python(label)] for label in unique_labels])
        return per_cluster[inverse]

    def _flag_anomalies(self, result: _Reconstruction, thresholds: Dict[Any, float]) -> np.ndarray:
        """1 for rows whose reconstruction error exceeds their cluster threshold, else 0."""
        row_thresholds = self._row_thresholds(result.labels, thresholds)
        return (result.scores > row_thresholds).astype("int64")

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def _build_evaluation_results(
        self,
        result: _Reconstruction,
        thresholds: Dict[Any, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold_ratio: float,
    ) -> Dict[str, Any]:
        """Compute cluster-level and global evaluation metrics."""
        metric_name = self._threshold_metric.value.upper()
        cluster_metrics = []
        confusion_matrices = {}

        for cluster_id, cluster in result.metrics.items():
            positions = cluster["positions"]
            threshold = thresholds[cluster_id]
            if self.verbose:
                print(f"Cluster {cluster_id} || Reconstruction Threshold: {round(threshold, 5)}")

            metrics = metrics_classification(y_true[positions], y_pred[positions])
            metrics["N_Cluster"] = cluster_id
            metrics["Threshold Metric"] = metric_name
            metrics["Threshold Value"] = round(threshold, 6)
            cluster_metrics.append(metrics)

            # labels=[0, 1] keeps the matrix 2x2 even when a cluster holds one class
            confusion_matrices[cluster_id] = {
                f"CM_{cluster_id}": confusion_matrix(
                    y_true[positions], y_pred[positions], labels=[0, 1]
                )
            }

        if self.verbose:
            print("")

        # Global predictions, in the order (and with the index) of the input data
        index = result.index
        if index.name is None and not isinstance(index, pd.MultiIndex):
            index = index.rename("index")
        ytpred = pd.DataFrame({"y_test": y_true, "Predicted Anomalies": y_pred}, index=index)

        # Global metrics
        global_metrics = metrics_classification(y_true, y_pred)
        global_metrics["Threshold Metric"] = metric_name
        global_metrics["Threshold Ratio"] = threshold_ratio

        return {
            "global metrics": global_metrics,
            "clusters metrics": pd.concat(cluster_metrics, ignore_index=True),
            "confusion matrix": confusion_matrices,
            "y_true vs y_pred": ytpred,
        }

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_autoencoder(self, ae_model: Any) -> None:
        """Validate that autoencoder has required interface."""
        if not isinstance(ae_model, AutoencoderProtocol) or not callable(ae_model.predict):
            raise ConfigurationError(
                "Autoencoder must have a 'predict' method. "
                "Ensure you're passing a trained Keras model or built-in autoencoder."
            )

    def _validate_clustering(self, cl_model: Clustering) -> None:
        """Validate clustering model."""
        if not isinstance(cl_model, Clustering):
            raise ConfigurationError(
                "cl_model must be an instance of Clustering", valid_options=["Clustering"]
            )
        if not cl_model.is_fitted:
            raise ModelNotFittedError(
                component="Clustering",
                message="Clustering model must be fitted before use. "
                "Call clustering_fit(X) method first.",
            )

    def _validate_reconstruction_input(
        self, input_data: pd.DataFrame, target_col: Optional[pd.Series]
    ) -> None:
        """Validate reconstruction inputs."""
        self._validate_input(input_data, "Input data")

        reserved = [str(col) for col in input_data.columns if col in _OUTPUT_COLUMNS]
        if reserved:
            raise ValidationError(
                f"Feature columns {reserved} clash with the columns added to the "
                f"detection output {list(_OUTPUT_COLUMNS)}",
                suggestion="Rename these feature columns before fitting",
            )

        if target_col is not None:
            if not isinstance(target_col, pd.Series):
                raise ValidationError(
                    f"target_col must be a pandas Series, got {type(target_col).__name__}",
                    suggestion="Convert to Series using pd.Series() or use DataFrame column",
                )
            if len(target_col) != len(input_data):
                raise ValidationError(
                    f"target_col length ({len(target_col)}) must match "
                    f"input_data length ({len(input_data)})"
                )

    def _validate_evaluation_input(self, input_data: pd.DataFrame, target_col: pd.Series) -> None:
        """Validate evaluation inputs."""
        self._validate_input(input_data, "Input data")

        if not isinstance(target_col, pd.Series):
            raise ValidationError(
                f"target_col must be a pandas Series, got {type(target_col).__name__}",
                suggestion="Use test[target_column] to extract Series",
            )

        if len(target_col) != len(input_data):
            raise ValidationError(
                f"target_col length ({len(target_col)}) must match "
                f"input_data length ({len(input_data)})"
            )

        validate_binary_target(target_col, "target_col")

    def _validate_input(self, X: pd.DataFrame, context: str = "Input") -> None:
        """Validate input DataFrame (non-empty, numeric, no missing or infinite values)."""
        validate_dataframe(X, context, allow_missing=False, require_numeric=True)

    def _validate_fitted_for_evaluation(self) -> None:
        """Validate that reconstruction has been performed."""
        if not self._is_reconstruction_fitted:
            raise ModelNotFittedError(
                component="SegmentAE",
                message="Must call reconstruction() before evaluation(). "
                "Example: sg.reconstruction(X_train, threshold_metric='mse')",
            )

    def _validate_fitted_for_detection(self) -> None:
        """Validate that reconstruction has been performed."""
        if not self._is_reconstruction_fitted:
            raise ModelNotFittedError(
                component="SegmentAE",
                message="Must call reconstruction() before detections(). "
                "Example: sg.reconstruction(X_train, threshold_metric='mse')",
            )

    def __repr__(self) -> str:
        """String representation of SegmentAE."""
        ae_name = type(self.ae_model).__name__
        return (
            f"SegmentAE("
            f"autoencoder={ae_name}, "
            f"clustering={self.cl_model.cluster_model}, "
            f"fitted={self._is_reconstruction_fitted})"
        )


def _align_target(input_data: pd.DataFrame, target_col: Optional[pd.Series]) -> Optional[pd.Series]:
    """
    Pair each label with its row and return the labels indexed like ``input_data``.

    Labels are matched by index label when the target carries exactly the index
    labels of the data (in any order), which is how pandas aligns two objects.
    Otherwise they are paired by position, so a target built from a plain array
    works with data that has any index.
    """
    if target_col is None or target_col.index.equals(input_data.index):
        return target_col

    same_labels = (
        input_data.index.is_unique
        and target_col.index.is_unique
        and target_col.index.isin(input_data.index).all()
    )
    if same_labels:
        return target_col.loc[input_data.index]

    return pd.Series(target_col.to_numpy(), index=input_data.index, name=target_col.name)


def _as_python(value: Any) -> Any:
    """Convert a NumPy scalar (e.g. a cluster label) to its Python equivalent."""
    return value.item() if isinstance(value, np.generic) else value


def _accepts_verbose(predict: Any) -> bool:
    """True if ``predict`` declares a ``verbose`` parameter (e.g. Keras models)."""
    try:
        return "verbose" in inspect.signature(predict).parameters
    except (TypeError, ValueError):
        return False
