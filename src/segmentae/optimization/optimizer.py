from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from segmentae.clusters.clustering import Clustering
from segmentae.clusters.models import normalize_cluster_model_name
from segmentae.core.constants import (
    ClusterModel,
    PerformanceMetric,
    ThresholdMetric,
    parse_performance_metric,
    parse_threshold_metric,
)
from segmentae.core.exceptions import ConfigurationError, ValidationError
from segmentae.core.types import AutoencoderProtocol
from segmentae.pipeline.segmentae import SegmentAE
from segmentae.utils.validation import (
    validate_binary_target,
    validate_columns_match,
    validate_dataframe,
)


class OptimizerConfig(BaseModel):
    """Configuration for grid search optimizer."""

    model_config = ConfigDict(use_enum_values=False)

    n_clusters_list: List[int] = [1, 2, 3, 4]
    cluster_models: List[str] = ["KMeans", "MiniBatchKMeans", "GMM"]
    threshold_ratios: List[float] = [0.75, 1, 1.5, 2, 3, 4]
    # Defaults are enum members: Pydantic does not run validators on default values
    performance_metric: PerformanceMetric = PerformanceMetric.F1_SCORE
    threshold_metric: ThresholdMetric = ThresholdMetric.MSE

    @field_validator("cluster_models", mode="before")
    @classmethod
    def convert_cluster_models(cls, v):
        """Accept names or ``ClusterModel`` members; store registered names."""
        if isinstance(v, (str, ClusterModel)):
            v = [v]
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            raise ValueError("cluster_models must be a non-empty list of model names")
        return [normalize_cluster_model_name(model) for model in v]

    @field_validator("n_clusters_list")
    @classmethod
    def validate_n_clusters(cls, v):
        """Validate all n_clusters values are positive."""
        if len(v) == 0:
            raise ValueError("n_clusters_list must contain at least one value")
        if any(n < 1 for n in v):
            raise ValueError("All n_clusters values must be >= 1")
        return v

    @field_validator("threshold_ratios")
    @classmethod
    def validate_ratios(cls, v):
        """Validate all threshold ratios are positive."""
        if len(v) == 0:
            raise ValueError("threshold_ratios must contain at least one value")
        if any(r <= 0 for r in v):
            raise ValueError("All threshold_ratios must be positive")
        return v

    @field_validator("performance_metric", mode="before")
    @classmethod
    def validate_metric(cls, v):
        """Validate performance metric name and convert it to the enum."""
        return parse_performance_metric(v)

    @field_validator("threshold_metric", mode="before")
    @classmethod
    def validate_threshold_metric(cls, v):
        """Validate threshold metric name and convert it to the enum."""
        return parse_threshold_metric(v)


class SegmentAE_Optimizer:
    """
    Grid search optimizer for SegmentAE configurations.

    Systematically evaluates combinations of autoencoders, clustering algorithms,
    cluster numbers, and threshold ratios to identify optimal configuration for
    anomaly detection performance.

    Note:
        The configuration is selected on the labeled data passed to ``optimize``
        (``X_test``/``y_test``), so the reported best score is a *validation*
        score. Measure final performance on data not used for the search.
    """

    def __init__(
        self,
        autoencoder_models: list,
        n_clusters_list: Optional[List[int]] = None,
        cluster_models: Optional[List[Union[str, ClusterModel]]] = None,
        threshold_ratios: Optional[List[float]] = None,
        performance_metric: Union[str, PerformanceMetric] = "f1_score",
        threshold_metric: Union[str, ThresholdMetric] = "mse",
        random_state: int = 0,
        verbose: bool = True,
    ):
        """
        Initialize grid search optimizer.

        Args:
            autoencoder_models: Trained autoencoders to compare (each must expose ``predict``).
            n_clusters_list: Numbers of clusters to try. Default: [1, 2, 3, 4].
            cluster_models: Clustering algorithms to try, as names or
                ``ClusterModel`` members. Default: KMeans, MiniBatchKMeans, GMM.
            threshold_ratios: Threshold ratios to try. Default: [0.75, 1, 1.5, 2, 3, 4].
            performance_metric: Metric to maximise: "accuracy", "precision",
                "recall" or "f1_score" (or a ``PerformanceMetric``).
            threshold_metric: Reconstruction metric used for thresholding:
                "mse", "mae", "rmse" or "max_error".
            random_state: Random seed for the clustering models.
            verbose: Print progress and the optimization summary.
        """
        if n_clusters_list is None:
            n_clusters_list = [1, 2, 3, 4]
        if cluster_models is None:
            cluster_models = ["KMeans", "MiniBatchKMeans", "GMM"]
        if threshold_ratios is None:
            threshold_ratios = [0.75, 1, 1.5, 2, 3, 4]

        self.config = OptimizerConfig(
            n_clusters_list=n_clusters_list,
            cluster_models=cluster_models,
            threshold_ratios=threshold_ratios,
            performance_metric=performance_metric,
            threshold_metric=threshold_metric,
        )

        # Store autoencoder models (a single model is accepted as a one-item list)
        if autoencoder_models is None:
            autoencoder_models = []
        elif not isinstance(autoencoder_models, (list, tuple)):
            autoencoder_models = [autoencoder_models]
        self.autoencoder_models = list(autoencoder_models)
        self.performance_metric = performance_metric
        self.random_state = random_state
        self.verbose = verbose

        self._validate_autoencoders()

        self._reset_results()

    def _reset_results(self) -> None:
        """Clear the outcome of any previous search."""
        self.optimal_segmentae: Optional[SegmentAE] = None
        self.best_threshold_ratio: Optional[float] = None
        self.best_n_clusters: Optional[int] = None
        self.best_autoencoder_index: Optional[int] = None
        self.best_performance: float = float("-inf")
        self.leaderboard: Optional[pd.DataFrame] = None

    def optimize(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> SegmentAE:
        """
        Execute grid search optimization.

        Evaluates all combinations of autoencoders, clustering algorithms,
        cluster numbers, and threshold ratios to find optimal configuration.

        Args:
            X_train: Training (reference) features, already preprocessed.
            X_test: Labeled features used to score each configuration.
            y_test: Binary labels for ``X_test`` (0 = normal, 1 = anomaly).

        Returns:
            The best ``SegmentAE`` pipeline; its ``results`` correspond to
            ``best_threshold_ratio``.
        """
        self._validate_inputs(X_train, X_test, y_test)
        self._reset_results()  # each call is an independent search

        # One frame, in the training column order, is reused by every configuration
        # (so its reconstruction is computed and cached once per autoencoder).
        X_test = validate_columns_match(X_test, list(X_train.columns), "X_test")

        total_configs = (
            len(self.config.n_clusters_list)
            * len(self.config.cluster_models)
            * len(self.autoencoder_models)
        )

        self._log(f"\n{'='*60}")
        self._log("Starting Grid Search Optimization")
        self._log(f"Total Configurations: {total_configs}")
        self._log(f"Performance Metric: {self.performance_metric}")
        self._log(f"{'='*60}\n")

        # An autoencoder's reconstruction of a dataset does not depend on the
        # clustering, so it is computed once per (autoencoder, dataset) and
        # shared by every configuration instead of once per configuration.
        reconstruction_cache: Dict = {}
        results = []
        iteration = 1

        try:
            # Grid search over all combinations. Clustering does not depend on the
            # autoencoder either, so each clustering model is fitted only once.
            for n_clusters in self.config.n_clusters_list:
                for cluster_model in self.config.cluster_models:
                    clustering = Clustering(
                        cluster_model=[cluster_model],
                        n_clusters=n_clusters,
                        random_state=self.random_state,
                    )
                    clustering.clustering_fit(X=X_train)

                    for ae_index, autoencoder in enumerate(self.autoencoder_models):
                        self._log(f"Iteration {iteration}/{total_configs}")
                        self._log(f"Cluster Model: {cluster_model}")
                        self._log(f"Number of Clusters: {n_clusters}")
                        self._log(f"Autoencoder: {type(autoencoder).__name__}")
                        self._log("")

                        config_results = self._evaluate_configuration(
                            autoencoder=autoencoder,
                            ae_index=ae_index,
                            clustering=clustering,
                            cluster_model=cluster_model,
                            n_clusters=n_clusters,
                            X_train=X_train,
                            X_test=X_test,
                            y_test=y_test,
                            reconstruction_cache=reconstruction_cache,
                        )

                        results.extend(config_results)
                        iteration += 1
        except BaseException:
            self._reset_results()  # never leave a half-finished search looking like a result
            raise
        finally:
            # Release the shared reconstructions; the best pipeline keeps its own results
            reconstruction_cache.clear()
            if self.optimal_segmentae is not None:
                self.optimal_segmentae._reconstruction_cache = None

        self.leaderboard = self._create_leaderboard(results)

        # Leave the returned pipeline in the state of the winning threshold ratio
        # (it was last evaluated with the final ratio of the grid, not the best one).
        self._restore_best_results(X_test, y_test)
        self._print_optimization_summary()

        return self.optimal_segmentae

    @property
    def n_clusters_list(self) -> List[int]:
        """Numbers of clusters explored by the search."""
        return list(self.config.n_clusters_list)

    @property
    def cluster_models(self) -> List[str]:
        """Names of the clustering algorithms explored by the search."""
        return list(self.config.cluster_models)

    @property
    def threshold_ratios(self) -> List[float]:
        """Threshold ratios explored by the search."""
        return list(self.config.threshold_ratios)

    @property
    def threshold_metric(self) -> str:
        """Reconstruction metric used for thresholding."""
        return self.config.threshold_metric.value

    @property
    def best_model(self) -> Optional[str]:
        """Name of the autoencoder in the best configuration."""
        if self.optimal_segmentae is None:
            return None
        return type(self.optimal_segmentae.ae_model).__name__

    @property
    def best_autoencoder(self) -> Optional[Any]:
        """The autoencoder object of the best configuration."""
        if self.optimal_segmentae is None:
            return None
        return self.optimal_segmentae.ae_model

    @property
    def best_cluster_model(self) -> Optional[str]:
        """Clustering algorithm used in the best configuration."""
        if self.optimal_segmentae is None:
            return None
        return self.optimal_segmentae.cl_model.cluster_model[0]

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Best configuration as a dictionary (None before ``optimize``)."""
        if self.optimal_segmentae is None:
            return None
        return {
            "autoencoder": self.best_model,
            "autoencoder_index": self.best_autoencoder_index,
            "cluster_model": self.best_cluster_model,
            "n_clusters": self.best_n_clusters,
            "threshold_ratio": self.best_threshold_ratio,
            "threshold_metric": self.threshold_metric,
        }

    def _evaluate_configuration(
        self,
        autoencoder: Any,
        ae_index: int,
        clustering: Clustering,
        cluster_model: str,
        n_clusters: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        reconstruction_cache: Dict,
    ) -> List[pd.DataFrame]:
        """Evaluate a single configuration across all threshold ratios."""
        sg = SegmentAE(ae_model=autoencoder, cl_model=clustering, verbose=self.verbose)
        sg._reconstruction_cache = reconstruction_cache
        sg.reconstruction(input_data=X_train, threshold_metric=self.config.threshold_metric)

        config_results = []
        for threshold_ratio in self.config.threshold_ratios:
            result = self._evaluate_single_threshold(
                sg=sg,
                autoencoder=autoencoder,
                ae_index=ae_index,
                cluster_model=cluster_model,
                n_clusters=n_clusters,
                threshold_ratio=threshold_ratio,
                X_test=X_test,
                y_test=y_test,
            )
            config_results.append(result)

        return config_results

    def _evaluate_single_threshold(
        self,
        sg: SegmentAE,
        autoencoder: Any,
        ae_index: int,
        cluster_model: str,
        n_clusters: int,
        threshold_ratio: float,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Evaluate a single threshold ratio."""
        evaluation = sg.evaluation(
            input_data=X_test, target_col=y_test, threshold_ratio=threshold_ratio
        )

        global_metrics = evaluation["global metrics"].copy()

        performance = float(global_metrics[self.config.performance_metric.value].iloc[0])

        # Update best model if necessary (the first of equal scores is kept)
        if performance > self.best_performance:
            self.best_performance = performance
            self.optimal_segmentae = sg
            self.best_threshold_ratio = threshold_ratio
            self.best_n_clusters = n_clusters
            self.best_autoencoder_index = ae_index

        global_metrics["Autoencoder"] = type(autoencoder).__name__
        global_metrics["Cluster"] = cluster_model
        global_metrics["N_Clusters"] = n_clusters
        global_metrics["Autoencoder_Index"] = ae_index

        return global_metrics

    def _create_leaderboard(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        """Create sorted leaderboard from results (ties keep the grid order)."""
        leaderboard = pd.concat(results, axis=0, ignore_index=True)
        sort_column = self.config.performance_metric.value

        return leaderboard.sort_values(by=sort_column, ascending=False, kind="stable").reset_index(
            drop=True
        )

    def _restore_best_results(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Re-evaluate the best pipeline at the best ratio so its results match."""
        if self.optimal_segmentae is None:
            return

        verbose = self.optimal_segmentae.verbose
        self.optimal_segmentae.verbose = False  # thresholds were already reported
        try:
            self.optimal_segmentae.evaluation(
                input_data=X_test, target_col=y_test, threshold_ratio=self.best_threshold_ratio
            )
        finally:
            self.optimal_segmentae.verbose = verbose

    def _print_optimization_summary(self) -> None:
        """Print optimization summary."""
        self._log(f"\n{'='*60}")
        self._log("OPTIMIZATION COMPLETE")
        self._log(f"{'='*60}")
        self._log(
            f"Best Performance ({self.performance_metric}): {round(self.best_performance, 6)}"
        )

        if len(self.autoencoder_models) > 1:
            self._log(f"Best Autoencoder: {self.best_model}")

        if len(self.config.cluster_models) > 1:
            self._log(f"Best Cluster Model: {self.best_cluster_model}")

        if len(self.config.n_clusters_list) > 1:
            self._log(f"Best Number of Clusters: {self.best_n_clusters}")

        if len(self.config.threshold_ratios) > 1:
            self._log(f"Best Threshold Ratio: {self.best_threshold_ratio}")

        self._log(f"{'='*60}\n")

    def _log(self, message: str) -> None:
        """Print a progress message when verbose."""
        if self.verbose:
            print(message)

    def _validate_autoencoders(self) -> None:
        """Validate autoencoder models."""
        if not self.autoencoder_models:
            raise ConfigurationError(
                "autoencoder_models list cannot be empty",
                valid_options=["Provide at least one trained autoencoder"],
            )

        for ae in self.autoencoder_models:
            if not isinstance(ae, AutoencoderProtocol) or not callable(ae.predict):
                raise ConfigurationError(
                    f"Autoencoder {type(ae).__name__} must have a 'predict' method. "
                    f"Ensure all autoencoders are properly trained."
                )

    def _validate_inputs(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        """Validate optimization inputs."""
        validate_dataframe(X_train, "X_train", allow_missing=False, require_numeric=True)
        validate_dataframe(X_test, "X_test", allow_missing=False, require_numeric=True)
        validate_columns_match(X_test, list(X_train.columns), "X_test")

        if not isinstance(y_test, pd.Series):
            raise ValidationError(f"y_test must be a pandas Series, got {type(y_test).__name__}")

        if len(X_test) != len(y_test):
            raise ValidationError(
                f"X_test length ({len(X_test)}) must match " f"y_test length ({len(y_test)})"
            )

        validate_binary_target(y_test, "y_test")

    def __repr__(self) -> str:
        """String representation of optimizer."""
        return (
            f"SegmentAE_Optimizer("
            f"n_autoencoders={len(self.autoencoder_models)}, "
            f"n_clusters={self.config.n_clusters_list}, "
            f"clusters={self.config.cluster_models}, "
            f"metric={self.performance_metric})"
        )
