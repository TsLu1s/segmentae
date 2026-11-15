from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from segmentae.clusters.models import ClusteringConfig
from segmentae.clusters.registry import ClusterRegistry
from segmentae.core.base import AbstractClusterModel
from segmentae.core.constants import ClusterModel
from segmentae.core.exceptions import ModelNotFittedError
from segmentae.utils.validation import validate_columns_match, validate_dataframe


class Clustering:
    """
    Main clustering orchestrator for SegmentAE.

    This class manages multiple clustering algorithms, handling fitting
    and prediction across different clustering approaches.

    Attributes:
        cluster_model: Names of the configured clustering algorithms (plain strings)
        n_clusters: Number of clusters to form
        random_state: Random seed for reproducibility
        covariance_type: Covariance type for GMM clustering
    """

    def __init__(
        self,
        cluster_model: Optional[Union[str, ClusterModel, List[Union[str, ClusterModel]]]] = None,
        n_clusters: int = 3,
        random_state: int = 0,
        covariance_type: str = "full",
    ):
        """
        Initialize clustering pipeline.

        Args:
            cluster_model: One algorithm or a list of algorithms, given as
                ``ClusterModel`` members or names ("KMeans", "MiniBatchKMeans",
                "GMM", or any name registered in ``ClusterRegistry``).
                Defaults to ``["KMeans"]``.
            n_clusters: Number of clusters to form
            random_state: Random seed for reproducibility
            covariance_type: Covariance type for GMM ('full', 'tied', 'diag', 'spherical')
        """
        if cluster_model is None:
            cluster_model = ["KMeans"]

        # Validate and store configuration
        self.config = ClusteringConfig(
            cluster_models=cluster_model,
            n_clusters=n_clusters,
            random_state=random_state,
            covariance_type=covariance_type,
        )

        # Public attributes mirror the validated configuration. ``cluster_model``
        # always holds plain string names, whatever form the caller used.
        self.cluster_model: List[str] = list(self.config.cluster_models)
        self.n_clusters = self.config.n_clusters
        self.random_state = self.config.random_state
        self.covariance_type = self.config.covariance_type

        # Internal state
        self._fitted_models: Dict[str, AbstractClusterModel] = {}
        self._feature_names: Optional[List] = None
        self._is_fitted: bool = False

    def clustering_fit(self, X: pd.DataFrame) -> "Clustering":
        """
        Fit all specified clustering models to data.

        This method creates and fits each specified clustering algorithm
        to the provided data, storing the fitted models for later prediction.

        Returns:
            The fitted ``Clustering`` instance (for chaining).
        """
        self._validate_input(X, "Training data")

        # Fit into a fresh dict so a failed refit never leaves a half-updated state
        fitted_models: Dict[str, AbstractClusterModel] = {}
        for model_name in self.config.cluster_models:
            model_instance = self._create_model(model_name)
            model_instance.fit(X)
            fitted_models[model_name] = model_instance

        self._fitted_models = fitted_models
        self._feature_names = list(X.columns)
        self._is_fitted = True
        return self

    def cluster_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict cluster assignments for all fitted models.

        Returns:
            DataFrame with one column of cluster labels per fitted model,
            indexed like ``X``.
        """
        self._validate_fitted()
        self._validate_input(X, "Prediction data")
        X = validate_columns_match(X, self._feature_names, "Prediction data")

        # np.asarray: labels are positional even if a custom model returns a Series
        return pd.DataFrame(
            {
                model_name: np.asarray(model.predict(X))
                for model_name, model in self._fitted_models.items()
            },
            index=X.index,
        )

    # Private methods
    def _create_model(self, model_name: str) -> AbstractClusterModel:
        """
        Create a clustering model instance with appropriate parameters.

        Each registered class maps the generic parameters to its own
        constructor through ``from_params`` (e.g. GMM uses ``n_components``).
        """
        return ClusterRegistry.get(model_name).from_params(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            covariance_type=self.config.covariance_type,
        )

    def _validate_input(self, X: pd.DataFrame, context: str = "Input") -> None:
        """
        Validate input DataFrame (non-empty, numeric, no missing or infinite values).
        """
        validate_dataframe(X, context, allow_missing=False, require_numeric=True)

    def _validate_fitted(self) -> None:
        """
        Check if clustering is fitted.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                component="Clustering",
                message="Clustering must be fitted before prediction. "
                "Call clustering_fit(X) method first.",
            )

    # Properties for accessing fitted models
    @property
    def fitted_models(self) -> Dict[str, AbstractClusterModel]:
        """Get dictionary of fitted clustering models."""
        return self._fitted_models.copy()

    @property
    def is_fitted(self) -> bool:
        """Check if clustering pipeline is fitted."""
        return self._is_fitted

    @property
    def feature_names(self) -> Optional[List]:
        """Feature columns seen during ``clustering_fit`` (None before fitting)."""
        return None if self._feature_names is None else list(self._feature_names)

    @property
    def clustering_dict(self) -> Dict[str, AbstractClusterModel]:
        """Get dictionary of fitted models (backward compatibility)."""
        return self._fitted_models.copy()

    @property
    def cmodel(self):
        """Get the last fitted model (backward compatibility)."""
        if not self._fitted_models:
            return None
        return list(self._fitted_models.values())[-1]

    def __repr__(self) -> str:
        """String representation of Clustering."""
        models_str = ", ".join(self.config.cluster_models)
        return (
            f"Clustering("
            f"models=[{models_str}], "
            f"n_clusters={self.config.n_clusters}, "
            f"fitted={self._is_fitted})"
        )
