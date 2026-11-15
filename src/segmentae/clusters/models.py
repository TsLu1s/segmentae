from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from segmentae.clusters.registry import ClusterRegistry
from segmentae.core.base import AbstractClusterModel
from segmentae.core.constants import ClusterModel
from segmentae.core.exceptions import ClusteringError

_COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]
_GMM_INIT_PARAMS = ["kmeans", "k-means++", "random", "random_from_data"]


def _require_at_least_one(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _require_covariance_type(value: str) -> str:
    if value not in _COVARIANCE_TYPES:
        raise ValueError(f"covariance_type must be one of {_COVARIANCE_TYPES}, got '{value}'")
    return value


class _SklearnClusterModel(AbstractClusterModel):
    """
    Shared fit / predict logic for the scikit-learn backed clustering models.

    Subclasses provide a validated ``config`` and build the estimator in
    ``_build_estimator``; fitting, prediction, input validation and error
    reporting live here.
    """

    config: Any  # the subclass's validated Config model

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._is_fitted: bool = False

    def _build_estimator(self) -> Any:
        """Create the (unfitted) scikit-learn estimator from ``self.config``."""
        raise NotImplementedError

    def fit(self, X: pd.DataFrame) -> None:
        """Fit the clustering model to data."""
        self._validate_input(X, "Training data")

        if len(X) < self.n_clusters:
            raise ClusteringError(
                f"{type(self).__name__} cannot form {self.n_clusters} clusters from "
                f"{len(X)} sample(s); reduce n_clusters or provide more data"
            )

        model = self._build_estimator()
        try:
            model.fit(X)
        except Exception as exc:
            raise ClusteringError(f"{type(self).__name__} failed to fit: {exc}") from exc

        self._model = model
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for data."""
        self._validate_fitted()
        self._validate_input(X, "Prediction data")
        try:
            return self._model.predict(X)
        except Exception as exc:
            raise ClusteringError(f"{type(self).__name__} failed to predict: {exc}") from exc

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit model and predict labels in one step."""
        self.fit(X)
        return self.predict(X)

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted


class KMeansCluster(_SklearnClusterModel):
    """
    K-Means clustering implementation.

    K-Means partitions data into n_clusters by minimizing within-cluster
    variance. It's efficient and works well for spherical clusters.
    """

    class Config(BaseModel):
        """Pydantic configuration for K-Means parameters."""

        n_clusters: int = 3
        random_state: int = 0
        max_iter: int = 300

        @field_validator("n_clusters")
        @classmethod
        def validate_n_clusters(cls, v):
            return _require_at_least_one(v, "n_clusters")

        @field_validator("max_iter")
        @classmethod
        def validate_max_iter(cls, v):
            return _require_at_least_one(v, "max_iter")

    def __init__(self, n_clusters: int = 3, random_state: int = 0, max_iter: int = 300):
        """
        Initialize K-Means clustering model.

        Args:
            n_clusters: Number of clusters to form
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for convergence
        """
        super().__init__()
        self.config = self.Config(
            n_clusters=n_clusters, random_state=random_state, max_iter=max_iter
        )

    def _build_estimator(self) -> KMeans:
        return KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
        )

    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self.config.n_clusters


class MiniBatchKMeansCluster(_SklearnClusterModel):
    """
    MiniBatch K-Means clustering implementation.

    A variant of K-Means that uses mini-batches to reduce computation time
    while approximating the standard K-Means algorithm. Ideal for large datasets.
    """

    class Config(BaseModel):
        """Pydantic configuration for MiniBatch K-Means parameters."""

        n_clusters: int = 3
        random_state: int = 0
        max_iter: int = 150

        @field_validator("n_clusters")
        @classmethod
        def validate_n_clusters(cls, v):
            return _require_at_least_one(v, "n_clusters")

        @field_validator("max_iter")
        @classmethod
        def validate_max_iter(cls, v):
            return _require_at_least_one(v, "max_iter")

    def __init__(self, n_clusters: int = 3, random_state: int = 0, max_iter: int = 150):
        """
        Initialize MiniBatch K-Means clustering model.

        Args:
            n_clusters: Number of clusters to form
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for convergence
        """
        super().__init__()
        self.config = self.Config(
            n_clusters=n_clusters, random_state=random_state, max_iter=max_iter
        )

    def _build_estimator(self) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
        )

    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self.config.n_clusters


class GaussianMixtureCluster(_SklearnClusterModel):
    """
    Gaussian Mixture Model clustering implementation.

    GMM assumes data is generated from a mixture of Gaussian distributions.
    Uses Expectation-Maximization algorithm for parameter estimation.
    Provides probabilistic cluster assignments.
    """

    class Config(BaseModel):
        """Pydantic configuration for GMM parameters."""

        n_components: int = 3
        covariance_type: str = "full"
        max_iter: int = 150
        init_params: str = "kmeans"
        random_state: int = 0

        @field_validator("n_components")
        @classmethod
        def validate_n_components(cls, v):
            return _require_at_least_one(v, "n_components")

        @field_validator("covariance_type")
        @classmethod
        def validate_covariance_type(cls, v):
            return _require_covariance_type(v)

        @field_validator("max_iter")
        @classmethod
        def validate_max_iter(cls, v):
            return _require_at_least_one(v, "max_iter")

        @field_validator("init_params")
        @classmethod
        def validate_init_params(cls, v):
            if v not in _GMM_INIT_PARAMS:
                raise ValueError(f"init_params must be one of {_GMM_INIT_PARAMS}, got '{v}'")
            return v

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        max_iter: int = 150,
        init_params: str = "kmeans",
        random_state: int = 0,
    ):
        """
        Initialize Gaussian Mixture Model.

        Args:
            n_components: Number of mixture components (clusters)
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            max_iter: Maximum EM iterations
            init_params: Initialization method for the responsibilities
                ('kmeans', 'k-means++', 'random', 'random_from_data')
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.config = self.Config(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            init_params=init_params,
            random_state=random_state,
        )

    @classmethod
    def from_params(
        cls, n_clusters: int = 3, random_state: int = 0, **options: Any
    ) -> "GaussianMixtureCluster":
        """Map the generic ``n_clusters`` to ``n_components`` and apply GMM options."""
        return cls(
            n_components=n_clusters,
            covariance_type=options.get("covariance_type", "full"),
            random_state=random_state,
        )

    def _build_estimator(self) -> GaussianMixture:
        return GaussianMixture(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            max_iter=self.config.max_iter,
            init_params=self.config.init_params,
            random_state=self.config.random_state,
        )

    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self.config.n_components


class ClusteringConfig(BaseModel):
    """
    Configuration for clustering pipeline.

    Attributes:
        cluster_models: Names of the clustering algorithms to use. Accepts
            ``ClusterModel`` members or strings (a single value or a list) and
            stores the registered names as plain strings.
        n_clusters: Number of clusters to form
        random_state: Random seed for reproducibility
        covariance_type: Covariance type for GMM
    """

    model_config = ConfigDict(use_enum_values=False)

    cluster_models: List[str]
    n_clusters: int = 3
    random_state: int = 0
    covariance_type: str = "full"

    @field_validator("cluster_models", mode="before")
    @classmethod
    def normalize_cluster_models(cls, v):
        """Accept one name or a list of names/enums; return registered names."""
        if isinstance(v, (str, ClusterModel)):
            v = [v]
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"cluster_models must be a list of model names, got {type(v)}")
        if len(v) == 0:
            raise ValueError("cluster_models must contain at least one model")

        return [normalize_cluster_model_name(model) for model in v]

    @field_validator("n_clusters")
    @classmethod
    def validate_n_clusters(cls, v):
        """Validate n_clusters is positive."""
        return _require_at_least_one(v, "n_clusters")

    @field_validator("covariance_type")
    @classmethod
    def validate_covariance_type(cls, v):
        """Validate covariance type for GMM."""
        return _require_covariance_type(v)


def normalize_cluster_model_name(model: Union[ClusterModel, str]) -> str:
    """
    Return the registered name for a cluster model given as enum or string.

    Raises:
        ValueError: If the model is not a registered clustering algorithm.
    """
    if isinstance(model, ClusterModel):
        name = model.value
    elif isinstance(model, str):
        name = model.strip()
    else:
        raise ValueError(f"Invalid cluster model type: {type(model)}")

    if not ClusterRegistry.is_registered(name):
        raise ValueError(
            f"Invalid cluster model: '{model}'. " f"Valid options: {ClusterRegistry.names()}"
        )
    return name


ClusterRegistry.register(ClusterModel.KMEANS, KMeansCluster, override=True)
ClusterRegistry.register(ClusterModel.MINIBATCH_KMEANS, MiniBatchKMeansCluster, override=True)
ClusterRegistry.register(ClusterModel.GMM, GaussianMixtureCluster, override=True)
