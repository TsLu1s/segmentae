import numpy as np
import pandas as pd

from typing import Optional, List, Union

from pydantic import BaseModel, field_validator
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering
)
from sklearn.mixture import GaussianMixture

from segmentae.core.base import AbstractClusterModel
from segmentae.core.constants import ClusterModel
from segmentae.clusters.registry import ClusterRegistry


class KMeansCluster(AbstractClusterModel):
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
        
        @field_validator('n_clusters')
        def validate_n_clusters(cls, v):
            if v < 1:
                raise ValueError("n_clusters must be >= 1")
            return v
        
        @field_validator('max_iter')
        def validate_max_iter(cls, v):
            if v < 1:
                raise ValueError("max_iter must be >= 1")
            return v
        
        class Config:
            use_enum_values = True
    
    def __init__(self,
                 n_clusters: int = 3,
                 random_state: int = 0,
                 max_iter: int = 300):
        """
        Initialize K-Means clustering model.
        
        Args:
            n_clusters: Number of clusters to form
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for convergence
        """
        self.config = self.Config(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter
        )
        self._model: Optional[KMeans] = None
        self._is_fitted: bool = False
    
    def fit(self, X: pd.DataFrame) -> None:
        """Fit K-Means model to data."""
        self._validate_input(X, "Training data")
        
        self._model = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            max_iter=self.config.max_iter
        )
        self._model.fit(X)
        self._is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for data."""
        self._validate_fitted()
        self._validate_input(X, "Prediction data")
        return self._model.predict(X)
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit model and predict labels in one step."""
        self.fit(X)
        return self.predict(X)
    
    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self.config.n_clusters
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted


class MiniBatchKMeansCluster(AbstractClusterModel):
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
        
        @field_validator('n_clusters')
        def validate_n_clusters(cls, v):
            if v < 1:
                raise ValueError("n_clusters must be >= 1")
            return v
        
        @field_validator('max_iter')
        def validate_max_iter(cls, v):
            if v < 1:
                raise ValueError("max_iter must be >= 1")
            return v
        
        class Config:
            use_enum_values = True
    
    def __init__(self,
                 n_clusters: int = 3,
                 random_state: int = 0,
                 max_iter: int = 150):
        """
        Initialize MiniBatch K-Means clustering model.
        
        Args:
            n_clusters: Number of clusters to form
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for convergence
        """
        self.config = self.Config(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter
        )
        self._model: Optional[MiniBatchKMeans] = None
        self._is_fitted: bool = False
    
    def fit(self, X: pd.DataFrame) -> None:
        """Fit MiniBatch K-Means model to data."""
        self._validate_input(X, "Training data")
        
        self._model = MiniBatchKMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            max_iter=self.config.max_iter
        )
        self._model.fit(X)
        self._is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for data."""
        self._validate_fitted()
        self._validate_input(X, "Prediction data")
        return self._model.predict(X)
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit model and predict labels in one step."""
        self.fit(X)
        return self.predict(X)
    
    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self.config.n_clusters
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted


class GaussianMixtureCluster(AbstractClusterModel):
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
        init_params: str = "k-means++"
        
        @field_validator('n_components')
        def validate_n_components(cls, v):
            if v < 1:
                raise ValueError("n_components must be >= 1")
            return v
        
        @field_validator('covariance_type')
        def validate_covariance_type(cls, v):
            valid_types = ['full', 'tied', 'diag', 'spherical']
            if v not in valid_types:
                raise ValueError(
                    f"covariance_type must be one of {valid_types}, got '{v}'"
                )
            return v
        
        @field_validator('max_iter')
        def validate_max_iter(cls, v):
            if v < 1:
                raise ValueError("max_iter must be >= 1")
            return v
        
        class Config:
            use_enum_values = True
    
    def __init__(self,
                 n_components: int = 3,
                 covariance_type: str = "full",
                 max_iter: int = 150,
                 init_params: str = "k-means++"):
        """
        Initialize Gaussian Mixture Model.
        
        Args:
            n_components: Number of mixture components (clusters)
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            max_iter: Maximum EM iterations
            init_params: Initialization method for parameters
        """
        self.config = self.Config(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            init_params=init_params
        )
        self._model: Optional[GaussianMixture] = None
        self._is_fitted: bool = False
    
    def fit(self, X: pd.DataFrame) -> None:
        """Fit Gaussian Mixture Model to data."""
        self._validate_input(X, "Training data")
        
        self._model = GaussianMixture(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            max_iter=self.config.max_iter
        )
        self._model.fit(X)
        self._is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for data."""
        self._validate_fitted()
        self._validate_input(X, "Prediction data")
        return self._model.predict(X)
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit model and predict labels in one step."""
        self.fit(X)
        return self.predict(X)
    
    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self.config.n_components
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted


class AgglomerativeCluster(AbstractClusterModel):
    """
    Agglomerative (Hierarchical) clustering implementation.
    
    Builds nested clusters by successively merging or splitting them.
    Works bottom-up, starting with each sample as its own cluster.
    """
    
    class Config(BaseModel):
        """Pydantic configuration for Agglomerative clustering parameters."""
        n_clusters: int = 3
        linkage: str = "ward"
        distance_threshold: Optional[float] = None
        
        @field_validator('n_clusters')
        def validate_n_clusters(cls, v):
            if v < 1:
                raise ValueError("n_clusters must be >= 1")
            return v
        
        @field_validator('linkage')
        def validate_linkage(cls, v):
            valid_linkage = ['ward', 'complete', 'average', 'single']
            if v not in valid_linkage:
                raise ValueError(
                    f"linkage must be one of {valid_linkage}, got '{v}'"
                )
            return v
        
        class Config:
            use_enum_values = True
    
    def __init__(self,
                 n_clusters: int = 3,
                 linkage: str = "ward",
                 distance_threshold: Optional[float] = None):
        """
        Initialize Agglomerative clustering model.
        
        Args:
            n_clusters: Number of clusters to find
            linkage: Linkage criterion to use
            distance_threshold: Distance threshold for merging clusters
        """
        self.config = self.Config(
            n_clusters=n_clusters,
            linkage=linkage,
            distance_threshold=distance_threshold
        )
        self._model: Optional[AgglomerativeClustering] = None
        self._labels: Optional[np.ndarray] = None
        self._is_fitted: bool = False
    
    def fit(self, X: pd.DataFrame) -> None:
        """Fit Agglomerative clustering model to data."""
        self._validate_input(X, "Training data")
        
        self._model = AgglomerativeClustering(
            n_clusters=self.config.n_clusters,
            linkage=self.config.linkage,
            distance_threshold=self.config.distance_threshold
        )
        self._labels = self._model.fit_predict(X)
        self._is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for data.
        
        Note: Agglomerative clustering doesn't support prediction on new data
        after fitting. This method re-fits the model on the new data.
        """
        self._validate_input(X, "Prediction data")
        
        model = AgglomerativeClustering(
            n_clusters=self.config.n_clusters,
            linkage=self.config.linkage,
            distance_threshold=self.config.distance_threshold
        )
        return model.fit_predict(X)
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fit model and predict labels in one step."""
        self.fit(X)
        return self._labels
    
    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self.config.n_clusters
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

class ClusteringConfig(BaseModel):
    """
    Configuration for clustering pipeline.
    
    Attributes:
        cluster_models: List of clustering algorithms to use
        n_clusters: Number of clusters to form
        random_state: Random seed for reproducibility
        covariance_type: Covariance type for GMM
    """
    
    cluster_models: List[Union[ClusterModel, str]]
    n_clusters: int = 3
    random_state: int = 0
    covariance_type: str = "full"
    
    @field_validator('cluster_models')
    def convert_to_enum_list(cls, v):
        """Convert cluster models to enum list."""
        if isinstance(v, str):
            v = [v]
        
        result = []
        for model in v:
            if isinstance(model, ClusterModel):
                result.append(model)
            elif isinstance(model, str):
                try:
                    result.append(ClusterModel(model))
                except ValueError:
                    valid_models = [m.value for m in ClusterModel]
                    raise ValueError(
                        f"Invalid cluster model: '{model}'. "
                        f"Valid options: {valid_models}"
                    )
            else:
                raise ValueError(f"Invalid cluster model type: {type(model)}")
        
        return result
    
    @field_validator('n_clusters')
    def validate_n_clusters(cls, v):
        """Validate n_clusters is positive."""
        if v < 1:
            raise ValueError("n_clusters must be >= 1")
        return v
    
    @field_validator('covariance_type')
    def validate_covariance_type(cls, v):
        """Validate covariance type for GMM."""
        valid_types = ['full', 'tied', 'diag', 'spherical']
        if v not in valid_types:
            raise ValueError(
                f"covariance_type must be one of {valid_types}, got '{v}'"
            )
        return v
    
    class Config:
        use_enum_values = False


# Auto-register all clustering models
ClusterRegistry.register(ClusterModel.KMEANS, KMeansCluster)
ClusterRegistry.register(ClusterModel.MINIBATCH_KMEANS, MiniBatchKMeansCluster)
ClusterRegistry.register(ClusterModel.GMM, GaussianMixtureCluster)
ClusterRegistry.register(ClusterModel.AGGLOMERATIVE, AgglomerativeCluster)