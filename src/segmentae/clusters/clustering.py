import pandas as pd

from typing import List, Dict

from segmentae.core.constants import ClusterModel
from segmentae.core.base import AbstractClusterModel
from segmentae.core.exceptions import ValidationError, ModelNotFittedError
from segmentae.clusters.registry import ClusterRegistry
from segmentae.clusters.models import ClusteringConfig

class Clustering:
    """
    Main clustering orchestrator for SegmentAE.
    
    This class manages multiple clustering algorithms, handling fitting
    and prediction across different clustering approaches.
    
    Attributes:
        cluster_model: List of clustering algorithm names
        n_clusters: Number of clusters to form
        random_state: Random seed for reproducibility
        covariance_type: Covariance type for GMM clustering
    """
    
    def __init__(self,
                 cluster_model: List[str] = ['KMeans'],
                 n_clusters: int = 3,
                 random_state: int = 0,
                 covariance_type: str = "full"):
        """
        Initialize clustering pipeline.
        """
        # Validate and store configuration
        self.config = ClusteringConfig(
            cluster_models=cluster_model,
            n_clusters=n_clusters,
            random_state=random_state,
            covariance_type=covariance_type
        )
        
        # Store for backward compatibility
        self.cluster_model = cluster_model
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.covariance_type = covariance_type
        
        # Internal state
        self._fitted_models: Dict[str, AbstractClusterModel] = {}
        self._is_fitted: bool = False
    
    def clustering_fit(self, X: pd.DataFrame) -> 'Clustering':
        """
        Fit all specified clustering models to data.
        
        This method creates and fits each specified clustering algorithm
        to the provided data, storing the fitted models for later prediction.
        """
        self._validate_input(X, "Training data")
        
        # Fit each specified clustering model
        for model_type in self.config.cluster_models:
            model_instance = self._create_model(model_type)
            model_instance.fit(X)
            self._fitted_models[model_type.value] = model_instance
        
        self._is_fitted = True
        return self
    
    def cluster_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict cluster assignments for all fitted models.
        """
        self._validate_fitted()
        self._validate_input(X, "Prediction data")
        
        results = pd.DataFrame()
        
        for model_name, model in self._fitted_models.items():
            predictions = model.predict(X)
            results[model_name] = predictions
        
        return results
    
    # Private methods
    def _create_model(self, model_type: ClusterModel) -> AbstractClusterModel:
        """
        Create a clustering model instance with appropriate parameters.
        """
        # Base parameters for all models
        kwargs = {
            'n_clusters': self.config.n_clusters,
            'random_state': self.config.random_state
        }
        
        # Special handling for GMM (uses n_components instead of n_clusters)
        if model_type == ClusterModel.GMM:
            kwargs = {
                'n_components': self.config.n_clusters,
                'covariance_type': self.config.covariance_type
            }
        
        # Remove n_clusters for Agglomerative if using distance_threshold
        if model_type == ClusterModel.AGGLOMERATIVE:
            kwargs = {'n_clusters': self.config.n_clusters}
        
        # MiniBatchKMeans uses different default max_iter
        if model_type == ClusterModel.MINIBATCH_KMEANS:
            kwargs['max_iter'] = 150
        
        return ClusterRegistry.create(model_type, **kwargs)
    
    def _validate_input(self, X: pd.DataFrame, context: str = "Input") -> None:
        """
        Validate input DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(
                f"{context} must be a pandas DataFrame, got {type(X).__name__}",
                suggestion="Convert your data to DataFrame using pd.DataFrame()"
            )
        
        if X.empty:
            raise ValidationError(
                f"{context} DataFrame is empty",
                suggestion="Ensure your dataset contains data"
            )
    
    def _validate_fitted(self) -> None:
        """
        Check if clustering is fitted.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                component="Clustering",
                message="Clustering must be fitted before prediction. "
                        "Call clustering_fit(X) method first."
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
        models_str = ", ".join([m.value for m in self.config.cluster_models])
        return (
            f"Clustering("
            f"models=[{models_str}], "
            f"n_clusters={self.config.n_clusters}, "
            f"fitted={self._is_fitted})"
        )