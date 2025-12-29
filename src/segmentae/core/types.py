from typing import Protocol, Union, Any, Dict, List
import pandas as pd
import numpy as np


# Type aliases for commonly used types
DataFrame = pd.DataFrame
Series = pd.Series
NDArray = np.ndarray
DictStrAny = Dict[str, Any]


class AutoencoderProtocol(Protocol):
    """
    Protocol defining the interface for autoencoder models.
    """
    
    def predict(self, X: Union[DataFrame, NDArray]) -> NDArray:
        """
        Generate reconstructions from input data.
        """
        ...


class ClusterModelProtocol(Protocol):
    """
    Protocol defining the interface for clustering models.
    """
    
    def fit(self, X: DataFrame) -> None:
        """Fit clustering model to data."""
        ...
    
    def predict(self, X: DataFrame) -> NDArray:
        """Predict cluster assignments."""
        ...
    
    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        ...


class PreprocessorProtocol(Protocol):
    """
    Protocol defining the interface for preprocessing components.
    """
    
    def fit(self, X: DataFrame) -> 'PreprocessorProtocol':
        """Fit preprocessor to data."""
        ...
    
    def transform(self, X: DataFrame) -> DataFrame:
        """Transform data using fitted preprocessor."""
        ...