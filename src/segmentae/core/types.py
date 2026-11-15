from typing import Any, Dict, Protocol, TypeAlias, Union, runtime_checkable

import numpy as np
import pandas as pd

# Type aliases for commonly used types
DataFrame: TypeAlias = pd.DataFrame
Series: TypeAlias = pd.Series
NDArray: TypeAlias = np.ndarray
DictStrAny: TypeAlias = Dict[str, Any]


@runtime_checkable
class AutoencoderProtocol(Protocol):
    """
    Protocol defining the interface for autoencoder models.

    Any object exposing ``predict(X)`` that returns one reconstructed row per
    input row (same number of features) can be used: the built-in autoencoders,
    a compiled Keras model, or a custom wrapper around another framework.
    The protocol is runtime-checkable, so ``isinstance(model, AutoencoderProtocol)``
    verifies the interface.
    """

    def predict(self, X: Union[DataFrame, NDArray]) -> NDArray:
        """
        Generate reconstructions from input data.
        """
        ...


@runtime_checkable
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


@runtime_checkable
class PreprocessorProtocol(Protocol):
    """
    Protocol defining the interface for preprocessing components.
    """

    def fit(self, X: DataFrame) -> "PreprocessorProtocol":
        """Fit preprocessor to data."""
        ...

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform data using fitted preprocessor."""
        ...
