from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from segmentae.core.exceptions import ModelNotFittedError
from segmentae.utils.validation import validate_dataframe


class AbstractClusterModel(ABC):
    """
    Abstract base class for all clustering implementations.

    Subclass it and register the class with ``ClusterRegistry.register`` to make
    a custom algorithm available to ``Clustering`` and ``SegmentAE_Optimizer``.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit clustering model to data.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster assignments for data.
        """
        pass

    @abstractmethod
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit model and predict cluster assignments in one step.
        """
        pass

    @property
    @abstractmethod
    def n_clusters(self) -> int:
        """
        Return the number of clusters.
        """
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """
        Check if model has been fitted.
        """
        pass

    @classmethod
    def from_params(
        cls, n_clusters: int = 3, random_state: int = 0, **options: Any
    ) -> "AbstractClusterModel":
        """
        Build a model from the generic parameters of the ``Clustering`` orchestrator.

        The default implementation forwards ``n_clusters`` and ``random_state``.
        Override it when a model names its parameters differently or consumes
        extra ``options`` (e.g. ``covariance_type`` for Gaussian Mixture).
        """
        return cls(n_clusters=n_clusters, random_state=random_state)

    def _validate_input(self, X: pd.DataFrame, context: str = "Input") -> None:
        """
        Validate input DataFrame (non-empty, numeric, no missing or infinite values).
        """
        validate_dataframe(X, context, allow_missing=False, require_numeric=True)

    def _validate_fitted(self) -> None:
        """
        Check if model is fitted, raise error if not.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                component=self.__class__.__name__,
                message=f"{self.__class__.__name__} must be fitted before prediction. "
                f"Call fit(X) method first.",
            )


class AbstractPreprocessor(ABC):
    """
    Abstract base class for preprocessing components.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "AbstractPreprocessor":
        """
        Fit preprocessor to data.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        """
        pass

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        """
        return self.fit(X).transform(X)
