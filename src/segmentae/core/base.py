from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from segmentae.core.exceptions import ValidationError, ModelNotFittedError


class AbstractClusterModel(ABC):
    """
    Abstract base class for all clustering implementations.
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
    
    def _validate_input(self, X: pd.DataFrame, context: str = "Input") -> None:
        """
        Validate input DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(
                f"{context} must be a pandas DataFrame, got {type(X).__name__}",
                suggestion="Convert your data to a pandas DataFrame using pd.DataFrame()"
            )
        
        if X.empty:
            raise ValidationError(
                f"{context} DataFrame is empty",
                suggestion="Ensure your dataset contains data before fitting"
            )
        
        if X.isnull().any().any():
            raise ValidationError(
                f"{context} contains missing values",
                suggestion="Handle missing values using preprocessing before clustering"
            )
    
    def _validate_fitted(self) -> None:
        """
        Check if model is fitted, raise error if not.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                component=self.__class__.__name__,
                message=f"{self.__class__.__name__} must be fitted before prediction. "
                        f"Call fit(X) method first."
            )


class AbstractPreprocessor(ABC):
    """
    Abstract base class for preprocessing components.
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> 'AbstractPreprocessor':
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