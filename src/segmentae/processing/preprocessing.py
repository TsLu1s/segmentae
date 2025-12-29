import pandas as pd

from typing import List, Optional, Union, Any

from pydantic import BaseModel, field_validator, ConfigDict

from segmentae.core.constants import EncoderType, ScalerType, ImputerType
from segmentae.core.exceptions import ValidationError, ModelNotFittedError
from segmentae.processing.simplifier import ComponentFactory

import warnings
warnings.filterwarnings("ignore", category=Warning)

class PreprocessingConfig(BaseModel):
    """
    Configuration for preprocessing pipeline.
    """

    encoder: Optional[Union[EncoderType, str]] = None           # Default: No encoding || Options: "IFrequencyEncoder", "LabelEncoder", "OneHotEncoder"
    scaler: Optional[Union[ScalerType, str]] = "MinMaxScaler"   # Default: MinMaxScaler || Options: "MinMaxScaler", "StandardScaler", "RobustScaler"
    imputer: Optional[Union[ImputerType, str]] = "Simple"       # Default: Simple Imputer || Options: "Simple"
    
    @field_validator('encoder', mode='before')
    def convert_encoder_to_enum(cls, v):
        """Convert string encoder to enum."""
        if v is None or isinstance(v, EncoderType):
            return v
        try:
            return EncoderType(v)
        except ValueError:
            valid_options = [e.value for e in EncoderType]
            raise ValueError(
                f"Invalid encoder type: '{v}'. "
                f"Valid options: {valid_options}"
            )
    
    @field_validator('scaler', mode='before')
    def convert_scaler_to_enum(cls, v):
        """Convert string scaler to enum."""
        if v is None or isinstance(v, ScalerType):
            return v
        try:
            return ScalerType(v)
        except ValueError:
            valid_options = [s.value for s in ScalerType]
            raise ValueError(
                f"Invalid scaler type: '{v}'. "
                f"Valid options: {valid_options}"
            )
    
    @field_validator('imputer', mode='before')
    def convert_imputer_to_enum(cls, v):
        """Convert string imputer to enum."""
        if v is None or isinstance(v, ImputerType):
            return v
        try:
            return ImputerType(v)
        except ValueError:
            valid_options = [i.value for i in ImputerType]
            raise ValueError(
                f"Invalid imputer type: '{v}'. "
                f"Valid options: {valid_options}"
            )
    
    model_config = ConfigDict(use_enum_values=False)

class Preprocessing:
    """
    Main preprocessing class for data transformation.
    
    This class orchestrates the preprocessing pipeline including categorical encoding, numerical scaling, and missing value imputation.
    It follows the scikit-learn fit/transform pattern.
    """
    
    def __init__(self,
                 encoder: Optional[Union[EncoderType, str]] = None,
                 scaler: Optional[Union[ScalerType, str]] = "MinMaxScaler",
                 imputer: Optional[Union[ImputerType, str]] = "Simple"):
        """
        Initialize preprocessing pipeline.
        """
        # Validate and store configuration
        self.config = PreprocessingConfig(
            encoder=encoder,
            scaler=scaler,
            imputer=imputer
        )
        
        # Internal component storage
        self._encoder: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._imputer: Optional[Any] = None
        
        # State tracking
        self._X: Optional[pd.DataFrame] = None
        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._is_fitted: bool = False
    
    def fit(self, X: pd.DataFrame) -> 'Preprocessing':
        """
        Fit preprocessing components to data.
        """
        self._validate_input(X, "Input for fitting")
        
        # Setup components in order
        self._setup_encoder(X)
        self._setup_scaler()
        self._setup_imputer()
        
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessing components.
        """
        self._validate_fitted()
        self._validate_input(X, "Input for transformation")
        
        return self._apply_transformations(X)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        """
        return self.fit(X).transform(X)
    
    def _setup_encoder(self, X: pd.DataFrame) -> None:
        """
        Setup encoder based on categorical columns.
        """
        self._cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if self.config.encoder is not None and self._cat_cols:
            self._encoder = ComponentFactory.create_encoder(self.config.encoder)
            self._encoder.fit(X[self._cat_cols])
            self._X = self._encoder.transform(X).copy()
        else:
            self._X = X.copy()
    
    def _setup_scaler(self) -> None:
        """
        Setup scaler based on numerical columns.
        """
        self._num_cols = self._X.select_dtypes(include=['int', 'float']).columns.tolist()
        
        if self.config.scaler is not None and self._num_cols:
            self._scaler = ComponentFactory.create_scaler(self.config.scaler)
            self._scaler.fit(self._X[self._num_cols])
    
    def _setup_imputer(self) -> None:
        """
        Setup imputer if missing values exist.
        """
        if self.config.imputer is None or self._X.isnull().sum().sum() == 0:
            return
        
        self._imputer = ComponentFactory.create_imputer(self.config.imputer)
        
        # Prepare data for imputer
        X_for_imputer = self._X.copy()
        
        # Scale numerical columns before imputation
        if self._scaler is not None and self._num_cols:
            X_for_imputer[self._num_cols] = self._scaler.transform(
                X_for_imputer[self._num_cols].copy()
            )
        
        # Fit imputer
        self._imputer.fit(X=X_for_imputer)
    
    def _apply_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all fitted transformations in correct order.
        """
        X_ = X.copy()
        
        # Apply encoder
        if self._encoder is not None:
            X_ = self._encoder.transform(X_)
        
        # Apply scaler
        if self._scaler is not None and self._num_cols:
            X_[self._num_cols] = self._scaler.transform(X_[self._num_cols].copy())
        
        # Apply imputer
        if self._imputer is not None:
            X_[self._num_cols] = self._imputer.transform(X=X_[self._num_cols].copy())
        
        return X_

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
        Check if preprocessing is fitted.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                component="Preprocessing",
                message="Preprocessing must be fitted before transform. "
                        "Call fit(X) method first."
            )

    @property
    def encoder(self) -> Optional[Any]:
        """Get fitted encoder component."""
        return self._encoder
    
    @property
    def scaler(self) -> Optional[Any]:
        """Get fitted scaler component."""
        return self._scaler
    
    @property
    def imputer(self) -> Optional[Any]:
        """Get fitted imputer component."""
        return self._imputer
    
    @property
    def cat_cols(self) -> List[str]:
        """Get list of categorical columns."""
        return self._cat_cols
    
    @property
    def num_cols(self) -> List[str]:
        """Get list of numerical columns."""
        return self._num_cols
    
    def __repr__(self) -> str:
        """String representation of Preprocessing."""
        return (
            f"Preprocessing("
            f"encoder={self.config.encoder.value if self.config.encoder else None}, "
            f"scaler={self.config.scaler.value if self.config.scaler else None}, "
            f"imputer={self.config.imputer.value if self.config.imputer else None})"
        )
    











