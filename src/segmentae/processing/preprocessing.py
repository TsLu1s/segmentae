import warnings
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype
from pydantic import BaseModel, ConfigDict, field_validator

from segmentae.core.constants import EncoderType, ImputerType, ScalerType
from segmentae.core.exceptions import ModelNotFittedError, ValidationError
from segmentae.processing.simplifier import ComponentFactory
from segmentae.utils.validation import validate_dataframe


class PreprocessingConfig(BaseModel):
    """
    Configuration for preprocessing pipeline.
    """

    model_config = ConfigDict(use_enum_values=False)

    # Defaults are enum members: Pydantic does not run validators on default values
    encoder: Optional[EncoderType] = None  # "IFrequencyEncoder", "LabelEncoder", "OneHotEncoder"
    scaler: Optional[ScalerType] = (
        ScalerType.MINMAX
    )  # "MinMaxScaler", "StandardScaler", "RobustScaler"
    imputer: Optional[ImputerType] = ImputerType.SIMPLE  # "Simple"

    @field_validator("encoder", mode="before")
    @classmethod
    def convert_encoder_to_enum(cls, v):
        """Convert string encoder to enum."""
        if v is None or isinstance(v, EncoderType):
            return v
        try:
            return EncoderType(v)
        except ValueError:
            valid_options = [e.value for e in EncoderType]
            raise ValueError(
                f"Invalid encoder type: '{v}'. " f"Valid options: {valid_options}"
            ) from None

    @field_validator("scaler", mode="before")
    @classmethod
    def convert_scaler_to_enum(cls, v):
        """Convert string scaler to enum."""
        if v is None or isinstance(v, ScalerType):
            return v
        try:
            return ScalerType(v)
        except ValueError:
            valid_options = [s.value for s in ScalerType]
            raise ValueError(
                f"Invalid scaler type: '{v}'. " f"Valid options: {valid_options}"
            ) from None

    @field_validator("imputer", mode="before")
    @classmethod
    def convert_imputer_to_enum(cls, v):
        """Convert string imputer to enum."""
        if v is None or isinstance(v, ImputerType):
            return v
        try:
            return ImputerType(v)
        except ValueError:
            valid_options = [i.value for i in ImputerType]
            raise ValueError(
                f"Invalid imputer type: '{v}'. " f"Valid options: {valid_options}"
            ) from None


class Preprocessing:
    """
    Main preprocessing class for data transformation.

    This class orchestrates the preprocessing pipeline including categorical encoding, numerical scaling, and missing value imputation.
    It follows the scikit-learn fit/transform pattern.

    Transformations are applied in the order encoder -> scaler -> imputer, always
    by position, so the index of the input frame is preserved and never has to
    be reset beforehand.
    """

    def __init__(
        self,
        encoder: Optional[Union[EncoderType, str]] = None,
        scaler: Optional[Union[ScalerType, str]] = "MinMaxScaler",
        imputer: Optional[Union[ImputerType, str]] = "Simple",
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            encoder: Categorical encoder ("IFrequencyEncoder", "LabelEncoder",
                "OneHotEncoder" or an ``EncoderType``); None leaves categorical
                columns untouched.
            scaler: Numerical scaler ("MinMaxScaler", "StandardScaler",
                "RobustScaler" or a ``ScalerType``); None disables scaling.
            imputer: Missing value imputer ("Simple" or an ``ImputerType``);
                None disables imputation.
        """
        # Validate and store configuration
        self.config = PreprocessingConfig(encoder=encoder, scaler=scaler, imputer=imputer)
        self._reset()

    def _reset(self) -> None:
        """Clear every fitted component so a refit starts from a clean state."""
        self._encoder: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._imputer: Optional[Any] = None

        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._feature_names_in: List[str] = []
        self._feature_names_out: List[str] = []
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "Preprocessing":
        """
        Fit preprocessing components to data.
        """
        self._validate_input(X, "Input for fitting")
        if X.columns.duplicated().any():
            duplicated = sorted({str(col) for col in X.columns[X.columns.duplicated()]})
            raise ValidationError(
                f"Input for fitting has duplicated column names: {duplicated}",
                suggestion="Give every feature column a unique name",
            )

        self._reset()
        self._feature_names_in = list(X.columns)

        # Fit components in the order they are applied
        X_encoded = self._fit_encoder(X)
        self._num_cols = _numerical_columns(X_encoded)
        self._require_imputable_columns(X_encoded)
        self._fit_scaler(X_encoded)
        self._fit_imputer(X_encoded)

        self._feature_names_out = list(X_encoded.columns)
        self._warn_unprocessed_columns(X_encoded)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessing components.

        The columns seen during ``fit`` must all be present; they are returned
        in the fitted order regardless of their order in ``X``. Columns that
        were not seen during ``fit`` are passed through unchanged (with a warning).
        """
        self._validate_fitted()
        self._validate_input(X, "Input for transformation")

        return self._apply_transformations(X)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        """
        return self.fit(X).transform(X)

    def _fit_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the encoder on the categorical columns and return the encoded frame.
        """
        self._cat_cols = _categorical_columns(X)

        if self.config.encoder is None or not self._cat_cols:
            return X

        self._encoder = ComponentFactory.create_encoder(self.config.encoder)
        self._encoder.fit(X[self._cat_cols])
        return self._encode(X)

    def _require_imputable_columns(self, X_encoded: pd.DataFrame) -> None:
        """
        Reject numerical columns with no observed value when an imputer is configured.

        Such a column has no mean to impute with (and no range to scale by), so
        it is reported up front rather than failing inside a component.
        """
        if self.config.imputer is None or not self._num_cols:
            return

        numeric = X_encoded[self._num_cols]
        empty = [str(col) for col in self._num_cols if numeric[col].isnull().all()]
        if empty:
            raise ValidationError(
                f"Columns with no observed values cannot be imputed: {empty}",
                suggestion="Drop columns that are entirely missing before fitting",
            )

    def _fit_scaler(self, X_encoded: pd.DataFrame) -> None:
        """
        Fit the scaler on the numerical columns of the encoded frame.
        """
        if self.config.scaler is not None and self._num_cols:
            self._scaler = ComponentFactory.create_scaler(self.config.scaler)
            self._scaler.fit(X_encoded[self._num_cols])

    def _fit_imputer(self, X_encoded: pd.DataFrame) -> None:
        """
        Fit the imputer on the (scaled) numerical columns.

        The imputer is fitted whenever it is configured, even if the training
        data has no missing values, so values that are missing only at
        inference time are still imputed. It is fitted on exactly the columns
        and scale it is later applied to.
        """
        if self.config.imputer is None or not self._num_cols:
            return

        scaled = self._scale(X_encoded[self._num_cols])
        self._imputer = ComponentFactory.create_imputer(self.config.imputer)
        self._imputer.fit(pd.DataFrame(scaled, columns=self._num_cols))

    def _apply_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all fitted transformations in correct order.
        """
        X_ = self._align_columns(X)

        # Apply encoder
        if self._encoder is not None:
            X_ = self._encode(X_)

        # Apply scaler
        if self._scaler is not None and self._num_cols:
            X_[self._num_cols] = self._scale(X_[self._num_cols])

        # Apply imputer (only when there is something to impute)
        if self._imputer is not None and X_[self._num_cols].isnull().to_numpy().any():
            X_[self._num_cols] = self._impute(X_[self._num_cols])

        # Fitted output columns first, in fitted order; columns unseen during fit last
        fitted = [col for col in self._feature_names_out if col in X_.columns]
        ordered = fitted + [col for col in X_.columns if col not in set(fitted)]
        return X_ if list(X_.columns) == ordered else X_[ordered]

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of ``X`` with the fitted columns in their fitted order.
        """
        present = set(X.columns)
        missing = [str(col) for col in self._feature_names_in if col not in present]
        if missing:
            raise ValidationError(
                f"Input for transformation is missing columns seen during fit: {missing}",
                suggestion="Pass data with the same feature columns used in fit()",
            )

        fitted = set(self._feature_names_in)
        extra = [col for col in X.columns if col not in fitted]
        if extra:
            warnings.warn(
                f"Columns not seen during fit are passed through unchanged: "
                f"{[str(col) for col in extra]}",
                UserWarning,
                stacklevel=4,
            )

        return X[self._feature_names_in + extra].copy()

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns, keeping the index of ``X``."""
        encoded = self._encoder.transform(X)
        if len(encoded) == len(X) and not encoded.index.equals(X.index):
            encoded.index = X.index

        # An encoder can hand back numbers in a non-numeric container: mapping a
        # ``category`` column keeps the category dtype when the codes are unique.
        # Store encoded columns as plain numbers so they are scaled like the rest.
        for col in self._cat_cols:
            if col in encoded.columns and not is_numeric_dtype(encoded[col].dtype):
                try:
                    encoded[col] = pd.to_numeric(encoded[col].astype(object))
                except (TypeError, ValueError):
                    pass  # genuinely non-numeric; reported by _warn_unprocessed_columns
        return encoded

    def _scale(self, numeric: pd.DataFrame) -> np.ndarray:
        """Scale the numerical columns; returns positional values (index-agnostic)."""
        if self._scaler is None:
            return numeric.to_numpy(dtype="float64", na_value=np.nan)
        return _positional_values(self._scaler.transform(numeric), self._num_cols)

    def _impute(self, numeric: pd.DataFrame) -> np.ndarray:
        """Impute the numerical columns; returns positional values (index-agnostic)."""
        # Hand the imputer its own frame: some implementations fill in place.
        return _positional_values(self._imputer.transform(numeric.copy()), self._num_cols)

    def _warn_unprocessed_columns(self, X_encoded: pd.DataFrame) -> None:
        """Warn about columns that stay non-numeric after preprocessing."""
        leftover = [
            str(col)
            for col, dtype in X_encoded.dtypes.items()
            if not is_numeric_dtype(dtype) and not is_bool_dtype(dtype)
        ]
        if leftover:
            warnings.warn(
                f"Columns left non-numeric by preprocessing: {leftover}. Clustering and "
                f"autoencoders need numeric input; set an encoder (e.g. "
                f"encoder='IFrequencyEncoder') or drop these columns.",
                UserWarning,
                stacklevel=3,
            )

    def _validate_input(self, X: pd.DataFrame, context: str = "Input") -> None:
        """
        Validate input DataFrame.
        """
        validate_dataframe(X, context)

    def _validate_fitted(self) -> None:
        """
        Check if preprocessing is fitted.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                component="Preprocessing",
                message="Preprocessing must be fitted before transform. "
                "Call fit(X) method first.",
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
        return list(self._cat_cols)

    @property
    def num_cols(self) -> List[str]:
        """Get list of numerical columns."""
        return list(self._num_cols)

    @property
    def feature_names_in(self) -> List[str]:
        """Columns seen during fit, in fitted order."""
        return list(self._feature_names_in)

    @property
    def feature_names_out(self) -> List[str]:
        """Columns produced by transform for the fitted features."""
        return list(self._feature_names_out)

    @property
    def is_fitted(self) -> bool:
        """Check if preprocessing is fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        """String representation of Preprocessing."""
        return (
            f"Preprocessing("
            f"encoder={self.config.encoder.value if self.config.encoder else None}, "
            f"scaler={self.config.scaler.value if self.config.scaler else None}, "
            f"imputer={self.config.imputer.value if self.config.imputer else None})"
        )


def _categorical_columns(X: pd.DataFrame) -> List[str]:
    """Columns holding categorical data (object, string or category dtype)."""
    return [
        col
        for col, dtype in X.dtypes.items()
        if is_object_dtype(dtype) or isinstance(dtype, (pd.CategoricalDtype, pd.StringDtype))
    ]


def _numerical_columns(X: pd.DataFrame) -> List[str]:
    """Columns holding numbers of any width or signedness (booleans excluded)."""
    return [
        col
        for col, dtype in X.dtypes.items()
        if is_numeric_dtype(dtype) and not is_bool_dtype(dtype)
    ]


def _positional_values(result: Any, columns: List[str]) -> np.ndarray:
    """
    Extract a float array from a component's output, in ``columns`` order.

    Components may return a DataFrame with a fresh index (Atlantic 1.x) or with
    the caller's index (Atlantic 2.x). Taking the values positionally makes the
    result independent of either behaviour.
    """
    if isinstance(result, pd.DataFrame):
        return result[columns].to_numpy(dtype="float64", na_value=np.nan)
    return np.asarray(result, dtype="float64")
