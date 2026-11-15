from importlib import import_module
from importlib.util import find_spec
from typing import Any, Dict, Optional, Tuple, Union

from segmentae.core.constants import (
    ENCODER_CLASS_MAP,
    IMPUTER_CLASS_MAP,
    SCALER_CLASS_MAP,
    EncoderType,
    ImputerType,
    ScalerType,
)
from segmentae.core.exceptions import ConfigurationError

# Module that provides each family of components, for the two Atlantic layouts:
# Atlantic >= 2.0 first, then the Atlantic 1.x paths.
_ATLANTIC_MODULES: Dict[str, Tuple[str, ...]] = {
    "encoder": ("atlantic.preprocessing.encoders", "atlantic.processing.encoders"),
    "scaler": ("atlantic.preprocessing.scalers", "atlantic.processing.scalers"),
    "imputer": ("atlantic.preprocessing.imputers", "atlantic.imputers.imputation"),
}


def _load_atlantic_class(kind: str, class_name: str) -> Any:
    """
    Import a preprocessing component from Atlantic on first use.

    Importing lazily keeps ``import segmentae`` light (Atlantic pulls in its
    whole AutoML stack) and lets the same code run on Atlantic 1.x and 2.x,
    which expose the same classes under different module paths.
    """
    last_error: Optional[Exception] = None
    for module_name in _ATLANTIC_MODULES[kind]:
        try:
            return getattr(import_module(module_name), class_name)
        except (ImportError, AttributeError) as exc:
            last_error = exc

    if find_spec("atlantic") is None:
        hint = "It is not installed; install it with: pip install atlantic"
    else:
        hint = (
            "The installed version could not be imported in this environment. "
            "Atlantic 2.x requires Python 3.11 or newer; on Python 3.10 install "
            "'atlantic<2'"
        )
    raise ConfigurationError(
        f"Could not load '{class_name}' from the 'atlantic' package ({last_error}). {hint}"
    ) from last_error


class ComponentFactory:
    """Factory class for creating preprocessing components."""

    @staticmethod
    def create_imputer(imputer_type: Optional[Union[ImputerType, str]]) -> Optional[Any]:
        """Create an imputer instance based on type."""
        if imputer_type is None:
            return None

        imputer_member: ImputerType = _coerce(imputer_type, ImputerType, "imputer")
        imputer_class = _load_atlantic_class("imputer", IMPUTER_CLASS_MAP[imputer_member])
        return imputer_class(strategy="mean")

    @staticmethod
    def create_encoder(encoder_type: Optional[Union[EncoderType, str]]) -> Optional[Any]:
        """Create an encoder instance based on type."""
        if encoder_type is None:
            return None

        encoder_member: EncoderType = _coerce(encoder_type, EncoderType, "encoder")
        return _load_atlantic_class("encoder", ENCODER_CLASS_MAP[encoder_member])()

    @staticmethod
    def create_scaler(scaler_type: Optional[Union[ScalerType, str]]) -> Optional[Any]:
        """Create a scaler instance based on type."""
        if scaler_type is None:
            return None

        scaler_member: ScalerType = _coerce(scaler_type, ScalerType, "scaler")
        return _load_atlantic_class("scaler", SCALER_CLASS_MAP[scaler_member])()


def _coerce(value: Any, enum_class: Any, label: str) -> Any:
    """Convert a string to its enum member, raising ConfigurationError if unknown."""
    if isinstance(value, enum_class):
        return value
    try:
        return enum_class(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"Unknown {label} type: {value}", valid_options=[member.value for member in enum_class]
        ) from exc
