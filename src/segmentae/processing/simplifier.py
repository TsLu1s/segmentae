from typing import Optional, Union, Any
from atlantic.imputers.imputation import AutoSimpleImputer # || #AutoKNNImputer, AutoIterativeImputer
from atlantic.processing.encoders import (
    AutoIFrequencyEncoder,
    AutoLabelEncoder,
    AutoOneHotEncoder
)
from atlantic.processing.scalers import (
    AutoMinMaxScaler,
    AutoStandardScaler,
    AutoRobustScaler
)

from segmentae.core.constants import EncoderType, ScalerType, ImputerType
from segmentae.core.exceptions import ConfigurationError


class ComponentFactory:
    """Factory class for creating preprocessing components."""
    
    @staticmethod
    def create_imputer(imputer_type: Optional[Union[ImputerType, str]]) -> Optional[Any]:
        """Create an imputer instance based on type."""
        if imputer_type is None:
            return None
        
        if isinstance(imputer_type, str):
            imputer_type = ImputerType(imputer_type)
        
        if imputer_type == ImputerType.SIMPLE:
            return AutoSimpleImputer(strategy="mean")
        
        raise ConfigurationError(
            f"Unknown imputer type: {imputer_type}",
            valid_options=list(ImputerType)
        )
    
    @staticmethod
    def create_encoder(encoder_type: Optional[Union[EncoderType, str]]) -> Optional[Any]:
        """Create an encoder instance based on type."""
        if encoder_type is None:
            return None
        
        if isinstance(encoder_type, str):
            encoder_type = EncoderType(encoder_type)
        
        match encoder_type:
            case EncoderType.IFREQUENCY:
                return AutoIFrequencyEncoder()
            case EncoderType.LABEL:
                return AutoLabelEncoder()
            case EncoderType.ONEHOT:
                return AutoOneHotEncoder()
            case _:
                raise ConfigurationError(
                    f"Unknown encoder type: {encoder_type}",
                    valid_options=list(EncoderType)
                )
    
    @staticmethod
    def create_scaler(scaler_type: Optional[Union[ScalerType, str]]) -> Optional[Any]:
        """Create a scaler instance based on type."""
        if scaler_type is None:
            return None
        
        if isinstance(scaler_type, str):
            scaler_type = ScalerType(scaler_type)
        
        match scaler_type:
            case ScalerType.MINMAX:
                return AutoMinMaxScaler()
            case ScalerType.STANDARD:
                return AutoStandardScaler()
            case ScalerType.ROBUST:
                return AutoRobustScaler()
            case _:
                raise ConfigurationError(
                    f"Unknown scaler type: {scaler_type}",
                    valid_options=list(ScalerType)
                )