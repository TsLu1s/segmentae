class SegmentAEError(Exception):
    """Base exception class for all SegmentAE errors."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ClusteringError(SegmentAEError):
    """Exception raised for clustering-related errors."""
    
    def __init__(self, message: str):
        super().__init__(f"Clustering Error: {message}")


class ReconstructionError(SegmentAEError):
    """Exception raised for reconstruction-related errors."""
    
    def __init__(self, message: str):
        super().__init__(f"Reconstruction Error: {message}")


class ValidationError(SegmentAEError):
    """Exception raised for input validation errors."""
    
    def __init__(self, message: str, suggestion: str = None):
        error_msg = f"Validation Error: {message}"
        if suggestion:
            error_msg += f"\nSuggestion: {suggestion}"
        super().__init__(error_msg)


class ModelNotFittedError(SegmentAEError):
    """Exception raised when attempting to use a model before fitting."""
    
    def __init__(self, message: str = None, component: str = "Model"):
        if message is None:
            message = (
                f"{component} must be fitted before use. "
                f"Please call the fit() or appropriate fitting method first."
            )
        super().__init__(message)


class ConfigurationError(SegmentAEError):
    """Exception raised for invalid configuration parameters."""
    
    def __init__(self, message: str, valid_options: list = None):
        error_msg = f"Configuration Error: {message}"
        if valid_options:
            options_str = ", ".join(str(opt) for opt in valid_options)
            error_msg += f"\nValid options: {options_str}"
        super().__init__(error_msg)


class AutoencoderError(SegmentAEError):
    """Exception raised for autoencoder-related errors."""
    
    def __init__(self, message: str):
        super().__init__(f"Autoencoder Error: {message}")