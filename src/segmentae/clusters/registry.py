from typing import Dict, Type, List
from segmentae.core.base import AbstractClusterModel
from segmentae.core.constants import ClusterModel
from segmentae.core.exceptions import ConfigurationError


class ClusterRegistry:
    """
    Registry for clustering model classes.
    
    This class implements the Registry pattern for managing clustering
    algorithm implementations. Models are registered at module load time
    and can be instantiated dynamically by type.
    """
    
    _models: Dict[ClusterModel, Type[AbstractClusterModel]] = {}
    
    @classmethod
    def register(cls,
                 model_type: ClusterModel,
                 model_class: Type[AbstractClusterModel]) -> None:
        """
        Register a clustering model class.
        """
        if model_type in cls._models:
            raise ConfigurationError(
                f"Cluster model '{model_type.value}' is already registered"
            )
        
        # Validate that model_class implements AbstractClusterModel
        if not issubclass(model_class, AbstractClusterModel):
            raise ConfigurationError(
                f"Model class must inherit from AbstractClusterModel"
            )
        
        cls._models[model_type] = model_class
    
    @classmethod
    def create(cls,
               model_type: ClusterModel,
               **kwargs) -> AbstractClusterModel:
        """
        Create a clustering model instance.
        """
        if model_type not in cls._models:
            available = [m.value for m in cls.list_available()]
            raise ConfigurationError(
                f"Unknown cluster model: '{model_type.value}'",
                valid_options=available
            )
        
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def list_available(cls) -> List[ClusterModel]:
        """
        List all registered clustering models.
        """
        return list(cls._models.keys())
    
    @classmethod
    def is_registered(cls, model_type: ClusterModel) -> bool:
        """
        Check if a model type is registered.
        """
        return model_type in cls._models
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered models.
        """
        cls._models.clear()