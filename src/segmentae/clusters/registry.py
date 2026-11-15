from typing import Dict, List, Type, Union

from segmentae.core.base import AbstractClusterModel
from segmentae.core.constants import ClusterModel
from segmentae.core.exceptions import ConfigurationError

ClusterModelName = Union[ClusterModel, str]


class ClusterRegistry:
    """
    Registry for clustering model classes.

    This class implements the Registry pattern for managing clustering
    algorithm implementations. The built-in models are registered at module
    load time; custom models can be added under any name:

        >>> ClusterRegistry.register("Birch", BirchCluster)
        >>> Clustering(cluster_model=["Birch"], n_clusters=4)

    Names are stored as plain strings, so a ``ClusterModel`` member and its
    string value (``ClusterModel.KMEANS`` / ``"KMeans"``) are interchangeable.
    """

    _models: Dict[str, Type[AbstractClusterModel]] = {}

    @staticmethod
    def _key(model_type: ClusterModelName) -> str:
        """Normalise an enum member or a string to the registry key."""
        if isinstance(model_type, ClusterModel):
            return model_type.value
        if isinstance(model_type, str) and model_type.strip():
            return model_type.strip()
        raise ConfigurationError(
            f"Cluster model name must be a non-empty string or ClusterModel, "
            f"got {type(model_type).__name__}"
        )

    @classmethod
    def register(
        cls,
        model_type: ClusterModelName,
        model_class: Type[AbstractClusterModel],
        override: bool = False,
    ) -> None:
        """
        Register a clustering model class.

        Args:
            model_type: Name (or ``ClusterModel`` member) the model is selected by.
            model_class: Class inheriting from ``AbstractClusterModel``.
            override: Replace an existing registration instead of raising.
        """
        key = cls._key(model_type)

        if not (isinstance(model_class, type) and issubclass(model_class, AbstractClusterModel)):
            raise ConfigurationError("Model class must inherit from AbstractClusterModel")

        if key in cls._models and not override:
            raise ConfigurationError(f"Cluster model '{key}' is already registered")

        cls._models[key] = model_class

    @classmethod
    def get(cls, model_type: ClusterModelName) -> Type[AbstractClusterModel]:
        """
        Return the class registered under ``model_type``.
        """
        key = cls._key(model_type)
        if key not in cls._models:
            raise ConfigurationError(f"Unknown cluster model: '{key}'", valid_options=cls.names())
        return cls._models[key]

    @classmethod
    def create(cls, model_type: ClusterModelName, **kwargs) -> AbstractClusterModel:
        """
        Create a clustering model instance from its constructor arguments.
        """
        return cls.get(model_type)(**kwargs)

    @classmethod
    def names(cls) -> List[str]:
        """
        List the names of all registered clustering models as plain strings.
        """
        return list(cls._models.keys())

    @classmethod
    def list_available(cls) -> List[ClusterModelName]:
        """
        List all registered clustering models.

        Built-in models are returned as ``ClusterModel`` members (which compare
        equal to their names); custom models as the names they were registered under.
        """
        builtin = {member.value: member for member in ClusterModel}
        return [builtin.get(name, name) for name in cls._models]

    @classmethod
    def is_registered(cls, model_type: ClusterModelName) -> bool:
        """
        Check if a model type is registered.
        """
        try:
            return cls._key(model_type) in cls._models
        except ConfigurationError:
            return False

    @classmethod
    def unregister(cls, model_type: ClusterModelName) -> None:
        """
        Remove a registered model (no-op if it is not registered).
        """
        cls._models.pop(cls._key(model_type), None)

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered models.
        """
        cls._models.clear()
