"""
SegmentAE: A Python Library for Anomaly Detection Optimization

SegmentAE enhances anomaly detection performance through the optimization of
reconstruction error by integrating clustering methods with tabular autoencoders.

Key Components:
    - Preprocessing: Data preparation with encoding, scaling, and imputation
    - Clustering: Multiple clustering algorithms (KMeans, MiniBatch KMeans, GMM)
    - SegmentAE: Main pipeline integrating autoencoders and clustering
    - Autoencoders: Dense, BatchNorm, and Ensemble implementations
    - Optimizer: Grid search for optimal configuration
"""

from importlib import import_module
from typing import TYPE_CHECKING

from segmentae._version import __version__

__author__ = "Luís Fernando da Silva Santos"

from segmentae.clusters import Clustering, ClusterRegistry
from segmentae.core import (
    ClusterModel,
    EncoderType,
    ImputerType,
    PerformanceMetric,
    PhaseType,
    ScalerType,
    ThresholdMetric,
)
from segmentae.data_sources import load_dataset
from segmentae.metrics import metrics_classification, metrics_regression
from segmentae.optimization import SegmentAE_Optimizer
from segmentae.pipeline import SegmentAE
from segmentae.processing.preprocessing import Preprocessing

if TYPE_CHECKING:  # make the lazily loaded names visible to type checkers and IDEs
    from segmentae.autoencoders.batch_norm import BatchNormAutoencoder
    from segmentae.autoencoders.dense import DenseAutoencoder
    from segmentae.autoencoders.ensemble import EnsembleAutoencoder

# The built-in autoencoders need TensorFlow/Keras, which takes seconds to import.
# They are loaded on first access (``segmentae.DenseAutoencoder`` or
# ``from segmentae import DenseAutoencoder``), so importing the package stays
# fast and pipelines built around a custom model never pay that cost.
_LAZY_ATTRIBUTES = {
    "DenseAutoencoder": "segmentae.autoencoders.dense",
    "BatchNormAutoencoder": "segmentae.autoencoders.batch_norm",
    "EnsembleAutoencoder": "segmentae.autoencoders.ensemble",
}


def __getattr__(name: str):
    if name == "autoencoders":  # the subpackage, e.g. segmentae.autoencoders.DenseAutoencoder
        return import_module("segmentae.autoencoders")
    if name in _LAZY_ATTRIBUTES:
        value = getattr(import_module(_LAZY_ATTRIBUTES[name]), name)
        globals()[name] = value  # cache so the import happens only once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES) | {"autoencoders"})


__all__ = [
    "__version__",
    "__author__",
    "PhaseType",
    "ClusterModel",
    "ThresholdMetric",
    "PerformanceMetric",
    "EncoderType",
    "ScalerType",
    "ImputerType",
    "Preprocessing",
    "Clustering",
    "ClusterRegistry",
    "SegmentAE",
    "SegmentAE_Optimizer",
    "metrics_classification",
    "metrics_regression",
    "load_dataset",
    "DenseAutoencoder",
    "BatchNormAutoencoder",
    "EnsembleAutoencoder",
]
